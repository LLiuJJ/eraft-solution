"""
Feature Adapter 自监督训练脚本

用对比学习训练轻量级 FeatureAdapter：
  - 正常 patch → 拉近到 memory bank（对比正样本损失）
  - 合成异常 patch → 推远到 memory bank（margin hinge loss）

合成异常策略:
  1. CutPaste: 从其他样本复制 patch 特征
  2. 高斯噪声: 添加 N(0, 0.5) 扰动
  3. 零化: 将特征置零

用法：
    uv run python -m src.train_adapter \
        --dinov2_weights weights/dinov2_vitb14_pretrain.pth \
        --epochs 30 --batch_size 4 --image_size 518
"""
import argparse
import os
import sys
import time

# RTX 50 系列 cuBLAS 兼容性
os.environ.setdefault("CUBLAS_FRONTEND", "1")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":16:8")
# 内存优化：避免 fragmentation
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from pathlib import Path

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.config import get_config
from src.data.dataset import build_dataloader
from src.models.dinov2_extractor import DINOv2Extractor
from src.models.feature_adapter import FeatureAdapter, generate_synthetic_anomaly
from src.utils.utils import set_seed


# ── 参数解析 ──────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Feature Adapter 自监督训练")
    p.add_argument("--dinov2_weights", type=str, required=True,
                   help="DINOv2 预训练权重路径")
    p.add_argument("--root_dir", type=str, default="data")
    p.add_argument("--image_size", type=int, default=518)
    p.add_argument("--batch_size", type=int, default=4,
                   help="训练 batch size（16GB 显存建议 4）")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--checkpoint_dir", type=str, default="checkpoints/adapter")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no_amp", action="store_true", help="禁用混合精度")
    p.add_argument("--amp", action="store_true", help="启用混合精度（sm_120 默认关闭）")
    return p.parse_args()


# ── Memory Bank 构建（每类少量正常特征）────────────────────────────

@torch.no_grad()
def build_adapter_memory_bank(
    dinov2: DINOv2Extractor,
    train_loader,
    device: torch.device,
    max_per_cat: int = 5000,
) -> dict:
    """
    扫描训练集，为每个类别构建 DINOv2 4 层拼接特征的 memory bank。
    用于对比学习中的正样本参考。

    Returns:
        dict: {category: Tensor[M, 3072]}
    """
    print("[Memory Bank] 扫描训练集构建正常特征库...")
    bank = {}
    n_h, n_w = None, None

    # sm_120 兼容性: 清理 CUDA 状态
    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    for batch in train_loader:
        views = batch["views"].to(device)      # [B, V, C, H, W]
        categories = batch["category"]
        B, V = views.shape[0], views.shape[1]

        feats = dinov2.extract_multi_view(views)
        if n_h is None:
            n_h = feats["num_patches_h"]
            n_w = feats["num_patches_w"]

        ms = feats["multi_scale_features"]
        multi = torch.cat(ms, dim=-1)  # [B, V, N, 4D=3072]

        for i in range(B):
            cat = categories[i]
            patches = multi[i].reshape(-1, multi.shape[-1]).cpu()  # [V*N, 3072]
            # 随机采样减少内存
            k = min(patches.shape[0], max_per_cat // 10)
            if patches.shape[0] > k:
                idx = torch.randperm(patches.shape[0])[:k]
                patches = patches[idx]

            if cat not in bank:
                bank[cat] = patches
            else:
                bank[cat] = torch.cat([bank[cat], patches], dim=0)
                if bank[cat].shape[0] > max_per_cat:
                    idx = torch.randperm(bank[cat].shape[0])[:max_per_cat]
                    bank[cat] = bank[cat][idx]

    # L2 归一化后移到 GPU
    for cat in bank:
        bank[cat] = F.normalize(bank[cat], dim=-1).to(device)
        print(f"  {cat}: {bank[cat].shape[0]} 个正常特征 (dim={bank[cat].shape[1]})")

    return bank


@torch.no_grad()
def build_adapted_bank(
    raw_bank: dict,
    adapter: FeatureAdapter,
) -> dict:
    """
    将原始 3072 维 bank 通过 adapter 转换为 1024 维并缓存。
    每 epoch 开始调用一次，避免训练时重复计算。

    Returns:
        dict: {category: Tensor[M, 1024]}
    """
    adapted_bank = {}
    for cat, features in raw_bank.items():
        adapted = adapter(features)  # [M, 1024]
        adapted_bank[cat] = F.normalize(adapted, dim=-1)
    return adapted_bank


# ── 单 epoch 训练 ──────────────────────────────────────────────────

def train_one_epoch(
    epoch: int,
    dinov2: DINOv2Extractor,
    adapter: FeatureAdapter,
    dataloader,
    memory_bank: dict,
    adapted_bank: dict,
    optimizer,
    device: torch.device,
    margin: float = 2.0,
    use_amp: bool = True,
):
    """
    训练一个 epoch:
    - 正样本损失: 正常 patch 经 adapter 后距 memory bank 最近邻的距离 → 最小化
    - 负样本损失: 合成异常 patch 经 adapter 后距 memory bank 最近邻的距离 → 最大化(hinge)
    """
    adapter.train()
    total_loss = 0.0
    total_pos_loss = 0.0
    total_neg_loss = 0.0
    n_batches = 0

    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and device.type == "cuda")

    for batch in dataloader:
        views = batch["views"].to(device)      # [B, V, C, H, W]
        categories = batch["category"]
        B, V = views.shape[0], views.shape[1]

        with torch.no_grad():
            feats = dinov2.extract_multi_view(views)
            ms = feats["multi_scale_features"]
            normal_patches = torch.cat(ms, dim=-1)  # [B, V, N, 4D]
            n_h = feats["num_patches_h"]
            n_w = feats["num_patches_w"]

        # 展平: [B*V*N, 4D]
        N = n_h * n_w
        patches_flat = normal_patches.reshape(B * V * N, -1)  # [B*V*N, 3072]

        # 生成合成异常
        patches_for_anomaly = normal_patches.reshape(B, V * N, -1)  # [B, V*N, 3072]
        corrupted, is_anomaly = generate_synthetic_anomaly(
            patches_for_anomaly, n_h, n_w, anomaly_ratio=0.3
        )
        corrupted_flat = corrupted.reshape(B * V * N, -1)  # [B*V*N, 3072]
        is_anomaly_flat = is_anomaly.reshape(B * V * N)     # [B*V*N]

        # 混合正常 + 异常: [2*B*V*N, 3072]
        combined = torch.cat([patches_flat, corrupted_flat], dim=0)
        labels = torch.cat([
            torch.zeros(B * V * N, dtype=torch.bool, device=device),  # 正常
            is_anomaly_flat,                                           # 含合成异常
        ], dim=0)

        with torch.amp.autocast("cuda", enabled=use_amp and device.type == "cuda"):
            # adapter forward
            adapted = adapter(combined)  # [2*B*V*N, 1024]
            adapted = F.normalize(adapted, dim=-1)

            # 分批次计算到 memory bank 的距离（避免 OOM）
            # 按类别分批：每个样本用对应类别的 bank
            loss_pos = torch.tensor(0.0, device=device)
            loss_neg = torch.tensor(0.0, device=device)
            n_pos = 0
            n_neg = 0

            for i in range(B):
                cat = categories[i]
                if cat not in adapted_bank:
                    continue
                bank_adapted = adapted_bank[cat]  # [M, 1024] 预计算的

                # 该样本的 patch 索引范围
                start = i * V * N
                end = (i + 1) * V * N

                # 正常部分
                normal_adapted = adapted[start:end]        # [V*N, 1024]
                # 异常部分
                anomaly_adapted = adapted[B * V * N + start:B * V * N + end]  # [V*N, 1024]

                # 正样本: 距最近邻的距离 → 最小化
                dist_normal = torch.cdist(
                    normal_adapted.unsqueeze(0), bank_adapted.unsqueeze(0)
                ).squeeze(0)  # [V*N, M]
                nn_dist_normal = dist_normal.min(dim=-1).values  # [V*N]
                loss_pos = loss_pos + nn_dist_normal.sum()
                n_pos += V * N

                # 负样本: 合成异常 patch 的距离 → 最大化 (margin hinge)
                anomaly_mask = is_anomaly[i].reshape(-1)  # [V*N]
                if anomaly_mask.any():
                    anomaly_feats = anomaly_adapted[anomaly_mask]  # [k, 1024]
                    dist_anomaly = torch.cdist(
                        anomaly_feats.unsqueeze(0), bank_adapted.unsqueeze(0)
                    ).squeeze(0)  # [k, M]
                    nn_dist_anomaly = dist_anomaly.min(dim=-1).values  # [k]
                    # hinge: max(0, margin - dist)
                    hinge = torch.clamp(margin - nn_dist_anomaly, min=0.0)
                    loss_neg = loss_neg + hinge.sum()
                    n_neg += anomaly_mask.sum().item()

            # 总损失
            loss_pos = loss_pos / max(n_pos, 1)
            loss_neg = loss_neg / max(n_neg, 1)
            loss = loss_pos + loss_neg

        # 反向传播
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(adapter.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        total_pos_loss += loss_pos.item()
        total_neg_loss += loss_neg.item()
        n_batches += 1

    avg_loss = total_loss / max(n_batches, 1)
    avg_pos = total_pos_loss / max(n_batches, 1)
    avg_neg = total_neg_loss / max(n_batches, 1)
    print(
        f"  Epoch {epoch:3d} | loss={avg_loss:.4f} "
        f"(pos={avg_pos:.4f} neg={avg_neg:.4f})"
    )
    return avg_loss


# ── Main ──────────────────────────────────────────────────────────

def main():
    args = parse_args()
    set_seed(args.seed)

    # 设备
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"[Device] {device}")

    use_amp = args.amp and not args.no_amp and device.type == "cuda"
    # RTX 5070 Ti (sm_120): AMP 可能触发未编译的 kernel，默认关闭
    if use_amp:
        print("[AMP] 混合精度已开启")
    else:
        print("[AMP] 混合精度已关闭 (sm_120 兼容性)")

    # 配置
    cfg = get_config()
    cfg.data.root_dir = args.root_dir
    cfg.data.image_size = args.image_size
    cfg.data.batch_size = args.batch_size
    cfg.data.num_workers = args.num_workers
    cfg.dinov2.weights_path = args.dinov2_weights

    # DINOv2（冻结）
    dinov2 = DINOv2Extractor(
        model_name=cfg.dinov2.model_name,
        out_indices=cfg.dinov2.out_indices,
        patch_size=cfg.dinov2.patch_size,
        frozen=True,
        weights_path=cfg.dinov2.weights_path,
    ).to(device).eval()

    # DataLoader
    train_loader = build_dataloader(cfg, split="Train")
    print(f"[Data] {len(train_loader.dataset)} 训练样本")

    # Memory Bank（正常特征参考）
    memory_bank = build_adapter_memory_bank(dinov2, train_loader, device, max_per_cat=2000)

    # Feature Adapter
    adapter = FeatureAdapter(
        input_dim=4 * cfg.dinov2.embed_dim,  # 3072
        output_dim=1024,
        hidden_dim=2048,
        dropout=0.1,
    ).to(device)
    print(f"[Adapter] {adapter.num_parameters() / 1e6:.2f}M 可训练参数")

    # 优化器
    optimizer = AdamW(adapter.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # Checkpoint 目录
    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    best_loss = float("inf")
    print(f"\n{'='*60}")
    print(f"Feature Adapter 训练 | {args.epochs} epochs | margin=2.0")
    print(f"{'='*60}\n")

    for epoch in range(args.epochs):
        # 每 epoch 开始预计算 adapted bank（1024 维），避免重复计算
        adapted_bank = build_adapted_bank(memory_bank, adapter)
        
        t0 = time.time()
        avg_loss = train_one_epoch(
            epoch, dinov2, adapter, train_loader, memory_bank, adapted_bank,
            optimizer, device, margin=2.0, use_amp=use_amp,
        )
        scheduler.step()
        elapsed = time.time() - t0
        lr = optimizer.param_groups[0]["lr"]
        print(f"  lr={lr:.2e}  time={elapsed:.1f}s")

        # 保存 best
        if avg_loss < best_loss:
            best_loss = avg_loss
            save_path = ckpt_dir / "best.pth"
            torch.save({
                "epoch": epoch,
                "adapter_state": adapter.state_dict(),
                "loss": avg_loss,
                "config": {
                    "input_dim": adapter.input_dim,
                    "output_dim": adapter.output_dim,
                    "hidden_dim": 2048,
                    "dropout": 0.1,
                },
            }, save_path)
            print(f"  >>> 保存 best.pth (loss={avg_loss:.4f})")

        # 每 10 epoch 保存一次
        if (epoch + 1) % 10 == 0:
            save_path = ckpt_dir / f"epoch_{epoch}.pth"
            torch.save({
                "epoch": epoch,
                "adapter_state": adapter.state_dict(),
                "loss": avg_loss,
                "optimizer_state": optimizer.state_dict(),
                "config": {
                    "input_dim": adapter.input_dim,
                    "output_dim": adapter.output_dim,
                    "hidden_dim": 2048,
                    "dropout": 0.1,
                },
            }, save_path)

    # 保存 last
    save_path = ckpt_dir / "last.pth"
    torch.save({
        "epoch": args.epochs - 1,
        "adapter_state": adapter.state_dict(),
        "loss": avg_loss,
        "config": {
            "input_dim": adapter.input_dim,
            "output_dim": adapter.output_dim,
            "hidden_dim": 2048,
            "dropout": 0.1,
        },
    }, save_path)
    print(f"\n训练完成 | best_loss={best_loss:.4f} | 保存至 {ckpt_dir}/")


if __name__ == "__main__":
    main()
