"""
训练主脚本
DINOv2 + INP-Former 在 Real-IAD Variety 上的无监督异常检测训练

用法：
    python -m src.train                                  # 训练所有类别
    python -m src.train --category battery               # 训练单一类别
    python -m src.train --epochs 50 --lr 2e-4            # 自定义超参数
    python -m src.train --dinov2_weights /data/dinov2_vitb14_pretrain.pth  # 离线权重
    python -m src.train --no_amp                          # 关闭混合精度
"""
import argparse
import os
import sys
import time

# RTX 50 系列 (sm_120) cuBLAS LT 兼容性修复
# 强制使用旧版 cuBLAS 后端，绕过 cublasLtMatmul 融合 GEMM 的崩溃问题
os.environ.setdefault("CUBLAS_FRONTEND", "1")

import torch
import torch.nn as nn

# 项目根目录
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.config import get_config
from src.data.dataset import build_dataloader
from src.models.dinov2_extractor import DINOv2Extractor
from src.models.inpformer import INPFormer
from src.losses.loss import INPFormerLoss
from src.utils.utils import (
    set_seed,
    MetricTracker,
    get_lr_scheduler,
    save_checkpoint,
    load_checkpoint,
)


def parse_args():
    parser = argparse.ArgumentParser(description="DINOv2 + INP-Former 训练")
    parser.add_argument("--category", type=str, default=None, help="指定单一类别训练")
    parser.add_argument("--epochs", type=int, default=None, help="训练轮数")
    parser.add_argument("--lr", type=float, default=None, help="学习率")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size")
    parser.add_argument("--resume", type=str, default=None, help="恢复训练的 checkpoint 路径")
    parser.add_argument("--root_dir", type=str, default="data", help="数据集根目录")
    parser.add_argument("--device", type=str, default=None, help="设备: cuda/cpu/mps")
    # GPU 服务器适配参数
    parser.add_argument(
        "--dinov2_weights", type=str, default=None,
        help="DINOv2 本地权重路径（离线 GPU 服务器使用）",
    )
    parser.add_argument(
        "--torch_home", type=str, default=None,
        help="torch hub 缓存目录（如 /data/cache/torch_home）",
    )
    parser.add_argument(
        "--no_amp", action="store_true",
        help="禁用混合精度训练（调试用）",
    )
    parser.add_argument(
        "--num_workers", type=int, default=None,
        help="DataLoader worker 进程数（GPU 服务器建议 8~16）",
    )
    parser.add_argument(
        "--image_size", type=int, default=None,
        help="输入图像尺寸（MPS/CPU 验证建议 224，GPU 训练默认 518）",
    )
    return parser.parse_args()


def train_one_epoch(
    epoch: int,
    dinov2: DINOv2Extractor,
    model: INPFormer,
    dataloader,
    criterion: INPFormerLoss,
    optimizer,
    device: str,
    cfg,
    scaler: torch.cuda.amp.GradScaler = None,
):
    """训练一个 epoch，支持 AMP 混合精度"""
    model.train()
    tracker = MetricTracker(["total_loss", "nll_cls", "nll_view", "reg_loss", "margin_loss", "log_det_reg"])
    use_amp = scaler is not None

    for batch_idx, batch in enumerate(dataloader):
        views = batch["views"].to(device, non_blocking=True)  # [B, V, C, H, W]

        if batch_idx == 0:
            print(f"  [batch 0] views={list(views.shape)}, dtype={views.dtype}, device={views.device}")

        # 1. DINOv2 提取特征（冻结，无梯度）
        with torch.no_grad():
            dinov2_feats = dinov2.extract_multi_view(views)

        if batch_idx == 0:
            pf = dinov2_feats["patch_features"]
            ms = dinov2_feats.get("multi_scale_features", [])
            print(f"  [batch 0] DINOv2: patch={list(pf.shape)}, ms_layers={len(ms)}")

        # 2 + 3. INP-Former 前向 + 损失计算（AMP 自动混合精度）
        try:
            with torch.cuda.amp.autocast(enabled=use_amp):
                outputs = model(dinov2_feats)
                losses = criterion(outputs)
        except Exception as e:
            print(f"\n!!! 前向传播失败 (batch {batch_idx}): {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)

        if batch_idx == 0:
            print(f"  [batch 0] loss={losses['total_loss']:.4f}, 开始反向传播...")

        # 4. 反向传播
        optimizer.zero_grad(set_to_none=True)
        if use_amp:
            scaler.scale(losses["total_loss"]).backward()
            scaler.unscale_(optimizer)
            if cfg.train.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            losses["total_loss"].backward()
            if cfg.train.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
            optimizer.step()

        # 记录指标
        B = views.size(0)
        for k, v in losses.items():
            tracker.update(k, v.item(), n=B)

        if (batch_idx + 1) % cfg.train.log_interval == 0:
            avg = tracker.avg_all()
            print(
                f"  [Epoch {epoch}][{batch_idx + 1}/{len(dataloader)}] "
                f"loss={avg['total_loss']:.4f} "
                f"nll_cls={avg['nll_cls']:.4f} "
                f"nll_view={avg['nll_view']:.4f} "
                f"reg={avg['reg_loss']:.4f} "
                f"margin={avg.get('margin_loss', 0.0):.4f} "
                f"log_det={avg['log_det_reg']:.4f}"
                f"{'  [AMP]' if use_amp else ''}"
            )

    return tracker.avg_all()


def main():
    args = parse_args()

    # ── 配置 ──────────────────────────────────────────────────────────────
    cfg = get_config(category=args.category)
    cfg.data.root_dir = args.root_dir

    if args.epochs is not None:
        cfg.train.epochs = args.epochs
    if args.lr is not None:
        cfg.train.lr = args.lr
    if args.batch_size is not None:
        cfg.data.batch_size = args.batch_size
    if args.device is not None:
        cfg.train.device = args.device
    if args.num_workers is not None:
        cfg.data.num_workers = args.num_workers
    if args.image_size is not None:
        cfg.data.image_size = args.image_size
        # image_size 必须是 patch_size 的整数倍
        assert cfg.data.image_size % cfg.dinov2.patch_size == 0, (
            f"image_size {cfg.data.image_size} 必须是 patch_size {cfg.dinov2.patch_size} 的整数倍"
        )
    if args.dinov2_weights is not None:
        cfg.dinov2.weights_path = args.dinov2_weights
    if args.torch_home is not None:
        cfg.dinov2.torch_home = args.torch_home
    if args.no_amp:
        cfg.train.amp = False

    set_seed(cfg.train.seed)
    device = torch.device(cfg.train.device)

    # cuDNN 自动寻优（固定输入尺寸时开启可加速 ~20%）
    if cfg.train.cudnn_benchmark and torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        print("[cuDNN] benchmark 模式已开启")

    print(f"设备: {device}")
    if cfg.train.amp and torch.cuda.is_available():
        print("[AMP] 混合精度训练已开启 (FP16)")

    # ── 数据 ──────────────────────────────────────────────────────────────
    train_loader = build_dataloader(cfg, split="Train", categories=cfg.categories)

    # ── 模型 ──────────────────────────────────────────────────────────────
    print("加载 DINOv2 特征提取器...")
    dinov2 = DINOv2Extractor(
        model_name=cfg.dinov2.model_name,
        out_indices=cfg.dinov2.out_indices,
        patch_size=cfg.dinov2.patch_size,
        frozen=cfg.dinov2.frozen,
        weights_path=cfg.dinov2.weights_path,
        torch_home=cfg.dinov2.torch_home,
    ).to(device)

    print("构建 INP-Former 模型...")
    model = INPFormer(
        dinov2_dim=cfg.dinov2.embed_dim,
        d_model=cfg.inpformer.d_model,
        n_heads=cfg.inpformer.n_heads,
        n_layers=cfg.inpformer.n_layers,
        dim_ff=cfg.inpformer.dim_ff,
        dropout=cfg.inpformer.dropout,
        num_views=cfg.data.num_views,
        n_flow_layers=cfg.inpformer.n_flow_layers,
        coupling_hidden=cfg.inpformer.coupling_hidden,
        score_type=cfg.inpformer.score_type,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"INP-Former 可训练参数: {n_params / 1e6:.2f}M")

    # ── 损失 + 优化器 + 调度器 ────────────────────────────────────────────
    criterion = INPFormerLoss().to(device)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg.train.lr,
        weight_decay=cfg.train.weight_decay,
    )

    scheduler = get_lr_scheduler(optimizer, cfg)

    # AMP 混合精度 GradScaler
    scaler = (
        torch.cuda.amp.GradScaler()
        if (cfg.train.amp and torch.cuda.is_available())
        else None
    )

    # 恢复训练
    start_epoch = 0
    if args.resume:
        start_epoch = load_checkpoint(args.resume, model, optimizer, device)

    # checkpoint 目录
    cat_name = args.category or "all"
    ckpt_dir = os.path.join(cfg.train.checkpoint_dir, cat_name)
    os.makedirs(ckpt_dir, exist_ok=True)

    # ── 训练循环 ──────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"开始训练: {cfg.train.epochs} 个 epoch")
    print(f"类别: {cfg.categories or '全部 50 类'}")
    if cfg.dinov2.weights_path:
        print(f"DINOv2 权重: {cfg.dinov2.weights_path}")
    print(f"{'='*60}\n")

    # 诊断: DataLoader 是否正常
    n_batches = len(train_loader)
    print(f"DataLoader: {len(train_loader.dataset)} 样本, {n_batches} 个 batch (batch_size={cfg.data.batch_size})")
    if n_batches == 0:
        print("错误: DataLoader 没有 batch，请检查数据目录")
        sys.exit(1)

    print(f"start_epoch={start_epoch}, total_epochs={cfg.train.epochs}")
    print(f"循环将执行: range({start_epoch}, {cfg.train.epochs}) = {cfg.train.epochs - start_epoch} 个 epoch")

    best_loss = float("inf")

    for epoch in range(start_epoch, cfg.train.epochs):
        t0 = time.time()
        print(f"\n>>> Epoch {epoch} 开始...", flush=True)

        metrics = train_one_epoch(
            epoch, dinov2, model, train_loader, criterion,
            optimizer, device, cfg, scaler,
        )

        if scheduler is not None:
            scheduler.step()

        elapsed = time.time() - t0
        lr = optimizer.param_groups[0]["lr"]
        print(
            f"[Epoch {epoch}/{cfg.train.epochs - 1}] "
            f"loss={metrics['total_loss']:.4f} "
            f"nll_cls={metrics['nll_cls']:.4f} "
            f"nll_view={metrics['nll_view']:.4f} "
            f"lr={lr:.2e} "
            f"({elapsed:.1f}s)"
        )

        if metrics["total_loss"] < best_loss:
            best_loss = metrics["total_loss"]
            save_checkpoint(
                {
                    "epoch": epoch + 1,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "best_loss": best_loss,
                    "config": cfg,
                },
                os.path.join(ckpt_dir, "best.pth"),
            )

        if (epoch + 1) % cfg.train.save_interval == 0:
            save_checkpoint(
                {
                    "epoch": epoch + 1,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "best_loss": best_loss,
                    "config": cfg,
                },
                os.path.join(ckpt_dir, f"epoch_{epoch + 1}.pth"),
            )

    save_checkpoint(
        {
            "epoch": cfg.train.epochs,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "best_loss": best_loss,
            "config": cfg,
        },
        os.path.join(ckpt_dir, "last.pth"),
    )

    print(f"\n训练完成! 最优损失: {best_loss:.4f}")
    print(f"模型保存在: {ckpt_dir}")


if __name__ == "__main__":
    main()
