"""
比赛提交生成脚本
生成 submission.csv + predicted_masks/ 并打包为 zip

两阶段推理：
  Phase 1: 扫描训练集，构建 per-category DINOv2 patch memory bank
  Phase 2: 推理测试集，用 memory bank 计算像素级异常 mask

支持 Test_A（已见类别）和 Test_B（含未见类别）

用法：
    # macOS / MPS
    uv run python -m src.submit \
        --checkpoint checkpoints/battery/best.pth \
        --test_split Test_A \
        --image_size 224 \
        --dinov2_weights weights/dinov2_vitb14_pretrain.pth

    # GPU
    uv run python -m src.submit \
        --checkpoint checkpoints/all/best.pth \
        --test_split Test_A \
        --image_size 518 \
        --dinov2_weights weights/dinov2_vitb14_pretrain.pth
"""
import argparse
import csv
import os
import sys
import zipfile
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.config import get_config
from src.data.dataset import build_dataloader
from src.models.dinov2_extractor import DINOv2Extractor
from src.models.inpformer import INPFormer


# ── 常量 ─────────────────────────────────────────────────────────
MASK_SIZE = 448  # 比赛要求的 mask 尺寸
MEMORY_BANK_MAX = 2000  # 每个类别 memory bank 最大特征数
K_NEIGHBORS = 5  # k-NN 近邻数


def parse_args():
    p = argparse.ArgumentParser(description="生成比赛提交包")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--root_dir", type=str, default="data")
    p.add_argument("--test_split", type=str, default="Test_A",
                   help="Test_A 或 Test_B")
    p.add_argument("--dinov2_weights", type=str, default=None)
    p.add_argument("--image_size", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--output_dir", type=str, default="submission",
                   help="提交文件输出目录")
    p.add_argument("--category", type=str, default=None,
                   help="仅处理指定类别（调试用，提交时不指定）")
    p.add_argument("--k_neighbors", type=int, default=K_NEIGHBORS,
                   help="k-NN 近邻数")
    p.add_argument("--memory_bank_max", type=int, default=MEMORY_BANK_MAX,
                   help="每个类别 memory bank 最大特征数")
    return p.parse_args()


# ── Phase 1: 构建 Memory Bank ─────────────────────────────────────

@torch.no_grad()
def build_memory_bank(
    dinov2: DINOv2Extractor,
    train_loader,
    device: torch.device,
    max_features: int = MEMORY_BANK_MAX,
) -> dict:
    """
    扫描训练集，为每个类别构建 DINOv2 patch 特征的 memory bank

    Returns:
        dict: {category_name: Tensor[max_features, dinov2_dim]}
    """
    print("\n[Phase 1] 构建 Memory Bank...")
    bank = {}  # {category: Tensor[≤max_features, D]}

    # 逐 batch 扫描，对每个类别增量收集并截断
    for batch_idx, batch in enumerate(tqdm(train_loader, desc="  扫描训练集")):
        views = batch["views"].to(device)
        categories = batch["category"]
        B, V = views.shape[0], views.shape[1]

        dinov2_feats = dinov2.extract_multi_view(views)
        patch_features = dinov2_feats["patch_features"]  # [B, V, N, D]

        for i in range(B):
            cat = categories[i]
            # 每个样本随机采样部分 patch（避免内存爆炸）
            patches = patch_features[i].reshape(-1, patch_features.shape[-1])  # [V*N, D]
            sample_k = min(patches.shape[0], max_features // 5)  # 每个样本最多取 1/5
            if patches.shape[0] > sample_k:
                idx = torch.randperm(patches.shape[0], device=device)[:sample_k]
                patches = patches[idx]
            patches_cpu = patches.cpu()

            if cat not in bank:
                bank[cat] = patches_cpu
            else:
                bank[cat] = torch.cat([bank[cat], patches_cpu], dim=0)
                # 超过上限时随机截断
                if bank[cat].shape[0] > max_features:
                    idx = torch.randperm(bank[cat].shape[0])[:max_features]
                    bank[cat] = bank[cat][idx]

        # 释放 GPU 显存
        del dinov2_feats, patch_features

    # L2 归一化并移到 GPU
    for cat in bank:
        bank[cat] = F.normalize(bank[cat], dim=-1).to(device)
        print(f"  {cat}: {bank[cat].shape[0]} 个特征")

    print(f"[Phase 1] 完成，共 {len(bank)} 个类别\n")
    return bank


# ── 像素级异常热力图生成 ────────────────────────────────────────

@torch.no_grad()
def compute_patch_anomaly_score(
    test_patches: torch.Tensor,   # [V, N, D] 测试样本的 DINOv2 patch 特征
    memory_bank: torch.Tensor,    # [M, D] 该类别的 memory bank（已 L2 归一化）
    k: int = K_NEIGHBORS,
) -> torch.Tensor:
    """
    计算每个 patch 的 k-NN 异常得分

    Args:
        test_patches: [V, N, D]
        memory_bank:  [M, D] 已 L2 归一化

    Returns:
        patch_score: [V, N] 每个 patch 的异常得分
    """
    V, N, D = test_patches.shape
    M = memory_bank.shape[0]

    # L2 归一化测试特征
    test_norm = F.normalize(test_patches.reshape(-1, D), dim=-1)  # [V*N, D]

    # 计算余弦相似度矩阵: [V*N, M]
    # 由于都已 L2 归一化，点积 = 余弦相似度
    sim_matrix = test_norm @ memory_bank.T  # [V*N, M]

    # 取 top-k 最近邻的平均相似度
    k = min(k, M)
    topk_sim, _ = sim_matrix.topk(k, dim=-1)  # [V*N, k]
    avg_sim = topk_sim.mean(dim=-1)  # [V*N]

    # 异常得分 = 1 - 平均相似度（越小越正常，越大越异常）
    patch_score = 1.0 - avg_sim  # [V*N]

    return patch_score.reshape(V, N)


@torch.no_grad()
def generate_pixel_mask(
    patch_score: torch.Tensor,   # [V, N]  每个视角的 patch 异常得分
    n_h: int,                    # 水平 patch 数
    n_w: int,                    # 垂直 patch 数
    mask_size: int = MASK_SIZE,
) -> np.ndarray:
    """
    将 patch 级异常得分上采样为像素级 mask

    Returns:
        masks: [V, mask_size, mask_size]  uint8 灰度图
    """
    V = patch_score.shape[0]

    # 1. reshape 为 2D grid: [V, 1, n_h, n_w]
    score_map = patch_score.reshape(V, 1, n_h, n_w)

    # 2. 双线性上采样到 mask_size x mask_size
    upsampled = F.interpolate(
        score_map,
        size=(mask_size, mask_size),
        mode="bilinear",
        align_corners=False,
    )  # [V, 1, mask_size, mask_size]

    # 3. 逐视角归一化到 [0, 255]
    masks = upsampled.squeeze(1)  # [V, mask_size, mask_size]
    result = np.zeros((V, mask_size, mask_size), dtype=np.uint8)

    for v in range(V):
        s = masks[v].cpu().float()
        s_min, s_max = s.min(), s.max()
        if s_max - s_min > 1e-8:
            s = (s - s_min) / (s_max - s_min) * 255.0
        else:
            s = torch.zeros_like(s)
        result[v] = s.numpy().astype(np.uint8)

    return result


# ── Phase 2: 推理主函数 ──────────────────────────────────────────

@torch.no_grad()
def run_inference(
    dinov2: DINOv2Extractor,
    model: INPFormer,
    dataloader,
    memory_bank: dict,
    device: torch.device,
    output_dir: str,
    k_neighbors: int = K_NEIGHBORS,
) -> list:
    """
    在测试集上推理，生成 submission.csv 和 predicted_masks/

    Returns:
        List[dict]: 所有样本的结果
    """
    model.eval()

    # 创建 mask 输出目录
    mask_dir = Path(output_dir) / "predicted_masks"
    mask_dir.mkdir(parents=True, exist_ok=True)

    csv_rows = []
    all_results = []
    raw_scores = []
    total = len(dataloader)
    missing_cats = set()

    for batch_idx, batch in enumerate(dataloader):
        views = batch["views"].to(device)     # [B, V, C, H, W]
        categories = batch["category"]         # List[str]
        sample_ids = batch["sample_id"]        # List[str]
        B, V = views.shape[0], views.shape[1]

        # ── 1. DINOv2 特征提取 ──
        dinov2_feats = dinov2.extract_multi_view(views)
        patch_features = dinov2_feats["patch_features"]  # [B, V, N, D_dino]
        n_h = dinov2_feats["num_patches_h"]
        n_w = dinov2_feats["num_patches_w"]

        # ── 2. INP-Former 推理（图像级得分）──
        out = model(dinov2_feats)

        # ── 3. 图像级异常得分（z 空间 L2 距离）──
        z_cls = out["z_cls"]          # [B, d_model]
        z_view = out["z_view"]        # [B, V, d_model]
        cls_dist = torch.norm(z_cls, dim=-1)
        view_dist = torch.norm(z_view, dim=-1).mean(dim=-1)
        image_score = cls_dist + view_dist  # [B]

        # ── 4. 像素级异常得分（Memory Bank k-NN）──
        for i in range(B):
            cat = categories[i]
            sid = sample_ids[i]
            group_folder = f"{cat}/{sid}"
            score = image_score[i].item()

            raw_scores.append(score)
            csv_rows.append({
                "group_folder": group_folder,
                "raw_score": score,
            })

            # 生成像素 mask
            sample_mask_dir = mask_dir / cat / sid
            sample_mask_dir.mkdir(parents=True, exist_ok=True)

            if cat in memory_bank:
                # 用 memory bank 计算 k-NN 距离
                test_patches = patch_features[i]  # [V, N, D_dino]
                ps = compute_patch_anomaly_score(
                    test_patches, memory_bank[cat], k=k_neighbors
                )
            else:
                # 未见类别：回退到 Transformer 编码特征
                if cat not in missing_cats:
                    print(f"  [警告] 类别 '{cat}' 无 memory bank，使用回退方案")
                    missing_cats.add(cat)
                # 使用 encoded patch 特征与 CLS 的距离
                encoded = out["patch_map"]  # [B, V, N, d_model]
                cls_feat = out["z_cls"][i]  # [d_model]
                patch_feats = encoded[i]    # [V, N, d_model]
                cos_sim = F.cosine_similarity(
                    patch_feats, cls_feat.unsqueeze(0).unsqueeze(0).expand_as(patch_feats), dim=-1
                )
                ps = 1.0 - cos_sim  # [V, N]

            masks = generate_pixel_mask(ps, n_h, n_w)

            for v in range(V):
                mask_path = sample_mask_dir / f"{v}_mask.png"
                Image.fromarray(masks[v], mode="L").save(str(mask_path))

            all_results.append({
                "group_folder": group_folder,
                "image_score": score,
            })

        if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == total:
            print(f"  推理进度: {batch_idx + 1}/{total}")

    # ── 5. Min-Max 归一化到 [0, 1] ──
    score_min = min(raw_scores)
    score_max = max(raw_scores)
    score_range = score_max - score_min if score_max > score_min else 1.0

    for row in csv_rows:
        normalized = (row["raw_score"] - score_min) / score_range
        row["anomaly_score"] = f"{normalized:.6f}"
        del row["raw_score"]

    print(f"  得分归一化: raw [{score_min:.2f}, {score_max:.2f}] → [0, 1]")

    return csv_rows


# ── 生成提交包 ──────────────────────────────────────────────────

def write_submission_csv(rows: list, output_path: str):
    """写入 submission.csv"""
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["group_folder", "anomaly_score"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"[CSV] 已写入 {len(rows)} 条记录 → {output_path}")


def create_submission_zip(submission_dir: str, zip_path: str):
    """将 submission 目录打包为 zip"""
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, dirs, files in os.walk(submission_dir):
            for fname in sorted(files):
                fpath = os.path.join(root, fname)
                arcname = os.path.relpath(fpath, submission_dir)
                zf.write(fpath, arcname)

    zip_size_mb = os.path.getsize(zip_path) / (1024 * 1024)
    print(f"[ZIP] 已打包 → {zip_path} ({zip_size_mb:.1f} MB)")


# ── Main ─────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # ── 配置 ──
    print(f"加载 checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = ckpt.get("config", get_config())

    if args.device:
        cfg.train.device = args.device
    if args.dinov2_weights:
        cfg.dinov2.weights_path = args.dinov2_weights
    if args.image_size:
        cfg.data.image_size = args.image_size
    if args.batch_size:
        cfg.data.batch_size = args.batch_size
    if args.num_workers is not None:
        cfg.data.num_workers = args.num_workers

    device = torch.device(cfg.train.device)

    # ── 模型 ──
    dinov2 = DINOv2Extractor(
        model_name=cfg.dinov2.model_name,
        out_indices=cfg.dinov2.out_indices,
        patch_size=cfg.dinov2.patch_size,
        frozen=True,
        weights_path=cfg.dinov2.weights_path,
    ).to(device)

    model = INPFormer(
        dinov2_dim=cfg.dinov2.embed_dim,
        d_model=cfg.inpformer.d_model,
        n_heads=cfg.inpformer.n_heads,
        n_layers=cfg.inpformer.n_layers,
        dim_ff=cfg.inpformer.dim_ff,
        dropout=0.0,
        num_views=cfg.data.num_views,
        n_flow_layers=cfg.inpformer.n_flow_layers,
        coupling_hidden=cfg.inpformer.coupling_hidden,
        score_type=cfg.inpformer.score_type,
    ).to(device)

    model.load_state_dict(ckpt["model"])
    print(f"已加载模型, epoch={ckpt.get('epoch', 'N/A')}")

    # ── Phase 1: 构建 Memory Bank ──
    train_categories = [args.category] if args.category else None
    train_loader = build_dataloader(cfg, split="Train", categories=train_categories)
    memory_bank = build_memory_bank(
        dinov2, train_loader, device, max_features=args.memory_bank_max
    )

    # ── Phase 2: 推理 ──
    test_loader = build_dataloader(
        cfg, split=args.test_split,
        categories=[args.category] if args.category else None,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[Phase 2] 开始推理 ({args.test_split})...")
    csv_rows = run_inference(
        dinov2, model, test_loader, memory_bank, device,
        str(output_dir), k_neighbors=args.k_neighbors
    )

    # ── 生成 CSV ──
    csv_path = output_dir / "submission.csv"
    write_submission_csv(csv_rows, str(csv_path))

    # ── 统计 ──
    scores = [float(r["anomaly_score"]) for r in csv_rows]
    print(f"\n推理完成:")
    print(f"  样本数: {len(csv_rows)}")
    print(f"  得分范围: [{min(scores):.4f}, {max(scores):.4f}]")
    print(f"  平均得分: {np.mean(scores):.4f} ± {np.std(scores):.4f}")

    # ── 打包 ZIP ──
    zip_path = output_dir.parent / f"{args.test_split}_submission.zip"
    create_submission_zip(str(output_dir), str(zip_path))

    # ── 验证 zip 结构 ──
    print(f"\n[验证] ZIP 内部结构:")
    with zipfile.ZipFile(str(zip_path), "r") as zf:
        names = zf.namelist()
        csv_count = sum(1 for n in names if n.endswith("submission.csv"))
        mask_count = sum(1 for n in names if n.endswith("_mask.png"))
        print(f"  submission.csv: {csv_count} 个")
        print(f"  mask 文件: {mask_count} 个")
        for n in sorted(names)[:10]:
            print(f"    {n}")
        if len(names) > 10:
            print(f"    ... (共 {len(names)} 个文件)")

    print(f"\n提交文件就绪: {zip_path}")


if __name__ == "__main__":
    main()
