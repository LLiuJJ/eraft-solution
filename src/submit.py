"""
比赛提交生成脚本
生成 submission.csv + predicted_masks/ 并打包为 zip

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

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.config import get_config
from src.data.dataset import build_dataloader
from src.models.dinov2_extractor import DINOv2Extractor
from src.models.inpformer import INPFormer


# ── 常量 ─────────────────────────────────────────────────────────
MASK_SIZE = 448  # 比赛要求的 mask 尺寸


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
    return p.parse_args()


# ── 像素级异常热力图生成 ────────────────────────────────────────

@torch.no_grad()
def generate_pixel_mask(
    patch_score: torch.Tensor,   # [V, N]  每个视角的 patch 异常得分
    n_h: int,                    # 水平 patch 数
    n_w: int,                    # 垂直 patch 数
    mask_size: int = MASK_SIZE,
) -> np.ndarray:
    """
    将 patch 级异常得分上采样为像素级 mask

    Args:
        patch_score: [V, N]
        n_h, n_w:    patch grid 尺寸
        mask_size:   输出 mask 尺寸

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


# ── 推理主函数 ──────────────────────────────────────────────────

@torch.no_grad()
def run_inference(
    dinov2: DINOv2Extractor,
    model: INPFormer,
    dataloader,
    device: torch.device,
    output_dir: str,
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

    csv_rows = []  # 临时存储，后续归一化
    all_results = []
    raw_scores = []  # 收集原始得分，用于最后归一化
    total = len(dataloader)

    for batch_idx, batch in enumerate(dataloader):
        views = batch["views"].to(device)     # [B, V, C, H, W]
        categories = batch["category"]         # List[str]
        sample_ids = batch["sample_id"]        # List[str]
        B, V = views.shape[0], views.shape[1]

        # ── 1. DINOv2 特征提取 ──
        dinov2_feats = dinov2.extract_multi_view(views)

        # ── 2. INP-Former 推理 ──
        out = model(dinov2_feats)

        # ── 3. 图像级异常得分（z 空间 L2 距离，比 NLL 更鲁棒）──
        z_cls = out["z_cls"]          # [B, d_model]
        z_view = out["z_view"]        # [B, V, d_model]
        cls_dist = torch.norm(z_cls, dim=-1)                          # [B]
        view_dist = torch.norm(z_view, dim=-1).mean(dim=-1)           # [B]
        image_score = cls_dist + view_dist  # [B]

        # ── 4. Patch 级异常得分（用于像素 mask）──
        patch_map = out["patch_map"]          # [B, V, N, d_model]
        n_h = dinov2_feats["num_patches_h"]
        n_w = dinov2_feats["num_patches_w"]

        cls_out = out["z_cls"]                # [B, d_model]
        patch_map_flat = patch_map.reshape(B, V, -1, cls_out.shape[-1])
        cls_exp = cls_out.unsqueeze(1).unsqueeze(2)
        cos_sim = F.cosine_similarity(
            patch_map_flat, cls_exp.expand_as(patch_map_flat), dim=-1
        )  # [B, V, N]
        patch_score = 1.0 - cos_sim  # [B, V, N]  越大越异常

        # ── 5. 收集结果（先保存 mask，得分稍后归一化）──
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

            # 生成并保存 mask
            sample_mask_dir = mask_dir / cat / sid
            sample_mask_dir.mkdir(parents=True, exist_ok=True)

            ps = patch_score[i]  # [V, N]
            masks = generate_pixel_mask(ps, n_h, n_w)  # [V, 448, 448] uint8

            for v in range(V):
                mask_path = sample_mask_dir / f"{v}_mask.png"
                Image.fromarray(masks[v], mode="L").save(str(mask_path))

            all_results.append({
                "group_folder": group_folder,
                "image_score": score,
            })

        if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == total:
            print(f"  推理进度: {batch_idx + 1}/{total}")

    # ── 6. Min-Max 归一化到 [0, 1] ──
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
    """
    将 submission 目录打包为 zip
    zip 内部结构：
        submission.csv
        predicted_masks/
            类别/
                样本/
                    0_mask.png ~ 4_mask.png
    """
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

    # ── 数据 ──
    test_loader = build_dataloader(
        cfg, split=args.test_split,
        categories=[args.category] if args.category else None,
    )

    # ── 输出目录 ──
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── 推理 ──
    print(f"\n开始推理 ({args.test_split})...")
    csv_rows = run_inference(dinov2, model, test_loader, device, str(output_dir))

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
        # 打印前几个文件
        for n in sorted(names)[:10]:
            print(f"    {n}")
        if len(names) > 10:
            print(f"    ... (共 {len(names)} 个文件)")

    print(f"\n提交文件就绪: {zip_path}")


if __name__ == "__main__":
    main()
