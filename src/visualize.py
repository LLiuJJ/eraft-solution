"""
异常可视化脚本
在测试集上推理，为每个样本输出：
  1. 异常热力图（原图 + jet colormap 叠加）
  2. 黑白灰度 mask（与提交格式一致）

用法：
    # 全量测试集，输出 Top-20 最异常样本
    uv run python -m src.visualize \
        --checkpoint checkpoints/all/best.pth \
        --test_split Test_A \
        --image_size 518 \
        --dinov2_weights weights/dinov2_vitb14_pretrain.pth

    # 指定类别 + 全部样本
    uv run python -m src.visualize \
        --checkpoint checkpoints/all/best.pth \
        --category battery \
        --top_n 999 \
        --dinov2_weights weights/dinov2_vitb14_pretrain.pth

    # macOS / MPS 调试
    uv run python -m src.visualize \
        --checkpoint checkpoints/effect_transistor/best.pth \
        --category effect_transistor \
        --image_size 224 \
        --batch_size 2 \
        --num_workers 0 \
        --top_n 5 \
        --dinov2_weights weights/dinov2_vitb14_pretrain.pth
"""
import argparse
import os
import sys

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
from src.submit import (
    build_memory_bank,
    compute_patch_anomaly_score,
    compute_flow_patch_score,
    generate_pixel_mask,
    percentile_normalize,
    MASK_SIZE,
    MEMORY_BANK_MAX,
    K_NEIGHBORS,
)


# ── 热力图生成 ──────────────────────────────────────────────────

def apply_jet_colormap(gray: np.ndarray) -> np.ndarray:
    """
    将 [0, 255] 灰度图转为 jet colormap 的 RGB 图。
    纯 numpy 实现，不依赖 matplotlib。
    """
    x = np.clip(gray.astype(np.float64) / 255.0, 0.0, 1.0)
    r = np.clip(1.5 - np.abs(4 * x - 3), 0, 1)
    g = np.clip(1.5 - np.abs(4 * x - 2), 0, 1)
    b = np.clip(1.5 - np.abs(4 * x - 1), 0, 1)
    rgb = np.stack([r, g, b], axis=-1) * 255
    return rgb.astype(np.uint8)


def overlay_heatmap(
    original: np.ndarray,   # [H, W, 3] uint8
    mask: np.ndarray,       # [H, W] uint8 [0, 255]
    alpha: float = 0.5,
) -> np.ndarray:
    """
    将热力图叠加到原图上。
    alpha=1.0 时只显示热力图，alpha=0.0 时只显示原图。
    """
    heatmap = apply_jet_colormap(mask)  # [H, W, 3]
    blended = (original.astype(np.float32) * (1 - alpha)
               + heatmap.astype(np.float32) * alpha)
    return np.clip(blended, 0, 255).astype(np.uint8)


def load_original_image(
    root_dir: str,
    split: str,
    category: str,
    sample_id: str,
    view_idx: int,
    target_size: int = 448,
) -> np.ndarray:
    """
    加载原始图像（不归一化），resize 到 target_size。
    """
    sample_dir = os.path.join(root_dir, split, category, sample_id)
    img_path = os.path.join(sample_dir, f"{view_idx}.png")
    if not os.path.exists(img_path):
        img_path = os.path.join(sample_dir, f"{view_idx}.jpg")
    img = Image.open(img_path).convert("RGB")
    img = img.resize((target_size, target_size), Image.BILINEAR)
    return np.array(img)


# ── 推理与可视化 ─────────────────────────────────────────────────

@torch.no_grad()
def run_visualization(
    dinov2: DINOv2Extractor,
    model: INPFormer,
    dataloader,
    memory_bank: dict,
    device: torch.device,
    output_dir: str,
    root_dir: str,
    test_split: str,
    k_neighbors: int = K_NEIGHBORS,
    smooth_sigma: float = 2.0,
    clip_percentile: float = 30.0,
    flow_patch_weight: float = 0.2,
    top_n: int = 20,
    heatmap_alpha: float = 0.5,
):
    """
    在测试集上推理，为 Top-N 异常样本生成热力图和灰度 mask。
    """
    model.eval()

    # 收集所有样本的得分
    all_results = []

    for batch_idx, batch in enumerate(dataloader):
        views = batch["views"].to(device)
        categories = batch["category"]
        sample_ids = batch["sample_id"]
        B, V = views.shape[0], views.shape[1]

        # DINOv2 特征提取
        dinov2_feats = dinov2.extract_multi_view(views)
        patch_features = dinov2_feats["patch_features"]
        n_h = dinov2_feats["num_patches_h"]
        n_w = dinov2_feats["num_patches_w"]

        # INP-Former 推理
        out = model(dinov2_feats)

        # 图像级得分（z 空间 L2 距离）
        z_cls = out["z_cls"]
        z_view = out["z_view"]
        cls_dist = torch.norm(z_cls, dim=-1)
        view_dist = torch.norm(z_view, dim=-1).mean(dim=-1)
        flow_score = cls_dist + view_dist

        for i in range(B):
            cat = categories[i]
            sid = sample_ids[i]

            if cat in memory_bank:
                ms = dinov2_feats["multi_scale_features"]
                test_patches = torch.cat([ms[-2][i], ms[-1][i]], dim=-1)

                # k-NN patch 得分
                knn_patch = compute_patch_anomaly_score(
                    test_patches, memory_bank[cat], k=k_neighbors
                )

                # Flow patch 得分融合
                if flow_patch_weight > 0:
                    patch_map_i = out["patch_map"][i]
                    flow_patch = compute_flow_patch_score(patch_map_i)
                    knn_norm = percentile_normalize(knn_patch.cpu().numpy())
                    flow_norm = percentile_normalize(flow_patch.cpu().numpy())
                    ps_np = (1 - flow_patch_weight) * knn_norm + flow_patch_weight * flow_norm
                    ps = torch.from_numpy(ps_np).to(knn_patch.device)
                else:
                    ps = knn_patch

                knn_score = float(ps.topk(max(1, ps.shape[1] // 10), dim=-1)[0].mean().item())
                pixel_max = float(ps.max().item())
            else:
                encoded = out["patch_map"]
                cls_feat = out["z_cls"][i]
                patch_feats = encoded[i]
                cos_sim = F.cosine_similarity(
                    patch_feats, cls_feat.unsqueeze(0).unsqueeze(0).expand_as(patch_feats), dim=-1
                )
                ps = 1.0 - cos_sim
                knn_score = 0.0
                pixel_max = float(ps.max().item())

            total_score = float(flow_score[i].item())

            all_results.append({
                "category": cat,
                "sample_id": sid,
                "flow_score": total_score,
                "knn_score": knn_score,
                "pixel_max": pixel_max,
                "patch_score": ps.cpu().numpy(),
                "n_h": n_h,
                "n_w": n_w,
            })

        if (batch_idx + 1) % 10 == 0:
            print(f"  推理进度: {batch_idx + 1}/{len(dataloader)}")

    # 按 flow_score 排序，取 Top-N
    all_results.sort(key=lambda x: x["flow_score"], reverse=True)
    top_results = all_results[:top_n]

    print(f"\n生成 Top-{len(top_results)} 异常样本可视化...")

    # 为每个样本生成可视化
    for rank, result in enumerate(top_results):
        cat = result["category"]
        sid = result["sample_id"]
        ps = result["patch_score"]
        n_h = result["n_h"]
        n_w = result["n_w"]

        # 生成灰度 mask
        masks = generate_pixel_mask(
            torch.from_numpy(ps), n_h, n_w,
            smooth_sigma=smooth_sigma,
            clip_percentile=clip_percentile,
        )  # [V, mask_size, mask_size] uint8

        V = masks.shape[0]

        # 为每个视角生成原图 + 热力图 + mask
        sample_dir = os.path.join(
            output_dir, f"{rank:03d}_{cat}_{sid}"
        )
        os.makedirs(sample_dir, exist_ok=True)

        for v in range(V):
            # 加载原图（resize 到 448×448）
            orig = load_original_image(
                root_dir, test_split, cat, sid, v, target_size=MASK_SIZE
            )

            mask = masks[v]  # [mask_size, mask_size]

            # 热力图叠加
            heatmap = overlay_heatmap(orig, mask, alpha=heatmap_alpha)

            # 保存
            Image.fromarray(orig).save(os.path.join(sample_dir, f"{v}_original.png"))
            Image.fromarray(heatmap).save(os.path.join(sample_dir, f"{v}_heatmap.png"))
            Image.fromarray(mask, mode="L").save(os.path.join(sample_dir, f"{v}_mask.png"))

        # 生成组合图：5 视角并排（原图 | 热力图 | mask）
        combined = create_combined_view(
            sample_dir, V, mask_size=MASK_SIZE
        )
        combined.save(os.path.join(sample_dir, "combined.png"))

        print(f"  [{rank+1:03d}] {cat}/{sid}  "
              f"score={result['flow_score']:.2f}  "
              f"pixel_max={result['pixel_max']:.4f}")

    # 保存得分排名 CSV
    csv_path = os.path.join(output_dir, "anomaly_scores.csv")
    with open(csv_path, "w") as f:
        f.write("rank,category,sample_id,flow_score,knn_score,pixel_max\n")
        for rank, r in enumerate(top_results):
            f.write(f"{rank},{r['category']},{r['sample_id']},"
                    f"{r['flow_score']:.4f},{r['knn_score']:.4f},"
                    f"{r['pixel_max']:.4f}\n")
    print(f"\n得分排名已保存: {csv_path}")


def create_combined_view(sample_dir: str, num_views: int, mask_size: int = 448) -> Image.Image:
    """
    将 num_views 个视角的 原图 | 热力图 | mask 拼成一张大图。
    布局：每行一个视角，3 列（原图、热力图、mask）
    """
    gap = 4
    cols = 3
    rows = num_views
    total_w = cols * mask_size + (cols + 1) * gap
    total_h = rows * mask_size + (rows + 1) * gap

    combined = Image.new("RGB", (total_w, total_h), (40, 40, 40))

    for v in range(num_views):
        orig = Image.open(os.path.join(sample_dir, f"{v}_original.png"))
        heat = Image.open(os.path.join(sample_dir, f"{v}_heatmap.png"))
        mask = Image.open(os.path.join(sample_dir, f"{v}_mask.png")).convert("RGB")

        y = gap + v * (mask_size + gap)
        combined.paste(orig, (gap, y))
        combined.paste(heat, (gap * 2 + mask_size, y))
        combined.paste(mask, (gap * 3 + mask_size * 2, y))

    return combined


# ── Main ─────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="异常可视化：热力图 + 灰度 mask")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--root_dir", type=str, default="data")
    p.add_argument("--test_split", type=str, default="Test_A")
    p.add_argument("--dinov2_weights", type=str, default=None)
    p.add_argument("--image_size", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--output_dir", type=str, default="visualization")
    p.add_argument("--category", type=str, default=None,
                   help="仅处理指定类别（调试用）")
    p.add_argument("--k_neighbors", type=int, default=K_NEIGHBORS)
    p.add_argument("--memory_bank_max", type=int, default=MEMORY_BANK_MAX)
    p.add_argument("--smooth_sigma", type=float, default=2.0)
    p.add_argument("--clip_percentile", type=float, default=30.0)
    p.add_argument("--flow_patch_weight", type=float, default=0.2)
    p.add_argument("--top_n", type=int, default=20,
                   help="输出得分最高的 N 个样本")
    p.add_argument("--heatmap_alpha", type=float, default=0.5,
                   help="热力图叠加透明度 (0=原图, 1=纯热力图)")
    return p.parse_args()


def main():
    args = parse_args()

    # 加载 checkpoint
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

    # 初始化模型
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

    # Phase 1: 构建 Memory Bank
    train_categories = [args.category] if args.category else None
    train_loader = build_dataloader(cfg, split="Train", categories=train_categories)
    memory_bank = build_memory_bank(
        dinov2, train_loader, device, max_features=args.memory_bank_max
    )

    # Phase 2: 推理 + 可视化
    test_loader = build_dataloader(
        cfg, split=args.test_split,
        categories=[args.category] if args.category else None,
    )

    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n[Phase 2] 开始推理 + 可视化 ({args.test_split})...")
    print(f"  输出目录: {output_dir}")
    print(f"  Top-N: {args.top_n}")
    print(f"  热力图透明度: {args.heatmap_alpha}")

    run_visualization(
        dinov2, model, test_loader, memory_bank, device,
        output_dir, args.root_dir, args.test_split,
        k_neighbors=args.k_neighbors,
        smooth_sigma=args.smooth_sigma,
        clip_percentile=args.clip_percentile,
        flow_patch_weight=args.flow_patch_weight,
        top_n=args.top_n,
        heatmap_alpha=args.heatmap_alpha,
    )

    print(f"\n可视化完成！输出目录: {output_dir}")
    print(f"  每个样本目录包含:")
    print(f"    0~4_original.png  原始图像")
    print(f"    0~4_heatmap.png   异常热力图（原图+jet叠加）")
    print(f"    0~4_mask.png      黑白灰度 mask")
    print(f"    combined.png       5视角组合大图")


if __name__ == "__main__":
    main()
