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
MEMORY_BANK_MAX = 5000  # P1-F: 增大 memory bank (2000→5000)
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
    p.add_argument("--no_multiscale", action="store_true",
                   help="禁用多尺度特征融合（仅用最后一层）")
    p.add_argument("--smooth_sigma", type=float, default=4.0,
                   help="异常图高斯平滑 σ")
    p.add_argument("--clip_percentile", type=float, default=0.0,
                   help="得分截断百分位 (已禁用，截断破坏 AP 排序)")
    p.add_argument("--flow_patch_weight", type=float, default=0.0,
                   help="Flow patch 得分融合权重 (0=仅用 k-NN, z 塌缩后默认关闭)")
    p.add_argument("--alpha_flow", type=float, default=0.0,
                   help="图像级融合权重: Flow 信号 (z 塌缩后默认关闭)")
    p.add_argument("--alpha_knn", type=float, default=0.6,
                   help="图像级融合权重: k-NN 信号")
    p.add_argument("--alpha_pixel", type=float, default=0.4,
                   help="图像级融合权重: 像素级 max 信号")
    return p.parse_args()


# ── P1-B: 多尺度特征提取 ────────────────────────────────────────

def get_multiscale_patches(dinov2_feats: dict) -> torch.Tensor:
    """
    拼接 DINOv2 全部 4 层多尺度 patch 特征，融合浅层纹理 + 中层结构 + 深层语义。

    层选择：layers [8, 9, 10, 11]（ViT-B/12 共 12 层，取后 4 层）
    更宽的特征维度（4×768=3072）提供更丰富的异常判据。

    Returns:
        patches: [B, V, N, 4*D]  拼接后的多尺度特征
    """
    ms = dinov2_feats["multi_scale_features"]  # List[[B, V, N, D]]
    # 拼接全部 4 层
    fused = torch.cat(ms, dim=-1)  # [B, V, N, 4D]
    return fused


def local_neighborhood_aggregate(
    patches: torch.Tensor,   # [B, V, N, D]  或 [V, N, D]
    n_h: int,
    n_w: int,
    kernel_size: int = 3,
) -> torch.Tensor:
    """
    局部邻域聚合（PatchCore 风格）：对每个 patch，将其特征与 3×3 邻域平均融合。

    作用：
    - 提供空间上下文，单个 patch 的异常会扩散到邻域
    - 减少“椒盐”噪声式的假阳性
    - 提升 F1max（边界更平滑）

    Returns:
        与输入同形状的特征 tensor
    """
    has_batch = patches.dim() == 4
    if not has_batch:
        patches = patches.unsqueeze(0)  # [1, V, N, D]

    B, V, N, D = patches.shape
    # reshape 为空间 grid: [B*V, D, H, W]
    x = patches.reshape(B * V, n_h, n_w, D).permute(0, 3, 1, 2)  # [B*V, D, H, W]

    pad = kernel_size // 2
    x = F.pad(x, (pad, pad, pad, pad), mode="replicate")
    # 均值卷积
    weight = torch.ones(D, 1, kernel_size, kernel_size, device=x.device, dtype=x.dtype)
    weight = weight / (kernel_size * kernel_size)
    x = F.conv2d(x, weight, groups=D)  # depthwise conv = neighborhood average

    # reshape 回 [B, V, N, D]
    x = x.permute(0, 2, 3, 1).reshape(B, V, N, D)

    if not has_batch:
        x = x.squeeze(0)
    return x


# ── Phase 1: 构建 Memory Bank ─────────────────────────────────────

@torch.no_grad()
def build_memory_bank(
    dinov2: DINOv2Extractor,
    train_loader,
    device: torch.device,
    max_features: int = MEMORY_BANK_MAX,
) -> dict:
    """
    扫描训练集，为每个类别构建多尺度 DINOv2 patch 特征的 memory bank

    Returns:
        dict: {category_name: Tensor[max_features, 2*dinov2_dim]}
    """
    print("\n[Phase 1] 构建 Memory Bank (4层多尺度 + 局部邻域聚合)...")
    bank = {}  # {category: Tensor[≤max_features, 4D]}

    # 逐 batch 扫描，对每个类别增量收集并截断
    for batch_idx, batch in enumerate(tqdm(train_loader, desc="  扫描训练集")):
        views = batch["views"].to(device)
        categories = batch["category"]
        B, V = views.shape[0], views.shape[1]

        dinov2_feats = dinov2.extract_multi_view(views)
        # 4 层多尺度拼接 + 局部邻域聚合
        ms_patches = get_multiscale_patches(dinov2_feats)  # [B, V, N, 4D]
        n_h = dinov2_feats["num_patches_h"]
        n_w = dinov2_feats["num_patches_w"]
        ms_patches = local_neighborhood_aggregate(ms_patches, n_h, n_w)  # [B, V, N, 4D]

        for i in range(B):
            cat = categories[i]
            patches = ms_patches[i].reshape(-1, ms_patches.shape[-1])  # [V*N, 4D]
            sample_k = min(patches.shape[0], max_features // 5)
            if patches.shape[0] > sample_k:
                idx = torch.randperm(patches.shape[0], device=device)[:sample_k]
                patches = patches[idx]
            patches_cpu = patches.cpu()

            if cat not in bank:
                bank[cat] = patches_cpu
            else:
                bank[cat] = torch.cat([bank[cat], patches_cpu], dim=0)
                if bank[cat].shape[0] > max_features:
                    idx = torch.randperm(bank[cat].shape[0])[:max_features]
                    bank[cat] = bank[cat][idx]

        del dinov2_feats, ms_patches

    # L2 归一化并移到 GPU
    for cat in bank:
        bank[cat] = F.normalize(bank[cat], dim=-1).to(device)
        print(f"  {cat}: {bank[cat].shape[0]} 个特征, dim={bank[cat].shape[1]}")

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
def compute_flow_patch_score(
    patch_map: torch.Tensor,   # [V, N, D] Flow 编码后的 patch 特征
) -> torch.Tensor:
    """
    用 Flow 编码特征计算 patch 级异常得分。
    正常 patch 的编码特征应该接近全局均值（因为训练时 Flow 将所有正常特征映射到原点附近）。
    异常 patch 偏离全局均值 → 得分高。

    Returns:
        patch_score: [V, N]
    """
    V, N, D = patch_map.shape
    flat = patch_map.reshape(-1, D)  # [V*N, D]
    center = flat.mean(dim=0, keepdim=True)  # [1, D]
    dist = torch.norm(flat - center, dim=-1)  # [V*N]
    return dist.reshape(V, N)


def percentile_normalize(
    scores: np.ndarray,
    clip_low: float = 0.0,
    clip_high: float = 100.0,
) -> np.ndarray:
    """
    基于百分位的鲁棒归一化到 [0, 1]。
    比 min-max 更稳定：不受极端离群值影响。

    Args:
        scores: 任意形状的 numpy 数组
        clip_low:  下界百分位（低于此值的被截断为 0）
        clip_high: 上界百分位（高于此值的被截断为 1）
    """
    lo = np.percentile(scores, clip_low)
    hi = np.percentile(scores, clip_high)
    rng = hi - lo if hi > lo else 1.0
    normalized = np.clip((scores - lo) / rng, 0.0, 1.0)
    return normalized


@torch.no_grad()
def generate_pixel_mask(
    patch_score: torch.Tensor,   # [V, N]  每个视角的 patch 异常得分
    n_h: int,                    # 水平 patch 数
    n_w: int,                    # 垂直 patch 数
    mask_size: int = MASK_SIZE,
    smooth_sigma: float = 4.0,   # 高斯平滑 σ
    clip_percentile: float = 0.0, # 已禁用：截断破坏 AP 排序信息
) -> np.ndarray:
    """
    将 patch 级异常得分上采样为像素级 mask。

    流程：reshape → 双线性上采样 → 高斯平滑 → 逐视角 min-max 归一化。

    为什么用逐视角 min-max（而非全局/百分位）：
    - AP 是排序指标，任何单调变换都不影响 AP；min-max 是单调的，不破坏排序
    - 百分位截断将低分置零，破坏排序信息，降低 AP
    - 逐视角归一化保持每个视角的动态范围，避免全局归一化压缩对比度

    Returns:
        masks: [V, mask_size, mask_size]  uint8 灰度图
    """
    V = patch_score.shape[0]

    # 1. reshape 为 2D grid
    score_map = patch_score.reshape(V, 1, n_h, n_w)

    # 2. 双线性上采样
    upsampled = F.interpolate(
        score_map,
        size=(mask_size, mask_size),
        mode="bilinear",
        align_corners=False,
    )

    # 3. 高斯平滑
    if smooth_sigma > 0:
        kernel_size = max(5, int(smooth_sigma * 4) | 1)
        if kernel_size % 2 == 0:
            kernel_size += 1
        x = torch.arange(kernel_size, dtype=torch.float32) - kernel_size // 2
        gauss_1d = torch.exp(-0.5 * (x / smooth_sigma) ** 2)
        gauss_1d = gauss_1d / gauss_1d.sum()
        gauss_2d = gauss_1d.unsqueeze(1) @ gauss_1d.unsqueeze(0)
        gauss_2d = gauss_2d.unsqueeze(0).unsqueeze(0).to(upsampled.device)
        pad = kernel_size // 2
        upsampled = F.conv2d(upsampled, gauss_2d, padding=pad)

    # 4. 逐视角 min-max 归一化到 [0, 255]
    masks = upsampled.squeeze(1)  # [V, mask_size, mask_size]
    result = np.zeros((V, mask_size, mask_size), dtype=np.uint8)

    for v in range(V):
        s = masks[v].cpu().float().numpy()
        s_min, s_max = s.min(), s.max()
        rng = s_max - s_min if s_max > s_min else 1.0
        result[v] = np.clip((s - s_min) / rng, 0.0, 1.0).astype(np.float32)
        result[v] = (result[v] * 255).astype(np.uint8)

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
    alpha_flow: float = 0.3,
    alpha_knn: float = 0.4,
    alpha_pixel: float = 0.3,
    use_multiscale: bool = True,   # P1-B: 是否使用多尺度特征
    smooth_sigma: float = 2.0,     # 异常图平滑 σ (降低以保留边界)
    clip_percentile: float = 30.0, # 截断百分位：低于此值的得分置零
    flow_patch_weight: float = 0.2,  # Flow patch 得分融合权重
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
    raw_flow_scores = []   # Flow z 空间 L2 距离
    raw_knn_scores = []    # Memory Bank k-NN 最大距离
    raw_pixel_scores = []  # 像素级最大得分
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

        # ── 2. INP-Former 推理（Flow 图像级得分）──
        out = model(dinov2_feats)

        # ── 3. Flow 图像级得分（z 空间 L2 距离）──
        z_cls = out["z_cls"]          # [B, d_model]
        z_view = out["z_view"]        # [B, V, d_model]
        cls_dist = torch.norm(z_cls, dim=-1)
        view_dist = torch.norm(z_view, dim=-1).mean(dim=-1)
        flow_score = cls_dist + view_dist  # [B]

        # ── 4. 像素级异常得分 + Memory Bank k-NN 图像级得分 ──
        for i in range(B):
            cat = categories[i]
            sid = sample_ids[i]
            group_folder = f"{cat}/{sid}"

            # 像素级 k-NN 异常图
            if cat in memory_bank:
                if use_multiscale:
                    # 4 层多尺度拼接 + 局部邻域聚合
                    test_patches = get_multiscale_patches(dinov2_feats)[i]  # [V, N, 4D]
                    test_patches = local_neighborhood_aggregate(
                        test_patches, n_h, n_w
                    )  # [V, N, 4D]
                else:
                    test_patches = patch_features[i]  # [V, N, D]

                # k-NN patch 得分
                knn_patch = compute_patch_anomaly_score(
                    test_patches, memory_bank[cat], k=k_neighbors
                )  # [V, N]

                # Flow patch 得分：patch_map 编码特征与全局均值的距离
                if flow_patch_weight > 0:
                    patch_map_i = out["patch_map"][i]  # [V, N, d_model]
                    flow_patch = compute_flow_patch_score(patch_map_i)  # [V, N]
                    # 融合：k-NN 为主，Flow 为辅
                    w_flow = flow_patch_weight
                    w_knn = 1.0 - w_flow
                    # 各自归一化后融合
                    knn_norm = percentile_normalize(knn_patch.cpu().numpy())
                    flow_norm = percentile_normalize(flow_patch.cpu().numpy())
                    ps_np = w_knn * knn_norm + w_flow * flow_norm
                    ps = torch.from_numpy(ps_np).to(knn_patch.device)
                else:
                    ps = knn_patch

                # 图像级 k-NN 信号：每个视角 top-10% patch 的平均距离
                V_s, N_s = ps.shape
                k_top = max(1, N_s // 10)
                topk_vals, _ = ps.topk(k_top, dim=-1)  # [V, k_top]
                knn_score = topk_vals.mean().item()  # 标量
                pixel_max = ps.max().item()
            else:
                if cat not in missing_cats:
                    print(f"  [警告] 类别 '{cat}' 无 memory bank，使用回退方案")
                    missing_cats.add(cat)
                encoded = out["patch_map"]  # [B, V, N, d_model]
                cls_feat = out["z_cls"][i]  # [d_model]
                patch_feats = encoded[i]    # [V, N, d_model]
                cos_sim = F.cosine_similarity(
                    patch_feats, cls_feat.unsqueeze(0).unsqueeze(0).expand_as(patch_feats), dim=-1
                )
                ps = 1.0 - cos_sim  # [V, N]
                knn_score = 0.0
                pixel_max = ps.max().item()

            # 收集原始得分
            raw_flow_scores.append(flow_score[i].item())
            raw_knn_scores.append(knn_score)
            raw_pixel_scores.append(pixel_max)

            csv_rows.append({
                "group_folder": group_folder,
            })

            # 生成并保存像素 mask
            sample_mask_dir = mask_dir / cat / sid
            sample_mask_dir.mkdir(parents=True, exist_ok=True)
            masks = generate_pixel_mask(
                ps, n_h, n_w,
                smooth_sigma=smooth_sigma,
                clip_percentile=clip_percentile,
            )
            for v in range(V):
                mask_path = sample_mask_dir / f"{v}_mask.png"
                Image.fromarray(masks[v], mode="L").save(str(mask_path))

        if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == total:
            print(f"  推理进度: {batch_idx + 1}/{total}")

    # ── 5. 三路融合 + Min-Max 归一化 ──
    # 对每种信号分别做 min-max 归一化到 [0, 1]
    def minmax_normalize(scores):
        s_min, s_max = min(scores), max(scores)
        rng = s_max - s_min if s_max > s_min else 1.0
        return [(s - s_min) / rng for s in scores], s_min, s_max

    flow_norm, f_min, f_max = minmax_normalize(raw_flow_scores)
    knn_norm, k_min, k_max = minmax_normalize(raw_knn_scores)
    pixel_norm, p_min, p_max = minmax_normalize(raw_pixel_scores)

    # 三路融合
    for idx, row in enumerate(csv_rows):
        fused = (
            alpha_flow * flow_norm[idx]
            + alpha_knn * knn_norm[idx]
            + alpha_pixel * pixel_norm[idx]
        )
        row["anomaly_score"] = f"{fused:.6f}"

    print(f"  融合归一化:")
    print(f"    Flow:  raw [{f_min:.2f}, {f_max:.2f}]")
    print(f"    k-NN:  raw [{k_min:.4f}, {k_max:.4f}]")
    print(f"    Pixel: raw [{p_min:.4f}, {p_max:.4f}]")
    print(f"    权重: Flow={alpha_flow}, k-NN={alpha_knn}, Pixel={alpha_pixel}")

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
        str(output_dir), k_neighbors=args.k_neighbors,
        alpha_flow=args.alpha_flow,
        alpha_knn=args.alpha_knn,
        alpha_pixel=args.alpha_pixel,
        use_multiscale=not args.no_multiscale,
        smooth_sigma=args.smooth_sigma,
        clip_percentile=args.clip_percentile,
        flow_patch_weight=args.flow_patch_weight,
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
