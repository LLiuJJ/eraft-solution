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
from typing import Optional

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
from src.models.feature_adapter import FeatureAdapter


# ── 常量 ─────────────────────────────────────────────────────────
MASK_SIZE = 448  # 比赛要求的 mask 尺寸
MEMORY_BANK_MAX = 3000  # CoreSet 采样后每类最大特征数
K_NEIGHBORS = 3  # k-NN 近邻数（PatchCore 默认）
PCA_DIM = 1024  # PCA 降维目标维度
CORESET_RATIO = 0.1  # CoreSet 保留比例（10%）


def parse_args():
    p = argparse.ArgumentParser(description="生成比赛提交包")
    p.add_argument("--checkpoint", type=str, default=None,
                   help="INP-Former checkpoint (可选，不指定则纯 k-NN)")
    p.add_argument("--adapter_checkpoint", type=str, default=None,
                   help="Feature Adapter checkpoint 路径 (推荐)")
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


# ── PatchCore 风格特征处理 ─────────────────────────────────────

class PatchCoreFeatures:
    """
    PatchCore 风格的特征处理：多尺度拼接 + PCA 降维 + 局部空间聚合。

    与之前方案的区别：
    - 4 层全部拼接（3072 维） → PCA 降到 1024 维（去除冗余，避免维度灾难）
    - 局部 3×3 空间聚合后再存入 memory bank
    - CoreSet 贪心采样替代随机采样
    - L2 距离替代余弦距离（PatchCore 标准做法）
    """

    def __init__(self, pca_dim: int = PCA_DIM):
        self.pca_dim = pca_dim
        self.mean = None          # PCA 均值
        self.components = None    # PCA 主成分 [D, pca_dim]

    def fit_pca(self, features: torch.Tensor) -> None:
        """
        在 memory bank 特征上拟合 PCA（使用 SVD）。
        features: [M, D]
        """
        M, D = features.shape
        # 减去均值
        self.mean = features.mean(dim=0)  # [D]
        X = features - self.mean  # [M, D]
        # SVD: X = U * S * V^T，取前 pca_dim 个主成分
        U, S, V = torch.svd(X.T)  # V: [D, min(M,D)]
        self.components = V[:, :self.pca_dim].contiguous()  # [D, pca_dim]
        print(f"  PCA: {D} → {self.pca_dim} 维, 解释方差: {(S[:self.pca_dim]**2).sum() / (S**2).sum():.2%}")

    def transform(self, features: torch.Tensor) -> torch.Tensor:
        """PCA 降维: [*, D] → [*, pca_dim]"""
        shape = features.shape
        flat = features.reshape(-1, shape[-1])
        projected = (flat - self.mean) @ self.components  # [*, pca_dim]
        # L2 归一化
        projected = F.normalize(projected, dim=-1)
        return projected.reshape(*shape[:-1], self.pca_dim)

    def fit_transform(self, features: torch.Tensor) -> torch.Tensor:
        """拟合并变换"""
        self.fit_pca(features)
        return self.transform(features)


def get_multiscale_patches(dinov2_feats: dict) -> torch.Tensor:
    """
    拼接 DINOv2 全部 4 层 patch 特征（3072 维），后续用 PCA 降维。
    """
    ms = dinov2_feats["multi_scale_features"]  # List[[B, V, N, D]]
    fused = torch.cat(ms, dim=-1)  # [B, V, N, 4D=3072]
    return fused


def local_neighborhood_aggregate(
    patches: torch.Tensor,   # [B, V, N, D]  或 [V, N, D]
    n_h: int,
    n_w: int,
    kernel_size: int = 3,
) -> torch.Tensor:
    """
    局部 3×3 空间均值聚合：每个 patch 特征融合其空间邻域。
    在 PCA 降维后执行（维度已降，depthwise conv 开销小）。
    """
    has_batch = patches.dim() == 4
    if not has_batch:
        patches = patches.unsqueeze(0)

    B, V, N, D = patches.shape
    x = patches.reshape(B * V, n_h, n_w, D).permute(0, 3, 1, 2)  # [B*V, D, H, W]

    pad = kernel_size // 2
    x = F.pad(x, (pad, pad, pad, pad), mode="replicate")
    weight = torch.ones(D, 1, kernel_size, kernel_size, device=x.device, dtype=x.dtype)
    weight = weight / (kernel_size * kernel_size)
    x = F.conv2d(x, weight, groups=D)

    x = x.permute(0, 2, 3, 1).reshape(B, V, N, D)
    if not has_batch:
        x = x.squeeze(0)
    return x


def coreset_sample(features: torch.Tensor, num_select: int) -> torch.Tensor:
    """
    CoreSet 贪心采样（PatchCore 的 greedy coreset）。

    贪心 farthest point sampling：每步选离已选集合最远的点。
    比随机采样更好地覆盖特征空间多样性。

    Args:
        features: [M, D] 候选特征
        num_select: 选取数量

    Returns:
        selected: [num_select, D]
    """
    M, D = features.shape
    num_select = min(num_select, M)

    if num_select >= M:
        return features

    # 初始化：选第一个点（距均值最远）
    center = features.mean(dim=0, keepdim=True)  # [1, D]
    dists = torch.cdist(features.unsqueeze(0), center.unsqueeze(0)).squeeze()  # [M]
    selected_idx = [dists.argmax().item()]

    # 贪心选择
    min_dists = torch.full((M,), float('inf'), device=features.device)
    for _ in range(num_select - 1):
        # 更新每个点到最近已选点的距离
        last = features[selected_idx[-1]]  # [D]
        d = torch.norm(features - last, dim=-1)  # [M]
        min_dists = torch.min(min_dists, d)
        # 选距离最大的点
        next_idx = min_dists.argmax().item()
        selected_idx.append(next_idx)

    return features[selected_idx]


# ── Phase 1: 构建 Memory Bank (PatchCore 风格) ─────────────────

@torch.no_grad()
def build_memory_bank(
    dinov2: DINOv2Extractor,
    train_loader,
    device: torch.device,
    max_features: int = MEMORY_BANK_MAX,
    coreset_ratio: float = CORESET_RATIO,
    adapter: Optional[FeatureAdapter] = None,
) -> dict:
    """
    PatchCore 风格 Memory Bank 构建流程：
    1. 收集训练集所有 patch 特征（4层拼接 3072 维）
    2. 降维：adapter (推荐) 或 PCA (3072 → 1024 维)
    3. 局部 3×3 空间聚合
    4. CoreSet 贪心采样

    Returns:
        dict: {category_name: {"features": Tensor, "adapter"/"pca": ..., "n_h": int, "n_w": int}}
    """
    mode = "adapter" if adapter is not None else "PCA"
    print(f"\n[Phase 1] 构建 Memory Bank (PatchCore: 4层+{mode}{PCA_DIM}+3×3聚合+CoreSet{coreset_ratio:.0%})")

    # Step 1: 收集所有原始多尺度特征
    raw_bank = {}  # {category: Tensor[≤collect_max, 4D]}
    collect_max = 20000  # 每类最多收集 20000 个候选
    n_h, n_w = None, None

    for batch_idx, batch in enumerate(tqdm(train_loader, desc="  [1/4] 收集特征")):
        views = batch["views"].to(device)
        categories = batch["category"]
        B, V = views.shape[0], views.shape[1]

        dinov2_feats = dinov2.extract_multi_view(views)
        if n_h is None:
            n_h = dinov2_feats["num_patches_h"]
            n_w = dinov2_feats["num_patches_w"]

        ms_patches = get_multiscale_patches(dinov2_feats)  # [B, V, N, 4D]

        for i in range(B):
            cat = categories[i]
            patches = ms_patches[i].reshape(-1, ms_patches.shape[-1]).cpu()  # [V*N, 4D]
            if cat not in raw_bank:
                raw_bank[cat] = patches
            else:
                raw_bank[cat] = torch.cat([raw_bank[cat], patches], dim=0)
                if raw_bank[cat].shape[0] > collect_max:
                    idx = torch.randperm(raw_bank[cat].shape[0])[:collect_max]
                    raw_bank[cat] = raw_bank[cat][idx]

        del dinov2_feats, ms_patches

    # Step 2-4: 按类别处理（降维 → 空间聚合 → CoreSet）
    final_bank = {}  # {category: {"features": Tensor, "adapter"/"pca": ..., "n_h", "n_w"}}

    for cat, features in raw_bank.items():
        print(f"\n  [{cat}] 候选: {features.shape[0]} × {features.shape[1]}")

        # Step 2: 降维 (adapter 或 PCA)
        if adapter is not None:
            # 用训练好的 adapter 降维
            reduced = adapter(features.to(device))  # [M, 1024]
            reduced = F.normalize(reduced, dim=-1)
            entry_key = "adapter"
            entry_val = adapter
        else:
            # 拟合 PCA 降维
            pca = PatchCoreFeatures(pca_dim=PCA_DIM)
            reduced = pca.fit_transform(features.to(device))  # [M, PCA_DIM]
            entry_key = "pca"
            entry_val = pca

        # Step 3: 局部空间聚合
        # reshape 为 [1, 1, N, PCA_DIM] 以应用空间聚合
        N = n_h * n_w
        num_views = features.shape[0] // N  # 训练样本数（近似）
        if num_views > 0:
            # 按样本分批做空间聚合
            batch_size = 100  # 避免 OOM
            aggregated = []
            for start in range(0, reduced.shape[0], batch_size):
                end = min(start + batch_size, reduced.shape[0])
                chunk = reduced[start:end]  # [k, PCA_DIM]
                # reshape: [k, N, PCA_DIM] → 但 k 可能不是 N 的整数倍
                # 简化：对每个样本的 patch 做空间聚合
                n_samples = chunk.shape[0] // N
                if n_samples == 0:
                    aggregated.append(chunk)  # 不够一个样本，直接保留
                    continue
                usable = chunk[:n_samples * N]  # [n_samples * N, PCA_DIM]
                usable = usable.reshape(n_samples, 1, N, PCA_DIM)  # [n, 1, N, D]
                usable = local_neighborhood_aggregate(usable, n_h, n_w, kernel_size=3)
                usable = F.normalize(usable.reshape(-1, PCA_DIM), dim=-1)  # 重新归一化
                aggregated.append(usable)
                # 处理剩余不足一个样本的部分
                remainder = chunk[n_samples * N:]
                if remainder.shape[0] > 0:
                    aggregated.append(remainder)
            reduced = torch.cat(aggregated, dim=0)

        # Step 4: CoreSet 采样
        num_select = max(100, int(reduced.shape[0] * coreset_ratio))
        num_select = min(num_select, max_features, reduced.shape[0])
        coreset = coreset_sample(reduced, num_select)  # [num_select, PCA_DIM]

        final_bank[cat] = {
            "features": coreset.to(device),
            entry_key: entry_val,
            "n_h": n_h,
            "n_w": n_w,
        }
        print(f"  [{cat}] CoreSet: {coreset.shape[0]} 个特征 ({mode})")

    print(f"\n[Phase 1] 完成，共 {len(final_bank)} 个类别\n")
    return final_bank


# ── 像素级异常热力图生成 ────────────────────────────────────────

@torch.no_grad()
def compute_patch_anomaly_score(
    test_patches: torch.Tensor,       # [V, N, D] 测试样本 patch 特征（原始多尺度）
    bank_entry: dict,                 # {"features": Tensor, "adapter"/"pca": ..., "n_h", "n_w"}
    k: int = K_NEIGHBORS,
) -> torch.Tensor:
    """
    PatchCore 风格 k-NN 异常得分计算：
    1. 用 adapter (推荐) 或 PCA 降维测试特征
    2. 局部 3×3 空间聚合
    3. L2 距离计算 k-NN

    Returns:
        patch_score: [V, N] 每个 patch 的异常得分
    """
    V, N, D_raw = test_patches.shape
    bank_feats = bank_entry["features"]  # [M, out_dim]
    n_h = bank_entry["n_h"]
    n_w = bank_entry["n_w"]
    M = bank_feats.shape[0]
    out_dim = bank_feats.shape[1]

    # Step 1: 降维测试特征 (adapter 或 PCA)
    if "adapter" in bank_entry:
        adapter = bank_entry["adapter"]
        test_reduced = adapter(test_patches.reshape(-1, D_raw))  # [V*N, 1024]
        test_reduced = F.normalize(test_reduced, dim=-1)
    else:
        pca = bank_entry["pca"]
        test_reduced = pca.transform(test_patches.reshape(-1, D_raw))  # [V*N, PCA_DIM]

    test_reduced = test_reduced.reshape(V, N, -1)  # [V, N, out_dim]

    # Step 2: 局部 3×3 空间聚合
    test_reduced = local_neighborhood_aggregate(test_reduced, n_h, n_w, kernel_size=3)  # [V, N, out_dim]
    test_reduced = F.normalize(test_reduced.reshape(-1, out_dim), dim=-1)  # 重新归一化

    # Step 3: L2 距离 k-NN（PatchCore 标准）
    # cdist: [V*N, M] L2 距离
    dists = torch.cdist(test_reduced.unsqueeze(0), bank_feats.unsqueeze(0)).squeeze(0)  # [V*N, M]

    k = min(k, M)
    topk_dist, _ = dists.topk(k, dim=-1, largest=False)  # [V*N, k] 最近 k 个距离
    # 异常得分 = k 个最近邻的平均距离
    avg_dist = topk_dist.mean(dim=-1)  # [V*N]

    return avg_dist.reshape(V, N)


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
        normalized = np.clip((s - s_min) / rng, 0.0, 1.0)  # float [0,1]
        result[v] = (normalized * 255).astype(np.uint8)   # uint8 [0,255]

    return result


# ── Phase 2: 推理主函数 ──────────────────────────────────────────

@torch.no_grad()
def run_inference(
    dinov2: DINOv2Extractor,
    model: Optional[INPFormer],  # None 时纯 k-NN，跳过 INP-Former
    dataloader,
    memory_bank: dict,
    device: torch.device,
    output_dir: str,
    k_neighbors: int = K_NEIGHBORS,
    alpha_flow: float = 0.0,
    alpha_knn: float = 0.6,
    alpha_pixel: float = 0.4,
    use_multiscale: bool = True,
    smooth_sigma: float = 4.0,
    clip_percentile: float = 0.0,
    flow_patch_weight: float = 0.0,
) -> list:
    """
    在测试集上推理，生成 submission.csv 和 predicted_masks/

    Returns:
        List[dict]: 所有样本的结果
    """
    if model is not None:
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

        # ── 2. INP-Former 推理（Flow 图像级得分，可选）──
        if model is not None:
            out = model(dinov2_feats)
            z_cls = out["z_cls"]
            z_view = out["z_view"]
            cls_dist = torch.norm(z_cls, dim=-1)
            view_dist = torch.norm(z_view, dim=-1).mean(dim=-1)
            flow_score = cls_dist + view_dist  # [B]
        else:
            # 纯 k-NN 模式，无 Flow 信号
            flow_score = torch.zeros(B, device=device)

        # ── 4. 像素级异常得分 + Memory Bank k-NN 图像级得分 ──
        for i in range(B):
            cat = categories[i]
            sid = sample_ids[i]
            group_folder = f"{cat}/{sid}"

            # 像素级 k-NN 异常图
            if cat in memory_bank:
                # 始终用 4 层多尺度拼接（与 memory bank 训练时的维度一致）
                ms = dinov2_feats["multi_scale_features"]
                test_patches = torch.cat(ms, dim=-1)[i]  # [V, N, 4D=3072]

                # k-NN patch 得分（PatchCore 风格：PCA + 空间聚合 + L2 距离）
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
                # 回退方案: 用 DINOv2 patch 特征的方差作为异常图
                ms = dinov2_feats["multi_scale_features"]
                test_patches_raw = torch.cat(ms, dim=-1)[i]  # [V, N, 4D]
                # 正常 patch 特征方差小，异常 patch 方差大
                ps = test_patches_raw.var(dim=-1)  # [V, N]
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

    # 至少需要一个 checkpoint
    if not args.checkpoint and not args.adapter_checkpoint:
        print("[错误] 必须指定 --checkpoint 或 --adapter_checkpoint 至少一个")
        sys.exit(1)

    # ── 配置 ──
    # 加载 INP-Former checkpoint (可选)
    if args.checkpoint:
        print(f"加载 INP-Former checkpoint: {args.checkpoint}")
        ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        cfg = ckpt.get("config", get_config())
    else:
        print("无 INP-Former checkpoint，纯 k-NN 模式")
        cfg = get_config()
        ckpt = None

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

    # ── DINOv2 ──
    dinov2 = DINOv2Extractor(
        model_name=cfg.dinov2.model_name,
        out_indices=cfg.dinov2.out_indices,
        patch_size=cfg.dinov2.patch_size,
        frozen=True,
        weights_path=cfg.dinov2.weights_path,
    ).to(device)

    # ── INP-Former (可选) ──
    model = None
    if ckpt is not None:
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
        print(f"已加载 INP-Former, epoch={ckpt.get('epoch', 'N/A')}")

    # ── Feature Adapter (推荐) ──
    adapter = None
    if args.adapter_checkpoint:
        print(f"加载 Feature Adapter: {args.adapter_checkpoint}")
        adapter_ckpt = torch.load(args.adapter_checkpoint, map_location=device, weights_only=False)
        adapter_cfg = adapter_ckpt.get("config", {})
        adapter = FeatureAdapter(
            input_dim=adapter_cfg.get("input_dim", 4 * cfg.dinov2.embed_dim),
            output_dim=adapter_cfg.get("output_dim", 1024),
            hidden_dim=adapter_cfg.get("hidden_dim", 2048),
            dropout=adapter_cfg.get("dropout", 0.1),
        ).to(device)
        adapter.load_state_dict(adapter_ckpt["adapter_state"])
        adapter.eval()
        print(f"已加载 Feature Adapter (loss={adapter_ckpt.get('loss', 'N/A'):.4f})")

    # ── Phase 1: 构建 Memory Bank ──
    train_categories = [args.category] if args.category else None
    train_loader = build_dataloader(cfg, split="Train", categories=train_categories)
    memory_bank = build_memory_bank(
        dinov2, train_loader, device,
        max_features=args.memory_bank_max,
        adapter=adapter,
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
