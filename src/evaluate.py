"""
本地评估脚本：计算 图像级AUROC、图像级AP、像素级AUROC、像素级AP、像素级F1max

输入：
  1. 提交结果目录 (submission.csv + predicted_masks/)
  2. Ground Truth 目录 (image_labels.csv + gt_masks/)

Ground Truth 目录结构：
  gt_dir/
    image_labels.csv        # 两列: group_folder, label (0=正常, 1=异常)
    gt_masks/               # 像素级 GT mask (448×448 灰度图, 255=异常区域)
      类别名/
        Sxxxx/
          0_mask.png        # 视角 0 的 GT mask
          1_mask.png
          ...

用法：
    # 完整评估
    uv run python -m src.evaluate \
        --submission_dir submission \
        --gt_dir gt

    # 只评估图像级指标 (无需 GT masks)
    uv run python -m src.evaluate \
        --submission_dir submission \
        --gt_dir gt \
        --image_only

    # 按类别查看结果
    uv run python -m src.evaluate \
        --submission_dir submission \
        --gt_dir gt \
        --per_category
"""
import argparse
import csv
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def parse_args():
    p = argparse.ArgumentParser(description="评估提交结果的各项指标")
    p.add_argument(
        "--submission_dir",
        type=str,
        required=True,
        help="提交结果目录 (含 submission.csv + predicted_masks/)",
    )
    p.add_argument(
        "--gt_dir",
        type=str,
        required=True,
        help="Ground Truth 目录 (含 image_labels.csv + gt_masks/)",
    )
    p.add_argument(
        "--image_only",
        action="store_true",
        help="只评估图像级指标 (不需要 GT masks)",
    )
    p.add_argument(
        "--per_category",
        action="store_true",
        help="输出每个类别的详细指标",
    )
    p.add_argument(
        "--mask_size",
        type=int,
        default=448,
        help="mask 尺寸 (默认 448)",
    )
    return p.parse_args()


# ─────────────────────────────────────────────────────────
# 数据加载
# ─────────────────────────────────────────────────────────


def load_submission_csv(path: str) -> dict:
    """加载 submission.csv，返回 {group_folder: anomaly_score}"""
    scores = {}
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            gf = row["group_folder"].strip()
            score = float(row["anomaly_score"])
            scores[gf] = score
    return scores


def load_gt_labels(path: str) -> dict:
    """加载 image_labels.csv，返回 {group_folder: label (0/1)}"""
    labels = {}
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            gf = row["group_folder"].strip()
            label = int(row["label"])
            labels[gf] = label
    return labels


def load_predicted_mask(mask_path: str, target_size: int) -> np.ndarray:
    """加载预测 mask，返回 [H, W] 归一化到 [0, 1] 的浮点数组"""
    img = Image.open(mask_path).convert("L")
    if img.size != (target_size, target_size):
        img = img.resize((target_size, target_size), Image.BILINEAR)
    return np.array(img, dtype=np.float32) / 255.0


def load_gt_mask(mask_path: str, target_size: int) -> np.ndarray:
    """加载 GT mask，返回 [H, W] 二值数组 (0=正常, 1=异常)"""
    img = Image.open(mask_path).convert("L")
    if img.size != (target_size, target_size):
        img = img.resize((target_size, target_size), Image.NEAREST)
    arr = np.array(img, dtype=np.float32)
    return (arr > 127).astype(np.float32)


# ─────────────────────────────────────────────────────────
# 纯 numpy 指标计算（不依赖 sklearn）
# ─────────────────────────────────────────────────────────


def _roc_auc_score(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """用 trapezoidal rule 计算 AUROC"""
    # 按分数降序排序
    desc_idx = np.argsort(-y_score)
    y_sorted = y_true[desc_idx]

    # 计算 TPR 和 FPR
    tps = np.cumsum(y_sorted)
    fps = np.cumsum(1 - y_sorted)
    total_pos = tps[-1]
    total_neg = fps[-1]

    if total_pos == 0 or total_neg == 0:
        return float("nan")

    tpr = tps / total_pos
    fpr = fps / total_neg

    # 加起点 (0, 0)
    tpr = np.concatenate([[0], tpr])
    fpr = np.concatenate([[0], fpr])

    # trapezoidal rule (numpy 2.x: trapezoid, 1.x: trapz)
    try:
        return float(np.trapezoid(tpr, fpr))
    except AttributeError:
        return float(np.trapz(tpr, fpr))


def _average_precision_score(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """计算 Average Precision (AP)"""
    desc_idx = np.argsort(-y_score)
    y_sorted = y_true[desc_idx]

    tps = np.cumsum(y_sorted)
    total = np.arange(1, len(y_sorted) + 1)
    precision = tps / total
    recall = tps / y_true.sum()

    # AP = sum of (recall_change * precision) at each positive
    ap = float(np.sum((y_sorted == 1) * precision) / max(1, y_true.sum()))
    return ap


def _f1_max(y_true: np.ndarray, y_score: np.ndarray, n_thresholds: int = 200) -> float:
    """搜索最优阈值下的最大 F1 分数"""
    thresholds = np.linspace(0.001, 0.999, n_thresholds)
    f1_best = 0.0
    for t in thresholds:
        pred = (y_score >= t).astype(np.float32)
        tp = np.sum(pred * y_true)
        fp = np.sum(pred * (1 - y_true))
        fn = np.sum((1 - pred) * y_true)
        if tp == 0:
            continue
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = 2 * precision * recall / (precision + recall)
        if f1 > f1_best:
            f1_best = f1
    return float(f1_best)


def compute_image_level_metrics(scores: list, labels: list) -> dict:
    """计算图像级 AUROC 和 AP"""
    scores = np.array(scores, dtype=np.float64)
    labels = np.array(labels, dtype=np.int32)

    if len(np.unique(labels)) < 2:
        return {"AUROC": float("nan"), "AP": float("nan")}

    auroc = _roc_auc_score(labels, scores)
    ap = _average_precision_score(labels, scores)
    return {"AUROC": auroc, "AP": ap}


def compute_pixel_level_metrics(
    pred_masks: list, gt_masks: list, n_thresholds: int = 200
) -> dict:
    """
    计算像素级 AUROC、AP、F1max
    pred_masks: list of [H, W] float arrays in [0, 1]
    gt_masks:   list of [H, W] binary arrays {0, 1}
    """
    all_pred = np.concatenate([p.flatten() for p in pred_masks])
    all_gt = np.concatenate([g.flatten() for g in gt_masks])

    if len(np.unique(all_gt)) < 2:
        return {"AUROC": float("nan"), "AP": float("nan"), "F1max": float("nan")}

    # 采样以加速（最多 500K 像素）
    max_pixels = 500_000
    if len(all_gt) > max_pixels:
        idx = np.random.default_rng(42).choice(len(all_gt), max_pixels, replace=False)
        sample_pred = all_pred[idx]
        sample_gt = all_gt[idx]
    else:
        sample_pred = all_pred
        sample_gt = all_gt

    auroc = _roc_auc_score(sample_gt, sample_pred)
    ap = _average_precision_score(sample_gt, sample_pred)
    f1m = _f1_max(sample_gt, sample_pred, n_thresholds)

    return {"AUROC": auroc, "AP": ap, "F1max": f1m}


# ─────────────────────────────────────────────────────────
# 主流程
# ─────────────────────────────────────────────────────────


def main():
    args = parse_args()

    sub_dir = Path(args.submission_dir)
    gt_dir = Path(args.gt_dir)
    mask_size = args.mask_size

    # ── 加载预测结果 ──
    csv_path = sub_dir / "submission.csv"
    if not csv_path.exists():
        print(f"错误: 找不到 {csv_path}")
        sys.exit(1)

    pred_scores = load_submission_csv(str(csv_path))
    print(f"加载预测: {len(pred_scores)} 个样本")

    # ── 加载 GT 标签 ──
    gt_labels_path = gt_dir / "image_labels.csv"
    if not gt_labels_path.exists():
        print(f"错误: 找不到 {gt_labels_path}")
        print("GT 目录需包含 image_labels.csv (两列: group_folder, label)")
        sys.exit(1)

    gt_labels = load_gt_labels(str(gt_labels_path))
    print(f"加载 GT: {len(gt_labels)} 个标签")

    # ── 匹配样本 ──
    common_keys = sorted(set(pred_scores.keys()) & set(gt_labels.keys()))
    if not common_keys:
        print("错误: 预测结果和 GT 没有匹配的样本")
        sys.exit(1)
    print(f"匹配样本: {len(common_keys)} 个")

    # 按类别分组
    cat_data = defaultdict(lambda: {"scores": [], "labels": [], "keys": []})
    for key in common_keys:
        cat = key.split("/")[0]  # e.g. "3_adapter/S0001" → "3_adapter"
        cat_data[cat]["scores"].append(pred_scores[key])
        cat_data[cat]["labels"].append(gt_labels[key])
        cat_data[cat]["keys"].append(key)

    categories = sorted(cat_data.keys())
    print(f"类别数: {len(categories)}")

    # ══════════════════════════════════════════════════════
    # 图像级评估
    # ══════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print("图像级评估 (Image-Level)")
    print(f"{'='*60}")

    cat_image_metrics = {}
    all_scores, all_labels = [], []

    for cat in categories:
        scores = cat_data[cat]["scores"]
        labels = cat_data[cat]["labels"]
        all_scores.extend(scores)
        all_labels.extend(labels)

        metrics = compute_image_level_metrics(scores, labels)
        cat_image_metrics[cat] = metrics

        if args.per_category:
            n_pos = sum(labels)
            n_neg = len(labels) - n_pos
            print(
                f"  {cat:35s}  AUROC={metrics['AUROC']:.4f}  "
                f"AP={metrics['AP']:.4f}  "
                f"(正常={n_neg}, 异常={n_pos})"
            )

    # 宏平均
    aurocs = [m["AUROC"] for m in cat_image_metrics.values() if not np.isnan(m["AUROC"])]
    aps = [m["AP"] for m in cat_image_metrics.values() if not np.isnan(m["AP"])]

    macro_auroc = np.mean(aurocs) if aurocs else float("nan")
    macro_ap = np.mean(aps) if aps else float("nan")

    # 全局指标 (所有样本混合计算)
    global_image = compute_image_level_metrics(all_scores, all_labels)

    print(f"\n  {'指标':<20s} {'宏平均':>10s}  {'全局':>10s}")
    print(f"  {'-'*44}")
    print(f"  {'AUROC':<20s} {macro_auroc:>10.4f}  {global_image['AUROC']:>10.4f}")
    print(f"  {'AP':<20s} {macro_ap:>10.4f}  {global_image['AP']:>10.4f}")

    # ══════════════════════════════════════════════════════
    # 像素级评估
    # ══════════════════════════════════════════════════════
    if not args.image_only:
        print(f"\n{'='*60}")
        print("像素级评估 (Pixel-Level)")
        print(f"{'='*60}")

        gt_masks_dir = gt_dir / "gt_masks"
        pred_masks_dir = sub_dir / "predicted_masks"

        if not gt_masks_dir.exists():
            print(f"警告: 找不到 GT masks 目录 {gt_masks_dir}")
            print("跳过像素级评估")
        elif not pred_masks_dir.exists():
            print(f"警告: 找不到预测 masks 目录 {pred_masks_dir}")
            print("跳过像素级评估")
        else:
            cat_pixel_metrics = {}
            all_pred_masks, all_gt_masks = [], []

            for cat in categories:
                cat_pred_masks = []
                cat_gt_masks = []

                for key in cat_data[cat]["keys"]:
                    # 只处理异常样本 (正常样本无 GT mask)
                    if gt_labels[key] == 0:
                        continue

                    # 加载每个视角的 mask
                    sample_dir = key  # e.g. "3_adapter/S0001"
                    for view_idx in range(5):
                        pred_path = (
                            pred_masks_dir / sample_dir / f"{view_idx}_mask.png"
                        )
                        gt_path = gt_masks_dir / sample_dir / f"{view_idx}_mask.png"

                        if not pred_path.exists():
                            # 缺失预测 mask → 全黑 (正常)
                            pred_mask = np.zeros((mask_size, mask_size), dtype=np.float32)
                        else:
                            pred_mask = load_predicted_mask(str(pred_path), mask_size)

                        if not gt_path.exists():
                            # 缺失 GT mask → 全黑 (无异常)
                            gt_mask = np.zeros((mask_size, mask_size), dtype=np.float32)
                        else:
                            gt_mask = load_gt_mask(str(gt_path), mask_size)

                        cat_pred_masks.append(pred_mask)
                        cat_gt_masks.append(gt_mask)

                if cat_pred_masks:
                    metrics = compute_pixel_level_metrics(cat_pred_masks, cat_gt_masks)
                    cat_pixel_metrics[cat] = metrics
                    all_pred_masks.extend(cat_pred_masks)
                    all_gt_masks.extend(cat_gt_masks)

                    if args.per_category:
                        print(
                            f"  {cat:35s}  AUROC={metrics['AUROC']:.4f}  "
                            f"AP={metrics['AP']:.4f}  "
                            f"F1max={metrics['F1max']:.4f}  "
                            f"(masks={len(cat_pred_masks)})"
                        )
                else:
                    cat_pixel_metrics[cat] = {
                        "AUROC": float("nan"),
                        "AP": float("nan"),
                        "F1max": float("nan"),
                    }

            # 宏平均
            p_aurocs = [
                m["AUROC"]
                for m in cat_pixel_metrics.values()
                if not np.isnan(m["AUROC"])
            ]
            p_aps = [
                m["AP"]
                for m in cat_pixel_metrics.values()
                if not np.isnan(m["AP"])
            ]
            p_f1s = [
                m["F1max"]
                for m in cat_pixel_metrics.values()
                if not np.isnan(m["F1max"])
            ]

            macro_p_auroc = np.mean(p_aurocs) if p_aurocs else float("nan")
            macro_p_ap = np.mean(p_aps) if p_aps else float("nan")
            macro_p_f1 = np.mean(p_f1s) if p_f1s else float("nan")

            # 全局像素级指标
            if all_pred_masks:
                global_pixel = compute_pixel_level_metrics(all_pred_masks, all_gt_masks)
            else:
                global_pixel = {"AUROC": float("nan"), "AP": float("nan"), "F1max": float("nan")}

            print(f"\n  {'指标':<20s} {'宏平均':>10s}  {'全局':>10s}")
            print(f"  {'-'*44}")
            print(f"  {'AUROC':<20s} {macro_p_auroc:>10.4f}  {global_pixel['AUROC']:>10.4f}")
            print(f"  {'AP':<20s} {macro_p_ap:>10.4f}  {global_pixel['AP']:>10.4f}")
            print(f"  {'F1max':<20s} {macro_p_f1:>10.4f}  {global_pixel['F1max']:>10.4f}")

    # ══════════════════════════════════════════════════════
    # 汇总
    # ══════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print("汇总")
    print(f"{'='*60}")
    print(f"  图像级 AUROC (宏平均):  {macro_auroc:.4f}")
    print(f"  图像级 AP   (宏平均):  {macro_ap:.4f}")

    if not args.image_only and all_pred_masks:
        print(f"  像素级 AUROC (宏平均):  {macro_p_auroc:.4f}")
        print(f"  像素级 AP   (宏平均):  {macro_p_ap:.4f}")
        print(f"  像素级 F1max(宏平均):  {macro_p_f1:.4f}")

    # 按比赛公式估算综合分 (S = 100 * (0.3*I-AUROC + 0.5*P-metrics + 0.2*ZS))
    # 这里只算已见类部分 (不含 zero-shot)
    if not args.image_only and all_pred_masks:
        s_cls = (macro_auroc + macro_ap) / 2
        s_seg = (macro_p_auroc + macro_p_ap + macro_p_f1) / 3
        estimated = 100 * (0.3 * s_cls + 0.5 * s_seg)  # 不含 0.2*S_zs
        print(f"\n  估算得分 (仅已见类):  {estimated:.2f} / 80")
        print(f"  (注: 总分 100 = 30*S_cls + 50*S_seg + 20*S_zs，此处缺 S_zs)")


if __name__ == "__main__":
    main()
