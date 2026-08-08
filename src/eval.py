"""
推理评估脚本
在 Test_A 上评估训练好的 INP-Former 模型，输出异常得分

用法：
    python -m src.eval --checkpoint checkpoints/battery/best.pth
    python -m src.eval --checkpoint checkpoints/all/best.pth --root_dir data
"""
import argparse
import json
import os
import sys
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.config import get_config
from src.data.dataset import build_dataloader
from src.models.dinov2_extractor import DINOv2Extractor
from src.models.inpformer import INPFormer
from src.utils.utils import load_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(description="INP-Former 推理评估")
    parser.add_argument("--checkpoint", type=str, required=True, help="模型 checkpoint 路径")
    parser.add_argument("--root_dir", type=str, default="data", help="数据集根目录")
    parser.add_argument("--output", type=str, default="results.json", help="输出结果文件")
    parser.add_argument("--device", type=str, default=None, help="设备")
    parser.add_argument(
        "--dinov2_weights", type=str, default=None,
        help="DINOv2 本地权重路径",
    )
    parser.add_argument(
        "--image_size", type=int, default=None,
        help="输入图像尺寸（MPS/CPU 验证建议 224，GPU 默认 518）",
    )
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=0)
    return parser.parse_args()


@torch.no_grad()
def evaluate(
    dinov2: DINOv2Extractor,
    model: INPFormer,
    test_loader,
    device: str,
) -> Dict:
    """
    在测试集上推理，收集异常得分

    Returns:
        dict: {
            "results": List[dict],  每个样本的异常得分
            "mean_score": float,    平均异常得分
        }
    """
    model.eval()
    all_results = []

    for batch_idx, batch in enumerate(test_loader):
        views = batch["views"].to(device)
        categories = batch["category"]
        sample_ids = batch["sample_id"]

        # DINOv2 特征提取
        dinov2_feats = dinov2.extract_multi_view(views)

        # 异常得分
        scores = model.anomaly_score(dinov2_feats)

        # 收集结果
        B = views.size(0)
        for i in range(B):
            result = {
                "category": categories[i],
                "sample_id": sample_ids[i],
                "image_score": scores["image_score"][i].item(),
                "view_scores": scores["view_score"][i].cpu().tolist(),
            }
            all_results.append(result)

        if (batch_idx + 1) % 10 == 0:
            print(f"  推理进度: {batch_idx + 1}/{len(test_loader)}")

    # 统计
    image_scores = [r["image_score"] for r in all_results]
    mean_score = float(np.mean(image_scores))
    std_score = float(np.std(image_scores))

    return {
        "results": all_results,
        "mean_score": mean_score,
        "std_score": std_score,
        "num_samples": len(all_results),
    }


def main():
    args = parse_args()

    # 加载 checkpoint
    print(f"加载 checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location="cpu")
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

    # 构建模型
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
        dropout=0.0,  # 推理时关闭 dropout
        num_views=cfg.data.num_views,
        n_flow_layers=cfg.inpformer.n_flow_layers,
        coupling_hidden=cfg.inpformer.coupling_hidden,
        score_type=cfg.inpformer.score_type,
    ).to(device)

    model.load_state_dict(ckpt["model"])
    print(f"已加载模型, epoch={ckpt.get('epoch', 'N/A')}")

    # 测试数据
    test_loader = build_dataloader(cfg, split="Test_A", categories=cfg.categories)

    # 推理
    print("\n开始推理...")
    results = evaluate(dinov2, model, test_loader, device)

    print(f"\n推理完成:")
    print(f"  样本数: {results['num_samples']}")
    print(f"  平均异常得分: {results['mean_score']:.4f} ± {results['std_score']:.4f}")

    # 保存结果
    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"结果已保存至: {args.output}")


if __name__ == "__main__":
    main()
