"""
Real-IAD Variety 数据集加载器
支持多视角样本加载，用于无监督异常检测训练与评估
"""
import os
import logging
from pathlib import Path
from typing import List, Optional, Tuple

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image, ImageFile

# 允许加载截断/损坏的 PNG 文件
ImageFile.LOAD_TRUNCATED_IMAGES = True

logger = logging.getLogger(__name__)


class RealIADDataset(Dataset):
    """
    Real-IAD Variety 数据集
    每个样本包含 num_views 个视角的图像
    """

    def __init__(
        self,
        root_dir: str,
        split: str = "Train",
        categories: Optional[List[str]] = None,
        transform=None,
        num_views: int = 5,
    ):
        super().__init__()
        self.root_dir = Path(root_dir) / split
        self.split = split
        self.num_views = num_views
        self.transform = transform or self._default_transform()

        # 获取所有类别
        if categories is None:
            self.categories = sorted(
                [d.name for d in self.root_dir.iterdir() if d.is_dir()]
            )
        else:
            self.categories = categories

        # 构建样本索引：(category, sample_id)
        self.samples: List[Tuple[str, str]] = []
        for cat in self.categories:
            cat_dir = self.root_dir / cat
            if not cat_dir.exists():
                continue
            for sample_dir in sorted(cat_dir.iterdir()):
                if sample_dir.is_dir():
                    self.samples.append((cat, sample_dir.name))

        print(f"[{split}] 加载 {len(self.categories)} 个类别, "
              f"{len(self.samples)} 个样本")

    def _default_transform(self):
        return transforms.Compose([
            transforms.Resize((518, 518)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        cat, sample_id = self.samples[idx]
        sample_dir = self.root_dir / cat / sample_id

        # 加载所有视角图像，遇到损坏图片时用黑色占位
        views = []
        for v in range(self.num_views):
            img_path = sample_dir / f"{v}.png"
            if not img_path.exists():
                img_path = sample_dir / f"{v}.jpg"
            try:
                img = Image.open(img_path).convert("RGB")
            except Exception as e:
                logger.warning(f"[损坏图像] {img_path}: {e}，使用黑色占位")
                # 创建黑色占位图
                img = Image.new("RGB", (518, 518), (0, 0, 0))
            img = self.transform(img)
            views.append(img)

        views = torch.stack(views, dim=0)

        return {
            "views": views,
            "category": cat,
            "sample_id": sample_id,
        }


class MultiViewBatchSampler:
    """
    自定义 batch 采样器
    确保每个 batch 来自同一类别，便于类内特征建模
    """

    def __init__(self, dataset: RealIADDataset, batch_size: int, shuffle: bool = True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle

        # 按类别分组索引
        self.cat_indices = {}
        for idx, (cat, _) in enumerate(dataset.samples):
            self.cat_indices.setdefault(cat, []).append(idx)

    def __iter__(self):
        import random
        all_indices = list(range(len(self.dataset)))
        if self.shuffle:
            random.shuffle(all_indices)

        batch = []
        for idx in all_indices:
            batch.append(idx)
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        if batch:
            yield batch

    def __len__(self):
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size


def build_dataloader(cfg, split="Train", categories=None) -> DataLoader:
    """构建 DataLoader"""
    transform = transforms.Compose([
        transforms.Resize((cfg.data.image_size, cfg.data.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=cfg.data.mean, std=cfg.data.std),
    ])

    dataset = RealIADDataset(
        root_dir=cfg.data.root_dir,
        split=split,
        categories=categories,
        transform=transform,
        num_views=cfg.data.num_views,
    )

    return DataLoader(
        dataset,
        batch_size=cfg.data.batch_size,
        shuffle=(split == "Train"),
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
        drop_last=(split == "Train"),
    )
