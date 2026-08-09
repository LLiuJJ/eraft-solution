"""
从训练日志绘制 loss 曲线
用法: uv run python -m src.plot_loss --log data/trainlog.txt
"""
import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt


def parse_args():
    p = argparse.ArgumentParser(description="绘制训练 loss 曲线")
    p.add_argument("--log", type=str, default="data/trainlog.txt", help="训练日志路径")
    p.add_argument("--output", type=str, default=None, help="输出图片路径 (默认与日志同目录)")
    return p.parse_args()


def parse_log(log_path: str):
    """解析训练日志，提取每个 epoch 的 loss 指标"""
    pattern = re.compile(
        r"\[Epoch (\d+)/(\d+)\] "
        r"loss=([0-9.]+) "
        r"nll_cls=([0-9.]+) "
        r"nll_view=([0-9.]+) "
        r"lr=([0-9.e-]+)"
    )

    epochs, losses, nll_cls_list, nll_view_list, lrs = [], [], [], [], []

    with open(log_path, "r") as f:
        for line in f:
            match = pattern.search(line)
            if match:
                epochs.append(int(match.group(1)))
                losses.append(float(match.group(3)))
                nll_cls_list.append(float(match.group(4)))
                nll_view_list.append(float(match.group(5)))
                lrs.append(float(match.group(6)))

    return epochs, losses, nll_cls_list, nll_view_list, lrs


def plot_curves(epochs, losses, nll_cls_list, nll_view_list, lrs, output_path: str):
    """绘制 loss 曲线图"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # 1. Total Loss
    ax = axes[0, 0]
    ax.plot(epochs, losses, "b-", linewidth=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Total Loss")
    ax.set_title("Total Loss")
    ax.grid(True, alpha=0.3)

    # 2. NLL CLS
    ax = axes[0, 1]
    ax.plot(epochs, nll_cls_list, "r-", linewidth=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("NLL CLS")
    ax.set_title("NLL CLS (Global Feature)")
    ax.grid(True, alpha=0.3)

    # 3. NLL View
    ax = axes[1, 0]
    ax.plot(epochs, nll_view_list, "g-", linewidth=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("NLL View")
    ax.set_title("NLL View (View Feature)")
    ax.grid(True, alpha=0.3)

    # 4. Learning Rate
    ax = axes[1, 1]
    ax.plot(epochs, lrs, "orange", linewidth=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Learning Rate")
    ax.set_title("Learning Rate (Cosine Decay)")
    ax.grid(True, alpha=0.3)

    plt.suptitle("INP-Former Training Loss Curves", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Loss 曲线已保存至: {output_path}")
    plt.close()


def main():
    args = parse_args()
    log_path = args.log

    if not Path(log_path).exists():
        print(f"错误: 日志文件不存在 {log_path}")
        return

    epochs, losses, nll_cls_list, nll_view_list, lrs = parse_log(log_path)

    if not epochs:
        print("错误: 日志中没有找到有效的 epoch 数据")
        return

    print(f"解析到 {len(epochs)} 个 epoch (0 ~ {epochs[-1]})")
    print(f"Loss: {losses[0]:.4f} → {losses[-1]:.4f}")
    print(f"NLL CLS: {nll_cls_list[0]:.4f} → {nll_cls_list[-1]:.4f}")
    print(f"NLL View: {nll_view_list[0]:.4f} → {nll_view_list[-1]:.4f}")

    output_path = args.output or str(Path(log_path).with_suffix(".png"))
    plot_curves(epochs, losses, nll_cls_list, nll_view_list, lrs, output_path)


if __name__ == "__main__":
    main()
