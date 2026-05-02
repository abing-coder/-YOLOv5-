"""Compare two YOLOv5 training runs and plot metric curves.

Usage:
    python tools/compare_train_results.py --base runs/train/exp_base/results.csv --new runs/train/exp_new/results.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description="Compare two YOLOv5 training results.csv files")
    parser.add_argument("--base", type=Path, required=True, help="Baseline results.csv path")
    parser.add_argument("--new", type=Path, required=True, help="Improved model results.csv path")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("runs/train/comparisons/yolov5_compare.png"),
        help="Output image path",
    )
    parser.add_argument("--base-name", type=str, default="YOLOv5s baseline", help="Legend name for baseline")
    parser.add_argument("--new-name", type=str, default="YOLOv5s-CBAM", help="Legend name for improved model")
    return parser.parse_args()


def load_csv(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"results file not found: {path}")
    return pd.read_csv(path)


def get_col(df: pd.DataFrame, name: str):
    cols = {c.strip(): c for c in df.columns}
    if name not in cols:
        raise KeyError(f"missing column: {name}")
    return df[cols[name]].astype(float)


def main():
    args = parse_args()
    base_df = load_csv(args.base)
    new_df = load_csv(args.new)

    metrics = [
        ("metrics/mAP_0.5", "mAP@0.5"),
        ("metrics/mAP_0.5:0.95", "mAP@0.5:0.95"),
        ("metrics/precision", "Precision"),
        ("metrics/recall", "Recall"),
        ("val/box_loss", "Val box loss"),
        ("val/obj_loss", "Val obj loss"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), tight_layout=True)
    axes = axes.ravel()

    x_base = get_col(base_df, "epoch")
    x_new = get_col(new_df, "epoch")
    for i, (col, title) in enumerate(metrics):
        y_base = get_col(base_df, col)
        y_new = get_col(new_df, col)
        axes[i].plot(x_base, y_base, marker="o", linewidth=1.8, markersize=3, label=args.base_name)
        axes[i].plot(x_new, y_new, marker="o", linewidth=1.8, markersize=3, label=args.new_name)
        axes[i].set_title(title)
        axes[i].set_xlabel("Epoch")
        axes[i].grid(alpha=0.25)
        if i in (0, 1, 2, 3):
            axes[i].set_ylim(0, 1)
        axes[i].legend()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=250)
    plt.close(fig)
    print(f"Saved comparison plot: {args.out}")


if __name__ == "__main__":
    main()
