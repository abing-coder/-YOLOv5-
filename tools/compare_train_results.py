r"""Compare YOLOv5 training runs and plot metric curves (English labels).

Two modes:
  (A) Legacy — two curves:
      python tools/compare_train_results.py --base PATH --new PATH ...
  (B) Multi — three or more curves:
      python tools/compare_train_results.py --series "Baseline:runs/train/exp1/results.csv" \\
          "CBAM:runs/train/exp2/results.csv" "Ghost:runs/train/exp3/results.csv" --out OUT.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description="Compare YOLOv5 training results.csv files")
    parser.add_argument(
        "--series",
        nargs="+",
        metavar="LABEL:PATH",
        help='Repeat for each model, e.g. "Baseline:runs/train/a/results.csv"',
    )
    parser.add_argument("--base", type=Path, default=None, help="(legacy) Baseline results.csv")
    parser.add_argument("--new", type=Path, default=None, help="(legacy) Second model results.csv")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("runs/train/comparisons/yolov5_compare.png"),
        help="Output image path",
    )
    parser.add_argument("--base-name", type=str, default="YOLOv5s baseline", help="(legacy) Legend baseline")
    parser.add_argument("--new-name", type=str, default="YOLOv5s-CBAM", help="(legacy) Legend second")
    parser.add_argument(
        "--print-summary",
        action="store_true",
        help="Print last-epoch metrics for each series to console",
    )
    return parser.parse_args()


def parse_series_entry(entry: str) -> tuple[str, Path]:
    if ":" not in entry:
        raise ValueError(f'Expected "Label:path/to/results.csv", got: {entry}')
    label, path = entry.split(":", 1)
    label, path = label.strip(), path.strip()
    if not label or not path:
        raise ValueError(f"Invalid --series entry: {entry}")
    return label, Path(path)


def load_csv(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"results file not found: {path}")
    return pd.read_csv(path)


def get_col(df: pd.DataFrame, name: str):
    cols = {c.strip(): c for c in df.columns}
    if name not in cols:
        raise KeyError(f"missing column: {name}")
    return df[cols[name]].astype(float)


def collect_series(args) -> list[tuple[str, pd.DataFrame]]:
    if args.series:
        out = []
        for e in args.series:
            label, path = parse_series_entry(e)
            out.append((label, load_csv(path)))
        return out
    if args.base is None or args.new is None:
        raise SystemExit("Use either --series (multi) or both --base and --new (legacy).")
    return [(args.base_name, load_csv(args.base)), (args.new_name, load_csv(args.new))]


def main():
    args = parse_args()
    series_list = collect_series(args)

    metrics = [
        ("metrics/mAP_0.5", "mAP@0.5"),
        ("metrics/mAP_0.5:0.95", "mAP@0.5:0.95"),
        ("metrics/precision", "Precision"),
        ("metrics/recall", "Recall"),
        ("val/box_loss", "Val box loss"),
        ("val/obj_loss", "Val obj loss"),
    ]

    cmap = plt.cm.tab10.colors
    fig, axes = plt.subplots(2, 3, figsize=(16, 8), tight_layout=True)
    axes = axes.ravel()

    for i, (col, title) in enumerate(metrics):
        ax = axes[i]
        for si, (label, df) in enumerate(series_list):
            color = cmap[si % len(cmap)]
            x = get_col(df, "epoch")
            y = get_col(df, col)
            ax.plot(
                x,
                y,
                marker="o",
                linewidth=1.8,
                markersize=2.5,
                label=label,
                color=color,
            )
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.grid(alpha=0.25)
        if i in (0, 1, 2, 3):
            ax.set_ylim(0, 1)
        ax.legend(fontsize=8)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=250)
    plt.close(fig)
    print(f"Saved comparison plot: {args.out}")

    if args.print_summary:
        print("\nLast-epoch summary (same columns as curves):")
        for label, df in series_list:
            try:
                last = df.iloc[-1]
                cols = {c.strip(): c for c in df.columns}
                ep = last[cols["epoch"]]
                m50 = last[cols["metrics/mAP_0.5"]]
                m5095 = last[cols["metrics/mAP_0.5:0.95"]]
                p = last[cols["metrics/precision"]]
                r = last[cols["metrics/recall"]]
                print(f"  {label}: epoch={int(ep)}  P={p:.4f} R={r:.4f}  mAP50={m50:.4f}  mAP50-95={m5095:.4f}")
            except Exception as ex:
                print(f"  {label}: (summary failed) {ex}")


if __name__ == "__main__":
    main()
