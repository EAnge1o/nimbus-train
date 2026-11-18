"""Plot a training log.csv into a loss-curve PNG (dev tool, needs matplotlib).

    python scripts/plot_log.py runs/tinystories-5m/log.csv results/loss_curve.png
"""
from __future__ import annotations

import csv
import sys


def main(log_path: str, out_path: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    steps, train, val = [], [], []
    with open(log_path, encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            steps.append(int(row["step"]))
            train.append(float(row["train_loss"]))
            if row["val_loss"]:
                val.append((int(row["step"]), float(row["val_loss"])))

    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)
    ax.plot(steps, train, linewidth=1.6, label="train loss", color="#2563eb")
    if val:
        vs, vl = zip(*val)
        ax.plot(vs, vl, marker="o", markersize=3, linewidth=1.4,
                label="val loss", color="#dc2626")
    ax.set_xlabel("step")
    ax.set_ylabel("cross-entropy loss")
    ax.set_title("nimbus-train · TinyStories-5M · pure-NumPy training")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
