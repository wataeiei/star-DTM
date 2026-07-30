#!/usr/bin/env python3
"""Plot LoRA layer importance at fixed noise ratios over training."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def normalize(values: pd.Series, mode: str) -> pd.Series:
    if mode == "max" and float(values.max()) > 0:
        return values / float(values.max())
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_csv", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--title", default="Noise-conditioned LoRA Layer Importance")
    parser.add_argument("--score", default="normalized_grad_score")
    parser.add_argument("--normalize_y", default="max", choices=["max", "none"])
    parser.add_argument("--dpi", type=int, default=180)
    args = parser.parse_args()

    frame = pd.read_csv(args.input_csv)
    required = {"train_step", "noise_ratio", "block", "block_index", args.score}
    missing = required - set(frame.columns)
    if missing:
        raise SystemExit(f"Missing CSV columns: {sorted(missing)}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # At each training checkpoint, compare all fixed noise ratios.
    for step in sorted(frame["train_step"].unique()):
        subset = frame[frame["train_step"] == step]
        fig, ax = plt.subplots(figsize=(17, 8))
        for ratio in sorted(subset["noise_ratio"].unique()):
            curve = subset[subset["noise_ratio"] == ratio].sort_values("block_index")
            values = normalize(curve[args.score].astype(float), args.normalize_y)
            ax.plot(
                curve["block_index"],
                values,
                marker="o",
                linewidth=2,
                label=f"noise={ratio:g}",
            )
        labels = (
            subset[["block_index", "block"]]
            .drop_duplicates()
            .sort_values("block_index")
        )
        ax.set_xticks(labels["block_index"])
        ax.set_xticklabels(labels["block"], rotation=50, ha="right")
        suffix = " (max-normalized per noise curve)" if args.normalize_y == "max" else ""
        ax.set_ylabel(args.score + suffix)
        ax.set_xlabel("Layer")
        ax.set_title(f"{args.title} - train step {int(step)}")
        ax.grid(alpha=0.25)
        ax.legend(ncol=2)
        fig.tight_layout()
        path = output_dir / f"noise_importance_step_{int(step):04d}.png"
        fig.savefig(path, dpi=args.dpi)
        plt.close(fig)
        print(f"Wrote {path}")

    # For each noise ratio, show how the layer pattern evolves during training.
    for ratio in sorted(frame["noise_ratio"].unique()):
        subset = frame[frame["noise_ratio"] == ratio]
        fig, ax = plt.subplots(figsize=(17, 8))
        for step in sorted(subset["train_step"].unique()):
            curve = subset[subset["train_step"] == step].sort_values("block_index")
            values = normalize(curve[args.score].astype(float), args.normalize_y)
            ax.plot(
                curve["block_index"],
                values,
                marker="o",
                linewidth=2,
                label=f"step={int(step)}",
            )
        labels = (
            subset[["block_index", "block"]]
            .drop_duplicates()
            .sort_values("block_index")
        )
        ax.set_xticks(labels["block_index"])
        ax.set_xticklabels(labels["block"], rotation=50, ha="right")
        suffix = " (max-normalized per step curve)" if args.normalize_y == "max" else ""
        ax.set_ylabel(args.score + suffix)
        ax.set_xlabel("Layer")
        ax.set_title(f"{args.title} - noise ratio {ratio:g}")
        ax.grid(alpha=0.25)
        ax.legend(ncol=2)
        fig.tight_layout()
        token = str(ratio).replace(".", "p")
        path = output_dir / f"training_evolution_noise_{token}.png"
        fig.savefig(path, dpi=args.dpi)
        plt.close(fig)
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
