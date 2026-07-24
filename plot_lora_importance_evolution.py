#!/usr/bin/env python3
"""Plot LoRA layer-importance changes over training."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_csv", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--title", default="All-LoRA Layer Importance over Training")
    parser.add_argument(
        "--score",
        default="normalized_grad_score",
        choices=["normalized_grad_score", "normalized_update_score"],
    )
    args = parser.parse_args()

    df = pd.read_csv(args.input_csv)
    required = {"train_step", "block", "block_index", args.score}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"Missing columns: {sorted(missing)}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    df = df.sort_values(["train_step", "block_index"])

    plt.figure(figsize=(15, 8))
    for step, group in df.groupby("train_step", sort=True):
        plt.plot(
            group["block_index"],
            group[args.score],
            marker="o",
            linewidth=2,
            label=f"Step {int(step)}",
        )
    first = df[df["train_step"] == df["train_step"].min()].sort_values("block_index")
    plt.xticks(first["block_index"], first["block"], rotation=45, ha="right")
    plt.xlabel("Transformer block")
    plt.ylabel(args.score)
    plt.title(args.title)
    plt.grid(alpha=0.25)
    plt.legend(ncol=2)
    plt.tight_layout()
    plt.savefig(output_dir / f"{args.score}_lines.png", dpi=200)
    plt.close()

    pivot = df.pivot(index="train_step", columns="block", values=args.score)
    ordered_blocks = (
        df[["block", "block_index"]]
        .drop_duplicates()
        .sort_values("block_index")["block"]
        .tolist()
    )
    pivot = pivot.reindex(columns=ordered_blocks)
    plt.figure(figsize=(16, 6))
    image = plt.imshow(pivot.values, aspect="auto", cmap="viridis")
    plt.colorbar(image, label=args.score)
    plt.xticks(range(len(pivot.columns)), pivot.columns, rotation=45, ha="right")
    plt.yticks(range(len(pivot.index)), [f"Step {int(step)}" for step in pivot.index])
    plt.xlabel("Transformer block")
    plt.ylabel("Training checkpoint")
    plt.title(args.title)
    plt.tight_layout()
    plt.savefig(output_dir / f"{args.score}_heatmap.png", dpi=200)
    plt.close()

    print(f"Wrote plots to {output_dir}")


if __name__ == "__main__":
    main()
