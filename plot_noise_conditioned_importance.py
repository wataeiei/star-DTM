#!/usr/bin/env python3
"""Plot LoRA layer importance at fixed noise ratios over training."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def normalize(values: pd.Series, mode: str) -> pd.Series:
    if mode == "max" and float(values.max()) > 0:
        return values / float(values.max())
    return values


def heatmap_matrix(frame: pd.DataFrame, score: str, normalize_y: str):
    pivot = frame.pivot(index="noise_ratio", columns="block_index", values=score)
    pivot = pivot.sort_index().sort_index(axis=1)
    values = pivot.to_numpy(dtype=float)
    if normalize_y == "max":
        maxima = np.nanmax(values, axis=1, keepdims=True)
        values = np.divide(
            values,
            maxima,
            out=np.zeros_like(values),
            where=np.isfinite(maxima) & (maxima > 0),
        )
    return pivot, values


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

    block_labels = (
        frame[["block_index", "block"]]
        .drop_duplicates()
        .sort_values("block_index")
        .set_index("block_index")["block"]
    )

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

    # One figure containing noise-by-layer heatmaps for every training checkpoint.
    steps = sorted(frame["train_step"].unique())
    columns = 2 if len(steps) > 1 else 1
    rows = math.ceil(len(steps) / columns)
    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(20, max(4.8 * rows, 6)),
        squeeze=False,
        constrained_layout=True,
    )
    image = None
    for axis, step in zip(axes.flat, steps):
        subset = frame[frame["train_step"] == step]
        pivot, values = heatmap_matrix(subset, args.score, args.normalize_y)
        image = axis.imshow(values, aspect="auto", cmap="viridis", origin="lower")
        axis.set_title(f"Train step {int(step)}")
        axis.set_yticks(range(len(pivot.index)))
        axis.set_yticklabels([f"{value:g}" for value in pivot.index])
        axis.set_ylabel("Noise ratio")
        axis.set_xticks(range(len(pivot.columns)))
        axis.set_xticklabels(
            [block_labels.get(index, str(index)) for index in pivot.columns],
            rotation=55,
            ha="right",
            fontsize=7,
        )
        axis.set_xlabel("Layer")
    for axis in axes.flat[len(steps) :]:
        axis.set_visible(False)
    if image is not None:
        label = args.score
        if args.normalize_y == "max":
            label += " (row max-normalized)"
        fig.colorbar(image, ax=axes.ravel().tolist(), shrink=0.75, label=label)
    fig.suptitle(args.title + " - all training checkpoints", fontsize=16)
    path = output_dir / "noise_layer_importance_all_steps.png"
    fig.savefig(path, dpi=args.dpi)
    plt.close(fig)
    print(f"Wrote {path}")

    # A compact overall heatmap, averaging the score over all training checkpoints.
    mean_frame = (
        frame.groupby(["noise_ratio", "block_index", "block"], as_index=False)[args.score]
        .mean()
    )
    pivot, values = heatmap_matrix(mean_frame, args.score, args.normalize_y)
    fig, ax = plt.subplots(figsize=(18, 6))
    image = ax.imshow(values, aspect="auto", cmap="viridis", origin="lower")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([f"{value:g}" for value in pivot.index])
    ax.set_ylabel("Noise ratio")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(
        [block_labels.get(index, str(index)) for index in pivot.columns],
        rotation=55,
        ha="right",
    )
    ax.set_xlabel("Layer")
    label = args.score
    if args.normalize_y == "max":
        label += " (row max-normalized)"
    fig.colorbar(image, ax=ax, label=label)
    ax.set_title(args.title + " - mean over training checkpoints")
    fig.tight_layout()
    path = output_dir / "noise_layer_importance_mean_over_training.png"
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
