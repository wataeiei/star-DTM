#!/usr/bin/env python3
"""Plot the three-seed quality-efficiency trade-off for the core methods."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


COLORS = {"All-LoRA": "#4C78A8", "GradSkip-LoRA": "#E45756"}
LABELS = {"All-LoRA": "All-LoRA", "GradSkip-LoRA": "GradSkip-LoRA (Ours)"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_csv",
        type=Path,
        default=Path(
            "paper_results/cvpr_core_multiseed/paper_core_mean_std.csv"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("paper_results/cvpr_core_multiseed/quality_efficiency_pareto"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame = pd.read_csv(args.input_csv)
    expected = {"All-LoRA", "GradSkip-LoRA"}
    methods = set(frame["method"])
    if methods != expected:
        raise SystemExit(f"Expected methods {sorted(expected)}, found {sorted(methods)}")

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.linewidth": 0.8,
        }
    )
    figure, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), constrained_layout=True)
    panels = [
        ("psnr", "PSNR (dB)", "Pixel fidelity"),
        ("clipiqa", "CLIP-IQA", "Perceptual quality"),
    ]

    for axis, (metric, ylabel, title) in zip(axes, panels):
        for _, row in frame.iterrows():
            method = row["method"]
            adapter_mb = float(row["adapter_size_mb_mean"])
            marker_area = 55.0 + 5.0 * adapter_mb
            axis.errorbar(
                float(row["train_step_time_s_mean"]),
                float(row[f"{metric}_mean"]),
                xerr=float(row["train_step_time_s_sample_std"]),
                yerr=float(row[f"{metric}_sample_std"]),
                fmt="none",
                ecolor=COLORS[method],
                elinewidth=1.0,
                capsize=2.5,
                alpha=0.85,
                zorder=2,
            )
            axis.scatter(
                float(row["train_step_time_s_mean"]),
                float(row[f"{metric}_mean"]),
                s=marker_area,
                color=COLORS[method],
                edgecolor="white",
                linewidth=0.9,
                label=LABELS[method],
                zorder=3,
            )
            offset = (7, 6) if method == "GradSkip-LoRA" else (-7, -15)
            alignment = "left" if method == "GradSkip-LoRA" else "right"
            axis.annotate(
                f"{LABELS[method]}\n{adapter_mb:.2f} MB",
                (
                    float(row["train_step_time_s_mean"]),
                    float(row[f"{metric}_mean"]),
                ),
                xytext=offset,
                textcoords="offset points",
                ha=alignment,
                va="bottom",
                fontsize=7.5,
            )

        axis.set_title(title)
        axis.set_xlabel("Training time for 1,000 steps (s)")
        axis.set_ylabel(ylabel)
        axis.grid(True, color="#D9D9D9", linewidth=0.6, alpha=0.7)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.margins(x=0.18, y=0.25)

    handles, labels = axes[0].get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    figure.legend(
        unique.values(),
        unique.keys(),
        loc="upper center",
        bbox_to_anchor=(0.5, 1.04),
        ncol=2,
        frameon=False,
    )
    figure.text(
        0.5,
        -0.02,
        "Markers scale with adapter size; error bars show sample standard deviation across three seeds.",
        ha="center",
        fontsize=7.5,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    png_path = args.output.with_suffix(".png")
    pdf_path = args.output.with_suffix(".pdf")
    figure.savefig(png_path, dpi=300, bbox_inches="tight")
    figure.savefig(pdf_path, bbox_inches="tight")
    plt.close(figure)
    print(f"Wrote {png_path}")
    print(f"Wrote {pdf_path}")


if __name__ == "__main__":
    main()
