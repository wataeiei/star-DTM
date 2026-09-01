#!/usr/bin/env python3
"""Compare block-skipping efficiency across model architectures and seeds."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REQUIRED_MANIFEST_COLUMNS = {
    "model",
    "total_blocks",
    "seed",
    "baseline_run_dir",
    "gradskip_run_dir",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--expected_steps", type=int, default=1000)
    return parser.parse_args()


def read_run(run_dir: Path, expected_steps: int) -> tuple[pd.Series, pd.DataFrame]:
    summary_path = run_dir / "summary.csv"
    log_path = run_dir / "train_log.csv"
    missing = [str(path) for path in (summary_path, log_path) if not path.is_file()]
    if missing:
        raise SystemExit("Missing run artifacts:\n  " + "\n  ".join(missing))

    summary = pd.read_csv(summary_path).iloc[0]
    log = pd.read_csv(log_path)
    final_step = int(log["step"].max())
    if final_step != expected_steps:
        raise SystemExit(
            f"{run_dir} ended at step {final_step}, expected {expected_steps}."
        )
    return summary, log


def mean_numeric(frame: pd.DataFrame, column: str) -> float:
    if column not in frame:
        return float("nan")
    return float(pd.to_numeric(frame[column], errors="coerce").mean())


def max_numeric(frame: pd.DataFrame, column: str) -> float:
    if column not in frame:
        return float("nan")
    return float(pd.to_numeric(frame[column], errors="coerce").max())


def reduction(current: float, reference: float) -> float:
    return 100.0 * (reference - current) / reference


def build_per_seed(manifest: pd.DataFrame, expected_steps: int) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for item in manifest.itertuples(index=False):
        baseline_summary, baseline_log = read_run(
            Path(item.baseline_run_dir), expected_steps
        )
        gradskip_summary, gradskip_log = read_run(
            Path(item.gradskip_run_dir), expected_steps
        )

        baseline_step_s = mean_numeric(baseline_log, "train_step_time_s")
        gradskip_step_s = mean_numeric(gradskip_log, "train_step_time_s")
        mean_skipped = mean_numeric(gradskip_log, "skipped_block_count")
        mean_requested = mean_numeric(gradskip_log, "requested_skip_count")
        total_blocks = int(item.total_blocks)

        baseline_peak = max_numeric(baseline_log, "train_peak_cuda_mem_mb")
        gradskip_peak = max_numeric(gradskip_log, "train_peak_cuda_mem_mb")
        if np.isnan(baseline_peak):
            baseline_peak = float(baseline_summary["peak_cuda_mem_mb"])
        if np.isnan(gradskip_peak):
            gradskip_peak = float(gradskip_summary["peak_cuda_mem_mb"])

        baseline_adapter = float(baseline_summary["adapter_size_mb"])
        gradskip_adapter = float(gradskip_summary["adapter_size_mb"])
        baseline_params = float(baseline_summary["trainable_lora_params"])
        gradskip_params = float(gradskip_summary["trainable_lora_params"])

        rows.append(
            {
                "model": item.model,
                "seed": int(item.seed),
                "total_blocks": total_blocks,
                "mean_skipped_blocks": mean_skipped,
                "mean_requested_blocks": mean_requested,
                "skip_ratio_pct": 100.0 * mean_skipped / total_blocks,
                "request_actual_abs_diff": abs(mean_requested - mean_skipped),
                "baseline_step_time_s": baseline_step_s,
                "gradskip_step_time_s": gradskip_step_s,
                "train_time_reduction_pct": reduction(
                    gradskip_step_s, baseline_step_s
                ),
                "train_speedup_x": baseline_step_s / gradskip_step_s,
                "baseline_peak_cuda_mb": baseline_peak,
                "gradskip_peak_cuda_mb": gradskip_peak,
                "peak_memory_reduction_pct": reduction(
                    gradskip_peak, baseline_peak
                ),
                "baseline_adapter_mb": baseline_adapter,
                "gradskip_adapter_mb": gradskip_adapter,
                "adapter_reduction_pct": reduction(
                    gradskip_adapter, baseline_adapter
                ),
                "baseline_trainable_params": baseline_params,
                "gradskip_trainable_params": gradskip_params,
                "parameter_reduction_pct": reduction(
                    gradskip_params, baseline_params
                ),
            }
        )
    return pd.DataFrame(rows)


def aggregate(per_seed: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "mean_skipped_blocks",
        "skip_ratio_pct",
        "baseline_step_time_s",
        "gradskip_step_time_s",
        "train_time_reduction_pct",
        "train_speedup_x",
        "peak_memory_reduction_pct",
        "adapter_reduction_pct",
        "parameter_reduction_pct",
    ]
    rows: list[dict[str, object]] = []
    for model, group in per_seed.groupby("model", sort=False):
        row: dict[str, object] = {
            "model": model,
            "num_seeds": len(group),
            "total_blocks": int(group["total_blocks"].iloc[0]),
        }
        for metric in metrics:
            values = pd.to_numeric(group[metric], errors="coerce")
            row[f"{metric}_mean"] = values.mean()
            row[f"{metric}_sample_std"] = values.std(ddof=1)
        rows.append(row)
    return pd.DataFrame(rows)


def plot_summary(summary: pd.DataFrame, output_dir: Path) -> None:
    labels = summary["model"].tolist()
    x = np.arange(len(labels))
    width = 0.34

    fig, ax = plt.subplots(figsize=(6.6, 3.7))
    skip = summary["skip_ratio_pct_mean"].to_numpy()
    speed = summary["train_time_reduction_pct_mean"].to_numpy()
    skip_std = summary["skip_ratio_pct_sample_std"].fillna(0).to_numpy()
    speed_std = summary["train_time_reduction_pct_sample_std"].fillna(0).to_numpy()

    bars_skip = ax.bar(
        x - width / 2,
        skip,
        width,
        yerr=skip_std,
        capsize=3,
        label="Blocks skipped",
        color="#4C78A8",
    )
    bars_speed = ax.bar(
        x + width / 2,
        speed,
        width,
        yerr=speed_std,
        capsize=3,
        label="Training time reduced",
        color="#E45756",
    )
    ax.set_ylabel("Percentage (%)")
    ax.set_xticks(x, labels)
    ax.set_ylim(0, max(float(np.max(skip)), float(np.max(speed))) * 1.28)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)

    for bars in (bars_skip, bars_speed):
        for bar in bars:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.7,
                f"{bar.get_height():.1f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(
            output_dir / f"cross_model_skip_efficiency.{suffix}",
            dpi=300,
            bbox_inches="tight",
        )
    plt.close(fig)


def main() -> None:
    args = parse_args()
    manifest = pd.read_csv(args.manifest)
    missing = REQUIRED_MANIFEST_COLUMNS - set(manifest.columns)
    if missing:
        raise SystemExit(f"Manifest is missing columns: {sorted(missing)}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    per_seed = build_per_seed(manifest, args.expected_steps)
    summary = aggregate(per_seed)

    per_seed_path = args.output_dir / "cross_model_skip_efficiency_per_seed.csv"
    summary_path = args.output_dir / "cross_model_skip_efficiency_mean_std.csv"
    per_seed.to_csv(per_seed_path, index=False)
    summary.to_csv(summary_path, index=False)
    plot_summary(summary, args.output_dir)

    print("Per-seed results:")
    print(per_seed.to_string(index=False))
    print("\nMean +/- sample std:")
    print(summary.to_string(index=False))
    print(f"\nWrote {per_seed_path}")
    print(f"Wrote {summary_path}")
    print(f"Wrote {args.output_dir / 'cross_model_skip_efficiency.png'}")
    print(f"Wrote {args.output_dir / 'cross_model_skip_efficiency.pdf'}")


if __name__ == "__main__":
    main()
