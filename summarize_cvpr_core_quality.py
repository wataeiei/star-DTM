#!/usr/bin/env python3
"""Build paper-ready three-seed quality and efficiency summaries."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


RUNS = {
    "All-LoRA": {
        42: {
            "sr": (
                "outputs/dit4sr_alllora1000_final_val419/sr_metrics_summary.csv",
                "All-LoRA-1000",
            ),
            "iqa": (
                "outputs/dit4sr_alllora1000_iqa_val419/iqa_metrics_summary.csv",
                "All-LoRA-1000",
            ),
            "semantic": (
                "outputs/dit4sr_alllora1000_semantic_val419/semantic_metrics_summary.csv",
                "All-LoRA-1000",
            ),
        },
        43: {
            "sr": (
                "outputs/dit4sr_all_gradskip_multiseed_new4_val419/sr_metrics_summary.csv",
                "All-LoRA-Seed43",
            ),
            "iqa": (
                "outputs/dit4sr_all_gradskip_multiseed_new4_iqa_val419/iqa_metrics_summary.csv",
                "All-LoRA-Seed43",
            ),
            "semantic": (
                "outputs/dit4sr_all_gradskip_multiseed_new4_semantic_val419/semantic_metrics_summary.csv",
                "All-LoRA-Seed43",
            ),
        },
        44: {
            "sr": (
                "outputs/dit4sr_all_gradskip_multiseed_new4_val419/sr_metrics_summary.csv",
                "All-LoRA-Seed44",
            ),
            "iqa": (
                "outputs/dit4sr_all_gradskip_multiseed_new4_iqa_val419/iqa_metrics_summary.csv",
                "All-LoRA-Seed44",
            ),
            "semantic": (
                "outputs/dit4sr_all_gradskip_multiseed_new4_semantic_val419/semantic_metrics_summary.csv",
                "All-LoRA-Seed44",
            ),
        },
    },
    "GradSkip-LoRA": {
        42: {
            "sr": (
                "outputs/dit4sr_gradtop8_noiseaware_1000_val419/sr_metrics_summary.csv",
                "Grad-Top8-NoiseAware-1000",
            ),
            "iqa": (
                "outputs/dit4sr_gradtop8_noiseaware_1000_iqa_val419/iqa_metrics_summary.csv",
                "Grad-Top8-NoiseAware-1000",
            ),
            "semantic": (
                "outputs/dit4sr_gradtop8_noiseaware_1000_semantic_val419/semantic_metrics_summary.csv",
                "Grad-Top8-NoiseAware-1000",
            ),
        },
        43: {
            "sr": (
                "outputs/dit4sr_all_gradskip_multiseed_new4_val419/sr_metrics_summary.csv",
                "GradSkip-LoRA-Seed43",
            ),
            "iqa": (
                "outputs/dit4sr_all_gradskip_multiseed_new4_iqa_val419/iqa_metrics_summary.csv",
                "GradSkip-LoRA-Seed43",
            ),
            "semantic": (
                "outputs/dit4sr_all_gradskip_multiseed_new4_semantic_val419/semantic_metrics_summary.csv",
                "GradSkip-LoRA-Seed43",
            ),
        },
        44: {
            "sr": (
                "outputs/dit4sr_all_gradskip_multiseed_new4_val419/sr_metrics_summary.csv",
                "GradSkip-LoRA-Seed44",
            ),
            "iqa": (
                "outputs/dit4sr_all_gradskip_multiseed_new4_iqa_val419/iqa_metrics_summary.csv",
                "GradSkip-LoRA-Seed44",
            ),
            "semantic": (
                "outputs/dit4sr_all_gradskip_multiseed_new4_semantic_val419/semantic_metrics_summary.csv",
                "GradSkip-LoRA-Seed44",
            ),
        },
    },
}

QUALITY_METRICS = [
    "psnr",
    "ssim",
    "lpips",
    "musiq",
    "maniqa",
    "clipiqa",
    "liqe",
    "top1_accuracy",
    "macro_f1",
    "balanced_accuracy",
    "mean_inference_time_s",
    "images_per_hour",
]

TRAINING_METRICS = [
    "train_step_time_s",
    "mean_train_step_time_s",
    "experiment_time_s",
    "profiling_time_s",
    "peak_cuda_mem_mb",
    "adapter_size_mb",
    "lora_module_count",
    "trainable_lora_params",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--expected_images", type=int, default=419)
    parser.add_argument(
        "--training_csv",
        type=Path,
        default=Path("outputs/cvpr_core_multiseed_training_summary.csv"),
    )
    parser.add_argument(
        "--output_dir", type=Path, default=Path("paper_results/cvpr_core_multiseed")
    )
    return parser.parse_args()


def select_method(path: Path, method: str) -> dict[str, object]:
    if not path.is_file():
        raise SystemExit(f"Missing result file: {path}")
    frame = pd.read_csv(path)
    if "method" not in frame.columns:
        raise SystemExit(f"Missing method column: {path}")
    selected = frame.loc[frame["method"] == method]
    if len(selected) != 1:
        available = ", ".join(frame["method"].astype(str).tolist())
        raise SystemExit(
            f"Expected one row for {method!r} in {path}; found {len(selected)}. "
            f"Available methods: {available}"
        )
    return selected.iloc[0].to_dict()


def collect_quality(root: Path, expected_images: int) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for method, seed_specs in RUNS.items():
        for seed, specs in seed_specs.items():
            sr = select_method(root / specs["sr"][0], specs["sr"][1])
            iqa = select_method(root / specs["iqa"][0], specs["iqa"][1])
            semantic = select_method(
                root / specs["semantic"][0], specs["semantic"][1]
            )
            counts = {
                int(sr["num_images"]),
                int(iqa["num_images"]),
                int(semantic["num_images"]),
            }
            if counts != {expected_images}:
                raise SystemExit(
                    f"{method} seed {seed} image counts are {sorted(counts)}, "
                    f"expected {expected_images}."
                )
            rows.append(
                {
                    "method": method,
                    "seed": seed,
                    "num_images": expected_images,
                    "psnr": sr["mean_psnr"],
                    "ssim": sr["mean_ssim"],
                    "mean_inference_time_s": sr["mean_inference_time_s"],
                    "images_per_hour": sr["images_per_hour"],
                    "lpips": iqa["mean_lpips"],
                    "musiq": iqa["mean_musiq"],
                    "maniqa": iqa["mean_maniqa"],
                    "clipiqa": iqa["mean_clipiqa"],
                    "liqe": iqa["mean_liqe"],
                    "top1_accuracy": semantic["top1_accuracy"],
                    "macro_f1": semantic["macro_f1"],
                    "balanced_accuracy": semantic["balanced_accuracy"],
                }
            )
    return pd.DataFrame(rows)


def aggregate(frame: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for method, group in frame.groupby("method", sort=False):
        row: dict[str, object] = {"method": method, "num_seeds": len(group)}
        for metric in metrics:
            values = pd.to_numeric(group[metric], errors="raise")
            row[f"{metric}_mean"] = values.mean()
            row[f"{metric}_sample_std"] = values.std(ddof=1)
        rows.append(row)
    return pd.DataFrame(rows)


def latex_value(row: pd.Series, metric: str, digits: int, scale: float = 1.0) -> str:
    mean = float(row[f"{metric}_mean"]) * scale
    std = float(row[f"{metric}_sample_std"]) * scale
    return f"{mean:.{digits}f} $\\pm$ {std:.{digits}f}"


def write_latex(path: Path, merged: pd.DataFrame) -> None:
    by_method = merged.set_index("method")
    all_lora = by_method.loc["All-LoRA"]
    ours = by_method.loc["GradSkip-LoRA"]
    lines = [
        "% Required packages: booktabs, graphicx",
        "\\begin{table*}[t]",
        "  \\centering",
        "  \\caption{Three-seed comparison on the disjoint 419-image UC Merced validation set. Results are mean $\\pm$ sample standard deviation.}",
        "  \\label{tab:core_multiseed}",
        "  \\setlength{\\tabcolsep}{3.2pt}",
        "  \\resizebox{\\textwidth}{!}{%",
        "  \\begin{tabular}{lcccccccc}",
        "    \\toprule",
        "    Method & PSNR$\\uparrow$ & SSIM$\\uparrow$ & LPIPS$\\downarrow$ & CLIP-IQA$\\uparrow$ & Top-1 (\\%)$\\uparrow$ & Train time (s)$\\downarrow$ & Peak mem. (MB)$\\downarrow$ & Adapter (MB)$\\downarrow$ \\\\",
        "    \\midrule",
        "    All-LoRA & "
        + " & ".join(
            [
                latex_value(all_lora, "psnr", 3),
                latex_value(all_lora, "ssim", 4),
                latex_value(all_lora, "lpips", 4),
                latex_value(all_lora, "clipiqa", 4),
                latex_value(all_lora, "top1_accuracy", 2, 100.0),
                latex_value(all_lora, "train_step_time_s", 1),
                latex_value(all_lora, "peak_cuda_mem_mb", 0),
                latex_value(all_lora, "adapter_size_mb", 2),
            ]
        )
        + " \\\\",
        "    \\textbf{GradSkip-LoRA (Ours)} & "
        + " & ".join(
            [
                latex_value(ours, "psnr", 3),
                latex_value(ours, "ssim", 4),
                latex_value(ours, "lpips", 4),
                "\\textbf{" + latex_value(ours, "clipiqa", 4) + "}",
                latex_value(ours, "top1_accuracy", 2, 100.0),
                "\\textbf{" + latex_value(ours, "train_step_time_s", 1) + "}",
                "\\textbf{" + latex_value(ours, "peak_cuda_mem_mb", 0) + "}",
                "\\textbf{" + latex_value(ours, "adapter_size_mb", 2) + "}",
            ]
        )
        + " \\\\",
        "    \\bottomrule",
        "  \\end{tabular}%",
        "  }",
        "\\end{table*}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    output_dir = root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    quality = collect_quality(root, args.expected_images)
    quality_path = output_dir / "quality_per_seed.csv"
    quality.to_csv(quality_path, index=False)
    quality_aggregate = aggregate(quality, QUALITY_METRICS)
    quality_aggregate_path = output_dir / "quality_mean_std.csv"
    quality_aggregate.to_csv(quality_aggregate_path, index=False)

    training_path = root / args.training_csv
    if not training_path.is_file():
        raise SystemExit(f"Missing training summary: {training_path}")
    training = pd.read_csv(training_path)
    expected_pairs = {(method, seed) for method in RUNS for seed in RUNS[method]}
    actual_pairs = set(zip(training["method"], training["seed"].astype(int)))
    if actual_pairs != expected_pairs:
        raise SystemExit(
            "Training summary method/seed pairs do not match quality results. "
            f"Expected {sorted(expected_pairs)}, found {sorted(actual_pairs)}"
        )
    training_aggregate = aggregate(training, TRAINING_METRICS)
    merged = quality_aggregate.merge(
        training_aggregate, on=["method", "num_seeds"], validate="one_to_one"
    )
    merged_path = output_dir / "paper_core_mean_std.csv"
    merged.to_csv(merged_path, index=False)

    all_row = merged.set_index("method").loc["All-LoRA"]
    ours_row = merged.set_index("method").loc["GradSkip-LoRA"]
    comparison = pd.DataFrame(
        [
            {
                "train_time_reduction_pct": 100.0
                * (1.0 - ours_row["train_step_time_s_mean"] / all_row["train_step_time_s_mean"]),
                "train_speedup_x": all_row["train_step_time_s_mean"]
                / ours_row["train_step_time_s_mean"],
                "inference_time_reduction_pct": 100.0
                * (1.0 - ours_row["mean_inference_time_s_mean"] / all_row["mean_inference_time_s_mean"]),
                "peak_memory_reduction_pct": 100.0
                * (1.0 - ours_row["peak_cuda_mem_mb_mean"] / all_row["peak_cuda_mem_mb_mean"]),
                "adapter_reduction_pct": 100.0
                * (1.0 - ours_row["adapter_size_mb_mean"] / all_row["adapter_size_mb_mean"]),
                "psnr_delta_db": ours_row["psnr_mean"] - all_row["psnr_mean"],
                "ssim_delta": ours_row["ssim_mean"] - all_row["ssim_mean"],
                "lpips_delta": ours_row["lpips_mean"] - all_row["lpips_mean"],
                "clipiqa_delta": ours_row["clipiqa_mean"] - all_row["clipiqa_mean"],
                "top1_delta_percentage_points": 100.0
                * (ours_row["top1_accuracy_mean"] - all_row["top1_accuracy_mean"]),
            }
        ]
    )
    comparison_path = output_dir / "ours_vs_all_lora.csv"
    comparison.to_csv(comparison_path, index=False)

    latex_path = output_dir / "table_core_multiseed.tex"
    write_latex(latex_path, merged)

    print("Three-seed quality results:")
    print(quality.to_string(index=False))
    print("\nPaper comparison:")
    print(comparison.T.to_string(header=False))
    for path in (
        quality_path,
        quality_aggregate_path,
        merged_path,
        comparison_path,
        latex_path,
    ):
        print(f"Wrote {path.relative_to(root)}")


if __name__ == "__main__":
    main()
