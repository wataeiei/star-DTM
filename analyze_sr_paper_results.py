#!/usr/bin/env python3
"""Create paper-ready paired statistics from per-image SR evaluation results."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np


METRIC_DIRECTIONS = {"psnr": 1.0, "ssim": 1.0, "lpips": -1.0}


def parse_labeled_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Expected LABEL=PATH.")
    label, path = value.split("=", 1)
    if not label.strip():
        raise argparse.ArgumentTypeError("LABEL cannot be empty.")
    return label.strip(), Path(path).expanduser()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def numeric(values) -> np.ndarray:
    return np.asarray([float(value) for value in values if value not in ("", None)], dtype=np.float64)


def bootstrap_mean_ci(
    values: np.ndarray, rng: np.random.Generator, samples: int, confidence: float
) -> tuple[float, float]:
    if values.size == 0:
        return math.nan, math.nan
    if values.size == 1 or samples <= 0:
        value = float(values.mean())
        return value, value
    means = np.empty(samples, dtype=np.float64)
    chunk_size = max(1, min(1000, samples))
    for start in range(0, samples, chunk_size):
        stop = min(start + chunk_size, samples)
        indices = rng.integers(0, values.size, size=(stop - start, values.size))
        means[start:stop] = values[indices].mean(axis=1)
    alpha = (1.0 - confidence) * 50.0
    low, high = np.percentile(means, [alpha, 100.0 - alpha])
    return float(low), float(high)


def sign_flip_p_value(
    improvements: np.ndarray, rng: np.random.Generator, samples: int
) -> float:
    if improvements.size == 0:
        return math.nan
    observed = abs(float(improvements.mean()))
    if observed == 0.0:
        return 1.0
    exceed = 0
    completed = 0
    chunk_size = max(1, min(1000, samples))
    while completed < samples:
        count = min(chunk_size, samples - completed)
        signs = rng.integers(0, 2, size=(count, improvements.size), dtype=np.int8)
        signs = signs.astype(np.float64) * 2.0 - 1.0
        permuted = np.abs((signs * improvements).mean(axis=1))
        exceed += int(np.count_nonzero(permuted >= observed))
        completed += count
    return (exceed + 1.0) / (samples + 1.0)


def build_metric_maps(rows: list[dict], allow_incomplete: bool):
    required = {"method", "image", "psnr", "ssim", "lpips"}
    missing_columns = required - set(rows[0]) if rows else required
    if missing_columns:
        raise SystemExit(f"Missing columns in per-image CSV: {sorted(missing_columns)}")

    method_order = []
    maps: dict[str, dict[str, dict[str, float]]] = {}
    for row in rows:
        method = row["method"]
        image = row["image"]
        if method not in maps:
            method_order.append(method)
            maps[method] = {}
        if image in maps[method]:
            raise SystemExit(f"Duplicate method/image row: {method} / {image}")
        metrics = {}
        for metric in METRIC_DIRECTIONS:
            value = row.get(metric, "")
            if value not in ("", None):
                metrics[metric] = float(value)
        maps[method][image] = metrics

    image_sets = {method: set(images) for method, images in maps.items()}
    reference_method = method_order[0]
    reference_images = image_sets[reference_method]
    mismatches = {
        method: {
            "missing": sorted(reference_images - images),
            "extra": sorted(images - reference_images),
        }
        for method, images in image_sets.items()
        if images != reference_images
    }
    if mismatches and not allow_incomplete:
        details = "; ".join(
            f"{method}: missing={len(value['missing'])}, extra={len(value['extra'])}"
            for method, value in mismatches.items()
        )
        raise SystemExit(
            f"Methods were not evaluated on identical image sets ({details}). "
            "Re-run evaluation or pass --allow_incomplete_methods for diagnostics only."
        )
    return method_order, maps, mismatches


def paired_stats(
    method: str,
    reference: str,
    metric: str,
    maps,
    rng: np.random.Generator,
    bootstrap_samples: int,
    confidence: float,
) -> dict:
    common = sorted(set(maps[method]) & set(maps[reference]))
    pairs = [
        (maps[method][image].get(metric), maps[reference][image].get(metric))
        for image in common
    ]
    pairs = [(current, base) for current, base in pairs if current is not None and base is not None]
    if not pairs:
        return {}
    raw_delta = np.asarray([current - base for current, base in pairs], dtype=np.float64)
    improvements = raw_delta * METRIC_DIRECTIONS[metric]
    ci_low, ci_high = bootstrap_mean_ci(improvements, rng, bootstrap_samples, confidence)
    return {
        "method": method,
        "reference_method": reference,
        "metric": metric,
        "direction": "higher_is_better" if METRIC_DIRECTIONS[metric] > 0 else "lower_is_better",
        "num_paired_images": int(improvements.size),
        "mean_method_minus_reference": float(raw_delta.mean()),
        "mean_improvement": float(improvements.mean()),
        "improvement_ci_low": ci_low,
        "improvement_ci_high": ci_high,
        "win_rate": float(np.mean(improvements > 0.0)),
        "tie_rate": float(np.mean(improvements == 0.0)),
        "sign_flip_p_value": sign_flip_p_value(improvements, rng, bootstrap_samples),
    }


def summarize_training_logs(
    specifications: list[tuple[str, Path]], baseline_method: str
) -> list[dict]:
    rows = []
    for method, path in specifications:
        log_rows = read_csv(path)
        step_time = numeric(row.get("train_step_time_s", "") for row in log_rows)
        cache_time = numeric(row.get("residual_cache_time_s", "") for row in log_rows)
        peak_memory = numeric(row.get("train_peak_cuda_mem_mb", "") for row in log_rows)
        skipped = numeric(row.get("skipped_block_count", "") for row in log_rows)
        mean_step = float(step_time.mean()) if step_time.size else math.nan
        mean_cache = float(cache_time.mean()) if cache_time.size else 0.0
        rows.append(
            {
                "method": method,
                "train_steps": len(log_rows),
                "mean_train_step_time_s": mean_step,
                "mean_cache_time_s": mean_cache,
                "mean_end_to_end_step_time_s": mean_step + mean_cache,
                "peak_train_cuda_mem_mb": float(peak_memory.max()) if peak_memory.size else math.nan,
                "mean_skipped_blocks": float(skipped.mean()) if skipped.size else 0.0,
                "train_time_reduction_vs_baseline_pct": math.nan,
            }
        )
    baseline = next((row for row in rows if row["method"] == baseline_method), None)
    if baseline and baseline["mean_end_to_end_step_time_s"] > 0:
        reference = baseline["mean_end_to_end_step_time_s"]
        for row in rows:
            row["train_time_reduction_vs_baseline_pct"] = 100.0 * (
                reference - row["mean_end_to_end_step_time_s"]
            ) / reference
    return rows


def fmt(value, digits: int) -> str:
    if value in ("", None) or (isinstance(value, float) and not math.isfinite(value)):
        return ""
    return f"{float(value):.{digits}f}"


def write_markdown(path: Path, rows: list[dict]) -> None:
    headers = [
        "Method", "PSNR", "SSIM", "LPIPS", "PSNR gain retention",
        "Train time reduction", "Adapter MB",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(["---"] + ["---:"] * (len(headers) - 1)) + "|",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["method"]),
                    str(row["psnr_mean_std"]),
                    str(row["ssim_mean_std"]),
                    str(row["lpips_mean_std"]),
                    (fmt(row["psnr_gain_retention_pct"], 1) + "%")
                    if row["psnr_gain_retention_pct"] != "" else "",
                    (fmt(row["train_time_reduction_vs_baseline_pct"], 1) + "%")
                    if row["train_time_reduction_vs_baseline_pct"] != "" else "",
                    fmt(row["adapter_size_mb"], 3),
                ]
            )
            + " |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("per_image_csv")
    parser.add_argument("--summary_csv", default="")
    parser.add_argument("--output_dir", default="")
    parser.add_argument("--base_method", default="Base-DiT4SR")
    parser.add_argument("--all_lora_method", default="All-LoRA")
    parser.add_argument("--training_log", action="append", type=parse_labeled_path, default=[])
    parser.add_argument("--training_baseline_method", default="All-LoRA")
    parser.add_argument("--bootstrap_samples", type=int, default=10000)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--min_images", type=int, default=100)
    parser.add_argument("--allow_incomplete_methods", action="store_true")
    args = parser.parse_args()

    per_image_path = Path(args.per_image_csv)
    output_dir = Path(args.output_dir) if args.output_dir else per_image_path.parent / "paper_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = read_csv(per_image_path)
    if not rows:
        raise SystemExit(f"No rows found in {per_image_path}")
    method_order, maps, mismatches = build_metric_maps(rows, args.allow_incomplete_methods)
    for required_method in (args.base_method, args.all_lora_method):
        if required_method not in maps:
            raise SystemExit(f"Required method not found: {required_method}")

    unique_images = len(set().union(*(set(method_map) for method_map in maps.values())))
    if unique_images < args.min_images:
        print(
            f"WARNING: only {unique_images} unique images; use at least {args.min_images} "
            "independent test images for paper results."
        )

    rng = np.random.default_rng(args.seed)
    metric_rows = []
    for method in method_order:
        for metric in METRIC_DIRECTIONS:
            values = numeric(
                metrics.get(metric)
                for metrics in maps[method].values()
                if metrics.get(metric) is not None
            )
            if not values.size:
                continue
            ci_low, ci_high = bootstrap_mean_ci(
                values, rng, args.bootstrap_samples, args.confidence
            )
            metric_rows.append(
                {
                    "method": method,
                    "metric": metric,
                    "num_images": int(values.size),
                    "mean": float(values.mean()),
                    "sample_std": float(values.std(ddof=1)) if values.size > 1 else 0.0,
                    "mean_ci_low": ci_low,
                    "mean_ci_high": ci_high,
                }
            )

    paired_rows = []
    for reference in (args.base_method, args.all_lora_method):
        for method in method_order:
            if method == reference:
                continue
            for metric in METRIC_DIRECTIONS:
                result = paired_stats(
                    method, reference, metric, maps, rng,
                    args.bootstrap_samples, args.confidence,
                )
                if result:
                    paired_rows.append(result)

    summary_path = Path(args.summary_csv) if args.summary_csv else per_image_path.parent / "sr_metrics_summary.csv"
    evaluation_summary = {
        row["method"]: row for row in read_csv(summary_path)
    } if summary_path.is_file() else {}
    training_rows = summarize_training_logs(args.training_log, args.training_baseline_method)
    training_summary = {row["method"]: row for row in training_rows}
    metric_lookup = {(row["method"], row["metric"]): row for row in metric_rows}
    paired_lookup = {
        (row["method"], row["reference_method"], row["metric"]): row
        for row in paired_rows
    }
    all_improvements = {
        metric: paired_lookup.get((args.all_lora_method, args.base_method, metric), {}).get(
            "mean_improvement", math.nan
        )
        for metric in METRIC_DIRECTIONS
    }

    paper_rows = []
    for method in method_order:
        output = {"method": method, "num_images": len(maps[method])}
        for metric, digits in (("psnr", 4), ("ssim", 5), ("lpips", 5)):
            stats = metric_lookup.get((method, metric), {})
            mean_value = stats.get("mean", math.nan)
            std_value = stats.get("sample_std", math.nan)
            output[f"mean_{metric}"] = mean_value
            output[f"std_{metric}"] = std_value
            output[f"{metric}_ci_low"] = stats.get("mean_ci_low", math.nan)
            output[f"{metric}_ci_high"] = stats.get("mean_ci_high", math.nan)
            output[f"{metric}_mean_std"] = (
                f"{mean_value:.{digits}f} +/- {std_value:.{digits}f}"
                if math.isfinite(mean_value) else ""
            )
            paired = paired_lookup.get((method, args.base_method, metric), {})
            improvement = 0.0 if method == args.base_method else paired.get("mean_improvement", math.nan)
            output[f"{metric}_improvement_vs_base"] = improvement
            denominator = all_improvements[metric]
            output[f"{metric}_gain_retention_pct"] = (
                100.0 * improvement / denominator
                if math.isfinite(improvement) and math.isfinite(denominator) and denominator > 0
                else ""
            )

        eval_row = evaluation_summary.get(method, {})
        train_row = training_summary.get(method, {})
        output["mean_inference_time_s"] = eval_row.get("mean_inference_time_s", "")
        output["peak_inference_cuda_mem_mb"] = eval_row.get("peak_cuda_mem_mb", "")
        output["adapter_size_mb"] = eval_row.get("adapter_size_mb", "")
        output["mean_train_step_time_s"] = train_row.get("mean_train_step_time_s", "")
        output["mean_end_to_end_step_time_s"] = train_row.get(
            "mean_end_to_end_step_time_s", ""
        )
        output["peak_train_cuda_mem_mb"] = train_row.get("peak_train_cuda_mem_mb", "")
        output["mean_skipped_blocks"] = train_row.get("mean_skipped_blocks", "")
        output["train_time_reduction_vs_baseline_pct"] = train_row.get(
            "train_time_reduction_vs_baseline_pct", ""
        )
        paper_rows.append(output)

    write_csv(output_dir / "metric_confidence_intervals.csv", metric_rows)
    write_csv(output_dir / "paired_comparisons.csv", paired_rows)
    if training_rows:
        write_csv(output_dir / "training_efficiency.csv", training_rows)
    write_csv(output_dir / "paper_ready_table.csv", paper_rows)
    write_markdown(output_dir / "paper_ready_table.md", paper_rows)
    metadata = {
        "per_image_csv": str(per_image_path),
        "summary_csv": str(summary_path) if summary_path.is_file() else "",
        "methods": method_order,
        "num_unique_images": unique_images,
        "base_method": args.base_method,
        "all_lora_method": args.all_lora_method,
        "bootstrap_samples": args.bootstrap_samples,
        "confidence": args.confidence,
        "seed": args.seed,
        "method_image_set_mismatches": mismatches,
    }
    (output_dir / "analysis_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(f"Wrote paper-ready results to {output_dir}")


if __name__ == "__main__":
    main()
