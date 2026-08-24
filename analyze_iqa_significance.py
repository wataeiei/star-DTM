#!/usr/bin/env python3
"""Run paper-ready paired significance tests for per-image IQA scores."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np


DEFAULT_METRIC_ORDER = ["lpips", "musiq", "maniqa", "clipiqa", "liqe"]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise SystemExit(f"Empty CSV: {path}")
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parse_bool(value: str) -> bool:
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no"}:
        return False
    raise SystemExit(f"Invalid Boolean value in lower_better column: {value!r}")


def bootstrap_mean_ci(
    values: np.ndarray,
    rng: np.random.Generator,
    samples: int,
    confidence: float,
) -> tuple[float, float]:
    if values.size == 0:
        return math.nan, math.nan
    if values.size == 1 or samples <= 0:
        mean = float(values.mean())
        return mean, mean
    bootstrap_means = np.empty(samples, dtype=np.float64)
    chunk_size = min(1000, samples)
    for start in range(0, samples, chunk_size):
        stop = min(samples, start + chunk_size)
        indices = rng.integers(0, values.size, size=(stop - start, values.size))
        bootstrap_means[start:stop] = values[indices].mean(axis=1)
    alpha = (1.0 - confidence) * 50.0
    low, high = np.percentile(bootstrap_means, [alpha, 100.0 - alpha])
    return float(low), float(high)


def holm_adjust(p_values: list[float]) -> list[float]:
    adjusted = [math.nan] * len(p_values)
    finite = [(index, value) for index, value in enumerate(p_values) if math.isfinite(value)]
    finite.sort(key=lambda item: item[1])
    running = 0.0
    total = len(finite)
    for rank, (original_index, p_value) in enumerate(finite):
        candidate = min(1.0, (total - rank) * p_value)
        running = max(running, candidate)
        adjusted[original_index] = running
    return adjusted


def wilcoxon_test(improvements: np.ndarray) -> tuple[float, float, float]:
    try:
        from scipy.stats import rankdata, wilcoxon
    except Exception as error:
        raise SystemExit(
            "SciPy is required for the Wilcoxon test. Install it with "
            "`python3 -m pip install scipy`. Original error: " + repr(error)
        ) from error

    nonzero = improvements[improvements != 0.0]
    if nonzero.size == 0:
        return 0.0, 1.0, 0.0
    result = wilcoxon(
        nonzero,
        zero_method="wilcox",
        correction=False,
        alternative="two-sided",
        method="auto",
    )
    ranks = rankdata(np.abs(nonzero), method="average")
    positive = float(ranks[nonzero > 0].sum())
    negative = float(ranks[nonzero < 0].sum())
    denominator = positive + negative
    rank_biserial = (positive - negative) / denominator if denominator else 0.0
    return float(result.statistic), float(result.pvalue), rank_biserial


def build_score_maps(rows: list[dict[str, str]]):
    required = {"method", "filename", "metric", "score", "lower_better"}
    missing = required - set(rows[0])
    if missing:
        raise SystemExit(f"Input CSV is missing columns: {sorted(missing)}")

    score_maps: dict[str, dict[str, dict[str, float]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    directions: dict[str, bool] = {}
    method_order = []
    metric_order = []
    class_names: dict[str, str] = {}
    for row in rows:
        method = row["method"]
        metric = row["metric"]
        filename = row["filename"]
        if method not in method_order:
            method_order.append(method)
        if metric not in metric_order:
            metric_order.append(metric)
        lower_better = parse_bool(row["lower_better"])
        if metric in directions and directions[metric] != lower_better:
            raise SystemExit(f"Inconsistent lower_better direction for metric {metric}")
        directions[metric] = lower_better
        if filename in score_maps[method][metric]:
            raise SystemExit(f"Duplicate score: {method} / {metric} / {filename}")
        score = float(row["score"])
        if not math.isfinite(score):
            raise SystemExit(f"Non-finite score: {method} / {metric} / {filename}")
        score_maps[method][metric][filename] = score
        if row.get("class_name"):
            class_names[filename] = row["class_name"]
    return method_order, metric_order, score_maps, directions, class_names


def ordered_metrics(found: list[str]) -> list[str]:
    preferred = [metric for metric in DEFAULT_METRIC_ORDER if metric in found]
    return preferred + sorted(set(found) - set(preferred))


def validate_pairing(
    target: str,
    references: list[str],
    metrics: list[str],
    score_maps,
    allow_incomplete: bool,
) -> list[dict]:
    diagnostics = []
    for reference in references:
        for metric in metrics:
            target_images = set(score_maps[target][metric])
            reference_images = set(score_maps[reference][metric])
            missing = sorted(reference_images - target_images)
            extra = sorted(target_images - reference_images)
            diagnostics.append(
                {
                    "target_method": target,
                    "reference_method": reference,
                    "metric": metric,
                    "target_images": len(target_images),
                    "reference_images": len(reference_images),
                    "common_images": len(target_images & reference_images),
                    "missing_from_target": len(missing),
                    "extra_in_target": len(extra),
                }
            )
            if (missing or extra) and not allow_incomplete:
                raise SystemExit(
                    f"Image sets differ for {target} vs {reference}, metric={metric}: "
                    f"missing={len(missing)}, extra={len(extra)}"
                )
    return diagnostics


def analyze_pair(
    target: str,
    reference: str,
    metric: str,
    target_scores: dict[str, float],
    reference_scores: dict[str, float],
    lower_better: bool,
    rng: np.random.Generator,
    args: argparse.Namespace,
) -> tuple[dict, list[dict]]:
    filenames = sorted(set(target_scores) & set(reference_scores))
    target_values = np.asarray([target_scores[name] for name in filenames], dtype=np.float64)
    reference_values = np.asarray(
        [reference_scores[name] for name in filenames], dtype=np.float64
    )
    raw_delta = target_values - reference_values
    improvements = -raw_delta if lower_better else raw_delta
    ci_low, ci_high = bootstrap_mean_ci(
        improvements, rng, args.bootstrap_samples, args.confidence
    )
    statistic, p_value, rank_biserial = wilcoxon_test(improvements)
    tolerance = args.tie_tolerance
    wins = int(np.count_nonzero(improvements > tolerance))
    losses = int(np.count_nonzero(improvements < -tolerance))
    ties = int(improvements.size - wins - losses)
    reference_mean = float(reference_values.mean())
    relative_improvement = (
        100.0 * float(improvements.mean()) / abs(reference_mean)
        if reference_mean != 0.0
        else math.nan
    )
    summary = {
        "target_method": target,
        "reference_method": reference,
        "metric": metric,
        "direction": "lower_is_better" if lower_better else "higher_is_better",
        "num_paired_images": len(filenames),
        "target_mean": float(target_values.mean()),
        "reference_mean": reference_mean,
        "mean_target_minus_reference": float(raw_delta.mean()),
        "mean_improvement": float(improvements.mean()),
        "relative_improvement_pct": relative_improvement,
        "median_improvement": float(np.median(improvements)),
        "bootstrap_ci_low": ci_low,
        "bootstrap_ci_high": ci_high,
        "bootstrap_ci_excludes_zero": bool(ci_low > 0.0 or ci_high < 0.0),
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "win_rate": wins / len(filenames),
        "wilcoxon_statistic": statistic,
        "wilcoxon_p_raw": p_value,
        "wilcoxon_p_holm": math.nan,
        "significant_holm_0_05": False,
        "rank_biserial_effect": rank_biserial,
    }
    per_image = [
        {
            "target_method": target,
            "reference_method": reference,
            "metric": metric,
            "filename": filename,
            "target_score": target,
            "reference_score": reference_value,
            "raw_target_minus_reference": raw,
            "improvement_positive_is_better": improvement,
        }
        for filename, target, reference_value, raw, improvement in zip(
            filenames, target_values, reference_values, raw_delta, improvements
        )
    ]
    return summary, per_image


def write_markdown(path: Path, rows: list[dict]) -> None:
    lines = [
        "| Comparison | Metric | Improvement | 95% CI | Wins/Losses/Ties | Holm p | Effect |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    f"{row['target_method']} vs {row['reference_method']}",
                    str(row["metric"]),
                    f"{float(row['mean_improvement']):.6f}",
                    f"[{float(row['bootstrap_ci_low']):.6f}, {float(row['bootstrap_ci_high']):.6f}]",
                    f"{row['wins']}/{row['losses']}/{row['ties']}",
                    f"{float(row['wilcoxon_p_holm']):.3e}",
                    f"{float(row['rank_biserial_effect']):.3f}",
                ]
            )
            + " |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("per_image_csv")
    parser.add_argument("--target_method", default="All-LoRA-1000")
    parser.add_argument(
        "--reference_method",
        action="append",
        default=[],
        help="Reference method; repeat for multiple comparisons.",
    )
    parser.add_argument("--output_dir", default="")
    parser.add_argument("--bootstrap_samples", type=int, default=10000)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--tie_tolerance", type=float, default=1e-12)
    parser.add_argument("--allow_incomplete", action="store_true")
    args = parser.parse_args()

    input_path = Path(args.per_image_csv)
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else input_path.parent / "significance_analysis"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    references = args.reference_method or ["Base-DiT4SR", "Bicubic"]

    rows = read_csv(input_path)
    method_order, found_metrics, score_maps, directions, _classes = build_score_maps(rows)
    metrics = ordered_metrics(found_metrics)
    for method in [args.target_method, *references]:
        if method not in score_maps:
            raise SystemExit(f"Method not found: {method}; available={method_order}")
        missing_metrics = [metric for metric in metrics if metric not in score_maps[method]]
        if missing_metrics:
            raise SystemExit(f"Method {method} is missing metrics: {missing_metrics}")

    pairing_rows = validate_pairing(
        args.target_method, references, metrics, score_maps, args.allow_incomplete
    )
    rng = np.random.default_rng(args.seed)
    comparison_rows = []
    per_image_rows = []
    for reference in references:
        for metric in metrics:
            summary, details = analyze_pair(
                args.target_method,
                reference,
                metric,
                score_maps[args.target_method][metric],
                score_maps[reference][metric],
                directions[metric],
                rng,
                args,
            )
            comparison_rows.append(summary)
            per_image_rows.extend(details)

    adjusted = holm_adjust(
        [float(row["wilcoxon_p_raw"]) for row in comparison_rows]
    )
    for row, adjusted_p in zip(comparison_rows, adjusted):
        row["wilcoxon_p_holm"] = adjusted_p
        row["significant_holm_0_05"] = bool(adjusted_p < 0.05)

    write_csv(output_dir / "iqa_paired_significance.csv", comparison_rows)
    write_csv(output_dir / "iqa_per_image_improvements.csv", per_image_rows)
    write_csv(output_dir / "pairing_audit.csv", pairing_rows)
    write_markdown(output_dir / "iqa_paired_significance.md", comparison_rows)
    metadata = {
        "input_csv": str(input_path),
        "target_method": args.target_method,
        "reference_methods": references,
        "metrics": metrics,
        "metric_directions": {
            metric: ("lower_is_better" if directions[metric] else "higher_is_better")
            for metric in metrics
        },
        "bootstrap_samples": args.bootstrap_samples,
        "confidence": args.confidence,
        "bootstrap_method": "paired percentile bootstrap of mean improvement",
        "statistical_test": "two-sided Wilcoxon signed-rank",
        "multiple_testing": f"Holm correction across {len(comparison_rows)} tests",
        "seed": args.seed,
    }
    (output_dir / "analysis_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(f"Wrote paired IQA statistics to {output_dir}")
    for row in comparison_rows:
        print(
            f"{row['target_method']} vs {row['reference_method']} "
            f"{row['metric']}: improvement={row['mean_improvement']:.6f} "
            f"CI=[{row['bootstrap_ci_low']:.6f}, {row['bootstrap_ci_high']:.6f}] "
            f"W/L/T={row['wins']}/{row['losses']}/{row['ties']} "
            f"p_holm={row['wilcoxon_p_holm']:.3e}"
        )


if __name__ == "__main__":
    main()
