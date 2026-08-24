#!/usr/bin/env python3
"""Paired significance tests for SR quality and semantic preservation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest, wilcoxon


DIRECTIONS = {
    "psnr": "higher",
    "ssim": "higher",
    "lpips": "lower",
    "musiq": "higher",
    "maniqa": "higher",
    "clipiqa": "higher",
    "liqe": "higher",
}


def read_many(paths: list[str]) -> pd.DataFrame:
    if not paths:
        return pd.DataFrame()
    return pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)


def bootstrap_mean_ci(
    values: np.ndarray, rng: np.random.Generator, samples: int
) -> tuple[float, float]:
    if len(values) == 0:
        return float("nan"), float("nan")
    means = np.empty(samples, dtype=np.float64)
    for start in range(0, samples, 1000):
        count = min(1000, samples - start)
        indices = rng.integers(0, len(values), size=(count, len(values)))
        means[start : start + count] = values[indices].mean(axis=1)
    return tuple(np.quantile(means, [0.025, 0.975]).tolist())


def holm_adjust(p_values: list[float]) -> list[float]:
    order = np.argsort(p_values)
    adjusted = np.empty(len(p_values), dtype=np.float64)
    running = 0.0
    total = len(p_values)
    for rank, index in enumerate(order):
        value = min(1.0, (total - rank) * float(p_values[index]))
        running = max(running, value)
        adjusted[index] = running
    return adjusted.tolist()


def paired_metric(
    frame: pd.DataFrame,
    key: str,
    metric: str,
    method_a: str,
    method_b: str,
    rng: np.random.Generator,
    bootstrap_samples: int,
) -> dict:
    selected = frame[frame["method"].isin([method_a, method_b])].copy()
    selected[metric] = pd.to_numeric(selected[metric], errors="coerce")
    selected = selected.dropna(subset=[metric])
    selected = selected.drop_duplicates(["method", key], keep="last")
    pivot = selected.pivot(index=key, columns="method", values=metric).dropna()
    if method_a not in pivot or method_b not in pivot:
        raise SystemExit(f"Missing paired {metric} values for {method_a} or {method_b}")

    a = pivot[method_a].to_numpy(dtype=np.float64)
    b = pivot[method_b].to_numpy(dtype=np.float64)
    difference = b - a
    if np.allclose(difference, 0.0):
        statistic, p_value = 0.0, 1.0
    else:
        result = wilcoxon(difference, zero_method="wilcox", alternative="two-sided")
        statistic, p_value = float(result.statistic), float(result.pvalue)
    ci_low, ci_high = bootstrap_mean_ci(difference, rng, bootstrap_samples)
    direction = DIRECTIONS[metric]
    better = difference > 0 if direction == "higher" else difference < 0
    worse = difference < 0 if direction == "higher" else difference > 0
    return {
        "metric": metric,
        "direction": direction,
        "num_pairs": len(pivot),
        f"mean_{method_a}": a.mean(),
        f"mean_{method_b}": b.mean(),
        "mean_difference_b_minus_a": difference.mean(),
        "median_difference_b_minus_a": np.median(difference),
        "mean_difference_ci95_low": ci_low,
        "mean_difference_ci95_high": ci_high,
        "b_better_count": int(better.sum()),
        "a_better_count": int(worse.sum()),
        "tie_count": int((difference == 0).sum()),
        "wilcoxon_statistic": statistic,
        "p_value": p_value,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sr_csv", action="append", default=[])
    parser.add_argument("--iqa_csv", action="append", default=[])
    parser.add_argument("--semantic_csv", action="append", default=[])
    parser.add_argument("--method_a", required=True)
    parser.add_argument("--method_b", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--bootstrap_samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    tests = []

    sr = read_many(args.sr_csv)
    if not sr.empty:
        sr["filename"] = sr["image"].map(lambda value: Path(str(value)).name)
        for metric in ("psnr", "ssim"):
            tests.append(
                paired_metric(
                    sr, "filename", metric, args.method_a, args.method_b,
                    rng, args.bootstrap_samples
                )
            )

    iqa = read_many(args.iqa_csv)
    if not iqa.empty:
        iqa = iqa[iqa["metric"].isin(DIRECTIONS)].copy()
        wide = iqa.pivot_table(
            index=["method", "filename"], columns="metric", values="score", aggfunc="last"
        ).reset_index()
        for metric in ("lpips", "musiq", "maniqa", "clipiqa", "liqe"):
            if metric in wide:
                tests.append(
                    paired_metric(
                        wide, "filename", metric, args.method_a, args.method_b,
                        rng, args.bootstrap_samples
                    )
                )

    if tests:
        adjusted = holm_adjust([row["p_value"] for row in tests])
        for row, value in zip(tests, adjusted):
            row["holm_adjusted_p"] = value
            row["significant_0_05"] = value < 0.05
        pd.DataFrame(tests).to_csv(
            output_dir / "paired_continuous_tests.csv", index=False
        )

    semantic = read_many(args.semantic_csv)
    mcnemar_row = None
    if not semantic.empty:
        semantic = semantic.drop_duplicates(["method", "filename"], keep="last")
        selected = semantic[semantic["method"].isin([args.method_a, args.method_b])]
        pivot = selected.pivot(index="filename", columns="method", values="correct").dropna()
        if args.method_a not in pivot or args.method_b not in pivot:
            raise SystemExit("Missing paired semantic predictions for one of the methods")
        a = pivot[args.method_a].map(
            lambda value: str(value).lower() in {"1", "true"}
        ).to_numpy()
        b = pivot[args.method_b].map(
            lambda value: str(value).lower() in {"1", "true"}
        ).to_numpy()
        a_wins = int((a & ~b).sum())
        b_wins = int((~a & b).sum())
        discordant = a_wins + b_wins
        p_value = (
            float(binomtest(b_wins, discordant, 0.5).pvalue)
            if discordant
            else 1.0
        )
        accuracy_difference = b.astype(float) - a.astype(float)
        ci_low, ci_high = bootstrap_mean_ci(
            accuracy_difference, rng, args.bootstrap_samples
        )
        mcnemar_row = {
            "method_a": args.method_a,
            "method_b": args.method_b,
            "num_pairs": len(pivot),
            "accuracy_a": a.mean(),
            "accuracy_b": b.mean(),
            "accuracy_difference_b_minus_a": accuracy_difference.mean(),
            "accuracy_difference_ci95_low": ci_low,
            "accuracy_difference_ci95_high": ci_high,
            "a_correct_b_wrong": a_wins,
            "a_wrong_b_correct": b_wins,
            "discordant_pairs": discordant,
            "exact_mcnemar_p": p_value,
        }
        pd.DataFrame([mcnemar_row]).to_csv(
            output_dir / "mcnemar_test.csv", index=False
        )

    metadata = {
        **vars(args),
        "continuous_metrics_test": "paired two-sided Wilcoxon signed-rank",
        "multiple_testing_correction": "Holm",
        "semantic_test": "exact McNemar (two-sided binomial)",
        "continuous_metric_count": len(tests),
        "semantic_test_written": mcnemar_row is not None,
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(f"Wrote paired tests to {output_dir}")


if __name__ == "__main__":
    main()
