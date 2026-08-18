#!/usr/bin/env python3
"""Summarize Grad-BlockSkip correctness, memory, timing, and selected blocks."""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path


def numeric(rows, key):
    values = []
    for row in rows:
        value = row.get(key, "")
        if value not in ("", None):
            values.append(float(value))
    return values


def mean(values):
    return sum(values) / len(values) if values else 0.0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("train_log")
    parser.add_argument("--baseline_log", default="")
    parser.add_argument("--top_blocks", type=int, default=12)
    args = parser.parse_args()

    path = Path(args.train_log)
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise SystemExit(f"No rows found in {path}")

    fallbacks = numeric(rows, "fallback_blocks")
    replayable = numeric(rows, "replayable_blocks")
    skipped = numeric(rows, "skipped_block_count")
    loss_diff = numeric(rows, "residual_loss_abs_diff")
    forward_diff = numeric(rows, "residual_forward_max_abs_diff")
    cache_time = numeric(rows, "residual_cache_time_s")
    train_time = numeric(rows, "train_step_time_s")
    train_peak = numeric(rows, "train_peak_cuda_mem_mb")
    cache_peak = numeric(rows, "cache_peak_cuda_mem_mb")
    cache_mb = numeric(rows, "residual_cache_mb")

    frequency = Counter()
    fallback_frequency = Counter()
    for row in rows:
        frequency.update(block for block in row.get("skipped_blocks", "").split(";") if block)
        fallback_frequency.update(
            block for block in row.get("fallback_block_names", "").split(";") if block
        )

    print(f"steps: {len(rows)}")
    print(f"mean skipped blocks: {mean(skipped):.2f}")
    print(f"mean replayable blocks: {mean(replayable):.2f}")
    print(f"total fallback block-events: {sum(fallbacks):.0f}")
    print(f"mean residual loss abs diff: {mean(loss_diff):.8g}")
    print(f"max residual loss abs diff: {max(loss_diff, default=0.0):.8g}")
    if forward_diff:
        print(f"max residual forward abs diff: {max(forward_diff):.8g}")
    print(f"mean cache time s: {mean(cache_time):.4f}")
    print(f"mean train step time s: {mean(train_time):.4f}")
    print(f"mean residual cache MB: {mean(cache_mb):.2f}")
    print(f"max cache-forward CUDA MB: {max(cache_peak, default=0.0):.1f}")
    print(f"max train CUDA MB: {max(train_peak, default=0.0):.1f}")
    print("most frequently skipped blocks:")
    for block, count in frequency.most_common(args.top_blocks):
        print(f"  {block}: {count}/{len(rows)} steps")
    if fallback_frequency:
        print("fallback blocks:")
        for block, count in fallback_frequency.most_common():
            print(f"  {block}: {count}")

    if args.baseline_log:
        with Path(args.baseline_log).open(newline="", encoding="utf-8") as handle:
            baseline = list(csv.DictReader(handle))
        baseline_time = mean(numeric(baseline, "train_step_time_s"))
        baseline_peak = max(numeric(baseline, "train_peak_cuda_mem_mb"), default=0.0)
        current_time = mean(train_time)
        current_peak = max(train_peak, default=0.0)
        online_total = current_time + mean(cache_time)

        def reduction(current, reference):
            return 100.0 * (reference - current) / reference if reference else 0.0

        print("comparison with baseline:")
        print(f"  baseline train step s: {baseline_time:.4f}")
        print(f"  residual-skip train step s: {current_time:.4f}")
        print(f"  online cache + train s: {online_total:.4f}")
        print(f"  train-step speed change: {reduction(current_time, baseline_time):+.2f}% reduction")
        print(f"  baseline peak CUDA MB: {baseline_peak:.1f}")
        print(f"  residual-skip peak CUDA MB: {current_peak:.1f}")
        print(f"  peak CUDA reduction: {reduction(current_peak, baseline_peak):+.2f}%")


if __name__ == "__main__":
    main()
