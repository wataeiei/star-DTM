#!/usr/bin/env python3
"""Summarize whether LoRA layer importance changes with diffusion noise ratio."""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path


def pearson(xs, ys):
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    dx, dy = [x - mx for x in xs], [y - my for y in ys]
    denom = math.sqrt(sum(x * x for x in dx) * sum(y * y for y in dy))
    return sum(x * y for x, y in zip(dx, dy)) / denom if denom else float("nan")


def read_rows(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path", help="lora_importance_evolution.csv")
    parser.add_argument("--output", default="", help="Defaults to timestep_importance_summary.csv")
    args = parser.parse_args()

    rows = read_rows(args.csv_path)
    required = {"train_step", "noise_ratio", "block", "importance_rank", "selected_topk"}
    missing = required - set(rows[0]) if rows else required
    if missing:
        raise SystemExit(f"Missing CSV columns: {sorted(missing)}")

    grouped = defaultdict(dict)
    for row in rows:
        key = (int(row["train_step"]), float(row["noise_ratio"]))
        grouped[key][row["block"]] = row

    summary = []
    for step in sorted({key[0] for key in grouped}):
        ratios = sorted(ratio for s, ratio in grouped if s == step)
        reference_ratio = ratios[0]
        reference = grouped[(step, reference_ratio)]
        reference_topk = {
            block for block, row in reference.items()
            if row["selected_topk"].lower() == "true"
        }
        for ratio in ratios:
            current = grouped[(step, ratio)]
            blocks = sorted(set(reference) & set(current))
            ranks_a = [float(reference[block]["importance_rank"]) for block in blocks]
            ranks_b = [float(current[block]["importance_rank"]) for block in blocks]
            scores_a = [float(reference[block]["normalized_grad_score"]) for block in blocks]
            scores_b = [float(current[block]["normalized_grad_score"]) for block in blocks]
            current_topk = {
                block for block, row in current.items()
                if row["selected_topk"].lower() == "true"
            }
            overlap = reference_topk & current_topk
            summary.append(
                {
                    "train_step": step,
                    "reference_noise_ratio": reference_ratio,
                    "noise_ratio": ratio,
                    "common_blocks": len(blocks),
                    "spearman_rank_vs_lowest_t": pearson(ranks_a, ranks_b),
                    "pearson_score_vs_lowest_t": pearson(scores_a, scores_b),
                    "topk_jaccard_vs_lowest_t": len(overlap)
                    / max(len(reference_topk | current_topk), 1),
                }
            )

    output = Path(args.output) if args.output else Path(args.csv_path).with_name(
        "timestep_importance_summary.csv"
    )
    if not summary:
        raise SystemExit("No timestep groups found in the input CSV.")
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
