#!/usr/bin/env python3
"""Build a fixed bypass policy from mean importance across noise anchors."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import adaptive_grad_blockskip as adaptive


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def contiguous_runs(selected: set[str], ordered: list[dict]) -> list[list[str]]:
    runs: list[list[str]] = []
    previous_index = None
    for row in ordered:
        block = str(row["block"])
        index = int(row["block_index"])
        if block not in selected:
            previous_index = None
            continue
        if previous_index is None or index != previous_index + 1:
            runs.append([])
        runs[-1].append(block)
        previous_index = index
    return runs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--importance_csv", required=True)
    parser.add_argument("--importance_step", type=int, default=0)
    parser.add_argument("--lora_selection_file", required=True)
    parser.add_argument("--reference_metadata", required=True)
    parser.add_argument("--skip_count", type=int, default=8)
    parser.add_argument("--protected_block", action="append", default=[])
    parser.add_argument("--cost_csv", default="")
    parser.add_argument("--score_key", default="normalized_grad_score")
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    rows = read_csv(Path(args.importance_csv))
    required = {"train_step", "noise_ratio", "block", "block_index", args.score_key}
    if not rows or required - set(rows[0]):
        raise SystemExit(
            "Importance CSV is empty or missing columns: "
            + ", ".join(sorted(required - (set(rows[0]) if rows else set())))
        )
    rows = [row for row in rows if int(row["train_step"]) == args.importance_step]
    if not rows:
        raise SystemExit(f"No importance rows at train step {args.importance_step}")

    anchors = sorted({float(row["noise_ratio"]) for row in rows})
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[str(row["block"])].append(row)

    averaged = []
    for block, block_rows in grouped.items():
        block_anchors = sorted(float(row["noise_ratio"]) for row in block_rows)
        indices = {int(row["block_index"]) for row in block_rows}
        if block_anchors != anchors or len(indices) != 1:
            raise SystemExit(f"Incomplete or inconsistent importance rows for {block}")
        averaged.append(
            {
                "train_step": args.importance_step,
                "noise_ratio": 0.0,
                "block": block,
                "block_index": indices.pop(),
                args.score_key: sum(float(row[args.score_key]) for row in block_rows)
                / len(block_rows),
            }
        )
    averaged.sort(key=lambda row: int(row["block_index"]))

    lora_payload = json.loads(
        Path(args.lora_selection_file).read_text(encoding="utf-8")
    )
    selected_lora = lora_payload.get("selected_blocks")
    if not isinstance(selected_lora, list) or not selected_lora:
        raise SystemExit("LoRA selection file has no selected_blocks list")
    reference = json.loads(Path(args.reference_metadata).read_text(encoding="utf-8"))
    min_run = int(reference["blockskip_min_run"])
    max_run = int(reference["blockskip_max_run"])
    max_runs = int(reference["blockskip_max_runs"])
    protected = set(selected_lora) | set(args.protected_block)

    selected = adaptive.select_low_score_runs(
        averaged,
        args.importance_step,
        0.0,
        args.skip_count,
        min_run,
        max_run,
        max_runs,
        score_key=args.score_key,
        excluded_blocks=protected,
    )
    selected_set = set(selected)
    runs = contiguous_runs(selected_set, averaged)
    if selected_set & protected:
        raise SystemExit("Generated policy intersects protected blocks")
    if len(selected) != args.skip_count:
        raise SystemExit("Generated policy has the wrong number of blocks")
    if len(runs) > max_runs or any(not min_run <= len(run) <= max_run for run in runs):
        raise SystemExit("Generated policy violates the reference run constraints")

    costs = {}
    estimated_gflops = None
    if args.cost_csv:
        for row in read_csv(Path(args.cost_csv)):
            costs[str(row["block"])] = float(row["reported_gflops_saved"])
        missing = selected_set - set(costs)
        if missing:
            raise SystemExit("Cost CSV is missing selected blocks: " + ", ".join(sorted(missing)))
        estimated_gflops = sum(costs[block] for block in selected)

    for row in averaged:
        row["protected"] = str(row["block"] in protected)
        row["selected_static_bypass"] = str(row["block"] in selected_set)
        row["reported_gflops_saved"] = costs.get(str(row["block"]), "")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "mean_importance_by_block.csv", averaged)
    (output_dir / "static_bypass_blocks.txt").write_text(
        " ".join(selected) + "\n", encoding="utf-8"
    )
    payload = {
        "policy": "mean-noise-importance-static-bypass",
        "importance_csv": args.importance_csv,
        "importance_step": args.importance_step,
        "noise_anchors": anchors,
        "score_key": args.score_key,
        "skip_count": args.skip_count,
        "min_run": min_run,
        "max_run": max_run,
        "max_runs": max_runs,
        "selected_lora_blocks": selected_lora,
        "additional_protected_blocks": args.protected_block,
        "selected_bypass_blocks": selected,
        "contiguous_runs": runs,
        "mean_importance_sum": sum(
            float(row[args.score_key]) for row in averaged
            if str(row["block"]) in selected_set
        ),
        "estimated_reported_gflops_saved": estimated_gflops,
    }
    (output_dir / "static_bypass_policy.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload, indent=2))
    print(f"Wrote static bypass policy to {output_dir}")


if __name__ == "__main__":
    main()
