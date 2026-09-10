#!/usr/bin/env python3
"""Build a strict score-threshold bypass schedule matched to a B8 target."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

def read_csv(path: str) -> list[dict]:
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def selected_blocks(path: str) -> list[str]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    blocks = payload.get("selected_blocks", payload.get("selected_lora_blocks"))
    if not isinstance(blocks, list) or not blocks:
        raise ValueError("Selection JSON needs selected_blocks or selected_lora_blocks")
    return [str(block) for block in blocks]


def schedule_text(schedule: dict[float, int]) -> str:
    return " ".join(f"{noise:g}:{schedule[noise]}" for noise in sorted(schedule))


def rows_at_noise(rows: list[dict], noise: float) -> list[dict]:
    return [
        row for row in rows
        if math.isclose(float(row["noise_ratio"]), noise, abs_tol=1e-8)
    ]


def normalized_score_shares(
    rows: list[dict], noise: float, score_key: str, protected: set[str]
) -> dict[str, float]:
    available = [
        row for row in rows_at_noise(rows, noise)
        if str(row["block"]) not in protected
    ]
    scores = {str(row["block"]): max(float(row[score_key]), 0.0) for row in available}
    total = sum(scores.values())
    if total <= 0.0:
        raise ValueError(f"Non-positive score sum at noise={noise:g}")
    return {block: score / total for block, score in scores.items()}


def exact_policy(
    rows: list[dict], train_step: int, noise: float, count: int,
    min_run: int, max_run: int, max_runs: int, score_key: str,
    protected: set[str],
) -> list[str] | None:
    ordered = sorted(
        (
            row for row in rows
            if int(row["train_step"]) == train_step
            and math.isclose(float(row["noise_ratio"]), noise, abs_tol=1e-8)
            and str(row["block"]) not in protected
        ),
        key=lambda row: int(row["block_index"]),
    )
    if count == 0:
        return []
    if count > len(ordered):
        return None

    def group_name(block: str) -> str:
        parts = block.split(".")
        if parts[0] in {"input_blocks", "output_blocks"} and len(parts) > 1:
            return ".".join(parts[:2])
        if parts[0] == "middle_block":
            return "middle_block"
        return parts[0]

    states: dict[tuple[int, int, int], tuple[float, tuple[int, ...]]] = {
        (0, 0, 0): (0.0, ())
    }
    for index, row in enumerate(ordered):
        group_changed = (
            index > 0
            and (
                group_name(str(row["block"]))
                != group_name(str(ordered[index - 1]["block"]))
                or int(row["block_index"])
                != int(ordered[index - 1]["block_index"]) + 1
            )
        )
        next_states: dict[tuple[int, int, int], tuple[float, tuple[int, ...]]] = {}

        def update(key, value):
            previous = next_states.get(key)
            if previous is None or value[0] < previous[0]:
                next_states[key] = value

        for (used, runs, original_length), (score, chosen) in states.items():
            run_length = original_length
            if group_changed and run_length > 0:
                if run_length < min_run:
                    continue
                run_length = 0
            if run_length == 0 or run_length >= min_run:
                update((used, runs, 0), (score, chosen))
            if used >= count:
                continue
            row_score = float(row[score_key])
            if run_length > 0 and run_length < max_run:
                update(
                    (used + 1, runs, run_length + 1),
                    (score + row_score, chosen + (index,)),
                )
            elif run_length == 0 and runs < max_runs:
                update(
                    (used + 1, runs + 1, 1),
                    (score + row_score, chosen + (index,)),
                )
        states = next_states

    candidates = [
        value
        for (used, _runs, length), value in states.items()
        if used == count and (length == 0 or length >= min_run)
    ]
    if not candidates:
        return None
    _, chosen = min(candidates, key=lambda item: item[0])
    return [str(ordered[index]["block"]) for index in chosen]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--importance_csv", required=True)
    parser.add_argument("--importance_step", type=int, default=0)
    parser.add_argument("--cost_csv", required=True)
    parser.add_argument("--selection_file", required=True)
    parser.add_argument("--protected_block", action="append", default=[])
    parser.add_argument("--score_key", default="normalized_grad_score")
    parser.add_argument(
        "--thresholds", type=float, nargs="+",
        default=[0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05, 0.06],
        help="Maximum per-noise score share allowed for every bypassed block.",
    )
    parser.add_argument("--target_budget", type=int, default=8)
    parser.add_argument("--min_count", type=int, default=0)
    parser.add_argument("--max_count", type=int, default=12)
    parser.add_argument("--min_run", type=int, default=2)
    parser.add_argument("--max_run", type=int, default=6)
    parser.add_argument("--max_runs", type=int, default=3)
    parser.add_argument(
        "--allow_constant_schedule", action="store_true",
        help="Allow the recommended threshold to produce the same count at every noise anchor.",
    )
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    if not 0 <= args.min_count <= args.max_count:
        raise SystemExit("Require 0 <= --min_count <= --max_count")
    if any(value < 0.0 for value in args.thresholds):
        raise SystemExit("--thresholds must be non-negative")

    all_rows = read_csv(args.importance_csv)
    required = {"train_step", "noise_ratio", "block", "block_index", args.score_key}
    missing = required - set(all_rows[0])
    if missing:
        raise SystemExit("Importance CSV is missing columns: " + ", ".join(sorted(missing)))
    rows = [row for row in all_rows if int(row["train_step"]) == args.importance_step]
    if not rows:
        raise SystemExit(f"No importance rows at train_step={args.importance_step}")
    noises = sorted({float(row["noise_ratio"]) for row in rows})
    protected = set(selected_blocks(args.selection_file)) | set(args.protected_block)
    costs = {
        str(row["block"]): float(row["reported_gflops_saved"])
        for row in read_csv(args.cost_csv)
    }

    policies: dict[tuple[float, int], list[str] | None] = {}
    for noise in noises:
        for count in range(0, args.max_count + 1):
            policies[(noise, count)] = exact_policy(
                rows, args.importance_step, noise, count,
                args.min_run, args.max_run, args.max_runs,
                args.score_key, protected,
            )

    target_costs = []
    for noise in noises:
        blocks = policies.get((noise, args.target_budget))
        if blocks is None:
            raise SystemExit(
                f"Target budget {args.target_budget} is infeasible at noise={noise:g}"
            )
        target_costs.append(sum(costs[block] for block in blocks))
    target_saved = sum(target_costs) / len(target_costs)

    scan_rows = []
    candidates = []
    candidate_details: dict[float, list[dict]] = {}
    for threshold in sorted(set(args.thresholds)):
        details = []
        schedule = {}
        for noise in noises:
            shares = normalized_score_shares(rows, noise, args.score_key, protected)
            feasible = []
            for count in range(args.min_count, args.max_count + 1):
                blocks = policies[(noise, count)]
                if blocks is None:
                    continue
                if all(shares[block] <= threshold + 1e-15 for block in blocks):
                    feasible.append((count, blocks))
            if feasible:
                count, blocks = max(feasible, key=lambda item: item[0])
            else:
                count, blocks = 0, []
            saved = sum(costs[block] for block in blocks)
            schedule[noise] = count
            details.append({
                "threshold": threshold,
                "noise_ratio": noise,
                "bypass_budget": count,
                "estimated_saved_gflops": saved,
                "max_selected_score_share": max((shares[block] for block in blocks), default=0.0),
                "skip_blocks": ";".join(blocks),
            })
        mean_saved = sum(row["estimated_saved_gflops"] for row in details) / len(details)
        difference = (mean_saved / target_saved - 1.0) * 100.0
        is_constant = len(set(schedule.values())) == 1
        scan_rows.append({
            "threshold": threshold,
            "mean_bypass_budget": sum(schedule.values()) / len(schedule),
            "min_bypass_budget": min(schedule.values()),
            "max_bypass_budget": max(schedule.values()),
            "is_constant_schedule": is_constant,
            "estimated_saved_gflops_per_step": mean_saved,
            "difference_vs_fixed_b8_pct": difference,
            "schedule": schedule_text(schedule),
        })
        candidate_details[threshold] = details
        if args.allow_constant_schedule or not is_constant:
            candidates.append((abs(difference), threshold))

    if not candidates:
        raise SystemExit(
            "No non-constant threshold schedule was found; extend --thresholds or "
            "use --allow_constant_schedule"
        )
    _, chosen_threshold = min(candidates)
    chosen_scan = next(row for row in scan_rows if row["threshold"] == chosen_threshold)
    chosen_details = candidate_details[chosen_threshold]

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "threshold_scan.csv", scan_rows)
    write_csv(output / "threshold_policy_by_noise.csv", chosen_details)
    report = {
        "score_normalization": "per-noise share over unprotected candidate blocks",
        "score_key": args.score_key,
        "chosen_threshold": chosen_threshold,
        "target_budget": args.target_budget,
        "target_saved_gflops_per_step": target_saved,
        "threshold_saved_gflops_per_step": chosen_scan["estimated_saved_gflops_per_step"],
        "difference_vs_fixed_b8_pct": chosen_scan["difference_vs_fixed_b8_pct"],
        "threshold_schedule": chosen_scan["schedule"],
        "protected_blocks": sorted(protected),
        "min_count": args.min_count,
        "max_count": args.max_count,
        "min_run": args.min_run,
        "max_run": args.max_run,
        "max_runs": args.max_runs,
    }
    (output / "threshold_schedule.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    (output / "threshold_schedule.txt").write_text(
        str(chosen_scan["schedule"]) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
