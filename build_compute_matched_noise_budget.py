#!/usr/bin/env python3
"""Build a noise-aware bypass schedule matched to a fixed-budget compute target."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
from pathlib import Path


def read_csv(path: str) -> list[dict]:
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def selected_blocks(path: str) -> list[str]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    blocks = payload.get("selected_blocks", payload.get("selected_lora_blocks"))
    if not isinstance(blocks, list) or not blocks:
        raise ValueError("Selection JSON needs selected_blocks or selected_lora_blocks")
    return [str(block) for block in blocks]


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def schedule_text(schedule: dict[float, int]) -> str:
    return " ".join(f"{noise:g}:{schedule[noise]}" for noise in sorted(schedule))


def select_low_score_runs(
    rows: list[dict], noise: float, count: int, min_run: int, max_run: int,
    max_runs: int, protected: set[str],
) -> list[str]:
    ordered = sorted(
        (row for row in rows
         if float(row["noise_ratio"]) == noise and row["block"] not in protected),
        key=lambda row: int(row["block_index"]),
    )
    states: dict[tuple[int, int, int], tuple[float, tuple[int, ...]]] = {
        (0, 0, 0): (0.0, ())
    }
    for index, row in enumerate(ordered):
        next_states = {}
        gap = (
            index > 0
            and int(row["block_index"]) != int(ordered[index - 1]["block_index"]) + 1
        )

        def update(key, value):
            previous = next_states.get(key)
            if previous is None or value < previous:
                next_states[key] = value

        for (used, runs, original_length), (score, chosen) in states.items():
            run_length = original_length
            if gap and run_length:
                if run_length < min_run:
                    continue
                run_length = 0
            if run_length == 0 or run_length >= min_run:
                update((used, runs, 0), (score, chosen))
            if used == count:
                continue
            row_score = float(row["normalized_grad_score"])
            if run_length and run_length < max_run:
                update((used + 1, runs, run_length + 1),
                       (score + row_score, chosen + (index,)))
            elif run_length == 0 and runs < max_runs:
                update((used + 1, runs + 1, 1),
                       (score + row_score, chosen + (index,)))
        states = next_states
    candidates = [value for (used, _runs, length), value in states.items()
                  if used == count and (length == 0 or length >= min_run)]
    if not candidates:
        raise ValueError(f"No feasible policy for noise={noise:g}, budget={count}")
    return [ordered[index]["block"] for index in min(candidates)[1]]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--importance_csv", required=True)
    parser.add_argument("--importance_step", type=int, default=0)
    parser.add_argument("--cost_csv", required=True)
    parser.add_argument("--fidelity_csv", required=True)
    parser.add_argument("--selection_file", required=True)
    parser.add_argument("--protected_block", action="append", default=[])
    parser.add_argument("--fixed_budget", type=int, default=8)
    parser.add_argument("--candidate_budgets", type=int, nargs="+", default=[6, 8, 10])
    parser.add_argument("--min_run", type=int, default=2)
    parser.add_argument("--max_run", type=int, default=6)
    parser.add_argument("--max_runs", type=int, default=3)
    parser.add_argument("--compute_tolerance_pct", type=float, default=1.0)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    importance = read_csv(args.importance_csv)
    importance = [row for row in importance if int(row["train_step"]) == args.importance_step]
    noises = sorted({float(row["noise_ratio"]) for row in importance})
    costs = {row["block"]: float(row["reported_gflops_saved"])
             for row in read_csv(args.cost_csv)}
    fidelity = {(float(row["noise_ratio"]), int(row["bypass_budget"])): row
                for row in read_csv(args.fidelity_csv)}
    protected = set(selected_blocks(args.selection_file)) | set(args.protected_block)
    budgets = sorted(set(args.candidate_budgets) | {args.fixed_budget})

    policies: dict[tuple[float, int], dict] = {}
    for noise in noises:
        for budget in budgets:
            key = (noise, budget)
            if key not in fidelity:
                raise ValueError(f"Missing fidelity row for noise={noise:g}, budget={budget}")
            blocks = select_low_score_runs(
                importance, noise, budget, args.min_run, args.max_run,
                args.max_runs, protected,
            )
            missing = sorted(set(blocks) - costs.keys())
            if missing:
                raise ValueError(f"Missing block costs: {missing}")
            policies[key] = {
                "blocks": blocks,
                "saved_gflops": sum(costs[block] for block in blocks),
                "penalty": float(fidelity[key]["mean_relative_gradient_error"]),
            }

    target = sum(policies[(noise, args.fixed_budget)]["saved_gflops"] for noise in noises) / len(noises)
    tolerance = target * args.compute_tolerance_pct / 100.0
    required_budget_sum = args.fixed_budget * len(noises)
    feasible = []
    for values in itertools.product(args.candidate_budgets, repeat=len(noises)):
        if sum(values) != required_budget_sum or len(set(values)) == 1:
            continue
        schedule = dict(zip(noises, values))
        saved = sum(policies[(noise, schedule[noise])]["saved_gflops"] for noise in noises) / len(noises)
        if abs(saved - target) > tolerance:
            continue
        penalty = sum(policies[(noise, schedule[noise])]["penalty"] for noise in noises) / len(noises)
        feasible.append((penalty, abs(saved - target), tuple(values), saved))
    if not feasible:
        raise ValueError("No compute-matched non-constant schedule; increase tolerance or candidate budgets")

    _, _, chosen_values, chosen_saved = min(feasible)
    chosen = dict(zip(noises, chosen_values))
    sensitivity = sorted(noises, key=lambda n: policies[(n, args.fixed_budget)]["penalty"], reverse=True)
    reverse_values = sorted(chosen_values, reverse=True)
    reverse = {noise: budget for noise, budget in zip(sensitivity, reverse_values)}

    rows = []
    for label, schedule in (("fixed", {n: args.fixed_budget for n in noises}),
                            ("noise_aware", chosen), ("reverse", reverse)):
        for noise in noises:
            policy = policies[(noise, schedule[noise])]
            rows.append({
                "method": label, "noise_ratio": noise, "bypass_budget": schedule[noise],
                "estimated_saved_gflops": policy["saved_gflops"],
                "relative_gradient_error": policy["penalty"],
                "skip_blocks": ";".join(policy["blocks"]),
            })

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "budget_policies.csv", rows)
    reverse_saved = sum(policies[(n, reverse[n])]["saved_gflops"] for n in noises) / len(noises)
    report = {
        "fixed_budget": args.fixed_budget,
        "target_saved_gflops_per_step": target,
        "compute_tolerance_pct": args.compute_tolerance_pct,
        "noise_aware_saved_gflops_per_step": chosen_saved,
        "noise_aware_difference_pct": (chosen_saved / target - 1.0) * 100.0,
        "reverse_saved_gflops_per_step": reverse_saved,
        "reverse_difference_pct": (reverse_saved / target - 1.0) * 100.0,
        "fixed_schedule": schedule_text({n: args.fixed_budget for n in noises}),
        "noise_aware_schedule": schedule_text(chosen),
        "reverse_schedule": schedule_text(reverse),
        "candidate_schedules_tested": len(feasible),
    }
    (output / "budget_schedule.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (output / "noise_aware_schedule.txt").write_text(report["noise_aware_schedule"] + "\n", encoding="utf-8")
    (output / "reverse_schedule.txt").write_text(report["reverse_schedule"] + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
