#!/usr/bin/env python3
"""Compare logged bypass sets with exact compute-budget dynamic programming.

Uses only the Python standard library. No model loading or training is performed.
"""

import argparse
import csv
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import time


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def number(value):
    result = Decimal(str(value))
    if not result.is_finite():
        raise ValueError(f"Non-finite value: {value}")
    return result


def group_name(block):
    parts = block.split(".")
    if parts[0] in {"input_blocks", "output_blocks"}:
        return ".".join(parts[:2])
    return parts[0]


def adjacent(left, right):
    return (right["index"] == left["index"] + 1
            and group_name(left["block"]) == group_name(right["block"]))


def runs_for(chosen, candidates):
    selected = set(chosen)
    runs = []
    previous = None
    for row in candidates:
        if row["block"] not in selected:
            previous = None
            continue
        if previous is None or not adjacent(previous, row):
            runs.append([])
        runs[-1].append(row["block"])
        previous = row
    return runs


def solve(candidates, target, min_run, max_run, max_runs, max_states=1000000):
    """Exact DP over cost, run count and active length; costs are Decimal.

    Preserve actual cost above target as well, so ties favor less overshoot.
    At identical states only the minimum-importance prefix can be optimal.
    """
    if not 1 <= min_run <= max_run or max_runs < 1 or target < 0:
        raise ValueError("Invalid run constraints or compute target")
    # key: (saved cost, number of runs, active length); value: (score, indices)
    states = {(Decimal(0), 0, 0): (Decimal(0), ())}
    peak_states = 1
    for index, row in enumerate(candidates):
        gap = index > 0 and not adjacent(candidates[index - 1], row)
        next_states = {}

        def update(key, value):
            old = next_states.get(key)
            if old is None or (value[0], len(value[1]), value[1]) < (
                    old[0], len(old[1]), old[1]):
                next_states[key] = value

        for (cost, runs, length), (score, chosen) in states.items():
            if gap and length:
                if length < min_run:
                    continue
                length = 0
            if length == 0 or length >= min_run:
                update((cost, runs, 0), (score, chosen))
            if length and length < max_run:
                update((cost + row["cost"], runs, length + 1),
                       (score + row["score"], chosen + (index,)))
            elif length == 0 and runs < max_runs:
                update((cost + row["cost"], runs + 1, 1),
                       (score + row["score"], chosen + (index,)))
        states = next_states
        peak_states = max(peak_states, len(states))
        if len(states) > max_states:
            raise ValueError("DP state limit exceeded; no approximate policy was emitted")
    feasible = [
        (score, cost, len(chosen), chosen)
        for (cost, _runs, length), (score, chosen) in states.items()
        if cost >= target and (length == 0 or length >= min_run)
    ]
    if not feasible:
        raise ValueError("No feasible bypass set reaches the compute target")
    score, cost, _, chosen = min(feasible)
    return [candidates[i]["block"] for i in chosen], score, cost, peak_states


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cost_csv", required=True)
    parser.add_argument("--importance_csv", required=True)
    parser.add_argument("--importance_step", type=int, default=0)
    parser.add_argument("--baseline_run", action="append", required=True,
                        help="BUDGET=run_directory; repeat for B8 and B12")
    parser.add_argument("--protected_block", action="append", default=[])
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()
    started = time.perf_counter()

    costs = {}
    for row in read_csv(args.cost_csv):
        block = row["block"]
        if block in costs:
            raise ValueError(f"Duplicate cost row: {block}")
        value = number(row["reported_gflops_saved"])
        if value <= 0:
            raise ValueError(f"Non-positive saved FLOPs: {block}")
        if number(row["max_loss_abs_diff"]) != 0:
            raise ValueError(f"Forward loss differs in cost audit: {block}")
        costs[block] = value

    importance = {}
    for row in read_csv(args.importance_csv):
        if int(row["train_step"]) != args.importance_step:
            continue
        noise, block = number(row["noise_ratio"]), row["block"]
        bucket = importance.setdefault(noise, {})
        if block in bucket:
            raise ValueError(f"Duplicate importance row: {noise}, {block}")
        score = number(row["normalized_grad_score"])
        if score < 0:
            raise ValueError(f"Negative importance: {block}")
        bucket[block] = {"block": block, "index": int(row["block_index"]),
                         "score": score}
    if not importance or not costs:
        raise ValueError("Missing cost rows or requested importance step")

    comparisons, policy_rows, provenance = [], [], []
    seen_budgets = set()
    for spec in args.baseline_run:
        label, directory = spec.split("=", 1)
        budget, run_dir = int(label), Path(directory)
        if budget <= 0 or budget in seen_budgets:
            raise ValueError(f"Invalid or duplicate budget: {budget}")
        seen_budgets.add(budget)
        meta_path, log_path = run_dir / "metadata.json", run_dir / "train_log.csv"
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("always_skip_blocks") or meta.get("fixed_skip_blocks"):
            raise ValueError("This comparison requires DP baselines without mandatory or fixed skip blocks")
        constraints = [int(meta[key]) for key in
                       ("blockskip_min_run", "blockskip_max_run", "blockskip_max_runs")]
        min_run, max_run, max_runs = constraints
        protected = set(meta["selected_lora_blocks"])
        protected.update(meta.get("blockskip_protected_blocks", []))
        protected.update(args.protected_block)
        logged = {}
        for row in read_csv(log_path):
            noise = number(row["noise_ratio"])
            chosen = frozenset(filter(None, row["skipped_blocks"].split(";")))
            if len(chosen) != budget or int(row["skipped_block_count"]) != budget:
                raise ValueError(f"Logged count mismatch in {log_path}, step {row['step']}")
            if chosen & protected:
                raise ValueError(f"Logged policy bypasses protected blocks: {chosen & protected}")
            if noise in logged and logged[noise][0] != chosen:
                raise ValueError(f"Multiple policies at noise {noise} in {run_dir}")
            logged[noise] = (chosen, logged.get(noise, (None, 0))[1] + 1)
        if not logged:
            raise ValueError(f"Empty training log: {log_path}")
        provenance.append({"budget": budget, "run_dir": str(run_dir),
                           "metadata_sha256": digest(meta_path),
                           "log_sha256": digest(log_path),
                           "protected_blocks": sorted(protected),
                           "min_run": min_run, "max_run": max_run, "max_runs": max_runs})
        for noise, (old_set, num_steps) in sorted(logged.items()):
            if noise not in importance:
                raise ValueError(f"Missing exact noise anchor {noise}")
            source = importance[noise]
            unknown = protected - source.keys()
            if unknown:
                raise ValueError(f"Unknown protected blocks: {sorted(unknown)}")
            eligible = set(source) - protected
            missing = eligible - costs.keys()
            if missing:
                raise ValueError(f"Missing costs for eligible blocks: {sorted(missing)}")
            if not old_set <= eligible:
                raise ValueError("Logged blocks absent from importance candidates")
            candidates = sorted(
                [dict(source[b], cost=costs[b]) for b in eligible], key=lambda r: r["index"])
            if len({r["index"] for r in candidates}) != len(candidates):
                raise ValueError("Duplicate block indices")
            old_runs = runs_for(old_set, candidates)
            if len(old_runs) > max_runs or any(not min_run <= len(r) <= max_run for r in old_runs):
                raise ValueError(f"Logged runs violate metadata constraints at {noise}")
            target = sum((costs[b] for b in old_set), Decimal(0))
            old_score = sum((source[b]["score"] for b in old_set), Decimal(0))
            begin = time.perf_counter()
            chosen, score, saved, peak = solve(candidates, target, *constraints)
            elapsed = time.perf_counter() - begin
            old = [r["block"] for r in candidates if r["block"] in old_set]
            new_set = set(chosen)
            row = {
                "baseline_budget": budget, "noise_ratio": float(noise),
                "num_logged_steps": num_steps,
                "old_count": len(old), "new_count": len(chosen),
                "selection_changed": old_set != new_set,
                "old_skip_indices": ";".join(str(source[b]["index"]) for b in old),
                "new_skip_indices": ";".join(str(source[b]["index"]) for b in chosen),
                "target_estimated_gflops": str(target), "new_estimated_gflops": str(saved),
                "overshoot_pct": float((saved / target - 1) * 100),
                "old_importance_sum": str(old_score), "new_importance_sum": str(score),
                "importance_reduction_pct": float((1 - score / old_score) * 100) if old_score else 0.0,
                "old_run_count": len(old_runs), "new_run_count": len(runs_for(chosen, candidates)),
                "old_skip_blocks": ";".join(old), "new_skip_blocks": ";".join(chosen),
                "dp_time_s": elapsed, "dp_peak_states": peak,
            }
            comparisons.append(row)
            for mode, blocks in (("layer_budget", old), ("compute_budget", chosen)):
                policy_rows.append({"baseline_budget": budget, "noise_ratio": float(noise),
                                    "mode": mode, "skip_count": len(blocks),
                                    "skip_blocks": ";".join(blocks)})
            print(f"B{budget} noise={noise}: {len(old)} -> {len(chosen)} blocks; "
                  f"changed={row['selection_changed']}; "
                  f"importance reduction={row['importance_reduction_pct']:.2f}%; "
                  f"compute overshoot={row['overshoot_pct']:.2f}%")

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "compute_budget_comparison.csv", comparisons)
    write_csv(output / "bypass_policies.csv", policy_rows)
    report = {
        "importance_csv": args.importance_csv, "importance_step": args.importance_step,
        "importance_sha256": digest(args.importance_csv),
        "cost_csv": args.cost_csv, "cost_sha256": digest(args.cost_csv),
        "baseline_runs": provenance, "num_policies": len(comparisons),
        "changed_policies": sum(row["selection_changed"] for row in comparisons),
        "total_dp_time_s": sum(row["dp_time_s"] for row in comparisons),
        "total_script_time_s": time.perf_counter() - started,
        "cost_note": "Sum of measured single-block supported-operator FLOPs savings; "
                     "not a measurement of joint bypass savings. Verify joint execution before training.",
        "solver": "Exact Decimal-cost DP; minimum importance, then minimum cost, then block count.",
    }
    (output / "selection_metadata.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote results to {output}")


if __name__ == "__main__":
    main()
