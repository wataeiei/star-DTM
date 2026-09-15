#!/usr/bin/env python3
"""Calibrate an unconstrained threshold against a measured DiT-SR skip policy."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path


def read_rows(path, required):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = set(required) - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path}: missing columns {sorted(missing)}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"Empty CSV: {path}")
    return rows


def finite(value):
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"Non-finite numeric value: {value}")
    return number


def split_blocks(value):
    return [part.strip() for part in value.split(";") if part.strip()]


def anchor_for(value, anchors):
    matches = [a for a in anchors if math.isclose(value, a, rel_tol=0, abs_tol=1e-8)]
    if len(matches) != 1:
        raise ValueError(f"Reference noise {value} does not match one calibration anchor")
    return matches[0]


def build(args):
    importance = read_rows(args.importance_csv, [
        "train_step", "noise_ratio", "block", "block_index", args.score_key,
    ])
    groups = defaultdict(dict)
    for row in importance:
        if int(row["train_step"]) != args.importance_step:
            continue
        noise, block = finite(row["noise_ratio"]), row["block"]
        if block in groups[noise]:
            raise ValueError(f"Duplicate importance row: {noise}, {block}")
        score = finite(row[args.score_key])
        if score < 0:
            raise ValueError(f"Negative importance: {block}")
        groups[noise][block] = (int(row["block_index"]), score)
    if not groups:
        raise ValueError("No importance rows at the requested step")
    anchors = sorted(groups)
    blocks = set(groups[anchors[0]])
    indices = {b: groups[anchors[0]][b][0] for b in blocks}
    if len(set(indices.values())) != len(indices):
        raise ValueError("Duplicate block indices")
    for noise in anchors:
        if set(groups[noise]) != blocks:
            raise ValueError(f"Inconsistent block coverage at noise={noise}")
        if any(groups[noise][b][0] != indices[b] for b in blocks):
            raise ValueError("Block indices differ between noise anchors")

    selection = json.loads(Path(args.selection_file).read_text(encoding="utf-8-sig"))
    selected = selection.get("selected_blocks", selection.get("selected_lora_blocks"))
    if not isinstance(selected, list) or not selected:
        raise ValueError("Selection JSON needs selected_blocks or selected_lora_blocks")
    protected = set(selected) | set(args.protected_block)
    excluded = set(args.exclude_block)
    unknown = (protected | excluded) - blocks
    if unknown:
        raise ValueError(f"Unknown protected/excluded blocks: {sorted(unknown)}")

    costs = {}
    for row in read_rows(args.cost_csv, ["block", "reported_gflops_saved"]):
        block = row["block"]
        if block in costs:
            raise ValueError(f"Duplicate cost row: {block}")
        costs[block] = finite(row["reported_gflops_saved"])
    eligible = blocks - protected - excluded
    if not eligible or eligible - costs.keys():
        raise ValueError(f"Empty candidate set or missing costs: {sorted(eligible - costs.keys())}")
    if any(costs[b] <= 0 for b in eligible):
        raise ValueError("Non-positive candidate costs: exclude those blocks explicitly")
    cap = len(eligible) if args.max_count is None else args.max_count
    if not 1 <= cap <= len(eligible):
        raise ValueError(f"max_count must be within 1..{len(eligible)}")

    ref_costs = defaultdict(list)
    steps = set()
    for row in read_rows(args.reference_log, [
        "step", "noise_ratio", "skipped_blocks", "skipped_block_count", "fallback_blocks",
    ]):
        step = int(row["step"])
        if step in steps:
            raise ValueError(f"Duplicate reference step: {step}")
        steps.add(step)
        noise = anchor_for(finite(row["noise_ratio"]), anchors)
        skip = split_blocks(row["skipped_blocks"])
        if len(set(skip)) != len(skip) or len(skip) != int(row["skipped_block_count"]):
            raise ValueError(f"Reference skip count mismatch at step {step}")
        if int(row["fallback_blocks"]) != 0:
            raise ValueError("Reference contains fallback events; audit actual execution first")
        if set(skip) - blocks or set(skip) - costs.keys() or set(skip) & protected:
            raise ValueError(f"Reference policy has unknown, uncosted or protected blocks at {step}")
        # Keep signed measured costs in the reference instead of silently clipping them.
        ref_costs[noise].append(sum(costs[b] for b in skip))
    if set(ref_costs) != set(anchors):
        raise ValueError("Reference log must cover every noise anchor")
    if args.noise_weighting == "uniform":
        weights = {a: 1 / len(anchors) for a in anchors}
    else:
        weights = {a: len(ref_costs[a]) / len(steps) for a in anchors}
    target = sum(weights[a] * sum(ref_costs[a]) / len(ref_costs[a]) for a in anchors)
    if target <= 0:
        raise ValueError("Reference saved-compute target must be positive")

    scores, rankings = {}, {}
    n = len(eligible)
    for noise in anchors:
        total = sum(groups[noise][b][1] for b in eligible)
        if total <= 0:
            raise ValueError(f"Non-positive importance sum at noise={noise}")
        scores[noise] = {b: n * groups[noise][b][1] / total for b in eligible}
        rankings[noise] = sorted(eligible, key=lambda b: (scores[noise][b], indices[b]))

    # Selections can change only at a score boundary; no arbitrary threshold grid is needed.
    thresholds = sorted({0.0} | {v for table in scores.values() for v in table.values()})
    scan, policies = [], {}
    for alpha in thresholds:
        policy = {}
        for noise in anchors:
            qualifying = [b for b in rankings[noise] if scores[noise][b] <= alpha]
            chosen = qualifying[:cap]
            policy[noise] = {
                "alpha": alpha, "score_share_threshold": alpha / n,
                "noise_ratio": noise, "bypass_budget": len(chosen),
                "qualifying_count": len(qualifying), "cap_applied": len(qualifying) > cap,
                "estimated_saved_gflops": sum(costs[b] for b in chosen),
                "skip_blocks": ";".join(sorted(chosen, key=indices.get)),
            }
        saved = sum(weights[a] * policy[a]["estimated_saved_gflops"] for a in anchors)
        schedule = " ".join(f"{a:g}:{policy[a]['bypass_budget']}" for a in anchors)
        scan.append({
            "alpha": alpha, "score_share_threshold": alpha / n,
            "mean_bypass_budget": sum(weights[a] * policy[a]["bypass_budget"] for a in anchors),
            "estimated_saved_gflops_per_step": saved,
            "difference_vs_reference_pct": 100 * (saved / target - 1),
            "cap_applied": any(row["cap_applied"] for row in policy.values()),
            "schedule": schedule,
        })
        policies[alpha] = policy
    chosen = min(scan, key=lambda r: (abs(r["difference_vs_reference_pct"]), r["alpha"]))
    report = {
        "policy": "mean-normalized-threshold-unconstrained",
        "importance_csv": str(args.importance_csv), "importance_step": args.importance_step,
        "score_key": args.score_key, "cost_csv": str(args.cost_csv),
        "reference_log": str(args.reference_log), "reference_steps": len(steps),
        "noise_weighting": args.noise_weighting, "noise_probabilities": weights,
        "candidate_count": n, "selected_lora_blocks": sorted(set(selected), key=indices.get),
        "protected_blocks": sorted(protected, key=indices.get),
        "excluded_blocks": sorted(excluded, key=indices.get),
        "max_count": cap, "chosen_alpha": chosen["alpha"],
        "chosen_score_share_threshold": chosen["score_share_threshold"],
        "target_saved_gflops_per_step": target,
        "threshold_saved_gflops_per_step": chosen["estimated_saved_gflops_per_step"],
        "difference_vs_reference_pct": chosen["difference_vs_reference_pct"],
        "within_tolerance": abs(chosen["difference_vs_reference_pct"]) <= args.compute_tolerance_pct,
        "compute_tolerance_pct": args.compute_tolerance_pct,
        "mean_bypass_budget": chosen["mean_bypass_budget"],
        "threshold_schedule": chosen["schedule"], "cap_applied": chosen["cap_applied"],
        "candidate_thresholds_tested": len(scan),
        "note": "Costs are additive estimates of profiler-supported FLOPs, not measured whole-policy savings. No quality-based safety guarantee.",
    }
    return report, scan, list(policies[chosen["alpha"]].values())


def write_rows(path, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--importance_csv", required=True)
    parser.add_argument("--importance_step", type=int, default=0)
    parser.add_argument("--score_key", default="normalized_grad_score")
    parser.add_argument("--cost_csv", required=True)
    parser.add_argument("--selection_file", required=True)
    parser.add_argument("--reference_log", required=True)
    parser.add_argument("--protected_block", action="append", default=[])
    parser.add_argument("--exclude_block", action="append", default=[])
    parser.add_argument("--max_count", type=int)
    parser.add_argument("--noise_weighting", choices=["uniform", "reference"], default="uniform")
    parser.add_argument("--compute_tolerance_pct", type=float, default=1.0)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()
    if not math.isfinite(args.compute_tolerance_pct) or args.compute_tolerance_pct < 0:
        parser.error("compute_tolerance_pct must be finite and non-negative")
    try:
        report, scan, policy = build(args)
    except (ValueError, KeyError, OSError) as exc:
        parser.error(str(exc))
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_rows(out / "threshold_scan.csv", scan)
    write_rows(out / "threshold_policy_by_noise.csv", policy)
    (out / "threshold_schedule.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    (out / "threshold_schedule.txt").write_text(report["threshold_schedule"] + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, allow_nan=False))
    if not report["within_tolerance"]:
        print("WARNING: no scanned threshold meets the compute tolerance; this policy is not compute-matched.")


if __name__ == "__main__":
    main()
