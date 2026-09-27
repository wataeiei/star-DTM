#!/usr/bin/env python3
"""Build joint sparse-LoRA and independent Threshold bypass candidates.

The policy is intentionally free of count, contiguity, run-length, and run-count
constraints. For one global threshold tau, every structurally safe frozen block
whose condition-normalized importance is at most tau is bypassed independently.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--importance_csv", required=True)
    parser.add_argument("--cost_csv", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--score_key", default="normalized_grad_score")
    parser.add_argument(
        "--importance_step",
        type=int,
        default=None,
        help="Exact train_step to use when the importance CSV contains train_step.",
    )
    parser.add_argument(
        "--noise_weights",
        nargs="*",
        default=[],
        metavar="NOISE:WEIGHT",
        help="Training-condition probabilities. Default: uniform over anchors.",
    )
    parser.add_argument(
        "--k_values",
        type=int,
        nargs="*",
        default=[],
        help="Explicit K candidates. Default uses approximately 10,20,33,50,100%%.",
    )
    parser.add_argument(
        "--k_fractions",
        type=float,
        nargs="+",
        default=[0.10, 0.20, 1.0 / 3.0, 0.50, 1.0],
    )
    parser.add_argument(
        "--threshold_metric",
        choices=["importance", "importance_per_saved_gflop"],
        default="importance",
    )
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="*",
        default=[],
        help="Explicit global tau candidates. Default derives policy breakpoints.",
    )
    parser.add_argument(
        "--max_threshold_candidates",
        type=int,
        default=15,
        help="Maximum automatically sampled policy breakpoints per K; 0 keeps all.",
    )
    parser.add_argument("--unsafe_block", action="append", default=[])
    parser.add_argument("--correctness_tolerance", type=float, default=0.0)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def block_index(block: str) -> int:
    try:
        return int(block.rsplit(".", 1)[1])
    except (IndexError, ValueError) as exc:
        raise SystemExit(f"Cannot infer block index from {block!r}") from exc


def parse_noise_weights(values: list[str], anchors: list[float]) -> dict[float, float]:
    if not values:
        return {anchor: 1.0 / len(anchors) for anchor in anchors}
    parsed: dict[float, float] = {}
    for value in values:
        try:
            noise_text, weight_text = value.split(":", 1)
            noise = float(noise_text)
            weight = float(weight_text)
        except ValueError as exc:
            raise SystemExit(
                f"Invalid noise weight {value!r}; expected NOISE:WEIGHT"
            ) from exc
        if weight < 0 or not math.isfinite(weight):
            raise SystemExit(f"Noise weight must be finite and non-negative: {value}")
        parsed[noise] = weight
    missing = [
        anchor for anchor in anchors
        if not any(math.isclose(anchor, key, abs_tol=1e-8) for key in parsed)
    ]
    extras = [
        key for key in parsed
        if not any(math.isclose(key, anchor, abs_tol=1e-8) for anchor in anchors)
    ]
    if missing or extras:
        raise SystemExit(
            f"Noise weights do not match anchors; missing={missing}, extras={extras}"
        )
    total = sum(parsed.values())
    if total <= 0:
        raise SystemExit("Noise weights must have positive total mass")
    return {
        anchor: next(
            weight for key, weight in parsed.items()
            if math.isclose(anchor, key, abs_tol=1e-8)
        ) / total
        for anchor in anchors
    }


def load_importance(
    path: Path,
    score_key: str,
    importance_step: int | None,
) -> tuple[dict[float, dict[str, float]], list[str]]:
    rows = read_csv(path)
    if not rows:
        raise SystemExit(f"Importance CSV is empty: {path}")
    required = {"noise_ratio", "block"}
    missing = required - set(rows[0])
    if missing:
        raise SystemExit(f"Importance CSV is missing columns: {sorted(missing)}")
    if score_key not in rows[0]:
        if score_key == "normalized_grad_score" and "score_share" in rows[0]:
            score_key = "score_share"
        else:
            raise SystemExit(f"Importance CSV has no score column {score_key!r}")
    if "train_step" in rows[0]:
        available_steps = sorted({int(float(row["train_step"])) for row in rows})
        selected_step = importance_step if importance_step is not None else available_steps[0]
        if selected_step not in available_steps:
            raise SystemExit(
                f"importance_step={selected_step} not found; available={available_steps}"
            )
        rows = [row for row in rows if int(float(row["train_step"])) == selected_step]
    elif importance_step is not None:
        raise SystemExit("--importance_step was set but CSV has no train_step column")

    grouped: dict[float, dict[str, float]] = {}
    for row in rows:
        ratio = float(row["noise_ratio"])
        block = str(row["block"])
        score = float(row[score_key])
        if score < 0 or not math.isfinite(score):
            raise SystemExit(f"Invalid importance score for {block} at {ratio:g}: {score}")
        if block in grouped.setdefault(ratio, {}):
            raise SystemExit(f"Duplicate importance row for {block} at {ratio:g}")
        grouped[ratio][block] = score

    anchors = sorted(grouped)
    reference_blocks = set(grouped[anchors[0]])
    if not reference_blocks:
        raise SystemExit("Importance CSV contains no candidate blocks")
    for ratio in anchors:
        blocks = set(grouped[ratio])
        if blocks != reference_blocks:
            raise SystemExit(
                f"Importance coverage differs at noise={ratio:g}: "
                f"missing={sorted(reference_blocks - blocks)}, "
                f"extra={sorted(blocks - reference_blocks)}"
            )
        total = sum(grouped[ratio].values())
        if total <= 0:
            raise SystemExit(f"Importance total is zero at noise={ratio:g}")
        grouped[ratio] = {
            block: score / total for block, score in grouped[ratio].items()
        }
    return grouped, sorted(reference_blocks, key=block_index)


def load_costs(
    path: Path,
    blocks: list[str],
    explicit_unsafe: set[str],
    tolerance: float,
) -> tuple[dict[str, float], set[str], list[dict]]:
    rows = read_csv(path)
    if not rows:
        raise SystemExit(f"Cost CSV is empty: {path}")
    by_block = {str(row["block"]): row for row in rows}
    missing = sorted(set(blocks) - set(by_block), key=block_index)
    if missing:
        raise SystemExit(
            "Cost profile does not cover every candidate block. Re-run "
            "profile_tsdsr_bypass_costs.py with --selected_k 0. Missing: "
            + ", ".join(missing)
        )
    unknown_unsafe = sorted(explicit_unsafe - set(blocks))
    if unknown_unsafe:
        raise SystemExit("Unknown --unsafe_block values: " + ", ".join(unknown_unsafe))

    costs: dict[str, float] = {}
    unsafe = set(explicit_unsafe)
    audit = []
    for block in blocks:
        row = by_block[block]
        fallback = int(float(row.get("fallback_events", 0) or 0))
        loss_diff = float(row.get("max_loss_abs_diff", 0) or 0)
        forward_diff = float(row.get("max_forward_abs_diff", 0) or 0)
        saved = float(row.get("reported_gflops_saved", "nan"))
        reasons = []
        if fallback:
            reasons.append("fallback")
        if loss_diff > tolerance:
            reasons.append("loss_diff")
        if forward_diff > tolerance:
            reasons.append("forward_diff")
        if not math.isfinite(saved) or saved <= 0:
            reasons.append("invalid_saved_gflops")
        if block in explicit_unsafe:
            reasons.append("explicitly_unsafe")
        if reasons:
            unsafe.add(block)
        costs[block] = saved
        audit.append({
            "block": block,
            "block_index": block_index(block),
            "safe_for_bypass": not reasons,
            "unsafe_reasons": ";".join(reasons),
            "reported_gflops_saved": saved,
            "max_loss_abs_diff": loss_diff,
            "max_forward_abs_diff": forward_diff,
            "fallback_events": fallback,
        })
    return costs, unsafe, audit


def derive_k_values(args: argparse.Namespace, count: int) -> list[int]:
    if args.k_values:
        values = args.k_values
    else:
        if any(fraction <= 0 or fraction > 1 for fraction in args.k_fractions):
            raise SystemExit("--k_fractions values must be in (0, 1]")
        values = [math.ceil(count * fraction) for fraction in args.k_fractions]
    values = sorted(set(values))
    if not values or values[0] <= 0 or values[-1] > count:
        raise SystemExit(f"K candidates must be within [1, {count}]: {values}")
    return values


def sample_breakpoints(values: list[float], maximum: int) -> list[float]:
    values = sorted(set([0.0, *values]))
    if maximum <= 0 or len(values) <= maximum:
        return values
    if maximum < 2:
        return [values[0]]
    indices = {
        round(position * (len(values) - 1) / (maximum - 1))
        for position in range(maximum)
    }
    return [values[index] for index in sorted(indices)]


def format_tau(value: float) -> str:
    return f"{value:.12g}"


def main() -> None:
    args = parse_args()
    if args.correctness_tolerance < 0:
        raise SystemExit("--correctness_tolerance must be non-negative")
    if args.max_threshold_candidates < 0:
        raise SystemExit("--max_threshold_candidates must be non-negative")

    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise SystemExit(f"Output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    importance, blocks = load_importance(
        Path(args.importance_csv), args.score_key, args.importance_step
    )
    anchors = sorted(importance)
    weights = parse_noise_weights(args.noise_weights, anchors)
    aggregate = {
        block: sum(weights[ratio] * importance[ratio][block] for ratio in anchors)
        for block in blocks
    }
    ranked = sorted(blocks, key=lambda block: (-aggregate[block], block_index(block)))
    costs, unsafe, safety_audit = load_costs(
        Path(args.cost_csv),
        blocks,
        set(args.unsafe_block),
        args.correctness_tolerance,
    )
    write_csv(output_dir / "block_safety_audit.csv", safety_audit)
    k_values = derive_k_values(args, len(blocks))

    ranking_rows = []
    cumulative = 0.0
    for rank, block in enumerate(ranked, 1):
        cumulative += aggregate[block]
        ranking_rows.append({
            "rank": rank,
            "block": block,
            "block_index": block_index(block),
            "aggregate_utility": aggregate[block],
            "cumulative_utility": cumulative,
            "reported_gflops_saved": costs[block],
            "safe_for_bypass": block not in unsafe,
        })
    write_csv(output_dir / "lora_importance_ranking.csv", ranking_rows)

    k_rows = []
    manifest = []
    for k in k_values:
        selected = set(ranked[:k])
        eligible = [
            block for block in blocks if block not in selected and block not in unsafe
        ]
        selected_utility = sum(aggregate[block] for block in selected)
        k_dir = output_dir / f"K{k:02d}"
        k_dir.mkdir(parents=True, exist_ok=True)
        selection_path = k_dir / "selection.json"
        selection = {
            "algorithm": "Threshold bypass",
            "selection_rule": "target-domain weighted gradient importance Top-K",
            "selected_k": k,
            "candidate_block_count": len(blocks),
            "selected_utility": selected_utility,
            "selected_lora_blocks": sorted(selected, key=block_index),
            "selected_blocks": sorted(selected, key=block_index),
            "noise_weights": {f"{ratio:g}": weights[ratio] for ratio in anchors},
            "importance_csv": str(Path(args.importance_csv)),
            "cost_csv": str(Path(args.cost_csv)),
        }
        selection_path.write_text(json.dumps(selection, indent=2), encoding="utf-8")
        k_rows.append({
            "selected_k": k,
            "block_fraction": k / len(blocks),
            "selected_utility": selected_utility,
            "eligible_bypass_blocks": len(eligible),
            "unsafe_frozen_blocks": len((set(blocks) - selected) & unsafe),
            "selection_file": str(selection_path),
            "selected_blocks": ";".join(sorted(selected, key=block_index)),
        })

        metric_by_noise: dict[float, dict[str, float]] = {}
        breakpoints = []
        for ratio in anchors:
            denominator = sum(importance[ratio][block] for block in eligible)
            condition = {}
            for block in eligible:
                normalized = (
                    importance[ratio][block] / denominator if denominator > 0 else 0.0
                )
                metric = normalized
                if args.threshold_metric == "importance_per_saved_gflop":
                    metric /= costs[block]
                condition[block] = metric
                breakpoints.append(metric)
            metric_by_noise[ratio] = condition

        thresholds = (
            sorted(set(args.thresholds))
            if args.thresholds
            else sample_breakpoints(breakpoints, args.max_threshold_candidates)
        )
        if not thresholds:
            thresholds = [0.0]
        if any(value < 0 or not math.isfinite(value) for value in thresholds):
            raise SystemExit("Threshold candidates must be finite and non-negative")

        seen_signatures = set()
        candidate_number = 0
        for threshold in thresholds:
            policy: dict[float, list[str]] = {}
            signature = []
            for ratio in anchors:
                chosen = sorted(
                    [
                        block for block, metric in metric_by_noise[ratio].items()
                        if metric <= threshold + 1e-15
                    ],
                    key=block_index,
                )
                policy[ratio] = chosen
                signature.append(tuple(chosen))
            signature_key = tuple(signature)
            if signature_key in seen_signatures:
                continue
            seen_signatures.add(signature_key)
            candidate_number += 1
            candidate_id = f"K{k:02d}-T{candidate_number:03d}"
            candidate_dir = k_dir / "threshold_candidates" / candidate_id
            candidate_dir.mkdir(parents=True, exist_ok=True)
            policy_path = candidate_dir / "bypass_policy_by_condition.csv"
            policy_rows = []
            weighted_count = 0.0
            weighted_saved = 0.0
            for ratio in anchors:
                chosen = policy[ratio]
                saved = sum(costs[block] for block in chosen)
                weighted_count += weights[ratio] * len(chosen)
                weighted_saved += weights[ratio] * saved
                policy_rows.append({
                    "algorithm": "Threshold bypass",
                    "candidate_id": candidate_id,
                    "selected_k": k,
                    "threshold_metric": args.threshold_metric,
                    "threshold": format_tau(threshold),
                    "noise_ratio": ratio,
                    "noise_weight": weights[ratio],
                    "eligible_block_count": len(eligible),
                    "bypass_budget": len(chosen),
                    "estimated_saved_gflops": saved,
                    "max_selected_metric": max(
                        (metric_by_noise[ratio][block] for block in chosen),
                        default=0.0,
                    ),
                    "skip_blocks": ";".join(chosen),
                })
            write_csv(policy_path, policy_rows)
            candidate_metadata = {
                "algorithm": "Threshold bypass",
                "candidate_id": candidate_id,
                "selected_k": k,
                "threshold_metric": args.threshold_metric,
                "global_threshold": threshold,
                "independent_block_selection": True,
                "count_constraint": None,
                "contiguity_constraint": None,
                "run_length_constraint": None,
                "run_count_constraint": None,
                "selected_lora_blocks": sorted(selected, key=block_index),
                "protected_unsafe_blocks": sorted(unsafe, key=block_index),
                "mean_bypass_blocks": weighted_count,
                "estimated_saved_gflops_per_step": weighted_saved,
                "selection_file": str(selection_path),
                "policy_csv": str(policy_path),
            }
            (candidate_dir / "metadata.json").write_text(
                json.dumps(candidate_metadata, indent=2), encoding="utf-8"
            )
            manifest.append({
                "candidate_id": candidate_id,
                "selected_k": k,
                "selected_utility": selected_utility,
                "threshold_metric": args.threshold_metric,
                "global_threshold": threshold,
                "mean_bypass_blocks": weighted_count,
                "estimated_saved_gflops_per_step": weighted_saved,
                "selection_file": str(selection_path),
                "policy_csv": str(policy_path),
            })

    write_csv(output_dir / "k_candidates.csv", k_rows)
    write_csv(output_dir / "threshold_candidate_manifest.csv", manifest)
    metadata = {
        "algorithm": "Threshold bypass",
        "objective": "joint measured-time minimization under quality constraints",
        "candidate_block_count": len(blocks),
        "k_candidates": k_values,
        "threshold_metric": args.threshold_metric,
        "noise_weights": {f"{ratio:g}": weights[ratio] for ratio in anchors},
        "safe_bypass_blocks": sorted(set(blocks) - unsafe, key=block_index),
        "unsafe_bypass_blocks": sorted(unsafe, key=block_index),
        "policy_rule": "independent safe frozen blocks with metric <= one global tau",
        "constraints_removed": [
            "fixed bypass count",
            "contiguous runs",
            "minimum run length",
            "maximum run length",
            "maximum run count",
        ],
        "importance_csv": args.importance_csv,
        "cost_csv": args.cost_csv,
        "candidate_count": len(manifest),
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2))
    print(f"Wrote {len(manifest)} Threshold bypass candidates to {output_dir}")


if __name__ == "__main__":
    main()
