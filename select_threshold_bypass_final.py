#!/usr/bin/env python3
"""Select the fastest quality-feasible Threshold bypass candidate."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


GAIN_METRICS = {
    "psnr": "higher",
    "ssim": "higher",
    "lpips": "lower",
    "acc": "higher",
}
NO_REFERENCE_METRICS = ("musiq", "maniqa", "clipiqa", "liqe")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results_csv", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--min_gain_retention", type=float, default=0.85)
    parser.add_argument("--max_no_ref_drop_fraction", type=float, default=0.05)
    parser.add_argument("--max_acc_drop_pp", type=float, default=1.0)
    parser.add_argument("--min_speedup_vs_controller_pct", type=float, default=2.0)
    parser.add_argument("--forward_diff_tolerance", type=float, default=0.0)
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


def number(row: dict[str, str], key: str) -> float:
    value = row.get(key, "")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise SystemExit(f"Missing or invalid {key!r} for {row.get('candidate_id')}") from exc
    if not math.isfinite(result):
        raise SystemExit(f"Non-finite {key!r} for {row.get('candidate_id')}")
    return result


def boolean(row: dict[str, str], key: str) -> bool:
    value = str(row.get(key, "")).strip().lower()
    if value in {"1", "true", "yes"}:
        return True
    if value in {"0", "false", "no"}:
        return False
    raise SystemExit(f"Missing or invalid boolean {key!r} for {row.get('candidate_id')}")


def gain_retention(base: float, all_lora: float, candidate: float, direction: str) -> float | None:
    if direction == "higher":
        gain = all_lora - base
        return (candidate - base) / gain if gain > 0 else None
    gain = base - all_lora
    return (base - candidate) / gain if gain > 0 else None


def main() -> None:
    args = parse_args()
    if not 0 <= args.min_gain_retention <= 1:
        raise SystemExit("--min_gain_retention must be in [0, 1]")
    if not 0 <= args.max_no_ref_drop_fraction <= 1:
        raise SystemExit("--max_no_ref_drop_fraction must be in [0, 1]")
    if args.max_acc_drop_pp < 0 or args.min_speedup_vs_controller_pct < 0:
        raise SystemExit("Drop and speedup limits must be non-negative")

    rows = read_csv(Path(args.results_csv))
    if not rows:
        raise SystemExit(f"Results CSV is empty: {args.results_csv}")
    required = {
        "candidate_id",
        "selected_k",
        "global_threshold",
        "mean_step_time_s",
        "controller_b0_step_time_s",
        "controller_overhead_in_step_time",
        "fallback_block_events",
        "max_forward_abs_diff",
        "nonfinite_loss_count",
    }
    for metric in GAIN_METRICS:
        required.update({f"base_{metric}", f"all_{metric}", f"candidate_{metric}"})
    for metric in NO_REFERENCE_METRICS:
        required.update({f"native_{metric}", f"candidate_{metric}"})
    required.add("native_acc")
    missing = required - set(rows[0])
    if missing:
        raise SystemExit("Results CSV is missing columns: " + ", ".join(sorted(missing)))

    decisions = []
    for row in rows:
        reasons = []
        candidate_time = number(row, "mean_step_time_s")
        controller_time = number(row, "controller_b0_step_time_s")
        speedup = (
            (controller_time - candidate_time) / controller_time * 100.0
            if controller_time > 0 else float("-inf")
        )
        if not boolean(row, "controller_overhead_in_step_time"):
            reasons.append("controller_overhead_not_timed")
        if number(row, "fallback_block_events") != 0:
            reasons.append("fallback_nonzero")
        if number(row, "max_forward_abs_diff") > args.forward_diff_tolerance:
            reasons.append("forward_difference")
        if number(row, "nonfinite_loss_count") != 0:
            reasons.append("nonfinite_loss")
        if speedup < args.min_speedup_vs_controller_pct:
            reasons.append("insufficient_speedup")

        retentions = {}
        for metric, direction in GAIN_METRICS.items():
            retention = gain_retention(
                number(row, f"base_{metric}"),
                number(row, f"all_{metric}"),
                number(row, f"candidate_{metric}"),
                direction,
            )
            retentions[metric] = retention
            if retention is not None and retention < args.min_gain_retention:
                reasons.append(f"{metric}_gain_retention")

        for metric in NO_REFERENCE_METRICS:
            native = number(row, f"native_{metric}")
            candidate = number(row, f"candidate_{metric}")
            minimum = native * (1.0 - args.max_no_ref_drop_fraction)
            if candidate < minimum:
                reasons.append(f"{metric}_drop_vs_native")

        acc_drop_pp = (
            number(row, "native_acc") - number(row, "candidate_acc")
        ) * 100.0
        if acc_drop_pp > args.max_acc_drop_pp:
            reasons.append("acc_drop_vs_native")

        decision = dict(row)
        decision.update({
            "speedup_vs_controller_pct": speedup,
            "psnr_gain_retention": retentions["psnr"],
            "ssim_gain_retention": retentions["ssim"],
            "lpips_gain_retention": retentions["lpips"],
            "acc_gain_retention": retentions["acc"],
            "acc_drop_vs_native_pp": acc_drop_pp,
            "feasible": not reasons,
            "rejection_reasons": ";".join(reasons),
        })
        decisions.append(decision)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "candidate_decisions.csv", decisions)
    feasible = [row for row in decisions if row["feasible"]]
    feasible.sort(key=lambda row: float(row["mean_step_time_s"]))
    report = {
        "algorithm": "Threshold bypass",
        "selection_objective": "minimum measured step time among quality-feasible candidates",
        "candidate_count": len(decisions),
        "feasible_count": len(feasible),
        "constraints": {
            "min_gain_retention": args.min_gain_retention,
            "max_no_ref_drop_fraction_vs_native_k": args.max_no_ref_drop_fraction,
            "max_acc_drop_pp_vs_native_k": args.max_acc_drop_pp,
            "min_speedup_vs_controller_b0_pct": args.min_speedup_vs_controller_pct,
            "forward_diff_tolerance": args.forward_diff_tolerance,
            "fallback": 0,
            "nonfinite_loss": 0,
        },
        "best_candidate": feasible[0] if feasible else None,
    }
    (output_dir / "selection_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    if not feasible:
        raise SystemExit("No Threshold bypass candidate satisfies every constraint")


if __name__ == "__main__":
    main()
