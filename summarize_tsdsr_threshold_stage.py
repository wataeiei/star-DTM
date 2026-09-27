#!/usr/bin/env python3
"""Summarize one serial Threshold bypass timing stage and shortlist candidates."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage_results", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--min_speedup_vs_controller_pct", type=float, default=2.0)
    parser.add_argument("--forward_diff_tolerance", type=float, default=0.0)
    parser.add_argument("--top_per_k", type=int, default=2)
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


def value(row: dict[str, str], key: str) -> float:
    try:
        result = float(row[key])
    except (KeyError, TypeError, ValueError) as exc:
        raise SystemExit(f"Missing or invalid {key!r} in stage results") from exc
    if not math.isfinite(result):
        raise SystemExit(f"Non-finite {key!r} in stage results")
    return result


def main() -> None:
    args = parse_args()
    if args.top_per_k <= 0:
        raise SystemExit("--top_per_k must be positive")
    rows = read_csv(Path(args.stage_results))
    if not rows:
        raise SystemExit(f"Stage results are empty: {args.stage_results}")

    controls: dict[int, dict[str, dict[str, str]]] = {}
    for row in rows:
        if row["run_kind"] in {"native", "controller_b0"}:
            controls.setdefault(int(row["selected_k"]), {})[row["run_kind"]] = row

    comparisons = []
    for row in rows:
        if row["run_kind"] != "threshold_bypass":
            continue
        k = int(row["selected_k"])
        if k not in controls or set(controls[k]) != {"native", "controller_b0"}:
            raise SystemExit(f"Missing Native-K or Controller-B0 control for K={k}")
        candidate_time = value(row, "mean_train_step_time_s")
        native_time = value(controls[k]["native"], "mean_train_step_time_s")
        controller_time = value(controls[k]["controller_b0"], "mean_train_step_time_s")
        speed_native = (native_time - candidate_time) / native_time * 100.0
        speed_controller = (controller_time - candidate_time) / controller_time * 100.0
        fallback = value(row, "fallback_block_events")
        forward_diff = value(row, "max_residual_forward_abs_diff")
        reasons = []
        if fallback != 0:
            reasons.append("fallback_nonzero")
        if forward_diff > args.forward_diff_tolerance:
            reasons.append("forward_difference")
        if speed_controller < args.min_speedup_vs_controller_pct:
            reasons.append("insufficient_speedup")
        comparisons.append({
            "candidate_id": row["candidate_id"],
            "selected_k": k,
            "global_threshold": row["global_threshold"],
            "mean_skipped_blocks": row["mean_skipped_blocks"],
            "candidate_step_time_s": candidate_time,
            "native_step_time_s": native_time,
            "controller_b0_step_time_s": controller_time,
            "speedup_vs_native_pct": speed_native,
            "speedup_vs_controller_pct": speed_controller,
            "fallback_block_events": fallback,
            "max_forward_abs_diff": forward_diff,
            "timing_correctness_pass": not reasons,
            "rejection_reasons": ";".join(reasons),
            "run_dir": row["run_dir"],
        })

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "timing_comparison.csv", comparisons)
    shortlist = []
    for k in sorted({int(row["selected_k"]) for row in comparisons}):
        candidates = [
            row for row in comparisons
            if int(row["selected_k"]) == k and row["timing_correctness_pass"]
        ]
        candidates.sort(key=lambda row: float(row["candidate_step_time_s"]))
        shortlist.extend(candidates[: args.top_per_k])
    write_csv(output_dir / "shortlist.csv", shortlist)
    report = {
        "algorithm": "Threshold bypass",
        "stage_candidate_count": len(comparisons),
        "timing_correctness_pass_count": sum(
            bool(row["timing_correctness_pass"]) for row in comparisons
        ),
        "shortlist_count": len(shortlist),
        "shortlisted_candidate_ids": [row["candidate_id"] for row in shortlist],
        "min_speedup_vs_controller_pct": args.min_speedup_vs_controller_pct,
        "forward_diff_tolerance": args.forward_diff_tolerance,
        "top_per_k": args.top_per_k,
    }
    (output_dir / "stage_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
