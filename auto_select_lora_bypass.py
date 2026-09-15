#!/usr/bin/env python3
"""Jointly select threshold LoRA placement and a noise-aware bypass threshold."""

from __future__ import annotations

import argparse
import csv
import json
import math
import shlex
import subprocess
import sys
from pathlib import Path

from build_dit_sr_threshold_lora import make_selection, read_csv


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def write_rows(path, rows):
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def finite(value, name):
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"Non-finite {name}: {value}")
    return result


def normalized_weights(anchors, values):
    if values is None:
        return {anchor: 1 / len(anchors) for anchor in anchors}
    if len(values) != len(anchors) or any(not math.isfinite(v) or v < 0 for v in values):
        raise ValueError("Noise weights must be finite, non-negative, and match sorted anchors")
    total = sum(values)
    if total <= 0:
        raise ValueError("Noise weights must have a positive sum")
    return {anchor: value / total for anchor, value in zip(anchors, values)}


def load_lora_candidates(policy_dir, compute_csv, utility, max_blocks):
    policy_dir = Path(policy_dir)
    manifest = load_json(policy_dir / "scan_manifest.json")
    measured = {row["label"]: row for row in read_csv(compute_csv)}
    candidates = []
    for entry in manifest:
        label = entry["label"]
        if label == "reference_top8" or label not in measured:
            continue
        metadata = load_json(policy_dir / entry["selection_file"])
        selected = metadata.get("selected_blocks", [])
        if not selected or len(selected) > max_blocks or len(selected) != len(set(selected)):
            continue
        unknown = set(selected) - utility.keys()
        if unknown:
            raise ValueError(f"{label}: unknown selected blocks: {sorted(unknown)}")
        row = measured[label]
        candidates.append({
            "label": label,
            "metadata": metadata,
            "selected_blocks": selected,
            "selected_count": len(selected),
            "utility": sum(utility[block] for block in selected),
            "reported_gflops": finite(row["reported_gflops"], f"{label} GFLOPs"),
            "mean_step_time_ms": finite(row["mean_step_time_ms"], f"{label} step time"),
            "peak_cuda_mb": finite(row["peak_cuda_mb"], f"{label} peak memory"),
        })
    if not candidates:
        raise ValueError("No measured threshold LoRA candidates were found")
    refs = [row for row in read_csv(compute_csv) if row["label"] == "reference_top8"]
    if len(refs) != 1:
        raise ValueError("Compute CSV must contain exactly one reference_top8 row")
    reference_cost = finite(refs[0]["reported_gflops"], "reference Top8 GFLOPs")
    return candidates, reference_cost


def choose_lora(candidates, retention, metric):
    reference_utility = max(row["utility"] for row in candidates)
    if reference_utility <= 0:
        raise ValueError("Threshold candidates have no positive calibration utility")
    rows = []
    for item in candidates:
        item = dict(item)
        item["utility_retention"] = item["utility"] / reference_utility
        item["utility_constraint_met"] = item["utility_retention"] + 1e-12 >= retention
        rows.append(item)
    feasible = [row for row in rows if row["utility_constraint_met"]]
    if not feasible:
        raise ValueError("No LoRA candidate satisfies the utility-retention constraint")
    chosen = min(feasible, key=lambda row: (row[metric], row["selected_count"], -row["utility"]))
    return chosen, rows, reference_utility


def profile_bypass_costs(args, out, chosen_lora, training_data_dir, source):
    """Measure the chosen LoRA placement's per-block bypass costs before training."""
    profile_dir = out / "bypass_cost_profile"
    selection_path = out / "lora_selection_for_cost_profile.json"
    metadata = dict(chosen_lora["metadata"])
    metadata["selected_blocks"] = list(chosen_lora["selected_blocks"])
    metadata["selected_lora_blocks"] = list(chosen_lora["selected_blocks"])
    metadata["topk_blocks"] = chosen_lora["selected_count"]
    write_json(selection_path, metadata)

    script = Path(args.cost_profile_script)
    if not script.is_absolute() and not script.is_file():
        sibling = Path(__file__).resolve().parent / script
        if sibling.is_file():
            script = sibling
    if not script.is_file():
        raise ValueError(f"Cost profiling script does not exist: {script}")
    command = [
        sys.executable,
        str(script),
        "--model", "dit-sr",
        "--selection_file", str(selection_path),
        "--data_dir", str(training_data_dir),
        "--output_dir", str(profile_dir),
        "--importance_csv", str(args.importance_csv),
        "--importance_step", str(args.importance_step),
        "--noise_ratio", str(args.cost_noise_ratio),
        "--batch_size", "1",
        "--warmup", str(args.cost_warmup),
        "--repeats", str(args.cost_repeats),
        "--num_workers", "0",
        "--seed", str(args.cost_seed),
    ]
    overrides = (
        ("--config_path", args.config_path or source.get("config_path", "")),
        ("--ckpt_path", args.ckpt_path or source.get("ckpt_path", "")),
        ("--autoencoder_ckpt", args.autoencoder_ckpt or source.get("autoencoder_ckpt", "")),
    )
    for flag, value in overrides:
        if value:
            command.extend([flag, str(value)])
    for block in args.protected_block:
        command.extend(["--protected_block", block])
    if args.cpu_cost_profile:
        command.append("--cpu")
    print(
        f"Profiling K={chosen_lora['selected_count']} LoRA-specific bypass costs "
        f"into {profile_dir}"
    )
    subprocess.run(command, check=True)
    cost_path = profile_dir / "block_backward_costs.csv"
    if not cost_path.is_file():
        raise ValueError(f"Cost profiler did not create {cost_path}")
    print(f"Using newly profiled bypass costs: {cost_path}")
    return cost_path, command


def load_costs(path, all_blocks, selected, explicit_protected):
    costs, invalid = {}, set()
    for row in read_csv(path):
        block = str(row["block"])
        if block in costs or block in invalid:
            raise ValueError(f"Duplicate bypass cost row: {block}")
        if block not in all_blocks:
            raise ValueError(f"Cost CSV contains an unknown block: {block}")
        if "max_loss_abs_diff" in row and finite(row["max_loss_abs_diff"], "loss difference") != 0:
            raise ValueError(f"Cost audit changed the forward loss for {block}")
        value = finite(row["reported_gflops_saved"], f"{block} saved GFLOPs")
        if value > 0:
            costs[block] = value
        else:
            invalid.add(block)
    protected = set(selected) | set(explicit_protected) | invalid
    unknown = protected - all_blocks
    if unknown:
        raise ValueError(f"Unknown protected blocks: {sorted(unknown)}")
    eligible = all_blocks - protected
    missing = eligible - costs.keys()
    if missing:
        raise ValueError(
            "Bypass cost profile does not match the selected LoRA placement; "
            f"missing {len(missing)} blocks, examples={sorted(missing)[:5]}"
        )
    if not eligible:
        raise ValueError("No positive-cost frozen bypass candidates remain")
    return costs, protected, invalid, eligible


def load_fidelity(path):
    if not path:
        return None
    table = {}
    for row in read_csv(path):
        if str(row.get("noise_ratio", "")).lower() == "mean":
            continue
        key = (round(finite(row["noise_ratio"], "fidelity noise"), 8), int(row["bypass_budget"]))
        if key in table:
            raise ValueError(f"Duplicate fidelity row: {key}")
        table[key] = (
            finite(row["mean_gradient_cosine"], "gradient cosine"),
            finite(row["mean_relative_gradient_error"], "relative gradient error"),
        )
    if not table:
        raise ValueError("Fidelity CSV contains no per-noise rows")
    return table


def choose_bypass(groups, indices, eligible, costs, weights, lora_cost, reference_cost, args):
    scores = {}
    boundaries = {0.0}
    cap = min(len(eligible), args.max_bypass_count or len(eligible),
              max(0, math.floor(args.max_bypass_fraction * len(groups[next(iter(groups))]))))
    if cap < 1:
        raise ValueError("Bypass cap leaves no eligible blocks")
    for noise in sorted(groups):
        total = sum(groups[noise][block] for block in eligible)
        if total <= 0:
            raise ValueError(f"Non-positive bypass importance total at noise={noise:g}")
        scores[noise] = {block: groups[noise][block] / total for block in eligible}
        boundaries.update(scores[noise].values())
    fidelity = load_fidelity(args.fidelity_csv)
    scans, policies = [], {}
    for alpha in sorted(boundaries):
        details = []
        valid = True
        for noise in sorted(groups):
            selected = sorted(
                (block for block in eligible if scores[noise][block] <= alpha + 1e-15),
                key=lambda block: (scores[noise][block], indices[block]),
            )[:cap]
            risk = sum(scores[noise][block] for block in selected)
            valid &= risk <= args.max_bypass_importance_mass + 1e-12
            cosine = relative_error = None
            if fidelity is not None:
                measurement = fidelity.get((round(noise, 8), len(selected)))
                if measurement is None:
                    valid = False
                else:
                    cosine, relative_error = measurement
                    valid &= cosine >= args.min_gradient_cosine
                    valid &= relative_error <= args.max_relative_gradient_error
            details.append({
                "bypass_threshold": alpha,
                "noise_ratio": noise,
                "bypass_budget": len(selected),
                "bypass_importance_mass": risk,
                "estimated_saved_gflops": sum(costs[block] for block in selected),
                "gradient_cosine": cosine,
                "relative_gradient_error": relative_error,
                "skip_blocks": ";".join(sorted(selected, key=indices.get)),
            })
        saved = sum(weights[row["noise_ratio"]] * row["estimated_saved_gflops"] for row in details)
        reduction = 100 * (1 - (lora_cost - saved) / reference_cost)
        expected_risk = sum(weights[row["noise_ratio"]] * row["bypass_importance_mass"] for row in details)
        scan = {
            "bypass_threshold": alpha,
            "valid": bool(valid),
            "mean_bypass_budget": sum(weights[row["noise_ratio"]] * row["bypass_budget"] for row in details),
            "expected_bypass_importance_mass": expected_risk,
            "estimated_saved_gflops": saved,
            "estimated_total_gflops": lora_cost - saved,
            "estimated_total_reduction_vs_top8_pct": reduction,
            "target_met": reduction + 1e-12 >= args.target_total_compute_reduction_pct,
            "schedule": " ".join(f"{row['noise_ratio']:g}:{row['bypass_budget']}" for row in details),
        }
        scans.append(scan)
        policies[alpha] = details
    valid = [row for row in scans if row["valid"]]
    if not valid:
        raise ValueError("No bypass threshold satisfies the configured safety constraints")
    meeting = [row for row in valid if row["target_met"]]
    if meeting:
        chosen = min(meeting, key=lambda row: (
            row["expected_bypass_importance_mass"],
            row["estimated_total_reduction_vs_top8_pct"], row["bypass_threshold"]))
    else:
        chosen = max(valid, key=lambda row: (
            row["estimated_total_reduction_vs_top8_pct"],
            -row["expected_bypass_importance_mass"]))
    return chosen, scans, policies[chosen["bypass_threshold"]], cap, fidelity is not None


def auto_select(args):
    source = load_json(args.source_metadata)
    training_data_dir = args.data_dir or source.get("data_dir")
    if not training_data_dir:
        raise ValueError("Training data directory is absent; provide --data_dir")
    importance = read_csv(args.importance_csv)
    _, details = make_selection(
        importance, source, threshold=0, step=args.importance_step,
        expected_blocks=args.expected_blocks, score_key=args.score_key,
        noise_weights=args.noise_weights,
    )
    all_blocks = {row["block"] for row in details}
    indices = {row["block"]: row["block_index"] for row in details}
    utility = {row["block"]: row["noise_weighted_score_share"] for row in details}
    anchors = sorted({float(row["noise_ratio"]) for row in importance
                      if int(row["train_step"]) == args.importance_step})
    weights = normalized_weights(anchors, args.noise_weights)
    groups = {noise: {} for noise in anchors}
    for row in importance:
        if int(row["train_step"]) == args.importance_step:
            groups[float(row["noise_ratio"])][row["block"]] = finite(row[args.score_key], args.score_key)

    candidates, reference_cost = load_lora_candidates(
        args.lora_policy_dir, args.lora_compute_csv, utility, args.max_lora_blocks)
    chosen_lora, candidate_rows, reference_utility = choose_lora(
        candidates, args.min_lora_utility_retention, args.lora_cost_metric)

    out = Path(args.output_dir)
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise ValueError(f"Output directory is not empty: {out}")
    out.mkdir(parents=True, exist_ok=True)
    if args.bypass_cost_csv:
        bypass_cost_path = Path(args.bypass_cost_csv)
        cost_source = "reused"
        cost_profile_command = None
    else:
        bypass_cost_path, cost_profile_command = profile_bypass_costs(
            args, out, chosen_lora, training_data_dir, source
        )
        cost_source = "profiled_pretraining"
    costs, protected, invalid_cost, eligible = load_costs(
        bypass_cost_path, all_blocks, chosen_lora["selected_blocks"], args.protected_block)
    chosen_bypass, bypass_scan, policy, cap, fidelity_checked = choose_bypass(
        groups, indices, eligible, costs, weights, chosen_lora["reported_gflops"],
        reference_cost, args)

    lora_meta = dict(chosen_lora["metadata"])
    lora_meta.update({
        "selection_policy": "auto-noise-normalized-threshold",
        "auto_selected_label": chosen_lora["label"],
        "auto_utility_retention": chosen_lora["utility_retention"],
        "auto_selection_report": str(out / "auto_selection.json"),
        "selected_lora_blocks": list(chosen_lora["selected_blocks"]),
    })
    write_json(out / "dit_sr_grad_metadata.json", lora_meta)
    public_candidates = [{key: row[key] for key in (
        "label", "selected_count", "utility", "utility_retention", "utility_constraint_met",
        "reported_gflops", "mean_step_time_ms", "peak_cuda_mb")}
        for row in candidate_rows]
    write_rows(out / "lora_candidates.csv", public_candidates)
    write_rows(out / "bypass_threshold_scan.csv", bypass_scan)
    write_rows(out / "bypass_policy_by_noise.csv", policy)
    schedule = chosen_bypass["schedule"].split()
    additional_protected = sorted(protected - set(chosen_lora["selected_blocks"]), key=indices.get)
    report = {
        "method": "joint-pretraining-calibration",
        "selected_lora_label": chosen_lora["label"],
        "selected_lora_threshold": chosen_lora["metadata"]["importance_threshold"],
        "selected_lora_blocks": chosen_lora["selected_blocks"],
        "selected_lora_count": chosen_lora["selected_count"],
        "lora_utility": chosen_lora["utility"],
        "lora_utility_reference": reference_utility,
        "lora_utility_retention": chosen_lora["utility_retention"],
        "min_lora_utility_retention": args.min_lora_utility_retention,
        "lora_cost_metric": args.lora_cost_metric,
        "lora_reported_gflops": chosen_lora["reported_gflops"],
        "reference_top8_reported_gflops": reference_cost,
        "bypass_cost_source": cost_source,
        "bypass_cost_csv": str(bypass_cost_path),
        "bypass_cost_profile_command": (
            shlex.join(cost_profile_command) if cost_profile_command else None
        ),
        "bypass_threshold": chosen_bypass["bypass_threshold"],
        "bypass_schedule": schedule,
        "mean_bypass_budget": chosen_bypass["mean_bypass_budget"],
        "estimated_bypass_saved_gflops": chosen_bypass["estimated_saved_gflops"],
        "estimated_total_gflops": chosen_bypass["estimated_total_gflops"],
        "estimated_total_reduction_vs_top8_pct": chosen_bypass["estimated_total_reduction_vs_top8_pct"],
        "target_total_compute_reduction_pct": args.target_total_compute_reduction_pct,
        "target_met": chosen_bypass["target_met"],
        "max_bypass_importance_mass": args.max_bypass_importance_mass,
        "max_bypass_count": cap,
        "additional_protected_blocks": additional_protected,
        "nonpositive_cost_blocks_protected": sorted(invalid_cost, key=indices.get),
        "gradient_fidelity_checked": fidelity_checked,
        "min_gradient_cosine": args.min_gradient_cosine if fidelity_checked else None,
        "max_relative_gradient_error": args.max_relative_gradient_error if fidelity_checked else None,
        "calibration_only": True,
        "note": "Bypass savings are additive estimates from supported profiler operators. Verify the whole policy and gradient fidelity before formal training.",
    }
    write_json(out / "auto_selection.json", report)
    write_json(out / "training_arguments.json", {
        "lora_selection": "metadata", "lora_selection_file": str(out / "dit_sr_grad_metadata.json"),
        "lora_block_budget": chosen_lora["selected_count"], "topk_blocks": chosen_lora["selected_count"],
        "blockskip_schedule": schedule, "blockskip_importance_csv": args.importance_csv,
        "blockskip_importance_step": args.importance_step,
        "protect_selected_lora_blocks": True,
        "blockskip_protected_blocks": additional_protected,
        "blockskip_min_run": 1, "blockskip_max_run": args.expected_blocks,
        "blockskip_max_runs": args.expected_blocks, "residual_execution": "single_pass",
    })
    command = [
        "python3", "train_dit_sr_all_lora_importance.py",
        "--config_path", source.get("config_path", "configs/realsr_DiT.yaml"),
        "--ckpt_path", source.get("ckpt_path", "weights/realsr.pth"),
        "--autoencoder_ckpt", source.get("autoencoder_ckpt", "weights/autoencoder_vq_f4.pth"),
        "--data_dir", training_data_dir, "--output_dir", args.train_output_dir,
        "--loss_mode", "official", "--image_size", str(source.get("image_size", 256)),
        "--lq_size", str(source.get("lq_size", 64)), "--target", source["target"],
        "--rank", str(source["rank"]), "--alpha", str(source["alpha"]),
        "--lora_selection", "metadata", "--lora_block_budget", str(chosen_lora["selected_count"]),
        "--topk_blocks", str(chosen_lora["selected_count"]),
        "--lora_selection_file", str(out / "dit_sr_grad_metadata.json"),
        "--blockskip_schedule", *schedule,
        "--blockskip_importance_csv", args.importance_csv,
        "--blockskip_importance_step", str(args.importance_step),
        "--protect_selected_lora_blocks",
        "--blockskip_min_run", "1", "--blockskip_max_run", str(args.expected_blocks),
        "--blockskip_max_runs", str(args.expected_blocks),
        "--residual_execution", "single_pass", "--train_steps", str(args.train_steps),
        "--profile_steps", "0", str(args.train_steps), "--profile_batches", "5",
        "--profile_noise_ratios", *[f"{noise:g}" for noise in anchors],
        "--train_noise_ratios", *[f"{noise:g}" for noise in anchors],
        "--batch_size", "1", "--lr", "1e-5", "--grad_clip", "1.0",
        "--max_images", "0", "--num_workers", "0", "--seed", str(args.seed),
        "--profile_seed", str(args.seed), "--log_every", "10",
    ]
    if additional_protected:
        insert_at = command.index("--blockskip_min_run")
        command[insert_at:insert_at] = ["--blockskip_protected_blocks", *additional_protected]
    run_script = out / "run_selected_training.sh"
    run_script.write_text(
        "#!/usr/bin/env bash\nset -eu\n\n" + shlex.join(command) + "\n",
        encoding="utf-8",
    )
    run_script.chmod(0o755)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--importance_csv", required=True)
    parser.add_argument("--source_metadata", required=True)
    parser.add_argument("--lora_policy_dir", required=True)
    parser.add_argument("--lora_compute_csv", required=True)
    parser.add_argument(
        "--bypass_cost_csv",
        default="",
        help="Reuse an existing K-specific cost CSV; omit to profile it automatically.",
    )
    parser.add_argument("--fidelity_csv", default="")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--data_dir", default="", help="Override the calibration metadata data directory")
    parser.add_argument("--cost_profile_script", default="profile_dit_bypass_block_costs.py")
    parser.add_argument("--config_path", default="")
    parser.add_argument("--ckpt_path", default="")
    parser.add_argument("--autoencoder_ckpt", default="")
    parser.add_argument("--cost_noise_ratio", type=float, default=0.4)
    parser.add_argument("--cost_warmup", type=int, default=3)
    parser.add_argument("--cost_repeats", type=int, default=10)
    parser.add_argument("--cost_seed", type=int, default=4242)
    parser.add_argument("--cpu_cost_profile", action="store_true")
    parser.add_argument("--train_output_dir", default="outputs/dit_sr_auto_lora_bypass_seed42")
    parser.add_argument("--train_steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--importance_step", type=int, default=0)
    parser.add_argument("--expected_blocks", type=int, default=58)
    parser.add_argument("--score_key", default="normalized_grad_score")
    parser.add_argument("--noise_weights", type=float, nargs="+")
    parser.add_argument("--max_lora_blocks", type=int, default=8)
    parser.add_argument("--min_lora_utility_retention", type=float, default=0.8)
    parser.add_argument("--lora_cost_metric", choices=["reported_gflops", "mean_step_time_ms"],
                        default="reported_gflops")
    parser.add_argument("--target_total_compute_reduction_pct", type=float, default=10.0)
    parser.add_argument("--max_bypass_importance_mass", type=float, default=0.2)
    parser.add_argument("--max_bypass_fraction", type=float, default=0.5)
    parser.add_argument("--max_bypass_count", type=int)
    parser.add_argument("--protected_block", action="append", default=[])
    parser.add_argument("--min_gradient_cosine", type=float, default=0.9)
    parser.add_argument("--max_relative_gradient_error", type=float, default=0.5)
    args = parser.parse_args()
    for name in ("min_lora_utility_retention", "max_bypass_importance_mass", "max_bypass_fraction"):
        value = getattr(args, name)
        if not math.isfinite(value) or not 0 < value <= 1:
            parser.error(f"{name} must be in (0, 1]")
    if not 0 <= args.target_total_compute_reduction_pct < 100:
        parser.error("target_total_compute_reduction_pct must be in [0, 100)")
    if args.train_steps < 1:
        parser.error("train_steps must be positive")
    if args.cost_warmup < 0 or args.cost_repeats < 1:
        parser.error("cost_warmup must be non-negative and cost_repeats must be positive")
    try:
        report = auto_select(args)
    except (OSError, ValueError, KeyError, TypeError, subprocess.CalledProcessError) as exc:
        parser.error(str(exc))
    print(json.dumps(report, indent=2, allow_nan=False))
    if not report["gradient_fidelity_checked"]:
        print("WARNING: no gradient-fidelity CSV was supplied; run a policy fidelity audit before formal training.")
    if not report["target_met"]:
        print("WARNING: the safest available threshold did not reach the requested compute target.")


if __name__ == "__main__":
    main()
