#!/usr/bin/env python3
"""Build and optionally test a complete threshold-LoRA ResShift bypass policy."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import shlex
import subprocess
import sys
from pathlib import Path

from train_resshift_lora_importance import (
    aggregate_importance,
    choose_blocks,
    read_csv,
)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def finite(value, label: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"Non-finite {label}: {value}")
    return result


def importance_groups(rows, step: int, score_key: str):
    chosen = [row for row in rows if int(row["train_step"]) == step]
    if not chosen:
        available = sorted({int(row["train_step"]) for row in rows})
        raise ValueError(f"Importance step {step} is unavailable; available={available}")
    groups = {}
    indices = {}
    for row in chosen:
        noise = float(row["noise_ratio"])
        block = str(row["block"])
        groups.setdefault(noise, {})[block] = finite(row[score_key], score_key)
        indices[block] = int(row["block_index"])
    expected = set(indices)
    for noise, scores in groups.items():
        if set(scores) != expected:
            raise ValueError(f"Importance coverage differs at noise={noise:g}")
    return groups, indices


def load_costs(path: Path, all_blocks: set[str], selected: set[str]):
    rows = read_csv(path)
    costs = {}
    invalid = set()
    for row in rows:
        block = str(row["block"])
        if block not in all_blocks:
            raise ValueError(f"Cost CSV contains unknown block: {block}")
        if finite(row.get("max_loss_abs_diff", 0), "loss difference") != 0:
            invalid.add(block)
            continue
        gflops = finite(row["reported_gflops_saved"], f"{block} saved GFLOPs")
        if gflops > 0:
            costs[block] = {
                "gflops": gflops,
                "milliseconds": finite(row["step_time_saved_ms"], f"{block} saved time"),
                "memory_mb": finite(row["peak_cuda_mem_saved_mb"], f"{block} saved memory"),
            }
        else:
            invalid.add(block)
    eligible = all_blocks - selected - invalid
    missing = eligible - costs.keys()
    if missing:
        raise ValueError(f"Cost CSV is missing eligible blocks: {sorted(missing)}")
    if not eligible:
        raise ValueError("No positive-FLOPs frozen bypass blocks remain")
    return costs, eligible, invalid


def choose_policy(groups, indices, costs, eligible, max_mass, max_count):
    rows = []
    for noise in sorted(groups):
        total = sum(groups[noise].values())
        if total <= 0:
            raise ValueError(f"Non-positive importance at noise={noise:g}")
        shares = {block: groups[noise][block] / total for block in eligible}
        candidates = sorted(eligible, key=indices.get)
        best = (0.0, 0.0, ())
        for count in range(1, min(max_count, len(candidates)) + 1):
            for subset in itertools.combinations(candidates, count):
                risk = sum(shares[block] for block in subset)
                if risk > max_mass + 1e-12:
                    continue
                saved = sum(costs[block]["gflops"] for block in subset)
                key = (saved, -risk, tuple(-indices[block] for block in subset))
                current_key = (best[0], -best[1], tuple(-indices[block] for block in best[2]))
                if key > current_key:
                    best = (saved, risk, subset)
        saved, risk, subset = best
        ordered = sorted(subset, key=indices.get)
        rows.append({
            "noise_ratio": noise,
            "bypass_budget": len(ordered),
            "bypass_importance_mass": risk,
            "estimated_saved_gflops": saved,
            "estimated_saved_time_ms": sum(costs[b]["milliseconds"] for b in ordered),
            "estimated_saved_memory_mb": sum(costs[b]["memory_mb"] for b in ordered),
            "skip_blocks": ";".join(ordered),
        })
    return rows


def training_command(args, policy_csv: Path, output_dir: str, mode: str):
    command = [
        sys.executable,
        "train_resshift_lora_importance.py",
        "--resshift_root", args.resshift_root,
        "--config_path", args.config_path,
        "--checkpoint", args.checkpoint,
        "--autoencoder_checkpoint", args.autoencoder_checkpoint,
        "--data_dir", args.data_dir,
        "--output_dir", output_dir,
        "--method_name", f"Threshold-K{args.selected_lora_count}-{mode}",
        "--importance_csv", args.importance_csv,
        "--importance_step", str(args.importance_step),
        "--score_key", args.score_key,
        "--lora_selection", "threshold",
        "--importance_threshold", str(args.importance_threshold),
        "--image_size", str(args.image_size),
        "--sr_scale", str(args.sr_scale),
        "--rank", str(args.rank),
        "--alpha", str(args.alpha),
        "--train_steps", str(args.gate_steps),
        "--train_noise_ratios", *[f"{value:g}" for value in args.noise_ratios],
        "--batch_size", "1",
        "--lr", str(args.lr),
        "--grad_clip", "1.0",
        "--dtype", args.dtype,
        "--max_images", "0",
        "--num_workers", "0",
        "--seed", str(args.seed),
        "--log_every", "1",
    ]
    if mode != "Native":
        command.extend([
            "--blockskip_policy_csv", str(policy_csv),
            "--blockskip_importance_csv", args.importance_csv,
            "--blockskip_importance_step", str(args.importance_step),
            "--blockskip_score_key", args.score_key,
            "--protect_selected_lora_blocks",
            "--residual_execution", "single_pass",
        ])
        if mode == "Controller-B0":
            command.append("--blockskip_controller_only")
    return command


def summarize_gate(run_dirs: dict[str, Path], warmup_steps: int, min_speedup: float):
    measurements = {}
    for mode, run_dir in run_dirs.items():
        rows = read_csv(run_dir / "train_log.csv")
        steady = [row for row in rows if int(row["step"]) > warmup_steps]
        if not steady:
            raise ValueError(f"No steady timing rows in {run_dir}")
        measurements[mode] = {
            "mean_step_time_s": sum(float(row["train_step_time_s"]) for row in rows) / len(rows),
            "steady_step_time_s": sum(float(row["train_step_time_s"]) for row in steady) / len(steady),
            "mean_skipped_blocks": sum(float(row["skipped_block_count"]) for row in steady) / len(steady),
            "fallback_block_events": sum(float(row["fallback_blocks"]) for row in steady),
            "peak_cuda_mem_mb": max(float(row["peak_cuda_mem_mb"]) for row in rows),
        }
    native = measurements["Native"]["steady_step_time_s"]
    controller = measurements["Controller-B0"]["steady_step_time_s"]
    bypass = measurements["Bypass"]["steady_step_time_s"]
    net = 100 * (native - bypass) / native
    gross = 100 * (controller - bypass) / controller
    overhead = 100 * (controller - native) / native
    fallback = measurements["Bypass"]["fallback_block_events"]
    return {
        "measurements": measurements,
        "controller_overhead_vs_native_pct": overhead,
        "bypass_saving_vs_controller_pct": gross,
        "net_bypass_speedup_vs_native_pct": net,
        "minimum_required_net_speedup_pct": min_speedup,
        "enable_bypass": bool(net >= min_speedup and fallback == 0),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resshift_root", default=".")
    parser.add_argument("--config_path", default="configs/realsr_swinunet_realesrgan256.yaml")
    parser.add_argument("--checkpoint", default="weights/resshift_realsrx4_s15_v1.pth")
    parser.add_argument("--autoencoder_checkpoint", default="weights/autoencoder_vq_f4.pth")
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--importance_csv", required=True)
    parser.add_argument("--importance_step", type=int, default=0)
    parser.add_argument("--score_key", default="normalized_grad_score")
    parser.add_argument("--importance_threshold", type=float, default=1.0)
    parser.add_argument("--cost_csv", default="")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--sr_scale", type=int, default=4)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=16.0)
    parser.add_argument("--noise_ratios", type=float, nargs="+", default=[0.05, 0.2, 0.4, 0.6, 0.8, 0.95])
    parser.add_argument("--max_bypass_importance_mass", type=float, default=0.10)
    parser.add_argument("--max_bypass_count", type=int, default=6)
    parser.add_argument("--cost_noise_ratio", type=float, default=0.4)
    parser.add_argument("--cost_warmup", type=int, default=2)
    parser.add_argument("--cost_repeats", type=int, default=8)
    parser.add_argument("--gate_steps", type=int, default=100)
    parser.add_argument("--gate_warmup_steps", type=int, default=10)
    parser.add_argument("--min_net_speedup_pct", type=float, default=1.0)
    parser.add_argument("--train_output_prefix", default="outputs/resshift_final_k6_gate")
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--dtype", choices=["fp32", "bf16", "fp16"], default="fp32")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run_gate", action="store_true")
    args = parser.parse_args()

    if not 0 < args.max_bypass_importance_mass <= 1:
        parser.error("max_bypass_importance_mass must be in (0, 1]")
    if args.max_bypass_count < 1:
        parser.error("max_bypass_count must be positive")
    if args.gate_steps < 2 or not 0 <= args.gate_warmup_steps < args.gate_steps:
        parser.error("Require 0 <= gate_warmup_steps < gate_steps and gate_steps >= 2")
    if args.min_net_speedup_pct < 0:
        parser.error("min_net_speedup_pct must be non-negative")

    out = Path(args.output_dir)
    if out.exists() and any(out.iterdir()):
        parser.error(f"Output directory is not empty: {out}")
    out.mkdir(parents=True, exist_ok=True)

    importance_rows = read_csv(Path(args.importance_csv))
    aggregated = aggregate_importance(importance_rows, args.importance_step, args.score_key)
    all_blocks = [row["block"] for row in sorted(aggregated, key=lambda row: row["block_index"])]
    selected, selection_report = choose_blocks(
        all_blocks, aggregated, "threshold", args.importance_threshold, 1
    )
    args.selected_lora_count = len(selected)

    if args.cost_csv:
        cost_csv = Path(args.cost_csv)
        cost_command = None
    else:
        profile_dir = out / "bypass_cost_profile"
        script = Path(__file__).resolve().parent / "profile_resshift_bypass_costs.py"
        cost_command = [
            sys.executable, str(script),
            "--resshift_root", args.resshift_root,
            "--config_path", args.config_path,
            "--checkpoint", args.checkpoint,
            "--autoencoder_checkpoint", args.autoencoder_checkpoint,
            "--data_dir", args.data_dir,
            "--output_dir", str(profile_dir),
            "--importance_csv", args.importance_csv,
            "--importance_step", str(args.importance_step),
            "--score_key", args.score_key,
            "--lora_selection", "threshold",
            "--importance_threshold", str(args.importance_threshold),
            "--image_size", str(args.image_size),
            "--sr_scale", str(args.sr_scale),
            "--rank", str(args.rank),
            "--alpha", str(args.alpha),
            "--noise_ratio", str(args.cost_noise_ratio),
            "--warmup", str(args.cost_warmup),
            "--repeats", str(args.cost_repeats),
            "--seed", str(args.seed + 4000),
        ]
        print("Profiling K-specific ResShift bypass costs...")
        subprocess.run(cost_command, check=True)
        cost_csv = profile_dir / "block_backward_costs.csv"

    groups, indices = importance_groups(importance_rows, args.importance_step, args.score_key)
    costs, eligible, invalid = load_costs(cost_csv, set(all_blocks), set(selected))
    policy = choose_policy(
        groups,
        indices,
        costs,
        eligible,
        args.max_bypass_importance_mass,
        args.max_bypass_count,
    )
    policy_csv = out / "bypass_policy_by_noise.csv"
    write_csv(policy_csv, policy)

    weights = 1 / len(policy)
    report = {
        "model": "ResShift",
        "selection": "gradient-threshold LoRA plus cost-aware noise-conditioned bypass",
        "selected_lora_blocks": selected,
        "selected_lora_count": len(selected),
        "selected_lora_utility": selection_report["selected_utility"],
        "importance_threshold": args.importance_threshold,
        "cost_csv": str(cost_csv),
        "cost_profile_command": shlex.join(cost_command) if cost_command else None,
        "eligible_positive_cost_blocks": sorted(eligible, key=indices.get),
        "nonpositive_or_inexact_blocks": sorted(invalid, key=indices.get),
        "max_bypass_importance_mass": args.max_bypass_importance_mass,
        "max_bypass_count": args.max_bypass_count,
        "mean_bypass_budget": sum(row["bypass_budget"] * weights for row in policy),
        "mean_estimated_saved_gflops": sum(row["estimated_saved_gflops"] * weights for row in policy),
        "mean_estimated_saved_time_ms": sum(row["estimated_saved_time_ms"] * weights for row in policy),
        "hardware_gate_required": True,
    }
    write_json(out / "auto_selection.json", report)

    commands = {}
    run_dirs = {}
    for mode, suffix in (
        ("Native", "native"),
        ("Controller-B0", "controller_b0"),
        ("Bypass", "bypass"),
    ):
        run_dirs[mode] = Path(
            f"{args.train_output_prefix}_{suffix}_{args.gate_steps}_seed{args.seed}"
        )
        commands[mode] = training_command(
            args,
            policy_csv,
            str(run_dirs[mode]),
            mode,
        )
    gate_script = out / "run_hardware_gate.sh"
    gate_script.write_text(
        "#!/usr/bin/env bash\nset -eu\n\n"
        + "\n\n".join(shlex.join(command) for command in commands.values())
        + "\n",
        encoding="utf-8",
    )
    gate_script.chmod(0o755)
    print(json.dumps(report, indent=2))
    print(f"Wrote policy to {policy_csv}")
    print(f"Wrote hardware gate to {gate_script}")
    if args.run_gate:
        subprocess.run(["bash", str(gate_script)], check=True)
        gate = summarize_gate(
            run_dirs, args.gate_warmup_steps, args.min_net_speedup_pct
        )
        write_json(out / "hardware_gate_summary.json", gate)
        print(json.dumps(gate, indent=2))


if __name__ == "__main__":
    main()
