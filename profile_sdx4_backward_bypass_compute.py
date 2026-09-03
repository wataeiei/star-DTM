#!/usr/bin/env python3
"""Measure actual compute changes from SD-x4 single-pass backward bypass."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import time
from pathlib import Path

import pandas as pd
import torch
from torch.profiler import ProfilerActivity, profile
from torch.utils.data import DataLoader

import adaptive_grad_blockskip as adaptive
import onboard_sandwich_lora_sr as core


def save_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def candidate_blocks(unet: torch.nn.Module) -> list[str]:
    blocks = []
    for name, module in unet.named_modules():
        if module.__class__.__name__ == "BasicTransformerBlock" and ".transformer_blocks." in name:
            blocks.append(name)
    return sorted(set(blocks), key=core.natural_key)


def load_bypass_blocks(
    args: argparse.Namespace,
    candidates: list[str],
    protected: set[str],
) -> list[str]:
    if args.bypass_blocks:
        selected = list(dict.fromkeys(args.bypass_blocks))
    else:
        if not args.importance_csv:
            raise SystemExit("Provide --bypass_blocks or --importance_csv")
        frame = pd.read_csv(args.importance_csv)
        if "block" not in frame.columns:
            raise SystemExit(f"Missing 'block' column in {args.importance_csv}")
        score_column = next(
            (
                key
                for key in (
                    args.score_column,
                    "normalized_grad_score",
                    "selection_score",
                    "grad_score",
                    "grad_norm",
                )
                if key and key in frame.columns
            ),
            None,
        )
        if score_column is None:
            raise SystemExit(
                "No score column found. Pass --score_column; available columns: "
                + ", ".join(frame.columns)
            )
        scores = (
            frame[["block", score_column]]
            .dropna()
            .groupby("block", as_index=False)[score_column]
            .mean()
            .sort_values(score_column, ascending=True)
        )
        selected = [
            str(block)
            for block in scores["block"]
            if str(block) in candidates and str(block) not in protected
        ][: args.skip_count]

    unknown = sorted(set(selected) - set(candidates))
    collisions = sorted(set(selected) & protected)
    if unknown:
        raise SystemExit(f"Unknown bypass blocks: {unknown}")
    if collisions:
        raise SystemExit(f"Protected LoRA blocks cannot be bypassed: {collisions}")
    if len(selected) != args.skip_count:
        raise SystemExit(f"Requested {args.skip_count} bypass blocks, found {len(selected)}")
    return selected


def device_kernel_time_ms(profiler) -> float:
    total_us = 0.0
    for event in profiler.events():
        if "cuda" not in str(getattr(event, "device_type", "")).lower():
            continue
        time_range = getattr(event, "time_range", None)
        if time_range is not None and hasattr(time_range, "elapsed_us"):
            total_us += float(time_range.elapsed_us())
        else:
            total_us += float(getattr(event, "device_time_total", 0.0) or 0.0)
    return total_us / 1000.0


def reported_flops(profiler) -> float:
    return float(sum(float(getattr(event, "flops", 0.0) or 0.0) for event in profiler.key_averages()))


def clear_grads(unet: torch.nn.Module) -> None:
    for parameter in unet.parameters():
        parameter.grad = None


def run_step(
    pipe,
    batch: dict,
    train_args: argparse.Namespace,
    device: torch.device,
    controller: adaptive.ResidualBlockController,
    mode: str,
    rng_state,
) -> float:
    clear_grads(pipe.unet)
    controller.set_mode(mode)
    adaptive.restore_rng(rng_state)
    loss, _prediction, _target = core.diffusion_train_loss(
        pipe, batch, train_args, device, torch.float32, use_amp=False
    )
    loss.backward()
    return float(loss.detach())


def timed_measurement(
    pipe,
    batch: dict,
    train_args: argparse.Namespace,
    device: torch.device,
    controller: adaptive.ResidualBlockController,
    mode: str,
    rng_state,
    repeats: int,
) -> tuple[list[float], list[float]]:
    elapsed_ms = []
    losses = []
    for _ in range(repeats):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        started = time.perf_counter()
        loss = run_step(pipe, batch, train_args, device, controller, mode, rng_state)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed_ms.append((time.perf_counter() - started) * 1000.0)
        losses.append(loss)
    return elapsed_ms, losses


def profiled_measurement(
    pipe,
    batch: dict,
    train_args: argparse.Namespace,
    device: torch.device,
    controller: adaptive.ResidualBlockController,
    mode: str,
    rng_state,
) -> tuple[float, float, float]:
    activities = [ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(ProfilerActivity.CUDA)
        torch.cuda.reset_peak_memory_stats(device)
    with profile(activities=activities, with_flops=True, profile_memory=True) as prof:
        loss = run_step(pipe, batch, train_args, device, controller, mode, rng_state)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
    peak_mb = (
        torch.cuda.max_memory_allocated(device) / (1024.0**2)
        if device.type == "cuda"
        else 0.0
    )
    return reported_flops(prof), device_kernel_time_ms(prof), peak_mb


def reduction(full: float, bypass: float) -> float:
    if not math.isfinite(full) or full <= 0:
        return float("nan")
    return (full - bypass) / full * 100.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lora_dir", required=True)
    parser.add_argument("--train_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--importance_csv", default="")
    parser.add_argument("--score_column", default="")
    parser.add_argument("--bypass_blocks", nargs="*", default=[])
    parser.add_argument("--skip_count", type=int, default=6)
    parser.add_argument("--hr_size", type=int, default=256)
    parser.add_argument("--lr_size", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--noise_level", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    if args.skip_count <= 0 or args.repeats <= 0:
        raise SystemExit("--skip_count and --repeats must be positive")

    core.set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    pipe = core.load_pipeline(dtype=torch.float32).to(device)
    pipe.vae.requires_grad_(False)
    pipe.text_encoder.requires_grad_(False)
    pipe.unet.requires_grad_(False)
    metadata = core.load_lora(pipe.unet, args.lora_dir, device, torch.float32)
    pipe.unet.train()

    candidates = candidate_blocks(pipe.unet)
    protected = set(metadata.get("topk_block_names") or metadata.get("selected_lora_blocks") or [])
    bypass_blocks = load_bypass_blocks(args, candidates, protected)
    controller = adaptive.ResidualBlockController(
        pipe.unet,
        {name: name for name in candidates},
        cache_device="cpu",
        cache_dtype=torch.float16,
    )
    controller.configure(bypass_blocks)

    dataset = core.HrImageDataset(args.train_dir, args.hr_size, args.lr_size)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    batch = next(iter(loader))
    train_args = argparse.Namespace(noise_level=args.noise_level)
    rng_state = adaptive.snapshot_rng(device)

    for _ in range(args.warmup):
        run_step(pipe, batch, train_args, device, controller, "full", rng_state)
        run_step(pipe, batch, train_args, device, controller, "single_skip", rng_state)

    full_ms, full_losses = timed_measurement(
        pipe, batch, train_args, device, controller, "full", rng_state, args.repeats
    )
    bypass_ms, bypass_losses = timed_measurement(
        pipe, batch, train_args, device, controller, "single_skip", rng_state, args.repeats
    )
    full_flops, full_kernel_ms, full_peak_mb = profiled_measurement(
        pipe, batch, train_args, device, controller, "full", rng_state
    )
    bypass_flops, bypass_kernel_ms, bypass_peak_mb = profiled_measurement(
        pipe, batch, train_args, device, controller, "single_skip", rng_state
    )

    rows = []
    for mode, times, losses, flops, kernel_ms, peak_mb, bypassed in (
        ("full_backward", full_ms, full_losses, full_flops, full_kernel_ms, full_peak_mb, 0),
        (
            "single_pass_backward_bypass",
            bypass_ms,
            bypass_losses,
            bypass_flops,
            bypass_kernel_ms,
            bypass_peak_mb,
            len(bypass_blocks),
        ),
    ):
        rows.append(
            {
                "mode": mode,
                "total_transformer_blocks": len(candidates),
                "forward_executed_blocks": len(candidates),
                "backward_bypassed_blocks": bypassed,
                "backward_bypass_ratio_pct": bypassed / len(candidates) * 100.0,
                "mean_step_time_ms": statistics.mean(times),
                "std_step_time_ms": statistics.stdev(times) if len(times) > 1 else 0.0,
                "reported_gflops": flops / 1e9,
                "cuda_kernel_time_ms": kernel_ms,
                "peak_cuda_mem_mb": peak_mb,
                "mean_loss": statistics.mean(losses),
            }
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_rows(output_dir / "compute_audit_modes.csv", rows)
    comparison = {
        "model": core.MODEL_ID,
        "adapter": str(args.lora_dir),
        "total_transformer_blocks": len(candidates),
        "forward_executed_blocks": len(candidates),
        "backward_bypassed_blocks": len(bypass_blocks),
        "backward_bypass_ratio_pct": len(bypass_blocks) / len(candidates) * 100.0,
        "bypass_blocks": bypass_blocks,
        "protected_lora_blocks": sorted(protected, key=core.natural_key),
        "full_mean_step_time_ms": statistics.mean(full_ms),
        "bypass_mean_step_time_ms": statistics.mean(bypass_ms),
        "step_time_reduction_pct": reduction(statistics.mean(full_ms), statistics.mean(bypass_ms)),
        "full_reported_gflops": full_flops / 1e9,
        "bypass_reported_gflops": bypass_flops / 1e9,
        "reported_flops_reduction_pct": reduction(full_flops, bypass_flops),
        "full_cuda_kernel_time_ms": full_kernel_ms,
        "bypass_cuda_kernel_time_ms": bypass_kernel_ms,
        "cuda_kernel_time_reduction_pct": reduction(full_kernel_ms, bypass_kernel_ms),
        "full_peak_cuda_mem_mb": full_peak_mb,
        "bypass_peak_cuda_mem_mb": bypass_peak_mb,
        "peak_cuda_memory_reduction_pct": reduction(full_peak_mb, bypass_peak_mb),
        "max_loss_abs_diff": max(abs(a - b) for a, b in zip(full_losses, bypass_losses)),
        "profiler_note": "PyTorch reported FLOPs cover supported operators only; CUDA time and wall time are measured separately.",
    }
    with (output_dir / "compute_audit_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(comparison, handle, indent=2)

    print(json.dumps(comparison, indent=2))
    print(f"Wrote compute audit to {output_dir}")


if __name__ == "__main__":
    main()
