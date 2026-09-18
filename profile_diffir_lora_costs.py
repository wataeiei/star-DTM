#!/usr/bin/env python3
"""Benchmark complete DiffIR-S2 sparse-LoRA candidate configurations.

This script consumes ``lora_importance_ranked.csv`` from
``profile_diffir_grad.py``.  It benchmarks the actual official S2 training
step for several ranking-prefix K values and also records structural LoRA cost
estimates for every DIRformer block.  Candidate timing is the primary result;
per-block FLOPs are an additive local-adapter estimate, not a substitute for
the end-to-end measurement.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from pathlib import Path
from typing import Any

import numpy as np

from profile_diffir_grad import (
    ImageFolderDataset,
    audit_overlap,
    block_from_qkv,
    inject_qkv_lora,
    load_models,
    natural_key,
    set_seed,
    write_csv,
)


def read_importance(path: Path, expected_blocks: int) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    required = {
        "block",
        "importance_rank",
        "normalized_grad_score",
        "score_share",
        "cumulative_utility",
    }
    if not rows:
        raise ValueError(f"Importance CSV is empty: {path}")
    missing = sorted(required - set(rows[0]))
    if missing:
        raise ValueError("Importance CSV is missing columns: " + ", ".join(missing))
    parsed = []
    for row in rows:
        parsed.append(
            {
                "block": row["block"],
                "importance_rank": int(row["importance_rank"]),
                "normalized_grad_score": float(row["normalized_grad_score"]),
                "score_share": float(row["score_share"]),
                "cumulative_utility": float(row["cumulative_utility"]),
            }
        )
    parsed.sort(key=lambda row: row["importance_rank"])
    if len(parsed) != expected_blocks:
        raise ValueError(f"Expected {expected_blocks} importance rows, found {len(parsed)}")
    if [row["importance_rank"] for row in parsed] != list(range(1, len(parsed) + 1)):
        raise ValueError("Importance ranks must be unique and contiguous from 1")
    if len({row["block"] for row in parsed}) != len(parsed):
        raise ValueError("Importance CSV contains duplicate blocks")
    return parsed


def configure_wrappers(injected, selected_blocks: set[str]) -> list[Any]:
    parameters = []
    for module_name, wrapper in injected.items():
        enabled = block_from_qkv(module_name) in selected_blocks
        wrapper.enabled = enabled
        wrapper.lora_down.weight.requires_grad_(enabled)
        wrapper.lora_up.weight.requires_grad_(enabled)
        wrapper.lora_down.weight.grad = None
        wrapper.lora_up.weight.grad = None
        if enabled:
            parameters.extend((wrapper.lora_down.weight, wrapper.lora_up.weight))
    return parameters


def sync(device) -> None:
    import torch

    if device.type == "cuda":
        torch.cuda.synchronize(device)


def official_step(s2, s1, batch, optimizer, args, device) -> tuple[float, float, float]:
    import torch
    import torch.nn.functional as F

    gt = batch["gt"].to(device, non_blocking=True)
    lq = batch["lq"].to(device, non_blocking=True)
    optimizer.zero_grad(set_to_none=True)
    with torch.no_grad():
        teacher_ipr, _ = s1.E(lq, gt)
    sr, pred_ipr_list = s2(lq, teacher_ipr)
    pixel_loss = F.l1_loss(sr, gt)
    prior_loss = F.l1_loss(pred_ipr_list[-1], teacher_ipr.detach())
    loss = args.pixel_weight * pixel_loss + args.prior_weight * prior_loss
    if not torch.isfinite(loss):
        raise RuntimeError("Non-finite DiffIR candidate-profile loss")
    loss.backward()
    optimizer.step()
    return (
        float(loss.detach().cpu()),
        float(pixel_loss.detach().cpu()),
        float(prior_loss.detach().cpu()),
    )


def collect_activation_shapes(s2, lq, injected) -> dict[str, tuple[int, ...]]:
    import torch

    shapes: dict[str, tuple[int, ...]] = {}
    hooks = []
    for module_name, wrapper in injected.items():
        block = block_from_qkv(module_name)

        def capture(_module, inputs, block_name=block):
            shapes[block_name] = tuple(int(value) for value in inputs[0].shape)

        hooks.append(wrapper.register_forward_pre_hook(capture))
    previous = {name: wrapper.enabled for name, wrapper in injected.items()}
    try:
        for wrapper in injected.values():
            wrapper.enabled = False
        s2.eval()
        with torch.no_grad():
            s2(lq)
    finally:
        for name, wrapper in injected.items():
            wrapper.enabled = previous[name]
        for hook in hooks:
            hook.remove()
    return shapes


def reported_gflops_for_step(s2, s1, batch, optimizer, args, device) -> float:
    import torch

    activities = [torch.profiler.ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    set_seed(args.seed + 700000)
    with torch.profiler.profile(activities=activities, with_flops=True) as profiler:
        official_step(s2, s1, batch, optimizer, args, device)
        sync(device)
    total = sum(float(event.flops or 0.0) for event in profiler.key_averages())
    return total / 1e9


def profile(args: argparse.Namespace) -> None:
    import torch
    from torch.utils.data import DataLoader

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is unavailable")
    if args.image_size != args.lq_size * args.sr_scale:
        raise SystemExit("Require image_size == lq_size * sr_scale")
    if args.sr_scale != 4:
        raise SystemExit("The released SISR DiffIR checkpoints used here are x4")
    candidate_ks = sorted(set(args.candidate_k))
    if not candidate_ks or candidate_ks[0] < 1 or candidate_ks[-1] > args.expected_blocks:
        raise SystemExit(f"candidate_k values must be in [1, {args.expected_blocks}]")

    set_seed(args.seed)
    device = torch.device(args.device)
    importance_path = Path(args.importance_csv).expanduser()
    importance = read_importance(importance_path, args.expected_blocks)
    dataset = ImageFolderDataset(
        Path(args.data_dir).expanduser(),
        args.image_size,
        args.lq_size,
        args.max_images,
        args.seed,
    )
    required = max(args.warmup_steps, args.timed_steps) * args.batch_size
    if len(dataset) < required:
        raise SystemExit(f"Need at least {required} calibration images, found {len(dataset)}")
    calibration_paths = dataset.paths[:required]
    protected_dirs = [Path(path).expanduser() for path in args.protected_eval_dir]
    overlap = audit_overlap(calibration_paths, protected_dirs)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    batches = []
    for batch in loader:
        batches.append(batch)
        if len(batches) >= max(args.warmup_steps, args.timed_steps):
            break

    s2, s1, loaded_keys = load_models(
        Path(args.s2_checkpoint).expanduser(),
        Path(args.s1_checkpoint).expanduser(),
        args.checkpoint_key,
        device,
    )
    injected = inject_qkv_lora(s2, args.rank, args.alpha)
    model_blocks = {block_from_qkv(name) for name in injected}
    ranked_blocks = {row["block"] for row in importance}
    if model_blocks != ranked_blocks:
        raise RuntimeError(
            "Importance/model block mismatch: "
            f"missing={sorted(model_blocks - ranked_blocks, key=natural_key)} "
            f"unknown={sorted(ranked_blocks - model_blocks, key=natural_key)}"
        )

    first_lq = batches[0]["lq"].to(device)
    shapes = collect_activation_shapes(s2, first_lq, injected)
    if set(shapes) != model_blocks:
        raise RuntimeError(f"Captured {len(shapes)} of {len(model_blocks)} block shapes")

    structural_rows = []
    importance_by_block = {row["block"]: row for row in importance}
    for module_name, wrapper in injected.items():
        block = block_from_qkv(module_name)
        batch, channels, height, width = shapes[block]
        out_channels = wrapper.base.out_channels
        local_macs = batch * height * width * args.rank * (channels + out_channels)
        # One forward plus approximately two equivalent backward matrix products.
        train_flops = 6 * local_macs
        parameters = wrapper.lora_down.weight.numel() + wrapper.lora_up.weight.numel()
        item = importance_by_block[block]
        structural_rows.append(
            {
                "block": block,
                "importance_rank": item["importance_rank"],
                "score_share": item["score_share"],
                "cumulative_utility": item["cumulative_utility"],
                "batch_size": batch,
                "input_channels": channels,
                "output_channels": out_channels,
                "height": height,
                "width": width,
                "lora_params": parameters,
                "estimated_lora_train_gflops": train_flops / 1e9,
                "importance_per_estimated_gflop": (
                    item["score_share"] / (train_flops / 1e9)
                    if train_flops > 0 else float("inf")
                ),
            }
        )
    structural_rows.sort(key=lambda row: row["importance_rank"])

    candidate_rows = []
    s2.train()
    for k in candidate_ks:
        selected = {row["block"] for row in importance[:k]}
        parameters = configure_wrappers(injected, selected)
        optimizer = torch.optim.Adam(parameters, lr=0.0, betas=(0.9, 0.99))
        utility = importance[k - 1]["cumulative_utility"]
        selected_structural = [row for row in structural_rows if row["block"] in selected]

        set_seed(args.seed + k * 1000)
        for index in range(args.warmup_steps):
            official_step(s2, s1, batches[index % len(batches)], optimizer, args, device)
        sync(device)

        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
        times = []
        losses = []
        pixels = []
        priors = []
        set_seed(args.seed + k * 1000 + 1)
        for index in range(args.timed_steps):
            sync(device)
            started = time.perf_counter()
            loss, pixel, prior = official_step(
                s2, s1, batches[index % len(batches)], optimizer, args, device
            )
            sync(device)
            times.append(time.perf_counter() - started)
            losses.append(loss)
            pixels.append(pixel)
            priors.append(prior)

        mean_time = float(np.mean(times))
        std_time = float(np.std(times, ddof=1)) if len(times) > 1 else 0.0
        peak_mb = (
            torch.cuda.max_memory_allocated(device) / (1024**2)
            if device.type == "cuda" else 0.0
        )
        reported_gflops = float("nan")
        if args.profile_flops:
            reported_gflops = reported_gflops_for_step(
                s2, s1, batches[0], optimizer, args, device
            )
        candidate_rows.append(
            {
                "label": f"K{k}",
                "selected_blocks": k,
                "block_fraction": k / args.expected_blocks,
                "utility_retained": utility,
                "lora_params": sum(row["lora_params"] for row in selected_structural),
                "estimated_lora_train_gflops": sum(
                    row["estimated_lora_train_gflops"] for row in selected_structural
                ),
                "reported_full_step_gflops": reported_gflops,
                "mean_step_time_s": mean_time,
                "std_step_time_s": std_time,
                "peak_cuda_mem_mb": peak_mb,
                "mean_loss": float(np.mean(losses)),
                "mean_pixel_loss": float(np.mean(pixels)),
                "mean_prior_l1_loss": float(np.mean(priors)),
                "warmup_steps": args.warmup_steps,
                "timed_steps": args.timed_steps,
                "selected_block_names": ";".join(
                    sorted(selected, key=natural_key)
                ),
            }
        )
        print(
            f"K{k}: utility={utility:.2%} time={mean_time:.6f}s "
            f"peak={peak_mb:.1f}MB params={candidate_rows[-1]['lora_params']}"
        )

    all_row = next((row for row in candidate_rows if row["selected_blocks"] == 44), None)
    if all_row is not None:
        for row in candidate_rows:
            row["time_reduction_vs_k44_pct"] = (
                100.0 * (all_row["mean_step_time_s"] - row["mean_step_time_s"])
                / all_row["mean_step_time_s"]
            )
            row["memory_reduction_vs_k44_pct"] = (
                100.0 * (all_row["peak_cuda_mem_mb"] - row["peak_cuda_mem_mb"])
                / all_row["peak_cuda_mem_mb"]
                if all_row["peak_cuda_mem_mb"] > 0 else float("nan")
            )

    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "block_lora_costs.csv", structural_rows)
    write_csv(output_dir / "candidate_compute_summary.csv", candidate_rows)
    metadata = {
        "model": "DiffIR-S2",
        "importance_csv": str(importance_path.resolve()),
        "s2_checkpoint": str(Path(args.s2_checkpoint).expanduser().resolve()),
        "s1_checkpoint": str(Path(args.s1_checkpoint).expanduser().resolve()),
        "checkpoint_keys": loaded_keys,
        "data_dir": str(Path(args.data_dir).expanduser().resolve()),
        "overlap_audit": overlap,
        "candidate_k": candidate_ks,
        "rank": args.rank,
        "alpha": args.alpha,
        "pixel_weight": args.pixel_weight,
        "prior_weight": args.prior_weight,
        "seed": args.seed,
        "profile_flops": args.profile_flops,
        "note": (
            "End-to-end candidate timing is authoritative. Per-block GFLOPs estimate "
            "only the local LoRA down/up operations and is additive."
        ),
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(f"Wrote DiffIR LoRA cost profile to {output_dir}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--importance_csv", required=True)
    parser.add_argument(
        "--s2_checkpoint", default="experiments/pretrained/SISR-DiffIRS2.pth"
    )
    parser.add_argument(
        "--s1_checkpoint", default="experiments/pretrained/SISR-DiffIRS1.pth"
    )
    parser.add_argument(
        "--checkpoint_key",
        choices=("auto", "params_ema", "params", "state_dict", "model", "root"),
        default="auto",
    )
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--protected_eval_dir", action="append", default=[])
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--candidate_k", type=int, nargs="+", default=[13, 21, 26, 29, 44])
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--lq_size", type=int, default=64)
    parser.add_argument("--sr_scale", type=int, default=4)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=16.0)
    parser.add_argument("--pixel_weight", type=float, default=1.0)
    parser.add_argument("--prior_weight", type=float, default=1.0)
    parser.add_argument("--expected_blocks", type=int, default=44)
    parser.add_argument("--warmup_steps", type=int, default=5)
    parser.add_argument("--timed_steps", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_images", type=int, default=20)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--profile_flops", action="store_true")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="cuda")
    return parser


def main() -> None:
    profile(build_parser().parse_args())


if __name__ == "__main__":
    main()
