#!/usr/bin/env python3
"""Profile frozen-block backward-bypass costs for a sparse TSD-SR LoRA policy."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
import time
from pathlib import Path

import pyiqa
import torch
from diffusers import FlowMatchEulerDiscreteScheduler
from torch.profiler import ProfilerActivity, profile
from torch.utils.data import DataLoader

import adaptive_grad_blockskip as adaptive
import profile_tsdsr_grad as core


BLOCK_PATTERN = re.compile(r"transformer_blocks\.\d+")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pretrained_model", required=True)
    parser.add_argument("--official_lora_dir", required=True)
    parser.add_argument("--teacher_lora_dir", required=True)
    parser.add_argument("--default_embedding_dir", required=True)
    parser.add_argument("--null_embedding_dir", required=True)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--importance_csv", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--selected_k", type=int, required=True)
    parser.add_argument("--profile_noise_ratio", type=float, default=0.4)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--sr_scale", type=int, default=4)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=16.0)
    parser.add_argument("--reg_rank", type=int, default=16)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--lambda_tsd", type=float, default=0.7)
    parser.add_argument("--lpips_weight", type=float, default=1.0)
    parser.add_argument("--latent_mse_weight", type=float, default=1.0)
    parser.add_argument("--tsd_weight", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--dtype", choices=["fp16", "bf16", "fp32"], default="fp16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--skip_profiler",
        action="store_true",
        help="Skip the slower PyTorch FLOP/kernel profiler during timing smoke tests.",
    )
    return parser.parse_args()


def read_csv(path: Path) -> list[dict]:
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
    return int(block.rsplit(".", 1)[1])


def aggregate_importance(path: Path) -> tuple[list[dict], list[dict]]:
    rows = read_csv(path)
    if not rows:
        raise SystemExit(f"Importance CSV is empty: {path}")
    required = {"block"}
    missing = required - set(rows[0])
    if missing:
        raise SystemExit(f"Importance CSV is missing columns: {sorted(missing)}")

    if "mean_score_share" in rows[0]:
        aggregate = [
            {
                "block": str(row["block"]),
                "mean_score_share": float(row["mean_score_share"]),
            }
            for row in rows
        ]
        evolution = []
    else:
        score_key = "score_share" if "score_share" in rows[0] else "normalized_grad_score"
        if score_key not in rows[0] or "noise_ratio" not in rows[0]:
            raise SystemExit(
                "Importance CSV must contain mean_score_share, or noise_ratio plus "
                "score_share/normalized_grad_score."
            )
        evolution = []
        by_noise: dict[float, list[dict]] = {}
        for row in rows:
            by_noise.setdefault(float(row["noise_ratio"]), []).append(row)
        for ratio, group in by_noise.items():
            total = sum(float(row[score_key]) for row in group)
            for row in group:
                evolution.append(
                    {
                        "noise_ratio": ratio,
                        "block": str(row["block"]),
                        "score_share": float(row[score_key]) / max(total, 1e-30),
                    }
                )
        blocks = sorted({row["block"] for row in evolution}, key=block_index)
        aggregate = []
        for block in blocks:
            values = [row["score_share"] for row in evolution if row["block"] == block]
            aggregate.append(
                {"block": block, "mean_score_share": statistics.mean(values)}
            )

    aggregate.sort(key=lambda row: row["mean_score_share"], reverse=True)
    cumulative = 0.0
    for rank, row in enumerate(aggregate, 1):
        cumulative += row["mean_score_share"]
        row["aggregate_rank"] = rank
        row["cumulative_utility"] = cumulative
    return aggregate, evolution


def module_has_adapter(module: torch.nn.Module, adapter_name: str) -> bool:
    for attribute in (
        "lora_A",
        "lora_B",
        "lora_embedding_A",
        "lora_embedding_B",
        "lora_magnitude_vector",
    ):
        container = getattr(module, attribute, None)
        if container is not None and adapter_name in container:
            return True
    return False


def configure_sparse_domain_adapter(
    student: torch.nn.Module,
    selected_blocks: set[str],
) -> tuple[int, int]:
    """Keep the AID adapter only in selected blocks and shared boundary modules."""
    selected_modules = 0
    boundary_modules = 0
    for name, module in student.named_modules():
        if not module_has_adapter(module, "aid"):
            continue
        match = BLOCK_PATTERN.search(name)
        if match is None:
            active = True
            boundary_modules += 1
        else:
            active = match.group(0) in selected_blocks
            selected_modules += int(active)
        module.set_adapter(["official", "aid"] if active else "official")

    trainable = 0
    for name, parameter in student.named_parameters():
        match = BLOCK_PATTERN.search(name)
        active_block = match is None or match.group(0) in selected_blocks
        parameter.requires_grad_(".aid." in name and active_block)
        if parameter.requires_grad:
            trainable += parameter.numel()
    return trainable, boundary_modules


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def clear_grads(model: torch.nn.Module) -> None:
    for parameter in model.parameters():
        parameter.grad = None


def run_step(
    student,
    controller,
    loss_context: dict,
    batch,
    ratio: float,
    mode: str,
    rng_state,
) -> tuple[float, int, float]:
    clear_grads(student)
    controller.set_mode(mode)
    adaptive.restore_rng(rng_state)
    args = loss_context["args"]
    device = loss_context["device"]
    dtype = loss_context["dtype"]
    with torch.autocast(
        device_type=device.type,
        dtype=dtype,
        enabled=device.type == "cuda" and dtype != torch.float32,
    ):
        values = core.profile_loss(
            student,
            loss_context["teacher"],
            loss_context["vae"],
            loss_context["lpips"],
            loss_context["scheduler"],
            batch,
            ratio,
            args,
            device,
            dtype,
            loss_context["default_prompt"],
            loss_context["default_pooled"],
            loss_context["null_prompt"],
            loss_context["null_pooled"],
        )
    loss = values[0]
    if not torch.isfinite(loss):
        raise RuntimeError(f"Non-finite loss in mode={mode}")
    loss.backward()
    sync(device)
    stats = controller.stats(0.0)
    return float(loss.detach().cpu()), stats.fallback_blocks, stats.max_reconstruction_abs_diff


def paired_timing(
    student,
    controller,
    loss_context,
    batch,
    ratio,
    rng_state,
    repeats,
) -> tuple[list[float], list[float], list[float], list[float], int, float]:
    values = {"full": [], "single_skip": []}
    losses = {"full": [], "single_skip": []}
    fallback = 0
    max_forward_diff = 0.0
    device = loss_context["device"]
    for repeat in range(repeats):
        modes = ("full", "single_skip") if repeat % 2 == 0 else ("single_skip", "full")
        for mode in modes:
            sync(device)
            started = time.perf_counter()
            loss, current_fallback, forward_diff = run_step(
                student, controller, loss_context, batch, ratio, mode, rng_state
            )
            values[mode].append((time.perf_counter() - started) * 1000.0)
            losses[mode].append(loss)
            fallback += current_fallback
            max_forward_diff = max(max_forward_diff, forward_diff)
    return (
        values["full"],
        values["single_skip"],
        losses["full"],
        losses["single_skip"],
        fallback,
        max_forward_diff,
    )


def profiler_totals(profiler) -> tuple[float, float]:
    flops = sum(
        float(getattr(event, "flops", 0.0) or 0.0)
        for event in profiler.key_averages()
    )
    cuda_us = 0.0
    for event in profiler.events():
        if "cuda" not in str(getattr(event, "device_type", "")).lower():
            continue
        time_range = getattr(event, "time_range", None)
        if time_range is not None and hasattr(time_range, "elapsed_us"):
            cuda_us += float(time_range.elapsed_us())
        else:
            cuda_us += float(getattr(event, "device_time_total", 0.0) or 0.0)
    return flops / 1e9, cuda_us / 1000.0


def profiled_step(student, controller, loss_context, batch, ratio, mode, rng_state):
    device = loss_context["device"]
    activities = [ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(ProfilerActivity.CUDA)
        torch.cuda.reset_peak_memory_stats(device)
    with profile(activities=activities, with_flops=True, profile_memory=True) as prof:
        loss, fallback, forward_diff = run_step(
            student, controller, loss_context, batch, ratio, mode, rng_state
        )
    gflops, kernel_ms = profiler_totals(prof)
    peak_mb = (
        torch.cuda.max_memory_allocated(device) / (1024.0**2)
        if device.type == "cuda"
        else 0.0
    )
    return gflops, kernel_ms, peak_mb, loss, fallback, forward_diff


def reduction(full: float, bypass: float) -> float:
    return (full - bypass) / full * 100.0 if full > 0 else float("nan")


def stdev(values: list[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else 0.0


def main() -> None:
    args = parse_args()
    if args.selected_k <= 0 or args.warmup < 0 or args.repeats <= 0:
        raise SystemExit("Require selected_k > 0, warmup >= 0, and repeats > 0")
    if not 0.0 <= args.profile_noise_ratio <= 1.0:
        raise SystemExit("--profile_noise_ratio must be in [0, 1]")

    core.set_seed(args.seed)
    device = torch.device(args.device)
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[args.dtype]
    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise SystemExit(f"Output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    aggregate, evolution = aggregate_importance(Path(args.importance_csv))
    if args.selected_k >= len(aggregate):
        raise SystemExit(
            f"selected_k={args.selected_k} leaves no frozen bypass candidates "
            f"among {len(aggregate)} blocks"
        )
    selected = {row["block"] for row in aggregate[: args.selected_k]}
    selected_utility = aggregate[args.selected_k - 1]["cumulative_utility"]
    all_blocks = sorted((row["block"] for row in aggregate), key=block_index)
    frozen = [block for block in all_blocks if block not in selected]

    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        args.pretrained_model,
        subfolder="scheduler",
        local_files_only=True,
    )
    student, teacher, vae = core.load_models(args, dtype, device)
    trainable_params, boundary_modules = configure_sparse_domain_adapter(student, selected)
    student.train()

    lpips = pyiqa.create_metric("lpips", as_loss=True, device=device)
    lpips.requires_grad_(False)
    dataset = core.PairedFolderDataset(
        args.data_dir, args.image_size, args.sr_scale, args.seed, max_images=1
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    try:
        batch = next(iter(loader))
    except StopIteration as exc:
        raise SystemExit(f"No images found in {args.data_dir}") from exc

    default_prompt, default_pooled = core.load_embeddings(
        args.default_embedding_dir, args.batch_size, device, dtype
    )
    null_prompt, null_pooled = core.load_embeddings(
        args.null_embedding_dir, args.batch_size, device, dtype
    )
    context = {
        "args": args,
        "device": device,
        "dtype": dtype,
        "teacher": teacher,
        "vae": vae,
        "lpips": lpips,
        "scheduler": scheduler,
        "default_prompt": default_prompt,
        "default_pooled": default_pooled,
        "null_prompt": null_prompt,
        "null_pooled": null_pooled,
    }

    controller = adaptive.ResidualBlockController(
        student,
        {block: block for block in all_blocks},
        cache_device="cpu",
        cache_dtype=torch.float16,
    )
    rows = []
    for block in frozen:
        controller.configure([block])
        rng_state = adaptive.snapshot_rng(device)
        for warmup_index in range(args.warmup):
            modes = ("full", "single_skip")
            if warmup_index % 2:
                modes = tuple(reversed(modes))
            for mode in modes:
                run_step(
                    student,
                    controller,
                    context,
                    batch,
                    args.profile_noise_ratio,
                    mode,
                    rng_state,
                )

        full_ms, bypass_ms, full_losses, bypass_losses, fallback, forward_diff = paired_timing(
            student,
            controller,
            context,
            batch,
            args.profile_noise_ratio,
            rng_state,
            args.repeats,
        )
        if args.skip_profiler:
            full_gflops = bypass_gflops = float("nan")
            full_kernel = bypass_kernel = float("nan")
            full_peak = bypass_peak = float("nan")
        else:
            full_gflops, full_kernel, full_peak, _, f1, d1 = profiled_step(
                student,
                controller,
                context,
                batch,
                args.profile_noise_ratio,
                "full",
                rng_state,
            )
            bypass_gflops, bypass_kernel, bypass_peak, _, f2, d2 = profiled_step(
                student,
                controller,
                context,
                batch,
                args.profile_noise_ratio,
                "single_skip",
                rng_state,
            )
            fallback += f1 + f2
            forward_diff = max(forward_diff, d1, d2)

        full_mean = statistics.mean(full_ms)
        bypass_mean = statistics.mean(bypass_ms)
        row = {
            "block": block,
            "block_index": block_index(block),
            "selected_k": args.selected_k,
            "selected_utility": selected_utility,
            "profile_noise_ratio": args.profile_noise_ratio,
            "full_step_time_ms": full_mean,
            "bypass_step_time_ms": bypass_mean,
            "step_time_saved_ms": full_mean - bypass_mean,
            "step_time_reduction_pct": reduction(full_mean, bypass_mean),
            "full_step_time_std_ms": stdev(full_ms),
            "bypass_step_time_std_ms": stdev(bypass_ms),
            "full_reported_gflops": full_gflops,
            "bypass_reported_gflops": bypass_gflops,
            "reported_gflops_saved": full_gflops - bypass_gflops,
            "full_cuda_kernel_time_ms": full_kernel,
            "bypass_cuda_kernel_time_ms": bypass_kernel,
            "cuda_kernel_time_saved_ms": full_kernel - bypass_kernel,
            "full_peak_cuda_mem_mb": full_peak,
            "bypass_peak_cuda_mem_mb": bypass_peak,
            "peak_cuda_mem_saved_mb": full_peak - bypass_peak,
            "max_loss_abs_diff": max(
                abs(full - bypass) for full, bypass in zip(full_losses, bypass_losses)
            ),
            "max_forward_abs_diff": forward_diff,
            "fallback_events": fallback,
        }
        rows.append(row)
        print(
            f"{block}: saved={row['step_time_saved_ms']:.3f} ms "
            f"GFLOPs={row['reported_gflops_saved']:.3f} "
            f"fallback={fallback}"
        )
        torch.cuda.empty_cache()

    rows.sort(key=lambda row: row["block_index"])
    write_csv(output_dir / "block_backward_costs.csv", rows)

    if evolution:
        cost_by_block = {row["block"]: row for row in rows}
        tradeoff = []
        for importance in evolution:
            cost = cost_by_block.get(importance["block"])
            if cost is None:
                continue
            saved = float(cost["reported_gflops_saved"])
            tradeoff.append(
                {
                    "noise_ratio": importance["noise_ratio"],
                    "block": importance["block"],
                    "block_index": cost["block_index"],
                    "score_share": importance["score_share"],
                    "reported_gflops_saved": saved,
                    "step_time_saved_ms": cost["step_time_saved_ms"],
                    "importance_per_saved_gflop": (
                        importance["score_share"] / saved
                        if math.isfinite(saved) and saved > 0
                        else float("inf")
                    ),
                }
            )
        tradeoff.sort(key=lambda row: (row["noise_ratio"], row["block_index"]))
        write_csv(output_dir / "importance_compute_tradeoff.csv", tradeoff)

    summary = {
        "model": "TSD-SR-MSE",
        "selected_k": args.selected_k,
        "selected_utility": selected_utility,
        "selected_lora_blocks": sorted(selected, key=block_index),
        "frozen_bypass_candidates": frozen,
        "trainable_domain_lora_params": trainable_params,
        "always_trained_boundary_lora_modules": boundary_modules,
        "profile_noise_ratio": args.profile_noise_ratio,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "skip_profiler": args.skip_profiler,
        "mean_full_step_time_ms": statistics.mean(
            row["full_step_time_ms"] for row in rows
        ),
        "max_loss_abs_diff": max(row["max_loss_abs_diff"] for row in rows),
        "max_forward_abs_diff": max(row["max_forward_abs_diff"] for row in rows),
        "total_fallback_events": sum(row["fallback_events"] for row in rows),
        "profiler_note": (
            "The official TSD-SR and VAE adapters remain frozen. The fresh domain "
            "adapter is active only in the selected K blocks and shared boundary "
            "modules. Single-pass bypass executes the exact forward under no_grad "
            "and removes only the selected frozen block's backward graph."
        ),
    }
    (output_dir / "block_backward_costs_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    print(f"Wrote block costs to {output_dir}")


if __name__ == "__main__":
    main()
