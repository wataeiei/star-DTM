#!/usr/bin/env python3
"""Audit FLOPs, latency, and memory of existing DiT GradSkip-LoRA runs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import time
from collections import Counter
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile
from torch.utils.data import DataLoader

import adaptive_grad_blockskip as adaptive


def read_csv(path: Path) -> list[dict]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def kernel_time_ms(profiler) -> float:
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


def profiler_flops(profiler) -> float:
    return float(
        sum(float(getattr(event, "flops", 0.0) or 0.0) for event in profiler.key_averages())
    )


def reduction(full: float, bypass: float) -> float:
    return (full - bypass) / full * 100.0 if full > 0 else float("nan")


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def clear_grads(model: torch.nn.Module) -> None:
    for parameter in model.parameters():
        parameter.grad = None


def modal_skip_sets(log_rows: list[dict]) -> list[dict]:
    grouped: dict[float, list[str]] = {}
    for row in log_rows:
        ratio = float(row["noise_ratio"])
        grouped.setdefault(ratio, []).append(row.get("skipped_blocks", "") or "")
    result = []
    for ratio in sorted(grouped):
        values = grouped[ratio]
        skip_text, occurrences = Counter(values).most_common(1)[0]
        result.append(
            {
                "noise_ratio": ratio,
                "weight": len(values),
                "modal_occurrences": occurrences,
                "skip_blocks": [value for value in skip_text.split(";") if value],
            }
        )
    return result


def load_experiment(args: argparse.Namespace, metadata: dict, device: torch.device):
    run_args = argparse.Namespace(**metadata)
    run_args.data_dir = args.data_dir
    run_args.max_images = 0
    run_args.batch_size = args.batch_size
    run_args.num_workers = args.num_workers

    if args.model == "dit4sr":
        import profile_hf_dit4sr_grad as core
        import train_hf_dit4sr_all_lora_importance as train_core

        if args.local_files_only:
            run_args.local_files_only = True
        pipe = core.load_pipe(run_args, device)
        model = core.get_transformer(pipe, run_args.component_name)
        model.requires_grad_(False)
        target_names = [
            name
            for name, module in model.named_modules()
            if isinstance(module, torch.nn.Linear)
            and core.block_key(name, run_args.block_regex)
            and core.target_match(name, run_args.target)
        ]
        candidates = train_core.candidate_lora_blocks(
            model, run_args.target, run_args.block_regex
        )
        selected = metadata["selected_lora_blocks"]
        core.inject_lora(
            model,
            run_args.target,
            run_args.rank,
            run_args.alpha,
            run_args.block_regex,
            selected_blocks=set(selected),
        )
        adapter_status = adaptive.load_lora_adapter(
            model, Path(args.run_dir) / "lora_adapter.pt"
        )
        dataset = core.ImageFolderDataset(args.data_dir, run_args.image_size, 0)

        def loss_fn(batch, ratio):
            run_args._profile_noise_ratio = ratio
            return train_core.batch_loss(pipe, model, batch, run_args, device)

        block_key = core.block_key
        block_regex = run_args.block_regex
        context = (pipe,)
    else:
        import profile_dit_sr_grad as core
        import train_dit_sr_all_lora_importance as train_core

        if args.config_path:
            run_args.config_path = args.config_path
        if args.ckpt_path:
            run_args.ckpt_path = args.ckpt_path
        if args.autoencoder_ckpt:
            run_args.autoencoder_ckpt = args.autoencoder_ckpt
        model = core.load_model(run_args, device)
        diffusion, autoencoder = core.load_official_objective(run_args, device)
        model.requires_grad_(False)
        target_names = [
            name
            for name, module in model.named_modules()
            if isinstance(module, torch.nn.Linear)
            and core.block_key(name, run_args.block_regex)
            and core.target_match(name, run_args.target)
        ]
        candidates = train_core.candidate_lora_blocks(
            model, run_args.target, run_args.block_regex
        )
        selected = metadata["selected_lora_blocks"]
        core.inject_lora(
            model,
            run_args.target,
            run_args.rank,
            run_args.alpha,
            run_args.block_regex,
            selected_blocks=set(selected),
        )
        adapter_status = adaptive.load_lora_adapter(
            model, Path(args.run_dir) / "lora_adapter.pt"
        )
        dataset = core.ImageFolderDataset(args.data_dir, run_args.image_size, 0)

        def loss_fn(batch, ratio):
            run_args._profile_noise_ratio = ratio
            return train_core.batch_loss(
                model, diffusion, autoencoder, batch, run_args, device
            )

        block_key = core.block_key
        block_regex = run_args.block_regex
        context = (diffusion, autoencoder)

    if adapter_status["missing"]:
        raise SystemExit(f"Adapter modules missing from model: {adapter_status['missing'][:5]}")
    model.train()
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    block_paths = adaptive.infer_block_module_paths(
        target_names, candidates, block_key, block_regex
    )
    controller = adaptive.ResidualBlockController(
        model, block_paths, cache_device="cpu", cache_dtype=torch.float16
    )
    return model, controller, iter(loader), loss_fn, candidates, selected, context


def run_step(model, controller, loss_fn, batch, ratio, mode, rng_state, device):
    clear_grads(model)
    controller.set_mode(mode)
    adaptive.restore_rng(rng_state)
    loss = loss_fn(batch, ratio)
    loss.backward()
    sync(device)
    return float(loss.detach().cpu())


def timed(model, controller, loss_fn, batch, ratio, mode, rng_state, device, repeats):
    values = []
    losses = []
    for _ in range(repeats):
        sync(device)
        started = time.perf_counter()
        losses.append(
            run_step(model, controller, loss_fn, batch, ratio, mode, rng_state, device)
        )
        values.append((time.perf_counter() - started) * 1000.0)
    return values, losses


def profiled(model, controller, loss_fn, batch, ratio, mode, rng_state, device):
    activities = [ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(ProfilerActivity.CUDA)
        torch.cuda.reset_peak_memory_stats(device)
    with profile(activities=activities, with_flops=True, profile_memory=True) as prof:
        loss = run_step(
            model, controller, loss_fn, batch, ratio, mode, rng_state, device
        )
    peak_mb = (
        torch.cuda.max_memory_allocated(device) / (1024.0**2)
        if device.type == "cuda"
        else 0.0
    )
    return profiler_flops(prof), kernel_time_ms(prof), peak_mb, loss


def weighted_mean(rows: list[dict], key: str) -> float:
    total = sum(int(row["training_steps_at_ratio"]) for row in rows)
    return sum(float(row[key]) * int(row["training_steps_at_ratio"]) for row in rows) / total


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=["dit4sr", "dit-sr"])
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=4242)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--config_path", default="")
    parser.add_argument("--ckpt_path", default="")
    parser.add_argument("--autoencoder_ckpt", default="")
    args = parser.parse_args()

    if args.warmup < 0 or args.repeats <= 0:
        raise SystemExit("Require --warmup >= 0 and --repeats > 0")
    run_dir = Path(args.run_dir)
    metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
    log_rows = read_csv(run_dir / "train_log.csv")
    policies = modal_skip_sets(log_rows)
    if not policies or all(not row["skip_blocks"] for row in policies):
        raise SystemExit("The run log does not contain any skipped blocks.")

    core_seed = metadata.get("seed", args.seed)
    torch.manual_seed(int(core_seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(core_seed))
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model, controller, loader, loss_fn, candidates, selected, keepalive = load_experiment(
        args, metadata, device
    )
    try:
        batch = next(loader)
    except StopIteration as exc:
        raise SystemExit(f"No images found in {args.data_dir}") from exc

    rows = []
    for policy in policies:
        ratio = float(policy["noise_ratio"])
        skipped = policy["skip_blocks"]
        unknown = sorted(set(skipped) - set(candidates))
        collisions = sorted(set(skipped) & set(selected))
        if unknown:
            raise SystemExit(f"Unknown logged blocks at noise ratio {ratio}: {unknown}")
        if collisions:
            raise SystemExit(f"Protected LoRA blocks were skipped at noise ratio {ratio}: {collisions}")
        controller.configure(skipped)
        rng_state = adaptive.snapshot_rng(device)
        for _ in range(args.warmup):
            run_step(model, controller, loss_fn, batch, ratio, "full", rng_state, device)
            run_step(model, controller, loss_fn, batch, ratio, "single_skip", rng_state, device)
        full_ms, full_losses = timed(
            model, controller, loss_fn, batch, ratio, "full", rng_state, device, args.repeats
        )
        bypass_ms, bypass_losses = timed(
            model, controller, loss_fn, batch, ratio, "single_skip", rng_state, device, args.repeats
        )
        full_flops, full_kernel, full_peak, _ = profiled(
            model, controller, loss_fn, batch, ratio, "full", rng_state, device
        )
        bypass_flops, bypass_kernel, bypass_peak, _ = profiled(
            model, controller, loss_fn, batch, ratio, "single_skip", rng_state, device
        )
        row = {
            "noise_ratio": ratio,
            "training_steps_at_ratio": policy["weight"],
            "modal_policy_occurrences": policy["modal_occurrences"],
            "total_blocks": len(candidates),
            "forward_executed_blocks": len(candidates),
            "backward_bypassed_blocks": len(skipped),
            "backward_bypass_ratio_pct": len(skipped) / len(candidates) * 100.0,
            "full_step_time_ms": statistics.mean(full_ms),
            "bypass_step_time_ms": statistics.mean(bypass_ms),
            "step_time_reduction_pct": reduction(statistics.mean(full_ms), statistics.mean(bypass_ms)),
            "full_reported_gflops": full_flops / 1e9,
            "bypass_reported_gflops": bypass_flops / 1e9,
            "reported_flops_reduction_pct": reduction(full_flops, bypass_flops),
            "full_cuda_kernel_time_ms": full_kernel,
            "bypass_cuda_kernel_time_ms": bypass_kernel,
            "cuda_kernel_time_reduction_pct": reduction(full_kernel, bypass_kernel),
            "full_peak_cuda_mem_mb": full_peak,
            "bypass_peak_cuda_mem_mb": bypass_peak,
            "peak_cuda_memory_reduction_pct": reduction(full_peak, bypass_peak),
            "max_loss_abs_diff": max(abs(a - b) for a, b in zip(full_losses, bypass_losses)),
            "bypassed_block_names": ";".join(skipped),
        }
        rows.append(row)
        print(
            f"sigma={ratio:.2f} bypass={len(skipped)}/{len(candidates)} "
            f"FLOPs={row['reported_flops_reduction_pct']:.2f}% "
            f"time={row['step_time_reduction_pct']:.2f}%"
        )

    summary = {
        "model": metadata.get("model", args.model),
        "run_dir": str(run_dir),
        "training_steps": sum(int(row["training_steps_at_ratio"]) for row in rows),
        "total_blocks": len(candidates),
        "weighted_backward_bypassed_blocks": weighted_mean(rows, "backward_bypassed_blocks"),
        "weighted_backward_bypass_ratio_pct": weighted_mean(rows, "backward_bypass_ratio_pct"),
        "weighted_full_step_time_ms": weighted_mean(rows, "full_step_time_ms"),
        "weighted_bypass_step_time_ms": weighted_mean(rows, "bypass_step_time_ms"),
        "weighted_step_time_reduction_pct": reduction(
            weighted_mean(rows, "full_step_time_ms"), weighted_mean(rows, "bypass_step_time_ms")
        ),
        "weighted_full_reported_gflops": weighted_mean(rows, "full_reported_gflops"),
        "weighted_bypass_reported_gflops": weighted_mean(rows, "bypass_reported_gflops"),
        "weighted_reported_flops_reduction_pct": reduction(
            weighted_mean(rows, "full_reported_gflops"),
            weighted_mean(rows, "bypass_reported_gflops"),
        ),
        "weighted_full_cuda_kernel_time_ms": weighted_mean(rows, "full_cuda_kernel_time_ms"),
        "weighted_bypass_cuda_kernel_time_ms": weighted_mean(rows, "bypass_cuda_kernel_time_ms"),
        "weighted_cuda_kernel_time_reduction_pct": reduction(
            weighted_mean(rows, "full_cuda_kernel_time_ms"),
            weighted_mean(rows, "bypass_cuda_kernel_time_ms"),
        ),
        "weighted_full_peak_cuda_mem_mb": weighted_mean(rows, "full_peak_cuda_mem_mb"),
        "weighted_bypass_peak_cuda_mem_mb": weighted_mean(rows, "bypass_peak_cuda_mem_mb"),
        "weighted_peak_cuda_memory_reduction_pct": reduction(
            weighted_mean(rows, "full_peak_cuda_mem_mb"),
            weighted_mean(rows, "bypass_peak_cuda_mem_mb"),
        ),
        "max_loss_abs_diff": max(float(row["max_loss_abs_diff"]) for row in rows),
        "profiler_note": "FLOPs include only operators supported by PyTorch profiler; timing and memory are measured separately.",
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "compute_audit_by_noise.csv", rows)
    (output_dir / "compute_audit_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    print(f"Wrote compute audit to {output_dir}")


if __name__ == "__main__":
    main()
