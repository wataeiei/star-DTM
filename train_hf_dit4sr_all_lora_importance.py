#!/usr/bin/env python3
"""Train All-LoRA on the official DiT4SR flow objective and track importance."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import profile_hf_dit4sr_grad as core
import adaptive_grad_blockskip as adaptive


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def batch_loss(pipe, transformer, batch, args, device):
    return core.flow_matching_batch_loss(pipe, transformer, batch, args, device)


def profile_importance(pipe, transformer, loader, args, device, train_step, noise_ratio):
    cpu_state = torch.random.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if device.type == "cuda" else None
    python_state = random.getstate()
    core.set_seed(args.profile_seed)
    args._profile_noise_ratio = noise_ratio
    transformer.zero_grad(set_to_none=True)
    iterator = iter(loader)
    valid = 0
    loss_sum = 0.0
    try:
        for _ in range(args.profile_batches):
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                batch = next(iterator)
            batch, _patch_size = adaptive.dynamic_patch_batch(
                batch,
                noise_ratio,
                args.patch_min_fraction,
                args.patch_max_fraction,
            )
            loss = batch_loss(pipe, transformer, batch, args, device)
            if not torch.isfinite(loss):
                transformer.zero_grad(set_to_none=True)
                continue
            loss.backward()
            valid += 1
            loss_sum += float(loss.detach().cpu())
        if valid == 0:
            raise SystemExit(f"No valid profile batches at step {train_step}.")

        scheduler = pipe.scheduler
        count = len(scheduler.timesteps)
        sigma_index = int(
            (scheduler.sigmas[:count].float() - noise_ratio).abs().argmin().item()
        )
        actual_sigma = float(scheduler.sigmas[sigma_index])
        actual_timestep = float(scheduler.timesteps[sigma_index])
        grouped = {}
        for name, module in core.iter_lora_modules(transformer):
            block = core.block_key(name, args.block_regex)
            if not block:
                continue
            item = grouped.setdefault(
                block, {"grad_sq": 0.0, "update_sq": 0.0, "params": 0, "modules": 0}
            )
            item["grad_sq"] += core.lora_grad_norm(module) ** 2
            delta = module.lora_up.weight.detach().float() @ module.lora_down.weight.detach().float()
            item["update_sq"] += float(delta.pow(2).sum().cpu())
            item["params"] += module.lora_down.weight.numel() + module.lora_up.weight.numel()
            item["modules"] += 1
        blocks = sorted(grouped, key=core.natural_key)
        rows = []
        for index, block in enumerate(blocks):
            item = grouped[block]
            params = int(item["params"])
            grad_norm = math.sqrt(item["grad_sq"])
            update_norm = math.sqrt(item["update_sq"])
            rows.append(
                {
                    "train_step": train_step,
                    "noise_ratio": noise_ratio,
                    "scheduler_index": sigma_index,
                    "timestep": actual_timestep,
                    "sigma": actual_sigma,
                    "block": block,
                    "block_index": index,
                    "grad_norm": grad_norm,
                    "lora_param_count": params,
                    "module_count": int(item["modules"]),
                    "normalized_grad_score": grad_norm / math.sqrt(max(params, 1)),
                    "update_norm": update_norm,
                    "normalized_update_score": update_norm / math.sqrt(max(params, 1)),
                    "probe_batches": valid,
                    "mean_probe_loss": loss_sum / valid,
                    "loss_mode": args.loss_mode,
                }
            )
        ranked = sorted(rows, key=lambda row: row["normalized_grad_score"], reverse=True)
        ranks = {row["block"]: rank for rank, row in enumerate(ranked, start=1)}
        for row in rows:
            row["importance_rank"] = ranks[row["block"]]
            row["selected_topk"] = ranks[row["block"]] <= args.topk_blocks
        return rows
    finally:
        del args._profile_noise_ratio
        transformer.zero_grad(set_to_none=True)
        random.setstate(python_state)
        torch.random.set_rng_state(cpu_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)


def topk_summary(rows):
    grouped = {}
    for row in rows:
        key = (int(row["train_step"]), float(row["noise_ratio"]))
        grouped.setdefault(key, []).append(row)
    output = []
    for step, ratio in sorted(grouped):
        baseline_key = (step, min(r for s, r in grouped if s == step))
        baseline = {
            row["block"] for row in grouped[baseline_key] if bool(row["selected_topk"])
        }
        selected = {
            row["block"] for row in grouped[(step, ratio)] if bool(row["selected_topk"])
        }
        overlap = len(baseline & selected)
        output.append(
            {
                "train_step": step,
                "noise_ratio": ratio,
                "topk_blocks": ";".join(
                    row["block"]
                    for row in sorted(grouped[(step, ratio)], key=lambda row: row["importance_rank"])
                    if row["selected_topk"]
                ),
                "topk_overlap_count_vs_lowest_t": overlap,
                "topk_overlap_ratio_vs_lowest_t": overlap / max(len(baseline), 1),
                "topk_jaccard_vs_lowest_t": overlap / max(len(baseline | selected), 1),
            }
        )
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_id", default="acceptee/DiT4SR")
    parser.add_argument("--base_model_id", default="stabilityai/stable-diffusion-3.5-medium")
    parser.add_argument("--loss_mode", default="official_flow", choices=["official_flow", "proxy"])
    parser.add_argument("--load_mode", default="transformer", choices=["transformer", "pipeline"])
    parser.add_argument("--model_impl", default="official", choices=["official", "standard"])
    parser.add_argument("--dit4sr_code_repo", default="acceptee/DiT4SR")
    parser.add_argument("--dit4sr_code_dir", default="")
    parser.add_argument("--variant", default="dit4sr_q", choices=["dit4sr_q", "dit4sr_f", "dit4sr_r1"])
    parser.add_argument("--transformer_subfolder", default="")
    parser.add_argument("--pipeline_subfolder", default="")
    parser.add_argument("--component_name", default="")
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--max_images", type=int, default=0)
    parser.add_argument("--target", default="qv", choices=["q", "v", "qv", "qkv", "qkvo"])
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--alpha", type=int, default=16)
    parser.add_argument("--train_steps", type=int, default=1000)
    parser.add_argument("--profile_steps", type=int, nargs="+", default=[0, 100, 250, 500, 750, 1000])
    parser.add_argument("--profile_batches", type=int, default=5)
    parser.add_argument(
        "--profile_noise_ratios", type=float, nargs="+",
        default=[0.05, 0.2, 0.4, 0.6, 0.8, 0.95],
        help="Target flow sigmas to scan at every profile step.",
    )
    parser.add_argument("--topk_blocks", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--profile_seed", type=int, default=2026)
    parser.add_argument("--dtype", default="bf16", choices=["fp32", "bf16", "fp16"])
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--block_regex", default="")
    parser.add_argument("--latent_channels", type=int, default=16)
    parser.add_argument("--latent_size", type=int, default=64)
    parser.add_argument("--caption_dim", type=int, default=4096)
    parser.add_argument("--prompt_seq_len", type=int, default=77)
    parser.add_argument("--timestep", type=int, default=500)
    parser.add_argument("--input_noise_std", type=float, default=0.0)
    parser.add_argument("--sr_scale", type=float, default=4.0)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--blockskip_count", type=int, default=0)
    parser.add_argument("--blockskip_min_run", type=int, default=2)
    parser.add_argument("--blockskip_max_run", type=int, default=4)
    parser.add_argument("--blockskip_max_runs", type=int, default=2)
    parser.add_argument("--residual_cache_device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--residual_cache_dtype", choices=["fp16", "bf16", "fp32"], default="fp16")
    parser.add_argument("--patch_min_fraction", type=float, default=1.0)
    parser.add_argument("--patch_max_fraction", type=float, default=1.0)
    parser.add_argument(
        "--train_noise_ratios", type=float, nargs="+", default=[],
        help="Discrete official-flow sigmas sampled during training. Defaults to profile ratios.",
    )
    args = parser.parse_args()

    if args.loss_mode != "official_flow":
        raise SystemExit("Noise-ratio profiling requires --loss_mode official_flow.")
    if not 0.0 < args.patch_min_fraction <= args.patch_max_fraction <= 1.0:
        raise SystemExit("Require 0 < --patch_min_fraction <= --patch_max_fraction <= 1.")
    core.set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    pipe = core.load_pipe(args, device)
    transformer = core.get_transformer(pipe, args.component_name)
    transformer.requires_grad_(False)
    injected = core.inject_lora(transformer, args.target, args.rank, args.alpha, args.block_regex)
    transformer.train()
    dataset = core.ImageFolderDataset(args.data_dir, args.image_size, args.max_images)
    train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    profile_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    params = [param for param in transformer.parameters() if param.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    profile_steps = {step for step in args.profile_steps if 0 <= step <= args.train_steps}
    profile_steps.update({0, args.train_steps})
    if any(not 0.0 <= ratio <= 1.0 for ratio in args.profile_noise_ratios):
        raise SystemExit("--profile_noise_ratios values must be in [0, 1].")
    importance_rows = []
    for ratio in args.profile_noise_ratios:
        importance_rows.extend(
            profile_importance(pipe, transformer, profile_loader, args, device, 0, ratio)
        )
    block_names = [
        row["block"]
        for row in sorted(importance_rows, key=lambda row: int(row["block_index"]))
        if float(row["noise_ratio"]) == float(args.profile_noise_ratios[0])
    ]
    cache_dtype = {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
    }[args.residual_cache_dtype]
    controller = None
    if args.blockskip_count > 0:
        block_paths = adaptive.infer_block_module_paths(
            (name for name, _module in core.iter_lora_modules(transformer)),
            block_names,
            core.block_key,
            args.block_regex,
        )
        controller = adaptive.ResidualBlockController(
            transformer,
            block_paths,
            cache_device=args.residual_cache_device,
            cache_dtype=cache_dtype,
        )
    train_noise_ratios = args.train_noise_ratios or args.profile_noise_ratios
    if any(not 0.0 <= ratio <= 1.0 for ratio in train_noise_ratios):
        raise SystemExit("--train_noise_ratios values must be in [0, 1].")
    train_rows = []
    iterator = iter(train_loader)
    write_csv(output_dir / "lora_importance_evolution.csv", importance_rows)
    write_csv(output_dir / "lora_importance_topk.csv", topk_summary(importance_rows))

    for step in range(1, args.train_steps + 1):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            batch = next(iterator)
        noise_ratio = float(random.choice(train_noise_ratios))
        args._profile_noise_ratio = noise_ratio
        batch, patch_size = adaptive.dynamic_patch_batch(
            batch,
            noise_ratio,
            args.patch_min_fraction,
            args.patch_max_fraction,
        )
        skip_blocks = []
        cache_stats = adaptive.CacheStats(0.0, 0.0, 0, 0)
        if controller is not None:
            skip_blocks = adaptive.select_low_score_runs(
                importance_rows,
                step,
                noise_ratio,
                args.blockskip_count,
                args.blockskip_min_run,
                args.blockskip_max_run,
                args.blockskip_max_runs,
            )
            controller.configure(skip_blocks)
            cache_stats = adaptive.populate_online_cache(
                controller,
                lambda: batch_loss(pipe, transformer, batch, args, device),
                device,
            )
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        train_start = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        loss = batch_loss(pipe, transformer, batch, args, device)
        if not torch.isfinite(loss):
            raise SystemExit(f"Non-finite training loss at step {step}.")
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(params, args.grad_clip)
        optimizer.step()
        if device.type == "cuda":
            torch.cuda.synchronize()
        train_step_time_s = time.perf_counter() - train_start
        train_peak_cuda_mem_mb = (
            torch.cuda.max_memory_allocated() / (1024.0 ** 2)
            if device.type == "cuda"
            else 0.0
        )
        if controller is not None:
            controller.set_mode("full")
        del args._profile_noise_ratio
        train_rows.append({
            "step": step,
            "loss": float(loss.detach().cpu()),
            "grad_norm": float(grad_norm),
            "noise_ratio": noise_ratio,
            "patch_size": patch_size,
            "skipped_blocks": ";".join(skip_blocks),
            "skipped_block_count": len(skip_blocks),
            "residual_cache_time_s": cache_stats.elapsed_s,
            "residual_cache_mb": cache_stats.cache_mb,
            "cache_teacher_loss": cache_stats.teacher_loss,
            "residual_loss_abs_diff": abs(float(loss.detach().cpu()) - cache_stats.teacher_loss)
            if controller is not None else 0.0,
            "replayable_blocks": cache_stats.replayable_blocks,
            "fallback_blocks": cache_stats.fallback_blocks,
            "fallback_block_names": cache_stats.fallback_names,
            "cache_peak_cuda_mem_mb": cache_stats.peak_cuda_mem_mb,
            "train_step_time_s": train_step_time_s,
            "train_peak_cuda_mem_mb": train_peak_cuda_mem_mb,
        })
        if step % args.log_every == 0 or step == 1:
            print(
                f"step {step:05d}/{args.train_steps} "
                f"loss={float(loss.detach().cpu()):.6f}"
            )
        if step in profile_steps:
            if controller is not None:
                controller.set_mode("full")
            current = []
            for ratio in args.profile_noise_ratios:
                current.extend(
                    profile_importance(pipe, transformer, profile_loader, args, device, step, ratio)
                )
            importance_rows.extend(current)
            write_csv(output_dir / "lora_importance_evolution.csv", importance_rows)
            write_csv(output_dir / "lora_importance_topk.csv", topk_summary(importance_rows))
            print(
                f"profile step {step}: "
                + ", ".join(
                    row["block"]
                    for row in sorted(current, key=lambda row: row["importance_rank"])
                    if row["selected_topk"]
                )
            )

    write_csv(output_dir / "train_log.csv", train_rows)
    metadata = vars(args) | {
        "profile_steps": sorted(profile_steps),
        "injected_module_count": len(injected),
        "model": "DiT4SR-HF",
        "objective": args.loss_mode,
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Wrote results to {output_dir}")


if __name__ == "__main__":
    main()
