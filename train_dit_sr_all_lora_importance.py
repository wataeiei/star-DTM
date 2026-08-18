#!/usr/bin/env python3
"""Train All-LoRA on DiT-SR and track layer importance over time."""

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

import profile_dit_sr_grad as core
import adaptive_grad_blockskip as adaptive


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def batch_loss(model, diffusion, autoencoder, batch, args, device):
    return core.diffusion_batch_loss(
        model, diffusion, autoencoder, batch, args, device
    )


def profile_importance(model, diffusion, autoencoder, loader, args, device, train_step, noise_ratio):
    cpu_state = torch.random.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if device.type == "cuda" else None
    python_state = random.getstate()
    core.set_seed(args.profile_seed)
    args._profile_noise_ratio = noise_ratio
    model.zero_grad(set_to_none=True)
    iterator = iter(loader)
    valid = 0
    total_loss = 0.0
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
            loss = batch_loss(model, diffusion, autoencoder, batch, args, device)
            if not torch.isfinite(loss):
                model.zero_grad(set_to_none=True)
                continue
            loss.backward()
            total_loss += float(loss.detach().cpu())
            valid += 1
        if valid == 0:
            raise SystemExit(f"No valid profile batches at step {train_step}.")

        by_block = {}
        for name, module in core.iter_lora_modules(model):
            block = core.block_key(name, args.block_regex)
            if not block:
                continue
            row = by_block.setdefault(
                block, {"grad_sq": 0.0, "update_sq": 0.0, "params": 0, "modules": 0}
            )
            row["grad_sq"] += core.lora_grad_norm(module) ** 2
            delta = module.lora_up.weight.detach().float() @ module.lora_down.weight.detach().float()
            row["update_sq"] += float(delta.pow(2).sum().cpu())
            row["params"] += module.lora_down.weight.numel() + module.lora_up.weight.numel()
            row["modules"] += 1

        blocks = sorted(by_block, key=core.natural_key)
        rows = []
        for index, block in enumerate(blocks):
            item = by_block[block]
            params = int(item["params"])
            grad_norm = math.sqrt(item["grad_sq"])
            update_norm = math.sqrt(item["update_sq"])
            rows.append(
                {
                    "train_step": train_step,
                    "noise_ratio": noise_ratio,
                    "timestep": round(noise_ratio * (diffusion.num_timesteps - 1)),
                    "block": block,
                    "block_index": index,
                    "grad_norm": grad_norm,
                    "lora_param_count": params,
                    "module_count": int(item["modules"]),
                    "normalized_grad_score": grad_norm / math.sqrt(max(params, 1)),
                    "update_norm": update_norm,
                    "normalized_update_score": update_norm / math.sqrt(max(params, 1)),
                    "probe_batches": valid,
                    "mean_probe_loss": total_loss / valid,
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
        model.zero_grad(set_to_none=True)
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
                    for row in sorted(
                        grouped[(step, ratio)], key=lambda row: int(row["importance_rank"])
                    )
                    if bool(row["selected_topk"])
                ),
                "topk_overlap_count_vs_lowest_t": overlap,
                "topk_overlap_ratio_vs_lowest_t": overlap / max(len(baseline), 1),
                "topk_jaccard_vs_lowest_t": overlap / max(len(baseline | selected), 1),
            }
        )
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config_path", default="configs/realsr_DiT.yaml")
    parser.add_argument("--ckpt_path", default="weights/realsr.pth")
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--lq_size", type=int, default=64)
    parser.add_argument("--loss_mode", default="official", choices=["official", "proxy"])
    parser.add_argument("--autoencoder_ckpt", default="")
    parser.add_argument("--target", default="qkv", choices=["q", "v", "qv", "qkv", "all_linear"])
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--alpha", type=int, default=16)
    parser.add_argument("--train_steps", type=int, default=1000)
    parser.add_argument("--profile_steps", type=int, nargs="+", default=[0, 100, 250, 500, 750, 1000])
    parser.add_argument("--profile_batches", type=int, default=5)
    parser.add_argument(
        "--profile_noise_ratios", type=float, nargs="+",
        default=[0.05, 0.2, 0.4, 0.6, 0.8, 0.95],
        help="Normalized noise ratios to scan at every profile step.",
    )
    parser.add_argument("--topk_blocks", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--timestep", type=int, default=500)
    parser.add_argument("--input_noise_std", type=float, default=0.0)
    parser.add_argument("--max_images", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--profile_seed", type=int, default=2026)
    parser.add_argument("--block_regex", default="")
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--blockskip_count", type=int, default=0)
    parser.add_argument(
        "--blockskip_schedule", nargs="*", default=[],
        metavar="SIGMA:COUNT",
        help="Noise-aware skip counts, for example 0.05:8 0.4:4 0.95:8.",
    )
    parser.add_argument("--blockskip_min_run", type=int, default=2)
    parser.add_argument("--blockskip_max_run", type=int, default=4)
    parser.add_argument("--blockskip_max_runs", type=int, default=2)
    parser.add_argument(
        "--fixed_skip_blocks", nargs="*", default=[],
        help="Explicit logical block names to skip on every step; overrides gradient selection.",
    )
    parser.add_argument(
        "--always_skip_blocks", nargs="*", default=[],
        help="Mandatory blocks included in every dynamic skip set.",
    )
    parser.add_argument("--residual_cache_device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--residual_cache_dtype", choices=["fp16", "bf16", "fp32"], default="fp16")
    parser.add_argument(
        "--residual_execution",
        choices=["two_pass", "single_pass"],
        default="two_pass",
        help="Separate teacher-cache forward or in-forward no-grad residual bypass.",
    )
    parser.add_argument("--patch_min_fraction", type=float, default=1.0)
    parser.add_argument("--patch_max_fraction", type=float, default=1.0)
    parser.add_argument(
        "--train_noise_ratios", type=float, nargs="+", default=[],
        help="Discrete normalized timesteps sampled during training. Defaults to profile ratios.",
    )
    args = parser.parse_args()

    if args.loss_mode != "official":
        raise SystemExit("Noise-ratio profiling requires --loss_mode official.")
    try:
        blockskip_schedule = adaptive.parse_noise_int_schedule(
            args.blockskip_schedule
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    if args.fixed_skip_blocks and args.always_skip_blocks:
        raise SystemExit("Use either --fixed_skip_blocks or --always_skip_blocks, not both.")
    if not 0.0 < args.patch_min_fraction <= args.patch_max_fraction <= 1.0:
        raise SystemExit("Require 0 < --patch_min_fraction <= --patch_max_fraction <= 1.")
    core.set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model = core.load_model(args, device)
    diffusion, autoencoder = core.load_official_objective(args, device)
    model.requires_grad_(False)
    injected = core.inject_lora(model, args.target, args.rank, args.alpha, args.block_regex)
    model.train()

    dataset = core.ImageFolderDataset(args.data_dir, args.image_size, args.max_images)
    train_loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers
    )
    profile_loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )
    optimizer = torch.optim.AdamW(
        [param for param in model.parameters() if param.requires_grad], lr=args.lr
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    profile_steps = {step for step in args.profile_steps if 0 <= step <= args.train_steps}
    profile_steps.update({0, args.train_steps})
    if any(not 0.0 <= ratio <= 1.0 for ratio in args.profile_noise_ratios):
        raise SystemExit("--profile_noise_ratios values must be in [0, 1].")
    importance_rows = []
    for ratio in args.profile_noise_ratios:
        importance_rows.extend(profile_importance(
            model, diffusion, autoencoder, profile_loader, args, device, 0, ratio
        ))
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
    if (
        args.blockskip_count > 0
        or blockskip_schedule
        or args.fixed_skip_blocks
        or args.always_skip_blocks
    ):
        configured_blocks = set(args.fixed_skip_blocks) | set(args.always_skip_blocks)
        unknown = sorted(configured_blocks - set(block_names))
        if unknown:
            raise SystemExit("Unknown explicitly configured blocks: " + ", ".join(unknown))
        block_paths = adaptive.infer_block_module_paths(
            (name for name, _module in core.iter_lora_modules(model)),
            block_names,
            core.block_key,
            args.block_regex,
        )
        controller = adaptive.ResidualBlockController(
            model,
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
        requested_skip_count = 0
        cache_stats = adaptive.CacheStats(0.0, 0.0, 0, 0)
        if controller is not None:
            if args.fixed_skip_blocks:
                requested_skip_count = len(args.fixed_skip_blocks)
                skip_blocks = list(args.fixed_skip_blocks)
            else:
                mandatory = list(dict.fromkeys(args.always_skip_blocks))
                requested_skip_count = max(
                    len(mandatory),
                    adaptive.noise_scheduled_int(
                        noise_ratio, blockskip_schedule, args.blockskip_count
                    ),
                )
                extra_count = requested_skip_count - len(mandatory)
                extras = (
                    adaptive.select_low_score_runs(
                        importance_rows,
                        step,
                        noise_ratio,
                        extra_count,
                        args.blockskip_min_run,
                        args.blockskip_max_run,
                        max(1, args.blockskip_max_runs - 1),
                        excluded_blocks=mandatory,
                    )
                    if extra_count > 0
                    else []
                )
                selected = set(mandatory) | set(extras)
                skip_blocks = [block for block in block_names if block in selected]
            controller.configure(skip_blocks)
            if args.residual_execution == "two_pass":
                cache_stats = adaptive.populate_online_cache(
                    controller,
                    lambda: batch_loss(model, diffusion, autoencoder, batch, args, device),
                    device,
                )
            else:
                controller.set_mode("single_skip")
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        train_start = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        loss = batch_loss(model, diffusion, autoencoder, batch, args, device)
        if not torch.isfinite(loss):
            raise SystemExit(f"Non-finite training loss at step {step}.")
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            [param for param in model.parameters() if param.requires_grad], args.grad_clip
        )
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
            if args.residual_execution == "single_pass":
                cache_stats = controller.stats(0.0)
            controller.set_mode("full")
        del args._profile_noise_ratio
        train_rows.append(
            {
                "step": step,
                "loss": float(loss.detach().cpu()),
                "grad_norm": float(grad_norm),
                "noise_ratio": noise_ratio,
                "patch_size": patch_size,
                "skipped_blocks": ";".join(skip_blocks),
                "requested_skip_count": requested_skip_count,
                "skipped_block_count": len(skip_blocks),
                "residual_cache_time_s": cache_stats.elapsed_s,
                "residual_cache_mb": cache_stats.cache_mb,
                "cache_teacher_loss": cache_stats.teacher_loss,
                "residual_loss_abs_diff": (
                    abs(float(loss.detach().cpu()) - cache_stats.teacher_loss)
                    if controller is not None and args.residual_execution == "two_pass"
                    else ""
                ),
                "replayable_blocks": cache_stats.replayable_blocks,
                "fallback_blocks": cache_stats.fallback_blocks,
                "fallback_block_names": cache_stats.fallback_names,
                "residual_forward_max_abs_diff": cache_stats.max_reconstruction_abs_diff,
                "cache_peak_cuda_mem_mb": cache_stats.peak_cuda_mem_mb,
                "train_step_time_s": train_step_time_s,
                "train_peak_cuda_mem_mb": train_peak_cuda_mem_mb,
            }
        )
        if step % args.log_every == 0 or step == 1:
            print(f"step {step:05d}/{args.train_steps} loss={float(loss):.6f}")
        if step in profile_steps:
            if controller is not None:
                controller.set_mode("full")
            current = []
            for ratio in args.profile_noise_ratios:
                current.extend(profile_importance(
                    model, diffusion, autoencoder, profile_loader, args, device, step, ratio
                ))
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
    adapter_summary = adaptive.save_lora_adapter(
        model, output_dir / "lora_adapter.pt"
    )
    metadata = vars(args) | {
        "parsed_blockskip_schedule": blockskip_schedule,
        "profile_steps": sorted(profile_steps),
        "injected_module_count": len(injected),
        "model": "DiT-SR",
    } | adapter_summary
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Wrote results to {output_dir}")


if __name__ == "__main__":
    main()
