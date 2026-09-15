#!/usr/bin/env python3
"""Fine-tune packed-qkv LoRA modules in ResShift with gradient-based placement.

Copy this script together with ``profile_resshift_grad.py`` and
``inspect_resshift_structure.py`` to the official ResShift repository root.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import torch
from torch.utils.data import DataLoader

import adaptive_grad_blockskip as adaptive
from inspect_resshift_structure import install_timm_layers_stub, load_checkpoint
from profile_resshift_grad import (
    ImageFolderDataset,
    block_from_qkv,
    inject_packed_qkv_lora,
    load_state,
    natural_key,
    noise_ratio_to_timestep,
    noise_sigma,
    resolve_under_root,
    set_seed,
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def aggregate_importance(
    rows: list[dict[str, str]],
    step: int,
    score_key: str,
) -> list[dict]:
    required = {"train_step", "noise_ratio", "block", "block_index", score_key}
    if not rows:
        raise ValueError("Importance CSV is empty")
    missing = sorted(required - set(rows[0]))
    if missing:
        raise ValueError("Importance CSV is missing columns: " + ", ".join(missing))

    chosen = [row for row in rows if int(row["train_step"]) == step]
    if not chosen:
        available = sorted({int(row["train_step"]) for row in rows})
        raise ValueError(f"Importance step {step} is unavailable; available={available}")

    by_noise: dict[float, list[dict[str, str]]] = {}
    for row in chosen:
        by_noise.setdefault(float(row["noise_ratio"]), []).append(row)
    blocks = {row["block"] for row in chosen}
    if any({row["block"] for row in group} != blocks for group in by_noise.values()):
        raise ValueError("Importance CSV does not cover the same blocks at every noise anchor")

    totals = {}
    for noise, group in by_noise.items():
        total = sum(float(row[score_key]) for row in group)
        if not math.isfinite(total) or total <= 0:
            raise ValueError(f"Non-positive importance total at noise_ratio={noise:g}")
        totals[noise] = total

    details = []
    for block in blocks:
        block_rows = [row for row in chosen if row["block"] == block]
        shares = [float(row[score_key]) / totals[float(row["noise_ratio"])] for row in block_rows]
        mean_share = sum(shares) / len(shares)
        details.append({
            "block": block,
            "block_index": int(block_rows[0]["block_index"]),
            "mean_score_share": mean_share,
            "mean_relative_importance": mean_share * len(blocks),
            "noise_anchor_count": len(block_rows),
        })
    details.sort(key=lambda row: (-row["mean_relative_importance"], row["block_index"]))
    cumulative = 0.0
    for rank, row in enumerate(details, start=1):
        cumulative += row["mean_score_share"]
        row["aggregate_rank"] = rank
        row["cumulative_utility"] = cumulative
    return details


def read_blockskip_importance(path: Path, step: int) -> list[dict]:
    rows = read_csv(path)
    required = {
        "train_step",
        "noise_ratio",
        "block",
        "block_index",
        "normalized_grad_score",
    }
    if not rows:
        raise ValueError("Block-skip importance CSV is empty")
    missing = sorted(required - set(rows[0]))
    if missing:
        raise ValueError(
            "Block-skip importance CSV is missing columns: " + ", ".join(missing)
        )
    chosen = [row for row in rows if int(row["train_step"]) == step]
    if not chosen:
        available = sorted({int(row["train_step"]) for row in rows})
        raise ValueError(
            f"Block-skip importance step {step} is unavailable; available={available}"
        )
    return chosen


def read_blockskip_policy(path: Path) -> dict[float, list[str]]:
    rows = read_csv(path)
    if not rows:
        raise ValueError("Block-skip policy CSV is empty")
    required = {"noise_ratio", "skip_blocks"}
    missing = sorted(required - set(rows[0]))
    if missing:
        raise ValueError("Block-skip policy CSV is missing columns: " + ", ".join(missing))
    result = {}
    for row in rows:
        noise = float(row["noise_ratio"])
        if noise in result:
            raise ValueError(f"Duplicate block-skip policy noise ratio: {noise:g}")
        result[noise] = [block for block in row["skip_blocks"].split(";") if block]
    return result


def choose_blocks(
    all_blocks: list[str],
    importance: list[dict] | None,
    selection: str,
    threshold: float,
    topk: int,
) -> tuple[list[str], dict[str, float | int | str]]:
    if selection == "all":
        return list(all_blocks), {
            "selection": "all",
            "selected_utility": 1.0,
            "selected_block_fraction": 1.0,
        }
    if importance is None:
        raise ValueError(f"--importance_csv is required for selection={selection}")
    known = {row["block"] for row in importance}
    if known != set(all_blocks):
        raise ValueError(
            "Importance blocks do not match model blocks: "
            f"missing={sorted(set(all_blocks) - known)} unknown={sorted(known - set(all_blocks))}"
        )
    if selection == "threshold":
        selected_rows = [
            row for row in importance
            if row["mean_relative_importance"] >= threshold
        ]
    else:
        if topk < 1 or topk > len(all_blocks):
            raise ValueError(f"topk must be in [1, {len(all_blocks)}]")
        selected_rows = importance[:topk]
    if not selected_rows:
        raise ValueError("LoRA selection produced no blocks")
    selected_set = {row["block"] for row in selected_rows}
    selected = [block for block in all_blocks if block in selected_set]
    utility = sum(row["mean_score_share"] for row in selected_rows)
    return selected, {
        "selection": selection,
        "importance_threshold": threshold if selection == "threshold" else None,
        "requested_topk": topk if selection == "topk" else None,
        "selected_utility": utility,
        "selected_block_fraction": len(selected) / len(all_blocks),
    }


def autocast_context(device: torch.device, dtype_name: str):
    if device.type != "cuda" or dtype_name == "fp32":
        return nullcontext()
    dtype = torch.bfloat16 if dtype_name == "bf16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def save_adapter(path: Path, injected, selected_blocks, args) -> None:
    modules = {}
    for name, module in injected.items():
        modules[name] = {
            "block": block_from_qkv(name),
            "lora_down": module.lora_down.weight.detach().cpu(),
            "lora_up": module.lora_up.weight.detach().cpu(),
            "rank": module.rank,
            "alpha": float(args.alpha),
        }
    torch.save({
        "format": "resshift_packed_qkv_lora_v1",
        "target": "packed_qkv",
        "rank": args.rank,
        "alpha": float(args.alpha),
        "selected_blocks": list(selected_blocks),
        "modules": modules,
    }, path)


def train(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    root = Path(args.resshift_root).resolve()
    config_path = resolve_under_root(root, args.config_path)
    checkpoint = resolve_under_root(root, args.checkpoint)
    autoencoder_checkpoint = resolve_under_root(root, args.autoencoder_checkpoint)
    for path in (config_path, checkpoint, autoencoder_checkpoint):
        if not path.is_file():
            raise FileNotFoundError(path)
    sys.path.insert(0, str(root))

    from omegaconf import OmegaConf
    from utils.util_common import get_obj_from_str

    install_timm_layers_stub(torch)
    config = OmegaConf.load(config_path)
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")

    model = get_obj_from_str(config.model.target)(**config.model.get("params", {}))
    missing, unexpected = load_checkpoint(model, checkpoint, torch)
    if missing or unexpected:
        raise RuntimeError(
            f"Model checkpoint mismatch: missing={missing[:10]} unexpected={unexpected[:10]}"
        )
    model.to(device).requires_grad_(False)

    autoencoder = get_obj_from_str(config.autoencoder.target)(
        **config.autoencoder.get("params", {})
    )
    missing, unexpected = load_state(autoencoder, autoencoder_checkpoint)
    if missing or unexpected:
        raise RuntimeError(
            "Autoencoder checkpoint mismatch: "
            f"missing={missing[:10]} unexpected={unexpected[:10]}"
        )
    autoencoder.to(device).eval().requires_grad_(False)
    diffusion = get_obj_from_str(config.diffusion.target)(
        **config.diffusion.get("params", {})
    )

    qkv_names = [
        name for name, module in model.named_modules()
        if isinstance(module, torch.nn.Linear)
        and name.endswith(".attn.qkv")
        and module.out_features == 3 * module.in_features
    ]
    all_blocks = sorted((block_from_qkv(name) for name in qkv_names), key=natural_key)
    importance = None
    if args.importance_csv:
        importance = aggregate_importance(
            read_csv(Path(args.importance_csv)), args.importance_step, args.score_key
        )
    selected, selection_report = choose_blocks(
        all_blocks,
        importance,
        args.lora_selection,
        args.importance_threshold,
        args.topk_blocks,
    )
    injected = inject_packed_qkv_lora(
        model, args.rank, args.alpha, selected_blocks=set(selected)
    )
    if len(injected) != len(selected):
        raise RuntimeError(f"Injected {len(injected)} modules for {len(selected)} blocks")

    blockskip_schedule = adaptive.parse_noise_int_schedule(args.blockskip_schedule)
    blockskip_policy = (
        read_blockskip_policy(Path(args.blockskip_policy_csv))
        if args.blockskip_policy_csv
        else {}
    )
    controller = None
    controller_blocks: list[str] = []
    blockskip_rows: list[dict] = []
    if blockskip_schedule or blockskip_policy or args.blockskip_controller_only:
        if not args.blockskip_importance_csv and not blockskip_policy:
            raise ValueError(
                "--blockskip_importance_csv is required when the bypass controller is enabled"
            )
        if args.blockskip_importance_csv:
            blockskip_rows = read_blockskip_importance(
                Path(args.blockskip_importance_csv), args.blockskip_importance_step
            )
            known = {str(row["block"]) for row in blockskip_rows}
            if known != set(all_blocks):
                raise ValueError(
                    "Block-skip importance blocks do not match the model: "
                    f"missing={sorted(set(all_blocks) - known)} "
                    f"unknown={sorted(known - set(all_blocks))}"
                )
        protected = set(selected) if args.protect_selected_lora_blocks else set()
        if blockskip_policy:
            potential = {block for blocks in blockskip_policy.values() for block in blocks}
            unknown = potential - set(all_blocks)
            if unknown:
                raise ValueError(f"Block-skip policy contains unknown blocks: {sorted(unknown)}")
            overlap = potential & protected
            if overlap:
                raise ValueError(f"Block-skip policy contains protected LoRA blocks: {sorted(overlap)}")
        else:
            potential = set()
            for ratio in sorted({float(row["noise_ratio"]) for row in blockskip_rows}):
                requested = adaptive.noise_scheduled_int(ratio, blockskip_schedule, 0)
                potential.update(
                    adaptive.select_low_score_runs(
                        blockskip_rows,
                        args.blockskip_importance_step,
                        ratio,
                        requested,
                        args.blockskip_min_run,
                        args.blockskip_max_run,
                        args.blockskip_max_runs,
                        score_key=args.blockskip_score_key,
                        excluded_blocks=protected,
                    )
                )
        controller_blocks = [block for block in all_blocks if block in potential]
        if not controller_blocks and not args.blockskip_controller_only:
            raise ValueError("The configured bypass policy never selects a block")
        if controller_blocks:
            controller = adaptive.ResidualBlockController(
                model,
                {block: block for block in controller_blocks},
                cache_device="cpu",
                cache_dtype=torch.float32,
            )
        print(
            f"Backward controller wrappers: {len(controller_blocks)}/{len(all_blocks)} "
            "candidate blocks"
        )
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable:
        raise RuntimeError("No trainable LoRA parameters")
    trainable_params = sum(parameter.numel() for parameter in trainable)
    optimizer = torch.optim.AdamW(
        trainable, lr=args.lr, betas=(0.9, 0.999), weight_decay=args.weight_decay
    )
    model.train()

    dataset = ImageFolderDataset(
        args.data_dir, args.image_size, args.sr_scale, args.max_images, args.seed
    )
    loader_generator = torch.Generator()
    loader_generator.manual_seed(args.seed)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        generator=loader_generator,
    )
    iterator = iter(loader)
    anchors = sorted(set(args.train_noise_ratios))
    noise_rng = random.Random(args.seed + 1009)
    latent_scale = 2 ** (len(config.autoencoder.params.ddconfig.ch_mult) - 1)
    latent_channels = int(config.autoencoder.params.embed_dim)

    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f"Output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    if importance is not None:
        selected_set = set(selected)
        importance_rows = []
        for row in importance:
            item = dict(row)
            item["selected"] = row["block"] in selected_set
            importance_rows.append(item)
        write_csv(output_dir / "aggregated_importance.csv", importance_rows)

    print(
        f"LoRA selection={args.lora_selection}: blocks={len(selected)}/{len(all_blocks)} "
        f"modules={len(injected)} params={trainable_params}"
    )
    print("Selected blocks: " + ", ".join(selected))

    rows = []
    experiment_started = time.perf_counter()
    train_time = 0.0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for step in range(1, args.train_steps + 1):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        ratio = anchors[noise_rng.randrange(len(anchors))]
        timestep = noise_ratio_to_timestep(ratio, diffusion.num_timesteps)
        gt = batch["gt"].to(device, non_blocking=True)
        lq = batch["lq"].to(device, non_blocking=True)
        tt = torch.full((gt.shape[0],), timestep, dtype=torch.long, device=device)
        latent_resolution = gt.shape[-1] // latent_scale
        noise = torch.randn(
            (gt.shape[0], latent_channels, latent_resolution, latent_resolution),
            device=device,
            dtype=gt.dtype,
        )

        if blockskip_policy:
            policy_ratio = min(blockskip_policy, key=lambda value: abs(value - ratio))
            policy_blocks = list(blockskip_policy[policy_ratio])
            requested_skip_count = len(policy_blocks)
        else:
            policy_blocks = []
            requested_skip_count = adaptive.noise_scheduled_int(
                ratio, blockskip_schedule, 0
            )
        skipped_blocks: list[str] = []
        if controller is not None:
            if not args.blockskip_controller_only and requested_skip_count > 0:
                if blockskip_policy:
                    skipped_blocks = policy_blocks
                else:
                    excluded = set(selected) if args.protect_selected_lora_blocks else set()
                    skipped_blocks = adaptive.select_low_score_runs(
                        blockskip_rows,
                        args.blockskip_importance_step,
                        ratio,
                        requested_skip_count,
                        args.blockskip_min_run,
                        args.blockskip_max_run,
                        args.blockskip_max_runs,
                        score_key=args.blockskip_score_key,
                        excluded_blocks=excluded,
                    )
            controller.configure(skipped_blocks)
            controller.set_mode(
                "single_run_skip"
                if args.residual_execution == "single_run"
                else "single_skip"
            )

        sync(device)
        step_started = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        with autocast_context(device, args.dtype):
            result = diffusion.training_losses(
                model,
                gt,
                lq,
                tt,
                first_stage_model=autoencoder,
                model_kwargs={"lq": lq},
                noise=noise,
            )
            terms = result[0] if isinstance(result, (tuple, list)) else result
            loss = terms["mse"].mean()
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite loss at step={step}")
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(trainable, args.grad_clip)
        optimizer.step()
        sync(device)
        elapsed = time.perf_counter() - step_started
        train_time += elapsed
        bypass_stats = (
            controller.stats(elapsed)
            if controller is not None
            else adaptive.CacheStats(
                elapsed_s=elapsed,
                cache_mb=0.0,
                replayable_blocks=0,
                fallback_blocks=0,
            )
        )
        peak_mb = (
            torch.cuda.max_memory_allocated(device) / (1024.0**2)
            if device.type == "cuda" else 0.0
        )
        row = {
            "step": step,
            "noise_ratio": ratio,
            "timestep": timestep,
            "sigma": noise_sigma(diffusion, timestep),
            "loss": float(loss.detach().cpu()),
            "grad_norm": float(torch.as_tensor(grad_norm).detach().cpu()),
            "train_step_time_s": elapsed,
            "peak_cuda_mem_mb": peak_mb,
            "train_peak_cuda_mem_mb": peak_mb,
            "cache_peak_cuda_mem_mb": 0.0,
            "residual_cache_time_s": 0.0,
            "residual_cache_mb": 0.0,
            "requested_skip_count": requested_skip_count,
            "skipped_block_count": len(skipped_blocks),
            "skipped_blocks": ";".join(skipped_blocks),
            "replayable_blocks": bypass_stats.replayable_blocks,
            "fallback_blocks": bypass_stats.fallback_blocks,
            "fallback_block_names": bypass_stats.fallback_names,
            "residual_forward_max_abs_diff": (
                bypass_stats.max_reconstruction_abs_diff
            ),
            "bypass_run_count": bypass_stats.bypass_run_count,
            "bypass_run_blocks": bypass_stats.bypass_run_blocks,
        }
        rows.append(row)
        if step == 1 or step % args.log_every == 0 or step == args.train_steps:
            print(
                f"step={step}/{args.train_steps} noise={ratio:g} "
                f"loss={row['loss']:.6f} time={elapsed:.4f}s"
            )

    adapter_path = output_dir / "lora_adapter.pt"
    checkpoint_started = time.perf_counter()
    save_adapter(adapter_path, injected, selected, args)
    checkpoint_time = time.perf_counter() - checkpoint_started
    experiment_time = time.perf_counter() - experiment_started
    write_csv(output_dir / "train_log.csv", rows)

    metadata = {
        "resshift_root": str(root),
        "config_path": str(config_path),
        "ckpt_path": str(checkpoint),
        "autoencoder_ckpt": str(autoencoder_checkpoint),
        "data_dir": str(Path(args.data_dir).resolve()),
        "image_size": args.image_size,
        "sr_scale": args.sr_scale,
        "target": "packed_qkv",
        "rank": args.rank,
        "alpha": args.alpha,
        "loss_mode": "official_resshift",
        "lora_selection": args.lora_selection,
        "importance_csv": args.importance_csv or None,
        "importance_step": args.importance_step,
        "score_key": args.score_key,
        "selected_blocks": selected,
        "selected_lora_blocks": selected,
        "selected_block_count": len(selected),
        "candidate_block_count": len(all_blocks),
        "trainable_lora_params": trainable_params,
        "train_noise_ratios": anchors,
        "train_steps": args.train_steps,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "seed": args.seed,
        "blockskip_schedule": blockskip_schedule,
        "blockskip_importance_csv": args.blockskip_importance_csv or None,
        "blockskip_policy_csv": args.blockskip_policy_csv or None,
        "blockskip_importance_step": args.blockskip_importance_step,
        "blockskip_score_key": args.blockskip_score_key,
        "protect_selected_lora_blocks": args.protect_selected_lora_blocks,
        "blockskip_controller_only": args.blockskip_controller_only,
        "controller_blocks": controller_blocks,
        "controller_block_count": len(controller_blocks),
        "residual_execution": args.residual_execution,
        **selection_report,
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    adapter_size_mb = adapter_path.stat().st_size / (1024.0**2)
    summary = [{
        "method": args.method_name,
        "train_steps": args.train_steps,
        "selected_lora_blocks": len(selected),
        "trainable_lora_params": trainable_params,
        "train_step_time_s": train_time,
        "mean_train_step_time_s": train_time / args.train_steps,
        "experiment_time_s": experiment_time,
        "checkpoint_time_s": checkpoint_time,
        "non_train_overhead_s": experiment_time - train_time,
        "peak_cuda_mem_mb": max(row["peak_cuda_mem_mb"] for row in rows),
        "adapter_size_mb": adapter_size_mb,
        "adapter_path": str(adapter_path),
    }]
    write_csv(output_dir / "summary.csv", summary)
    print(json.dumps(summary[0], indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resshift_root", default=".")
    parser.add_argument("--config_path", default="configs/realsr_swinunet_realesrgan256.yaml")
    parser.add_argument("--checkpoint", default="weights/resshift_realsrx4_s15_v1.pth")
    parser.add_argument("--autoencoder_checkpoint", default="weights/autoencoder_vq_f4.pth")
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--method_name", default="ResShift-LoRA")
    parser.add_argument("--importance_csv", default="")
    parser.add_argument("--importance_step", type=int, default=0)
    parser.add_argument("--score_key", default="normalized_grad_score")
    parser.add_argument("--lora_selection", choices=["all", "threshold", "topk"], default="threshold")
    parser.add_argument("--importance_threshold", type=float, default=1.0)
    parser.add_argument("--topk_blocks", type=int, default=6)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--sr_scale", type=int, default=4)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=16.0)
    parser.add_argument("--train_steps", type=int, default=100)
    parser.add_argument("--train_noise_ratios", type=float, nargs="+", default=[0.05, 0.2, 0.4, 0.6, 0.8, 0.95])
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--dtype", choices=["fp32", "bf16", "fp16"], default="fp32")
    parser.add_argument("--max_images", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--blockskip_schedule", nargs="*", default=[])
    parser.add_argument("--blockskip_importance_csv", default="")
    parser.add_argument("--blockskip_policy_csv", default="")
    parser.add_argument("--blockskip_importance_step", type=int, default=0)
    parser.add_argument("--blockskip_score_key", default="normalized_grad_score")
    parser.add_argument("--blockskip_min_run", type=int, default=1)
    parser.add_argument("--blockskip_max_run", type=int, default=18)
    parser.add_argument("--blockskip_max_runs", type=int, default=18)
    parser.add_argument("--protect_selected_lora_blocks", action="store_true")
    parser.add_argument("--blockskip_controller_only", action="store_true")
    parser.add_argument(
        "--residual_execution",
        choices=["single_pass", "single_run"],
        default="single_pass",
    )
    parser.add_argument("--cpu", action="store_true")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.train_steps < 1 or args.batch_size < 1 or args.rank < 1:
        parser.error("train_steps, batch_size, and rank must be positive")
    if args.image_size < 1 or args.sr_scale < 1 or args.image_size % args.sr_scale:
        parser.error("image_size must be positive and divisible by sr_scale")
    if args.lr <= 0 or args.grad_clip <= 0:
        parser.error("lr and grad_clip must be positive")
    if args.importance_threshold < 0:
        parser.error("importance_threshold must be non-negative")
    if args.blockskip_min_run < 1:
        parser.error("blockskip_min_run must be positive")
    if args.blockskip_max_run < args.blockskip_min_run:
        parser.error("blockskip_max_run must be >= blockskip_min_run")
    if args.blockskip_max_runs < 1:
        parser.error("blockskip_max_runs must be positive")
    if any(not 0.0 <= ratio <= 1.0 for ratio in args.train_noise_ratios):
        parser.error("train_noise_ratios must be in [0, 1]")
    try:
        train(args)
    except (OSError, RuntimeError, ValueError, KeyError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
