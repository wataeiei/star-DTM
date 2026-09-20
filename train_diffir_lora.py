#!/usr/bin/env python3
"""Fine-tune DiffIR-S2 with all or importance-selected packed-QKV LoRA.

Copy this file together with ``profile_diffir_grad.py`` and
``profile_diffir_lora_costs.py`` to the DiffIR-SRGAN repository root. The
training objective and optimizer defaults match the released DiffIR-S2 x4
configuration: pixel L1 plus final-prior L1, optimized with Adam.
"""

from __future__ import annotations

import argparse
import csv
import json
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
)
from profile_diffir_lora_costs import configure_wrappers, read_importance


def sync(device) -> None:
    import torch

    if device.type == "cuda":
        torch.cuda.synchronize(device)


def atomic_torch_save(payload: Any, path: Path) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def adapter_payload(
    injected,
    selected_blocks: set[str],
    args: argparse.Namespace,
    step: int,
) -> dict[str, Any]:
    modules = {}
    for module_name, wrapper in injected.items():
        block = block_from_qkv(module_name)
        if block not in selected_blocks:
            continue
        modules[module_name] = {
            "lora_down": wrapper.lora_down.weight.detach().cpu(),
            "lora_up": wrapper.lora_up.weight.detach().cpu(),
        }
    return {
        "format": "diffir_packed_qkv_lora_v1",
        "model": "DiffIR-S2",
        "step": step,
        "rank": args.rank,
        "alpha": float(args.alpha),
        "selection": args.selection,
        "topk": len(selected_blocks),
        "selected_blocks": sorted(selected_blocks, key=natural_key),
        "modules": modules,
    }


def grad_norm(parameters) -> float:
    import torch

    squared_norms = []
    for parameter in parameters:
        if parameter.grad is not None:
            squared_norms.append(parameter.grad.detach().float().pow(2).sum())
    if not squared_norms:
        return 0.0
    # Perform one device-to-host synchronization instead of one per parameter.
    return float(torch.stack(squared_norms).sum().sqrt().cpu())


def train(args: argparse.Namespace) -> None:
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is unavailable")
    if args.image_size != args.lq_size * args.sr_scale:
        raise SystemExit("Require image_size == lq_size * sr_scale")
    if args.sr_scale != 4:
        raise SystemExit("The released SISR DiffIR-S1/S2 checkpoints are x4")
    if args.train_steps < 1 or args.batch_size < 1:
        raise SystemExit("train_steps and batch_size must be positive")
    if args.rank < 1 or args.alpha <= 0 or args.lr <= 0:
        raise SystemExit("rank, alpha, and lr must be positive")
    if args.selection == "topk" and not args.importance_csv:
        raise SystemExit("--selection topk requires --importance_csv")

    output_dir = Path(args.output_dir).expanduser()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise SystemExit(
            f"Output directory is not empty: {output_dir}. "
            "Use a new directory to preserve the experiment."
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)
    device = torch.device(args.device)
    experiment_started = time.perf_counter()

    dataset = ImageFolderDataset(
        Path(args.data_dir).expanduser(),
        args.image_size,
        args.lq_size,
        args.max_images,
        args.seed,
    )
    overlap = audit_overlap(
        dataset.paths,
        [Path(path).expanduser() for path in args.protected_eval_dir],
    )
    generator = torch.Generator()
    generator.manual_seed(args.seed)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )

    s2, s1, loaded_keys = load_models(
        Path(args.s2_checkpoint).expanduser(),
        Path(args.s1_checkpoint).expanduser(),
        args.checkpoint_key,
        device,
    )
    injected = inject_qkv_lora(s2, args.rank, args.alpha)
    model_blocks = {block_from_qkv(name) for name in injected}
    if len(model_blocks) != args.expected_blocks:
        raise RuntimeError(
            f"Expected {args.expected_blocks} DiffIR blocks, found {len(model_blocks)}"
        )

    importance = None
    if args.importance_csv:
        importance = read_importance(
            Path(args.importance_csv).expanduser(), args.expected_blocks
        )
        ranked_blocks = {row["block"] for row in importance}
        if ranked_blocks != model_blocks:
            raise RuntimeError(
                "Importance/model block mismatch: "
                f"missing={sorted(model_blocks - ranked_blocks, key=natural_key)} "
                f"unknown={sorted(ranked_blocks - model_blocks, key=natural_key)}"
            )

    if args.selection == "all":
        selected_blocks = set(model_blocks)
        selected_utility = 1.0
    else:
        assert importance is not None
        if not 1 <= args.topk <= len(importance):
            raise SystemExit(f"topk must be in [1, {len(importance)}]")
        selected_blocks = {row["block"] for row in importance[: args.topk]}
        selected_utility = float(importance[args.topk - 1]["cumulative_utility"])

    parameters = configure_wrappers(injected, selected_blocks)
    if not parameters:
        raise RuntimeError("No trainable LoRA parameters were selected")
    optimizer = torch.optim.Adam(
        parameters,
        lr=args.lr,
        betas=(args.beta1, args.beta2),
        weight_decay=args.weight_decay,
    )
    trainable_params = sum(parameter.numel() for parameter in parameters)

    metadata = {
        "model": "DiffIR-S2",
        "s2_checkpoint": str(Path(args.s2_checkpoint).expanduser().resolve()),
        "s1_checkpoint": str(Path(args.s1_checkpoint).expanduser().resolve()),
        "checkpoint_keys": loaded_keys,
        "data_dir": str(Path(args.data_dir).expanduser().resolve()),
        "overlap_audit": overlap,
        "selection": args.selection,
        "importance_csv": (
            str(Path(args.importance_csv).expanduser().resolve())
            if args.importance_csv else None
        ),
        "selected_block_count": len(selected_blocks),
        "selected_block_fraction": len(selected_blocks) / len(model_blocks),
        "selected_utility": selected_utility,
        "selected_blocks": sorted(selected_blocks, key=natural_key),
        "rank": args.rank,
        "alpha": float(args.alpha),
        "trainable_lora_params": trainable_params,
        "train_steps": args.train_steps,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "optimizer": "Adam",
        "betas": [args.beta1, args.beta2],
        "weight_decay": args.weight_decay,
        "grad_clip": args.grad_clip,
        "pixel_weight": args.pixel_weight,
        "prior_weight": args.prior_weight,
        "precision": "fp32",
        "loss_mode": "official_diffir_s2_pixel_l1_plus_final_prior_l1",
        "seed": args.seed,
        "dataset_size": len(dataset),
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )

    print(
        f"Selection={args.selection} blocks={len(selected_blocks)}/{len(model_blocks)} "
        f"utility={selected_utility:.2%} params={trainable_params}"
    )
    for block in sorted(selected_blocks, key=natural_key):
        print(f"  {block}")

    log_path = output_dir / "train_log.csv"
    fieldnames = [
        "step",
        "epoch",
        "image",
        "loss",
        "pixel_loss",
        "prior_l1_loss",
        "grad_norm",
        "learning_rate",
        "train_step_time_s",
        "peak_cuda_mem_mb",
    ]
    checkpoint_time = 0.0
    train_time = 0.0
    losses: list[float] = []
    data_iterator = iter(loader)
    epoch = 1
    s2.train()
    s1.eval()
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    with log_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()

        for step in range(1, args.train_steps + 1):
            try:
                batch = next(data_iterator)
            except StopIteration:
                epoch += 1
                data_iterator = iter(loader)
                batch = next(data_iterator)

            gt = batch["gt"].to(device, non_blocking=True)
            lq = batch["lq"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            sync(device)
            step_started = time.perf_counter()
            with torch.no_grad():
                teacher_ipr, _ = s1.E(lq, gt)
            sr, pred_ipr_list = s2(lq, teacher_ipr.detach())
            pixel_loss = F.l1_loss(sr, gt)
            prior_loss = F.l1_loss(pred_ipr_list[-1], teacher_ipr.detach())
            loss = args.pixel_weight * pixel_loss + args.prior_weight * prior_loss
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss at step {step}: {loss}")
            loss.backward()
            if args.grad_clip > 0:
                current_grad_norm_tensor = torch.nn.utils.clip_grad_norm_(
                    parameters, args.grad_clip
                )
            else:
                current_grad_norm_tensor = None
            optimizer.step()
            sync(device)
            step_time = time.perf_counter() - step_started
            train_time += step_time
            current_grad_norm = (
                float(current_grad_norm_tensor.detach().cpu())
                if current_grad_norm_tensor is not None
                else grad_norm(parameters)
            )

            loss_value = float(loss.detach().cpu())
            pixel_value = float(pixel_loss.detach().cpu())
            prior_value = float(prior_loss.detach().cpu())
            losses.append(loss_value)
            peak_mb = (
                torch.cuda.max_memory_allocated(device) / 2**20
                if device.type == "cuda" else 0.0
            )
            image_names = [Path(path).name for path in batch["path"]]
            writer.writerow(
                {
                    "step": step,
                    "epoch": epoch,
                    "image": ";".join(image_names),
                    "loss": loss_value,
                    "pixel_loss": pixel_value,
                    "prior_l1_loss": prior_value,
                    "grad_norm": current_grad_norm,
                    "learning_rate": optimizer.param_groups[0]["lr"],
                    "train_step_time_s": step_time,
                    "peak_cuda_mem_mb": peak_mb,
                }
            )
            if step % args.log_every == 0 or step == 1 or step == args.train_steps:
                handle.flush()
                recent = losses[-min(args.log_every, len(losses)) :]
                print(
                    f"step {step:04d}/{args.train_steps}: "
                    f"loss={loss_value:.6f} recent={np.mean(recent):.6f} "
                    f"time={step_time:.4f}s peak={peak_mb:.1f}MB",
                    flush=True,
                )

            if args.checkpoint_every > 0 and step % args.checkpoint_every == 0:
                sync(device)
                checkpoint_started = time.perf_counter()
                atomic_torch_save(
                    adapter_payload(injected, selected_blocks, args, step),
                    output_dir / f"lora_adapter_step_{step:05d}.pt",
                )
                checkpoint_time += time.perf_counter() - checkpoint_started

    sync(device)
    checkpoint_started = time.perf_counter()
    adapter_path = output_dir / "lora_adapter.pt"
    atomic_torch_save(
        adapter_payload(injected, selected_blocks, args, args.train_steps),
        adapter_path,
    )
    checkpoint_time += time.perf_counter() - checkpoint_started
    experiment_time = time.perf_counter() - experiment_started
    peak_mb = (
        torch.cuda.max_memory_allocated(device) / 2**20
        if device.type == "cuda" else 0.0
    )
    nonzero_modules = sum(
        int(torch.count_nonzero(wrapper.lora_up.weight.detach()).item() > 0)
        for module_name, wrapper in injected.items()
        if block_from_qkv(module_name) in selected_blocks
    )

    summary = {
        "method": args.method,
        "train_steps": args.train_steps,
        "selected_lora_blocks": len(selected_blocks),
        "selected_utility": selected_utility,
        "trainable_lora_params": trainable_params,
        "train_step_time_s": train_time,
        "mean_train_step_time_s": train_time / args.train_steps,
        "experiment_time_s": experiment_time,
        "checkpoint_time_s": checkpoint_time,
        "non_train_overhead_s": experiment_time - train_time,
        "peak_cuda_mem_mb": peak_mb,
        "final_loss": losses[-1],
        "mean_last100_loss": float(np.mean(losses[-min(100, len(losses)) :])),
        "adapter_size_mb": adapter_path.stat().st_size / 2**20,
        "lora_module_count": len(selected_blocks),
        "nonzero_lora_module_count": nonzero_modules,
        "adapter_path": str(adapter_path),
    }
    with (output_dir / "summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary))
        writer.writeheader()
        writer.writerow(summary)
    metadata["summary"] = summary
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote results to {output_dir}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
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
    parser.add_argument("--method", required=True)
    parser.add_argument("--selection", choices=("all", "topk"), required=True)
    parser.add_argument("--importance_csv", default="")
    parser.add_argument("--topk", type=int, default=21)
    parser.add_argument("--expected_blocks", type=int, default=44)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--lq_size", type=int, default=64)
    parser.add_argument("--sr_scale", type=int, default=4)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=16.0)
    parser.add_argument("--train_steps", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_images", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.99)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--grad_clip", type=float, default=0.0)
    parser.add_argument("--pixel_weight", type=float, default=1.0)
    parser.add_argument("--prior_weight", type=float, default=1.0)
    parser.add_argument("--checkpoint_every", type=int, default=250)
    parser.add_argument("--log_every", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    return parser


def main() -> None:
    train(build_parser().parse_args())


if __name__ == "__main__":
    main()
