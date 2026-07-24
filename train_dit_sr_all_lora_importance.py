#!/usr/bin/env python3
"""Train All-LoRA on DiT-SR and track layer importance over time."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import profile_dit_sr_grad as core


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def batch_loss(model, batch, args, device):
    image = batch["image"].to(device)
    lq = F.interpolate(
        image, size=(args.lq_size, args.lq_size), mode="bicubic", align_corners=False
    )
    model_input = image
    if args.input_noise_std > 0:
        model_input = model_input + torch.randn_like(model_input) * args.input_noise_std
    timesteps = torch.full(
        (image.shape[0],), args.timestep, device=device, dtype=torch.long
    )
    output = model(model_input, timesteps, lq=lq)
    if output.shape == image.shape:
        return F.mse_loss(output.float(), image.float())
    return output.float().pow(2).mean()


def profile_importance(model, loader, args, device, train_step):
    cpu_state = torch.random.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if device.type == "cuda" else None
    python_state = random.getstate()
    core.set_seed(args.profile_seed)
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
            loss = batch_loss(model, batch, args, device)
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
                }
            )
        ranked = sorted(rows, key=lambda row: row["normalized_grad_score"], reverse=True)
        ranks = {row["block"]: rank for rank, row in enumerate(ranked, start=1)}
        for row in rows:
            row["importance_rank"] = ranks[row["block"]]
            row["selected_topk"] = ranks[row["block"]] <= args.topk_blocks
        return rows
    finally:
        model.zero_grad(set_to_none=True)
        random.setstate(python_state)
        torch.random.set_rng_state(cpu_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)


def topk_summary(rows):
    grouped = {}
    for row in rows:
        grouped.setdefault(int(row["train_step"]), []).append(row)
    first_step = min(grouped)
    baseline = {
        row["block"] for row in grouped[first_step] if bool(row["selected_topk"])
    }
    output = []
    for step in sorted(grouped):
        selected = {
            row["block"] for row in grouped[step] if bool(row["selected_topk"])
        }
        overlap = len(baseline & selected)
        output.append(
            {
                "train_step": step,
                "topk_blocks": ";".join(
                    row["block"]
                    for row in sorted(
                        grouped[step], key=lambda row: int(row["importance_rank"])
                    )
                    if bool(row["selected_topk"])
                ),
                "topk_overlap_count_vs_step0": overlap,
                "topk_overlap_ratio_vs_step0": overlap / max(len(baseline), 1),
                "topk_jaccard_vs_step0": overlap / max(len(baseline | selected), 1),
            }
        )
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config_path", default="configs/realsr_DiT.yaml")
    parser.add_argument("--ckpt_path", default="weights/realsr.pth")
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--image_size", type=int, default=64)
    parser.add_argument("--lq_size", type=int, default=64)
    parser.add_argument("--target", default="qkv", choices=["q", "v", "qv", "qkv", "all_linear"])
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--alpha", type=int, default=16)
    parser.add_argument("--train_steps", type=int, default=1000)
    parser.add_argument("--profile_steps", type=int, nargs="+", default=[0, 100, 250, 500, 750, 1000])
    parser.add_argument("--profile_batches", type=int, default=5)
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
    args = parser.parse_args()

    core.set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model = core.load_model(args, device)
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
    importance_rows = profile_importance(model, profile_loader, args, device, 0)
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
        optimizer.zero_grad(set_to_none=True)
        loss = batch_loss(model, batch, args, device)
        if not torch.isfinite(loss):
            raise SystemExit(f"Non-finite training loss at step {step}.")
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            [param for param in model.parameters() if param.requires_grad], args.grad_clip
        )
        optimizer.step()
        train_rows.append(
            {"step": step, "loss": float(loss.detach().cpu()), "grad_norm": float(grad_norm)}
        )
        if step % args.log_every == 0 or step == 1:
            print(f"step {step:05d}/{args.train_steps} loss={float(loss):.6f}")
        if step in profile_steps:
            current = profile_importance(model, profile_loader, args, device, step)
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
        "model": "DiT-SR",
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Wrote results to {output_dir}")


if __name__ == "__main__":
    main()
