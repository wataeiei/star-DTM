#!/usr/bin/env python3
"""Train a noise-conditioned Grad-TopK Parallel Adapter on official DiT4SR."""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import dit4sr_parallel_adapter as parallel
import profile_hf_dit4sr_grad as core


def write_csv(path: str | Path, rows: list[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def prepare_flow_batch(pipe, transformer, batch, args, device, noise_ratio):
    images = batch["image"]
    clean = core.image_to_hidden_states(pipe, images, transformer, args, device)
    control_image = F.interpolate(
        images,
        scale_factor=1.0 / args.sr_scale,
        mode="bicubic",
        align_corners=False,
    )
    control_image = F.interpolate(
        control_image,
        size=images.shape[-2:],
        mode="bicubic",
        align_corners=False,
    )
    control = core.image_to_hidden_states(pipe, control_image, transformer, args, device)

    scheduler = pipe.scheduler
    count = len(scheduler.timesteps)
    scheduler_sigmas = scheduler.sigmas[:count].float()
    index = int((scheduler_sigmas - float(noise_ratio)).abs().argmin().item())
    indices = torch.full((clean.shape[0],), index, device=device, dtype=torch.long)
    timesteps = scheduler.timesteps.to(device=device)[indices]
    sigma_values = scheduler.sigmas.to(device=device, dtype=clean.dtype)[indices]
    sigmas = sigma_values.view(-1, *([1] * (clean.ndim - 1)))
    noise = torch.randn_like(clean)
    noisy = (1.0 - sigmas) * clean + sigmas * noise
    target = noise - clean
    return noisy, control, timesteps, target, float(sigma_values[0].float().cpu())


def parallel_flow_loss(pipe, transformer, adapter, batch, args, device, noise_ratio):
    noisy, control, timesteps, target, actual_sigma = prepare_flow_batch(
        pipe, transformer, batch, args, device, noise_ratio
    )

    def base_forward():
        return core.output_tensor(
            core.transformer_forward(
                transformer,
                noisy,
                args,
                controlnet_image=control,
                timesteps=timesteps,
            )
        )

    prediction, active_blocks = adapter.forward_prediction(
        base_forward,
        noise_ratio=actual_sigma,
        enable_grad=True,
    )
    loss = F.mse_loss(prediction.float(), target.float())
    return loss, actual_sigma, active_blocks


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_id", default="acceptee/DiT4SR")
    parser.add_argument("--base_model_id", default="stabilityai/stable-diffusion-3.5-medium")
    parser.add_argument("--variant", default="dit4sr_q", choices=["dit4sr_q", "dit4sr_f", "dit4sr_r1"])
    parser.add_argument("--dit4sr_code_repo", default="acceptee/DiT4SR")
    parser.add_argument("--dit4sr_code_dir", default="")
    parser.add_argument("--transformer_subfolder", default="")
    parser.add_argument("--component_name", default="")
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--importance_csv",
        required=True,
        help="Official-flow All-LoRA importance CSV used to build the noise schedule.",
    )
    parser.add_argument(
        "--importance_train_step",
        type=int,
        default=0,
        help="Use one profiling step; -1 is a retrospective all-step oracle, not a fair main baseline.",
    )
    parser.add_argument("--active_topk", type=int, default=8)
    parser.add_argument("--anchor_count", type=int, default=2)
    parser.add_argument(
        "--side_dim",
        type=int,
        default=104,
        help="Side width; 104 approximately matches the existing 20.94 MB FP32 All-LoRA budget.",
    )
    parser.add_argument("--side_mlp_ratio", type=float, default=2.0)
    parser.add_argument(
        "--side_conv",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use a depthwise 3x3 token-grid convolution inside active adapters.",
    )
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--max_images", type=int, default=0)
    parser.add_argument("--sr_scale", type=float, default=4.0)
    parser.add_argument("--train_steps", type=int, default=20)
    parser.add_argument(
        "--train_noise_ratios",
        type=float,
        nargs="+",
        default=[],
        help="Discrete flow sigmas; defaults to all ratios present in the importance CSV.",
    )
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dtype", default="bf16", choices=["fp32", "bf16", "fp16"])
    parser.add_argument("--power_w", type=float, default=30.0)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--latent_channels", type=int, default=16)
    parser.add_argument("--latent_size", type=int, default=64)
    parser.add_argument("--caption_dim", type=int, default=4096)
    parser.add_argument("--prompt_seq_len", type=int, default=77)
    parser.add_argument("--timestep", type=int, default=500)
    parser.add_argument("--input_noise_std", type=float, default=0.0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.side_dim <= 0:
        raise SystemExit("--side_dim must be positive.")
    if args.train_steps <= 0:
        raise SystemExit("--train_steps must be positive.")
    if any(not 0.0 <= ratio <= 1.0 for ratio in args.train_noise_ratios):
        raise SystemExit("--train_noise_ratios values must be in [0, 1].")
    importance_path = Path(args.importance_csv)
    if not importance_path.is_file():
        raise SystemExit(f"Missing importance CSV: {importance_path}")

    args.loss_mode = "official_flow"
    args.load_mode = "transformer"
    args.model_impl = "official"
    args.pipeline_subfolder = ""
    core.set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    pipe = core.load_pipe(args, device)
    transformer = core.get_transformer(pipe, args.component_name)
    transformer.requires_grad_(False).eval()

    block_names = [name for name, _block in parallel.transformer_block_items(transformer)]
    anchors, schedule, score_table = parallel.build_noise_schedule(
        importance_path,
        block_names,
        active_topk=args.active_topk,
        anchor_count=args.anchor_count,
        importance_train_step=args.importance_train_step,
    )
    adapter = parallel.NoiseConditionedParallelAdapter(
        transformer,
        side_dim=args.side_dim,
        active_schedule=schedule,
        mlp_ratio=args.side_mlp_ratio,
        use_depthwise_conv=args.side_conv,
    ).to(device=device, dtype=torch.float32)
    adapter.train()
    trainable_params = sum(param.numel() for param in adapter.parameters())
    print(f"Parallel Adapter parameters: {trainable_params:,}")
    print("Anchor blocks: " + ", ".join(anchors))
    for noise_ratio, blocks in schedule.items():
        print(f"sigma={noise_ratio:.2f}: " + ", ".join(blocks))

    dataset = core.ImageFolderDataset(args.data_dir, args.image_size, args.max_images)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    iterator = iter(loader)
    optimizer = torch.optim.AdamW(
        adapter.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    train_noise_ratios = args.train_noise_ratios or sorted(schedule)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    started_training = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    try:
        for step in range(1, args.train_steps + 1):
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                batch = next(iterator)
            requested_sigma = float(random.choice(train_noise_ratios))
            optimizer.zero_grad(set_to_none=True)
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
                torch.cuda.synchronize(device)
            step_started = time.perf_counter()
            loss, actual_sigma, active_blocks = parallel_flow_loss(
                pipe,
                transformer,
                adapter,
                batch,
                args,
                device,
                requested_sigma,
            )
            if not torch.isfinite(loss):
                raise SystemExit(f"Non-finite loss at step {step}.")
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(adapter.parameters(), args.grad_clip)
            optimizer.step()
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - step_started
            peak_mb = (
                torch.cuda.max_memory_allocated(device) / (1024.0**2)
                if device.type == "cuda"
                else 0.0
            )
            rows.append(
                {
                    "step": step,
                    "loss": float(loss.detach().cpu()),
                    "grad_norm": float(grad_norm),
                    "requested_noise_ratio": requested_sigma,
                    "noise_ratio": actual_sigma,
                    "active_block_count": len(active_blocks),
                    "active_blocks": ";".join(active_blocks),
                    "train_step_time_s": elapsed,
                    "train_peak_cuda_mem_mb": peak_mb,
                }
            )
            if step == 1 or step % args.log_every == 0 or step == args.train_steps:
                print(
                    f"step {step:05d}/{args.train_steps} "
                    f"loss={float(loss.detach().cpu()):.6f} "
                    f"sigma={actual_sigma:.3f} active={len(active_blocks)} "
                    f"time={elapsed:.3f}s peak={peak_mb:.1f}MB"
                )
    finally:
        adapter._enable_grad = False

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    train_time_s = time.perf_counter() - started_training
    write_csv(output_dir / "train_log.csv", rows)
    checkpoint_summary = parallel.save_parallel_adapter(
        adapter,
        output_dir / "parallel_adapter.pt",
        anchors,
        score_table,
    )
    peak_cuda_mb = max((row["train_peak_cuda_mem_mb"] for row in rows), default=0.0)
    summary = {
        "train_time_s": train_time_s,
        "sec_per_step": train_time_s / args.train_steps,
        "estimated_energy_wh": args.power_w * train_time_s / 3600.0,
        "peak_cuda_mem_mb": peak_cuda_mb,
        "side_dim": args.side_dim,
        "active_topk": args.active_topk,
        "anchor_count": args.anchor_count,
        **checkpoint_summary,
    }
    write_csv(output_dir / "summary.csv", [summary])
    metadata = {
        **vars(args),
        "model": "DiT4SR-HF",
        "objective": "official_flow",
        "method": "noise_conditioned_grad_topk_parallel_adapter",
        "block_names": block_names,
        "anchors": anchors,
        "active_schedule": schedule,
        "score_table": score_table,
        **summary,
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    adapter.close()
    print(f"Wrote results to {output_dir}")


if __name__ == "__main__":
    main()
