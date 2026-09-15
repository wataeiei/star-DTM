#!/usr/bin/env python3
"""Profile per-block backward-bypass savings for a selected ResShift LoRA layout."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
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
    resolve_under_root,
    set_seed,
)
from train_resshift_lora_importance import (
    aggregate_importance,
    choose_blocks,
    read_csv,
)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def load_experiment(args):
    set_seed(args.seed)
    root = Path(args.resshift_root).resolve()
    config_path = resolve_under_root(root, args.config_path)
    checkpoint = resolve_under_root(root, args.checkpoint)
    autoencoder_path = resolve_under_root(root, args.autoencoder_checkpoint)
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
    missing, unexpected = load_state(autoencoder, autoencoder_path)
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
        name
        for name, module in model.named_modules()
        if isinstance(module, torch.nn.Linear)
        and name.endswith(".attn.qkv")
        and module.out_features == 3 * module.in_features
    ]
    all_blocks = sorted((block_from_qkv(name) for name in qkv_names), key=natural_key)
    importance = aggregate_importance(
        read_csv(Path(args.importance_csv)), args.importance_step, args.score_key
    )
    selected, report = choose_blocks(
        all_blocks,
        importance,
        args.lora_selection,
        args.importance_threshold,
        args.topk_blocks,
    )
    inject_packed_qkv_lora(model, args.rank, args.alpha, set(selected))
    model.train()

    frozen = [block for block in all_blocks if block not in set(selected)]
    controller = adaptive.ResidualBlockController(
        model,
        {block: block for block in frozen},
        cache_device="cpu",
        cache_dtype=torch.float32,
    )

    dataset = ImageFolderDataset(
        args.data_dir, args.image_size, args.sr_scale, args.max_images, args.seed
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    batch = next(iter(loader))
    gt = batch["gt"].to(device)
    lq = batch["lq"].to(device)
    timestep = noise_ratio_to_timestep(args.noise_ratio, diffusion.num_timesteps)
    tt = torch.full((1,), timestep, dtype=torch.long, device=device)
    latent_scale = 2 ** (len(config.autoencoder.params.ddconfig.ch_mult) - 1)
    latent_resolution = gt.shape[-1] // latent_scale
    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed + 99991)
    noise = torch.randn(
        (1, int(config.autoencoder.params.embed_dim), latent_resolution, latent_resolution),
        generator=generator,
        device=device,
        dtype=gt.dtype,
    )

    def run_step(block: str, bypass: bool) -> float:
        controller.configure([block])
        controller.set_mode("single_skip" if bypass else "full")
        model.zero_grad(set_to_none=True)
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
        loss.backward()
        sync(device)
        return float(loss.detach().cpu())

    return device, model, controller, all_blocks, selected, frozen, report, run_step


def profile_flops(run_step, block: str, bypass: bool, device: torch.device):
    activities = [torch.profiler.ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)
        torch.cuda.reset_peak_memory_stats(device)
    with torch.profiler.profile(activities=activities, with_flops=True) as profiler:
        loss = run_step(block, bypass)
    flops = sum(float(event.flops or 0) for event in profiler.key_averages())
    peak = (
        torch.cuda.max_memory_allocated(device) / (1024.0**2)
        if device.type == "cuda"
        else 0.0
    )
    return flops / 1e9, peak, loss


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resshift_root", default=".")
    parser.add_argument("--config_path", default="configs/realsr_swinunet_realesrgan256.yaml")
    parser.add_argument("--checkpoint", default="weights/resshift_realsrx4_s15_v1.pth")
    parser.add_argument("--autoencoder_checkpoint", default="weights/autoencoder_vq_f4.pth")
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--importance_csv", required=True)
    parser.add_argument("--importance_step", type=int, default=0)
    parser.add_argument("--score_key", default="normalized_grad_score")
    parser.add_argument("--lora_selection", choices=["threshold", "topk"], default="threshold")
    parser.add_argument("--importance_threshold", type=float, default=1.0)
    parser.add_argument("--topk_blocks", type=int, default=6)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--sr_scale", type=int, default=4)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=16.0)
    parser.add_argument("--noise_ratio", type=float, default=0.4)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=8)
    parser.add_argument("--max_images", type=int, default=1)
    parser.add_argument("--seed", type=int, default=4242)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    if args.warmup < 0 or args.repeats < 1:
        parser.error("warmup must be non-negative and repeats must be positive")
    try:
        (
            device,
            _model,
            _controller,
            all_blocks,
            selected,
            frozen,
            selection_report,
            run_step,
        ) = load_experiment(args)
    except (OSError, RuntimeError, ValueError, KeyError) as exc:
        parser.error(str(exc))

    rows = []
    for block_index, block in enumerate(all_blocks):
        if block not in frozen:
            continue
        for _ in range(args.warmup):
            run_step(block, False)
            run_step(block, True)

        full_ms, bypass_ms, full_losses, bypass_losses = [], [], [], []
        for repeat in range(args.repeats):
            modes = (False, True) if repeat % 2 == 0 else (True, False)
            for bypass in modes:
                sync(device)
                started = time.perf_counter()
                loss = run_step(block, bypass)
                elapsed = (time.perf_counter() - started) * 1000.0
                (bypass_ms if bypass else full_ms).append(elapsed)
                (bypass_losses if bypass else full_losses).append(loss)

        full_gflops, full_peak, full_loss = profile_flops(run_step, block, False, device)
        bypass_gflops, bypass_peak, bypass_loss = profile_flops(run_step, block, True, device)
        row = {
            "block": block,
            "block_index": block_index,
            "profile_noise_ratio": args.noise_ratio,
            "full_step_time_ms": statistics.mean(full_ms),
            "bypass_step_time_ms": statistics.mean(bypass_ms),
            "step_time_saved_ms": statistics.mean(full_ms) - statistics.mean(bypass_ms),
            "full_step_time_std_ms": statistics.stdev(full_ms) if len(full_ms) > 1 else 0.0,
            "bypass_step_time_std_ms": statistics.stdev(bypass_ms) if len(bypass_ms) > 1 else 0.0,
            "full_reported_gflops": full_gflops,
            "bypass_reported_gflops": bypass_gflops,
            "reported_gflops_saved": full_gflops - bypass_gflops,
            "full_peak_cuda_mem_mb": full_peak,
            "bypass_peak_cuda_mem_mb": bypass_peak,
            "peak_cuda_mem_saved_mb": full_peak - bypass_peak,
            "max_loss_abs_diff": max(
                [abs(a - b) for a, b in zip(full_losses, bypass_losses)]
                + [abs(full_loss - bypass_loss)]
            ),
        }
        rows.append(row)
        print(
            f"{block}: {row['reported_gflops_saved']:.3f} GFLOPs, "
            f"{row['step_time_saved_ms']:.3f} ms, "
            f"{row['peak_cuda_mem_saved_mb']:.1f} MB"
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "block_backward_costs.csv", rows)
    summary = {
        "model": "ResShift",
        "selected_lora_blocks": selected,
        "selected_lora_count": len(selected),
        "selection_report": selection_report,
        "total_blocks": len(all_blocks),
        "profiled_frozen_blocks": len(frozen),
        "profile_noise_ratio": args.noise_ratio,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "note": "Reported FLOPs include only operators supported by PyTorch profiler.",
    }
    (output_dir / "block_backward_costs_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(f"Wrote ResShift bypass costs to {output_dir}")


if __name__ == "__main__":
    main()
