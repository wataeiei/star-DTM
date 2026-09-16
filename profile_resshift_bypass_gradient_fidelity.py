#!/usr/bin/env python3
"""Audit ResShift LoRA-gradient fidelity under a noise-conditioned bypass policy."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
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
    read_blockskip_policy,
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


def mean(values: list[float]) -> float:
    return statistics.mean(values) if values else float("nan")


def stdev(values: list[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else 0.0


def flatten_gradients(parameters: list[torch.nn.Parameter]) -> torch.Tensor:
    values = []
    for parameter in parameters:
        if parameter.grad is None:
            values.append(torch.zeros_like(parameter, dtype=torch.float32).reshape(-1))
        else:
            values.append(parameter.grad.detach().float().reshape(-1))
    return torch.cat(values)


def gradient_metrics(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, float]:
    reference_norm = float(reference.norm().cpu())
    candidate_norm = float(candidate.norm().cpu())
    difference_norm = float((candidate - reference).norm().cpu())
    if reference_norm == 0 or candidate_norm == 0:
        cosine = 1.0 if reference_norm == candidate_norm == 0 else 0.0
    else:
        cosine = float(
            torch.nn.functional.cosine_similarity(
                reference.unsqueeze(0), candidate.unsqueeze(0), dim=1
            ).cpu()
        )
    return {
        "gradient_cosine": cosine,
        "relative_gradient_error": (
            difference_norm / reference_norm if reference_norm else float("inf")
        ),
        "gradient_norm_ratio": (
            candidate_norm / reference_norm if reference_norm else float("inf")
        ),
        "reference_gradient_norm": reference_norm,
        "bypass_gradient_norm": candidate_norm,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resshift_root", default=".")
    parser.add_argument("--config_path", default="configs/realsr_swinunet_realesrgan256.yaml")
    parser.add_argument("--checkpoint", default="weights/resshift_realsrx4_s15_v1.pth")
    parser.add_argument("--autoencoder_checkpoint", default="weights/autoencoder_vq_f4.pth")
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--importance_csv", required=True)
    parser.add_argument("--importance_step", type=int, default=0)
    parser.add_argument("--score_key", default="normalized_grad_score")
    parser.add_argument("--importance_threshold", type=float, default=1.0)
    parser.add_argument("--policy_csv", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--sr_scale", type=int, default=4)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=16.0)
    parser.add_argument("--noise_ratios", type=float, nargs="+", default=[0.05, 0.2, 0.4, 0.6, 0.8, 0.95])
    parser.add_argument("--probe_batches", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_images", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--min_cosine", type=float, default=0.90)
    parser.add_argument("--max_relative_error", type=float, default=0.50)
    parser.add_argument("--max_loss_abs_diff", type=float, default=1e-7)
    parser.add_argument("--seed", type=int, default=4242)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    if args.probe_batches < 1 or args.batch_size < 1:
        parser.error("probe_batches and batch_size must be positive")
    if not -1 <= args.min_cosine <= 1:
        parser.error("min_cosine must be in [-1, 1]")
    if args.max_relative_error < 0 or args.max_loss_abs_diff < 0:
        parser.error("error thresholds must be non-negative")

    set_seed(args.seed)
    root = Path(args.resshift_root).resolve()
    config_path = resolve_under_root(root, args.config_path)
    checkpoint = resolve_under_root(root, args.checkpoint)
    autoencoder_path = resolve_under_root(root, args.autoencoder_checkpoint)
    sys.path.insert(0, str(root))

    try:
        from omegaconf import OmegaConf
        from utils.util_common import get_obj_from_str

        install_timm_layers_stub(torch)
        config = OmegaConf.load(config_path)
        device = torch.device(
            "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"
        )

        model = get_obj_from_str(config.model.target)(**config.model.get("params", {}))
        missing, unexpected = load_checkpoint(model, checkpoint, torch)
        if missing or unexpected:
            raise RuntimeError(
                f"Model checkpoint mismatch: missing={missing[:10]} "
                f"unexpected={unexpected[:10]}"
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
        all_blocks = sorted(
            (block_from_qkv(name) for name in qkv_names), key=natural_key
        )
        importance = aggregate_importance(
            read_csv(Path(args.importance_csv)), args.importance_step, args.score_key
        )
        selected, selection_report = choose_blocks(
            all_blocks,
            importance,
            "threshold",
            args.importance_threshold,
            1,
        )
        injected = inject_packed_qkv_lora(
            model, args.rank, args.alpha, selected_blocks=set(selected)
        )
        trainable = [parameter for module in injected.values() for parameter in (
            module.lora_down.weight, module.lora_up.weight
        )]
        model.train()

        policy = read_blockskip_policy(Path(args.policy_csv))
        policy_union = {block for blocks in policy.values() for block in blocks}
        unknown = policy_union - set(all_blocks)
        overlap = policy_union & set(selected)
        if unknown:
            raise ValueError(f"Policy contains unknown blocks: {sorted(unknown)}")
        if overlap:
            raise ValueError(f"Policy bypasses selected LoRA blocks: {sorted(overlap)}")
        controller = adaptive.ResidualBlockController(
            model,
            {block: block for block in all_blocks if block in policy_union},
            cache_device="cpu",
            cache_dtype=torch.float32,
        )

        dataset = ImageFolderDataset(
            args.data_dir,
            args.image_size,
            args.sr_scale,
            args.max_images,
            args.seed,
        )
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
            if len(batches) >= args.probe_batches:
                break
        if not batches:
            raise RuntimeError("No calibration batches were loaded")

        latent_scale = 2 ** (len(config.autoencoder.params.ddconfig.ch_mult) - 1)
        latent_channels = int(config.autoencoder.params.embed_dim)
        detail_rows = []

        for anchor_index, ratio in enumerate(sorted(set(args.noise_ratios))):
            policy_ratio = min(policy, key=lambda value: abs(value - ratio))
            skip_blocks = policy[policy_ratio]
            timestep = noise_ratio_to_timestep(ratio, diffusion.num_timesteps)
            for batch_index, batch in enumerate(batches):
                gt = batch["gt"].to(device, non_blocking=True)
                lq = batch["lq"].to(device, non_blocking=True)
                tt = torch.full(
                    (gt.shape[0],), timestep, dtype=torch.long, device=device
                )
                latent_resolution = gt.shape[-1] // latent_scale
                generator = torch.Generator(device=device)
                generator.manual_seed(
                    args.seed + anchor_index * 100000 + batch_index
                )
                noise = torch.randn(
                    (
                        gt.shape[0],
                        latent_channels,
                        latent_resolution,
                        latent_resolution,
                    ),
                    generator=generator,
                    device=device,
                    dtype=gt.dtype,
                )

                def run(mode: str):
                    model.zero_grad(set_to_none=True)
                    controller.configure(skip_blocks)
                    controller.set_mode(mode)
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
                    if device.type == "cuda":
                        torch.cuda.synchronize(device)
                    return float(loss.detach().cpu()), flatten_gradients(trainable)

                rng = adaptive.snapshot_rng(device)
                adaptive.restore_rng(rng)
                full_loss, full_gradient = run("full")
                adaptive.restore_rng(rng)
                bypass_loss, bypass_gradient = run("single_skip")
                metrics = gradient_metrics(full_gradient, bypass_gradient)
                loss_difference = abs(bypass_loss - full_loss)
                detail_rows.append({
                    "noise_ratio": ratio,
                    "batch_index": batch_index,
                    "bypass_budget": len(skip_blocks),
                    "skip_blocks": ";".join(skip_blocks),
                    "full_loss": full_loss,
                    "bypass_loss": bypass_loss,
                    "loss_abs_diff": loss_difference,
                    **metrics,
                })
                print(
                    f"noise={ratio:g} batch={batch_index} K={len(skip_blocks)} "
                    f"cos={metrics['gradient_cosine']:.6f} "
                    f"rel_err={metrics['relative_gradient_error']:.6f} "
                    f"loss_diff={loss_difference:.3g}"
                )

        summary_rows = []
        for ratio in sorted({row["noise_ratio"] for row in detail_rows}):
            values = [row for row in detail_rows if row["noise_ratio"] == ratio]
            row = {
                "noise_ratio": ratio,
                "bypass_budget": values[0]["bypass_budget"],
                "num_batches": len(values),
                "mean_gradient_cosine": mean([v["gradient_cosine"] for v in values]),
                "std_gradient_cosine": stdev([v["gradient_cosine"] for v in values]),
                "mean_relative_gradient_error": mean([v["relative_gradient_error"] for v in values]),
                "std_relative_gradient_error": stdev([v["relative_gradient_error"] for v in values]),
                "mean_gradient_norm_ratio": mean([v["gradient_norm_ratio"] for v in values]),
                "max_loss_abs_diff": max(v["loss_abs_diff"] for v in values),
            }
            row["safe"] = bool(
                row["mean_gradient_cosine"] >= args.min_cosine
                and row["mean_relative_gradient_error"] <= args.max_relative_error
                and row["max_loss_abs_diff"] <= args.max_loss_abs_diff
            )
            summary_rows.append(row)

        overall = {
            "noise_ratio": "mean",
            "bypass_budget": mean([float(row["bypass_budget"]) for row in detail_rows]),
            "num_batches": len(detail_rows),
            "mean_gradient_cosine": mean([row["gradient_cosine"] for row in detail_rows]),
            "std_gradient_cosine": stdev([row["gradient_cosine"] for row in detail_rows]),
            "mean_relative_gradient_error": mean([row["relative_gradient_error"] for row in detail_rows]),
            "std_relative_gradient_error": stdev([row["relative_gradient_error"] for row in detail_rows]),
            "mean_gradient_norm_ratio": mean([row["gradient_norm_ratio"] for row in detail_rows]),
            "max_loss_abs_diff": max(row["loss_abs_diff"] for row in detail_rows),
        }
        overall["safe"] = bool(all(row["safe"] for row in summary_rows))
        summary_rows.append(overall)

        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        write_csv(output_dir / "gradient_fidelity_details.csv", detail_rows)
        write_csv(output_dir / "gradient_fidelity_summary.csv", summary_rows)
        report = {
            "model": "ResShift",
            "selected_lora_blocks": selected,
            "selected_lora_count": len(selected),
            "selected_lora_utility": selection_report["selected_utility"],
            "policy_csv": str(args.policy_csv),
            "policy_union": sorted(policy_union, key=all_blocks.index),
            "probe_batches_per_noise": len(batches),
            "min_cosine": args.min_cosine,
            "max_relative_error": args.max_relative_error,
            "max_loss_abs_diff": args.max_loss_abs_diff,
            "all_noise_anchors_safe": overall["safe"],
        }
        (output_dir / "gradient_fidelity_report.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )
        print(json.dumps(report, indent=2))
    except (OSError, RuntimeError, ValueError, KeyError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
