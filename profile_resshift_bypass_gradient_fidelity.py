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


def load_block_costs(path: Path) -> dict[str, dict[str, float]]:
    costs = {}
    for row in read_csv(path):
        block = str(row["block"])
        gflops = float(row["reported_gflops_saved"])
        if not math.isfinite(gflops) or gflops <= 0:
            continue
        costs[block] = {
            "gflops": gflops,
            "milliseconds": float(row.get("step_time_saved_ms", 0.0)),
            "memory_mb": float(row.get("peak_cuda_mem_saved_mb", 0.0)),
        }
    if not costs:
        raise ValueError(f"No positive block costs were found in {path}")
    return costs


def importance_by_noise(rows, step: int, score_key: str):
    selected = [row for row in rows if int(row["train_step"]) == step]
    if not selected:
        available = sorted({int(row["train_step"]) for row in rows})
        raise ValueError(
            f"Importance step {step} is unavailable; available={available}"
        )
    groups = {}
    for row in selected:
        groups.setdefault(float(row["noise_ratio"]), {})[str(row["block"])] = float(
            row[score_key]
        )
    return groups


def build_nested_candidates(
    blocks: list[str],
    importance_scores: dict[str, float],
    costs: dict[str, dict[str, float]],
    minimum_count: int,
) -> list[dict]:
    missing_costs = set(blocks) - costs.keys()
    missing_scores = set(blocks) - importance_scores.keys()
    if missing_costs:
        raise ValueError(f"Missing costs for policy blocks: {sorted(missing_costs)}")
    if missing_scores:
        raise ValueError(
            f"Missing importance scores for policy blocks: {sorted(missing_scores)}"
        )
    total_importance = sum(importance_scores.values())
    if total_importance <= 0:
        raise ValueError("Total block importance must be positive")
    shares = {
        block: importance_scores[block] / total_importance
        for block in importance_scores
    }
    current = list(blocks)
    candidates = []
    while len(current) >= minimum_count:
        candidates.append({
            "skip_blocks": tuple(current),
            "bypass_budget": len(current),
            "bypass_importance_mass": sum(shares[block] for block in current),
            "effective_score_threshold": max(
                (shares[block] for block in current), default=0.0
            ),
            "estimated_saved_gflops": sum(costs[block]["gflops"] for block in current),
            "estimated_saved_time_ms": sum(
                costs[block]["milliseconds"] for block in current
            ),
            "estimated_saved_memory_mb": sum(
                costs[block]["memory_mb"] for block in current
            ),
        })
        if len(current) == minimum_count:
            break
        # Remove the block with the largest importance cost per saved GFLOP.
        # This preserves as much estimated compute benefit as possible while
        # progressively reducing the gradient-risk proxy.
        removed = max(
            current,
            key=lambda block: (
                shares[block] / costs[block]["gflops"],
                shares[block],
                -costs[block]["gflops"],
                block,
            ),
        )
        current.remove(removed)
    return candidates


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
    parser.add_argument("--auto_refine_policy", action="store_true")
    parser.add_argument("--cost_csv", default="")
    parser.add_argument("--min_bypass_count", type=int, default=0)
    parser.add_argument("--refined_policy_csv", default="")
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
    if args.min_bypass_count < 0:
        parser.error("min_bypass_count must be non-negative")
    if args.auto_refine_policy and not args.cost_csv:
        parser.error("--auto_refine_policy requires --cost_csv")

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
        importance_rows = read_csv(Path(args.importance_csv))
        importance = aggregate_importance(
            importance_rows, args.importance_step, args.score_key
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
        candidate_metadata = {}
        if args.auto_refine_policy:
            costs = load_block_costs(Path(args.cost_csv))
            importance_groups = importance_by_noise(
                importance_rows, args.importance_step, args.score_key
            )
        else:
            costs = None
            importance_groups = None

        for anchor_index, ratio in enumerate(sorted(set(args.noise_ratios))):
            policy_ratio = min(policy, key=lambda value: abs(value - ratio))
            initial_skip_blocks = policy[policy_ratio]
            if args.auto_refine_policy:
                importance_ratio = min(
                    importance_groups,
                    key=lambda value: abs(value - ratio),
                )
                candidates = build_nested_candidates(
                    initial_skip_blocks,
                    importance_groups[importance_ratio],
                    costs,
                    args.min_bypass_count,
                )
            else:
                candidates = [{
                    "skip_blocks": tuple(initial_skip_blocks),
                    "bypass_budget": len(initial_skip_blocks),
                    "bypass_importance_mass": float("nan"),
                    "effective_score_threshold": float("nan"),
                    "estimated_saved_gflops": float("nan"),
                    "estimated_saved_time_ms": float("nan"),
                    "estimated_saved_memory_mb": float("nan"),
                }]
            candidate_metadata[ratio] = candidates
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

                def run(mode: str, skip_blocks):
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
                full_loss, full_gradient = run("full", ())
                for candidate_index, candidate in enumerate(candidates):
                    skip_blocks = candidate["skip_blocks"]
                    adaptive.restore_rng(rng)
                    bypass_loss, bypass_gradient = run("single_skip", skip_blocks)
                    metrics = gradient_metrics(full_gradient, bypass_gradient)
                    loss_difference = abs(bypass_loss - full_loss)
                    detail_rows.append({
                        "noise_ratio": ratio,
                        "candidate_index": candidate_index,
                        "batch_index": batch_index,
                        "bypass_budget": len(skip_blocks),
                        "bypass_importance_mass": candidate["bypass_importance_mass"],
                        "effective_score_threshold": candidate["effective_score_threshold"],
                        "estimated_saved_gflops": candidate["estimated_saved_gflops"],
                        "skip_blocks": ";".join(skip_blocks),
                        "full_loss": full_loss,
                        "bypass_loss": bypass_loss,
                        "loss_abs_diff": loss_difference,
                        **metrics,
                    })
                    print(
                        f"noise={ratio:g} candidate={candidate_index} "
                        f"batch={batch_index} K={len(skip_blocks)} "
                        f"cos={metrics['gradient_cosine']:.6f} "
                        f"rel_err={metrics['relative_gradient_error']:.6f} "
                        f"loss_diff={loss_difference:.3g}"
                    )

        candidate_rows = []
        candidate_keys = sorted({
            (row["noise_ratio"], row["candidate_index"])
            for row in detail_rows
        })
        for ratio, candidate_index in candidate_keys:
            values = [
                row for row in detail_rows
                if row["noise_ratio"] == ratio
                and row["candidate_index"] == candidate_index
            ]
            row = {
                "noise_ratio": ratio,
                "candidate_index": candidate_index,
                "bypass_budget": values[0]["bypass_budget"],
                "bypass_importance_mass": values[0]["bypass_importance_mass"],
                "effective_score_threshold": values[0]["effective_score_threshold"],
                "estimated_saved_gflops": values[0]["estimated_saved_gflops"],
                "skip_blocks": values[0]["skip_blocks"],
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
            candidate_rows.append(row)

        selected_rows = []
        for ratio in sorted({row["noise_ratio"] for row in candidate_rows}):
            candidates = [
                row for row in candidate_rows
                if row["noise_ratio"] == ratio and row["safe"]
            ]
            if not candidates:
                if args.auto_refine_policy:
                    raise RuntimeError(
                        f"No fidelity-safe bypass candidate at noise={ratio:g}"
                    )
                candidates = [
                    row for row in candidate_rows
                    if row["noise_ratio"] == ratio
                ]
            if args.auto_refine_policy:
                selected_row = max(
                    candidates,
                    key=lambda row: (
                        row["estimated_saved_gflops"],
                        -row["bypass_importance_mass"],
                    ),
                )
            else:
                selected_row = candidates[0]
            selected_rows.append(dict(selected_row))

        overall = {
            "noise_ratio": "mean",
            "candidate_index": "selected",
            "bypass_budget": mean([float(row["bypass_budget"]) for row in selected_rows]),
            "bypass_importance_mass": mean([
                float(row["bypass_importance_mass"])
                for row in selected_rows
                if math.isfinite(float(row["bypass_importance_mass"]))
            ]),
            "effective_score_threshold": mean([
                float(row["effective_score_threshold"])
                for row in selected_rows
                if math.isfinite(float(row["effective_score_threshold"]))
            ]),
            "estimated_saved_gflops": mean([
                float(row["estimated_saved_gflops"])
                for row in selected_rows
                if math.isfinite(float(row["estimated_saved_gflops"]))
            ]),
            "skip_blocks": "",
            "num_batches": sum(int(row["num_batches"]) for row in selected_rows),
            "mean_gradient_cosine": mean([row["mean_gradient_cosine"] for row in selected_rows]),
            "std_gradient_cosine": stdev([row["mean_gradient_cosine"] for row in selected_rows]),
            "mean_relative_gradient_error": mean([row["mean_relative_gradient_error"] for row in selected_rows]),
            "std_relative_gradient_error": stdev([row["mean_relative_gradient_error"] for row in selected_rows]),
            "mean_gradient_norm_ratio": mean([row["mean_gradient_norm_ratio"] for row in selected_rows]),
            "max_loss_abs_diff": max(row["max_loss_abs_diff"] for row in selected_rows),
        }
        overall["safe"] = bool(all(row["safe"] for row in selected_rows))
        summary_rows = selected_rows + [overall]

        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        write_csv(output_dir / "gradient_fidelity_details.csv", detail_rows)
        write_csv(output_dir / "gradient_fidelity_candidates.csv", candidate_rows)
        write_csv(output_dir / "gradient_fidelity_summary.csv", summary_rows)
        refined_policy_path = None
        if args.auto_refine_policy:
            refined_policy_path = Path(args.refined_policy_csv) if args.refined_policy_csv else (
                output_dir / "refined_bypass_policy_by_noise.csv"
            )
            refined_policy_rows = []
            for row in selected_rows:
                metadata = candidate_metadata[float(row["noise_ratio"])][
                    int(row["candidate_index"])
                ]
                refined_policy_rows.append({
                    "noise_ratio": row["noise_ratio"],
                    "bypass_budget": row["bypass_budget"],
                    "bypass_importance_mass": row["bypass_importance_mass"],
                    "effective_score_threshold": row["effective_score_threshold"],
                    "estimated_saved_gflops": row["estimated_saved_gflops"],
                    "estimated_saved_time_ms": metadata["estimated_saved_time_ms"],
                    "estimated_saved_memory_mb": metadata["estimated_saved_memory_mb"],
                    "mean_gradient_cosine": row["mean_gradient_cosine"],
                    "mean_relative_gradient_error": row["mean_relative_gradient_error"],
                    "mean_gradient_norm_ratio": row["mean_gradient_norm_ratio"],
                    "max_loss_abs_diff": row["max_loss_abs_diff"],
                    "skip_blocks": row["skip_blocks"],
                })
            write_csv(refined_policy_path, refined_policy_rows)
        report = {
            "model": "ResShift",
            "selected_lora_blocks": selected,
            "selected_lora_count": len(selected),
            "selected_lora_utility": selection_report["selected_utility"],
            "policy_csv": str(args.policy_csv),
            "auto_refine_policy": args.auto_refine_policy,
            "cost_csv": args.cost_csv or None,
            "refined_policy_csv": str(refined_policy_path) if refined_policy_path else None,
            "policy_union": sorted(policy_union, key=all_blocks.index),
            "probe_batches_per_noise": len(batches),
            "min_cosine": args.min_cosine,
            "max_relative_error": args.max_relative_error,
            "max_loss_abs_diff": args.max_loss_abs_diff,
            "all_noise_anchors_safe": overall["safe"],
            "mean_selected_bypass_budget": overall["bypass_budget"],
            "mean_selected_saved_gflops": overall["estimated_saved_gflops"],
        }
        (output_dir / "gradient_fidelity_report.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )
        print(json.dumps(report, indent=2))
    except (OSError, RuntimeError, ValueError, KeyError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
