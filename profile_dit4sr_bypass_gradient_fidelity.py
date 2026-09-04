#!/usr/bin/env python3
"""Measure full-vs-bypassed LoRA gradient fidelity across noise and budgets."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader

import adaptive_grad_blockskip as adaptive
import profile_hf_dit4sr_grad as core
import train_hf_dit4sr_all_lora_importance as train_core


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_id", default="acceptee/DiT4SR")
    parser.add_argument(
        "--base_model_id", default="stabilityai/stable-diffusion-3.5-medium"
    )
    parser.add_argument("--variant", default="dit4sr_q")
    parser.add_argument("--dit4sr_code_repo", default="acceptee/DiT4SR")
    parser.add_argument("--dit4sr_code_dir", default="")
    parser.add_argument("--transformer_subfolder", default="")
    parser.add_argument("--pipeline_subfolder", default="")
    parser.add_argument("--component_name", default="")
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--selection_file", required=True)
    parser.add_argument("--importance_csv", required=True)
    parser.add_argument("--importance_step", type=int, default=0)
    parser.add_argument("--adapter", default="")
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--max_images", type=int, default=0)
    parser.add_argument("--probe_batches", type=int, default=5)
    parser.add_argument(
        "--noise_ratios",
        type=float,
        nargs="+",
        default=[0.05, 0.2, 0.4, 0.6, 0.8, 0.95],
    )
    parser.add_argument("--bypass_budgets", type=int, nargs="+", default=[0, 2, 4, 6, 8])
    parser.add_argument("--blockskip_min_run", type=int, default=2)
    parser.add_argument("--blockskip_max_run", type=int, default=4)
    parser.add_argument("--blockskip_max_runs", type=int, default=2)
    parser.add_argument("--min_cosine", type=float, default=0.95)
    parser.add_argument("--max_relative_error", type=float, default=0.10)
    parser.add_argument("--target", default="qv")
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--alpha", type=int, default=16)
    parser.add_argument("--block_regex", default="")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval_seed", type=int, default=4242)
    parser.add_argument("--dtype", default="bf16", choices=["fp32", "bf16", "fp16"])
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--sr_scale", type=float, default=4.0)
    parser.add_argument("--latent_channels", type=int, default=16)
    parser.add_argument("--latent_size", type=int, default=64)
    parser.add_argument("--caption_dim", type=int, default=4096)
    parser.add_argument("--prompt_seq_len", type=int, default=77)
    parser.add_argument("--timestep", type=int, default=500)
    parser.add_argument("--input_noise_std", type=float, default=0.0)
    args = parser.parse_args()
    args.loss_mode = "official_flow"
    args.load_mode = "transformer"
    args.model_impl = "official"
    return args


def selected_blocks(path: str | Path) -> list[str]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    selected = payload.get("selected_blocks", payload.get("selected_lora_blocks"))
    if not isinstance(selected, list) or not selected:
        raise SystemExit(
            f"{path} must contain selected_blocks or selected_lora_blocks."
        )
    return selected


def flatten_gradients(parameters: list[torch.nn.Parameter]) -> torch.Tensor:
    values = []
    for parameter in parameters:
        if parameter.grad is None:
            values.append(torch.zeros_like(parameter, dtype=torch.float32).flatten())
        else:
            values.append(parameter.grad.detach().float().flatten())
    return torch.cat(values)


def gradient_metrics(reference: torch.Tensor, candidate: torch.Tensor) -> dict:
    reference_norm = torch.linalg.vector_norm(reference)
    candidate_norm = torch.linalg.vector_norm(candidate)
    difference_norm = torch.linalg.vector_norm(candidate - reference)
    denominator = reference_norm * candidate_norm
    cosine = (
        float(torch.dot(reference, candidate) / denominator)
        if float(denominator) > 0.0
        else float("nan")
    )
    return {
        "gradient_cosine": cosine,
        "relative_gradient_error": float(difference_norm / reference_norm)
        if float(reference_norm) > 0.0
        else float("nan"),
        "gradient_norm_ratio": float(candidate_norm / reference_norm)
        if float(reference_norm) > 0.0
        else float("nan"),
        "full_gradient_norm": float(reference_norm),
        "bypass_gradient_norm": float(candidate_norm),
    }


def backward_gradient(
    pipe,
    transformer,
    controller,
    parameters,
    batch,
    args,
    device,
    mode,
    skip_blocks,
    draw_seed,
):
    controller.configure(skip_blocks)
    controller.set_mode(mode)
    transformer.zero_grad(set_to_none=True)
    core.set_seed(draw_seed)
    loss = core.flow_matching_batch_loss(pipe, transformer, batch, args, device)
    if not torch.isfinite(loss):
        raise RuntimeError("Encountered non-finite calibration loss.")
    loss.backward()
    gradient = flatten_gradients(parameters)
    stats = controller.stats(0.0)
    return float(loss.detach().cpu()), gradient, stats


def aggregate_rows(rows: list[dict], args: argparse.Namespace) -> tuple[list[dict], list[dict]]:
    grouped: dict[tuple[float, int], list[dict]] = {}
    for row in rows:
        grouped.setdefault((row["noise_ratio"], row["bypass_budget"]), []).append(row)
    summary = []
    for (noise_ratio, budget), values in sorted(grouped.items()):
        feasible = [row for row in values if row["status"] == "ok"]
        result = {
            "noise_ratio": noise_ratio,
            "bypass_budget": budget,
            "num_batches": len(values),
            "num_feasible_batches": len(feasible),
            "all_feasible": len(feasible) == len(values),
        }
        for key in (
            "gradient_cosine",
            "relative_gradient_error",
            "gradient_norm_ratio",
            "loss_abs_diff",
            "fallback_blocks",
        ):
            numbers = [float(row[key]) for row in feasible if math.isfinite(float(row[key]))]
            result[f"mean_{key}"] = sum(numbers) / len(numbers) if numbers else float("nan")
            if len(numbers) > 1:
                mean = result[f"mean_{key}"]
                result[f"std_{key}"] = math.sqrt(
                    sum((number - mean) ** 2 for number in numbers) / (len(numbers) - 1)
                )
            else:
                result[f"std_{key}"] = 0.0 if numbers else float("nan")
        result["safe"] = bool(
            result["all_feasible"]
            and result["mean_fallback_blocks"] == 0
            and result["mean_gradient_cosine"] >= args.min_cosine
            and result["mean_relative_gradient_error"] <= args.max_relative_error
        )
        summary.append(result)

    recommendations = []
    for noise_ratio in sorted({row["noise_ratio"] for row in summary}):
        candidates = [
            row for row in summary if row["noise_ratio"] == noise_ratio and row["safe"]
        ]
        chosen = max(candidates, key=lambda row: row["bypass_budget"]) if candidates else None
        recommendations.append(
            {
                "noise_ratio": noise_ratio,
                "recommended_bypass_budget": chosen["bypass_budget"] if chosen else 0,
                "mean_gradient_cosine": chosen["mean_gradient_cosine"] if chosen else float("nan"),
                "mean_relative_gradient_error": chosen["mean_relative_gradient_error"] if chosen else float("nan"),
                "min_cosine_threshold": args.min_cosine,
                "max_relative_error_threshold": args.max_relative_error,
            }
        )
    return summary, recommendations


def plot_summary(summary: list[dict], output_dir: Path, args: argparse.Namespace) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib unavailable; skipped plots.")
        return
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))
    noise_ratios = sorted({row["noise_ratio"] for row in summary})
    for noise_ratio in noise_ratios:
        values = sorted(
            [row for row in summary if row["noise_ratio"] == noise_ratio],
            key=lambda row: row["bypass_budget"],
        )
        budgets = [row["bypass_budget"] for row in values]
        axes[0].plot(
            budgets,
            [row["mean_gradient_cosine"] for row in values],
            marker="o",
            label=f"sigma={noise_ratio:g}",
        )
        axes[1].plot(
            budgets,
            [row["mean_relative_gradient_error"] for row in values],
            marker="o",
            label=f"sigma={noise_ratio:g}",
        )
    axes[0].set_xlabel("Bypass budget K")
    axes[0].set_ylabel("Gradient cosine similarity")
    axes[0].set_ylim(-0.05, 1.02)
    axes[0].axhline(
        args.min_cosine, color="#555555", linestyle="--", linewidth=1.0,
        label=f"threshold={args.min_cosine:g}",
    )
    axes[1].set_xlabel("Bypass budget K")
    axes[1].set_ylabel("Relative gradient error")
    axes[1].axhline(
        args.max_relative_error, color="#555555", linestyle="--", linewidth=1.0,
        label=f"threshold={args.max_relative_error:g}",
    )
    for axis in axes:
        axis.grid(alpha=0.25)
    axes[0].legend(frameon=False, fontsize=8, ncol=2)
    axes[1].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "bypass_gradient_fidelity.png", dpi=300)
    fig.savefig(output_dir / "bypass_gradient_fidelity.pdf")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.probe_batches < 1:
        raise SystemExit("--probe_batches must be positive.")
    if any(not 0.0 <= ratio <= 1.0 for ratio in args.noise_ratios):
        raise SystemExit("--noise_ratios values must be in [0, 1].")
    if any(budget < 0 for budget in args.bypass_budgets):
        raise SystemExit("--bypass_budgets values must be non-negative.")

    core.set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    pipe = core.load_pipe(args, device)
    transformer = core.get_transformer(pipe, args.component_name)
    transformer.requires_grad_(False)
    target_module_names = [
        name
        for name, module in transformer.named_modules()
        if isinstance(module, torch.nn.Linear)
        and core.block_key(name, args.block_regex)
        and core.target_match(name, args.target)
    ]
    candidates = train_core.candidate_lora_blocks(
        transformer, args.target, args.block_regex
    )
    protected = selected_blocks(args.selection_file)
    unknown = sorted(set(protected) - set(candidates), key=core.natural_key)
    if unknown:
        raise SystemExit("Selection file contains unknown blocks: " + ", ".join(unknown))
    injected = core.inject_lora(
        transformer,
        args.target,
        args.rank,
        args.alpha,
        args.block_regex,
        selected_blocks=set(protected),
    )
    if args.adapter:
        report = adaptive.load_lora_adapter(transformer, args.adapter)
        if report["missing"]:
            raise SystemExit(f"Adapter has unknown modules: {report['missing'][:5]}")
    core.set_seed(args.seed)
    transformer.train()
    parameters = [parameter for parameter in transformer.parameters() if parameter.requires_grad]
    if not parameters:
        raise SystemExit("No trainable LoRA parameters found.")

    importance_rows = train_core.read_blockskip_importance(
        args.importance_csv, args.importance_step
    )
    first_ratio = min(float(row["noise_ratio"]) for row in importance_rows)
    block_names = [
        row["block"]
        for row in sorted(importance_rows, key=lambda row: int(row["block_index"]))
        if math.isclose(float(row["noise_ratio"]), first_ratio, abs_tol=1e-8)
    ]
    block_paths = adaptive.infer_block_module_paths(
        target_module_names,
        block_names,
        core.block_key,
        args.block_regex,
    )
    controller = adaptive.ResidualBlockController(
        transformer, block_paths, cache_device="cpu", cache_dtype=torch.float32
    )

    dataset = core.ImageFolderDataset(args.data_dir, args.image_size, args.max_images)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    batches = []
    iterator = iter(loader)
    for _ in range(args.probe_batches):
        try:
            batches.append(next(iterator))
        except StopIteration:
            iterator = iter(loader)
            batches.append(next(iterator))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    budgets = sorted(set(args.bypass_budgets) | {0})
    for ratio_index, noise_ratio in enumerate(args.noise_ratios):
        args._profile_noise_ratio = float(noise_ratio)
        for batch_index, batch in enumerate(batches):
            draw_seed = args.eval_seed + ratio_index * 10000 + batch_index
            full_loss, full_gradient, _ = backward_gradient(
                pipe,
                transformer,
                controller,
                parameters,
                batch,
                args,
                device,
                "full",
                [],
                draw_seed,
            )
            for budget in budgets:
                if budget == 0:
                    skip_blocks = []
                else:
                    try:
                        skip_blocks = adaptive.select_low_score_runs(
                            importance_rows,
                            args.importance_step,
                            noise_ratio,
                            budget,
                            args.blockskip_min_run,
                            args.blockskip_max_run,
                            args.blockskip_max_runs,
                            excluded_blocks=protected,
                        )
                    except ValueError as exc:
                        rows.append(
                            {
                                "noise_ratio": noise_ratio,
                                "batch_index": batch_index,
                                "bypass_budget": budget,
                                "actual_bypassed_blocks": 0,
                                "bypassed_blocks": "",
                                "status": f"infeasible: {exc}",
                                "full_loss": full_loss,
                                "bypass_loss": float("nan"),
                                "loss_abs_diff": float("nan"),
                                "gradient_cosine": float("nan"),
                                "relative_gradient_error": float("nan"),
                                "gradient_norm_ratio": float("nan"),
                                "full_gradient_norm": float(torch.linalg.vector_norm(full_gradient)),
                                "bypass_gradient_norm": float("nan"),
                                "fallback_blocks": 0,
                            }
                        )
                        continue
                mode = "full" if budget == 0 else "single_skip"
                bypass_loss, bypass_gradient, stats = backward_gradient(
                    pipe,
                    transformer,
                    controller,
                    parameters,
                    batch,
                    args,
                    device,
                    mode,
                    skip_blocks,
                    draw_seed,
                )
                metrics = gradient_metrics(full_gradient, bypass_gradient)
                rows.append(
                    {
                        "noise_ratio": noise_ratio,
                        "batch_index": batch_index,
                        "bypass_budget": budget,
                        "actual_bypassed_blocks": len(skip_blocks),
                        "bypassed_blocks": ";".join(skip_blocks),
                        "status": "ok",
                        "full_loss": full_loss,
                        "bypass_loss": bypass_loss,
                        "loss_abs_diff": abs(bypass_loss - full_loss),
                        **metrics,
                        "fallback_blocks": stats.fallback_blocks,
                    }
                )
                print(
                    f"sigma={noise_ratio:.2f} batch={batch_index + 1}/{len(batches)} "
                    f"K={budget} cos={metrics['gradient_cosine']:.6f} "
                    f"rel_err={metrics['relative_gradient_error']:.6f}"
                )
        del args._profile_noise_ratio

    summary, recommendations = aggregate_rows(rows, args)
    write_csv(output_dir / "gradient_fidelity_per_batch.csv", rows)
    write_csv(output_dir / "gradient_fidelity_summary.csv", summary)
    write_csv(output_dir / "recommended_bypass_schedule.csv", recommendations)
    schedule = " ".join(
        f"{row['noise_ratio']:g}:{row['recommended_bypass_budget']}"
        for row in recommendations
    )
    (output_dir / "recommended_bypass_schedule.txt").write_text(
        schedule + "\n", encoding="utf-8"
    )
    metadata = {
        "model_id": args.model_id,
        "selection_file": args.selection_file,
        "importance_csv": args.importance_csv,
        "importance_step": args.importance_step,
        "adapter": args.adapter,
        "protected_lora_blocks": protected,
        "injected_lora_modules": len(injected),
        "noise_ratios": args.noise_ratios,
        "bypass_budgets": budgets,
        "probe_batches": args.probe_batches,
        "min_cosine": args.min_cosine,
        "max_relative_error": args.max_relative_error,
        "blockskip_min_run": args.blockskip_min_run,
        "blockskip_max_run": args.blockskip_max_run,
        "blockskip_max_runs": args.blockskip_max_runs,
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    plot_summary(summary, output_dir, args)
    print("Recommended --blockskip_schedule " + schedule)
    print(f"Wrote gradient-fidelity audit to {output_dir}")


if __name__ == "__main__":
    main()
