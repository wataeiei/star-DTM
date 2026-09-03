#!/usr/bin/env python3
"""Prepare a controlled single-block LoRA gradient-utility experiment."""

import argparse
import csv
import json
import shlex
from collections import defaultdict
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--importance_csv", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--train_data_dir", default="data/ucmerced/train_hr")
    parser.add_argument("--val_data_dir", default="data/ucmerced/val_hr")
    parser.add_argument("--exclude_image", action="append", default=["val_0418.png"])
    parser.add_argument("--importance_step", type=int, default=0)
    parser.add_argument("--group_size", type=int, default=4)
    parser.add_argument("--train_steps", type=int, default=100)
    parser.add_argument("--eval_batches", type=int, default=20)
    parser.add_argument("--model_id", default="acceptee/DiT4SR")
    parser.add_argument("--base_model_id", default="stabilityai/stable-diffusion-3.5-medium")
    parser.add_argument("--variant", default="dit4sr_q")
    parser.add_argument("--target", default="qv")
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--alpha", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--profile_seed", type=int, default=42)
    parser.add_argument("--eval_seed", type=int, default=4242)
    parser.add_argument("--dtype", default="bf16")
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--sr_scale", type=float, default=4.0)
    parser.add_argument(
        "--noise_ratios",
        type=float,
        nargs="+",
        default=[0.05, 0.2, 0.4, 0.6, 0.8, 0.95],
    )
    return parser.parse_args()


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def shell_command(parts):
    return " \\\n  ".join(shlex.quote(str(part)) for part in parts)


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    metadata_dir = output_dir / "selection_metadata"
    runs_dir = output_dir / "runs"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    runs_dir.mkdir(parents=True, exist_ok=True)

    scores = defaultdict(list)
    params = defaultdict(list)
    modules = defaultdict(list)
    indices = defaultdict(list)
    with Path(args.importance_csv).open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if int(float(row["train_step"])) != args.importance_step:
                continue
            block = row["block"]
            scores[block].append(float(row["normalized_grad_score"]))
            params[block].append(int(float(row.get("lora_param_count", 0) or 0)))
            modules[block].append(int(float(row.get("module_count", 0) or 0)))
            if row.get("block_index", "") != "":
                indices[block].append(int(float(row["block_index"])))
    if not scores:
        raise SystemExit(
            f"No rows found for train_step={args.importance_step} in {args.importance_csv}"
        )

    ranking = []
    for block, values in scores.items():
        ranking.append(
            {
                "block": block,
                "block_index": max(indices[block]) if indices[block] else "",
                "gradient_importance": sum(values) / len(values),
                "num_noise_anchors": len(values),
                "lora_param_count": max(params[block]),
                "module_count": max(modules[block]),
            }
        )
    ranking.sort(key=lambda row: (-row["gradient_importance"], row["block"]))
    for rank, row in enumerate(ranking, 1):
        row["importance_rank"] = rank
    write_csv(output_dir / "all_block_ranking.csv", ranking)

    size = args.group_size
    if size < 1 or len(ranking) < 3 * size:
        raise SystemExit("Require at least three non-overlapping groups of --group_size blocks.")
    middle_start = (len(ranking) - size) // 2
    selections = [
        ("Top", ranking[:size]),
        ("Middle", ranking[middle_start:middle_start + size]),
        ("Bottom", ranking[-size:]),
    ]

    manifest = []
    training_commands = []
    validation_parts = [
        "python3", "eval_hf_dit4sr_lora_loss.py",
        "--model_id", args.model_id,
        "--base_model_id", args.base_model_id,
        "--variant", args.variant,
        "--data_dir", args.val_data_dir,
        "--output_dir", output_dir / "validation",
        "--image_size", args.image_size,
        "--max_images", 0,
        "--eval_batches", args.eval_batches,
        "--noise_ratios", *args.noise_ratios,
        "--eval_seed", args.eval_seed,
        "--target", args.target,
        "--rank", args.rank,
        "--alpha", args.alpha,
        "--dtype", args.dtype,
        "--sr_scale", args.sr_scale,
        "--allow_sparse_adapter",
    ]
    for image_name in args.exclude_image:
        validation_parts.extend(["--exclude_image", image_name])

    for group, rows in selections:
        for within_group_rank, row in enumerate(rows, 1):
            block = row["block"]
            label = f"{group}-{within_group_rank}-{block.replace('.', '_')}"
            run_dir = runs_dir / label
            adapter_path = run_dir / "lora_adapter.pt"
            selection_file = metadata_dir / f"{label}.json"
            selection = {
                "model_id": args.model_id,
                "base_model_id": args.base_model_id,
                "loss_mode": "official_flow",
                "target": args.target,
                "rank": args.rank,
                "alpha": args.alpha,
                "topk_blocks": 1,
                "selected_blocks": [block],
                "source_importance_csv": args.importance_csv,
                "source_importance_step": args.importance_step,
                "selection_group": group,
            }
            selection_file.write_text(json.dumps(selection, indent=2), encoding="utf-8")
            manifest.append(
                {
                    "label": label,
                    "group": group,
                    "within_group_rank": within_group_rank,
                    "block": block,
                    "block_index": row["block_index"],
                    "importance_rank": row["importance_rank"],
                    "gradient_importance": row["gradient_importance"],
                    "lora_param_count": row["lora_param_count"],
                    "module_count": row["module_count"],
                    "selection_file": selection_file,
                    "run_dir": run_dir,
                    "adapter_path": adapter_path,
                }
            )
            train_command = shell_command(
                [
                        "python3", "train_hf_dit4sr_all_lora_importance.py",
                        "--model_id", args.model_id,
                        "--base_model_id", args.base_model_id,
                        "--variant", args.variant,
                        "--data_dir", args.train_data_dir,
                        "--output_dir", run_dir,
                        "--loss_mode", "official_flow",
                        "--image_size", args.image_size,
                        "--max_images", 0,
                        "--sr_scale", args.sr_scale,
                        "--dtype", args.dtype,
                        "--target", args.target,
                        "--rank", args.rank,
                        "--alpha", args.alpha,
                        "--lora_selection", "metadata",
                        "--lora_block_budget", 1,
                        "--lora_selection_file", selection_file,
                        "--train_steps", args.train_steps,
                        "--batch_size", 1,
                        "--lr", args.lr,
                        "--grad_clip", 1.0,
                        "--profile_noise_ratios", *args.noise_ratios,
                        "--train_noise_ratios", *args.noise_ratios,
                        "--disable_profiling",
                        "--reset_seed_after_lora_injection",
                        "--log_every", 10,
                        "--num_workers", 0,
                        "--seed", args.seed,
                        "--profile_seed", args.profile_seed,
                ]
            )
            training_commands.append(
                f"if [ -f {shlex.quote(str(adapter_path))} ]; then\n"
                f"  echo {shlex.quote('Skipping completed run: ' + label)}\n"
                "else\n"
                f"  {train_command}\n"
                "fi"
            )
            validation_parts.extend(["--adapter", f"{label}={adapter_path}"])

    write_csv(output_dir / "single_block_manifest.csv", manifest)
    training_script = output_dir / "run_single_block_training.sh"
    training_script.write_text("#!/usr/bin/env bash\n\n" + "\n\n".join(training_commands) + "\n", encoding="utf-8")
    validation_script = output_dir / "run_single_block_validation.sh"
    validation_script.write_text("#!/usr/bin/env bash\n\n" + shell_command(validation_parts) + "\n", encoding="utf-8")

    print(f"Selected {len(manifest)} blocks from {len(ranking)} candidates.")
    for group, rows in selections:
        print(f"{group}: " + ", ".join(row["block"] for row in rows))
    print(f"Wrote experiment files to {output_dir}")


if __name__ == "__main__":
    main()
