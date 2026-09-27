#!/usr/bin/env python3
"""Run one serial stage of the TSD-SR Threshold bypass search."""

from __future__ import annotations

import argparse
import csv
import json
import random
import subprocess
import sys
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate_manifest", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--train_script", default="train_tsdsr_lora.py")
    parser.add_argument("--pretrained_model", required=True)
    parser.add_argument("--official_lora_dir", required=True)
    parser.add_argument("--teacher_lora_dir", required=True)
    parser.add_argument("--default_embedding_dir", required=True)
    parser.add_argument("--null_embedding_dir", required=True)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--train_steps", type=int, default=20)
    parser.add_argument("--timing_warmup_steps", type=int, default=0)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--sr_scale", type=int, default=4)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=16.0)
    parser.add_argument("--reg_rank", type=int, default=16)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--reg_lr", type=float, default=1e-6)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dtype", choices=["fp16", "bf16", "fp32"], default="fp16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--include_all_lora", action="store_true")
    parser.add_argument("--k", type=int, action="append", default=[])
    parser.add_argument("--candidate_id", action="append", default=[])
    parser.add_argument("--include_zero_bypass", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--shuffle_run_order", action="store_true")
    parser.add_argument("--run_order_seed", type=int, default=0)
    parser.add_argument("--cooldown_seconds", type=float, default=0.0)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def common_command(args: argparse.Namespace, run_dir: Path, method: str) -> list[str]:
    return [
        sys.executable,
        args.train_script,
        "--pretrained_model", args.pretrained_model,
        "--official_lora_dir", args.official_lora_dir,
        "--teacher_lora_dir", args.teacher_lora_dir,
        "--default_embedding_dir", args.default_embedding_dir,
        "--null_embedding_dir", args.null_embedding_dir,
        "--data_dir", args.data_dir,
        "--output_dir", str(run_dir),
        "--method", method,
        "--image_size", str(args.image_size),
        "--sr_scale", str(args.sr_scale),
        "--rank", str(args.rank),
        "--alpha", str(args.alpha),
        "--reg_rank", str(args.reg_rank),
        "--train_steps", str(args.train_steps),
        "--timing_warmup_steps", str(args.timing_warmup_steps),
        "--batch_size", str(args.batch_size),
        "--num_workers", str(args.num_workers),
        "--lr", str(args.lr),
        "--reg_lr", str(args.reg_lr),
        "--weight_decay", str(args.weight_decay),
        "--grad_clip", str(args.grad_clip),
        "--checkpoint_every", "0",
        "--seed", str(args.seed),
        "--dtype", args.dtype,
        "--device", args.device,
        "--log_every", str(args.log_every),
    ]


def resolve_manifest_path(value: str, manifest_path: Path) -> Path:
    path = Path(value)
    if path.is_absolute() or path.exists():
        return path
    candidate = manifest_path.parent / path
    return candidate if candidate.exists() else path


def main() -> None:
    args = parse_args()
    if args.train_steps <= 0:
        raise SystemExit("--train_steps must be positive")
    if not 0 <= args.timing_warmup_steps < args.train_steps:
        raise SystemExit(
            "--timing_warmup_steps must be non-negative and smaller than "
            "--train_steps"
        )
    if args.cooldown_seconds < 0:
        raise SystemExit("--cooldown_seconds must be non-negative")
    manifest_path = Path(args.candidate_manifest)
    candidates = read_csv(manifest_path)
    if not candidates:
        raise SystemExit(f"Candidate manifest is empty: {manifest_path}")
    if args.k:
        candidates = [row for row in candidates if int(row["selected_k"]) in args.k]
    if args.candidate_id:
        wanted = set(args.candidate_id)
        candidates = [row for row in candidates if row["candidate_id"] in wanted]
    if not args.include_zero_bypass:
        candidates = [row for row in candidates if float(row["mean_bypass_blocks"]) > 0]
    if not candidates and not args.include_all_lora:
        raise SystemExit("No candidates remain after filtering")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plan = []
    if args.include_all_lora:
        run_dir = output_dir / "all_lora"
        plan.append({
            "run_kind": "all_lora",
            "candidate_id": "",
            "selected_k": 24,
            "global_threshold": "",
            "run_dir": str(run_dir),
            "command": common_command(args, run_dir, "All-LoRA")
            + ["--selection", "all"],
        })

    by_k: dict[int, list[dict[str, str]]] = {}
    for row in candidates:
        by_k.setdefault(int(row["selected_k"]), []).append(row)
    for k, group in sorted(by_k.items()):
        selection = resolve_manifest_path(group[0]["selection_file"], manifest_path)
        k_dir = output_dir / f"K{k:02d}"
        for run_kind, method, extra in (
            ("native", "Native-K", []),
            ("controller_b0", "Controller-B0", ["--controller_b0"]),
        ):
            run_dir = k_dir / run_kind
            plan.append({
                "run_kind": run_kind,
                "candidate_id": "",
                "selected_k": k,
                "global_threshold": "",
                "run_dir": str(run_dir),
                "command": common_command(args, run_dir, method)
                + ["--selection", "metadata", "--selection_file", str(selection)]
                + extra,
            })
        for row in sorted(group, key=lambda item: item["candidate_id"]):
            policy = resolve_manifest_path(row["policy_csv"], manifest_path)
            run_dir = k_dir / "threshold_bypass" / row["candidate_id"]
            plan.append({
                "run_kind": "threshold_bypass",
                "candidate_id": row["candidate_id"],
                "selected_k": k,
                "global_threshold": row["global_threshold"],
                "run_dir": str(run_dir),
                "command": common_command(args, run_dir, "Threshold bypass")
                + [
                    "--selection", "metadata",
                    "--selection_file", str(selection),
                    "--bypass_policy_csv", str(policy),
                ],
            })

    if args.shuffle_run_order:
        random.Random(args.run_order_seed).shuffle(plan)

    plan_rows = [
        {**{key: value for key, value in item.items() if key != "command"},
         "command": json.dumps(item["command"])}
        for item in plan
    ]
    write_csv(output_dir / "run_plan.csv", plan_rows)
    if args.dry_run:
        print(f"Wrote {len(plan)} serial runs to {output_dir / 'run_plan.csv'}")
        return

    results = []
    for index, item in enumerate(plan, 1):
        if index > 1 and args.cooldown_seconds > 0:
            print(f"Cooling down for {args.cooldown_seconds:g} seconds")
            time.sleep(args.cooldown_seconds)
        run_dir = Path(item["run_dir"])
        summary_path = run_dir / "summary.csv"
        if args.resume and summary_path.exists():
            print(f"[{index}/{len(plan)}] Resume existing {run_dir}")
        else:
            if run_dir.exists() and any(run_dir.iterdir()):
                raise SystemExit(
                    f"Run directory is non-empty without a completed summary: {run_dir}"
                )
            print(f"[{index}/{len(plan)}] Running {item['run_kind']} {item['candidate_id']}")
            subprocess.run(item["command"], check=True)
        summary_rows = read_csv(summary_path)
        if len(summary_rows) != 1:
            raise SystemExit(f"Expected one summary row in {summary_path}")
        results.append({
            "run_kind": item["run_kind"],
            "candidate_id": item["candidate_id"],
            "selected_k": item["selected_k"],
            "global_threshold": item["global_threshold"],
            **summary_rows[0],
            "run_dir": str(run_dir),
        })
        write_csv(output_dir / "stage_results.csv", results)
    print(f"Completed {len(results)} serial runs; wrote {output_dir / 'stage_results.csv'}")


if __name__ == "__main__":
    main()
