#!/usr/bin/env python3
"""Train the historical Top8 and threshold K6/K3 with a shared no-bypass protocol."""

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path


LABELS = {"reference_top8": 8, "threshold_k6": 6, "threshold_k3": 3}


def build_jobs(args):
    root = Path(args.root).resolve()
    trainer = root / "train_dit_sr_all_lora_importance.py"
    required = [trainer, root / "configs/realsr_DiT.yaml", root / "weights/realsr.pth",
                root / "weights/autoencoder_vq_f4.pth", Path(args.data_dir)]
    for path in required:
        if not path.exists():
            raise ValueError(f"Required path missing: {path}")
    jobs = []
    for label in args.labels:
        count = LABELS[label]
        selection = root / args.policy_dir / label / "dit_sr_grad_metadata.json"
        meta = json.loads(selection.read_text(encoding="utf-8-sig"))
        expected = dict(target="qv", rank=8, alpha=16, loss_mode="official", topk_blocks=count)
        for key, value in expected.items():
            if meta.get(key) != value:
                raise ValueError(f"{label}: invalid {key}: {meta.get(key)!r}")
        blocks = meta.get("selected_blocks", [])
        if len(blocks) != count or len(set(blocks)) != count:
            raise ValueError(f"{label}: selected block count/uniqueness mismatch")
        out = root / args.output_root / f"{label}_nobypass_{args.steps}_seed{args.seed}"
        if out.exists():
            raise ValueError(f"Output exists, refusing to overwrite: {out}")
        cmd = [sys.executable, str(trainer),
               "--config_path", "configs/realsr_DiT.yaml", "--ckpt_path", "weights/realsr.pth",
               "--autoencoder_ckpt", "weights/autoencoder_vq_f4.pth", "--data_dir", args.data_dir,
               "--output_dir", str(out), "--loss_mode", "official", "--image_size", "256",
               "--lq_size", "64", "--target", "qv", "--rank", "8", "--alpha", "16",
               "--lora_selection", "metadata", "--lora_block_budget", str(count),
               "--topk_blocks", str(count), "--lora_selection_file", str(selection),
               "--blockskip_count", "0", "--train_steps", str(args.steps),
               "--profile_steps", "0", str(args.steps), "--profile_batches", "5",
               "--profile_noise_ratios", "0.05", "0.2", "0.4", "0.6", "0.8", "0.95",
               "--train_noise_ratios", "0.05", "0.2", "0.4", "0.6", "0.8", "0.95",
               "--batch_size", "1", "--lr", "1e-5", "--grad_clip", "1.0",
               "--max_images", "0", "--num_workers", "0", "--seed", str(args.seed),
               "--profile_seed", "42", "--log_every", "10"]
        jobs.append((label, out, cmd))
    return root, jobs


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", default="/mnt/disk1T/liyijuan/dit-sr")
    p.add_argument("--policy_dir", default="outputs/dit_sr_lora_threshold_compute_scan")
    p.add_argument("--output_root", default="outputs/dit_sr_lora_threshold_candidates")
    p.add_argument("--data_dir", default="/mnt/disk1T/liyijuan/star-DTM/data/ucmerced/train_hr")
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--labels", nargs="+", choices=list(LABELS), default=list(LABELS))
    p.add_argument("--dry_run", action="store_true")
    args = p.parse_args()
    if args.steps < 1 or len(set(args.labels)) != len(args.labels):
        p.error("steps must be positive and labels must be unique")
    try:
        root, jobs = build_jobs(args)
        for label, out, cmd in jobs:
            print(f"\n{label}: bypass disabled; output={out}", flush=True)
            print(shlex.join(cmd), flush=True)
            if not args.dry_run:
                subprocess.run(cmd, cwd=root, check=True)
                summary = out / "summary.csv"
                if not summary.is_file():
                    raise ValueError(f"Training finished without summary: {summary}")
                print(summary.read_text(encoding="utf-8"), flush=True)
    except (OSError, ValueError, subprocess.CalledProcessError) as exc:
        p.exit(1, f"Stopped: {exc}\n")


if __name__ == "__main__":
    main()
