#!/usr/bin/env python3
"""Full-parameter fine-tuning for DGMR with the official task loss."""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path

import torch

import profile_dgmr_grad as prof


def write_csv(path: str | Path, rows: list[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_full_state(model, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "net_G": model.net_G.state_dict() if hasattr(model, "net_G") else None,
            "diffusion": model.diffusion.state_dict() if hasattr(model, "diffusion") else None,
        },
        path,
    )


def file_size_mb(path: str | Path) -> float:
    return Path(path).stat().st_size / (1024.0 * 1024.0)


def train(args: argparse.Namespace) -> None:
    prof.set_seed(args.seed)
    prof.install_runtime_stubs()
    sen12_dir = prof.locate_sen12_dir(Path.cwd())
    sys.path.insert(0, str(sen12_dir))
    os.chdir(sen12_dir)

    from dgmr import DGMR

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model = DGMR(prof.make_opts(args))
    prof.disable_checkpoint_saves(model)

    trainable_params = [p for p in prof.iter_dgmr_parameters(model) if p.requires_grad]
    trainable_count = sum(p.numel() for p in trainable_params)
    data_iter = prof.load_data_iter(args, sen12_dir, device)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_rows = []
    start = time.time()
    if torch.cuda.is_available() and not args.cpu:
        torch.cuda.reset_peak_memory_stats()

    for step in range(1, args.train_steps + 1):
        batch = next(data_iter)
        model.set_input(batch)
        model.optimize_parameters(epoch=0)
        loss_tensor = getattr(model, "loss_G", None)
        loss_value = float(loss_tensor.detach().cpu()) if torch.is_tensor(loss_tensor) else float("nan")

        elapsed = time.time() - start
        row = {"step": step, "loss": loss_value, "elapsed_s": elapsed}
        if torch.cuda.is_available() and not args.cpu:
            row["cuda_mem_allocated_mb"] = torch.cuda.memory_allocated() / (1024.0 * 1024.0)
            row["cuda_max_mem_allocated_mb"] = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
        log_rows.append(row)

        if step == 1 or step % args.log_every == 0 or step == args.train_steps:
            print(f"step {step:05d}/{args.train_steps} loss={loss_value:.6f} elapsed={elapsed:.1f}s")

    state_path = out_dir / "full_dgmr_state.pt"
    save_full_state(model, state_path)
    train_time = time.time() - start
    state_mb = file_size_mb(state_path)
    peak_mem = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0) if torch.cuda.is_available() and not args.cpu else 0.0

    write_csv(out_dir / "train_log.csv", log_rows)
    write_csv(
        out_dir / "summary.csv",
        [
            {
                "train_method": "full_finetune",
                "train_steps": args.train_steps,
                "train_time_s": train_time,
                "sec_per_step": train_time / max(args.train_steps, 1),
                "lr": args.lr,
                "trainable_params": trainable_count,
                "full_state_size_mb": state_mb,
                "upload_time_1mbps_s": state_mb * 8.0,
                "peak_cuda_mem_mb": peak_mem,
                "state_path": str(state_path),
            }
        ],
    )
    print(f"Saved full DGMR state to {state_path}")
    print(f"Wrote {out_dir / 'summary.csv'}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_root", default="")
    parser.add_argument("--data_list_filepath", default="")
    parser.add_argument("--hf_dataset", default="")
    parser.add_argument("--hf_split", default="train")
    parser.add_argument("--hf_max_samples", type=int, default=0)
    parser.add_argument("--cached_sample_dir", default="")
    parser.add_argument("--ckpt_path", default="")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--train_steps", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--split", default="train", choices=["all", "train", "val", "test"])
    parser.add_argument("--region", default="all")
    parser.add_argument("--load_size", type=int, default=128)
    parser.add_argument("--crop_size", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--gpu_ids", default="0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    train(args)
    if args.hf_dataset:
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)


if __name__ == "__main__":
    main()
