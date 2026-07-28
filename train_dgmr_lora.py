#!/usr/bin/env python3
"""Parameter-efficient LoRA training for DGMR with the official task loss.

Copy this file together with profile_dgmr_grad.py into the DGMR repo root.
It freezes the original DGMR weights, injects LoRA into selected Linear
modules, and trains only LoRA parameters through DGMR.optimize_parameters().
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

import profile_dgmr_grad as prof


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int, alpha: float) -> None:
        super().__init__()
        self.base = base
        self.rank = rank
        self.alpha = alpha
        self.scale = alpha / max(rank, 1)
        self.lora_down = nn.Linear(base.in_features, rank, bias=False)
        self.lora_up = nn.Linear(rank, base.out_features, bias=False)
        nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_up.weight)
        for param in self.base.parameters():
            param.requires_grad_(False)

    def forward(self, x):
        base_out = self.base(x)
        lora_out = self.lora_up(self.lora_down(x.float())) * self.scale
        return base_out + lora_out.to(dtype=base_out.dtype)


class NullOptimizer:
    def zero_grad(self, *args, **kwargs):
        return None

    def step(self, *args, **kwargs):
        return None


class ClippedOptimizer:
    def __init__(self, optimizer: torch.optim.Optimizer, params: list[torch.nn.Parameter], grad_clip: float) -> None:
        self.optimizer = optimizer
        self.params = params
        self.grad_clip = grad_clip

    def zero_grad(self, *args, **kwargs):
        return self.optimizer.zero_grad(*args, **kwargs)

    def step(self, *args, **kwargs):
        if self.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.params, self.grad_clip)
        return self.optimizer.step(*args, **kwargs)


def write_csv(path: str | Path, rows: list[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def parent_and_leaf(root, full_name: str):
    parts = full_name.split(".")
    obj = getattr(root, parts[0])
    for part in parts[1:-1]:
        if part.isdigit() and hasattr(obj, "__getitem__"):
            obj = obj[int(part)]
        else:
            obj = getattr(obj, part)
    return obj, parts[-1]


def read_topk_blocks(path: str | Path, topk: int) -> set[str]:
    rows = []
    with Path(path).open("r", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            row["score"] = float(row.get("selection_score") or row.get("normalized_grad_score") or 0.0)
            row["selected_bool"] = str(row.get("selected", "")).lower() in ("true", "1", "yes")
            rows.append(row)
    selected = {row["block"] for row in rows if row["selected_bool"]}
    if selected:
        return selected
    return {row["block"] for row in sorted(rows, key=lambda r: r["score"], reverse=True)[:topk]}


def selected_blocks(args: argparse.Namespace) -> set[str] | None:
    if args.lora_scheme == "all_linear":
        return None
    if args.lora_scheme == "sar_attn":
        return {"net_G.sar_sa.attn"}
    if args.lora_scheme == "dual_attn":
        return {"net_G.sar_sa.attn", "net_G.opt_sa.attn"}
    if args.lora_scheme == "fusion_attn":
        return {"net_G.sar_sa.attn", "net_G.opt_sa.attn", "net_G.csa.tr"}
    if args.lora_scheme == "grad_topk":
        if not args.grad_scores_csv:
            raise SystemExit("--grad_scores_csv is required for --lora_scheme grad_topk")
        return read_topk_blocks(args.grad_scores_csv, args.topk_blocks)
    if args.lora_scheme == "custom":
        return {part.strip() for part in args.custom_blocks.split(",") if part.strip()}
    raise SystemExit(f"Unknown lora_scheme={args.lora_scheme}")


def inject_lora(model, args: argparse.Namespace) -> list[tuple[str, LoRALinear]]:
    blocks = selected_blocks(args)
    injected = []
    candidates = list(prof.iter_dgmr_named_modules(model))
    for name, module in candidates:
        if not isinstance(module, nn.Linear):
            continue
        block = prof.block_key(name, args.block_regex)
        if blocks is not None and block not in blocks:
            continue
        parent, leaf = parent_and_leaf(model, name)
        lora = LoRALinear(module, args.rank, args.alpha)
        lora.to(device=module.weight.device, dtype=module.weight.dtype)
        setattr(parent, leaf, lora)
        injected.append((name, lora))

    if not injected:
        raise SystemExit("No LoRA modules injected. Check --lora_scheme/--grad_scores_csv/--block_regex.")
    return injected


def lora_state_dict(injected: list[tuple[str, LoRALinear]]) -> dict:
    state = {}
    for name, module in injected:
        state[f"{name}.lora_down.weight"] = module.lora_down.weight.detach().cpu()
        state[f"{name}.lora_up.weight"] = module.lora_up.weight.detach().cpu()
        state[f"{name}.alpha"] = torch.tensor(float(module.alpha))
        state[f"{name}.rank"] = torch.tensor(int(module.rank))
    return state


def file_size_mb(path: str | Path) -> float:
    return Path(path).stat().st_size / (1024.0 * 1024.0)


def make_opts(args: argparse.Namespace):
    return prof.make_opts(args)


def train(args: argparse.Namespace) -> None:
    prof.set_seed(args.seed)
    prof.install_runtime_stubs()
    sen12_dir = prof.locate_sen12_dir(Path.cwd())
    sys.path.insert(0, str(sen12_dir))
    os.chdir(sen12_dir)

    from dgmr import DGMR

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model = DGMR(make_opts(args))
    prof.disable_checkpoint_saves(model)

    for param in prof.iter_dgmr_parameters(model):
        param.requires_grad_(False)

    injected = inject_lora(model, args)
    trainable_params = [p for _, module in injected for p in module.parameters() if p.requires_grad]
    trainable_count = sum(p.numel() for p in trainable_params)
    total_count = sum(p.numel() for p in prof.iter_dgmr_parameters(model))

    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    model.optimizer_G = ClippedOptimizer(optimizer, trainable_params, args.grad_clip)
    model.optimizer_G1 = NullOptimizer()
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

    adapter_path = out_dir / "dgmr_lora_adapter.pt"
    torch.save(lora_state_dict(injected), adapter_path)
    train_time = time.time() - start
    adapter_mb = file_size_mb(adapter_path)
    peak_mem = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0) if torch.cuda.is_available() and not args.cpu else 0.0

    write_csv(out_dir / "train_log.csv", log_rows)
    summary = [
        {
            "lora_scheme": args.lora_scheme,
            "train_steps": args.train_steps,
            "train_time_s": train_time,
            "sec_per_step": train_time / max(args.train_steps, 1),
            "rank": args.rank,
            "alpha": args.alpha,
            "lr": args.lr,
            "lora_module_count": len(injected),
            "trainable_params": trainable_count,
            "total_params_with_lora": total_count,
            "trainable_param_pct": trainable_count / max(total_count, 1) * 100.0,
            "adapter_size_mb": adapter_mb,
            "upload_time_1mbps_s": adapter_mb * 8.0,
            "peak_cuda_mem_mb": peak_mem,
            "adapter_path": str(adapter_path),
            "injected_blocks": ";".join(sorted({prof.block_key(name, args.block_regex) for name, _ in injected})),
        }
    ]
    write_csv(out_dir / "summary.csv", summary)
    print(f"Saved LoRA adapter to {adapter_path}")
    print(f"Wrote {out_dir / 'summary.csv'}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_root", default="")
    parser.add_argument("--data_list_filepath", default="")
    parser.add_argument("--hf_dataset", default="")
    parser.add_argument("--hf_split", default="train")
    parser.add_argument("--hf_max_samples", type=int, default=0)
    parser.add_argument("--cached_sample_dir", default="", help="Directory of cached SEN12MS-CR .npz samples")
    parser.add_argument("--ckpt_path", default="")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--lora_scheme",
        default="dual_attn",
        choices=["sar_attn", "dual_attn", "fusion_attn", "grad_topk", "all_linear", "custom"],
    )
    parser.add_argument("--grad_scores_csv", default="")
    parser.add_argument("--custom_blocks", default="")
    parser.add_argument("--topk_blocks", type=int, default=8)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=16.0)
    parser.add_argument("--train_steps", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--split", default="train", choices=["all", "train", "val", "test"])
    parser.add_argument("--region", default="all")
    parser.add_argument("--load_size", type=int, default=128)
    parser.add_argument("--crop_size", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--gpu_ids", default="0")
    parser.add_argument("--block_regex", default="")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    return parser


def main() -> None:
    train(build_parser().parse_args())


if __name__ == "__main__":
    main()
