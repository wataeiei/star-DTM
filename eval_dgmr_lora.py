#!/usr/bin/env python3
"""Evaluate DGMR LoRA adapters on cached SEN12MS-CR samples."""

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
import torch.nn.functional as F

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
        for param in self.base.parameters():
            param.requires_grad_(False)

    def forward(self, x):
        base_out = self.base(x)
        lora_out = self.lora_up(self.lora_down(x.float())) * self.scale
        return base_out + lora_out.to(dtype=base_out.dtype)


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_csv(path: str | Path, rows: list[dict]) -> None:
    path = Path(path)
    ensure_dir(path.parent)
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


def adapter_module_names(state: dict) -> list[str]:
    names = []
    for key in state:
        if key.endswith(".lora_down.weight"):
            names.append(key[: -len(".lora_down.weight")])
    return sorted(names, key=prof.natural_key)


def load_lora_adapter(model, adapter_path: str | Path) -> list[str]:
    state = torch.load(adapter_path, map_location="cpu")
    injected = []
    for name in adapter_module_names(state):
        parent, leaf = parent_and_leaf(model, name)
        base = getattr(parent, leaf)
        if not isinstance(base, nn.Linear):
            raise SystemExit(f"Adapter target is not nn.Linear: {name}")
        rank = int(state.get(f"{name}.rank", torch.tensor(state[f"{name}.lora_down.weight"].shape[0])).item())
        alpha = float(state.get(f"{name}.alpha", torch.tensor(rank)).item())
        lora = LoRALinear(base, rank, alpha)
        lora.lora_down.weight.data.copy_(state[f"{name}.lora_down.weight"])
        lora.lora_up.weight.data.copy_(state[f"{name}.lora_up.weight"])
        lora.to(device=base.weight.device, dtype=base.weight.dtype)
        setattr(parent, leaf, lora)
        injected.append(name)
    if not injected:
        raise SystemExit(f"No LoRA modules found in {adapter_path}")
    return injected


def load_full_state(model, state_path: str | Path) -> None:
    state = torch.load(state_path, map_location="cpu")
    if state.get("net_G") is not None:
        model.net_G.load_state_dict(state["net_G"], strict=True)
    if state.get("diffusion") is not None:
        model.diffusion.load_state_dict(state["diffusion"], strict=True)


def file_size_mb(path: str | Path) -> float:
    return Path(path).stat().st_size / (1024.0 * 1024.0)


def psnr(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-10) -> torch.Tensor:
    mse = (pred - target).float().pow(2).flatten(1).mean(dim=1)
    return 10.0 * torch.log10(1.0 / torch.clamp(mse, min=eps))


def ssim_global(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    x = pred.float().flatten(1)
    y = target.float().flatten(1)
    mu_x = x.mean(dim=1)
    mu_y = y.mean(dim=1)
    var_x = ((x - mu_x[:, None]) ** 2).mean(dim=1)
    var_y = ((y - mu_y[:, None]) ** 2).mean(dim=1)
    cov = ((x - mu_x[:, None]) * (y - mu_y[:, None])).mean(dim=1)
    c1 = 0.01**2
    c2 = 0.03**2
    return ((2 * mu_x * mu_y + c1) * (2 * cov + c2)) / ((mu_x**2 + mu_y**2 + c1) * (var_x + var_y + c2) + eps)


def sam_degrees(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    x = pred.float()
    y = target.float()
    dot = (x * y).sum(dim=1)
    denom = torch.linalg.vector_norm(x, dim=1) * torch.linalg.vector_norm(y, dim=1)
    cos = torch.clamp(dot / torch.clamp(denom, min=eps), -1.0, 1.0)
    return torch.rad2deg(torch.acos(cos)).flatten(1).mean(dim=1)


def resize_like(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if pred.shape[-2:] == target.shape[-2:]:
        return pred
    return F.interpolate(pred, size=target.shape[-2:], mode="bilinear", align_corners=False)


def evaluate(args: argparse.Namespace) -> None:
    prof.set_seed(args.seed)
    prof.install_runtime_stubs()
    sen12_dir = prof.locate_sen12_dir(Path.cwd())
    sys.path.insert(0, str(sen12_dir))
    os.chdir(sen12_dir)

    from dgmr import DGMR

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model = DGMR(prof.make_opts(args))
    prof.disable_checkpoint_saves(model)
    for param in prof.iter_dgmr_parameters(model):
        param.requires_grad_(False)

    injected = []
    adapter_mb = 0.0
    if args.full_state_path:
        load_full_state(model, args.full_state_path)
        adapter_mb = file_size_mb(args.full_state_path)
        print(f"Loaded full DGMR state from {args.full_state_path}")

    if args.adapter_path:
        injected = load_lora_adapter(model, args.adapter_path)
        adapter_mb = file_size_mb(args.adapter_path)
        print(f"Loaded {len(injected)} LoRA modules from {args.adapter_path}")

    data_iter = prof.load_data_iter(args, sen12_dir, device)
    rows = []
    total_time = 0.0

    if torch.cuda.is_available() and not args.cpu:
        torch.cuda.reset_peak_memory_stats()

    model.net_G.eval()
    model.diffusion.eval()
    with torch.no_grad():
        for idx in range(1, args.eval_samples + 1):
            batch = next(data_iter)
            model.set_input(batch)
            start = time.time()
            output = model.forward()
            if isinstance(output, tuple):
                pred = output[0]
            else:
                pred = output
            if not torch.is_tensor(pred):
                raise SystemExit("DGMR forward did not return a tensor prediction.")
            if torch.cuda.is_available() and not args.cpu:
                torch.cuda.synchronize()
            infer_s = time.time() - start
            total_time += infer_s

            target = getattr(model, "cloudfree_data", batch["target"]["S2"])
            pred = resize_like(pred, target).clamp(0.0, 1.0)
            target = target.clamp(0.0, 1.0)

            batch_psnr = psnr(pred, target)
            batch_ssim = ssim_global(pred, target)
            batch_mae = (pred - target).abs().float().flatten(1).mean(dim=1)
            batch_sam = sam_degrees(pred, target)

            for b in range(pred.shape[0]):
                rows.append(
                    {
                        "sample_index": len(rows),
                        "psnr": float(batch_psnr[b].cpu()),
                        "ssim": float(batch_ssim[b].cpu()),
                        "mae": float(batch_mae[b].cpu()),
                        "sam_deg": float(batch_sam[b].cpu()),
                        "inference_time_s": infer_s / max(pred.shape[0], 1),
                    }
                )
            print(f"[{idx}/{args.eval_samples}] PSNR={float(batch_psnr.mean().cpu()):.3f} SSIM={float(batch_ssim.mean().cpu()):.4f}")

    out_dir = ensure_dir(args.output_dir)
    write_csv(out_dir / "per_sample_metrics.csv", rows)
    mean = lambda key: sum(row[key] for row in rows) / max(len(rows), 1)
    peak_mem = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0) if torch.cuda.is_available() and not args.cpu else 0.0
    summary = [
        {
            "method": args.method,
            "num_samples": len(rows),
            "mean_psnr": mean("psnr"),
            "mean_ssim": mean("ssim"),
            "mean_mae": mean("mae"),
            "mean_sam_deg": mean("sam_deg"),
            "mean_inference_time_s": mean("inference_time_s"),
            "total_eval_time_s": total_time,
            "peak_cuda_mem_mb": peak_mem,
            "adapter_size_mb": adapter_mb,
            "upload_time_1mbps_s": adapter_mb * 8.0,
            "adapter_path": args.adapter_path,
            "lora_module_count": len(injected),
            "injected_modules": ";".join(injected),
        }
    ]
    write_csv(out_dir / "eval_summary.csv", summary)
    print(f"Wrote {out_dir / 'eval_summary.csv'}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_root", default="")
    parser.add_argument("--data_list_filepath", default="")
    parser.add_argument("--hf_dataset", default="")
    parser.add_argument("--hf_split", default="train")
    parser.add_argument("--hf_max_samples", type=int, default=0)
    parser.add_argument("--cached_sample_dir", default="")
    parser.add_argument("--ckpt_path", default="")
    parser.add_argument("--adapter_path", default="")
    parser.add_argument("--full_state_path", default="")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--method", default="DGMR")
    parser.add_argument("--eval_samples", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--load_size", type=int, default=128)
    parser.add_argument("--crop_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--split", default="train", choices=["all", "train", "val", "test"])
    parser.add_argument("--region", default="all")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--gpu_ids", default="0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    evaluate(args)
    if args.hf_dataset:
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)


if __name__ == "__main__":
    main()
