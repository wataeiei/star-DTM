#!/usr/bin/env python3
"""Gradient layer-importance profiling for DiT-style super-resolution models.

This script is intentionally model-agnostic. DiT-SR and DiT4SR have different
model builders, checkpoint formats, and forward signatures, so the only required
project-specific part is a small Python adapter module that provides:

  build_model(args) -> torch.nn.Module
  build_dataloader(args) -> iterable batches
  compute_loss(model, batch, args, device) -> torch.Tensor

The profiler will:
  1. Load the model through the adapter.
  2. Freeze the base model.
  3. Insert temporary probe LoRA modules into attention Linear layers.
  4. Run a few calibration batches.
  5. Score each Transformer block by probe-LoRA gradient norm.
  6. Write CSV scores and selected blocks.

Example:
  python3 profile_dit_grad_topk.py \
    --adapter_module dit_profile_adapter_ditsr \
    --checkpoint path/to/model.pth \
    --data_dir data/ucmerced/train_hr \
    --output_dir outputs/ditsr_grad_profile_ucmerced \
    --target qv \
    --rank 8 \
    --alpha 16 \
    --topk_blocks 8 \
    --probe_batches 20 \
    --seed 42
"""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import math
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
import torch.nn as nn


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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


def natural_key(text: str) -> list[int | str]:
    key: list[int | str] = []
    for part in re.split(r"(\d+)", text):
        if not part:
            continue
        key.append(int(part) if part.isdigit() else part)
    return key


def target_keywords(target: str) -> tuple[str, ...]:
    mapping = {
        "q": ("q", "to_q", "q_proj"),
        "v": ("v", "to_v", "v_proj"),
        "qv": ("q", "to_q", "q_proj", "v", "to_v", "v_proj"),
        "qkv": ("qkv", "q", "k", "v", "to_q", "to_k", "to_v", "q_proj", "k_proj", "v_proj"),
        "qkvo": (
            "qkv",
            "q",
            "k",
            "v",
            "proj",
            "to_q",
            "to_k",
            "to_v",
            "to_out",
            "q_proj",
            "k_proj",
            "v_proj",
            "out_proj",
        ),
    }
    if target not in mapping:
        raise SystemExit(f"Unknown target={target}; choose one of {sorted(mapping)}")
    return mapping[target]


def module_name_matches_target(name: str, target: str) -> bool:
    lname = name.lower()
    leaf = lname.split(".")[-1]
    keywords = target_keywords(target)
    if target == "qkv" and leaf in ("qkv", "qkv_proj", "to_qkv"):
        return True
    if target == "qkvo" and any(k in leaf for k in keywords):
        return True
    if target in ("q", "v", "qv"):
        allowed = set()
        if "q" in target:
            allowed.update(["q", "to_q", "q_proj"])
        if "v" in target:
            allowed.update(["v", "to_v", "v_proj"])
        return leaf in allowed or any(leaf.endswith("." + k) for k in allowed)
    return any(k in leaf for k in keywords)


def infer_block_key(name: str, block_regex: str = "") -> str:
    if block_regex:
        match = re.search(block_regex, name)
        if match:
            return match.group(0)

    parts = name.split(".")
    for marker in ("blocks", "layers", "transformer_blocks", "dit_blocks", "body"):
        if marker in parts:
            idx = parts.index(marker)
            if idx + 1 < len(parts) and parts[idx + 1].isdigit():
                return ".".join(parts[: idx + 2])

    # Fallback for names like block0.attn.qkv or layers_3_attn_q.
    match = re.search(r"(?:block|layer|blocks|layers)[._-]?(\d+)", name, flags=re.IGNORECASE)
    if match:
        return match.group(0)
    return ""


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int, alpha: int) -> None:
        super().__init__()
        self.base = base
        self.rank = rank
        self.alpha = alpha
        self.scale = alpha / rank
        self.lora_down = nn.Linear(base.in_features, rank, bias=False)
        self.lora_up = nn.Linear(rank, base.out_features, bias=False)
        nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_up.weight)
        for p in self.base.parameters():
            p.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        lora_out = self.lora_up(self.lora_down(x.float())) * self.scale
        return base_out + lora_out.to(dtype=base_out.dtype)


def split_parent_name(root: nn.Module, dotted_name: str) -> tuple[nn.Module, str]:
    parts = dotted_name.split(".")
    parent = root
    for part in parts[:-1]:
        parent = parent[int(part)] if part.isdigit() else getattr(parent, part)
    return parent, parts[-1]


def iter_lora_modules(root: nn.Module) -> Iterable[tuple[str, LoRALinear]]:
    for name, module in root.named_modules():
        if isinstance(module, LoRALinear):
            yield name, module


@dataclass
class InjectionInfo:
    module_names: list[str]
    block_names: list[str]
    lora_param_count: int


def inject_probe_lora(model: nn.Module, args: argparse.Namespace) -> InjectionInfo:
    replacements: list[tuple[str, nn.Linear]] = []
    block_names = set()
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        block_key = infer_block_key(name, args.block_regex)
        if not block_key:
            continue
        if module_name_matches_target(name, args.target):
            replacements.append((name, module))
            block_names.add(block_key)

    if not replacements:
        raise SystemExit(
            "No probe LoRA targets found. Try a different --target or provide --block_regex.\n"
            "Use --inspect_only first to print candidate module names."
        )

    for name, module in replacements:
        parent, child_name = split_parent_name(model, name)
        setattr(parent, child_name, LoRALinear(module, rank=args.rank, alpha=args.alpha))

    lora_params = sum(p.numel() for _n, m in iter_lora_modules(model) for p in (m.lora_down.weight, m.lora_up.weight))
    return InjectionInfo([name for name, _m in replacements], sorted(block_names, key=natural_key), lora_params)


def move_lora_to_device(model: nn.Module, device: torch.device) -> None:
    for _name, module in iter_lora_modules(model):
        module.lora_down.to(device=device, dtype=torch.float32)
        module.lora_up.to(device=device, dtype=torch.float32)


def lora_grad_norm(module: LoRALinear) -> float:
    total = 0.0
    for param in (module.lora_down.weight, module.lora_up.weight):
        if param.grad is not None:
            total += float(param.grad.detach().float().pow(2).sum().cpu())
    return math.sqrt(total)


def block_order(blocks: Iterable[str]) -> dict[str, int]:
    ordered = sorted(set(blocks), key=natural_key)
    return {block: idx + 1 for idx, block in enumerate(ordered)}


def select_blocks(rows: list[dict], topk: int) -> list[str]:
    ranked = sorted(rows, key=lambda row: row["selection_score"], reverse=True)
    return [row["block"] for row in ranked[:topk]]


def inspect_model(model: nn.Module, args: argparse.Namespace) -> None:
    print("== Transformer/block-like modules ==")
    for name, module in model.named_modules():
        cls = module.__class__.__name__
        if any(k in cls.lower() for k in ("block", "attention", "transformer", "dit")):
            print(name, cls)

    print("\n== Linear candidates ==")
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            block_key = infer_block_key(name, args.block_regex)
            mark = "*" if block_key and module_name_matches_target(name, args.target) else " "
            print(f"{mark} {name} [{module.in_features}->{module.out_features}] block={block_key}")


def profile(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    adapter = importlib.import_module(args.adapter_module)
    model = adapter.build_model(args).to(device)
    model.train()

    if args.inspect_only:
        inspect_model(model, args)
        return

    for p in model.parameters():
        p.requires_grad_(False)

    info = inject_probe_lora(model, args)
    move_lora_to_device(model, device)
    loader = adapter.build_dataloader(args)

    rows_by_block: dict[str, dict] = {}
    for name, module in iter_lora_modules(model):
        block = infer_block_key(name, args.block_regex)
        if not block:
            continue
        row = rows_by_block.setdefault(
            block,
            {
                "block": block,
                "lora_param_count": 0,
                "module_count": 0,
                "grad_sq": 0.0,
            },
        )
        row["lora_param_count"] += module.lora_down.weight.numel() + module.lora_up.weight.numel()
        row["module_count"] += 1

    data_iter = iter(loader)
    valid_batches = 0
    total_loss = 0.0
    model.zero_grad(set_to_none=True)
    for idx in range(1, args.probe_batches + 1):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch = next(data_iter)

        loss = adapter.compute_loss(model, batch, args, device)
        if not torch.is_tensor(loss):
            raise SystemExit("adapter.compute_loss must return a torch.Tensor loss")
        if not torch.isfinite(loss):
            print(f"probe batch {idx}/{args.probe_batches} skipped: non-finite loss")
            model.zero_grad(set_to_none=True)
            continue
        loss.backward()
        valid_batches += 1
        loss_value = float(loss.detach().cpu())
        total_loss += loss_value
        print(f"probe batch {idx:03d}/{args.probe_batches} loss={loss_value:.6f}")

    if valid_batches == 0:
        raise SystemExit("All probe batches failed with non-finite loss.")

    for name, module in iter_lora_modules(model):
        block = infer_block_key(name, args.block_regex)
        if block in rows_by_block:
            rows_by_block[block]["grad_sq"] += lora_grad_norm(module) ** 2

    order = block_order(rows_by_block)
    total_blocks = len(order)
    rows = []
    for block, row in rows_by_block.items():
        grad_norm = math.sqrt(row["grad_sq"])
        p_count = int(row["lora_param_count"])
        normalized = grad_norm / math.sqrt(max(p_count, 1)) if args.normalize else grad_norm
        bp_cost = total_blocks - order[block] + 1
        selection = normalized
        if args.compute_lambda > 0:
            selection = normalized / (p_count + args.compute_lambda * bp_cost)
        rows.append(
            {
                "block": block,
                "block_index": order[block],
                "grad_norm": grad_norm,
                "lora_param_count": p_count,
                "module_count": int(row["module_count"]),
                "normalized_grad_score": normalized,
                "bp_cost": bp_cost,
                "compute_lambda": args.compute_lambda,
                "selection_score": selection,
                "probe_batches": valid_batches,
                "mean_probe_loss": total_loss / valid_batches,
                "selected": False,
            }
        )

    selected = set(select_blocks(rows, args.topk_blocks))
    for row in rows:
        row["selected"] = row["block"] in selected
    rows.sort(key=lambda row: row["block_index"])

    out = ensure_dir(args.output_dir)
    write_csv(out / "dit_grad_topk_scores.csv", rows)
    metadata = {
        "adapter_module": args.adapter_module,
        "checkpoint": args.checkpoint,
        "data_dir": args.data_dir,
        "target": args.target,
        "rank": args.rank,
        "alpha": args.alpha,
        "topk_blocks": args.topk_blocks,
        "probe_batches": args.probe_batches,
        "seed": args.seed,
        "block_regex": args.block_regex,
        "selected_blocks": sorted(selected, key=natural_key),
        "injected_module_count": len(info.module_names),
        "candidate_block_count": len(info.block_names),
    }
    (out / "dit_grad_topk_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Wrote {out / 'dit_grad_topk_scores.csv'}")
    print(f"Wrote {out / 'dit_grad_topk_metadata.json'}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter_module", required=True, help="Python module with build_model/build_dataloader/compute_loss")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--config", default="")
    parser.add_argument("--data_dir", default="")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--target", default="qv", choices=["q", "v", "qv", "qkv", "qkvo"])
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--alpha", type=int, default=16)
    parser.add_argument("--topk_blocks", type=int, default=8)
    parser.add_argument("--probe_batches", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--hr_size", type=int, default=256)
    parser.add_argument("--lr_size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--block_regex", default="", help="Optional regex to identify a block prefix from module names")
    parser.add_argument("--compute_lambda", type=float, default=0.0)
    parser.add_argument("--normalize", dest="normalize", action="store_true", default=True)
    parser.add_argument("--no_normalize", dest="normalize", action="store_false")
    parser.add_argument("--inspect_only", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    profile(args)


if __name__ == "__main__":
    main()
