#!/usr/bin/env python3
"""Standalone DiT-style SR gradient profiling.

This script does not require the DiT-SR or DiT4SR repositories. It builds a
small DiT-style super-resolution model locally, inserts temporary probe LoRA
modules into selected attention projections, and profiles layer importance on
UC Merced / SpaceNet-style image folders.

It is intended for quick mechanism validation:
  - Does a DiT-style SR model show stable early/late gradient patterns?
  - Do different calibration subsets / noise levels / seeds select similar blocks?

Example:
  python3 standalone_dit_sr_grad_profile.py \
    --data_dir data/ucmerced/train_hr \
    --output_dir outputs/standalone_dit_profile_noise20 \
    --hr_size 256 \
    --lr_size 64 \
    --depth 16 \
    --embed_dim 256 \
    --num_heads 8 \
    --target qv \
    --topk_blocks 8 \
    --probe_batches 20 \
    --noise_level 20 \
    --seed 42
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from pathlib import Path
from typing import Iterable

from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def list_images(root: str | Path) -> list[Path]:
    root = Path(root)
    return sorted([p for p in root.rglob("*") if p.suffix.lower() in IMAGE_EXTS])


def write_csv(path: str | Path, rows: list[dict]) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


class SRImageDataset(Dataset):
    def __init__(self, root: str | Path, hr_size: int, lr_size: int, max_images: int = 0) -> None:
        self.paths = list_images(root)
        if max_images > 0:
            self.paths = self.paths[:max_images]
        if not self.paths:
            raise SystemExit(f"No images found in {root}")
        self.hr_size = hr_size
        self.lr_size = lr_size

    def __len__(self) -> int:
        return len(self.paths)

    @staticmethod
    def to_tensor(image: Image.Image, size: int) -> torch.Tensor:
        image = image.convert("RGB").resize((size, size), Image.BICUBIC)
        x = torch.frombuffer(bytearray(image.tobytes()), dtype=torch.uint8)
        x = x.view(size, size, 3).permute(2, 0, 1).float() / 255.0
        return x

    def __getitem__(self, idx: int) -> dict:
        image = Image.open(self.paths[idx]).convert("RGB")
        hr = self.to_tensor(image, self.hr_size)
        lr_img = image.resize((self.lr_size, self.lr_size), Image.BICUBIC)
        lr = self.to_tensor(lr_img, self.lr_size)
        lr_up = F.interpolate(lr.unsqueeze(0), size=(self.hr_size, self.hr_size), mode="bicubic", align_corners=False)[0]
        return {"lr_up": lr_up, "hr": hr, "path": str(self.paths[idx])}


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


class DiTAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, n_tokens, dim = x.shape
        q = self.q(x).view(bsz, n_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k(x).view(bsz, n_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v(x).view(bsz, n_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).contiguous().view(bsz, n_tokens, dim)
        return self.proj(out)


class DiTBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float) -> None:
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.norm1 = nn.LayerNorm(dim)
        self.attn = DiTAttention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class TinyDiTSR(nn.Module):
    def __init__(
        self,
        image_size: int,
        patch_size: int,
        in_chans: int,
        embed_dim: int,
        depth: int,
        num_heads: int,
        mlp_ratio: float,
    ) -> None:
        super().__init__()
        if image_size % patch_size != 0:
            raise ValueError("hr_size must be divisible by patch_size")
        self.image_size = image_size
        self.patch_size = patch_size
        self.grid_size = image_size // patch_size
        self.num_patches = self.grid_size * self.grid_size
        self.patch_embed = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
        self.blocks = nn.ModuleList([DiTBlock(embed_dim, num_heads, mlp_ratio) for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, patch_size * patch_size * 3)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        bsz, n_tokens, patch_dim = x.shape
        p = self.patch_size
        h = w = self.grid_size
        x = x.view(bsz, h, w, 3, p, p)
        x = x.permute(0, 3, 1, 4, 2, 5).contiguous()
        return x.view(bsz, 3, h * p, w * p)

    def forward(self, lr_up: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(lr_up)
        x = x.flatten(2).transpose(1, 2)
        x = x + self.pos_embed
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        patches = self.head(x)
        residual = self.unpatchify(patches)
        return (lr_up + residual).clamp(0.0, 1.0)


def split_parent_name(root: nn.Module, dotted_name: str) -> tuple[nn.Module, str]:
    parts = dotted_name.split(".")
    parent = root
    for part in parts[:-1]:
        parent = parent[int(part)] if part.isdigit() else getattr(parent, part)
    return parent, parts[-1]


def block_key(name: str) -> str:
    parts = name.split(".")
    if "blocks" in parts:
        idx = parts.index("blocks")
        if idx + 1 < len(parts):
            return ".".join(parts[: idx + 2])
    return ""


def target_leafs(target: str) -> set[str]:
    mapping = {
        "q": {"q"},
        "v": {"v"},
        "qv": {"q", "v"},
        "qkv": {"q", "k", "v"},
        "qkvo": {"q", "k", "v", "proj"},
    }
    if target not in mapping:
        raise SystemExit(f"Unknown target={target}")
    return mapping[target]


def iter_lora_modules(root: nn.Module) -> Iterable[tuple[str, LoRALinear]]:
    for name, module in root.named_modules():
        if isinstance(module, LoRALinear):
            yield name, module


def inject_probe_lora(model: nn.Module, target: str, rank: int, alpha: int) -> list[str]:
    leafs = target_leafs(target)
    replacements = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and block_key(name) and name.split(".")[-1] in leafs:
            replacements.append((name, module))
    if not replacements:
        raise SystemExit(f"No LoRA targets found for target={target}")
    for name, module in replacements:
        parent, child_name = split_parent_name(model, name)
        setattr(parent, child_name, LoRALinear(module, rank=rank, alpha=alpha))
    return [name for name, _ in replacements]


def lora_grad_norm(module: LoRALinear) -> float:
    total = 0.0
    for param in (module.lora_down.weight, module.lora_up.weight):
        if param.grad is not None:
            total += float(param.grad.detach().float().pow(2).sum().cpu())
    return math.sqrt(total)


def profile(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    dataset = SRImageDataset(args.data_dir, args.hr_size, args.lr_size, args.max_images)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    model = TinyDiTSR(
        image_size=args.hr_size,
        patch_size=args.patch_size,
        in_chans=3,
        embed_dim=args.embed_dim,
        depth=args.depth,
        num_heads=args.num_heads,
        mlp_ratio=args.mlp_ratio,
    ).to(device)
    model.train()
    for p in model.parameters():
        p.requires_grad_(False)
    injected = inject_probe_lora(model, args.target, args.rank, args.alpha)
    model.zero_grad(set_to_none=True)

    data_iter = iter(loader)
    valid_batches = 0
    total_loss = 0.0
    for idx in range(1, args.probe_batches + 1):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch = next(data_iter)
        lr_up = batch["lr_up"].to(device)
        hr = batch["hr"].to(device)
        if args.noise_level > 0:
            # noise_level is interpreted as pixel noise std in [0, 255].
            lr_up = (lr_up + torch.randn_like(lr_up) * (args.noise_level / 255.0)).clamp(0.0, 1.0)
        pred = model(lr_up)
        loss = F.l1_loss(pred.float(), hr.float())
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
        raise SystemExit("All probe batches failed.")

    block_rows: dict[str, dict] = {}
    for name, module in iter_lora_modules(model):
        block = block_key(name)
        row = block_rows.setdefault(
            block,
            {
                "block": block,
                "block_index": int(block.split(".")[-1]),
                "grad_sq": 0.0,
                "lora_param_count": 0,
                "module_count": 0,
            },
        )
        row["grad_sq"] += lora_grad_norm(module) ** 2
        row["lora_param_count"] += module.lora_down.weight.numel() + module.lora_up.weight.numel()
        row["module_count"] += 1

    total_blocks = len(block_rows)
    rows = []
    for block, row in block_rows.items():
        grad_norm = math.sqrt(row["grad_sq"])
        p_count = int(row["lora_param_count"])
        normalized = grad_norm / math.sqrt(max(p_count, 1))
        bp_cost = total_blocks - int(row["block_index"])
        selection_score = normalized
        if args.compute_lambda > 0:
            selection_score = normalized / (p_count + args.compute_lambda * bp_cost)
        rows.append(
            {
                "block": block,
                "block_index": int(row["block_index"]),
                "grad_norm": grad_norm,
                "lora_param_count": p_count,
                "module_count": int(row["module_count"]),
                "normalized_grad_score": normalized,
                "bp_cost": bp_cost,
                "compute_lambda": args.compute_lambda,
                "selection_score": selection_score,
                "probe_batches": valid_batches,
                "mean_probe_loss": total_loss / valid_batches,
                "selected": False,
            }
        )

    selected = set([row["block"] for row in sorted(rows, key=lambda r: r["selection_score"], reverse=True)[: args.topk_blocks]])
    for row in rows:
        row["selected"] = row["block"] in selected
    rows.sort(key=lambda row: row["block_index"])

    out = ensure_dir(args.output_dir)
    write_csv(out / "dit_grad_topk_scores.csv", rows)
    metadata = {
        "model": "standalone_tiny_dit_sr",
        "data_dir": args.data_dir,
        "hr_size": args.hr_size,
        "lr_size": args.lr_size,
        "patch_size": args.patch_size,
        "depth": args.depth,
        "embed_dim": args.embed_dim,
        "num_heads": args.num_heads,
        "target": args.target,
        "rank": args.rank,
        "alpha": args.alpha,
        "topk_blocks": args.topk_blocks,
        "probe_batches": args.probe_batches,
        "noise_level": args.noise_level,
        "seed": args.seed,
        "selected_blocks": sorted(selected, key=lambda x: int(x.split(".")[-1])),
        "injected_module_count": len(injected),
    }
    (out / "dit_grad_topk_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Wrote {out / 'dit_grad_topk_scores.csv'}")
    print(f"Wrote {out / 'dit_grad_topk_metadata.json'}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--hr_size", type=int, default=256)
    parser.add_argument("--lr_size", type=int, default=64)
    parser.add_argument("--patch_size", type=int, default=16)
    parser.add_argument("--depth", type=int, default=16)
    parser.add_argument("--embed_dim", type=int, default=256)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--mlp_ratio", type=float, default=4.0)
    parser.add_argument("--target", default="qv", choices=["q", "v", "qv", "qkv", "qkvo"])
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--alpha", type=int, default=16)
    parser.add_argument("--topk_blocks", type=int, default=8)
    parser.add_argument("--probe_batches", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--noise_level", type=float, default=0.0)
    parser.add_argument("--compute_lambda", type=float, default=0.0)
    parser.add_argument("--max_images", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    profile(args)


if __name__ == "__main__":
    main()
