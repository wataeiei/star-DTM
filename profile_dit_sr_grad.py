#!/usr/bin/env python3
"""Gradient profiling for the official DiT-SR repository.

Copy this file into the DiT-SR repository root, then run it there:

  python3 profile_dit_sr_grad.py \
    --config_path configs/realsr_DiT.yaml \
    --ckpt_path weights/realsr.pth \
    --data_dir /mnt/disk1T/liyijuan/star-DTM/data/ucmerced/train_hr \
    --output_dir outputs/dit_sr_grad_profile_ucmerced \
    --target qkv \
    --probe_batches 20

The profiler inserts temporary probe LoRA modules into attention projection
Linear layers, measures gradient norms, and writes a CSV compatible with
plot_cross_model_grad_scores.py.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import math
import random
import re
import sys
import types
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


def instantiate_from_config(config):
    install_timm_layers_stub()
    install_torchvision_stub_for_timm()
    target = config.get("target")
    if not target:
        raise ValueError("Config section has no target field.")
    module_name, cls_name = target.rsplit(".", 1)
    cls = getattr(importlib.import_module(module_name), cls_name)
    params = config.get("params", {})
    return cls(**params)


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        return x.div(keep_prob) * random_tensor


def to_2tuple(x):
    return x if isinstance(x, tuple) else (x, x)


def trunc_normal_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0):
    return nn.init.trunc_normal_(tensor, mean=mean, std=std, a=a, b=b)


def install_timm_layers_stub() -> None:
    """Provide only the timm symbols DiT-SR imports.

    The full timm import pulls torchvision datasets/models during module init.
    On Jetson, torchvision often mismatches the NVIDIA PyTorch wheel. DiT-SR's
    Swin code only needs these three layer helpers, so a small stub is safer.
    """
    if "timm.models.layers" in sys.modules:
        return
    timm = types.ModuleType("timm")
    timm_models = types.ModuleType("timm.models")
    timm_layers = types.ModuleType("timm.models.layers")
    timm_layers.DropPath = DropPath
    timm_layers.to_2tuple = to_2tuple
    timm_layers.trunc_normal_ = trunc_normal_
    timm_models.layers = timm_layers
    timm.models = timm_models
    sys.modules["timm"] = timm
    sys.modules["timm.models"] = timm_models
    sys.modules["timm.models.layers"] = timm_layers


def install_torchvision_stub_for_timm() -> None:
    """Avoid importing an incompatible torchvision just for timm feature hooks.

    DiT-SR only needs timm.layers.DropPath/to_2tuple/trunc_normal_ through
    timm.models.layers. Recent timm imports torchvision feature_extraction at
    module import time, which can fail on Jetson when torchvision does not match
    the installed NVIDIA PyTorch wheel. A tiny stub is enough for this profiler.
    """
    if "torchvision" in sys.modules:
        return
    torchvision = types.ModuleType("torchvision")
    models = types.ModuleType("torchvision.models")
    feature_extraction = types.ModuleType("torchvision.models.feature_extraction")
    ops = types.ModuleType("torchvision.ops")
    ops_misc = types.ModuleType("torchvision.ops.misc")

    def create_feature_extractor(*args, **kwargs):
        raise RuntimeError("torchvision feature_extraction is not available in this profiling environment.")

    feature_extraction.create_feature_extractor = create_feature_extractor
    ops_misc.FrozenBatchNorm2d = nn.BatchNorm2d
    ops.misc = ops_misc
    models.feature_extraction = feature_extraction
    torchvision.models = models
    torchvision.ops = ops
    sys.modules["torchvision"] = torchvision
    sys.modules["torchvision.models"] = models
    sys.modules["torchvision.models.feature_extraction"] = feature_extraction
    sys.modules["torchvision.ops"] = ops
    sys.modules["torchvision.ops.misc"] = ops_misc


class ImageFolderDataset(Dataset):
    def __init__(self, root: str | Path, image_size: int, max_images: int = 0) -> None:
        root = Path(root)
        paths = [p for p in root.rglob("*") if p.suffix.lower() in IMAGE_EXTS]
        if not paths:
            raise FileNotFoundError(f"No images found under {root}")
        paths = sorted(paths)
        if max_images > 0:
            paths = paths[:max_images]
        self.paths = paths
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> dict:
        image = Image.open(self.paths[idx]).convert("RGB").resize((self.image_size, self.image_size), Image.BICUBIC)
        x = torch.frombuffer(bytearray(image.tobytes()), dtype=torch.uint8)
        x = x.view(self.image_size, self.image_size, 3).permute(2, 0, 1).float() / 255.0
        x = x * 2.0 - 1.0
        return {"image": x, "path": str(self.paths[idx])}


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int, alpha: int) -> None:
        super().__init__()
        self.base = base
        self.rank = rank
        self.alpha = alpha
        self.scale = alpha / rank
        self.lora_down = nn.Linear(base.in_features, rank, bias=False)
        self.lora_up = nn.Linear(rank, base.out_features, bias=False)
        self.lora_down.to(device=base.weight.device, dtype=torch.float32)
        self.lora_up.to(device=base.weight.device, dtype=torch.float32)
        nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_up.weight)
        for param in self.base.parameters():
            param.requires_grad_(False)

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


def target_match(name: str, target: str) -> bool:
    leaf = name.split(".")[-1]
    if target == "qkv":
        return leaf in {"qkv", "to_qkv", "q", "k", "v", "to_q", "to_k", "to_v", "q_proj", "k_proj", "v_proj"}
    if target == "qv":
        return leaf in {"qkv", "to_qkv", "q", "v", "to_q", "to_v", "q_proj", "v_proj"}
    if target == "q":
        return leaf in {"q", "to_q", "q_proj"}
    if target == "v":
        return leaf in {"v", "to_v", "v_proj"}
    if target == "all_linear":
        return True
    raise SystemExit(f"Unknown target={target}")


def block_key(name: str, block_regex: str = "") -> str:
    if block_regex:
        m = re.search(block_regex, name)
        if m:
            return m.group(1) if m.groups() else m.group(0)
    parts = name.split(".")
    if "input_blocks" in parts:
        i = parts.index("input_blocks")
        if i + 1 < len(parts):
            if "blocks" in parts:
                j = parts.index("blocks")
                if j + 1 < len(parts):
                    return f"input_blocks.{parts[i + 1]}.blocks.{parts[j + 1]}"
            return f"input_blocks.{parts[i + 1]}"
    if "middle_block" in parts:
        if "blocks" in parts:
            j = parts.index("blocks")
            if j + 1 < len(parts):
                return f"middle_block.blocks.{parts[j + 1]}"
        return "middle_block"
    if "output_blocks" in parts:
        i = parts.index("output_blocks")
        if i + 1 < len(parts):
            if "blocks" in parts:
                j = parts.index("blocks")
                if j + 1 < len(parts):
                    return f"output_blocks.{parts[i + 1]}.blocks.{parts[j + 1]}"
            return f"output_blocks.{parts[i + 1]}"
    return ""


def natural_key(text: str) -> list:
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", text)]


def iter_lora_modules(root: nn.Module) -> Iterable[tuple[str, LoRALinear]]:
    for name, module in root.named_modules():
        if isinstance(module, LoRALinear):
            yield name, module


def inject_lora(root: nn.Module, target: str, rank: int, alpha: int, block_regex: str) -> list[str]:
    replacements = []
    for name, module in root.named_modules():
        if isinstance(module, nn.Linear) and block_key(name, block_regex) and target_match(name, target):
            replacements.append((name, module))
    if not replacements:
        raise SystemExit("No target Linear modules found. Run --inspect_only to check module names.")
    for name, module in replacements:
        parent, child_name = split_parent_name(root, name)
        setattr(parent, child_name, LoRALinear(module, rank=rank, alpha=alpha))
    return [name for name, _ in replacements]


def lora_grad_norm(module: LoRALinear) -> float:
    total = 0.0
    for param in (module.lora_down.weight, module.lora_up.weight):
        if param.grad is not None:
            total += float(param.grad.detach().float().pow(2).sum().cpu())
    return math.sqrt(total)


def load_model(args: argparse.Namespace, device: torch.device) -> nn.Module:
    try:
        from omegaconf import OmegaConf
    except ImportError as exc:
        raise SystemExit("Missing package: omegaconf\nInstall with: pip3 install omegaconf") from exc

    sys.path.insert(0, str(Path.cwd()))
    config = OmegaConf.load(args.config_path)
    model = instantiate_from_config(config.model)
    ckpt = torch.load(args.ckpt_path, map_location="cpu")
    state = ckpt
    for key in ("state_dict", "model", "params", "ema"):
        if isinstance(state, dict) and key in state and isinstance(state[key], dict):
            state = state[key]
            break
    if isinstance(state, dict):
        cleaned = {}
        for key, value in state.items():
            new_key = key
            for prefix in ("module.", "model.", "model_ema."):
                if new_key.startswith(prefix):
                    new_key = new_key[len(prefix) :]
            cleaned[new_key] = value
        missing, unexpected = model.load_state_dict(cleaned, strict=False)
        print(f"Loaded checkpoint: missing={len(missing)} unexpected={len(unexpected)}")
        if missing:
            print("First missing keys:", missing[:10])
        if unexpected:
            print("First unexpected keys:", unexpected[:10])
    else:
        raise SystemExit(f"Unsupported checkpoint format: {type(ckpt)}")
    model.to(device)
    model.eval()
    return model


def _load_state(module: nn.Module, path: str | Path, label: str) -> None:
    ckpt = torch.load(path, map_location="cpu")
    state = ckpt
    for key in ("state_dict", "model", "params", "ema"):
        if isinstance(state, dict) and key in state and isinstance(state[key], dict):
            state = state[key]
            break
    if not isinstance(state, dict):
        raise SystemExit(f"Unsupported {label} checkpoint format: {type(ckpt)}")
    cleaned = {}
    for key, value in state.items():
        new_key = key
        for prefix in ("module.", "model.", "model_ema.", "autoencoder."):
            if new_key.startswith(prefix):
                new_key = new_key[len(prefix) :]
        cleaned[new_key] = value
    missing, unexpected = module.load_state_dict(cleaned, strict=False)
    print(f"Loaded {label}: missing={len(missing)} unexpected={len(unexpected)}")
    if missing:
        print(f"First missing {label} keys:", missing[:10])
    if unexpected:
        print(f"First unexpected {label} keys:", unexpected[:10])


def load_official_objective(args: argparse.Namespace, device: torch.device):
    """Load the diffusion process and first-stage model used by DiT-SR training."""
    if args.loss_mode != "official":
        return None, None
    from omegaconf import OmegaConf

    config = OmegaConf.load(args.config_path)
    diffusion = instantiate_from_config(config.diffusion)
    autoencoder = instantiate_from_config(config.autoencoder)
    ae_path = args.autoencoder_ckpt or str(config.autoencoder.get("ckpt_path", ""))
    if not ae_path:
        raise SystemExit("Official DiT-SR loss requires --autoencoder_ckpt or autoencoder.ckpt_path.")
    _load_state(autoencoder, ae_path, "autoencoder")
    autoencoder.to(device).eval().requires_grad_(False)
    print(
        "Using official DiT-SR diffusion objective: "
        f"predict_type={getattr(diffusion, 'predict_type', 'config-defined')}"
    )
    return diffusion, autoencoder


def diffusion_batch_loss(
    model: nn.Module,
    diffusion,
    autoencoder: nn.Module | None,
    batch: dict,
    args: argparse.Namespace,
    device: torch.device,
) -> torch.Tensor:
    image = batch["image"].to(device)
    lq = F.interpolate(
        image, size=(args.lq_size, args.lq_size), mode="bicubic", align_corners=False
    )
    if args.loss_mode == "proxy":
        timesteps = torch.full(
            (image.shape[0],), args.timestep, device=device, dtype=torch.long
        )
        output = model(image, timesteps, lq=lq)
        return (
            F.mse_loss(output.float(), image.float())
            if output.shape == image.shape
            else output.float().pow(2).mean()
        )

    timesteps = torch.randint(
        0, diffusion.num_timesteps, (image.shape[0],), device=device, dtype=torch.long
    )
    model_kwargs = {"lq": lq}
    result = diffusion.training_losses(
        model,
        image,
        lq,
        timesteps,
        first_stage_model=autoencoder,
        model_kwargs=model_kwargs,
        noise=None,
    )
    losses = result[0] if isinstance(result, (tuple, list)) else result
    if not isinstance(losses, dict):
        raise RuntimeError(f"Unexpected DiT-SR training_losses result: {type(losses)}")
    loss = losses.get("mse", losses.get("loss"))
    if loss is None:
        raise RuntimeError(f"DiT-SR loss dictionary has no mse/loss key: {list(losses)}")
    return loss.mean()


def inspect_model(model: nn.Module, args: argparse.Namespace) -> None:
    print("== Attention/BasicLayer-like modules ==")
    for name, module in model.named_modules():
        cls = module.__class__.__name__
        if any(key in cls.lower() for key in ("basiclayer", "attention", "swin", "block")):
            print(name, cls)
    print("\n== Candidate Linear modules ==")
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            bkey = block_key(name, args.block_regex)
            mark = "*" if bkey and target_match(name, args.target) else " "
            print(f"{mark} {name} [{module.in_features}->{module.out_features}] block={bkey}")


def write_csv(path: str | Path, rows: list[dict]) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def profile(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model = load_model(args, device)
    diffusion, autoencoder = load_official_objective(args, device)

    if args.inspect_only:
        inspect_model(model, args)
        return

    for param in model.parameters():
        param.requires_grad_(False)
    injected = inject_lora(model, args.target, args.rank, args.alpha, args.block_regex)
    model.train()

    dataset = ImageFolderDataset(args.data_dir, args.image_size, args.max_images)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    data_iter = iter(loader)

    total_loss = 0.0
    valid = 0
    model.zero_grad(set_to_none=True)
    for idx in range(1, args.probe_batches + 1):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch = next(data_iter)

        loss = diffusion_batch_loss(model, diffusion, autoencoder, batch, args, device)
        if not torch.isfinite(loss):
            print(f"probe batch {idx}/{args.probe_batches} skipped: non-finite loss")
            model.zero_grad(set_to_none=True)
            continue
        loss.backward()
        valid += 1
        total_loss += float(loss.detach().cpu())
        print(f"probe batch {idx:03d}/{args.probe_batches} loss={float(loss.detach().cpu()):.6f}")

    if valid == 0:
        raise SystemExit("No valid probe batches.")

    by_block: dict[str, dict] = {}
    for name, module in iter_lora_modules(model):
        bkey = block_key(name, args.block_regex)
        if not bkey:
            continue
        row = by_block.setdefault(bkey, {"grad_norm": 0.0, "lora_param_count": 0, "module_count": 0})
        row["grad_norm"] += lora_grad_norm(module)
        row["lora_param_count"] += module.lora_down.weight.numel() + module.lora_up.weight.numel()
        row["module_count"] += 1

    blocks = sorted(by_block, key=natural_key)
    block_index = {block: idx for idx, block in enumerate(blocks)}
    total_blocks = max(len(blocks), 1)
    rows = []
    for block in blocks:
        row = by_block[block]
        p_count = int(row["lora_param_count"])
        norm_score = row["grad_norm"] / math.sqrt(max(p_count, 1))
        bp_cost = total_blocks - block_index[block]
        selection = norm_score / (p_count + args.compute_lambda * bp_cost) if args.compute_lambda > 0 else norm_score
        rows.append(
            {
                "block": block,
                "block_index": block_index[block],
                "grad_norm": row["grad_norm"],
                "lora_param_count": p_count,
                "module_count": int(row["module_count"]),
                "normalized_grad_score": norm_score,
                "bp_cost": bp_cost,
                "compute_lambda": args.compute_lambda,
                "selection_score": selection,
                "probe_batches": valid,
                "mean_probe_loss": total_loss / valid,
                "loss_mode": args.loss_mode,
                "selected": False,
            }
        )

    selected = set(r["block"] for r in sorted(rows, key=lambda r: r["selection_score"], reverse=True)[: args.topk_blocks])
    for row in rows:
        row["selected"] = row["block"] in selected

    out_dir = ensure_dir(args.output_dir)
    write_csv(out_dir / "dit_sr_grad_scores.csv", rows)
    metadata = {
        "config_path": args.config_path,
        "ckpt_path": args.ckpt_path,
        "data_dir": args.data_dir,
        "target": args.target,
        "rank": args.rank,
        "alpha": args.alpha,
        "topk_blocks": args.topk_blocks,
        "probe_batches": args.probe_batches,
        "seed": args.seed,
        "loss_mode": args.loss_mode,
        "selected_blocks": sorted(selected, key=natural_key),
        "injected_module_count": len(injected),
    }
    (out_dir / "dit_sr_grad_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Wrote {out_dir / 'dit_sr_grad_scores.csv'}")
    print(f"Wrote {out_dir / 'dit_sr_grad_metadata.json'}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config_path", default="configs/realsr_DiT.yaml")
    parser.add_argument("--ckpt_path", default="weights/realsr.pth")
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--lq_size", type=int, default=64)
    parser.add_argument("--loss_mode", default="official", choices=["official", "proxy"])
    parser.add_argument("--autoencoder_ckpt", default="")
    parser.add_argument("--max_images", type=int, default=0)
    parser.add_argument("--target", default="qkv", choices=["q", "v", "qv", "qkv", "all_linear"])
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--alpha", type=int, default=16)
    parser.add_argument("--topk_blocks", type=int, default=8)
    parser.add_argument("--probe_batches", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timestep", type=int, default=500)
    parser.add_argument("--input_noise_std", type=float, default=0.0)
    parser.add_argument("--compute_lambda", type=float, default=0.0)
    parser.add_argument("--block_regex", default="")
    parser.add_argument("--inspect_only", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    profile(args)


if __name__ == "__main__":
    main()
