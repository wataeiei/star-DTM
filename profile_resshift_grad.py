#!/usr/bin/env python3
"""Profile packed-qkv LoRA gradients with ResShift's official objective.

Copy this script and ``inspect_resshift_structure.py`` to the official
ResShift repository root before running it there.  The output schema matches
the importance tables used by the existing threshold-selection tools.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
import sys
from pathlib import Path

from PIL import Image, ImageFile

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from inspect_resshift_structure import install_timm_layers_stub, load_checkpoint


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
ImageFile.LOAD_TRUNCATED_IMAGES = True


def natural_key(text: str) -> list[int | str]:
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", text)]


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_under_root(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def noise_ratio_to_timestep(noise_ratio: float, num_timesteps: int) -> int:
    if not 0.0 <= noise_ratio <= 1.0:
        raise ValueError(f"Noise ratio must be in [0, 1], got {noise_ratio}")
    if num_timesteps < 1:
        raise ValueError(f"num_timesteps must be positive, got {num_timesteps}")
    return min(num_timesteps - 1, max(0, round(noise_ratio * (num_timesteps - 1))))


def block_from_qkv(name: str) -> str:
    marker = ".attn.qkv"
    if not name.endswith(marker):
        raise ValueError(f"Not a packed Swin qkv module: {name}")
    return name[: -len(marker)]


def split_parent_name(root: nn.Module, dotted_name: str) -> tuple[nn.Module, str]:
    parts = dotted_name.split(".")
    parent = root
    for part in parts[:-1]:
        parent = parent[int(part)] if part.isdigit() else getattr(parent, part)
    return parent, parts[-1]


class ImageFolderDataset(Dataset):
    def __init__(self, root: str | Path, image_size: int, scale: int, max_images: int) -> None:
        root = Path(root)
        self.paths = sorted(
            (path for path in root.rglob("*") if path.suffix.lower() in IMAGE_EXTS),
            key=lambda path: natural_key(str(path)),
        )
        if max_images > 0:
            self.paths = self.paths[:max_images]
        if not self.paths:
            raise FileNotFoundError(f"No images found under {root}")
        if image_size % scale:
            raise ValueError(f"image_size={image_size} must be divisible by scale={scale}")
        self.image_size = image_size
        self.lq_size = image_size // scale

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        path = self.paths[index]
        with Image.open(path) as image:
            image = image.convert("RGB").resize(
                (self.image_size, self.image_size), Image.Resampling.BICUBIC
            )
            data = torch.frombuffer(bytearray(image.tobytes()), dtype=torch.uint8)
        gt = data.view(self.image_size, self.image_size, 3).permute(2, 0, 1)
        gt = gt.float().div(127.5).sub(1.0)
        lq = F.interpolate(
            gt.unsqueeze(0),
            size=(self.lq_size, self.lq_size),
            mode="bicubic",
            align_corners=False,
            antialias=True,
        ).squeeze(0).clamp(-1.0, 1.0)
        return {"gt": gt, "lq": lq, "path": str(path)}


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int, alpha: float) -> None:
        super().__init__()
        self.base = base
        self.rank = rank
        self.scale = float(alpha) / rank
        self.lora_down = nn.Linear(base.in_features, rank, bias=False)
        self.lora_up = nn.Linear(rank, base.out_features, bias=False)
        self.lora_down.to(device=base.weight.device, dtype=torch.float32)
        self.lora_up.to(device=base.weight.device, dtype=torch.float32)
        nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_up.weight)
        self.base.requires_grad_(False)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        base = self.base(value)
        update = self.lora_up(self.lora_down(value.float())) * self.scale
        return base + update.to(dtype=base.dtype)


def inject_packed_qkv_lora(model: nn.Module, rank: int, alpha: float) -> dict[str, LoRALinear]:
    replacements = []
    for name, module in model.named_modules():
        if (
            isinstance(module, nn.Linear)
            and name.endswith(".attn.qkv")
            and module.out_features == 3 * module.in_features
        ):
            replacements.append((name, module))
    if not replacements:
        raise RuntimeError("No packed *.attn.qkv Linear modules were found")
    result = {}
    for name, module in replacements:
        parent, child = split_parent_name(model, name)
        wrapper = LoRALinear(module, rank, alpha)
        setattr(parent, child, wrapper)
        result[name] = wrapper
    return result


def load_state(module: nn.Module, path: Path, torch_module=torch) -> tuple[list[str], list[str]]:
    payload = torch_module.load(path, map_location="cpu")
    state = payload.get("state_dict", payload) if isinstance(payload, dict) else payload
    if not isinstance(state, dict):
        raise ValueError(f"Unsupported checkpoint payload: {path}")
    cleaned = {}
    for name, value in state.items():
        while name.startswith("module."):
            name = name[len("module.") :]
        cleaned[name] = value
    status = module.load_state_dict(cleaned, strict=False)
    return list(status.missing_keys), list(status.unexpected_keys)


def noise_sigma(diffusion, timestep: int) -> float:
    etas = getattr(diffusion, "etas", None)
    kappa = float(getattr(diffusion, "kappa", 1.0))
    if etas is None:
        return float("nan")
    return kappa * math.sqrt(float(etas[timestep]))


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def profile(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    root = Path(args.resshift_root).resolve()
    config_path = resolve_under_root(root, args.config_path)
    checkpoint = resolve_under_root(root, args.checkpoint)
    autoencoder_checkpoint = resolve_under_root(root, args.autoencoder_checkpoint)
    for path in (config_path, checkpoint, autoencoder_checkpoint):
        if not path.is_file():
            raise FileNotFoundError(path)
    sys.path.insert(0, str(root))

    from omegaconf import OmegaConf
    from utils.util_common import get_obj_from_str

    install_timm_layers_stub(torch)
    config = OmegaConf.load(config_path)
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")

    model = get_obj_from_str(config.model.target)(**config.model.get("params", {}))
    missing, unexpected = load_checkpoint(model, checkpoint, torch)
    if missing or unexpected:
        raise RuntimeError(
            f"Model checkpoint mismatch: missing={missing[:10]} unexpected={unexpected[:10]}"
        )
    model.to(device).requires_grad_(False)

    autoencoder = get_obj_from_str(config.autoencoder.target)(
        **config.autoencoder.get("params", {})
    )
    missing, unexpected = load_state(autoencoder, autoencoder_checkpoint)
    if missing or unexpected:
        raise RuntimeError(
            "Autoencoder checkpoint mismatch: "
            f"missing={missing[:10]} unexpected={unexpected[:10]}"
        )
    autoencoder.to(device).eval().requires_grad_(False)

    diffusion = get_obj_from_str(config.diffusion.target)(
        **config.diffusion.get("params", {})
    )
    injected = inject_packed_qkv_lora(model, args.rank, args.alpha)
    blocks = sorted((block_from_qkv(name) for name in injected), key=natural_key)
    block_index = {block: index for index, block in enumerate(blocks)}
    model.train()

    dataset = ImageFolderDataset(args.data_dir, args.image_size, args.sr_scale, args.max_images)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    batches = []
    for batch in loader:
        batches.append(batch)
        if len(batches) >= args.probe_batches:
            break
    if not batches:
        raise RuntimeError("No calibration batches were loaded")

    rows = []
    anchors = sorted(set(args.noise_ratios))
    for anchor_index, ratio in enumerate(anchors):
        timestep = noise_ratio_to_timestep(ratio, diffusion.num_timesteps)
        model.zero_grad(set_to_none=True)
        losses = []
        for batch_index, batch in enumerate(batches):
            gt = batch["gt"].to(device, non_blocking=True)
            lq = batch["lq"].to(device, non_blocking=True)
            tt = torch.full((gt.shape[0],), timestep, dtype=torch.long, device=device)
            generator = torch.Generator(device=device)
            generator.manual_seed(args.seed + anchor_index * 100000 + batch_index)
            latent_scale = 2 ** (len(config.autoencoder.params.ddconfig.ch_mult) - 1)
            latent_resolution = gt.shape[-1] // latent_scale
            noise = torch.randn(
                (gt.shape[0], int(config.autoencoder.params.embed_dim), latent_resolution, latent_resolution),
                generator=generator,
                device=device,
                dtype=gt.dtype,
            )
            result = diffusion.training_losses(
                model,
                gt,
                lq,
                tt,
                first_stage_model=autoencoder,
                model_kwargs={"lq": lq},
                noise=noise,
            )
            terms = result[0] if isinstance(result, (tuple, list)) else result
            loss = terms["mse"].mean()
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss at noise_ratio={ratio}, batch={batch_index}")
            (loss / len(batches)).backward()
            losses.append(float(loss.detach().cpu()))

        anchor_rows = []
        for module_name, module in injected.items():
            block = block_from_qkv(module_name)
            squared = 0.0
            parameter_count = 0
            for parameter in (module.lora_down.weight, module.lora_up.weight):
                parameter_count += parameter.numel()
                if parameter.grad is not None:
                    squared += float(parameter.grad.detach().float().square().sum().cpu())
            grad_norm = math.sqrt(squared)
            anchor_rows.append(
                {
                    "train_step": 0,
                    "noise_ratio": ratio,
                    "scheduler_index": timestep,
                    "timestep": timestep,
                    "sigma": noise_sigma(diffusion, timestep),
                    "block": block,
                    "block_index": block_index[block],
                    "grad_norm": grad_norm,
                    "lora_param_count": parameter_count,
                    "module_count": 1,
                    "normalized_grad_score": grad_norm / math.sqrt(parameter_count),
                    "update_norm": 0.0,
                    "normalized_update_score": 0.0,
                    "probe_batches": len(batches),
                    "mean_probe_loss": sum(losses) / len(losses),
                    "loss_mode": "official_resshift",
                    "importance_rank": 0,
                    "selected_topk": False,
                }
            )
        ranked = sorted(anchor_rows, key=lambda row: row["normalized_grad_score"], reverse=True)
        for rank, row in enumerate(ranked, start=1):
            row["importance_rank"] = rank
            row["selected_topk"] = rank <= args.report_topk
        rows.extend(sorted(anchor_rows, key=lambda row: row["block_index"]))
        print(
            f"noise_ratio={ratio:g} timestep={timestep}/{diffusion.num_timesteps - 1} "
            f"mean_loss={sum(losses) / len(losses):.6f}"
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    importance_path = output_dir / "lora_importance_evolution.csv"
    write_csv(importance_path, rows)
    metadata = {
        "resshift_root": str(root),
        "config_path": str(config_path),
        "ckpt_path": str(checkpoint),
        "autoencoder_ckpt": str(autoencoder_checkpoint),
        "data_dir": str(Path(args.data_dir).resolve()),
        "image_size": args.image_size,
        "sr_scale": args.sr_scale,
        "target": "packed_qkv",
        "rank": args.rank,
        "alpha": args.alpha,
        "loss_mode": "official_resshift",
        "candidate_block_count": len(blocks),
        "injected_module_count": len(injected),
        "noise_anchors": anchors,
        "probe_batches_per_anchor": len(batches),
        "seed": args.seed,
        "importance_csv": str(importance_path),
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Wrote {len(rows)} importance rows to {importance_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resshift_root", default=".")
    parser.add_argument("--config_path", default="configs/realsr_swinunet_realesrgan256.yaml")
    parser.add_argument("--checkpoint", default="weights/resshift_realsrx4_s15_v1.pth")
    parser.add_argument("--autoencoder_checkpoint", default="weights/autoencoder_vq_f4.pth")
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--sr_scale", type=int, default=4)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=16.0)
    parser.add_argument("--noise_ratios", type=float, nargs="+", default=[0.05, 0.2, 0.4, 0.6, 0.8, 0.95])
    parser.add_argument("--probe_batches", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_images", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--report_topk", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.rank < 1 or args.probe_batches < 1 or args.batch_size < 1:
        raise SystemExit("rank, probe_batches, and batch_size must be positive")
    if args.sr_scale < 1 or args.image_size < 1:
        raise SystemExit("sr_scale and image_size must be positive")
    profile(args)


if __name__ == "__main__":
    main()
