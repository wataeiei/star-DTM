#!/usr/bin/env python3
"""All-LoRA vs Grad-BlockSkip training on the Stable Diffusion x4 Upscaler.

This standalone script mirrors the DiT-SR / DiT4SR ``train_*_all_lora_importance``
trainers but targets the on-board SD-x4 Upscaler
(``stabilityai/stable-diffusion-x4-upscaler``). It lets you run two configurations
and compare whether residual-cache Grad-BlockSkip actually accelerates training:

  * ``--method all_lora``      : train LoRA on every attention projection
      (All-LoRA baseline). Optional per-noise-ratio importance profiling.
  * ``--method grad_blockskip``: All-LoRA + skip low-gradient transformer blocks
      and replay their residuals (two-pass teacher cache or single-pass bypass),
      intended to reduce peak memory and per-step time.

The block-skip machinery is reused from ``adaptive_grad_blockskip`` which is
model-agnostic; the SD-x4 specific parts (pipeline load, diffusion loss, low-res
conditioning, UNet block naming) are implemented here.

Growing evidence from DiT-SR / DiT4SR says the important-layer ordering shifts with
the noise ratio, so the profiler records LoRA gradient importance at several
normalized timesteps and block skipping is noise-aware.

Example All-LoRA baseline:
  python train_sdx4_all_lora_blockskip.py \
    --data_dir data/ucmerced/train_hr \
    --output_dir outputs/sdx4_all_lora \
    --method all_lora --train_steps 200

Example Grad-BlockSkip:
  python train_sdx4_all_lora_blockskip.py \
    --data_dir data/ucmerced/train_hr \
    --output_dir outputs/sdx4_blockskip \
    --method grad_blockskip --train_steps 200 \
    --blockskip_count 6 --blockskip_min_run 2 --blockskip_max_run 4 --blockskip_max_runs 2 \
    --profile_noise_ratios 0.05 0.2 0.4 0.6 0.8 0.95 \
    --train_noise_ratios 0.05 0.2 0.4 0.6 0.8 0.95

Before a long run, inspect module / candidate-block naming:
  python train_sdx4_all_lora_blockskip.py --inspect_only
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
import time
from pathlib import Path

from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import adaptive_grad_blockskip as adaptive


MODEL_ID = "stabilityai/stable-diffusion-x4-upscaler"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


# --------------------------------------------------------------------------- #
# Small shared helpers (kept local so the script is self-contained).
# --------------------------------------------------------------------------- #
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
    return sorted(p for p in root.rglob("*") if p.suffix.lower() in IMAGE_EXTS)


def write_csv(path: str | Path, rows: list[dict]) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def natural_key(text: str) -> list[int | str]:
    return [
        int(part) if part.isdigit() else part
        for part in re.split(r"(\d+)", text)
        if part
    ]


def load_pipeline(dtype: torch.dtype):
    from diffusers import StableDiffusionUpscalePipeline

    return StableDiffusionUpscalePipeline.from_pretrained(MODEL_ID, torch_dtype=dtype)


def get_prompt_embeds(pipe, batch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    tokens = pipe.tokenizer(
        [""] * batch_size,
        padding="max_length",
        max_length=pipe.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    ).input_ids.to(device)
    return pipe.text_encoder(tokens)[0].to(dtype=dtype)


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
class Sdx4HrDataset(Dataset):
    """HR images in [-1, 1]; the LR conditioning is built inside the loss."""

    def __init__(self, root: str | Path, image_size: int, max_images: int = 0) -> None:
        self.paths = list_images(root)
        if max_images > 0:
            self.paths = self.paths[:max_images]
        if not self.paths:
            raise SystemExit(f"No images found in {root}")
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> dict:
        image = Image.open(self.paths[idx]).convert("RGB").resize(
            (self.image_size, self.image_size), Image.BICUBIC
        )
        arr = torch.frombuffer(bytearray(image.tobytes()), dtype=torch.uint8)
        arr = arr.view(self.image_size, self.image_size, 3).permute(2, 0, 1).float() / 255.0
        arr = arr * 2.0 - 1.0
        return {"image": arr.contiguous(), "path": str(self.paths[idx])}


# --------------------------------------------------------------------------- #
# LoRA + block naming (SD-x4 UNet, robust to diffusers naming variants)
# --------------------------------------------------------------------------- #
class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int, alpha: int) -> None:
        super().__init__()
        self.base = base
        self.rank = rank
        self.alpha = alpha
        self.scale = alpha / rank
        self.lora_down = nn.Linear(base.in_features, rank, bias=False)
        self.lora_up = nn.Linear(rank, base.out_features, bias=False)
        device = base.weight.device
        self.lora_down.to(device=device, dtype=torch.float32)
        self.lora_up.to(device=device, dtype=torch.float32)
        nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_up.weight)
        for p in self.base.parameters():
            p.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        lora_out = self.lora_up(self.lora_down(x.float())) * self.scale
        return base_out + lora_out.to(dtype=base_out.dtype)


# Typical diffusers block markers for SD-family UNets.
_KEY_MARKERS = (
    "down_blocks",
    "up_blocks",
    "mid_block",
    "input_blocks",
    "output_blocks",
    "middle_block",
)
_CONTAINER_MARKERS = ("transformer_blocks", "blocks")


def transformer_block_key(name: str, block_regex: str = "") -> str:
    """Return a logical block key pointing at a shape-preserving transformer block.

    The returned key is itself a valid module path (e.g.
    ``down_blocks.1.attentions.0.transformer_blocks.2``), which makes
    residual-cache wrapping straightforward and version-robust.
    """
    if block_regex:
        match = re.search(block_regex, name)
        if match:
            return match.group(1) if match.groups() else match.group(0)
    parts = name.split(".")
    for container in _CONTAINER_MARKERS:
        if container in parts:
            index = parts.index(container)
            if index + 1 < len(parts) and parts[index + 1].isdigit():
                return ".".join(parts[: index + 2])
    return ""


def target_suffixes(target: str) -> tuple[str, ...]:
    mapping = {
        "q": ("to_q",),
        "v": ("to_v",),
        "qv": ("to_q", "to_v"),
        "qkvo": ("to_q", "to_k", "to_v", "to_out.0"),
    }
    if target not in mapping:
        raise SystemExit(f"Unknown target={target}; choose one of {sorted(mapping)}")
    return mapping[target]


def split_parent_name(root: nn.Module, dotted_name: str) -> tuple[nn.Module, str]:
    parts = dotted_name.split(".")
    parent = root
    for part in parts[:-1]:
        parent = parent[int(part)] if part.isdigit() else getattr(parent, part)
    return parent, parts[-1]


def inject_lora(
    root: nn.Module,
    target: str,
    rank: int,
    alpha: int,
    block_regex: str = "",
    selected_blocks: set[str] | None = None,
) -> list[str]:
    suffixes = target_suffixes(target)
    replacements: list[tuple[str, nn.Linear]] = []
    for name, module in root.named_modules():
        block = transformer_block_key(name, block_regex)
        if (
            isinstance(module, nn.Linear)
            and block
            and any(name.endswith(suffix) for suffix in suffixes)
            and (selected_blocks is None or block in selected_blocks)
        ):
            replacements.append((name, module))
    if not replacements:
        raise SystemExit(
            f"No LoRA targets found for target={target}. Run with --inspect_only to see module names."
        )
    for name, module in replacements:
        parent, child_name = split_parent_name(root, name)
        setattr(parent, child_name, LoRALinear(module, rank=rank, alpha=alpha))
    return [name for name, _ in replacements]


def iter_lora_modules(root: nn.Module):
    for name, module in root.named_modules():
        if isinstance(module, LoRALinear):
            yield name, module


def lora_grad_norm(module: LoRALinear) -> float:
    total = 0.0
    for param in (module.lora_down.weight, module.lora_up.weight):
        if param.grad is not None:
            total += float(param.grad.detach().float().pow(2).sum().cpu())
    return math.sqrt(total)


def resolve_block_paths(lora_module_names, block_names, block_regex: str = "") -> dict[str, str]:
    """Map logical transformer-block keys to concrete module paths.

    Since the logical key is already a module path (the transformer-block
    container), we confirm it by checking that some LoRA module lives inside.
    """
    wanted = set(block_names)
    result: dict[str, str] = {}
    for module_name in lora_module_names:
        block = transformer_block_key(module_name, block_regex)
        if block in wanted:
            result[block] = block
    missing = sorted(wanted - set(result))
    if missing:
        raise ValueError(
            "Could not resolve UNet modules for importance blocks: " + ", ".join(missing)
        )
    return result


# --------------------------------------------------------------------------- #
# SD-x4 diffusion loss with noise-ratio timestep control
# --------------------------------------------------------------------------- #
def sdx4_diffusion_loss(pipe, batch: dict, args: argparse.Namespace, device: torch.device, dtype: torch.dtype):
    image = batch["image"].to(device=device, dtype=dtype)
    bsz = image.shape[0]
    with torch.no_grad():
        latents = pipe.vae.encode(image).latent_dist.sample()
        latents = latents * pipe.vae.config.scaling_factor
        noise = torch.randn_like(latents)

        n_ts = pipe.scheduler.config.num_train_timesteps
        profile_ratio = getattr(args, "_profile_noise_ratio", None)
        if profile_ratio is None:
            timesteps = torch.randint(0, n_ts, (bsz,), device=device, dtype=torch.long)
        else:
            step = int(round(float(profile_ratio) * (n_ts - 1)))
            timesteps = torch.full((bsz,), step, device=device, dtype=torch.long)
        noisy_latents = pipe.scheduler.add_noise(latents, noise, timesteps)

        noise_level = torch.full((bsz,), args.noise_level, device=device, dtype=torch.long)
        lr_size = max(1, image.shape[-2] // args.sr_scale)
        lr_cond_clean = F.interpolate(
            image.float(), size=(lr_size, lr_size), mode="bicubic", align_corners=False
        )
        lr_cond_clean = F.interpolate(
            lr_cond_clean,
            size=noisy_latents.shape[-2:],
            mode="bicubic",
            align_corners=False,
        ).to(dtype=dtype)
        low_noise = torch.randn_like(lr_cond_clean)
        if hasattr(pipe, "low_res_scheduler"):
            lr_cond = pipe.low_res_scheduler.add_noise(lr_cond_clean, low_noise, noise_level)
        else:
            lr_cond = lr_cond_clean
        model_input = torch.cat([noisy_latents, lr_cond], dim=1)
        prompt_embeds = get_prompt_embeds(pipe, bsz, device, dtype)

        prediction_type = getattr(pipe.scheduler.config, "prediction_type", "epsilon")
        if prediction_type == "v_prediction":
            target = pipe.scheduler.get_velocity(latents, noise, timesteps)
        else:
            target = noise

    pred = pipe.unet(
        model_input, timesteps, prompt_embeds, class_labels=noise_level
    ).sample
    return F.mse_loss(pred.float(), target.float())


# --------------------------------------------------------------------------- #
# Noise-aware importance profiling (mirrors the DiT trainers)
# --------------------------------------------------------------------------- #
def profile_importance(pipe, loader, args, device, dtype, train_step, noise_ratio):
    cpu_state = torch.random.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if device.type == "cuda" else None
    python_state = random.getstate()
    set_seed(args.profile_seed)
    args._profile_noise_ratio = noise_ratio
    pipe.unet.zero_grad(set_to_none=True)
    iterator = iter(loader)
    valid = 0
    total_loss = 0.0
    try:
        for _ in range(args.profile_batches):
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                batch = next(iterator)
            batch, _patch = adaptive.dynamic_patch_batch(
                batch, noise_ratio, args.patch_min_fraction, args.patch_max_fraction
            )
            loss = sdx4_diffusion_loss(pipe, batch, args, device, dtype)
            if not torch.isfinite(loss):
                pipe.unet.zero_grad(set_to_none=True)
                continue
            loss.backward()
            valid += 1
            total_loss += float(loss.detach().cpu())
        if valid == 0:
            raise SystemExit(f"No valid profile batches at step {train_step}.")

        by_block: dict[str, dict] = {}
        for name, module in iter_lora_modules(pipe.unet):
            block = transformer_block_key(name, args.block_regex)
            if not block:
                continue
            row = by_block.setdefault(
                block, {"grad_sq": 0.0, "update_sq": 0.0, "params": 0, "modules": 0}
            )
            row["grad_sq"] += lora_grad_norm(module) ** 2
            delta = module.lora_up.weight.detach().float() @ module.lora_down.weight.detach().float()
            row["update_sq"] += float(delta.pow(2).sum().cpu())
            row["params"] += module.lora_down.weight.numel() + module.lora_up.weight.numel()
            row["modules"] += 1

        blocks = sorted(by_block, key=natural_key)
        rows = []
        for index, block in enumerate(blocks):
            item = by_block[block]
            params = int(item["params"])
            grad_norm = math.sqrt(item["grad_sq"])
            update_norm = math.sqrt(item["update_sq"])
            rows.append(
                {
                    "train_step": train_step,
                    "noise_ratio": noise_ratio,
                    "timestep": round(noise_ratio * (pipe.scheduler.config.num_train_timesteps - 1)),
                    "block": block,
                    "block_index": index,
                    "grad_norm": grad_norm,
                    "lora_param_count": params,
                    "module_count": int(item["modules"]),
                    "normalized_grad_score": grad_norm / math.sqrt(max(params, 1)),
                    "update_norm": update_norm,
                    "normalized_update_score": update_norm / math.sqrt(max(params, 1)),
                    "probe_batches": valid,
                    "mean_probe_loss": total_loss / valid,
                }
            )
        ranked = sorted(rows, key=lambda row: row["normalized_grad_score"], reverse=True)
        ranks = {row["block"]: rank for rank, row in enumerate(ranked, start=1)}
        for row in rows:
            row["importance_rank"] = ranks[row["block"]]
            row["selected_topk"] = ranks[row["block"]] <= args.topk_blocks
        return rows
    finally:
        del args._profile_noise_ratio
        pipe.unet.zero_grad(set_to_none=True)
        random.setstate(python_state)
        torch.random.set_rng_state(cpu_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)


def topk_summary(rows):
    grouped: dict[tuple[int, float], list[dict]] = {}
    for row in rows:
        grouped.setdefault((int(row["train_step"]), float(row["noise_ratio"])), []).append(row)
    output = []
    for step, ratio in sorted(grouped):
        baseline_key = (step, min(r for s, r in grouped if s == step))
        baseline = {row["block"] for row in grouped[baseline_key] if bool(row["selected_topk"])}
        selected = {row["block"] for row in grouped[(step, ratio)] if bool(row["selected_topk"])}
        overlap = len(baseline & selected)
        output.append(
            {
                "train_step": step,
                "noise_ratio": ratio,
                "topk_blocks": ";".join(
                    row["block"]
                    for row in sorted(grouped[(step, ratio)], key=lambda row: row["importance_rank"])
                    if row["selected_topk"]
                ),
                "topk_overlap_count_vs_lowest_t": overlap,
                "topk_overlap_ratio_vs_lowest_t": overlap / max(len(baseline), 1),
                "topk_jaccard_vs_lowest_t": overlap / max(len(baseline | selected), 1),
            }
        )
    return output


def read_blockskip_importance(path, train_step):
    path = Path(path)
    with path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise SystemExit(f"Block-skip importance CSV is empty: {path}")
    required = {"block", "block_index", "normalized_grad_score"}
    missing = sorted(required - set(rows[0]))
    if missing:
        raise SystemExit(
            "Block-skip importance CSV is missing columns: " + ", ".join(missing)
        )
    available = sorted({int(row.get("train_step", 0)) for row in rows})
    if train_step not in available:
        raise SystemExit(
            f"Block-skip importance step {train_step} is unavailable; available steps: {available}"
        )
    selected = [
        dict(row, train_step=0)
        for row in rows
        if int(row.get("train_step", 0)) == train_step
    ]
    for row in selected:
        row.setdefault("noise_ratio", 0.5)
    keys = [(row["block"], float(row["noise_ratio"])) for row in selected]
    if len(keys) != len(set(keys)):
        raise SystemExit(
            "Block-skip importance CSV contains duplicate block/noise rows "
            f"at train step {train_step}."
        )
    return selected


def inspect(pipe, args) -> None:
    print(f"== Model: {MODEL_ID} ==")
    print("== Attention-like blocks ==")
    seen = set()
    for name, module in pipe.unet.named_modules():
        cls = module.__class__.__name__
        if any(k in cls.lower() for k in ("transformerblock", "crossattn", "attention", "upblock", "downblock", "midblock")):
            idx = name.rfind(".")
            short = name[idx + 1:] if idx >= 0 else name
            if short not in seen:
                seen.add(short)
            print(name, cls)
    print("\n== Candidate LoRA Linear modules (target=%s) ==" % args.target)
    suffixes = target_suffixes(args.target)
    for name, module in pipe.unet.named_modules():
        if isinstance(module, nn.Linear) and any(name.endswith(s) for s in suffixes):
            bkey = transformer_block_key(name, args.block_regex)
            print(f"  {name} [{module.in_features}->{module.out_features}] block={bkey}")
    print("\n== Unique transformer-block keys ==")
    keys = {
        transformer_block_key(n, args.block_regex)
        for n, m in pipe.unet.named_modules()
        if isinstance(m, nn.Linear) and transformer_block_key(n, args.block_regex)
    }
    for key in sorted(keys, key=natural_key):
        print("  " + key)


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_id", default=MODEL_ID)
    parser.add_argument("--method", choices=["all_lora", "grad_blockskip"], default="all_lora")
    parser.add_argument("--data_dir", default="data/ucmerced/train_hr")
    parser.add_argument("--output_dir", default="outputs/sdx4_all_lora")
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--max_images", type=int, default=0)
    parser.add_argument("--sr_scale", type=int, default=4)
    parser.add_argument("--noise_level", type=int, default=20)
    parser.add_argument("--target", default="qv", choices=["q", "v", "qv", "qkvo"])
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--alpha", type=int, default=16)
    parser.add_argument("--train_steps", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--profile_seed", type=int, default=2026)
    parser.add_argument("--dtype", default="fp32", choices=["fp32", "fp16", "bf16"])
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--checkpoint_every", type=int, default=0)

    # importance profiling
    parser.add_argument("--profile_steps", type=int, nargs="+", default=[0, 50, 100, 150, 200])
    parser.add_argument("--profile_batches", type=int, default=5)
    parser.add_argument(
        "--profile_noise_ratios", type=float, nargs="+",
        default=[0.05, 0.2, 0.4, 0.6, 0.8, 0.95],
    )
    parser.add_argument(
        "--train_noise_ratios", type=float, nargs="+", default=[],
        help="Discrete normalized timesteps sampled during training; defaults to profile ratios.",
    )
    parser.add_argument("--topk_blocks", type=int, default=8)
    parser.add_argument("--disable_profiling", action="store_true")
    parser.add_argument("--block_regex", default="")

    # block skipping
    parser.add_argument("--blockskip_count", type=int, default=0)
    parser.add_argument(
        "--blockskip_schedule", nargs="*", default=[], metavar="SIGMA:COUNT",
        help="Noise-aware skip counts, e.g. 0.05:8 0.4:4 0.95:8.",
    )
    parser.add_argument(
        "--blockskip_fraction_schedule", nargs="*", default=[], metavar="SIGMA:FRACTION",
    )
    parser.add_argument("--blockskip_min_run", type=int, default=2)
    parser.add_argument("--blockskip_max_run", type=int, default=4)
    parser.add_argument("--blockskip_max_runs", type=int, default=2)
    parser.add_argument("--fixed_skip_blocks", nargs="*", default=[])
    parser.add_argument("--always_skip_blocks", nargs="*", default=[])
    parser.add_argument("--blockskip_importance_csv", default="")
    parser.add_argument("--blockskip_importance_step", type=int, default=0)
    parser.add_argument("--protect_selected_lora_blocks", action="store_true")
    parser.add_argument("--residual_cache_device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--residual_cache_dtype", choices=["fp16", "bf16", "fp32"], default="fp16")
    parser.add_argument(
        "--residual_execution", choices=["two_pass", "single_pass"], default="two_pass",
    )
    parser.add_argument("--patch_min_fraction", type=float, default=1.0)
    parser.add_argument("--patch_max_fraction", type=float, default=1.0)
    parser.add_argument("--inspect_only", action="store_true")
    args = parser.parse_args()

    if args.inspect_only:
        dtype = torch.float32
        device = torch.device("cpu")
        pipe = load_pipeline(dtype=dtype).to(device)
        pipe.unet.eval()
        inspect(pipe, args)
        return

    try:
        blockskip_schedule = adaptive.parse_noise_int_schedule(args.blockskip_schedule)
        blockskip_fraction_schedule = adaptive.parse_noise_float_schedule(
            args.blockskip_fraction_schedule
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    if blockskip_schedule and blockskip_fraction_schedule:
        raise SystemExit("Use either --blockskip_schedule or --blockskip_fraction_schedule, not both.")
    if args.fixed_skip_blocks and args.always_skip_blocks:
        raise SystemExit("Use either --fixed_skip_blocks or --always_skip_blocks, not both.")
    if not 0.0 < args.patch_min_fraction <= args.patch_max_fraction <= 1.0:
        raise SystemExit("Require 0 < --patch_min_fraction <= --patch_max_fraction <= 1.")
    if args.checkpoint_every < 0:
        raise SystemExit("--checkpoint_every must be non-negative.")

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    dtype = {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }[args.dtype]
    pipe = load_pipeline(dtype=dtype).to(device)
    pipe.vae.eval().requires_grad_(False)
    pipe.text_encoder.eval().requires_grad_(False)
    if not hasattr(pipe, "low_res_scheduler"):
        print("NOTE: pipeline has no low_res_scheduler; using clean LR conditioning.")
    pipe.unet.requires_grad_(False)

    target_module_names = [
        name
        for name, module in pipe.unet.named_modules()
        if isinstance(module, nn.Linear)
        and transformer_block_key(name, args.block_regex)
        and any(name.endswith(s) for s in target_suffixes(args.target))
    ]
    if not target_module_names:
        raise SystemExit(
            "No candidate LoRA modules found. Run with --inspect_only to check UNet naming."
        )
    candidate_blocks = sorted(
        {transformer_block_key(n, args.block_regex) for n in target_module_names},
        key=natural_key,
    )
    injected = inject_lora(
        pipe.unet,
        args.target,
        args.rank,
        args.alpha,
        args.block_regex,
        selected_blocks=None,
    )
    pipe.unet.train()
    print(
        f"All-LoRA: blocks={len(candidate_blocks)} modules={len(injected)} "
        f"rank={args.rank} alpha={args.alpha}"
    )

    dataset = Sdx4HrDataset(args.data_dir, args.image_size, args.max_images)
    train_loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers
    )
    profile_loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )
    params = [p for p in pipe.unet.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr)

    output_dir = ensure_dir(args.output_dir)
    experiment_start = time.perf_counter()
    profiling_time_s = 0.0
    checkpoint_time_s = 0.0
    max_cuda_mem_mb = 0.0
    max_cuda_reserved_mb = 0.0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    profile_steps = {s for s in args.profile_steps if 0 <= s <= args.train_steps}
    profile_steps.update({0, args.train_steps})
    if args.disable_profiling:
        profile_steps.clear()
    for ratio in args.profile_noise_ratios:
        if not 0.0 <= ratio <= 1.0:
            raise SystemExit("--profile_noise_ratios values must be in [0, 1].")

    importance_rows = []
    if not args.disable_profiling:
        for ratio in args.profile_noise_ratios:
            profile_start = time.perf_counter()
            importance_rows.extend(
                profile_importance(pipe, profile_loader, args, device, dtype, 0, ratio)
            )
            if device.type == "cuda":
                torch.cuda.synchronize()
                max_cuda_mem_mb = max(max_cuda_mem_mb, torch.cuda.max_memory_allocated() / (1024.0 ** 2))
                max_cuda_reserved_mb = max(max_cuda_reserved_mb, torch.cuda.max_memory_reserved() / (1024.0 ** 2))
            profiling_time_s += time.perf_counter() - profile_start

    block_names = list(candidate_blocks)
    blockskip_importance_rows = importance_rows
    if args.blockskip_importance_csv:
        blockskip_importance_rows = read_blockskip_importance(
            args.blockskip_importance_csv, args.blockskip_importance_step
        )
    if args.protect_selected_lora_blocks and not args.blockskip_importance_csv:
        raise SystemExit(
            "--protect_selected_lora_blocks requires --blockskip_importance_csv "
            "from a full-model probe."
        )

    cache_dtype = {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
    }[args.residual_cache_dtype]

    enable_blockskip = (
        args.method == "grad_blockskip"
        and (
            args.blockskip_count > 0
            or blockskip_schedule
            or blockskip_fraction_schedule
            or args.fixed_skip_blocks
            or args.always_skip_blocks
        )
    )
    if args.method == "grad_blockskip" and not enable_blockskip:
        raise SystemExit(
            "--method grad_blockskip needs a skip trigger: --blockskip_count, "
            "--blockskip_schedule, --blockskip_fraction_schedule, "
            "--fixed_skip_blocks, or --always_skip_blocks."
        )
    controller = None
    if enable_blockskip:
        configured = set(args.fixed_skip_blocks) | set(args.always_skip_blocks)
        unknown = sorted(configured - set(block_names))
        if unknown:
            raise SystemExit("Unknown explicitly configured blocks: " + ", ".join(unknown))
        if args.disable_profiling and not args.fixed_skip_blocks:
            raise SystemExit(
                "Dynamic block skipping requires profiling; remove --disable_profiling "
                "or use --fixed_skip_blocks."
            )
        block_paths = resolve_block_paths(target_module_names, block_names, args.block_regex)
        controller = adaptive.ResidualBlockController(
            pipe.unet,
            block_paths,
            cache_device=args.residual_cache_device,
            cache_dtype=cache_dtype,
        )
        print(
            f"Grad-BlockSkip enabled: count={args.blockskip_count} "
            f"min_run={args.blockskip_min_run} max_run={args.blockskip_max_run} "
            f"max_runs={args.blockskip_max_runs} exec={args.residual_execution}"
        )

    train_noise_ratios = args.train_noise_ratios or args.profile_noise_ratios
    if any(not 0.0 <= ratio <= 1.0 for ratio in train_noise_ratios):
        raise SystemExit("--train_noise_ratios values must be in [0, 1].")

    train_rows = []
    iterator = iter(train_loader)
    write_csv(output_dir / "lora_importance_evolution.csv", importance_rows)
    write_csv(output_dir / "lora_importance_topk.csv", topk_summary(importance_rows))

    for step in range(1, args.train_steps + 1):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            batch = next(iterator)
        noise_ratio = float(random.choice(train_noise_ratios))
        args._profile_noise_ratio = noise_ratio
        batch, patch_size = adaptive.dynamic_patch_batch(
            batch, noise_ratio, args.patch_min_fraction, args.patch_max_fraction
        )
        skip_blocks = []
        requested_skip_count = 0
        cache_stats = adaptive.CacheStats(0.0, 0.0, 0, 0)
        if controller is not None:
            if args.fixed_skip_blocks:
                requested_skip_count = len(args.fixed_skip_blocks)
                skip_blocks = list(args.fixed_skip_blocks)
            else:
                mandatory = list(dict.fromkeys(args.always_skip_blocks))
                if blockskip_fraction_schedule:
                    fraction = adaptive.noise_scheduled_float(
                        noise_ratio, blockskip_fraction_schedule, 0.0
                    )
                    scheduled_count = round(fraction * len(block_names))
                else:
                    scheduled_count = adaptive.noise_scheduled_int(
                        noise_ratio, blockskip_schedule, args.blockskip_count
                    )
                requested_skip_count = max(len(mandatory), scheduled_count)
                extra_count = requested_skip_count - len(mandatory)
                extras = (
                    adaptive.select_low_score_runs(
                        blockskip_importance_rows,
                        step,
                        noise_ratio,
                        extra_count,
                        args.blockskip_min_run,
                        args.blockskip_max_run,
                        args.blockskip_max_runs,
                        excluded_blocks=set(mandatory),
                    )
                    if extra_count > 0
                    else []
                )
                selected = set(mandatory) | set(extras)
                skip_blocks = [b for b in block_names if b in selected]
            controller.configure(skip_blocks)
            if args.residual_execution == "two_pass":
                cache_stats = adaptive.populate_online_cache(
                    controller,
                    lambda: sdx4_diffusion_loss(pipe, batch, args, device, dtype),
                    device,
                )
            else:
                controller.set_mode("single_skip")

        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        train_start = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        loss = sdx4_diffusion_loss(pipe, batch, args, device, dtype)
        if not torch.isfinite(loss):
            raise SystemExit(f"Non-finite training loss at step {step}.")
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(params, args.grad_clip)
        optimizer.step()
        if device.type == "cuda":
            torch.cuda.synchronize()
        train_step_time_s = time.perf_counter() - train_start
        train_peak_cuda_mem_mb = (
            torch.cuda.max_memory_allocated() / (1024.0 ** 2)
            if device.type == "cuda"
            else 0.0
        )
        train_peak_cuda_reserved_mb = (
            torch.cuda.max_memory_reserved() / (1024.0 ** 2)
            if device.type == "cuda"
            else 0.0
        )
        max_cuda_mem_mb = max(max_cuda_mem_mb, train_peak_cuda_mem_mb)
        max_cuda_reserved_mb = max(max_cuda_reserved_mb, train_peak_cuda_reserved_mb)
        if controller is not None:
            if args.residual_execution == "single_pass":
                cache_stats = controller.stats(0.0)
            controller.set_mode("full")
        del args._profile_noise_ratio
        train_rows.append(
            {
                "step": step,
                "loss": float(loss.detach().cpu()),
                "grad_norm": float(grad_norm),
                "noise_ratio": noise_ratio,
                "patch_size": patch_size,
                "skipped_blocks": ";".join(skip_blocks),
                "requested_skip_count": requested_skip_count,
                "skipped_block_count": len(skip_blocks),
                "residual_cache_time_s": cache_stats.elapsed_s,
                "residual_cache_mb": cache_stats.cache_mb,
                "cache_teacher_loss": cache_stats.teacher_loss,
                "residual_loss_abs_diff": (
                    abs(float(loss.detach().cpu()) - cache_stats.teacher_loss)
                    if controller is not None and args.residual_execution == "two_pass"
                    else ""
                ),
                "replayable_blocks": cache_stats.replayable_blocks,
                "fallback_blocks": cache_stats.fallback_blocks,
                "fallback_block_names": cache_stats.fallback_names,
                "residual_forward_max_abs_diff": cache_stats.max_reconstruction_abs_diff,
                "cache_peak_cuda_mem_mb": cache_stats.peak_cuda_mem_mb,
                "train_step_time_s": train_step_time_s,
                "train_peak_cuda_mem_mb": train_peak_cuda_mem_mb,
                "train_peak_cuda_reserved_mb": train_peak_cuda_reserved_mb,
            }
        )
        if step % args.log_every == 0 or step == 1:
            print(f"step {step:05d}/{args.train_steps} loss={float(loss):.6f}")
        if step in profile_steps:
            if controller is not None:
                controller.set_mode("full")
            current = []
            for ratio in args.profile_noise_ratios:
                profile_start = time.perf_counter()
                current.extend(
                    profile_importance(pipe, profile_loader, args, device, dtype, step, ratio)
                )
                if device.type == "cuda":
                    torch.cuda.synchronize()
                    max_cuda_mem_mb = max(max_cuda_mem_mb, torch.cuda.max_memory_allocated() / (1024.0 ** 2))
                    max_cuda_reserved_mb = max(max_cuda_reserved_mb, torch.cuda.max_memory_reserved() / (1024.0 ** 2))
                profiling_time_s += time.perf_counter() - profile_start
            importance_rows.extend(current)
            write_csv(output_dir / "lora_importance_evolution.csv", importance_rows)
            write_csv(output_dir / "lora_importance_topk.csv", topk_summary(importance_rows))
            print(
                f"profile step {step}: "
                + ", ".join(
                    row["block"]
                    for row in sorted(current, key=lambda row: row["importance_rank"])
                    if row["selected_topk"]
                )
            )
        if args.checkpoint_every > 0 and (
            step % args.checkpoint_every == 0 or step == args.train_steps
        ):
            write_csv(output_dir / "train_log.csv", train_rows)
            ckpt = output_dir / f"lora_adapter_step_{step:05d}.pt"
            checkpoint_start = time.perf_counter()
            adaptive.save_lora_adapter(pipe.unet, ckpt)
            checkpoint_time_s += time.perf_counter() - checkpoint_start
            print(f"Saved checkpoint to {ckpt}")

    write_csv(output_dir / "train_log.csv", train_rows)
    checkpoint_start = time.perf_counter()
    adapter_summary = adaptive.save_lora_adapter(pipe.unet, output_dir / "lora_adapter.pt")
    checkpoint_time_s += time.perf_counter() - checkpoint_start
    if device.type == "cuda":
        torch.cuda.synchronize()
    experiment_time_s = time.perf_counter() - experiment_start
    train_step_time_s = sum(float(r["train_step_time_s"]) for r in train_rows)
    summary = {
        "method": args.method,
        "train_steps": args.train_steps,
        "train_step_time_s": train_step_time_s,
        "mean_train_step_time_s": train_step_time_s / max(len(train_rows), 1),
        "experiment_time_s": experiment_time_s,
        "profiling_time_s": profiling_time_s,
        "checkpoint_time_s": checkpoint_time_s,
        "non_train_overhead_s": max(0.0, experiment_time_s - train_step_time_s),
        "peak_cuda_mem_mb": max_cuda_mem_mb,
        "peak_cuda_reserved_mb": max_cuda_reserved_mb,
    } | adapter_summary
    write_csv(output_dir / "summary.csv", [summary])
    metadata = vars(args) | {
        "parsed_blockskip_schedule": blockskip_schedule,
        "parsed_blockskip_fraction_schedule": blockskip_fraction_schedule,
        "profile_steps": sorted(profile_steps),
        "injected_module_count": len(injected),
        "candidate_blocks": candidate_blocks,
        "blockskip_enabled": controller is not None,
        "model": "SD-x4-Upscaler",
        "model_id": args.model_id,
    } | summary
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Wrote results to {output_dir}")
    if controller is not None:
        print(
            "Block-skip health: "
            f"replayable={sum(float(r['replayable_blocks']) for r in train_rows)} "
            f"fallback={sum(float(r['fallback_blocks']) for r in train_rows)}"
        )


if __name__ == "__main__":
    main()
