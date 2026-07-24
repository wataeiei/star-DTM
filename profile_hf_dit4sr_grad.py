#!/usr/bin/env python3
"""Gradient profiling for Hugging Face DiT4SR Diffusers model.

This script does not require cloning the DiT4SR GitHub repository. It loads the
Hugging Face Diffusers model directly:

  acceptee/DiT4SR

Important:
  This profiler targets the DiT4SR transformer component and measures probe-LoRA
  gradient sensitivity under image/latent-conditioned synthetic transformer loss.
  It is meant for layer-importance pattern analysis, not for reporting final SR
  PSNR/SSIM.

Example:
  python3 profile_hf_dit4sr_grad.py \
    --model_id acceptee/DiT4SR \
    --data_dir data/ucmerced/train_hr \
    --output_dir outputs/hf_dit4sr_grad_profile \
    --target qv \
    --topk_blocks 8 \
    --probe_batches 20 \
    --seed 42
"""

from __future__ import annotations

import argparse
import csv
import inspect
import json
import math
import random
import re
import sys
import types
from enum import Enum
from importlib.machinery import ModuleSpec
from pathlib import Path
from typing import Iterable

from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def install_torchvision_stub() -> None:
    """Bypass a binary-incompatible torchvision when only tensor models are used."""
    try:
        import torchvision  # noqa: F401
        return
    except Exception as exc:
        print(f"torchvision is unavailable/incompatible ({exc}); using a minimal import stub.")

    for name in list(sys.modules):
        if name == "torchvision" or name.startswith("torchvision."):
            sys.modules.pop(name, None)

    class InterpolationMode(Enum):
        NEAREST = "nearest"
        NEAREST_EXACT = "nearest-exact"
        BILINEAR = "bilinear"
        BICUBIC = "bicubic"
        BOX = "box"
        HAMMING = "hamming"
        LANCZOS = "lanczos"

    class ImageReadMode(Enum):
        UNCHANGED = 0
        GRAY = 1
        GRAY_ALPHA = 2
        RGB = 3
        RGB_ALPHA = 4

    def module(name: str, package: bool = False):
        result = types.ModuleType(name)
        result.__spec__ = ModuleSpec(name, loader=None, is_package=package)
        result.__file__ = f"<{name}-stub>"
        result.__loader__ = None
        result.__cached__ = None
        if package:
            result.__path__ = []
        return result

    torchvision = module("torchvision", package=True)
    transforms = module("torchvision.transforms", package=True)
    transforms_functional = module("torchvision.transforms.functional")
    transforms_v2 = module("torchvision.transforms.v2", package=True)
    transforms_v2_functional = module("torchvision.transforms.v2.functional")
    io = module("torchvision.io")
    models = module("torchvision.models", package=True)
    ops = module("torchvision.ops", package=True)
    datasets = module("torchvision.datasets", package=True)
    utils = module("torchvision.utils")

    def unavailable(*_args, **_kwargs):
        raise RuntimeError("This DiT4SR experiment does not provide torchvision image operators.")

    def missing_torchvision_attr(name: str):
        if name.startswith("__"):
            raise AttributeError(name)
        return unavailable

    for stub_module in (
        transforms,
        transforms_functional,
        transforms_v2,
        transforms_v2_functional,
        io,
        models,
        ops,
        datasets,
        utils,
    ):
        stub_module.__getattr__ = missing_torchvision_attr

    transforms.InterpolationMode = InterpolationMode
    transforms.functional = transforms_functional
    transforms.v2 = transforms_v2
    transforms_v2.functional = transforms_v2_functional
    io.ImageReadMode = ImageReadMode
    io.decode_image = unavailable
    utils.make_grid = unavailable
    torchvision.transforms = transforms
    torchvision.io = io
    torchvision.models = models
    torchvision.ops = ops
    torchvision.datasets = datasets
    torchvision.utils = utils

    sys.modules.update(
        {
            "torchvision": torchvision,
            "torchvision.transforms": transforms,
            "torchvision.transforms.functional": transforms_functional,
            "torchvision.transforms.v2": transforms_v2,
            "torchvision.transforms.v2.functional": transforms_v2_functional,
            "torchvision.io": io,
            "torchvision.models": models,
            "torchvision.ops": ops,
            "torchvision.datasets": datasets,
            "torchvision.utils": utils,
        }
    )


def require_packages() -> None:
    missing = []
    for name in ("diffusers", "transformers", "accelerate", "huggingface_hub"):
        try:
            __import__(name)
        except ImportError:
            missing.append(name)
    if missing:
        raise SystemExit("Missing packages: " + ", ".join(missing) + "\nInstall with: pip3 install " + " ".join(missing))


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
    out: list[int | str] = []
    for part in re.split(r"(\d+)", text):
        if part:
            out.append(int(part) if part.isdigit() else part)
    return out


def list_images(root: str | Path) -> list[Path]:
    root = Path(root)
    return sorted([p for p in root.rglob("*") if p.suffix.lower() in IMAGE_EXTS])


class ImageFolderDataset(Dataset):
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


def split_parent_name(root: nn.Module, dotted_name: str) -> tuple[nn.Module, str]:
    parts = dotted_name.split(".")
    parent = root
    for part in parts[:-1]:
        parent = parent[int(part)] if part.isdigit() else getattr(parent, part)
    return parent, parts[-1]


def block_key(name: str, block_regex: str = "") -> str:
    if block_regex:
        m = re.search(block_regex, name)
        if m:
            return m.group(0)
    parts = name.split(".")
    for marker in ("transformer_blocks", "blocks", "layers", "joint_transformer_blocks", "single_transformer_blocks"):
        if marker in parts:
            idx = parts.index(marker)
            if idx + 1 < len(parts) and parts[idx + 1].isdigit():
                return ".".join(parts[: idx + 2])
    m = re.search(r"(?:block|layer|transformer_blocks)[._-]?(\d+)", name, flags=re.IGNORECASE)
    return m.group(0) if m else ""


def target_match(name: str, target: str) -> bool:
    leaf = name.lower().split(".")[-1]
    if target == "q":
        return leaf in {"q", "to_q", "q_proj"}
    if target == "v":
        return leaf in {"v", "to_v", "v_proj"}
    if target == "qv":
        return leaf in {"q", "v", "to_q", "to_v", "q_proj", "v_proj"}
    if target == "qkv":
        return leaf in {"qkv", "to_qkv", "q", "k", "v", "to_q", "to_k", "to_v", "q_proj", "k_proj", "v_proj"}
    if target == "qkvo":
        return leaf in {
            "qkv",
            "to_qkv",
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
        }
    raise SystemExit(f"Unknown target={target}")


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


class TransformerOnlyPipe:
    def __init__(self, transformer: nn.Module):
        self.transformer = transformer
        self.vae = None


def model_dtype(args: argparse.Namespace) -> torch.dtype:
    if args.dtype == "bf16":
        return torch.bfloat16
    if args.dtype == "fp16":
        return torch.float16
    return torch.float32


def load_pipe(args: argparse.Namespace, device: torch.device):
    install_torchvision_stub()
    require_packages()
    dtype = model_dtype(args)
    if args.load_mode == "transformer":
        try:
            from diffusers import SD3Transformer2DModel
        except ImportError as exc:
            raise SystemExit(
                "Your diffusers version does not expose SD3Transformer2DModel.\n"
                "Upgrade with: pip3 install -U diffusers transformers accelerate"
            ) from exc
        subfolder = args.transformer_subfolder or f"{args.variant}/transformer"
        transformer = SD3Transformer2DModel.from_pretrained(
            args.model_id,
            subfolder=subfolder,
            torch_dtype=dtype,
            local_files_only=args.local_files_only,
        )
        return TransformerOnlyPipe(transformer.to(device))

    from diffusers import DiffusionPipeline

    pipe_kwargs = {
        "torch_dtype": dtype,
        "trust_remote_code": True,
        "local_files_only": args.local_files_only,
    }
    if args.pipeline_subfolder:
        pipe_kwargs["subfolder"] = args.pipeline_subfolder
    pipe = DiffusionPipeline.from_pretrained(args.model_id, **pipe_kwargs)
    pipe = pipe.to(device)
    if hasattr(pipe, "set_progress_bar_config"):
        pipe.set_progress_bar_config(disable=True)
    return pipe


def get_transformer(pipe, component_name: str) -> nn.Module:
    if component_name:
        module = getattr(pipe, component_name, None)
        if module is None:
            raise SystemExit(f"Pipeline has no component named {component_name}")
        return module
    for name in ("transformer", "dit", "model", "unet"):
        module = getattr(pipe, name, None)
        if isinstance(module, nn.Module):
            return module
    raise SystemExit("Could not find transformer component. Use --component_name.")


def inspect_component(component: nn.Module, args: argparse.Namespace) -> None:
    print("== Block/attention-like modules ==")
    for name, module in component.named_modules():
        cls = module.__class__.__name__
        if any(k in cls.lower() for k in ("block", "attention", "transformer", "joint")):
            print(name, cls)
    print("\n== Candidate Linear modules ==")
    for name, module in component.named_modules():
        if isinstance(module, nn.Linear):
            bkey = block_key(name, args.block_regex)
            mark = "*" if bkey and target_match(name, args.target) else " "
            print(f"{mark} {name} [{module.in_features}->{module.out_features}] block={bkey}")


def image_to_hidden_states(pipe, images: torch.Tensor, transformer: nn.Module, args: argparse.Namespace, device: torch.device):
    bsz = images.shape[0]
    config = getattr(transformer, "config", None)
    in_channels = int(getattr(config, "in_channels", args.latent_channels))
    sample_size = int(getattr(config, "sample_size", args.latent_size))
    dtype = next(transformer.parameters()).dtype
    if sample_size <= 0:
        sample_size = args.latent_size

    if hasattr(pipe, "vae") and pipe.vae is not None:
        with torch.no_grad():
            latent = pipe.vae.encode(images.to(device)).latent_dist.sample()
            scale = getattr(getattr(pipe.vae, "config", None), "scaling_factor", 1.0)
            latent = latent * scale
            latent = F.interpolate(latent.float(), size=(sample_size, sample_size), mode="bilinear", align_corners=False)
            if latent.shape[1] != in_channels:
                if latent.shape[1] > in_channels:
                    latent = latent[:, :in_channels]
                else:
                    pad = torch.zeros(bsz, in_channels - latent.shape[1], sample_size, sample_size, device=device)
                    latent = torch.cat([latent, pad], dim=1)
            return latent.to(dtype=dtype)

    x = F.interpolate(images.to(device).float(), size=(sample_size, sample_size), mode="bilinear", align_corners=False)
    if x.shape[1] != in_channels:
        if x.shape[1] > in_channels:
            x = x[:, :in_channels]
        else:
            x = torch.cat([x, torch.zeros(bsz, in_channels - x.shape[1], sample_size, sample_size, device=device)], dim=1)
    return x.to(dtype=dtype)


def zeros_like_signature_arg(name: str, param, bsz: int, transformer: nn.Module, args: argparse.Namespace, device: torch.device):
    config = getattr(transformer, "config", None)
    caption_dim = int(getattr(config, "joint_attention_dim", args.caption_dim))
    pooled_dim = int(getattr(config, "pooled_projection_dim", caption_dim))
    seq_len = args.prompt_seq_len
    dtype = next(transformer.parameters()).dtype
    if name in ("timestep", "timesteps"):
        return torch.full((bsz,), args.timestep, device=device, dtype=torch.long)
    if name in ("encoder_hidden_states", "prompt_embeds"):
        return torch.zeros(bsz, seq_len, caption_dim, device=device, dtype=dtype)
    if name in ("pooled_projections", "pooled_prompt_embeds"):
        return torch.zeros(bsz, pooled_dim, device=device, dtype=dtype)
    if name in ("return_dict",):
        return True
    if name in ("joint_attention_kwargs", "attention_kwargs"):
        return None
    if param.default is not inspect._empty:
        return param.default
    raise SystemExit(f"Do not know how to build required transformer argument: {name}")


def transformer_forward(transformer: nn.Module, hidden_states: torch.Tensor, args: argparse.Namespace):
    sig = inspect.signature(transformer.forward)
    kwargs = {}
    bsz = hidden_states.shape[0]
    for name, param in sig.parameters.items():
        if name == "hidden_states":
            kwargs[name] = hidden_states
        elif name == "sample":
            kwargs[name] = hidden_states
        else:
            kwargs[name] = zeros_like_signature_arg(name, param, bsz, transformer, args, hidden_states.device)
    return transformer(**kwargs)


def output_tensor(output):
    if torch.is_tensor(output):
        return output
    if hasattr(output, "sample"):
        return output.sample
    if isinstance(output, dict):
        for key in ("sample", "hidden_states", "output"):
            if key in output and torch.is_tensor(output[key]):
                return output[key]
    if isinstance(output, (tuple, list)):
        for item in output:
            if torch.is_tensor(item):
                return item
    raise SystemExit("Could not find tensor output from transformer forward.")


def profile(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    pipe = load_pipe(args, device)
    transformer = get_transformer(pipe, args.component_name)

    if args.inspect_only:
        inspect_component(transformer, args)
        return

    for p in transformer.parameters():
        p.requires_grad_(False)
    injected = inject_lora(transformer, args.target, args.rank, args.alpha, args.block_regex)
    transformer.train()

    dataset = ImageFolderDataset(args.data_dir, args.image_size, args.max_images)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    data_iter = iter(loader)
    valid = 0
    total_loss = 0.0
    transformer.zero_grad(set_to_none=True)
    for idx in range(1, args.probe_batches + 1):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch = next(data_iter)
        hidden = image_to_hidden_states(pipe, batch["image"], transformer, args, device)
        if args.input_noise_std > 0:
            hidden = hidden + torch.randn_like(hidden) * args.input_noise_std
        output = transformer_forward(transformer, hidden, args)
        out = output_tensor(output).float()
        if out.shape == hidden.shape:
            loss = F.mse_loss(out, hidden.float())
        else:
            loss = out.pow(2).mean()
        if not torch.isfinite(loss):
            print(f"probe batch {idx}/{args.probe_batches} skipped: non-finite loss")
            transformer.zero_grad(set_to_none=True)
            continue
        loss.backward()
        valid += 1
        total_loss += float(loss.detach().cpu())
        print(f"probe batch {idx:03d}/{args.probe_batches} loss={float(loss.detach().cpu()):.6f}")

    if valid == 0:
        raise SystemExit("All probe batches failed.")

    rows_by_block: dict[str, dict] = {}
    for name, module in iter_lora_modules(transformer):
        bkey = block_key(name, args.block_regex)
        row = rows_by_block.setdefault(
            bkey,
            {"block": bkey, "grad_sq": 0.0, "lora_param_count": 0, "module_count": 0},
        )
        row["grad_sq"] += lora_grad_norm(module) ** 2
        row["lora_param_count"] += module.lora_down.weight.numel() + module.lora_up.weight.numel()
        row["module_count"] += 1

    ordered = sorted(rows_by_block, key=natural_key)
    block_index = {block: idx + 1 for idx, block in enumerate(ordered)}
    total_blocks = len(ordered)
    rows = []
    for bkey, row in rows_by_block.items():
        grad_norm = math.sqrt(row["grad_sq"])
        p_count = int(row["lora_param_count"])
        normalized = grad_norm / math.sqrt(max(p_count, 1))
        bp_cost = total_blocks - block_index[bkey] + 1
        selection = normalized
        if args.compute_lambda > 0:
            selection = normalized / (p_count + args.compute_lambda * bp_cost)
        rows.append(
            {
                "block": bkey,
                "block_index": block_index[bkey],
                "grad_norm": grad_norm,
                "lora_param_count": p_count,
                "module_count": int(row["module_count"]),
                "normalized_grad_score": normalized,
                "bp_cost": bp_cost,
                "compute_lambda": args.compute_lambda,
                "selection_score": selection,
                "probe_batches": valid,
                "mean_probe_loss": total_loss / valid,
                "selected": False,
            }
        )
    selected = set([r["block"] for r in sorted(rows, key=lambda r: r["selection_score"], reverse=True)[: args.topk_blocks]])
    for row in rows:
        row["selected"] = row["block"] in selected
    rows.sort(key=lambda r: r["block_index"])

    out_dir = ensure_dir(args.output_dir)
    write_csv(out_dir / "hf_dit4sr_grad_scores.csv", rows)
    metadata = {
        "model_id": args.model_id,
        "component_name": args.component_name or "auto",
        "data_dir": args.data_dir,
        "target": args.target,
        "rank": args.rank,
        "alpha": args.alpha,
        "topk_blocks": args.topk_blocks,
        "probe_batches": args.probe_batches,
        "seed": args.seed,
        "selected_blocks": sorted(selected, key=natural_key),
        "injected_module_count": len(injected),
    }
    (out_dir / "hf_dit4sr_grad_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Wrote {out_dir / 'hf_dit4sr_grad_scores.csv'}")
    print(f"Wrote {out_dir / 'hf_dit4sr_grad_metadata.json'}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_id", default="acceptee/DiT4SR")
    parser.add_argument("--load_mode", default="transformer", choices=["transformer", "pipeline"])
    parser.add_argument("--variant", default="dit4sr_q", choices=["dit4sr_q", "dit4sr_f", "dit4sr_r1"])
    parser.add_argument("--transformer_subfolder", default="", help="Override transformer subfolder, e.g. dit4sr_q/transformer")
    parser.add_argument("--pipeline_subfolder", default="", help="Optional pipeline subfolder if a repo contains model_index.json there")
    parser.add_argument("--component_name", default="", help="Pipeline component to profile, default auto: transformer/dit/model/unet")
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--max_images", type=int, default=0)
    parser.add_argument("--target", default="qv", choices=["q", "v", "qv", "qkv", "qkvo"])
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--alpha", type=int, default=16)
    parser.add_argument("--topk_blocks", type=int, default=8)
    parser.add_argument("--probe_batches", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dtype", default="fp32", choices=["fp32", "bf16", "fp16"])
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--inspect_only", action="store_true")
    parser.add_argument("--block_regex", default="")
    parser.add_argument("--latent_channels", type=int, default=16)
    parser.add_argument("--latent_size", type=int, default=64)
    parser.add_argument("--caption_dim", type=int, default=4096)
    parser.add_argument("--prompt_seq_len", type=int, default=77)
    parser.add_argument("--timestep", type=int, default=500)
    parser.add_argument("--input_noise_std", type=float, default=0.0)
    parser.add_argument("--compute_lambda", type=float, default=0.0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    profile(args)


if __name__ == "__main__":
    main()
