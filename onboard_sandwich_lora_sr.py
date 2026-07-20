#!/usr/bin/env python3
"""On-board Sandwich-LoRA experiment for remote-sensing super resolution.

Modes:
  prepare_ucmerced  Download/split UC Merced HR images.
  comm              Estimate adapter upload feasibility.
  train             Fine-tune selected UNet attention projections with LoRA.
  eval              Evaluate base or LoRA-adapted SD x4 upscaler.
  parse_tegrastats  Parse Jetson tegrastats logs into resource metrics.
  inspect_lora      Inspect whether a saved LoRA adapter is nonzero and loadable.

The LoRA implementation is intentionally local to this script so the experiment
does not depend on PEFT/diffusers adapter API changes.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from PIL import Image

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset
except ImportError:
    torch = None
    F = None
    DataLoader = None

    class _MissingNN:
        Module = object
        Linear = object

    class Dataset:
        pass

    nn = _MissingNN()


MODEL_ID = "stabilityai/stable-diffusion-x4-upscaler"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def require_packages(*names: str) -> None:
    missing = []
    for name in names:
        try:
            __import__(name)
        except ImportError:
            missing.append(name)
    if missing:
        raise SystemExit(
            "Missing packages: "
            + ", ".join(missing)
            + "\nInstall them with: pip3 install "
            + " ".join(missing)
        )


def require_torch() -> None:
    if torch is None:
        raise SystemExit(
            "Missing package: torch\n"
            "Install the NVIDIA Jetson-compatible PyTorch wheel before running train/eval."
        )


def set_seed(seed: int) -> None:
    random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def list_images(root: str | Path) -> list[Path]:
    root = Path(root)
    paths = [p for p in root.rglob("*") if p.suffix.lower() in IMAGE_EXTS]
    return sorted(paths)


def pil_to_tensor(image: Image.Image, size: int) -> torch.Tensor:
    image = image.convert("RGB").resize((size, size), Image.BICUBIC)
    arr = torch.frombuffer(bytearray(image.tobytes()), dtype=torch.uint8)
    arr = arr.view(size, size, 3).permute(2, 0, 1).float() / 255.0
    return arr * 2.0 - 1.0


def tensor_to_pil(x: torch.Tensor) -> Image.Image:
    x = x.detach().float().cpu().clamp(-1, 1)
    x = ((x + 1.0) * 127.5).round().byte()
    x = x.permute(1, 2, 0).numpy()
    return Image.fromarray(x, mode="RGB")


def save_csv(path: str | Path, rows: list[dict]) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def upload_seconds(size_mb: float, uplink_mbps: float, eta: float) -> float:
    mb_per_s = eta * uplink_mbps / 8.0
    return size_mb / mb_per_s if mb_per_s > 0 else float("inf")


class HrImageDataset(Dataset):
    def __init__(self, root: str | Path, hr_size: int, lr_size: int) -> None:
        self.paths = list_images(root)
        if not self.paths:
            raise SystemExit(f"No images found in {root}")
        self.hr_size = hr_size
        self.lr_size = lr_size

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> dict:
        image = Image.open(self.paths[idx]).convert("RGB")
        hr = pil_to_tensor(image, self.hr_size)
        lr = image.resize((self.lr_size, self.lr_size), Image.BICUBIC)
        lr_up = pil_to_tensor(lr, self.hr_size)
        return {"hr": hr, "lr_up": lr_up, "path": str(self.paths[idx])}


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


def natural_key(text: str) -> list[int | str]:
    key: list[int | str] = []
    for part in text.replace(".", " ").split():
        key.append(int(part) if part.isdigit() else part)
    return key


def transformer_block_key(name: str) -> str:
    parts = name.split(".")
    if "transformer_blocks" not in parts:
        return ""
    idx = parts.index("transformer_blocks")
    if idx + 1 >= len(parts):
        return ""
    return ".".join(parts[: idx + 2])


def block_group_index(block: str, group: str) -> int | None:
    parts = block.split(".")
    if len(parts) >= 2 and parts[0] == group and parts[1].isdigit():
        return int(parts[1])
    return None


def sandwich_core_blocks(block_keys: Iterable[str]) -> list[str]:
    keys = sorted(block_keys, key=natural_key)
    down_indices = [idx for key in keys if (idx := block_group_index(key, "down_blocks")) is not None]
    up_indices = [idx for key in keys if (idx := block_group_index(key, "up_blocks")) is not None]
    core: list[str] = []
    if down_indices:
        first_down = min(down_indices)
        core.extend([key for key in keys if block_group_index(key, "down_blocks") == first_down])
    if up_indices:
        last_up = max(up_indices)
        core.extend([key for key in keys if block_group_index(key, "up_blocks") == last_up])
    return list(dict.fromkeys(core))


def select_topk_blocks(block_keys: set[str], k: int, policy: str) -> list[str]:
    if k <= 0:
        raise SystemExit("--topk_blocks must be positive when --lora_scope topk")
    keys = sorted(block_keys, key=natural_key)
    if not keys:
        return []
    if k >= len(keys):
        return keys
    if policy == "early":
        return keys[:k]
    if policy == "late":
        return keys[-k:]
    if policy == "uniform":
        if k == 1:
            return [keys[len(keys) // 2]]
        indices = sorted({round(i * (len(keys) - 1) / (k - 1)) for i in range(k)})
        return [keys[i] for i in indices][:k]
    if policy in ("sandwich_plus", "grad_sandwich_plus"):
        core = sandwich_core_blocks(keys)
        selected = core[:k]
        for key in [key for key in keys if key.startswith("mid_block.")] + keys:
            if len(selected) >= k:
                break
            if key not in selected:
                selected.append(key)
        return selected[:k]
    if policy == "balanced":
        down = [key for key in keys if key.startswith("down_blocks.")]
        mid = [key for key in keys if key.startswith("mid_block.")]
        up = [key for key in keys if key.startswith("up_blocks.")]
        selected: list[str] = []
        i = 0
        while len(selected) < k and (i < len(down) or i < len(up)):
            if i < len(down):
                selected.append(down[i])
            if len(selected) >= k:
                break
            if i < len(up):
                selected.append(up[-1 - i])
            i += 1
        for key in mid + keys:
            if len(selected) >= k:
                break
            if key not in selected:
                selected.append(key)
        return selected[:k]
    raise SystemExit(f"Unknown topk_policy={policy}")


def module_in_scope(
    name: str,
    scope: str,
    shallow_down_indices: set[int] | None = None,
    last_up_indices: set[int] | None = None,
    topk_block_keys: set[str] | None = None,
) -> bool:
    if scope == "all":
        return True
    if scope in ("topk", "grad_topk"):
        topk_block_keys = topk_block_keys or set()
        return transformer_block_key(name) in topk_block_keys
    if scope == "shallow":
        shallow_down_indices = shallow_down_indices or set()
        return any(f"down_blocks.{idx}." in name for idx in shallow_down_indices)
    if scope == "last2_up":
        last_up_indices = last_up_indices or set()
        return any(f"up_blocks.{idx}." in name for idx in last_up_indices)
    if scope == "shallow_deep":
        return module_in_scope(
            name, "shallow", shallow_down_indices, last_up_indices, topk_block_keys
        ) or module_in_scope(
            name, "last2_up", shallow_down_indices, last_up_indices, topk_block_keys
        )
    raise SystemExit(f"Unknown lora_scope={scope}")


def split_parent_name(root: nn.Module, dotted_name: str) -> tuple[nn.Module, str]:
    parts = dotted_name.split(".")
    parent = root
    for part in parts[:-1]:
        parent = parent[int(part)] if part.isdigit() else getattr(parent, part)
    return parent, parts[-1]


def unwrap_lora(root: nn.Module) -> None:
    for name, module in list(iter_lora_modules(root)):
        parent, child_name = split_parent_name(root, name)
        setattr(parent, child_name, module.base)


@dataclass
class LoraInfo:
    names: list[str]
    trainable_params: int
    total_params: int
    selected_blocks: list[str]


def inject_lora(
    unet: nn.Module,
    rank: int,
    alpha: int,
    target: str,
    scope: str,
    topk_blocks: int = 5,
    topk_policy: str = "balanced",
    topk_block_names: list[str] | None = None,
) -> LoraInfo:
    suffixes = target_suffixes(target)
    replacements: list[tuple[str, nn.Linear]] = []
    down_indices_with_targets = set()
    candidate_block_keys = set()
    up_indices = set()
    for name, module in unet.named_modules():
        parts = name.split(".")
        block_key = transformer_block_key(name)
        if isinstance(module, nn.Linear) and block_key and any(name.endswith(suffix) for suffix in suffixes):
            candidate_block_keys.add(block_key)
        if (
            isinstance(module, nn.Linear)
            and len(parts) >= 2
            and parts[0] == "down_blocks"
            and parts[1].isdigit()
            and any(name.endswith(suffix) for suffix in suffixes)
        ):
            down_indices_with_targets.add(int(parts[1]))
        if len(parts) >= 2 and parts[0] == "up_blocks" and parts[1].isdigit():
            up_indices.add(int(parts[1]))
    shallow_down_indices = set(sorted(down_indices_with_targets)[:1])
    last_up_indices = set(sorted(up_indices)[-2:])
    if scope in ("topk", "grad_topk"):
        selected_blocks = topk_block_names or select_topk_blocks(candidate_block_keys, topk_blocks, topk_policy)
        topk_block_keys = set(selected_blocks)
    else:
        selected_blocks = []
        topk_block_keys = set()

    for name, module in unet.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if not module_in_scope(name, scope, shallow_down_indices, last_up_indices, topk_block_keys):
            continue
        if any(name.endswith(suffix) for suffix in suffixes):
            replacements.append((name, module))

    if not replacements:
        raise SystemExit(
            f"No LoRA targets found for scope={scope}, target={target}. "
            "Check the UNet module names for this diffusers version."
        )

    for name, module in replacements:
        parent, child_name = split_parent_name(unet, name)
        setattr(parent, child_name, LoRALinear(module, rank=rank, alpha=alpha))

    trainable = sum(p.numel() for p in unet.parameters() if p.requires_grad)
    total = sum(p.numel() for p in unet.parameters())
    return LoraInfo([name for name, _ in replacements], trainable, total, selected_blocks)


def iter_lora_modules(root: nn.Module) -> Iterable[tuple[str, LoRALinear]]:
    for name, module in root.named_modules():
        if isinstance(module, LoRALinear):
            yield name, module


def move_lora_to_device(root: nn.Module, device: torch.device) -> None:
    for _name, module in iter_lora_modules(root):
        module.lora_down.to(device=device, dtype=torch.float32)
        module.lora_up.to(device=device, dtype=torch.float32)


def lora_block_key(name: str) -> str:
    return transformer_block_key(name)


def lora_param_vector(module: LoRALinear) -> torch.Tensor:
    return torch.cat(
        [
            module.lora_down.weight.detach().float().flatten().cpu(),
            module.lora_up.weight.detach().float().flatten().cpu(),
        ]
    )


def lora_grad_norm(module: LoRALinear) -> float:
    total = 0.0
    for param in (module.lora_down.weight, module.lora_up.weight):
        if param.grad is not None:
            total += float(param.grad.detach().float().pow(2).sum().cpu())
    return math.sqrt(total)


def set_lora_trainable(module: LoRALinear, trainable: bool) -> None:
    module.lora_down.weight.requires_grad_(trainable)
    module.lora_up.weight.requires_grad_(trainable)


def shallow_lora_blocks(root: nn.Module) -> set[str]:
    blocks = {lora_block_key(name) for name, _module in iter_lora_modules(root) if name.startswith("down_blocks.")}
    return {block for block in blocks if block}


def init_freeze_state(root: nn.Module) -> dict:
    snapshots: dict[str, torch.Tensor] = {}
    for name, module in iter_lora_modules(root):
        block = lora_block_key(name)
        if block:
            current = lora_param_vector(module)
            snapshots[block] = current if block not in snapshots else torch.cat([snapshots[block], current])
    return {"snapshots": snapshots, "patience": {}, "frozen": set(), "events": []}


def dynamic_freeze_check(root: nn.Module, step: int, args: argparse.Namespace, state: dict) -> None:
    shallow_blocks = shallow_lora_blocks(root)
    if not shallow_blocks:
        return
    current_vectors: dict[str, torch.Tensor] = {}
    grad_sq: dict[str, float] = {}
    modules_by_block: dict[str, list[LoRALinear]] = {}
    for name, module in iter_lora_modules(root):
        block = lora_block_key(name)
        if block not in shallow_blocks:
            continue
        vec = lora_param_vector(module)
        current_vectors[block] = vec if block not in current_vectors else torch.cat([current_vectors[block], vec])
        grad_sq[block] = grad_sq.get(block, 0.0) + lora_grad_norm(module) ** 2
        modules_by_block.setdefault(block, []).append(module)

    for block, current in current_vectors.items():
        previous = state["snapshots"].get(block)
        if previous is None:
            state["snapshots"][block] = current
            continue
        update_ratio = float(torch.linalg.vector_norm(current - previous) / (torch.linalg.vector_norm(previous) + 1e-12))
        grad_norm = math.sqrt(grad_sq.get(block, 0.0))
        should_count = update_ratio < args.freeze_tau_update and grad_norm < args.freeze_tau_grad
        state["patience"][block] = state["patience"].get(block, 0) + 1 if should_count else 0
        frozen_now = False
        if state["patience"][block] >= args.freeze_patience and block not in state["frozen"]:
            for module in modules_by_block.get(block, []):
                set_lora_trainable(module, False)
            state["frozen"].add(block)
            frozen_now = True
            print(
                f"Dynamic freeze: froze {block} at step {step} "
                f"(update_ratio={update_ratio:.3e}, grad_norm={grad_norm:.3e})"
            )
        state["events"].append(
            {
                "step": step,
                "block": block,
                "update_ratio": update_ratio,
                "grad_norm": grad_norm,
                "patience_count": state["patience"][block],
                "frozen": block in state["frozen"],
                "frozen_now": frozen_now,
            }
        )
        state["snapshots"][block] = current


def lora_structure_rows(unet: nn.Module) -> list[dict]:
    rows = []
    for name, module in iter_lora_modules(unet):
        params = module.lora_down.weight.numel() + module.lora_up.weight.numel()
        rows.append(
            {
                "module": name,
                "rank": module.rank,
                "alpha": module.alpha,
                "in_features": module.base.in_features,
                "out_features": module.base.out_features,
                "lora_params": params,
            }
        )
    return rows


def lora_save_dtype(dtype_name: str):
    if dtype_name == "fp32":
        return torch.float32
    if dtype_name == "fp16":
        return torch.float16
    raise SystemExit(f"Unknown save_lora_dtype={dtype_name}; choose fp32 or fp16")


def save_lora(unet: nn.Module, output_dir: str | Path, metadata: dict, save_dtype: str = "fp32") -> Path:
    require_packages("safetensors")
    from safetensors.torch import save_file

    output_dir = ensure_dir(output_dir)
    tensor_dtype = lora_save_dtype(save_dtype)
    tensors = {}
    for name, module in iter_lora_modules(unet):
        tensors[f"{name}.lora_down.weight"] = module.lora_down.weight.detach().cpu().to(dtype=tensor_dtype)
        tensors[f"{name}.lora_up.weight"] = module.lora_up.weight.detach().cpu().to(dtype=tensor_dtype)
    if not tensors:
        raise SystemExit("No LoRA modules to save")
    weight_path = output_dir / "pytorch_lora_weights.safetensors"
    save_file(tensors, str(weight_path))
    metadata["save_lora_dtype"] = save_dtype
    with (output_dir / "lora_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    return weight_path


def load_lora(unet: nn.Module, lora_dir: str | Path, device: torch.device, dtype: torch.dtype) -> dict:
    require_packages("safetensors")
    from safetensors.torch import load_file

    lora_dir = Path(lora_dir)
    with (lora_dir / "lora_metadata.json").open("r", encoding="utf-8") as f:
        metadata = json.load(f)
    inject_lora(
        unet,
        rank=int(metadata["rank"]),
        alpha=int(metadata["alpha"]),
        target=metadata["target"],
        scope=metadata["lora_scope"],
        topk_blocks=int(metadata.get("topk_blocks", 5)),
        topk_policy=metadata.get("topk_policy", "balanced"),
        topk_block_names=metadata.get("topk_block_names") or None,
    )
    move_lora_to_device(unet, device)
    tensors = load_file(str(lora_dir / "pytorch_lora_weights.safetensors"))
    modules = dict(iter_lora_modules(unet))
    for key, value in tensors.items():
        module_name, weight_name, _ = key.rsplit(".", 2)
        module = modules[module_name]
        layer = module.lora_down if weight_name == "lora_down" else module.lora_up
        layer.weight.data.copy_(value.to(device=device, dtype=torch.float32))
    return metadata


def inspect_lora(args: argparse.Namespace) -> None:
    require_packages("safetensors")
    from safetensors.torch import load_file

    lora_dir = Path(args.lora_dir)
    if not lora_dir:
        raise SystemExit("--lora_dir is required for --mode inspect_lora")
    metadata_path = lora_dir / "lora_metadata.json"
    weight_path = lora_dir / "pytorch_lora_weights.safetensors"
    if not metadata_path.exists():
        raise SystemExit(f"Missing metadata: {metadata_path}")
    if not weight_path.exists():
        raise SystemExit(f"Missing weights: {weight_path}")

    with metadata_path.open("r", encoding="utf-8") as f:
        metadata = json.load(f)
    tensors = load_file(str(weight_path))
    rows = []
    total_params = 0
    total_l1 = 0.0
    global_max_abs = 0.0
    zero_tensors = 0
    for name, tensor in sorted(tensors.items()):
        t = tensor.float()
        numel = t.numel()
        l1 = float(t.abs().sum().item())
        max_abs = float(t.abs().max().item()) if numel else 0.0
        mean_abs = l1 / numel if numel else 0.0
        is_all_zero = max_abs == 0.0
        rows.append(
            {
                "tensor": name,
                "shape": "x".join(str(v) for v in tensor.shape),
                "num_params": numel,
                "mean_abs": mean_abs,
                "max_abs": max_abs,
                "l1_sum": l1,
                "all_zero": is_all_zero,
            }
        )
        total_params += numel
        total_l1 += l1
        global_max_abs = max(global_max_abs, max_abs)
        zero_tensors += int(is_all_zero)

    expected_modules = metadata.get("lora_module_names", [])
    summary = {
        "lora_dir": str(lora_dir),
        "rank": metadata.get("rank", ""),
        "alpha": metadata.get("alpha", ""),
        "target": metadata.get("target", ""),
        "lora_scope": metadata.get("lora_scope", ""),
        "metadata_module_count": len(expected_modules),
        "tensor_count": len(tensors),
        "total_lora_params": total_params,
        "zero_tensor_count": zero_tensors,
        "nonzero_tensor_count": len(tensors) - zero_tensors,
        "global_max_abs": global_max_abs,
        "global_mean_abs": total_l1 / total_params if total_params else 0.0,
        "adapter_size_mb": weight_path.stat().st_size / (1024 * 1024),
        "looks_nonzero": global_max_abs > args.inspect_zero_tol,
        "deep_model_check": bool(args.inspect_load_model),
    }

    if args.inspect_load_model:
        require_torch()
        device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
        dtype = torch.float16 if device.type == "cuda" and args.fp16 else torch.float32
        pipe = load_pipeline(dtype=dtype).to(device)
        info = inject_lora(
            pipe.unet,
            rank=int(metadata["rank"]),
            alpha=int(metadata["alpha"]),
            target=metadata["target"],
            scope=metadata["lora_scope"],
            topk_blocks=int(metadata.get("topk_blocks", 5)),
            topk_policy=metadata.get("topk_policy", "balanced"),
            topk_block_names=metadata.get("topk_block_names") or None,
        )
        actual_modules = set(info.names)
        expected_set = set(expected_modules)
        modules_from_tensors = {key.rsplit(".", 2)[0] for key in tensors}
        summary.update(
            {
                "actual_injected_module_count": len(actual_modules),
                "metadata_modules_missing_in_model": len(expected_set - actual_modules),
                "tensor_modules_missing_in_model": len(modules_from_tensors - actual_modules),
                "model_modules_missing_in_tensors": len(actual_modules - modules_from_tensors),
                "module_names_match": not (expected_set - actual_modules)
                and not (modules_from_tensors - actual_modules)
                and not (actual_modules - modules_from_tensors),
            }
        )

    output_dir = ensure_dir(args.output_dir)
    save_csv(output_dir / "lora_inspect_tensors.csv", rows)
    save_csv(output_dir / "lora_inspect_summary.csv", [summary])
    print(json.dumps(summary, indent=2))


def quantize_lora(args: argparse.Namespace) -> None:
    require_packages("safetensors")
    from safetensors.torch import load_file, save_file

    src_dir = Path(args.lora_dir)
    if not src_dir:
        raise SystemExit("--lora_dir is required for --mode quantize_lora")
    src_meta = src_dir / "lora_metadata.json"
    src_weights = src_dir / "pytorch_lora_weights.safetensors"
    if not src_meta.exists():
        raise SystemExit(f"Missing metadata: {src_meta}")
    if not src_weights.exists():
        raise SystemExit(f"Missing weights: {src_weights}")

    out_dir = ensure_dir(args.output_dir)
    with src_meta.open("r", encoding="utf-8") as f:
        metadata = json.load(f)
    target_dtype = lora_save_dtype(args.save_lora_dtype)
    tensors = load_file(str(src_weights))
    quantized = {name: tensor.detach().cpu().to(dtype=target_dtype) for name, tensor in tensors.items()}
    out_weights = out_dir / "pytorch_lora_weights.safetensors"
    save_file(quantized, str(out_weights))
    metadata["source_lora_dir"] = str(src_dir)
    metadata["save_lora_dtype"] = args.save_lora_dtype
    metadata["quantized_from"] = str(src_weights)
    with (out_dir / "lora_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    src_mb = src_weights.stat().st_size / (1024 * 1024)
    out_mb = out_weights.stat().st_size / (1024 * 1024)
    summary = {
        "source_lora_dir": str(src_dir),
        "output_dir": str(out_dir),
        "save_lora_dtype": args.save_lora_dtype,
        "source_size_mb": src_mb,
        "quantized_size_mb": out_mb,
        "adapter_size_mb": out_mb,
        "size_ratio": safe_div(src_mb, out_mb),
        "upload_time_1mbps_s": upload_seconds(out_mb, 1.0, args.eta),
    }
    save_csv(out_dir / "quantize_lora_summary.csv", [summary])
    print(json.dumps(summary, indent=2))


def prepare_ucmerced(args: argparse.Namespace) -> None:
    require_packages("datasets")
    from datasets import load_dataset

    set_seed(args.seed)
    out = ensure_dir(args.data_root)
    train_dir = ensure_dir(out / "train_hr")
    val_dir = ensure_dir(out / "val_hr")
    split_path = out / "split.json"

    ds = load_dataset(args.dataset_name, split=args.dataset_split)
    indices = list(range(len(ds)))
    random.shuffle(indices)
    n_train = int(len(indices) * args.train_ratio)
    split = {"train": indices[:n_train], "val": indices[n_train:], "seed": args.seed}

    def export(indices_to_save: list[int], dst: Path, prefix: str) -> None:
        for j, idx in enumerate(indices_to_save):
            item = ds[idx]
            image = item.get("image") or item.get("img")
            if image is None:
                raise SystemExit("Dataset item does not contain an image/img field")
            image = image.convert("RGB").resize((args.hr_size, args.hr_size), Image.BICUBIC)
            image.save(dst / f"{prefix}_{j:04d}.png")

    export(split["train"], train_dir, "train")
    export(split["val"], val_dir, "val")
    with split_path.open("w", encoding="utf-8") as f:
        json.dump(split, f, indent=2)
    print(f"Saved {len(split['train'])} train and {len(split['val'])} val images to {out}")


def comm(args: argparse.Namespace) -> None:
    rows = []
    for size_mb in args.update_sizes_mb:
        for rate_mbps in args.uplink_mbps:
            upload_s = upload_seconds(size_mb, rate_mbps, args.eta)
            rows.append(
                {
                    "update_size_mb": size_mb,
                    "uplink_mbps": rate_mbps,
                    "eta": args.eta,
                    "upload_time_s": upload_s,
                    "contact_window_s": args.contact_window_s,
                    "feasible_in_one_pass": upload_s <= args.contact_window_s,
                }
            )
    save_csv(args.output_csv, rows)
    for row in rows:
        ok = "OK" if row["feasible_in_one_pass"] else "NO"
        print(
            f"{row['update_size_mb']:8.1f} MB @ {row['uplink_mbps']:6.3f} Mbps "
            f"=> {row['upload_time_s']:8.1f}s [{ok}]"
        )


def load_pipeline(dtype: torch.dtype):
    require_packages("diffusers", "transformers", "accelerate")
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


def make_grad_scaler(device: torch.device, enabled: bool):
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler(device.type, enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


def autocast_context(device: torch.device, enabled: bool):
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        return torch.amp.autocast(device_type=device.type, enabled=enabled)
    return torch.cuda.amp.autocast(enabled=enabled)


def cuda_memory_stats_mb(device: torch.device) -> dict:
    if device.type != "cuda":
        return {
            "cuda_mem_allocated_mb": 0.0,
            "cuda_mem_reserved_mb": 0.0,
            "cuda_max_mem_allocated_mb": 0.0,
            "cuda_max_mem_reserved_mb": 0.0,
        }
    return {
        "cuda_mem_allocated_mb": torch.cuda.memory_allocated(device) / (1024 * 1024),
        "cuda_mem_reserved_mb": torch.cuda.memory_reserved(device) / (1024 * 1024),
        "cuda_max_mem_allocated_mb": torch.cuda.max_memory_allocated(device) / (1024 * 1024),
        "cuda_max_mem_reserved_mb": torch.cuda.max_memory_reserved(device) / (1024 * 1024),
    }


def diffusion_train_loss(pipe, batch: dict, args: argparse.Namespace, device: torch.device, dtype: torch.dtype, use_amp: bool):
    hr = batch["hr"].to(device=device, dtype=dtype)
    lr_up = batch["lr_up"].to(device=device, dtype=dtype)
    bsz = hr.shape[0]

    with torch.no_grad():
        latents = pipe.vae.encode(hr).latent_dist.sample()
        latents = latents * pipe.vae.config.scaling_factor
        noise = torch.randn_like(latents)
        timesteps = torch.randint(
            0,
            pipe.scheduler.config.num_train_timesteps,
            (bsz,),
            device=device,
            dtype=torch.long,
        )
        noisy_latents = pipe.scheduler.add_noise(latents, noise, timesteps)
        noise_level = torch.full((bsz,), args.noise_level, device=device, dtype=torch.long)
        lr_cond_clean = F.interpolate(
            lr_up.float(),
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

    with autocast_context(device, enabled=use_amp):
        pred = pipe.unet(model_input, timesteps, prompt_embeds, class_labels=noise_level).sample
        loss = F.mse_loss(pred.float(), target.float())
    return loss, pred, target


def param_matches_target(name: str, suffixes: tuple[str, ...]) -> bool:
    return any(f".{suffix}.weight" in name for suffix in suffixes)


def block_order(keys: Iterable[str]) -> dict[str, int]:
    ordered = sorted(keys, key=natural_key)
    return {key: idx + 1 for idx, key in enumerate(ordered)}


def select_gradient_blocks(rows: list[dict], args: argparse.Namespace) -> list[str]:
    row_by_block = {row["block"]: row for row in rows}
    if args.topk_policy == "grad_sandwich_plus":
        core = sandwich_core_blocks(row_by_block)
        selected = core[: args.topk_blocks]
        used = sum(int(row_by_block[block]["lora_param_count"]) for block in selected if block in row_by_block)
        ranked = sorted(rows, key=lambda row: row["selection_score"], reverse=True)
        for row in ranked:
            block = row["block"]
            if block in selected:
                continue
            params = int(row["lora_param_count"])
            if args.grad_probe_budget_params > 0 and used + params > args.grad_probe_budget_params:
                continue
            selected.append(block)
            used += params
            if len(selected) >= args.topk_blocks:
                break
        return selected[: args.topk_blocks]

    rows.sort(key=lambda row: row["selection_score"], reverse=True)
    if args.grad_probe_budget_params > 0:
        selected = []
        used = 0
        for row in rows:
            params = int(row["lora_param_count"])
            if used + params <= args.grad_probe_budget_params:
                selected.append(row["block"])
                used += params
            if len(selected) >= args.topk_blocks:
                break
        if selected:
            return selected
    return [row["block"] for row in rows[: args.topk_blocks]]


def probe_gradient_topk_blocks(
    pipe,
    loader: DataLoader,
    args: argparse.Namespace,
    device: torch.device,
    dtype: torch.dtype,
) -> list[str]:
    pipe.unet.requires_grad_(False)
    probe_info = inject_lora(
        pipe.unet,
        args.rank,
        args.alpha,
        args.target,
        "all",
    )
    move_lora_to_device(pipe.unet, device)
    pipe.unet.zero_grad(set_to_none=True)

    block_param_counts: dict[str, int] = {}
    for name, module in iter_lora_modules(pipe.unet):
        block_key = transformer_block_key(name)
        if block_key:
            block_param_counts[block_key] = block_param_counts.get(block_key, 0) + (
                module.lora_down.weight.numel() + module.lora_up.weight.numel()
            )

    if not block_param_counts:
        raise SystemExit(f"No gradient probe candidates found for target={args.target}")

    data_iter = iter(loader)
    rows = []
    total_loss = 0.0
    valid_batches = 0
    use_amp = False
    probe_batches = max(1, args.grad_probe_batches)
    for probe_idx in range(1, probe_batches + 1):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch = next(data_iter)

        loss, pred, target = diffusion_train_loss(pipe, batch, args, device, dtype, use_amp)
        if not torch.isfinite(loss):
            print(
                f"gradient probe batch {probe_idx}/{probe_batches} skipped: "
                f"loss_nonfinite,pred_finite={bool(torch.isfinite(pred).all().item())},"
                f"target_finite={bool(torch.isfinite(target).all().item())}"
            )
            pipe.unet.zero_grad(set_to_none=True)
            continue
        loss.backward()
        total_loss += float(loss.detach().cpu())
        valid_batches += 1
        print(f"gradient probe batch {probe_idx:03d}/{probe_batches} loss={float(loss.detach().cpu()):.6f}")

    if valid_batches == 0:
        raise SystemExit("Gradient probe failed: all probe batches produced non-finite loss.")

    block_grad_sq: dict[str, float] = {key: 0.0 for key in block_param_counts}
    for name, module in iter_lora_modules(pipe.unet):
        block_key = transformer_block_key(name)
        if block_key not in block_grad_sq:
            continue
        for param in (module.lora_down.weight, module.lora_up.weight):
            if param.grad is not None:
                block_grad_sq[block_key] += float(param.grad.detach().float().pow(2).sum().cpu())

    order = block_order(block_param_counts)
    total_blocks = len(order)
    for block, grad_sq in block_grad_sq.items():
        grad_norm = math.sqrt(grad_sq)
        param_count = block_param_counts[block]
        normalized_grad = grad_norm / math.sqrt(max(param_count, 1)) if args.grad_probe_normalize else grad_norm
        bp_cost = total_blocks - order[block] + 1
        selection_score = normalized_grad
        if args.grad_probe_compute_lambda > 0:
            selection_score = normalized_grad / (param_count + args.grad_probe_compute_lambda * bp_cost)
        rows.append(
            {
                "block": block,
                "block_index": order[block],
                "grad_norm": grad_norm,
                "lora_param_count": param_count,
                "normalized_grad_score": normalized_grad,
                "bp_cost": bp_cost,
                "compute_lambda": args.grad_probe_compute_lambda,
                "selection_score": selection_score,
                "probe_batches": valid_batches,
                "mean_probe_loss": total_loss / valid_batches,
                "selected": False,
            }
        )

    selected = select_gradient_blocks(rows, args)
    selected_set = set(selected)
    for row in rows:
        row["selected"] = row["block"] in selected_set

    output_dir = ensure_dir(args.output_dir)
    rows.sort(key=lambda row: row["selection_score"], reverse=True)
    save_csv(output_dir / "grad_topk_scores.csv", rows)
    unwrap_lora(pipe.unet)
    pipe.unet.zero_grad(set_to_none=True)
    pipe.unet.requires_grad_(False)
    del probe_info
    print(f"Gradient-TopK selected blocks: {selected}")
    return selected


def train(args: argparse.Namespace) -> None:
    require_torch()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    force_fp32 = args.train_method == "lora" and args.lora_scope == "grad_topk"
    if force_fp32 and args.fp16:
        print("Gradient-TopK-LoRA uses FP32 for stable gradient probing; ignoring --fp16 during training.")
    dtype = torch.float16 if device.type == "cuda" and args.fp16 and not force_fp32 else torch.float32
    pipe = load_pipeline(dtype=dtype).to(device)
    pipe.vae.requires_grad_(False)
    pipe.text_encoder.requires_grad_(False)
    pipe.unet.train()

    dataset = HrImageDataset(args.train_dir, args.hr_size, args.lr_size)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    if args.train_method == "lora":
        pipe.unet.requires_grad_(False)
        topk_block_names = None
        if args.lora_scope == "grad_topk":
            topk_block_names = probe_gradient_topk_blocks(pipe, loader, args, device, dtype)
            if args.reset_seed_after_probe:
                set_seed(args.seed)
        info = inject_lora(
            pipe.unet,
            args.rank,
            args.alpha,
            args.target,
            args.lora_scope,
            topk_blocks=args.topk_blocks,
            topk_policy=args.topk_policy,
            topk_block_names=topk_block_names,
        )
        move_lora_to_device(pipe.unet, device)
        trainable_label = "lora"
    elif args.train_method == "full_unet":
        pipe.unet.requires_grad_(True)
        info = LoraInfo(
            names=[],
            trainable_params=sum(p.numel() for p in pipe.unet.parameters()),
            total_params=sum(p.numel() for p in pipe.unet.parameters()),
            selected_blocks=[],
        )
        trainable_label = "full_unet"
    else:
        raise SystemExit(f"Unknown train_method={args.train_method}")

    params = [p for p in pipe.unet.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr)

    data_iter = iter(loader)

    use_amp = device.type == "cuda" and args.fp16 and not force_fp32
    scaler = make_grad_scaler(device, enabled=use_amp)
    logs = []
    start = time.time()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    start_mem = cuda_memory_stats_mb(device)
    optimizer.zero_grad(set_to_none=True)
    skipped_nonfinite_steps = 0
    freeze_state = init_freeze_state(pipe.unet) if args.dynamic_shallow_freeze and args.train_method == "lora" else None

    for step in range(1, args.train_steps + 1):
        accum_loss = 0.0
        step_nonfinite = False
        nonfinite_reason = ""
        did_backward = False
        for _ in range(args.grad_accum):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(loader)
                batch = next(data_iter)

            hr = batch["hr"].to(device=device, dtype=dtype)
            lr_up = batch["lr_up"].to(device=device, dtype=dtype)
            bsz = hr.shape[0]

            with torch.no_grad():
                latents = pipe.vae.encode(hr).latent_dist.sample()
                latents = latents * pipe.vae.config.scaling_factor
                noise = torch.randn_like(latents)
                timesteps = torch.randint(
                    0,
                    pipe.scheduler.config.num_train_timesteps,
                    (bsz,),
                    device=device,
                    dtype=torch.long,
                )
                noisy_latents = pipe.scheduler.add_noise(latents, noise, timesteps)
                noise_level = torch.full((bsz,), args.noise_level, device=device, dtype=torch.long)
                lr_cond_clean = F.interpolate(
                    lr_up.float(),
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

            with autocast_context(device, enabled=use_amp):
                pred = pipe.unet(model_input, timesteps, prompt_embeds, class_labels=noise_level).sample
                loss = F.mse_loss(pred.float(), target.float()) / args.grad_accum

            if not torch.isfinite(loss):
                step_nonfinite = True
                pred_finite = bool(torch.isfinite(pred).all().item())
                target_finite = bool(torch.isfinite(target).all().item())
                nonfinite_reason = f"loss_nonfinite,pred_finite={pred_finite},target_finite={target_finite}"
                break

            scaler.scale(loss).backward()
            did_backward = True
            accum_loss += float(loss.detach().cpu()) * args.grad_accum

        if step_nonfinite:
            skipped_nonfinite_steps += 1
            optimizer.zero_grad(set_to_none=True)
            if use_amp and did_backward:
                scaler.update()
        else:
            if args.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(params, args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            if (
                freeze_state is not None
                and step >= args.freeze_warmup_steps
                and args.freeze_check_interval > 0
                and step % args.freeze_check_interval == 0
            ):
                dynamic_freeze_check(pipe.unet, step, args, freeze_state)
            optimizer.zero_grad(set_to_none=True)

        elapsed = time.time() - start
        mem_stats = cuda_memory_stats_mb(device)
        logs.append(
            {
                "step": step,
                "loss": accum_loss if not step_nonfinite else "nan",
                "elapsed_s": elapsed,
                "step_nonfinite": step_nonfinite,
                "nonfinite_reason": nonfinite_reason,
                "did_backward": did_backward,
                "skipped_nonfinite_steps": skipped_nonfinite_steps,
                "frozen_shallow_blocks": len(freeze_state["frozen"]) if freeze_state is not None else 0,
                **mem_stats,
            }
        )
        if step % args.log_every == 0 or step == 1 or step == args.train_steps:
            mem_msg = ""
            if device.type == "cuda":
                mem_msg = (
                    f" cuda_alloc={mem_stats['cuda_mem_allocated_mb']:.1f}MB"
                    f" cuda_peak={mem_stats['cuda_max_mem_allocated_mb']:.1f}MB"
                    f" cuda_reserved={mem_stats['cuda_mem_reserved_mb']:.1f}MB"
                )
            if step_nonfinite:
                print(
                    f"step {step:05d}/{args.train_steps} loss=nan skipped_nonfinite={skipped_nonfinite_steps} "
                    f"reason={nonfinite_reason} elapsed={elapsed:.1f}s{mem_msg}"
                )
            else:
                print(f"step {step:05d}/{args.train_steps} loss={accum_loss:.6f} elapsed={elapsed:.1f}s{mem_msg}")

    output_dir = ensure_dir(args.output_dir)
    metadata = {
        "model_id": MODEL_ID,
        "train_method": args.train_method,
        "rank": args.rank,
        "alpha": args.alpha,
        "target": args.target,
        "lora_scope": args.lora_scope,
        "hr_size": args.hr_size,
        "lr_size": args.lr_size,
        "train_steps": args.train_steps,
        "seed": args.seed,
        "lora_module_names": info.names,
        "topk_blocks": args.topk_blocks,
        "topk_policy": args.topk_policy,
        "topk_block_names": info.selected_blocks,
        "grad_probe_batches": args.grad_probe_batches,
        "grad_probe_normalize": args.grad_probe_normalize,
        "grad_probe_compute_lambda": args.grad_probe_compute_lambda,
        "grad_probe_budget_params": args.grad_probe_budget_params,
        "reset_seed_after_probe": args.reset_seed_after_probe,
        "dynamic_shallow_freeze": args.dynamic_shallow_freeze,
        "freeze_warmup_steps": args.freeze_warmup_steps,
        "freeze_check_interval": args.freeze_check_interval,
        "freeze_patience": args.freeze_patience,
        "freeze_tau_update": args.freeze_tau_update,
        "freeze_tau_grad": args.freeze_tau_grad,
    }
    if args.train_method == "lora":
        weight_path = save_lora(pipe.unet, output_dir, metadata, save_dtype=args.save_lora_dtype)
        update_size_mb = weight_path.stat().st_size / (1024 * 1024)
        saved_update_path = str(weight_path)
    else:
        unet_dir = ensure_dir(output_dir / "unet")
        pipe.unet.save_pretrained(unet_dir, safe_serialization=True)
        metadata_path = output_dir / "full_unet_metadata.json"
        with metadata_path.open("w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        update_size_mb = sum(p.stat().st_size for p in unet_dir.rglob("*") if p.is_file()) / (1024 * 1024)
        saved_update_path = str(unet_dir)

    elapsed = time.time() - start
    energy_wh = args.power_w * elapsed / 3600.0
    effective_samples = args.train_steps * args.batch_size * args.grad_accum
    full_model_size_mb = args.full_model_size_mb
    compression_ratio = safe_div(full_model_size_mb, update_size_mb)
    trainable_param_pct = safe_div(info.trainable_params * 100.0, info.total_params)
    final_trainable_params = sum(p.numel() for p in pipe.unet.parameters() if p.requires_grad)
    final_trainable_param_pct = safe_div(final_trainable_params * 100.0, info.total_params)
    end_mem = cuda_memory_stats_mb(device)
    save_csv(output_dir / "train_log.csv", logs)
    if args.train_method == "lora":
        save_csv(output_dir / "lora_structure.csv", lora_structure_rows(pipe.unet))
    if freeze_state is not None:
        save_csv(output_dir / "dynamic_freeze_log.csv", freeze_state["events"])
    save_csv(
        output_dir / "summary.csv",
        [
            {
                "train_method": trainable_label,
                "saved_update_path": saved_update_path,
                "train_time_s": elapsed,
                "sec_per_step": safe_div(elapsed, args.train_steps),
                "effective_samples": effective_samples,
                "samples_per_second": safe_div(effective_samples, elapsed),
                "images_per_hour": safe_div(effective_samples * 3600.0, elapsed),
                "estimated_energy_wh": energy_wh,
                "fp16": args.fp16,
                "save_lora_dtype": args.save_lora_dtype if args.train_method == "lora" else "",
                "grad_clip": args.grad_clip,
                "skipped_nonfinite_steps": skipped_nonfinite_steps,
                "cuda_start_mem_allocated_mb": start_mem["cuda_mem_allocated_mb"],
                "cuda_start_mem_reserved_mb": start_mem["cuda_mem_reserved_mb"],
                "cuda_end_mem_allocated_mb": end_mem["cuda_mem_allocated_mb"],
                "cuda_end_mem_reserved_mb": end_mem["cuda_mem_reserved_mb"],
                "peak_cuda_mem_mb": end_mem["cuda_max_mem_allocated_mb"],
                "peak_cuda_reserved_mb": end_mem["cuda_max_mem_reserved_mb"],
                "adapter_size_mb": update_size_mb,
                "update_size_mb": update_size_mb,
                "full_model_size_mb": full_model_size_mb,
                "compression_ratio_full_to_adapter": compression_ratio,
                "upload_time_0_5mbps_s": upload_seconds(update_size_mb, 0.5, args.eta),
                "upload_time_1mbps_s": upload_seconds(update_size_mb, 1.0, args.eta),
                "upload_time_5mbps_s": upload_seconds(update_size_mb, 5.0, args.eta),
                "lora_module_count": len(info.names),
                "trainable_params": info.trainable_params,
                "final_trainable_params": final_trainable_params,
                "total_unet_params": info.total_params,
                "trainable_param_pct": trainable_param_pct,
                "final_trainable_param_pct": final_trainable_param_pct,
                "dynamic_shallow_freeze": args.dynamic_shallow_freeze,
                "frozen_shallow_blocks": len(freeze_state["frozen"]) if freeze_state is not None else 0,
                "freeze_warmup_steps": args.freeze_warmup_steps if freeze_state is not None else 0,
                "freeze_check_interval": args.freeze_check_interval if freeze_state is not None else 0,
                "freeze_patience": args.freeze_patience if freeze_state is not None else 0,
                "freeze_tau_update": args.freeze_tau_update if freeze_state is not None else 0,
                "freeze_tau_grad": args.freeze_tau_grad if freeze_state is not None else 0,
            }
        ],
    )
    print(f"Saved {args.train_method} update to {saved_update_path}")


def psnr(pred, target) -> float:
    import numpy as np

    pred = pred.astype("float32") / 255.0
    target = target.astype("float32") / 255.0
    mse = float(np.mean((pred - target) ** 2))
    return 99.0 if mse == 0 else 20.0 * math.log10(1.0 / math.sqrt(mse))


def eval_model(args: argparse.Namespace) -> None:
    require_torch()
    require_packages("skimage")
    import numpy as np
    from skimage.metrics import structural_similarity

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    dtype = torch.float16 if device.type == "cuda" and args.fp16 else torch.float32
    pipe = load_pipeline(dtype=dtype).to(device)
    pipe.set_progress_bar_config(disable=args.disable_progress)
    if args.unet_dir:
        require_packages("diffusers")
        from diffusers import UNet2DConditionModel

        pipe.unet = UNet2DConditionModel.from_pretrained(args.unet_dir, torch_dtype=dtype).to(device)
    if args.lora_dir:
        load_lora(pipe.unet, args.lora_dir, device=device, dtype=dtype)
    pipe.unet.eval()

    out = ensure_dir(args.output_dir)
    vis_dir = ensure_dir(out / "visuals")
    paths = list_images(args.val_dir)
    if args.eval_max_images > 0:
        paths = paths[: args.eval_max_images]
    rows = []
    start = time.time()

    for i, path in enumerate(paths):
        hr = Image.open(path).convert("RGB").resize((args.hr_size, args.hr_size), Image.BICUBIC)
        lr = hr.resize((args.lr_size, args.lr_size), Image.BICUBIC)
        t0 = time.time()
        with torch.no_grad():
            result = pipe(
                prompt="",
                image=lr,
                num_inference_steps=args.num_inference_steps,
                noise_level=args.noise_level,
            ).images[0]
        infer_s = time.time() - t0
        result = result.resize((args.hr_size, args.hr_size), Image.BICUBIC)
        pred_np = np.array(result)
        hr_np = np.array(hr)
        bic_np = np.array(lr.resize((args.hr_size, args.hr_size), Image.BICUBIC))
        row = {
            "image": str(path),
            "psnr": psnr(pred_np, hr_np),
            "ssim": structural_similarity(pred_np, hr_np, channel_axis=2, data_range=255),
            "bicubic_psnr": psnr(bic_np, hr_np),
            "bicubic_ssim": structural_similarity(bic_np, hr_np, channel_axis=2, data_range=255),
            "inference_time_s": infer_s,
        }
        row["delta_psnr_vs_bicubic"] = row["psnr"] - row["bicubic_psnr"]
        row["delta_ssim_vs_bicubic"] = row["ssim"] - row["bicubic_ssim"]
        rows.append(row)

        if i < args.save_visuals:
            canvas = Image.new("RGB", (args.hr_size * 3, args.hr_size))
            canvas.paste(lr.resize((args.hr_size, args.hr_size), Image.BICUBIC), (0, 0))
            canvas.paste(result, (args.hr_size, 0))
            canvas.paste(hr, (args.hr_size * 2, 0))
            canvas.save(vis_dir / f"vis_{i:03d}.png")
        print(f"[{i + 1}/{len(paths)}] PSNR={row['psnr']:.3f} SSIM={row['ssim']:.4f}")

    save_csv(out / "eval_metrics.csv", rows)
    if rows:
        mean_psnr = sum(r["psnr"] for r in rows) / len(rows)
        mean_ssim = sum(r["ssim"] for r in rows) / len(rows)
        mean_bicubic_psnr = sum(r["bicubic_psnr"] for r in rows) / len(rows)
        mean_bicubic_ssim = sum(r["bicubic_ssim"] for r in rows) / len(rows)
        mean_inference_time_s = sum(r["inference_time_s"] for r in rows) / len(rows)
        adapter_mb = 0.0
        estimated_energy_wh = 0.0
        if args.train_summary_csv:
            with Path(args.train_summary_csv).open("r", encoding="utf-8") as f:
                train_summary = next(csv.DictReader(f))
            adapter_mb = float(train_summary.get("adapter_size_mb") or 0)
            estimated_energy_wh = float(train_summary.get("estimated_energy_wh") or 0)

        summary = {
            "num_images": len(rows),
            "mean_psnr": mean_psnr,
            "mean_ssim": mean_ssim,
            "mean_bicubic_psnr": mean_bicubic_psnr,
            "mean_bicubic_ssim": mean_bicubic_ssim,
            "delta_psnr_vs_bicubic": mean_psnr - mean_bicubic_psnr,
            "delta_ssim_vs_bicubic": mean_ssim - mean_bicubic_ssim,
            "mean_inference_time_s": mean_inference_time_s,
            "images_per_hour_inference": safe_div(len(rows) * 3600.0, sum(r["inference_time_s"] for r in rows)),
            "total_eval_time_s": time.time() - start,
        }
        if args.base_summary_csv:
            with Path(args.base_summary_csv).open("r", encoding="utf-8") as f:
                base_summary = next(csv.DictReader(f))
            base_psnr = float(base_summary.get("mean_psnr") or 0)
            base_ssim = float(base_summary.get("mean_ssim") or 0)
            summary["base_mean_psnr"] = base_psnr
            summary["base_mean_ssim"] = base_ssim
            summary["delta_psnr_vs_base"] = mean_psnr - base_psnr
            summary["delta_ssim_vs_base"] = mean_ssim - base_ssim
        if adapter_mb:
            psnr_gain = summary.get("delta_psnr_vs_base", summary["delta_psnr_vs_bicubic"])
            summary["adapter_size_mb"] = adapter_mb
            summary["psnr_gain_per_mb"] = safe_div(psnr_gain, adapter_mb)
            if estimated_energy_wh:
                summary["estimated_energy_wh"] = estimated_energy_wh
                summary["psnr_gain_per_wh"] = safe_div(psnr_gain, estimated_energy_wh)
        save_csv(out / "eval_summary.csv", [summary])
        print(json.dumps(summary, indent=2))


def parse_tegrastats_line(line: str, interval_s: float) -> dict:
    ram = re.search(r"RAM\s+(\d+)/(\d+)MB", line)
    temps = [float(x) for x in re.findall(r"@([0-9.]+)C", line)]
    power_tokens = re.findall(r"\b(\d+)mW/\d+mW\b", line)
    current_mw = [float(token) for token in power_tokens]
    return {
        "ram_used_mb": float(ram.group(1)) if ram else 0.0,
        "ram_total_mb": float(ram.group(2)) if ram else 0.0,
        "max_temp_c": max(temps) if temps else 0.0,
        "avg_temp_c": safe_div(sum(temps), len(temps)),
        "total_power_w": sum(current_mw) / 1000.0,
        "sample_energy_wh": sum(current_mw) / 1000.0 * interval_s / 3600.0,
    }


def parse_tegrastats(args: argparse.Namespace) -> None:
    log_path = Path(args.tegrastats_log)
    rows = []
    with log_path.open("r", encoding="utf-8", errors="ignore") as f:
        for idx, line in enumerate(f):
            row = parse_tegrastats_line(line, args.tegrastats_interval_s)
            row["sample"] = idx
            rows.append(row)
    if not rows:
        raise SystemExit(f"No tegrastats samples found in {log_path}")

    save_csv(args.output_csv, rows)
    summary = {
        "num_samples": len(rows),
        "duration_s": len(rows) * args.tegrastats_interval_s,
        "mean_power_w": sum(r["total_power_w"] for r in rows) / len(rows),
        "peak_power_w": max(r["total_power_w"] for r in rows),
        "estimated_energy_wh_from_log": sum(r["sample_energy_wh"] for r in rows),
        "peak_ram_used_mb": max(r["ram_used_mb"] for r in rows),
        "ram_total_mb": max(r["ram_total_mb"] for r in rows),
        "peak_temp_c": max(r["max_temp_c"] for r in rows),
        "mean_temp_c": sum(r["avg_temp_c"] for r in rows) / len(rows),
    }
    summary_path = Path(args.output_csv).with_name(Path(args.output_csv).stem + "_summary.csv")
    save_csv(summary_path, [summary])
    print(json.dumps(summary, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        required=True,
        choices=["prepare_ucmerced", "comm", "train", "eval", "parse_tegrastats", "inspect_lora", "quantize_lora"],
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--fp16", dest="fp16", action="store_true", default=True)
    parser.add_argument("--no_fp16", dest="fp16", action="store_false")

    parser.add_argument("--data_root", default="data/ucmerced")
    parser.add_argument("--dataset_name", default="blanchon/UC_Merced")
    parser.add_argument("--dataset_split", default="train")
    parser.add_argument("--train_ratio", type=float, default=0.8)

    parser.add_argument("--update_sizes_mb", type=float, nargs="+", default=[1200, 80, 30, 15, 7.5])
    parser.add_argument("--uplink_mbps", type=float, nargs="+", default=[0.128, 0.5, 1, 5, 10])
    parser.add_argument("--contact_window_s", type=float, default=600)
    parser.add_argument("--eta", type=float, default=0.7)
    parser.add_argument("--output_csv", default="outputs/comm_feasibility.csv")

    parser.add_argument("--train_dir", default="data/ucmerced/train_hr")
    parser.add_argument("--val_dir", default="data/ucmerced/val_hr")
    parser.add_argument("--output_dir", default="outputs/lora_sandwich_r8")
    parser.add_argument("--lora_dir", default="")
    parser.add_argument("--unet_dir", default="")
    parser.add_argument("--hr_size", type=int, default=256)
    parser.add_argument("--lr_size", type=int, default=64)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--alpha", type=int, default=16)
    parser.add_argument("--target", default="qv", choices=["q", "v", "qv", "qkvo"])
    parser.add_argument(
        "--lora_scope",
        default="shallow_deep",
        choices=["shallow", "last2_up", "shallow_deep", "topk", "grad_topk", "all"],
    )
    parser.add_argument("--topk_blocks", type=int, default=5)
    parser.add_argument(
        "--topk_policy",
        default="balanced",
        choices=["balanced", "uniform", "early", "late", "sandwich_plus", "grad_sandwich_plus"],
    )
    parser.add_argument("--grad_probe_batches", type=int, default=20)
    parser.add_argument("--grad_probe_normalize", dest="grad_probe_normalize", action="store_true", default=True)
    parser.add_argument("--no_grad_probe_normalize", dest="grad_probe_normalize", action="store_false")
    parser.add_argument("--grad_probe_compute_lambda", type=float, default=0.0)
    parser.add_argument("--grad_probe_budget_params", type=int, default=0)
    parser.add_argument("--reset_seed_after_probe", dest="reset_seed_after_probe", action="store_true", default=True)
    parser.add_argument("--no_reset_seed_after_probe", dest="reset_seed_after_probe", action="store_false")
    parser.add_argument("--dynamic_shallow_freeze", action="store_true")
    parser.add_argument("--freeze_warmup_steps", type=int, default=20)
    parser.add_argument("--freeze_check_interval", type=int, default=10)
    parser.add_argument("--freeze_patience", type=int, default=3)
    parser.add_argument("--freeze_tau_update", type=float, default=1e-3)
    parser.add_argument("--freeze_tau_grad", type=float, default=1e-5)
    parser.add_argument("--save_lora_dtype", default="fp32", choices=["fp32", "fp16"])
    parser.add_argument("--train_method", default="lora", choices=["lora", "full_unet"])
    parser.add_argument("--train_steps", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--power_w", type=float, default=30)
    parser.add_argument("--full_model_size_mb", type=float, default=1200)
    parser.add_argument("--noise_level", type=int, default=20)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--log_every", type=int, default=10)

    parser.add_argument("--eval_max_images", type=int, default=20)
    parser.add_argument("--num_inference_steps", type=int, default=25)
    parser.add_argument("--save_visuals", type=int, default=8)
    parser.add_argument("--base_summary_csv", default="")
    parser.add_argument("--train_summary_csv", default="")
    parser.add_argument("--disable_progress", action="store_true")

    parser.add_argument("--tegrastats_log", default="outputs/tegrastats_sandwich_train.log")
    parser.add_argument("--tegrastats_interval_s", type=float, default=1.0)
    parser.add_argument("--inspect_load_model", action="store_true")
    parser.add_argument("--inspect_zero_tol", type=float, default=0.0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.mode == "prepare_ucmerced":
        prepare_ucmerced(args)
    elif args.mode == "comm":
        comm(args)
    elif args.mode == "train":
        train(args)
    elif args.mode == "eval":
        eval_model(args)
    elif args.mode == "parse_tegrastats":
        parse_tegrastats(args)
    elif args.mode == "inspect_lora":
        inspect_lora(args)
    elif args.mode == "quantize_lora":
        quantize_lora(args)
    else:
        raise SystemExit(f"Unsupported mode: {args.mode}")


if __name__ == "__main__":
    main()
