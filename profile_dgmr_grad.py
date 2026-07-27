#!/usr/bin/env python3
"""Gradient profiling for DGMR SEN12MS-CR.

Copy this file into the DGMR repository root or run it from
dgmr_code/SEN12MS-CR. It profiles DGMR's official optimize_parameters loss
without saving checkpoints and without applying optimizer updates.
"""

from __future__ import annotations

import argparse
import csv
import importlib.machinery
import math
import os
import random
import sys
import types
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


def install_runtime_stubs() -> None:
    """Avoid optional import failures on minimal Jetson environments."""
    if "torchvision" not in sys.modules:
        tv = types.ModuleType("torchvision")
        tv.__spec__ = importlib.machinery.ModuleSpec("torchvision", loader=None)
        tv.transforms = types.ModuleType("torchvision.transforms")
        tv.transforms.__spec__ = importlib.machinery.ModuleSpec("torchvision.transforms", loader=None)
        tv.utils = types.ModuleType("torchvision.utils")
        tv.utils.__spec__ = importlib.machinery.ModuleSpec("torchvision.utils", loader=None)
        tv.utils.make_grid = lambda x, *args, **kwargs: x
        sys.modules["torchvision"] = tv
        sys.modules["torchvision.transforms"] = tv.transforms
        sys.modules["torchvision.utils"] = tv.utils

    if "timm" not in sys.modules:
        timm = types.ModuleType("timm")
        models = types.ModuleType("timm.models")
        layers = types.ModuleType("timm.models.layers")

        class DropPath(nn.Identity):
            pass

        def to_2tuple(x):
            return x if isinstance(x, tuple) else (x, x)

        def trunc_normal_(tensor, mean=0.0, std=1.0, *args, **kwargs):
            return nn.init.trunc_normal_(tensor, mean=mean, std=std)

        layers.DropPath = DropPath
        layers.to_2tuple = to_2tuple
        layers.trunc_normal_ = trunc_normal_
        models.layers = layers
        timm.models = models
        sys.modules["timm"] = timm
        sys.modules["timm.models"] = models
        sys.modules["timm.models.layers"] = layers

    if "focal_frequency_loss" not in sys.modules:
        ffl_mod = types.ModuleType("focal_frequency_loss")

        class FocalFrequencyLoss(nn.Module):
            def __init__(self, *args, **kwargs):
                super().__init__()

            def forward(self, x, target):
                return torch.mean(torch.abs(x - target))

        ffl_mod.FocalFrequencyLoss = FocalFrequencyLoss
        sys.modules["focal_frequency_loss"] = ffl_mod


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def natural_key(text: str) -> list:
    import re

    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", text)]


def block_key(name: str, block_regex: str = "") -> str:
    import re

    if block_regex:
        m = re.search(block_regex, name)
        if m:
            return m.group(1) if m.groups() else m.group(0)

    parts = name.split(".")
    for marker in (
        "layers",
        "blocks",
        "encoder",
        "decoder",
        "down",
        "up",
        "body",
        "resblocks",
        "patch_embed",
        "patch_unembed",
    ):
        if marker in parts:
            idx = parts.index(marker)
            if idx + 1 < len(parts):
                return ".".join(parts[max(idx - 1, 0) : idx + 2])
            return marker

    # Keep major DGMR sub-networks separate when no obvious block name exists.
    if name.startswith("net_G."):
        return ".".join(parts[: min(3, len(parts))])
    if name.startswith("diffusion."):
        return ".".join(parts[: min(4, len(parts))])
    return ".".join(parts[: min(2, len(parts))]) or "__root__"


def target_match(name: str, target: str) -> bool:
    leaf = name.split(".")[-1]
    full = name.lower()
    if target == "qkv":
        return leaf in {"qkv", "q", "k", "v", "to_q", "to_k", "to_v", "q_proj", "k_proj", "v_proj"}
    if target == "attention_linear":
        return any(key in full for key in ("attn", "attention", "qkv", "proj"))
    if target == "all_linear":
        return True
    if target == "all_conv_linear":
        return True
    raise SystemExit(f"Unknown target={target}")


def iter_dgmr_named_modules(root):
    if isinstance(root, nn.Module):
        yield from root.named_modules()
        return
    for prefix in ("net_G", "diffusion"):
        module = getattr(root, prefix, None)
        if isinstance(module, nn.Module):
            for name, child in module.named_modules():
                full_name = prefix if not name else f"{prefix}.{name}"
                yield full_name, child


def iter_dgmr_parameters(root):
    if isinstance(root, nn.Module):
        yield from root.parameters()
        return
    for prefix in ("net_G", "diffusion"):
        module = getattr(root, prefix, None)
        if isinstance(module, nn.Module):
            yield from module.parameters()


def iter_target_modules(root, target: str, block_regex: str):
    module_types = (nn.Linear, nn.Conv2d) if target == "all_conv_linear" else (nn.Linear,)
    for name, module in iter_dgmr_named_modules(root):
        if isinstance(module, module_types) and target_match(name, target):
            bkey = block_key(name, block_regex)
            if bkey:
                yield name, module


def enable_target_grads(root, target: str, block_regex: str):
    modules = list(iter_target_modules(root, target, block_regex))
    if not modules:
        raise SystemExit("No target modules found. Try --target all_conv_linear or run --inspect_only.")
    for _, module in modules:
        for param in module.parameters(recurse=False):
            param.requires_grad_(True)
    return modules


def module_grad_norm_and_params(module: nn.Module) -> tuple[float, int]:
    total = 0.0
    count = 0
    for param in module.parameters(recurse=False):
        count += param.numel()
        if param.grad is not None:
            total += float(param.grad.detach().float().pow(2).sum().cpu())
    return math.sqrt(total), count


def write_csv(path: str | Path, rows: list[dict]) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    if not rows:
        raise SystemExit(f"No rows to write: {path}")
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def make_opts(args: argparse.Namespace):
    opts = argparse.Namespace()
    opts.batch_sz = args.batch_size
    opts.load_size = args.load_size
    opts.crop_size = args.crop_size
    opts.input_data_folder = args.data_root
    opts.is_use_cloudmask = True
    opts.cloud_threshold = 0.2
    opts.data_list_filepath = args.data_list_filepath
    opts.optimizer = "Adam"
    opts.lr = args.lr
    opts.lr_step = 5
    opts.lr_start_epoch_decay = 0
    opts.max_epochs = 1
    opts.save_freq = 999999
    opts.log_freq = 999999
    opts.save_model_dir = str(Path(args.output_dir) / "_no_save_dgmr")
    opts.save_model_dir1 = str(Path(args.output_dir) / "_no_save_cdgm")
    opts.is_test = False
    opts.load_pretrained_model = bool(args.ckpt_path)
    opts.pretrained_model = args.ckpt_path
    opts.gpu_ids = args.gpu_ids
    return opts


def locate_sen12_dir(start: Path) -> Path:
    if (start / "dgmr.py").exists() and (start / "dataload_new_128.py").exists():
        return start
    candidate = start / "dgmr_code" / "SEN12MS-CR"
    if (candidate / "dgmr.py").exists():
        return candidate
    raise SystemExit("Run from DGMR root or dgmr_code/SEN12MS-CR; cannot find dgmr.py.")


def make_synthetic_batch(batch_size: int, crop_size: int, device: torch.device) -> dict:
    return {
        "input": {
            "S2": torch.rand(batch_size, 13, crop_size, crop_size, device=device),
            "S1": torch.rand(batch_size, 2, crop_size, crop_size, device=device),
            "masks": torch.ones(batch_size, crop_size, crop_size, device=device),
        },
        "target": {
            "S2": torch.rand(batch_size, 13, crop_size, crop_size, device=device),
        },
    }


def move_to_device(value, device: torch.device):
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, dict):
        return {k: move_to_device(v, device) for k, v in value.items()}
    if isinstance(value, list):
        return [move_to_device(v, device) for v in value]
    if isinstance(value, tuple):
        return tuple(move_to_device(v, device) for v in value)
    return value


def chw_array(array: np.ndarray) -> np.ndarray:
    if array.ndim == 3 and array.shape[-1] in (1, 2, 3, 4, 13):
        return np.transpose(array, (2, 0, 1))
    if array.ndim == 3:
        return array
    raise ValueError(f"Expected HWC or CHW array, got shape={array.shape}")


def normalize_optical(array: np.ndarray) -> np.ndarray:
    array = array.astype(np.float32, copy=False)
    if float(np.nanmax(array)) > 2.0:
        array = array / 10000.0
    return np.clip(array, 0.0, 1.0)


def normalize_sar(array: np.ndarray) -> np.ndarray:
    array = array.astype(np.float32, copy=False)
    if float(np.nanmin(array)) < -1.0 or float(np.nanmax(array)) > 2.0:
        array = np.clip(array, -25.0, 0.0)
        array = (array + 25.0) / 25.0
    return np.clip(array, 0.0, 1.0)


def crop_chw(array: np.ndarray, crop_size: int, rng: random.Random) -> np.ndarray:
    _, height, width = array.shape
    if height < crop_size or width < crop_size:
        tensor = torch.from_numpy(array).unsqueeze(0)
        tensor = torch.nn.functional.interpolate(
            tensor,
            size=(crop_size, crop_size),
            mode="bilinear",
            align_corners=False,
        )
        return tensor.squeeze(0).numpy()
    top = rng.randint(0, height - crop_size) if height > crop_size else 0
    left = rng.randint(0, width - crop_size) if width > crop_size else 0
    return array[:, top : top + crop_size, left : left + crop_size]


def decode_hf_sen12mscr_sample(sample: dict, crop_size: int, rng: random.Random) -> dict:
    sar = np.frombuffer(sample["sar"], dtype=np.float32).reshape(sample["sar_shape"])
    cloudy = np.frombuffer(sample["cloudy"], dtype=np.int16).reshape(sample["opt_shape"])
    target = np.frombuffer(sample["target"], dtype=np.int16).reshape(sample["opt_shape"])

    sar = crop_chw(normalize_sar(chw_array(sar)), crop_size, rng)
    cloudy = crop_chw(normalize_optical(chw_array(cloudy)), crop_size, rng)
    target = crop_chw(normalize_optical(chw_array(target)), crop_size, rng)

    return {
        "input": {
            "S2": torch.from_numpy(np.ascontiguousarray(cloudy)),
            "S1": torch.from_numpy(np.ascontiguousarray(sar)),
            "masks": torch.ones(crop_size, crop_size, dtype=torch.float32),
        },
        "target": {
            "S2": torch.from_numpy(np.ascontiguousarray(target)),
        },
    }


def collate_dgmr_batches(samples: list[dict], device: torch.device) -> dict:
    return {
        "input": {
            "S2": torch.stack([s["input"]["S2"] for s in samples], dim=0).to(device),
            "S1": torch.stack([s["input"]["S1"] for s in samples], dim=0).to(device),
            "masks": torch.stack([s["input"]["masks"] for s in samples], dim=0).to(device),
        },
        "target": {
            "S2": torch.stack([s["target"]["S2"] for s in samples], dim=0).to(device),
        },
    }


def hf_sen12mscr_iter(args: argparse.Namespace, device: torch.device):
    from datasets import load_dataset

    ds = load_dataset(args.hf_dataset, split=args.hf_split, streaming=True)
    rng = random.Random(args.seed)
    batch = []
    seen = 0
    while True:
        for sample in ds:
            batch.append(decode_hf_sen12mscr_sample(sample, args.crop_size, rng))
            seen += 1
            if len(batch) == args.batch_size:
                yield collate_dgmr_batches(batch, device)
                batch = []
            if args.hf_max_samples > 0 and seen >= args.hf_max_samples:
                seen = 0
                break


def cached_sample_iter(args: argparse.Namespace, device: torch.device):
    cache_dir = Path(args.cached_sample_dir)
    files = sorted(cache_dir.glob("*.npz"), key=lambda p: natural_key(p.name))
    if not files:
        raise SystemExit(f"No cached .npz samples found in {cache_dir}")

    idx = 0
    while True:
        batch = []
        for _ in range(args.batch_size):
            path = files[idx % len(files)]
            idx += 1
            with np.load(path) as sample:
                batch.append(
                    {
                        "input": {
                            "S2": torch.from_numpy(sample["cloudy"].astype(np.float32, copy=False)),
                            "S1": torch.from_numpy(sample["sar"].astype(np.float32, copy=False)),
                            "masks": torch.from_numpy(sample["mask"].astype(np.float32, copy=False)),
                        },
                        "target": {
                            "S2": torch.from_numpy(sample["target"].astype(np.float32, copy=False)),
                        },
                    }
                )
        yield collate_dgmr_batches(batch, device)


def load_data_iter(args: argparse.Namespace, sen12_dir: Path, device: torch.device):
    if args.cached_sample_dir:
        yield from cached_sample_iter(args, device)

    if args.hf_dataset:
        yield from hf_sen12mscr_iter(args, device)

    if args.synthetic:
        while True:
            yield make_synthetic_batch(args.batch_size, args.crop_size, device)

    from dataload_new_128 import SEN12MSCR_train

    dataset = SEN12MSCR_train(args.data_root, split=args.split, region=args.region)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=False,
    )
    while True:
        for batch in loader:
            yield move_to_device(batch, device)


def disable_optimizer_steps(model) -> None:
    for opt_name in ("optimizer_G", "optimizer_G1"):
        opt = getattr(model, opt_name, None)
        if opt is not None:
            opt.step = lambda *args, **kwargs: None
    model.save_checkpoint = lambda *args, **kwargs: None


def inspect_model(model, args: argparse.Namespace) -> None:
    print("== Candidate modules ==")
    for name, module in iter_dgmr_named_modules(model):
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            if args.target == "all_conv_linear":
                mark = "*"
            else:
                mark = "*" if isinstance(module, nn.Linear) and target_match(name, args.target) else " "
            shape = ""
            if isinstance(module, nn.Linear):
                shape = f"{module.in_features}->{module.out_features}"
            elif isinstance(module, nn.Conv2d):
                shape = f"{module.in_channels}->{module.out_channels} k={module.kernel_size}"
            print(f"{mark} {name} [{module.__class__.__name__} {shape}] block={block_key(name, args.block_regex)}")


def profile(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    install_runtime_stubs()
    sen12_dir = locate_sen12_dir(Path.cwd())
    sys.path.insert(0, str(sen12_dir))
    os.chdir(sen12_dir)

    from dgmr import DGMR

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    opts = make_opts(args)
    model = DGMR(opts)
    disable_optimizer_steps(model)

    if args.inspect_only:
        inspect_model(model, args)
        return

    for param in iter_dgmr_parameters(model):
        param.requires_grad_(False)
    targets = enable_target_grads(model, args.target, args.block_regex)

    data_iter = load_data_iter(args, sen12_dir, device)
    total_loss = 0.0
    valid = 0
    accum: dict[str, dict] = {}

    for step in range(1, args.probe_batches + 1):
        batch = next(data_iter)
        model.set_input(batch)
        loss_value = model.optimize_parameters(epoch=0)
        loss_tensor = getattr(model, "loss_G", None)
        if not torch.is_tensor(loss_tensor):
            print(f"probe batch {step}/{args.probe_batches} skipped: no tensor loss_G")
            continue
        if not torch.isfinite(loss_tensor.detach()):
            print(f"probe batch {step}/{args.probe_batches} skipped: non-finite loss")
            continue

        valid += 1
        total_loss += float(loss_tensor.detach().cpu())
        print(f"probe batch {step:03d}/{args.probe_batches} loss={float(loss_tensor.detach().cpu()):.6f}")

        for name, module in targets:
            bkey = block_key(name, args.block_regex)
            grad_norm, param_count = module_grad_norm_and_params(module)
            row = accum.setdefault(bkey, {"grad_norm": 0.0, "param_count": 0, "module_count": 0})
            row["grad_norm"] += grad_norm
            row["param_count"] += param_count
            row["module_count"] += 1

    if valid == 0:
        raise SystemExit("No valid probe batches.")

    rows = []
    blocks = sorted(accum, key=natural_key)
    total_blocks = max(len(blocks), 1)
    for idx, block in enumerate(blocks):
        row = accum[block]
        p_count = max(int(row["param_count"]), 1)
        score = row["grad_norm"] / math.sqrt(p_count)
        rows.append(
            {
                "block": block,
                "block_index": idx,
                "grad_norm": row["grad_norm"],
                "weight_param_count": int(row["param_count"]),
                "module_count": int(row["module_count"]),
                "normalized_grad_score": score,
                "bp_cost": total_blocks - idx,
                "selection_score": score,
                "probe_batches": valid,
                "mean_probe_loss": total_loss / valid,
                "selected": False,
            }
        )

    selected = {r["block"] for r in sorted(rows, key=lambda r: r["selection_score"], reverse=True)[: args.topk_blocks]}
    for row in rows:
        row["selected"] = row["block"] in selected

    out_dir = ensure_dir(args.output_dir)
    write_csv(out_dir / "dgmr_grad_scores.csv", rows)
    print(f"Wrote {out_dir / 'dgmr_grad_scores.csv'}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_root", default="", help="SEN12MS-CR root directory")
    parser.add_argument("--data_list_filepath", default="")
    parser.add_argument("--hf_dataset", default="", help="Streaming Hugging Face dataset id, e.g. Hermanni/sen12mscr")
    parser.add_argument("--hf_split", default="train")
    parser.add_argument("--hf_max_samples", type=int, default=0, help="Restart stream after this many samples; 0 means no limit")
    parser.add_argument("--cached_sample_dir", default="", help="Directory of cached SEN12MS-CR .npz samples")
    parser.add_argument("--ckpt_path", default="")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--target", default="all_linear", choices=["qkv", "attention_linear", "all_linear", "all_conv_linear"])
    parser.add_argument("--split", default="train", choices=["all", "train", "val", "test"])
    parser.add_argument("--region", default="all")
    parser.add_argument("--load_size", type=int, default=128)
    parser.add_argument("--crop_size", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--probe_batches", type=int, default=2)
    parser.add_argument("--topk_blocks", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--gpu_ids", default="0")
    parser.add_argument("--block_regex", default="")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic SEN12MS-like tensors for a smoke test")
    parser.add_argument("--inspect_only", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    profile(args)
    if args.hf_dataset:
        # pyarrow/aiohttp streaming can crash during interpreter finalization on
        # some Jetson Python builds. The profiling outputs are already flushed.
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)


if __name__ == "__main__":
    main()
