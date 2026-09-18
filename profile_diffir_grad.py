#!/usr/bin/env python3
"""Profile packed-QKV LoRA importance for the official DiffIR-S2 model.

Copy this file to the ``DiffIR-SRGAN`` repository root and run it there.  The
profiler keeps the pretrained network frozen, inserts zero-initialized LoRA
probes into every DIRformer ``*.attn.qkv`` 1x1 convolution, and differentiates
DiffIR-S2's official stage-two objective (pixel L1 plus final-prior L1).

DiffIR's four diffusion iterations estimate a compact restoration-prior vector;
the 44 DIRformer blocks run once after that process.  Consequently block LoRA
importance is reported once over the complete objective, while per-iteration
prior errors and gradient contributions are written to a separate CSV.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import re
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageFile


IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
ImageFile.LOAD_TRUNCATED_IMAGES = True


def natural_key(value: str) -> list[int | str]:
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", value)]


def set_seed(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write an empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def pixel_sha256(path: Path) -> str:
    with Image.open(path) as image:
        image = image.convert("RGB")
        digest = hashlib.sha256()
        digest.update(f"RGB:{image.width}x{image.height}:".encode("ascii"))
        digest.update(image.tobytes())
    return digest.hexdigest()


def discover_images(root: Path, max_images: int, seed: int) -> list[Path]:
    if not root.is_dir():
        raise FileNotFoundError(root)
    paths = sorted(
        (
            path
            for path in root.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        ),
        key=lambda path: natural_key(str(path)),
    )
    if not paths:
        raise FileNotFoundError(f"No images found under {root}")
    if 0 < max_images < len(paths):
        paths = sorted(
            random.Random(seed).sample(paths, max_images),
            key=lambda path: natural_key(str(path)),
        )
    return paths


def audit_overlap(
    calibration_paths: list[Path], eval_dirs: list[Path]
) -> dict[str, Any]:
    if not eval_dirs:
        return {"enabled": False}
    for eval_dir in eval_dirs:
        if not eval_dir.is_dir():
            raise FileNotFoundError(eval_dir)
    eval_paths = []
    for eval_dir in eval_dirs:
        eval_paths.extend(
            path
            for path in eval_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        )
    eval_hashes = {pixel_sha256(path) for path in eval_paths}
    overlaps = [str(path) for path in calibration_paths if pixel_sha256(path) in eval_hashes]
    if overlaps:
        raise RuntimeError(
            "Calibration images overlap the protected evaluation set; first matches: "
            + ", ".join(overlaps[:5])
        )
    print(
        f"Dataset overlap audit: calibration={len(calibration_paths)} "
        f"protected_eval={len(eval_paths)} overlap=0"
    )
    return {
        "enabled": True,
        "protected_eval_dirs": [str(path.resolve()) for path in eval_dirs],
        "num_protected_eval_images": len(eval_paths),
        "overlap_count": 0,
    }


class ImageFolderDataset:
    def __init__(
        self,
        root: Path,
        image_size: int,
        lq_size: int,
        max_images: int,
        seed: int,
    ) -> None:
        self.paths = discover_images(root, max_images, seed)
        self.image_size = image_size
        self.lq_size = lq_size

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> dict[str, Any]:
        import torch
        from basicsr.utils.matlab_functions import imresize

        path = self.paths[index]
        with Image.open(path) as image:
            image = image.convert("RGB")
            if image.size != (self.image_size, self.image_size):
                image = image.resize(
                    (self.image_size, self.image_size), Image.Resampling.BICUBIC
                )
            array = np.asarray(image, dtype=np.float32) / 255.0
        gt = torch.from_numpy(array).permute(2, 0, 1).contiguous()
        lq = imresize(gt, scale=self.lq_size / self.image_size, antialiasing=True)
        lq = lq.clamp(0.0, 1.0).contiguous()
        return {"gt": gt, "lq": lq, "path": str(path)}


def extract_state_dict(payload: Any, checkpoint_key: str) -> tuple[dict[str, Any], str]:
    if not isinstance(payload, dict):
        raise TypeError(f"Checkpoint must be a dictionary, found {type(payload).__name__}")
    if checkpoint_key != "auto":
        if checkpoint_key == "root":
            state = payload
        elif checkpoint_key in payload and isinstance(payload[checkpoint_key], dict):
            state = payload[checkpoint_key]
        else:
            raise KeyError(f"Checkpoint has no dictionary key {checkpoint_key!r}")
        selected_key = checkpoint_key
    else:
        state = payload
        selected_key = "root"
        for key in ("params_ema", "params", "state_dict", "model"):
            if key in payload and isinstance(payload[key], dict):
                state = payload[key]
                selected_key = key
                break
    for prefix in ("module.", "net_g."):
        if state and all(str(key).startswith(prefix) for key in state):
            state = {str(key)[len(prefix) :]: value for key, value in state.items()}
            selected_key += f" (stripped {prefix})"
    return state, selected_key


def model_kwargs() -> dict[str, Any]:
    return {
        "n_encoder_res": 9,
        "inp_channels": 3,
        "out_channels": 3,
        "scale": 4,
        "dim": 64,
        "num_blocks": [13, 1, 1, 1],
        "num_refinement_blocks": 13,
        "heads": [1, 2, 4, 8],
        "ffn_expansion_factor": 2.2,
        "bias": False,
        "LayerNorm_type": "BiasFree",
    }


def load_models(s2_path: Path, s1_path: Path, checkpoint_key: str, device):
    import torch
    from DiffIR.archs.S1_arch import DiffIRS1
    from DiffIR.archs.S2_arch import DiffIRS2

    s2 = DiffIRS2(
        **model_kwargs(),
        n_denoise_res=1,
        linear_start=0.1,
        linear_end=0.99,
        timesteps=4,
    )
    s1 = DiffIRS1(**model_kwargs())

    loaded_keys = {}
    for label, model, path in (("s2", s2, s2_path), ("s1", s1, s1_path)):
        payload = torch.load(path, map_location="cpu")
        state, selected_key = extract_state_dict(payload, checkpoint_key)
        try:
            model.load_state_dict(state, strict=True)
        except RuntimeError as error:
            raise RuntimeError(f"Strict {label.upper()} checkpoint load failed:\n{error}") from error
        model.to(device=device, dtype=torch.float32).requires_grad_(False)
        loaded_keys[label] = selected_key
        print(f"Strict {label.upper()} load: OK; key={selected_key}; tensors={len(state)}")
    s1.eval()
    return s2, s1, loaded_keys


def split_parent_name(root, dotted_name: str):
    parent = root
    parts = dotted_name.split(".")
    for part in parts[:-1]:
        parent = parent[int(part)] if part.isdigit() else getattr(parent, part)
    return parent, parts[-1]


def block_from_qkv(module_name: str) -> str:
    suffix = ".attn.qkv"
    if not module_name.endswith(suffix):
        raise ValueError(f"Not a DiffIR packed-QKV module: {module_name}")
    return module_name[: -len(suffix)]


class LoRAConv2d:
    """Factory namespace so importing this script does not require PyTorch."""

    @staticmethod
    def create(base, rank: int, alpha: float):
        import torch
        import torch.nn as nn

        if base.kernel_size != (1, 1) or base.groups != 1:
            raise ValueError("DiffIR QKV LoRA expects an ungrouped 1x1 convolution")

        class Wrapper(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.base = base
                self.rank = rank
                self.alpha = float(alpha)
                self.scale = self.alpha / rank
                self.lora_down = nn.Conv2d(base.in_channels, rank, 1, bias=False)
                self.lora_up = nn.Conv2d(rank, base.out_channels, 1, bias=False)
                self.lora_down.to(device=base.weight.device, dtype=torch.float32)
                self.lora_up.to(device=base.weight.device, dtype=torch.float32)
                nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
                nn.init.zeros_(self.lora_up.weight)
                self.base.requires_grad_(False)

            def forward(self, value):
                result = self.base(value)
                update = self.lora_up(self.lora_down(value.float())) * self.scale
                return result + update.to(dtype=result.dtype)

        return Wrapper()


def inject_qkv_lora(model, rank: int, alpha: float) -> dict[str, Any]:
    import torch.nn as nn

    replacements = []
    for name, module in model.named_modules():
        if (
            isinstance(module, nn.Conv2d)
            and name.endswith(".attn.qkv")
            and module.out_channels == 3 * module.in_channels
            and module.kernel_size == (1, 1)
        ):
            replacements.append((name, module))
    if not replacements:
        raise RuntimeError("No DiffIR *.attn.qkv packed 1x1 convolutions were found")
    injected = {}
    for name, module in replacements:
        parent, child = split_parent_name(model, name)
        wrapper = LoRAConv2d.create(module, rank, alpha)
        setattr(parent, child, wrapper)
        injected[name] = wrapper
    return injected


def parameter_grad_norm(module) -> tuple[float, int]:
    squared = 0.0
    count = 0
    for parameter in (module.lora_down.weight, module.lora_up.weight):
        count += parameter.numel()
        if parameter.grad is not None:
            squared += float(parameter.grad.detach().float().pow(2).sum().cpu())
    return math.sqrt(squared), count


def mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    mean = sum(values) / len(values)
    if len(values) == 1:
        return mean, 0.0
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return mean, math.sqrt(variance)


def profile(args: argparse.Namespace) -> None:
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader

    set_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is unavailable")
    if args.lq_size * args.sr_scale != args.image_size:
        raise SystemExit("Require lq_size * sr_scale == image_size")
    if args.sr_scale != 4:
        raise SystemExit("The released SISR DiffIR-S1/S2 checkpoints used here are x4")
    if args.rank < 1 or args.alpha <= 0:
        raise SystemExit("rank and alpha must be positive")

    s2_path = Path(args.s2_checkpoint).expanduser()
    s1_path = Path(args.s1_checkpoint).expanduser()
    for path in (s2_path, s1_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    dataset = ImageFolderDataset(
        Path(args.data_dir).expanduser(),
        args.image_size,
        args.lq_size,
        args.max_images,
        args.seed,
    )
    needed_images = args.probe_batches * args.batch_size
    if len(dataset) < needed_images:
        raise SystemExit(
            f"Need at least {needed_images} calibration images, found {len(dataset)}"
        )
    calibration_paths = dataset.paths[:needed_images]
    protected_dirs = [Path(path).expanduser() for path in args.protected_eval_dir]
    overlap = audit_overlap(calibration_paths, protected_dirs)

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    s2, s1, loaded_keys = load_models(s2_path, s1_path, args.checkpoint_key, device)
    injected = inject_qkv_lora(s2, args.rank, args.alpha)
    blocks = sorted((block_from_qkv(name) for name in injected), key=natural_key)
    if args.expected_blocks and len(blocks) != args.expected_blocks:
        raise RuntimeError(
            f"Expected {args.expected_blocks} DIRformer blocks, found {len(blocks)}"
        )
    block_index = {block: index for index, block in enumerate(blocks)}
    print(f"Injected packed-QKV LoRA probes: blocks={len(blocks)} modules={len(injected)}")

    s2.train()
    s1.eval()
    block_scores: dict[str, list[float]] = {block: [] for block in blocks}
    block_raw_norms: dict[str, list[float]] = {block: [] for block in blocks}
    pixel_losses: list[float] = []
    prior_losses: list[float] = []
    total_losses: list[float] = []
    step_errors: dict[int, list[float]] = {step: [] for step in range(4)}
    step_grad_norms: dict[int, list[float]] = {step: [] for step in range(4)}

    for batch_index, batch in enumerate(loader, start=1):
        if batch_index > args.probe_batches:
            break
        gt = batch["gt"].to(device, non_blocking=True)
        lq = batch["lq"].to(device, non_blocking=True)
        s2.zero_grad(set_to_none=True)

        with torch.no_grad():
            teacher_ipr, _ = s1.E(lq, gt)
        # This leaf enables diagnostic gradients through the four compact-prior
        # denoising steps without unfreezing the teacher or S2 base parameters.
        teacher_ipr = teacher_ipr.detach().requires_grad_(True)
        sr, pred_ipr_list = s2(lq, teacher_ipr)
        if len(pred_ipr_list) != 4:
            raise RuntimeError(f"Expected four DiffIR prior predictions, got {len(pred_ipr_list)}")
        for prediction in pred_ipr_list:
            prediction.retain_grad()

        pixel_loss = F.l1_loss(sr, gt)
        prior_loss = F.l1_loss(pred_ipr_list[-1], teacher_ipr.detach())
        total_loss = args.pixel_weight * pixel_loss + args.prior_weight * prior_loss
        if not torch.isfinite(total_loss):
            raise RuntimeError(f"Non-finite loss in probe batch {batch_index}")
        total_loss.backward()

        pixel_losses.append(float(pixel_loss.detach().cpu()))
        prior_losses.append(float(prior_loss.detach().cpu()))
        total_losses.append(float(total_loss.detach().cpu()))

        for sequence_index, prediction in enumerate(pred_ipr_list):
            timestep = len(pred_ipr_list) - 1 - sequence_index
            error = F.l1_loss(prediction.detach(), teacher_ipr.detach())
            grad_norm = (
                float(prediction.grad.detach().float().norm().cpu())
                if prediction.grad is not None
                else 0.0
            )
            step_errors[timestep].append(float(error.cpu()))
            step_grad_norms[timestep].append(grad_norm)

        for module_name, module in injected.items():
            block = block_from_qkv(module_name)
            grad_norm, parameter_count = parameter_grad_norm(module)
            block_raw_norms[block].append(grad_norm)
            block_scores[block].append(grad_norm / math.sqrt(parameter_count))

        print(
            f"probe {batch_index:02d}/{args.probe_batches}: "
            f"pixel={pixel_losses[-1]:.6f} prior={prior_losses[-1]:.6f} "
            f"total={total_losses[-1]:.6f}"
        )

    importance_rows = []
    for module_name, module in injected.items():
        block = block_from_qkv(module_name)
        grad_mean, grad_std = mean_std(block_raw_norms[block])
        score_mean, score_std = mean_std(block_scores[block])
        parameter_count = module.lora_down.weight.numel() + module.lora_up.weight.numel()
        importance_rows.append(
            {
                "train_step": 0,
                "profile_scope": "complete_official_s2_objective",
                "block": block,
                "block_index": block_index[block],
                "module": module_name,
                "grad_norm": grad_mean,
                "std_grad_norm": grad_std,
                "lora_param_count": parameter_count,
                "module_count": 1,
                "normalized_grad_score": score_mean,
                "std_normalized_grad_score": score_std,
                "probe_batches": len(total_losses),
                "mean_pixel_loss": sum(pixel_losses) / len(pixel_losses),
                "mean_prior_l1_loss": sum(prior_losses) / len(prior_losses),
                "mean_probe_loss": sum(total_losses) / len(total_losses),
                "loss_mode": "official_diffir_s2_pixel_l1_plus_final_prior_l1",
                "importance_rank": 0,
                "selected_topk": False,
            }
        )
    ranked = sorted(
        importance_rows,
        key=lambda row: (-row["normalized_grad_score"], row["block_index"]),
    )
    total_score = sum(row["normalized_grad_score"] for row in ranked)
    cumulative = 0.0
    for rank, row in enumerate(ranked, start=1):
        share = row["normalized_grad_score"] / total_score if total_score > 0 else 0.0
        cumulative += share
        row["importance_rank"] = rank
        row["score_share"] = share
        row["cumulative_utility"] = cumulative
        row["selected_topk"] = rank <= args.report_topk
    importance_rows = sorted(ranked, key=lambda row: row["block_index"])

    step_rows = []
    for timestep in reversed(range(4)):
        error_mean, error_std = mean_std(step_errors[timestep])
        grad_mean, grad_std = mean_std(step_grad_norms[timestep])
        step_rows.append(
            {
                "timestep": timestep,
                "reverse_sequence_index": 3 - timestep,
                "mean_prior_l1_to_teacher": error_mean,
                "std_prior_l1_to_teacher": error_std,
                "mean_output_gradient_norm": grad_mean,
                "std_output_gradient_norm": grad_std,
                "probe_batches": len(step_errors[timestep]),
            }
        )

    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    importance_path = output_dir / "lora_importance_evolution.csv"
    ranked_path = output_dir / "lora_importance_ranked.csv"
    step_path = output_dir / "diffusion_step_contributions.csv"
    metadata_path = output_dir / "metadata.json"
    write_csv(importance_path, importance_rows)
    write_csv(ranked_path, ranked)
    write_csv(step_path, step_rows)

    metadata = {
        "model": "DiffIR-S2",
        "architecture": model_kwargs(),
        "s2_checkpoint": str(s2_path.resolve()),
        "s1_checkpoint": str(s1_path.resolve()),
        "checkpoint_keys": loaded_keys,
        "data_dir": str(Path(args.data_dir).expanduser().resolve()),
        "protected_eval_overlap_audit": overlap,
        "image_size": args.image_size,
        "lq_size": args.lq_size,
        "sr_scale": args.sr_scale,
        "target": "packed_qkv_conv2d",
        "rank": args.rank,
        "alpha": float(args.alpha),
        "candidate_block_count": len(blocks),
        "injected_module_count": len(injected),
        "diffusion_timesteps": 4,
        "probe_batches": len(total_losses),
        "batch_size": args.batch_size,
        "seed": args.seed,
        "pixel_weight": args.pixel_weight,
        "prior_weight": args.prior_weight,
        "loss_mode": "official_diffir_s2_pixel_l1_plus_final_prior_l1",
        "calibration_images": [str(path) for path in calibration_paths],
        "importance_csv": str(importance_path),
        "ranked_csv": str(ranked_path),
        "diffusion_step_csv": str(step_path),
        "note": (
            "DIRformer blocks execute once after compact-prior diffusion. "
            "Per-timestep diagnostics are therefore separate from block importance."
        ),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("\nTop blocks:")
    for row in ranked[: args.report_topk]:
        print(
            f"  {row['importance_rank']:2d}. {row['block']} "
            f"score={row['normalized_grad_score']:.8g} "
            f"cumulative={row['cumulative_utility']:.2%}"
        )
    print(f"\nWrote DiffIR gradient profile to {output_dir}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Profile DiffIR-S2 DIRformer packed-QKV LoRA importance."
    )
    parser.add_argument(
        "--s2_checkpoint",
        default="experiments/pretrained/SISR-DiffIRS2.pth",
    )
    parser.add_argument(
        "--s1_checkpoint",
        default="experiments/pretrained/SISR-DiffIRS1.pth",
    )
    parser.add_argument(
        "--checkpoint_key",
        choices=("auto", "params_ema", "params", "state_dict", "model", "root"),
        default="auto",
    )
    parser.add_argument("--data_dir", required=True)
    parser.add_argument(
        "--protected_eval_dir",
        action="append",
        default=[],
        help="Protected validation/test directory; repeat this option for multiple sets.",
    )
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--lq_size", type=int, default=64)
    parser.add_argument("--sr_scale", type=int, default=4)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=16.0)
    parser.add_argument("--probe_batches", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_images", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--pixel_weight", type=float, default=1.0)
    parser.add_argument("--prior_weight", type=float, default=1.0)
    parser.add_argument("--expected_blocks", type=int, default=44)
    parser.add_argument("--report_topk", type=int, default=12)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="cuda")
    return parser


def main() -> None:
    profile(build_parser().parse_args())


if __name__ == "__main__":
    main()
