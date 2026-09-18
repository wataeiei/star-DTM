#!/usr/bin/env python3
"""Evaluate a pretrained DiffIR-S2 model on an HR image directory.

Run this file from the DiffIR-SRGAN repository root. HR images are resized to
``image_size`` first, then degraded with BasicSR's MATLAB-compatible bicubic
resizer. PSNR and SSIM are calculated from the exact uint8 PNG pixels written
to the output directory.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write an empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def discover_images(data_dir: Path, eval_manifest: Path | None, max_images: int) -> list[Path]:
    if not data_dir.is_dir():
        raise SystemExit(f"Evaluation directory not found: {data_dir}")

    by_name: dict[str, Path] = {}
    for path in sorted(data_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            if path.name in by_name:
                raise SystemExit(f"Duplicate evaluation filename: {path.name}")
            by_name[path.name] = path

    if eval_manifest is None:
        paths = list(by_name.values())
    else:
        if not eval_manifest.is_file():
            raise SystemExit(f"Evaluation manifest not found: {eval_manifest}")
        with eval_manifest.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        if not rows or "filename" not in rows[0]:
            raise SystemExit("Evaluation manifest must contain a filename column")
        excluded = {
            "1", "true", "yes"
        }
        names = [
            row["filename"]
            for row in rows
            if str(row.get("exclude_from_eval", "false")).strip().lower() not in excluded
        ]
        missing = [name for name in names if name not in by_name]
        if missing:
            raise SystemExit(
                f"Evaluation directory is missing {len(missing)} manifest images; "
                f"examples: {missing[:5]}"
            )
        paths = [by_name[name] for name in names]

    if max_images > 0:
        paths = paths[:max_images]
    if not paths:
        raise SystemExit(f"No evaluation images found in {data_dir}")
    return paths


def pixel_sha256(path: Path) -> str:
    with Image.open(path) as image:
        image = image.convert("RGB")
        digest = hashlib.sha256()
        digest.update(f"RGB:{image.width}x{image.height}:".encode("ascii"))
        digest.update(image.tobytes())
    return digest.hexdigest()


def audit_overlap(eval_paths: list[Path], train_dir: Path | None) -> dict[str, Any]:
    if train_dir is None:
        return {"enabled": False}
    if not train_dir.is_dir():
        raise SystemExit(f"Training directory not found: {train_dir}")

    train_paths = sorted(
        path for path in train_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    train_hashes = {pixel_sha256(path) for path in train_paths}
    overlaps = [str(path) for path in eval_paths if pixel_sha256(path) in train_hashes]
    print(
        f"Dataset overlap audit: train={len(train_paths)} "
        f"eval={len(eval_paths)} overlap={len(overlaps)}"
    )
    if overlaps:
        raise SystemExit(
            "Evaluation set overlaps training data; first matches: "
            + ", ".join(overlaps[:5])
        )
    return {
        "enabled": True,
        "train_dir": str(train_dir.resolve()),
        "num_train_images": len(train_paths),
        "num_eval_images": len(eval_paths),
        "overlap_count": 0,
    }


def load_hr(path: Path, image_size: int):
    import torch

    with Image.open(path) as image:
        image = image.convert("RGB")
        if image.size != (image_size, image_size):
            image = image.resize((image_size, image_size), Image.Resampling.BICUBIC)
        array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


def tensor_to_uint8(image) -> np.ndarray:
    return (
        image.detach().float().clamp(0, 1).mul(255.0).round().byte()
        .permute(1, 2, 0).cpu().numpy()
    )


def calculate_metrics(prediction: np.ndarray, target: np.ndarray, crop_border: int) -> tuple[float, float]:
    from skimage.metrics import peak_signal_noise_ratio, structural_similarity

    if prediction.shape != target.shape:
        raise ValueError(f"Prediction/target shape mismatch: {prediction.shape} vs {target.shape}")
    if crop_border > 0:
        if min(target.shape[:2]) <= 2 * crop_border:
            raise ValueError(f"crop_border={crop_border} is too large for {target.shape[:2]}")
        prediction = prediction[crop_border:-crop_border, crop_border:-crop_border]
        target = target[crop_border:-crop_border, crop_border:-crop_border]
    return (
        float(peak_signal_noise_ratio(target, prediction, data_range=255)),
        float(structural_similarity(target, prediction, channel_axis=2, data_range=255)),
    )


def extract_state_dict(payload: Any, checkpoint_key: str) -> tuple[dict[str, Any], str]:
    if not isinstance(payload, dict):
        raise TypeError(f"Checkpoint must be a dictionary, found {type(payload).__name__}")

    if checkpoint_key != "auto":
        if checkpoint_key == "root":
            state = payload
        else:
            if checkpoint_key not in payload or not isinstance(payload[checkpoint_key], dict):
                raise KeyError(f"Checkpoint has no dictionary key {checkpoint_key!r}")
            state = payload[checkpoint_key]
        selected_key = checkpoint_key
    else:
        selected_key = "root"
        state = payload
        for key in ("params_ema", "params", "state_dict", "model"):
            if key in payload and isinstance(payload[key], dict):
                selected_key = key
                state = payload[key]
                break

    prefixes = ("module.", "net_g.")
    for prefix in prefixes:
        if state and all(str(key).startswith(prefix) for key in state):
            state = {str(key)[len(prefix):]: value for key, value in state.items()}
            selected_key += f" (stripped {prefix})"
    return state, selected_key


def build_model(checkpoint: Path, checkpoint_key: str, device):
    import torch
    from DiffIR.archs.S2_arch import DiffIRS2

    model = DiffIRS2(
        n_encoder_res=9,
        inp_channels=3,
        out_channels=3,
        scale=4,
        dim=64,
        num_blocks=[13, 1, 1, 1],
        num_refinement_blocks=13,
        heads=[1, 2, 4, 8],
        ffn_expansion_factor=2.2,
        bias=False,
        LayerNorm_type="BiasFree",
        n_denoise_res=1,
        linear_start=0.1,
        linear_end=0.99,
        timesteps=4,
    )
    payload = torch.load(checkpoint, map_location="cpu")
    state, selected_key = extract_state_dict(payload, checkpoint_key)
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as error:
        raise SystemExit(f"Strict checkpoint loading failed:\n{error}") from error
    model.eval().requires_grad_(False).to(device=device, dtype=torch.float32)
    print(
        f"Strict checkpoint load: OK; key={selected_key}; "
        f"tensors={len(state)}"
    )
    return model, selected_key


def atomic_save_png(array: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    Image.fromarray(array, mode="RGB").save(temporary, format="PNG")
    temporary.replace(path)


def mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def evaluate(args: argparse.Namespace) -> None:
    import torch
    from basicsr.utils.matlab_functions import imresize

    if args.sr_scale != 4:
        raise SystemExit("The released SISR-DiffIRS2 checkpoint evaluated here is x4 only")
    if args.lq_size * args.sr_scale != args.image_size:
        raise SystemExit("Require lq_size * sr_scale == image_size")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is unavailable")

    random.seed(args.eval_seed)
    np.random.seed(args.eval_seed)
    torch.manual_seed(args.eval_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.eval_seed)

    data_dir = Path(args.data_dir).expanduser()
    manifest = Path(args.eval_manifest).expanduser() if args.eval_manifest else None
    train_dir = (
        Path(args.train_dir_for_overlap_check).expanduser()
        if args.train_dir_for_overlap_check else None
    )
    output_dir = Path(args.output_dir).expanduser()
    checkpoint = Path(args.checkpoint).expanduser()
    if not checkpoint.is_file():
        raise SystemExit(f"Checkpoint not found: {checkpoint}")

    eval_paths = discover_images(data_dir, manifest, args.max_images)
    overlap = audit_overlap(eval_paths, train_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    bicubic_dir = output_dir / "images" / "Bicubic"
    base_dir = output_dir / "images" / args.method
    if args.save_images:
        bicubic_dir.mkdir(parents=True, exist_ok=True)
        base_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    model, selected_key = build_model(checkpoint, args.checkpoint_key, device)
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    rows: list[dict[str, Any]] = []
    timed_seconds: list[float] = []
    for index, image_path in enumerate(eval_paths):
        hr = load_hr(image_path, args.image_size)
        lr = imresize(hr, scale=1.0 / args.sr_scale).clamp(0, 1)
        if tuple(lr.shape[-2:]) != (args.lq_size, args.lq_size):
            raise RuntimeError(
                f"Unexpected LR shape for {image_path.name}: {tuple(lr.shape)}"
            )
        bicubic = imresize(lr, scale=float(args.sr_scale)).clamp(0, 1)
        hr_uint8 = tensor_to_uint8(hr)
        bicubic_uint8 = tensor_to_uint8(bicubic)

        lr_batch = lr.unsqueeze(0).to(device=device, dtype=torch.float32)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        with torch.inference_mode():
            prediction = model(lr_batch)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - start
        prediction_uint8 = tensor_to_uint8(prediction[0])

        bicubic_psnr, bicubic_ssim = calculate_metrics(
            bicubic_uint8, hr_uint8, args.crop_border
        )
        base_psnr, base_ssim = calculate_metrics(
            prediction_uint8, hr_uint8, args.crop_border
        )
        timed = index >= args.warmup_images
        if timed:
            timed_seconds.append(elapsed)

        rows.extend([
            {
                "method": "Bicubic",
                "image": image_path.name,
                "psnr": bicubic_psnr,
                "ssim": bicubic_ssim,
                "inference_time_s": 0.0,
            },
            {
                "method": args.method,
                "image": image_path.name,
                "psnr": base_psnr,
                "ssim": base_ssim,
                "inference_time_s": elapsed if timed else "",
            },
        ])
        if args.save_images:
            atomic_save_png(bicubic_uint8, bicubic_dir / image_path.name)
            atomic_save_png(prediction_uint8, base_dir / image_path.name)

        if (index + 1) == 1 or (index + 1) % args.log_every == 0 or (index + 1) == len(eval_paths):
            print(
                f"{args.method:24s} [{index + 1:03d}/{len(eval_paths):03d}] "
                f"PSNR={base_psnr:.3f} SSIM={base_ssim:.4f} time={elapsed:.3f}s"
            )

    peak_mb = (
        torch.cuda.max_memory_allocated(device) / 2**20
        if device.type == "cuda" else 0.0
    )
    by_method = {
        label: [row for row in rows if row["method"] == label]
        for label in ("Bicubic", args.method)
    }
    summary: list[dict[str, Any]] = []
    for label, method_rows in by_method.items():
        inference_time = mean(timed_seconds) if label == args.method else 0.0
        summary.append({
            "method": label,
            "num_images": len(method_rows),
            "mean_psnr": mean([float(row["psnr"]) for row in method_rows]),
            "mean_ssim": mean([float(row["ssim"]) for row in method_rows]),
            "mean_inference_time_s": inference_time,
            "images_per_hour": 3600.0 / inference_time if inference_time > 0 else "",
            "peak_cuda_mem_mb": peak_mb if label == args.method else 0.0,
            "adapter_size_mb": 0.0,
        })

    write_csv(output_dir / "sr_metrics_per_image.csv", rows)
    write_csv(output_dir / "sr_metrics_summary.csv", summary)
    metadata = {
        "model": "DiffIR-S2",
        "method": args.method,
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_key": selected_key,
        "strict_checkpoint_load": True,
        "data_dir": str(data_dir.resolve()),
        "eval_manifest": str(manifest.resolve()) if manifest else None,
        "num_images": len(eval_paths),
        "image_names": [path.name for path in eval_paths],
        "image_size": args.image_size,
        "lq_size": args.lq_size,
        "sr_scale": args.sr_scale,
        "crop_border": args.crop_border,
        "precision": "fp32",
        "lr_degradation": "BasicSR MATLAB-compatible bicubic",
        "metrics": "skimage RGB uint8 PSNR/SSIM on saved pixels",
        "warmup_images": args.warmup_images,
        "num_timed_images": len(timed_seconds),
        "eval_seed": args.eval_seed,
        "save_images": args.save_images,
        "overlap_audit": overlap,
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote results to {output_dir}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate the released x4 DiffIR-S2 checkpoint on HR images."
    )
    parser.add_argument(
        "--checkpoint",
        default="experiments/pretrained/SISR-DiffIRS2.pth",
    )
    parser.add_argument(
        "--checkpoint_key",
        choices=("auto", "params_ema", "params", "state_dict", "model", "root"),
        default="auto",
    )
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--eval_manifest", default="")
    parser.add_argument("--train_dir_for_overlap_check", default="")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--method", default="Base-DiffIR-S2")
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--lq_size", type=int, default=64)
    parser.add_argument("--sr_scale", type=int, default=4)
    parser.add_argument("--crop_border", type=int, default=4)
    parser.add_argument("--max_images", type=int, default=0)
    parser.add_argument("--warmup_images", type=int, default=1)
    parser.add_argument("--eval_seed", type=int, default=4242)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--log_every", type=int, default=25)
    parser.add_argument("--save_images", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.image_size <= 0 or args.lq_size <= 0 or args.sr_scale <= 0:
        raise SystemExit("Image sizes and scale must be positive")
    if args.crop_border < 0 or args.max_images < 0 or args.warmup_images < 0:
        raise SystemExit("crop_border, max_images, and warmup_images cannot be negative")
    if args.log_every <= 0:
        raise SystemExit("log_every must be positive")
    evaluate(args)


if __name__ == "__main__":
    main()
