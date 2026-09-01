#!/usr/bin/env python3
"""Evaluate official DiT-SR sampling with base or custom LoRA adapters.

Copy this script, ``profile_dit_sr_grad.py``, and
``adaptive_grad_blockskip.py`` into the DiT-SR repository root before use.
The script deliberately calls the repository's official ``Sampler`` so the
reported reconstruction quality follows the same diffusion/autoencoder path
as the released inference code.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import time
import types
from contextlib import nullcontext
from pathlib import Path

from PIL import Image

import torch
import torch.nn.functional as F

import adaptive_grad_blockskip as adaptive
import profile_dit_sr_grad as core


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def parse_adapter(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--adapter must use LABEL=PATH")
    label, path = value.split("=", 1)
    if not label.strip():
        raise argparse.ArgumentTypeError("Adapter label cannot be empty.")
    return label.strip(), Path(path).expanduser()


def safe_name(value: str) -> str:
    return "".join(char if char.isalnum() or char in "-_." else "_" for char in value)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def image_paths(directory: str | Path, excluded: set[str], max_images: int) -> list[Path]:
    directory = Path(directory)
    paths = sorted(
        path for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS and path.name not in excluded
    )
    if max_images > 0:
        paths = paths[:max_images]
    if not paths:
        raise SystemExit(f"No evaluation images found in {directory}")
    return paths


def image_sha256(path: Path) -> str:
    image = Image.open(path).convert("RGB")
    digest = hashlib.sha256()
    digest.update(f"{image.width}x{image.height}:RGB:".encode("ascii"))
    digest.update(image.tobytes())
    return digest.hexdigest()


def audit_overlap(eval_paths: list[Path], train_dir: str) -> dict:
    if not train_dir:
        return {"enabled": False}
    train_paths = image_paths(train_dir, set(), 0)
    train_hashes = {image_sha256(path) for path in train_paths}
    overlaps = [str(path) for path in eval_paths if image_sha256(path) in train_hashes]
    print(
        f"Dataset overlap audit: train={len(train_paths)} eval={len(eval_paths)} "
        f"overlap={len(overlaps)}"
    )
    if overlaps:
        raise SystemExit(
            "Evaluation set overlaps the training set. First matches: "
            + ", ".join(overlaps[:5])
        )
    return {
        "enabled": True,
        "train_dir": train_dir,
        "num_train_images": len(train_paths),
        "num_eval_images": len(eval_paths),
        "overlap_count": 0,
    }


def pil_to_tensor(image: Image.Image, size: int) -> torch.Tensor:
    image = image.convert("RGB").resize((size, size), Image.Resampling.BICUBIC)
    data = torch.frombuffer(bytearray(image.tobytes()), dtype=torch.uint8)
    return data.view(size, size, 3).permute(2, 0, 1).float() / 255.0


def tensor_to_pil(image: torch.Tensor) -> Image.Image:
    array = (
        image.detach().clamp(0, 1).mul(255).round().byte()
        .permute(1, 2, 0).cpu().numpy()
    )
    return Image.fromarray(array, mode="RGB")


def make_lr(hr: torch.Tensor, scale: int) -> tuple[torch.Tensor, torch.Tensor]:
    lr = F.interpolate(
        hr.unsqueeze(0), scale_factor=1.0 / scale, mode="bicubic", align_corners=False
    ).clamp(0, 1)
    bicubic = F.interpolate(
        lr, size=hr.shape[-2:], mode="bicubic", align_corners=False
    ).clamp(0, 1)
    return lr.squeeze(0), bicubic.squeeze(0)


def crop(image: torch.Tensor, border: int) -> torch.Tensor:
    return image if border <= 0 else image[..., border:-border, border:-border]


def psnr(prediction: torch.Tensor, target: torch.Tensor) -> float:
    mse = F.mse_loss(prediction.float(), target.float()).item()
    return 100.0 if mse <= 1e-10 else -10.0 * math.log10(mse)


def ssim(prediction: torch.Tensor, target: torch.Tensor) -> float:
    channels = prediction.shape[0]
    coords = torch.arange(11, dtype=torch.float32) - 5
    kernel_1d = torch.exp(-(coords.square()) / (2 * 1.5**2))
    kernel_1d /= kernel_1d.sum()
    kernel = (kernel_1d[:, None] @ kernel_1d[None, :]).view(1, 1, 11, 11)
    kernel = kernel.repeat(channels, 1, 1, 1)
    x = F.pad(prediction.float().unsqueeze(0), (5, 5, 5, 5), mode="reflect")
    y = F.pad(target.float().unsqueeze(0), (5, 5, 5, 5), mode="reflect")
    mu_x = F.conv2d(x, kernel, groups=channels)
    mu_y = F.conv2d(y, kernel, groups=channels)
    mu_x2, mu_y2, mu_xy = mu_x.square(), mu_y.square(), mu_x * mu_y
    sigma_x = F.conv2d(x * x, kernel, groups=channels) - mu_x2
    sigma_y = F.conv2d(y * y, kernel, groups=channels) - mu_y2
    sigma_xy = F.conv2d(x * y, kernel, groups=channels) - mu_xy
    c1, c2 = 0.01**2, 0.03**2
    score = ((2 * mu_xy + c1) * (2 * sigma_xy + c2)) / (
        (mu_x2 + mu_y2 + c1) * (sigma_x + sigma_y + c2)
    )
    return float(score.mean())


def adapter_blocks(paths: list[Path], block_regex: str) -> set[str]:
    blocks = set()
    for path in paths:
        payload = torch.load(path, map_location="cpu")
        if payload.get("format") != "custom_lora_linear_v1":
            raise SystemExit(f"Unsupported adapter format: {path}")
        for name in payload["modules"]:
            block = core.block_key(name, block_regex)
            if block:
                blocks.add(block)
    return blocks


def reset_lora(model: torch.nn.Module) -> None:
    for _name, module in core.iter_lora_modules(model):
        module.lora_down.weight.data.zero_()
        module.lora_up.weight.data.zero_()


def install_unused_sampler_dataset_stub() -> None:
    """Skip BasicSR dataset imports unused by direct ``sample_func`` calls."""
    if "datapipe.datasets" in sys.modules:
        return
    import datapipe

    datasets = types.ModuleType("datapipe.datasets")

    def create_dataset(*_args, **_kwargs):
        raise RuntimeError(
            "Folder-based Sampler.inference is unavailable in this evaluator. "
            "Use eval_dit_sr_sr_metrics.py --data_dir instead."
        )

    datasets.create_dataset = create_dataset
    datapipe.datasets = datasets
    sys.modules["datapipe.datasets"] = datasets


def build_sampler(args):
    from omegaconf import OmegaConf

    # Install the same lightweight import guards used by the training scripts.
    core.install_timm_layers_stub()
    core.install_torchvision_stub_for_timm()
    install_unused_sampler_dataset_stub()
    from sampler import Sampler

    configs = OmegaConf.load(args.config_path)
    configs.model.ckpt_path = args.ckpt_path
    configs.diffusion.params.sf = args.sr_scale
    configs.autoencoder.ckpt_path = args.autoencoder_ckpt
    return Sampler(
        configs,
        sf=args.sr_scale,
        chop_size=args.lq_size,
        chop_stride=max(1, args.lq_size - 16),
        chop_bs=1,
        use_amp=not args.fp32,
        seed=args.eval_seed,
        padding_offset=max(int(configs.model.params.get("lq_size", 64)), 64),
    )


@torch.no_grad()
def sample(sampler, lr: torch.Tensor, seed: int, fp32: bool) -> torch.Tensor:
    sampler.setup_seed(seed)
    device = next(sampler.model.parameters()).device
    lq = (lr.unsqueeze(0).to(device) * 2.0 - 1.0)
    context = nullcontext if fp32 or device.type != "cuda" else torch.cuda.amp.autocast
    with context():
        output = sampler.sample_func(lq, noise_repeat=False)
    return (output[0].float().cpu() * 0.5 + 0.5).clamp(0, 1)


def summarize(rows: list[dict], adapter_sizes: dict[str, float]) -> list[dict]:
    methods = list(dict.fromkeys(row["method"] for row in rows))
    summaries = []
    for method in methods:
        selected = [row for row in rows if row["method"] == method]
        elapsed = sum(float(row["inference_time_s"]) for row in selected)
        summaries.append({
            "method": method,
            "num_images": len(selected),
            "mean_psnr": sum(float(row["psnr"]) for row in selected) / len(selected),
            "mean_ssim": sum(float(row["ssim"]) for row in selected) / len(selected),
            "mean_inference_time_s": elapsed / len(selected),
            "images_per_hour": 3600.0 * len(selected) / elapsed if elapsed else "",
            "peak_cuda_mem_mb": max(float(row["peak_cuda_mem_mb"]) for row in selected),
            "adapter_size_mb": adapter_sizes.get(method, 0.0),
        })
    base = next(row for row in summaries if row["method"] == "Base-DiT-SR")
    for row in summaries:
        row["delta_psnr_vs_base"] = row["mean_psnr"] - base["mean_psnr"]
        row["delta_ssim_vs_base"] = row["mean_ssim"] - base["mean_ssim"]
    return summaries


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config_path", required=True)
    parser.add_argument("--ckpt_path", required=True)
    parser.add_argument("--autoencoder_ckpt", required=True)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--train_dir_for_overlap_check", default="")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--adapter", action="append", type=parse_adapter, default=[])
    parser.add_argument("--exclude_image", action="append", default=[])
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--lq_size", type=int, default=64)
    parser.add_argument("--sr_scale", type=int, default=4)
    parser.add_argument("--max_images", type=int, default=0)
    parser.add_argument("--eval_seed", type=int, default=4242)
    parser.add_argument("--warmup_images", type=int, default=1)
    parser.add_argument("--crop_border", type=int, default=4)
    parser.add_argument("--target", choices=["q", "v", "qv", "qkv", "all_linear"], default="qv")
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--alpha", type=int, default=16)
    parser.add_argument("--block_regex", default="")
    parser.add_argument("--save_images", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fp32", action="store_true")
    args = parser.parse_args()

    if args.image_size != args.lq_size * args.sr_scale:
        raise SystemExit("Require --image_size == --lq_size * --sr_scale.")
    missing = [str(path) for _label, path in args.adapter if not path.is_file()]
    if missing:
        raise SystemExit("Missing adapter checkpoint(s): " + ", ".join(missing))

    paths = image_paths(args.data_dir, set(args.exclude_image), args.max_images)
    overlap_report = audit_overlap(paths, args.train_dir_for_overlap_check)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sampler = build_sampler(args)
    union_blocks = adapter_blocks([path for _label, path in args.adapter], args.block_regex)
    injected = []
    if union_blocks:
        injected = core.inject_lora(
            sampler.model, args.target, args.rank, args.alpha, args.block_regex,
            selected_blocks=union_blocks,
        )
    sampler.model.eval()
    print(f"Injected union LoRA targets: blocks={len(union_blocks)} modules={len(injected)}")

    image_rows = []
    adapter_sizes = {"Bicubic": 0.0, "Base-DiT-SR": 0.0}
    load_reports = {}

    # Bicubic needs no diffusion inference and is written once.
    bicubic_dir = output_dir / "images" / "Bicubic"
    if args.save_images:
        bicubic_dir.mkdir(parents=True, exist_ok=True)
    for path in paths:
        hr = pil_to_tensor(Image.open(path), args.image_size)
        _lr, bicubic = make_lr(hr, args.sr_scale)
        target, prediction = crop(hr, args.crop_border), crop(bicubic, args.crop_border)
        image_rows.append({
            "method": "Bicubic", "image": path.name,
            "psnr": psnr(prediction, target), "ssim": ssim(prediction, target),
            "inference_time_s": 0.0, "peak_cuda_mem_mb": 0.0,
        })
        if args.save_images:
            tensor_to_pil(bicubic).save(bicubic_dir / path.name)

    methods = [("Base-DiT-SR", None), *args.adapter]
    for method, adapter_path in methods:
        reset_lora(sampler.model)
        if adapter_path is not None:
            report = adaptive.load_lora_adapter(sampler.model, adapter_path)
            if report["missing"]:
                raise RuntimeError(f"{method}: missing adapter targets: {report['missing'][:5]}")
            load_reports[method] = report
            adapter_sizes[method] = adapter_path.stat().st_size / (1024.0**2)
            print(f"{method}: loaded={report['loaded']} inactive_union_modules={len(report['unexpected'])}")

        if args.warmup_images > 0:
            warm_hr = pil_to_tensor(Image.open(paths[0]), args.image_size)
            warm_lr, _warm_bicubic = make_lr(warm_hr, args.sr_scale)
            for warm_index in range(args.warmup_images):
                sample(sampler, warm_lr, args.eval_seed + 100000 + warm_index, args.fp32)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        method_dir = output_dir / "images" / safe_name(method)
        if args.save_images:
            method_dir.mkdir(parents=True, exist_ok=True)

        for index, path in enumerate(paths):
            hr = pil_to_tensor(Image.open(path), args.image_size)
            lr, _bicubic = make_lr(hr, args.sr_scale)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            started = time.perf_counter()
            prediction = sample(sampler, lr, args.eval_seed + index, args.fp32)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            peak_mb = (
                torch.cuda.max_memory_allocated() / (1024.0**2)
                if torch.cuda.is_available() else 0.0
            )
            target_eval = crop(hr, args.crop_border)
            prediction_eval = crop(prediction, args.crop_border)
            row = {
                "method": method, "image": path.name,
                "psnr": psnr(prediction_eval, target_eval),
                "ssim": ssim(prediction_eval, target_eval),
                "inference_time_s": elapsed, "peak_cuda_mem_mb": peak_mb,
            }
            image_rows.append(row)
            if args.save_images:
                tensor_to_pil(prediction).save(method_dir / path.name)
            print(
                f"{method:<28} [{index + 1:03d}/{len(paths)}] "
                f"PSNR={row['psnr']:.3f} SSIM={row['ssim']:.4f} time={elapsed:.2f}s"
            )

    summaries = summarize(image_rows, adapter_sizes)
    write_csv(output_dir / "sr_metrics_per_image.csv", image_rows)
    write_csv(output_dir / "sr_metrics_summary.csv", summaries)
    metadata = {
        "config_path": args.config_path,
        "ckpt_path": args.ckpt_path,
        "autoencoder_ckpt": args.autoencoder_ckpt,
        "data_dir": args.data_dir,
        "num_images": len(paths),
        "excluded_images": args.exclude_image,
        "eval_seed": args.eval_seed,
        "official_sampler": True,
        "union_lora_blocks": sorted(union_blocks, key=core.natural_key),
        "injected_module_count": len(injected),
        "overlap_audit": overlap_report,
        "load_reports": load_reports,
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, default=str), encoding="utf-8"
    )
    print(f"Wrote results to {output_dir}")


if __name__ == "__main__":
    main()
