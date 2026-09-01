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


def read_csv_rows(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def append_csv_row(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.is_file() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


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


def load_complete_image(path: Path, size: int) -> torch.Tensor | None:
    """Load a complete saved prediction, returning None for partial files."""
    try:
        with Image.open(path) as image:
            image.load()
            if image.size != (size, size):
                raise ValueError(
                    f"expected {(size, size)}, found {image.size}"
                )
            return pil_to_tensor(image, size)
    except (OSError, ValueError) as error:
        print(f"Invalid saved image; rerunning {path}: {error}")
        return None


def atomic_save_png(image: torch.Tensor, path: Path) -> None:
    """Write a PNG completely before replacing its final destination."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    tensor_to_pil(image).save(temporary, format="PNG")
    temporary.replace(path)


def make_lr(hr: torch.Tensor, scale: int) -> tuple[torch.Tensor, torch.Tensor]:
    lr = F.interpolate(
        hr.unsqueeze(0),
        scale_factor=1.0 / scale,
        mode="bicubic",
        align_corners=False,
        antialias=True,
    ).clamp(0, 1)
    bicubic = F.interpolate(
        lr,
        size=hr.shape[-2:],
        mode="bicubic",
        align_corners=False,
        antialias=True,
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
        # The official sampler uses ``mask=False`` as its function default but
        # tests ``mask is not None`` internally. Pass None explicitly for SR so
        # the model follows the no-mask branch used by Sampler.inference().
        output = sampler.sample_func(lq, noise_repeat=False, mask=None)
    return (output[0].float().cpu() * 0.5 + 0.5).clamp(0, 1)


def summarize(rows: list[dict], adapter_sizes: dict[str, float]) -> list[dict]:
    methods = list(dict.fromkeys(row["method"] for row in rows))
    summaries = []
    for method in methods:
        selected = [row for row in rows if row["method"] == method]
        times = [
            float(row["inference_time_s"])
            for row in selected
            if row.get("inference_time_s", "") not in ("", None)
        ]
        peaks = [
            float(row["peak_cuda_mem_mb"])
            for row in selected
            if row.get("peak_cuda_mem_mb", "") not in ("", None)
        ]
        elapsed = sum(times)
        summaries.append({
            "method": method,
            "num_images": len(selected),
            "mean_psnr": sum(float(row["psnr"]) for row in selected) / len(selected),
            "mean_ssim": sum(float(row["ssim"]) for row in selected) / len(selected),
            "mean_inference_time_s": elapsed / len(times) if times else "",
            "num_timed_images": len(times),
            "images_per_hour": 3600.0 * len(times) / elapsed if elapsed else "",
            "peak_cuda_mem_mb": max(peaks) if peaks else "",
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
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse saved method images and append recoverable per-image metrics.",
    )
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
    if args.resume and not args.save_images:
        raise SystemExit("--resume requires --save_images so completed outputs can be reused.")

    metrics_path = output_dir / "sr_metrics_per_image.csv"
    recovery_path = output_dir / "sr_metrics_recovery.csv"
    previous_rows = read_csv_rows(metrics_path) if args.resume else []
    if args.resume and recovery_path.is_file():
        previous_rows.extend(read_csv_rows(recovery_path))
    image_rows_by_key: dict[tuple[str, str], dict] = {}
    for row in previous_rows:
        image_rows_by_key[(row["method"], row["image"])] = row
    recovered_saved_images = 0

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

    adapter_sizes = {"Bicubic": 0.0, "Base-DiT-SR": 0.0}
    load_reports = {}

    # Bicubic needs no diffusion inference and is written once.
    bicubic_dir = output_dir / "images" / "Bicubic"
    if args.save_images:
        bicubic_dir.mkdir(parents=True, exist_ok=True)
    for path in paths:
        hr = pil_to_tensor(Image.open(path), args.image_size)
        _lr, bicubic = make_lr(hr, args.sr_scale)
        output_path = bicubic_dir / path.name
        saved_bicubic = (
            load_complete_image(output_path, args.image_size)
            if args.resume and output_path.is_file()
            else None
        )
        if saved_bicubic is not None:
            bicubic = saved_bicubic
        target, prediction = crop(hr, args.crop_border), crop(bicubic, args.crop_border)
        row = {
            "method": "Bicubic", "image": path.name,
            "psnr": psnr(prediction, target), "ssim": ssim(prediction, target),
            "inference_time_s": 0.0, "peak_cuda_mem_mb": 0.0,
        }
        image_rows_by_key[("Bicubic", path.name)] = row
        if args.save_images and saved_bicubic is None:
            atomic_save_png(bicubic, output_path)

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

        method_dir = output_dir / "images" / safe_name(method)
        if args.save_images:
            method_dir.mkdir(parents=True, exist_ok=True)
        pending_inference = any(
            not (args.resume and load_complete_image(
                method_dir / path.name, args.image_size
            ) is not None)
            for path in paths
            if (method_dir / path.name).is_file()
        ) or any(not (method_dir / path.name).is_file() for path in paths)

        if args.warmup_images > 0 and pending_inference:
            warm_hr = pil_to_tensor(Image.open(paths[0]), args.image_size)
            warm_lr, _warm_bicubic = make_lr(warm_hr, args.sr_scale)
            for warm_index in range(args.warmup_images):
                sample(sampler, warm_lr, args.eval_seed + 100000 + warm_index, args.fp32)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        for index, path in enumerate(paths):
            hr = pil_to_tensor(Image.open(path), args.image_size)
            lr, _bicubic = make_lr(hr, args.sr_scale)
            output_path = method_dir / path.name
            key = (method, path.name)
            saved_prediction = (
                load_complete_image(output_path, args.image_size)
                if args.resume and output_path.is_file()
                else None
            )
            if saved_prediction is not None:
                if key not in image_rows_by_key:
                    prediction = saved_prediction
                    target_eval = crop(hr, args.crop_border)
                    prediction_eval = crop(prediction, args.crop_border)
                    row = {
                        "method": method,
                        "image": path.name,
                        "psnr": psnr(prediction_eval, target_eval),
                        "ssim": ssim(prediction_eval, target_eval),
                        "inference_time_s": "",
                        "peak_cuda_mem_mb": "",
                    }
                    image_rows_by_key[key] = row
                    append_csv_row(recovery_path, row)
                    recovered_saved_images += 1
                print(
                    f"{method:<28} [{index + 1:03d}/{len(paths)}] "
                    "resume=existing"
                )
                continue
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
            image_rows_by_key[key] = row
            append_csv_row(recovery_path, row)
            if args.save_images:
                atomic_save_png(prediction, output_path)
            print(
                f"{method:<28} [{index + 1:03d}/{len(paths)}] "
                f"PSNR={row['psnr']:.3f} SSIM={row['ssim']:.4f} time={elapsed:.2f}s"
            )

    method_order = ["Bicubic", "Base-DiT-SR", *(label for label, _ in args.adapter)]
    image_rows = sorted(
        image_rows_by_key.values(),
        key=lambda row: (method_order.index(row["method"]), row["image"]),
    )
    summaries = summarize(image_rows, adapter_sizes)
    write_csv(metrics_path, image_rows)
    write_csv(output_dir / "sr_metrics_summary.csv", summaries)
    metadata = {
        "config_path": args.config_path,
        "ckpt_path": args.ckpt_path,
        "autoencoder_ckpt": args.autoencoder_ckpt,
        "data_dir": args.data_dir,
        "num_images": len(paths),
        "excluded_images": args.exclude_image,
        "eval_seed": args.eval_seed,
        "lr_degradation": "torch_bicubic_antialias",
        "sr_scale": args.sr_scale,
        "official_sampler": True,
        "union_lora_blocks": sorted(union_blocks, key=core.natural_key),
        "injected_module_count": len(injected),
        "overlap_audit": overlap_report,
        "load_reports": load_reports,
        "resume": args.resume,
        "recovered_saved_images_without_timing": recovered_saved_images,
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, default=str), encoding="utf-8"
    )
    if recovered_saved_images:
        print(
            f"Recovered {recovered_saved_images} saved predictions without timing; "
            "see num_timed_images in the summary."
        )
    print(f"Wrote results to {output_dir}")


if __name__ == "__main__":
    main()
