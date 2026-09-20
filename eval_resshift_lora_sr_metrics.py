#!/usr/bin/env python3
"""Evaluate ResShift base and packed-qkv LoRA adapters with official sampling."""

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

from inspect_resshift_structure import install_timm_layers_stub
from profile_resshift_grad import inject_packed_qkv_lora


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def parse_adapter(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--adapter must use LABEL=PATH")
    label, path = value.split("=", 1)
    if not label.strip():
        raise argparse.ArgumentTypeError("Adapter label cannot be empty")
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


def image_paths(
    directory: str | Path,
    max_images: int,
    excluded: set[str] | None = None,
) -> list[Path]:
    excluded = excluded or set()
    paths = sorted(
        path for path in Path(directory).rglob("*")
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTS
        and path.name not in excluded
    )
    if max_images > 0:
        paths = paths[:max_images]
    if not paths:
        raise SystemExit(f"No evaluation images found in {directory}")
    return paths


def image_sha256(path: Path) -> str:
    with Image.open(path) as image:
        image = image.convert("RGB")
        digest = hashlib.sha256()
        digest.update(f"{image.width}x{image.height}:RGB:".encode("ascii"))
        digest.update(image.tobytes())
    return digest.hexdigest()


def audit_overlap(eval_paths: list[Path], train_dir: str) -> dict:
    if not train_dir:
        return {"enabled": False}
    train_paths = image_paths(train_dir, 0)
    train_hashes = {image_sha256(path) for path in train_paths}
    overlaps = [str(path) for path in eval_paths if image_sha256(path) in train_hashes]
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
        "train_dir": str(Path(train_dir).resolve()),
        "num_train_images": len(train_paths),
        "num_eval_images": len(eval_paths),
        "overlap_count": 0,
    }


def pil_to_tensor(image: Image.Image, size: int) -> torch.Tensor:
    image = image.convert("RGB").resize((size, size), Image.Resampling.BICUBIC)
    data = torch.frombuffer(bytearray(image.tobytes()), dtype=torch.uint8)
    return data.view(size, size, 3).permute(2, 0, 1).float().div(255.0)


def tensor_to_pil(image: torch.Tensor) -> Image.Image:
    array = (
        image.detach().clamp(0, 1).mul(255).round().byte()
        .permute(1, 2, 0).cpu().numpy()
    )
    return Image.fromarray(array, mode="RGB")


def load_complete_image(path: Path, size: int) -> torch.Tensor | None:
    try:
        with Image.open(path) as image:
            image.load()
            if image.size != (size, size):
                raise ValueError(f"expected {(size, size)}, found {image.size}")
            return pil_to_tensor(image, size)
    except (OSError, ValueError):
        return None


def atomic_save_png(image: torch.Tensor, path: Path) -> None:
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


def install_sampler_dataset_stub() -> None:
    datasets = types.ModuleType("datapipe.datasets")

    def create_dataset(*_args, **_kwargs):
        raise RuntimeError("Folder inference is disabled in this evaluator")

    datasets.create_dataset = create_dataset
    try:
        import datapipe
    except ImportError:
        datapipe = types.ModuleType("datapipe")
        sys.modules["datapipe"] = datapipe
    datapipe.datasets = datasets
    sys.modules["datapipe.datasets"] = datasets


def build_sampler(args):
    from omegaconf import OmegaConf

    install_timm_layers_stub(torch)
    install_sampler_dataset_stub()
    from sampler import ResShiftSampler

    config = OmegaConf.load(args.config_path)
    config.model.ckpt_path = args.checkpoint
    config.autoencoder.ckpt_path = args.autoencoder_checkpoint
    config.diffusion.params.sf = args.sr_scale
    return ResShiftSampler(
        config,
        sf=args.sr_scale,
        use_amp=not args.fp32,
        chop_size=args.lq_size,
        chop_stride=max(1, args.lq_size - 16),
        chop_bs=1,
        padding_offset=max(int(config.model.params.get("lq_size", 64)), 64),
        seed=args.eval_seed,
    )


def inspect_adapters(adapters: list[tuple[str, Path]]) -> tuple[set[str], dict]:
    blocks = set()
    payloads = {}
    for label, path in adapters:
        payload = torch.load(path, map_location="cpu")
        if payload.get("format") != "resshift_packed_qkv_lora_v1":
            raise SystemExit(f"Unsupported ResShift adapter format: {path}")
        payloads[label] = payload
        blocks.update(payload["selected_blocks"])
    return blocks, payloads


def reset_lora(injected) -> None:
    for module in injected.values():
        module.lora_down.weight.data.zero_()
        module.lora_up.weight.data.zero_()


def load_adapter(injected, payload: dict) -> dict:
    loaded = []
    missing = []
    for name, weights in payload["modules"].items():
        module = injected.get(name)
        if module is None:
            missing.append(name)
            continue
        if module.lora_down.weight.shape != weights["lora_down"].shape:
            raise ValueError(f"LoRA-down shape mismatch for {name}")
        if module.lora_up.weight.shape != weights["lora_up"].shape:
            raise ValueError(f"LoRA-up shape mismatch for {name}")
        module.lora_down.weight.data.copy_(
            weights["lora_down"].to(module.lora_down.weight)
        )
        module.lora_up.weight.data.copy_(
            weights["lora_up"].to(module.lora_up.weight)
        )
        loaded.append(name)
    return {"loaded": loaded, "missing": missing}


@torch.no_grad()
def sample(sampler, lr: torch.Tensor, seed: int, fp32: bool) -> torch.Tensor:
    sampler.setup_seed(seed)
    device = next(sampler.model.parameters()).device
    lq = lr.unsqueeze(0).to(device).mul(2.0).sub(1.0)
    context = nullcontext if fp32 or device.type != "cuda" else torch.cuda.amp.autocast
    with context():
        output = sampler.sample_func(lq, noise_repeat=False, mask=None)
    return output[0].float().cpu().mul(0.5).add(0.5).clamp(0, 1)


def read_base_summary(path: str) -> dict[str, float] | None:
    if not path:
        return None
    with Path(path).open(newline="", encoding="utf-8") as handle:
        matches = [
            row for row in csv.DictReader(handle)
            if row.get("method") == "Base-ResShift"
        ]
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one Base-ResShift row in {path}; found {len(matches)}"
        )
    return {
        "mean_psnr": float(matches[0]["mean_psnr"]),
        "mean_ssim": float(matches[0]["mean_ssim"]),
    }


def summarize(
    rows: list[dict],
    adapter_sizes: dict[str, float],
    external_base: dict[str, float] | None = None,
) -> list[dict]:
    summaries = []
    for method in dict.fromkeys(row["method"] for row in rows):
        selected = [row for row in rows if row["method"] == method]
        times = [
            float(row["inference_time_s"])
            for row in selected if row["inference_time_s"] != ""
        ]
        peaks = [
            float(row["peak_cuda_mem_mb"])
            for row in selected if row["peak_cuda_mem_mb"] != ""
        ]
        elapsed = sum(times)
        summaries.append({
            "method": method,
            "num_images": len(selected),
            "mean_psnr": sum(float(row["psnr"]) for row in selected) / len(selected),
            "mean_ssim": sum(float(row["ssim"]) for row in selected) / len(selected),
            "mean_inference_time_s": elapsed / len(times) if times else "",
            "num_timed_images": len(times),
            "images_per_hour": 3600 * len(times) / elapsed if elapsed else "",
            "peak_cuda_mem_mb": max(peaks) if peaks else "",
            "adapter_size_mb": adapter_sizes.get(method, 0.0),
        })
    base = next(
        (row for row in summaries if row["method"] == "Base-ResShift"),
        external_base,
    )
    if base is None:
        raise ValueError(
            "Base-ResShift was skipped but --base_metrics_csv was not supplied"
        )
    for row in summaries:
        row["delta_psnr_vs_base"] = row["mean_psnr"] - base["mean_psnr"]
        row["delta_ssim_vs_base"] = row["mean_ssim"] - base["mean_ssim"]
    return summaries


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config_path", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--autoencoder_checkpoint", required=True)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--train_dir_for_overlap_check", default="")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--adapter", action="append", type=parse_adapter, default=[])
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--lq_size", type=int, default=64)
    parser.add_argument("--sr_scale", type=int, default=4)
    parser.add_argument("--max_images", type=int, default=0)
    parser.add_argument("--exclude_image", action="append", default=[])
    parser.add_argument("--eval_seed", type=int, default=4242)
    parser.add_argument("--warmup_images", type=int, default=1)
    parser.add_argument("--crop_border", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip_base_eval", action="store_true")
    parser.add_argument(
        "--base_metrics_csv",
        default="",
        help="Existing summary containing one Base-ResShift row; required with --skip_base_eval",
    )
    parser.add_argument("--fp32", action="store_true")
    args = parser.parse_args()

    if args.image_size != args.lq_size * args.sr_scale:
        parser.error("Require image_size == lq_size * sr_scale")
    missing = [str(path) for _label, path in args.adapter if not path.is_file()]
    if missing:
        parser.error("Missing adapters: " + ", ".join(missing))
    if args.skip_base_eval and not args.adapter:
        parser.error("--skip_base_eval requires at least one --adapter")
    if args.skip_base_eval and not args.base_metrics_csv:
        parser.error("--skip_base_eval requires --base_metrics_csv")
    if args.base_metrics_csv and not Path(args.base_metrics_csv).is_file():
        parser.error(f"Missing base metrics CSV: {args.base_metrics_csv}")

    paths = image_paths(args.data_dir, args.max_images, set(args.exclude_image))
    overlap = audit_overlap(paths, args.train_dir_for_overlap_check)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    union_blocks, payloads = inspect_adapters(args.adapter)
    sampler = build_sampler(args)
    injected = inject_packed_qkv_lora(
        sampler.model,
        next(iter(payloads.values()))["rank"] if payloads else 8,
        next(iter(payloads.values()))["alpha"] if payloads else 16,
        selected_blocks=union_blocks,
    ) if union_blocks else {}
    sampler.model.eval()
    print(
        f"Official ResShift sampler ready: images={len(paths)} "
        f"union_blocks={len(union_blocks)} adapters={len(args.adapter)}"
    )

    adapter_sizes = {"Bicubic": 0.0, "Base-ResShift": 0.0}
    load_reports = {}
    rows = []

    bicubic_dir = output_dir / "images" / "Bicubic"
    for path in paths:
        hr = pil_to_tensor(Image.open(path), args.image_size)
        _lr, bicubic = make_lr(hr, args.sr_scale)
        output_path = bicubic_dir / path.name
        if not (args.resume and output_path.is_file()):
            atomic_save_png(bicubic, output_path)
        target = crop(hr, args.crop_border)
        prediction = crop(bicubic, args.crop_border)
        rows.append({
            "method": "Bicubic", "image": path.name,
            "psnr": psnr(prediction, target), "ssim": ssim(prediction, target),
            "inference_time_s": 0.0, "peak_cuda_mem_mb": 0.0,
        })

    methods = list(args.adapter) if args.skip_base_eval else [("Base-ResShift", None), *args.adapter]
    for method, adapter_path in methods:
        reset_lora(injected)
        if adapter_path is not None:
            report = load_adapter(injected, payloads[method])
            if report["missing"]:
                raise RuntimeError(f"{method}: missing targets {report['missing'][:5]}")
            load_reports[method] = report
            adapter_sizes[method] = adapter_path.stat().st_size / (1024**2)
            print(f"{method}: loaded LoRA modules={len(report['loaded'])}")

        method_dir = output_dir / "images" / safe_name(method)
        pending = any(not (method_dir / path.name).is_file() for path in paths)
        if pending:
            warm_hr = pil_to_tensor(Image.open(paths[0]), args.image_size)
            warm_lr, _ = make_lr(warm_hr, args.sr_scale)
            for warm_index in range(args.warmup_images):
                sample(sampler, warm_lr, args.eval_seed + 100000 + warm_index, args.fp32)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

        for index, path in enumerate(paths):
            hr = pil_to_tensor(Image.open(path), args.image_size)
            lr, _ = make_lr(hr, args.sr_scale)
            output_path = method_dir / path.name
            saved = load_complete_image(output_path, args.image_size) if args.resume else None
            if saved is not None:
                prediction = saved
                elapsed = ""
                peak_mb = ""
            else:
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                started = time.perf_counter()
                prediction = sample(sampler, lr, args.eval_seed + index, args.fp32)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                elapsed = time.perf_counter() - started
                peak_mb = (
                    torch.cuda.max_memory_allocated() / (1024**2)
                    if torch.cuda.is_available() else 0.0
                )
                atomic_save_png(prediction, output_path)

            target_eval = crop(hr, args.crop_border)
            prediction_eval = crop(prediction, args.crop_border)
            row = {
                "method": method,
                "image": path.name,
                "psnr": psnr(prediction_eval, target_eval),
                "ssim": ssim(prediction_eval, target_eval),
                "inference_time_s": elapsed,
                "peak_cuda_mem_mb": peak_mb,
            }
            rows.append(row)
            print(
                f"{method:<32} [{index + 1:03d}/{len(paths)}] "
                f"PSNR={row['psnr']:.3f} SSIM={row['ssim']:.4f}"
            )

    summaries = summarize(rows, adapter_sizes, read_base_summary(args.base_metrics_csv))
    write_csv(output_dir / "sr_metrics_per_image.csv", rows)
    write_csv(output_dir / "sr_metrics_summary.csv", summaries)
    metadata = {
        "config_path": str(Path(args.config_path).resolve()),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "autoencoder_checkpoint": str(Path(args.autoencoder_checkpoint).resolve()),
        "data_dir": str(Path(args.data_dir).resolve()),
        "num_images": len(paths),
        "excluded_images": args.exclude_image,
        "eval_seed": args.eval_seed,
        "official_sampler": True,
        "base_evaluation_skipped": args.skip_base_eval,
        "base_metrics_csv": args.base_metrics_csv or None,
        "inference_bypass_enabled": False,
        "union_lora_blocks": sorted(union_blocks),
        "load_reports": load_reports,
        "overlap_audit": overlap,
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, default=str), encoding="utf-8"
    )
    print(f"Wrote results to {output_dir}")


if __name__ == "__main__":
    main()
