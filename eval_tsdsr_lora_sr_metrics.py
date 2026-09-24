#!/usr/bin/env python3
"""Evaluate official TSD-SR and one or more domain LoRA checkpoints."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from diffusers import SD3Transformer2DModel, StableDiffusion3Pipeline
from peft import LoraConfig
from PIL import Image

from basicsr.utils.matlab_functions import imresize
from models.autoencoder_kl import AutoencoderKL
from utils.util import load_lora_state_dict
from utils.wavelet_color_fix import adain_color_fix, wavelet_color_fix


IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
TRANSFORMER_TARGETS = [
    "to_k",
    "to_q",
    "to_v",
    "to_out.0",
    "add_q_proj",
    "add_k_proj",
    "add_v_proj",
    "proj",
    "linear",
    "proj_out",
]
VAE_TARGETS = [
    "encoder.conv_in",
    "encoder.down_blocks.0.resnets.0.conv1",
    "encoder.down_blocks.0.resnets.0.conv2",
    "encoder.down_blocks.0.resnets.1.conv1",
    "encoder.down_blocks.0.resnets.1.conv2",
    "encoder.down_blocks.0.downsamplers.0.conv",
    "encoder.down_blocks.1.resnets.0.conv1",
    "encoder.down_blocks.1.resnets.0.conv2",
    "encoder.down_blocks.1.resnets.0.conv_shortcut",
    "encoder.down_blocks.1.resnets.1.conv1",
    "encoder.down_blocks.1.resnets.1.conv2",
    "encoder.down_blocks.1.downsamplers.0.conv",
    "encoder.down_blocks.2.resnets.0.conv1",
    "encoder.down_blocks.2.resnets.0.conv2",
    "encoder.down_blocks.2.resnets.0.conv_shortcut",
    "encoder.down_blocks.2.resnets.1.conv1",
    "encoder.down_blocks.2.resnets.1.conv2",
    "encoder.down_blocks.2.downsamplers.0.conv",
    "encoder.down_blocks.3.resnets.0.conv1",
    "encoder.down_blocks.3.resnets.0.conv2",
    "encoder.down_blocks.3.resnets.1.conv1",
    "encoder.down_blocks.3.resnets.1.conv2",
    "encoder.mid_block.attentions.0.to_q",
    "encoder.mid_block.attentions.0.to_k",
    "encoder.mid_block.attentions.0.to_v",
    "encoder.mid_block.attentions.0.to_out.0",
    "encoder.mid_block.resnets.0.conv1",
    "encoder.mid_block.resnets.0.conv2",
    "encoder.mid_block.resnets.1.conv1",
    "encoder.mid_block.resnets.1.conv2",
    "encoder.conv_out",
    "quant_conv",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pretrained_model", required=True)
    parser.add_argument("--official_lora_dir", required=True)
    parser.add_argument("--embedding_dir", required=True)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--eval_manifest", default="")
    parser.add_argument("--train_dir_for_overlap_check", default="")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--adapter",
        action="append",
        default=[],
        metavar="LABEL=PATH",
        help="Domain adapter checkpoint; may be repeated.",
    )
    parser.add_argument("--skip_base_eval", action="store_true")
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--sr_scale", type=int, default=4)
    parser.add_argument("--max_images", type=int, default=0)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=16.0)
    parser.add_argument("--eval_seed", type=int, default=4242)
    parser.add_argument("--warmup_images", type=int, default=1)
    parser.add_argument("--crop_border", type=int, default=4)
    parser.add_argument(
        "--color_fix", choices=["wavelet", "adain", "none"], default="wavelet"
    )
    parser.add_argument("--dtype", choices=["fp16", "bf16", "fp32"], default="fp16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--save_images", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--log_every", type=int, default=25)
    return parser.parse_args()


def parse_adapters(values: list[str]) -> list[tuple[str, Path]]:
    adapters = []
    labels = set()
    for value in values:
        if "=" not in value:
            raise SystemExit(f"--adapter must be LABEL=PATH, found {value!r}")
        label, raw_path = value.split("=", 1)
        label = label.strip()
        path = Path(raw_path).expanduser()
        if not label or label in labels:
            raise SystemExit(f"Adapter label is empty or duplicated: {label!r}")
        if not path.is_file():
            raise SystemExit(f"Adapter checkpoint not found: {path}")
        labels.add(label)
        adapters.append((label, path))
    return adapters


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write an empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def discover_images(data_dir: Path, manifest_path: Path | None, max_images: int) -> list[Path]:
    if not data_dir.is_dir():
        raise SystemExit(f"Evaluation directory not found: {data_dir}")
    paths = sorted(
        path for path in data_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    by_name = {path.name: path for path in paths}
    if len(by_name) != len(paths):
        raise SystemExit("Evaluation directory contains duplicate image filenames")
    if manifest_path is not None:
        with manifest_path.open(newline="", encoding="utf-8-sig") as handle:
            manifest = list(csv.DictReader(handle))
        names = [
            row["filename"] for row in manifest
            if str(row.get("exclude_from_eval", "false")).lower() not in {"1", "true", "yes"}
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
        raise SystemExit("No evaluation images were found")
    return paths


def pixel_sha256(path: Path) -> str:
    with Image.open(path) as image:
        image = image.convert("RGB")
        digest = hashlib.sha256()
        digest.update(f"RGB:{image.width}x{image.height}:".encode("ascii"))
        digest.update(image.tobytes())
    return digest.hexdigest()


def audit_overlap(eval_paths: list[Path], train_dir: Path | None) -> dict:
    if train_dir is None:
        return {"enabled": False}
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


def add_adapter(model, name: str, rank: int, alpha: float, targets: list[str], init) -> None:
    model.add_adapter(
        LoraConfig(
            r=rank,
            lora_alpha=alpha,
            init_lora_weights=init,
            target_modules=targets,
        ),
        adapter_name=name,
    )


def load_adapter(model, path: Path, filename: str, adapter_name: str) -> None:
    directory = path if path.is_dir() else path.parent
    weight_name = filename if path.is_dir() else path.name
    state = StableDiffusion3Pipeline.lora_state_dict(
        str(directory), weight_name=weight_name
    )
    load_lora_state_dict(state, model, adapter_name=adapter_name)


def load_models(args, adapters: list[tuple[str, Path]], dtype, device):
    transformer = SD3Transformer2DModel.from_pretrained(
        args.pretrained_model,
        subfolder="transformer",
        torch_dtype=dtype,
        local_files_only=True,
    )
    add_adapter(transformer, "official", 64, 64, TRANSFORMER_TARGETS, "gaussian")
    load_adapter(
        transformer,
        Path(args.official_lora_dir),
        "transformer.safetensors",
        "official",
    )
    internal_names = {}
    for index, (label, checkpoint) in enumerate(adapters):
        internal_name = f"domain_{index}"
        add_adapter(
            transformer,
            internal_name,
            args.rank,
            args.alpha,
            TRANSFORMER_TARGETS,
            True,
        )
        load_adapter(transformer, checkpoint, checkpoint.name, internal_name)
        internal_names[label] = internal_name

    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model,
        subfolder="vae",
        torch_dtype=dtype,
        local_files_only=True,
    )
    add_adapter(vae, "official", 64, 64, VAE_TARGETS, "gaussian")
    load_adapter(vae, Path(args.official_lora_dir), "vae.safetensors", "official")
    vae.set_adapter("official")

    transformer.requires_grad_(False).eval().to(device)
    vae.requires_grad_(False).eval().to(device)
    return transformer, vae, internal_names


def load_embeddings(directory: Path, device, dtype):
    prompt = torch.load(
        directory / "prompt_embeds.pt", map_location=device, weights_only=True
    ).to(dtype=dtype)
    pooled = torch.load(
        directory / "pool_embeds.pt", map_location=device, weights_only=True
    ).to(dtype=dtype)
    return prompt[:1], pooled[:1]


def image_to_tensor(image: Image.Image) -> torch.Tensor:
    array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


def tensor_to_pil(image: torch.Tensor) -> Image.Image:
    array = (
        image.detach().float().clamp(0, 1).mul(255).round().byte()
        .permute(1, 2, 0).cpu().numpy()
    )
    return Image.fromarray(array, mode="RGB")


def calculate_metrics(prediction: np.ndarray, target: np.ndarray, crop_border: int):
    from skimage.metrics import peak_signal_noise_ratio, structural_similarity

    if crop_border > 0:
        prediction = prediction[crop_border:-crop_border, crop_border:-crop_border]
        target = target[crop_border:-crop_border, crop_border:-crop_border]
    return (
        float(peak_signal_noise_ratio(target, prediction, data_range=255)),
        float(structural_similarity(target, prediction, channel_axis=2, data_range=255)),
    )


@torch.inference_mode()
def infer(transformer, vae, lr, prompt, pooled, image_size, device, dtype):
    upsampled = F.interpolate(
        lr.unsqueeze(0),
        size=(image_size, image_size),
        mode="bicubic",
        align_corners=False,
    )
    model_input = upsampled.mul(2).sub(1).to(device=device, dtype=dtype).clamp(-1, 1)
    latent = vae.encode(model_input).latent_dist.sample() * vae.config.scaling_factor
    timestep = torch.tensor([1000.0], device=device, dtype=dtype)
    prediction = transformer(
        hidden_states=latent,
        timestep=timestep,
        encoder_hidden_states=prompt,
        pooled_projections=pooled,
        return_dict=False,
    )[0]
    clean = latent - prediction
    decoded = vae.decode(
        clean / vae.config.scaling_factor, return_dict=False
    )[0].squeeze(0)
    return decoded.float().clamp(-1, 1).mul(0.5).add(0.5)


def main() -> None:
    args = parse_args()
    adapters = parse_adapters(args.adapter)
    if args.skip_base_eval and not adapters:
        raise SystemExit("Nothing to evaluate: base is skipped and no adapters were given")
    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise SystemExit(f"Output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = Path(args.eval_manifest) if args.eval_manifest else None
    eval_paths = discover_images(Path(args.data_dir), manifest, args.max_images)
    overlap = audit_overlap(
        eval_paths,
        Path(args.train_dir_for_overlap_check)
        if args.train_dir_for_overlap_check else None,
    )
    device = torch.device(args.device)
    dtype = {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
    }[args.dtype]
    transformer, vae, internal_names = load_models(args, adapters, dtype, device)
    prompt, pooled = load_embeddings(Path(args.embedding_dir), device, dtype)

    methods = [] if args.skip_base_eval else [("Base-TSD-SR", None, None)]
    methods.extend(
        (label, internal_names[label], checkpoint)
        for label, checkpoint in adapters
    )
    image_dirs = {}
    if args.save_images:
        for label, _, _ in methods:
            image_dirs[label] = output_dir / "images" / label
            image_dirs[label].mkdir(parents=True, exist_ok=True)

    rows = []
    method_times = {label: [] for label, _, _ in methods}
    torch.cuda.reset_peak_memory_stats(device)
    for image_index, hr_path in enumerate(eval_paths):
        with Image.open(hr_path) as source:
            hr_pil = source.convert("RGB").resize(
                (args.image_size, args.image_size), Image.Resampling.BICUBIC
            )
        hr = image_to_tensor(hr_pil)
        lr = imresize(hr, scale=1.0 / args.sr_scale).clamp(0, 1)
        lr_pil = tensor_to_pil(lr)
        lr_upscaled_pil = lr_pil.resize(
            (args.image_size, args.image_size), Image.Resampling.BICUBIC
        )

        for label, internal_name, _ in methods:
            transformer.set_adapter(
                "official" if internal_name is None else ["official", internal_name]
            )
            random.seed(args.eval_seed + image_index)
            torch.manual_seed(args.eval_seed + image_index)
            torch.cuda.manual_seed_all(args.eval_seed + image_index)
            torch.cuda.synchronize(device)
            started = time.perf_counter()
            prediction = infer(
                transformer,
                vae,
                lr,
                prompt,
                pooled,
                args.image_size,
                device,
                dtype,
            )
            torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - started
            if image_index >= args.warmup_images:
                method_times[label].append(elapsed)

            prediction_pil = tensor_to_pil(prediction)
            if args.color_fix == "wavelet":
                prediction_pil = wavelet_color_fix(prediction_pil, lr_upscaled_pil)
            elif args.color_fix == "adain":
                prediction_pil = adain_color_fix(prediction_pil, lr_upscaled_pil)
            prediction_array = np.asarray(prediction_pil.convert("RGB"), dtype=np.uint8)
            target_array = np.asarray(hr_pil, dtype=np.uint8)
            psnr, ssim = calculate_metrics(
                prediction_array, target_array, args.crop_border
            )
            rows.append({
                "method": label,
                "image_index": image_index,
                "filename": hr_path.name,
                "psnr": psnr,
                "ssim": ssim,
                "inference_time_s": elapsed,
            })
            if args.save_images:
                prediction_pil.save(
                    image_dirs[label] / f"{image_index:04d}_{hr_path.stem}.png"
                )
            if image_index == 0 or (image_index + 1) % args.log_every == 0:
                print(
                    f"{label:<28} [{image_index + 1:03d}/{len(eval_paths)}] "
                    f"PSNR={psnr:.3f} SSIM={ssim:.4f} time={elapsed:.3f}s"
                )

    summary = []
    adapter_sizes = {
        label: checkpoint.stat().st_size / 2**20
        for label, checkpoint in adapters
    }
    for label, _, _ in methods:
        selected = [row for row in rows if row["method"] == label]
        times = method_times[label]
        mean_time = float(np.mean(times)) if times else 0.0
        summary.append({
            "method": label,
            "num_images": len(selected),
            "mean_psnr": float(np.mean([row["psnr"] for row in selected])),
            "mean_ssim": float(np.mean([row["ssim"] for row in selected])),
            "mean_inference_time_s": mean_time,
            "num_timed_images": len(times),
            "images_per_hour": 3600.0 / mean_time if mean_time else "",
            "peak_cuda_mem_mb": torch.cuda.max_memory_allocated(device) / 2**20,
            "adapter_size_mb": adapter_sizes.get(label, 0.0),
        })
    write_csv(output_dir / "sr_metrics_per_image.csv", rows)
    write_csv(output_dir / "sr_metrics_summary.csv", summary)
    metadata = {
        **vars(args),
        "adapters": [
            {"label": label, "path": str(path), "size_mb": adapter_sizes[label]}
            for label, path in adapters
        ],
        "overlap_audit": overlap,
        "num_eval_images": len(eval_paths),
        "paired_vae_sampling": True,
        "official_transformer_and_vae_adapters_loaded": True,
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(f"Wrote results to {output_dir}")


if __name__ == "__main__":
    main()
