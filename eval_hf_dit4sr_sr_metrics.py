#!/usr/bin/env python3
"""Evaluate DiT4SR LoRA or Parallel Adapters with deterministic SR sampling."""

from __future__ import annotations

import argparse
import csv
import hashlib
import inspect
import json
import math
import re
import time
from pathlib import Path

from PIL import Image

import torch
import torch.nn.functional as F

import adaptive_grad_blockskip as adaptive
import dit4sr_parallel_adapter as parallel
import profile_hf_dit4sr_grad as core


def parse_adapter(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--adapter must use LABEL=PATH")
    label, path = value.split("=", 1)
    if not label.strip():
        raise argparse.ArgumentTypeError("Adapter label cannot be empty.")
    return label.strip(), Path(path).expanduser()


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def reset_lora_to_base(transformer: torch.nn.Module) -> None:
    for _name, module in core.iter_lora_modules(transformer):
        module.lora_down.weight.data.zero_()
        module.lora_up.weight.data.zero_()
        module.lora_enabled = False


def enable_nonzero_lora(transformer: torch.nn.Module) -> int:
    active = 0
    for _name, module in core.iter_lora_modules(transformer):
        enabled = bool(torch.count_nonzero(module.lora_up.weight).item())
        module.lora_enabled = enabled
        active += int(enabled)
    return active


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "method"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit_train_eval_overlap(
    train_dir: str, eval_paths: list[Path], allow_overlap: bool
) -> dict:
    if not train_dir:
        return {"enabled": False}
    train_paths = [Path(path) for path in core.list_images(train_dir)]
    train_hashes: dict[str, list[str]] = {}
    for path in train_paths:
        train_hashes.setdefault(file_sha256(path), []).append(str(path))

    overlaps = []
    for path in eval_paths:
        digest = file_sha256(Path(path))
        if digest in train_hashes:
            overlaps.append(
                {
                    "eval_image": str(path),
                    "train_images": train_hashes[digest],
                    "sha256": digest,
                }
            )
    report = {
        "enabled": True,
        "train_dir": train_dir,
        "num_train_images": len(train_paths),
        "num_eval_images": len(eval_paths),
        "num_overlapping_eval_images": len(overlaps),
        "overlaps": overlaps,
    }
    if overlaps and not allow_overlap:
        examples = ", ".join(item["eval_image"] for item in overlaps[:3])
        raise SystemExit(
            f"Found {len(overlaps)} evaluation images duplicated in the training set "
            f"(examples: {examples}). Use a disjoint test set, or pass --allow_overlap "
            f"only for an explicitly labeled training-set diagnostic."
        )
    return report


def pil_to_tensor(image: Image.Image, size: int) -> torch.Tensor:
    image = image.convert("RGB").resize((size, size), Image.Resampling.BICUBIC)
    data = torch.frombuffer(bytearray(image.tobytes()), dtype=torch.uint8)
    return data.view(size, size, 3).permute(2, 0, 1).float().div(255.0)


def tensor_to_pil(image: torch.Tensor) -> Image.Image:
    image = image.detach().float().clamp(0, 1).mul(255).round().byte().cpu()
    array = image.permute(1, 2, 0).contiguous().numpy()
    return Image.fromarray(array, mode="RGB")


def make_lr_condition(hr: torch.Tensor, sr_scale: int) -> tuple[torch.Tensor, torch.Tensor]:
    lr_size = hr.shape[-1] // sr_scale
    lr = F.interpolate(
        hr.unsqueeze(0), size=(lr_size, lr_size), mode="bicubic", align_corners=False,
        antialias=True,
    ).clamp(0, 1)
    bicubic = F.interpolate(
        lr, size=hr.shape[-2:], mode="bicubic", align_corners=False, antialias=True
    ).clamp(0, 1)
    return lr.squeeze(0), bicubic.squeeze(0)


@torch.no_grad()
def encode_control(vae, image: torch.Tensor, device: torch.device) -> torch.Tensor:
    dtype = next(vae.parameters()).dtype
    encoded = vae.encode(image.unsqueeze(0).to(device=device, dtype=dtype) * 2.0 - 1.0)
    latent = encoded.latent_dist.mode()
    scale = float(getattr(vae.config, "scaling_factor", 1.0))
    shift = float(getattr(vae.config, "shift_factor", 0.0))
    return ((latent - shift) * scale).to(dtype=dtype)


@torch.no_grad()
def decode_latent(vae, latent: torch.Tensor) -> torch.Tensor:
    scale = float(getattr(vae.config, "scaling_factor", 1.0))
    shift = float(getattr(vae.config, "shift_factor", 0.0))
    decoded = vae.decode(latent / scale + shift, return_dict=False)[0]
    return (decoded.float() / 2.0 + 0.5).clamp(0, 1)


def inference_forward(
    transformer: torch.nn.Module,
    latent: torch.Tensor,
    control: torch.Tensor,
    timestep: torch.Tensor,
    prompt_embeds: torch.Tensor,
    pooled_prompt_embeds: torch.Tensor,
    parallel_adapter=None,
    noise_ratio: float = 0.0,
):
    kwargs = {}
    for name, param in inspect.signature(transformer.forward).parameters.items():
        if name in {"hidden_states", "sample"}:
            kwargs[name] = latent
        elif name == "controlnet_image":
            kwargs[name] = control
        elif name in {"timestep", "timesteps"}:
            kwargs[name] = timestep.expand(latent.shape[0])
        elif name in {"encoder_hidden_states", "prompt_embeds"}:
            kwargs[name] = prompt_embeds
        elif name in {"pooled_projections", "pooled_prompt_embeds"}:
            kwargs[name] = pooled_prompt_embeds
        elif name in {"joint_attention_kwargs", "attention_kwargs"}:
            kwargs[name] = None
        elif name == "return_dict":
            kwargs[name] = False
        elif param.default is not inspect._empty:
            kwargs[name] = param.default
        else:
            raise RuntimeError(f"Unsupported required transformer argument: {name}")
    if parallel_adapter is None:
        return core.output_tensor(transformer(**kwargs))

    def base_forward():
        return core.output_tensor(transformer(**kwargs))

    prediction, _active = parallel_adapter.forward_prediction(
        base_forward,
        noise_ratio=noise_ratio,
        enable_grad=False,
    )
    return prediction


def adain_color_fix(output: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
    out_flat = output.flatten(2)
    src_flat = source.flatten(2)
    out_mean = out_flat.mean(dim=2).view(output.shape[0], output.shape[1], 1, 1)
    src_mean = src_flat.mean(dim=2).view(source.shape[0], source.shape[1], 1, 1)
    out_std = (out_flat.var(dim=2, unbiased=False) + 1e-5).sqrt().view_as(out_mean)
    src_std = (src_flat.var(dim=2, unbiased=False) + 1e-5).sqrt().view_as(src_mean)
    return ((output - out_mean) / out_std * src_std + src_mean).clamp(0, 1)


@torch.no_grad()
def sample_sr(
    pipe,
    transformer: torch.nn.Module,
    bicubic: torch.Tensor,
    args: argparse.Namespace,
    device: torch.device,
    seed: int,
    parallel_adapter=None,
) -> torch.Tensor:
    control = encode_control(pipe.vae, bicubic, device)
    dtype = next(transformer.parameters()).dtype
    generator = torch.Generator(device=device).manual_seed(seed)
    latent = torch.randn(control.shape, generator=generator, device=device, dtype=dtype)
    pipe.scheduler.set_timesteps(args.num_inference_steps, device=device)

    config = getattr(transformer, "config", None)
    caption_dim = int(getattr(config, "joint_attention_dim", args.caption_dim))
    pooled_dim = int(getattr(config, "pooled_projection_dim", args.pooled_dim))
    prompt = torch.zeros(
        latent.shape[0], args.prompt_seq_len, caption_dim, device=device, dtype=dtype
    )
    pooled = torch.zeros(latent.shape[0], pooled_dim, device=device, dtype=dtype)

    scheduler_sigmas = pipe.scheduler.sigmas
    for step_index, timestep in enumerate(pipe.scheduler.timesteps):
        noise_ratio = float(scheduler_sigmas[step_index].float().cpu())
        prediction = inference_forward(
            transformer,
            latent,
            control,
            timestep,
            prompt,
            pooled,
            parallel_adapter=parallel_adapter,
            noise_ratio=noise_ratio,
        )
        latent = pipe.scheduler.step(
            prediction, timestep, latent, return_dict=False
        )[0]

    output = decode_latent(pipe.vae, latent)
    if args.color_fix == "adain":
        source = bicubic.unsqueeze(0).to(device=device, dtype=output.dtype)
        output = adain_color_fix(output, source)
    return output.squeeze(0).cpu()


def crop(image: torch.Tensor, border: int) -> torch.Tensor:
    if border <= 0:
        return image
    if image.shape[-1] <= 2 * border or image.shape[-2] <= 2 * border:
        raise ValueError("--crop_border is too large for the evaluation image.")
    return image[..., border:-border, border:-border]


def psnr(prediction: torch.Tensor, target: torch.Tensor) -> float:
    mse = F.mse_loss(prediction.float(), target.float()).item()
    return 100.0 if mse <= 1e-10 else -10.0 * math.log10(mse)


def ssim(prediction: torch.Tensor, target: torch.Tensor) -> float:
    channels = prediction.shape[0]
    coords = torch.arange(11, dtype=torch.float32) - 5
    kernel_1d = torch.exp(-(coords.pow(2)) / (2 * 1.5**2))
    kernel_1d /= kernel_1d.sum()
    kernel = (kernel_1d[:, None] @ kernel_1d[None, :]).view(1, 1, 11, 11)
    kernel = kernel.repeat(channels, 1, 1, 1)
    x = prediction.float().unsqueeze(0)
    y = target.float().unsqueeze(0)
    x = F.pad(x, (5, 5, 5, 5), mode="reflect")
    y = F.pad(y, (5, 5, 5, 5), mode="reflect")
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


def build_lpips(enabled: bool, device_name: str):
    if not enabled:
        return None, "disabled"
    try:
        import lpips

        model = lpips.LPIPS(net="alex").eval().to(device_name)
        return model, "enabled"
    except Exception as exc:
        return None, f"unavailable: {exc}"


@torch.no_grad()
def lpips_score(model, prediction: torch.Tensor, target: torch.Tensor, device: str):
    if model is None:
        return ""
    x = prediction.unsqueeze(0).to(device) * 2.0 - 1.0
    y = target.unsqueeze(0).to(device) * 2.0 - 1.0
    return float(model(x, y).detach().cpu().flatten()[0])


def summarize(
    rows: list[dict], base_method: str, adapter_sizes_mb: dict[str, float]
) -> list[dict]:
    methods = list(dict.fromkeys(row["method"] for row in rows))
    summaries = []
    for method in methods:
        selected = [row for row in rows if row["method"] == method]
        numeric_lpips = [float(row["lpips"]) for row in selected if row["lpips"] != ""]
        total_inference_time = sum(
            float(row["inference_time_s"]) for row in selected
        )
        summaries.append(
            {
                "method": method,
                "num_images": len(selected),
                "mean_psnr": sum(float(row["psnr"]) for row in selected) / len(selected),
                "mean_ssim": sum(float(row["ssim"]) for row in selected) / len(selected),
                "mean_lpips": sum(numeric_lpips) / len(numeric_lpips) if numeric_lpips else "",
                "mean_inference_time_s": total_inference_time / len(selected),
                "images_per_hour": (
                    3600.0 * len(selected) / total_inference_time
                    if total_inference_time > 0
                    else ""
                ),
                "peak_cuda_mem_mb": max(float(row["peak_cuda_mem_mb"]) for row in selected),
                "adapter_size_mb": adapter_sizes_mb.get(method, 0.0),
            }
        )
    base = next(row for row in summaries if row["method"] == base_method)
    for row in summaries:
        row["delta_psnr_vs_base"] = float(row["mean_psnr"]) - float(base["mean_psnr"])
        row["delta_ssim_vs_base"] = float(row["mean_ssim"]) - float(base["mean_ssim"])
        row["delta_lpips_vs_base"] = (
            float(row["mean_lpips"]) - float(base["mean_lpips"])
            if row["mean_lpips"] != "" and base["mean_lpips"] != ""
            else ""
        )
    return summaries


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_id", default="acceptee/DiT4SR")
    parser.add_argument("--base_model_id", default="stabilityai/stable-diffusion-3.5-medium")
    parser.add_argument("--variant", default="dit4sr_q")
    parser.add_argument("--dit4sr_code_repo", default="acceptee/DiT4SR")
    parser.add_argument("--dit4sr_code_dir", default="")
    parser.add_argument("--transformer_subfolder", default="")
    parser.add_argument("--component_name", default="")
    parser.add_argument("--data_dir", required=True)
    parser.add_argument(
        "--exclude_image",
        action="append",
        default=[],
        help="Exclude an evaluation image by basename or full path; repeat as needed.",
    )
    parser.add_argument(
        "--train_dir_for_overlap_check",
        default="",
        help="Optional training directory used for a SHA-256 leakage check.",
    )
    parser.add_argument(
        "--allow_overlap",
        action="store_true",
        help="Allow train/eval duplicates for explicitly labeled diagnostics.",
    )
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--adapter", action="append", type=parse_adapter, default=[])
    parser.add_argument(
        "--parallel_adapter",
        action="append",
        type=parse_adapter,
        default=[],
        help="Noise-conditioned Parallel Adapter as LABEL=PATH; repeat as needed.",
    )
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--max_images", type=int, default=20)
    parser.add_argument("--sr_scale", type=int, default=4)
    parser.add_argument("--num_inference_steps", type=int, default=20)
    parser.add_argument("--eval_seed", type=int, default=4242)
    parser.add_argument("--warmup_images", type=int, default=1)
    parser.add_argument("--crop_border", type=int, default=4)
    parser.add_argument("--color_fix", choices=["none", "adain"], default="adain")
    parser.add_argument("--save_images", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--lpips", action="store_true")
    parser.add_argument("--lpips_device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--target", default="qv")
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--alpha", type=int, default=16)
    parser.add_argument("--block_regex", default="")
    parser.add_argument("--dtype", default="bf16", choices=["fp32", "bf16", "fp16"])
    parser.add_argument("--prompt_seq_len", type=int, default=77)
    parser.add_argument("--caption_dim", type=int, default=4096)
    parser.add_argument("--pooled_dim", type=int, default=2048)
    parser.add_argument("--latent_channels", type=int, default=16)
    parser.add_argument("--latent_size", type=int, default=64)
    parser.add_argument("--timestep", type=int, default=500)
    parser.add_argument("--input_noise_std", type=float, default=0.0)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--local_files_only", action="store_true")
    args = parser.parse_args()

    if args.image_size % args.sr_scale:
        raise SystemExit("--image_size must be divisible by --sr_scale.")
    missing = [
        str(path)
        for _label, path in [*args.adapter, *args.parallel_adapter]
        if not path.is_file()
    ]
    if missing:
        raise SystemExit("Missing adapter checkpoint(s): " + ", ".join(missing))

    lpips_model, lpips_status = build_lpips(args.lpips, args.lpips_device)
    print(f"LPIPS: {lpips_status}")
    args.loss_mode = "official_flow"
    args.load_mode = "transformer"
    args.model_impl = "official"
    args.pipeline_subfolder = ""
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    if args.lpips_device == "cuda" and device.type != "cuda":
        raise SystemExit("--lpips_device cuda requires CUDA.")

    pipe = core.load_pipe(args, device)
    transformer = core.get_transformer(pipe, args.component_name)
    transformer.requires_grad_(False)
    injected = core.inject_lora(transformer, args.target, args.rank, args.alpha, args.block_regex)
    transformer.eval()

    all_paths = [Path(path) for path in core.list_images(args.data_dir)]
    excluded_tokens = set(args.exclude_image)
    excluded_paths = [
        path for path in all_paths
        if str(path) in excluded_tokens or path.name in excluded_tokens
    ]
    matched_tokens = {
        token for token in excluded_tokens
        if any(str(path) == token or path.name == token for path in excluded_paths)
    }
    unmatched_tokens = sorted(excluded_tokens - matched_tokens)
    if unmatched_tokens:
        raise SystemExit(f"--exclude_image did not match: {unmatched_tokens}")
    paths = [path for path in all_paths if path not in set(excluded_paths)]
    if excluded_paths:
        print(
            "Excluded evaluation images: "
            + ", ".join(str(path) for path in excluded_paths)
        )
    if args.max_images > 0:
        paths = paths[: args.max_images]
    if not paths:
        raise SystemExit(f"No images found in {args.data_dir}")
    overlap_audit = audit_train_eval_overlap(
        args.train_dir_for_overlap_check, [Path(path) for path in paths], args.allow_overlap
    )
    if overlap_audit.get("enabled"):
        print(
            "Dataset overlap audit: "
            f"train={overlap_audit['num_train_images']} "
            f"eval={overlap_audit['num_eval_images']} "
            f"overlap={overlap_audit['num_overlapping_eval_images']}"
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    methods: list[tuple[str, str, Path | None]] = [
        ("Base-DiT4SR", "base", None),
        *((label, "lora", path) for label, path in args.adapter),
        *((label, "parallel", path) for label, path in args.parallel_adapter),
    ]
    image_rows = []
    load_reports = {}

    bicubic_rows = []
    for image_index, path in enumerate(paths):
        hr = pil_to_tensor(Image.open(path), args.image_size)
        _lr, bicubic = make_lr_condition(hr, args.sr_scale)
        metric_hr, metric_bicubic = crop(hr, args.crop_border), crop(bicubic, args.crop_border)
        bicubic_rows.append(
            {
                "method": "Bicubic",
                "image": str(path),
                "psnr": psnr(metric_bicubic, metric_hr),
                "ssim": ssim(metric_bicubic, metric_hr),
                "lpips": lpips_score(lpips_model, bicubic, hr, args.lpips_device),
                "inference_time_s": 0.0,
                "peak_cuda_mem_mb": 0.0,
                "output_path": "",
            }
        )
    image_rows.extend(bicubic_rows)

    for method, method_type, adapter_path in methods:
        reset_lora_to_base(transformer)
        parallel_controller = None
        if method_type == "lora" and adapter_path is not None:
            report = adaptive.load_lora_adapter(transformer, adapter_path)
            if report["missing"] or report["unexpected"]:
                raise RuntimeError(
                    f"{method}: adapter mismatch: missing={report['missing'][:5]} "
                    f"unexpected={report['unexpected'][:5]}"
                )
            report["active_lora_modules"] = enable_nonzero_lora(transformer)
            load_reports[method] = report
            print(
                f"{method}: active LoRA modules="
                f"{report['active_lora_modules']}/{len(injected)}"
            )
        elif method_type == "parallel" and adapter_path is not None:
            parallel_controller, report = parallel.load_parallel_adapter(
                transformer, adapter_path, device
            )
            load_reports[method] = report
            print(
                f"{method}: Parallel Adapter noise buckets="
                f"{len(report['active_schedule'])} anchors={len(report['anchors'])}"
            )

        if args.warmup_images > 0:
            warm_hr = pil_to_tensor(Image.open(paths[0]), args.image_size)
            _warm_lr, warm_bicubic = make_lr_condition(warm_hr, args.sr_scale)
            for warm_index in range(args.warmup_images):
                sample_sr(
                    pipe, transformer, warm_bicubic, args, device,
                    args.eval_seed + 100000 + warm_index,
                    parallel_adapter=parallel_controller,
                )

        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)

        method_dir = output_dir / "images" / safe_name(method)
        if args.save_images:
            method_dir.mkdir(parents=True, exist_ok=True)

        for image_index, path in enumerate(paths):
            hr = pil_to_tensor(Image.open(path), args.image_size)
            _lr, bicubic = make_lr_condition(hr, args.sr_scale)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            started = time.perf_counter()
            output = sample_sr(
                pipe,
                transformer,
                bicubic,
                args,
                device,
                args.eval_seed + image_index,
                parallel_adapter=parallel_controller,
            )
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - started
            peak_mb = (
                torch.cuda.max_memory_allocated(device) / (1024**2)
                if device.type == "cuda"
                else 0.0
            )

            output_path = ""
            if args.save_images:
                output_path_obj = method_dir / f"{image_index:04d}_{path.stem}.png"
                tensor_to_pil(output).save(output_path_obj)
                output_path = str(output_path_obj)
            metric_hr, metric_output = crop(hr, args.crop_border), crop(output, args.crop_border)
            row = {
                "method": method,
                "image": str(path),
                "psnr": psnr(metric_output, metric_hr),
                "ssim": ssim(metric_output, metric_hr),
                "lpips": lpips_score(lpips_model, output, hr, args.lpips_device),
                "inference_time_s": elapsed,
                "peak_cuda_mem_mb": peak_mb,
                "output_path": output_path,
            }
            image_rows.append(row)
            print(
                f"{method:28s} [{image_index + 1:03d}/{len(paths):03d}] "
                f"PSNR={row['psnr']:.3f} SSIM={row['ssim']:.4f} time={elapsed:.2f}s"
            )
        if parallel_controller is not None:
            parallel_controller.close()

    adapter_sizes_mb = {
        "Bicubic": 0.0,
        "Base-DiT4SR": 0.0,
        **{
            label: path.stat().st_size / (1024**2)
            for label, path in [*args.adapter, *args.parallel_adapter]
        },
    }
    summaries = summarize(image_rows, "Base-DiT4SR", adapter_sizes_mb)
    write_csv(output_dir / "sr_metrics_per_image.csv", image_rows)
    write_csv(output_dir / "sr_metrics_summary.csv", summaries)
    metadata = {
        "model_id": args.model_id,
        "base_model_id": args.base_model_id,
        "variant": args.variant,
        "data_dir": args.data_dir,
        "excluded_images": [str(path) for path in excluded_paths],
        "dataset_overlap_audit": overlap_audit,
        "image_size": args.image_size,
        "sr_scale": args.sr_scale,
        "num_inference_steps": args.num_inference_steps,
        "eval_seed": args.eval_seed,
        "crop_border": args.crop_border,
        "color_fix": args.color_fix,
        "prompt_condition": f"zeros[{args.prompt_seq_len},{args.caption_dim}]",
        "lpips_status": lpips_status,
        "injected_module_count": len(injected),
        "parallel_adapter_count": len(args.parallel_adapter),
        "load_reports": load_reports,
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, default=str), encoding="utf-8"
    )
    print(f"Wrote results to {output_dir}")


if __name__ == "__main__":
    main()
