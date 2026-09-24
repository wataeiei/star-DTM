#!/usr/bin/env python3
"""Profile TSD-SR domain-LoRA block importance on paired SR images."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
from pathlib import Path

import pyiqa
import torch
import torch.nn.functional as F
from diffusers import (
    FlowMatchEulerDiscreteScheduler,
    SD3Transformer2DModel,
    StableDiffusion3Pipeline,
)
from peft import LoraConfig
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from basicsr.utils.matlab_functions import imresize
from models.autoencoder_kl import AutoencoderKL
from utils.util import load_lora_state_dict


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
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
REG_TARGETS = [
    "to_k",
    "to_q",
    "to_v",
    "to_out.0",
    "add_q_proj",
    "add_k_proj",
    "add_v_proj",
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
    parser.add_argument("--teacher_lora_dir", required=True)
    parser.add_argument("--default_embedding_dir", required=True)
    parser.add_argument("--null_embedding_dir", required=True)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--sr_scale", type=int, default=4)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=16.0)
    parser.add_argument("--reg_rank", type=int, default=16)
    parser.add_argument("--probe_batches", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_images", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--noise_ratios", type=float, nargs="+", default=[0.05, 0.2, 0.4, 0.6, 0.8, 0.95])
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--lambda_tsd", type=float, default=0.7)
    parser.add_argument("--lpips_weight", type=float, default=1.0)
    parser.add_argument("--latent_mse_weight", type=float, default=1.0)
    parser.add_argument("--tsd_weight", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--dtype", choices=["fp16", "bf16", "fp32"], default="fp16")
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


class PairedFolderDataset(Dataset):
    def __init__(self, root: str, image_size: int, scale: int, seed: int, max_images: int) -> None:
        paths = sorted(
            path for path in Path(root).rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES
        )
        if not paths:
            raise SystemExit(f"No images found in {root}")
        random.Random(seed).shuffle(paths)
        self.paths = paths[:max_images] if max_images > 0 else paths
        self.image_size = image_size
        self.scale = scale

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> dict:
        image = Image.open(self.paths[index]).convert("RGB")
        image = image.resize((self.image_size, self.image_size), Image.Resampling.BICUBIC)
        array = torch.frombuffer(bytearray(image.tobytes()), dtype=torch.uint8)
        hr = array.view(self.image_size, self.image_size, 3).permute(2, 0, 1).float() / 255.0
        lr = imresize(hr, scale=1.0 / self.scale).clamp(0, 1)
        return {"hr": hr, "lr": lr, "path": str(self.paths[index])}


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


def load_adapter(model, directory: str, filename: str, adapter_name: str) -> None:
    state = StableDiffusion3Pipeline.lora_state_dict(directory, weight_name=filename)
    load_lora_state_dict(state, model, adapter_name=adapter_name)


def freeze_except_domain(model) -> None:
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(".aid." in name)


def load_models(args, dtype, device):
    student = SD3Transformer2DModel.from_pretrained(
        args.pretrained_model,
        subfolder="transformer",
        torch_dtype=dtype,
        local_files_only=True,
    )
    add_adapter(student, "official", 64, 64, TRANSFORMER_TARGETS, "gaussian")
    load_adapter(student, args.official_lora_dir, "transformer.safetensors", "official")
    add_adapter(student, "aid", args.rank, args.alpha, TRANSFORMER_TARGETS, True)
    student.set_adapter(["official", "aid"])
    freeze_except_domain(student)

    teacher = SD3Transformer2DModel.from_pretrained(
        args.pretrained_model,
        subfolder="transformer",
        torch_dtype=dtype,
        local_files_only=True,
    )
    add_adapter(teacher, "teacher", 64, 64, TRANSFORMER_TARGETS, "gaussian")
    load_adapter(teacher, args.teacher_lora_dir, "teacher.safetensors", "teacher")
    add_adapter(teacher, "reg", args.reg_rank, args.reg_rank, REG_TARGETS, "gaussian")
    teacher.requires_grad_(False)

    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model,
        subfolder="vae",
        torch_dtype=dtype,
        local_files_only=True,
    )
    add_adapter(vae, "official", 64, 64, VAE_TARGETS, "gaussian")
    load_adapter(vae, args.official_lora_dir, "vae.safetensors", "official")
    vae.set_adapter("official")
    vae.requires_grad_(False)

    student.to(device).train()
    teacher.to(device).eval()
    vae.to(device).eval()
    return student, teacher, vae


def load_embeddings(directory: str, batch_size: int, device, dtype):
    root = Path(directory)
    prompt = torch.load(root / "prompt_embeds.pt", map_location=device, weights_only=True).to(dtype=dtype)
    pooled = torch.load(root / "pool_embeds.pt", map_location=device, weights_only=True).to(dtype=dtype)
    return prompt.repeat(batch_size, 1, 1), pooled.repeat(batch_size, 1)


def scheduler_sigma(scheduler, index: int, latent: torch.Tensor) -> torch.Tensor:
    sigma = scheduler.sigmas[index].to(device=latent.device, dtype=latent.dtype)
    return sigma.reshape(1, *([1] * (latent.ndim - 1)))


def cfg_forward(model, latent, timestep, cond, pooled, null_cond, null_pooled, scale):
    prediction = model(
        hidden_states=torch.cat([latent, latent]),
        timestep=torch.cat([timestep, timestep]),
        encoder_hidden_states=torch.cat([null_cond, cond]),
        pooled_projections=torch.cat([null_pooled, pooled]),
        return_dict=False,
    )[0]
    uncond, conditional = prediction.chunk(2)
    return uncond + scale * (conditional - uncond)


def make_latents(vae, lr, hr, image_size: int):
    lr_up = F.interpolate(lr, size=(image_size, image_size), mode="bicubic", align_corners=False)
    lr_input = lr_up.mul(2).sub(1).clamp(-1, 1)
    hr_input = hr.mul(2).sub(1).clamp(-1, 1)
    with torch.no_grad():
        lr_latent = vae.encode(lr_input).latent_dist.sample() * vae.config.scaling_factor
        hr_latent = vae.encode(hr_input).latent_dist.sample() * vae.config.scaling_factor
    return lr_latent, hr_latent, hr_input


def profile_loss(
    student,
    teacher,
    vae,
    lpips,
    scheduler,
    batch,
    ratio,
    args,
    device,
    dtype,
    default_prompt,
    default_pooled,
    null_prompt,
    null_pooled,
):
    hr = batch["hr"].to(device=device, dtype=dtype)
    lr = batch["lr"].to(device=device, dtype=dtype)
    lr_latent, hr_latent, hr_input = make_latents(vae, lr, hr, args.image_size)
    batch_size = lr_latent.shape[0]

    cond = default_prompt[:batch_size]
    pooled = default_pooled[:batch_size]
    null_cond = null_prompt[:batch_size]
    null_pool = null_pooled[:batch_size]
    student_timestep = torch.full((batch_size,), 1000.0, device=device, dtype=dtype)

    prediction = student(
        hidden_states=lr_latent,
        timestep=student_timestep,
        encoder_hidden_states=cond,
        pooled_projections=pooled,
        return_dict=False,
    )[0]
    latent_student = lr_latent - prediction

    teacher_index = min(max(round(ratio * (len(scheduler.timesteps) - 1)), 50), 949)
    teacher_timestep = scheduler.timesteps[teacher_index].to(device=device, dtype=dtype)
    teacher_timestep = teacher_timestep.repeat(batch_size)
    sigma = scheduler_sigma(scheduler, teacher_index, lr_latent)
    noise = torch.randn_like(lr_latent)
    noisy_student = sigma * noise + (1 - sigma) * latent_student
    noisy_hr = sigma * noise + (1 - sigma) * hr_latent

    with torch.no_grad():
        teacher.set_adapter("teacher")
        teacher_student = cfg_forward(
            teacher, noisy_student, teacher_timestep, cond, pooled,
            null_cond, null_pool, args.guidance_scale,
        )
        teacher_hr = cfg_forward(
            teacher, noisy_hr, teacher_timestep, cond, pooled,
            null_cond, null_pool, args.guidance_scale,
        )
        teacher.set_adapter("reg")
        reg_student = cfg_forward(
            teacher, noisy_student, teacher_timestep, cond, pooled,
            null_cond, null_pool, args.guidance_scale,
        )
        grad_vsd = torch.nan_to_num((teacher_student - reg_student) * sigma.square())
        grad_tsm = torch.nan_to_num((teacher_student - teacher_hr) * sigma.square())
        pseudo_gradient = args.lambda_tsd * grad_vsd + (1 - args.lambda_tsd) * grad_tsm
        tsd_target = (latent_student - pseudo_gradient).detach()

    tsd_loss = 0.5 * F.mse_loss(latent_student.float(), tsd_target.float())
    latent_mse = F.mse_loss(latent_student.float(), hr_latent.float().detach())
    decoded = vae.decode(
        latent_student / vae.config.scaling_factor,
        return_dict=False,
    )[0].clamp(-1, 1)
    lpips_loss = lpips(decoded.mul(0.5).add(0.5), hr_input.mul(0.5).add(0.5)).mean()
    total = (
        args.tsd_weight * tsd_loss
        + args.latent_mse_weight * latent_mse
        + args.lpips_weight * lpips_loss.float()
    )
    return total, tsd_loss, latent_mse, lpips_loss, teacher_index, float(teacher_timestep[0].item())


def collect_block_scores(student) -> list[dict]:
    blocks: dict[str, dict] = {}
    boundary_params = 0
    boundary_modules = set()
    for name, parameter in student.named_parameters():
        if ".aid." not in name or "lora_" not in name:
            continue
        match = re.search(r"transformer_blocks\.\d+", name)
        if match is None:
            boundary_params += parameter.numel()
            boundary_modules.add(name.split(".lora_", 1)[0])
            continue
        block = match.group(0)
        row = blocks.setdefault(
            block,
            {"block": block, "grad_sq": 0.0, "lora_param_count": 0, "modules": set()},
        )
        if parameter.grad is not None:
            row["grad_sq"] += float(parameter.grad.detach().float().square().sum().cpu())
        row["lora_param_count"] += parameter.numel()
        row["modules"].add(name.split(".lora_", 1)[0])

    rows = []
    for block, row in blocks.items():
        grad_norm = math.sqrt(row["grad_sq"])
        count = row["lora_param_count"]
        rows.append({
            "block": block,
            "block_index": int(block.rsplit(".", 1)[1]),
            "grad_norm": grad_norm,
            "lora_param_count": count,
            "module_count": len(row["modules"]),
            "normalized_grad_score": grad_norm / math.sqrt(max(count, 1)),
        })
    rows.sort(key=lambda row: row["block_index"])
    return rows, boundary_params, len(boundary_modules)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[args.dtype]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        args.pretrained_model, subfolder="scheduler", local_files_only=True
    )
    student, teacher, vae = load_models(args, dtype, device)
    lpips = pyiqa.create_metric("lpips", as_loss=True, device=device)
    lpips.requires_grad_(False)

    dataset = PairedFolderDataset(
        args.data_dir, args.image_size, args.sr_scale, args.seed, args.max_images
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    batches = []
    for batch in loader:
        batches.append(batch)
        if len(batches) >= args.probe_batches:
            break
    if not batches:
        raise SystemExit("No probe batches were loaded")

    default_prompt, default_pooled = load_embeddings(
        args.default_embedding_dir, args.batch_size, device, dtype
    )
    null_prompt, null_pooled = load_embeddings(
        args.null_embedding_dir, args.batch_size, device, dtype
    )

    all_rows = []
    boundary_params = 0
    boundary_modules = 0
    for ratio in args.noise_ratios:
        student.zero_grad(set_to_none=True)
        sums = {"loss": 0.0, "tsd": 0.0, "mse": 0.0, "lpips": 0.0}
        teacher_index = None
        timestep = None
        valid = 0
        for batch_index, batch in enumerate(batches):
            set_seed(args.seed + batch_index)
            with torch.autocast(device_type=device.type, dtype=dtype, enabled=dtype != torch.float32):
                values = profile_loss(
                    student, teacher, vae, lpips, scheduler, batch, ratio, args,
                    device, dtype, default_prompt, default_pooled, null_prompt, null_pooled,
                )
            loss, tsd, mse, perceptual, teacher_index, timestep = values
            if not torch.isfinite(loss):
                raise SystemExit(f"Non-finite probe loss at noise ratio {ratio}")
            loss.backward()
            valid += 1
            sums["loss"] += float(loss.detach().cpu())
            sums["tsd"] += float(tsd.detach().cpu())
            sums["mse"] += float(mse.detach().cpu())
            sums["lpips"] += float(perceptual.detach().cpu())
            print(
                f"noise={ratio:.2f} batch={batch_index + 1}/{len(batches)} "
                f"loss={float(loss.detach().cpu()):.6f}"
            )

        rows, boundary_params, boundary_modules = collect_block_scores(student)
        total_score = sum(row["normalized_grad_score"] for row in rows)
        ranked = sorted(rows, key=lambda row: row["normalized_grad_score"], reverse=True)
        rank_by_block = {row["block"]: rank + 1 for rank, row in enumerate(ranked)}
        for row in rows:
            row.update({
                "noise_ratio": ratio,
                "teacher_schedule_index": teacher_index,
                "timestep": timestep,
                "importance_rank": rank_by_block[row["block"]],
                "score_share": row["normalized_grad_score"] / max(total_score, 1e-30),
                "probe_batches": valid,
                "mean_probe_loss": sums["loss"] / valid,
                "mean_tsd_loss": sums["tsd"] / valid,
                "mean_latent_mse": sums["mse"] / valid,
                "mean_lpips_loss": sums["lpips"] / valid,
            })
            all_rows.append(row)
        torch.cuda.empty_cache()

    all_rows.sort(key=lambda row: (row["noise_ratio"], row["block_index"]))
    write_csv(output_dir / "lora_importance_evolution.csv", all_rows)

    blocks = sorted({row["block"] for row in all_rows}, key=lambda value: int(value.rsplit(".", 1)[1]))
    aggregate = []
    for block in blocks:
        selected = [row for row in all_rows if row["block"] == block]
        mean_share = sum(row["score_share"] for row in selected) / len(selected)
        mean_rank = sum(row["importance_rank"] for row in selected) / len(selected)
        aggregate.append({
            "block": block,
            "block_index": selected[0]["block_index"],
            "mean_score_share": mean_share,
            "mean_rank": mean_rank,
            "lora_param_count": selected[0]["lora_param_count"],
            "module_count": selected[0]["module_count"],
        })
    aggregate.sort(key=lambda row: row["mean_score_share"], reverse=True)
    cumulative = 0.0
    for rank, row in enumerate(aggregate, 1):
        cumulative += row["mean_score_share"]
        row["aggregate_rank"] = rank
        row["cumulative_utility"] = cumulative
    write_csv(output_dir / "lora_importance_aggregate.csv", aggregate)

    metadata = {
        "model": "TSD-SR-MSE",
        "objective": "official_tsd_plus_latent_mse_plus_lpips",
        "pretrained_model": args.pretrained_model,
        "official_lora_dir": args.official_lora_dir,
        "teacher_lora_dir": args.teacher_lora_dir,
        "data_dir": args.data_dir,
        "image_size": args.image_size,
        "sr_scale": args.sr_scale,
        "rank": args.rank,
        "alpha": args.alpha,
        "candidate_block_count": len(blocks),
        "boundary_lora_module_count": boundary_modules,
        "boundary_lora_param_count": boundary_params,
        "noise_anchors": args.noise_ratios,
        "probe_batches_per_anchor": len(batches),
        "seed": args.seed,
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Wrote {output_dir / 'lora_importance_evolution.csv'}")
    print(f"Wrote {output_dir / 'lora_importance_aggregate.csv'}")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
