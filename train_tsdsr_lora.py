#!/usr/bin/env python3
"""Train a fresh domain LoRA on top of the frozen official TSD-SR adapters."""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import time
from pathlib import Path

import pyiqa
import torch
import torch.nn.functional as F
from diffusers import FlowMatchEulerDiscreteScheduler, StableDiffusion3Pipeline
from peft.utils import get_peft_model_state_dict
from torch.utils.data import DataLoader

import profile_tsdsr_grad as core


BLOCK_PATTERN = re.compile(r"transformer_blocks\.\d+")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pretrained_model", required=True)
    parser.add_argument("--official_lora_dir", required=True)
    parser.add_argument("--teacher_lora_dir", required=True)
    parser.add_argument("--default_embedding_dir", required=True)
    parser.add_argument("--null_embedding_dir", required=True)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--method", default="TSD-SR-All-LoRA-AID-1000")
    parser.add_argument("--selection", choices=["all", "metadata"], default="all")
    parser.add_argument("--selection_file", default="")
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--sr_scale", type=int, default=4)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=16.0)
    parser.add_argument("--reg_rank", type=int, default=16)
    parser.add_argument("--train_steps", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--reg_lr", type=float, default=1e-6)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--checkpoint_every", type=int, default=250)
    parser.add_argument(
        "--train_noise_ratios",
        type=float,
        nargs="+",
        default=[0.05, 0.2, 0.4, 0.6, 0.8, 0.95],
    )
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--lambda_tsd", type=float, default=0.7)
    parser.add_argument("--lpips_weight", type=float, default=1.0)
    parser.add_argument("--latent_mse_weight", type=float, default=1.0)
    parser.add_argument("--tsd_weight", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dtype", choices=["fp16", "bf16", "fp32"], default="fp16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--log_every", type=int, default=10)
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_selected_blocks(path: str) -> set[str]:
    payload = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    blocks = payload.get("selected_lora_blocks", payload.get("selected_blocks"))
    if not isinstance(blocks, list) or not blocks:
        raise SystemExit("Selection metadata contains no selected LoRA blocks")
    return {str(block) for block in blocks}


def module_has_adapter(module: torch.nn.Module, adapter_name: str) -> bool:
    for attribute in (
        "lora_A",
        "lora_B",
        "lora_embedding_A",
        "lora_embedding_B",
        "lora_magnitude_vector",
    ):
        container = getattr(module, attribute, None)
        if container is not None and adapter_name in container:
            return True
    return False


def configure_domain_adapter(
    student: torch.nn.Module,
    selection: str,
    selected_blocks: set[str],
) -> tuple[int, int, int, list[str]]:
    active_module_names = []
    boundary_modules = 0
    active_blocks = set()
    for name, module in student.named_modules():
        if not module_has_adapter(module, "aid"):
            continue
        match = BLOCK_PATTERN.search(name)
        is_boundary = match is None
        block = match.group(0) if match else ""
        active = selection == "all" or is_boundary or block in selected_blocks
        module.set_adapter(["official", "aid"] if active else "official")
        if active:
            active_module_names.append(name)
            if is_boundary:
                boundary_modules += 1
            else:
                active_blocks.add(block)

    trainable_params = 0
    for name, parameter in student.named_parameters():
        match = BLOCK_PATTERN.search(name)
        active = (
            selection == "all"
            or match is None
            or match.group(0) in selected_blocks
        )
        parameter.requires_grad_(".aid." in name and active)
        if parameter.requires_grad:
            parameter.data = parameter.data.float()
            trainable_params += parameter.numel()
    return trainable_params, len(active_module_names), boundary_modules, sorted(active_blocks)


def set_reg_trainable(teacher: torch.nn.Module) -> list[torch.nn.Parameter]:
    parameters = []
    for name, parameter in teacher.named_parameters():
        active = ".reg." in name
        parameter.requires_grad_(active)
        if active:
            parameter.data = parameter.data.float()
            parameters.append(parameter)
    return parameters


def teacher_location(scheduler, ratio: float) -> tuple[int, torch.Tensor]:
    index = min(max(round(ratio * (len(scheduler.timesteps) - 1)), 50), 949)
    return index, scheduler.timesteps[index]


def student_loss(
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
    lr_latent, hr_latent, hr_input = core.make_latents(
        vae, lr, hr, args.image_size
    )
    batch_size = lr_latent.shape[0]
    cond = default_prompt[:batch_size]
    pooled = default_pooled[:batch_size]
    null_cond = null_prompt[:batch_size]
    null_pool = null_pooled[:batch_size]
    student_timestep = torch.full(
        (batch_size,), 1000.0, device=device, dtype=dtype
    )
    prediction = student(
        hidden_states=lr_latent,
        timestep=student_timestep,
        encoder_hidden_states=cond,
        pooled_projections=pooled,
        return_dict=False,
    )[0]
    latent_student = lr_latent - prediction

    teacher_index, teacher_timestep = teacher_location(scheduler, ratio)
    teacher_timestep = teacher_timestep.to(device=device, dtype=dtype).repeat(batch_size)
    sigma = core.scheduler_sigma(scheduler, teacher_index, lr_latent)
    noise = torch.randn_like(lr_latent)
    noisy_student = sigma * noise + (1 - sigma) * latent_student
    noisy_hr = sigma * noise + (1 - sigma) * hr_latent

    with torch.no_grad():
        teacher.set_adapter("teacher")
        teacher_student = core.cfg_forward(
            teacher,
            noisy_student,
            teacher_timestep,
            cond,
            pooled,
            null_cond,
            null_pool,
            args.guidance_scale,
        )
        teacher_hr = core.cfg_forward(
            teacher,
            noisy_hr,
            teacher_timestep,
            cond,
            pooled,
            null_cond,
            null_pool,
            args.guidance_scale,
        )
        teacher.set_adapter("reg")
        reg_student = core.cfg_forward(
            teacher,
            noisy_student,
            teacher_timestep,
            cond,
            pooled,
            null_cond,
            null_pool,
            args.guidance_scale,
        )
        grad_vsd = torch.nan_to_num(
            (teacher_student - reg_student) * sigma.square()
        )
        grad_tsm = torch.nan_to_num(
            (teacher_student - teacher_hr) * sigma.square()
        )
        pseudo_gradient = (
            args.lambda_tsd * grad_vsd + (1 - args.lambda_tsd) * grad_tsm
        )
        target = (latent_student - pseudo_gradient).detach()

    tsd = 0.5 * F.mse_loss(latent_student.float(), target.float())
    latent_mse = F.mse_loss(
        latent_student.float(), hr_latent.float().detach()
    )
    decoded = vae.decode(
        latent_student / vae.config.scaling_factor,
        return_dict=False,
    )[0].clamp(-1, 1)
    perceptual = lpips(
        decoded.mul(0.5).add(0.5), hr_input.mul(0.5).add(0.5)
    ).mean()
    total = (
        args.tsd_weight * tsd
        + args.latent_mse_weight * latent_mse
        + args.lpips_weight * perceptual.float()
    )
    context = {
        "latent_student": latent_student.detach(),
        "lr_latent": lr_latent.detach(),
        "cond": cond,
        "pooled": pooled,
        "teacher_index": teacher_index,
        "teacher_timestep": teacher_timestep,
        "sigma": sigma,
    }
    return total, tsd, latent_mse, perceptual, context


def regularizer_loss(teacher, scheduler, context, device) -> torch.Tensor:
    latent_student = context["latent_student"]
    sigma = context["sigma"]
    noisy_student = sigma * torch.randn_like(latent_student) + (1 - sigma) * latent_student
    teacher.set_adapter("reg")
    prediction = teacher(
        hidden_states=noisy_student,
        timestep=context["teacher_timestep"],
        encoder_hidden_states=context["cond"],
        pooled_projections=context["pooled"],
        return_dict=False,
    )[0]
    predicted_clean = noisy_student - sigma * prediction
    u = torch.normal(
        mean=0.0,
        std=1.0,
        size=(latent_student.shape[0],),
        device=device,
    )
    weight = torch.sigmoid(u).view(-1, 1, 1, 1)
    return torch.nan_to_num(
        0.5
        * weight
        * F.mse_loss(
            predicted_clean.float(),
            latent_student.float(),
            reduction="none",
        )
    ).mean()


def save_adapter(student, directory: Path) -> tuple[str, float]:
    directory.mkdir(parents=True, exist_ok=True)
    state = get_peft_model_state_dict(student, adapter_name="aid")
    StableDiffusion3Pipeline.save_lora_weights(
        str(directory),
        transformer_lora_layers=state,
        weight_name="transformer.safetensors",
    )
    path = directory / "transformer.safetensors"
    return str(path), path.stat().st_size / (1024.0**2)


def main() -> None:
    args = parse_args()
    if args.train_steps <= 0 or args.batch_size <= 0:
        raise SystemExit("--train_steps and --batch_size must be positive")
    if args.selection == "metadata" and not args.selection_file:
        raise SystemExit("--selection metadata requires --selection_file")
    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise SystemExit(f"Output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    core.set_seed(args.seed)
    device = torch.device(args.device)
    dtype = {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
    }[args.dtype]
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        args.pretrained_model,
        subfolder="scheduler",
        local_files_only=True,
    )
    student, teacher, vae = core.load_models(args, dtype, device)
    selected_blocks = (
        read_selected_blocks(args.selection_file)
        if args.selection == "metadata"
        else set()
    )
    trainable_params, active_modules, boundary_modules, active_blocks = (
        configure_domain_adapter(student, args.selection, selected_blocks)
    )
    reg_parameters = set_reg_trainable(teacher)
    student_parameters = [
        parameter for parameter in student.parameters() if parameter.requires_grad
    ]
    if not student_parameters or not reg_parameters:
        raise RuntimeError("No trainable student or regularizer parameters were found")

    optimizer = torch.optim.AdamW(
        student_parameters,
        lr=args.lr,
        betas=(0.9, 0.999),
        weight_decay=args.weight_decay,
        eps=1e-8,
    )
    reg_optimizer = torch.optim.AdamW(
        reg_parameters,
        lr=args.reg_lr,
        betas=(0.9, 0.999),
        weight_decay=args.weight_decay,
        eps=1e-8,
    )
    lpips = pyiqa.create_metric("lpips", as_loss=True, device=device)
    lpips.requires_grad_(False)

    dataset = core.PairedFolderDataset(
        args.data_dir,
        args.image_size,
        args.sr_scale,
        args.seed,
        max_images=0,
    )
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        generator=generator,
    )
    iterator = iter(loader)
    default_prompt, default_pooled = core.load_embeddings(
        args.default_embedding_dir, args.batch_size, device, dtype
    )
    null_prompt, null_pooled = core.load_embeddings(
        args.null_embedding_dir, args.batch_size, device, dtype
    )

    rows = []
    train_elapsed = 0.0
    experiment_started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    for step in range(1, args.train_steps + 1):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        ratio = random.choice(args.train_noise_ratios)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        started = time.perf_counter()

        optimizer.zero_grad(set_to_none=True)
        reg_optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type,
            dtype=dtype,
            enabled=device.type == "cuda" and dtype != torch.float32,
        ):
            loss, tsd, latent_mse, perceptual, context = student_loss(
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
            )
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite student loss at step {step}")
        loss.backward()
        student_grad_norm = torch.nn.utils.clip_grad_norm_(
            student_parameters, args.grad_clip
        )
        optimizer.step()

        set_reg_trainable(teacher)
        with torch.autocast(
            device_type=device.type,
            dtype=dtype,
            enabled=device.type == "cuda" and dtype != torch.float32,
        ):
            reg_loss = regularizer_loss(teacher, scheduler, context, device)
        if not torch.isfinite(reg_loss):
            raise RuntimeError(f"Non-finite regularizer loss at step {step}")
        reg_loss.backward()
        reg_grad_norm = torch.nn.utils.clip_grad_norm_(
            reg_parameters, args.grad_clip
        )
        reg_optimizer.step()

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - started
        train_elapsed += elapsed
        row = {
            "step": step,
            "noise_ratio": ratio,
            "teacher_schedule_index": context["teacher_index"],
            "loss": float(loss.detach().cpu()),
            "tsd_loss": float(tsd.detach().cpu()),
            "latent_mse_loss": float(latent_mse.detach().cpu()),
            "lpips_loss": float(perceptual.detach().cpu()),
            "regularizer_loss": float(reg_loss.detach().cpu()),
            "student_grad_norm": float(student_grad_norm.detach().cpu()),
            "regularizer_grad_norm": float(reg_grad_norm.detach().cpu()),
            "train_step_time_s": elapsed,
        }
        rows.append(row)
        if step == 1 or step % args.log_every == 0:
            print(
                f"step={step}/{args.train_steps} noise={ratio:.2f} "
                f"loss={row['loss']:.6f} reg={row['regularizer_loss']:.6f} "
                f"time={elapsed:.3f}s"
            )
        if args.checkpoint_every > 0 and step % args.checkpoint_every == 0:
            save_adapter(student, output_dir / f"checkpoint-{step:05d}")

    adapter_path, adapter_size_mb = save_adapter(student, output_dir)
    experiment_elapsed = time.perf_counter() - experiment_started
    peak_mb = (
        torch.cuda.max_memory_allocated(device) / (1024.0**2)
        if device.type == "cuda"
        else 0.0
    )
    write_csv(output_dir / "train_log.csv", rows)
    summary = [{
        "method": args.method,
        "train_steps": args.train_steps,
        "selection": args.selection,
        "selected_lora_blocks": len(active_blocks) if args.selection == "metadata" else 24,
        "active_domain_lora_modules": active_modules,
        "boundary_lora_modules": boundary_modules,
        "trainable_domain_lora_params": trainable_params,
        "train_step_time_s": train_elapsed,
        "mean_train_step_time_s": train_elapsed / args.train_steps,
        "experiment_time_s": experiment_elapsed,
        "non_train_overhead_s": experiment_elapsed - train_elapsed,
        "peak_cuda_mem_mb": peak_mb,
        "final_loss": rows[-1]["loss"],
        "mean_last100_loss": sum(row["loss"] for row in rows[-100:]) / min(100, len(rows)),
        "adapter_size_mb": adapter_size_mb,
        "adapter_path": adapter_path,
    }]
    write_csv(output_dir / "summary.csv", summary)
    metadata = {
        **vars(args),
        "model": "TSD-SR-MSE",
        "objective": "official_tsd_plus_latent_mse_plus_lpips_with_auxiliary_regularizer",
        "official_transformer_and_vae_adapters_frozen": True,
        "selected_lora_blocks": active_blocks,
        "active_domain_lora_modules": active_modules,
        "boundary_lora_modules": boundary_modules,
        "trainable_domain_lora_params": trainable_params,
        "regularizer_trainable_params": sum(parameter.numel() for parameter in reg_parameters),
        "adapter_path": adapter_path,
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary[0], indent=2))
    print(f"Wrote training results to {output_dir}")


if __name__ == "__main__":
    main()
