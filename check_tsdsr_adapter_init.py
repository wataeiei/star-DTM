#!/usr/bin/env python3
"""Verify that a fresh domain LoRA preserves the official TSD-SR output."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from diffusers import SD3Transformer2DModel, StableDiffusion3Pipeline
from peft import LoraConfig
from peft.utils import get_peft_model_state_dict

from utils.util import load_lora_state_dict


TARGET_MODULES = [
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pretrained_model", required=True)
    parser.add_argument("--official_lora_dir", required=True)
    parser.add_argument("--embedding_dir", required=True)
    parser.add_argument("--output_json", default="")
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=16.0)
    parser.add_argument("--latent_size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["fp16", "bf16", "fp32"], default="fp16")
    return parser.parse_args()


def activate_adapters(model: SD3Transformer2DModel, names: str | list[str]) -> None:
    """Use the singular API exposed by diffusers 0.29.x."""
    model.set_adapter(names)


def main() -> None:
    args = parse_args()
    dtype = {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
    }[args.dtype]

    model = SD3Transformer2DModel.from_pretrained(
        args.pretrained_model,
        subfolder="transformer",
        torch_dtype=dtype,
        local_files_only=True,
    )

    official_config = LoraConfig(
        r=64,
        lora_alpha=64,
        init_lora_weights="gaussian",
        target_modules=TARGET_MODULES,
    )
    model.add_adapter(official_config, adapter_name="official")

    official_state = StableDiffusion3Pipeline.lora_state_dict(
        args.official_lora_dir,
        weight_name="transformer.safetensors",
    )
    load_lora_state_dict(official_state, model, adapter_name="official")

    model.to(args.device)
    model.eval()
    activate_adapters(model, "official")

    embedding_dir = Path(args.embedding_dir)
    prompt = torch.load(
        embedding_dir / "prompt_embeds.pt",
        map_location=args.device,
        weights_only=True,
    ).to(dtype=dtype)
    pooled = torch.load(
        embedding_dir / "pool_embeds.pt",
        map_location=args.device,
        weights_only=True,
    ).to(dtype=dtype)

    torch.manual_seed(args.seed)
    hidden = torch.randn(
        1,
        model.config.in_channels,
        args.latent_size,
        args.latent_size,
        device=args.device,
        dtype=dtype,
    )
    timestep = torch.tensor([1000.0], device=args.device, dtype=dtype)

    with torch.no_grad():
        official_output = model(
            hidden_states=hidden,
            timestep=timestep,
            encoder_hidden_states=prompt,
            pooled_projections=pooled,
            return_dict=False,
        )[0]

    domain_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.alpha,
        init_lora_weights=True,
        target_modules=TARGET_MODULES,
    )
    model.add_adapter(domain_config, adapter_name="aid")
    activate_adapters(model, ["official", "aid"])
    model.eval()

    for name, parameter in model.named_parameters():
        parameter.requires_grad_(".aid." in name)

    with torch.no_grad():
        stacked_output = model(
            hidden_states=hidden,
            timestep=timestep,
            encoder_hidden_states=prompt,
            pooled_projections=pooled,
            return_dict=False,
        )[0]

    domain_state = get_peft_model_state_dict(model, adapter_name="aid")
    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    module_count = sum("lora_A" in key for key in domain_state)
    difference = (official_output - stacked_output).abs()

    report = {
        "model": "TSD-SR-MSE",
        "official_adapter": str(Path(args.official_lora_dir)),
        "domain_adapter": "aid",
        "domain_rank": args.rank,
        "domain_alpha": args.alpha,
        "domain_lora_modules": module_count,
        "domain_trainable_parameters": trainable_parameters,
        "domain_raw_fp32_mb": trainable_parameters * 4 / 2**20,
        "output_max_abs_diff": difference.max().item(),
        "output_mean_abs_diff": difference.mean().item(),
        "finite_output": bool(torch.isfinite(stacked_output).all().item()),
        "initialization_compatible": bool(difference.max().item() == 0.0),
    }

    print(json.dumps(report, indent=2))
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    if not report["finite_output"] or not report["initialization_compatible"]:
        raise SystemExit("Fresh domain adapter changed the initial TSD-SR output")


if __name__ == "__main__":
    main()
