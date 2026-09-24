import json
import math
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


class VOSRLoRALinear(torch.nn.Module):
    def __init__(self, base_layer, rank=8, alpha=16.0):
        super().__init__()
        if not isinstance(base_layer, torch.nn.Linear):
            raise TypeError(type(base_layer))

        self.base_layer = base_layer
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.rank

        for parameter in self.base_layer.parameters():
            parameter.requires_grad = False

        self.lora_A = torch.nn.Linear(
            base_layer.in_features,
            self.rank,
            bias=False,
        )
        self.lora_B = torch.nn.Linear(
            self.rank,
            base_layer.out_features,
            bias=False,
        )

        torch.nn.init.kaiming_uniform_(
            self.lora_A.weight,
            a=math.sqrt(5),
        )
        torch.nn.init.zeros_(self.lora_B.weight)

    def forward(self, x):
        base_output = self.base_layer(x)
        update = self.lora_B(self.lora_A(x))
        return base_output + update.to(base_output.dtype) * self.scaling


def inject_vosr_lora(
    model,
    selected_blocks=None,
    rank=8,
    alpha=16.0,
):
    if not hasattr(model, "blocks"):
        raise RuntimeError("Model has no blocks attribute")

    num_blocks = len(model.blocks)
    if selected_blocks is None:
        selected_blocks = list(range(num_blocks))
    selected_blocks = sorted(set(int(x) for x in selected_blocks))

    invalid = [
        index for index in selected_blocks
        if index < 0 or index >= num_blocks
    ]
    if invalid:
        raise ValueError(f"Invalid block indices: {invalid}")

    for parameter in model.parameters():
        parameter.requires_grad = False

    injected = []
    for block_index in selected_blocks:
        block = model.blocks[block_index]
        targets = (
            (block.attn, "qkv", "attn.qkv"),
            (
                block.cross_attn,
                "q_linear",
                "cross_attn.q_linear",
            ),
            (
                block.cross_attn,
                "v_linear",
                "cross_attn.v_linear",
            ),
        )

        for parent, attribute, suffix in targets:
            base_layer = getattr(parent, attribute)
            if isinstance(base_layer, VOSRLoRALinear):
                raise RuntimeError(
                    f"Already wrapped: blocks.{block_index}.{suffix}"
                )

            setattr(
                parent,
                attribute,
                VOSRLoRALinear(
                    base_layer,
                    rank=rank,
                    alpha=alpha,
                ),
            )
            injected.append(f"blocks.{block_index}.{suffix}")

    return injected


def get_lora_state_dict(model):
    return {
        name: tensor.detach().cpu().contiguous()
        for name, tensor in model.state_dict().items()
        if ".lora_A." in name or ".lora_B." in name
    }


def validate_lora_model(
    model,
    expected_blocks,
    rank=8,
):
    state = get_lora_state_dict(model)
    a = {k: v for k, v in state.items() if ".lora_A." in k}
    b = {k: v for k, v in state.items() if ".lora_B." in k}

    expected_modules = expected_blocks * 3
    expected_parameters = expected_blocks * 65536

    trainable_names = [
        name for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    unexpected_trainable = [
        name for name in trainable_names
        if ".lora_A." not in name and ".lora_B." not in name
    ]
    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    if len(a) != expected_modules or len(b) != expected_modules:
        raise RuntimeError(
            f"Expected {expected_modules} A/B tensors, "
            f"found A={len(a)} B={len(b)}"
        )
    if trainable_parameters != expected_parameters:
        raise RuntimeError(
            f"Expected {expected_parameters} trainable parameters, "
            f"found {trainable_parameters}"
        )
    if unexpected_trainable:
        raise RuntimeError(
            f"Unexpected trainable parameters: "
            f"{unexpected_trainable[:10]}"
        )

    return {
        "lora_module_count": expected_modules,
        "trainable_lora_params": trainable_parameters,
        "rank": rank,
    }


def save_vosr_lora_adapter(
    model,
    output_dir,
    selected_blocks,
    global_step,
    final_loss,
    method,
    rank=8,
    alpha=16.0,
):
    output_dir = Path(output_dir)
    adapter_dir = output_dir / "lora_adapter"
    adapter_dir.mkdir(parents=True, exist_ok=True)

    state = get_lora_state_dict(model)
    a = {k: v for k, v in state.items() if ".lora_A." in k}
    b = {k: v for k, v in state.items() if ".lora_B." in k}

    expected_modules = len(selected_blocks) * 3
    expected_parameters = len(selected_blocks) * 65536

    if len(a) != expected_modules or len(b) != expected_modules:
        raise RuntimeError(
            f"Save validation failed: A={len(a)} B={len(b)}, "
            f"expected={expected_modules}"
        )

    stored_parameters = sum(x.numel() for x in state.values())
    if stored_parameters != expected_parameters:
        raise RuntimeError(
            f"Stored parameters={stored_parameters}, "
            f"expected={expected_parameters}"
        )

    nonfinite = sum(
        int((~torch.isfinite(x)).sum().item())
        for x in state.values()
    )
    if nonfinite:
        raise RuntimeError(f"Non-finite LoRA values: {nonfinite}")

    adapter_path = adapter_dir / "adapter_model.safetensors"
    save_file(state, str(adapter_path))

    config = {
        "format": "vosr_native_lora_v1",
        "model": "VOSR-0.5B-ms",
        "rank": rank,
        "alpha": alpha,
        "selected_blocks": list(selected_blocks),
        "target_suffixes": [
            "attn.qkv",
            "cross_attn.q_linear",
            "cross_attn.v_linear",
        ],
    }
    (adapter_dir / "adapter_config.json").write_text(
        json.dumps(config, indent=2),
        encoding="utf-8",
    )

    metadata = {
        "method": method,
        "train_steps": int(global_step),
        "selected_block_count": len(selected_blocks),
        "lora_module_count": len(b),
        "nonzero_lora_module_count": sum(
            int(x.count_nonzero().item() > 0)
            for x in b.values()
        ),
        "trainable_lora_params": stored_parameters,
        "nonfinite_values": nonfinite,
        "final_loss": float(final_loss),
        "peak_cuda_mem_mb": (
            torch.cuda.max_memory_allocated() / 2**20
        ),
        "adapter_path": str(adapter_path),
        "adapter_size_mb": adapter_path.stat().st_size / 2**20,
    }
    (output_dir / "lora_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    return metadata


def load_vosr_lora_adapter(model, adapter_path):
    state = load_file(str(adapter_path))
    result = model.load_state_dict(state, strict=False)
    if result.unexpected_keys:
        raise RuntimeError(
            f"Unexpected adapter keys: {result.unexpected_keys[:10]}"
        )
    return len(state)
