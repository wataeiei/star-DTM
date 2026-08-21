#!/usr/bin/env python3
"""Noise-conditioned parallel adapters for the official DiT4SR transformer."""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F


CHECKPOINT_FORMAT = "dit4sr_noise_parallel_adapter_v1"


def _natural_key(text: str) -> list[int | str]:
    parts: list[int | str] = []
    token = ""
    numeric = False
    for char in text:
        if char.isdigit() != numeric and token:
            parts.append(int(token) if numeric else token)
            token = ""
        token += char
        numeric = char.isdigit()
    if token:
        parts.append(int(token) if numeric else token)
    return parts


def transformer_block_items(transformer: nn.Module) -> list[tuple[str, nn.Module]]:
    blocks = getattr(transformer, "transformer_blocks", None)
    if blocks is None:
        raise ValueError("The transformer has no transformer_blocks container.")
    return [(f"transformer_blocks.{index}", block) for index, block in enumerate(blocks)]


def infer_hidden_dim(transformer: nn.Module) -> int:
    value = getattr(transformer, "inner_dim", None)
    if isinstance(value, int) and value > 0:
        return value
    config = getattr(transformer, "config", None)
    heads = getattr(config, "num_attention_heads", None)
    head_dim = getattr(config, "attention_head_dim", None)
    if isinstance(heads, int) and isinstance(head_dim, int):
        return heads * head_dim
    for name, module in transformer.named_modules():
        if isinstance(module, nn.Linear) and name.endswith("attn.to_q"):
            return int(module.in_features)
    raise ValueError("Could not infer the DiT hidden dimension.")


def output_layout(transformer: nn.Module) -> tuple[int, int]:
    config = getattr(transformer, "config", None)
    out_channels = int(getattr(config, "out_channels", 16))
    patch_size = getattr(config, "patch_size", 2)
    if isinstance(patch_size, (tuple, list)):
        if len(patch_size) != 2 or patch_size[0] != patch_size[1]:
            raise ValueError(f"Only square patch sizes are supported, got {patch_size}.")
        patch_size = patch_size[0]
    return out_channels, int(patch_size)


def _tensor_candidates(value) -> Iterable[torch.Tensor]:
    if torch.is_tensor(value):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _tensor_candidates(item)
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from _tensor_candidates(item)


def image_tokens(block_output, hidden_dim: int) -> torch.Tensor:
    candidates = [
        tensor
        for tensor in _tensor_candidates(block_output)
        if tensor.ndim == 3 and tensor.shape[-1] == hidden_dim
    ]
    if not candidates:
        shapes = [tuple(tensor.shape) for tensor in _tensor_candidates(block_output)]
        raise RuntimeError(
            f"Could not find [batch, tokens, {hidden_dim}] image tokens in block output {shapes}."
        )
    tokens = max(candidates, key=lambda tensor: tensor.shape[1])
    # Official DiT4SR concatenates noisy-image and LR-control tokens, then
    # splits them in half after the last transformer block. The first half is
    # the image stream consumed by the output head.
    if tokens.shape[1] % 2 == 0:
        image_count = tokens.shape[1] // 2
        side = int(round(math.sqrt(image_count)))
        if side * side == image_count:
            return tokens[:, :image_count]
    return tokens


def noise_features(noise_ratio: float, batch: int, device: torch.device) -> torch.Tensor:
    sigma = torch.full((batch, 1), float(noise_ratio), device=device, dtype=torch.float32)
    return torch.cat(
        [sigma, sigma.square(), torch.sin(math.pi * sigma), torch.cos(math.pi * sigma)],
        dim=1,
    )


class ParallelAdapterLayer(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        side_dim: int,
        mlp_ratio: float,
        use_depthwise_conv: bool,
    ) -> None:
        super().__init__()
        mlp_dim = max(side_dim, int(round(side_dim * mlp_ratio)))
        self.down = nn.Linear(hidden_dim, side_dim)
        self.noise_projection = nn.Sequential(
            nn.Linear(4, side_dim),
            nn.SiLU(),
            nn.Linear(side_dim, side_dim),
        )
        self.norm = nn.LayerNorm(side_dim)
        self.mlp = nn.Sequential(
            nn.Linear(side_dim, mlp_dim),
            nn.SiLU(),
            nn.Linear(mlp_dim, side_dim),
        )
        self.depthwise = (
            nn.Conv2d(side_dim, side_dim, 3, padding=1, groups=side_dim)
            if use_depthwise_conv
            else None
        )
        self.mix_logit = nn.Parameter(torch.zeros(()))

    def forward(
        self,
        backbone_tokens: torch.Tensor,
        previous: torch.Tensor | None,
        noise_ratio: float,
    ) -> torch.Tensor:
        backbone = self.down(backbone_tokens.float())
        noise = self.noise_projection(
            noise_features(noise_ratio, backbone.shape[0], backbone.device)
        ).unsqueeze(1)
        if previous is None:
            fused = backbone + noise
        else:
            mix = torch.sigmoid(self.mix_logit)
            fused = mix * backbone + (1.0 - mix) * previous + noise
        update = self.mlp(self.norm(fused))
        if self.depthwise is not None:
            side = int(round(math.sqrt(update.shape[1])))
            if side * side == update.shape[1]:
                spatial = update.transpose(1, 2).reshape(
                    update.shape[0], update.shape[2], side, side
                )
                update = update + self.depthwise(spatial).flatten(2).transpose(1, 2)
        return fused + update


def _normalised_importance_rows(
    path: str | Path,
    block_names: list[str],
    importance_train_step: int,
) -> list[dict]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    required = {"block", "noise_ratio", "normalized_grad_score"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"{path} must contain columns {sorted(required)}.")
    known = set(block_names)
    rows = [row for row in rows if row["block"] in known]
    if importance_train_step >= 0:
        available = sorted({int(float(row.get("train_step", 0))) for row in rows})
        if importance_train_step not in available:
            raise ValueError(
                f"Importance step {importance_train_step} is unavailable; found {available}."
            )
        rows = [
            row
            for row in rows
            if int(float(row.get("train_step", 0))) == importance_train_step
        ]

    groups: dict[tuple[int, float], list[dict]] = {}
    for row in rows:
        key = (int(float(row.get("train_step", 0))), float(row["noise_ratio"]))
        groups.setdefault(key, []).append(row)
    output = []
    for group in groups.values():
        maximum = max(float(row["normalized_grad_score"]) for row in group)
        for row in group:
            output.append(
                {
                    "block": row["block"],
                    "noise_ratio": float(row["noise_ratio"]),
                    "score": float(row["normalized_grad_score"]) / max(maximum, 1e-12),
                }
            )
    return output


def build_noise_schedule(
    importance_csv: str | Path,
    block_names: list[str],
    active_topk: int,
    anchor_count: int,
    importance_train_step: int = -1,
) -> tuple[list[str], dict[float, list[str]], dict[float, dict[str, float]]]:
    if not 1 <= active_topk <= len(block_names):
        raise ValueError(f"active_topk must be in [1, {len(block_names)}].")
    if not 0 <= anchor_count <= active_topk:
        raise ValueError("anchor_count must be between 0 and active_topk.")
    rows = _normalised_importance_rows(
        importance_csv, block_names, importance_train_step
    )
    if not rows:
        raise ValueError("No matching importance rows were found.")

    by_noise: dict[float, dict[str, list[float]]] = {}
    global_scores: dict[str, list[float]] = {block: [] for block in block_names}
    for row in rows:
        by_noise.setdefault(row["noise_ratio"], {}).setdefault(row["block"], []).append(
            row["score"]
        )
        global_scores[row["block"]].append(row["score"])
    mean_global = {
        block: sum(values) / len(values) if values else 0.0
        for block, values in global_scores.items()
    }
    anchors = sorted(
        block_names, key=lambda block: (-mean_global[block], _natural_key(block))
    )[:anchor_count]

    schedule: dict[float, list[str]] = {}
    score_table: dict[float, dict[str, float]] = {}
    for noise_ratio in sorted(by_noise):
        scores = {
            block: sum(by_noise[noise_ratio].get(block, [0.0]))
            / len(by_noise[noise_ratio].get(block, [0.0]))
            for block in block_names
        }
        ranked = sorted(
            (block for block in block_names if block not in anchors),
            key=lambda block: (-scores[block], _natural_key(block)),
        )
        selected = set(anchors + ranked[: active_topk - len(anchors)])
        schedule[noise_ratio] = [block for block in block_names if block in selected]
        score_table[noise_ratio] = scores
    return anchors, schedule, score_table


class NoiseConditionedParallelAdapter(nn.Module):
    """Parallel side network that reads detached DiT block activations."""

    def __init__(
        self,
        transformer: nn.Module,
        side_dim: int,
        active_schedule: dict[float, list[str]],
        mlp_ratio: float = 2.0,
        use_depthwise_conv: bool = True,
    ) -> None:
        super().__init__()
        self.block_items = transformer_block_items(transformer)
        self.block_names = [name for name, _block in self.block_items]
        self.hidden_dim = infer_hidden_dim(transformer)
        self.out_channels, self.patch_size = output_layout(transformer)
        self.side_dim = int(side_dim)
        self.mlp_ratio = float(mlp_ratio)
        self.use_depthwise_conv = bool(use_depthwise_conv)
        self.layers = nn.ModuleDict(
            {
                name.replace(".", "_"): ParallelAdapterLayer(
                    self.hidden_dim,
                    self.side_dim,
                    self.mlp_ratio,
                    self.use_depthwise_conv,
                )
                for name in self.block_names
            }
        )
        self.output_head = nn.Linear(
            self.side_dim,
            self.out_channels * self.patch_size * self.patch_size,
        )
        nn.init.zeros_(self.output_head.weight)
        nn.init.zeros_(self.output_head.bias)
        self.active_schedule = {
            float(noise): list(blocks) for noise, blocks in active_schedule.items()
        }
        self._state: torch.Tensor | None = None
        self._noise_ratio = 0.0
        self._active: set[str] = set()
        self._enable_grad = False
        self._seen: list[str] = []
        self._handles = [
            block.register_forward_hook(self._make_hook(name))
            for name, block in self.block_items
        ]

    def _make_hook(self, block_name: str):
        def hook(_module, _inputs, output):
            if block_name not in self._active:
                return output
            tokens = image_tokens(output, self.hidden_dim).detach()
            context = torch.enable_grad() if self._enable_grad else torch.no_grad()
            with context:
                layer = self.layers[block_name.replace(".", "_")]
                self._state = layer(tokens, self._state, self._noise_ratio)
            self._seen.append(block_name)
            return output

        return hook

    def nearest_noise(self, noise_ratio: float) -> float:
        if not self.active_schedule:
            raise RuntimeError("The Parallel Adapter has no noise schedule.")
        return min(self.active_schedule, key=lambda value: abs(value - float(noise_ratio)))

    def begin(self, noise_ratio: float, enable_grad: bool) -> list[str]:
        selected_noise = self.nearest_noise(noise_ratio)
        self._noise_ratio = float(noise_ratio)
        self._active = set(self.active_schedule[selected_noise])
        self._state = None
        self._enable_grad = bool(enable_grad)
        self._seen = []
        return [name for name in self.block_names if name in self._active]

    def correction(self, base_prediction: torch.Tensor) -> torch.Tensor:
        if self._state is None:
            raise RuntimeError("No Parallel Adapter layer ran during the transformer forward.")
        if set(self._seen) != self._active:
            missing = sorted(self._active - set(self._seen), key=_natural_key)
            raise RuntimeError(f"Parallel Adapter hooks did not run for: {missing}")
        tokens = self.output_head(self._state)
        batch, token_count, _channels = tokens.shape
        grid = int(round(math.sqrt(token_count)))
        if grid * grid != token_count:
            raise RuntimeError(f"Image token count {token_count} is not a square grid.")
        patch = self.patch_size
        channels = self.out_channels
        correction = tokens.view(batch, grid, grid, patch, patch, channels)
        correction = correction.permute(0, 5, 1, 3, 2, 4).reshape(
            batch, channels, grid * patch, grid * patch
        )
        if correction.shape[-2:] != base_prediction.shape[-2:]:
            correction = F.interpolate(
                correction,
                size=base_prediction.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        if correction.shape[1] != base_prediction.shape[1]:
            raise RuntimeError(
                f"Parallel head has {correction.shape[1]} channels but base output has "
                f"{base_prediction.shape[1]}."
            )
        return correction.to(dtype=base_prediction.dtype)

    def forward_prediction(
        self,
        transformer_forward_fn,
        noise_ratio: float,
        enable_grad: bool,
    ) -> tuple[torch.Tensor, list[str]]:
        active = self.begin(noise_ratio, enable_grad=enable_grad)
        with torch.no_grad():
            base = transformer_forward_fn().detach()
        correction = self.correction(base)
        return base + correction, active

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles = []


def save_parallel_adapter(
    adapter: NoiseConditionedParallelAdapter,
    path: str | Path,
    anchors: list[str],
    score_table: dict[float, dict[str, float]],
) -> dict:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": CHECKPOINT_FORMAT,
        "config": {
            "side_dim": adapter.side_dim,
            "mlp_ratio": adapter.mlp_ratio,
            "use_depthwise_conv": adapter.use_depthwise_conv,
            "hidden_dim": adapter.hidden_dim,
            "out_channels": adapter.out_channels,
            "patch_size": adapter.patch_size,
            "block_names": adapter.block_names,
            "active_schedule": adapter.active_schedule,
            "anchors": anchors,
            "score_table": score_table,
        },
        "state_dict": {
            name: tensor.detach().cpu() for name, tensor in adapter.state_dict().items()
        },
    }
    torch.save(payload, path)
    return {
        "adapter_path": str(path),
        "adapter_size_mb": path.stat().st_size / (1024.0**2),
        "trainable_params": sum(param.numel() for param in adapter.parameters()),
    }


def load_parallel_adapter(
    transformer: nn.Module,
    path: str | Path,
    device: torch.device,
) -> tuple[NoiseConditionedParallelAdapter, dict]:
    payload = torch.load(path, map_location="cpu")
    if payload.get("format") != CHECKPOINT_FORMAT:
        raise ValueError(f"Unsupported Parallel Adapter checkpoint: {path}")
    config = payload["config"]
    adapter = NoiseConditionedParallelAdapter(
        transformer,
        side_dim=int(config["side_dim"]),
        active_schedule={float(key): value for key, value in config["active_schedule"].items()},
        mlp_ratio=float(config["mlp_ratio"]),
        use_depthwise_conv=bool(config["use_depthwise_conv"]),
    ).to(device=device, dtype=torch.float32)
    if adapter.block_names != list(config["block_names"]):
        adapter.close()
        raise ValueError("Parallel Adapter block layout does not match the transformer.")
    report = adapter.load_state_dict(payload["state_dict"], strict=True)
    adapter.eval()
    return adapter, {
        "missing": list(report.missing_keys),
        "unexpected": list(report.unexpected_keys),
        "anchors": list(config.get("anchors", [])),
        "active_schedule": config["active_schedule"],
    }
