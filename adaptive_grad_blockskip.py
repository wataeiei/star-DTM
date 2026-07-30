#!/usr/bin/env python3
"""Shared utilities for stage/noise-aware Grad-BlockSkip LoRA experiments."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F


def dynamic_patch_batch(
    batch: dict,
    noise_ratio: float,
    min_fraction: float,
    max_fraction: float,
) -> tuple[dict, int]:
    """Crop a noise-conditioned HR patch and resize it to the original tensor size."""
    image = batch["image"]
    height, width = image.shape[-2:]
    fraction = min_fraction + float(noise_ratio) * (max_fraction - min_fraction)
    fraction = max(0.0, min(1.0, fraction))
    crop_size = max(1, min(height, width, int(round(min(height, width) * fraction))))
    if crop_size >= min(height, width):
        return batch, min(height, width)

    top = int(torch.randint(0, height - crop_size + 1, (1,)).item())
    left = int(torch.randint(0, width - crop_size + 1, (1,)).item())
    cropped = image[..., top : top + crop_size, left : left + crop_size]
    resized = F.interpolate(
        cropped,
        size=(height, width),
        mode="bicubic",
        align_corners=False,
    )
    output = dict(batch)
    output["image"] = resized
    return output, crop_size


def _score_rows(
    rows: Iterable[dict],
    train_step: int,
    noise_ratio: float,
) -> list[dict]:
    steps = sorted({int(row["train_step"]) for row in rows if int(row["train_step"]) <= train_step})
    if not steps:
        raise ValueError("No importance rows are available at or before this training step.")
    selected_step = steps[-1]
    ratios = sorted(
        {
            float(row["noise_ratio"])
            for row in rows
            if int(row["train_step"]) == selected_step
        }
    )
    selected_ratio = min(ratios, key=lambda value: abs(value - float(noise_ratio)))
    return sorted(
        [
            row
            for row in rows
            if int(row["train_step"]) == selected_step
            and math.isclose(float(row["noise_ratio"]), selected_ratio, abs_tol=1e-8)
        ],
        key=lambda row: int(row["block_index"]),
    )


def select_low_score_runs(
    rows: Iterable[dict],
    train_step: int,
    noise_ratio: float,
    skip_count: int,
    min_run: int,
    max_run: int,
    max_runs: int,
    score_key: str = "normalized_grad_score",
) -> list[str]:
    """Select exactly ``skip_count`` low-score blocks in a few contiguous runs."""
    ordered = _score_rows(rows, train_step, noise_ratio)
    count = min(max(int(skip_count), 0), len(ordered))
    if count == 0:
        return []
    min_run = max(1, int(min_run))
    max_run = max(min_run, int(max_run))
    max_runs = max(1, int(max_runs))

    def group_name(block: str) -> str:
        parts = block.split(".")
        if parts[0] in {"input_blocks", "output_blocks"} and len(parts) > 1:
            return ".".join(parts[:2])
        if parts[0] == "middle_block":
            return "middle_block"
        return parts[0]

    # State: (number skipped, completed/active runs, current run length).
    states: dict[tuple[int, int, int], tuple[float, tuple[int, ...]]] = {
        (0, 0, 0): (0.0, ())
    }
    for index, row in enumerate(ordered):
        score = float(row[score_key])
        group_changed = (
            index > 0
            and group_name(str(row["block"])) != group_name(str(ordered[index - 1]["block"]))
        )
        next_states: dict[tuple[int, int, int], tuple[float, tuple[int, ...]]] = {}

        def update(key, value):
            previous = next_states.get(key)
            if previous is None or value[0] < previous[0]:
                next_states[key] = value

        for (used, runs, original_run_len), (cost, chosen) in states.items():
            run_len = original_run_len
            if group_changed and run_len > 0:
                if run_len < min_run:
                    continue
                run_len = 0
            if run_len == 0 or run_len >= min_run:
                update((used, runs, 0), (cost, chosen))
            if used >= count:
                continue
            if run_len > 0 and run_len < max_run:
                update(
                    (used + 1, runs, run_len + 1),
                    (cost + score, chosen + (index,)),
                )
            elif run_len == 0 and runs < max_runs:
                update(
                    (used + 1, runs + 1, 1),
                    (cost + score, chosen + (index,)),
                )
        states = next_states

    candidates = [
        value
        for (used, _runs, run_len), value in states.items()
        if used == count and (run_len == 0 or run_len >= min_run)
    ]
    if not candidates:
        raise ValueError(
            "Cannot satisfy the requested contiguous skip policy. "
            "Reduce --blockskip_count/--blockskip_min_run or increase --blockskip_max_runs."
        )
    _, chosen = min(candidates, key=lambda item: item[0])
    return [ordered[index]["block"] for index in chosen]


def get_module(root: nn.Module, dotted_name: str) -> nn.Module:
    module = root
    for part in dotted_name.split("."):
        module = module[int(part)] if part.isdigit() else getattr(module, part)
    return module


def set_module(root: nn.Module, dotted_name: str, module: nn.Module) -> None:
    parts = dotted_name.split(".")
    parent = root
    for part in parts[:-1]:
        parent = parent[int(part)] if part.isdigit() else getattr(parent, part)
    child = parts[-1]
    if child.isdigit():
        parent[int(child)] = module
    else:
        setattr(parent, child, module)


def infer_block_module_paths(
    lora_module_names: Iterable[str],
    block_names: Iterable[str],
    block_key_fn,
    block_regex: str = "",
) -> dict[str, str]:
    """Map CSV block keys to concrete module paths, including nested containers."""
    wanted = set(block_names)
    result = {}
    for module_name in lora_module_names:
        block = block_key_fn(module_name, block_regex)
        if block not in wanted or block in result:
            continue
        parts = module_name.split(".")
        key_parts = block.split(".")
        marker = key_parts[0]
        try:
            marker_index = parts.index(marker)
        except ValueError:
            continue

        if marker == "transformer_blocks":
            result[block] = ".".join(parts[: marker_index + 2])
            continue

        if marker in {"input_blocks", "output_blocks"} and "blocks" in key_parts:
            inner_index = parts.index("blocks", marker_index + 2)
            result[block] = ".".join(parts[: inner_index + 2])
            continue

        if marker == "middle_block" and "blocks" in key_parts:
            inner_index = parts.index("blocks", marker_index + 1)
            result[block] = ".".join(parts[: inner_index + 2])
            continue

        result[block] = ".".join(parts[: marker_index + 2])

    missing = sorted(wanted - set(result))
    if missing:
        raise ValueError(
            "Could not resolve model modules for importance blocks: " + ", ".join(missing)
        )
    return result


def _tensor_inputs(args: tuple, kwargs: dict) -> list[torch.Tensor]:
    refs = []
    for name in ("encoder_hidden_states", "hidden_states", "sample", "x"):
        value = kwargs.get(name)
        if torch.is_tensor(value):
            refs.append(value)
    refs.extend(value for value in args if torch.is_tensor(value))
    unique = []
    for tensor in refs:
        if not any(tensor is previous for previous in unique):
            unique.append(tensor)
    return unique


def _cache_tensor(tensor: torch.Tensor, device: str, dtype: torch.dtype) -> torch.Tensor:
    target = tensor.detach().to(dtype=dtype)
    if device == "cpu":
        target = target.cpu()
    return target


def _encode_residual(
    output: Any,
    refs: list[torch.Tensor],
    used_refs: set[int],
    cache_device: str,
    cache_dtype: torch.dtype,
) -> Any:
    if torch.is_tensor(output):
        matches = [
            index
            for index, ref in enumerate(refs)
            if index not in used_refs and ref.shape == output.shape
        ]
        if not matches:
            raise ValueError("Block output has no shape-compatible residual input.")
        index = matches[0]
        used_refs.add(index)
        # Subtract in FP32 so BF16/FP16 round-off does not accumulate across skips.
        delta = _cache_tensor(
            output.detach().float() - refs[index].detach().float(),
            cache_device,
            cache_dtype,
        )
        return ("residual", index, delta)
    if isinstance(output, tuple):
        return ("tuple", tuple(
            _encode_residual(item, refs, used_refs, cache_device, cache_dtype)
            for item in output
        ))
    if isinstance(output, list):
        return ("list", [
            _encode_residual(item, refs, used_refs, cache_device, cache_dtype)
            for item in output
        ])
    if output is None or isinstance(output, (bool, int, float, str)):
        return ("constant", output)
    raise ValueError(f"Unsupported block output type for residual replay: {type(output)}")


def _decode_residual(encoded: Any, refs: list[torch.Tensor]) -> Any:
    kind = encoded[0]
    if kind == "residual":
        ref = refs[encoded[1]]
        delta = encoded[2].to(device=ref.device, dtype=torch.float32, non_blocking=True)
        return (ref.float() + delta).to(dtype=ref.dtype)
    if kind == "tuple":
        return tuple(_decode_residual(item, refs) for item in encoded[1])
    if kind == "list":
        return [_decode_residual(item, refs) for item in encoded[1]]
    if kind == "constant":
        return encoded[1]
    raise RuntimeError(f"Unknown residual encoding: {kind}")


class ResidualBlockWrapper(nn.Module):
    def __init__(self, name: str, block: nn.Module, controller: "ResidualBlockController"):
        super().__init__()
        self.name = name
        self.block = block
        self.controller = controller
        self.cached = None
        self.cached_bytes = 0
        self.replayable = True

    def forward(self, *args, **kwargs):
        selected = self.name in self.controller.skip_blocks
        if self.controller.mode == "skip" and selected and self.cached is not None and self.replayable:
            refs = _tensor_inputs(args, kwargs)
            return _decode_residual(self.cached, refs)

        output = self.block(*args, **kwargs)
        if self.controller.mode == "cache" and selected:
            refs = _tensor_inputs(args, kwargs)
            try:
                self.cached = _encode_residual(
                    output,
                    refs,
                    set(),
                    self.controller.cache_device,
                    self.controller.cache_dtype,
                )
                self.cached_bytes = _encoded_nbytes(self.cached)
                self.replayable = True
            except ValueError as exc:
                self.cached = None
                self.cached_bytes = 0
                self.replayable = False
                self.controller.fallbacks[self.name] = str(exc)
        return output


def _encoded_nbytes(encoded: Any) -> int:
    if encoded[0] == "residual":
        tensor = encoded[2]
        return tensor.numel() * tensor.element_size()
    if encoded[0] == "constant":
        return 0
    return sum(_encoded_nbytes(item) for item in encoded[1])


@dataclass
class CacheStats:
    elapsed_s: float
    cache_mb: float
    replayable_blocks: int
    fallback_blocks: int
    teacher_loss: float = 0.0
    peak_cuda_mem_mb: float = 0.0
    fallback_names: str = ""


class ResidualBlockController:
    """Wrap model blocks and switch between full, cache, and residual-skip modes."""

    def __init__(
        self,
        root: nn.Module,
        block_paths: dict[str, str],
        cache_device: str = "cpu",
        cache_dtype: torch.dtype = torch.float16,
    ) -> None:
        self.mode = "full"
        self.skip_blocks: set[str] = set()
        self.cache_device = cache_device
        self.cache_dtype = cache_dtype
        self.fallbacks: dict[str, str] = {}
        self.wrappers: dict[str, ResidualBlockWrapper] = {}
        modules = {
            logical_name: (path, get_module(root, path))
            for logical_name, path in block_paths.items()
        }
        for logical_name, (path, module) in modules.items():
            wrapper = ResidualBlockWrapper(logical_name, module, self)
            set_module(root, path, wrapper)
            self.wrappers[logical_name] = wrapper

    def configure(self, skip_blocks: Iterable[str]) -> None:
        self.skip_blocks = set(skip_blocks)
        self.fallbacks.clear()
        for wrapper in self.wrappers.values():
            wrapper.cached = None
            wrapper.cached_bytes = 0
            wrapper.replayable = True

    def set_mode(self, mode: str) -> None:
        if mode not in {"full", "cache", "skip"}:
            raise ValueError(f"Unknown block controller mode: {mode}")
        self.mode = mode

    def stats(
        self,
        elapsed_s: float,
        teacher_loss: float = 0.0,
        peak_cuda_mem_mb: float = 0.0,
    ) -> CacheStats:
        selected = [self.wrappers[name] for name in self.skip_blocks]
        return CacheStats(
            elapsed_s=elapsed_s,
            cache_mb=sum(wrapper.cached_bytes for wrapper in selected) / (1024.0 ** 2),
            replayable_blocks=sum(wrapper.replayable and wrapper.cached is not None for wrapper in selected),
            fallback_blocks=sum(not wrapper.replayable for wrapper in selected),
            teacher_loss=teacher_loss,
            peak_cuda_mem_mb=peak_cuda_mem_mb,
            fallback_names=";".join(sorted(self.fallbacks)),
        )


def snapshot_rng(device: torch.device) -> tuple[torch.Tensor, list[torch.Tensor] | None]:
    cpu_state = torch.random.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if device.type == "cuda" else None
    return cpu_state, cuda_state


def restore_rng(state: tuple[torch.Tensor, list[torch.Tensor] | None]) -> None:
    cpu_state, cuda_state = state
    torch.random.set_rng_state(cpu_state)
    if cuda_state is not None:
        torch.cuda.set_rng_state_all(cuda_state)


def populate_online_cache(
    controller: ResidualBlockController,
    loss_fn,
    device: torch.device,
) -> CacheStats:
    """Run a deterministic no-grad teacher forward and retain only block residuals."""
    state = snapshot_rng(device)
    controller.set_mode("cache")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    with torch.no_grad():
        loss = loss_fn()
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite loss during residual-cache forward.")
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    peak_cuda_mem_mb = (
        torch.cuda.max_memory_allocated() / (1024.0 ** 2)
        if device.type == "cuda"
        else 0.0
    )
    restore_rng(state)
    controller.set_mode("skip")
    return controller.stats(
        elapsed,
        teacher_loss=float(loss.detach().cpu()),
        peak_cuda_mem_mb=peak_cuda_mem_mb,
    )
