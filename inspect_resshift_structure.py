#!/usr/bin/env python3
"""Inspect an official ResShift model for LoRA and backward-bypass integration."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def stage_from_name(name: str) -> str:
    if name.startswith("input_blocks."):
        return "input"
    if name.startswith("middle_block."):
        return "middle"
    if name.startswith("output_blocks."):
        return "output"
    return "other"


def block_from_target(name: str) -> str:
    marker = ".attn.qkv"
    return name[: name.index(marker)] if marker in name else ""


def load_checkpoint(model, checkpoint: Path, torch) -> tuple[list[str], list[str]]:
    payload = torch.load(checkpoint, map_location="cpu")
    state = payload.get("state_dict", payload) if isinstance(payload, dict) else payload
    if not isinstance(state, dict):
        raise ValueError(f"Unsupported checkpoint payload: {checkpoint}")
    cleaned = {}
    for name, value in state.items():
        while name.startswith("module."):
            name = name[len("module.") :]
        cleaned[name] = value
    status = model.load_state_dict(cleaned, strict=False)
    return list(status.missing_keys), list(status.unexpected_keys)


def inspect_model(model, torch) -> tuple[list[dict], list[dict], list[dict]]:
    blocks = []
    targets = []
    boundaries = []
    block_names = set()

    for name, module in model.named_modules():
        class_name = type(module).__name__
        if class_name == "SwinTransformerBlock":
            block_names.add(name)
            resolution = getattr(module, "input_resolution", "")
            if isinstance(resolution, (tuple, list)):
                resolution = "x".join(str(value) for value in resolution)
            blocks.append(
                {
                    "block": name,
                    "stage": stage_from_name(name),
                    "class": class_name,
                    "input_resolution": resolution,
                    "dim": getattr(module, "dim", ""),
                    "num_heads": getattr(getattr(module, "attn", None), "num_heads", ""),
                    "shift_size": getattr(module, "shift_size", ""),
                }
            )
        if class_name in {"Downsample", "Upsample", "ResBlock"}:
            changes_shape = class_name in {"Downsample", "Upsample"} or bool(
                getattr(module, "updown", False)
            )
            if changes_shape:
                boundaries.append(
                    {
                        "module": name,
                        "stage": stage_from_name(name),
                        "class": class_name,
                        "protection_reason": "changes spatial resolution or residual shape",
                    }
                )

    modules = dict(model.named_modules())
    for name, module in modules.items():
        if not isinstance(module, torch.nn.Linear) or not name.endswith(".attn.qkv"):
            continue
        block = block_from_target(name)
        targets.append(
            {
                "module": name,
                "block": block,
                "stage": stage_from_name(block),
                "in_features": module.in_features,
                "out_features": module.out_features,
                "packed_qkv": module.out_features == 3 * module.in_features,
                "candidate_block_found": block in block_names,
            }
        )

    blocks.sort(key=lambda row: row["block"])
    targets.sort(key=lambda row: row["module"])
    boundaries.sort(key=lambda row: row["module"])
    for index, row in enumerate(blocks):
        row["block_index"] = index
    return blocks, targets, boundaries


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resshift_root", required=True)
    parser.add_argument("--config_path", required=True)
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    root = Path(args.resshift_root).resolve()
    config_path = Path(args.config_path)
    if not config_path.is_absolute():
        config_path = root / config_path
    if not (root / "models").is_dir() or not (root / "utils").is_dir():
        parser.error(f"Not an official ResShift repository: {root}")
    if not config_path.is_file():
        parser.error(f"Missing ResShift config: {config_path}")
    sys.path.insert(0, str(root))

    try:
        import torch
        from omegaconf import OmegaConf
        from utils.util_common import get_obj_from_str
    except ImportError as exc:
        parser.error(f"ResShift environment is incomplete: {exc}")

    config = OmegaConf.load(config_path)
    model = get_obj_from_str(config.model.target)(**config.model.get("params", {}))
    missing = []
    unexpected = []
    checkpoint_path = None
    if args.checkpoint:
        checkpoint_path = Path(args.checkpoint)
        if not checkpoint_path.is_absolute():
            checkpoint_path = root / checkpoint_path
        if not checkpoint_path.is_file():
            parser.error(f"Missing checkpoint: {checkpoint_path}")
        try:
            missing, unexpected = load_checkpoint(model, checkpoint_path, torch)
        except (OSError, RuntimeError, ValueError) as exc:
            parser.error(str(exc))

    blocks, targets, boundaries = inspect_model(model, torch)
    if not blocks:
        parser.error("No SwinTransformerBlock modules were found")
    if not targets:
        parser.error("No packed Swin attention qkv Linear modules were found")
    unmatched = [row["module"] for row in targets if not row["candidate_block_found"]]
    if unmatched:
        parser.error(f"Could not map qkv targets to Swin blocks: {unmatched[:5]}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "resshift_candidate_blocks.csv", blocks)
    write_csv(output_dir / "resshift_lora_targets.csv", targets)
    write_csv(output_dir / "resshift_structural_boundaries.csv", boundaries)
    summary = {
        "resshift_root": str(root),
        "config_path": str(config_path),
        "model_target": str(config.model.target),
        "checkpoint": str(checkpoint_path) if checkpoint_path else None,
        "checkpoint_loaded": checkpoint_path is not None,
        "checkpoint_missing_keys": missing,
        "checkpoint_unexpected_keys": unexpected,
        "candidate_swin_blocks": len(blocks),
        "packed_qkv_lora_modules": len(targets),
        "structural_boundary_modules": len(boundaries),
        "recommended_lora_target": "packed Swin *.attn.qkv Linear modules",
        "recommended_bypass_unit": "SwinTransformerBlock",
        "note": (
            "ResShift packs query, key, and value into one qkv projection. A selective "
            "q/v-only adapter therefore requires a sliced packed-qkv LoRA wrapper; the "
            "first compatibility experiment should adapt the complete qkv projection."
        ),
    }
    (output_dir / "resshift_structure_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    print(json.dumps(summary, indent=2))
    print("\nCandidate Swin blocks:")
    for row in blocks:
        print(f"  {row['block']}")
    print(f"\nWrote ResShift structure audit to {output_dir}")


if __name__ == "__main__":
    main()
