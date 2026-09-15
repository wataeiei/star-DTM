#!/usr/bin/env python3
"""Select DiT-SR LoRA blocks using noise-normalized calibration importance."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    if not rows:
        raise ValueError(f"Empty importance CSV: {path}")
    return rows


def make_selection(rows, source, threshold=1.0, step=0, expected_blocks=58,
                   score_key="normalized_grad_score", noise_weights=None):
    if not math.isfinite(threshold) or threshold < 0:
        raise ValueError("Importance threshold must be finite and non-negative")
    required = {"train_step", "noise_ratio", "block", "block_index",
                "lora_param_count", "module_count", score_key}
    if not rows or required - set(rows[0]):
        raise ValueError(f"Missing importance columns: {sorted(required - set(rows[0] if rows else {}))}")
    for field in ("loss_mode", "target", "rank", "alpha"):
        if field not in source:
            raise ValueError(f"Source metadata lacks {field}")
    if source["loss_mode"] != "official":
        raise ValueError("Expected an official-objective calibration")
    if source.get("lora_selection") != "all":
        raise ValueError("Source metadata must record lora_selection=all for full-block calibration")
    if not isinstance(source["rank"], int) or source["rank"] <= 0:
        raise ValueError("Invalid LoRA rank in source metadata")

    groups = defaultdict(dict)
    for row in rows:
        if int(row["train_step"]) != step:
            continue
        noise, score = float(row["noise_ratio"]), float(row[score_key])
        if not math.isfinite(noise) or not math.isfinite(score) or score < 0:
            raise ValueError("Noise and scores must be finite; scores must be non-negative")
        if row.get("loss_mode", "official") != "official":
            raise ValueError("Importance CSV objective differs from source metadata")
        block = row["block"]
        if block in groups[noise]:
            raise ValueError(f"Duplicate block {block} at noise={noise}")
        params, modules = int(row["lora_param_count"]), int(row["module_count"])
        if params <= 0 or modules <= 0:
            raise ValueError(f"Invalid parameter/module count for {block}")
        groups[noise][block] = (int(row["block_index"]), params, modules, score)
    if not groups:
        raise ValueError(f"No importance rows at step={step}")
    anchors = sorted(groups)
    first = groups[anchors[0]]
    if expected_blocks and len(first) != expected_blocks:
        raise ValueError(f"Expected {expected_blocks} blocks, found {len(first)}")
    blocks = sorted(first, key=lambda b: first[b][0])
    if len({first[b][0] for b in blocks}) != len(blocks):
        raise ValueError("Duplicate block indices")
    for noise in anchors:
        if set(groups[noise]) != set(blocks):
            raise ValueError(f"Incomplete block coverage at noise={noise}")
        if any(groups[noise][b][:3] != first[b][:3] for b in blocks):
            raise ValueError("Block indices or parameter/module counts vary across noise anchors")
    if noise_weights is None:
        weights = {a: 1 / len(anchors) for a in anchors}
    else:
        if len(noise_weights) != len(anchors):
            raise ValueError("Provide one noise weight per sorted calibration anchor")
        if any(not math.isfinite(w) or w < 0 for w in noise_weights) or sum(noise_weights) <= 0:
            raise ValueError("Noise weights must be finite, non-negative and have a positive sum")
        weights = {a: w / sum(noise_weights) for a, w in zip(anchors, noise_weights)}

    aggregate = dict.fromkeys(blocks, 0.0)
    for noise in anchors:
        total = sum(groups[noise][b][3] for b in blocks)
        if not math.isfinite(total) or total <= 0:
            raise ValueError(f"Non-positive/non-finite score sum at noise={noise}")
        for b in blocks:
            aggregate[b] += weights[noise] * groups[noise][b][3] / total
    n = len(blocks)
    ranks = {b: i + 1 for i, b in enumerate(sorted(blocks, key=lambda b: (-aggregate[b], first[b][0])))}
    details = []
    for b in blocks:
        relative = n * aggregate[b]
        details.append({
            "block": b, "block_index": first[b][0],
            "lora_param_count": first[b][1], "module_count": first[b][2],
            "noise_weighted_score_share": aggregate[b],
            "relative_mean_importance": relative,
            "importance_rank": ranks[b], "selected": relative >= threshold,
        })
    selected = [row["block"] for row in details if row["selected"]]
    if not selected:
        raise ValueError("Threshold selected zero blocks; no automatic Top-K fallback was applied")
    total_params = sum(row["lora_param_count"] for row in details)
    selected_params = sum(row["lora_param_count"] for row in details if row["selected"])
    metadata = {key: source[key] for key in (
        "config_path", "ckpt_path", "autoencoder_ckpt", "data_dir", "image_size",
        "lq_size", "target", "rank", "alpha", "loss_mode", "block_regex",
    ) if key in source}
    metadata.update({
        "selection_policy": "noise-normalized-importance-threshold",
        "importance_step": step, "score_key": score_key,
        "importance_threshold": threshold, "noise_anchors": anchors,
        "noise_probabilities": weights, "candidate_block_count": n,
        # The trainer requires this field; it stores the resulting count, not a preset budget.
        "topk_blocks": len(selected), "selected_blocks": selected,
        "selected_block_fraction": len(selected) / n,
        "all_candidate_lora_params": total_params,
        "selected_lora_params": selected_params,
        "selected_lora_parameter_fraction": selected_params / total_params,
        "selected_lora_module_count": sum(row["module_count"] for row in details if row["selected"]),
        "all_blocks_selected": len(selected) == n,
        "normalization": "Per-noise shares over all candidate blocks, weighted across noise anchors, multiplied by block count",
    })
    return metadata, details


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--importance_csv", required=True)
    parser.add_argument("--source_metadata", required=True)
    parser.add_argument("--importance_step", type=int, default=0)
    parser.add_argument("--importance_threshold", type=float, default=1.0)
    parser.add_argument("--expected_blocks", type=int, default=58)
    parser.add_argument("--score_key", default="normalized_grad_score")
    parser.add_argument("--noise_weights", type=float, nargs="+", help="Weights in sorted noise-anchor order; default is uniform")
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()
    out = Path(args.output_dir)
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        parser.error("Output directory is not empty; choose a new directory to preserve previous policies")
    try:
        source = json.loads(Path(args.source_metadata).read_text(encoding="utf-8-sig"))
        metadata, details = make_selection(
            read_csv(args.importance_csv), source, args.importance_threshold,
            args.importance_step, args.expected_blocks, args.score_key, args.noise_weights,
        )
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.error(str(exc))
    metadata.update(importance_csv=str(args.importance_csv), source_metadata=str(args.source_metadata))
    out.mkdir(parents=True, exist_ok=True)
    with (out / "lora_selection_scores.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(details[0]))
        writer.writeheader()
        writer.writerows(details)
    (out / "dit_sr_grad_metadata.json").write_text(json.dumps(metadata, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(metadata, indent=2, allow_nan=False))
    if metadata["all_blocks_selected"]:
        print("WARNING: this threshold selected every block, so this is not sparse LoRA.")


if __name__ == "__main__":
    main()
