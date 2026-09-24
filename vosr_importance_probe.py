import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import torch

from vosr_lora_adapter import VOSRLoRALinear


DEFAULT_FLOW_ANCHORS = (0.05, 0.2, 0.4, 0.6, 0.8, 0.95)
TARGET_SUFFIXES = (
    "attn.qkv",
    "cross_attn.q_linear",
    "cross_attn.v_linear",
)


def configure_importance_probe(args):
    """Normalize probe settings and make the run update-free."""
    enabled = bool(getattr(args, "importance_probe_only", False))
    if not enabled:
        return None

    anchors = tuple(
        float(value)
        for value in getattr(
            args,
            "importance_noise_ratios",
            DEFAULT_FLOW_ANCHORS,
        )
    )
    batches_per_anchor = int(
        getattr(args, "importance_batches_per_anchor", 4)
    )
    if not anchors:
        raise ValueError("importance_noise_ratios must not be empty")
    if len(set(anchors)) != len(anchors):
        raise ValueError("importance_noise_ratios contains duplicates")
    if any(value < 0.0 or value > 1.0 for value in anchors):
        raise ValueError("importance noise ratios must be in [0, 1]")
    if batches_per_anchor <= 0:
        raise ValueError("importance_batches_per_anchor must be positive")

    expected_steps = len(anchors) * batches_per_anchor
    args.max_train_steps = expected_steps
    args.gradient_accumulation_steps = 1
    args.use_ema = False
    args.report_to = None
    args.resume_ckpt = None

    return {
        "anchors": anchors,
        "batches_per_anchor": batches_per_anchor,
        "expected_steps": expected_steps,
    }


def flow_anchor_for_step(probe_config, step):
    anchors = probe_config["anchors"]
    return anchors[int(step) % len(anchors)]


def _block_lora_modules(block):
    return (
        block.attn.qkv,
        block.cross_attn.q_linear,
        block.cross_attn.v_linear,
    )


def capture_block_gradients(
    model,
    probe_step,
    flow_t,
    loss,
    num_anchors=len(DEFAULT_FLOW_ANCHORS),
):
    """Capture one post-backward gradient norm for every VOSR block."""
    rows = []
    for block_index, block in enumerate(model.blocks):
        modules = _block_lora_modules(block)
        invalid = [
            TARGET_SUFFIXES[index]
            for index, module in enumerate(modules)
            if not isinstance(module, VOSRLoRALinear)
        ]
        if invalid:
            raise RuntimeError(
                f"blocks.{block_index} has unwrapped LoRA targets: {invalid}"
            )

        squared_norm = 0.0
        parameter_count = 0
        tensor_count = 0
        for module in modules:
            for parameter in (
                module.lora_A.weight,
                module.lora_B.weight,
            ):
                if parameter.grad is None:
                    raise RuntimeError(
                        "Missing LoRA gradient at "
                        f"blocks.{block_index} on probe step {probe_step}"
                    )
                gradient = parameter.grad.detach().float()
                squared_norm += gradient.square().sum().item()
                parameter_count += parameter.numel()
                tensor_count += 1

        grad_norm = math.sqrt(squared_norm)
        normalized = grad_norm / math.sqrt(parameter_count)
        row = {
            "probe_step": int(probe_step),
            "anchor_probe_index": (
                int(probe_step) // int(num_anchors) + 1
            ),
            "flow_t": float(flow_t),
            "noise_ratio": float(flow_t),
            "block": f"blocks.{block_index}",
            "block_index": block_index,
            "grad_norm": grad_norm,
            "normalized_grad_score": normalized,
            "parameter_count": parameter_count,
            "gradient_tensor_count": tensor_count,
            "probe_loss": float(loss),
        }
        if not all(
            math.isfinite(float(row[key]))
            for key in (
                "grad_norm",
                "normalized_grad_score",
                "probe_loss",
            )
        ):
            raise RuntimeError(
                f"Non-finite importance value: {row}"
            )
        rows.append(row)
    return rows


def _mean(values):
    return sum(values) / len(values)


def _write_csv(path, rows, fieldnames):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _aggregate_rows(raw_rows, probe_config, expected_blocks):
    grouped = defaultdict(list)
    for row in raw_rows:
        grouped[(row["flow_t"], row["block_index"])].append(row)

    aggregate = []
    for flow_t in probe_config["anchors"]:
        anchor_rows = []
        for block_index in range(expected_blocks):
            samples = grouped[(flow_t, block_index)]
            if len(samples) != probe_config["batches_per_anchor"]:
                raise RuntimeError(
                    f"Expected {probe_config['batches_per_anchor']} samples "
                    f"for flow_t={flow_t}, blocks.{block_index}; "
                    f"found {len(samples)}"
                )
            anchor_rows.append(
                {
                    "train_step": 0,
                    "flow_t": flow_t,
                    "noise_ratio": flow_t,
                    "block": f"blocks.{block_index}",
                    "block_index": block_index,
                    "grad_norm": _mean(
                        [row["grad_norm"] for row in samples]
                    ),
                    "normalized_grad_score": _mean(
                        [
                            row["normalized_grad_score"]
                            for row in samples
                        ]
                    ),
                    "parameter_count": samples[0]["parameter_count"],
                    "mean_probe_loss": _mean(
                        [row["probe_loss"] for row in samples]
                    ),
                    "probe_batches": len(samples),
                }
            )

        denominator = sum(
            row["normalized_grad_score"] for row in anchor_rows
        )
        if denominator <= 0.0 or not math.isfinite(denominator):
            raise RuntimeError(
                f"Invalid score denominator for flow_t={flow_t}: "
                f"{denominator}"
            )
        ranked = sorted(
            anchor_rows,
            key=lambda row: (
                -row["normalized_grad_score"],
                row["block_index"],
            ),
        )
        for rank, row in enumerate(ranked, start=1):
            row["score_share"] = (
                row["normalized_grad_score"] / denominator
            )
            row["importance_rank"] = rank
        aggregate.extend(sorted(ranked, key=lambda row: row["block_index"]))
    return aggregate


def _summarize_rows(aggregate, expected_blocks):
    by_block = defaultdict(list)
    for row in aggregate:
        by_block[row["block_index"]].append(row)

    summary = []
    for block_index in range(expected_blocks):
        rows = by_block[block_index]
        summary.append(
            {
                "block": f"blocks.{block_index}",
                "block_index": block_index,
                "mean_normalized_grad_score": _mean(
                    [row["normalized_grad_score"] for row in rows]
                ),
                "mean_score_share": _mean(
                    [row["score_share"] for row in rows]
                ),
                "mean_rank": _mean(
                    [row["importance_rank"] for row in rows]
                ),
            }
        )

    summary.sort(
        key=lambda row: (-row["mean_score_share"], row["block_index"])
    )
    cumulative = 0.0
    for rank, row in enumerate(summary, start=1):
        cumulative += row["mean_score_share"]
        row["overall_rank"] = rank
        row["cumulative_utility"] = cumulative
    return summary


def write_importance_reports(
    output_dir,
    raw_rows,
    probe_config,
    expected_blocks=28,
    metadata=None,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    expected_raw = (
        expected_blocks
        * len(probe_config["anchors"])
        * probe_config["batches_per_anchor"]
    )
    if len(raw_rows) != expected_raw:
        raise RuntimeError(
            f"Expected {expected_raw} raw importance rows, "
            f"found {len(raw_rows)}"
        )

    aggregate = _aggregate_rows(
        raw_rows,
        probe_config,
        expected_blocks,
    )
    expected_aggregate = expected_blocks * len(probe_config["anchors"])
    if len(aggregate) != expected_aggregate:
        raise RuntimeError(
            f"Expected {expected_aggregate} aggregate rows, "
            f"found {len(aggregate)}"
        )
    summary = _summarize_rows(aggregate, expected_blocks)

    raw_path = output_dir / "lora_importance_probe_batches.csv"
    aggregate_path = output_dir / "lora_importance_evolution.csv"
    summary_path = output_dir / "lora_importance_summary.csv"
    metadata_path = output_dir / "importance_metadata.json"

    _write_csv(raw_path, raw_rows, list(raw_rows[0]))
    _write_csv(aggregate_path, aggregate, list(aggregate[0]))
    _write_csv(summary_path, summary, list(summary[0]))

    report = {
        "model": "VOSR-0.5B-ms",
        "method": "All-LoRA importance probe",
        "importance_probe_only": True,
        "flow_anchors": list(probe_config["anchors"]),
        "probe_batches_per_anchor": probe_config["batches_per_anchor"],
        "backward_probes": probe_config["expected_steps"],
        "optimizer_steps": 0,
        "candidate_block_count": expected_blocks,
        "lora_module_count": expected_blocks * 3,
        "raw_row_count": len(raw_rows),
        "aggregate_row_count": len(aggregate),
        "nonfinite_values": 0,
        "score_normalization": "grad_norm / sqrt(block_lora_parameter_count)",
        "importance_csv": str(aggregate_path),
        "summary_csv": str(summary_path),
    }
    if metadata:
        report.update(metadata)
    metadata_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )
    return report
