"""Paired full-vs-bypass gradient fidelity audit for VOSR."""

from __future__ import annotations

import csv
import json
import math
import statistics
from pathlib import Path

import torch

import adaptive_grad_blockskip as adaptive


def _read_csv(path):
    with Path(path).open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path, rows):
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def configure_bypass_fidelity_probe(args):
    if not bool(getattr(args, "bypass_fidelity_probe_only", False)):
        return None

    selection_path = Path(args.lora_selection_file)
    policy_path = Path(args.bypass_threshold_policy_csv)
    if not selection_path.is_file():
        raise FileNotFoundError(f"LoRA selection file not found: {selection_path}")
    if not policy_path.is_file():
        raise FileNotFoundError(f"Threshold policy CSV not found: {policy_path}")

    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    selected = sorted(int(value) for value in selection["selected_block_indices"])
    frozen = sorted(set(range(28)) - set(selected))
    rows = _read_csv(policy_path)
    if not rows:
        raise ValueError("Threshold policy CSV is empty")

    policies = []
    seen = set()
    for row in rows:
        flow_t = float(row.get("flow_t", row.get("noise_ratio")))
        if flow_t in seen:
            raise ValueError(f"Duplicate threshold policy for flow_t={flow_t:g}")
        seen.add(flow_t)
        blocks = [value for value in row["skip_blocks"].split(";") if value]
        indices = [int(block.rsplit(".", 1)[-1]) for block in blocks]
        unknown = sorted(set(indices) - set(frozen))
        if unknown:
            raise ValueError(
                f"Threshold policy bypasses protected blocks at flow_t={flow_t:g}: "
                f"{unknown}"
            )
        declared = int(row["bypass_budget"])
        if declared != len(blocks):
            raise ValueError(
                f"Policy count mismatch at flow_t={flow_t:g}: "
                f"declared={declared}, blocks={len(blocks)}"
            )
        policies.append(
            {
                "flow_t": flow_t,
                "threshold": float(row["threshold"]),
                "bypass_budget": declared,
                "skip_blocks": blocks,
                "estimated_saved_gflops": float(row["estimated_saved_gflops"]),
            }
        )
    policies.sort(key=lambda row: row["flow_t"])

    probe_batches = int(getattr(args, "bypass_fidelity_batches", 4))
    if probe_batches <= 0:
        raise ValueError("bypass_fidelity_batches must be positive")

    args.gradient_accumulation_steps = 1
    args.use_ema = False
    args.report_to = None
    args.resume_ckpt = None
    return {
        "selection_path": str(selection_path),
        "policy_path": str(policy_path),
        "selection": selection,
        "selected_indices": selected,
        "frozen_indices": frozen,
        "policies": policies,
        "probe_batches": probe_batches,
        "min_cosine": float(getattr(args, "bypass_fidelity_min_cosine", 0.90)),
        "min_descent_retention": float(
            getattr(args, "bypass_fidelity_min_descent_retention", 0.50)
        ),
        "strict_min_cosine": float(
            getattr(args, "bypass_fidelity_strict_min_cosine", 0.95)
        ),
        "max_relative_error": float(
            getattr(args, "bypass_fidelity_max_relative_error", 0.10)
        ),
        "max_loss_abs_diff": float(
            getattr(args, "bypass_fidelity_max_loss_abs_diff", 1e-7)
        ),
    }


def _flatten_gradients(parameters):
    values = []
    for parameter in parameters:
        if parameter.grad is None:
            values.append(torch.zeros_like(parameter, dtype=torch.float32).reshape(-1))
        else:
            values.append(parameter.grad.detach().float().reshape(-1))
    return torch.cat(values)


def _gradient_metrics(reference, candidate):
    reference_norm = torch.linalg.vector_norm(reference)
    candidate_norm = torch.linalg.vector_norm(candidate)
    difference_norm = torch.linalg.vector_norm(candidate - reference)
    dot = torch.dot(reference, candidate)
    reference_value = float(reference_norm)
    candidate_value = float(candidate_norm)
    denominator = reference_value * candidate_value
    return {
        "gradient_cosine": float(dot) / denominator if denominator > 0 else float("nan"),
        "descent_retention": (
            float(dot) / (reference_value * reference_value)
            if reference_value > 0 else float("nan")
        ),
        "relative_gradient_error": (
            float(difference_norm) / reference_value
            if reference_value > 0 else float("nan")
        ),
        "gradient_norm_ratio": (
            candidate_value / reference_value
            if reference_value > 0 else float("nan")
        ),
        "full_gradient_norm": reference_value,
        "bypass_gradient_norm": candidate_value,
    }


def _backward(model, controller, parameters, loss_fn, mode, rng_state, device):
    model.zero_grad(set_to_none=True)
    controller.set_mode(mode)
    adaptive.restore_rng(rng_state)
    loss = loss_fn()
    if not torch.isfinite(loss):
        raise RuntimeError(f"Non-finite loss in mode={mode}")
    loss.backward()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    gradient = _flatten_gradients(parameters)
    stats = controller.stats(0.0)
    return float(loss.detach().float().cpu()), gradient, stats


def _mean(values):
    return statistics.mean(values) if values else float("nan")


def _std(values):
    return statistics.stdev(values) if len(values) > 1 else 0.0


def profile_threshold_gradient_fidelity(
    model,
    controller,
    parameters,
    prepared_batches,
    loss_factory,
    probe_config,
    output_dir,
    device,
    metadata=None,
):
    if len(prepared_batches) != probe_config["probe_batches"]:
        raise ValueError(
            f"Expected {probe_config['probe_batches']} prepared batches, "
            f"found {len(prepared_batches)}"
        )

    rows = []
    for policy in probe_config["policies"]:
        controller.configure(policy["skip_blocks"])
        for batch_index, batch in enumerate(prepared_batches):
            loss_fn = loss_factory(batch, policy["flow_t"])
            rng_state = adaptive.snapshot_rng(device)
            full_loss, full_gradient, _ = _backward(
                model, controller, parameters, loss_fn, "full", rng_state, device
            )
            bypass_loss, bypass_gradient, stats = _backward(
                model,
                controller,
                parameters,
                loss_fn,
                "single_skip",
                rng_state,
                device,
            )
            metrics = _gradient_metrics(full_gradient, bypass_gradient)
            row = {
                "flow_t": policy["flow_t"],
                "noise_ratio": policy["flow_t"],
                "batch_index": batch_index,
                "threshold": policy["threshold"],
                "bypass_budget": policy["bypass_budget"],
                "bypassed_blocks": ";".join(policy["skip_blocks"]),
                "estimated_saved_gflops": policy["estimated_saved_gflops"],
                "full_loss": full_loss,
                "bypass_loss": bypass_loss,
                "loss_abs_diff": abs(bypass_loss - full_loss),
                **metrics,
                "fallback_blocks": stats.fallback_blocks,
                "max_forward_abs_diff": stats.max_reconstruction_abs_diff,
            }
            if not all(
                math.isfinite(float(row[key]))
                for key in (
                    "gradient_cosine",
                    "descent_retention",
                    "relative_gradient_error",
                    "gradient_norm_ratio",
                    "loss_abs_diff",
                )
            ):
                raise RuntimeError(f"Non-finite fidelity row: {row}")
            rows.append(row)
            print(
                f"flow_t={policy['flow_t']:.2f} "
                f"batch={batch_index + 1}/{len(prepared_batches)} "
                f"B={policy['bypass_budget']} "
                f"cos={metrics['gradient_cosine']:.6f} "
                f"descent={metrics['descent_retention']:.6f} "
                f"rel_err={metrics['relative_gradient_error']:.6f}"
            )

    summary = []
    for policy in probe_config["policies"]:
        values = [row for row in rows if row["flow_t"] == policy["flow_t"]]
        result = {
            "flow_t": policy["flow_t"],
            "noise_ratio": policy["flow_t"],
            "threshold": policy["threshold"],
            "bypass_budget": policy["bypass_budget"],
            "num_batches": len(values),
            "estimated_saved_gflops": policy["estimated_saved_gflops"],
        }
        for key in (
            "gradient_cosine",
            "descent_retention",
            "relative_gradient_error",
            "gradient_norm_ratio",
            "loss_abs_diff",
            "fallback_blocks",
            "max_forward_abs_diff",
        ):
            numbers = [float(row[key]) for row in values]
            result[f"mean_{key}"] = _mean(numbers)
            result[f"std_{key}"] = _std(numbers)
            result[f"max_{key}"] = max(numbers)
        result["descent_safe"] = bool(
            result["mean_gradient_cosine"] >= probe_config["min_cosine"]
            and result["mean_descent_retention"]
            >= probe_config["min_descent_retention"]
            and result["max_loss_abs_diff"]
            <= probe_config["max_loss_abs_diff"]
            and result["max_fallback_blocks"] == 0
            and result["max_max_forward_abs_diff"] == 0
        )
        result["strict_fidelity_safe"] = bool(
            result["mean_gradient_cosine"]
            >= probe_config["strict_min_cosine"]
            and result["mean_relative_gradient_error"]
            <= probe_config["max_relative_error"]
            and result["max_loss_abs_diff"]
            <= probe_config["max_loss_abs_diff"]
            and result["max_fallback_blocks"] == 0
            and result["max_max_forward_abs_diff"] == 0
        )
        result["safe"] = result["descent_safe"]
        summary.append(result)

    controller.set_mode("full")
    controller.configure([])
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "gradient_fidelity_per_batch.csv", rows)
    _write_csv(output_dir / "gradient_fidelity_summary.csv", summary)

    all_descent_safe = all(row["descent_safe"] for row in summary)
    all_strict_safe = all(row["strict_fidelity_safe"] for row in summary)
    report = {
        "model": "VOSR-0.5B-ms",
        "method": "Threshold bypass gradient fidelity",
        "selection_file": probe_config["selection_path"],
        "threshold_policy_csv": probe_config["policy_path"],
        "threshold": probe_config["policies"][0]["threshold"],
        "schedule": " ".join(
            f"{row['flow_t']:g}:{row['bypass_budget']}"
            for row in probe_config["policies"]
        ),
        "probe_batches_per_anchor": probe_config["probe_batches"],
        "backward_pairs": len(rows),
        "optimizer_steps": 0,
        "min_cosine": probe_config["min_cosine"],
        "min_descent_retention": probe_config["min_descent_retention"],
        "strict_min_cosine": probe_config["strict_min_cosine"],
        "max_relative_error": probe_config["max_relative_error"],
        "max_loss_abs_diff": probe_config["max_loss_abs_diff"],
        "all_flow_anchors_descent_safe": all_descent_safe,
        "all_flow_anchors_strict_fidelity_safe": all_strict_safe,
        "all_flow_anchors_safe": all_descent_safe,
    }
    if metadata:
        report.update(metadata)
    (output_dir / "gradient_fidelity_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report
