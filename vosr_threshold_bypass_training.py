"""Runtime policy helpers for VOSR Threshold bypass training."""

from __future__ import annotations

import csv
import json
from pathlib import Path


def _read_csv(path):
    with Path(path).open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def configure_threshold_bypass_training(args):
    if not bool(getattr(args, "threshold_bypass_training", False)):
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
    policies = []
    for row in _read_csv(policy_path):
        flow_t = float(row.get("flow_t", row.get("noise_ratio")))
        blocks = [value for value in row["skip_blocks"].split(";") if value]
        indices = [int(block.rsplit(".", 1)[-1]) for block in blocks]
        unknown = sorted(set(indices) - set(frozen))
        if unknown:
            raise ValueError(
                f"Policy bypasses protected LoRA blocks at flow_t={flow_t:g}: "
                f"{unknown}"
            )
        budget = int(row["bypass_budget"])
        if budget != len(blocks):
            raise ValueError(
                f"Policy count mismatch at flow_t={flow_t:g}: "
                f"declared={budget}, blocks={len(blocks)}"
            )
        policies.append(
            {
                "flow_t": flow_t,
                "threshold": float(row["threshold"]),
                "bypass_budget": budget,
                "skip_blocks": blocks,
                "estimated_saved_gflops": float(row["estimated_saved_gflops"]),
            }
        )
    policies.sort(key=lambda row: row["flow_t"])
    if not policies:
        raise ValueError("Threshold policy CSV is empty")
    if len({row["flow_t"] for row in policies}) != len(policies):
        raise ValueError("Threshold policy contains duplicate flow anchors")
    if int(args.train_batch_size) != 1:
        raise ValueError(
            "Threshold bypass currently requires train_batch_size=1 so each "
            "step has one unambiguous flow-conditioned policy"
        )

    return {
        "selection_path": str(selection_path),
        "policy_path": str(policy_path),
        "selection": selection,
        "selected_indices": selected,
        "frozen_indices": frozen,
        "policies": policies,
        "threshold": policies[0]["threshold"],
    }


def nearest_policy(config, sampled_flow_t):
    value = float(sampled_flow_t)
    return min(
        config["policies"],
        key=lambda row: (abs(row["flow_t"] - value), row["flow_t"]),
    )


def write_threshold_bypass_training_report(output_dir, rows, config, metadata):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("No Threshold bypass training rows were recorded")

    log_path = output_dir / "train_log.csv"
    with log_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    total_time = sum(float(row["train_step_time_s"]) for row in rows)
    summary = {
        "method": "Threshold bypass",
        "train_steps": len(rows),
        "selected_lora_blocks": len(config["selected_indices"]),
        "selected_utility": config["selection"].get("selected_utility"),
        "threshold": config["threshold"],
        "mean_bypassed_blocks": sum(
            float(row["skipped_block_count"]) for row in rows
        ) / len(rows),
        "fallback_blocks": sum(int(row["fallback_blocks"]) for row in rows),
        "train_step_time_s": total_time,
        "mean_train_step_time_s": total_time / len(rows),
        "mean_loss": sum(float(row["loss"]) for row in rows) / len(rows),
        "final_loss": float(rows[-1]["loss"]),
        "schedule": " ".join(
            f"{row['flow_t']:g}:{row['bypass_budget']}"
            for row in config["policies"]
        ),
        **metadata,
    }
    summary_path = output_dir / "threshold_bypass_training_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
