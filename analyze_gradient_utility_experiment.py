#!/usr/bin/env python3
"""Analyze whether calibration gradients predict single-block LoRA utility."""

import argparse
import json
from pathlib import Path

import pandas as pd
from scipy.stats import kendalltau, spearmanr


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--validation_csv", required=True)
    parser.add_argument("--output_dir", required=True)
    return parser.parse_args()


def correlation_row(name, x, y):
    if name == "spearman":
        coefficient, p_value = spearmanr(x, y)
    else:
        coefficient, p_value = kendalltau(x, y)
    return {
        "test": name,
        "coefficient": coefficient,
        "p_value": p_value,
        "num_blocks": len(x),
    }


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_csv(args.manifest)
    validation = pd.read_csv(args.validation_csv)
    validation["noise_ratio"] = validation["noise_ratio"].astype(str)
    means = validation[validation["noise_ratio"].eq("mean")].copy()
    base_rows = means[means["method"].str.contains("Base", case=False, na=False)]
    if len(base_rows) != 1:
        raise SystemExit(f"Expected one Base mean row, found {len(base_rows)}.")
    base_loss = float(base_rows.iloc[0]["mean_flow_loss"])

    method_losses = means[["method", "mean_flow_loss"]].rename(
        columns={"method": "label", "mean_flow_loss": "single_block_flow_loss"}
    )
    per_block = manifest.merge(method_losses, on="label", how="left", validate="one_to_one")
    if per_block["single_block_flow_loss"].isna().any():
        missing = per_block.loc[per_block["single_block_flow_loss"].isna(), "label"].tolist()
        raise SystemExit("Missing validation results for: " + ", ".join(missing))
    per_block["base_flow_loss"] = base_loss
    per_block["adaptation_utility"] = base_loss - per_block["single_block_flow_loss"]
    per_block["utility_per_million_lora_params"] = (
        per_block["adaptation_utility"] / per_block["lora_param_count"] * 1_000_000
    )
    per_block = per_block.sort_values("importance_rank")
    per_block.to_csv(output_dir / "gradient_utility_per_block.csv", index=False)

    correlations = pd.DataFrame(
        [
            correlation_row(
                "spearman",
                per_block["gradient_importance"],
                per_block["adaptation_utility"],
            ),
            correlation_row(
                "kendall",
                per_block["gradient_importance"],
                per_block["adaptation_utility"],
            ),
        ]
    )
    correlations.to_csv(output_dir / "gradient_utility_correlations.csv", index=False)

    group_summary = (
        per_block.groupby("group", sort=False)
        .agg(
            num_blocks=("block", "count"),
            mean_gradient_importance=("gradient_importance", "mean"),
            mean_flow_loss=("single_block_flow_loss", "mean"),
            std_flow_loss=("single_block_flow_loss", "std"),
            mean_adaptation_utility=("adaptation_utility", "mean"),
            std_adaptation_utility=("adaptation_utility", "std"),
            mean_utility_per_million_params=("utility_per_million_lora_params", "mean"),
        )
        .reset_index()
    )
    group_summary.to_csv(output_dir / "gradient_utility_group_summary.csv", index=False)

    summary = {
        "base_flow_loss": base_loss,
        "num_tested_blocks": int(len(per_block)),
        "spearman_rho": float(correlations.loc[correlations.test.eq("spearman"), "coefficient"].iloc[0]),
        "spearman_p": float(correlations.loc[correlations.test.eq("spearman"), "p_value"].iloc[0]),
        "kendall_tau": float(correlations.loc[correlations.test.eq("kendall"), "coefficient"].iloc[0]),
        "kendall_p": float(correlations.loc[correlations.test.eq("kendall"), "p_value"].iloc[0]),
    }
    (output_dir / "gradient_utility_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    try:
        import matplotlib.pyplot as plt

        colors = {"Top": "#d62728", "Middle": "#1f77b4", "Bottom": "#7f7f7f"}
        fig, ax = plt.subplots(figsize=(7.2, 4.8))
        for group, rows in per_block.groupby("group", sort=False):
            ax.scatter(
                rows["gradient_importance"],
                rows["adaptation_utility"],
                s=60,
                label=group,
                color=colors.get(group),
            )
            for _, row in rows.iterrows():
                label = f"B{int(row['block_index'])}" if pd.notna(row["block_index"]) else row["block"]
                ax.annotate(label, (row["gradient_importance"], row["adaptation_utility"]), xytext=(4, 4), textcoords="offset points", fontsize=8)
        ax.axhline(0.0, color="#888888", linewidth=0.8, linestyle="--")
        ax.set_xlabel("Calibration gradient importance")
        ax.set_ylabel("Single-block utility (base flow loss - adapted flow loss)")
        ax.set_title(f"Gradient importance predicts adaptation utility (Spearman rho={summary['spearman_rho']:.3f})")
        ax.grid(alpha=0.2)
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(output_dir / "gradient_importance_vs_utility.png", dpi=300)
        fig.savefig(output_dir / "gradient_importance_vs_utility.pdf")
        plt.close(fig)
    except ImportError:
        print("matplotlib unavailable; skipped scatter plot.")

    print(json.dumps(summary, indent=2))
    print(f"Wrote analysis to {output_dir}")


if __name__ == "__main__":
    main()
