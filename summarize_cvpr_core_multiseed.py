#!/usr/bin/env python3
"""Summarize three-seed training efficiency for the CVPR core methods."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


DEFAULT_RUNS = {
    "All-LoRA": {
        42: "outputs/dit4sr_all_lora_1000_saved",
        43: "outputs/dit4sr_all_lora_1000_seed43_final",
        44: "outputs/dit4sr_all_lora_1000_seed44_final",
    },
    "GradSkip-LoRA": {
        42: "outputs/dit4sr_gradtop8_noiseaware_singlepass_1000",
        43: "outputs/dit4sr_gradtop8_noiseaware_singlepass_1000_seed43_final",
        44: "outputs/dit4sr_gradtop8_noiseaware_singlepass_1000_seed44_final",
    },
}

METRICS = [
    "train_step_time_s",
    "mean_train_step_time_s",
    "experiment_time_s",
    "profiling_time_s",
    "peak_cuda_mem_mb",
    "adapter_size_mb",
    "lora_module_count",
    "trainable_lora_params",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/cvpr_core_multiseed_training_summary.csv"),
    )
    parser.add_argument("--expected_steps", type=int, default=1000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows: list[dict[str, object]] = []

    for method, seed_runs in DEFAULT_RUNS.items():
        for seed, relative_dir in seed_runs.items():
            run_dir = args.root / relative_dir
            summary_path = run_dir / "summary.csv"
            log_path = run_dir / "train_log.csv"
            adapter_path = run_dir / "lora_adapter.pt"
            missing = [
                str(path)
                for path in (summary_path, log_path, adapter_path)
                if not path.is_file()
            ]
            if missing:
                raise SystemExit("Missing run artifacts:\n  " + "\n  ".join(missing))

            log = pd.read_csv(log_path)
            final_step = int(log["step"].max())
            if final_step != args.expected_steps:
                raise SystemExit(
                    f"{method} seed {seed} ended at step {final_step}, "
                    f"expected {args.expected_steps}."
                )

            summary = pd.read_csv(summary_path).iloc[0].to_dict()
            row = {"method": method, "seed": seed, "final_step": final_step}
            row.update({metric: summary.get(metric) for metric in METRICS})
            rows.append(row)

    per_seed = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    per_seed.to_csv(args.output, index=False)

    aggregate_rows = []
    for method, group in per_seed.groupby("method", sort=False):
        row: dict[str, object] = {"method": method, "num_seeds": len(group)}
        for metric in METRICS:
            values = pd.to_numeric(group[metric], errors="coerce")
            row[f"{metric}_mean"] = values.mean()
            row[f"{metric}_sample_std"] = values.std(ddof=1)
        aggregate_rows.append(row)

    aggregate = pd.DataFrame(aggregate_rows)
    aggregate_path = args.output.with_name(
        args.output.stem + "_mean_std" + args.output.suffix
    )
    aggregate.to_csv(aggregate_path, index=False)

    print("Per-seed results:")
    print(per_seed.to_string(index=False))
    print("\nThree-seed mean +/- sample std:")
    for _, row in aggregate.iterrows():
        print(f"\n{row['method']}")
        for metric in METRICS:
            mean = row[f"{metric}_mean"]
            std = row[f"{metric}_sample_std"]
            print(f"  {metric}: {mean:.6f} +/- {std:.6f}")
    print(f"\nWrote {args.output}")
    print(f"Wrote {aggregate_path}")


if __name__ == "__main__":
    main()
