#!/usr/bin/env python3
"""Summarize DGMR LoRA training/evaluation results and plot comparison figures."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def require_matplotlib():
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit("Missing package: matplotlib\nInstall with: pip3 install matplotlib") from exc
    return plt


def read_one_csv(path: str | Path) -> dict:
    path = Path(path)
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit(f"Empty CSV: {path}")
    return rows[0]


def to_float(row: dict, key: str, default: float = 0.0) -> float:
    value = row.get(key, "")
    if value == "":
        return default
    return float(value)


def to_int(row: dict, key: str, default: int = 0) -> int:
    value = row.get(key, "")
    if value == "":
        return default
    return int(float(value))


def parse_item(text: str) -> tuple[str, Path, Path | None]:
    parts = text.split(":")
    if len(parts) == 2:
        method, eval_csv = parts
        return method, Path(eval_csv), None
    if len(parts) == 3:
        method, eval_csv, train_csv = parts
        return method, Path(eval_csv), Path(train_csv)
    raise SystemExit("--item format: Method:eval_summary.csv[:train_summary.csv]")


def method_step(method: str, train_row: dict | None) -> int:
    if train_row is not None:
        return to_int(train_row, "train_steps", 0)
    import re

    m = re.search(r"(\d+)$", method)
    return int(m.group(1)) if m else 0


def write_csv(path: str | Path, rows: list[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def build_rows(args: argparse.Namespace) -> list[dict]:
    items = [parse_item(item) for item in args.item]
    rows = []
    base_eval = None
    for method, eval_csv, train_csv in items:
        eval_row = read_one_csv(eval_csv)
        train_row = read_one_csv(train_csv) if train_csv else None
        row = {
            "method": method,
            "steps": method_step(method, train_row),
            "mean_psnr": to_float(eval_row, "mean_psnr"),
            "mean_ssim": to_float(eval_row, "mean_ssim"),
            "mean_mae": to_float(eval_row, "mean_mae"),
            "mean_sam_deg": to_float(eval_row, "mean_sam_deg"),
            "train_time_s": to_float(train_row or {}, "train_time_s"),
            "sec_per_step": to_float(train_row or {}, "sec_per_step"),
            "peak_cuda_mem_mb": max(to_float(eval_row, "peak_cuda_mem_mb"), to_float(train_row or {}, "peak_cuda_mem_mb")),
            "adapter_size_mb": to_float(eval_row, "adapter_size_mb", to_float(train_row or {}, "adapter_size_mb")),
            "upload_time_1mbps_s": to_float(eval_row, "upload_time_1mbps_s", to_float(train_row or {}, "upload_time_1mbps_s")),
            "trainable_params": to_int(train_row or {}, "trainable_params"),
            "lora_module_count": to_int(eval_row, "lora_module_count", to_int(train_row or {}, "lora_module_count")),
        }
        if args.base_method and method == args.base_method:
            base_eval = row
        rows.append(row)

    if base_eval is None:
        base_eval = rows[0]

    for row in rows:
        row["delta_psnr_vs_base"] = row["mean_psnr"] - base_eval["mean_psnr"]
        row["delta_ssim_vs_base"] = row["mean_ssim"] - base_eval["mean_ssim"]
        row["delta_mae_vs_base"] = row["mean_mae"] - base_eval["mean_mae"]
        row["delta_sam_vs_base"] = row["mean_sam_deg"] - base_eval["mean_sam_deg"]
        row["psnr_gain_per_mb"] = row["delta_psnr_vs_base"] / row["adapter_size_mb"] if row["adapter_size_mb"] > 0 else ""
        row["psnr_gain_per_train_s"] = row["delta_psnr_vs_base"] / row["train_time_s"] if row["train_time_s"] > 0 else ""
    return rows


def plot_quality(rows: list[dict], out_png: Path) -> None:
    plt = require_matplotlib()
    series = {}
    for row in rows:
        if row["steps"] <= 0:
            continue
        if row["method"].startswith("Dual-Attn"):
            key = "Dual-Attn-LoRA"
        elif row["method"].startswith("All-Linear"):
            key = "All-Linear-LoRA"
        else:
            continue
        series.setdefault(key, []).append(row)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    metrics = [("mean_psnr", "PSNR ↑"), ("mean_ssim", "SSIM ↑"), ("mean_sam_deg", "SAM ↓")]
    colors = {"Dual-Attn-LoRA": "#2563eb", "All-Linear-LoRA": "#dc2626"}
    for ax, (metric, ylabel) in zip(axes, metrics):
        for name, points in series.items():
            points = sorted(points, key=lambda r: r["steps"])
            ax.plot(
                [r["steps"] for r in points],
                [r[metric] for r in points],
                marker="o",
                linewidth=2.0,
                label=name,
                color=colors.get(name),
            )
        ax.set_xlabel("Training steps")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)
    axes[0].legend()
    fig.suptitle("DGMR LoRA Quality across Training Steps")
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=180)
    print(f"Wrote {out_png}")


def plot_cost(rows: list[dict], out_png: Path) -> None:
    plt = require_matplotlib()
    wanted = [r for r in rows if r["steps"] in (0, 100, 300, 500)]
    labels = [r["method"] for r in wanted]
    x = range(len(wanted))
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    metrics = [
        ("train_time_s", "Train time (s)"),
        ("adapter_size_mb", "Adapter size (MB)"),
        ("upload_time_1mbps_s", "Upload time @1Mbps (s)"),
    ]
    for ax, (metric, ylabel) in zip(axes, metrics):
        ax.bar(x, [r[metric] for r in wanted], color="#2563eb")
        ax.set_ylabel(ylabel)
        ax.set_xticks(list(x))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("DGMR LoRA Cost Comparison")
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=180)
    print(f"Wrote {out_png}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--item",
        action="append",
        required=True,
        help="Method:eval_summary.csv[:train_summary.csv]",
    )
    parser.add_argument("--base_method", default="Base-DGMR")
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--quality_png", default="")
    parser.add_argument("--cost_png", default="")
    args = parser.parse_args()

    rows = build_rows(args)
    write_csv(args.output_csv, rows)
    print(f"Wrote {args.output_csv}")
    if args.quality_png:
        plot_quality(rows, Path(args.quality_png))
    if args.cost_png:
        plot_cost(rows, Path(args.cost_png))


if __name__ == "__main__":
    main()
