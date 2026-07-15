#!/usr/bin/env python3
"""Summarize Base vs one or more fine-tuning methods.

Examples:
  python3 summarize_base_sandwich.py \
    --base_eval_summary outputs/eval_base_gpu_full/eval_summary.csv \
    --method Sandwich-LoRA,outputs/eval_sandwich/eval_summary.csv,outputs/lora_sandwich/summary.csv \
    --method All-LoRA,outputs/eval_all/eval_summary.csv,outputs/lora_all/summary.csv

The older --sandwich_* and --full_* arguments are still supported.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def read_one_csv(path: str | Path) -> dict:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"CSV is empty: {path}")
    return rows[0]


def as_float(row: dict, key: str, default: float = 0.0) -> float:
    value = row.get(key, "")
    if value in ("", None):
        return default
    try:
        return float(value)
    except ValueError:
        return default


def fmt(value: float, digits: int = 4) -> str:
    return f"{value:.{digits}f}"


def write_csv(path: str | Path, rows: list[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def markdown_table(rows: list[dict], columns: list[str]) -> str:
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = ["| " + " | ".join(str(row.get(col, "")) for col in columns) + " |" for row in rows]
    return "\n".join([header, sep, *body])


def parse_method_spec(spec: str) -> tuple[str, str, str]:
    parts = [part.strip() for part in spec.split(",", 2)]
    if len(parts) != 3 or not all(parts):
        raise SystemExit(
            "Invalid --method/--extra_method. Expected: name,eval_summary_csv,train_summary_csv"
        )
    return parts[0], parts[1], parts[2]


def method_row(name: str, eval_csv: str, train_csv: str, base_psnr: float, base_ssim: float) -> dict:
    eval_row = read_one_csv(eval_csv)
    train_row = read_one_csv(train_csv)
    psnr = as_float(eval_row, "mean_psnr")
    ssim = as_float(eval_row, "mean_ssim")
    delta_psnr = psnr - base_psnr
    delta_ssim = ssim - base_ssim
    update_mb = as_float(train_row, "update_size_mb", as_float(train_row, "adapter_size_mb"))
    energy_wh = as_float(train_row, "estimated_energy_wh")
    return {
        "method": name,
        "mean_psnr": fmt(psnr, 4),
        "mean_ssim": fmt(ssim, 5),
        "delta_psnr_vs_base": fmt(delta_psnr, 4),
        "delta_ssim_vs_base": fmt(delta_ssim, 5),
        "train_time_s": fmt(as_float(train_row, "train_time_s"), 1),
        "estimated_energy_wh": fmt(energy_wh, 4),
        "peak_cuda_mem_mb": fmt(as_float(train_row, "peak_cuda_mem_mb"), 1),
        "adapter_size_mb": fmt(update_mb, 4),
        "upload_time_1mbps_s": fmt(as_float(train_row, "upload_time_1mbps_s"), 1),
        "psnr_gain_per_mb": fmt(delta_psnr / update_mb, 6) if update_mb else "",
        "psnr_gain_per_wh": fmt(delta_psnr / energy_wh, 6) if energy_wh else "",
    }


def collect_method_specs(args: argparse.Namespace) -> list[tuple[str, str, str]]:
    specs = []
    if args.sandwich_eval_summary and args.sandwich_train_summary:
        specs.append(("Sandwich-LoRA", args.sandwich_eval_summary, args.sandwich_train_summary))
    if args.full_eval_summary and args.full_train_summary:
        specs.append(("Full UNet Fine-tune", args.full_eval_summary, args.full_train_summary))
    for spec in args.method + args.extra_method:
        specs.append(parse_method_spec(spec))
    return specs


def build_summary(args: argparse.Namespace) -> tuple[list[dict], str]:
    base_eval = read_one_csv(args.base_eval_summary)
    base_psnr = as_float(base_eval, "mean_psnr")
    base_ssim = as_float(base_eval, "mean_ssim")
    rows = [
        {
            "method": "Base DLM",
            "mean_psnr": fmt(base_psnr, 4),
            "mean_ssim": fmt(base_ssim, 5),
            "delta_psnr_vs_base": "0.0000",
            "delta_ssim_vs_base": "0.00000",
            "train_time_s": "0.0",
            "estimated_energy_wh": "0.0000",
            "peak_cuda_mem_mb": "",
            "adapter_size_mb": "0.0000",
            "upload_time_1mbps_s": "0.0",
            "psnr_gain_per_mb": "",
            "psnr_gain_per_wh": "",
        }
    ]

    method_specs = collect_method_specs(args)
    if not method_specs:
        raise SystemExit("Provide at least one --method or the legacy --sandwich_* arguments.")
    for name, eval_csv, train_csv in method_specs:
        rows.append(method_row(name, eval_csv, train_csv, base_psnr, base_ssim))

    best_quality = max(rows[1:], key=lambda r: float(r["mean_psnr"]))
    best_efficiency = max(
        [r for r in rows[1:] if r["psnr_gain_per_mb"]],
        key=lambda r: float(r["psnr_gain_per_mb"]),
        default=None,
    )
    columns = list(rows[0].keys())
    conclusion = [
        f"Best PSNR: {best_quality['method']} ({best_quality['mean_psnr']} dB).",
    ]
    if best_efficiency:
        conclusion.append(
            f"Best PSNR gain per MB: {best_efficiency['method']} ({best_efficiency['psnr_gain_per_mb']})."
        )
    report = "\n".join(
        [
            "# Method Comparison Summary",
            "",
            markdown_table(rows, columns),
            "",
            "## Conclusion",
            "",
            " ".join(conclusion),
            "",
        ]
    )
    return rows, report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base_eval_summary", default="outputs/eval_base/eval_summary.csv")
    parser.add_argument("--sandwich_eval_summary", default="")
    parser.add_argument("--sandwich_train_summary", default="")
    parser.add_argument("--full_eval_summary", default="")
    parser.add_argument("--full_train_summary", default="")
    parser.add_argument(
        "--method",
        action="append",
        default=[],
        help="Repeatable: name,eval_summary_csv,train_summary_csv",
    )
    parser.add_argument(
        "--extra_method",
        action="append",
        default=[],
        help="Alias of --method for compatibility with requested wording.",
    )
    parser.add_argument("--output_csv", default="outputs/method_comparison_summary.csv")
    parser.add_argument("--output_md", default="outputs/method_comparison_report.md")
    args = parser.parse_args()

    required = [args.base_eval_summary]
    for _name, eval_csv, train_csv in collect_method_specs(args):
        required.extend([eval_csv, train_csv])
    missing = [str(path) for path in required if not path or not Path(path).exists()]
    if missing:
        print("Missing required result files:")
        for path in missing:
            print(f"  - {path}")
        raise SystemExit(1)

    rows, report = build_summary(args)
    write_csv(args.output_csv, rows)
    output_md = Path(args.output_md)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text(report, encoding="utf-8")
    print(f"Wrote {args.output_csv}")
    print(f"Wrote {args.output_md}")


if __name__ == "__main__":
    main()
