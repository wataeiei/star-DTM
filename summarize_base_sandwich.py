#!/usr/bin/env python3
"""Summarize Base vs Sandwich-LoRA and optional full fine-tuning results.

Reads the CSV files produced by onboard_sandwich_lora_sr.py and writes:
  outputs/base_vs_sandwich_summary.csv
  outputs/base_vs_sandwich_report.md
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def read_one_csv(path: str | Path) -> dict:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as f:
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


def build_summary(args: argparse.Namespace) -> tuple[list[dict], str]:
    base_eval = read_one_csv(args.base_eval_summary)
    sandwich_eval = read_one_csv(args.sandwich_eval_summary)
    train = read_one_csv(args.sandwich_train_summary)

    base_psnr = as_float(base_eval, "mean_psnr")
    base_ssim = as_float(base_eval, "mean_ssim")
    sandwich_psnr = as_float(sandwich_eval, "mean_psnr")
    sandwich_ssim = as_float(sandwich_eval, "mean_ssim")
    delta_psnr = sandwich_psnr - base_psnr
    delta_ssim = sandwich_ssim - base_ssim

    adapter_mb = as_float(train, "adapter_size_mb")
    energy_wh = as_float(train, "estimated_energy_wh")
    upload_1mbps = as_float(train, "upload_time_1mbps_s")
    peak_mem = as_float(train, "peak_cuda_mem_mb")
    train_time = as_float(train, "train_time_s")

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
        },
        {
            "method": "Sandwich-LoRA",
            "mean_psnr": fmt(sandwich_psnr, 4),
            "mean_ssim": fmt(sandwich_ssim, 5),
            "delta_psnr_vs_base": fmt(delta_psnr, 4),
            "delta_ssim_vs_base": fmt(delta_ssim, 5),
            "train_time_s": fmt(train_time, 1),
            "estimated_energy_wh": fmt(energy_wh, 4),
            "peak_cuda_mem_mb": fmt(peak_mem, 1),
            "adapter_size_mb": fmt(adapter_mb, 4),
            "upload_time_1mbps_s": fmt(upload_1mbps, 1),
            "psnr_gain_per_mb": fmt(delta_psnr / adapter_mb, 6) if adapter_mb else "",
            "psnr_gain_per_wh": fmt(delta_psnr / energy_wh, 6) if energy_wh else "",
        },
    ]

    if args.full_eval_summary and args.full_train_summary:
        full_eval = read_one_csv(args.full_eval_summary)
        full_train = read_one_csv(args.full_train_summary)
        full_psnr = as_float(full_eval, "mean_psnr")
        full_ssim = as_float(full_eval, "mean_ssim")
        full_delta_psnr = full_psnr - base_psnr
        full_delta_ssim = full_ssim - base_ssim
        full_update_mb = as_float(full_train, "update_size_mb", as_float(full_train, "adapter_size_mb"))
        full_energy_wh = as_float(full_train, "estimated_energy_wh")
        rows.append(
            {
                "method": "Full UNet Fine-tune",
                "mean_psnr": fmt(full_psnr, 4),
                "mean_ssim": fmt(full_ssim, 5),
                "delta_psnr_vs_base": fmt(full_delta_psnr, 4),
                "delta_ssim_vs_base": fmt(full_delta_ssim, 5),
                "train_time_s": fmt(as_float(full_train, "train_time_s"), 1),
                "estimated_energy_wh": fmt(full_energy_wh, 4),
                "peak_cuda_mem_mb": fmt(as_float(full_train, "peak_cuda_mem_mb"), 1),
                "adapter_size_mb": fmt(full_update_mb, 4),
                "upload_time_1mbps_s": fmt(as_float(full_train, "upload_time_1mbps_s"), 1),
                "psnr_gain_per_mb": fmt(full_delta_psnr / full_update_mb, 6) if full_update_mb else "",
                "psnr_gain_per_wh": fmt(full_delta_psnr / full_energy_wh, 6) if full_energy_wh else "",
            }
        )

    verdict_bits = []
    if delta_psnr > 0:
        verdict_bits.append(f"Sandwich-LoRA improves PSNR by {delta_psnr:.4f} dB over Base.")
    elif delta_psnr < 0:
        verdict_bits.append(f"Sandwich-LoRA is {abs(delta_psnr):.4f} dB lower than Base in PSNR.")
    else:
        verdict_bits.append("Sandwich-LoRA matches Base PSNR.")

    if delta_ssim > 0:
        verdict_bits.append(f"SSIM improves by {delta_ssim:.5f}.")
    elif delta_ssim < 0:
        verdict_bits.append(f"SSIM decreases by {abs(delta_ssim):.5f}.")
    else:
        verdict_bits.append("SSIM is unchanged.")

    if adapter_mb:
        verdict_bits.append(
            f"The LoRA adapter is {adapter_mb:.2f} MB and takes about {upload_1mbps:.1f} s "
            "to upload at 1 Mbps with the configured link efficiency."
        )

    columns = list(rows[0].keys())
    report = "\n".join(
        [
            "# Base vs Sandwich-LoRA Summary",
            "",
            markdown_table(rows, columns),
            "",
            "## Conclusion",
            "",
            " ".join(verdict_bits),
            "",
            "For the paper/report, interpret the result as a quality-resource trade-off: "
            "Sandwich-LoRA is useful when its quality gain over Base is achieved with a small adapter, "
            "short upload time, and acceptable training energy/memory on Jetson.",
            "",
        ]
    )
    return rows, report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base_eval_summary", default="outputs/eval_base/eval_summary.csv")
    parser.add_argument("--sandwich_eval_summary", default="outputs/eval_sandwich_r8/eval_summary.csv")
    parser.add_argument("--sandwich_train_summary", default="outputs/lora_sandwich_r8/summary.csv")
    parser.add_argument("--full_eval_summary", default="")
    parser.add_argument("--full_train_summary", default="")
    parser.add_argument("--output_csv", default="outputs/base_vs_sandwich_summary.csv")
    parser.add_argument("--output_md", default="outputs/base_vs_sandwich_report.md")
    args = parser.parse_args()

    required = [args.base_eval_summary, args.sandwich_eval_summary, args.sandwich_train_summary]
    if args.full_eval_summary or args.full_train_summary:
        required.extend([args.full_eval_summary, args.full_train_summary])
    missing = [str(path) for path in required if not path or not Path(path).exists()]
    if missing:
        print("Missing required result files:")
        for path in missing:
            print(f"  - {path}")
        print("\nRun Base eval, Sandwich-LoRA training, and Sandwich-LoRA eval first.")
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
