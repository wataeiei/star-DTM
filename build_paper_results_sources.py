#!/usr/bin/env python3
"""Build a unified paper-results table from a manifest of experiment CSV files."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


MANIFEST_COLUMNS = [
    "method",
    "seed",
    "source_method",
    "train_summary",
    "sr_summary",
    "sr_per_image",
    "iqa_summary",
    "iqa_per_image",
    "semantic_summary",
    "semantic_per_image",
]

MASTER_COLUMNS = [
    "method",
    "seed",
    "num_images",
    "psnr",
    "ssim",
    "lpips",
    "musiq",
    "maniqa",
    "clipiqa",
    "liqe",
    "top1_accuracy",
    "macro_f1",
    "balanced_accuracy",
    "train_steps",
    "train_time_s",
    "experiment_time_s",
    "mean_step_time_s",
    "profiling_time_s",
    "peak_train_cuda_mem_mb",
    "peak_inference_cuda_mem_mb",
    "adapter_size_mb",
    "trainable_params",
    "lora_module_count",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def clean_path(value: str, root: Path) -> Path | None:
    value = (value or "").strip()
    if not value:
        return None
    path = Path(value).expanduser()
    return path if path.is_absolute() else root / path


def select_method_row(path: Path, method: str) -> dict[str, str]:
    rows = read_csv(path)
    if not rows:
        raise ValueError(f"CSV is empty: {path}")
    if "method" not in rows[0]:
        if len(rows) == 1:
            return rows[0]
        raise ValueError(f"CSV has multiple rows but no method column: {path}")
    matches = [row for row in rows if row.get("method", "").strip() == method]
    if len(matches) != 1:
        available = sorted({row.get("method", "") for row in rows})
        raise ValueError(
            f"Expected one row for method {method!r} in {path}, found {len(matches)}. "
            f"Available methods: {available}"
        )
    return matches[0]


def first_value(row: dict[str, str], *keys: str) -> str:
    for key in keys:
        value = row.get(key, "")
        if value not in (None, ""):
            return value
    return ""


def numeric_or_blank(value: str):
    if value in (None, ""):
        return ""
    number = float(value)
    return number if math.isfinite(number) else ""


def count_method_images(path: Path, method: str) -> int:
    rows = read_csv(path)
    if not rows:
        return 0
    if "method" not in rows[0]:
        return len(rows)
    return sum(row.get("method", "").strip() == method for row in rows)


def add_values(output: dict, source: dict[str, str], mapping: dict[str, tuple[str, ...]]) -> None:
    for destination, candidates in mapping.items():
        value = first_value(source, *candidates)
        if value not in (None, ""):
            output[destination] = numeric_or_blank(value)


def build_row(entry: dict[str, str], root: Path, expected_images: int) -> tuple[dict, list[str]]:
    method = entry["method"].strip()
    source_method = (entry.get("source_method") or method).strip()
    output = {column: "" for column in MASTER_COLUMNS}
    output.update({"method": method, "seed": entry.get("seed", "").strip()})
    warnings = []

    train_path = clean_path(entry.get("train_summary", ""), root)
    if train_path is not None:
        if not train_path.is_file():
            warnings.append(f"missing train_summary: {train_path}")
        else:
            train = select_method_row(train_path, source_method)
            add_values(
                output,
                train,
                {
                    "train_steps": ("train_steps",),
                    "train_time_s": ("train_step_time_s", "train_time_s"),
                    "experiment_time_s": ("experiment_time_s",),
                    "mean_step_time_s": ("mean_train_step_time_s", "sec_per_step"),
                    "profiling_time_s": ("profiling_time_s",),
                    "peak_train_cuda_mem_mb": ("peak_cuda_mem_mb", "train_peak_cuda_mem_mb"),
                    "adapter_size_mb": ("adapter_size_mb", "update_size_mb"),
                    "trainable_params": ("trainable_lora_params", "trainable_params"),
                    "lora_module_count": ("lora_module_count",),
                },
            )

    sr_path = clean_path(entry.get("sr_summary", ""), root)
    if sr_path is not None:
        if not sr_path.is_file():
            warnings.append(f"missing sr_summary: {sr_path}")
        else:
            sr = select_method_row(sr_path, source_method)
            add_values(
                output,
                sr,
                {
                    "num_images": ("num_images",),
                    "psnr": ("mean_psnr",),
                    "ssim": ("mean_ssim",),
                    "lpips": ("mean_lpips",),
                    "peak_inference_cuda_mem_mb": ("peak_cuda_mem_mb",),
                    "adapter_size_mb": ("adapter_size_mb",),
                },
            )

    iqa_path = clean_path(entry.get("iqa_summary", ""), root)
    if iqa_path is not None:
        if not iqa_path.is_file():
            warnings.append(f"missing iqa_summary: {iqa_path}")
        else:
            iqa = select_method_row(iqa_path, source_method)
            add_values(
                output,
                iqa,
                {
                    "num_images": ("num_images",),
                    "lpips": ("mean_lpips",),
                    "musiq": ("mean_musiq",),
                    "maniqa": ("mean_maniqa",),
                    "clipiqa": ("mean_clipiqa",),
                    "liqe": ("mean_liqe",),
                },
            )

    semantic_path = clean_path(entry.get("semantic_summary", ""), root)
    if semantic_path is not None:
        if not semantic_path.is_file():
            warnings.append(f"missing semantic_summary: {semantic_path}")
        else:
            semantic = select_method_row(semantic_path, source_method)
            add_values(
                output,
                semantic,
                {
                    "num_images": ("num_images",),
                    "top1_accuracy": ("top1_accuracy",),
                    "macro_f1": ("macro_f1",),
                    "balanced_accuracy": ("balanced_accuracy",),
                },
            )

    per_image_counts = []
    for column in ("sr_per_image", "iqa_per_image", "semantic_per_image"):
        path = clean_path(entry.get(column, ""), root)
        if path is None:
            continue
        if not path.is_file():
            warnings.append(f"missing {column}: {path}")
            continue
        count = count_method_images(path, source_method)
        per_image_counts.append((column, count))
        if expected_images > 0 and count != expected_images:
            warnings.append(f"{column} has {count} rows, expected {expected_images}")
    if per_image_counts and len({count for _, count in per_image_counts}) > 1:
        warnings.append(f"per-image row counts disagree: {per_image_counts}")

    return output, warnings


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--root",
        default=".",
        help="Root used to resolve relative paths in the manifest (default: current directory).",
    )
    parser.add_argument("--expected_images", type=int, default=419)
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    entries = read_csv(manifest_path)
    missing = set(MANIFEST_COLUMNS) - set(entries[0] if entries else {})
    if missing:
        raise SystemExit(f"Manifest is empty or missing columns: {sorted(missing)}")

    root = Path(args.root).expanduser().resolve()
    outputs = []
    issue_count = 0
    for entry in entries:
        if not entry.get("method", "").strip():
            continue
        try:
            output, warnings = build_row(entry, root, args.expected_images)
        except (OSError, ValueError) as error:
            raise SystemExit(f"{entry.get('method', '<unknown>')}: {error}") from error
        outputs.append(output)
        for warning in warnings:
            issue_count += 1
            print(f"WARNING [{output['method']} seed={output['seed']}]: {warning}")

    output_path = Path(args.output)
    write_csv(output_path, outputs, MASTER_COLUMNS)
    print(f"Wrote {len(outputs)} rows to {output_path}")
    print(f"Validation warnings: {issue_count}")


if __name__ == "__main__":
    main()
