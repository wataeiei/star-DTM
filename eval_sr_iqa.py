#!/usr/bin/env python3
"""Evaluate full-reference and no-reference IQA metrics for SR outputs.

The script intentionally creates one PyIQA model at a time and writes progress
after every source. This keeps peak memory bounded and makes long evaluations
resumable on edge devices.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
DEFAULT_METRICS = ["lpips", "musiq", "maniqa", "clipiqa", "liqe"]
FULL_REFERENCE_METRICS = {"lpips"}


def parse_source(value: str) -> tuple[str, Path]:
    label, separator, path = value.partition("=")
    if not separator or not label.strip() or not path.strip():
        raise argparse.ArgumentTypeError("--source must use LABEL=IMAGE_DIR")
    return label.strip(), Path(path.strip())


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise SystemExit(f"Empty manifest: {path}")
    required = {"filename", "class_name"}
    missing = required - set(rows[0])
    if missing:
        raise SystemExit(f"Manifest is missing columns: {sorted(missing)}")
    if "exclude_from_eval" in rows[0]:
        rows = [row for row in rows if row["exclude_from_eval"].lower() != "true"]
    return rows


def image_name_index(source_dir: Path) -> dict[str, Path]:
    if not source_dir.is_dir():
        raise SystemExit(f"Image source directory not found: {source_dir}")
    paths = sorted(
        path for path in source_dir.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES
    )
    index: dict[str, Path] = {}
    for path in paths:
        index[path.name] = path
        match = re.match(r"^\d+_(.+)$", path.name)
        if match:
            index[match.group(1)] = path
    return index


def resolve_source(
    manifest_rows: list[dict[str, str]], source_dir: Path
) -> list[dict[str, str]]:
    index = image_name_index(source_dir)
    resolved = []
    missing = []
    for row in manifest_rows:
        image_path = index.get(row["filename"])
        if image_path is None:
            missing.append(row["filename"])
            continue
        item = dict(row)
        item["resolved_path"] = str(image_path.resolve())
        resolved.append(item)
    if missing:
        raise SystemExit(
            f"{source_dir} is missing {len(missing)} manifest images; "
            f"examples: {missing[:5]}"
        )
    return resolved


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def read_existing(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def scalar_score(value) -> float:
    if isinstance(value, (tuple, list)):
        if len(value) != 1:
            raise RuntimeError(f"Expected one IQA score, received {len(value)} values")
        value = value[0]
    if torch.is_tensor(value):
        value = value.detach().float().mean().cpu().item()
    score = float(value)
    if not math.isfinite(score):
        raise RuntimeError(f"IQA model returned a non-finite score: {score}")
    return score


def summarize(rows: list[dict]) -> list[dict]:
    methods = sorted({str(row["method"]) for row in rows})
    metrics = [name for name in DEFAULT_METRICS if any(row["metric"] == name for row in rows)]
    metrics.extend(
        sorted({str(row["metric"]) for row in rows} - set(metrics))
    )
    summary = []
    for method in methods:
        method_rows = [row for row in rows if row["method"] == method]
        item: dict[str, object] = {"method": method}
        counts = []
        for metric_name in metrics:
            values = [
                float(row["score"])
                for row in method_rows
                if row["metric"] == metric_name
            ]
            if not values:
                item[f"mean_{metric_name}"] = ""
                item[f"std_{metric_name}"] = ""
                continue
            counts.append(len(values))
            item[f"mean_{metric_name}"] = float(np.mean(values))
            item[f"std_{metric_name}"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        item["num_images"] = min(counts) if counts else 0
        summary.append(item)
    return summary


def import_pyiqa():
    try:
        import pyiqa
    except Exception as error:
        raise SystemExit(
            "Could not import PyIQA. Install it in an environment with a working, "
            "Torch-compatible torchvision build. Original error: " + repr(error)
        ) from error
    return pyiqa


def evaluate(args: argparse.Namespace) -> None:
    pyiqa = import_pyiqa()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    detail_path = output_dir / "iqa_per_image.csv"
    summary_path = output_dir / "iqa_metrics_summary.csv"

    manifest_rows = read_manifest(Path(args.eval_manifest))
    if args.max_images > 0:
        manifest_rows = manifest_rows[: args.max_images]
    references = resolve_source(manifest_rows, Path(args.hr_dir))
    reference_by_name = {row["filename"]: row["resolved_path"] for row in references}
    sources = [
        (label, resolve_source(manifest_rows, source_dir))
        for label, source_dir in args.source
    ]

    available = set(pyiqa.list_models())
    missing_metrics = [name for name in args.metrics if name not in available]
    if missing_metrics:
        raise SystemExit(
            f"PyIQA does not recognize metrics {missing_metrics}. "
            "Run `python -c \"import pyiqa; print(pyiqa.list_models())\"`."
        )

    existing = read_existing(detail_path) if args.resume else []
    results: list[dict] = [dict(row) for row in existing]
    completed = {
        (str(row["method"]), str(row["filename"]), str(row["metric"]))
        for row in existing
    }
    device = torch.device(args.device)
    started = time.perf_counter()

    for metric_name in args.metrics:
        print(f"Loading PyIQA metric: {metric_name}", flush=True)
        metric = pyiqa.create_metric(metric_name, device=device)
        metric.eval()
        lower_better = bool(metric.lower_better)
        metric_started = time.perf_counter()

        for label, source_rows in sources:
            pending = [
                row
                for row in source_rows
                if (label, row["filename"], metric_name) not in completed
            ]
            if not pending:
                print(f"  {label}: already complete", flush=True)
                continue
            source_started = time.perf_counter()
            for position, row in enumerate(pending, start=1):
                distorted_path = row["resolved_path"]
                with torch.inference_mode():
                    if metric_name in FULL_REFERENCE_METRICS:
                        value = metric(distorted_path, reference_by_name[row["filename"]])
                    else:
                        value = metric(distorted_path)
                score = scalar_score(value)
                result = {
                    "method": label,
                    "filename": row["filename"],
                    "class_name": row["class_name"],
                    "metric": metric_name,
                    "score": score,
                    "lower_better": lower_better,
                }
                results.append(result)
                completed.add((label, row["filename"], metric_name))
                if position == 1 or position % args.log_every == 0 or position == len(pending):
                    print(
                        f"  {label}: {position:04d}/{len(pending):04d} "
                        f"{metric_name}={score:.6f}",
                        flush=True,
                    )
            write_csv(detail_path, results)
            write_csv(summary_path, summarize(results))
            print(
                f"  {label}: completed in {time.perf_counter() - source_started:.1f}s",
                flush=True,
            )

        del metric
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(
            f"Finished {metric_name} in {time.perf_counter() - metric_started:.1f}s",
            flush=True,
        )

    metadata = {
        "eval_manifest": args.eval_manifest,
        "hr_dir": args.hr_dir,
        "sources": [{"method": label, "directory": str(path)} for label, path in args.source],
        "metrics": args.metrics,
        "metric_direction": {
            name: ("lower_is_better" if name in FULL_REFERENCE_METRICS else "higher_is_better")
            for name in args.metrics
        },
        "num_manifest_images": len(manifest_rows),
        "device": str(device),
        "torch_version": torch.__version__,
        "pyiqa_version": getattr(pyiqa, "__version__", "unknown"),
        "elapsed_s": time.perf_counter() - started,
    }
    (output_dir / "iqa_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(f"Wrote {detail_path}")
    print(f"Wrote {summary_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval_manifest", required=True)
    parser.add_argument("--hr_dir", required=True)
    parser.add_argument("--source", action="append", type=parse_source, required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--metrics", nargs="+", default=DEFAULT_METRICS)
    parser.add_argument("--max_images", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--log_every", type=int, default=25)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but torch.cuda.is_available() is False")
    evaluate(args)


if __name__ == "__main__":
    main()
