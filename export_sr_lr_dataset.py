#!/usr/bin/env python3
"""Export the exact bicubic LR inputs used by the SR evaluation pipeline."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise SystemExit(f"Empty manifest: {path}")
    if "filename" not in rows[0]:
        raise SystemExit(f"Manifest has no filename column: {path}")
    if "exclude_from_eval" in rows[0]:
        rows = [row for row in rows if row["exclude_from_eval"].lower() != "true"]
    return rows


def pil_to_tensor(image: Image.Image, size: int) -> torch.Tensor:
    image = image.convert("RGB").resize((size, size), Image.Resampling.BICUBIC)
    data = torch.frombuffer(bytearray(image.tobytes()), dtype=torch.uint8)
    return data.view(size, size, 3).permute(2, 0, 1).float().div(255.0)


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    array = (
        tensor.detach()
        .clamp(0, 1)
        .mul(255)
        .round()
        .to(torch.uint8)
        .permute(1, 2, 0)
        .contiguous()
        .cpu()
        .numpy()
    )
    return Image.fromarray(array, mode="RGB")


def make_lr(hr: torch.Tensor, lr_size: int) -> torch.Tensor:
    return F.interpolate(
        hr.unsqueeze(0),
        size=(lr_size, lr_size),
        mode="bicubic",
        align_corners=False,
        antialias=True,
    ).squeeze(0).clamp(0, 1)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def export(args: argparse.Namespace) -> None:
    if args.image_size <= 0 or args.sr_scale <= 0:
        raise SystemExit("--image_size and --sr_scale must be positive")
    if args.image_size % args.sr_scale:
        raise SystemExit("--image_size must be divisible by --sr_scale")

    manifest_path = Path(args.eval_manifest)
    hr_dir = Path(args.hr_dir)
    output_dir = Path(args.output_dir)
    if not hr_dir.is_dir():
        raise SystemExit(f"HR directory not found: {hr_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = read_manifest(manifest_path)
    if args.max_images > 0:
        rows = rows[: args.max_images]
    lr_size = args.image_size // args.sr_scale
    exported = []

    for index, row in enumerate(rows, start=1):
        source_path = hr_dir / row["filename"]
        if not source_path.is_file():
            raise SystemExit(f"Missing HR image: {source_path}")
        output_path = output_dir / f"{Path(row['filename']).stem}.png"
        if output_path.exists() and not args.overwrite:
            with Image.open(output_path) as existing:
                if existing.size != (lr_size, lr_size):
                    raise SystemExit(
                        f"Existing LR image has wrong size {existing.size}: {output_path}"
                    )
        else:
            with Image.open(source_path) as image:
                hr = pil_to_tensor(image, args.image_size)
            tensor_to_pil(make_lr(hr, lr_size)).save(output_path)

        exported.append(
            {
                "filename": output_path.name,
                "hr_filename": row["filename"],
                "class_id": row.get("class_id", ""),
                "class_name": row.get("class_name", ""),
                "hr_width": args.image_size,
                "hr_height": args.image_size,
                "lr_width": lr_size,
                "lr_height": lr_size,
                "sr_scale": args.sr_scale,
                "lr_sha256": sha256(output_path),
            }
        )
        if index == 1 or index % args.log_every == 0 or index == len(rows):
            print(f"[{index}/{len(rows)}] {source_path.name} -> {output_path.name}")

    write_csv(output_dir / "lr_manifest.csv", exported)
    metadata = {
        "eval_manifest": str(manifest_path),
        "hr_dir": str(hr_dir),
        "output_dir": str(output_dir),
        "num_images": len(exported),
        "image_size": args.image_size,
        "lr_size": lr_size,
        "sr_scale": args.sr_scale,
        "downsample": {
            "library": "torch.nn.functional.interpolate",
            "mode": "bicubic",
            "align_corners": False,
            "antialias": True,
        },
        "storage": "RGB PNG, uint8",
        "torch_version": torch.__version__,
    }
    (output_dir / "lr_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(f"Wrote {output_dir / 'lr_manifest.csv'}")
    print(f"Wrote {output_dir / 'lr_metadata.json'}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval_manifest", required=True)
    parser.add_argument("--hr_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--sr_scale", type=int, default=4)
    parser.add_argument("--max_images", type=int, default=0)
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument(
        "--overwrite", action=argparse.BooleanOptionalAction, default=False
    )
    return parser


if __name__ == "__main__":
    export(build_parser().parse_args())
