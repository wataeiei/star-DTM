#!/usr/bin/env python3
"""Build a reproducible, class-balanced AID-2.1K split.

The expected source layout is one directory per class::

    aid_full/
      airport/*.jpg
      bare_land/*.jpg
      ...

By default the script selects 70 source images from each of AID's 30 classes,
placing 56 per class in ``train_hr`` and 14 per class in ``val_hr``.  It writes
manifests compatible with ``ucmerced_semantic_classifier.py`` and audits pixel
duplicates before materializing the split.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}


@dataclass(frozen=True)
class SourceImage:
    path: Path
    class_name: str
    pixel_sha256: str


def pixel_sha256(path: Path) -> str:
    from PIL import Image

    with Image.open(path) as image:
        rgb = image.convert("RGB")
        digest = hashlib.sha256()
        digest.update(f"RGB:{rgb.width}x{rgb.height}:".encode("ascii"))
        digest.update(rgb.tobytes())
        return digest.hexdigest()


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write an empty manifest: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def discover_images(source_dir: Path) -> dict[str, list[Path]]:
    if not source_dir.is_dir():
        raise SystemExit(f"Source directory not found: {source_dir}")
    grouped: dict[str, list[Path]] = defaultdict(list)
    for path in sorted(source_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            grouped[path.parent.name].append(path)
    if not grouped:
        raise SystemExit(f"No class-organized images found under {source_dir}")
    return dict(sorted(grouped.items()))


def audit_candidates(grouped_paths: dict[str, list[Path]]) -> tuple[
    dict[str, list[SourceImage]], list[dict]
]:
    unique_by_class: dict[str, list[SourceImage]] = {}
    hash_owners: dict[str, list[tuple[str, Path]]] = defaultdict(list)
    duplicate_rows: list[dict] = []

    for class_name, paths in grouped_paths.items():
        print(f"Auditing class {class_name}: {len(paths)} source images")
        by_hash: dict[str, list[Path]] = defaultdict(list)
        for path in paths:
            digest = pixel_sha256(path)
            by_hash[digest].append(path)
            hash_owners[digest].append((class_name, path))

        unique_by_class[class_name] = [
            SourceImage(paths[0], class_name, digest)
            for digest, paths in sorted(by_hash.items(), key=lambda item: str(item[1][0]))
        ]
        for digest, duplicates in by_hash.items():
            if len(duplicates) > 1:
                duplicate_rows.append(
                    {
                        "pixel_sha256": digest,
                        "class_names": class_name,
                        "duplicate_count": len(duplicates),
                        "paths": ";".join(str(path) for path in duplicates),
                    }
                )

    conflicts = []
    for digest, owners in hash_owners.items():
        classes = sorted({class_name for class_name, _path in owners})
        if len(classes) > 1:
            conflicts.append((digest, classes, owners))
    if conflicts:
        digest, classes, owners = conflicts[0]
        examples = ", ".join(str(path) for _class_name, path in owners[:4])
        raise SystemExit(
            "Identical pixels have conflicting class labels. "
            f"hash={digest} classes={classes} examples={examples}"
        )
    return unique_by_class, duplicate_rows


def materialize(source: Path, destination: Path, mode: str) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if mode == "copy":
        shutil.copy2(source, destination)
        return "copy"
    if mode == "symlink":
        destination.symlink_to(source.resolve())
        return "symlink"
    try:
        os.link(source, destination)
        return "hardlink"
    except OSError:
        shutil.copy2(source, destination)
        return "copy_fallback"


def build_split(
    source_dir: Path,
    output_dir: Path,
    *,
    seed: int,
    expected_classes: int,
    samples_per_class: int,
    train_per_class: int,
    materialize_mode: str,
) -> dict:
    if train_per_class <= 0 or train_per_class >= samples_per_class:
        raise ValueError("train_per_class must be between 1 and samples_per_class - 1")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise SystemExit(
            f"Output directory is not empty: {output_dir}\n"
            "Choose a new directory or remove the incomplete output explicitly."
        )

    grouped_paths = discover_images(source_dir)
    if expected_classes > 0 and len(grouped_paths) != expected_classes:
        raise SystemExit(
            f"Expected {expected_classes} classes, found {len(grouped_paths)}: "
            f"{sorted(grouped_paths)}"
        )
    unique_by_class, duplicate_rows = audit_candidates(grouped_paths)
    for class_name, images in unique_by_class.items():
        if len(images) < samples_per_class:
            raise SystemExit(
                f"Class {class_name!r} has only {len(images)} unique images; "
                f"need {samples_per_class}"
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    train_dir = output_dir / "train_hr"
    val_dir = output_dir / "val_hr"
    train_dir.mkdir()
    val_dir.mkdir()

    rng = random.Random(seed)
    class_names = sorted(unique_by_class)
    class_ids = {name: index for index, name in enumerate(class_names)}
    selected: dict[str, list[SourceImage]] = {"train": [], "val": []}
    selected_by_class: dict[str, dict[str, list[SourceImage]]] = {}

    for class_name in class_names:
        candidates = list(unique_by_class[class_name])
        rng.shuffle(candidates)
        chosen = candidates[:samples_per_class]
        class_train = chosen[:train_per_class]
        class_val = chosen[train_per_class:]
        selected["train"].extend(class_train)
        selected["val"].extend(class_val)
        selected_by_class[class_name] = {"train": class_train, "val": class_val}

    # Remove class ordering from the exported filenames while retaining a fixed seed.
    rng.shuffle(selected["train"])
    rng.shuffle(selected["val"])

    rows: list[dict] = []
    realized_modes = Counter()
    for split_name, images in selected.items():
        split_dir = train_dir if split_name == "train" else val_dir
        for position, item in enumerate(images):
            suffix = item.path.suffix.lower()
            filename = f"{split_name}_{position:04d}{suffix}"
            destination = split_dir / filename
            realized_modes[materialize(item.path, destination, materialize_mode)] += 1
            rows.append(
                {
                    "split": split_name,
                    "split_position": position,
                    "filename": filename,
                    "relative_path": destination.as_posix(),
                    "source_path": item.path.resolve().as_posix(),
                    "source_relative_path": item.path.relative_to(source_dir).as_posix(),
                    "class_id": class_ids[item.class_name],
                    "class_name": item.class_name,
                    "pixel_sha256": item.pixel_sha256,
                    "duplicate_count": 1,
                    "duplicate_splits": split_name,
                    "duplicate_across_splits": False,
                    "exclude_from_eval": False,
                    "exclude_reason": "",
                }
            )

    train_rows = [row for row in rows if row["split"] == "train"]
    val_rows = [row for row in rows if row["split"] == "val"]
    selected_hashes = [row["pixel_sha256"] for row in rows]
    if len(selected_hashes) != len(set(selected_hashes)):
        raise AssertionError("Selected split unexpectedly contains duplicate pixels")

    write_csv(output_dir / "split_manifest.csv", rows)
    write_csv(output_dir / "train_manifest.csv", train_rows)
    write_csv(output_dir / "val_manifest.csv", val_rows)
    write_csv(output_dir / "val_eval_manifest.csv", val_rows)
    if duplicate_rows:
        write_csv(output_dir / "source_duplicate_pixels.csv", duplicate_rows)

    split_json = {
        "seed": seed,
        "source_dir": source_dir.resolve().as_posix(),
        "samples_per_class": samples_per_class,
        "train_per_class": train_per_class,
        "val_per_class": samples_per_class - train_per_class,
        "class_names": class_names,
        "train": [row["source_relative_path"] for row in train_rows],
        "val": [row["source_relative_path"] for row in val_rows],
    }
    (output_dir / "split.json").write_text(
        json.dumps(split_json, indent=2), encoding="utf-8"
    )

    class_counts = {
        split_name: dict(
            sorted(Counter(row["class_name"] for row in split_rows).items())
        )
        for split_name, split_rows in (("train", train_rows), ("val", val_rows))
    }
    summary = {
        "dataset": "AID-2.1K",
        "source_dir": source_dir.resolve().as_posix(),
        "split_seed": seed,
        "num_source_images": sum(len(paths) for paths in grouped_paths.values()),
        "num_source_unique_images": sum(
            len(images) for images in unique_by_class.values()
        ),
        "num_classes": len(class_names),
        "samples_per_class": samples_per_class,
        "num_selected_images": len(rows),
        "num_train_images": len(train_rows),
        "num_val_images": len(val_rows),
        "num_val_eval_images": len(val_rows),
        "num_source_duplicate_pixel_groups": len(duplicate_rows),
        "num_cross_split_duplicate_groups": 0,
        "requested_materialize_mode": materialize_mode,
        "realized_materialize_modes": dict(sorted(realized_modes.items())),
        "class_counts": class_counts,
    }
    (output_dir / "split_manifest_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source_dir", required=True)
    parser.add_argument("--output_dir", default="data/aid2100")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--expected_classes", type=int, default=30)
    parser.add_argument("--samples_per_class", type=int, default=70)
    parser.add_argument("--train_per_class", type=int, default=56)
    parser.add_argument(
        "--materialize",
        choices=["hardlink", "symlink", "copy"],
        default="hardlink",
        help="How selected files are placed in train_hr/val_hr. Hardlink falls back to copy.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = build_split(
        Path(args.source_dir),
        Path(args.output_dir),
        seed=args.seed,
        expected_classes=args.expected_classes,
        samples_per_class=args.samples_per_class,
        train_per_class=args.train_per_class,
        materialize_mode=args.materialize,
    )
    print(json.dumps(summary, indent=2))
    print(f"Wrote AID-2.1K split to {args.output_dir}")


if __name__ == "__main__":
    main()
