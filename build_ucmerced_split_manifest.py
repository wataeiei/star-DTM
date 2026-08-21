#!/usr/bin/env python3
"""Recover UC Merced labels for an exported split and audit pixel duplicates."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path


def pixel_sha256(path: Path) -> str:
    from PIL import Image

    with Image.open(path) as image:
        rgb = image.convert("RGB")
        digest = hashlib.sha256()
        digest.update(f"RGB:{rgb.width}x{rgb.height}:".encode("ascii"))
        digest.update(rgb.tobytes())
        return digest.hexdigest()


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"Refusing to write an empty manifest: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_root", default="data/ucmerced")
    parser.add_argument("--dataset_name", default="blanchon/UC_Merced")
    parser.add_argument("--dataset_split", default="train")
    parser.add_argument("--label_column", default="label")
    parser.add_argument("--output_csv", default="")
    parser.add_argument("--summary_json", default="")
    return parser.parse_args()


def main() -> None:
    from datasets import load_dataset

    args = parse_args()
    root = Path(args.data_root)
    split_path = root / "split.json"
    if not split_path.is_file():
        raise SystemExit(f"Missing split file: {split_path}")

    split = json.loads(split_path.read_text(encoding="utf-8"))
    for name in ("train", "val"):
        if name not in split or not isinstance(split[name], list):
            raise SystemExit(f"{split_path} does not contain a valid '{name}' index list")

    dataset = load_dataset(args.dataset_name, split=args.dataset_split)
    label_feature = dataset.features.get(args.label_column)
    label_names = getattr(label_feature, "names", None)
    if not label_names:
        raise SystemExit(
            f"Dataset column '{args.label_column}' is not a ClassLabel with names"
        )

    rows: list[dict] = []
    for split_name in ("train", "val"):
        indices = split[split_name]
        image_dir = root / f"{split_name}_hr"
        for position, source_index in enumerate(indices):
            filename = f"{split_name}_{position:04d}.png"
            image_path = image_dir / filename
            if not image_path.is_file():
                raise SystemExit(f"Missing exported image: {image_path}")
            if not 0 <= int(source_index) < len(dataset):
                raise SystemExit(f"Source index out of range: {source_index}")
            label_id = int(dataset[int(source_index)][args.label_column])
            rows.append(
                {
                    "split": split_name,
                    "split_position": position,
                    "filename": filename,
                    "relative_path": image_path.as_posix(),
                    "source_index": int(source_index),
                    "class_id": label_id,
                    "class_name": label_names[label_id],
                    "pixel_sha256": pixel_sha256(image_path),
                }
            )

    by_hash: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_hash[row["pixel_sha256"]].append(row)

    for group in by_hash.values():
        splits = sorted({row["split"] for row in group})
        val_rows = sorted(
            (row for row in group if row["split"] == "val"),
            key=lambda row: int(row["split_position"]),
        )
        has_train = any(row["split"] == "train" for row in group)
        for row in group:
            exclude = False
            reason = ""
            if row["split"] == "val" and has_train:
                exclude = True
                reason = "pixel_duplicate_in_train"
            elif row["split"] == "val" and len(val_rows) > 1 and row is not val_rows[0]:
                exclude = True
                reason = "duplicate_of_earlier_val_image"
            row.update(
                {
                    "duplicate_count": len(group),
                    "duplicate_splits": ";".join(splits),
                    "duplicate_across_splits": len(splits) > 1,
                    "exclude_from_eval": exclude,
                    "exclude_reason": reason,
                }
            )

    output_csv = Path(args.output_csv) if args.output_csv else root / "split_manifest.csv"
    summary_json = (
        Path(args.summary_json) if args.summary_json else root / "split_manifest_summary.json"
    )
    train_rows = [row for row in rows if row["split"] == "train"]
    val_rows = [row for row in rows if row["split"] == "val"]
    val_eval_rows = [row for row in val_rows if not row["exclude_from_eval"]]

    write_csv(output_csv, rows)
    write_csv(root / "train_manifest.csv", train_rows)
    write_csv(root / "val_manifest.csv", val_rows)
    write_csv(root / "val_eval_manifest.csv", val_eval_rows)

    duplicate_groups = [group for group in by_hash.values() if len(group) > 1]
    cross_split_groups = [
        group for group in duplicate_groups
        if {row["split"] for row in group} == {"train", "val"}
    ]
    class_counts = {
        "train": dict(sorted(Counter(row["class_name"] for row in train_rows).items())),
        "val": dict(sorted(Counter(row["class_name"] for row in val_rows).items())),
        "val_eval": dict(
            sorted(Counter(row["class_name"] for row in val_eval_rows).items())
        ),
    }
    summary = {
        "dataset_name": args.dataset_name,
        "dataset_split": args.dataset_split,
        "split_seed": split.get("seed"),
        "num_dataset_images": len(dataset),
        "num_train_images": len(train_rows),
        "num_val_images": len(val_rows),
        "num_val_eval_images": len(val_eval_rows),
        "num_duplicate_pixel_groups": len(duplicate_groups),
        "num_cross_split_duplicate_groups": len(cross_split_groups),
        "excluded_val_images": [
            {
                "filename": row["filename"],
                "class_name": row["class_name"],
                "reason": row["exclude_reason"],
                "pixel_sha256": row["pixel_sha256"],
            }
            for row in val_rows
            if row["exclude_from_eval"]
        ],
        "class_counts": class_counts,
    }
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Wrote {output_csv}")
    print(f"Wrote {root / 'train_manifest.csv'}")
    print(f"Wrote {root / 'val_manifest.csv'}")
    print(f"Wrote {root / 'val_eval_manifest.csv'}")
    print(f"Wrote {summary_json}")
    print(
        f"train={len(train_rows)} val={len(val_rows)} "
        f"val_eval={len(val_eval_rows)} "
        f"cross_split_duplicate_groups={len(cross_split_groups)}"
    )
    for row in val_rows:
        if row["exclude_from_eval"]:
            print(
                f"exclude {row['filename']}: {row['exclude_reason']} "
                f"class={row['class_name']}"
            )


if __name__ == "__main__":
    main()
