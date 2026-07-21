#!/usr/bin/env python3
"""Export a Hugging Face image dataset to a local image folder.

This is useful for keeping all profiling scripts on the same simple interface:

  --data_dir path/to/images

Example:
  python3 export_hf_image_dataset.py \
    --dataset_id blanchon/UC_Merced \
    --split train \
    --output_dir data/ucmerced_hf/train_hr \
    --image_column image \
    --label_column label \
    --max_images 0
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_id", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--image_column", default="image")
    parser.add_argument("--label_column", default="")
    parser.add_argument("--filename_column", default="")
    parser.add_argument("--max_images", type=int, default=0, help="0 means export all images")
    parser.add_argument("--image_format", default="png", choices=["png", "jpg"])
    args = parser.parse_args()

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit("Missing package: datasets\nInstall with: pip3 install datasets") from exc

    out_dir = ensure_dir(args.output_dir)
    dataset = load_dataset(args.dataset_id, split=args.split)
    if args.max_images > 0:
        dataset = dataset.select(range(min(args.max_images, len(dataset))))

    label_names = None
    if args.label_column:
        feature = dataset.features.get(args.label_column)
        label_names = getattr(feature, "names", None)

    rows = []
    for idx, item in enumerate(dataset):
        image = item[args.image_column].convert("RGB")
        label_value = item.get(args.label_column) if args.label_column else None
        if label_names is not None and isinstance(label_value, int):
            label_name = str(label_names[label_value])
        elif label_value is None:
            label_name = "images"
        else:
            label_name = str(label_value)

        class_dir = ensure_dir(out_dir / label_name)
        if args.filename_column and item.get(args.filename_column):
            stem = Path(str(item[args.filename_column])).stem
        else:
            stem = f"{idx:06d}"
        filename = f"{stem}.{args.image_format}"
        save_path = class_dir / filename
        image.save(save_path)
        rows.append(
            {
                "index": idx,
                "path": str(save_path),
                "label": label_value,
                "label_name": label_name,
            }
        )

    metadata = {
        "dataset_id": args.dataset_id,
        "split": args.split,
        "image_column": args.image_column,
        "label_column": args.label_column,
        "num_images": len(rows),
    }
    (out_dir / "export_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    with (out_dir / "export_manifest.csv").open("w", encoding="utf-8") as f:
        f.write("index,path,label,label_name\n")
        for row in rows:
            f.write(f"{row['index']},{row['path']},{row['label']},{row['label_name']}\n")
    print(f"Exported {len(rows)} images to {out_dir}")


if __name__ == "__main__":
    main()
