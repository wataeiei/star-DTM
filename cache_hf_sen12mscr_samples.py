#!/usr/bin/env python3
"""Cache a few streaming SEN12MS-CR samples from Hugging Face as local .npz files."""

from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path

import numpy as np


def chw_array(array: np.ndarray) -> np.ndarray:
    if array.ndim == 3 and array.shape[-1] in (1, 2, 3, 4, 13):
        return np.transpose(array, (2, 0, 1))
    if array.ndim == 3:
        return array
    raise ValueError(f"Expected HWC or CHW array, got shape={array.shape}")


def normalize_optical(array: np.ndarray) -> np.ndarray:
    array = array.astype(np.float32, copy=False)
    if float(np.nanmax(array)) > 2.0:
        array = array / 10000.0
    return np.clip(array, 0.0, 1.0)


def normalize_sar(array: np.ndarray) -> np.ndarray:
    array = array.astype(np.float32, copy=False)
    if float(np.nanmin(array)) < -1.0 or float(np.nanmax(array)) > 2.0:
        array = np.clip(array, -25.0, 0.0)
        array = (array + 25.0) / 25.0
    return np.clip(array, 0.0, 1.0)


def crop_chw(array: np.ndarray, crop_size: int, rng: random.Random) -> np.ndarray:
    _, height, width = array.shape
    if height < crop_size or width < crop_size:
        raise ValueError(f"Sample {array.shape} is smaller than crop_size={crop_size}")
    top = rng.randint(0, height - crop_size) if height > crop_size else 0
    left = rng.randint(0, width - crop_size) if width > crop_size else 0
    return array[:, top : top + crop_size, left : left + crop_size]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="Hermanni/sen12mscr")
    parser.add_argument("--split", default="train")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--num_samples", type=int, default=8)
    parser.add_argument("--crop_size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    from datasets import load_dataset

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    ds = load_dataset(args.dataset, split=args.split, streaming=True)
    for idx, sample in enumerate(ds):
        if idx >= args.num_samples:
            break

        sar = np.frombuffer(sample["sar"], dtype=np.float32).reshape(sample["sar_shape"])
        cloudy = np.frombuffer(sample["cloudy"], dtype=np.int16).reshape(sample["opt_shape"])
        target = np.frombuffer(sample["target"], dtype=np.int16).reshape(sample["opt_shape"])

        sar = crop_chw(normalize_sar(chw_array(sar)), args.crop_size, rng)
        cloudy = crop_chw(normalize_optical(chw_array(cloudy)), args.crop_size, rng)
        target = crop_chw(normalize_optical(chw_array(target)), args.crop_size, rng)
        mask = np.ones((args.crop_size, args.crop_size), dtype=np.float32)

        path = out_dir / f"sample_{idx:05d}.npz"
        np.savez_compressed(
            path,
            sar=np.ascontiguousarray(sar),
            cloudy=np.ascontiguousarray(cloudy),
            target=np.ascontiguousarray(target),
            mask=mask,
        )
        print(f"wrote {path}")

    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
