#!/usr/bin/env python3
"""Overlay gradient-profile curves from different models or datasets.

Use this when curves have different block names/counts, for example:

  DiT4SR transformer_blocks.0..23
  Stable Diffusion UNet down/mid/up attention blocks

The x-axis is relative depth in [0, 1]. The y-axis can be kept raw or normalized
per curve so architectural patterns can be compared fairly.
"""

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


def read_score_csv(path: str | Path, score_key: str) -> list[dict]:
    path = Path(path)
    with path.open("r", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit(f"Empty score CSV: {path}")
    for row in rows:
        row["block_index"] = int(float(row.get("block_index") or 0))
        row["score"] = float(row.get(score_key) or 0.0)
        row["selected_bool"] = str(row.get("selected", "")).lower() in ("true", "1", "yes")
    return sorted(rows, key=lambda row: row["block_index"])


def short_block_label(block: str) -> str:
    parts = block.split(".")
    if block.startswith("down_blocks.") and len(parts) >= 4:
        return f"D{parts[1]}-A{parts[3]}"
    if block.startswith("up_blocks.") and len(parts) >= 4:
        return f"U{parts[1]}-A{parts[3]}"
    if block.startswith("mid_block."):
        return "MID"
    if "transformer_blocks" in parts:
        idx = parts.index("transformer_blocks")
        if idx + 1 < len(parts):
            return f"T{parts[idx + 1]}"
    return block


def normalize_scores(values: list[float], mode: str) -> list[float]:
    if mode == "none":
        return values
    if mode == "max":
        denom = max(values) if values else 0.0
        return [v / denom if denom else 0.0 for v in values]
    if mode == "sum":
        denom = sum(values)
        return [v / denom if denom else 0.0 for v in values]
    raise SystemExit(f"Unknown normalize mode: {mode}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--score_csv", action="append", required=True)
    parser.add_argument("--label", action="append", default=[])
    parser.add_argument("--output_png", required=True)
    parser.add_argument("--score_key", default="normalized_grad_score", choices=["normalized_grad_score", "selection_score", "grad_norm"])
    parser.add_argument("--normalize_y", default="max", choices=["max", "sum", "none"])
    parser.add_argument("--title", default="Cross-model Gradient Score Patterns")
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument("--annotate_topk", type=int, default=5)
    args = parser.parse_args()

    labels = args.label or [Path(path).parent.name for path in args.score_csv]
    if len(labels) != len(args.score_csv):
        raise SystemExit("--label count must match --score_csv count")

    plt = require_matplotlib()
    fig, ax = plt.subplots(figsize=(12.5, 6.0))
    colors = ["#2563eb", "#dc2626", "#16a34a", "#9333ea", "#ea580c", "#0891b2", "#be123c"]

    for idx, (path, label) in enumerate(zip(args.score_csv, labels)):
        rows = read_score_csv(path, args.score_key)
        scores = normalize_scores([row["score"] for row in rows], args.normalize_y)
        denom = max(len(rows) - 1, 1)
        xs = [i / denom for i in range(len(rows))]
        color = colors[idx % len(colors)]
        ax.plot(xs, scores, marker="o", linewidth=2.0, markersize=4.8, color=color, label=label)

        selected_x = [xs[i] for i, row in enumerate(rows) if row["selected_bool"]]
        selected_y = [scores[i] for i, row in enumerate(rows) if row["selected_bool"]]
        ax.scatter(selected_x, selected_y, s=58, facecolors="white", edgecolors=color, linewidths=1.8, zorder=3)

        if args.annotate_topk > 0:
            top_indices = sorted(range(len(rows)), key=lambda i: scores[i], reverse=True)[: args.annotate_topk]
            for i in top_indices:
                ax.annotate(
                    short_block_label(rows[i]["block"]),
                    (xs[i], scores[i]),
                    textcoords="offset points",
                    xytext=(0, 8),
                    ha="center",
                    fontsize=8,
                    color=color,
                )

    ax.set_xlabel("Relative depth")
    ylabel = args.score_key if args.normalize_y == "none" else f"{args.score_key} ({args.normalize_y}-normalized per curve)"
    ax.set_ylabel(ylabel)
    ax.set_title(args.title)
    ax.set_xlim(-0.02, 1.02)
    ax.grid(axis="both", alpha=0.25)
    ax.legend(loc="upper left")
    fig.tight_layout()

    out = Path(args.output_png)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=args.dpi)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
