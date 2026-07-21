#!/usr/bin/env python3
"""Plot DiT gradient profiling scores as line charts."""

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


def read_csv(path: str | Path) -> list[dict]:
    path = Path(path)
    with path.open("r", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        row["block_index"] = int(float(row.get("block_index") or 0))
        row["selected_bool"] = str(row.get("selected", "")).lower() in ("true", "1", "yes")
        for key in ("normalized_grad_score", "selection_score", "grad_norm"):
            row[key] = float(row.get(key) or 0.0)
    return sorted(rows, key=lambda row: row["block_index"])


def short_label(block: str) -> str:
    parts = block.split(".")
    for marker in ("blocks", "layers", "transformer_blocks", "dit_blocks"):
        if marker in parts:
            idx = parts.index(marker)
            if idx + 1 < len(parts):
                return f"{marker[:1].upper()}{parts[idx + 1]}"
    return block.replace(".transformer_blocks.0", "")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--score_csv", action="append", required=True)
    parser.add_argument("--label", action="append", default=[])
    parser.add_argument("--output_png", required=True)
    parser.add_argument("--score_key", default="normalized_grad_score", choices=["normalized_grad_score", "selection_score", "grad_norm"])
    parser.add_argument("--title", default="DiT Gradient Layer Scores")
    parser.add_argument("--dpi", type=int, default=180)
    args = parser.parse_args()

    plt = require_matplotlib()
    labels = args.label or [Path(p).parent.name for p in args.score_csv]
    if len(labels) != len(args.score_csv):
        raise SystemExit("--label count must match --score_csv count")

    all_rows = [read_csv(path) for path in args.score_csv]
    x_labels = [short_label(row["block"]) for row in all_rows[0]]
    x = list(range(len(x_labels)))

    fig_width = max(10, 0.65 * len(x_labels))
    fig, ax = plt.subplots(figsize=(fig_width, 5.8))
    colors = ["#2563eb", "#dc2626", "#16a34a", "#9333ea", "#ea580c", "#0891b2"]
    for idx, (rows, label) in enumerate(zip(all_rows, labels)):
        color = colors[idx % len(colors)]
        y = [row[args.score_key] for row in rows]
        ax.plot(x, y, marker="o", linewidth=1.9, markersize=4.8, label=label, color=color)
        sx = [i for i, row in enumerate(rows) if row["selected_bool"]]
        sy = [y[i] for i in sx]
        ax.scatter(sx, sy, s=56, facecolors="white", edgecolors=color, linewidths=1.8, zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=45, ha="right")
    ax.set_xlabel("DiT block")
    ax.set_ylabel(args.score_key)
    ax.set_title(args.title)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="upper right")
    fig.tight_layout()

    out = Path(args.output_png)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=args.dpi)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
