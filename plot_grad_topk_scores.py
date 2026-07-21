#!/usr/bin/env python3
"""Plot Gradient-TopK layer scores.

Examples:
  python3 plot_grad_topk_scores.py \
    --score_csv outputs/profile_gradtop8_seed42/grad_topk_scores.csv \
    --output_png outputs/profile_gradtop8_seed42/grad_topk_scores.png

  python3 plot_grad_topk_scores.py \
    --score_csv outputs/profile_gradtop8_seed42/grad_topk_scores.csv \
    --score_csv outputs/profile_gradtop8_seed43/grad_topk_scores.csv \
    --score_csv outputs/profile_gradtop8_seed44/grad_topk_scores.csv \
    --output_png outputs/profile_gradtop8_multiseed_scores.png
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path


def require_matplotlib():
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit("Missing package: matplotlib\nInstall with: pip3 install matplotlib") from exc
    return plt


def read_score_csv(path: str | Path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"Empty score CSV: {path}")
    for row in rows:
        row["_source"] = path.parent.name
        row["block_index"] = int(float(row.get("block_index") or 0))
        row["selected_bool"] = str(row.get("selected", "")).lower() in ("true", "1", "yes")
        for key in ("normalized_grad_score", "selection_score", "grad_norm"):
            row[key] = float(row.get(key) or 0.0)
    return rows


def block_short_name(block: str) -> str:
    parts = block.split(".")
    if block.startswith("down_blocks.") and len(parts) >= 4:
        return f"D{parts[1]}-A{parts[3]}"
    if block.startswith("up_blocks.") and len(parts) >= 4:
        return f"U{parts[1]}-A{parts[3]}"
    if block.startswith("mid_block."):
        return "MID"
    return block.replace(".transformer_blocks.0", "")


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def std(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    m = mean(values)
    return math.sqrt(sum((v - m) ** 2 for v in values) / (len(values) - 1))


def plot_scores(
    score_paths: list[str],
    output_png: str,
    score_key: str,
    title: str,
    dpi: int,
    plot_type: str,
) -> None:
    plt = require_matplotlib()
    all_rows = [row for path in score_paths for row in read_score_csv(path)]
    by_block: dict[str, list[dict]] = defaultdict(list)
    for row in all_rows:
        by_block[row["block"]].append(row)

    blocks = sorted(by_block, key=lambda b: mean([r["block_index"] for r in by_block[b]]))
    labels = [block_short_name(block) for block in blocks]
    values = [mean([r[score_key] for r in by_block[block]]) for block in blocks]
    errors = [std([r[score_key] for r in by_block[block]]) for block in blocks]
    selected_counts = [sum(1 for r in by_block[block] if r["selected_bool"]) for block in blocks]
    colors = ["#2563eb" if count else "#94a3b8" for count in selected_counts]

    fig_width = max(10, 0.65 * len(blocks))
    fig, ax = plt.subplots(figsize=(fig_width, 5.8))
    x = list(range(len(blocks)))
    if plot_type == "bar":
        ax.bar(x, values, yerr=errors if len(score_paths) > 1 else None, color=colors, capsize=4)
    else:
        ax.plot(x, values, color="#2563eb", linewidth=2.2, marker="o", markersize=5)
        if len(score_paths) > 1:
            lower = [v - e for v, e in zip(values, errors)]
            upper = [v + e for v, e in zip(values, errors)]
            ax.fill_between(x, lower, upper, color="#2563eb", alpha=0.16, linewidth=0)
        selected_x = [idx for idx, count in enumerate(selected_counts) if count]
        selected_y = [values[idx] for idx in selected_x]
        ax.scatter(selected_x, selected_y, color="#dc2626", s=58, zorder=3, label="Selected")
        for idx in selected_x:
            ax.axvline(idx, color="#dc2626", alpha=0.12, linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel(score_key)
    ax.set_xlabel("Transformer block")
    ax.set_title(title or f"Gradient-TopK {score_key}")
    ax.grid(axis="y", alpha=0.25)

    selected_label = "Selected"
    not_selected_label = "Not selected"
    from matplotlib.patches import Patch

    if plot_type == "bar":
        ax.legend(
            handles=[
                Patch(facecolor="#2563eb", label=selected_label),
                Patch(facecolor="#94a3b8", label=not_selected_label),
            ],
            loc="upper right",
        )
    else:
        ax.legend(loc="upper right")

    for idx, count in enumerate(selected_counts):
        if count and len(score_paths) > 1:
            ax.text(idx, values[idx], f"{count}/{len(score_paths)}", ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    out = Path(output_png)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi)
    print(f"Wrote {out}")


def write_rank_csv(score_paths: list[str], output_csv: str, score_key: str) -> None:
    all_rows = [row for path in score_paths for row in read_score_csv(path)]
    by_block: dict[str, list[dict]] = defaultdict(list)
    for row in all_rows:
        by_block[row["block"]].append(row)
    rows = []
    for block, items in by_block.items():
        scores = [row[score_key] for row in items]
        rows.append(
            {
                "block": block,
                "short_name": block_short_name(block),
                "block_index": int(mean([row["block_index"] for row in items])),
                f"mean_{score_key}": mean(scores),
                f"std_{score_key}": std(scores),
                "selected_count": sum(1 for row in items if row["selected_bool"]),
                "num_runs": len(items),
            }
        )
    rows.sort(key=lambda row: row[f"mean_{score_key}"], reverse=True)
    out = Path(output_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--score_csv", action="append", required=True, help="Repeatable grad_topk_scores.csv path")
    parser.add_argument("--output_png", required=True)
    parser.add_argument("--output_rank_csv", default="")
    parser.add_argument(
        "--score_key",
        default="normalized_grad_score",
        choices=["normalized_grad_score", "selection_score", "grad_norm"],
    )
    parser.add_argument("--title", default="")
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument("--plot_type", default="line", choices=["line", "bar"])
    args = parser.parse_args()

    plot_scores(args.score_csv, args.output_png, args.score_key, args.title, args.dpi, args.plot_type)
    if args.output_rank_csv:
        write_rank_csv(args.score_csv, args.output_rank_csv, args.score_key)


if __name__ == "__main__":
    main()
