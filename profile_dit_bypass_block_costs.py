#!/usr/bin/env python3
"""Profile the backward-bypass cost of each frozen DiT block."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path

import torch

import adaptive_grad_blockskip as adaptive
import profile_dit_gradskip_compute as audit


def read_csv(path: Path) -> list[dict]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def paired_timed(
    model,
    controller,
    loss_fn,
    batch,
    ratio: float,
    rng_state,
    device: torch.device,
    repeats: int,
) -> tuple[list[float], list[float], list[float], list[float]]:
    full_ms: list[float] = []
    bypass_ms: list[float] = []
    full_losses: list[float] = []
    bypass_losses: list[float] = []

    for repeat in range(repeats):
        modes = ("full", "single_skip")
        if repeat % 2:
            modes = tuple(reversed(modes))
        for mode in modes:
            audit.sync(device)
            started = audit.time.perf_counter()
            loss = audit.run_step(
                model, controller, loss_fn, batch, ratio, mode, rng_state, device
            )
            elapsed_ms = (audit.time.perf_counter() - started) * 1000.0
            if mode == "full":
                full_ms.append(elapsed_ms)
                full_losses.append(loss)
            else:
                bypass_ms.append(elapsed_ms)
                bypass_losses.append(loss)
    return full_ms, bypass_ms, full_losses, bypass_losses


def stdev(values: list[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else 0.0


def join_importance(
    cost_rows: list[dict],
    importance_path: Path,
    importance_step: int,
) -> list[dict]:
    importance_rows = read_csv(importance_path)
    required = {
        "train_step",
        "noise_ratio",
        "block",
        "normalized_grad_score",
    }
    if not importance_rows:
        raise SystemExit(f"Importance CSV is empty: {importance_path}")
    missing = sorted(required - set(importance_rows[0]))
    if missing:
        raise SystemExit(
            "Importance CSV is missing columns: " + ", ".join(missing)
        )

    selected_importance = [
        row
        for row in importance_rows
        if int(row["train_step"]) == importance_step
    ]
    if not selected_importance:
        available = sorted({int(row["train_step"]) for row in importance_rows})
        raise SystemExit(
            f"Importance step {importance_step} is unavailable; available={available}"
        )

    cost_by_block = {str(row["block"]): row for row in cost_rows}
    joined = []
    for importance in selected_importance:
        block = str(importance["block"])
        cost = cost_by_block.get(block)
        if cost is None:
            continue
        importance_value = float(importance["normalized_grad_score"])
        saved_gflops = float(cost["reported_gflops_saved"])
        saved_ms = float(cost["step_time_saved_ms"])
        joined.append(
            {
                "train_step": importance_step,
                "noise_ratio": float(importance["noise_ratio"]),
                "block": block,
                "block_index": int(cost["block_index"]),
                "normalized_grad_score": importance_value,
                "reported_gflops_saved": saved_gflops,
                "step_time_saved_ms": saved_ms,
                "peak_cuda_mem_saved_mb": float(cost["peak_cuda_mem_saved_mb"]),
                "importance_per_saved_gflop": (
                    importance_value / saved_gflops if saved_gflops > 0 else float("inf")
                ),
                "importance_per_saved_ms": (
                    importance_value / saved_ms if saved_ms > 0 else float("inf")
                ),
            }
        )
    return sorted(joined, key=lambda row: (row["noise_ratio"], row["block_index"]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=["dit4sr", "dit-sr"])
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--importance_csv", default="")
    parser.add_argument("--importance_step", type=int, default=0)
    parser.add_argument("--noise_ratio", type=float, default=0.4)
    parser.add_argument("--protected_block", action="append", default=[])
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=4242)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--config_path", default="")
    parser.add_argument("--ckpt_path", default="")
    parser.add_argument("--autoencoder_ckpt", default="")
    args = parser.parse_args()

    if args.warmup < 0 or args.repeats <= 0:
        raise SystemExit("Require --warmup >= 0 and --repeats > 0")

    run_dir = Path(args.run_dir)
    metadata_path = run_dir / "metadata.json"
    adapter_path = run_dir / "lora_adapter.pt"
    if not metadata_path.is_file() or not adapter_path.is_file():
        raise SystemExit(f"Missing metadata.json or lora_adapter.pt in {run_dir}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    core_seed = int(metadata.get("seed", args.seed))
    torch.manual_seed(core_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(core_seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    model, controller, loader, loss_fn, candidates, selected, keepalive = (
        audit.load_experiment(args, metadata, device)
    )
    try:
        batch = next(loader)
    except StopIteration as exc:
        raise SystemExit(f"No images found in {args.data_dir}") from exc

    protected = set(args.protected_block)
    unknown_protected = sorted(protected - set(candidates))
    if unknown_protected:
        raise SystemExit(f"Unknown protected blocks: {unknown_protected}")
    frozen_candidates = [
        block
        for block in candidates
        if block not in set(selected) and block not in protected
    ]
    if not frozen_candidates:
        raise SystemExit("No frozen bypass candidates remain after protection.")

    rows = []
    for block in frozen_candidates:
        controller.configure([block])
        rng_state = adaptive.snapshot_rng(device)
        for warmup_index in range(args.warmup):
            modes = ("full", "single_skip")
            if warmup_index % 2:
                modes = tuple(reversed(modes))
            for mode in modes:
                audit.run_step(
                    model,
                    controller,
                    loss_fn,
                    batch,
                    args.noise_ratio,
                    mode,
                    rng_state,
                    device,
                )

        full_ms, bypass_ms, full_losses, bypass_losses = paired_timed(
            model,
            controller,
            loss_fn,
            batch,
            args.noise_ratio,
            rng_state,
            device,
            args.repeats,
        )
        full_flops, full_kernel, full_peak, _ = audit.profiled(
            model,
            controller,
            loss_fn,
            batch,
            args.noise_ratio,
            "full",
            rng_state,
            device,
        )
        bypass_flops, bypass_kernel, bypass_peak, _ = audit.profiled(
            model,
            controller,
            loss_fn,
            batch,
            args.noise_ratio,
            "single_skip",
            rng_state,
            device,
        )

        full_mean = statistics.mean(full_ms)
        bypass_mean = statistics.mean(bypass_ms)
        row = {
            "block": block,
            "block_index": candidates.index(block),
            "profile_noise_ratio": args.noise_ratio,
            "full_step_time_ms": full_mean,
            "bypass_step_time_ms": bypass_mean,
            "step_time_saved_ms": full_mean - bypass_mean,
            "step_time_reduction_pct": audit.reduction(full_mean, bypass_mean),
            "full_step_time_std_ms": stdev(full_ms),
            "bypass_step_time_std_ms": stdev(bypass_ms),
            "full_reported_gflops": full_flops / 1e9,
            "bypass_reported_gflops": bypass_flops / 1e9,
            "reported_gflops_saved": (full_flops - bypass_flops) / 1e9,
            "reported_flops_reduction_pct": audit.reduction(full_flops, bypass_flops),
            "full_cuda_kernel_time_ms": full_kernel,
            "bypass_cuda_kernel_time_ms": bypass_kernel,
            "cuda_kernel_time_saved_ms": full_kernel - bypass_kernel,
            "full_peak_cuda_mem_mb": full_peak,
            "bypass_peak_cuda_mem_mb": bypass_peak,
            "peak_cuda_mem_saved_mb": full_peak - bypass_peak,
            "max_loss_abs_diff": max(
                abs(full - bypass)
                for full, bypass in zip(full_losses, bypass_losses)
            ),
        }
        rows.append(row)
        print(
            f"{block}: saved={row['reported_gflops_saved']:.3f} GFLOPs "
            f"time={row['step_time_saved_ms']:.3f} ms "
            f"memory={row['peak_cuda_mem_saved_mb']:.1f} MB"
        )

    rows.sort(key=lambda row: row["block_index"])
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "block_backward_costs.csv", rows)

    if args.importance_csv:
        joined = join_importance(
            rows, Path(args.importance_csv), args.importance_step
        )
        write_csv(output_dir / "importance_compute_tradeoff.csv", joined)

    summary = {
        "model": args.model,
        "run_dir": str(run_dir),
        "profile_noise_ratio": args.noise_ratio,
        "importance_csv": args.importance_csv,
        "importance_step": args.importance_step,
        "total_blocks": len(candidates),
        "selected_lora_blocks": list(selected),
        "additional_protected_blocks": sorted(protected),
        "profiled_frozen_blocks": frozen_candidates,
        "num_profiled_frozen_blocks": len(frozen_candidates),
        "warmup": args.warmup,
        "repeats": args.repeats,
        "profiler_note": (
            "Reported FLOPs cover only operators supported by PyTorch profiler. "
            "Single-pass bypass still executes the forward path; saved FLOPs are "
            "the reported difference in the training step."
        ),
    }
    (output_dir / "block_backward_costs_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(f"Wrote block costs to {output_dir}")


if __name__ == "__main__":
    main()
