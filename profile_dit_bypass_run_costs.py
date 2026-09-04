#!/usr/bin/env python3
"""Profile the real compute savings of every legal contiguous bypass run."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import torch

import adaptive_grad_blockskip as adaptive
import profile_dit_gradskip_compute as compute


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=["dit4sr", "dit-sr"])
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--noise_ratio", type=float, default=0.4)
    parser.add_argument("--min_run", type=int, default=2)
    parser.add_argument("--max_run", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=4242)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--config_path", default="")
    parser.add_argument("--ckpt_path", default="")
    parser.add_argument("--autoencoder_ckpt", default="")
    return parser.parse_args()


def architecture_group(block: str) -> str:
    parts = block.split(".")
    if parts[0] in {"input_blocks", "output_blocks"} and len(parts) > 1:
        return ".".join(parts[:2])
    if parts[0] == "middle_block":
        return "middle_block"
    return parts[0]


def legal_runs(
    candidates: list[str], protected: set[str], min_run: int, max_run: int
) -> list[list[str]]:
    segments: list[list[str]] = []
    current: list[str] = []
    previous_index = -2
    previous_group = ""
    for index, block in enumerate(candidates):
        group = architecture_group(block)
        if block in protected:
            if current:
                segments.append(current)
                current = []
            previous_index = -2
            previous_group = ""
            continue
        if current and (index != previous_index + 1 or group != previous_group):
            segments.append(current)
            current = []
        current.append(block)
        previous_index = index
        previous_group = group
    if current:
        segments.append(current)

    runs = []
    for segment in segments:
        for length in range(min_run, max_run + 1):
            for start in range(0, len(segment) - length + 1):
                runs.append(segment[start : start + length])
    return runs


def mean(values) -> float:
    return statistics.mean(values)


def main() -> None:
    args = parse_args()
    if args.min_run < 1 or args.max_run < args.min_run:
        raise SystemExit("Require 1 <= --min_run <= --max_run.")
    if args.warmup < 0 or args.repeats < 1:
        raise SystemExit("Require --warmup >= 0 and --repeats >= 1.")
    if not 0.0 <= args.noise_ratio <= 1.0:
        raise SystemExit("--noise_ratio must be in [0, 1].")

    run_dir = Path(args.run_dir)
    metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
    model_seed = int(metadata.get("seed", args.seed))
    torch.manual_seed(model_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(model_seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model, controller, loader, loss_fn, candidates, selected, keepalive = (
        compute.load_experiment(args, metadata, device)
    )
    try:
        batch = next(loader)
    except StopIteration as exc:
        raise SystemExit(f"No images found in {args.data_dir}") from exc

    protected = set(selected)
    runs = legal_runs(candidates, protected, args.min_run, args.max_run)
    if not runs:
        raise SystemExit("No legal contiguous bypass runs were found.")
    rng_state = adaptive.snapshot_rng(device)

    controller.configure([])
    for _ in range(args.warmup):
        compute.run_step(
            model, controller, loss_fn, batch, args.noise_ratio, "full", rng_state, device
        )
    full_ms, full_losses = compute.timed(
        model,
        controller,
        loss_fn,
        batch,
        args.noise_ratio,
        "full",
        rng_state,
        device,
        args.repeats,
    )
    full_flops, full_kernel_ms, full_peak_mb, full_profile_loss = compute.profiled(
        model,
        controller,
        loss_fn,
        batch,
        args.noise_ratio,
        "full",
        rng_state,
        device,
    )
    full_time_ms = mean(full_ms)

    rows = []
    candidate_index = {block: index for index, block in enumerate(candidates)}
    for run_index, blocks in enumerate(runs, 1):
        controller.configure(blocks)
        for _ in range(args.warmup):
            compute.run_step(
                model,
                controller,
                loss_fn,
                batch,
                args.noise_ratio,
                "single_skip",
                rng_state,
                device,
            )
        bypass_ms, bypass_losses = compute.timed(
            model,
            controller,
            loss_fn,
            batch,
            args.noise_ratio,
            "single_skip",
            rng_state,
            device,
            args.repeats,
        )
        bypass_flops, bypass_kernel_ms, bypass_peak_mb, bypass_profile_loss = (
            compute.profiled(
                model,
                controller,
                loss_fn,
                batch,
                args.noise_ratio,
                "single_skip",
                rng_state,
                device,
            )
        )
        stats = controller.stats(0.0)
        bypass_time_ms = mean(bypass_ms)
        saved_flops = full_flops - bypass_flops
        row = {
            "run_id": run_index,
            "architecture_group": architecture_group(blocks[0]),
            "start_block_index": candidate_index[blocks[0]],
            "end_block_index": candidate_index[blocks[-1]],
            "run_length": len(blocks),
            "block_names": ";".join(blocks),
            "full_reported_gflops": full_flops / 1e9,
            "bypass_reported_gflops": bypass_flops / 1e9,
            "saved_reported_gflops": saved_flops / 1e9,
            "reported_flops_reduction_pct": compute.reduction(full_flops, bypass_flops),
            "full_step_time_ms": full_time_ms,
            "bypass_step_time_ms": bypass_time_ms,
            "step_time_reduction_pct": compute.reduction(full_time_ms, bypass_time_ms),
            "full_cuda_kernel_time_ms": full_kernel_ms,
            "bypass_cuda_kernel_time_ms": bypass_kernel_ms,
            "cuda_kernel_time_reduction_pct": compute.reduction(
                full_kernel_ms, bypass_kernel_ms
            ),
            "full_peak_cuda_mem_mb": full_peak_mb,
            "bypass_peak_cuda_mem_mb": bypass_peak_mb,
            "peak_cuda_memory_reduction_pct": compute.reduction(
                full_peak_mb, bypass_peak_mb
            ),
            "max_timed_loss_abs_diff": max(
                abs(full - bypass)
                for full, bypass in zip(full_losses, bypass_losses)
            ),
            "profile_loss_abs_diff": abs(full_profile_loss - bypass_profile_loss),
            "fallback_blocks": stats.fallback_blocks,
            "replayable_blocks": stats.replayable_blocks,
        }
        rows.append(row)
        print(
            f"[{run_index:03d}/{len(runs):03d}] len={len(blocks)} "
            f"blocks={blocks[0]}..{blocks[-1]} "
            f"FLOPs={row['reported_flops_reduction_pct']:.3f}% "
            f"time={row['step_time_reduction_pct']:.3f}%"
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    compute.write_csv(output_dir / "bypass_run_compute_costs.csv", rows)
    by_length = []
    for length in sorted({row["run_length"] for row in rows}):
        values = [row for row in rows if row["run_length"] == length]
        by_length.append(
            {
                "run_length": length,
                "num_runs": len(values),
                "mean_saved_reported_gflops": mean(
                    row["saved_reported_gflops"] for row in values
                ),
                "mean_reported_flops_reduction_pct": mean(
                    row["reported_flops_reduction_pct"] for row in values
                ),
                "mean_step_time_reduction_pct": mean(
                    row["step_time_reduction_pct"] for row in values
                ),
                "mean_peak_cuda_memory_reduction_pct": mean(
                    row["peak_cuda_memory_reduction_pct"] for row in values
                ),
            }
        )
    compute.write_csv(output_dir / "bypass_run_compute_by_length.csv", by_length)
    summary = {
        "model": args.model,
        "run_dir": str(run_dir),
        "noise_ratio": args.noise_ratio,
        "total_candidate_blocks": len(candidates),
        "protected_lora_blocks": sorted(protected),
        "num_legal_runs": len(runs),
        "run_lengths": [args.min_run, args.max_run],
        "full_reported_gflops": full_flops / 1e9,
        "full_step_time_ms": full_time_ms,
        "full_peak_cuda_mem_mb": full_peak_mb,
        "profiler_note": (
            "Reported FLOPs include only operators supported by PyTorch profiler. "
            "Wall time and CUDA memory are measured separately."
        ),
    }
    (output_dir / "bypass_run_compute_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    print(f"Wrote contiguous-run compute costs to {output_dir}")


if __name__ == "__main__":
    main()
