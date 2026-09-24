import csv
import json
import math
import statistics
import time
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile

import adaptive_grad_blockskip as adaptive


def configure_bypass_cost_probe(args):
    enabled = bool(getattr(args, "bypass_cost_probe_only", False))
    if not enabled:
        return None

    selection_path = Path(args.lora_selection_file)
    if not selection_path.is_file():
        raise FileNotFoundError(
            f"LoRA selection file not found: {selection_path}"
        )
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    selected = sorted(
        int(value) for value in selection["selected_block_indices"]
    )
    frozen = sorted(set(range(28)) - set(selected))
    declared_frozen = sorted(
        int(value)
        for value in selection.get("frozen_block_indices", frozen)
    )
    if frozen != declared_frozen:
        raise ValueError(
            "Selection file frozen blocks do not match the complement "
            "of selected blocks"
        )
    if not frozen:
        raise ValueError("No frozen blocks remain for bypass profiling")

    warmup = int(getattr(args, "bypass_cost_warmup", 3))
    repeats = int(getattr(args, "bypass_cost_repeats", 10))
    flow_t = float(getattr(args, "bypass_cost_flow_t", 0.4))
    if warmup < 0 or repeats <= 0:
        raise ValueError("Require warmup >= 0 and repeats > 0")
    if not 0.0 <= flow_t <= 1.0:
        raise ValueError("bypass_cost_flow_t must be in [0, 1]")

    args.gradient_accumulation_steps = 1
    args.use_ema = False
    args.report_to = None
    args.resume_ckpt = None

    return {
        "selection_path": str(selection_path),
        "selection": selection,
        "selected_indices": selected,
        "frozen_indices": frozen,
        "warmup": warmup,
        "repeats": repeats,
        "flow_t": flow_t,
        "importance_csv": str(
            getattr(args, "bypass_cost_importance_csv", "")
        ),
    }


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _clear_grads(model):
    for parameter in model.parameters():
        parameter.grad = None


def _stdev(values):
    return statistics.stdev(values) if len(values) > 1 else 0.0


def _reduction(full, bypass):
    return (full - bypass) / full * 100.0 if full > 0 else float("nan")


def _profiler_flops(profiler):
    return float(
        sum(
            float(getattr(event, "flops", 0.0) or 0.0)
            for event in profiler.key_averages()
        )
    )


def _kernel_time_ms(profiler):
    total_us = 0.0
    for event in profiler.events():
        if "cuda" not in str(
            getattr(event, "device_type", "")
        ).lower():
            continue
        time_range = getattr(event, "time_range", None)
        if time_range is not None and hasattr(time_range, "elapsed_us"):
            total_us += float(time_range.elapsed_us())
        else:
            total_us += float(
                getattr(event, "device_time_total", 0.0) or 0.0
            )
    return total_us / 1000.0


def _run_step(model, controller, loss_fn, mode, rng_state, device):
    _clear_grads(model)
    controller.set_mode(mode)
    adaptive.restore_rng(rng_state)
    loss = loss_fn()
    if not torch.isfinite(loss):
        raise RuntimeError(f"Non-finite loss in mode={mode}")
    loss.backward()
    _sync(device)
    stats = controller.stats(0.0)
    return float(loss.detach().float().cpu()), stats


def _timed_pair(
    model,
    controller,
    loss_fn,
    rng_state,
    device,
    repeats,
):
    full_ms = []
    bypass_ms = []
    full_losses = []
    bypass_losses = []
    bypass_stats = []

    for repeat in range(repeats):
        modes = ("full", "single_skip")
        if repeat % 2:
            modes = tuple(reversed(modes))
        for mode in modes:
            _sync(device)
            started = time.perf_counter()
            loss, stats = _run_step(
                model,
                controller,
                loss_fn,
                mode,
                rng_state,
                device,
            )
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            if mode == "full":
                full_ms.append(elapsed_ms)
                full_losses.append(loss)
            else:
                bypass_ms.append(elapsed_ms)
                bypass_losses.append(loss)
                bypass_stats.append(stats)
    return full_ms, bypass_ms, full_losses, bypass_losses, bypass_stats


def _profiled(model, controller, loss_fn, mode, rng_state, device):
    activities = [ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(ProfilerActivity.CUDA)
        torch.cuda.reset_peak_memory_stats(device)
    with profile(
        activities=activities,
        with_flops=True,
        profile_memory=True,
    ) as profiler:
        loss, stats = _run_step(
            model,
            controller,
            loss_fn,
            mode,
            rng_state,
            device,
        )
    peak_mb = (
        torch.cuda.max_memory_allocated(device) / 2**20
        if device.type == "cuda"
        else 0.0
    )
    return {
        "flops": _profiler_flops(profiler),
        "kernel_time_ms": _kernel_time_ms(profiler),
        "peak_cuda_mb": peak_mb,
        "loss": loss,
        "stats": stats,
    }


def _write_csv(path, rows):
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path):
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _join_importance(cost_rows, importance_path):
    if not importance_path:
        return []
    path = Path(importance_path)
    if not path.is_file():
        raise FileNotFoundError(f"Importance CSV not found: {path}")
    importance_rows = _read_csv(path)
    costs = {row["block"]: row for row in cost_rows}
    joined = []
    for importance in importance_rows:
        cost = costs.get(str(importance["block"]))
        if cost is None:
            continue
        saved_gflops = float(cost["reported_gflops_saved"])
        score = float(importance["normalized_grad_score"])
        share = float(importance.get("score_share", "nan"))
        joined.append(
            {
                "train_step": int(importance.get("train_step", 0)),
                "flow_t": float(
                    importance.get(
                        "flow_t",
                        importance.get("noise_ratio", 0.0),
                    )
                ),
                "noise_ratio": float(
                    importance.get(
                        "noise_ratio",
                        importance.get("flow_t", 0.0),
                    )
                ),
                "block": cost["block"],
                "block_index": int(cost["block_index"]),
                "normalized_grad_score": score,
                "score_share": share,
                "reported_gflops_saved": saved_gflops,
                "step_time_saved_ms": float(
                    cost["step_time_saved_ms"]
                ),
                "peak_cuda_mem_saved_mb": float(
                    cost["peak_cuda_mem_saved_mb"]
                ),
                "importance_per_saved_gflop": (
                    score / saved_gflops
                    if saved_gflops > 0 else float("inf")
                ),
            }
        )
    return joined


def profile_frozen_block_costs(
    model,
    controller,
    loss_fn,
    probe_config,
    output_dir,
    device,
    metadata=None,
):
    rows = []
    for block_index in probe_config["frozen_indices"]:
        block = f"blocks.{block_index}"
        controller.configure([block])
        rng_state = adaptive.snapshot_rng(device)

        for warmup_index in range(probe_config["warmup"]):
            modes = ("full", "single_skip")
            if warmup_index % 2:
                modes = tuple(reversed(modes))
            for mode in modes:
                _run_step(
                    model,
                    controller,
                    loss_fn,
                    mode,
                    rng_state,
                    device,
                )

        (
            full_ms,
            bypass_ms,
            full_losses,
            bypass_losses,
            bypass_stats,
        ) = _timed_pair(
            model,
            controller,
            loss_fn,
            rng_state,
            device,
            probe_config["repeats"],
        )
        full_profile = _profiled(
            model,
            controller,
            loss_fn,
            "full",
            rng_state,
            device,
        )
        bypass_profile = _profiled(
            model,
            controller,
            loss_fn,
            "single_skip",
            rng_state,
            device,
        )

        max_loss_diff = max(
            [
                abs(full - bypass)
                for full, bypass in zip(full_losses, bypass_losses)
            ]
            + [
                abs(
                    full_profile["loss"]
                    - bypass_profile["loss"]
                )
            ]
        )
        fallbacks = max(
            [stats.fallback_blocks for stats in bypass_stats]
            + [bypass_profile["stats"].fallback_blocks]
        )
        max_forward_diff = max(
            [
                stats.max_reconstruction_abs_diff
                for stats in bypass_stats
            ]
            + [
                bypass_profile[
                    "stats"
                ].max_reconstruction_abs_diff
            ]
        )

        full_mean = statistics.mean(full_ms)
        bypass_mean = statistics.mean(bypass_ms)
        row = {
            "block": block,
            "block_index": block_index,
            "profile_flow_t": probe_config["flow_t"],
            "warmup": probe_config["warmup"],
            "repeats": probe_config["repeats"],
            "full_step_time_ms": full_mean,
            "bypass_step_time_ms": bypass_mean,
            "step_time_saved_ms": full_mean - bypass_mean,
            "step_time_reduction_pct": _reduction(
                full_mean, bypass_mean
            ),
            "full_step_time_std_ms": _stdev(full_ms),
            "bypass_step_time_std_ms": _stdev(bypass_ms),
            "full_reported_gflops": full_profile["flops"] / 1e9,
            "bypass_reported_gflops": (
                bypass_profile["flops"] / 1e9
            ),
            "reported_gflops_saved": (
                full_profile["flops"]
                - bypass_profile["flops"]
            ) / 1e9,
            "reported_flops_reduction_pct": _reduction(
                full_profile["flops"],
                bypass_profile["flops"],
            ),
            "full_cuda_kernel_time_ms": full_profile[
                "kernel_time_ms"
            ],
            "bypass_cuda_kernel_time_ms": bypass_profile[
                "kernel_time_ms"
            ],
            "cuda_kernel_time_saved_ms": (
                full_profile["kernel_time_ms"]
                - bypass_profile["kernel_time_ms"]
            ),
            "full_peak_cuda_mem_mb": full_profile["peak_cuda_mb"],
            "bypass_peak_cuda_mem_mb": bypass_profile[
                "peak_cuda_mb"
            ],
            "peak_cuda_mem_saved_mb": (
                full_profile["peak_cuda_mb"]
                - bypass_profile["peak_cuda_mb"]
            ),
            "max_loss_abs_diff": max_loss_diff,
            "fallback_blocks": fallbacks,
            "max_forward_abs_diff": max_forward_diff,
        }
        if not all(
            math.isfinite(float(row[key]))
            for key in (
                "full_step_time_ms",
                "bypass_step_time_ms",
                "reported_gflops_saved",
                "max_loss_abs_diff",
                "max_forward_abs_diff",
            )
        ):
            raise RuntimeError(f"Non-finite cost row: {row}")
        rows.append(row)
        print(
            f"{block}: "
            f"saved={row['reported_gflops_saved']:.3f} GFLOPs "
            f"time={row['step_time_saved_ms']:.3f} ms "
            f"memory={row['peak_cuda_mem_saved_mb']:.1f} MB "
            f"fallback={fallbacks}"
        )

    controller.set_mode("full")
    controller.configure([])
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    costs_path = output_dir / "block_backward_costs.csv"
    _write_csv(costs_path, rows)

    tradeoff = _join_importance(
        rows,
        probe_config["importance_csv"],
    )
    tradeoff_path = output_dir / "importance_compute_tradeoff.csv"
    if tradeoff:
        _write_csv(tradeoff_path, tradeoff)

    report = {
        "model": "VOSR-0.5B-ms",
        "method": "Threshold bypass cost profile",
        "selection_file": probe_config["selection_path"],
        "selected_lora_blocks": [
            f"blocks.{index}"
            for index in probe_config["selected_indices"]
        ],
        "selected_lora_count": len(
            probe_config["selected_indices"]
        ),
        "profiled_frozen_blocks": [
            f"blocks.{index}"
            for index in probe_config["frozen_indices"]
        ],
        "num_profiled_frozen_blocks": len(rows),
        "profile_flow_t": probe_config["flow_t"],
        "warmup": probe_config["warmup"],
        "repeats": probe_config["repeats"],
        "optimizer_steps": 0,
        "loss_mismatches": sum(
            float(row["max_loss_abs_diff"]) != 0.0 for row in rows
        ),
        "fallback_blocks": sum(
            int(row["fallback_blocks"]) for row in rows
        ),
        "max_forward_abs_diff": max(
            float(row["max_forward_abs_diff"]) for row in rows
        ),
        "cost_csv": str(costs_path),
        "importance_compute_tradeoff_csv": (
            str(tradeoff_path) if tradeoff else None
        ),
    }
    if metadata:
        report.update(metadata)
    summary_path = output_dir / "block_backward_costs_summary.json"
    summary_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )
    return report
