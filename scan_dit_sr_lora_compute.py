#!/usr/bin/env python3
"""Prepare threshold LoRA candidates, then audit their no-bypass training cost."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import random
import statistics
import time
from pathlib import Path

from build_dit_sr_threshold_lora import make_selection, read_csv


def dump(path, payload):
    path.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def new_directory(path):
    path = Path(path)
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise ValueError(f"Output is not empty; use a new directory: {path}")
    path.mkdir(parents=True, exist_ok=True)
    return path


def candidates_from_rows(rows, source, reference, max_blocks=8, expected_blocks=58):
    if max_blocks < 1:
        raise ValueError("max_blocks must be positive")
    full, details = make_selection(rows, source, threshold=0, expected_blocks=expected_blocks)
    known = {row["block"]: row for row in details}
    original = reference.get("selected_blocks", reference.get("selected_lora_blocks"))
    if not isinstance(original, list) or len(original) != 8 or len(set(original)) != 8:
        raise ValueError("Reference must contain exactly eight unique selected LoRA blocks")
    if set(original) - known.keys():
        raise ValueError("Reference contains blocks absent from calibration")
    for key in ("target", "rank", "alpha", "loss_mode"):
        if reference.get(key) != source.get(key):
            raise ValueError(f"Reference/calibration mismatch: {key}")
    # Never call the newly aggregated top eight the historical Top8 reference.
    old = dict(full)
    old.update(
        selection_policy="original-top8-reference", importance_threshold=None,
        topk_blocks=8, selected_blocks=sorted(original, key=lambda b: known[b]["block_index"]),
        selected_block_fraction=8 / len(known), all_blocks_selected=len(known) == 8,
        selected_lora_params=sum(known[b]["lora_param_count"] for b in original),
        selected_lora_module_count=sum(known[b]["module_count"] for b in original),
    )
    old["selected_lora_parameter_fraction"] = old["selected_lora_params"] / full["all_candidate_lora_params"]
    result = [("reference_top8", old)]
    thresholds = sorted({row["relative_mean_importance"] for row in details})
    seen = set()
    for threshold in thresholds:
        meta, _ = make_selection(rows, source, threshold=threshold, expected_blocks=expected_blocks)
        k = meta["topk_blocks"]
        if k > max_blocks:
            continue
        signature = tuple(meta["selected_blocks"])
        if signature in seen:
            continue
        seen.add(signature)
        result.append((f"threshold_k{k}", meta))
    if len(result) == 1:
        raise ValueError("No threshold produces a non-empty selection within the block cap (check tied scores)")
    return result


def prepare(args):
    source = json.loads(Path(args.source_metadata).read_text(encoding="utf-8-sig"))
    reference = json.loads(Path(args.reference_metadata).read_text(encoding="utf-8-sig"))
    policies = candidates_from_rows(read_csv(args.importance_csv), source, reference,
                                    args.max_blocks, args.expected_blocks)
    out = new_directory(args.policy_dir)
    manifest = []
    for label, metadata in policies:
        metadata.update(importance_csv=str(args.importance_csv), source_metadata=str(args.source_metadata))
        dest = out / label
        dest.mkdir()
        dump(dest / "dit_sr_grad_metadata.json", metadata)
        row = dict(label=label, importance_threshold=metadata["importance_threshold"],
                   selected_blocks=metadata["topk_blocks"],
                   lora_params=metadata["selected_lora_params"],
                   parameter_fraction=metadata["selected_lora_parameter_fraction"],
                   same_blocks_as_reference=set(metadata["selected_blocks"]) == set(policies[0][1]["selected_blocks"]),
                   selection_file=f"{label}/dit_sr_grad_metadata.json")
        manifest.append(row)
        print(f"{label}: K={row['selected_blocks']}, threshold={row['importance_threshold']}, params={row['lora_params']}")
    dump(out / "scan_manifest.json", manifest)
    dump(out / "source_metadata.json", source)
    write_csv(out / "candidate_summary.csv", manifest)
    print(f"Wrote {len(manifest)} policies to {out}")


def benchmark_one(label, selection, source, args, order_index):
    import torch
    from torch.profiler import ProfilerActivity, profile
    from torch.utils.data import default_collate
    import profile_dit_sr_grad as core
    import train_dit_sr_all_lora_importance as train

    if not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable; this command is a GPU cost audit")
    device = torch.device("cuda")
    run = argparse.Namespace(**source)
    run.config_path = args.config_path or run.config_path
    run.ckpt_path = args.ckpt_path or run.ckpt_path
    run.autoencoder_ckpt = args.autoencoder_ckpt or run.autoencoder_ckpt
    run.data_dir = args.data_dir
    run.block_regex = getattr(run, "block_regex", "")
    core.set_seed(args.seed)
    model = core.load_model(run, device)
    diffusion, autoencoder = core.load_official_objective(run, device)
    model.requires_grad_(False)
    selected = set(selection["selected_blocks"])
    actual = set(train.candidate_lora_blocks(model, run.target, run.block_regex))
    if selected - actual:
        raise ValueError(f"Unknown selected blocks for {label}: {sorted(selected - actual)}")
    core.inject_lora(model, run.target, run.rank, run.alpha, run.block_regex, selected_blocks=selected)
    model.train()
    params = [p for p in model.parameters() if p.requires_grad]
    count = sum(p.numel() for p in params)
    if count != selection["selected_lora_params"]:
        raise ValueError(f"Parameter count mismatch for {label}: {count} != {selection['selected_lora_params']}")
    optimizer = torch.optim.AdamW(params, lr=getattr(run, "lr", 1e-5))
    dataset = core.ImageFolderDataset(args.data_dir, run.image_size, 0)
    anchors = selection["noise_anchors"]
    # Identical file choices and per-call random seeds across placements; loading is outside timing.
    batches = [default_collate([dataset[(i * args.batch_size + j) % len(dataset)]
                               for j in range(args.batch_size)]) for i in range(len(anchors))]
    def step(batch, ratio):
        optimizer.zero_grad(set_to_none=True)
        run._profile_noise_ratio = ratio
        loss = train.batch_loss(model, diffusion, autoencoder, batch, run, device)
        if not torch.isfinite(loss).item():
            raise ValueError(f"Non-finite audit loss: {label}, {ratio}")
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(params, getattr(run, "grad_clip", 1.0), error_if_nonfinite=True)
        optimizer.step()
        return loss.detach(), norm.detach()

    by_noise, times = [], []
    for i, (ratio, batch) in enumerate(zip(anchors, batches)):
        for warmup in range(args.warmup):
            core.set_seed(args.seed + 10000 + i * 1000 + warmup)
            step(batch, ratio)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        values = []
        for repeat in range(args.repeats):
            core.set_seed(args.seed + i * 1000 + repeat)
            torch.cuda.synchronize()
            start = time.perf_counter()
            step(batch, ratio)
            torch.cuda.synchronize()
            elapsed = (time.perf_counter() - start) * 1000
            values.append(elapsed)
            times.append(dict(label=label, noise_ratio=ratio, repeat=repeat, step_time_ms=elapsed))
        peak = torch.cuda.max_memory_allocated() / 1024**2
        core.set_seed(args.seed + 500000 + i)
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], with_flops=True) as prof:
            step(batch, ratio)
            torch.cuda.synchronize()
        gflops = sum(event.flops or 0 for event in prof.key_averages()) / 1e9
        by_noise.append(dict(label=label, noise_ratio=ratio,
                             mean_step_time_ms=statistics.mean(values),
                             std_step_time_ms=statistics.stdev(values) if len(values) > 1 else 0,
                             reported_gflops=gflops, peak_cuda_mb=peak))
        print(f"{label}, noise={ratio:g}: {gflops:.3f} reported GFLOPs, {statistics.mean(values):.2f} ms", flush=True)
    summary = dict(
        label=label, order_index=order_index, importance_threshold=selection["importance_threshold"],
        selected_blocks=len(selected), lora_params=count,
        mean_reported_gflops=statistics.mean(r["reported_gflops"] for r in by_noise),
        mean_step_time_ms=statistics.mean(r["mean_step_time_ms"] for r in by_noise),
        peak_cuda_mb=max(r["peak_cuda_mb"] for r in by_noise),
        warmup_per_noise=args.warmup, repeats_per_noise=args.repeats,
        gpu=torch.cuda.get_device_name(), torch_version=torch.__version__,
        input_paths=[b["path"] for b in batches],
        note="Short no-bypass audit starting from fresh LoRA. Forward objective, backward, clipping and AdamW update included. Disk loading, checkpointing and calibration excluded. Profiler FLOPs are incomplete; audit losses are not validation results.",
    )
    return summary, by_noise, times


def comparison_rows(summaries, target_pct):
    refs = [s for s in summaries if s["label"] == "reference_top8"]
    if len(refs) != 1:
        raise ValueError("Exactly one reference_top8 result is required")
    ref = refs[0]
    if ref["mean_reported_gflops"] <= 0 or ref["mean_step_time_ms"] <= 0:
        raise ValueError("Non-positive reference FLOPs/time")
    rows = []
    for s in summaries:
        reduction = 100 * (1 - s["mean_reported_gflops"] / ref["mean_reported_gflops"])
        rows.append(dict(
            label=s["label"], selected_blocks=s["selected_blocks"], lora_params=s["lora_params"],
            importance_threshold=s["importance_threshold"],
            reported_gflops=s["mean_reported_gflops"],
            mean_step_time_ms=s["mean_step_time_ms"], peak_cuda_mb=s["peak_cuda_mb"],
            flops_reduction_vs_top8_pct=reduction,
            time_reduction_vs_top8_pct=100 * (1 - s["mean_step_time_ms"] / ref["mean_step_time_ms"]),
            meets_compute_target=reduction >= target_pct,
        ))
    return rows


def measure(args):
    policy_dir = Path(args.policy_dir)
    manifest = json.loads((policy_dir / "scan_manifest.json").read_text(encoding="utf-8"))
    source = json.loads((policy_dir / "source_metadata.json").read_text(encoding="utf-8"))
    if args.labels:
        wanted = set(args.labels) | {"reference_top8"}
        missing = wanted - {row["label"] for row in manifest}
        if missing:
            raise ValueError(f"Unknown labels: {sorted(missing)}")
        manifest = [row for row in manifest if row["label"] in wanted]
    random.Random(args.seed).shuffle(manifest)
    out = new_directory(args.audit_dir)
    dump(out / "audit_settings.json", vars(args) | {"execution_order": [r["label"] for r in manifest]})
    summaries = []
    for index, row in enumerate(manifest):
        selection = json.loads((policy_dir / row["selection_file"]).read_text(encoding="utf-8"))
        print(f"[{index + 1}/{len(manifest)}] {row['label']} (no bypass)", flush=True)
        summary, by_noise, timings = benchmark_one(row["label"], selection, source, args, index)
        dump(out / f"{row['label']}_summary.json", summary)
        write_csv(out / f"{row['label']}_by_noise.csv", by_noise)
        write_csv(out / f"{row['label']}_timings.csv", timings)
        summaries.append(summary)
        gc.collect()
        import torch
        torch.cuda.empty_cache()
    rows = comparison_rows(summaries, args.target_reduction_pct)
    write_csv(out / "compute_comparison.csv", rows)
    print(json.dumps(rows, indent=2, allow_nan=False))
    print("These are cost-screening results, not quality results. Repeat timing before drawing small speed conclusions.")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=["prepare", "measure"])
    p.add_argument("--importance_csv", default="outputs/dit_sr_fullprobe_ucmerced/lora_importance_evolution.csv")
    p.add_argument("--source_metadata", default="outputs/dit_sr_fullprobe_ucmerced/metadata.json")
    p.add_argument("--reference_metadata", default="outputs/dit_sr_gradskip_ucmerced_1000_seed42_final/metadata.json")
    p.add_argument("--policy_dir", default="outputs/dit_sr_lora_threshold_compute_scan")
    p.add_argument("--audit_dir", default="outputs/dit_sr_lora_threshold_compute_audit")
    p.add_argument("--expected_blocks", type=int, default=58)
    p.add_argument("--max_blocks", type=int, default=8)
    p.add_argument("--labels", nargs="+")
    p.add_argument("--data_dir", default="/mnt/disk1T/liyijuan/star-DTM/data/ucmerced/train_hr")
    p.add_argument("--config_path", default="configs/realsr_DiT.yaml")
    p.add_argument("--ckpt_path", default="weights/realsr.pth")
    p.add_argument("--autoencoder_ckpt", default="weights/autoencoder_vq_f4.pth")
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--repeats", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--seed", type=int, default=4242)
    p.add_argument("--target_reduction_pct", type=float, default=5.0)
    args = p.parse_args()
    if args.warmup < 1 or args.repeats < 2 or args.batch_size < 1:
        p.error("Require warmup>=1, repeats>=2, batch_size>=1")
    if not math.isfinite(args.target_reduction_pct) or not 0 <= args.target_reduction_pct < 100:
        p.error("target_reduction_pct must be in [0, 100)")
    try:
        (prepare if args.mode == "prepare" else measure)(args)
    except (OSError, ValueError, KeyError) as exc:
        p.error(str(exc))


if __name__ == "__main__":
    main()
