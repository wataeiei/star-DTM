#!/usr/bin/env python3
"""Compare DiT4SR LoRA adapters with identical official-flow validation draws."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader

import adaptive_grad_blockskip as adaptive
import profile_hf_dit4sr_grad as core


def parse_adapter(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError(
            "--adapter must use LABEL=PATH, for example All-LoRA=outputs/run/lora_adapter.pt"
        )
    label, path = value.split("=", 1)
    label = label.strip()
    if not label:
        raise argparse.ArgumentTypeError("Adapter label cannot be empty.")
    return label, Path(path).expanduser()


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def reset_lora_to_base(transformer: torch.nn.Module) -> None:
    for _name, module in core.iter_lora_modules(transformer):
        module.lora_down.weight.data.zero_()
        module.lora_up.weight.data.zero_()


@torch.no_grad()
def evaluate_ratio(
    pipe,
    transformer: torch.nn.Module,
    loader: DataLoader,
    args: argparse.Namespace,
    device: torch.device,
    noise_ratio: float,
    ratio_index: int,
) -> dict:
    # Reset before every method/ratio pair so VAE samples and flow noise match exactly.
    core.set_seed(args.eval_seed + ratio_index)
    args._profile_noise_ratio = noise_ratio
    losses = []
    iterator = iter(loader)
    try:
        for _ in range(args.eval_batches):
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                batch = next(iterator)
            loss = core.flow_matching_batch_loss(
                pipe, transformer, batch, args, device
            )
            value = float(loss.detach().cpu())
            if math.isfinite(value):
                losses.append(value)
    finally:
        del args._profile_noise_ratio

    if not losses:
        raise RuntimeError(f"No finite validation loss at noise ratio {noise_ratio}.")
    tensor = torch.tensor(losses, dtype=torch.float64)
    return {
        "noise_ratio": noise_ratio,
        "mean_flow_loss": float(tensor.mean()),
        "std_flow_loss": float(tensor.std(unbiased=False)),
        "num_batches": len(losses),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_id", default="acceptee/DiT4SR")
    parser.add_argument(
        "--base_model_id", default="stabilityai/stable-diffusion-3.5-medium"
    )
    parser.add_argument("--variant", default="dit4sr_q")
    parser.add_argument("--dit4sr_code_repo", default="acceptee/DiT4SR")
    parser.add_argument("--dit4sr_code_dir", default="")
    parser.add_argument("--transformer_subfolder", default="")
    parser.add_argument("--pipeline_subfolder", default="")
    parser.add_argument("--component_name", default="")
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--adapter",
        action="append",
        type=parse_adapter,
        default=[],
        metavar="LABEL=PATH",
        help="Repeat for every custom lora_adapter.pt checkpoint.",
    )
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--max_images", type=int, default=0)
    parser.add_argument("--eval_batches", type=int, default=20)
    parser.add_argument(
        "--noise_ratios",
        type=float,
        nargs="+",
        default=[0.05, 0.2, 0.4, 0.6, 0.8, 0.95],
    )
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--eval_seed", type=int, default=4242)
    parser.add_argument("--target", default="qv")
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--alpha", type=int, default=16)
    parser.add_argument("--block_regex", default="")
    parser.add_argument("--dtype", default="bf16", choices=["fp32", "bf16", "fp16"])
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--sr_scale", type=float, default=4.0)
    parser.add_argument("--latent_channels", type=int, default=16)
    parser.add_argument("--latent_size", type=int, default=64)
    parser.add_argument("--caption_dim", type=int, default=4096)
    parser.add_argument("--prompt_seq_len", type=int, default=77)
    parser.add_argument("--timestep", type=int, default=500)
    parser.add_argument("--input_noise_std", type=float, default=0.0)
    args = parser.parse_args()

    if any(not 0.0 <= ratio <= 1.0 for ratio in args.noise_ratios):
        raise SystemExit("--noise_ratios values must be in [0, 1].")
    missing_paths = [str(path) for _label, path in args.adapter if not path.is_file()]
    if missing_paths:
        raise SystemExit("Missing adapter checkpoint(s): " + ", ".join(missing_paths))

    # These attributes are consumed by the shared official DiT4SR loader.
    args.loss_mode = "official_flow"
    args.load_mode = "transformer"
    args.model_impl = "official"

    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    )
    pipe = core.load_pipe(args, device)
    transformer = core.get_transformer(pipe, args.component_name)
    transformer.requires_grad_(False)
    injected = core.inject_lora(
        transformer, args.target, args.rank, args.alpha, args.block_regex
    )
    transformer.eval()

    dataset = core.ImageFolderDataset(
        args.data_dir, args.image_size, args.max_images
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    methods: list[tuple[str, Path | None]] = [("Base-DiT4SR", None)]
    methods.extend(args.adapter)
    rows = []
    load_reports = {}
    for method, adapter_path in methods:
        reset_lora_to_base(transformer)
        if adapter_path is not None:
            report = adaptive.load_lora_adapter(transformer, adapter_path)
            if report["missing"] or report["unexpected"]:
                raise RuntimeError(
                    f"{method}: adapter mismatch: "
                    f"missing={report['missing'][:5]} "
                    f"unexpected={report['unexpected'][:5]}"
                )
            load_reports[method] = report

        method_rows = []
        for ratio_index, ratio in enumerate(args.noise_ratios):
            result = evaluate_ratio(
                pipe,
                transformer,
                loader,
                args,
                device,
                ratio,
                ratio_index,
            )
            row = {"method": method, **result}
            rows.append(row)
            method_rows.append(row)
            print(
                f"{method:20s} sigma={ratio:.2f} "
                f"loss={result['mean_flow_loss']:.6f}"
            )
        rows.append(
            {
                "method": method,
                "noise_ratio": "mean",
                "mean_flow_loss": sum(
                    row["mean_flow_loss"] for row in method_rows
                )
                / len(method_rows),
                "std_flow_loss": "",
                "num_batches": sum(row["num_batches"] for row in method_rows),
            }
        )

    base_losses = {
        str(row["noise_ratio"]): float(row["mean_flow_loss"])
        for row in rows
        if row["method"] == "Base-DiT4SR"
    }
    for row in rows:
        base_loss = base_losses[str(row["noise_ratio"])]
        delta = float(row["mean_flow_loss"]) - base_loss
        row["delta_loss_vs_base"] = delta
        row["relative_loss_change_vs_base_pct"] = 100.0 * delta / base_loss

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "official_flow_validation.csv", rows)
    metadata = {
        "model_id": args.model_id,
        "base_model_id": args.base_model_id,
        "variant": args.variant,
        "data_dir": args.data_dir,
        "image_size": args.image_size,
        "eval_batches_per_noise_ratio": args.eval_batches,
        "noise_ratios": args.noise_ratios,
        "eval_seed": args.eval_seed,
        "target": args.target,
        "rank": args.rank,
        "alpha": args.alpha,
        "injected_module_count": len(injected),
        "load_reports": load_reports,
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, default=str), encoding="utf-8"
    )
    print(f"Wrote results to {output_dir}")


if __name__ == "__main__":
    main()
