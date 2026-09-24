import argparse

from train_vosr_all_lora import main


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Profile VOSR LoRA gradient importance without updates."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--suffix",
        default="_importance_probe24",
    )
    parser.add_argument(
        "--noise_ratios",
        nargs="+",
        type=float,
        default=[0.05, 0.2, 0.4, 0.6, 0.8, 0.95],
    )
    parser.add_argument("--batches_per_anchor", type=int, default=4)
    parser.add_argument("--seed", type=int)
    arguments = parser.parse_args()
    overrides = {
        "output_dir": arguments.output_dir,
        "suffix": arguments.suffix,
        "importance_noise_ratios": arguments.noise_ratios,
        "importance_batches_per_anchor": arguments.batches_per_anchor,
    }
    if arguments.seed is not None:
        overrides["seed"] = arguments.seed
    main(
        arguments.config,
        force_importance_probe=True,
        config_overrides=overrides,
    )
