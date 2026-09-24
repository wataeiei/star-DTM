import argparse

from train_vosr_all_lora import main


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Profile VOSR frozen-block backward bypass costs."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--selection_file", required=True)
    parser.add_argument("--importance_csv", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--flow_t", type=float, default=0.4)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--seed", type=int)
    arguments = parser.parse_args()

    overrides = {
        "output_dir": arguments.output_dir,
        "suffix": "_bypass_cost_profile",
        "lora_selection_file": arguments.selection_file,
        "bypass_cost_importance_csv": arguments.importance_csv,
        "bypass_cost_flow_t": arguments.flow_t,
        "bypass_cost_warmup": arguments.warmup,
        "bypass_cost_repeats": arguments.repeats,
    }
    if arguments.seed is not None:
        overrides["seed"] = arguments.seed

    main(
        arguments.config,
        force_bypass_cost_probe=True,
        config_overrides=overrides,
    )
