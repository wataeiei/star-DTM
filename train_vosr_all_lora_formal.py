import argparse

from train_vosr_all_lora import main


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train formal VOSR All-LoRA.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--train_steps", type=int, default=1000)
    parser.add_argument("--checkpoint_every", type=int, default=250)
    parser.add_argument("--seed", type=int, default=42)
    arguments = parser.parse_args()

    main(
        arguments.config,
        config_overrides={
            "output_dir": arguments.output_dir,
            "suffix": "_all_lora",
            "max_train_steps": arguments.train_steps,
            "checkpointing_steps": arguments.checkpoint_every,
            "seed": arguments.seed,
            "importance_probe_only": False,
            "bypass_cost_probe_only": False,
            "bypass_fidelity_probe_only": False,
            "threshold_bypass_training": False,
        },
    )
