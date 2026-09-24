import argparse

from train_vosr_all_lora import main


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train VOSR Threshold bypass.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--selection_file", required=True)
    parser.add_argument("--threshold_policy_csv", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--train_steps", type=int, default=1000)
    parser.add_argument("--checkpoint_every", type=int, default=250)
    parser.add_argument("--seed", type=int, default=42)
    arguments = parser.parse_args()

    main(
        arguments.config,
        force_threshold_bypass_training=True,
        config_overrides={
            "output_dir": arguments.output_dir,
            "suffix": "_threshold_bypass",
            "lora_selection_file": arguments.selection_file,
            "bypass_threshold_policy_csv": arguments.threshold_policy_csv,
            "max_train_steps": arguments.train_steps,
            "checkpointing_steps": arguments.checkpoint_every,
            "seed": arguments.seed,
        },
    )
