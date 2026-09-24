import argparse

from train_vosr_all_lora import main


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Audit VOSR Threshold bypass gradient fidelity."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--selection_file", required=True)
    parser.add_argument("--threshold_policy_csv", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--probe_batches", type=int, default=4)
    parser.add_argument("--min_cosine", type=float, default=0.90)
    parser.add_argument("--min_descent_retention", type=float, default=0.50)
    parser.add_argument("--strict_min_cosine", type=float, default=0.95)
    parser.add_argument("--max_relative_error", type=float, default=0.10)
    parser.add_argument("--max_loss_abs_diff", type=float, default=1e-7)
    parser.add_argument("--seed", type=int)
    arguments = parser.parse_args()

    overrides = {
        "output_dir": arguments.output_dir,
        "suffix": "_bypass_gradient_fidelity",
        "lora_selection_file": arguments.selection_file,
        "bypass_threshold_policy_csv": arguments.threshold_policy_csv,
        "bypass_fidelity_batches": arguments.probe_batches,
        "bypass_fidelity_min_cosine": arguments.min_cosine,
        "bypass_fidelity_min_descent_retention": (
            arguments.min_descent_retention
        ),
        "bypass_fidelity_strict_min_cosine": arguments.strict_min_cosine,
        "bypass_fidelity_max_relative_error": arguments.max_relative_error,
        "bypass_fidelity_max_loss_abs_diff": arguments.max_loss_abs_diff,
    }
    if arguments.seed is not None:
        overrides["seed"] = arguments.seed

    main(
        arguments.config,
        force_bypass_fidelity_probe=True,
        config_overrides=overrides,
    )
