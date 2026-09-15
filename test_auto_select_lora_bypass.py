import argparse
import csv
import json
import tempfile
import unittest
from pathlib import Path

from auto_select_lora_bypass import auto_select


def write_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


class AutoSelectionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.source = dict(loss_mode="official", target="qv", rank=8, alpha=16,
                           lora_selection="all")
        (self.root / "source.json").write_text(json.dumps(self.source))
        rows = []
        for noise in (0.2, 0.8):
            for i, score in enumerate((1, 2, 3, 4, 5, 6)):
                rows.append(dict(train_step=0, noise_ratio=noise, block=f"b{i}", block_index=i,
                                 normalized_grad_score=score, lora_param_count=100,
                                 module_count=1, loss_mode="official"))
        write_csv(self.root / "importance.csv", rows)
        policy = self.root / "policies"
        manifest = []
        configurations = {
            "reference_top8": [f"b{i}" for i in range(6)],
            "threshold_k3": ["b3", "b4", "b5"],
            "threshold_k2": ["b4", "b5"],
            "threshold_k1": ["b5"],
        }
        for label, blocks in configurations.items():
            dest = policy / label
            dest.mkdir(parents=True)
            meta = dict(self.source, selected_blocks=blocks, topk_blocks=len(blocks),
                        importance_threshold=None if label == "reference_top8" else 1.0,
                        selection_policy="original-top8-reference" if label == "reference_top8"
                        else "noise-normalized-importance-threshold")
            (dest / "dit_sr_grad_metadata.json").write_text(json.dumps(meta))
            manifest.append(dict(label=label,
                                 selection_file=f"{label}/dit_sr_grad_metadata.json"))
        (policy / "scan_manifest.json").write_text(json.dumps(manifest))
        write_csv(self.root / "compute.csv", [
            dict(label="reference_top8", reported_gflops=100, mean_step_time_ms=100,
                 peak_cuda_mb=1000),
            dict(label="threshold_k3", reported_gflops=95, mean_step_time_ms=90,
                 peak_cuda_mb=900),
            dict(label="threshold_k2", reported_gflops=90, mean_step_time_ms=80,
                 peak_cuda_mb=800),
            dict(label="threshold_k1", reported_gflops=85, mean_step_time_ms=70,
                 peak_cuda_mb=700),
        ])
        write_csv(self.root / "cost.csv", [
            dict(block=f"b{i}", reported_gflops_saved=2, max_loss_abs_diff=0)
            for i in range(4)
        ])

    def args(self, output="out"):
        return argparse.Namespace(
            importance_csv=str(self.root / "importance.csv"),
            source_metadata=str(self.root / "source.json"),
            lora_policy_dir=str(self.root / "policies"),
            lora_compute_csv=str(self.root / "compute.csv"),
            bypass_cost_csv=str(self.root / "cost.csv"), fidelity_csv="",
            output_dir=str(self.root / output), importance_step=0, expected_blocks=6,
            score_key="normalized_grad_score", noise_weights=None, max_lora_blocks=3,
            min_lora_utility_retention=0.7, lora_cost_metric="reported_gflops",
            target_total_compute_reduction_pct=14, max_bypass_importance_mass=0.35,
            max_bypass_fraction=0.5, max_bypass_count=None, protected_block=[],
            min_gradient_cosine=0.9, max_relative_gradient_error=0.5,
            train_output_dir="outputs/auto", train_steps=100, seed=42,
            data_dir=str(self.root / "data"),
        )

    def test_joint_selection_and_training_outputs(self):
        report = auto_select(self.args())
        self.assertEqual(report["selected_lora_label"], "threshold_k2")
        self.assertEqual(report["selected_lora_count"], 2)
        self.assertEqual(report["bypass_schedule"], ["0.2:2", "0.8:2"])
        self.assertAlmostEqual(report["estimated_total_reduction_vs_top8_pct"], 14)
        self.assertTrue(report["target_met"])
        self.assertFalse(report["gradient_fidelity_checked"])
        training = json.loads((self.root / "out/training_arguments.json").read_text())
        self.assertEqual(training["lora_block_budget"], 2)
        self.assertEqual(training["blockskip_min_run"], 1)
        self.assertEqual(training["blockskip_schedule"], ["0.2:2", "0.8:2"])
        with (self.root / "out/bypass_policy_by_noise.csv").open() as handle:
            policy = list(csv.DictReader(handle))
        self.assertEqual(policy[0]["skip_blocks"], "b0;b1")
        script = (self.root / "out/run_selected_training.sh").read_text()
        self.assertIn("--blockskip_schedule 0.2:2 0.8:2", script)
        self.assertIn("--lora_block_budget 2", script)

    def test_cost_profile_must_match_auto_selected_lora(self):
        with (self.root / "cost.csv").open() as handle:
            rows = list(csv.DictReader(handle))[:-1]
        write_csv(self.root / "bad_cost.csv", rows)
        args = self.args("bad")
        args.bypass_cost_csv = str(self.root / "bad_cost.csv")
        with self.assertRaisesRegex(ValueError, "does not match"):
            auto_select(args)

    def test_nonpositive_cost_is_automatically_protected(self):
        with (self.root / "cost.csv").open() as handle:
            rows = list(csv.DictReader(handle))
        rows[3]["reported_gflops_saved"] = 0
        write_csv(self.root / "cost_zero.csv", rows)
        args = self.args("zero")
        args.bypass_cost_csv = str(self.root / "cost_zero.csv")
        report = auto_select(args)
        self.assertEqual(report["nonpositive_cost_blocks_protected"], ["b3"])
        self.assertIn("b3", report["additional_protected_blocks"])


if __name__ == "__main__":
    unittest.main()
