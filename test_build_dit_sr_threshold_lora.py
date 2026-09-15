import copy
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from build_dit_sr_threshold_lora import make_selection


class ThresholdLoRATests(unittest.TestCase):
    def setUp(self):
        self.source = dict(loss_mode="official", target="qv", rank=8, alpha=16,
                           lora_selection="all", topk_blocks=8)
        self.rows = []
        for noise in [0.2, 0.8]:
            for i, score in enumerate([1.0, 2.0, 3.0, 6.0]):
                self.rows.append(dict(train_step=0, noise_ratio=noise, block=f"b{i}",
                                      block_index=i, normalized_grad_score=score,
                                      lora_param_count=(i + 1) * 100, module_count=1))

    def run_selection(self, **kwargs):
        return make_selection(self.rows, self.source, expected_blocks=4, **kwargs)

    def test_threshold_selects_actual_k_and_parameter_fraction(self):
        meta, rows = self.run_selection()
        self.assertEqual(meta["selected_blocks"], ["b2", "b3"])
        self.assertEqual(meta["topk_blocks"], 2)
        self.assertEqual(meta["selected_lora_parameter_fraction"], 0.7)
        self.assertEqual(meta["selected_block_fraction"], 0.5)
        self.assertEqual(meta["alpha"], 16)
        self.assertEqual(meta["importance_threshold"], 1)
        self.assertEqual(rows[2]["relative_mean_importance"], 1)

    def test_per_noise_scale_invariant(self):
        first = self.run_selection()
        for row in self.rows:
            if row["noise_ratio"] == 0.8:
                row["normalized_grad_score"] *= 1000
        self.assertEqual(self.run_selection(), first)

    def test_custom_weights(self):
        for row in self.rows:
            if row["noise_ratio"] == 0.8:
                row["normalized_grad_score"] = 6 if row["block"] == "b0" else 1
        meta, _ = self.run_selection(noise_weights=[0, 1])
        self.assertEqual(meta["selected_blocks"], ["b0"])

    def test_no_silent_topk_fallback(self):
        with self.assertRaisesRegex(ValueError, "zero blocks"):
            self.run_selection(threshold=3)

    def test_all_selection_reported(self):
        meta, _ = self.run_selection(threshold=0)
        self.assertTrue(meta["all_blocks_selected"])

    def test_reject_incomplete_coverage(self):
        self.rows.pop()
        with self.assertRaisesRegex(ValueError, "coverage"):
            self.run_selection()

    def test_reject_duplicate_block(self):
        self.rows.append(dict(self.rows[0]))
        with self.assertRaisesRegex(ValueError, "Duplicate block"):
            self.run_selection()

    def test_reject_invalid_scores_and_totals(self):
        original = copy.deepcopy(self.rows)
        for value in [float("nan"), float("inf"), -1]:
            self.rows = copy.deepcopy(original)
            self.rows[0]["normalized_grad_score"] = value
            with self.assertRaises(ValueError):
                self.run_selection()
        for row in self.rows:
            row["normalized_grad_score"] = 0
        with self.assertRaisesRegex(ValueError, "score sum"):
            self.run_selection()

    def test_reject_parameter_mismatch(self):
        self.rows[-1]["lora_param_count"] += 1
        with self.assertRaisesRegex(ValueError, "counts vary"):
            self.run_selection()

    def test_reject_sparse_calibration(self):
        self.source["lora_selection"] = "metadata"
        with self.assertRaisesRegex(ValueError, "lora_selection=all"):
            self.run_selection()

    def test_cli_outputs_and_refuses_overwrite(self):
        import csv
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, importance, out = root / "source.json", root / "importance.csv", root / "out"
            source.write_text(json.dumps(self.source))
            with importance.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(self.rows[0]))
                writer.writeheader()
                writer.writerows(self.rows)
            cmd = [sys.executable, str(Path(__file__).with_name("build_dit_sr_threshold_lora.py")),
                   "--importance_csv", str(importance), "--source_metadata", str(source),
                   "--expected_blocks", "4", "--output_dir", str(out)]
            result = subprocess.run(cmd, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            metadata = json.loads((out / "dit_sr_grad_metadata.json").read_text())
            self.assertEqual(metadata["topk_blocks"], 2)
            self.assertTrue((out / "lora_selection_scores.csv").is_file())
            second = subprocess.run(cmd, capture_output=True, text=True)
            self.assertNotEqual(second.returncode, 0)
            self.assertIn("not empty", second.stderr)


if __name__ == "__main__":
    unittest.main()
