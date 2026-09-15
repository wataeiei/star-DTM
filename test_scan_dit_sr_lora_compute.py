import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from scan_dit_sr_lora_compute import candidates_from_rows, comparison_rows


class ComputeScanTests(unittest.TestCase):
    def setUp(self):
        self.source = dict(loss_mode="official", target="qv", rank=8, alpha=16,
                           lora_selection="all")
        self.reference = dict(self.source, selected_lora_blocks=[f"b{i}" for i in range(8)])
        self.rows = [dict(train_step=0, noise_ratio=noise, block=f"b{i}", block_index=i,
                          normalized_grad_score=i + 1, lora_param_count=(i + 1) * 100,
                          module_count=1) for noise in (0.2, 0.8) for i in range(10)]

    def candidates(self):
        return candidates_from_rows(self.rows, self.source, self.reference, expected_blocks=10)

    def test_nested_candidates_and_historical_reference(self):
        result = self.candidates()
        self.assertEqual([m["topk_blocks"] for _, m in result], [8, 8, 7, 6, 5, 4, 3, 2, 1])
        self.assertEqual(result[0][1]["selected_blocks"], [f"b{i}" for i in range(8)])
        self.assertEqual(result[0][1]["selected_lora_params"], 3600)
        self.assertEqual(result[-1][1]["selected_blocks"], ["b9"])
        for (_, a), (_, b) in zip(result[1:], result[2:]):
            self.assertLess(set(b["selected_blocks"]), set(a["selected_blocks"]))

    def test_ties_are_not_split(self):
        for row in self.rows:
            row["normalized_grad_score"] = 2 if row["block_index"] >= 7 else 1
        self.assertEqual([m["topk_blocks"] for _, m in self.candidates()], [8, 3])

    def test_all_tied_cannot_meet_cap(self):
        for row in self.rows:
            row["normalized_grad_score"] = 1
        with self.assertRaisesRegex(ValueError, "No threshold"):
            self.candidates()

    def test_reference_mismatch(self):
        self.reference["rank"] = 4
        with self.assertRaisesRegex(ValueError, "mismatch: rank"):
            self.candidates()

    def test_measured_cost_not_parameter_proxy(self):
        ref = dict(label="reference_top8", mean_reported_gflops=100, mean_step_time_ms=10,
                   selected_blocks=8, lora_params=800, importance_threshold=None, peak_cuda_mb=1000)
        trial = dict(ref, label="threshold_k2", mean_reported_gflops=90,
                     mean_step_time_ms=11, selected_blocks=2, lora_params=200)
        row = comparison_rows([ref, trial], 5)[1]
        self.assertAlmostEqual(row["flops_reduction_vs_top8_pct"], 10)
        self.assertAlmostEqual(row["time_reduction_vs_top8_pct"], -10)
        self.assertTrue(row["meets_compute_target"])
        self.assertFalse(comparison_rows([ref, trial], 15)[1]["meets_compute_target"])

    def test_reference_required(self):
        with self.assertRaisesRegex(ValueError, "Exactly one"):
            comparison_rows([], 5)

    def test_prepare_cli_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, ref, imp, out = [root / name for name in ("source.json", "ref.json", "imp.csv", "out")]
            source.write_text(json.dumps(self.source))
            ref.write_text(json.dumps(self.reference))
            with imp.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(self.rows[0]))
                writer.writeheader()
                writer.writerows(self.rows)
            cmd = [sys.executable, str(Path(__file__).with_name("scan_dit_sr_lora_compute.py")), "prepare",
                   "--importance_csv", str(imp), "--source_metadata", str(source),
                   "--reference_metadata", str(ref), "--expected_blocks", "10", "--policy_dir", str(out)]
            result = subprocess.run(cmd, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            manifest = json.loads((out / "scan_manifest.json").read_text())
            self.assertEqual(len(manifest), 9)
            self.assertTrue((out / "candidate_summary.csv").is_file())
            for row in manifest:
                self.assertTrue((out / row["selection_file"]).is_file())
            second = subprocess.run(cmd, capture_output=True, text=True)
            self.assertNotEqual(second.returncode, 0)
            self.assertIn("not empty", second.stderr)


if __name__ == "__main__":
    unittest.main()
