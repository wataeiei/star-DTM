import csv
import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

from build_dit_sr_threshold_policy import build


class ThresholdPolicyTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.imp = []
        for noise, values in [(0.2, [10, 1, 2, 3]), (0.8, [10, 3, 1, 2])]:
            for i, score in enumerate(values):
                self.imp.append(dict(train_step=0, noise_ratio=noise, block=f"b{i}",
                                     block_index=i, normalized_grad_score=score))
        self.costs = [dict(block=f"b{i}", reported_gflops_saved=2) for i in range(1, 4)]
        self.log = [dict(step=i + 1, noise_ratio=n, skipped_blocks="b1;b2",
                         skipped_block_count=2, fallback_blocks=0)
                    for i, n in enumerate([0.2, 0.8])]
        self.args = Namespace(
            importance_csv=self.root / "importance.csv", importance_step=0,
            score_key="normalized_grad_score", selection_file=self.root / "selection.json",
            cost_csv=self.root / "costs.csv", reference_log=self.root / "log.csv",
            protected_block=[], exclude_block=[], max_count=None,
            noise_weighting="uniform", compute_tolerance_pct=1.0,
        )
        self.args.selection_file.write_text(json.dumps({"selected_lora_blocks": ["b0"]}))

    def run_build(self):
        for path, rows in [(self.args.importance_csv, self.imp),
                           (self.args.cost_csv, self.costs), (self.args.reference_log, self.log)]:
            with path.open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
        return build(self.args)

    def test_exact_match_and_fixed_count_is_allowed(self):
        report, scan, rows = self.run_build()
        self.assertEqual(report["chosen_alpha"], 1)
        self.assertEqual(report["threshold_schedule"], "0.2:2 0.8:2")
        self.assertEqual(report["target_saved_gflops_per_step"], 4)
        self.assertTrue(report["within_tolerance"])
        self.assertEqual(rows[0]["skip_blocks"], "b1;b2")
        self.assertEqual(rows[1]["skip_blocks"], "b2;b3")
        self.assertEqual(len(scan), 4)

    def test_cap_does_not_silently_claim_match(self):
        self.args.max_count = 1
        report, _, _ = self.run_build()
        self.assertFalse(report["within_tolerance"])
        self.assertEqual(report["threshold_saved_gflops_per_step"], 2)

    def test_reference_weights(self):
        self.log.append(dict(self.log[0], step=3))
        self.log[1].update(skipped_blocks="b1", skipped_block_count=1)
        report, _, _ = self.run_build()
        self.assertEqual(report["target_saved_gflops_per_step"], 3)
        self.args.noise_weighting = "reference"
        report, _, _ = self.run_build()
        self.assertAlmostEqual(report["target_saved_gflops_per_step"], 10 / 3)

    def test_reject_duplicate_importance(self):
        self.imp.append(dict(self.imp[0]))
        with self.assertRaisesRegex(ValueError, "Duplicate importance"):
            self.run_build()

    def test_reject_missing_cost(self):
        self.costs.pop()
        with self.assertRaisesRegex(ValueError, "missing costs"):
            self.run_build()

    def test_exclude_nonpositive_cost_and_preserve_reference(self):
        self.costs[0]["reported_gflops_saved"] = -0.1
        with self.assertRaisesRegex(ValueError, "Non-positive"):
            self.run_build()
        self.args.exclude_block = ["b1"]
        report, _, rows = self.run_build()
        self.assertAlmostEqual(report["target_saved_gflops_per_step"], 1.9)
        self.assertEqual(report["candidate_count"], 2)
        self.assertTrue(all("b1" not in r["skip_blocks"] for r in rows))

    def test_reject_reference_count_mismatch(self):
        self.log[0]["skipped_block_count"] = 3
        with self.assertRaisesRegex(ValueError, "count mismatch"):
            self.run_build()

    def test_reject_reference_fallback(self):
        self.log[0]["fallback_blocks"] = 1
        with self.assertRaisesRegex(ValueError, "fallback"):
            self.run_build()

    def test_scale_invariant(self):
        first = self.run_build()[0]
        for row in self.imp:
            row["normalized_grad_score"] *= 100
        second = self.run_build()[0]
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
