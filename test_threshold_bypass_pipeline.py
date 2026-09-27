import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent
BUILDER = ROOT / "build_threshold_bypass_candidates.py"
SELECTOR = ROOT / "select_threshold_bypass_final.py"
RUNNER = ROOT / "run_tsdsr_threshold_bypass_search.py"
STAGE_SUMMARY = ROOT / "summarize_tsdsr_threshold_stage.py"


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


class ThresholdBypassPipelineTest(unittest.TestCase):
    def test_independent_global_threshold_policies(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            importance = root / "importance.csv"
            costs = root / "costs.csv"
            output = root / "policies"
            score_rows = []
            scores = {
                0.1: [8.0, 4.0, 2.0, 1.0],
                0.9: [1.0, 2.0, 4.0, 8.0],
            }
            for ratio, values in scores.items():
                for index, score in enumerate(values):
                    score_rows.append({
                        "noise_ratio": ratio,
                        "block": f"transformer_blocks.{index}",
                        "normalized_grad_score": score,
                    })
            write_csv(importance, score_rows)
            write_csv(costs, [
                {
                    "block": f"transformer_blocks.{index}",
                    "reported_gflops_saved": 10.0 + index,
                    "max_loss_abs_diff": 0.0,
                    "max_forward_abs_diff": 0.0,
                    "fallback_events": 0,
                }
                for index in range(4)
            ])
            subprocess.run(
                [
                    sys.executable,
                    str(BUILDER),
                    "--importance_csv", str(importance),
                    "--cost_csv", str(costs),
                    "--output_dir", str(output),
                    "--k_values", "1", "2", "4",
                    "--thresholds", "0", "0.34", "1",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            with (output / "threshold_candidate_manifest.csv").open(
                newline="", encoding="utf-8"
            ) as handle:
                manifest = list(csv.DictReader(handle))
            self.assertTrue(manifest)
            for candidate in manifest:
                selected = set(json.loads(
                    Path(candidate["selection_file"]).read_text(encoding="utf-8")
                )["selected_lora_blocks"])
                with Path(candidate["policy_csv"]).open(
                    newline="", encoding="utf-8"
                ) as handle:
                    policy_rows = list(csv.DictReader(handle))
                for row in policy_rows:
                    skipped = {value for value in row["skip_blocks"].split(";") if value}
                    self.assertFalse(selected & skipped)
                    self.assertEqual(int(row["bypass_budget"]), len(skipped))
            metadata = json.loads((output / "metadata.json").read_text())
            self.assertIn("fixed bypass count", metadata["constraints_removed"])
            self.assertEqual(metadata["policy_rule"],
                             "independent safe frozen blocks with metric <= one global tau")

            search_output = root / "search"
            subprocess.run(
                [
                    sys.executable,
                    str(RUNNER),
                    "--candidate_manifest",
                    str(output / "threshold_candidate_manifest.csv"),
                    "--output_dir", str(search_output),
                    "--pretrained_model", "sd3",
                    "--official_lora_dir", "official",
                    "--teacher_lora_dir", "teacher",
                    "--default_embedding_dir", "default",
                    "--null_embedding_dir", "null",
                    "--data_dir", "train",
                    "--include_all_lora",
                    "--dry_run",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            with (search_output / "run_plan.csv").open(
                newline="", encoding="utf-8"
            ) as handle:
                plan = list(csv.DictReader(handle))
            self.assertEqual(sum(row["run_kind"] == "all_lora" for row in plan), 1)
            self.assertEqual(sum(row["run_kind"] == "native" for row in plan), 2)
            self.assertEqual(sum(row["run_kind"] == "controller_b0" for row in plan), 2)
            self.assertTrue(any(row["run_kind"] == "threshold_bypass" for row in plan))

    def test_quality_selector_chooses_fastest_feasible_candidate(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            results = root / "results.csv"
            output = root / "selection"
            common = {
                "selected_k": 8,
                "global_threshold": 0.05,
                "controller_b0_step_time_s": 1.0,
                "controller_overhead_in_step_time": True,
                "fallback_block_events": 0,
                "max_forward_abs_diff": 0,
                "nonfinite_loss_count": 0,
                "base_psnr": 20,
                "all_psnr": 22,
                "candidate_psnr": 21.8,
                "base_ssim": 0.5,
                "all_ssim": 0.6,
                "candidate_ssim": 0.59,
                "base_lpips": 0.4,
                "all_lpips": 0.2,
                "candidate_lpips": 0.22,
                "base_acc": 0.7,
                "all_acc": 0.8,
                "candidate_acc": 0.79,
                "native_acc": 0.795,
                "native_musiq": 50,
                "candidate_musiq": 49,
                "native_maniqa": 0.4,
                "candidate_maniqa": 0.39,
                "native_clipiqa": 0.5,
                "candidate_clipiqa": 0.49,
                "native_liqe": 3.0,
                "candidate_liqe": 2.9,
            }
            write_csv(results, [
                {"candidate_id": "slower", "mean_step_time_s": 0.96, **common},
                {"candidate_id": "faster", "mean_step_time_s": 0.90, **common},
            ])
            subprocess.run(
                [
                    sys.executable,
                    str(SELECTOR),
                    "--results_csv", str(results),
                    "--output_dir", str(output),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            report = json.loads((output / "selection_report.json").read_text())
            self.assertEqual(report["best_candidate"]["candidate_id"], "faster")

    def test_stage_shortlist_uses_controller_b0_speedup(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            results = root / "stage.csv"
            output = root / "summary"
            common = {
                "selected_k": 8,
                "mean_skipped_blocks": 4,
                "fallback_block_events": 0,
                "max_residual_forward_abs_diff": 0,
                "run_dir": "run",
            }
            write_csv(results, [
                {"run_kind": "native", "candidate_id": "", "global_threshold": "",
                 "mean_train_step_time_s": 1.02, **common},
                {"run_kind": "controller_b0", "candidate_id": "", "global_threshold": "",
                 "mean_train_step_time_s": 1.00, **common},
                {"run_kind": "threshold_bypass", "candidate_id": "pass",
                 "global_threshold": 0.1, "mean_train_step_time_s": 0.95, **common},
                {"run_kind": "threshold_bypass", "candidate_id": "fail",
                 "global_threshold": 0.2, "mean_train_step_time_s": 0.99, **common},
            ])
            subprocess.run(
                [
                    sys.executable,
                    str(STAGE_SUMMARY),
                    "--stage_results", str(results),
                    "--output_dir", str(output),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            with (output / "shortlist.csv").open(
                newline="", encoding="utf-8"
            ) as handle:
                shortlist = list(csv.DictReader(handle))
            self.assertEqual([row["candidate_id"] for row in shortlist], ["pass"])


if __name__ == "__main__":
    unittest.main()
