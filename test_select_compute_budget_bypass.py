import csv
from decimal import Decimal
import itertools
import json
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import unittest

from select_compute_budget_bypass import runs_for, solve


def candidate(index, cost, score, group="transformer_blocks"):
    return dict(block=f"{group}.{index}", index=index,
                cost=Decimal(str(cost)), score=Decimal(str(score)))


class ComputeBudgetTests(unittest.TestCase):
    def test_matches_exhaustive_search(self):
        rng = random.Random(42)
        for _ in range(80):
            rows = [candidate(i, rng.choice([3.25, 4.75]), rng.randint(0, 9))
                    for i in [0, 1, 2, 4, 5, 6, 7]]
            minimum, maximum, nruns = rng.choice([(1, 3, 2), (2, 3, 2), (1, 2, 3)])
            target = Decimal(rng.randint(0, 25))
            options = []
            for mask in itertools.product([False, True], repeat=len(rows)):
                indices = tuple(i for i, include in enumerate(mask) if include)
                blocks = [rows[i]["block"] for i in indices]
                runs = runs_for(blocks, rows)
                if len(runs) > nruns or any(not minimum <= len(r) <= maximum for r in runs):
                    continue
                cost = sum((rows[i]["cost"] for i in indices), Decimal(0))
                if cost >= target:
                    score = sum((rows[i]["score"] for i in indices), Decimal(0))
                    options.append((score, cost, len(indices), indices))
            if not options:
                with self.assertRaises(ValueError):
                    solve(rows, target, minimum, maximum, nruns)
            else:
                score, cost, _, indices = min(options)
                actual = solve(rows, target, minimum, maximum, nruns)
                self.assertEqual(actual[:3], ([rows[i]["block"] for i in indices], score, cost))

    def test_can_change_count(self):
        rows = [candidate(0, 10, 1), candidate(1, 5, 3), candidate(2, 5, 4)]
        self.assertEqual(solve(rows, Decimal(10), 1, 3, 2)[0], ["transformer_blocks.0"])

    def test_group_boundary_is_not_contiguous(self):
        rows = [candidate(0, 5, 1, "input_blocks.0.blocks"),
                candidate(1, 5, 1, "output_blocks.0.blocks")]
        with self.assertRaises(ValueError):
            solve(rows, Decimal(10), 2, 3, 2)

    def test_decimal_cost_threshold(self):
        rows = [candidate(0, "0.1", 1), candidate(1, "0.2", 1)]
        self.assertEqual(solve(rows, Decimal("0.3"), 1, 2, 1)[2], Decimal("0.3"))

    def test_cli_reads_constraints_and_rejects_missing_cost(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            run = root / "run"
            run.mkdir()
            (run / "metadata.json").write_text(json.dumps({
                "selected_lora_blocks": ["transformer_blocks.3"],
                "blockskip_protected_blocks": [], "blockskip_min_run": 1,
                "blockskip_max_run": 3, "blockskip_max_runs": 2,
            }), encoding="utf-8")

            def write(name, rows):
                with (root / name).open("w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=list(rows[0]))
                    writer.writeheader()
                    writer.writerows(rows)

            write("run/train_log.csv", [dict(step=1, noise_ratio="0.4", skipped_block_count=2,
                  skipped_blocks="transformer_blocks.1;transformer_blocks.2")])
            cost_rows = [dict(block=f"transformer_blocks.{i}", reported_gflops_saved=c,
                             max_loss_abs_diff=0) for i, c in enumerate([10, 5, 5])]
            write("cost.csv", cost_rows)
            write("importance.csv", [dict(train_step=0, noise_ratio="0.4",
                  block=f"transformer_blocks.{i}", block_index=i, normalized_grad_score=s)
                  for i, s in enumerate([1, 3, 4, 10])])
            command = [sys.executable, str(Path(__file__).with_name("select_compute_budget_bypass.py")),
                       "--cost_csv", str(root / "cost.csv"), "--importance_csv", str(root / "importance.csv"),
                       "--baseline_run", f"2={run}", "--output_dir", str(root / "result")]
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            with (root / "result/compute_budget_comparison.csv").open() as f:
                row = next(csv.DictReader(f))
            self.assertEqual(row["new_skip_indices"], "0")
            self.assertEqual(row["new_count"], "1")
            self.assertEqual(row["overshoot_pct"], "0.0")
            write("cost.csv", cost_rows[:-1])
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Missing costs", result.stderr)


if __name__ == "__main__":
    unittest.main()
