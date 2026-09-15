import argparse
import json
import tempfile
import unittest
from pathlib import Path

from run_dit_sr_lora_candidates import LABELS, build_jobs


class CandidateRunnerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        for name in ("train_dit_sr_all_lora_importance.py", "configs/realsr_DiT.yaml",
                     "weights/realsr.pth", "weights/autoencoder_vq_f4.pth"):
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()
        (self.root / "data").mkdir()
        for label, k in LABELS.items():
            path = self.root / "policies" / label / "dit_sr_grad_metadata.json"
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps(dict(target="qv", rank=8, alpha=16, loss_mode="official",
                                            topk_blocks=k, selected_blocks=[f"b{i}" for i in range(k)])))
        self.args = argparse.Namespace(root=str(self.root), data_dir=str(self.root / "data"),
                                       policy_dir="policies", output_root="results", labels=list(LABELS),
                                       steps=100, seed=42)

    def test_same_training_protocol_with_actual_selection_counts(self):
        root, jobs = build_jobs(self.args)
        self.assertEqual(root, self.root.resolve())
        self.assertEqual(len(jobs), 3)
        for label, out, cmd in jobs:
            for flag, value in (("--train_steps", "100"), ("--seed", "42"),
                                ("--blockskip_count", "0"), ("--lr", "1e-5"),
                                ("--lora_block_budget", str(LABELS[label]))):
                self.assertEqual(cmd[cmd.index(flag) + 1], value)
            self.assertFalse(out.exists())

    def test_existing_output_stops_preflight(self):
        (self.root / "results/reference_top8_nobypass_100_seed42").mkdir(parents=True)
        with self.assertRaisesRegex(ValueError, "refusing to overwrite"):
            build_jobs(self.args)

    def test_bad_selection_stops_preflight(self):
        path = self.root / "policies/threshold_k6/dit_sr_grad_metadata.json"
        meta = json.loads(path.read_text())
        meta["selected_blocks"] = ["b0"] * 6
        path.write_text(json.dumps(meta))
        with self.assertRaisesRegex(ValueError, "uniqueness"):
            build_jobs(self.args)


if __name__ == "__main__":
    unittest.main()
