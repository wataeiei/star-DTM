import unittest

from train_resshift_lora_importance import aggregate_importance, choose_blocks


class ResShiftSelectionTests(unittest.TestCase):
    def test_per_noise_normalization_selects_above_mean_blocks(self):
        rows = []
        for noise, scores in ((0.1, [6.0, 3.0, 1.0]), (0.9, [3.0, 6.0, 1.0])):
            for index, score in enumerate(scores):
                rows.append({
                    "train_step": "0",
                    "noise_ratio": str(noise),
                    "block": f"block.{index}",
                    "block_index": str(index),
                    "normalized_grad_score": str(score),
                })
        details = aggregate_importance(rows, 0, "normalized_grad_score")
        selected, report = choose_blocks(
            ["block.0", "block.1", "block.2"], details, "threshold", 1.0, 2
        )
        self.assertEqual(selected, ["block.0", "block.1"])
        self.assertAlmostEqual(report["selected_utility"], 0.9)

    def test_topk_preserves_model_order(self):
        details = [
            {"block": "block.2", "mean_score_share": 0.5},
            {"block": "block.0", "mean_score_share": 0.3},
            {"block": "block.1", "mean_score_share": 0.2},
        ]
        selected, _ = choose_blocks(
            ["block.0", "block.1", "block.2"], details, "topk", 0.0, 2
        )
        self.assertEqual(selected, ["block.0", "block.2"])


if __name__ == "__main__":
    unittest.main()
