#!/usr/bin/env python3

import unittest

import torch
from torch import nn

import eval_hf_dit4sr_sr_metrics as evaluate
import profile_hf_dit4sr_grad as core


class MergeLoRAForEvalTests(unittest.TestCase):
    def test_merge_preserves_forward_and_removes_wrappers(self):
        torch.manual_seed(7)
        model = nn.Sequential(nn.Linear(5, 4, bias=True))
        original = model[0]
        wrapper = core.LoRALinear(original, rank=2, alpha=4)
        wrapper.lora_down.weight.data.normal_()
        wrapper.lora_up.weight.data.normal_()
        model[0] = wrapper

        value = torch.randn(3, 5)
        expected = model(value)
        report = evaluate.merge_lora_into_base_for_eval(model)
        actual = model(value)

        self.assertEqual(report["merged_module_count"], 1)
        self.assertEqual(report["active_merged_module_count"], 1)
        self.assertIs(model[0], original)
        self.assertEqual(list(core.iter_lora_modules(model)), [])
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)

    def test_disabled_zero_lora_is_removed_without_changing_weight(self):
        model = nn.Sequential(nn.Linear(3, 2, bias=False))
        original = model[0]
        wrapper = core.LoRALinear(original, rank=1, alpha=1)
        wrapper.lora_enabled = False
        model[0] = wrapper
        weight = original.weight.detach().clone()

        report = evaluate.merge_lora_into_base_for_eval(model)

        self.assertEqual(report["merged_module_count"], 1)
        self.assertEqual(report["active_merged_module_count"], 0)
        torch.testing.assert_close(model[0].weight, weight)


if __name__ == "__main__":
    unittest.main()
