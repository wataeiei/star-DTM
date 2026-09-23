import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn as nn

import adaptive_grad_blockskip as adaptive
import profile_hf_dit4sr_grad as core
from train_hf_dit4sr_all_lora_importance import (
    configure_trainable_lora_blocks,
    load_initial_lora_adapter,
    merge_lora_into_base,
)


class TinyTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer_blocks = nn.ModuleList(
            [nn.ModuleDict({"to_q": nn.Linear(4, 4, bias=False)}) for _ in range(2)]
        )


def inject_all(model):
    model.requires_grad_(False)
    return core.inject_lora(
        model,
        target="q",
        rank=2,
        alpha=4,
        block_regex="",
        selected_blocks={"transformer_blocks.0", "transformer_blocks.1"},
    )


class InitialAdapterTests(unittest.TestCase):
    def test_strict_load_and_selected_train_scope(self):
        source = TinyTransformer()
        inject_all(source)
        for index, (_name, module) in enumerate(core.iter_lora_modules(source), start=1):
            module.lora_down.weight.data.fill_(index)
            module.lora_up.weight.data.fill_(index + 0.5)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "adapter.pt"
            adaptive.save_lora_adapter(source, path)

            target = TinyTransformer()
            inject_all(target)
            report = load_initial_lora_adapter(target, path, strict=True)
            scope = configure_trainable_lora_blocks(
                target, {"transformer_blocks.1"}, ""
            )

        self.assertEqual(report["loaded"], 2)
        self.assertEqual(len(scope["trainable_modules"]), 1)
        self.assertEqual(len(scope["frozen_modules"]), 1)
        modules = dict(core.iter_lora_modules(target))
        self.assertFalse(modules["transformer_blocks.0.to_q"].lora_up.weight.requires_grad)
        self.assertTrue(modules["transformer_blocks.1.to_q"].lora_up.weight.requires_grad)
        self.assertTrue(
            torch.allclose(
                modules["transformer_blocks.1.to_q"].lora_up.weight,
                torch.full_like(modules["transformer_blocks.1.to_q"].lora_up.weight, 2.5),
            )
        )

    def test_strict_load_rejects_module_mismatch(self):
        source = TinyTransformer()
        inject_all(source)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "adapter.pt"
            adaptive.save_lora_adapter(source, path)

            target = TinyTransformer()
            target.requires_grad_(False)
            core.inject_lora(
                target,
                target="q",
                rank=2,
                alpha=4,
                block_regex="",
                selected_blocks={"transformer_blocks.0"},
            )
            with self.assertRaises(SystemExit):
                load_initial_lora_adapter(target, path, strict=True)

    def test_merge_preserves_forward_and_removes_wrappers(self):
        model = TinyTransformer()
        inject_all(model)
        for index, (_name, module) in enumerate(core.iter_lora_modules(model), start=1):
            module.lora_down.weight.data.fill_(index * 0.1)
            module.lora_up.weight.data.fill_(index * 0.2)

        inputs = [torch.randn(3, 4), torch.randn(3, 4)]
        before = [
            model.transformer_blocks[index]["to_q"](value)
            for index, value in enumerate(inputs)
        ]
        report = merge_lora_into_base(model)
        after = [
            model.transformer_blocks[index]["to_q"](value)
            for index, value in enumerate(inputs)
        ]

        self.assertEqual(report["merged_module_count"], 2)
        self.assertEqual(list(core.iter_lora_modules(model)), [])
        for expected, actual in zip(before, after):
            self.assertTrue(torch.allclose(expected, actual, atol=1e-6, rtol=1e-6))


if __name__ == "__main__":
    unittest.main()
