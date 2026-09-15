import unittest

from inspect_resshift_structure import block_from_target, stage_from_name


class ResShiftStructureTests(unittest.TestCase):
    def test_stage_mapping(self):
        self.assertEqual(stage_from_name("input_blocks.1.1.blocks.0"), "input")
        self.assertEqual(stage_from_name("middle_block.1.blocks.0"), "middle")
        self.assertEqual(stage_from_name("output_blocks.2.1.blocks.0"), "output")

    def test_qkv_target_maps_to_swin_block(self):
        target = "input_blocks.1.1.blocks.0.attn.qkv"
        self.assertEqual(block_from_target(target), "input_blocks.1.1.blocks.0")


if __name__ == "__main__":
    unittest.main()
