import unittest

from profile_resshift_grad import block_from_qkv, noise_ratio_to_timestep


class ResShiftGradientProfilerTests(unittest.TestCase):
    def test_noise_ratio_mapping_for_fifteen_steps(self):
        ratios = [0.05, 0.2, 0.4, 0.6, 0.8, 0.95]
        self.assertEqual(
            [noise_ratio_to_timestep(value, 15) for value in ratios],
            [1, 3, 6, 8, 11, 13],
        )

    def test_qkv_module_maps_to_block(self):
        name = "output_blocks.9.1.blocks.1.attn.qkv"
        self.assertEqual(block_from_qkv(name), "output_blocks.9.1.blocks.1")

    def test_invalid_qkv_name_is_rejected(self):
        with self.assertRaises(ValueError):
            block_from_qkv("output_blocks.9.1.blocks.1.attn.proj")


if __name__ == "__main__":
    unittest.main()
