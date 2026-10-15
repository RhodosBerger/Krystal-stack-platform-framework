import unittest

from krystal_kernel.scene_constructor_presets import VITAL_MAX_HP
from krystal_kernel.verification.authority_bits import (
    ALL_PROFILE_BITS,
    AuthorityProfileBit,
    BIT_TO_PROFILE,
    PROFILE_TO_BIT,
    mask_contains,
    profile_mask,
    reduce_authority_masks,
)


class TestAuthorityBitsContract(unittest.TestCase):
    """Contract tests for the generated authority bit representation."""

    def test_constitutional_invariant(self):
        self.assertEqual(VITAL_MAX_HP, 6)

    def test_profile_bits_are_unique_single_bit_values(self):
        values = [int(v) for v in PROFILE_TO_BIT.values()]
        self.assertEqual(len(values), len(set(values)))
        for value in values:
            self.assertGreater(value, 0)
            self.assertEqual(value & (value - 1), 0, f"{value:#x} is not a power-of-two bit")

    def test_only_bits_zero_through_five_are_allocated(self):
        self.assertEqual(ALL_PROFILE_BITS, 0x3F)
        self.assertEqual(ALL_PROFILE_BITS >> 6, 0)

    def test_forward_reverse_mapping_is_bijective(self):
        for profile_id, bit in PROFILE_TO_BIT.items():
            self.assertEqual(BIT_TO_PROFILE[int(bit)], profile_id)

    def test_profile_mask_and_contains(self):
        mask = profile_mask("ultra_saver", "balanced_npu", "performance_gpu")
        self.assertTrue(mask_contains(mask, "ultra_saver"))
        self.assertTrue(mask_contains(mask, "balanced_npu"))
        self.assertTrue(mask_contains(mask, "performance_gpu"))
        self.assertFalse(mask_contains(mask, "max_burst"))

    def test_reduce_authority_masks_is_true_intersection(self):
        requested = profile_mask("balanced_npu", "balanced_hybrid", "performance_gpu")
        thermal = profile_mask("ultra_saver", "balanced_npu", "balanced_hybrid")
        battery = profile_mask("balanced_npu", "balanced_hybrid")
        capability = profile_mask("balanced_npu", "performance_gpu")
        evidence = ALL_PROFILE_BITS
        policy = profile_mask("balanced_npu", "performance_gpu")

        result = reduce_authority_masks(
            requested,
            thermal,
            battery,
            capability,
            evidence,
            policy,
        )
        self.assertEqual(result, int(AuthorityProfileBit.BALANCED_NPU))

    def test_any_empty_gate_vetoes_everything(self):
        full = ALL_PROFILE_BITS
        for gate_index in range(6):
            gates = [full] * 6
            gates[gate_index] = 0
            with self.subTest(gate_index=gate_index):
                self.assertEqual(reduce_authority_masks(*gates), 0)


if __name__ == "__main__":
    unittest.main()
