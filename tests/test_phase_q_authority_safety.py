import unittest

from krystal_kernel.scene_constructor_presets import VITAL_MAX_HP
from krystal_kernel.verification.authority_bits import ALL_PROFILE_BITS, PROFILE_TO_BIT, profile_mask
from krystal_kernel.verification.capability_registry import (
    CapabilityRegistry,
    CapabilityState,
    HardwareBackend,
)
from krystal_kernel.verification.config_catalog import CATALOG
from krystal_kernel.verification.phase_q_authority import (
    CapabilitySnapshotCache,
    PhaseQAuthorityReducer,
)


class TestPhaseQAuthoritySafety(unittest.TestCase):
    """Safety, veto, and TOCTOU properties for Phase Q."""

    def setUp(self):
        self.assertEqual(VITAL_MAX_HP, 6)
        self.caps = CapabilityRegistry(auto_detect=False)
        self.cache = CapabilitySnapshotCache(self.caps, CATALOG)
        self.reducer = PhaseQAuthorityReducer(CATALOG, self.cache)

    def test_requested_mask_is_respected(self):
        requested = profile_mask("ultra_saver", "balanced_npu")
        result, _ = self.reducer.feasible_mask(
            battery_band="PLUGGED",
            thermal_band="NOMINAL",
            requested_mask=requested,
        )
        self.assertEqual(result & ~requested, 0)

    def test_evidence_veto_cannot_be_bypassed(self):
        result, _ = self.reducer.feasible_mask(
            battery_band="PLUGGED",
            thermal_band="NOMINAL",
            evidence_mask=0,
        )
        self.assertEqual(result, 0)

    def test_policy_veto_cannot_be_bypassed(self):
        result, _ = self.reducer.feasible_mask(
            battery_band="PLUGGED",
            thermal_band="NOMINAL",
            policy_mask=0,
        )
        self.assertEqual(result, 0)

    def test_empty_requested_set_remains_empty(self):
        result, _ = self.reducer.feasible_mask(
            battery_band="PLUGGED",
            thermal_band="NOMINAL",
            requested_mask=0,
        )
        self.assertEqual(result, 0)

    def test_reserved_bits_are_removed(self):
        reserved_only = 1 << 63
        result, _ = self.reducer.feasible_mask(
            battery_band="PLUGGED",
            thermal_band="NOMINAL",
            requested_mask=ALL_PROFILE_BITS | reserved_only,
            evidence_mask=ALL_PROFILE_BITS | reserved_only,
            policy_mask=ALL_PROFILE_BITS | reserved_only,
        )
        self.assertEqual(result & ~ALL_PROFILE_BITS, 0)

    def test_generation_is_stable_without_capability_change(self):
        a = self.cache.get_snapshot()
        b = self.cache.get_snapshot()
        c = self.cache.get_snapshot()
        self.assertEqual(a.generation, b.generation)
        self.assertEqual(b.generation, c.generation)

    def test_generation_changes_exactly_once_per_state_change(self):
        s0 = self.cache.get_snapshot()
        self.caps.set_state(HardwareBackend.OPENVINO_GPU, CapabilityState.VERIFIED)
        s1 = self.cache.get_snapshot()
        s2 = self.cache.get_snapshot()
        self.assertEqual(s1.generation, s0.generation + 1)
        self.assertEqual(s2.generation, s1.generation)

    def test_capability_mask_changes_when_gpu_becomes_unavailable(self):
        self.caps.set_state(HardwareBackend.OPENVINO_GPU, CapabilityState.VERIFIED)
        before = self.cache.get_snapshot()
        self.assertTrue(before.capability_mask & int(PROFILE_TO_BIT["performance_gpu"]))

        self.caps.set_state(HardwareBackend.OPENVINO_GPU, CapabilityState.UNAVAILABLE)
        self.caps.set_state(HardwareBackend.VULKAN, CapabilityState.UNAVAILABLE)
        after = self.cache.get_snapshot()
        self.assertFalse(after.capability_mask & int(PROFILE_TO_BIT["performance_gpu"]))
        self.assertGreater(after.generation, before.generation)

    def test_profiles_from_mask_roundtrip(self):
        mask = profile_mask("ultra_saver", "balanced_npu", "balanced_hybrid")
        profiles = self.reducer.profiles_from_mask(mask)
        self.assertEqual({p.name for p in profiles}, {"ultra_saver", "balanced_npu", "balanced_hybrid"})


if __name__ == "__main__":
    unittest.main()
