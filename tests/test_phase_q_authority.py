import unittest

from krystal_kernel.scene_constructor_presets import VITAL_MAX_HP
from krystal_kernel.verification.authority_bits import PROFILE_TO_BIT
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


class TestPhaseQAuthority(unittest.TestCase):
    def setUp(self):
        self.assertEqual(VITAL_MAX_HP, 6)
        self.caps = CapabilityRegistry(auto_detect=False)
        self.cache = CapabilitySnapshotCache(self.caps, CATALOG)
        self.reducer = PhaseQAuthorityReducer(CATALOG, self.cache)

    def test_critical_battery_forces_ultra_saver(self):
        mask, _ = self.reducer.feasible_mask(
            battery_band="CRITICAL", thermal_band="NOMINAL"
        )
        self.assertEqual(mask, int(PROFILE_TO_BIT["ultra_saver"]))

    def test_emergency_thermal_forces_ultra_saver(self):
        mask, _ = self.reducer.feasible_mask(
            battery_band="PLUGGED", thermal_band="EMERGENCY"
        )
        self.assertEqual(mask, int(PROFILE_TO_BIT["ultra_saver"]))

    def test_normal_battery_excludes_performance_profiles(self):
        mask, _ = self.reducer.feasible_mask(
            battery_band="NORMAL", thermal_band="NOMINAL"
        )
        self.assertFalse(mask & int(PROFILE_TO_BIT["performance_gpu"]))
        self.assertFalse(mask & int(PROFILE_TO_BIT["max_burst"]))
        self.assertTrue(mask & int(PROFILE_TO_BIT["balanced_npu"]))

    def test_plugged_excludes_short_burst(self):
        mask, _ = self.reducer.feasible_mask(
            battery_band="PLUGGED", thermal_band="NOMINAL"
        )
        self.assertFalse(mask & int(PROFILE_TO_BIT["max_burst"]))

    def test_capability_generation_changes_only_on_state_change(self):
        s1 = self.cache.get_snapshot()
        s2 = self.cache.get_snapshot()
        self.assertEqual(s1.generation, s2.generation)

        self.caps.set_state(HardwareBackend.OPENVINO_GPU, CapabilityState.VERIFIED)
        s3 = self.cache.get_snapshot()
        self.assertGreater(s3.generation, s2.generation)

    def test_gpu_profile_removed_when_no_gpu_backend_is_usable(self):
        self.caps.set_state(HardwareBackend.OPENVINO_GPU, CapabilityState.UNAVAILABLE)
        self.caps.set_state(HardwareBackend.VULKAN, CapabilityState.UNAVAILABLE)
        mask, _ = self.reducer.feasible_mask(
            battery_band="PLUGGED", thermal_band="NOMINAL"
        )
        self.assertFalse(mask & int(PROFILE_TO_BIT["performance_gpu"]))


if __name__ == "__main__":
    unittest.main()
