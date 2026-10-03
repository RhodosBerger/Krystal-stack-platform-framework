"""
Unit Tests for Vulkan Iris Xe Custom Engine, Driver Harness & CPU Whisperer
===========================================================================
Verifies:
1. Iris Xe 96-EU hardware execution profile and zero-copy latency reduction.
2. CPU Instruction Whisperer flags, AVX2 SIMD profiles, and thread affinity.
3. 3-Tier Grid Memory Hierarchy (L1/L2, Host DDR4/DDR5, NVMe SSD).
4. Dispatch simulation and memory transition prefetch times.
5. Strict adherence to the 6 Max HP Vital Invariant.
"""

import unittest
from krystal_web_hub.economic_engine.vulkan_iris_xe_engine import (
    VulkanIrisXeEngine,
    IrisXeExecutionUnitsProfile,
    CpuWhispererInstructionProfile,
    MemoryHierarchyStatus,
    GLOBAL_VULKAN_IRIS_XE_ENGINE,
    VITAL_MAX_HP
)


class TestVulkanIrisXeEngine(unittest.TestCase):

    def setUp(self):
        self.engine = VulkanIrisXeEngine()

    def test_vital_max_hp_invariant(self):
        """Ensures the platform-wide 6 Max HP rule is strictly maintained."""
        self.assertEqual(VITAL_MAX_HP, 6)
        self.assertEqual(self.engine.iris_profile.vital_max_hp, 6)
        
        telemetry = self.engine.get_full_telemetry()
        self.assertEqual(telemetry["vital_max_hp_rule"], 6)

        dispatch = self.engine.simulate_dispatch(workload_chunks=8)
        self.assertEqual(dispatch["vital_max_hp"], 6)

        swap = self.engine.trigger_memory_swap((2, 3, 1))
        self.assertEqual(swap["vital_max_hp"], 6)

    def test_iris_xe_hardware_profile(self):
        """Verifies Intel Iris Xe 96 Execution Units and hardware thread calculation."""
        prof = self.engine.iris_profile
        self.assertEqual(prof.total_eus, 96)
        self.assertEqual(prof.subslice_count, 6)
        self.assertEqual(prof.eus_per_subslice, 16)
        self.assertEqual(prof.threads_per_eu, 7)
        self.assertEqual(prof.total_concurrent_threads, 672)
        
        # Verify latency reduction when zero-copy is enabled
        self.assertLess(prof.krystal_bypass_latency_us, prof.wddm_driver_latency_us)
        reduction = (1.0 - (prof.krystal_bypass_latency_us / prof.wddm_driver_latency_us)) * 100.0
        self.assertGreater(reduction, 90.0)

    def test_cpu_whisperer_instruction_profile(self):
        """Checks AVX2 cache-line prefetching and thread-switch latency reductions."""
        whisp = self.engine.whisperer_profile
        self.assertEqual(whisp.cache_line_bytes, 64)
        self.assertEqual(whisp.simd_width_bits, 256)
        self.assertEqual(whisp.prefetch_instruction, "PREFETCHT0")
        self.assertEqual(whisp.streaming_store_instruction, "VMOVNTPS")
        self.assertEqual(whisp.p_core_affinity_mask, "0x000F")
        self.assertEqual(whisp.e_core_affinity_mask, "0x00F0")
        self.assertLess(whisp.thread_switch_penalty_whisperer_ns, whisp.thread_switch_penalty_legacy_ns)

    def test_3_tier_memory_grid_hierarchy(self):
        """Validates the 3D Grid Memory tiers (L1/L2, Host VRAM, NVMe SSD)."""
        grid = self.engine.grid_status
        self.assertEqual(len(grid.tiers), 3)
        self.assertEqual(grid.tiers[0].tier_level, 0)
        self.assertEqual(grid.tiers[1].tier_level, 1)
        self.assertEqual(grid.tiers[2].tier_level, 2)
        self.assertGreater(grid.tiers[0].bandwidth_gb_s, grid.tiers[1].bandwidth_gb_s)
        self.assertGreater(grid.tiers[1].bandwidth_gb_s, grid.tiers[2].bandwidth_gb_s)

    def test_dispatch_simulation(self):
        """Verifies dispatch time calculation with and without driver bypass."""
        self.engine.zero_copy_enabled = True
        res_fast = self.engine.simulate_dispatch(workload_chunks=16, eu_load_percent=100.0)
        self.assertEqual(res_fast["active_execution_units"], 96)
        self.assertEqual(res_fast["active_hardware_threads"], 672)

        self.engine.zero_copy_enabled = False
        res_slow = self.engine.simulate_dispatch(workload_chunks=16, eu_load_percent=100.0)
        self.assertGreater(res_slow["dispatch_duration_us"], res_fast["dispatch_duration_us"])

    def test_tuning_and_swap_trigger(self):
        """Tests dynamic tuning of flags and spatial memory swap transitions."""
        tune_res = self.engine.tune_whisperer(
            enable_whisperer=True,
            enable_zero_copy=True,
            enable_npu=True,
            enable_ssd_prefetch=True
        )
        self.assertTrue(tune_res["success"])
        self.assertTrue(self.engine.whisperer_hints_enabled)
        self.assertTrue(self.engine.zero_copy_enabled)

        swap_res = self.engine.trigger_memory_swap((1, 0, 2))
        self.assertIn("grid_sector_1_0_2", swap_res["chunk_id"])
        self.assertTrue(swap_res["cache_hit"])
        self.assertGreater(swap_res["prefetch_time_ms"], 1.0)


if __name__ == '__main__':
    unittest.main()
