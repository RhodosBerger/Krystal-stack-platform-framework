#!/usr/bin/env python3
"""
==============================================================================
UNIT TESTS: KRYSTAL-STACK LINUX HARDWARE MONITOR & ALLOCATION UTILITY
==============================================================================
Tests for:
  - HardwareResourceCategory enumeration (8 canonical domains)
  - Allocation dataclasses and invariant validation (VITAL_MAX_HP == 6)
  - LinuxHwMonitorUtility sampler and metric resolution
  - Category filtering and CLI dashboard formatting
  - ModelCascade runtime integration in NPU_CASCADE domain
==============================================================================
"""

import unittest
from typing import Dict, Any

from krystal_kernel.scene_constructor_presets import VITAL_MAX_HP
from krystal_kernel.linux_hw_monitor_utility import (
    HardwareResourceCategory,
    HardwareAllocationSnapshot,
    CpuSchedulerAllocation,
    MemoryUmaAllocation,
    GpuAcceleratorAllocation,
    NpuCascadeAllocation,
    PowerThermalAllocation,
    StorageIoAllocation,
    NetworkSocketsAllocation,
    EpistemicInvariantsAllocation,
    LinuxHwMonitorUtility,
    GLOBAL_LINUX_HW_MONITOR
)


class TestLinuxHwMonitorUtility(unittest.TestCase):

    def setUp(self):
        self.utility = LinuxHwMonitorUtility()

    def test_01_invariant_vital_max_hp(self):
        """Verify universal invariant VITAL_MAX_HP == 6 across all category models."""
        self.assertEqual(VITAL_MAX_HP, 6)
        snap = self.utility.sample_snapshot()
        self.assertEqual(snap.vital_max_hp, 6)
        self.assertEqual(snap.cpu.vital_max_hp, 6)
        self.assertEqual(snap.memory.vital_max_hp, 6)
        self.assertEqual(snap.gpu.vital_max_hp, 6)
        self.assertEqual(snap.npu.vital_max_hp, 6)
        self.assertEqual(snap.power_thermal.vital_max_hp, 6)
        self.assertEqual(snap.storage.vital_max_hp, 6)
        self.assertEqual(snap.network.vital_max_hp, 6)
        self.assertEqual(snap.invariants.vital_max_hp, 6)

    def test_02_all_eight_categories_present(self):
        """Verify presence of all 8 specialized hardware categories."""
        expected_categories = [
            "CPU_SCHEDULER",
            "MEMORY_UMA",
            "GPU_ACCELERATOR",
            "NPU_CASCADE",
            "POWER_THERMAL",
            "STORAGE_IO",
            "NETWORK_SOCKETS",
            "EPISTEMIC_INVARIANTS"
        ]
        enum_values = [c.value for c in HardwareResourceCategory]
        for exp in expected_categories:
            self.assertIn(exp, enum_values)

        snap = self.utility.sample_snapshot()
        snap_dict = snap.to_dict()
        self.assertIn("categories", snap_dict)
        for exp in expected_categories:
            self.assertIn(exp, snap_dict["categories"])

    def test_03_cpu_and_scheduler_metrics(self):
        """Verify CPU and scheduler allocation metrics."""
        snap = self.utility.sample_snapshot()
        c = snap.cpu
        self.assertGreater(c.total_logical_cores, 0)
        self.assertGreater(c.average_frequency_mhz, 0.0)
        self.assertGreaterEqual(c.cpu_utilization_pct, 0.0)
        self.assertLessEqual(c.cpu_utilization_pct, 100.0)
        self.assertIn("slice", c.cgroups_slice)

    def test_04_memory_and_uma_metrics(self):
        """Verify RAM, Swap, and Iris Xe UMA shared allocation metrics."""
        snap = self.utility.sample_snapshot()
        m = snap.memory
        self.assertGreater(m.ram_total_mb, 0.0)
        self.assertGreater(m.ram_available_mb, 0.0)
        self.assertGreaterEqual(m.uma_shared_vram_mb, 0.0)
        self.assertGreaterEqual(m.ram_utilization_pct, 0.0)

    def test_05_npu_cascade_integration(self):
        """Verify NPU category reflects loaded ModelCascade nodes."""
        snap = self.utility.sample_snapshot()
        n = snap.npu
        self.assertTrue(n.openvino_available)
        self.assertTrue(n.dp4a_vector_mode)
        self.assertGreater(len(n.cascade_nodes_loaded), 0)
        self.assertGreater(n.total_model_vram_mb, 0.0)

        # Check that standard nodes are present
        node_ids = [node["node_id"] for node in n.cascade_nodes_loaded]
        self.assertTrue(any("classifier" in nid for nid in node_ids))
        self.assertTrue(any("verifier" in nid for nid in node_ids))

    def test_06_power_thermal_envelope(self):
        """Verify RAPL power clamps, TjMax, and thermal headroom."""
        snap = self.utility.sample_snapshot()
        p = snap.power_thermal
        self.assertGreater(p.package_power_watts, 0.0)
        self.assertGreater(p.tj_max_c, 0.0)
        self.assertGreater(p.thermal_headroom_c, 0.0)
        self.assertIsInstance(p.rapl_clamped, bool)
        self.assertIsInstance(p.thermal_throttling_engaged, bool)

    def test_07_dashboard_formatting_and_filtering(self):
        """Verify CLI dashboard rendering and category filtering."""
        snap = self.utility.sample_snapshot()
        # Full dashboard
        full_dash = self.utility.format_cli_dashboard(snap)
        self.assertIn("KRYSTAL-STACK // LINUX HARDWARE ALLOCATION MONITOR", full_dash)
        self.assertIn("VITAL_MAX_HP==6", full_dash)
        self.assertIn("[1] CPU & SCHEDULER", full_dash)
        self.assertIn("[2] MEMORY & UMA", full_dash)
        self.assertIn("[3] GPU & HARDWARE ACCELERATORS", full_dash)
        self.assertIn("[4] NPU & MODEL CASCADE", full_dash)
        self.assertIn("[5] THERMAL & POWER ENVELOPE", full_dash)
        self.assertIn("[6] STORAGE & DIRECT I/O", full_dash)
        self.assertIn("[7] NETWORK & SPATIAL SOCKETS", full_dash)
        self.assertIn("[8] EPISTEMIC INVARIANTS", full_dash)

        # Filtered dashboard (only CPU)
        cpu_dash = self.utility.format_cli_dashboard(snap, category_filter="cpu")
        self.assertIn("[1] CPU & SCHEDULER", cpu_dash)
        self.assertNotIn("[2] MEMORY & UMA", cpu_dash)
        self.assertNotIn("[6] STORAGE & DIRECT I/O", cpu_dash)


if __name__ == "__main__":
    unittest.main()
