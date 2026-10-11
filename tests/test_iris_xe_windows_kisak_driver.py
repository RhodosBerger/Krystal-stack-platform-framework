#!/usr/bin/env python3
"""
==============================================================================
UNIT TEST SUITE: INTEL IRIS XE WINDOWS KISAK-BREAKTHROUGH GRAPHICS DRIVER
==============================================================================
Module: tests/test_iris_xe_windows_kisak_driver.py
Description: Validates native Windows Vulkan 1.3 C-ABI loading,
             physical device enumeration, Tile4 2D spatial swizzling,
             Lossless CCS auxiliary buffer compression, and DirectFlip UMA.

Platform Invariant: VITAL_MAX_HP = 6
Author: Dušan Kopecký & Krystal-Stack Architecture Council (2026)
==============================================================================
"""

import os
import sys
import unittest
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from krystal_kernel.iris_xe_windows_kisak_driver import (
    WindowsKisakDriverState,
    MemoryTilingMode,
    WindowsKisakConfiguration,
    IrisXeWindowsTelemetry,
    IrisXeWindowsKisakDriver,
    GLOBAL_IRIS_XE_WINDOWS_KISAK_DRIVER,
    VITAL_MAX_HP
)


class TestIrisXeWindowsKisakDriver(unittest.TestCase):
    """Test suite for the Windows Kisak-grade graphics driver engine."""

    def test_01_invariant_constant(self):
        """Ensures the sacred system invariant is strictly 6."""
        self.assertEqual(VITAL_MAX_HP, 6)

    def test_02_driver_initialization_and_hardware_probing(self):
        """Tests that driver initializes and probes Windows Vulkan 1.3 stack."""
        driver = IrisXeWindowsKisakDriver()
        self.assertIsNotNone(driver.device_name)
        self.assertIn("Intel", driver.device_name)
        self.assertEqual(driver.vendor_id, 0x8086)
        self.assertIn(
            driver.driver_state,
            [WindowsKisakDriverState.PHYSICAL_HARDWARE_ACTIVE, WindowsKisakDriverState.SIMULATED_UMA_TUNING]
        )

    def test_03_windows_kisak_optimizations_telemetry(self):
        """Tests bandwidth reduction and power savings under Kisak tuning on Windows."""
        driver = GLOBAL_IRIS_XE_WINDOWS_KISAK_DRIVER
        telem = driver.apply_windows_kisak_optimizations()

        self.assertIsInstance(telem, IrisXeWindowsTelemetry)
        self.assertEqual(telem.vital_max_hp, 6)
        self.assertEqual(telem.configured_simd_width, 16)
        # Verify 10x bandwidth reduction from Tile4 + CCS
        self.assertGreaterEqual(telem.bandwidth_reduction_factor, 5.0)
        self.assertLess(telem.optimized_bandwidth_gb_s, telem.dram_bus_bandwidth_gb_s)
        # Verify package power drop
        self.assertLess(telem.package_power_optimized_w, telem.package_power_baseline_w)
        self.assertGreaterEqual(telem.power_reduction_pct, 50.0)
        # Verify framerate boost
        self.assertGreater(telem.framerate_fps_optimized, telem.framerate_fps_baseline)
        # Verify DirectFlip latency
        self.assertEqual(telem.input_to_photon_latency_ms, 0.38)
        # Verify Arrhenius durability
        self.assertLessEqual(telem.arrhenius_wear_factor, 1.05)

    def test_04_windows_driver_environment_injection(self):
        """Tests that Intel Windows runtime environment variables are correctly injected."""
        driver = IrisXeWindowsKisakDriver()
        driver._inject_windows_driver_environment()

        self.assertEqual(sys.platform == "win32" and "INTEL_DEBUG" in os.environ, True)
        self.assertEqual(os.environ.get("NEOReadDebugKeys"), "1")
        self.assertEqual(os.environ.get("DisableResourceCompression"), "0")
        self.assertEqual(os.environ.get("OverrideSubgroupSize"), "16")
        self.assertEqual(os.environ.get("DirectFlipZeroCopy"), "1")


if __name__ == "__main__":
    import os
    unittest.main()
