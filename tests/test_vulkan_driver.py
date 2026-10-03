"""
Unit Tests for Vulkan Compute Driver (Zero-Compiler Ctypes Engine)
==================================================================
Tests:
1. Dynamic loading and device enumeration (Intel Iris Xe discovery).
2. Telemetry structure and hardware property validation.
3. Dual SSBO buffer generation and dimension verification.
4. Color packing and glyph index bounds.
5. Autonomous CPU SIMD fallback validation.
"""

import unittest
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.python.vulkan_compute_driver import VulkanComputeDriver


class TestVulkanComputeDriver(unittest.TestCase):

    def setUp(self):
        self.driver = VulkanComputeDriver()

    def test_01_device_discovery(self):
        """Driver initializes and attempts to find physical Vulkan GPU."""
        telemetry = self.driver.get_telemetry()
        self.assertIn("vulkan_available", telemetry)
        self.assertIn("device_name", telemetry)
        self.assertIn("vendor_id", telemetry)
        self.assertIn("ssbo_readback_kb_per_frame", telemetry)
        # On Windows host with vulkan-1.dll, Intel Iris Xe should be discovered
        if os.path.exists("C:\\Windows\\System32\\vulkan-1.dll"):
            self.assertTrue(telemetry["vulkan_available"])
            self.assertTrue(telemetry["gpu_accelerated"])
            self.assertIn("Intel", telemetry["device_name"])

    def test_02_dual_ssbo_execution(self):
        """Raymarching pass returns dual SSBO buffers with exact pixel count."""
        w, h = 48, 20
        glyphs, colors = self.driver.execute_raymarch(width=w, height=h, t=0.5)

        self.assertEqual(len(glyphs), w * h)
        self.assertEqual(len(colors), w * h)

        # Check bounds: glyphs in [0..4]
        for g in glyphs:
            self.assertGreaterEqual(g, 0)
            self.assertLessEqual(g, 4)

        # Check colors: packed RGB in [0..0xFFFFFF]
        for c in colors:
            self.assertGreaterEqual(c, 0)
            self.assertLessEqual(c, 0xFFFFFF)

    def test_03_telemetry_metrics_update(self):
        """Executing a dispatch updates latency and dispatch count metrics."""
        self.driver.execute_raymarch(width=32, height=16, t=1.0)
        telemetry = self.driver.get_telemetry()

        self.assertGreater(telemetry["dispatch_count"], 0)
        self.assertGreater(telemetry["last_dispatch_us"], 0.0)
        self.assertEqual(telemetry["ssbo_readback_kb_per_frame"], 19.2)

    def test_04_fallback_state(self):
        """Driver handles toggling to CPU fallback without crashing."""
        self.driver.gpu_accelerated = False
        self.assertFalse(self.driver.gpu_accelerated)
        g, c = self.driver.execute_raymarch(width=16, height=8, t=0.0)
        self.assertEqual(len(g), 16 * 8)
        self.assertEqual(len(c), 16 * 8)


if __name__ == "__main__":
    unittest.main()
