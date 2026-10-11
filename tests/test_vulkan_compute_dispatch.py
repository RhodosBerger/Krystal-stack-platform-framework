"""
Krystal-Stack Automated Verification: Native Vulkan Compute Dispatch on Intel Iris Xe
=====================================================================================
Standard discoverable unittest suite verifying:
1. Physical device discovery & Intel Iris Xe queue family identification
2. Logical device creation (vkCreateDevice)
3. Compute queue creation & retrieval (vkGetDeviceQueue)
4. Host-visible SSBO & UBO buffer allocations
5. Descriptor set layout, pool, and descriptor set creation
6. SPIR-V compute shader loading & pipeline creation (vkCreateComputePipelines)
7. Dispatch dimension calculations (workgroups x, y, z)
8. CPU reference fallback guarantee upon hardware bypass/failure
9. Output tensor shapes (glyphs & packed truecolor RGB)
10. Correctness comparison: GPU hardware output vs CPU mathematical oracle
11. Real GPU dispatch flag lifecycle & telemetry contract
12. Resource cleanup and destruction idempotency
"""

import sys
import os
import unittest

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.python.vulkan_compute_driver import VulkanComputeDriver


class TestVulkanComputeDispatch(unittest.TestCase):
    """
    Verification suite for native Vulkan compute hardware dispatch on Intel Iris Xe.
    """

    def setUp(self):
        self.driver = VulkanComputeDriver(prefer_gpu=True)

    def tearDown(self):
        if self.driver:
            self.driver.cleanup()
            self.driver = None

    def test_device_discovery(self):
        """Verifies Vulkan physical device detection and Intel Iris Xe enumeration."""
        if not self.driver.vulkan_available:
            self.skipTest("Vulkan runtime / physical device unavailable on this host.")

        self.assertTrue(self.driver.device_detected)
        self.assertIsNotNone(self.driver.physical_device)
        self.assertGreaterEqual(self.driver.compute_queue_family, 0)
        self.assertTrue(self.driver.is_intel_iris or self.driver.gpu_accelerated)

    def test_logical_device_creation(self):
        """Verifies vkCreateDevice creates a valid logical device handle."""
        if not self.driver.vulkan_available:
            self.skipTest("Vulkan runtime unavailable.")

        self.assertTrue(self.driver.logical_device_created)
        self.assertIsNotNone(self.driver.device)
        self.assertNotEqual(self.driver.device.value, 0)

    def test_compute_queue_creation(self):
        """Verifies compute queue retrieval via vkGetDeviceQueue."""
        if not self.driver.vulkan_available:
            self.skipTest("Vulkan runtime unavailable.")

        self.assertTrue(self.driver.compute_queue_ready)
        self.assertIsNotNone(self.driver.queue)
        self.assertNotEqual(self.driver.queue.value, 0)

    def test_buffer_allocation(self):
        """Verifies host-visible coherent SSBO and UBO allocations for 96x40 viewport."""
        if not self.driver.pipeline_ready:
            self.skipTest("Hardware pipeline not initialized.")

        self.driver._ensure_buffers(96, 40)
        self.assertIsNotNone(self.driver.glyph_buffer)
        self.assertIsNotNone(self.driver.glyph_mapped)
        self.assertIsNotNone(self.driver.rgb_buffer)
        self.assertIsNotNone(self.driver.rgb_mapped)
        self.assertIsNotNone(self.driver.ubo_buffer)
        self.assertIsNotNone(self.driver.ubo_mapped)
        self.assertEqual(self.driver.buffer_width, 96)
        self.assertEqual(self.driver.buffer_height, 40)

    def test_descriptor_layout(self):
        """Verifies descriptor set layout with bindings 0, 1 (storage) and 2 (uniform)."""
        if not self.driver.pipeline_ready:
            self.skipTest("Hardware pipeline not initialized.")

        self.assertIsNotNone(self.driver.descriptor_set_layout)
        self.assertNotEqual(self.driver.descriptor_set_layout.value, 0)
        self.assertIsNotNone(self.driver.pipeline_layout)
        self.assertNotEqual(self.driver.pipeline_layout.value, 0)

    def test_compute_pipeline_creation(self):
        """Verifies SPIR-V shader module and vkCreateComputePipelines readiness."""
        if not self.driver.vulkan_available:
            self.skipTest("Vulkan runtime unavailable.")

        self.assertTrue(self.driver.pipeline_ready)
        self.assertIsNotNone(self.driver.shader_module)
        self.assertIsNotNone(self.driver.pipeline)
        self.assertNotEqual(self.driver.pipeline.value, 0)

    def test_dispatch_dimensions(self):
        """Verifies workgroup dispatch dimension calculations with ceil rounding."""
        width, height = 96, 40
        lx, ly, lz = self.driver.workgroup_size
        self.assertEqual(lx, 8)
        self.assertEqual(ly, 8)
        self.assertEqual(lz, 1)

        groups_x = (width + lx - 1) // lx
        groups_y = (height + ly - 1) // ly
        self.assertEqual(groups_x, 12)
        self.assertEqual(groups_y, 5)

        # Test non-divisible dimensions
        w2, h2 = 97, 41
        g2_x = (w2 + lx - 1) // lx
        g2_y = (h2 + ly - 1) // ly
        self.assertEqual(g2_x, 13)
        self.assertEqual(g2_y, 6)

    def test_cpu_fallback(self):
        """Verifies deterministic CPU fallback when prefer_gpu is False or on error."""
        driver_cpu = VulkanComputeDriver(prefer_gpu=False)
        try:
            glyphs, colors = driver_cpu.execute_raymarch(96, 40, 0.5)
            self.assertEqual(len(glyphs), 3840)
            self.assertEqual(len(colors), 3840)
            self.assertFalse(driver_cpu.real_gpu_dispatch)
            self.assertEqual(driver_cpu.compute_backend, "CPU_EMULATION")
        finally:
            driver_cpu.cleanup()

    def test_output_shape(self):
        """Verifies output shapes and element bounds for 96x40 viewport."""
        if not self.driver.pipeline_ready:
            self.skipTest("Hardware pipeline not initialized.")

        glyphs, colors = self.driver.execute_raymarch(96, 40, 0.5)
        self.assertEqual(len(glyphs), 3840)
        self.assertEqual(len(colors), 3840)

        # Invariant checks on outputs
        self.assertTrue(all(0 <= g <= 4 for g in glyphs))
        self.assertTrue(all(0 <= c <= 0xFFFFFF for c in colors))

    def test_gpu_vs_cpu_correctness(self):
        """Verifies numerical match between native GPU dispatch and CPU reference oracle."""
        if not self.driver.pipeline_ready:
            self.skipTest("Hardware pipeline not initialized.")

        res = self.driver.verify_output_against_cpu(96, 40, 0.5)
        self.assertEqual(res["status"], "VERIFIED")
        metrics = res["metrics"]

        # Documented acceptance criteria: >= 99.0% glyph & pixel match
        self.assertGreaterEqual(metrics["glyph_match_ratio"], 0.99)
        self.assertGreaterEqual(metrics["pixel_match_ratio"], 0.99)
        self.assertLessEqual(metrics["mean_absolute_error"], 5.0)

    def test_real_gpu_dispatch_flag(self):
        """Verifies real_gpu_dispatch flag is strictly True during GPU execution."""
        if not self.driver.pipeline_ready:
            self.skipTest("Hardware pipeline not initialized.")

        self.driver.execute_raymarch(96, 40, 0.5)
        self.assertTrue(self.driver.real_gpu_dispatch)
        self.assertEqual(self.driver.compute_backend, "VULKAN_GPU")

        telemetry = self.driver.get_telemetry()
        self.assertTrue(telemetry["real_gpu_dispatch"])
        self.assertEqual(telemetry["compute_backend"], "VULKAN_GPU")
        self.assertFalse(telemetry["readback_is_simulated"])
        self.assertGreater(telemetry["readback_mb_s"], 0.0)

    def test_cleanup_idempotency(self):
        """Verifies driver.cleanup() can be called multiple times safely without error."""
        self.driver.cleanup()
        self.assertIsNone(self.driver.device)
        self.assertIsNone(self.driver.pipeline)
        # Calling cleanup again must succeed cleanly
        self.driver.cleanup()


if __name__ == "__main__":
    unittest.main()
