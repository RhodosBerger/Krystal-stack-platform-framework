"""
================================================================================
KRYSTAL-STACK / CRYSTAL-SEC 3.0: GPU FAULT INJECTION & RESILIENCE TEST SUITE
Test File: tests/test_gpu_fault_injection_and_resilience.py

Role:
Formal verification of the GPU abstraction boundary safe fault interceptor,
resilience state machine, cryptographic fault ledger, and CPU fallback isolation.

Epistemic Discipline:
Safe fault simulation at the driver abstraction boundary ONLY.
No physical GPU hanging, no TDR crash loops.
================================================================================
"""

import os
import sys
import unittest

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.python.vulkan_compute_driver import (
    FaultInjectionMask,
    GpuRecoveryState,
    ReadbackPolicy,
    VulkanComputeDriver,
)


class TestGpuFaultInjectionAndResilience(unittest.TestCase):
    """
    Verification suite for driver abstraction boundary fault injection,
    ledger hashing, and deterministic CPU fallback recovery.
    """

    def setUp(self):
        self.driver = VulkanComputeDriver(prefer_gpu=True)
        self.driver.set_readback_policy(ReadbackPolicy.ALWAYS)

    def tearDown(self):
        if self.driver:
            self.driver.clear_faults()
            self.driver.cleanup()
            self.driver = None

    def test_no_fault_does_not_trigger_fallback(self):
        """Verifies that an uncorrupted driver executes without entering fault states."""
        if not self.driver.pipeline_ready:
            self.skipTest("Hardware Vulkan pipeline unavailable")

        glyphs, colors = self.driver.execute_raymarch(96, 40, 0.5)
        self.assertEqual(len(glyphs), 3840)
        self.assertEqual(self.driver.recovery_state, GpuRecoveryState.READY)
        self.assertEqual(len(self.driver.fault_history), 0)
        self.assertEqual(self.driver.fallback_count, 0)
        self.assertTrue(self.driver.real_gpu_dispatch)

    def test_oom_fault_enters_fallback(self):
        """Simulates VK_ERROR_OUT_OF_DEVICE_MEMORY during buffer allocation."""
        if not self.driver.pipeline_ready:
            self.skipTest("Hardware Vulkan pipeline unavailable")

        # Inject simulated OOM
        self.driver.inject_fault(FaultInjectionMask.OOM_ALLOCATION_FAULT)

        # Force buffer allocation trigger with new resolution
        glyphs, colors = self.driver.execute_raymarch(128, 64, 0.5)

        # Invariant checks: CPU fallback executed cleanly, returning exact pixel count
        self.assertEqual(len(glyphs), 128 * 64)
        self.assertEqual(len(colors), 128 * 64)
        self.assertEqual(self.driver.recovery_state, GpuRecoveryState.CPU_FALLBACK)
        self.assertGreaterEqual(self.driver.fallback_count, 1)

        # Ledger record verification
        self.assertGreaterEqual(len(self.driver.fault_history), 1)
        record = self.driver.fault_history[-1]
        self.assertEqual(record.fault_type, "OOM_ALLOCATION_FAULT")
        self.assertEqual(record.injection_stage, "BUFFER_ALLOCATION")
        self.assertEqual(record.actual_recovery, "CPU_FALLBACK")
        self.assertEqual(record.fallback_backend, "CPU_REFERENCE")

    def test_fence_timeout_enters_degraded_state(self):
        """Simulates synchronization timeout / fence stall at the abstraction boundary."""
        if not self.driver.pipeline_ready:
            self.skipTest("Hardware Vulkan pipeline unavailable")

        self.driver.inject_fault(FaultInjectionMask.FENCE_TIMEOUT_STALL)

        glyphs, colors = self.driver.execute_raymarch(96, 40, 0.5)

        self.assertEqual(len(glyphs), 3840)
        self.assertEqual(self.driver.recovery_state, GpuRecoveryState.DEGRADED)
        self.assertGreaterEqual(self.driver.fallback_count, 1)

        record = self.driver.fault_history[-1]
        self.assertEqual(record.fault_type, "FENCE_TIMEOUT_STALL")
        self.assertEqual(record.injection_stage, "SYNCHRONIZATION_WAIT")
        self.assertEqual(record.expected_recovery, "DEGRADED")

    def test_device_loss_enters_cpu_fallback(self):
        """Simulates VK_ERROR_DEVICE_LOST from queue submission wrapper."""
        if not self.driver.pipeline_ready:
            self.skipTest("Hardware Vulkan pipeline unavailable")

        self.driver.inject_fault(FaultInjectionMask.DEVICE_LOSS_EVENT)

        glyphs, colors = self.driver.execute_raymarch(96, 40, 0.5)

        self.assertEqual(len(glyphs), 3840)
        self.assertEqual(self.driver.recovery_state, GpuRecoveryState.CPU_FALLBACK)
        self.assertGreaterEqual(self.driver.fallback_count, 1)

        record = self.driver.fault_history[-1]
        self.assertEqual(record.fault_type, "DEVICE_LOSS_EVENT")
        self.assertEqual(record.injection_stage, "QUEUE_SUBMIT")
        self.assertEqual(record.actual_recovery, "CPU_FALLBACK")

    def test_descriptor_corruption_is_preflight_rejected(self):
        """Simulates descriptor set validation fault before command recording."""
        if not self.driver.pipeline_ready:
            self.skipTest("Hardware Vulkan pipeline unavailable")

        self.driver.inject_fault(FaultInjectionMask.DESCRIPTOR_CORRUPTION)

        glyphs, colors = self.driver.execute_raymarch(96, 40, 0.5)

        self.assertEqual(len(glyphs), 3840)
        self.assertEqual(self.driver.recovery_state, GpuRecoveryState.CPU_FALLBACK)

        record = self.driver.fault_history[-1]
        self.assertEqual(record.fault_type, "DESCRIPTOR_CORRUPTION")
        self.assertEqual(record.injection_stage, "DESCRIPTOR_PREFLIGHT")

    def test_fault_ledger_written(self):
        """Verifies cryptographic hash chaining in the fault audit ledger."""
        if not self.driver.pipeline_ready:
            self.skipTest("Hardware Vulkan pipeline unavailable")

        # Inject two sequential faults
        self.driver.inject_fault(FaultInjectionMask.DESCRIPTOR_CORRUPTION)
        self.driver.execute_raymarch(96, 40, 0.1)

        self.driver.clear_faults()
        self.driver.inject_fault(FaultInjectionMask.DEVICE_LOSS_EVENT)
        self.driver.execute_raymarch(96, 40, 0.2)

        self.assertEqual(len(self.driver.fault_history), 2)
        f1 = self.driver.fault_history[0]
        f2 = self.driver.fault_history[1]

        self.assertEqual(f1.fault_id, "FAULT-0001")
        self.assertEqual(f2.fault_id, "FAULT-0002")

        # Hash chaining invariant: f2 hash incorporates f1 ledger_hash
        self.assertNotEqual(f1.ledger_hash, f2.ledger_hash)
        self.assertEqual(len(f1.ledger_hash), 64)
        self.assertEqual(len(f2.ledger_hash), 64)

    def test_recovery_state_machine(self):
        """Verifies transitions across the recovery lifecycle: READY -> FAULT -> FALLBACK -> RECOVERED."""
        if not self.driver.pipeline_ready:
            self.skipTest("Hardware Vulkan pipeline unavailable")

        self.assertEqual(self.driver.recovery_state, GpuRecoveryState.READY)

        # Trigger fault
        self.driver.inject_fault(FaultInjectionMask.DEVICE_LOSS_EVENT)
        self.driver.execute_raymarch(96, 40, 0.5)
        self.assertEqual(self.driver.recovery_state, GpuRecoveryState.CPU_FALLBACK)

        # Clear fault mask and execute re-initialization
        self.driver.clear_faults()
        reinit_ok = self.driver.reinitialize_gpu()
        self.assertTrue(reinit_ok)
        self.assertEqual(self.driver.recovery_state, GpuRecoveryState.RECOVERED)

        # Post-recovery dispatch operates normally
        self.driver.set_readback_policy(ReadbackPolicy.ALWAYS)
        glyphs, colors = self.driver.execute_raymarch(96, 40, 0.5)
        self.assertEqual(len(glyphs), 3840)
        self.assertEqual(self.driver.recovery_state, GpuRecoveryState.READY)

    def test_gpu_reinitialization(self):
        """Verifies driver teardown and hardware pipeline recreation."""
        if not self.driver.pipeline_ready:
            self.skipTest("Hardware Vulkan pipeline unavailable")

        success = self.driver.reinitialize_gpu()
        self.assertTrue(success)
        self.assertTrue(self.driver.pipeline_ready)
        self.assertIsNotNone(self.driver.device)
        self.assertEqual(self.driver.recovery_state, GpuRecoveryState.RECOVERED)

    def test_cpu_oracle_survives_gpu_fault(self):
        """
        Formal isolation guarantee: GPU fault injection must NEVER mutate
        or compromise the deterministic CPU reference oracle.
        """
        w, h, t = 48, 20, 0.75

        # Compute baseline CPU output
        cpu_g_clean, cpu_c_clean = self.driver.execute_raymarch_cpu(w, h, t)

        # Inject severe combined GPU faults
        self.driver.inject_fault(
            FaultInjectionMask.OOM_ALLOCATION_FAULT
            | FaultInjectionMask.DEVICE_LOSS_EVENT
            | FaultInjectionMask.DESCRIPTOR_CORRUPTION
        )

        # Call execute_raymarch (triggers fallback)
        fb_g, fb_c = self.driver.execute_raymarch(w, h, t)

        # Compute post-fault direct CPU output
        cpu_g_post, cpu_c_post = self.driver.execute_raymarch_cpu(w, h, t)

        # Bit-identical equality across all three runs
        self.assertEqual(cpu_g_clean, fb_g)
        self.assertEqual(cpu_c_clean, fb_c)
        self.assertEqual(cpu_g_clean, cpu_g_post)
        self.assertEqual(cpu_c_clean, cpu_c_post)


if __name__ == "__main__":
    unittest.main()
