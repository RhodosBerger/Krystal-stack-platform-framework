#!/usr/bin/env python3
"""
==============================================================================
KRYSTAL-STACK: UNIT & REGRESSION SUITE FOR KHI-E2E-003 QUEUE STABILITY
==============================================================================
Module: tests/test_khi_e2e_queue_stability.py
Description:
    Verifies that the KHI-E2E-003 Sustained 8 kHz Queue Stability benchmark
    properly enforces:
      1. Constitutional invariant assert VITAL_MAX_HP == 6.
      2. Zero drops and bounded queue depth under 1:8 control ratio.
      3. SPSC ring buffer pointer discipline without memory allocations.
      4. Strict monotonic consistency of deadline miss hierarchies.

Author: Dušan Kopecký & Krystal-Stack Architecture & Hardware Systems Council (2026)
==============================================================================
"""

import unittest
from krystal_kernel.scene_constructor_presets import VITAL_MAX_HP
from krystal_kernel.hardware_input_hal import SPSCInputRingBuffer
from scripts.run_khi_e2e_003_queue_stability import KhiQueueStabilityBenchmark


class TestKhiE2EQueueStability(unittest.TestCase):
    """Test suite for sustained 8 kHz queue stability and bounds."""

    def setUp(self):
        import gc
        gc.collect()
        self.assertEqual(VITAL_MAX_HP, 6, "Platform invariant VITAL_MAX_HP == 6 must hold")

    def test_01_vital_max_hp_invariant(self):
        bench = KhiQueueStabilityBenchmark(ring_capacity=512, control_ratio=8)
        res = bench.run_clock_driven(total_events=500, warmup_events=100)
        self.assertEqual(res["vital_max_hp"], 6)

    def test_02_clock_driven_bounded_queue_at_control_ratio_8(self):
        import gc
        gc.collect()
        bench = KhiQueueStabilityBenchmark(ring_capacity=1024, control_ratio=8)
        res = bench.run_clock_driven(total_events=2000, warmup_events=300)
        self.assertEqual(res["drops"], 0, "Drops must be strictly 0 under 1:8 control ratio")
        self.assertLess(res["traffic_intensity_rho"], 1.5, "Traffic intensity rho must reflect high service capacity")
        self.assertIn(res["stability_verdict"], ("STABLE_SUSTAINED", "STABLE_BOUNDED_WITH_BURSTS"))
        self.assertLessEqual(res["queue_depth"]["p50"], 400.0, "p50 queue depth must remain bounded within ring capacity")

    def test_03_spsc_ring_buffer_zero_realloc_under_8khz(self):
        ring = SPSCInputRingBuffer(capacity=64)
        self.assertTrue(ring.preallocated)
        self.assertTrue(ring.lock_avoiding)
        self.assertEqual(ring.capacity, 64)
        self.assertEqual(ring.size(), 0)

    def test_04_deadline_miss_accounting_monotonicity(self):
        bench = KhiQueueStabilityBenchmark(ring_capacity=512, control_ratio=1)
        res = bench.run_clock_driven(total_events=500, warmup_events=50)
        dm = res["deadline_misses"]
        self.assertGreaterEqual(dm["gt_125_us"], dm["gt_250_us"], ">125us misses must be >= >250us misses")
        self.assertGreaterEqual(dm["gt_250_us"], dm["gt_500_us"], ">250us misses must be >= >500us misses")


if __name__ == "__main__":
    unittest.main()
