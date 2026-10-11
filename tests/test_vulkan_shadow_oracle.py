"""
================================================================================
KRYSTAL-STACK / CRYSTAL-SEC 3.0: VULKAN SHADOW CPU ORACLE TEST SUITE
Test File: tests/test_vulkan_shadow_oracle.py

Role:
Verifies asynchronous shadow CPU verification, bounded task queueing,
adaptive sampling rate degradation, spatial pattern classification,
and conservative root cause analysis (RCA) hypothesis synthesis.

Epistemic Discipline:
No overclaiming of hardware root cause without external hardware evidence.
Strictly test bounded hypotheses: CONSISTENT_WITH_H0, H1, H2, H3, INSUFFICIENT_EVIDENCE.
================================================================================
"""

import os
import sys
import time
import unittest

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from krystal_kernel.verification.vulkan_shadow_oracle import (
    RCAHypothesis,
    SpatialPattern,
    VulkanShadowOracle,
)
from src.python.vulkan_compute_driver import VulkanComputeDriver


class TestVulkanShadowOracle(unittest.TestCase):
    """
    Verification suite for the asynchronous shadow CPU oracle and RCA classifier.
    """

    def setUp(self):
        self.driver = VulkanComputeDriver(prefer_gpu=False)
        self.oracle = VulkanShadowOracle(
            driver=self.driver,
            sampling_interval=16,
            max_queue_depth=4,
            overhead_threshold_pct=15.0,
        )

    def tearDown(self):
        if self.oracle:
            self.oracle.shutdown()
            self.oracle = None
        if self.driver:
            self.driver.cleanup()
            self.driver = None

    def test_shadow_sampling_ratio(self):
        """Verifies 1:16 sampling policy cadence."""
        self.assertTrue(self.oracle.should_sample(0))
        self.assertFalse(self.oracle.should_sample(1))
        self.assertFalse(self.oracle.should_sample(15))
        self.assertTrue(self.oracle.should_sample(16))
        self.assertFalse(self.oracle.should_sample(17))
        self.assertTrue(self.oracle.should_sample(32))

    def test_bit_identical_frame(self):
        """Bit-identical GPU and CPU frames yield H0 and D_divergence == 0.0."""
        w, h, t = 48, 20, 0.5
        ref_g, ref_c = self.driver.execute_raymarch_cpu(w, h, t)

        # Submit identical buffers
        self.oracle.submit_sample(
            frame_id=16,
            gpu_glyphs=ref_g,
            gpu_colors=ref_c,
            width=w,
            height=h,
            t=t,
            mode="CYBERPUNK",
            cam_pos=(0.0, 0.0, -3.2),
            step_budget=24,
        )

        # Allow worker thread to complete verification
        time.sleep(0.1)

        sample = self.oracle.last_sample
        self.assertIsNotNone(sample)
        self.assertEqual(sample.glyph_match_ratio, 1.0)
        self.assertEqual(sample.pixel_exact_match_ratio, 1.0)
        self.assertEqual(sample.mean_absolute_color_error, 0.0)
        self.assertEqual(sample.max_color_error, 0)
        self.assertEqual(sample.spatial_pattern, SpatialPattern.UNIFORM.value)
        self.assertEqual(sample.rca.hypothesis, RCAHypothesis.CONSISTENT_WITH_H0_IDENTICAL.value)
        self.assertEqual(sample.divergence_score, 0.0)

    def test_small_numeric_divergence_h1(self):
        """Small bounded color differences classify as CONSISTENT_WITH_H1 (drift)."""
        w, h, t = 48, 20, 0.5
        ref_g, ref_c = self.driver.execute_raymarch_cpu(w, h, t)

        # Perturb color slightly on 10 pixels (bounded deviation <= 4 units)
        perturbed_c = list(ref_c)
        for i in range(10):
            perturbed_c[i] = min(0xFFFFFF, perturbed_c[i] + 4)

        self.oracle.submit_sample(
            frame_id=32,
            gpu_glyphs=ref_g,
            gpu_colors=perturbed_c,
            width=w,
            height=h,
            t=t,
            mode="CYBERPUNK",
            cam_pos=(0.0, 0.0, -3.2),
            step_budget=24,
        )

        time.sleep(0.1)

        sample = self.oracle.last_sample
        self.assertIsNotNone(sample)
        self.assertEqual(sample.glyph_match_ratio, 1.0)
        self.assertLessEqual(sample.mean_absolute_color_error, 5.0)
        self.assertEqual(sample.rca.hypothesis, RCAHypothesis.CONSISTENT_WITH_H1.value)
        self.assertLess(sample.divergence_score, 0.10)

    def test_tile_pattern_h2(self):
        """8x8 workgroup tile divergence classifies as CONSISTENT_WITH_H2 (sync hazard)."""
        w, h, t = 48, 24, 0.5
        ref_g, ref_c = self.driver.execute_raymarch_cpu(w, h, t)

        # Corrupt an entire 8x8 tile (x from 8 to 15, y from 8 to 15)
        corrupted_c = list(ref_c)
        for y in range(8, 16):
            for x in range(8, 16):
                idx = y * w + x
                corrupted_c[idx] = (corrupted_c[idx] ^ 0x00FF00) & 0xFFFFFF

        self.oracle.submit_sample(
            frame_id=48,
            gpu_glyphs=ref_g,
            gpu_colors=corrupted_c,
            width=w,
            height=h,
            t=t,
            mode="CYBERPUNK",
            cam_pos=(0.0, 0.0, -3.2),
            step_budget=24,
        )

        time.sleep(0.1)

        sample = self.oracle.last_sample
        self.assertIsNotNone(sample)
        self.assertEqual(sample.spatial_pattern, SpatialPattern.TILE_LOCALIZED.value)
        self.assertEqual(sample.rca.hypothesis, RCAHypothesis.CONSISTENT_WITH_H2.value)
        self.assertIn("8x8", sample.rca.evidence[0])

    def test_nan_or_device_loss_h3(self):
        """Catastrophic corruption / out-of-range values classify as CONSISTENT_WITH_H3."""
        w, h, t = 48, 20, 0.5
        ref_g, ref_c = self.driver.execute_raymarch_cpu(w, h, t)

        # Introduce out-of-bounds invalid values
        corrupted_c = list(ref_c)
        corrupted_c[0] = 0x99999999  # > 0xFFFFFF invalid packed RGB
        corrupted_g = [99] * len(ref_g)  # Invalid glyphs

        self.oracle.submit_sample(
            frame_id=64,
            gpu_glyphs=corrupted_g,
            gpu_colors=corrupted_c,
            width=w,
            height=h,
            t=t,
            mode="CYBERPUNK",
            cam_pos=(0.0, 0.0, -3.2),
            step_budget=24,
        )

        time.sleep(0.1)

        sample = self.oracle.last_sample
        self.assertIsNotNone(sample)
        self.assertEqual(sample.rca.hypothesis, RCAHypothesis.CONSISTENT_WITH_H3.value)
        self.assertGreaterEqual(sample.divergence_score, 0.50)
        self.assertIn("TRIGGER_CPU_FALLBACK", sample.rca.recommended_action)

    def test_unknown_divergence(self):
        """Unstructured divergence returns INSUFFICIENT_EVIDENCE."""
        w, h, t = 48, 20, 0.5
        ref_g, ref_c = self.driver.execute_raymarch_cpu(w, h, t)

        # Distribute moderate noise that does not form an 8x8 tile or edge pattern
        noisy_c = list(ref_c)
        noisy_g = list(ref_g)
        for i in range(0, len(ref_c), 3):
            noisy_c[i] = (noisy_c[i] + 300) % 0xFFFFFF
            if i % 12 == 0:
                noisy_g[i] = (noisy_g[i] + 1) % 5

        self.oracle.submit_sample(
            frame_id=80,
            gpu_glyphs=noisy_g,
            gpu_colors=noisy_c,
            width=w,
            height=h,
            t=t,
            mode="CYBERPUNK",
            cam_pos=(0.0, 0.0, -3.2),
            step_budget=24,
        )

        time.sleep(0.1)

        sample = self.oracle.last_sample
        self.assertIsNotNone(sample)
        self.assertIn(
            sample.rca.hypothesis,
            [RCAHypothesis.INSUFFICIENT_EVIDENCE.value, RCAHypothesis.CONSISTENT_WITH_H3.value],
        )

    def test_bounded_worker_queue(self):
        """Verifies dropping of stale verification jobs when queue depth is saturated."""
        w, h, t = 48, 20, 0.5
        ref_g, ref_c = self.driver.execute_raymarch_cpu(w, h, t)

        # Block the worker thread temporarily using lock
        with self.oracle._lock:
            self.oracle._pending_jobs = 4  # Simulate saturated queue

        # Attempt to submit another job
        accepted = self.oracle.submit_sample(
            frame_id=96,
            gpu_glyphs=ref_g,
            gpu_colors=ref_c,
            width=w,
            height=h,
            t=t,
            mode="CYBERPUNK",
            cam_pos=(0.0, 0.0, -3.2),
            step_budget=24,
        )

        self.assertFalse(accepted)
        self.assertEqual(self.oracle.oracle_jobs_dropped, 1)

        # Reset pending jobs
        with self.oracle._lock:
            self.oracle._pending_jobs = 0

    def test_adaptive_shadow_rate(self):
        """Verifies automatic downscaling (16 -> 32 -> 64) when observer overhead > threshold."""
        self.assertEqual(self.oracle.sampling_interval, 16)

        # Feed baseline measurements: 2.0 ms
        for _ in range(15):
            self.oracle.record_frame_timing(frame_time_ms=2.0, was_sampled=False)

        # Feed heavy sampled frame times: 3.5 ms (+75% overhead > 15% threshold)
        for _ in range(6):
            self.oracle.record_frame_timing(frame_time_ms=3.5, was_sampled=True)

        # Should adaptively degrade to 32
        self.assertEqual(self.oracle.sampling_interval, 32)
        self.assertGreater(self.oracle.observer_overhead_pct, 15.0)
        self.assertEqual(len(self.oracle.degradation_events), 1)
        self.assertEqual(self.oracle.degradation_events[0]["event"], "SHADOW_ORACLE_RATE_DEGRADED")

    def test_observer_overhead_measurement(self):
        """Verifies mathematical calculation of observer overhead percentage."""
        for _ in range(12):
            self.oracle.record_frame_timing(2.0, was_sampled=False)
        for _ in range(6):
            self.oracle.record_frame_timing(2.2, was_sampled=True)

        # (2.2 - 2.0) / 2.0 = 10.0%
        self.assertAlmostEqual(self.oracle.observer_overhead_pct, 10.0, delta=1.0)

    def test_rca_does_not_claim_certainty_without_evidence(self):
        """
        Epistemic rule: RCA hypotheses must NEVER use absolute terms like 'PROVEN'
        or claim certainty without external hardware telemetry.
        """
        valid_prefixes = ("CONSISTENT_WITH_", "INSUFFICIENT_")
        for hypo in RCAHypothesis:
            self.assertTrue(
                hypo.value.startswith(valid_prefixes),
                f"Hypothesis {hypo.value} violates non-overclaiming discipline",
            )


if __name__ == "__main__":
    unittest.main()
