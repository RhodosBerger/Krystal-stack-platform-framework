#!/usr/bin/env python3
"""
==============================================================================
KRYSTAL-STACK: KHI-E2E-001 BENCHMARK VERTICAL PROOF INTEGRATION TEST
==============================================================================
Module: tests/test_khi_e2e_benchmark.py
Description:
    Formal integration test verifying the KHI-E2E-001 vertical proof scenario:
    8 kHz mouse report -> HAL timestamp -> SPSC ring -> normalization ->
    CalibrationProfile -> InputEvidenceEnvelope -> OBS_INPUT_ACTIVITY ->
    TypedDependencyRuntime -> Governor request -> EvidenceGate ->
    SafetyGate -> ExecutionEnvelope -> Planner -> trace + replay.

Platform Invariant: assert VITAL_MAX_HP == 6
Author: Dušan Kopecký & Krystal-Stack Architecture & Hardware Systems Council (2026)
==============================================================================
"""

import os
import sys
import unittest

WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

from krystal_kernel.scene_constructor_presets import VITAL_MAX_HP
from krystal_kernel.hardware_input_driver import HardwareInputDriver
from krystal_kernel.hardware_input_toolkit import HardwareInputToolkit


class TestKhiE2EBenchmark(unittest.TestCase):
    """End-to-End Vertical Proof Test Suite for KHI-E2E-001."""

    def setUp(self):
        self.assertEqual(VITAL_MAX_HP, 6)
        self.driver = HardwareInputDriver(mock_mode=True)
        self.toolkit = HardwareInputToolkit(driver=self.driver)

    def test_khi_e2e_001_vertical_benchmark_pipeline(self):
        """
        Executes KHI-E2E-001 and validates stage timings, metrics, and governor authority.
        """
        res = self.toolkit.run_khi_e2e_benchmark(sample_count=50)

        self.assertEqual(res["benchmark_id"], "KHI-E2E-001")
        self.assertEqual(res["sample_count"], 50)
        self.assertEqual(res["vital_max_hp"], 6)

        # Validate all 7 vertical stages are measured
        expected_stages = [
            "T_device_report",
            "T_driver_delivery",
            "T_ring",
            "T_normalize",
            "T_evidence",
            "T_governor",
            "T_planner"
        ]
        stages = res["stages"]
        for st in expected_stages:
            self.assertIn(st, stages)
            s_dict = stages[st]
            self.assertIn("p50_us", s_dict)
            self.assertIn("p95_us", s_dict)
            self.assertIn("p99_us", s_dict)
            self.assertIn("max_us", s_dict)
            self.assertIn("min_us", s_dict)
            self.assertGreater(s_dict["p50_us"], 0.0)
            self.assertGreaterEqual(s_dict["p95_us"], s_dict["p50_us"])
            self.assertGreaterEqual(s_dict["max_us"], s_dict["p99_us"])

        # Validate Authority and Planning decisions
        self.assertEqual(res["last_committable_profile"], "performance_gpu")
        self.assertEqual(res["last_authority_verdict"], "COMMITTED_NOMINAL")
        self.assertEqual(res["planned_tasks_count"], 1)
        self.assertEqual(res["drops"], 0)


if __name__ == "__main__":
    unittest.main()
