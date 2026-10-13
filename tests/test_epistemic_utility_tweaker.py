#!/usr/bin/env python3
"""
==============================================================================
KRYSTAL-STACK: UNIT TEST SUITE FOR EPISTEMIC UTILITY TWEAKER & OPENVINO NPU
==============================================================================
File: tests/test_epistemic_utility_tweaker.py
Description:
    Validates:
      1. Multi-dimensional level scanning and derived thermal headroom.
      2. NPU OpenVINO candidate plan generation and utility ranking.
      3. Epistemic Dual-Gate authority vetoing during thermal stress.
      4. Cross-platform execution plan transitions (Windows, Linux, NPU offload).
      5. Constitutional invariant: assert VITAL_MAX_HP == 6.
==============================================================================
"""

import unittest

from krystal_kernel.scene_constructor_presets import VITAL_MAX_HP
from krystal_kernel.epistemic_utility_tweaker import (
    TweakerPlanId,
    ScannedSystemLevels,
    CandidateExecutionPlan,
    NpuOpenVinoPlanEvaluator,
    KrystalEpistemicUtilityTweaker,
    GLOBAL_UTILITY_TWEAKER
)
from krystal_kernel.verification.semantic_protocol import (
    AuthorityState,
    SafetyState,
    EpistemicState,
    to_wire_token
)


class TestEpistemicUtilityTweaker(unittest.TestCase):
    """Test suite for Epistemic Utility Tweaker and OpenVINO NPU Plan Optimizer."""

    def setUp(self):
        self.tweaker = KrystalEpistemicUtilityTweaker()

    def test_constitutional_invariant(self):
        """Invariant: VITAL_MAX_HP == 6."""
        self.assertEqual(VITAL_MAX_HP, 6)
        levels = self.tweaker.scan_system_levels()
        self.assertEqual(levels.vital_hp, 6)

    def test_scan_system_levels(self):
        """Scans multi-dimensional levels and validates derived metrics."""
        levels = self.tweaker.scan_system_levels()
        self.assertIsInstance(levels, ScannedSystemLevels)
        self.assertGreater(levels.temperature_c, 0.0)
        self.assertGreater(levels.tj_max_c, levels.temperature_c)
        # Derived thermal headroom check
        self.assertAlmostEqual(levels.thermal_headroom_c, levels.tj_max_c - levels.temperature_c, places=2)
        # Pressure score must be bounded in [0, 1]
        self.assertGreaterEqual(levels.pressure_score, 0.0)
        self.assertLessEqual(levels.pressure_score, 1.0)
        self.assertIn(levels.active_platform, ["Windows", "Linux", "POSIX"])
        self.assertEqual(levels.epistemic_status, to_wire_token(EpistemicState.EPI_VALIDATED))

    def test_candidate_plan_generation_and_ranking(self):
        """Generates 5 distinct plans evaluated by NPU OpenVINO and ranked by utility."""
        plans = self.tweaker.generate_candidate_plans()
        self.assertEqual(len(plans), 5)

        # Verify all 5 canonical plans are present
        plan_ids = {p.plan_id for p in plans}
        expected_ids = {
            TweakerPlanId.PLAN_BURST_PEAK_OVERDRIVE,
            TweakerPlanId.PLAN_SUSTAINED_EFFICIENCY,
            TweakerPlanId.PLAN_LOW_POWER_STEALTH,
            TweakerPlanId.PLAN_THERMAL_RESCUE_RECOVERY,
            TweakerPlanId.PLAN_SHADOW_COUNTERFACTUAL
        }
        self.assertEqual(plan_ids, expected_ids)

        # Verify descending order by utility score
        for i in range(len(plans) - 1):
            self.assertGreaterEqual(
                plans[i].utility_score,
                plans[i + 1].utility_score,
                f"Plans must be sorted by utility descending: {plans[i].utility_score} vs {plans[i+1].utility_score}"
            )

        # Shadow simulation plan must carry AUTH_SHADOW_ONLY
        shadow_plan = next(p for p in plans if p.plan_id == TweakerPlanId.PLAN_SHADOW_COUNTERFACTUAL)
        self.assertEqual(shadow_plan.authority_verdict, to_wire_token(AuthorityState.AUTH_SHADOW_ONLY))

    def test_dual_gate_veto_during_thermal_crisis(self):
        """Dual-Gate must veto PLAN_BURST_PEAK_OVERDRIVE when thermal levels are critical."""
        # Create an artificial thermal crisis scenario: Temp = 92°C (Headroom = 8°C)
        crisis_levels = ScannedSystemLevels(
            timestamp_ns=1000,
            temperature_c=92.0,
            tj_max_c=100.0,
            thermal_headroom_c=8.0,
            package_power_watts=48.0,
            pl1_clamp_watts=35.0,
            pl2_burst_watts=54.0,
            cpu_utilization_pct=98.0,
            gpu_utilization_pct=85.0,
            memory_pressure_pct=92.0,
            frequency_mhz=4200.0,
            vital_hp=6,
            pressure_score=0.95,
            active_platform="Windows",
            epistemic_status=to_wire_token(EpistemicState.EPI_VALIDATED)
        )

        crisis_plans = self.tweaker.generate_candidate_plans(crisis_levels)
        burst_plan = next(p for p in crisis_plans if p.plan_id == TweakerPlanId.PLAN_BURST_PEAK_OVERDRIVE)

        # Must be vetoed by SafetyGate (SAFE_BREACH -> AUTH_VETOED)
        self.assertEqual(burst_plan.safety_verdict, to_wire_token(SafetyState.SAFE_BREACH))
        self.assertEqual(burst_plan.authority_verdict, to_wire_token(AuthorityState.AUTH_VETOED))

    def test_apply_device_plan_success(self):
        """Applying an approved plan dispatches cross-platform directives."""
        report = self.tweaker.apply_device_plan(TweakerPlanId.PLAN_SUSTAINED_EFFICIENCY)
        self.assertTrue(report["success"])
        self.assertEqual(report["applied_plan"]["plan_id"], TweakerPlanId.PLAN_SUSTAINED_EFFICIENCY.value)
        self.assertEqual(report["vital_max_hp_rule"], 6)
        self.assertGreater(len(report["dispatched_actions"]), 0)
        self.assertEqual(self.tweaker.active_plan_id, TweakerPlanId.PLAN_SUSTAINED_EFFICIENCY)

    def test_apply_device_plan_vetoed_failure(self):
        """Attempting to apply an authority-vetoed plan must be rejected."""
        # Force a thermal crisis state in the tweaker
        self.tweaker.scan_system_levels = lambda: ScannedSystemLevels(
            timestamp_ns=1000,
            temperature_c=95.0,
            tj_max_c=100.0,
            thermal_headroom_c=5.0,
            package_power_watts=50.0,
            pl1_clamp_watts=35.0,
            pl2_burst_watts=54.0,
            cpu_utilization_pct=99.0,
            gpu_utilization_pct=90.0,
            memory_pressure_pct=95.0,
            frequency_mhz=4400.0,
            vital_hp=6,
            pressure_score=0.98,
            active_platform="Windows",
            epistemic_status=to_wire_token(EpistemicState.EPI_VALIDATED)
        )

        res = self.tweaker.apply_device_plan(TweakerPlanId.PLAN_BURST_PEAK_OVERDRIVE)
        self.assertFalse(res["success"])
        self.assertIn("rejected by authority gate", res["error"])

    def test_npu_evaluator_device_and_tensor_inference(self):
        """OpenVINO NPU evaluator correctly identifies target device and runs tensor math."""
        evaluator = NpuOpenVinoPlanEvaluator()
        self.assertIn(evaluator.device, ["NPU", "GPU", "CPU"])
        features = [0.7, 0.4, 0.5, 0.3, 0.5, 1.0, 0.6]
        d_perf, d_energy, r_therm, p_breach = evaluator.forward_tensor_inference(features, plan_bias=0.5)

        self.assertGreaterEqual(d_perf, 0.0)
        self.assertGreaterEqual(r_therm, 0.0)
        self.assertLessEqual(r_therm, 1.0)
        self.assertGreaterEqual(p_breach, 0.0)
        self.assertLessEqual(p_breach, 1.0)


if __name__ == "__main__":
    unittest.main()
