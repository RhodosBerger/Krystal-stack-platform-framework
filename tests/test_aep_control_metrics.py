#!/usr/bin/env python3
"""
==============================================================================
KRYSTAL-STACK: AEP NINE-LAYER CONTROL METRICS (v2) TEST SUITE
==============================================================================
Module: tests/test_aep_control_metrics.py
Normative Reference:
    - docs/architecture/AEP_CANONICAL_SPEC.md (L0..L8)
    - Four-Axis Contract:
      ControlValidity x RuntimePosture x PredictionQuality x VerifiedUtility
      => VALIDATED_USEFUL_TRANSITION
Constitutional Invariant: VITAL_MAX_HP == 6
==============================================================================
"""

import unittest
from krystal_kernel.constants import VITAL_MAX_HP, ContractViolation
from krystal_kernel.aep_control_metrics import (
    AepControlMetricsEngine,
    ControlValidity,
    RuntimePosture,
    PredictionQualityGrade,
    UtilityAccretionStatus,
    TransitionCertification,
    ExecutionLane,
    AepLayerMetricsSnapshot,
    AepDashboardMinimum,
    AepFourAxisContract,
    AepReceiptValidationResult,
)


class TestAepControlMetricsV2(unittest.TestCase):
    """Rigorous formal verification of AEP 9-Layer Control Metrics v2."""

    def setUp(self) -> None:
        self.engine = AepControlMetricsEngine(vital_max_hp=VITAL_MAX_HP)

    def test_constitutional_invariant(self) -> None:
        """Verify initialization with vital_max_hp != 6 raises ContractViolation."""
        with self.assertRaises(ContractViolation):
            AepControlMetricsEngine(vital_max_hp=5)

    def test_l0_contract_integrity_rate(self) -> None:
        """Verify L0 frame admission, rejection accounting, and CIR."""
        for _ in range(8):
            self.engine.record_l0_frame(is_valid=True)
        self.engine.record_l0_frame(is_valid=False, is_checksum_failure=True)
        self.engine.record_l0_frame(is_valid=False, is_capability_rejection=True)

        metrics = self.engine.compute_all_layer_metrics()
        self.assertEqual(metrics.cir_contract_integrity_rate, 0.8)
        self.assertEqual(metrics.checksum_failures, 1)
        self.assertEqual(metrics.capability_rejections, 1)

    def test_correction_1_decouple_invalid_from_unhealthy(self) -> None:
        """
        CORRECTION 1: Decouple formal INVALID from physical RED.
        A negative pressure margin (overload) must set RuntimePosture = RED,
        but ControlValidity MUST remain VALID (physical strain is not a constitutional breach).
        """
        # Inject physical thermal overload (observed 95 C > limit 90 C)
        self.engine.update_pressure_observations({"thermal_celsius": 95.0})

        contract = self.engine.evaluate_four_axis_contract()
        self.assertEqual(contract.control_validity, ControlValidity.VALID)  # Formally clean!
        self.assertEqual(contract.runtime_posture, RuntimePosture.RED)     # Physically overloaded!
        self.assertEqual(contract.certification, TransitionCertification.PHYSICAL_ENVELOPE_BREACH)

    def test_correction_2_dimensionally_normalized_margins(self) -> None:
        """
        CORRECTION 2: Dimensionally normalized pressure margins.
        PM_i^norm = (Limit_i - Observed_i) / SafetyRange_i.
        All margins are dimensionless ratios in comparable units.
        """
        # Nominal observation: memory = 24.5 GB/s (limit 68, range 50) => (68-24.5)/50 = 0.87
        margins = self.engine.compute_normalized_pressure_margins()
        self.assertIn("memory_gbps", margins)
        self.assertIn("thermal_celsius", margins)
        self.assertAlmostEqual(margins["memory_gbps"], (68.0 - 24.5) / 50.0, places=3)
        self.assertAlmostEqual(margins["thermal_celsius"], (90.0 - 68.0) / 45.0, places=3)

        metrics = self.engine.compute_all_layer_metrics()
        self.assertGreater(metrics.minimum_normalized_pressure_margin, 0.20)

    def test_correction_3_dimension_specific_tolerance_tr(self) -> None:
        """
        CORRECTION 3: Dimension-specific tolerance for Transition Reliability.
        NR_i = |Actual_i - Predicted_i| / Tolerance_i
        TR = 1 - mean(min(1.0, NR_i)).
        """
        # Dim 0 (Core ratio): diff 0.02 / tol 0.05 => NR = 0.40
        # Dim 1 (Thermal): diff 1.0 C / tol 2.0 C => NR = 0.50
        # Dim 2 (Bus): diff 0.01 / tol 0.04 => NR = 0.25
        pred = [0.50, 65.0, 0.30]
        actual = [0.52, 66.0, 0.31]

        tr = self.engine.compute_dimension_tolerant_tr(pred, actual)
        expected_nr_mean = (0.40 + 0.50 + 0.25) / 3.0
        self.assertAlmostEqual(tr, 1.0 - expected_nr_mean, places=3)
        self.assertGreater(tr, 0.60)

    def test_correction_4_windowed_calibration_trend_ema(self) -> None:
        """
        CORRECTION 4: Windowed calibration trend with EMA.
        CalibrationTrend_N = MAE_baseline - EMA(MAE_N).
        Single receipt CG > 0 does NOT prove learning; windowed trend does.
        """
        # Historical baseline: 0.080, current EMA: ~0.038
        trend, is_confirmed = self.engine.compute_windowed_calibration_trend()
        self.assertGreater(trend, 0.030) # Baseline 0.080 - EMA 0.038 = ~0.042
        self.assertTrue(is_confirmed)

    def test_master_kpi_validated_useful_transition(self) -> None:
        """
        Verify that when all 4 axes are satisfied:
          ControlValidity == VALID
          RuntimePosture in (GREEN, AMBER)
          PredictionGrade != DEGRADED
          UtilityStatus == VALUE_ACCRETIVE
        The transition receives the master certification: VALIDATED_USEFUL_TRANSITION.
        """
        contract = self.engine.evaluate_four_axis_contract()
        self.assertEqual(contract.control_validity, ControlValidity.VALID)
        self.assertIn(contract.runtime_posture, (RuntimePosture.GREEN, RuntimePosture.AMBER))
        self.assertEqual(contract.prediction_grade, PredictionQualityGrade.HIGH_CONFIDENCE)
        self.assertEqual(contract.utility_status, UtilityAccretionStatus.VALUE_ACCRETIVE)
        self.assertEqual(contract.certification, TransitionCertification.VALIDATED_USEFUL_TRANSITION)

    def test_constitutional_breach_forces_blocked_invalid(self) -> None:
        """
        CONSTITUTIONAL MANDATE:
        If a tenant isolation breach (cross-runtime) occurs,
        ControlValidity immediately collapses to INVALID and transition is BLOCKED_INVALID.
        """
        self.engine.record_l1_operation("session-leak", is_cross_runtime_access=True)

        contract = self.engine.evaluate_four_axis_contract()
        self.assertEqual(contract.control_validity, ControlValidity.INVALID)
        self.assertEqual(contract.certification, TransitionCertification.BLOCKED_INVALID)
        self.assertIn("[HAZ]", contract.conduct_flags)
        self.assertIn("[VIT=BREACH]", contract.conduct_flags)

    def test_l3_observation_purity_delta_s_zero(self) -> None:
        """Verify L3 observation causes zero state mutation: Delta S_obs == 0 => OPS == 1.0."""
        self.engine.record_observation_cycle(telemetry_freshness=0.99, mutations_caused=0)
        metrics = self.engine.compute_all_layer_metrics()
        self.assertEqual(metrics.ops_observation_purity_score, 1.0)
        self.assertGreaterEqual(metrics.telemetry_freshness, 0.90)

    def test_l8_commit_integrity_contracts(self) -> None:
        """Verify L8 candidate transactional machine contracts."""
        self.engine.record_candidate_transition(
            candidate_passed_invariants=True,
            candidate_committed=True,
            dirty_mutation=False,
            invalid_commit=False,
            receipt_emitted=True,
        )

        metrics = self.engine.compute_all_layer_metrics()
        self.assertEqual(metrics.dirty_mutation_rate, 0.0)
        self.assertEqual(metrics.invalid_commit_rate, 0.0)
        self.assertEqual(metrics.receipt_coverage, 1.0)

    def test_aep_receipt_quality_audit(self) -> None:
        """Verify audit checklist on canonical AepReceipt JSON."""
        valid_receipt = {
            "$schema": "https://krystal.stack/schemas/aep-receipt-v1.json",
            "operation_id": "01JAX7K89M4V7N01B000000001",
            "runtime_id": "session-match-1876-0042",
            "workload": "GAME",
            "action": "DUEL_ROUND",
            "state_before": "sha256:7f83b1657ff1fc53b92dc18148a1d65dfc2d4b1fa3d677284addd200126d9069",
            "state_after": "sha256:3b9ac411c8d6263e8b090a9ec03f9dc242c75a918a3ec37d13b4c100142857e1",
            "policy": {"id": "interactive-low-latency", "revision": 2},
            "execution": {"lane": "REALTIME", "backend": "c11", "device": "CPU"},
            "telemetry": {"queue_depth": 1, "execution_us": 624, "bus_pressure": 0.12},
            "verdict": "COMMITTED"
        }

        audit = self.engine.verify_aep_receipt(valid_receipt)
        self.assertEqual(audit.receipt_completeness, 1.0)
        self.assertTrue(audit.receipt_hash_validity)
        self.assertTrue(audit.policy_revision_known)
        self.assertTrue(audit.telemetry_attached)
        self.assertTrue(audit.verdict_consistent)


if __name__ == "__main__":
    unittest.main()
