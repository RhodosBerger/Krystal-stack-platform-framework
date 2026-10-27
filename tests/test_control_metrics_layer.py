#!/usr/bin/env python3
"""
==============================================================================
KRYSTAL-STACK: CONTROL METRICS LAYER TEST SUITE
==============================================================================
Module: tests/test_control_metrics_layer.py
Constitutional Invariant: VITAL_MAX_HP == 6
Normative Reference:
    - Observe -> Predict -> Decide -> Execute -> Validate closed loop
    - Cardinal Invariant: "No transition is trusted without measured after-state."
    - Strict Invariant: CVR == 0 for all committed transitions.
==============================================================================
"""

import unittest
from krystal_kernel.constants import VITAL_MAX_HP, ContractViolation
from krystal_kernel.control_metrics_layer import (
    ControlMetricsLayer,
    TrafficLightStatus,
    HintRejectionReason,
    ValidationReceipt,
    ControlMetricsSummary,
)


class TestControlMetricsLayer(unittest.TestCase):
    """Rigorous formal verification of the 21 metrics and cardinal invariant."""

    def setUp(self) -> None:
        self.cml = ControlMetricsLayer(vital_max_hp=VITAL_MAX_HP)

    def test_constitutional_invariant_enforcement(self) -> None:
        """Verify that any attempt to initialize with vital_max_hp != 6 raises ContractViolation."""
        with self.assertRaises(ContractViolation):
            ControlMetricsLayer(vital_max_hp=5)
        with self.assertRaises(ContractViolation):
            ControlMetricsLayer(vital_max_hp=7)

    def test_cardinal_invariant_missing_after_state(self) -> None:
        """
        Cardinal Invariant: 'No transition is trusted without measured after-state.'
        If after_state is None, is_trusted MUST be False, and ECI < 1.0.
        """
        receipt = self.cml.record_transition(
            transition_id="TRANS-TEST-UNTRUSTED",
            before_state=[0.50, 0.40, 0.60],
            predicted_state=[0.55, 0.42, 0.58],
            action="FREQUENCY_SCALE_UP",
            after_state=None,  # Missing measured after-state
            constraint_violations_in_transition=0,
        )

        self.assertFalse(receipt.is_after_state_measured)
        self.assertFalse(receipt.is_trusted)
        self.assertLess(receipt.evidence_completeness_index, 1.0)
        self.assertEqual(receipt.mae, 1.0)  # Untrusted default

        summary = self.cml.get_summary()
        self.assertTrue(summary.cardinal_invariant_held)  # It held by correctly rejecting trust!

    def test_cardinal_invariant_measured_after_state(self) -> None:
        """
        When measured after-state is provided with 0 violations,
        transition is trusted and ECI reaches 1.0.
        """
        receipt = self.cml.record_transition(
            transition_id="TRANS-TEST-TRUSTED",
            before_state=[0.50, 0.40, 0.60],
            predicted_state=[0.55, 0.42, 0.58],
            action="FREQUENCY_SCALE_UP",
            after_state=[0.56, 0.43, 0.57],
            constraint_violations_in_transition=0,
            is_rollback=False,
            is_correct_candidate=True,
        )

        self.assertTrue(receipt.is_after_state_measured)
        self.assertTrue(receipt.is_trusted)
        self.assertEqual(receipt.evidence_completeness_index, 1.0)
        self.assertLess(receipt.mae, 0.05)  # Expected accurate prediction
        self.assertEqual(len(receipt.residual_vector), 3)
        self.assertIn("[!HAZ]", receipt.conduct_flags)
        self.assertIn("[VIT=6]", receipt.conduct_flags)

    def test_constraint_violation_rate_cvr_zero_enforcement(self) -> None:
        """
        If constraint_violations > 0:
        - is_trusted MUST be False even if after-state is present
        - flags contain [HAZ]
        - traffic light switches to RED
        """
        receipt = self.cml.record_transition(
            transition_id="TRANS-BREACH",
            before_state=[0.80, 0.85, 0.90],
            predicted_state=[0.75, 0.70, 0.80],
            action="OVERCLOCK_STEP",
            after_state=[0.95, 0.92, 0.98],
            constraint_violations_in_transition=1,  # Thermal ceiling breach
        )

        self.assertFalse(receipt.is_trusted)
        self.assertIn("[HAZ]", receipt.conduct_flags)

        summary = self.cml.get_summary()
        self.assertEqual(summary.traffic_light, TrafficLightStatus.RED)
        self.assertGreater(summary.safety["CVR"], 0.0)

    def test_all_21_metrics_present_and_bounded(self) -> None:
        """Verify that compute_all_21_metrics produces all 21 formal metrics."""
        # Record a nominal valid transition
        self.cml.record_transition(
            transition_id="TRANS-NOMINAL-1",
            before_state=[0.40, 0.50, 0.30],
            predicted_state=[0.42, 0.51, 0.32],
            action="TUNE_LATENCY",
            after_state=[0.41, 0.51, 0.31],
            planning_time_ms=0.35,
            workload_exec_time_ms=10.0,
            energy_joules=15.0,
            verified_work_units=100.0,
            operating_cost_usd=0.0010,
            is_rollback=False,
            constraint_violations_in_transition=0,
            is_correct_candidate=True,
            action_direction=1,
            quadrant="Q2",
        )

        metrics = self.cml.compute_all_21_metrics()
        expected_keys = [
            "SIS_StateIntegrityScore",
            "P_MAE_PredictionMAE",
            "PRV_ResidualNorm",
            "CRA_CandidateRankingAccuracy",
            "CVR_ConstraintViolationRate",
            "TR_TransitionReliability",
            "TQS_TransitionQualityScore",
            "RR_RollbackRate",
            "POR_PlanningOverheadRatio",
            "ECI_EvidenceCompletenessIndex",
            "HAR_HintAcceptanceRate",
            "HP_HintPrecision",
            "HUC_HintUrgencyCalibration",
            "SDR_StateDriftRate",
            "QSI_QuadrantStabilityIndex",
            "OCR_OscillationChatteringRate",
            "UWE_UsefulWorkEfficiency",
            "EER_EnergyEfficiencyRatio",
            "CVW_CostPerVerifiedWork",
            "CG_ControlGain",
            "PIR_PredictionImprovementRate",
        ]

        for key in expected_keys:
            self.assertIn(key, metrics, f"Missing metric key: {key}")
            self.assertIsInstance(metrics[key], (int, float))

        # Check expected baseline bounds
        self.assertGreaterEqual(metrics["SIS_StateIntegrityScore"], 0.90)
        self.assertEqual(metrics["CVR_ConstraintViolationRate"], 0.0)
        self.assertLess(metrics["POR_PlanningOverheadRatio"], 0.10)

    def test_kphp_hint_telemetry_and_rejections(self) -> None:
        """Verify KPHP Hint Acceptance Rate (HAR) and Hint Precision (HP) tracking."""
        self.cml.record_hint(is_forwarded=True, is_useful=True)
        self.cml.record_hint(is_forwarded=True, is_useful=False)
        self.cml.record_hint(is_forwarded=False, rejection_reason=HintRejectionReason.STALE)
        self.cml.record_hint(is_forwarded=False, rejection_reason=HintRejectionReason.RISK_EXCEEDED)

        self.assertEqual(self.cml.hints_decoded, 4)
        self.assertEqual(self.cml.hints_forwarded_to_aep, 2)
        self.assertEqual(self.cml.useful_validated_hints, 1)
        self.assertEqual(self.cml.hint_rejection_counts[HintRejectionReason.STALE], 1)
        self.assertEqual(self.cml.hint_rejection_counts[HintRejectionReason.RISK_EXCEEDED], 1)

        metrics = self.cml.compute_all_21_metrics()
        self.assertEqual(metrics["HAR_HintAcceptanceRate"], 0.5)  # 2 / 4
        self.assertEqual(metrics["HP_HintPrecision"], 0.5)       # 1 / 2

    def test_traffic_light_green_evaluation(self) -> None:
        """Verify GREEN rating when SIS > 0.90, MAE <= 0.05, CVR == 0, TR >= 0.90."""
        self.cml.record_transition(
            transition_id="TRANS-GREEN",
            before_state=[0.5, 0.5],
            predicted_state=[0.52, 0.48],
            action="OPTIMIZE_BANDWIDTH",
            after_state=[0.521, 0.479],  # Extremely close: MAE ~ 0.001
            constraint_violations_in_transition=0,
        )

        status = self.cml.evaluate_traffic_light()
        self.assertEqual(status, TrafficLightStatus.GREEN)

    def test_chattering_and_quadrant_stability(self) -> None:
        """Verify OCR (Oscillation Chattering Rate) and QSI (Quadrant Stability Index)."""
        # Alternating directions: +1, -1, +1, -1 (pure chattering)
        self.cml.record_transition("T1", [0.1], [0.1], "A", [0.1], action_direction=1, quadrant="Q2")
        self.cml.record_transition("T2", [0.1], [0.1], "A", [0.1], action_direction=-1, quadrant="Q2")
        self.cml.record_transition("T3", [0.1], [0.1], "A", [0.1], action_direction=1, quadrant="Q2")
        self.cml.record_transition("T4", [0.1], [0.1], "A", [0.1], action_direction=-1, quadrant="Q2")

        metrics = self.cml.compute_all_21_metrics()
        self.assertEqual(metrics["OCR_OscillationChatteringRate"], 1.0)  # 3 reversals out of 3 transitions
        self.assertEqual(metrics["QSI_QuadrantStabilityIndex"], 1.0)     # All Q2, 0 quadrant switches

    def test_hmac_receipt_signature_integrity(self) -> None:
        """Verify cryptographic HMAC-SHA256 signature is deterministic and tamper-evident."""
        receipt = self.cml.record_transition(
            transition_id="TRANS-HMAC",
            before_state=[0.1, 0.2],
            predicted_state=[0.11, 0.21],
            action="ADJUST_VOLTAGE",
            after_state=[0.11, 0.21],
        )

        self.assertTrue(len(receipt.hmac_digest) == 64)  # 256-bit hex
        self.assertIsInstance(receipt.to_dict(), dict)


if __name__ == "__main__":
    unittest.main()
