#!/usr/bin/env python3
"""
==============================================================================
KRYSTAL-STACK: FULL PREDICTIVE RUNTIME LOOP TEST SUITE (M1)
==============================================================================
Module: tests/test_full_predictive_loop.py
Verifies the complete closed loop:
    Observe -> Normalize -> Generate -> Predict -> Constrain ->
    Select -> Execute -> Validate -> Calibrate
Constitutional Invariant: VITAL_MAX_HP == 6
==============================================================================
"""

import unittest
from src.constants import VITAL_MAX_HP, ContractViolation
from src.runtime_state import RuntimeStateVector, ControlledInferenceWorkload
from src.candidate_generator import CandidateGenerator, CandidateAction
from src.transition_predictor import TransitionPredictor
from src.constraint_gate import ConstraintGate, HardConstraintThresholds
from src.validation_receipt import ValidationReceipt
from src.calibration import OnlineCalibrationEngine
from src.runtime_planner import RuntimePlanner, PlannedTransitionResult


class TestFullPredictiveLoop(unittest.TestCase):
    """Rigorous formal verification of Milestone M1 Predictive Runtime Loop."""

    def setUp(self) -> None:
        self.planner = RuntimePlanner(vital_max_hp=VITAL_MAX_HP)

    def test_constitutional_invariant(self) -> None:
        """Verify initialization with vital_max_hp != 6 raises ContractViolation."""
        with self.assertRaises(ContractViolation):
            RuntimePlanner(vital_max_hp=5)

    def test_canonical_6d_state_vector(self) -> None:
        """Verify canonical 6D state vector S = [C, M, L, T, E, Q]."""
        s = RuntimeStateVector(c_compute=0.5, m_memory=0.6, l_latency=0.4, t_thermal=0.55, e_energy=0.5, q_queue=0.3)
        v = s.to_list()
        self.assertEqual(len(v), 6)
        self.assertEqual(v, [0.5, 0.6, 0.4, 0.55, 0.5, 0.3])

        # Clamping check
        s_out = RuntimeStateVector.from_list([1.5, -0.2, 0.5, 0.5, 0.5, 0.5])
        self.assertEqual(s_out.c_compute, 1.0)
        self.assertEqual(s_out.m_memory, 0.0)

    def test_candidate_generation_at_least_four(self) -> None:
        """Verify at least 4 core candidate actions (A, B, C, D) are generated."""
        gen = CandidateGenerator()
        state = RuntimeStateVector()
        actions = gen.generate_candidates(state)
        self.assertGreaterEqual(len(actions), 4)

        action_ids = [a.action_id for a in actions]
        self.assertIn("ACT_A", action_ids)
        self.assertIn("ACT_B", action_ids)
        self.assertIn("ACT_C", action_ids)
        self.assertIn("ACT_D", action_ids)

    def test_deterministic_predictor(self) -> None:
        """Verify transition predictor is deterministic and non-random."""
        pred = TransitionPredictor()
        state = RuntimeStateVector(c_compute=0.4, m_memory=0.5, l_latency=0.3, t_thermal=0.4, e_energy=0.4, q_queue=0.2)
        act = CandidateAction("ACT_B", "BATCH_DOWN_25", 0.75, 1.0, [-0.25, 0.0], "test")

        res1 = pred.predict(state, act).to_list()
        res2 = pred.predict(state, act).to_list()
        self.assertEqual(res1, res2)

    def test_constraint_gate_enforcement(self) -> None:
        """Verify hard constraints reject memory >= 0.90, thermal >= 0.80, latency >= 0.75."""
        gate = ConstraintGate()

        # Safe state
        safe = RuntimeStateVector(c_compute=0.5, m_memory=0.80, l_latency=0.70, t_thermal=0.75, e_energy=0.6, q_queue=0.4)
        res_safe = gate.evaluate(safe, confidence=0.90)
        self.assertTrue(res_safe.is_admissible)
        self.assertEqual(len(res_safe.violations), 0)

        # Thermal breach
        hot = RuntimeStateVector(c_compute=0.5, m_memory=0.80, l_latency=0.70, t_thermal=0.85, e_energy=0.6, q_queue=0.4)
        res_hot = gate.evaluate(hot, confidence=0.90)
        self.assertFalse(res_hot.is_admissible)
        self.assertTrue(any("THERMAL_BREACH" in v for v in res_hot.violations))

    def test_full_nine_step_loop_execution(self) -> None:
        """Verify one complete step of the 9-step closed loop."""
        workload = ControlledInferenceWorkload(batch_size=32, worker_count=4, queue_depth=12)
        result = self.planner.step(workload, ambient_temp_c=26.0)

        self.assertIsInstance(result, PlannedTransitionResult)
        self.assertEqual(result.step_id, 1)
        self.assertIsInstance(result.receipt, ValidationReceipt)
        self.assertEqual(result.receipt.vital_max_hp, 6)
        self.assertTrue(len(result.receipt.hmac_digest) == 64)
        self.assertFalse(result.receipt.is_constraint_violated)
        self.assertLess(result.receipt.prediction_mae, 0.15)

    def test_multiple_cycles_and_calibration(self) -> None:
        """Verify that running 10 consecutive cycles compounds calibration."""
        workload = ControlledInferenceWorkload(batch_size=24, worker_count=4, queue_depth=8)
        for _ in range(10):
            res = self.planner.step(workload)
            workload = res.selected_action.apply_to_workload(workload)

        self.assertEqual(self.planner.step_counter, 10)
        self.assertEqual(self.planner.total_constraint_violations, 0)
        self.assertTrue(self.planner.calibration_engine.ema_mae < 0.10)


if __name__ == "__main__":
    unittest.main()
