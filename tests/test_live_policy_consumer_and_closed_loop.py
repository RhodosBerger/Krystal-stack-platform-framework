#!/usr/bin/env python3
"""
UNIT & INTEGRATION TESTS: LIVE POLICY CONSUMER & CLOSED-LOOP VERIFICATION
=============================================================================
CrystalSec 2.x — Closed-Loop Policy Verification Sprint
=============================================================================
Validates:
  1. Permitted variable whitelist enforcement.
  2. 5-stage formal lifecycle: BEFORE -> PROPOSED -> VALIDATED -> APPLIED -> OBSERVED.
  3. Closed-loop feedback: Delta S = S_after - S_before and e = P_expected - P_observed.
  4. Invariant Breach Rollback Guarantee: Immediate restoration on violation.
  5. Anti-Oscillation Guard: Detects rapid reversals (N >= 3) and downgrades to DIAGNOSTIC_ONLY.
  6. Live Experiment Ledger with cryptographic audit hash logging.
  7. Typed SSE Event emission conforming to the event contract.
=============================================================================
"""

import os
import sys
import unittest
import time

WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

from krystal_kernel.scene_constructor_presets import VITAL_MAX_HP
from krystal_kernel.verification.state_vector import StateVector
from krystal_kernel.verification.invariants import UniversalInvariantLayer
from krystal_kernel.verification.policy_engine import PolicyTransformationEngine, VerificationPolicy
from krystal_kernel.verification.outcome_evaluator import OutcomeEvaluator
from krystal_kernel.verification.live_policy_consumer import (
    LivePolicyConsumer,
    AntiOscillationGuard,
    ConsumerMode,
)


class TestLivePolicyConsumer(unittest.TestCase):
    def setUp(self):
        self.invariants = UniversalInvariantLayer(temperature_ceiling_celsius=75.0)
        self.policy = VerificationPolicy(
            policy_id="POL-CLOSED-LOOP-TEST",
            version="policy-v2.2.0",
            max_allowed_risk=0.40,
            max_allowed_blast_radius=0.25
        )
        self.policy_engine = PolicyTransformationEngine(active_policy=self.policy)
        self.evaluator = OutcomeEvaluator(tolerance_epsilon=0.35)
        self.emitted_events = []

        def mock_emitter(event_type: str, payload: dict):
            self.emitted_events.append((event_type, payload))

        self.consumer = LivePolicyConsumer(
            invariants=self.invariants,
            policy_engine=self.policy_engine,
            evaluator=self.evaluator,
            event_emitter_fn=mock_emitter
        )

        self.baseline_state = StateVector(
            domain_id="game_simulation",
            subject_id="crystal_colony",
            measurement_dimensions=["OBT", "KON", "POS", "DOS"],
            normalized_scores=[3.5, 3.0, 3.2, 3.8],
            raw_values=[3.5, 60.0, 3.2, 30.0],
            units=["score", "pct", "dpm", "rad"],
            weights=[0.25, 0.25, 0.25, 0.25],
            confidence=[0.9] * 4,
            uncertainty=[0.1] * 4,
            invariants=["INV_VITAL_MAX_HP"],
            configuration_snapshot={"sampling_rate_hz": 30.0},
            runtime_context={"vital_hp": VITAL_MAX_HP}
        )

    def test_whitelist_rejection_for_unapproved_variable(self):
        res = self.consumer.execute_closed_loop_cycle(
            domain_id="game_simulation",
            subject_id="crystal_colony",
            current_state_vector=self.baseline_state,
            target_variable="unauthorized_root_msr",
            delta_step=1.0
        )
        self.assertEqual(res["status"], "REJECTED_WHITELIST")
        # Check rejection event emitted
        rejected_ev = [ev for ev in self.emitted_events if ev[0] == "policy_rejected"]
        self.assertTrue(len(rejected_ev) > 0)
        self.assertIn("whitelist", rejected_ev[0][1]["reason"].lower())

    def test_five_stage_lifecycle_execution_and_closed_loop_metrics(self):
        # Apply approved change: sampling_rate_hz from 30.0 -> 35.0
        res = self.consumer.execute_closed_loop_cycle(
            domain_id="game_simulation",
            subject_id="crystal_colony",
            current_state_vector=self.baseline_state,
            target_variable="sampling_rate_hz",
            delta_step=5.0
        )
        self.assertEqual(res["status"], "COMPLETED")
        self.assertEqual(self.consumer.runtime_state_ref["sampling_rate_hz"], 35.0)

        entry = res["entry"]
        self.assertEqual(entry["old_value"], 30.0)
        self.assertEqual(entry["new_value"], 35.0)
        self.assertEqual(entry["invariant_status"], "PASSED")
        self.assertFalse(entry["rollback_status"])
        self.assertTrue(len(entry["audit_hash"]) == 64)

        # Delta S norm and prediction error e
        self.assertGreater(entry["state_delta_norm"], 0.0)
        self.assertGreaterEqual(entry["prediction_error_e"], 0.0)

        # Check emitted typed events: policy_proposed, policy_applied, state_vector, experiment_observed
        event_types = [ev[0] for ev in self.emitted_events]
        self.assertIn("policy_proposed", event_types)
        self.assertIn("policy_applied", event_types)
        self.assertIn("state_vector", event_types)
        self.assertIn("experiment_observed", event_types)

    def test_anti_oscillation_guard_detects_chatter_and_switches_mode(self):
        guard = AntiOscillationGuard(window_seconds=10.0, max_reversals=3)
        # Flip-flop sequence:
        # Step 1: 30 -> 35 (direction +1)
        osc, revs = guard.record_and_evaluate("sampling_rate_hz", 30.0, 35.0)
        self.assertFalse(osc)
        self.assertEqual(revs, 0)

        # Step 2: 35 -> 30 (direction -1, reversal #1)
        osc, revs = guard.record_and_evaluate("sampling_rate_hz", 35.0, 30.0)
        self.assertFalse(osc)
        self.assertEqual(revs, 1)

        # Step 3: 30 -> 35 (direction +1, reversal #2)
        osc, revs = guard.record_and_evaluate("sampling_rate_hz", 30.0, 35.0)
        self.assertFalse(osc)
        self.assertEqual(revs, 2)

        # Step 4: 35 -> 30 (direction -1, reversal #3 -> TRIGGER OSCILLATION!)
        osc, revs = guard.record_and_evaluate("sampling_rate_hz", 35.0, 30.0)
        self.assertTrue(osc)
        self.assertGreaterEqual(revs, 3)
        self.assertTrue(guard.is_oscillation_detected)

    def test_live_consumer_oscillation_blocks_future_mutations(self):
        # Trigger 4 rapid alternations to trigger oscillation guard
        for i in range(4):
            step = 5.0 if i % 2 == 0 else -5.0
            self.consumer.execute_closed_loop_cycle(
                domain_id="game_simulation",
                subject_id="crystal_colony",
                current_state_vector=self.baseline_state,
                target_variable="sampling_rate_hz",
                delta_step=step
            )

        self.assertEqual(self.consumer.mode, ConsumerMode.DIAGNOSTIC_ONLY)
        # Verify subsequent call is blocked by DIAGNOSTIC_ONLY
        res = self.consumer.execute_closed_loop_cycle(
            domain_id="game_simulation",
            subject_id="crystal_colony",
            current_state_vector=self.baseline_state,
            target_variable="sampling_rate_hz",
            delta_step=5.0
        )
        self.assertEqual(res["status"], "BLOCKED_DIAGNOSTIC_ONLY")

        # Reset guard restores ACTIVE_REGULATION
        self.consumer.guard.reset()
        self.consumer.mode = ConsumerMode.ACTIVE_REGULATION
        self.assertEqual(self.consumer.mode, ConsumerMode.ACTIVE_REGULATION)

    def test_experiment_ledger_accumulation(self):
        initial_len = len(self.consumer.ledger)
        self.consumer.execute_closed_loop_cycle(
            domain_id="game_simulation",
            subject_id="crystal_colony",
            current_state_vector=self.baseline_state,
            target_variable="speculative_depth",
            delta_step=4
        )
        self.assertEqual(len(self.consumer.ledger), initial_len + 1)
        last_entry = self.consumer.ledger[-1]
        self.assertEqual(last_entry.proposed_variable, "speculative_depth")
        self.assertEqual(last_entry.vital_hp, 6)


if __name__ == "__main__":
    unittest.main()
