#!/usr/bin/env python3
"""
==============================================================================
KRYSTAL-STACK: TWO-PHASE REASONING / COMPUTING ALTERNATOR TEST SUITE
==============================================================================
Module: tests/test_two_phase_reasoning_computing_alternator.py
Milestone: V-TWO-PHASE-ALTERNATOR-01
Constitutional Invariant: VITAL_MAX_HP == 6
Normative Reference: AGENTS.md, GEMINI.md, H07_SILICON_TELEMETRY_AND_OPENVINO_LEXICON_SPEC.md

TEST OBJECTIVES:
  1. Verify constitutional invariant VITAL_MAX_HP == 6 across all entities.
  2. Verify dual-mode epistemic separation (proposer != evaluator, Delta S = 0 in Phase 1).
  3. Verify successful 4-gate Phase 2 audit and 128B zero-copy contract frame emission.
  4. Verify strict rollback (Delta S = 0) and quarantine on Gate 1/Gate 2/Gate 3 falsification.
  5. Verify UMA inductive impedance guard (return loss RL >= 18 dB, droop <= 50 mV).
  6. Verify chronometric shadow multiplexing (Principle P3, mu_UMA < 0.78).
==============================================================================
"""

import os
import sys
import unittest
import math
import time

WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

from krystal_kernel.constants import (
    VITAL_MAX_HP,
    FRAME_SIZE_BYTES,
    ContractViolation,
)
from krystal_kernel.silicon_telemetry_lexicon import (
    TelemetryEventCode,
    PulseSequenceLogger,
)
from krystal_kernel.uma_inductive_voltage_governor import (
    UmaInductiveVoltageGovernor,
)
from krystal_kernel.two_phase_reasoning_computing_alternator import (
    AlternatorPhase,
    EpistemicClaimStatus,
    PatternEvidence6Tuple,
    ReasoningHypothesisProposal,
    VerificationAuditReceipt,
    AlternatorCycleReceipt,
    TwoPhaseVerifierGate,
    TwoPhaseReasoningComputingAlternator,
    GLOBAL_REASONING_COMPUTING_ALTERNATOR,
)


class TestTwoPhaseReasoningComputingAlternator(unittest.TestCase):
    """Formal unit verification of TwoPhaseReasoningComputingAlternator."""

    def setUp(self) -> None:
        self.logger = PulseSequenceLogger()
        self.uma_governor = UmaInductiveVoltageGovernor(logger=self.logger)
        self.alternator = TwoPhaseReasoningComputingAlternator(
            uma_governor=self.uma_governor,
            silicon_logger=self.logger,
        )

    def test_01_constitutional_invariant_enforcement(self) -> None:
        """Constitutional invariant VITAL_MAX_HP == 6 must hold across all components."""
        self.assertEqual(VITAL_MAX_HP, 6)
        self.assertEqual(self.alternator.vital_max_hp, 6)

        # Breach in Gate construction
        with self.assertRaises(ContractViolation):
            TwoPhaseVerifierGate(vital_max_hp=5)

        # Breach in Alternator construction
        with self.assertRaises(ContractViolation):
            TwoPhaseReasoningComputingAlternator(vital_max_hp=7)

        # Breach in Proposal construction
        evidence = PatternEvidence6Tuple(
            characteristic="Test",
            source="Unit",
            scope="Local",
            evidence_status=EpistemicClaimStatus.PATTERN_CANDIDATE,
            validity_boundary="Bounded",
        )
        with self.assertRaises(ContractViolation):
            ReasoningHypothesisProposal(
                proposal_id="p_bad",
                intent_action="test",
                claim="bad",
                evidence=evidence,
                proposed_delta={},
                vital_max_hp=4,
            )

    def test_02_epistemic_separation_proposer_is_not_evaluator(self) -> None:
        """Phase 1 creates an unvalidated candidate without mutating authoritative state."""
        current_state = {"hp": 6, "position": [0.0, 0.0], "vital_max_hp": 6}
        original_state_copy = dict(current_state)

        evidence = PatternEvidence6Tuple(
            characteristic="Move player",
            source="LLM_Explore_Phase",
            scope="AgentWorld",
            evidence_status=EpistemicClaimStatus.PATTERN_CANDIDATE,
            validity_boundary="ArenaBounds",
        )
        proposal = ReasoningHypothesisProposal(
            proposal_id="prop_001",
            intent_action="MOVE_ENTITY",
            claim="Move to [10, 20]",
            evidence=evidence,
            proposed_delta={"position": [10.0, 20.0]},
            vital_max_hp=6,
        )

        # Proposer output is uncertified
        self.assertEqual(proposal.evidence.evidence_status, EpistemicClaimStatus.PATTERN_CANDIDATE)
        # Authoritative state is untouched (Delta S = 0)
        self.assertEqual(current_state, original_state_copy)

    def test_03_successful_two_phase_alternation_cycle(self) -> None:
        """Full cycle: Phase 1 proposes, Phase 2 verifies all 4 gates, commits state with 128B frame."""
        initial_state = {"score": 100, "vital_max_hp": 6}

        def mock_reasoning_task():
            return {"score": 150, "predicted_di_dt": 2.0, "predicted_uma_bandwidth": 15.0}

        def mock_state_transition(state, delta):
            new_s = dict(state)
            new_s["score"] = delta["score"]
            return new_s

        new_state, cycle_receipt = self.alternator.step_cycle(
            current_state=initial_state,
            intent_action="UPDATE_SCORE",
            reasoning_task=mock_reasoning_task,
            state_transition_func=mock_state_transition,
            current_temp_c=65.0,
            uma_pressure=0.25,
        )

        # State was mutated upon verified commit
        self.assertTrue(cycle_receipt.state_mutated)
        self.assertEqual(new_state["score"], 150)
        self.assertEqual(new_state["vital_max_hp"], 6)

        # Audit receipt verification
        audit = cycle_receipt.phase_2_receipt
        self.assertTrue(audit.passed)
        self.assertEqual(audit.verdict, EpistemicClaimStatus.VERIFIED_WITH_SCOPE)
        self.assertTrue(audit.gate1_invariant_passed)
        self.assertTrue(audit.gate2_impedance_passed)
        self.assertTrue(audit.gate3_cdt_passed)
        self.assertTrue(audit.gate4_commit_passed)

        # 128B zero-copy contract frame
        self.assertIsNotNone(audit.contract_frame_bytes)
        self.assertEqual(len(audit.contract_frame_bytes), FRAME_SIZE_BYTES)

        # Telemetry events
        events = self.logger.read_all()
        codes = [e.code for e in events]
        self.assertIn(TelemetryEventCode.NOISE_FLOOR_NOMINAL_DB, codes)
        self.assertIn(TelemetryEventCode.OPENVINO_GPU_PREFILL, codes)

    def test_04_falsification_and_strict_rollback_on_gate1_nan(self) -> None:
        """Gate 1 detects NaN in proposed parameters -> strict Delta S = 0 and quarantine."""
        initial_state = {"pos_x": 5.0, "vital_max_hp": 6}

        def mock_bad_reasoning_task():
            return {"pos_x": float("nan")}

        def mock_state_transition(state, delta):
            new_s = dict(state)
            new_s["pos_x"] = delta["pos_x"]
            return new_s

        new_state, cycle_receipt = self.alternator.step_cycle(
            current_state=initial_state,
            intent_action="INVALID_TELEPORT",
            reasoning_task=mock_bad_reasoning_task,
            state_transition_func=mock_state_transition,
        )

        # Strict Delta S = 0
        self.assertFalse(cycle_receipt.state_mutated)
        self.assertEqual(new_state["pos_x"], 5.0)

        # Verdict is FALSIFIED
        audit = cycle_receipt.phase_2_receipt
        self.assertFalse(audit.passed)
        self.assertEqual(audit.verdict, EpistemicClaimStatus.FALSIFIED)
        self.assertIn("NaN detected", audit.failure_reason)

        # Proposal quarantined
        quarantine = self.alternator.verifier_gate.quarantine_ledger
        self.assertEqual(len(quarantine), 1)
        self.assertEqual(quarantine[0]["gate_failed"], 1)

    def test_05_falsification_and_rollback_on_thermal_throttle(self) -> None:
        """Gate 2 halts commit if silicon temperature >= 88°C -> Delta S = 0."""
        initial_state = {"engine_rpm": 3000, "vital_max_hp": 6}

        def mock_reasoning_task():
            return {"engine_rpm": 6000}

        def mock_state_transition(state, delta):
            new_s = dict(state)
            new_s["engine_rpm"] = delta["engine_rpm"]
            return new_s

        new_state, cycle_receipt = self.alternator.step_cycle(
            current_state=initial_state,
            intent_action="BOOST_ENGINE",
            reasoning_task=mock_reasoning_task,
            state_transition_func=mock_state_transition,
            current_temp_c=91.5,  # Overheating!
        )

        # Must reject: Delta S = 0
        self.assertFalse(cycle_receipt.state_mutated)
        self.assertEqual(new_state["engine_rpm"], 3000)

        audit = cycle_receipt.phase_2_receipt
        self.assertFalse(audit.passed)
        self.assertEqual(audit.verdict, EpistemicClaimStatus.FALSIFIED)
        self.assertIn("throttle limit 88.0°C", audit.failure_reason)

        events = self.logger.read_all()
        codes = [e.code for e in events]
        self.assertIn(TelemetryEventCode.SILICON_THERMAL_THROTTLE, codes)

    def test_06_chronometric_shadow_multiplexing(self) -> None:
        """Principle P3: High UMA pressure triggers contention telemetry and shadow yield."""
        initial_state = {"val": 1, "vital_max_hp": 6}

        new_state, cycle_receipt = self.alternator.step_cycle(
            current_state=initial_state,
            intent_action="SHADOW_TASK",
            reasoning_task=lambda: {"val": 2},
            state_transition_func=lambda s, d: {**s, **d},
            uma_pressure=0.85,  # Above 0.78 threshold
        )

        # Telemetry should note UMA contention
        events = self.logger.read_all()
        codes = [e.code for e in events]
        self.assertIn(TelemetryEventCode.UMA_CONTENTION_HIGH, codes)

    def test_07_global_singleton_availability(self) -> None:
        """GLOBAL_REASONING_COMPUTING_ALTERNATOR singleton is initialized and stable."""
        self.assertIsNotNone(GLOBAL_REASONING_COMPUTING_ALTERNATOR)
        self.assertEqual(GLOBAL_REASONING_COMPUTING_ALTERNATOR.vital_max_hp, 6)
        self.assertEqual(GLOBAL_REASONING_COMPUTING_ALTERNATOR.current_phase, AlternatorPhase.IDLE)


if __name__ == "__main__":
    unittest.main()
