#!/usr/bin/env python3
"""
==============================================================================
KRYSTAL-STACK: EXPERIMENTAL THEORY DEPENDENCY TEST SUITE
==============================================================================
Module: tests/test_theory_dependency_graph.py
Milestone: V-THEORY-DEP-01
Constitutional Invariant: VITAL_MAX_HP == 6
Normative Reference: AGENTS.md, GEMINI.md, krystal_theory/REGISTER_TVRDENI.md

TEST OBJECTIVES:
  1. Verify constitutional invariant VITAL_MAX_HP == 6 across TheoryNode and Graph.
  2. Verify canonical bootstrap theory registration across categories.
  3. Verify topological cycle detection (TheoryDependencyCycleError).
  4. Verify dependency validation: rejection of ZAMIETNUTE (refuted) and STIAHNUTE (withdrawn).
  5. Verify end-to-end integration: TwoPhaseVerifierGate Gate 1.5 rejects refuted theory.
  6. Verify successful commit when valid theory chain is declared.
==============================================================================
"""

import os
import sys
import unittest
import time

WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

from krystal_kernel.constants import VITAL_MAX_HP, ContractViolation
from krystal_kernel.verification.theory_dependency_graph import (
    TheoryNode,
    TheoryCategory,
    TheoryEpistemicStatus,
    TheoryDependencyCycleError,
    ExperimentalTheoryDependencyGraph,
    GLOBAL_THEORY_DEPENDENCY_GRAPH,
)
from krystal_kernel.two_phase_reasoning_computing_alternator import (
    TwoPhaseReasoningComputingAlternator,
    EpistemicClaimStatus,
)


class TestTheoryDependencyGraph(unittest.TestCase):
    """Formal unit verification of ExperimentalTheoryDependencyGraph and Gate 1.5."""

    def setUp(self) -> None:
        self.graph = ExperimentalTheoryDependencyGraph()
        self.alternator = TwoPhaseReasoningComputingAlternator(theory_graph=self.graph)

    def test_01_constitutional_invariant_enforcement(self) -> None:
        """Constitutional invariant VITAL_MAX_HP == 6 must hold across TheoryNode and Graph."""
        self.assertEqual(VITAL_MAX_HP, 6)
        self.assertEqual(self.graph.vital_max_hp, 6)

        with self.assertRaises(ContractViolation):
            ExperimentalTheoryDependencyGraph(vital_max_hp=5)

        with self.assertRaises(ContractViolation):
            TheoryNode(
                theory_id="THEORY-BAD-HP",
                title="Bad HP Theory",
                category=TheoryCategory.EPISTEMIC_TRIAGE,
                formal_ref="test",
                falsification_condition="test",
                epistemic_status=TheoryEpistemicStatus.MODEL,
                vital_max_hp=4,
            )

    def test_02_canonical_bootstrap_theories_present(self) -> None:
        """Foundational theories from krystal_theory, PDN, and Alternator must be registered."""
        pdn = self.graph.get_theory("THEORY-H07-FARADAY-LENZ-PDN")
        self.assertIsNotNone(pdn)
        self.assertEqual(pdn.category, TheoryCategory.ELECTRICAL_PDN)
        self.assertEqual(pdn.epistemic_status, TheoryEpistemicStatus.NAMERANE)

        p3 = self.graph.get_theory("THEORY-P3-CHRONOMETRIC-SHADOW")
        self.assertIsNotNone(p3)
        self.assertEqual(p3.epistemic_status, TheoryEpistemicStatus.DOKAZANE)

        veta32 = self.graph.get_theory("THEORY-VETA-3.2-SCHMITT-HYSTERESIS")
        self.assertIsNotNone(veta32)
        self.assertEqual(veta32.category, TheoryCategory.SCHEDULING_WSPT)

    def test_03_topological_cycle_detection(self) -> None:
        """Graph must detect and raise TheoryDependencyCycleError on circular theory dependencies."""
        graph = ExperimentalTheoryDependencyGraph()

        node_a = TheoryNode(
            theory_id="THEORY-CYCLE-A",
            title="Cycle Node A",
            category=TheoryCategory.EPISTEMIC_TRIAGE,
            formal_ref="test",
            falsification_condition="test",
            epistemic_status=TheoryEpistemicStatus.MODEL,
            parent_theories=("THEORY-CYCLE-B",),
        )
        node_b = TheoryNode(
            theory_id="THEORY-CYCLE-B",
            title="Cycle Node B",
            category=TheoryCategory.EPISTEMIC_TRIAGE,
            formal_ref="test",
            falsification_condition="test",
            epistemic_status=TheoryEpistemicStatus.MODEL,
            parent_theories=("THEORY-CYCLE-A",),
        )

        graph.register_theory(node_a)
        with self.assertRaises(TheoryDependencyCycleError):
            graph.register_theory(node_b)

    def test_04_dependency_validation_rejects_withdrawn_or_refuted(self) -> None:
        """validate_dependency_chain must reject STIAHNUTE and ZAMIETNUTE theories."""
        # Valid chain
        valid_chain = ("THEORY-H07-FARADAY-LENZ-PDN", "THEORY-P3-CHRONOMETRIC-SHADOW")
        is_valid, reason = self.graph.validate_dependency_chain(valid_chain)
        self.assertTrue(is_valid)
        self.assertIsNone(reason)

        # STIAHNUTE chain (Tvrdenie B1)
        withdrawn_chain = ("THEORY-HISTORICAL-EMBEDDING-SAVING-B1",)
        is_valid, reason = self.graph.validate_dependency_chain(withdrawn_chain)
        self.assertFalse(is_valid)
        self.assertIn("WITHDRAWN/STIAHNUTE", reason)

        # Non-existent theory
        missing_chain = ("THEORY-NON-EXISTENT",)
        is_valid, reason = self.graph.validate_dependency_chain(missing_chain)
        self.assertFalse(is_valid)
        self.assertIn("not registered", reason)

    def test_05_gate_1_5_rejection_of_refuted_theory_in_alternator(self) -> None:
        """Alternator Gate 1.5 must reject proposal relying on a withdrawn theory with Delta S = 0."""
        initial_state = {"power_budget_w": 15.0, "vital_max_hp": 6}

        new_state, cycle_receipt = self.alternator.step_cycle(
            current_state=initial_state,
            intent_action="COMPRESS_EMBEDDINGS_HISTORICAL",
            reasoning_task=lambda: {"power_budget_w": 25.0},
            state_transition_func=lambda s, d: {**s, **d},
            theory_dependencies=("THEORY-HISTORICAL-EMBEDDING-SAVING-B1",),  # STIAHNUTE!
        )

        # Strict rollback Delta S = 0
        self.assertFalse(cycle_receipt.state_mutated)
        self.assertEqual(new_state["power_budget_w"], 15.0)

        # Falsified receipt at Gate 1.5
        audit = cycle_receipt.phase_2_receipt
        self.assertFalse(audit.passed)
        self.assertEqual(audit.verdict, EpistemicClaimStatus.FALSIFIED)
        self.assertIn("GATE 1.5 FAIL", audit.failure_reason)

        # Quarantine verification
        quarantine = self.alternator.verifier_gate.quarantine_ledger
        self.assertTrue(any("GATE 1.5 FAIL" in q["reason"] for q in quarantine))

    def test_06_successful_commit_with_valid_theory_dependencies(self) -> None:
        """Proposal declaring valid theoretical foundation passes Gate 1.5 and commits state."""
        initial_state = {"voltage_compensation": 0.0, "vital_max_hp": 6}

        new_state, cycle_receipt = self.alternator.step_cycle(
            current_state=initial_state,
            intent_action="ENGAGE_FARADAY_CHOKE",
            reasoning_task=lambda: {"voltage_compensation": 0.015},
            state_transition_func=lambda s, d: {**s, **d},
            theory_dependencies=(
                "THEORY-H07-FARADAY-LENZ-PDN",
                "THEORY-VETA-3.2-SCHMITT-HYSTERESIS",
            ),
        )

        self.assertTrue(cycle_receipt.state_mutated)
        self.assertEqual(new_state["voltage_compensation"], 0.015)
        self.assertEqual(new_state["vital_max_hp"], 6)

        audit = cycle_receipt.phase_2_receipt
        self.assertTrue(audit.passed)
        self.assertEqual(audit.verdict, EpistemicClaimStatus.VERIFIED_WITH_SCOPE)
        self.assertIn("THEORY-H07-FARADAY-LENZ-PDN", audit.theory_dependencies)


if __name__ == "__main__":
    unittest.main()
