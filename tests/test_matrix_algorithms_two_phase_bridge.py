#!/usr/bin/env python3
"""
==============================================================================
KRYSTAL-STACK: MATRIX ALGORITHMS TWO-PHASE BRIDGE TEST SUITE
==============================================================================
Module: tests/test_matrix_algorithms_two_phase_bridge.py
Milestone: V-MATRIX-ALTERNATOR-01
Constitutional Invariant: VITAL_MAX_HP == 6
Normative Reference: AGENTS.md, GEMINI.md, THEORY_OF_PHASE_LOCKED_EPISTEMIC_ACTION.md

TEST OBJECTIVES:
  1. Verify constitutional invariant VITAL_MAX_HP == 6 across all bridge structures.
  2. Verify Symplectic Phase Space integration cycle execution through the Alternator.
  3. Verify Golden Ratio D6 Harmonic Tensor synthesis cycle and 6-cycle symmetry.
  4. Verify Chained 6-Stage Convolutional Matrix Pipeline under Gate 1.5 Theory verification.
  5. Verify strict rollback (Delta S = 0) and contract frame packaging.
==============================================================================
"""

import os
import sys
import unittest

WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

from krystal_kernel.constants import VITAL_MAX_HP, ContractViolation, FRAME_SIZE_BYTES
from krystal_kernel.two_phase_reasoning_computing_alternator import (
    EpistemicClaimStatus,
)
from krystal_kernel.matrix_algorithms_two_phase_bridge import (
    MatrixExecutionReceipt,
    MatrixAlgorithmsTwoPhaseBridge,
    GLOBAL_MATRIX_ALGORITHMS_BRIDGE,
)


class TestMatrixAlgorithmsTwoPhaseBridge(unittest.TestCase):
    """Formal unit verification of MatrixAlgorithmsTwoPhaseBridge."""

    def setUp(self) -> None:
        self.bridge = MatrixAlgorithmsTwoPhaseBridge()

    def test_01_constitutional_invariant_enforcement(self) -> None:
        """Constitutional invariant VITAL_MAX_HP == 6 must hold across bridge components."""
        self.assertEqual(VITAL_MAX_HP, 6)
        self.assertEqual(self.bridge.vital_max_hp, 6)

        with self.assertRaises(ContractViolation):
            MatrixAlgorithmsTwoPhaseBridge(vital_max_hp=5)

    def test_02_symplectic_governor_cycle_under_alternator(self) -> None:
        """Symplectic leapfrog integration must execute through the two-phase alternator."""
        initial_state = {"q": 1.0, "p": 0.0, "hamiltonian": 0.5, "vital_max_hp": 6}

        new_state, receipt = self.bridge.execute_symplectic_cycle(
            current_state=initial_state,
            q=1.0,
            p=0.5,
            delta_t=0.01,
            omega=1.0,
        )

        # Invariant preserved
        self.assertEqual(new_state["vital_max_hp"], 6)
        self.assertTrue(receipt.cycle_receipt.state_mutated)

        # Audit receipt check
        audit = receipt.cycle_receipt.phase_2_receipt
        self.assertTrue(audit.passed)
        self.assertEqual(audit.verdict, EpistemicClaimStatus.VERIFIED_WITH_SCOPE)
        self.assertTrue(audit.gate1_invariant_passed)
        self.assertTrue(audit.gate2_impedance_passed)

        # 128B contract frame
        self.assertIsNotNone(receipt.contract_frame_bytes)
        self.assertEqual(len(receipt.contract_frame_bytes), FRAME_SIZE_BYTES)

        # Metrics
        self.assertIn("q", receipt.matrix_metrics)
        self.assertIn("p", receipt.matrix_metrics)
        self.assertIn("hamiltonian", receipt.matrix_metrics)

    def test_03_golden_ratio_d6_synthesis_under_alternator(self) -> None:
        """Golden ratio D6 harmonic tensor synthesis must conserve 6-cycle symmetry."""
        initial_state = {"points_count": 0, "vital_max_hp": 6}

        new_state, receipt = self.bridge.execute_golden_ratio_d6_synthesis(
            current_state=initial_state,
            frequency=2.0,
            amplitude=1.5,
        )

        self.assertEqual(new_state["vital_max_hp"], 6)
        self.assertEqual(new_state["d6_cycle_points_count"], 6)  # D6 = 6 harmonic points!
        self.assertTrue(receipt.cycle_receipt.state_mutated)

        # Audit
        self.assertTrue(receipt.cycle_receipt.phase_2_receipt.passed)
        self.assertAlmostEqual(new_state["golden_ratio_phi"], 1.6180339887, places=4)

    def test_04_chained_convolution_pipeline_under_alternator(self) -> None:
        """Chained 6-stage matrix pipeline must execute through the two-phase alternator."""
        initial_state = {"stages_completed": 0, "vital_max_hp": 6}

        new_state, receipt = self.bridge.execute_chained_convolution_pipeline(
            current_state=initial_state,
            initial_density=0.25,
        )

        self.assertEqual(new_state["vital_max_hp"], 6)
        self.assertEqual(new_state["stages_completed"], 6)  # All 6 stages completed
        self.assertTrue(receipt.cycle_receipt.state_mutated)

        # Verify theory dependency passed Gate 1.5
        audit = receipt.cycle_receipt.phase_2_receipt
        self.assertTrue(audit.passed)
        self.assertIn("THEORY-CHAINED-BOOLEAN-CONVOLUTION", audit.theory_dependencies)

    def test_05_global_singleton_availability(self) -> None:
        """GLOBAL_MATRIX_ALGORITHMS_BRIDGE is initialized and healthy."""
        self.assertIsNotNone(GLOBAL_MATRIX_ALGORITHMS_BRIDGE)
        self.assertEqual(GLOBAL_MATRIX_ALGORITHMS_BRIDGE.vital_max_hp, 6)


if __name__ == "__main__":
    unittest.main()
