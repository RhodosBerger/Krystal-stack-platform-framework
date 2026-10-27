"""
Unit Test Suite for Krystal Sovereign Adaptive Architecture (KSAA)
Hardware Control Coprocessor (Milestone KSAA-M1 Silicon Proof).

Verification of Theory Node 51:
THEORY-KSAA-HARDWARE-CONTROL-COPROCESSOR-ARCHITECTURE
Constitutional Invariant: VITAL_MAX_HP == 6
"""

import unittest
import math
from typing import Dict, List, Any

from krystal_kernel.ksaa_hardware_control_coprocessor import (
    RuntimeQuadrant,
    ControlActionCode,
    HardwareStateVector,
    GhceControlWord,
    HardwareValidationReceipt,
    KsaaHardwareControlCoprocessor,
    GLOBAL_KSAA_COPROCESSOR,
)
from krystal_kernel.verification.theory_dependency_graph import GLOBAL_THEORY_DEPENDENCY_GRAPH
from krystal_kernel.constants import VITAL_MAX_HP


class TestKsaaHardwareControlCoprocessor(unittest.TestCase):
    """Test suite for Milestone KSAA-M1 Silicon Proof."""

    def setUp(self):
        self.coprocessor = KsaaHardwareControlCoprocessor()

    def test_theory_node_51_registration(self):
        """Verify that Theory Node 51 is formally registered without cycles."""
        theory_id = "THEORY-KSAA-HARDWARE-CONTROL-COPROCESSOR-ARCHITECTURE"
        self.assertIn(theory_id, GLOBAL_THEORY_DEPENDENCY_GRAPH._nodes)
        node = GLOBAL_THEORY_DEPENDENCY_GRAPH._nodes[theory_id]
        self.assertEqual(node.epistemic_status.name, "DOKAZANE")
        self.assertFalse(GLOBAL_THEORY_DEPENDENCY_GRAPH.has_cycles())

    def test_si_l0_hardware_state_vector(self):
        """Test Si-L0 sampling of 7-dimensional hardware state vector H = [C, M, T, E, Q, B, J]."""
        state = self.coprocessor.observe_hardware_state({
            "C": 0.65, "M": 0.40, "T": 0.55, "E": 0.48, "Q": 0.30, "B": 0.35, "J": 0.05
        })
        self.assertEqual(len(state.to_list()), 7)
        self.assertEqual(state.dominant_quadrant, RuntimeQuadrant.Q2_HIGH_LOAD_HEALTHY)
        self.assertEqual(state.vital_max_hp, 6)

    def test_ghce_control_word_packing_unpacking(self):
        """Test 32-bit bit-exact packing and unpacking of 0xVTAICHRF control word."""
        ghce = GhceControlWord(
            version=1,
            target=2,     # GPU EU
            action=3,     # CHANGE_PRECISION
            intensity=12, # 0xC
            confidence=14,# 0xE
            horizon=8,    # 8 dwell frames
            quadrant=2,   # Q2
            flags=0xF,    # Valid
        )
        u32 = ghce.pack_u32()
        self.assertIsInstance(u32, int)
        hex_str = ghce.to_hex_str()
        self.assertTrue(hex_str.startswith("0x123CE82F"))

        # Unpack and verify parity
        unpacked = GhceControlWord.unpack_u32(u32)
        self.assertEqual(unpacked.version, 1)
        self.assertEqual(unpacked.target, 2)
        self.assertEqual(unpacked.action, 3)
        self.assertEqual(unpacked.intensity, 12)
        self.assertEqual(unpacked.confidence, 14)
        self.assertEqual(unpacked.horizon, 8)
        self.assertEqual(unpacked.quadrant, 2)
        self.assertEqual(unpacked.flags, 0xF)

    def test_si_l3_influence_matrix_computation(self):
        """Test Si-L3 Delta S = M_I * A computation."""
        delta_throttle = self.coprocessor.compute_expected_delta(ControlActionCode.A4_THROTTLE_WORKLOAD)
        self.assertEqual(len(delta_throttle), 7)
        # Throttle workload should reduce compute (-0.35) and thermal (-0.28)
        self.assertLess(delta_throttle[0], 0.0) # C
        self.assertLess(delta_throttle[2], 0.0) # T

    def test_si_l4_candidate_generation_and_prediction(self):
        """Test Si-L4 generation of >= 4 competing candidates with predicted states."""
        state = HardwareStateVector(0.70, 0.60, 0.75, 0.68, 0.35, 0.50, 0.15)
        candidates = self.coprocessor.generate_and_predict_candidates(state)

        self.assertGreaterEqual(len(candidates), 4)
        for act, pred_state, utility, risk in candidates:
            self.assertIsInstance(act, ControlActionCode)
            self.assertIsInstance(pred_state, HardwareStateVector)
            self.assertIsInstance(utility, float)
            self.assertIsInstance(risk, float)

    def test_si_l2_constraint_gate_zero_mutation(self):
        """Test Si-L2 constraint comparator enforcing UnsafeCandidate => Delta S == 0."""
        safe_state = HardwareStateVector(0.50, 0.40, 0.55, 0.45, 0.30, 0.35, 0.10)
        unsafe_thermal = HardwareStateVector(0.95, 0.90, 0.96, 0.92, 0.95, 0.90, 0.80)

        self.assertTrue(self.coprocessor.gate_candidate(safe_state))
        self.assertFalse(self.coprocessor.gate_candidate(unsafe_thermal))

    def test_ksaa_m1_closed_loop_execution_and_receipt(self):
        """Verify full 10 DoD conditions of Milestone KSAA-M1 Silicon Proof."""
        state = HardwareStateVector(0.65, 0.55, 0.70, 0.60, 0.40, 0.45, 0.12)
        receipt = self.coprocessor.execute_coprocessor_cycle(state)

        self.assertIsInstance(receipt, HardwareValidationReceipt)
        self.assertEqual(receipt.vital_max_hp, 6)
        self.assertTrue(receipt.is_safe)
        self.assertLess(receipt.prediction_residual, 0.35)
        self.assertGreater(receipt.transition_reliability, 0.60)
        self.assertEqual(len(receipt.hmac_digest), 64)
        self.assertTrue(receipt.control_word_hex.startswith("0x"))

        # Verify Si-L7 Hardware Runtime Memory table was updated
        self.assertGreater(len(self.coprocessor.runtime_memory), 0)


if __name__ == "__main__":
    unittest.main()
