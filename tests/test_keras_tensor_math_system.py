#!/usr/bin/env python3
"""
==============================================================================
UNIT TEST SUITE: KERAS TENSOR MATH & MULTI-DOMAIN OBSERVATION MATRIX ENGINE
==============================================================================
Test Target: krystal_kernel/keras_tensor_math_engine.py
Framework: unittest
Invariants: VITAL_MAX_HP == 6 strictly enforced across all states and outputs.
==============================================================================
"""

import os
import sys
import math
import unittest

WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

from krystal_kernel.scene_constructor_presets import VITAL_MAX_HP
from krystal_kernel.keras_tensor_math_engine import (
    TensorMatrix,
    CombinedObservationTensor,
    MultiDomainMatrixCompiler,
    KerasMatrixObserverModelSpec,
    GLOBAL_MATRIX_COMPILER
)
from krystal_kernel.cnc_adaptive_thermal_governor import StateSpaceTelemetryCoupler


class TestKerasTensorMathEngine(unittest.TestCase):
    """Rigorous mathematical tests for TensorMatrix linear algebra and discretization."""

    def test_tensor_matrix_basics(self):
        """Test instantiation, shape properties, sequence indexing and copying."""
        mat = TensorMatrix([[1.0, 2.0], [3.0, 4.0]], name="test_mat")
        self.assertEqual(mat.shape, (2, 2))
        self.assertEqual(len(mat), 2)
        self.assertEqual(mat[0], [1.0, 2.0])
        self.assertEqual(mat[1][1], 4.0)

        # Copy isolation
        copied = mat.copy()
        copied.data[0][0] = 99.0
        self.assertEqual(mat[0][0], 1.0)
        self.assertEqual(copied[0][0], 99.0)

    def test_tensor_matrix_algebra(self):
        """Test matrix addition, subtraction, transpose, and scalar scaling."""
        A = TensorMatrix([[1.0, 2.0], [3.0, 4.0]])
        B = TensorMatrix([[5.0, 6.0], [7.0, 8.0]])

        # A + B
        C_add = A.add(B)
        self.assertEqual(C_add.to_list(), [[6.0, 8.0], [10.0, 12.0]])

        # B - A
        C_sub = B.sub(A)
        self.assertEqual(C_sub.to_list(), [[4.0, 4.0], [4.0, 4.0]])

        # Scalar scaling
        C_scale = A.scale(2.5)
        self.assertEqual(C_scale.to_list(), [[2.5, 5.0], [7.5, 10.0]])

        # Transpose
        rect = TensorMatrix([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        rect_T = rect.transpose()
        self.assertEqual(rect_T.shape, (3, 2))
        self.assertEqual(rect_T.to_list(), [[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]])

    def test_tensor_matmul_and_vector_dot(self):
        """Test C = A x B matrix multiplication and y = A x vector."""
        A = TensorMatrix([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])  # (2x3)
        B = TensorMatrix([[7.0, 8.0], [9.0, 1.0], [2.0, 3.0]])  # (3x2)

        C = A.matmul(B)  # (2x2)
        self.assertEqual(C.shape, (2, 2))
        # row 0: 1*7 + 2*9 + 3*2 = 7 + 18 + 6 = 31; 1*8 + 2*1 + 3*3 = 8 + 2 + 9 = 19
        # row 1: 4*7 + 5*9 + 6*2 = 28 + 45 + 12 = 85; 4*8 + 5*1 + 6*3 = 32 + 5 + 18 = 55
        self.assertEqual(C.to_list(), [[31.0, 19.0], [85.0, 55.0]])

        # Vector dot
        vec = [2.0, -1.0, 3.0]
        y = A.vector_dot(vec)
        # y[0] = 1*2 + 2*(-1) + 3*3 = 2 - 2 + 9 = 9
        # y[1] = 4*2 + 5*(-1) + 6*3 = 8 - 5 + 18 = 21
        self.assertEqual(y, [9.0, 21.0])

    def test_tensor_matrix_norms_and_spectral_radius(self):
        """Test trace, Frobenius norm, and Gershgorin spectral radius upper bound."""
        diag_mat = TensorMatrix([[3.0, 0.0], [0.0, 4.0]])
        self.assertEqual(diag_mat.trace(), 7.0)
        self.assertEqual(diag_mat.frobenius_norm(), 5.0)  # sqrt(9 + 16) = 5.0
        self.assertEqual(diag_mat.spectral_radius_upper_bound(), 4.0)

        general_mat = TensorMatrix([[2.0, -1.0], [1.0, 3.0]])
        # Gershgorin bounds: row 0: |2| + |-1| = 3.0; row 1: |3| + |1| = 4.0 -> max is 4.0
        self.assertEqual(general_mat.spectral_radius_upper_bound(), 4.0)

    def test_pade_matrix_exponential_discretization(self):
        """Test continuous-to-discrete state transition: exp(A * dt)."""
        # Nilpotent test matrix: A = [[0, 1], [0, 0]] -> A^2 = 0 -> exp(A*dt) = I + A*dt
        A = TensorMatrix([[0.0, 1.0], [0.0, 0.0]])
        dt = 0.1
        exp_A = A.matrix_exp_pade(dt, order=4)
        self.assertAlmostEqual(exp_A[0][0], 1.0, places=6)
        self.assertAlmostEqual(exp_A[0][1], 0.1, places=6)
        self.assertAlmostEqual(exp_A[1][0], 0.0, places=6)
        self.assertAlmostEqual(exp_A[1][1], 1.0, places=6)

        # Integration test with StateSpaceTelemetryCoupler
        coupler = StateSpaceTelemetryCoupler()
        Ad = coupler.discretize_dynamics(dt=0.05)
        self.assertEqual(Ad.shape, (5, 5))
        # Diagonal elements of exp(A*dt) must be close to 1.0 for small dt
        for i in range(5):
            self.assertGreater(Ad[i][i], 0.0)


class TestMultiDomainObservationCompiler(unittest.TestCase):
    """Tests compilation and structure of physical, software, and unified matrices."""

    def test_physical_observation_matrix(self):
        """Matrix C_physical must map 5 physical states to 4 telemetry sensors."""
        C_phys = MultiDomainMatrixCompiler.compile_physical_observation_matrix()
        self.assertEqual(C_phys.shape, (4, 5))
        # Row 0: Spindle current (Amps) ~ 2.8 * Torque (col 2)
        self.assertEqual(C_phys[0][2], 2.80)
        # Row 1: Temperature delta ~ 1.0 * Delta_T (col 0)
        self.assertEqual(C_phys[1][0], 1.00)
        # Row 2: Coolant pressure ~ 1.0 * Coolant_P (col 4)
        self.assertEqual(C_phys[2][4], 1.00)
        # Row 3: Vibration RMS ~ 0.08 * Torque + 12.5 * VB
        self.assertEqual(C_phys[3][2], 0.08)
        self.assertEqual(C_phys[3][3], 12.5)

    def test_software_architecture_observation_matrix(self):
        """Matrix C_software must map 5 architectural states to 5 runtime metrics."""
        C_soft = MultiDomainMatrixCompiler.compile_software_architecture_observation_matrix()
        self.assertEqual(C_soft.shape, (5, 5))
        # Row 0: Latency ~ 1.0 * tau_exec
        self.assertEqual(C_soft[0][0], 1.00)
        # Row 1: IPC ~ 1.0 * IPC
        self.assertEqual(C_soft[1][1], 1.00)
        # Row 2: Cache hit rate ~ -25.0 * entropy
        self.assertEqual(C_soft[2][2], -25.0)
        # Row 3: Heap alloc rate ~ 1.2 * latency + 120.0 * entropy
        self.assertEqual(C_soft[3][0], 1.20)
        self.assertEqual(C_soft[3][2], 120.0)

    def test_unified_hyper_observation_block_diagonal_structure(self):
        """Unified matrix C_unified must be 9 x 10 with exact block-diagonal zero off-blocks."""
        C_unified = MultiDomainMatrixCompiler.compile_unified_hyper_observation_matrix()
        self.assertEqual(C_unified.shape, (9, 10))

        # Top-right cross block (rows 0-3, cols 5-9) MUST be strictly zero
        for r in range(4):
            for c in range(5, 10):
                self.assertEqual(C_unified[r][c], 0.0, f"Non-zero leak in physical-to-software block at ({r},{c})")

        # Bottom-left cross block (rows 4-8, cols 0-4) MUST be strictly zero
        for r in range(4, 9):
            for c in range(5):
                self.assertEqual(C_unified[r][c], 0.0, f"Non-zero leak in software-to-physical block at ({r},{c})")

        # Top-left block must match physical C
        C_phys = MultiDomainMatrixCompiler.compile_physical_observation_matrix()
        for r in range(4):
            for c in range(5):
                self.assertEqual(C_unified[r][c], C_phys[r][c])

        # Bottom-right block must match software C
        C_soft = MultiDomainMatrixCompiler.compile_software_architecture_observation_matrix()
        for r in range(5):
            for c in range(5):
                self.assertEqual(C_unified[4 + r][5 + c], C_soft[r][c])

    def test_combined_observation_vector_packaging(self):
        """Test CombinedObservationTensor vector serialization."""
        obs = CombinedObservationTensor(
            timestamp=100.0,
            spindle_current_amps=14.2,
            delta_temperature_c=185.0,
            coolant_pressure_mpa=2.4,
            vibration_rms_g=0.32,
            execution_latency_us=18.5,
            instructions_per_cycle_ipc=1.45,
            l2_cache_hit_rate_pct=94.2,
            heap_alloc_rate_kb_s=45.0,
            thread_thrashing_index=0.08,
            tool_vital_hp=6
        )
        self.assertEqual(len(obs.to_physical_vector()), 4)
        self.assertEqual(len(obs.to_software_vector()), 5)
        self.assertEqual(len(obs.to_unified_vector()), 9)
        self.assertEqual(obs.vital_max_hp, VITAL_MAX_HP)
        self.assertEqual(VITAL_MAX_HP, 6)


class TestKerasModelSpecificationAndInference(unittest.TestCase):
    """Tests for the Keras 3 Functional API graph spec and vectorized forward pass."""

    def test_keras_graph_specification_structure(self):
        """Verify Keras model specification matches Keras 3 Functional graph standard."""
        spec = KerasMatrixObserverModelSpec.get_keras_model_graph_spec()
        self.assertEqual(spec["model_type"], "Keras_Functional_API_v3")
        self.assertEqual(spec["vital_max_hp"], 6)

        # Check dual inputs
        self.assertEqual(len(spec["inputs"]), 2)
        input_names = [inp["name"] for inp in spec["inputs"]]
        self.assertIn("physical_telemetry_input", input_names)
        self.assertIn("software_telemetry_input", input_names)

        # Check multi-head outputs
        output_names = [out["name"] for out in spec["outputs"]]
        self.assertIn("reconstructed_hyper_state", output_names)
        self.assertIn("vital_hp_logits", output_names)
        self.assertIn("collapse_hazard_probabilities", output_names)

        # Validate vital HP distribution units == 7 (for HP in {0, 1, 2, 3, 4, 5, 6})
        hp_head = next(o for o in spec["outputs"] if o["name"] == "vital_hp_logits")
        self.assertEqual(hp_head["units"], 7)
        self.assertEqual(hp_head["activation"], "softmax")

    def test_nominal_forward_pass_preserves_vital_invariant(self):
        """In nominal operations, HP must equal 6 (VITAL_MAX_HP)."""
        obs = CombinedObservationTensor(
            timestamp=1.0,
            spindle_current_amps=10.0,
            delta_temperature_c=120.0,
            coolant_pressure_mpa=2.5,
            vibration_rms_g=0.20,
            execution_latency_us=12.0,
            instructions_per_cycle_ipc=1.8,
            l2_cache_hit_rate_pct=98.0,
            heap_alloc_rate_kb_s=20.0,
            thread_thrashing_index=0.02,
            tool_vital_hp=6
        )

        res = KerasMatrixObserverModelSpec.evaluate_dual_domain_forward_pass(obs)
        self.assertEqual(res["status"], "EVALUATED")
        self.assertEqual(res["estimated_vital_hp"], 6)
        self.assertEqual(res["vital_max_hp"], 6)
        self.assertLess(res["aggregate_strain_index"], 0.35)
        self.assertEqual(len(res["reconstructed_hyper_state"]), 10)
        self.assertLess(res["hazard_probabilities"]["spindle_torque_stall"], 0.05)
        self.assertLess(res["hazard_probabilities"]["coolant_pressure_loss"], 0.05)

    def test_stressed_forward_pass_degradation_and_hazard_detection(self):
        """Under thermal runaway and thrashing, HP degrades gracefully but stays in [0, 6]."""
        stressed_obs = CombinedObservationTensor(
            timestamp=2.0,
            spindle_current_amps=32.0,       # Severe overload
            delta_temperature_c=650.0,       # Extreme cutting tip heat
            coolant_pressure_mpa=0.4,        # Coolant loss
            vibration_rms_g=1.8,             # Chatter vibration
            execution_latency_us=120.0,      # Software pipeline lagging
            instructions_per_cycle_ipc=0.45, # Execution stall
            l2_cache_hit_rate_pct=65.0,      # Cache misses
            heap_alloc_rate_kb_s=450.0,      # Heavy allocation thrash
            thread_thrashing_index=0.88,     # High contention
            tool_vital_hp=6
        )

        res = KerasMatrixObserverModelSpec.evaluate_dual_domain_forward_pass(stressed_obs)
        self.assertEqual(res["status"], "EVALUATED")
        # HP must degrade
        self.assertLessEqual(res["estimated_vital_hp"], 3)
        self.assertGreaterEqual(res["estimated_vital_hp"], 0)
        self.assertEqual(res["vital_max_hp"], 6)

        # Hazards must be triggered
        self.assertGreater(res["hazard_probabilities"]["spindle_torque_stall"], 0.50)
        self.assertGreater(res["hazard_probabilities"]["coolant_pressure_loss"], 0.50)
        self.assertGreater(res["hazard_probabilities"]["thread_thrashing_hazard"], 0.50)


if __name__ == "__main__":
    unittest.main()
