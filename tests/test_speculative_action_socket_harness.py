"""
Unit Test Suite for Speculative Action Interpreter, Socket Sub-Unit Anti-Oscillation
Hysteresis, and 2D Multi-Spectral Computer Vision Graph Encoders.

Verification of Theory Node 49:
THEORY-SPECULATIVE-ACTION-INTERPRETER-HYSTERESIS-SOCKET-NETWORKS
Constitutional Invariant: VITAL_MAX_HP == 6
"""

import unittest
import time
from typing import List

from krystal_kernel.speculative_action_socket_harness import (
    SocketSubUnitType,
    SpeculativeActionType,
    GraphVisualTensor,
    SpeculativeActionCandidate,
    GovernorSplicerReceipt,
    CvGraphImageEncoder,
    SpeculativeActionSocketHarness,
    GLOBAL_SPECULATIVE_SOCKET_HARNESS,
)
from krystal_kernel.graphql_cv_task_gtd_engine import (
    GraphQLDagNode,
    HeterogeneousDeviceTarget,
    VITAL_MAX_HP,
)
from krystal_kernel.verification.theory_dependency_graph import GLOBAL_THEORY_DEPENDENCY_GRAPH


class TestSpeculativeActionSocketHarness(unittest.TestCase):
    """Test suite for Theory Node 49 verification."""

    def setUp(self):
        self.harness = SpeculativeActionSocketHarness()
        self.encoder = CvGraphImageEncoder(raster_width=16, raster_height=16)

    def test_theory_node_49_registration(self):
        """Verify that Theory Node 49 is formally registered without cycles."""
        theory_id = "THEORY-SPECULATIVE-ACTION-INTERPRETER-HYSTERESIS-SOCKET-NETWORKS"
        self.assertIn(theory_id, GLOBAL_THEORY_DEPENDENCY_GRAPH._nodes)
        node = GLOBAL_THEORY_DEPENDENCY_GRAPH._nodes[theory_id]
        self.assertEqual(node.epistemic_status.name, "DOKAZANE")
        self.assertFalse(GLOBAL_THEORY_DEPENDENCY_GRAPH.has_cycles())

    def test_cv_multi_spectral_tensor_encoding(self):
        """Test encoding GraphQL DAG into a 4-channel 2D multi-spectral tensor."""
        nodes = [
            GraphQLDagNode("node_01", "preprocess", [], 50000.0, 1048576, 1, HeterogeneousDeviceTarget.CPU_AVX512),
            GraphQLDagNode("node_02", "tensor_matmul", ["node_01"], 2000000.0, 8388608, 1, HeterogeneousDeviceTarget.GPU_IRIS_XE),
            GraphQLDagNode("node_03", "neural_infer", ["node_01"], 1500000.0, 4194304, 1, HeterogeneousDeviceTarget.NPU_HAILO),
        ]

        tensor = self.encoder.encode_dag_to_visual_tensor(nodes)
        self.assertIsInstance(tensor, GraphVisualTensor)
        self.assertEqual(tensor.channels, 4)
        self.assertEqual(tensor.width, 16)
        self.assertEqual(tensor.height, 16)
        self.assertGreater(tensor.spatial_entropy, 0.0)
        self.assertGreater(tensor.uma_contention_hazard, 0.0)
        self.assertEqual(len(tensor.perceptual_hash), 64)
        self.assertEqual(tensor.vital_max_hp, 6)

    def test_socket_hysteresis_redundant_threshold(self):
        """Test that redundant hysteresis h_socket >= 2*delta + kappa*sigma_strain is strictly maintained."""
        h_pcore = self.harness.compute_socket_hysteresis_threshold("P_CORE_0")
        self.assertGreaterEqual(h_pcore, 0.05)

        # High strain unit should produce higher threshold to suppress jitter
        self.harness.socket_units["P_CORE_0"].temperature_celsius = 78.0
        self.harness.socket_units["P_CORE_0"].voltage_droop_mv = 32.0
        h_strained = self.harness.compute_socket_hysteresis_threshold("P_CORE_0")
        self.assertGreater(h_strained, h_pcore)

    def test_speculative_action_candidates_and_asymmetry(self):
        """Test candidate generation and asymmetric transition weighting."""
        dummy_nodes = [GraphQLDagNode("n1", "compute", [], 1e6, 2e6, 1)]
        tensor = self.encoder.encode_dag_to_visual_tensor(dummy_nodes)
        candidates = self.harness.generate_speculative_candidates(tensor)

        self.assertGreaterEqual(len(candidates), 4)
        # Verify asymmetric transition: upscale has higher momentum than downscale
        upscale = next(c for c in candidates if c.action_type == SpeculativeActionType.ASYMMETRIC_UPSCALE)
        downscale = next(c for c in candidates if c.action_type == SpeculativeActionType.HYSTERETIC_DOWNSCALE)
        self.assertGreater(upscale.asymmetric_weight, downscale.asymmetric_weight)

    def test_governor_splicer_cycle_and_oscillation_suppression(self):
        """Test complete governor & splicer cycle under sandwich architecture."""
        nodes = [
            GraphQLDagNode("task_a", "matrix_batch", [], 1.2e7, 1.6e7, 1, HeterogeneousDeviceTarget.GPU_IRIS_XE),
            GraphQLDagNode("task_b", "audit_tail", ["task_a"], 5e5, 1024, 5, HeterogeneousDeviceTarget.CPU_AVX512),
        ]

        # Trigger oscillation in P_CORE_0
        self.harness.socket_units["P_CORE_0"].oscillation_rate_hz = 1.8

        receipt = self.harness.execute_governor_splicer_cycle(
            graphql_query_str="{ batchMatrix { ops { id } } }",
            nodes=nodes,
        )

        self.assertIsInstance(receipt, GovernorSplicerReceipt)
        self.assertEqual(receipt.vital_max_hp, 6)
        self.assertEqual(receipt.delta_s, 0.0)
        self.assertTrue(receipt.is_safe)
        self.assertGreaterEqual(receipt.control_validity_quota, 0.90)
        self.assertGreater(receipt.asymmetric_transition_ratio, 1.0)
        self.assertEqual(len(receipt.hmac_digest), 64)

    def test_reverse_dependency_engineering(self):
        """Test causal reverse dependency tracing (Action -> Reaction -> Deduction)."""
        nodes = [GraphQLDagNode("n1", "compute", [], 1e6, 1024, 1)]
        receipt = self.harness.execute_governor_splicer_cycle("query", nodes)
        diag = self.harness.reverse_engineer_socket_dependencies(receipt)

        self.assertIn("receipt_id", diag)
        self.assertIn("action_executed", diag)
        self.assertIn("reverse_inferred_choke_points", diag)
        self.assertEqual(diag["vital_max_hp"], 6)


if __name__ == "__main__":
    unittest.main()
