"""
Unit Test Suite for GraphQL DAG Computer Vision Inspector,
Heterogeneous Machine-Scale GTD Scheduler, and LLM-RAG Atomic Task Decomposer.

Verification of Theory Node 48:
THEORY-GRAPHQL-CV-PREDICTIVE-TASK-TACTIC-GTD
Constitutional Invariant: VITAL_MAX_HP == 6
Zero-Dirty Mutation Contract: Delta S == 0 on breach
"""

import unittest
import time
import uuid
import hashlib
from typing import List, Dict, Any

from krystal_kernel.graphql_cv_task_gtd_engine import (
    HeterogeneousDeviceTarget,
    GtdTaskPhase,
    BranchEfficiencyStatus,
    GraphQLDagNode,
    CvGraphInspectionDigest,
    TraceableSyntheticDataPayload,
    GtdExecutionReceipt,
    ComputerVisionGraphInspector,
    HeterogeneousGtdScheduler,
    LlmRagTaskDecomposer,
    GLOBAL_GRAPHQL_CV_GTD_SCHEDULER,
    GLOBAL_LLM_RAG_DECOMPOSER,
    VITAL_MAX_HP,
)
from krystal_kernel.verification.theory_dependency_graph import GLOBAL_THEORY_DEPENDENCY_GRAPH


class TestGraphQLCvTaskGtdEngine(unittest.TestCase):
    """Test suite for Theory Node 48 verification."""

    def setUp(self):
        self.inspector = ComputerVisionGraphInspector(raster_width=64, raster_height=32)
        self.scheduler = HeterogeneousGtdScheduler()
        self.decomposer = LlmRagTaskDecomposer()

    def test_theory_node_registration(self):
        """Verify that Theory Node 48 is registered in the global dependency graph."""
        theory_id = "THEORY-GRAPHQL-CV-PREDICTIVE-TASK-TACTIC-GTD"
        self.assertIn(theory_id, GLOBAL_THEORY_DEPENDENCY_GRAPH._nodes)
        node = GLOBAL_THEORY_DEPENDENCY_GRAPH._nodes[theory_id]
        self.assertEqual(node.epistemic_status.name, "DOKAZANE")
        self.assertFalse(GLOBAL_THEORY_DEPENDENCY_GRAPH.has_cycles())
        self.assertIn("VITAL_MAX_HP == 6", node.assumptions)

    def test_dag_visual_inspection_topological(self):
        """Test computer vision DAG topological inspection and critical path detection."""
        nodes = [
            GraphQLDagNode(
                node_id="query_entry",
                field_name="viewer",
                depth=0,
                dependencies=[],
                est_compute_flops=10000.0,
                est_memory_bytes=4096,
                is_parallelizable=False,
            ),
            GraphQLDagNode(
                node_id="sub_profile",
                field_name="userProfile",
                depth=1,
                dependencies=["query_entry"],
                est_compute_flops=50000.0,
                est_memory_bytes=16384,
                is_parallelizable=True,
            ),
            GraphQLDagNode(
                node_id="sub_orders",
                field_name="orderHistory",
                depth=1,
                dependencies=["query_entry"],
                est_compute_flops=120000.0,
                est_memory_bytes=65536,
                is_parallelizable=True,
            ),
            GraphQLDagNode(
                node_id="sub_analytics",
                field_name="spendAnalytics",
                depth=2,
                dependencies=["sub_orders"],
                est_compute_flops=350000.0,
                est_memory_bytes=262144,
                is_parallelizable=True,
            ),
        ]

        digest = self.inspector.inspect_dag(nodes)
        self.assertIsInstance(digest, CvGraphInspectionDigest)
        self.assertEqual(digest.node_count, 4)
        self.assertGreaterEqual(digest.edge_count, 3)
        self.assertGreaterEqual(digest.max_topological_depth, 2)
        self.assertIn("sub_analytics", digest.critical_path)
        self.assertGreater(digest.critical_path_est_us, 0.0)
        self.assertEqual(digest.vital_max_hp, VITAL_MAX_HP)
        self.assertEqual(len(digest.visual_feature_signature), 64)

    def test_cv_fan_out_choke_point_and_barrier_hazard(self):
        """Test detection of fan-out choke points and memory barrier hazard scoring."""
        # Create a hub node with large fan-out
        nodes = [
            GraphQLDagNode("root", "feed", 0, [], 1000.0, 1024, False),
        ]
        for i in range(8):
            nodes.append(GraphQLDagNode(f"child_{i}", f"item_{i}", 1, ["root"], 5000.0, 8192, True))

        digest = self.inspector.inspect_dag(nodes)
        self.assertIn("root", digest.fan_out_choke_points)
        self.assertGreater(digest.memory_barrier_hazard_score, 0.1)

    def test_gtd_six_phase_heterogeneous_scheduling(self):
        """Test the 6-phase machine GTD execution cycle and device recommendation."""
        query_str = """
        query FetchTelemetryDashboard {
            clusterState {
                nodes {
                    id
                    cpuUsage
                    thermalEnvelope
                }
                gpuMatrix {
                    id
                    tensorOpsPerSec
                }
            }
        }
        """

        receipt = self.scheduler.execute_gtd_cycle(
            query_str=query_str,
            preferred_device=HeterogeneousDeviceTarget.AUTO_SELECT,
        )

        self.assertIsInstance(receipt, GtdExecutionReceipt)
        self.assertEqual(receipt.vital_max_hp, 6)
        self.assertTrue(receipt.state_committed)
        self.assertEqual(receipt.delta_s, 0.0)
        self.assertIn(receipt.recommended_device, [
            HeterogeneousDeviceTarget.CPU_AVX512,
            HeterogeneousDeviceTarget.GPU_IRIS_XE,
            HeterogeneousDeviceTarget.NPU_HAILO,
        ])
        self.assertEqual(receipt.active_phase, GtdTaskPhase.REVIEW)
        self.assertGreater(receipt.actual_execution_us, 0.0)
        self.assertLessEqual(receipt.prediction_error_residual, 0.35)
        self.assertTrue(receipt.traceable_payload.run_uuid)
        self.assertEqual(len(receipt.hmac_signature), 64)

    def test_traceable_synthetic_data_generation_and_attribution(self):
        """Test generation of unique UUID-tagged synthetic test loads and branch efficiency evaluation."""
        nodes = [
            GraphQLDagNode("compute_heavy", "heavyTensorMatrix", 0, [], 800000.0, 524288, True),
            GraphQLDagNode("io_light", "pingStatus", 0, [], 500.0, 64, False),
        ]

        payload = self.scheduler.generate_traceable_synthetic_payload(
            nodes=nodes,
            target_device=HeterogeneousDeviceTarget.GPU_IRIS_XE,
        )

        self.assertIsInstance(payload, TraceableSyntheticDataPayload)
        self.assertTrue(uuid.UUID(payload.run_uuid))
        self.assertIn("branch_compute_heavy", payload.branch_efficiency_attribution)
        self.assertIn("branch_io_light", payload.branch_efficiency_attribution)
        self.assertIn(payload.branch_efficiency_attribution["branch_compute_heavy"], [
            BranchEfficiencyStatus.EFFECTIVE.value,
            BranchEfficiencyStatus.BOTTLENECKED.value,
            BranchEfficiencyStatus.THROTTLED.value,
        ])

    def test_llm_rag_atomic_task_decomposition(self):
        """Test decomposing complex query into atomic sub-tasks with RAG retrieval."""
        query_str = """
        query ComplexEnterpriseReport {
            enterprise(id: "krystal_corp") {
                financialLedger {
                    quarterlyRevenues
                    ebitdaMargin
                }
                datacenterFleet {
                    racks {
                        pue
                        powerDrawWatts
                    }
                }
            }
        }
        """

        res = self.decomposer.decompose_query(query_str)
        self.assertIn("atomic_sub_tasks", res)
        sub_tasks = res["atomic_sub_tasks"]
        self.assertGreaterEqual(len(sub_tasks), 3)

        # Ensure all sub-tasks are GraphQLDagNodes
        for task in sub_tasks:
            self.assertIsInstance(task, GraphQLDagNode)
            self.assertGreater(task.est_compute_flops, 0.0)

        # Check RAG match
        self.assertIn("rag_pattern_match", res)
        self.assertIn("rag_similarity_score", res)
        self.assertGreaterEqual(res["rag_similarity_score"], 0.0)

    def test_constitutional_invariants_and_error_handling(self):
        """Test that VITAL_MAX_HP == 6 is enforced and invalid execution triggers zero-dirty mutation."""
        empty_nodes: List[GraphQLDagNode] = []
        digest = self.inspector.inspect_dag(empty_nodes)
        self.assertEqual(digest.node_count, 0)
        self.assertEqual(digest.vital_max_hp, 6)

        # Test execution with empty nodes - should complete cleanly without corrupting state
        receipt = self.scheduler.execute_gtd_cycle(
            query_str="{ __schema { types { name } } }",
            dag_nodes=[],
        )
        self.assertEqual(receipt.vital_max_hp, 6)
        self.assertEqual(receipt.delta_s, 0.0)
        self.assertTrue(receipt.state_committed)


if __name__ == "__main__":
    unittest.main()
