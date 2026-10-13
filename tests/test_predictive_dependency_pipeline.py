#!/usr/bin/env python3
"""
Unit and integration test suite for Predictive Dependency Pipeline
Validates:
  1. fetch_dependencies (upstream, downstream, both)
  2. synchronize (vector clocks, lease TTL, drift tolerance)
  3. overlap (Jaccard index, sliding window temporal overlap, cosine similarity)
  4. complement (algebraic 1 - x, missing dependencies relative to universe)
  5. add (element-wise addition, topological union)
  6. subtract (element-wise subtraction, residual error delta)
  7. multiply (scaling factor, element-wise bundle product)
  8. divide (element-wise division, guarded epsilon against zero division)
  9. calculate_via_x (quadratic manifold mapping via latent x, vital_hp constraints)
  10. hybrid MCP tool registration and execution

INVARIANT: assert VITAL_MAX_HP == 6
"""

import unittest
import math
import time

from krystal_kernel.scene_constructor_presets import VITAL_MAX_HP
from krystal_kernel.verification.epistemic_taxonomies import (
    PredictiveDependencyGraph,
    DependencyEdgeKind,
)
from krystal_kernel.verification.predictive_dependency_pipeline import (
    PredictiveDependencyPipeline,
    DependencyBundle,
    SyncReport,
    OverlapReport,
    ComplementReport,
    GLOBAL_DEPENDENCY_PIPELINE,
    EPSILON,
)
from krystal_web_hub.hybrid_mcp_antigravity_connector import GLOBAL_HYBRID_MCP_CONNECTOR


class TestPredictiveDependencyPipeline(unittest.TestCase):

    def setUp(self):
        assert VITAL_MAX_HP == 6, "Invariant VITAL_MAX_HP must be 6"
        self.pipeline = PredictiveDependencyPipeline()
        self.graph = PredictiveDependencyGraph()

        # Build a representative epistemic dependency network
        # Pipeline: sensor_thermal -> compute_scheduler -> thermal_governor -> actuator_fan
        #                          -> memory_allocator  -> swap_daemon
        self.graph.add_node(
            node_id="sensor_thermal",
            domain="thermal_celsius",
            value=65.0,
            interval=(0.0, 100.0),
            clock={"node_0": 10},
            lease_ttl_ns=10_000_000_000,
            confidence=0.98
        )
        self.graph.add_node(
            node_id="compute_scheduler",
            domain="cpu_load_pct",
            value=80.0,
            interval=(10.0, 90.0),
            clock={"node_0": 10, "node_1": 15},
            lease_ttl_ns=5_000_000_000,
            confidence=0.92
        )
        self.graph.add_node(
            node_id="thermal_governor",
            domain="vital_hp",
            value=4.5,
            interval=(20.0, 120.0),
            clock={"node_0": 10, "node_1": 15, "node_2": 8},
            lease_ttl_ns=8_000_000_000,
            confidence=0.95
        )
        self.graph.add_node(
            node_id="actuator_fan",
            domain="gpu_power_watts",
            value=25.0,
            interval=(30.0, 150.0),
            clock={"node_2": 8, "node_3": 20},
            lease_ttl_ns=12_000_000_000,
            confidence=0.90
        )
        self.graph.add_node(
            node_id="memory_allocator",
            domain="uma_pressure",
            value=0.6,
            interval=(15.0, 95.0),
            clock={"node_1": 15, "node_4": 5},
            lease_ttl_ns=4_000_000_000,
            confidence=0.88
        )
        self.graph.add_node(
            node_id="swap_daemon",
            domain="swap_util_pct",
            value=35.0,
            interval=(25.0, 105.0),
            clock={"node_4": 5, "node_5": 12},
            lease_ttl_ns=6_000_000_000,
            confidence=0.85
        )

        # Edges
        self.graph.add_edge("sensor_thermal", "compute_scheduler", DependencyEdgeKind.EPISTEMIC_CAUSAL, 1.0)
        self.graph.add_edge("compute_scheduler", "thermal_governor", DependencyEdgeKind.EPISTEMIC_CAUSAL, 0.9)
        self.graph.add_edge("compute_scheduler", "memory_allocator", DependencyEdgeKind.EPISTEMIC_CAUSAL, 0.8)
        self.graph.add_edge("thermal_governor", "actuator_fan", DependencyEdgeKind.CORRELATIONAL, 0.75)
        self.graph.add_edge("memory_allocator", "swap_daemon", DependencyEdgeKind.TEMPORAL, 0.65)

    def test_01_fetch_dependencies(self):
        """Test fetch_dependencies in upstream, downstream, and both directions."""
        # Upstream from thermal_governor -> should yield thermal_governor, compute_scheduler, sensor_thermal
        bundle_up = self.pipeline.fetch_dependencies(self.graph, "thermal_governor", direction="upstream")
        self.assertIn("thermal_governor", bundle_up.nodes)
        self.assertIn("compute_scheduler", bundle_up.nodes)
        self.assertIn("sensor_thermal", bundle_up.nodes)
        self.assertNotIn("actuator_fan", bundle_up.nodes)
        self.assertEqual(bundle_up.vital_max_hp, 6)

        # Downstream from compute_scheduler -> thermal_governor, memory_allocator, actuator_fan, swap_daemon
        bundle_down = self.pipeline.fetch_dependencies(self.graph, "compute_scheduler", direction="downstream")
        self.assertIn("compute_scheduler", bundle_down.nodes)
        self.assertIn("thermal_governor", bundle_down.nodes)
        self.assertIn("memory_allocator", bundle_down.nodes)
        self.assertIn("actuator_fan", bundle_down.nodes)
        self.assertIn("swap_daemon", bundle_down.nodes)
        self.assertNotIn("sensor_thermal", bundle_down.nodes)

        # Both from compute_scheduler -> includes sensor_thermal and all descendants
        bundle_both = self.pipeline.fetch_dependencies(self.graph, "compute_scheduler", direction="both")
        self.assertEqual(len(bundle_both.nodes), 6)
        self.assertGreater(len(bundle_both.topological_order), 0)

    def test_02_synchronize(self):
        """Test synchronization, vector clock consensus, and lease validation."""
        bundle = self.pipeline.fetch_dependencies(self.graph, "thermal_governor", direction="both")
        now_ns = time.time_ns()

        report = self.pipeline.synchronize(bundle, now_ns=now_ns, max_drift_ns=100_000_000_000)
        self.assertIsInstance(report, SyncReport)
        self.assertTrue(report.is_synchronized)
        self.assertEqual(len(report.expired_nodes), 0)
        self.assertGreater(len(report.vector_clock_consensus), 0)
        self.assertEqual(report.vital_max_hp, 6)

        # Simulate expired lease on one node
        self.graph.nodes["sensor_thermal"].lease_ttl_ns = 1  # 1 ns
        self.graph.nodes["sensor_thermal"].timestamp_ns = 1000
        bundle_expired = self.pipeline.fetch_dependencies(self.graph, "thermal_governor", direction="both")
        report_exp = self.pipeline.synchronize(bundle_expired, now_ns=now_ns, max_drift_ns=1000)
        self.assertIn("sensor_thermal", report_exp.expired_nodes)
        self.assertFalse(report_exp.is_synchronized)

    def test_03_overlap(self):
        """Test topological Jaccard index, temporal sliding overlap, and cosine similarity."""
        bundle_a = self.pipeline.fetch_dependencies(self.graph, "thermal_governor", direction="upstream")
        bundle_b = self.pipeline.fetch_dependencies(self.graph, "actuator_fan", direction="upstream")

        report = self.pipeline.overlap(bundle_a, bundle_b)
        self.assertIsInstance(report, OverlapReport)
        self.assertGreater(report.jaccard_index, 0.0)
        self.assertLessEqual(report.jaccard_index, 1.0)
        self.assertIn("thermal_governor", report.intersection_node_ids)
        self.assertIn("compute_scheduler", report.intersection_node_ids)
        self.assertIn("sensor_thermal", report.intersection_node_ids)
        self.assertGreater(report.cosine_similarity, 0.0)
        self.assertEqual(report.vital_max_hp, 6)

    def test_04_complement(self):
        """Test algebraic complement (1 - x) and set complement."""
        bundle = self.pipeline.fetch_dependencies(self.graph, "thermal_governor", direction="upstream")
        universe = set(self.graph.nodes.keys())

        comp_report = self.pipeline.complement(bundle, universe_node_ids=universe)
        self.assertIsInstance(comp_report, ComplementReport)
        self.assertEqual(len(comp_report.missing_dependencies), 3)  # actuator_fan, memory_allocator, swap_daemon
        self.assertIn("actuator_fan", comp_report.missing_dependencies)

        # Verify algebraic complement (1 - x) for each active node
        for nid, node_comp in comp_report.algebraic_complements.items():
            orig_val = bundle.values[nid]
            dom_id = bundle.domains.get(nid)
            if dom_id in self.pipeline.transformer._domains:
                solved_x, _, _ = self.pipeline.transformer._domains[dom_id].solve_x(orig_val)
                expected_comp_x = round(1.0 - (solved_x if solved_x is not None else 0.5), 5)
            else:
                norm_val = bundle.intervals[nid]
                diff = norm_val[1] - norm_val[0]
                expected_x = (orig_val - norm_val[0]) / (diff if diff != 0 else 1.0)
                expected_comp_x = round(1.0 - max(0.0, min(1.0, expected_x)), 5)
            self.assertAlmostEqual(node_comp["complement_latent_x"], expected_comp_x, places=4)
        self.assertEqual(comp_report.vital_max_hp, 6)

    def test_05_arithmetic_operations(self):
        """Test add, subtract, multiply, and divide with epsilon protections."""
        bundle_a = self.pipeline.fetch_dependencies(self.graph, "compute_scheduler", direction="both")
        bundle_b = self.pipeline.fetch_dependencies(self.graph, "compute_scheduler", direction="both")

        # 1. ADD
        added = self.pipeline.add(bundle_a, bundle_b)
        for nid, val in bundle_a.values.items():
            self.assertAlmostEqual(added.values[nid], val * 2.0, places=4)

        # 2. SUBTRACT
        subbed = self.pipeline.subtract(bundle_a, bundle_b)
        for nid in bundle_a.values:
            self.assertAlmostEqual(subbed.values[nid], 0.0, places=4)

        # 3. MULTIPLY (scalar & bundle)
        scaled = self.pipeline.multiply(bundle_a, 2.5)
        for nid, val in bundle_a.values.items():
            self.assertAlmostEqual(scaled.values[nid], val * 2.5, places=4)

        multiplied = self.pipeline.multiply(bundle_a, bundle_b)
        for nid, val in bundle_a.values.items():
            self.assertAlmostEqual(multiplied.values[nid], val * val, places=4)

        # 4. DIVIDE (guarded against zero)
        divided = self.pipeline.divide(bundle_a, bundle_b)
        for nid, val in bundle_a.values.items():
            self.assertAlmostEqual(divided.values[nid], 1.0, places=4)

        # Division by zero protection:
        zero_bundle = self.pipeline.subtract(bundle_a, bundle_b)
        safe_div = self.pipeline.divide(bundle_a, zero_bundle, epsilon=1e-12)
        for nid in bundle_a.values:
            self.assertFalse(math.isnan(safe_div.values[nid]))
            self.assertFalse(math.isinf(safe_div.values[nid]))

    def test_06_calculate_via_x(self):
        """Test quadratic variable manifold projection via latent x with invariant assert VITAL_MAX_HP == 6."""
        bundle = self.pipeline.fetch_dependencies(self.graph, "thermal_governor", direction="both")

        # Target: vital_hp
        res_vital = self.pipeline.calculate_via_x(
            bundle,
            target_domain="vital_hp",
            latent_operation="identity"
        )
        self.assertTrue(res_vital["constitutional_compliance"])
        self.assertEqual(res_vital["vital_max_hp_rule"], 6)
        self.assertLessEqual(res_vital["aggregate"]["max"], 6.0)

        # Target: cpu_load_pct with harmonic operation
        res_cpu = self.pipeline.calculate_via_x(
            bundle,
            target_domain="cpu_load_pct",
            latent_operation="harmonic"
        )
        self.assertGreater(res_cpu["aggregate"]["count"], 0)
        self.assertGreaterEqual(res_cpu["aggregate"]["min"], 0.0)

    def test_07_mcp_tool_execution(self):
        """Verify the 'krystal-dependency-pipeline' tool registered in hybrid MCP connector."""
        # 1. Fetch action via MCP
        res_fetch = GLOBAL_HYBRID_MCP_CONNECTOR.execute_tool("krystal-dependency-pipeline", {
            "action": "fetch",
            "root_node_id": "compute_scheduler",
            "direction": "both"
        })
        self.assertEqual(res_fetch["status"], "OK")
        self.assertEqual(res_fetch["vital_max_hp"], 6)
        self.assertIn("bundle", res_fetch)

        # 2. Arithmetic action via MCP (multiply)
        res_mul = GLOBAL_HYBRID_MCP_CONNECTOR.execute_tool("krystal-dependency-pipeline", {
            "action": "multiply",
            "root_node_id": "compute_scheduler",
            "scalar_factor": 1.5
        })
        self.assertEqual(res_mul["status"], "OK")
        self.assertIn("bundle", res_mul)

        # 3. Calculate via x action via MCP
        res_x = GLOBAL_HYBRID_MCP_CONNECTOR.execute_tool("krystal-dependency-pipeline", {
            "action": "calculate_via_x",
            "root_node_id": "compute_scheduler",
            "target_domain": "vital_hp",
            "latent_operation": "complement"
        })
        self.assertEqual(res_x["status"], "OK")
        self.assertTrue(res_x["transformation"]["constitutional_compliance"])


if __name__ == "__main__":
    unittest.main()
