#!/usr/bin/env python3
"""
==============================================================================
KRYSTAL-STACK: UNIT TESTS FOR PREDICTIVE DEPENDENCY PIPELINE & ARITHMETIC
==============================================================================
Module: tests/test_predictive_dependency_pipeline.py
Description:
    Exhaustive verification of the PredictiveDependencyPipeline:
    - fetch_dependencies (upstream, downstream, both, cycle-safety, topological sort)
    - synchronize (monotonic clock skew, lease TTL, vector clock consensus)
    - overlap (Jaccard similarity, sliding window interval, cosine value similarity)
    - complement (algebraic complement 1 - x, domain universe padding)
    - divide (element-wise division, scalar division, epsilon zero-guard)
    - multiply (element-wise, scalar, confidence decay)
    - subtract (element-wise, residual L2 norm error)
    - add (element-wise, scalar, confidence bounds)
    - calculate_via_x (Quadratic Variable Manifold, latent shifts, VITAL_MAX_HP == 6)

Constitutional Invariant:
    assert VITAL_MAX_HP == 6
==============================================================================
"""

import math
import time
import unittest

from krystal_kernel.scene_constructor_presets import VITAL_MAX_HP
from krystal_kernel.verification.epistemic_taxonomies import (
    DependencyEdgeKind,
    PredictiveDependencyGraph,
    PredictiveDependencyNode,
)
from krystal_kernel.verification.predictive_dependency_pipeline import (
    DependencyBundle,
    PredictiveDependencyPipeline,
    SyncReport,
    OverlapReport,
    ComplementReport,
    EPSILON,
)
from krystal_web_hub.economic_engine.quadratic_variable_transformer import (
    GLOBAL_QUADRATIC_TRANSFORMER,
    INV_GOLDEN_RATIO,
)


class TestPredictiveDependencyPipeline(unittest.TestCase):

    def setUp(self):
        assert VITAL_MAX_HP == 6, f"Constitutional invariant breach: {VITAL_MAX_HP}"
        self.now = time.time_ns()

        # Build standard test graph
        self.graph = PredictiveDependencyGraph()

        # Nodes
        self.node_sensor = PredictiveDependencyNode(
            node_id="hw_sensor",
            subsystem="HARDWARE_SENSOR",
            epistemic_kind="MEASURED",
            lease_ttl_ns=10_000_000_000,
            value=25.0,
            domain_id="memory_latency_ns",
            confidence=0.98,
            valid_from_ns=self.now,
            interval=(0.0, 50.0),
            vector_clock={"hw_sensor": 1},
        )
        self.node_forecast = PredictiveDependencyNode(
            node_id="kernel_forecast",
            subsystem="BACKEND_KERNEL",
            epistemic_kind="PREDICTED",
            lease_ttl_ns=5_000_000_000,
            value=40.0,
            domain_id="memory_latency_ns",
            confidence=0.90,
            valid_from_ns=self.now,
            interval=(20.0, 70.0),
            vector_clock={"hw_sensor": 1, "kernel_forecast": 2},
        )
        self.node_ui = PredictiveDependencyNode(
            node_id="ui_hud_view",
            subsystem="FRONTEND_UI",
            epistemic_kind="VIEW_MODEL",
            lease_ttl_ns=2_000_000_000,
            value=5.0,
            domain_id="vital_hp",
            confidence=0.95,
            valid_from_ns=self.now,
            interval=(30.0, 80.0),
            vector_clock={"ui_hud_view": 1},
        )
        self.node_vram = PredictiveDependencyNode(
            node_id="vram_allocator",
            subsystem="BACKEND_KERNEL",
            epistemic_kind="DERIVED",
            lease_ttl_ns=8_000_000_000,
            value=2048.0,
            domain_id="vram_allocation_mb",
            confidence=0.88,
            valid_from_ns=self.now,
            interval=(10.0, 60.0),
            vector_clock={"vram_allocator": 3},
        )

        self.graph.add_node(self.node_sensor)
        self.graph.add_node(self.node_forecast)
        self.graph.add_node(self.node_ui)
        self.graph.add_node(self.node_vram)

        # Edges: sensor -> forecast -> ui; sensor -> vram
        self.graph.add_edge("hw_sensor", "kernel_forecast", DependencyEdgeKind.PREDICTS)
        self.graph.add_edge("kernel_forecast", "ui_hud_view", DependencyEdgeKind.INVALIDATES)
        self.graph.add_edge("hw_sensor", "vram_allocator", DependencyEdgeKind.PREDICTS)

        self.pipeline = PredictiveDependencyPipeline(graph=self.graph)

    # ─── 1. FETCH DEPENDENCIES TESTS ─────────────────────────────────────────

    def test_01_fetch_dependencies_downstream(self):
        """Tests fetching downstream dependencies from root sensor."""
        bundle = self.pipeline.fetch_dependencies(node_id="hw_sensor", direction="downstream")
        self.assertEqual(bundle.root_node_id, "hw_sensor")
        self.assertEqual(bundle.node_count, 4)
        self.assertIn("hw_sensor", bundle.nodes)
        self.assertIn("kernel_forecast", bundle.nodes)
        self.assertIn("ui_hud_view", bundle.nodes)
        self.assertIn("vram_allocator", bundle.nodes)

        # In downstream direction, sensor must precede forecast and ui in topological order
        order = bundle.topological_order
        self.assertLess(order.index("hw_sensor"), order.index("kernel_forecast"))
        self.assertLess(order.index("kernel_forecast"), order.index("ui_hud_view"))

    def test_02_fetch_dependencies_upstream(self):
        """Tests fetching upstream causal dependencies for UI view."""
        bundle = self.pipeline.fetch_dependencies(node_id="ui_hud_view", direction="upstream")
        self.assertEqual(bundle.node_count, 3)
        self.assertIn("ui_hud_view", bundle.nodes)
        self.assertIn("kernel_forecast", bundle.nodes)
        self.assertIn("hw_sensor", bundle.nodes)
        self.assertNotIn("vram_allocator", bundle.nodes)

    def test_03_fetch_dependencies_cycle_safety(self):
        """Tests that cyclical graphs do not cause infinite loops and terminate gracefully."""
        self.graph.add_edge("ui_hud_view", "hw_sensor", DependencyEdgeKind.SUBSCRIBES)
        bundle = self.pipeline.fetch_dependencies(node_id="hw_sensor", direction="both", max_depth=16)
        self.assertEqual(bundle.node_count, 4)
        self.assertEqual(len(bundle.topological_order), 4)

    def test_04_fetch_dependencies_nonexistent_node(self):
        """Tests KeyError on unknown root node."""
        with self.assertRaises(KeyError):
            self.pipeline.fetch_dependencies(node_id="ghost_node")

    # ─── 2. SYNCHRONIZE TESTS ────────────────────────────────────────────────

    def test_05_synchronize_active_and_vector_clock(self):
        """Tests clock skew validation and vector clock reconciliation."""
        bundle = self.pipeline.fetch_dependencies(node_id="hw_sensor", direction="both")
        report = self.pipeline.synchronize(bundle, now_ns=self.now + 1_000_000_000)

        self.assertTrue(report.is_synchronized)
        self.assertEqual(len(report.expired_nodes), 0)
        self.assertEqual(len(report.active_nodes), 4)
        self.assertEqual(report.vital_max_hp, 6)

        # Consensus clock should contain max of all keys
        consensus = report.vector_clock_consensus
        self.assertEqual(consensus.get("hw_sensor"), 1)
        self.assertEqual(consensus.get("kernel_forecast"), 2)
        self.assertEqual(consensus.get("vram_allocator"), 3)

        # All nodes in reconciled bundle should share consensus clock
        for nid, clock in report.reconciled_bundle.vector_clocks.items():
            self.assertEqual(clock, consensus)

    def test_06_synchronize_lease_expiry_and_drift(self):
        """Tests identification of expired temporal leases."""
        bundle = self.pipeline.fetch_dependencies(node_id="hw_sensor", direction="both")
        # Advance time by 3 seconds: ui_hud_view has lease_ttl 2s, so it must expire
        future_ns = self.now + 3_500_000_000
        report = self.pipeline.synchronize(bundle, now_ns=future_ns, max_drift_ns=500_000_000)

        self.assertFalse(report.is_synchronized)
        self.assertIn("ui_hud_view", report.expired_nodes)
        self.assertIn("hw_sensor", report.active_nodes)
        # Confidence of expired node must be degraded
        self.assertLess(report.reconciled_bundle.confidences["ui_hud_view"], 0.95)

    # ─── 3. OVERLAP TESTS ────────────────────────────────────────────────────

    def test_07_overlap_jaccard_temporal_cosine(self):
        """Tests topological Jaccard similarity, sliding window overlap, and cosine value overlap."""
        bundle_a = self.pipeline.fetch_dependencies(node_id="hw_sensor", direction="downstream")
        bundle_b = self.pipeline.fetch_dependencies(node_id="kernel_forecast", direction="both")

        report = self.pipeline.overlap(bundle_a, bundle_b)
        self.assertEqual(report.vital_max_hp, 6)
        self.assertIn("kernel_forecast", report.common_nodes)
        self.assertIn("ui_hud_view", report.common_nodes)
        self.assertIn("hw_sensor", report.common_nodes)
        self.assertGreater(report.jaccard_similarity, 0.5)

        # Cosine similarity must be in [0.0, 1.0]
        self.assertGreaterEqual(report.cosine_value_similarity, 0.0)
        self.assertLessEqual(report.cosine_value_similarity, 1.0)

        # Temporal sliding window overlap for kernel_forecast [20, 70]
        self.assertIn("kernel_forecast", report.temporal_overlap_ratios)
        self.assertAlmostEqual(report.temporal_overlap_ratios["kernel_forecast"], 1.0, places=2)

    # ─── 4. COMPLEMENT TESTS ─────────────────────────────────────────────────

    def test_08_complement_algebraic_and_set_universe(self):
        """Tests algebraic complement 1 - x and domain universe padding."""
        bundle = self.pipeline.fetch_dependencies(node_id="ui_hud_view", direction="upstream")
        universe = {"hw_sensor", "kernel_forecast", "ui_hud_view", "vram_allocator", "gpu_clock_sentinel"}

        report = self.pipeline.complement(bundle, universe_ids=universe, default_value=1200.0, default_domain="gpu_clock_mhz")
        self.assertEqual(report.vital_max_hp, 6)
        self.assertIn("vram_allocator", report.missing_nodes)
        self.assertIn("gpu_clock_sentinel", report.missing_nodes)

        # Completed bundle must contain the padded missing nodes
        self.assertIn("gpu_clock_sentinel", report.completed_bundle.nodes)
        self.assertEqual(report.completed_bundle.values["gpu_clock_sentinel"], 1200.0)

        # Algebraic complement x values must satisfy x_comp = 1 - x
        for nid, x_comp in report.algebraic_complement_x.items():
            self.assertGreaterEqual(x_comp, 0.0)
            self.assertLessEqual(x_comp, 1.0)

    # ─── 5. ARITHMETIC TESTS: ADD, SUBTRACT, MULTIPLY, DIVIDE ────────────────

    def test_09_arithmetic_add_and_scalar(self):
        """Tests element-wise and scalar addition on dependency bundle."""
        bundle = self.pipeline.fetch_dependencies(node_id="hw_sensor", direction="both")
        orig_val = bundle.values["hw_sensor"]  # 25.0

        # Scalar addition
        added_scalar = self.pipeline.add(bundle, 10.0)
        self.assertEqual(added_scalar.values["hw_sensor"], orig_val + 10.0)

        # Bundle addition
        added_bundle = self.pipeline.add(bundle, bundle)
        self.assertEqual(added_bundle.values["hw_sensor"], orig_val * 2.0)

    def test_10_arithmetic_subtract_and_residual_l2(self):
        """Tests element-wise subtraction and residual error calculation."""
        bundle_a = self.pipeline.fetch_dependencies(node_id="hw_sensor", direction="both")
        bundle_b = self.pipeline.fetch_dependencies(node_id="hw_sensor", direction="both")
        bundle_b.set_value("hw_sensor", 28.0)  # delta = 3.0

        res_bundle, l2_err = self.pipeline.subtract(bundle_a, bundle_b)
        self.assertAlmostEqual(res_bundle.values["hw_sensor"], -3.0, places=4)
        self.assertAlmostEqual(l2_err, 3.0, places=4)

    def test_11_arithmetic_multiply_and_confidence(self):
        """Tests element-wise multiplication and probabilistic confidence scaling."""
        bundle = self.pipeline.fetch_dependencies(node_id="hw_sensor", direction="both")
        orig_conf = bundle.confidences["hw_sensor"]  # 0.98

        scaled = self.pipeline.multiply(bundle, 1.5)
        self.assertAlmostEqual(scaled.values["hw_sensor"], 25.0 * 1.5, places=4)

        # Multiply by another bundle decays confidence: conf_C = conf_A * conf_B
        bundle_mult = self.pipeline.multiply(bundle, bundle)
        expected_conf = round(orig_conf * orig_conf, 4)
        self.assertAlmostEqual(bundle_mult.confidences["hw_sensor"], expected_conf, places=4)

    def test_12_arithmetic_divide_and_epsilon_zero_guard(self):
        """Tests division with strict epsilon protection against zero division."""
        bundle = self.pipeline.fetch_dependencies(node_id="hw_sensor", direction="both")

        # Standard division
        div_half = self.pipeline.divide(bundle, 2.0)
        self.assertAlmostEqual(div_half.values["hw_sensor"], 12.5, places=4)

        # Dangerous division by zero: must NOT crash, must remain finite
        div_zero = self.pipeline.divide(bundle, 0.0)
        self.assertTrue(math.isfinite(div_zero.values["hw_sensor"]))
        self.assertGreater(abs(div_zero.values["hw_sensor"]), 1e10)

        # Bundle division where a divisor value is zero
        bundle_zero = bundle.clone()
        bundle_zero.set_value("hw_sensor", 0.0)
        div_bundle = self.pipeline.divide(bundle, bundle_zero)
        self.assertTrue(math.isfinite(div_bundle.values["hw_sensor"]))

    # ─── 6. CALCULATE VIA X TESTS (QUADRATIC VARIABLE MANIFOLD) ───────────────

    def test_13_calculate_via_x_vital_max_hp_invariant(self):
        """Tests cross-domain projection through latent x into vital_hp strictly enforcing 6 HP."""
        bundle = self.pipeline.fetch_dependencies(node_id="hw_sensor", direction="both")
        res = self.pipeline.calculate_via_x(
            bundle,
            target_domain="vital_hp",
            latent_operation="identity",
        )
        self.assertEqual(res["vital_max_hp_rule"], 6)
        self.assertTrue(res["constitutional_compliance"])

        # Check each node projection
        for nid, node_info in res["transformed_nodes"].items():
            self.assertEqual(node_info["target_domain"], "vital_hp")
            hp_val = node_info["target_value"]
            self.assertGreaterEqual(hp_val, 0.0)
            self.assertLessEqual(hp_val, 6.0)
            self.assertTrue(node_info["vital_hp_bound"])

    def test_14_calculate_via_x_latent_operations(self):
        """Tests latent manifold operations: shift, complement, scale, harmonic mean."""
        bundle = self.pipeline.fetch_dependencies(node_id="hw_sensor", direction="both")

        # 1. Shift
        res_shift = self.pipeline.calculate_via_x(bundle, target_domain="vital_hp", delta_x=0.2, latent_operation="shift")
        for nid, info in res_shift["transformed_nodes"].items():
            self.assertGreaterEqual(info["latent_x_transformed"], info["latent_x_solved"] - 1e-4)

        # 2. Complement: x_trans == 1.0 - x_solved
        res_comp = self.pipeline.calculate_via_x(bundle, target_domain="vital_hp", latent_operation="complement")
        for nid, info in res_comp["transformed_nodes"].items():
            self.assertAlmostEqual(info["latent_x_transformed"], 1.0 - info["latent_x_solved"], places=4)

        # 3. Scale: x_trans == x_solved * alpha
        res_scale = self.pipeline.calculate_via_x(bundle, target_domain="vital_hp", latent_operation="scale", alpha=0.5)
        for nid, info in res_scale["transformed_nodes"].items():
            self.assertAlmostEqual(info["latent_x_transformed"], min(1.0, info["latent_x_solved"] * 0.5), places=4)

        # 4. Harmonic Mean
        res_harmonic = self.pipeline.calculate_via_x(bundle, target_domain="vital_hp", latent_operation="harmonic")
        self.assertEqual(res_harmonic["vital_max_hp_rule"], 6)
        self.assertGreaterEqual(res_harmonic["aggregate"]["mean"], 0.0)
        self.assertLessEqual(res_harmonic["aggregate"]["max"], 6.0)

    def test_15_graph_fetch_dependencies_convenience_method(self):
        """Tests that PredictiveDependencyGraph.fetch_dependencies delegates correctly."""
        bundle = self.graph.fetch_dependencies(node_id="hw_sensor", direction="both")
        self.assertIsInstance(bundle, DependencyBundle)
        self.assertEqual(bundle.root_node_id, "hw_sensor")
        self.assertEqual(bundle.node_count, 4)


if __name__ == "__main__":
    unittest.main()
