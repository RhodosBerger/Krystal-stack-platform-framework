#!/usr/bin/env python3
"""
==============================================================================
KRYSTAL-STACK: UNIT TEST SUITE FOR TECTONIC MATRIX ADAPTABILITY & SPECULATION
==============================================================================
File: tests/test_tectonic_matrix_adaptability.py
Tests:
  - TectonicMatrixTensor 8x8 grid topology, elevations, and polarities
  - Zero-Cost Copy-on-Write (CoW) shadow cloning isolation
  - IF-THEN primary condition triggers and baseline proposals
  - Nested WHAT-IF 1st and 2nd order counterfactual shock projections
  - THAN-IF multi-criteria comparative arbitration override
  - ELSE-IF cascading fallback branches
  - IF-GUARD invariant verification (VITAL_MAX_HP == 6)
  - Sub-millisecond execution latency benchmark (< 600 us)
==============================================================================
"""

import unittest
import time
from krystal_kernel.tectonic_matrix_adaptability_engine import (
    VITAL_MAX_HP,
    ConditionOp,
    ConditionClause,
    TectonicCell,
    TectonicMatrixTensor,
    IfThenRule,
    WhatIfSpeculationNode,
    ThanIfComparator,
    ElseIfBranch,
    TectonicMatrixAdaptabilityEngine,
    GLOBAL_TECTONIC_ADAPTABILITY_ENGINE
)


class TestTectonicMatrixAdaptability(unittest.TestCase):
    """Test suite covering the Tectonic Matrix Adaptability Engine and Speculative Rules."""

    def setUp(self):
        self.engine = TectonicMatrixAdaptabilityEngine(width=8, height=8)

    def test_01_matrix_initialization_and_topology(self):
        """Verifies 8x8 grid contains 64 cells with valid physical properties."""
        matrix = self.engine.live_matrix
        self.assertEqual(len(matrix.cells), 64)

        center_cell = matrix.cells[(4, 4)]
        self.assertGreater(center_cell.elevation_z, 8.0)
        self.assertEqual(center_cell.vital_hp, VITAL_MAX_HP)
        self.assertIn("crystal", center_cell.polarities)

        metrics = matrix.compute_metrics()
        self.assertIn("stability_index", metrics)
        self.assertIn("mean_stress", metrics)
        self.assertIn("mean_temp_c", metrics)
        self.assertTrue(0.0 <= metrics["stability_index"] <= 1.0)

    def test_02_shadow_cloning_isolation(self):
        """Verifies shadow matrix can be mutated without altering live matrix state."""
        live_metrics_before = self.engine.live_matrix.compute_metrics()

        shadow = self.engine.live_matrix.clone_shadow()
        shadow.apply_shock("SEISMIC_SHEAR", magnitude=0.8, center=(4, 4))
        shadow_metrics = shadow.compute_metrics()

        live_metrics_after = self.engine.live_matrix.compute_metrics()

        # Shadow should show increased stress and decreased stability
        self.assertGreater(shadow_metrics["mean_stress"], live_metrics_before["mean_stress"])
        # Live matrix must be completely unchanged
        self.assertEqual(live_metrics_before["mean_stress"], live_metrics_after["mean_stress"])
        self.assertEqual(live_metrics_before["stability_index"], live_metrics_after["stability_index"])

    def test_03_if_then_rule_trigger(self):
        """Tests that when live stress > 0.40, IF-THEN triggers baseline proposal."""
        # Inject shear stress into live matrix
        self.engine.live_matrix.apply_shock("SEISMIC_SHEAR", magnitude=0.5, center=(4, 4))
        metrics = self.engine.live_matrix.compute_metrics()
        self.assertGreater(metrics["mean_stress"], 0.40)

        plan = self.engine.speculate()
        self.assertEqual(plan.baseline_action, "BASELINE_SLUDGE_QUENCH")
        self.assertIn("BASELINE_SLUDGE_QUENCH", [t.action_proposed for t in plan.trace if t.node_type == "IF_THEN"])

    def test_04_nested_what_if_and_than_if_override(self):
        """Verifies nested WHAT-IF shocks trigger and THAN-IF overrides baseline when superior."""
        # Elevate stress so IF-THEN triggers
        self.engine.live_matrix.apply_shock("SEISMIC_SHEAR", magnitude=0.5, center=(4, 4))

        plan = self.engine.speculate()

        # Check trace for WHAT_IF_1 and WHAT_IF_2
        trace_types = [t.node_type for t in plan.trace]
        self.assertIn("WHAT_IF_1", trace_types)
        self.assertIn("WHAT_IF_2", trace_types)
        self.assertIn("THAN_IF", trace_types)

        # In default rule set, amber catalyzed reinforcement grants high stability, triggering THAN_IF override
        self.assertTrue(plan.is_speculative_override)
        self.assertEqual(plan.chosen_action, "COMMIT_AMBER_CATALYZED_REINFORCE")
        self.assertGreater(plan.projected_metrics["stability_index"], 0.5)

    def test_05_else_if_fallback_branch(self):
        """Verifies ELSE-IF triggers when primary IF condition is false but secondary is true."""
        # Live stress <= 0.40 (IF fails), but thermal temp > 65.0 (ELSE-IF passes)
        for cell in self.engine.live_matrix.cells.values():
            cell.shear_stress = 0.20  # Keep stress low
            cell.temperature_c = 70.0  # Elevate temperature

        plan = self.engine.speculate()
        self.assertIsNone(plan.baseline_action)
        self.assertFalse(plan.is_speculative_override)
        self.assertEqual(plan.chosen_action, "EMERGENCY_COOLING_SLAG")

    def test_06_if_guard_invariant_vital_hp_6(self):
        """Verifies IF-GUARD asserts and clamps all cells to VITAL_MAX_HP == 6."""
        # Forcibly violate invariant
        self.engine.live_matrix.cells[(0, 0)].vital_hp = 999

        plan = self.engine.speculate()
        self.assertEqual(plan.vital_max_hp, VITAL_MAX_HP)
        self.assertEqual(self.engine.live_matrix.cells[(0, 0)].vital_hp, VITAL_MAX_HP)
        self.assertIn("IF_GUARD", [t.node_type for t in plan.trace])

    def test_07_adapt_commit_execution(self):
        """Verifies adapt() commits changes to the live matrix."""
        # Elevate stress
        self.engine.live_matrix.apply_shock("SEISMIC_SHEAR", magnitude=0.5, center=(4, 4))
        metrics_before = self.engine.live_matrix.compute_metrics()

        res = self.engine.adapt()
        self.assertEqual(res["status"], "ADAPTED")
        self.assertIn("committed_action", res)

        metrics_after = self.engine.live_matrix.compute_metrics()
        self.assertNotEqual(metrics_before, metrics_after)

    def test_08_sub_millisecond_latency_benchmark(self):
        """Verifies speculative decision pipeline executes in <= 600 us."""
        latencies = []
        for _ in range(50):
            plan = self.engine.speculate()
            latencies.append(plan.execution_latency_us)

        mean_latency = sum(latencies) / len(latencies)
        self.assertLess(mean_latency, 600.0)  # Must be well within frame budget

    def test_09_global_singleton(self):
        """Verifies GLOBAL_TECTONIC_ADAPTABILITY_ENGINE is available."""
        self.assertIsNotNone(GLOBAL_TECTONIC_ADAPTABILITY_ENGINE)
        metrics = GLOBAL_TECTONIC_ADAPTABILITY_ENGINE.live_matrix.compute_metrics()
        self.assertIn("stability_index", metrics)


if __name__ == "__main__":
    unittest.main()
