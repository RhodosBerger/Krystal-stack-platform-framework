#!/usr/bin/env python3
"""
==============================================================================
KRYSTAL-STACK: UNIT TESTS FOR SOVEREIGN INTROSPECTIVE SEARCH ENGINE
==============================================================================
Module: tests/test_sovereign_introspective_search_engine.py
Description:
    Validates:
      1. 10 Industry Verticals evaluation and distinct cyber-physical models.
      2. Self-Aware Internal Process Introspection (Cognitive Load, Entropy, Memory).
      3. Self-Error Detection, Residual Auditing & Missing Step Auto-Discovery.
      4. Tactical Divergence Mutation (Different tactics across consecutive cycles).
      5. Sovereign Game Bot Core Integration.
      6. Theory DAG Predicted Dependency Nodes.
      7. Constitutional Invariant VITAL_MAX_HP == 6.

Author: Dušan Kopecký & Krystal-Stack Architecture Council (2026)
==============================================================================
"""

import unittest
from krystal_kernel.constants import VITAL_MAX_HP, ContractViolation
from krystal_kernel.sovereign_introspective_search_engine import (
    IndustryVertical,
    OperatingTactic,
    IntrospectionProcessState,
    ProcessMetrics,
    VerticalAnalysisResult,
    SelfErrorAudit,
    IntrospectiveSearchCycleReceipt,
    SovereignIntrospectiveSearchEngine,
    GLOBAL_INTROSPECTIVE_SEARCH_ENGINE
)


class TestSovereignIntrospectiveSearchEngine(unittest.TestCase):
    """Test suite validating the self-aware introspective search engine."""

    def setUp(self) -> None:
        self.engine = SovereignIntrospectiveSearchEngine()

    def test_ten_industry_verticals_coverage(self) -> None:
        """Verifies all 10 canonical industry verticals are evaluated with valid models."""
        results = self.engine.evaluate_ten_verticals({})
        self.assertEqual(len(results), 10)

        observed_verticals = {res.vertical for res in results}
        expected_verticals = set(IndustryVertical)
        self.assertEqual(observed_verticals, expected_verticals)

        for res in results:
            self.assertGreater(len(res.governing_equation), 10)
            self.assertGreater(res.confidence_score, 0.85)
            self.assertGreater(len(res.status), 0)

    def test_internal_process_introspection(self) -> None:
        """Verifies metacognitive introspection of internal engine processes."""
        import time
        t0 = time.perf_counter_ns()
        metrics = self.engine.introspect_internal_processes(t0, depth=5, branches=12)

        self.assertIsInstance(metrics, ProcessMetrics)
        self.assertGreater(metrics.cognitive_load_pct, 0.0)
        self.assertLessEqual(metrics.cognitive_load_pct, 100.0)
        self.assertEqual(metrics.search_depth, 5)
        self.assertEqual(metrics.branches_explored, 12)
        self.assertGreater(metrics.allocated_memory_kb, 100.0)
        self.assertGreaterEqual(metrics.internal_entropy, 0.0)

    def test_self_error_detection_and_missing_steps(self) -> None:
        """Verifies self-error residual norm computation and missing prerequisite auto-discovery."""
        # 1. Nominal case: predictions match observations closely
        pred_nom = [1.0, 2.0, 3.0]
        obs_nom = [1.02, 1.98, 3.01]
        audit_nom = self.engine.detect_self_errors(pred_nom, obs_nom)

        self.assertFalse(audit_nom.is_anomaly_detected)
        self.assertLess(audit_nom.residual_norm, 0.15)
        self.assertEqual(len(audit_nom.auto_discovered_missing_steps), 0)
        self.assertEqual(audit_nom.remediation_action, "PRESERVE_ACTIVE_TRAJECTORY")

        # 2. Anomaly case: large residual discrepancy
        pred_anom = [1.0, 2.0, 3.0]
        obs_anom = [1.35, 2.45, 3.60]
        audit_anom = self.engine.detect_self_errors(pred_anom, obs_anom)

        self.assertTrue(audit_anom.is_anomaly_detected)
        self.assertGreater(audit_anom.residual_norm, 0.15)
        self.assertGreater(len(audit_anom.auto_discovered_missing_steps), 0)
        self.assertIn("INJECT_THERMAL_STRAIN_TENSOR_RECALIBRATION", audit_anom.auto_discovered_missing_steps)
        self.assertEqual(audit_anom.remediation_action, "MUTATE_TACTIC_TO_CONSERVATIVE_EVASION")

    def test_tactical_mutation_across_cycles(self) -> None:
        """Verifies that an anomaly or constraint mutates operating tactics for the subsequent cycle."""
        pred_anom = [1.0, 2.0]
        obs_anom = [1.8, 2.9]
        error_audit = self.engine.detect_self_errors(pred_anom, obs_anom)
        metrics = ProcessMetrics(50.0, 4, 8, 2, 256.0, 1.2, 1.35)

        # Starting from Apex Exploitation, an anomaly forces Evasion
        next_tactic = self.engine.mutate_tactic_for_next_cycle(
            OperatingTactic.GREEDY_APEX_EXPLOITATION,
            error_audit,
            metrics
        )
        self.assertEqual(next_tactic, OperatingTactic.HOMOTOPIC_THERMAL_EVASION)

    def test_full_cycle_execution_receipt(self) -> None:
        """Verifies full execution of introspective cycle returning a valid self-service receipt."""
        req = {
            "search_depth": 4,
            "branches_count": 10,
            "predicted_state_vector": [0.5, 0.5],
            "observed_state_vector": [0.51, 0.52],
            "game_bot_input": {
                "current_hp": 6,
                "active_hex_coord": 42
            }
        }
        receipt = self.engine.execute_introspective_cycle(req)

        self.assertIsInstance(receipt, IntrospectiveSearchCycleReceipt)
        self.assertEqual(receipt.vital_max_hp, 6)
        self.assertIn("[VIT_MAX_HP==6]", receipt.conduct_flags)
        self.assertEqual(len(receipt.verticals_evaluated), 10)
        self.assertEqual(receipt.game_bot_recommendation["bot_health_status"], "VITAL_MAX_HP_HELD")
        self.assertEqual(receipt.game_bot_recommendation["active_hex_subspace"], 42)
        self.assertGreater(len(receipt.predicted_dag_nodes), 0)
        self.assertGreater(len(receipt.hmac_signature), 32)

    def test_constitutional_invariant_rejection(self) -> None:
        """Verifies engine rejects initialization with vital_max_hp != 6."""
        with self.assertRaises(ContractViolation):
            SovereignIntrospectiveSearchEngine(vital_max_hp=5)


if __name__ == "__main__":
    unittest.main()
