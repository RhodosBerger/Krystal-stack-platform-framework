"""
Unit Test Suite for Unified Transition Envelope Architecture (Milestone M2)
and Cross-Domain Quad-Pole Adapters (CNC, Silicon, Sovereign, Farm).

Verification of Theory Node 50:
THEORY-UNIFIED-TRANSITION-ENVELOPE-CROSS-DOMAIN-ADAPTERS
Constitutional Invariant: VITAL_MAX_HP == 6
"""

import unittest
from typing import Dict, List, Any

from krystal_kernel.transition.models import (
    DomainType,
    TransitionVerdict,
    DomainStateVector,
    ExecutionCandidate,
    ConstraintSet,
    KrystalTransitionEnvelope,
)
from krystal_kernel.transition.adapters import (
    CncAdapter,
    SiliconAdapter,
    SovereignAdapter,
    FarmAdapter,
)
from krystal_kernel.transition.runtime_core import (
    KrystalRuntimeCore,
    GLOBAL_KRYSTAL_RUNTIME_CORE,
)
from krystal_kernel.verification.theory_dependency_graph import GLOBAL_THEORY_DEPENDENCY_GRAPH


class TestUnifiedTransitionAdapters(unittest.TestCase):
    """Test suite for Milestone M2 Unified Transition Proof."""

    def setUp(self):
        self.core = KrystalRuntimeCore()

    def test_theory_node_50_registration(self):
        """Verify that Theory Node 50 is formally registered without cycles."""
        theory_id = "THEORY-UNIFIED-TRANSITION-ENVELOPE-CROSS-DOMAIN-ADAPTERS"
        self.assertIn(theory_id, GLOBAL_THEORY_DEPENDENCY_GRAPH._nodes)
        node = GLOBAL_THEORY_DEPENDENCY_GRAPH._nodes[theory_id]
        self.assertEqual(node.epistemic_status.name, "DOKAZANE")
        self.assertFalse(GLOBAL_THEORY_DEPENDENCY_GRAPH.has_cycles())

    def test_domain_adapters_candidate_generation(self):
        """Test that all 4 domain adapters generate >= 3 valid execution candidates."""
        adapters = [CncAdapter(), SiliconAdapter(), SovereignAdapter(), FarmAdapter()]
        for adapter in adapters:
            raw = adapter.observe()
            norm = adapter.normalize(raw)
            candidates = adapter.generate_candidates(norm)
            self.assertGreaterEqual(
                len(candidates), 3,
                f"Domain {adapter.domain_name} generated fewer than 3 candidates: {len(candidates)}"
            )
            for c in candidates:
                self.assertTrue(c.candidate_id)
                self.assertTrue(c.action_name)
                self.assertGreater(c.expected_utility, 0.0)

    def test_cross_domain_transition_execution(self):
        """Test that all 4 domains execute and validate through KrystalTransitionEnvelope."""
        domains = ["CNC", "SILICON", "SOVEREIGN", "FARM"]
        for domain in domains:
            envelope = self.core.execute_unified_transition(domain)
            self.assertIsInstance(envelope, KrystalTransitionEnvelope)
            self.assertEqual(envelope.verdict, TransitionVerdict.APPROVED_COMMITTED.value)
            self.assertIsNotNone(envelope.receipt_hash)
            self.assertIsNotNone(envelope.prediction_residual)
            self.assertIsNotNone(envelope.utility_gain)
            self.assertIsNotNone(envelope.state_after)
            self.assertEqual(envelope.vital_max_hp, 6)

    def test_cross_domain_platform_metrics(self):
        """Test calculation of CDTV, SRR, PTG, and VUG metrics."""
        # Execute transitions across all domains to establish multi-domain sample set
        for domain in ["CNC", "SILICON", "SOVEREIGN", "FARM"]:
            self.core.execute_unified_transition(domain)

        metrics = self.core.compute_cross_domain_metrics()
        self.assertEqual(metrics["CDTV"], 1.0) # 100% of transitions used shared core
        self.assertGreaterEqual(metrics["SRR"], 0.80)
        self.assertGreater(metrics["VUG"], 0.0)
        self.assertEqual(metrics["domains_covered"], 4)
        self.assertEqual(metrics["vital_max_hp"], 6)

    def test_envelope_rejection_and_isolation(self):
        """Test that constitutional constraints cannot be violated."""
        # Attempt to create an envelope with vital_max_hp != 6
        with self.assertRaises(Exception):
            KrystalTransitionEnvelope(
                runtime_id="BREACH-ID",
                domain="CNC",
                state_before=None, # type: ignore
                candidate=None,    # type: ignore
                predicted_state=None, # type: ignore
                constraints=None,  # type: ignore
                vital_max_hp=5,    # Contract breach
            )


if __name__ == "__main__":
    unittest.main()
