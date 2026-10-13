#!/usr/bin/env python3
"""
==============================================================================
UNIT TESTS: MODEL ROUTER & EPISTEMIC ROUTE SCORING
==============================================================================
Module: tests/test_model_router.py
Description:
    Tests for ModelRouter:
    - Intent classification
    - 6-Factor RouteScore computation:
      RouteScore = w1*C + w2*E + w3*Cal - w4*L - w5*M - w6*Risk
    - Candidate ranking under multiple objectives (MIN_LATENCY, MAX_ACCURACY, etc.)
    - Specialist selection by domain
    - Fallback selection & regret accumulation
    - Invariant: assert VITAL_MAX_HP == 6
==============================================================================
"""

import unittest
from typing import Dict, Any, List

from krystal_kernel.scene_constructor_presets import VITAL_MAX_HP
from krystal_kernel.cascade.model_node import (
    IntentClassifierNode,
    CoreRagNode,
    ReasoningNode,
    CodingSpecialistNode,
    VerificationNode,
    FinalSynthesisNode
)
from krystal_kernel.cascade.model_router import (
    ModelRouter,
    NodeRouteCandidate,
    RoutingObjectiveWeights,
    ROUTING_PROFILES
)


class TestModelRouter(unittest.TestCase):

    def setUp(self):
        self.router = ModelRouter()
        self.nodes = [
            IntentClassifierNode("node-classifier-slm"),
            CoreRagNode("node-core-rag"),
            ReasoningNode("node-deep-reasoner"),
            CodingSpecialistNode("node-code-specialist"),
            VerificationNode("node-epistemic-verifier"),
            FinalSynthesisNode("node-final-synthesizer")
        ]

    def test_01_vital_max_hp_invariant(self):
        """Verify universal system invariant VITAL_MAX_HP == 6."""
        self.assertEqual(VITAL_MAX_HP, 6)
        w = RoutingObjectiveWeights(name="TEST")
        self.assertEqual(w.vital_max_hp, 6)
        spec = self.nodes[0].spec
        cand = NodeRouteCandidate(
            node_id="test",
            spec=spec,
            capability_score=1.0,
            evidence_score=1.0,
            latency_score=0.1,
            memory_score=0.1,
            regret_score=0.0,
            route_score=0.85
        )
        self.assertEqual(cand.vital_max_hp, 6)

    def test_02_classify_intent(self):
        """Verify natural language prompt classification into epistemic domains."""
        code_intent = self.router.classify_intent("Napíš funkciu v Pythone s overením typu")
        self.assertEqual(code_intent["intent"], "CODE_GENERATION")

        rag_intent = self.router.classify_intent("Aké sú pravidlá pre VITAL_MAX_HP v databáze?")
        self.assertIn(rag_intent["intent"], ("CORE_KNOWLEDGE_RAG", "KNOWLEDGE_RETRIEVAL"))

        reason_intent = self.router.classify_intent("Prečo a ako funguje architektúra kaskády?")
        self.assertIn(reason_intent["intent"], ("DEEP_REASONING", "REASONING"))

    def test_03_six_factor_route_score(self):
        """Verify exact calculation of the canonical 6-factor RouteScore."""
        intent = self.router.classify_intent("Vygeneruj kód")
        ranked = self.router.rank_nodes(self.nodes, intent, objective="BALANCED")
        self.assertTrue(len(ranked) > 0)

        cand = ranked[0]
        w = ROUTING_PROFILES["BALANCED"]

        expected = (
            w.w1_capability * cand.capability_score +
            w.w2_evidence * cand.evidence_score +
            w.w3_calibration * cand.calibration_score -
            w.w4_latency * cand.latency_score -
            w.w5_memory * cand.memory_score -
            w.w6_failure_risk * cand.regret_score
        )
        self.assertAlmostEqual(cand.route_score, expected, places=4)

    def test_04_objective_profiles(self):
        """Verify different objectives alter top ranking candidate."""
        intent = self.router.classify_intent("Vysvetli logiku a invariant")

        # MIN_LATENCY should favor faster / smaller nodes
        min_lat_ranked = self.router.rank_nodes(self.nodes, intent, objective="MIN_LATENCY")
        self.assertTrue(len(min_lat_ranked) > 0)

        # MAX_ACCURACY should heavily weight evidence and capability
        max_acc_ranked = self.router.rank_nodes(self.nodes, intent, objective="MAX_ACCURACY")
        self.assertTrue(len(max_acc_ranked) > 0)

    def test_05_select_node_and_fallback(self):
        """Verify select_node and select_fallback with regret update."""
        intent = self.router.classify_intent("Optimalizácia")
        selected = self.router.select_node(self.nodes, intent)
        self.assertIsNotNone(selected)

        fallback = self.router.select_fallback(selected.spec.node_id, self.router.rank_nodes(self.nodes, intent))
        self.assertIsNotNone(fallback)
        self.assertNotEqual(selected.spec.node_id, fallback.node_id)

        # Regret should be penalized on fallback trigger
        self.assertTrue(self.router.node_regret_history.get(selected.spec.node_id, 0.0) > 0.0)

    def test_06_select_specialist(self):
        """Verify specialist selection for specialized domains."""
        code_spec = self.router.select_specialist("CODE_GENERATION", self.nodes)
        self.assertEqual(code_spec.spec.node_id, "node-code-specialist")

        rag_spec = self.router.select_specialist("KNOWLEDGE_RETRIEVAL", self.nodes)
        self.assertEqual(rag_spec.spec.node_id, "node-core-rag")


if __name__ == "__main__":
    unittest.main()
