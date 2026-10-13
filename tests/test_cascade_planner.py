#!/usr/bin/env python3
"""
==============================================================================
UNIT TESTS: CASCADE PLANNER & GRAPH COMPILATION
==============================================================================
Module: tests/test_cascade_planner.py
Description:
    Tests for CascadePlanner:
    - CascadeEdge (condition, transform, timeout, evidence_requirement, fallback)
    - CascadePlan (6 canonical objectives, state version, budget bounds)
    - Plan compilation across 8 execution modes
    - Plan validation (connectivity, cycle detection, invariant checks)
    - Cost, latency, and memory estimation
    - Dynamic fallback path generation (generate_fallback_path)
    - Invariant: assert VITAL_MAX_HP == 6
==============================================================================
"""

import unittest
from typing import Dict, Any

from krystal_kernel.scene_constructor_presets import VITAL_MAX_HP
from krystal_kernel.cascade.model_node import (
    IntentClassifierNode,
    CoreRagNode,
    ReasoningNode,
    CodingSpecialistNode,
    VerificationNode,
    FinalSynthesisNode
)
from krystal_kernel.cascade.model_router import ModelRouter
from krystal_kernel.cascade.cascade_planner import CascadeEdge, CascadePlan, CascadePlanner


class TestCascadePlanner(unittest.TestCase):

    def setUp(self):
        self.router = ModelRouter()
        self.planner = CascadePlanner(router=self.router)
        self.nodes = {
            "node-classifier-slm": IntentClassifierNode("node-classifier-slm"),
            "node-core-rag": CoreRagNode("node-core-rag"),
            "node-deep-reasoner": ReasoningNode("node-deep-reasoner"),
            "node-code-specialist": CodingSpecialistNode("node-code-specialist"),
            "node-epistemic-verifier": VerificationNode("node-epistemic-verifier"),
            "node-final-synthesizer": FinalSynthesisNode("node-final-synthesizer")
        }

    def test_01_vital_max_hp_invariant(self):
        """Universal platform invariant check across planner structures."""
        self.assertEqual(VITAL_MAX_HP, 6)
        edge = CascadeEdge(
            source="n1",
            target="n2",
            condition="CONFIDENCE > 0.8",
            transform="EXTRACT_CODE",
            timeout=2.5,
            evidence_requirement=["DocRef"]
        )
        self.assertEqual(edge.vital_max_hp, 6)
        self.assertEqual(edge.condition, "CONFIDENCE > 0.8")
        self.assertEqual(edge.transform, "EXTRACT_CODE")

    def test_02_build_canonical_rag_grounded_plan(self):
        """Verify compilation of the canonical vertical slice: RAG_GROUNDED."""
        req = {
            "prompt": "Vysvetli mechanizmus pamäte",
            "strategy": "CASCADE_RAG_GROUNDED",
            "objective": "MAX_GROUNDING"
        }
        plan = self.planner.build_plan(req, self.nodes)
        self.assertEqual(plan.strategy, "CASCADE_RAG_GROUNDED")
        self.assertEqual(plan.objective, "MAX_GROUNDING")
        self.assertEqual(plan.vital_max_hp, 6)
        self.assertTrue(len(plan.nodes) >= 3)
        self.assertTrue(len(plan.edges) >= 2)

        valid, errs = self.planner.validate_plan(plan)
        self.assertTrue(valid, f"Plan validation errors: {errs}")

    def test_03_all_six_objectives(self):
        """Verify planner builds plans under all 6 canonical objectives."""
        objectives = [
            "MIN_LATENCY", "MAX_ACCURACY", "MIN_MEMORY",
            "MAX_GROUNDING", "BALANCED", "EXPERIMENTAL"
        ]
        for obj in objectives:
            req = {"prompt": "Test objective", "objective": obj}
            plan = self.planner.build_plan(req, self.nodes)
            self.assertEqual(plan.objective, obj)
            self.assertTrue(plan.expected_latency_ms > 0.0)
            self.assertTrue(plan.expected_memory_mb > 0.0)

    def test_04_plan_cost_and_latency_estimation(self):
        """Verify estimation functions calculate non-zero projections."""
        req = {"prompt": "Generuj komplexný kód", "strategy": "CASCADE_SEQUENTIAL"}
        plan = self.planner.build_plan(req, self.nodes)

        lat = self.planner.estimate_latency(plan, self.nodes)
        mem = self.planner.estimate_memory(plan, self.nodes)
        cost = self.planner.estimate_cost(plan, self.nodes)

        self.assertTrue(lat > 0.0)
        self.assertTrue(mem > 0.0)
        self.assertTrue(cost["estimated_cost"] > 0.0)

    def test_05_generate_fallback_path(self):
        """Verify dynamic detour plan generation when a node fails."""
        req = {"prompt": "Dotaz", "strategy": "CASCADE_SEQUENTIAL"}
        plan = self.planner.build_plan(req, self.nodes)

        fallback_plan = self.planner.generate_fallback_path(plan, "node-deep-reasoner")
        self.assertIsNotNone(fallback_plan)
        self.assertNotIn("node-deep-reasoner", [n.node_id for n in fallback_plan.nodes])
        self.assertIn("FALLBACK", fallback_plan.plan_id)
        self.assertEqual(fallback_plan.vital_max_hp, 6)

    def test_06_plan_validation_catches_invalid_edge(self):
        """Verify validator flags disconnected or invalid edges."""
        req = {"prompt": "Dotaz"}
        plan = self.planner.build_plan(req, self.nodes)
        # Add phantom node edge
        plan.edges.append(CascadeEdge(source="phantom-node-xyz", target="node-final-synthesizer"))

        valid, errs = self.planner.validate_plan(plan)
        self.assertFalse(valid)
        self.assertTrue(any("phantom-node-xyz" in e for e in errs))


if __name__ == "__main__":
    unittest.main()
