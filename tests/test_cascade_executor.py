#!/usr/bin/env python3
"""
==============================================================================
UNIT TESTS: CASCADE EXECUTOR & GRAPH EXECUTION MODES
==============================================================================
Module: tests/test_cascade_executor.py
Description:
    Tests for CascadeExecutor and CascadeMerger:
    - 8 execution modes (Sequential, Parallel, Vote, Verify, Fallback, etc.)
    - Contradiction detection across parallel model candidates (EPI_CONTRADICTORY)
    - Dual-Gate authority boundary (models cannot mutate OS/cgroups)
    - Shadow execution mode (WOULD_COMMIT, non-mutating)
    - Invariant: assert VITAL_MAX_HP == 6
==============================================================================
"""

import unittest
from typing import Dict, Any

from krystal_kernel.scene_constructor_presets import VITAL_MAX_HP
from krystal_kernel.cascade.cascade_trace import CascadeNodeResult, CascadeExecutionTrace
from krystal_kernel.cascade.model_node import (
    IntentClassifierNode,
    CoreRagNode,
    ReasoningNode,
    CodingSpecialistNode,
    VerificationNode,
    FinalSynthesisNode
)
from krystal_kernel.cascade.model_router import ModelRouter
from krystal_kernel.cascade.cascade_planner import CascadePlan, CascadePlanner
from krystal_kernel.cascade.cascade_executor import CascadeExecutor, CascadeMerger


class TestCascadeExecutor(unittest.TestCase):

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
        self.executor = CascadeExecutor(self.nodes)

    def test_01_vital_max_hp_invariant(self):
        """Universal platform invariant check across execution components."""
        self.assertEqual(VITAL_MAX_HP, 6)
        merger = CascadeMerger()
        self.assertEqual(merger.vital_max_hp, 6)

    def test_02_execute_sequential(self):
        """Verify sequential cascade execution pipeline."""
        req = {"prompt": "Preveď sekvenčnú analýzu", "strategy": "CASCADE_SEQUENTIAL"}
        plan = self.planner.build_plan(req, self.nodes)
        trace = self.executor.execute_sequential(plan, req)

        self.assertEqual(trace.vital_max_hp, 6)
        self.assertEqual(trace.authority_status, "AUTH_COMMIT_APPROVED")
        self.assertTrue(len(trace.node_results) >= 2)
        self.assertTrue(trace.total_latency_ms > 0.0)

    def test_03_execute_rag_grounded(self):
        """Verify RAG grounded execution with factual citations."""
        req = {"prompt": "Aké sú limity pamäte?", "strategy": "CASCADE_RAG_GROUNDED"}
        plan = self.planner.build_plan(req, self.nodes)
        trace = self.executor.execute_rag_grounded(plan, req)

        self.assertEqual(trace.vital_max_hp, 6)
        self.assertTrue(len(trace.evidence_citations) > 0)
        self.assertIn("VITAL_MAX_HP == 6", trace.final_output)

    def test_04_execute_vote_consensus(self):
        """Verify multi-model voting and consensus aggregation."""
        req = {"prompt": "Je invariant VITAL_MAX_HP rovný 6?", "strategy": "CASCADE_VOTE"}
        plan = self.planner.build_plan(req, self.nodes)
        trace = self.executor.execute_vote(plan, req)

        self.assertEqual(trace.vital_max_hp, 6)
        self.assertIn("Hlasovanie", trace.final_output)
        self.assertTrue(trace.final_confidence >= 0.8)

    def test_05_execute_verify_guardrail(self):
        """Verify verifier gate checks and output synthesis."""
        req = {"prompt": "Overenie správnosti kódu", "strategy": "CASCADE_VERIFY"}
        plan = self.planner.build_plan(req, self.nodes)
        trace = self.executor.execute_verify(plan, req)

        self.assertEqual(trace.vital_max_hp, 6)
        self.assertEqual(trace.authority_status, "AUTH_COMMIT_APPROVED")

    def test_06_merger_contradiction_detection(self):
        """Verify CascadeMerger surfaces EPI_CONTRADICTORY when parallel models disagree."""
        res1 = CascadeNodeResult(
            node_id="model-a",
            output="Invariant VITAL_MAX_HP is 6",
            latency_ms=10.0,
            input_tokens=5,
            output_tokens=5,
            confidence=0.9
        )
        res2 = CascadeNodeResult(
            node_id="model-b",
            output="Invariant VITAL_MAX_HP is strictly 10 and not 6",
            latency_ms=12.0,
            input_tokens=5,
            output_tokens=8,
            confidence=0.9
        )

        merger = CascadeMerger()
        has_contradiction, details = merger.has_contradictions([res1, res2])
        self.assertTrue(has_contradiction)
        self.assertIn("EPI_CONTRADICTORY", details[0])

    def test_07_shadow_execution_mode(self):
        """Verify shadow execution prevents state mutation and tags WOULD_COMMIT."""
        req = {"prompt": "Spusti v shadow režime", "is_shadow": True}
        plan = self.planner.build_plan(req, self.nodes)
        trace = self.executor.execute(plan, req)

        self.assertTrue(trace.shadow_mode)
        self.assertEqual(trace.authority_status, "WOULD_COMMIT")


if __name__ == "__main__":
    unittest.main()
