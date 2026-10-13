#!/usr/bin/env python3
"""
==============================================================================
UNIT TESTS: KRYSTAL-STACK MODEL CASCADE RUNTIME & EXECUTION FABRIC
==============================================================================
Tests for:
  - ModelNode, ModelNodeSpec, and 6 specialized concrete nodes
  - ModelRouter (epistemic scoring, ranking, filters, regret updates)
  - CascadePlanner (plan compilation, validation, cost estimation, optimization)
  - CascadeExecutor (8 execution modes, merger, contradiction detection)
  - ModelCascadeRuntime & GLOBAL_MODEL_CASCADE_RUNTIME
  - Platform invariant: assert VITAL_MAX_HP == 6
==============================================================================
"""

import unittest
from typing import Dict, Any, List

from krystal_kernel.scene_constructor_presets import VITAL_MAX_HP
from krystal_kernel.cascade import (
    ModelNode,
    ModelNodeSpec,
    IntentClassifierNode,
    CoreRagNode,
    ReasoningNode,
    CodingSpecialistNode,
    VerificationNode,
    FinalSynthesisNode,
    ModelRouter,
    NodeRouteCandidate,
    RoutingObjectiveWeights,
    ROUTING_PROFILES,
    CascadeEdge,
    CascadePlan,
    CascadePlanner,
    CascadeMerger,
    CascadeExecutor,
    CascadeNodeResult,
    CascadeExecutionTrace,
    ModelCascadeRuntime,
    GLOBAL_MODEL_CASCADE_RUNTIME,
)


class TestModelCascadeRuntime(unittest.TestCase):

    def setUp(self):
        self.runtime = ModelCascadeRuntime()
        self.router = self.runtime.router
        self.planner = self.runtime.planner
        self.executor = self.runtime.executor

    def test_01_invariant_vital_max_hp(self):
        """Verify universal system invariant VITAL_MAX_HP == 6 across all cascade components."""
        self.assertEqual(VITAL_MAX_HP, 6)
        spec = ModelNodeSpec(
            node_id="test-node",
            model_id="Test-1B",
            role="CLASSIFIER",
            backend="CPU_SIMD",
            context_limit=1024,
            memory_budget_mb=256.0,
            capabilities={"test"}
        )
        self.assertEqual(spec.vital_max_hp, 6)

        edge = CascadeEdge(source="a", target="b")
        self.assertEqual(edge.vital_max_hp, 6)

        res = CascadeNodeResult(
            node_id="test",
            output="ok",
            latency_ms=1.0,
            input_tokens=2,
            output_tokens=2,
            confidence=0.9
        )
        self.assertEqual(res.vital_max_hp, 6)

    def test_02_specialized_nodes_inference(self):
        """Verify inference of all 6 specialized nodes."""
        classifier = IntentClassifierNode()
        c_res = classifier.infer({"prompt": "Napíš kód v Pythone"})
        self.assertIn("CODE_GENERATION", c_res.output)
        self.assertEqual(c_res.status, "SUCCESS")
        self.assertEqual(c_res.vital_max_hp, 6)

        rag = CoreRagNode()
        r_res = rag.infer({"prompt": "univerzálny invariant"})
        self.assertEqual(r_res.status, "SUCCESS")
        self.assertTrue(len(r_res.evidence_refs) > 0)

        reasoner = ReasoningNode()
        re_res = reasoner.infer({"prompt": "Prečo je potrebný invariant?", "rag_context": "Invariant je VITAL_MAX_HP == 6"})
        self.assertIn("<think>", re_res.output)
        self.assertEqual(re_res.status, "SUCCESS")

        coder = CodingSpecialistNode()
        co_res = coder.infer({"prompt": "funkcia pre validáciu"})
        self.assertIn("def execute_verified_workload", co_res.output)
        self.assertIn("VITAL_MAX_HP == 6", co_res.output)

        verifier = VerificationNode()
        v_res = verifier.infer({"candidate_output": "def safe_code(): pass"})
        self.assertEqual(v_res.output, "AUTH_COMMIT_APPROVED")

        # Test safety veto
        v_veto = verifier.infer({"candidate_output": "import os; os.system('rm -rf /')"})
        self.assertEqual(v_veto.status, "VETOED")
        self.assertIn("SAFETY_VETO", v_veto.output)

        synthesizer = FinalSynthesisNode()
        s_res = synthesizer.infer({"body": "Overená odpoveď", "citations": ["ref://1", "ref://2"]})
        self.assertIn("Overená odpoveď", s_res.output)
        self.assertIn("ref://1", s_res.output)
        self.assertIn("VITAL_MAX_HP == 6", s_res.output)

    def test_03_model_router_epistemic_scoring(self):
        """Verify epistemic route score formula and candidate ranking."""
        intent_info = self.router.classify_intent("Optimalizuj Python algoritmus")
        self.assertEqual(intent_info["intent"], "CODE_GENERATION")

        nodes = list(self.runtime.nodes.values())
        ranked = self.router.rank_nodes(
            nodes=nodes,
            intent_info=intent_info,
            objective="ACCURACY",
            max_latency_ms=200.0,
            max_memory_mb=4096.0
        )
        self.assertTrue(len(ranked) > 0)
        top = ranked[0]
        self.assertTrue(top.is_eligible)
        self.assertTrue(hasattr(top, "route_score"))
        # Verify formula: RouteScore_i = w1*C_i + w2*E_i - w3*L_i - w4*M_i - w5*R_i
        w = ROUTING_PROFILES["ACCURACY"]
        expected_score = (
            w.w_capability * top.capability_score +
            w.w_evidence * top.evidence_score -
            w.w_latency * top.latency_score -
            w.w_memory * top.memory_score -
            w.w_regret * top.regret_score
        )
        self.assertAlmostEqual(top.route_score, expected_score, places=4)

    def test_04_model_router_fallback_and_regret(self):
        """Verify fallback node selection and regret penalty accumulation."""
        intent_info = self.router.classify_intent("Analyzuj systém")
        ranked = self.router.rank_nodes(list(self.runtime.nodes.values()), intent_info)
        primary = ranked[0]
        fallback = self.router.select_fallback(primary.node_id, ranked)
        self.assertIsNotNone(fallback)
        self.assertNotEqual(primary.node_id, fallback.node_id)
        # Verify regret increased
        self.assertGreater(self.router.node_regret_history[primary.node_id], 0.0)

    def test_05_cascade_planner_compilation_and_validation(self):
        """Verify graph plan compilation across different strategies and validation."""
        req_seq = {"prompt": "Bežná otázka o platforme", "strategy": "CASCADE_SEQUENTIAL"}
        plan_seq = self.planner.build_plan(req_seq, self.runtime.nodes)
        self.assertEqual(plan_seq.strategy, "CASCADE_SEQUENTIAL")
        valid, errors = self.planner.validate_plan(plan_seq)
        self.assertTrue(valid, f"Validation errors: {errors}")
        self.assertEqual(plan_seq.vital_max_hp, 6)

        req_par = {"prompt": "Paralelná syntéza", "strategy": "CASCADE_PARALLEL"}
        plan_par = self.planner.build_plan(req_par, self.runtime.nodes)
        self.assertEqual(plan_par.strategy, "CASCADE_PARALLEL")
        valid, errors = self.planner.validate_plan(plan_par)
        self.assertTrue(valid, f"Validation errors: {errors}")

        req_vote = {"prompt": "Hlasovanie", "strategy": "CASCADE_VOTE"}
        plan_vote = self.planner.build_plan(req_vote, self.runtime.nodes)
        self.assertEqual(plan_vote.strategy, "CASCADE_VOTE")
        valid, errors = self.planner.validate_plan(plan_vote)
        self.assertTrue(valid, f"Validation errors: {errors}")

    def test_06_cascade_merger_consensus_and_contradictions(self):
        """Verify CascadeMerger output blending, contradiction alerts, and consensus."""
        merger = CascadeMerger()
        r1 = CascadeNodeResult("node-1", "Stav je stabilný a bezpečný.", 10.0, 5, 10, 0.95)
        r2 = CascadeNodeResult("node-2", "Stav je plne stabilný.", 12.0, 5, 8, 0.88)
        r3 = CascadeNodeResult("node-3", "Stav zlyhal pri vykonávaní.", 14.0, 5, 8, 0.70)

        merged = merger.merge_outputs([r1, r2])
        self.assertIn("node-1", merged)
        self.assertIn("node-2", merged)

        auth = merger.select_authoritative_result([r1, r2, r3])
        self.assertEqual(auth.node_id, "node-1")

        contradictions = merger.detect_contradictions([r1, r3])
        self.assertTrue(len(contradictions) > 0)
        self.assertIn("Polarity conflict", contradictions[0])

    def test_07_all_eight_cascade_execution_modes(self):
        """Verify execution of all 8 canonical cascade strategies."""
        modes = [
            "CASCADE_SEQUENTIAL",
            "CASCADE_PARALLEL",
            "CASCADE_VOTE",
            "CASCADE_VERIFY",
            "CASCADE_FALLBACK",
            "CASCADE_SPECIALIST_ROUTE",
            "CASCADE_RAG_GROUNDED",
            "CASCADE_TOOL_EXECUTION"
        ]

        for mode in modes:
            req = {"prompt": "Testovacia požiadavka", "strategy": mode}
            plan = self.planner.build_plan(req, self.runtime.nodes)
            trace = self.executor.execute(plan, req)
            self.assertEqual(trace.vital_max_hp, 6, f"Failed for {mode}")
            self.assertEqual(trace.strategy, mode)
            self.assertTrue(len(trace.node_results) > 0)
            self.assertTrue(len(trace.final_output) > 0)

    def test_08_global_model_cascade_runtime_facade(self):
        """Verify top-level GLOBAL_MODEL_CASCADE_RUNTIME API."""
        prompt = "Vygeneruj overený kód pre Krystal Stack s VITAL_MAX_HP == 6"
        result = GLOBAL_MODEL_CASCADE_RUNTIME.execute(
            prompt=prompt,
            objective="ACCURACY",
            max_latency_ms=150,
            max_memory_mb=4096,
            require_grounding=True
        )

        self.assertIsInstance(result, dict)
        self.assertIn("final_output", result)
        self.assertIn("trace_id", result)
        self.assertEqual(result["vital_max_hp"], 6)
        self.assertEqual(result["objective"], "ACCURACY")
        self.assertTrue(len(result["node_results"]) >= 4)
        self.assertIn("VITAL_MAX_HP == 6", result["final_output"])

        # Check telemetry
        telemetry = GLOBAL_MODEL_CASCADE_RUNTIME.get_telemetry()
        self.assertGreater(telemetry["total_executions"], 0)
        self.assertEqual(telemetry["vital_max_hp"], 6)
        self.assertEqual(telemetry["registered_nodes_count"], 6)


if __name__ == "__main__":
    unittest.main()
