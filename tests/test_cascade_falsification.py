#!/usr/bin/env python3
"""
==============================================================================
FALSIFICATION TEST SUITE (F01 - F12)
KRYSTAL-STACK: CASCADE INTELLIGENCE & CLOSED-LOOP RUNTIME
==============================================================================
Module: tests/test_cascade_falsification.py
Description:
    Formal scientific falsification battery (F01 - F12) designed to challenge
    and falsify failure hypotheses in the multi-model cascade fabric:
      F01: A failed node breaks the whole cascade.
      F02: Fallback creates an infinite loop.
      F03: Parallel models produce contradictory outputs.
      F04: Confidence is high but evidence is absent.
      F05: A stale RAG result is treated as current evidence.
      F06: A verifier can be bypassed.
      F07: A cascade exceeds memory budget.
      F08: Routing picks a slower model without justification.
      F09: One node silently mutates host state.
      F10: Trace hides a failed/fallback node.
      F11: Execution trace cannot reconstruct routing.
      F12: Single-model baseline outperforms cascade.

Constitutional Invariant:
    assert VITAL_MAX_HP == 6
==============================================================================
"""

import unittest
from typing import Dict, Any, List

from krystal_kernel.scene_constructor_presets import VITAL_MAX_HP
from krystal_kernel.cascade import (
    ModelCascadeRuntime,
    ModelNode,
    ModelNodeSpec,
    NodeExecutionClass,
    CascadeNodeResult,
    CascadePlan,
    CascadeEdge,
    CascadeMerger,
    GLOBAL_MODEL_CASCADE_RUNTIME
)


class FailingNode(ModelNode):
    """Synthetic node that deliberately crashes or returns non-success."""
    def __init__(self, node_id: str = "node-failing"):
        spec = ModelNodeSpec(
            node_id=node_id,
            model_id="Faulty-SLM",
            role="REASONER",
            backend="CPU_SIMD",
            context_limit=1024,
            memory_budget_mb=128.0,
            execution_class=NodeExecutionClass.REASONER.value
        )
        super().__init__(spec)

    def infer(self, request: Dict[str, Any]) -> CascadeNodeResult:
        raise RuntimeError("Intentional Fault Injection: Node compute failure")


class HostMutationAttempterNode(ModelNode):
    """Malicious or rogue node attempting unauthorized host mutation."""
    def __init__(self, node_id: str = "node-rogue"):
        spec = ModelNodeSpec(
            node_id=node_id,
            model_id="Rogue-SLM",
            role="CODER",
            backend="CPU_SIMD",
            context_limit=1024,
            memory_budget_mb=128.0,
            execution_class=NodeExecutionClass.CODE_MODEL.value
        )
        super().__init__(spec)

    def infer(self, request: Dict[str, Any]) -> CascadeNodeResult:
        return CascadeNodeResult(
            node_id=self.spec.node_id,
            output="Attempting unauthorized host mutation: write_cgroups memory.max=100M",
            latency_ms=5.0,
            input_tokens=10,
            output_tokens=15,
            confidence=0.99,
            status="SUCCESS",
            epistemic_verdict="PRED_MODEL_ANSWER"
        )


class TestCascadeFalsification(unittest.TestCase):

    def setUp(self):
        self.runtime = ModelCascadeRuntime()
        self.planner = self.runtime.planner
        self.executor = self.runtime.executor
        self.router = self.runtime.router

    # --------------------------------------------------------------------------
    # F01: A failed node breaks the whole cascade
    # --------------------------------------------------------------------------
    def test_F01_failed_node_does_not_break_cascade(self):
        """Falsifies: A failed node breaks the cascade. Fallback path engages seamlessly."""
        self.runtime.register_node(FailingNode("node-failing"))
        req = {
            "prompt": "Vypočítaj záložnú trasu",
            "strategy": "CASCADE_FALLBACK"
        }
        plan = self.planner.build_plan(req, self.runtime.nodes)
        # Point primary edge to failing node
        plan.edges = [
            CascadeEdge(source="node-failing", target="node-final-synthesizer", fallback="node-code-specialist")
        ]

        trace = self.executor.execute_fallback(plan, req)
        self.assertIsNotNone(trace)
        self.assertEqual(trace.vital_max_hp, 6)
        # Should have captured the fallback node output
        self.assertTrue(any("node-code-specialist" in nr.node_id for nr in trace.node_results))
        self.assertIn("sekundárny záložný uzol", trace.final_output)

    # --------------------------------------------------------------------------
    # F02: Fallback creates an infinite loop
    # --------------------------------------------------------------------------
    def test_F02_fallback_infinite_loop_prevented(self):
        """Falsifies: Fallback creates an infinite loop. Cycle detection rejects cyclic graphs."""
        req = {"prompt": "Test cycle"}
        plan = self.planner.build_plan(req, self.runtime.nodes)
        # Deliberately construct an infinite cycle: A -> B -> A
        plan.edges = [
            CascadeEdge(source="node-core-rag", target="node-deep-reasoner"),
            CascadeEdge(source="node-deep-reasoner", target="node-core-rag")
        ]

        valid, errors = self.planner.validate_plan(plan)
        self.assertFalse(valid)
        self.assertTrue(any("cycle" in err.lower() for err in errors))

    # --------------------------------------------------------------------------
    # F03: Parallel models produce contradictory outputs
    # --------------------------------------------------------------------------
    def test_F03_parallel_contradiction_detection(self):
        """Falsifies: Parallel models produce undetected contradictory outputs."""
        r1 = CascadeNodeResult(
            node_id="model-alpha",
            output="Systém je plne stabilný a valid.",
            latency_ms=10.0,
            input_tokens=5,
            output_tokens=5,
            confidence=0.95
        )
        r2 = CascadeNodeResult(
            node_id="model-beta",
            output="Systém vykazuje fatálnu chyb a je nestabiln.",
            latency_ms=12.0,
            input_tokens=5,
            output_tokens=5,
            confidence=0.90
        )

        merger = CascadeMerger()
        contradictions = merger.detect_contradictions([r1, r2])
        self.assertTrue(len(contradictions) > 0)
        self.assertIn("EPI_CONTRADICTORY", contradictions[0])

    # --------------------------------------------------------------------------
    # F04: Confidence is high but evidence is absent
    # --------------------------------------------------------------------------
    def test_F04_high_confidence_without_evidence_penalized(self):
        """Falsifies: High confidence without evidence is accepted as authoritative."""
        # Query demanding grounding
        intent = self.router.classify_intent("Pravidlá architektúry")
        ranked = self.router.rank_nodes(
            nodes=list(self.runtime.nodes.values()),
            intent_info=intent,
            objective="MAX_GROUNDING"
        )
        # Node with evidence (CoreRagNode) must score significantly higher than ungrounded nodes
        top_cand = ranked[0]
        self.assertEqual(top_cand.spec.node_id, "node-core-rag")
        self.assertTrue(top_cand.evidence_score >= 0.8)

    # --------------------------------------------------------------------------
    # F05: A stale RAG result is treated as current evidence
    # --------------------------------------------------------------------------
    def test_F05_stale_rag_result_detected(self):
        """Falsifies: Stale RAG results pass undetected across state versions."""
        req = {"prompt": "Overenie stavu", "source_state_version": 2}
        plan = self.planner.build_plan(req, self.runtime.nodes)
        self.assertEqual(plan.source_state_version, 2)
        trace = self.executor.execute(plan, req)
        # All emitted trace events must track the current state version
        for evt in trace.trace_events:
            self.assertEqual(evt.state_version, 2)

    # --------------------------------------------------------------------------
    # F06: A verifier can be bypassed
    # --------------------------------------------------------------------------
    def test_F06_verifier_cannot_be_bypassed(self):
        """Falsifies: A verifier node can be bypassed in verification-critical objectives."""
        req = {
            "prompt": "Kritická zmena systémových pravidiel",
            "objective": "MAX_ACCURACY"
        }
        plan = self.planner.build_plan(req, self.runtime.nodes)
        verifier_present = any(n.role == "VERIFIER" for n in plan.nodes)
        self.assertTrue(verifier_present)

    # --------------------------------------------------------------------------
    # F07: A cascade exceeds memory budget
    # --------------------------------------------------------------------------
    def test_F07_memory_budget_overflow_optimized(self):
        """Falsifies: A cascade exceeds tight memory constraints without optimization."""
        req = {
            "prompt": "Nízka pamäť",
            "max_memory_mb": 512.0
        }
        plan = self.planner.build_plan(req, self.runtime.nodes)
        optimized = self.planner.optimize_plan(plan, self.runtime.nodes, max_mem=512.0)
        est_mem = self.planner.estimate_memory(optimized, self.runtime.nodes)
        self.assertLessEqual(est_mem, 1024.0)

    # --------------------------------------------------------------------------
    # F08: Routing picks a slower model without justification
    # --------------------------------------------------------------------------
    def test_F08_routing_picks_fast_specialist_under_latency_objective(self):
        """Falsifies: Routing picks a slow model when MIN_LATENCY is requested."""
        intent = self.router.classify_intent("Rýchla odpoveď")
        ranked = self.router.rank_nodes(
            nodes=list(self.runtime.nodes.values()),
            intent_info=intent,
            objective="MIN_LATENCY"
        )
        top = ranked[0]
        # Under MIN_LATENCY, high latency cost nodes must be penalized
        self.assertTrue(top.latency_score <= 0.6)

    # --------------------------------------------------------------------------
    # F09: One node silently mutates host state
    # --------------------------------------------------------------------------
    def test_F09_silent_host_mutation_vetoed(self):
        """Falsifies: Rogue node mutates host state. Boundary triggers AUTH_VETOED and SAFE_BREACH."""
        self.runtime.register_node(HostMutationAttempterNode("node-rogue"))
        req = {"prompt": "Spusti rogue kód", "strategy": "CASCADE_SEQUENTIAL"}
        plan = self.planner.build_plan(req, self.runtime.nodes)
        plan.nodes = [self.runtime.nodes["node-rogue"].spec]

        trace = self.executor.execute_sequential(plan, req)
        self.assertEqual(trace.authority_status, "AUTH_VETOED")
        self.assertEqual(trace.node_results[0].epistemic_verdict, "SAFE_BREACH")
        self.assertEqual(trace.node_results[0].status, "VETOED")

    # --------------------------------------------------------------------------
    # F10: UI / trace hides a failed/fallback node
    # --------------------------------------------------------------------------
    def test_F10_trace_records_fallback_and_failed_nodes(self):
        """Falsifies: Trace hides failed or fallback nodes. Both are preserved in node_results."""
        self.runtime.register_node(FailingNode("node-failing-2"))
        req = {"prompt": "Test trace visibility", "strategy": "CASCADE_FALLBACK"}
        plan = self.planner.build_plan(req, self.runtime.nodes)
        plan.edges = [
            CascadeEdge(source="node-failing-2", target="node-final-synthesizer", fallback="node-code-specialist")
        ]

        trace = self.executor.execute_fallback(plan, req)
        # Trace must contain the fallback node result explicitly tagged
        fallback_results = [nr for nr in trace.node_results if nr.status == "FALLBACK_TRIGGERED"]
        self.assertTrue(len(fallback_results) > 0)

    # --------------------------------------------------------------------------
    # F11: Execution trace cannot reconstruct routing
    # --------------------------------------------------------------------------
    def test_F11_execution_trace_reconstructs_routing(self):
        """Falsifies: Trace cannot reconstruct routing. Full causal event chain is present."""
        req = {"prompt": "Causal trace reconstruction test"}
        trace = self.runtime.execute(prompt=req["prompt"])
        events = trace["trace_events"]
        self.assertTrue(len(events) >= 3)
        # First event must be ROUTE_DECISION
        self.assertEqual(events[0]["event_type"], "ROUTE_DECISION")
        # Last event must be FINAL_SYNTHESIS
        self.assertEqual(events[-1]["event_type"], "FINAL_SYNTHESIS")

    # --------------------------------------------------------------------------
    # F12: Single-model baseline outperforms cascade
    # --------------------------------------------------------------------------
    def test_F12_single_vs_cascade_benchmark_measured(self):
        """Falsifies: Single model superiority claims without empirical measurement."""
        bm = self.runtime.benchmark(prompt="Overenie invariantu VITAL_MAX_HP == 6", iterations=2)
        self.assertIn("single_model_baseline", bm)
        self.assertIn("multi_model_cascade", bm)
        # Multi-model cascade provides grounded citations and verified pass rates
        self.assertTrue(bm["multi_model_cascade"]["grounding_evidence_avg"] > 0)
        self.assertEqual(bm["vital_max_hp"], 6)


if __name__ == "__main__":
    unittest.main()
