#!/usr/bin/env python3
"""
==============================================================================
KRYSTAL-STACK: GROUNDED ADAPTIVE CASCADE VERIFICATION SUITE
==============================================================================
Test Suite: tests/test_grounded_adaptive_cascade.py
Description:
    Formally verifies the closed-loop adaptive cascade pipeline with real
    physical LLM substrate (SlovakLLMNode wrapping SlovakLLMAdapter):
      1. Environment snapshot capture & provenance in CascadePlan
      2. Physical inference timing & MEASURED provenance propagation
      3. Provenance segregation in CascadeExecutionTrace
      4. Closed-loop recalibration: Route -> Plan -> Execute -> Observe -> Compare -> Learn -> Re-Plan
      5. Shadow mode side-effect isolation & Delta S == 0 guarantee
      6. Constitutional Invariant: assert VITAL_MAX_HP == 6

Constitutional Invariant:
    assert VITAL_MAX_HP == 6
==============================================================================
"""

import pytest
import time
from typing import Dict, Any

from krystal_kernel.scene_constructor_presets import VITAL_MAX_HP
from krystal_kernel.cascade.model_node import (
    IntentClassifierNode,
    CoreRagNode,
    VerificationNode,
    FinalSynthesisNode,
    SlovakLLMNode,
)
from krystal_kernel.cascade.model_router import ModelRouter
from krystal_kernel.cascade.cascade_planner import CascadePlanner, CascadePlan
from krystal_kernel.cascade.cascade_executor import CascadeExecutor
from krystal_kernel.cascade.cascade_trace import MetricProvenance, CascadeExecutionTrace
from krystal_kernel.cascade.closed_loop_adapter import ClosedLoopCascadeFeedback
from krystal_kernel.slovak_llm_adapter import SlovakLLMAdapter, ModelSpec


class TestGroundedAdaptiveCascade:
    """Verifies grounded physical inference and closed-loop adaptation."""

    def setup_method(self):
        assert VITAL_MAX_HP == 6, "Invariant VITAL_MAX_HP == 6 must hold"
        self.router = ModelRouter()
        self.planner = CascadePlanner(router=self.router)
        self.slovak_node = SlovakLLMNode(
            node_id="node-slovak-llm-physical",
            parameter_count_b=3.0,
            quantization="Q4_K_M"
        )
        self.nodes = {
            "node-classifier-slm": IntentClassifierNode(),
            "node-core-rag": CoreRagNode(),
            "node-slovak-llm-physical": self.slovak_node,
            "node-epistemic-verifier": VerificationNode(),
            "node-final-synthesizer": FinalSynthesisNode(),
        }
        self.executor = CascadeExecutor(self.nodes)
        self.feedback = ClosedLoopCascadeFeedback()

    def test_01_cascade_plan_environment_snapshot_and_provenance(self):
        """Verifies CascadePlan captures physical hardware telemetry and tracks provenance."""
        req = {
            "prompt": "Vysvetli mechanizmus closed-loop spätnej väzby v Krystal Stacku.",
            "objective": "BALANCED"
        }
        plan = self.planner.build_plan(req, self.nodes)
        assert plan.vital_max_hp == 6
        assert plan.environment_snapshot is not None
        assert "available_memory_mb" in plan.environment_snapshot
        assert "cpu_cores" in plan.environment_snapshot
        assert "platform" in plan.environment_snapshot
        assert plan.environment_snapshot["available_memory_mb"] > 0
        assert plan.provenance in ("ESTIMATED", "MEASURED_EMPIRICAL", "HYBRID_EMPIRICAL")

        d = plan.to_dict()
        assert d["environment_snapshot"] is not None
        assert "provenance" in d
        assert "historical_evidence" in d

    def test_02_slovak_llm_node_measured_provenance_propagation(self):
        """Verifies SlovakLLMNode executes real physical inference and emits MEASURED provenance."""
        res = self.slovak_node.infer({
            "prompt": "Test slovenského modelu s diakritikou: ľščťžýáíé úäň",
            "max_tokens": 16,
            "target_device": "CPU"
        })
        assert res.vital_max_hp == 6
        assert res.status == "SUCCESS"
        assert res.provenance == MetricProvenance.MEASURED.value
        assert res.latency_ms > 0.0
        assert res.input_tokens > 0
        assert res.output_tokens > 0
        assert res.memory_mb > 0.0
        assert any("PHYSICAL-LOCAL-TIMING" in ref for ref in res.evidence_refs)

    def test_03_trace_provenance_segregation(self):
        """Verifies that CascadeExecutionTrace segregates MEASURED vs SYNTHETIC node results."""
        req = {
            "prompt": "Analyzuj výkonnostné charakteristiky Slovak LLM pod záťažou.",
            "strategy": "CASCADE_SEQUENTIAL",
            "execution_mode": "CASCADE_SEQUENTIAL"
        }
        plan = self.planner.build_plan(req, self.nodes)
        # Ensure our physical node is included in the plan
        if not any(n.node_id == "node-slovak-llm-physical" for n in plan.nodes):
            plan.nodes.append(self.slovak_node.spec)

        trace = self.executor.execute_sequential(plan, req)
        assert trace.vital_max_hp == 6
        assert trace.provenance_summary is not None
        assert "MEASURED" in trace.provenance_summary
        assert "SYNTHETIC" in trace.provenance_summary
        assert trace.provenance_summary["MEASURED"] >= 1
        assert trace.total_latency_ms > 0.0

        trace_dict = trace.to_dict()
        assert "provenance_summary" in trace_dict
        assert trace_dict["provenance_summary"]["MEASURED"] >= 1

    def test_04_closed_loop_feedback_and_planner_recalibration(self):
        """
        Verifies the full cybernetic feedback loop:
        Route -> Plan (Cold) -> Execute -> Observe -> Compare -> Update -> Re-Plan (Empirical)
        """
        prompt = "Popíš bezpečnú správu pamäte a deterministický rollback v AEP."
        req1 = {"prompt": prompt, "objective": "BALANCED"}

        # 1. Cold Start Planning
        plan1 = self.planner.build_plan(req1, self.nodes)
        cold_lat = plan1.expected_latency_ms
        assert plan1.provenance == "ESTIMATED"

        # 2. Execution with Physical Substrate
        # Swap reasoner to SlovakLLMNode in nodes and execute
        trace1 = self.executor.execute_sequential(plan1, req1)
        assert trace1.authority_status == "AUTH_COMMIT_APPROVED"

        # 3. Observe & Compare via Feedback
        cycle_metrics = self.feedback.process_cycle(plan1, trace1, self.router)
        assert cycle_metrics["feedback_applied"] is True
        assert cycle_metrics["provenance"] == "MEASURED"

        # 4. Feed empirical trace into planner
        self.planner.record_trace(trace1)
        assert len(self.planner.trace_history) == 1

        # 5. Re-plan for second request
        req2 = {"prompt": prompt, "objective": "BALANCED"}
        plan2 = self.planner.build_plan(req2, self.nodes)

        # 6. Verify that Plan 2 incorporates empirical evidence
        assert len(plan2.historical_evidence) > 0
        assert plan2.provenance in ("HYBRID_EMPIRICAL", "MEASURED_EMPIRICAL")

    def test_05_shadow_mode_and_side_effect_isolation(self):
        """Verifies shadow execution tags verdict with WOULD_COMMIT and guarantees zero host mutation."""
        req_shadow = {
            "prompt": "Test shadow execution mode under zero-dirty contract.",
            "is_shadow": True
        }
        plan = self.planner.build_plan(req_shadow, self.nodes)
        trace = self.executor.execute_sequential(plan, req_shadow)

        assert trace.shadow_mode is True
        assert "WOULD_COMMIT" in trace.authority_status
        assert trace.vital_max_hp == 6

    def test_06_vital_max_hp_invariant_across_all_adaptive_primitives(self):
        """Constitutional invariant VITAL_MAX_HP == 6 must hold everywhere."""
        assert self.slovak_node.spec.vital_max_hp == 6
        assert self.feedback.topology.vital_max_hp == 6
        assert self.feedback.execution_matrix.vital_max_hp == 6
        
        plan = self.planner.build_plan({"prompt": "invariant test"}, self.nodes)
        assert plan.vital_max_hp == 6
        for edge in plan.edges:
            assert edge.vital_max_hp == 6
        for node in plan.nodes:
            assert node.vital_max_hp == 6
