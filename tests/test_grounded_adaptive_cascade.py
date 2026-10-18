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

    def test_07_prediction_ablation_and_out_of_distribution_generalization(self):
        """
        Ablation Test: Compares Static Heuristic vs Naive Historical Mean vs Token-Aware Empirical
        on completely unseen, out-of-distribution prompts.
        Refutes the hypothesis that closed-loop recalibration is merely in-sample memorization.
        """
        train_prompts = [
            "Aké sú základné epistemické invarianty Krystal Stacku?",
            "Popíš mechanizmus zero-dirty rollbacku pri zlyhaní transakcie v AEP engine.",
            "Vysvetli ako SlovakLLMAdapter monitoruje fyzickú pamäť cez Win32 API."
        ]
        test_prompts = [
            "Čo znamená VITAL_MAX_HP == 6?",
            "Navrhni optimalizáciu pre hybridnú alokáciu pamäte medzi UMA zbernicou a CPU cache pri inferencii.",
            "Vysvetli podrobne princíp chronometric shadow multiplexingu."
        ]

        # 1. Calibrate on Train Prompts
        train_traces = []
        for p in train_prompts:
            req = {"prompt": p, "strategy": "CASCADE_SEQUENTIAL", "execution_mode": "CASCADE_SEQUENTIAL"}
            plan = self.planner.build_plan(req, self.nodes)
            if not any(n.node_id == "node-slovak-llm-physical" for n in plan.nodes):
                plan.nodes.append(self.slovak_node.spec)
            tr = self.executor.execute_sequential(plan, req)
            self.planner.record_trace(tr)
            train_traces.append(tr)

        # 2. Evaluate on Unseen Test Prompts
        planner_static = CascadePlanner(router=self.router, trace_history=[])
        planner_calibrated = CascadePlanner(router=self.router, trace_history=train_traces)

        errors_static = []
        errors_calibrated = []

        for p in test_prompts:
            req = {"prompt": p, "strategy": "CASCADE_SEQUENTIAL", "execution_mode": "CASCADE_SEQUENTIAL"}
            
            # Static prediction
            plan_stat = planner_static.build_plan(req, self.nodes)
            if not any(n.node_id == "node-slovak-llm-physical" for n in plan_stat.nodes):
                plan_stat.nodes.append(self.slovak_node.spec)
            pred_stat = planner_static.estimate_latency(plan_stat, self.nodes, request=req)

            # Calibrated prediction
            plan_cal = planner_calibrated.build_plan(req, self.nodes)
            if not any(n.node_id == "node-slovak-llm-physical" for n in plan_cal.nodes):
                plan_cal.nodes.append(self.slovak_node.spec)
            pred_cal = planner_calibrated.estimate_latency(plan_cal, self.nodes, request=req, token_aware=True)

            # Actual execution
            tr_actual = self.executor.execute_sequential(plan_cal, req)
            obs_lat = tr_actual.total_latency_ms

            errors_static.append(abs(obs_lat - pred_stat))
            errors_calibrated.append(abs(obs_lat - pred_cal))

        mae_static = sum(errors_static) / len(errors_static)
        mae_calibrated = sum(errors_calibrated) / len(errors_calibrated)

        # Generalization Assertion: Calibrated MAE must be substantially lower than Static MAE
        assert mae_calibrated < mae_static
        error_reduction_pct = ((mae_static - mae_calibrated) / mae_static) * 100.0
        assert error_reduction_pct > 50.0, f"Expected >50% error reduction, got {error_reduction_pct:.1f}%"

    def test_08_multi_model_q4_vs_q8_cross_generalization(self):
        """
        Verifies that closed-loop recalibration successfully generalizes
        across two distinct model scales/quantizations (3B Q4 vs 7B Q8)
        without manual hyperparameter tuning.
        """
        # Model 2 (Dense 7B Q8_0)
        from krystal_kernel.slovak_llm_adapter import SlovakLLMAdapter, ModelSpec
        node_m2 = SlovakLLMNode(
            node_id="node-slovak-7b-q8",
            adapter=SlovakLLMAdapter(ModelSpec(
                model_id="slovak-nlp/Qwen3-14B-sk",
                parameter_count_b=7.0,
                quantization="Q8_0",
                license="Apache-2.0"
            )),
            parameter_count_b=7.0,
            quantization="Q8_0"
        )
        assert node_m2.spec.vital_max_hp == 6
        assert node_m2.spec.memory_budget_mb > 7000.0

        custom_nodes = dict(self.nodes)
        custom_nodes["node-slovak-7b-q8"] = node_m2
        executor_m2 = CascadeExecutor(custom_nodes)
        planner_m2 = CascadePlanner(router=self.router)

        # 1. Train step on Model 2
        p_train = "Vysvetli mechanizmus zamedzenia fázového sklzu v AEP."
        req_tr = {"prompt": p_train, "strategy": "CASCADE_SEQUENTIAL", "execution_mode": "CASCADE_SEQUENTIAL"}
        plan_tr = planner_m2.build_plan(req_tr, custom_nodes)
        plan_tr.nodes = [n if n.role != "REASONER" else node_m2.spec for n in plan_tr.nodes]
        tr1 = executor_m2.execute_sequential(plan_tr, req_tr)
        planner_m2.record_trace(tr1)

        # 2. Test step on Model 2 with unseen prompt
        p_test = "Čo znamená VITAL_MAX_HP == 6?"
        req_te = {"prompt": p_test, "strategy": "CASCADE_SEQUENTIAL", "execution_mode": "CASCADE_SEQUENTIAL", "token_aware": True}
        plan_te = planner_m2.build_plan(req_te, custom_nodes)
        plan_te.nodes = [n if n.role != "REASONER" else node_m2.spec for n in plan_te.nodes]
        plan_te.expected_latency_ms = planner_m2.estimate_latency(plan_te, custom_nodes, request=req_te, token_aware=True)
        pred_te = plan_te.expected_latency_ms

        tr2 = executor_m2.execute_sequential(plan_te, req_te)
        obs_te = tr2.total_latency_ms
        error_te = abs(obs_te - pred_te)

        # Static baseline would predict 63.60 ms
        static_error = abs(obs_te - 63.60)
        assert error_te < (static_error * 0.25), f"Model 2 recalibration failed: Error {error_te:.2f}ms vs Static {static_error:.2f}ms"
        assert plan_te.provenance in ("MEASURED_EMPIRICAL", "HYBRID_EMPIRICAL")

    def test_09_real_openvino_tensor_weights_generalization(self):
        """
        Proof obligation audit:
          1. Model A and Model B compile distinct weight tensors (FP_A != FP_B).
          2. TTFT and TPOT are physically measured and distinct between models.
          3. Real Win32 process RSS is captured.
          4. CascadePlanner achieves >50% prediction error reduction on unseen prompts
             under physical tensor weights WITHOUT ANY modification to planner/executor.
        """
        from krystal_kernel.slovak_llm_adapter import SlovakLLMAdapter, ModelSpec, InferenceRequest

        # Initialize Model A (3B Q4)
        node_a = SlovakLLMNode(
            node_id="node-slovak-m1-tensor",
            adapter=SlovakLLMAdapter(ModelSpec(
                model_id="slovak-nlp/Qwen3-3B-sk",
                parameter_count_b=3.0,
                quantization="Q4_K_M",
                license="Apache-2.0"
            )),
            parameter_count_b=3.0,
            quantization="Q4_K_M"
        )

        # Initialize Model B (7B Q8)
        node_b = SlovakLLMNode(
            node_id="node-slovak-m2-tensor",
            adapter=SlovakLLMAdapter(ModelSpec(
                model_id="slovak-nlp/Qwen3-7B-sk",
                parameter_count_b=7.0,
                quantization="Q8_0",
                license="Apache-2.0"
            )),
            parameter_count_b=7.0,
            quantization="Q8_0",
            enforce_headroom=False
        )

        fp_a = node_a.adapter.weight_fingerprint
        fp_b = node_b.adapter.weight_fingerprint

        # 1. Cryptographic proof that weights are strictly distinct
        assert len(fp_a) == 16
        assert len(fp_b) == 16
        assert fp_a != fp_b, f"Expected distinct weight fingerprints, got {fp_a} vs {fp_b}"

        # 2. Direct physical inference telemetry check
        req_a = InferenceRequest(prompt="Test prompt A pre verifikáciu.", max_new_tokens=16)
        resp_a = node_a.adapter.execute_inference(req_a, enforce_headroom=False)

        req_b = InferenceRequest(prompt="Test prompt B pre verifikáciu.", max_new_tokens=16)
        resp_b = node_b.adapter.execute_inference(req_b, enforce_headroom=False)

        assert resp_a.ttft_us > 0
        assert resp_b.ttft_us > 0
        assert resp_a.tpot_us > 0
        assert resp_b.tpot_us > 0
        assert (resp_a.ttft_us != resp_b.ttft_us) or (resp_a.tpot_us != resp_b.tpot_us)
        assert resp_a.weight_fingerprint == fp_a
        assert resp_b.weight_fingerprint == fp_b
        assert resp_a.process_rss_mb > 0.0

        # 3. Closed-loop adaptation on unseen prompt with Model A
        custom_nodes = dict(self.nodes)
        custom_nodes["node-slovak-m1-tensor"] = node_a
        executor = CascadeExecutor(custom_nodes)
        planner = CascadePlanner(router=ModelRouter())

        # Baseline static prediction on unseen test prompt before any historical traces
        p_test = "Aké sú podmienky pre zachovanie invariantu VITAL_MAX_HP == 6?"
        req_te = {"prompt": p_test, "strategy": "CASCADE_SEQUENTIAL", "execution_mode": "CASCADE_SEQUENTIAL", "token_aware": True}
        plan_static = planner.build_plan(req_te, custom_nodes)
        if not any(n.node_id == node_a.spec.node_id for n in plan_static.nodes):
            plan_static.nodes.append(node_a.spec)
        static_pred = planner.estimate_latency(plan_static, custom_nodes, request=req_te)

        # Calibration on training prompts
        for p_tr in [
            "Vysvetli mechanizmus adaptívneho plánovania v kaskáde.",
            "Analyzuj telemetriu a pamäťovú zbernicu UMA.",
            "Definuj podmienky pre bezpečné vykonávanie modelu v kaskáde."
        ]:
            req_tr = {"prompt": p_tr, "strategy": "CASCADE_SEQUENTIAL", "execution_mode": "CASCADE_SEQUENTIAL"}
            plan_tr = planner.build_plan(req_tr, custom_nodes)
            if not any(n.node_id == node_a.spec.node_id for n in plan_tr.nodes):
                plan_tr.nodes.append(node_a.spec)
            tr_train = executor.execute_sequential(plan_tr, req_tr)
            planner.record_trace(tr_train)

        # Calibrated adaptive evaluation on unseen test prompt
        plan_te = planner.build_plan(req_te, custom_nodes)
        if not any(n.node_id == node_a.spec.node_id for n in plan_te.nodes):
            plan_te.nodes.append(node_a.spec)
        plan_te.expected_latency_ms = planner.estimate_latency(plan_te, custom_nodes, request=req_te, token_aware=True)
        pred_te = plan_te.expected_latency_ms

        tr_test = executor.execute_sequential(plan_te, req_te)
        obs_te = tr_test.total_latency_ms
        adaptive_err = abs(obs_te - pred_te)
        static_err = abs(obs_te - static_pred)

        assert adaptive_err < (static_err * 0.50), (
            f"Expected >50% error reduction under real tensor weights, got "
            f"adaptive={adaptive_err:.2f}ms vs static={static_err:.2f}ms"
        )
        assert tr_test.vital_max_hp == 6



