#!/usr/bin/env python3
"""
==============================================================================
KRYSTAL-STACK: MODEL CASCADE RUNTIME & GLOBAL ORCHESTRATION FABRIC
==============================================================================
Module: krystal_kernel/cascade/cascade_runtime.py
Description:
    Central runtime fabric orchestrating multi-model execution graphs,
    epistemic routing, Dual-Gate invariant verification, and per-node tracing.
    Evolves the single-model LM-Studio pattern into a deterministic multi-model
    fabric.

Canonical Usage:
    from krystal_kernel.cascade import GLOBAL_MODEL_CASCADE_RUNTIME

    result = GLOBAL_MODEL_CASCADE_RUNTIME.execute(
        prompt="Vygeneruj funkciu v Pythone s overením invariantu VITAL_MAX_HP == 6",
        objective="ACCURACY",
        max_latency_ms=150,
        max_memory_mb=4096,
        require_grounding=True,
    )
    print(result["final_output"])

Constitutional Invariant:
    assert VITAL_MAX_HP == 6
==============================================================================
"""

import time
import uuid
from typing import Dict, Any, List, Optional, Set, Tuple

from krystal_kernel.scene_constructor_presets import VITAL_MAX_HP
from krystal_kernel.cascade.cascade_trace import CascadeNodeResult, CascadeExecutionTrace
from krystal_kernel.cascade.model_node import (
    ModelNode,
    ModelNodeSpec,
    IntentClassifierNode,
    CoreRagNode,
    ReasoningNode,
    CodingSpecialistNode,
    VerificationNode,
    FinalSynthesisNode
)
from krystal_kernel.cascade.model_router import ModelRouter, NodeRouteCandidate, ROUTING_PROFILES
from krystal_kernel.cascade.cascade_planner import CascadeEdge, CascadePlan, CascadePlanner
from krystal_kernel.cascade.cascade_executor import CascadeExecutor, CascadeMerger
from krystal_kernel.cascade.closed_loop_adapter import (
    ThreadTopology,
    ModelExecutionMatrix,
    ClosedLoopCascadeFeedback,
    CascadeBenchmarkEngine,
    RuntimeHypothesis,
    HypothesisState
)


class ModelCascadeRuntime:
    """
    Central orchestration fabric managing node catalogs, epistemic routing,
    graph compilation, closed-loop cybernetic feedback, and execution across 8 cascade modes.
    """

    def __init__(self):
        assert VITAL_MAX_HP == 6, "Invariant VITAL_MAX_HP == 6 must hold"
        self.router = ModelRouter()
        self.planner = CascadePlanner(router=self.router)
        self.nodes: Dict[str, ModelNode] = {}
        self.executor = CascadeExecutor(self.nodes)
        self.trace_history: List[CascadeExecutionTrace] = []
        self.closed_loop = ClosedLoopCascadeFeedback()
        self.benchmark_engine = CascadeBenchmarkEngine(self)

        # Bootstrap default production node catalog
        self._register_default_nodes()

    def _register_default_nodes(self):
        """Instantiates canonical specialized nodes."""
        default_nodes = [
            IntentClassifierNode("node-classifier-slm"),
            CoreRagNode("node-core-rag"),
            ReasoningNode("node-deep-reasoner"),
            CodingSpecialistNode("node-code-specialist"),
            VerificationNode("node-epistemic-verifier"),
            FinalSynthesisNode("node-final-synthesizer")
        ]
        for node in default_nodes:
            self.register_node(node)

    def register_node(self, node: ModelNode) -> bool:
        """Registers a model node into the runtime catalog."""
        assert node.spec.vital_max_hp == 6, "Node invariant violation"
        self.nodes[node.spec.node_id] = node
        return True

    def unregister_node(self, node_id: str) -> bool:
        """Removes a model node from the runtime catalog."""
        if node_id in self.nodes:
            del self.nodes[node_id]
            return True
        return False

    def get_node(self, node_id: str) -> Optional[ModelNode]:
        """Fetches a node by identifier."""
        return self.nodes.get(node_id)

    def list_nodes(self) -> List[Dict[str, Any]]:
        """Returns metadata specifications and health of all registered nodes."""
        catalog = []
        for n in self.nodes.values():
            h = n.health()
            spec_d = n.spec.to_dict()
            spec_d["health"] = h
            catalog.append(spec_d)
        return catalog

    def load_node(self, node_id: str) -> Dict[str, Any]:
        """Explicitly loads a model node into working memory."""
        node = self.get_node(node_id)
        if not node:
            return {"status": "ERROR", "error": f"Node {node_id} not found"}
        ok = node.load()
        return {"status": "SUCCESS" if ok else "ERROR", "node_id": node_id, "loaded": node.loaded}

    def unload_node(self, node_id: str) -> Dict[str, Any]:
        """Unloads a model node from working memory."""
        node = self.get_node(node_id)
        if not node:
            return {"status": "ERROR", "error": f"Node {node_id} not found"}
        ok = node.unload()
        return {"status": "SUCCESS" if ok else "ERROR", "node_id": node_id, "loaded": node.loaded}

    def warmup_node(self, node_id: str) -> Dict[str, Any]:
        """Executes a calibration warmup inference on the specified node."""
        node = self.get_node(node_id)
        if not node:
            return {"status": "ERROR", "error": f"Node {node_id} not found"}
        t0 = time.perf_counter_ns()
        res = node.infer({"prompt": "warmup_probe_ping"})
        lat_ms = (time.perf_counter_ns() - t0) / 1_000_000.0
        return {
            "status": "SUCCESS",
            "node_id": node_id,
            "warmup_latency_ms": round(lat_ms, 2),
            "output_tokens": res.output_tokens
        }

    def plan(
        self,
        prompt: str,
        objective: str = "BALANCED",
        max_latency_ms: float = 200.0,
        max_memory_mb: float = 4096.0,
        require_grounding: bool = True,
        strategy: Optional[str] = None,
        **kwargs
    ) -> CascadePlan:
        """
        Compiles an execution plan without executing it. Useful for dry-runs,
        graph inspection, and UI previews.
        """
        request = {
            "prompt": prompt,
            "objective": objective,
            "max_latency_ms": max_latency_ms,
            "max_memory_mb": max_memory_mb,
            "require_grounding": require_grounding,
            "strategy": strategy,
            **kwargs
        }
        return self.planner.build_plan(request, self.nodes)

    def execute(
        self,
        prompt: str,
        objective: str = "BALANCED",
        max_latency_ms: float = 200.0,
        max_memory_mb: float = 4096.0,
        require_grounding: bool = True,
        strategy: Optional[str] = None,
        plan: Optional[CascadePlan] = None,
        is_shadow: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Primary execution entry point.
        Compiles plan (if not provided), executes across registered nodes,
        enforces Dual-Gate invariants, feeds closed-loop regret back to router,
        and returns complete forensic trace.
        """
        assert VITAL_MAX_HP == 6, "Invariant VITAL_MAX_HP == 6 must hold"

        request = {
            "prompt": prompt,
            "objective": objective,
            "max_latency_ms": max_latency_ms,
            "max_memory_mb": max_memory_mb,
            "require_grounding": require_grounding,
            "strategy": strategy,
            "is_shadow": is_shadow,
            **kwargs
        }

        # 1. Compile or validate plan
        if plan is None:
            exec_plan = self.planner.build_plan(request, self.nodes)
        else:
            exec_plan = plan

        valid, errors = self.planner.validate_plan(exec_plan)
        if not valid:
            raise ValueError(f"CascadePlan validation failed: {errors}")

        # 2. Execute plan
        trace = self.executor.execute(exec_plan, request)

        # 3. Closed-loop feedback cycle: OBSERVE -> COMPARE -> LEARN -> REPLAN
        feedback_summary = self.closed_loop.process_cycle(exec_plan, trace, self.router)

        # 4. Store trace in ledger (ring buffer of last 100 traces)
        self.trace_history.append(trace)
        if len(self.trace_history) > 100:
            self.trace_history.pop(0)

        result_dict = trace.to_dict()
        result_dict["closed_loop_feedback"] = feedback_summary
        return result_dict

    def benchmark(
        self,
        prompt: str = "Vysvetli mechanizmus invariantu VITAL_MAX_HP == 6",
        iterations: int = 5
    ) -> Dict[str, Any]:
        """Runs comparative statistical benchmark: single model vs multi-model cascade."""
        return self.benchmark_engine.run_comparative_benchmark(prompt=prompt, iterations=iterations)

    def get_topology(self) -> Dict[str, Any]:
        """Returns thread and worker allocation topology."""
        return self.closed_loop.topology.to_dict()

    def get_execution_matrix(self) -> Dict[str, Any]:
        """Returns current model execution matrix M."""
        return self.closed_loop.execution_matrix.to_dict()

    def perturb_execution_matrix(self, delta_rule: str) -> Dict[str, Any]:
        """Applies a perturbation ΔM to explore alternate execution layouts."""
        self.closed_loop.execution_matrix = self.closed_loop.execution_matrix.perturb(delta_rule)
        return self.closed_loop.execution_matrix.to_dict()

    def get_hypotheses(self) -> List[Dict[str, Any]]:
        """Returns empirical hypothesis ledger with validation statuses."""
        return [h.to_dict() for h in self.closed_loop.hypotheses.values()]

    def get_telemetry(self) -> Dict[str, Any]:
        """Aggregates execution metrics across all traces and nodes."""
        total_executions = len(self.trace_history)
        avg_lat = (sum(t.total_latency_ms for t in self.trace_history) / total_executions) if total_executions > 0 else 0.0
        avg_conf = (sum(t.final_confidence for t in self.trace_history) / total_executions) if total_executions > 0 else 1.0

        return {
            "total_executions": total_executions,
            "average_latency_ms": round(avg_lat, 2),
            "average_confidence": round(avg_conf, 4),
            "registered_nodes_count": len(self.nodes),
            "supported_objectives": list(ROUTING_PROFILES.keys()),
            "supported_strategies": [
                "CASCADE_SEQUENTIAL",
                "CASCADE_PARALLEL",
                "CASCADE_VOTE",
                "CASCADE_VERIFY",
                "CASCADE_FALLBACK",
                "CASCADE_SPECIALIST_ROUTE",
                "CASCADE_RAG_GROUNDED",
                "CASCADE_TOOL_EXECUTION"
            ],
            "active_hypotheses_count": len(self.closed_loop.hypotheses),
            "thread_topology": self.closed_loop.topology.to_dict(),
            "vital_max_hp": VITAL_MAX_HP
        }


# Global singleton instance
GLOBAL_MODEL_CASCADE_RUNTIME = ModelCascadeRuntime()
