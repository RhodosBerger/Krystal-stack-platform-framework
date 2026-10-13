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


class ModelCascadeRuntime:
    """
    Central orchestration fabric managing node catalogs, epistemic routing,
    graph compilation, and execution across 8 cascade modes.
    """

    def __init__(self):
        assert VITAL_MAX_HP == 6, "Invariant VITAL_MAX_HP == 6 must hold"
        self.router = ModelRouter()
        self.planner = CascadePlanner(router=self.router)
        self.nodes: Dict[str, ModelNode] = {}
        self.executor = CascadeExecutor(self.nodes)
        self.trace_history: List[CascadeExecutionTrace] = []

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
        **kwargs
    ) -> Dict[str, Any]:
        """
        Primary execution entry point.
        Compiles plan (if not provided), executes across registered nodes,
        enforces Dual-Gate invariants, and returns complete forensic trace.
        """
        assert VITAL_MAX_HP == 6, "Invariant VITAL_MAX_HP == 6 must hold"

        request = {
            "prompt": prompt,
            "objective": objective,
            "max_latency_ms": max_latency_ms,
            "max_memory_mb": max_memory_mb,
            "require_grounding": require_grounding,
            "strategy": strategy,
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

        # 3. Store trace in ledger (ring buffer of last 100 traces)
        self.trace_history.append(trace)
        if len(self.trace_history) > 100:
            self.trace_history.pop(0)

        return trace.to_dict()

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
            "vital_max_hp": VITAL_MAX_HP
        }


# Global singleton instance
GLOBAL_MODEL_CASCADE_RUNTIME = ModelCascadeRuntime()
