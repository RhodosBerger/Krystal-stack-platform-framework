#!/usr/bin/env python3
"""
==============================================================================
KRYSTAL-STACK: CASCADE PLANNER & EXECUTION GRAPH COMPILER
==============================================================================
Module: krystal_kernel/cascade/cascade_planner.py
Description:
    Compiles, validates, estimates, optimizes, and generates fallback paths
    for multi-model execution graphs (CascadePlans). Connects specialized
    ModelNodes via conditional directed CascadeEdges with transform,
    evidence requirement, timeout, and fallback specifications.

Constitutional Invariant:
    assert VITAL_MAX_HP == 6
==============================================================================
"""

import uuid
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Optional, Set, Tuple

from krystal_kernel.scene_constructor_presets import VITAL_MAX_HP
from krystal_kernel.cascade.model_node import ModelNode, ModelNodeSpec
from krystal_kernel.cascade.model_router import ModelRouter, NodeRouteCandidate


@dataclass
class CascadeEdge:
    """Directed execution edge between two nodes in a cascade graph."""
    source: str
    target: str
    condition: str = "ON_SUCCESS"   # ALWAYS, ON_SUCCESS, ON_FAILURE, REQUIRES_EVIDENCE, IF_CODE
    transform: Optional[str] = None # e.g. "EXTRACT_CODE", "SYNTHESIZE_QUERY", "IDENTITY"
    evidence_requirement: Optional[str] = None
    timeout: float = 120.0
    timeout_ms: float = 120.0       # Alias for backwards compatibility
    fallback: Optional[str] = None
    vital_max_hp: int = VITAL_MAX_HP

    def __post_init__(self):
        assert self.vital_max_hp == 6, f"Invariant violation: vital_max_hp must be 6, got {self.vital_max_hp}"
        if self.timeout != 120.0 and self.timeout_ms == 120.0:
            self.timeout_ms = self.timeout
        elif self.timeout_ms != 120.0 and self.timeout == 120.0:
            self.timeout = self.timeout_ms

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["timeout"] = self.timeout_ms
        d["timeout_ms"] = self.timeout_ms
        return d


@dataclass
class CascadePlan:
    """Canonical multi-model execution plan graph."""
    plan_id: str
    source_state_version: int = 1
    nodes: List[ModelNodeSpec] = field(default_factory=list)
    edges: List[CascadeEdge] = field(default_factory=list)
    execution_mode: str = "CASCADE_SEQUENTIAL"   # CASCADE_SEQUENTIAL, CASCADE_PARALLEL, CASCADE_VOTE, CASCADE_VERIFY, CASCADE_FALLBACK, etc.
    strategy: str = "CASCADE_SEQUENTIAL"         # Alias
    expected_latency_ms: float = 0.0
    estimated_latency_ms: float = 0.0            # Alias
    expected_memory_mb: float = 0.0
    estimated_memory_mb: float = 0.0             # Alias
    objective: str = "BALANCED"                  # MIN_LATENCY, MAX_ACCURACY, MIN_MEMORY, MAX_GROUNDING, BALANCED, EXPERIMENTAL
    evidence_requirements: List[str] = field(default_factory=list)
    fallback_policy: Dict[str, Any] = field(default_factory=dict)
    require_grounding: bool = True
    environment_snapshot: Optional[Dict[str, Any]] = None
    historical_evidence: List[Dict[str, Any]] = field(default_factory=list)
    provenance: str = "ESTIMATED"
    vital_max_hp: int = VITAL_MAX_HP

    def __post_init__(self):
        assert self.vital_max_hp == 6, f"Invariant violation: vital_max_hp must be 6, got {self.vital_max_hp}"
        if not self.plan_id:
            self.plan_id = f"PLAN_{uuid.uuid4().hex[:10].upper()}"
        if self.strategy != "CASCADE_SEQUENTIAL" and self.execution_mode == "CASCADE_SEQUENTIAL":
            self.execution_mode = self.strategy
        elif self.execution_mode != "CASCADE_SEQUENTIAL" and self.strategy == "CASCADE_SEQUENTIAL":
            self.strategy = self.execution_mode
        if self.expected_latency_ms == 0.0 and self.estimated_latency_ms > 0.0:
            self.expected_latency_ms = self.estimated_latency_ms
        elif self.estimated_latency_ms == 0.0 and self.expected_latency_ms > 0.0:
            self.estimated_latency_ms = self.expected_latency_ms
        if self.expected_memory_mb == 0.0 and self.estimated_memory_mb > 0.0:
            self.expected_memory_mb = self.estimated_memory_mb
        elif self.estimated_memory_mb == 0.0 and self.expected_memory_mb > 0.0:
            self.estimated_memory_mb = self.expected_memory_mb

    def to_dict(self) -> Dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "source_state_version": self.source_state_version,
            "execution_mode": self.execution_mode or self.strategy,
            "strategy": self.strategy or self.execution_mode,
            "objective": self.objective,
            "require_grounding": self.require_grounding,
            "expected_latency_ms": round(self.expected_latency_ms or self.estimated_latency_ms, 2),
            "estimated_latency_ms": round(self.estimated_latency_ms or self.expected_latency_ms, 2),
            "expected_memory_mb": round(self.expected_memory_mb or self.estimated_memory_mb, 2),
            "estimated_memory_mb": round(self.estimated_memory_mb or self.expected_memory_mb, 2),
            "evidence_requirements": self.evidence_requirements,
            "fallback_policy": self.fallback_policy,
            "environment_snapshot": self.environment_snapshot,
            "historical_evidence": self.historical_evidence,
            "provenance": self.provenance,
            "nodes": [n.to_dict() for n in self.nodes],
            "edges": [e.to_dict() for e in self.edges],
            "vital_max_hp": self.vital_max_hp
        }


class CascadePlanner:
    """
    Compiles requests and available node catalogs into deterministic execution plans.
    Maintains closed-loop trace history to ground future latency and resource estimations.
    """

    def __init__(
        self,
        router: Optional[ModelRouter] = None,
        trace_history: Optional[List[Any]] = None
    ):
        assert VITAL_MAX_HP == 6, "Invariant VITAL_MAX_HP == 6 must hold"
        self.router = router or ModelRouter()
        self.trace_history: List[Any] = []
        self.node_empirical_latencies: Dict[str, List[float]] = {}
        self.node_empirical_memories: Dict[str, List[float]] = {}
        self.node_empirical_tokens: Dict[str, List[int]] = {}
        if trace_history:
            for tr in trace_history:
                self.record_trace(tr)

    def record_trace(self, trace: Any) -> None:
        """Incorporates an observed execution trace into empirical planning evidence."""
        if not trace:
            return
        if trace not in self.trace_history:
            self.trace_history.append(trace)

        node_results = getattr(trace, "node_results", [])
        for nr in node_results:
            nid = getattr(nr, "node_id", "")
            lat = getattr(nr, "latency_ms", 0.0)
            mem = getattr(nr, "memory_mb", 0.0) or getattr(nr, "memory_used", 0.0)
            in_tok = getattr(nr, "input_tokens", 0)
            out_tok = getattr(nr, "output_tokens", 0)
            total_tok = max(1, in_tok + out_tok)
            if nid and lat > 0.0:
                if nid not in self.node_empirical_latencies:
                    self.node_empirical_latencies[nid] = []
                    self.node_empirical_tokens[nid] = []
                self.node_empirical_latencies[nid].append(lat)
                self.node_empirical_tokens[nid].append(total_tok)
            if nid and mem > 0.0:
                if nid not in self.node_empirical_memories:
                    self.node_empirical_memories[nid] = []
                self.node_empirical_memories[nid].append(mem)

    @staticmethod
    def capture_environment_snapshot() -> Dict[str, Any]:
        """Captures hardware and platform runtime telemetry with granular provenance."""
        import sys
        import os
        avail_ram = 16384.0
        try:
            from krystal_kernel.slovak_llm_adapter import SlovakLLMAdapter
            adapter = SlovakLLMAdapter()
            avail_ram = adapter.get_physical_memory_available_mb()
        except Exception:
            pass

        return {
            "platform": sys.platform,
            "cpu_cores": os.cpu_count() or 4,
            "available_memory_mb": round(avail_ram, 2),
            "accelerators": ["CPU", "GPU"],
            "bus_pressure": 0.12,
            "provenance_breakdown": {
                "available_memory_mb": "MEASURED_WIN32_API" if sys.platform == "win32" else "ESTIMATED_DEFAULT",
                "cpu_cores": "OBSERVED_SYSTEM_CALL",
                "platform": "OBSERVED_SYSTEM_CALL",
                "accelerators": "ESTIMATED_DEVICE_CATALOG",
                "bus_pressure": "SYNTHETIC_HEURISTIC_CONSTANT"
            },
            "overall_provenance": "HYBRID_MEASURED_SYNTHETIC",
            "vital_max_hp": VITAL_MAX_HP
        }

    def build_plan(
        self,
        request: Dict[str, Any],
        available_nodes: Dict[str, ModelNode]
    ) -> CascadePlan:
        """
        Builds a tailored CascadePlan based on request requirements,
        intent classification, and epistemic routing.
        """
        prompt = request.get("prompt", "")
        objective = request.get("objective", "BALANCED")
        strategy_override = request.get("strategy") or request.get("execution_mode")
        max_lat = float(request.get("max_latency_ms", 200.0))
        max_mem = float(request.get("max_memory_mb", 4096.0))
        require_grounding = bool(request.get("require_grounding", True))

        # 1. Classify Intent
        intent_info = self.router.classify_intent(prompt)
        intent = intent_info["intent"]

        # 2. Rank candidate nodes
        node_list = list(available_nodes.values())
        ranked = self.router.rank_nodes(
            nodes=node_list,
            intent_info=intent_info,
            objective=objective,
            max_latency_ms=max_lat,
            max_memory_mb=max_mem
        )

        # 3. Determine Strategy
        if strategy_override:
            strategy = strategy_override
        elif intent == "CODE_GENERATION":
            strategy = "CASCADE_SPECIALIST_ROUTE"
        elif intent == "VERIFICATION":
            strategy = "CASCADE_VERIFY"
        elif intent == "DEEP_REASONING":
            strategy = "CASCADE_RAG_GROUNDED"
        else:
            strategy = "CASCADE_RAG_GROUNDED" if require_grounding else "CASCADE_SEQUENTIAL"

        # 4. Construct Nodes & Edges Pipeline
        plan_nodes: List[ModelNodeSpec] = []
        plan_edges: List[CascadeEdge] = []

        # Find key role nodes
        classifier_c = self.router.select_node(ranked, preferred_role="CLASSIFIER")
        retriever_c = self.router.select_node(ranked, preferred_role="RETRIEVER")
        reasoner_c = self.router.select_node(ranked, preferred_role="REASONER")
        coder_c = self.router.select_node(ranked, preferred_role="CODER")
        verifier_c = self.router.select_node(ranked, preferred_role="VERIFIER")
        synthesizer_c = self.router.select_node(ranked, preferred_role="SYNTHESIZER")

        # Fallback candidate for primary processing
        fallback_c = self.router.select_fallback(
            failed_node_id=coder_c.node_id if intent == "CODE_GENERATION" and coder_c else (reasoner_c.node_id if reasoner_c else ""),
            candidates=ranked
        )
        fallback_id = fallback_c.node_id if fallback_c else None

        if strategy == "CASCADE_PARALLEL":
            if classifier_c:
                plan_nodes.append(classifier_c.spec)
            if retriever_c:
                plan_nodes.append(retriever_c.spec)
            if reasoner_c:
                plan_nodes.append(reasoner_c.spec)
            if synthesizer_c:
                plan_nodes.append(synthesizer_c.spec)

            if classifier_c and retriever_c:
                plan_edges.append(CascadeEdge(source=classifier_c.node_id, target=retriever_c.node_id, condition="ALWAYS", timeout=80.0))
            if classifier_c and reasoner_c:
                plan_edges.append(CascadeEdge(source=classifier_c.node_id, target=reasoner_c.node_id, condition="ALWAYS", timeout=100.0))
            if retriever_c and synthesizer_c:
                plan_edges.append(CascadeEdge(source=retriever_c.node_id, target=synthesizer_c.node_id, condition="ON_SUCCESS", timeout=60.0))
            if reasoner_c and synthesizer_c:
                plan_edges.append(CascadeEdge(source=reasoner_c.node_id, target=synthesizer_c.node_id, condition="ON_SUCCESS", timeout=60.0))

        elif strategy == "CASCADE_VOTE":
            if classifier_c:
                plan_nodes.append(classifier_c.spec)
            for c in ranked:
                if c.spec.role in ["REASONER", "CODER"] and c.is_eligible:
                    plan_nodes.append(c.spec)
                    if classifier_c:
                        plan_edges.append(CascadeEdge(source=classifier_c.node_id, target=c.node_id, condition="ALWAYS", timeout=100.0))
            if verifier_c:
                plan_nodes.append(verifier_c.spec)
            if synthesizer_c:
                plan_nodes.append(synthesizer_c.spec)

        elif strategy == "CASCADE_SPECIALIST_ROUTE":
            if classifier_c:
                plan_nodes.append(classifier_c.spec)
            if require_grounding and retriever_c:
                plan_nodes.append(retriever_c.spec)
                if classifier_c:
                    plan_edges.append(CascadeEdge(source=classifier_c.node_id, target=retriever_c.node_id, condition="ON_SUCCESS", timeout=80.0))
            
            specialist = coder_c if coder_c else reasoner_c
            if specialist:
                plan_nodes.append(specialist.spec)
                prev_node = retriever_c.node_id if (require_grounding and retriever_c) else (classifier_c.node_id if classifier_c else None)
                if prev_node:
                    plan_edges.append(CascadeEdge(
                        source=prev_node,
                        target=specialist.node_id,
                        condition="ON_SUCCESS",
                        timeout=120.0,
                        fallback=fallback_id
                    ))
            if verifier_c and specialist:
                plan_nodes.append(verifier_c.spec)
                plan_edges.append(CascadeEdge(source=specialist.node_id, target=verifier_c.node_id, condition="ON_SUCCESS", timeout=50.0))
            if synthesizer_c and verifier_c:
                plan_nodes.append(synthesizer_c.spec)
                plan_edges.append(CascadeEdge(source=verifier_c.node_id, target=synthesizer_c.node_id, condition="ON_SUCCESS", timeout=50.0))

        else:
            # Default Canonical Krystal Cascade:
            # IntentClassifier -> CoreRagNode -> ReasoningNode -> VerificationNode -> FinalSynthesisNode
            if classifier_c:
                plan_nodes.append(classifier_c.spec)
            if retriever_c:
                plan_nodes.append(retriever_c.spec)
            if reasoner_c:
                plan_nodes.append(reasoner_c.spec)
            if verifier_c:
                plan_nodes.append(verifier_c.spec)
            if synthesizer_c:
                plan_nodes.append(synthesizer_c.spec)

            # Sequential flow
            node_ids = [n.node_id for n in plan_nodes]
            for i in range(len(node_ids) - 1):
                fb = fallback_id if node_ids[i] == (reasoner_c.node_id if reasoner_c else None) else None
                plan_edges.append(CascadeEdge(
                    source=node_ids[i],
                    target=node_ids[i + 1],
                    condition="ON_SUCCESS",
                    timeout=90.0,
                    fallback=fb
                ))

        env_snapshot = request.get("environment_snapshot") or self.capture_environment_snapshot()
        plan = CascadePlan(
            plan_id="",
            source_state_version=int(request.get("source_state_version", 1)),
            nodes=plan_nodes,
            edges=plan_edges,
            execution_mode=strategy,
            strategy=strategy,
            objective=objective,
            require_grounding=require_grounding,
            evidence_requirements=["DualGateAuthority://SafetyGate", "Invariant://VITAL_MAX_HP==6"],
            fallback_policy={"primary_fallback": fallback_id, "max_retry_hops": 2},
            environment_snapshot=env_snapshot,
            vital_max_hp=VITAL_MAX_HP
        )

        plan.expected_latency_ms = self.estimate_latency(
            plan,
            available_nodes,
            request=request,
            token_aware=bool(request.get("token_aware", False))
        )
        plan.estimated_latency_ms = plan.expected_latency_ms
        plan.expected_memory_mb = self.estimate_memory(plan, available_nodes)
        plan.estimated_memory_mb = plan.expected_memory_mb

        # Optimize if budget tight and strategy was not explicitly requested
        if not strategy_override and (plan.expected_latency_ms > max_lat or plan.expected_memory_mb > max_mem):
            plan = self.optimize_plan(plan, available_nodes, max_lat, max_mem)

        return plan

    def validate_plan(self, plan: CascadePlan) -> Tuple[bool, List[str]]:
        """
        Validates graph structure, absence of orphan edges, cycle freedom, and invariant assertion.
        """
        assert plan.vital_max_hp == 6, "Invariant VITAL_MAX_HP == 6 must hold"
        errors: List[str] = []

        node_ids = {n.node_id for n in plan.nodes}
        for edge in plan.edges:
            assert edge.vital_max_hp == 6, "Edge invariant violation"
            if edge.source not in node_ids:
                errors.append(f"Edge source '{edge.source}' not defined in plan nodes")
            if edge.target not in node_ids:
                errors.append(f"Edge target '{edge.target}' not defined in plan nodes")
            if edge.fallback and edge.fallback not in node_ids:
                errors.append(f"Edge fallback '{edge.fallback}' not defined in plan nodes")

        # Cycle detection in execution graph
        adj: Dict[str, List[str]] = {nid: [] for nid in node_ids}
        for edge in plan.edges:
            if edge.source in adj:
                adj[edge.source].append(edge.target)

        visited: Dict[str, int] = {nid: 0 for nid in node_ids}  # 0=unvisited, 1=visiting, 2=visited
        has_cycle = False

        def dfs(u: str) -> bool:
            visited[u] = 1
            for v in adj.get(u, []):
                if v not in visited:
                    continue
                if visited[v] == 1:
                    return True
                elif visited[v] == 0:
                    if dfs(v):
                        return True
            visited[u] = 2
            return False

        for nid in node_ids:
            if visited[nid] == 0:
                if dfs(nid):
                    has_cycle = True
                    break

        if has_cycle:
            errors.append("Cycle detected in cascade execution graph; DAG invariant violated")

        return (len(errors) == 0, errors)

    def estimate_latency(
        self,
        plan: CascadePlan,
        available_nodes: Optional[Dict[str, ModelNode]] = None,
        request: Optional[Dict[str, Any]] = None,
        token_aware: bool = False
    ) -> float:
        """Estimates end-to-end latency according to plan strategy with empirical closed-loop grounding."""
        nodes_dict = available_nodes or {}
        latencies: List[float] = []
        empirical_node_count = 0
        plan.historical_evidence = []
        req_dict = request or {}
        prompt_str = req_dict.get("prompt", "")
        est_req_tokens = max(16, len(prompt_str.split()) + 32) if prompt_str else 32

        for n in plan.nodes:
            nid = n.node_id
            if nid in self.node_empirical_latencies and len(self.node_empirical_latencies[nid]) > 0:
                empirical_samples = self.node_empirical_latencies[nid]
                sample_tokens = self.node_empirical_tokens.get(nid, [32] * len(empirical_samples))

                if token_aware and sum(sample_tokens) > 0:
                    slope_ms_per_tok = sum(empirical_samples) / float(sum(sample_tokens))
                    lat = max(1.0, slope_ms_per_tok * est_req_tokens)
                else:
                    lat = sum(empirical_samples) / len(empirical_samples)

                empirical_node_count += 1
                plan.historical_evidence.append({
                    "node_id": nid,
                    "empirical_samples": len(empirical_samples),
                    "mean_latency_ms": round(lat, 3),
                    "provenance": "MEASURED"
                })
            else:
                node_obj = nodes_dict.get(nid)
                lat = node_obj.estimate_latency({}) if node_obj else 15.0
            latencies.append(lat)

        if empirical_node_count == len(plan.nodes) and len(plan.nodes) > 0:
            plan.provenance = "MEASURED_EMPIRICAL"
        elif empirical_node_count > 0:
            plan.provenance = "HYBRID_EMPIRICAL"
        else:
            plan.provenance = "ESTIMATED"

        if plan.execution_mode == "CASCADE_PARALLEL" or plan.strategy == "CASCADE_PARALLEL":
            total = sum(latencies[:1]) + max(latencies[1:-1] if len(latencies) > 2 else [0.0]) + sum(latencies[-1:])
        else:
            total = sum(latencies)
        return round(total, 2)

    def estimate_memory(self, plan: CascadePlan, available_nodes: Optional[Dict[str, ModelNode]] = None) -> float:
        """Estimates peak concurrent memory utilization with empirical grounding."""
        if plan.execution_mode == "CASCADE_PARALLEL" or plan.strategy == "CASCADE_PARALLEL":
            return round(sum(n.memory_budget_mb for n in plan.nodes), 2)
        else:
            mems = []
            for n in plan.nodes:
                nid = n.node_id
                if nid in self.node_empirical_memories and len(self.node_empirical_memories[nid]) > 0:
                    mems.append(max(self.node_empirical_memories[nid]))
                else:
                    mems.append(n.memory_budget_mb)
            max_single = max(mems, default=256.0)
            return round(max_single * 1.15, 2)

    def estimate_cost(self, plan: CascadePlan, available_nodes: Optional[Dict[str, ModelNode]] = None) -> Dict[str, float]:
        """Provides holistic resource cost, latency, and memory projections."""
        lat = self.estimate_latency(plan, available_nodes)
        mem = self.estimate_memory(plan, available_nodes)
        return {
            "estimated_latency_ms": lat,
            "estimated_memory_mb": mem,
            "estimated_tokens": sum(n.context_limit // 16 for n in plan.nodes),
            "estimated_cost": round(len(plan.nodes) * 0.0002, 5),
            "estimated_cost_usd": round(len(plan.nodes) * 0.0002, 5)
        }

    def optimize_plan(
        self,
        plan: CascadePlan,
        available_nodes: Optional[Dict[str, ModelNode]] = None,
        max_lat: float = 200.0,
        max_mem: float = 4096.0
    ) -> CascadePlan:
        """
        Applies graph pruning and node shedding to satisfy tight resource limits.
        """
        nodes_dict = available_nodes or {}
        optimized_nodes = [n for n in plan.nodes]
        if (plan.expected_latency_ms or plan.estimated_latency_ms) > max_lat and len(optimized_nodes) > 3:
            optimized_nodes = [n for n in optimized_nodes if n.role != "REASONER"]

        opt_ids = [n.node_id for n in optimized_nodes]
        opt_edges: List[CascadeEdge] = []
        for i in range(len(opt_ids) - 1):
            opt_edges.append(CascadeEdge(
                source=opt_ids[i],
                target=opt_ids[i + 1],
                condition="ON_SUCCESS",
                timeout=min(80.0, max_lat / max(1, len(opt_ids)))
            ))

        new_plan = CascadePlan(
            plan_id=plan.plan_id + "_OPT",
            source_state_version=plan.source_state_version + 1,
            nodes=optimized_nodes,
            edges=opt_edges,
            execution_mode="CASCADE_SEQUENTIAL",
            strategy="CASCADE_SEQUENTIAL",
            objective=plan.objective,
            require_grounding=plan.require_grounding,
            evidence_requirements=plan.evidence_requirements,
            fallback_policy=plan.fallback_policy,
            environment_snapshot=plan.environment_snapshot,
            historical_evidence=list(plan.historical_evidence),
            provenance=plan.provenance,
            vital_max_hp=VITAL_MAX_HP
        )
        new_plan.expected_latency_ms = self.estimate_latency(new_plan, nodes_dict)
        new_plan.estimated_latency_ms = new_plan.expected_latency_ms
        new_plan.expected_memory_mb = self.estimate_memory(new_plan, nodes_dict)
        new_plan.estimated_memory_mb = new_plan.expected_memory_mb
        return new_plan

    def generate_fallback_path(
        self,
        plan: CascadePlan,
        failed_node_id: str,
        available_nodes: Optional[Dict[str, ModelNode]] = None
    ) -> CascadePlan:
        """
        Synthesizes a resilient detour plan bypassing a failed or timed-out node.
        """
        detour_nodes = [n for n in plan.nodes if n.node_id != failed_node_id]
        if not detour_nodes:
            # If all were removed, keep at least one verifier or classifier
            detour_nodes = [plan.nodes[0]]

        detour_ids = [n.node_id for n in detour_nodes]
        detour_edges: List[CascadeEdge] = []
        for i in range(len(detour_ids) - 1):
            detour_edges.append(CascadeEdge(
                source=detour_ids[i],
                target=detour_ids[i + 1],
                condition="ALWAYS",
                timeout=60.0
            ))

        fallback_plan = CascadePlan(
            plan_id=f"{plan.plan_id}_FALLBACK",
            source_state_version=plan.source_state_version + 1,
            nodes=detour_nodes,
            edges=detour_edges,
            execution_mode="CASCADE_FALLBACK",
            strategy="CASCADE_FALLBACK",
            objective=plan.objective,
            require_grounding=plan.require_grounding,
            evidence_requirements=plan.evidence_requirements,
            fallback_policy={"failed_node_id": failed_node_id, "detour_applied": True},
            environment_snapshot=plan.environment_snapshot,
            historical_evidence=list(plan.historical_evidence),
            provenance=plan.provenance,
            vital_max_hp=VITAL_MAX_HP
        )
        fallback_plan.expected_latency_ms = self.estimate_latency(fallback_plan, available_nodes)
        fallback_plan.estimated_latency_ms = fallback_plan.expected_latency_ms
        fallback_plan.expected_memory_mb = self.estimate_memory(fallback_plan, available_nodes)
        fallback_plan.estimated_memory_mb = fallback_plan.expected_memory_mb
        return fallback_plan
