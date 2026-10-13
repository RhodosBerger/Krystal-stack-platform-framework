#!/usr/bin/env python3
"""
==============================================================================
KRYSTAL-STACK: CASCADE PLANNER & EXECUTION GRAPH COMPILER
==============================================================================
Module: krystal_kernel/cascade/cascade_planner.py
Description:
    Compiles, validates, estimates, and optimizes multi-model execution graphs (plans).
    Connects specialized ModelNodes (IntentClassifier, CoreRag, Reasoner,
    Coder, Verifier, Synthesizer) via conditional directed edges with
    timeout and fallback specifications.

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
    timeout_ms: float = 120.0
    fallback: Optional[str] = None
    evidence_requirement: Optional[str] = None
    vital_max_hp: int = VITAL_MAX_HP

    def __post_init__(self):
        assert self.vital_max_hp == 6, f"Invariant violation: vital_max_hp must be 6, got {self.vital_max_hp}"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CascadePlan:
    """Canonical multi-model execution plan graph."""
    plan_id: str
    nodes: List[ModelNodeSpec]
    edges: List[CascadeEdge]
    strategy: str                   # CASCADE_SEQUENTIAL, CASCADE_PARALLEL, CASCADE_VOTE, CASCADE_VERIFY, etc.
    estimated_latency_ms: float = 0.0
    estimated_memory_mb: float = 0.0
    objective: str = "BALANCED"
    require_grounding: bool = True
    vital_max_hp: int = VITAL_MAX_HP

    def __post_init__(self):
        assert self.vital_max_hp == 6, f"Invariant violation: vital_max_hp must be 6, got {self.vital_max_hp}"
        if not self.plan_id:
            self.plan_id = f"PLAN_{uuid.uuid4().hex[:10].upper()}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "strategy": self.strategy,
            "objective": self.objective,
            "require_grounding": self.require_grounding,
            "estimated_latency_ms": round(self.estimated_latency_ms, 2),
            "estimated_memory_mb": round(self.estimated_memory_mb, 2),
            "nodes": [n.to_dict() for n in self.nodes],
            "edges": [e.to_dict() for e in self.edges],
            "vital_max_hp": self.vital_max_hp
        }


class CascadePlanner:
    """
    Compiles requests and available node catalogs into deterministic execution plans.
    """

    def __init__(self, router: Optional[ModelRouter] = None):
        assert VITAL_MAX_HP == 6, "Invariant VITAL_MAX_HP == 6 must hold"
        self.router = router or ModelRouter()

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
        strategy_override = request.get("strategy")
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
        ranked_dict = {c.node_id: c for c in ranked}

        # 3. Determine Strategy
        if strategy_override:
            strategy = strategy_override
        elif intent == "CODE_GENERATION":
            strategy = "CASCADE_SPECIALIST_ROUTE"
        elif intent == "EPISTEMIC_VERIFICATION":
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
            # IntentClassifier -> (Retriever || Reasoner) -> Merger/Synthesizer
            if classifier_c:
                plan_nodes.append(classifier_c.spec)
            if retriever_c:
                plan_nodes.append(retriever_c.spec)
            if reasoner_c:
                plan_nodes.append(reasoner_c.spec)
            if synthesizer_c:
                plan_nodes.append(synthesizer_c.spec)

            if classifier_c and retriever_c:
                plan_edges.append(CascadeEdge(source=classifier_c.node_id, target=retriever_c.node_id, condition="ALWAYS", timeout_ms=80.0))
            if classifier_c and reasoner_c:
                plan_edges.append(CascadeEdge(source=classifier_c.node_id, target=reasoner_c.node_id, condition="ALWAYS", timeout_ms=100.0))
            if retriever_c and synthesizer_c:
                plan_edges.append(CascadeEdge(source=retriever_c.node_id, target=synthesizer_c.node_id, condition="ON_SUCCESS", timeout_ms=60.0))
            if reasoner_c and synthesizer_c:
                plan_edges.append(CascadeEdge(source=reasoner_c.node_id, target=synthesizer_c.node_id, condition="ON_SUCCESS", timeout_ms=60.0))

        elif strategy == "CASCADE_VOTE":
            # Broadcast to multiple reasoners/coders and vote
            if classifier_c:
                plan_nodes.append(classifier_c.spec)
            for c in ranked:
                if c.spec.role in ["REASONER", "CODER"] and c.is_eligible:
                    plan_nodes.append(c.spec)
                    if classifier_c:
                        plan_edges.append(CascadeEdge(source=classifier_c.node_id, target=c.node_id, condition="ALWAYS", timeout_ms=100.0))
            if verifier_c:
                plan_nodes.append(verifier_c.spec)
            if synthesizer_c:
                plan_nodes.append(synthesizer_c.spec)

        elif strategy == "CASCADE_SPECIALIST_ROUTE":
            # Classifier -> (CoreRag if grounding) -> CodingSpecialist (with fallback to Reasoner) -> Verifier -> Synthesizer
            if classifier_c:
                plan_nodes.append(classifier_c.spec)
            if require_grounding and retriever_c:
                plan_nodes.append(retriever_c.spec)
                if classifier_c:
                    plan_edges.append(CascadeEdge(source=classifier_c.node_id, target=retriever_c.node_id, condition="ON_SUCCESS", timeout_ms=80.0))
            
            specialist = coder_c if coder_c else reasoner_c
            if specialist:
                plan_nodes.append(specialist.spec)
                prev_node = retriever_c.node_id if (require_grounding and retriever_c) else (classifier_c.node_id if classifier_c else None)
                if prev_node:
                    plan_edges.append(CascadeEdge(
                        source=prev_node,
                        target=specialist.node_id,
                        condition="ON_SUCCESS",
                        timeout_ms=120.0,
                        fallback=fallback_id
                    ))
            if verifier_c and specialist:
                plan_nodes.append(verifier_c.spec)
                plan_edges.append(CascadeEdge(source=specialist.node_id, target=verifier_c.node_id, condition="ON_SUCCESS", timeout_ms=50.0))
            if synthesizer_c and verifier_c:
                plan_nodes.append(synthesizer_c.spec)
                plan_edges.append(CascadeEdge(source=verifier_c.node_id, target=synthesizer_c.node_id, condition="ON_SUCCESS", timeout_ms=50.0))

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
                    timeout_ms=90.0,
                    fallback=fb
                ))

        plan = CascadePlan(
            plan_id="",
            nodes=plan_nodes,
            edges=plan_edges,
            strategy=strategy,
            objective=objective,
            require_grounding=require_grounding,
            vital_max_hp=VITAL_MAX_HP
        )

        plan.estimated_latency_ms = self.estimate_latency(plan, available_nodes)
        plan.estimated_memory_mb = self.estimate_memory(plan, available_nodes)

        # Optimize if budget tight and strategy was not explicitly requested
        if not strategy_override and (plan.estimated_latency_ms > max_lat or plan.estimated_memory_mb > max_mem):
            plan = self.optimize_plan(plan, available_nodes, max_lat, max_mem)

        return plan

    def validate_plan(self, plan: CascadePlan) -> Tuple[bool, List[str]]:
        """
        Validates graph structure, absence of orphan edges, and invariant assertion.
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

        return (len(errors) == 0, errors)

    def estimate_latency(self, plan: CascadePlan, available_nodes: Dict[str, ModelNode]) -> float:
        """Estimates end-to-end latency according to plan strategy."""
        total = 0.0
        if plan.strategy == "CASCADE_PARALLEL":
            # Max of parallel branch + sequential stages
            latencies = []
            for n in plan.nodes:
                node_obj = available_nodes.get(n.node_id)
                cost = node_obj.estimate_cost({}) if node_obj else {"estimated_latency_ms": 15.0}
                latencies.append(cost["estimated_latency_ms"])
            total = sum(latencies[:1]) + max(latencies[1:-1] if len(latencies) > 2 else [0.0]) + sum(latencies[-1:])
        else:
            for n in plan.nodes:
                node_obj = available_nodes.get(n.node_id)
                cost = node_obj.estimate_cost({}) if node_obj else {"estimated_latency_ms": 15.0}
                total += cost["estimated_latency_ms"]
        return round(total, 2)

    def estimate_memory(self, plan: CascadePlan, available_nodes: Dict[str, ModelNode]) -> float:
        """Estimates peak concurrent memory utilization."""
        if plan.strategy == "CASCADE_PARALLEL":
            # Concurrent nodes sum up
            return round(sum(n.memory_budget_mb for n in plan.nodes), 2)
        else:
            # Sequential pipeline: max of single node footprint plus buffer
            max_single = max((n.memory_budget_mb for n in plan.nodes), default=256.0)
            return round(max_single * 1.15, 2)

    def optimize_plan(
        self,
        plan: CascadePlan,
        available_nodes: Dict[str, ModelNode],
        max_lat: float,
        max_mem: float
    ) -> CascadePlan:
        """
        Applies graph pruning and node shedding to satisfy tight resource limits.
        """
        # If latency is tight, skip secondary reasoning or unneeded intermediate nodes
        optimized_nodes = [n for n in plan.nodes]
        if plan.estimated_latency_ms > max_lat and len(optimized_nodes) > 3:
            # Drop heavy reasoning node if simple RAG retrieval is sufficient
            optimized_nodes = [n for n in optimized_nodes if n.role != "REASONER"]

        # Rebuild edges
        opt_ids = [n.node_id for n in optimized_nodes]
        opt_edges: List[CascadeEdge] = []
        for i in range(len(opt_ids) - 1):
            opt_edges.append(CascadeEdge(
                source=opt_ids[i],
                target=opt_ids[i + 1],
                condition="ON_SUCCESS",
                timeout_ms=min(80.0, max_lat / max(1, len(opt_ids)))
            ))

        new_plan = CascadePlan(
            plan_id=plan.plan_id + "_OPT",
            nodes=optimized_nodes,
            edges=opt_edges,
            strategy="CASCADE_SEQUENTIAL",
            objective=plan.objective,
            require_grounding=plan.require_grounding,
            vital_max_hp=VITAL_MAX_HP
        )
        new_plan.estimated_latency_ms = self.estimate_latency(new_plan, available_nodes)
        new_plan.estimated_memory_mb = self.estimate_memory(new_plan, available_nodes)
        return new_plan
