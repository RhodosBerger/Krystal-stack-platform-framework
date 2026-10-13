#!/usr/bin/env python3
"""
==============================================================================
KRYSTAL-STACK: EPISTEMIC MODEL ROUTER & MULTI-OBJECTIVE RANKING
==============================================================================
Module: krystal_kernel/cascade/model_router.py
Description:
    Epistemic model routing engine that dynamically filters and ranks candidate
    model nodes using the multi-objective scoring formula:

        RouteScore_i = w1 * C_i + w2 * E_i - w3 * L_i - w4 * M_i - w5 * R_i

    where:
        C_i = capability match [0.0 .. 1.0]
        E_i = evidence / confidence estimate [0.0 .. 1.0]
        L_i = normalized latency expectation [0.0 .. 1.0]
        M_i = normalized memory footprint [0.0 .. 1.0]
        R_i = historical regret / error risk [0.0 .. 1.0]

    Routing Pipeline:
        Intent -> Candidate Nodes -> Capability Filter -> Evidence Filter
               -> Resource Filter -> Route Ranking -> Cascade Plan Selection

Constitutional Invariant:
    assert VITAL_MAX_HP == 6
==============================================================================
"""

import math
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Optional, Set, Tuple

from krystal_kernel.scene_constructor_presets import VITAL_MAX_HP
from krystal_kernel.cascade.model_node import ModelNode, ModelNodeSpec


@dataclass
class RoutingObjectiveWeights:
    """Weight vector [w1, w2, w3, w4, w5] for multi-objective route scoring."""
    name: str
    w_capability: float    # w1
    w_evidence: float      # w2
    w_latency: float       # w3
    w_memory: float        # w4
    w_regret: float        # w5
    vital_max_hp: int = VITAL_MAX_HP

    def __post_init__(self):
        assert self.vital_max_hp == 6, f"Invariant violation: vital_max_hp must be 6, got {self.vital_max_hp}"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# Canonical Weight Profiles
ROUTING_PROFILES: Dict[str, RoutingObjectiveWeights] = {
    "ACCURACY": RoutingObjectiveWeights(
        name="ACCURACY",
        w_capability=0.35,
        w_evidence=0.40,
        w_latency=0.10,
        w_memory=0.05,
        w_regret=0.10
    ),
    "LATENCY": RoutingObjectiveWeights(
        name="LATENCY",
        w_capability=0.25,
        w_evidence=0.15,
        w_latency=0.45,
        w_memory=0.10,
        w_regret=0.05
    ),
    "MEMORY": RoutingObjectiveWeights(
        name="MEMORY",
        w_capability=0.20,
        w_evidence=0.15,
        w_latency=0.10,
        w_memory=0.45,
        w_regret=0.10
    ),
    "BALANCED": RoutingObjectiveWeights(
        name="BALANCED",
        w_capability=0.25,
        w_evidence=0.25,
        w_latency=0.20,
        w_memory=0.15,
        w_regret=0.15
    ),
    "VERIFICATION_CRITICAL": RoutingObjectiveWeights(
        name="VERIFICATION_CRITICAL",
        w_capability=0.30,
        w_evidence=0.45,
        w_latency=0.05,
        w_memory=0.05,
        w_regret=0.15
    )
}


@dataclass
class NodeRouteCandidate:
    """Scored model candidate evaluated by ModelRouter."""
    node_id: str
    spec: ModelNodeSpec
    capability_score: float   # C_i
    evidence_score: float     # E_i
    latency_score: float      # L_i (normalized cost)
    memory_score: float       # M_i (normalized cost)
    regret_score: float       # R_i (normalized risk)
    route_score: float        # Final weighted score
    is_eligible: bool = True
    disqualification_reason: Optional[str] = None
    vital_max_hp: int = VITAL_MAX_HP

    def __post_init__(self):
        assert self.vital_max_hp == 6, f"Invariant violation: vital_max_hp must be 6, got {self.vital_max_hp}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_id": self.node_id,
            "role": self.spec.role,
            "backend": self.spec.backend,
            "capability_score": round(self.capability_score, 4),
            "evidence_score": round(self.evidence_score, 4),
            "latency_score": round(self.latency_score, 4),
            "memory_score": round(self.memory_score, 4),
            "regret_score": round(self.regret_score, 4),
            "route_score": round(self.route_score, 4),
            "is_eligible": self.is_eligible,
            "disqualification_reason": self.disqualification_reason,
            "vital_max_hp": self.vital_max_hp
        }


class ModelRouter:
    """
    Epistemic router determining node selection, candidate ranking,
    and fallback policies for multi-model cascades.
    """

    def __init__(self, default_objective: str = "BALANCED"):
        assert VITAL_MAX_HP == 6, "Invariant VITAL_MAX_HP == 6 must hold"
        self.default_objective = default_objective
        self.node_regret_history: Dict[str, float] = {}  # node_id -> cumulative regret/error rate

    def classify_intent(self, prompt: str) -> Dict[str, Any]:
        """
        Classifies incoming prompt into intent category, domain, and required capabilities.
        """
        lower = prompt.lower()
        required_caps: Set[str] = set()

        if any(w in lower for w in ["kód", "code", "python", "script", "def ", "class ", "rust", "function", "gcode", "shader"]):
            intent = "CODE_GENERATION"
            required_caps.update(["code_generation"])
        elif any(w in lower for w in ["over", "audit", "invariant", "verifikuj", "paradox", "safe", "kontrola"]):
            intent = "EPISTEMIC_VERIFICATION"
            required_caps.update(["safety_gate", "invariant_assertion"])
        elif any(w in lower for w in ["výkon", "profiluj", "isa", "roofline", "latencia", "benchmark", "hw"]):
            intent = "HARDWARE_PROFILING"
            required_caps.update(["fast_dispatch"])
        elif any(w in lower for w in ["prečo", "vysvetli", "analyzuj", "dôvod", "argumentuj", "hypotéza", "reason"]):
            intent = "DEEP_REASONING"
            required_caps.update(["causal_reasoning", "counterfactual_deduction"])
        else:
            intent = "CORE_KNOWLEDGE_RAG"
            required_caps.update(["rag_retrieval", "core_knowledge"])

        # Determine if grounding is explicitly requested or implied
        requires_grounding = intent in ["CORE_KNOWLEDGE_RAG", "EPISTEMIC_VERIFICATION", "DEEP_REASONING"]

        return {
            "intent": intent,
            "required_capabilities": required_caps,
            "requires_grounding": requires_grounding,
            "vital_max_hp": VITAL_MAX_HP
        }

    def calculate_route_score(
        self,
        node: ModelNode,
        intent: str,
        required_caps: Set[str],
        objective: str = "BALANCED",
        max_latency_ms: float = 200.0,
        max_memory_mb: float = 4096.0
    ) -> NodeRouteCandidate:
        """
        Applies epistemic multi-objective scoring formula:
        RouteScore_i = w1*C_i + w2*E_i - w3*L_i - w4*M_i - w5*R_i
        """
        weights = ROUTING_PROFILES.get(objective, ROUTING_PROFILES["BALANCED"])
        spec = node.spec

        # 1. Capability Match (C_i in [0, 1])
        if not required_caps:
            c_score = 0.8
        else:
            intersection = spec.capabilities.intersection(required_caps)
            c_score = len(intersection) / float(len(required_caps))

        # Role alignment bonus
        role_alignment_map = {
            "CODE_GENERATION": "CODER",
            "EPISTEMIC_VERIFICATION": "VERIFIER",
            "HARDWARE_PROFILING": "CLASSIFIER",
            "DEEP_REASONING": "REASONER",
            "CORE_KNOWLEDGE_RAG": "RETRIEVER"
        }
        if spec.role == role_alignment_map.get(intent):
            c_score = min(1.0, c_score + 0.3)

        # 2. Evidence / Confidence Estimate (E_i in [0, 1])
        # Higher confidence for dedicated verifiers and retrieval-grounded nodes
        base_confidence = 0.85
        if spec.role in ["VERIFIER", "RETRIEVER"]:
            base_confidence = 0.95
        elif "INT4" in str(spec.quantization):
            base_confidence = 0.82
        elif spec.quantization in ["FP16", "FP32"]:
            base_confidence = 0.94
        e_score = base_confidence

        # 3. Normalized Latency Expectation (L_i in [0, 1])
        cost_est = node.estimate_cost({"prompt": "benchmark"})
        est_lat = cost_est.get("estimated_latency_ms", 15.0)
        l_score = min(1.0, est_lat / max(1.0, max_latency_ms))

        # 4. Normalized Memory Cost (M_i in [0, 1])
        est_mem = spec.memory_budget_mb
        m_score = min(1.0, est_mem / max(1.0, max_memory_mb))

        # 5. Historical Regret / Risk (R_i in [0, 1])
        r_score = self.node_regret_history.get(spec.node_id, 0.05)

        # Compute RouteScore_i = w1*C_i + w2*E_i - w3*L_i - w4*M_i - w5*R_i
        route_score = (
            weights.w_capability * c_score +
            weights.w_evidence * e_score -
            weights.w_latency * l_score -
            weights.w_memory * m_score -
            weights.w_regret * r_score
        )

        # Check disqualification filters
        is_eligible = True
        reason = None
        if est_mem > max_memory_mb:
            is_eligible = False
            reason = f"Memory budget exceeded: {est_mem}MB > {max_memory_mb}MB"
        elif est_lat > (max_latency_ms * 1.5):
            is_eligible = False
            reason = f"Latency budget exceeded: {est_lat}ms > {max_latency_ms * 1.5}ms"

        return NodeRouteCandidate(
            node_id=spec.node_id,
            spec=spec,
            capability_score=c_score,
            evidence_score=e_score,
            latency_score=l_score,
            memory_score=m_score,
            regret_score=r_score,
            route_score=route_score,
            is_eligible=is_eligible,
            disqualification_reason=reason,
            vital_max_hp=VITAL_MAX_HP
        )

    def rank_nodes(
        self,
        nodes: List[ModelNode],
        intent_info: Dict[str, Any],
        objective: str = "BALANCED",
        max_latency_ms: float = 200.0,
        max_memory_mb: float = 4096.0
    ) -> List[NodeRouteCandidate]:
        """
        Filters and ranks candidate nodes based on capabilities, resources, and score.
        """
        intent = intent_info.get("intent", "CORE_KNOWLEDGE_RAG")
        required_caps = set(intent_info.get("required_capabilities", []))

        candidates: List[NodeRouteCandidate] = []
        for node in nodes:
            c = self.calculate_route_score(
                node=node,
                intent=intent,
                required_caps=required_caps,
                objective=objective,
                max_latency_ms=max_latency_ms,
                max_memory_mb=max_memory_mb
            )
            candidates.append(c)

        # Sort descending by route score
        candidates.sort(key=lambda x: (x.is_eligible, x.route_score), reverse=True)
        return candidates

    def select_node(
        self,
        candidates: List[NodeRouteCandidate],
        preferred_role: Optional[str] = None
    ) -> Optional[NodeRouteCandidate]:
        """Selects the best eligible candidate node, strictly filtered by role if preferred_role is specified."""
        if preferred_role:
            for c in candidates:
                if c.is_eligible and c.spec.role == preferred_role:
                    return c
            return None

        # If no role preference, take top eligible
        for c in candidates:
            if c.is_eligible:
                return c
        return candidates[0] if candidates else None

    def select_fallback(
        self,
        failed_node_id: str,
        candidates: List[NodeRouteCandidate]
    ) -> Optional[NodeRouteCandidate]:
        """
        Selects a secondary/fallback node if the primary node times out or fails.
        Penalizes failed node regret.
        """
        # Update regret history for failed node
        current_regret = self.node_regret_history.get(failed_node_id, 0.0)
        self.node_regret_history[failed_node_id] = min(1.0, current_regret + 0.25)

        for c in candidates:
            if c.node_id != failed_node_id and c.is_eligible:
                return c
        return None
