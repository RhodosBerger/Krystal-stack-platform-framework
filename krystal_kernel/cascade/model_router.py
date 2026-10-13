#!/usr/bin/env python3
"""
==============================================================================
KRYSTAL-STACK: EPISTEMIC MODEL ROUTER & MULTI-OBJECTIVE RANKING
==============================================================================
Module: krystal_kernel/cascade/model_router.py
Description:
    Epistemic model routing engine that dynamically classifies intent, ranks
    candidate model nodes, selects optimal nodes, and synthesizes fallback paths
    using the calibrated multi-objective scoring formula:

        RouteScore_i =
            w1 * capability_match
          + w2 * evidence_score
          + w3 * calibration_score
          - w4 * latency_cost
          - w5 * memory_cost
          - w6 * failure_risk

    Supports dynamic policy calibration, per-node historical regret updates,
    and specialist routing across code, reasoning, retrieval, and verification.

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
    """Weight vector [w1..w6] for multi-objective route scoring."""
    name: str
    w1_capability: float = 0.25   # capability_match
    w2_evidence: float = 0.25     # evidence_score
    w3_calibration: float = 0.15  # calibration_score
    w4_latency: float = 0.15      # latency_cost
    w5_memory: float = 0.10       # memory_cost
    w6_failure_risk: float = 0.10 # failure_risk
    # Backwards-compatible aliases
    w_capability: float = 0.25
    w_evidence: float = 0.25
    w_calibration: float = 0.15
    w_latency: float = 0.15
    w_memory: float = 0.10
    w_regret: float = 0.10
    vital_max_hp: int = VITAL_MAX_HP

    def __post_init__(self):
        assert self.vital_max_hp == 6, f"Invariant violation: vital_max_hp must be 6, got {self.vital_max_hp}"
        if self.w_capability != 0.25 and self.w1_capability == 0.25:
            self.w1_capability = self.w_capability
        elif self.w1_capability != 0.25 and self.w_capability == 0.25:
            self.w_capability = self.w1_capability
        if self.w_evidence != 0.25 and self.w2_evidence == 0.25:
            self.w2_evidence = self.w_evidence
        elif self.w2_evidence != 0.25 and self.w_evidence == 0.25:
            self.w_evidence = self.w2_evidence
        if self.w_latency != 0.15 and self.w4_latency == 0.15:
            self.w4_latency = self.w_latency
        elif self.w4_latency != 0.15 and self.w_latency == 0.15:
            self.w_latency = self.w4_latency
        if self.w_memory != 0.10 and self.w5_memory == 0.10:
            self.w5_memory = self.w_memory
        elif self.w5_memory != 0.10 and self.w_memory == 0.10:
            self.w_memory = self.w5_memory
        if self.w_regret != 0.10 and self.w6_failure_risk == 0.10:
            self.w6_failure_risk = self.w_regret
        elif self.w6_failure_risk != 0.10 and self.w_regret == 0.10:
            self.w_regret = self.w6_failure_risk

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# Canonical Weight Profiles
ROUTING_PROFILES: Dict[str, RoutingObjectiveWeights] = {
    "ACCURACY": RoutingObjectiveWeights(
        name="ACCURACY",
        w1_capability=0.30,
        w2_evidence=0.35,
        w3_calibration=0.15,
        w4_latency=0.08,
        w5_memory=0.04,
        w6_failure_risk=0.08,
        w_capability=0.30,
        w_evidence=0.35,
        w_latency=0.08,
        w_memory=0.04,
        w_regret=0.08
    ),
    "MAX_ACCURACY": RoutingObjectiveWeights(
        name="MAX_ACCURACY",
        w1_capability=0.30,
        w2_evidence=0.35,
        w3_calibration=0.15,
        w4_latency=0.08,
        w5_memory=0.04,
        w6_failure_risk=0.08,
        w_capability=0.30,
        w_evidence=0.35,
        w_latency=0.08,
        w_memory=0.04,
        w_regret=0.08
    ),
    "LATENCY": RoutingObjectiveWeights(
        name="LATENCY",
        w1_capability=0.20,
        w2_evidence=0.10,
        w3_calibration=0.10,
        w4_latency=0.45,
        w5_memory=0.05,
        w6_failure_risk=0.10,
        w_capability=0.20,
        w_evidence=0.10,
        w_latency=0.45,
        w_memory=0.05,
        w_regret=0.10
    ),
    "MIN_LATENCY": RoutingObjectiveWeights(
        name="MIN_LATENCY",
        w1_capability=0.20,
        w2_evidence=0.10,
        w3_calibration=0.10,
        w4_latency=0.45,
        w5_memory=0.05,
        w6_failure_risk=0.10,
        w_capability=0.20,
        w_evidence=0.10,
        w_latency=0.45,
        w_memory=0.05,
        w_regret=0.10
    ),
    "MEMORY": RoutingObjectiveWeights(
        name="MEMORY",
        w1_capability=0.20,
        w2_evidence=0.10,
        w3_calibration=0.10,
        w4_latency=0.10,
        w5_memory=0.40,
        w6_failure_risk=0.10,
        w_capability=0.20,
        w_evidence=0.10,
        w_latency=0.10,
        w_memory=0.40,
        w_regret=0.10
    ),
    "MIN_MEMORY": RoutingObjectiveWeights(
        name="MIN_MEMORY",
        w1_capability=0.20,
        w2_evidence=0.10,
        w3_calibration=0.10,
        w4_latency=0.10,
        w5_memory=0.40,
        w6_failure_risk=0.10,
        w_capability=0.20,
        w_evidence=0.10,
        w_latency=0.10,
        w_memory=0.40,
        w_regret=0.10
    ),
    "MAX_GROUNDING": RoutingObjectiveWeights(
        name="MAX_GROUNDING",
        w1_capability=0.20,
        w2_evidence=0.45,
        w3_calibration=0.15,
        w4_latency=0.05,
        w5_memory=0.05,
        w6_failure_risk=0.10,
        w_capability=0.20,
        w_evidence=0.45,
        w_latency=0.05,
        w_memory=0.05,
        w_regret=0.10
    ),
    "BALANCED": RoutingObjectiveWeights(
        name="BALANCED",
        w1_capability=0.25,
        w2_evidence=0.25,
        w3_calibration=0.10,
        w4_latency=0.15,
        w5_memory=0.15,
        w6_failure_risk=0.10,
        w_capability=0.25,
        w_evidence=0.25,
        w_latency=0.15,
        w_memory=0.15,
        w_regret=0.10
    ),
    "EXPERIMENTAL": RoutingObjectiveWeights(
        name="EXPERIMENTAL",
        w1_capability=0.20,
        w2_evidence=0.20,
        w3_calibration=0.20,
        w4_latency=0.15,
        w5_memory=0.15,
        w6_failure_risk=0.10,
        w_capability=0.20,
        w_evidence=0.20,
        w_latency=0.15,
        w_memory=0.15,
        w_regret=0.10
    ),
    "VERIFICATION_CRITICAL": RoutingObjectiveWeights(
        name="VERIFICATION_CRITICAL",
        w1_capability=0.25,
        w2_evidence=0.35,
        w3_calibration=0.20,
        w4_latency=0.05,
        w5_memory=0.05,
        w6_failure_risk=0.10,
        w_capability=0.25,
        w_evidence=0.35,
        w_latency=0.05,
        w_memory=0.05,
        w_regret=0.10
    )
}


@dataclass
class NodeRouteCandidate:
    """Scored model candidate evaluated by ModelRouter."""
    node_id: str
    spec: ModelNodeSpec
    capability_score: float   # w1 * capability_match
    evidence_score: float     # w2 * evidence_score
    latency_score: float      # w4 * latency_cost
    memory_score: float       # w5 * memory_cost
    regret_score: float       # w6 * failure_risk
    route_score: float        # Final weighted score
    calibration_score: float = 0.90 # w3 * calibration_score
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
            "calibration_score": round(self.calibration_score, 4),
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
            required_caps.update(["code_generation", "code_synthesis"])
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
        RouteScore_i = w1*C_i + w2*E_i + w3*Cal_i - w4*L_i - w5*M_i - w6*R_i
        """
        weights = ROUTING_PROFILES.get(objective, ROUTING_PROFILES["BALANCED"])
        spec = node.spec

        # 1. Capability Match (C_i in [0, 1])
        if not required_caps:
            c_score = 0.8
        else:
            intersection = spec.capabilities.intersection(required_caps)
            c_score = len(intersection) / float(len(required_caps))

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
        base_confidence = 0.85
        if spec.role in ["VERIFIER", "RETRIEVER"]:
            base_confidence = 0.95
        elif "INT4" in str(spec.quantization):
            base_confidence = 0.82
        elif spec.quantization in ["FP16", "FP32"]:
            base_confidence = 0.94
        e_score = base_confidence

        # 3. Calibration Score (Cal_i in [0, 1])
        cal_score = 0.90 if spec.role in ["VERIFIER", "RETRIEVER"] else 0.82

        # 4. Normalized Latency Cost (L_i in [0, 1])
        est_lat = node.estimate_latency({"prompt": "benchmark"})
        l_score = min(1.0, est_lat / max(1.0, max_latency_ms))

        # 5. Normalized Memory Cost (M_i in [0, 1])
        est_mem = spec.memory_budget_mb
        m_score = min(1.0, est_mem / max(1.0, max_memory_mb))

        # 6. Historical Regret / Risk (R_i in [0, 1])
        r_score = self.node_regret_history.get(spec.node_id, 0.05)

        # RouteScore_i = w1*C_i + w2*E_i + w3*Cal_i - w4*L_i - w5*M_i - w6*R_i
        route_score = (
            weights.w1_capability * c_score +
            weights.w2_evidence * e_score +
            weights.w3_calibration * cal_score -
            weights.w4_latency * l_score -
            weights.w5_memory * m_score -
            weights.w6_failure_risk * r_score
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
            calibration_score=cal_score,
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

        candidates.sort(key=lambda x: (x.is_eligible, x.route_score), reverse=True)
        return candidates

    def select_node(
        self,
        candidates: List[Any],
        preferred_role: Optional[Any] = None
    ) -> Optional[Any]:
        """Selects the best eligible candidate node, strictly filtered by role if preferred_role is specified."""
        target_role = None
        if isinstance(preferred_role, str):
            target_role = preferred_role
        elif isinstance(preferred_role, dict):
            intent_val = preferred_role.get("intent", "")
            intent_map = {
                "CODE_GENERATION": "CODER",
                "CORE_KNOWLEDGE_RAG": "RETRIEVER",
                "KNOWLEDGE_RETRIEVAL": "RETRIEVER",
                "DEEP_REASONING": "REASONER",
                "REASONING": "REASONER",
                "VERIFICATION": "VERIFIER",
                "FINAL_SYNTHESIS": "SYNTHESIZER"
            }
            target_role = intent_map.get(intent_val)

        if target_role:
            for c in candidates:
                role = c.spec.role if hasattr(c, "spec") else getattr(c, "role", "")
                eligible = getattr(c, "is_eligible", True)
                if eligible and role == target_role:
                    return c

        for c in candidates:
            if getattr(c, "is_eligible", True):
                return c
        return candidates[0] if candidates else None

    def select_fallback(
        self,
        failed_node_id: str,
        candidates: List[Any]
    ) -> Optional[Any]:
        """
        Selects a secondary/fallback node if the primary node times out or fails.
        Penalizes failed node regret.
        """
        current_regret = self.node_regret_history.get(failed_node_id, 0.0)
        self.node_regret_history[failed_node_id] = min(1.0, current_regret + 0.25)

        for c in candidates:
            n_id = getattr(c, "node_id", None) or (c.spec.node_id if hasattr(c, "spec") else "")
            eligible = getattr(c, "is_eligible", True)
            if n_id != failed_node_id and eligible:
                return c
        return None

    def select_specialist(
        self,
        domain: str,
        candidates: Optional[List[Any]] = None
    ) -> Optional[Any]:
        """
        Selects the most suitable specialist node for a domain ('code', 'reasoning', 'rag', 'verifier', 'tool').
        """
        domain_role_map = {
            "code": "CODER",
            "coding": "CODER",
            "code_generation": "CODER",
            "reasoning": "REASONER",
            "deep_reasoning": "REASONER",
            "logic": "REASONER",
            "rag": "RETRIEVER",
            "retrieval": "RETRIEVER",
            "core_knowledge_rag": "RETRIEVER",
            "knowledge_retrieval": "RETRIEVER",
            "verifier": "VERIFIER",
            "verification": "VERIFIER",
            "tool": "TOOL",
            "classifier": "CLASSIFIER",
            "synthesis": "SYNTHESIZER"
        }
        target_role = domain_role_map.get(domain.lower(), "REASONER")
        cand_list = candidates or []
        for c in cand_list:
            role = c.spec.role if hasattr(c, "spec") else getattr(c, "role", "")
            eligible = getattr(c, "is_eligible", True)
            if eligible and role == target_role:
                return c
        return cand_list[0] if cand_list else None
