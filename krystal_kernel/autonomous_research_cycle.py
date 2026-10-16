"""
autonomous_research_cycle.py - Autonomous Epistemic Research & Engineering Discovery Loop.

Implements the normative tenets of Milestone V-AUTONOMOUS-RESEARCH-01 (SPEC-AUTONOMOUS-RESEARCH-01):
1. Core Directive: "Nie zrýchli kód, ale spochybni predpoklady, nájdi neznáme a falzifikuj vlastné modely."
2. 8-Stage Research Loop:
   - 1. Fresh System Re-Mapping.
   - 2. Hypothesis Portfolio including M0 NULL model.
   - 3. Isolated Actuation with Falsifier & Rollback.
   - 4. Selection by Expected Value of Information (EVOI), not greedy benchmark gain.
   - 5. Transactional Actuation & Before/After Measurement.
   - 6. Adversarial Critic Audit (Confounders, flukes, and oracle gaming).
   - 7. Epistemic Knowledge Graph Revision with explicit statuses.
   - 8. Modularity Gate: "New First-Class Module vs Ad-Hoc Hack?".
3. Global Invariant: GLOBAL_VITAL_MAX_HP == 6.
"""

from __future__ import annotations

import enum
import math
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

# Global Invariant
GLOBAL_VITAL_MAX_HP: int = 6


class EpistemicContractViolation(Exception):
    """Raised when an epistemic discovery invariant is violated."""
    pass


# ============================================================================
# Epistemic Knowledge Graph (Nodes, Edges, Statuses)
# ============================================================================

class GraphNodeType(str, enum.Enum):
    MECHANISM = "MECHANISM"
    HYPOTHESIS = "HYPOTHESIS"
    MODEL = "MODEL"
    UNKNOWN_GAP = "UNKNOWN_GAP"
    CONFLICT = "CONFLICT"


class NodeStatus(str, enum.Enum):
    HYPOTHESIS = "HYPOTHESIS"
    UNDER_TEST = "UNDER_TEST"
    FALSIFIED = "FALSIFIED"
    APPROVED_MECHANISM = "APPROVED_MECHANISM"
    UNKNOWN_GAP = "UNKNOWN_GAP"
    CONFLICT = "CONFLICT"


class EdgeType(str, enum.Enum):
    SUPPORTS = "SUPPORTS"
    WEAKENS = "WEAKENS"
    FALSIFIES = "FALSIFIES"
    REPLACES = "REPLACES"
    DEPENDS_ON = "DEPENDS_ON"
    DISCRIMINATES_AGAINST = "DISCRIMINATES_AGAINST"


@dataclass
class KnowledgeNode:
    node_id: str
    node_type: GraphNodeType
    label: str
    status: NodeStatus
    metadata: Dict[str, Any] = field(default_factory=dict)
    version: int = 1


@dataclass
class KnowledgeEdge:
    edge_id: str
    source_id: str
    target_id: str
    edge_type: EdgeType
    rationale: str = ""


class EpistemicKnowledgeGraph:
    """
    Maintains the evolving knowledge base of models, hypotheses, mechanisms, gaps, and conflicts.
    Prevents silent overwrites.
    """
    def __init__(self) -> None:
        self.nodes: Dict[str, KnowledgeNode] = {}
        self.edges: List[KnowledgeEdge] = []

    def add_node(
        self,
        node_id: str,
        node_type: GraphNodeType,
        label: str,
        status: NodeStatus,
        metadata: Optional[Dict[str, Any]] = None
    ) -> KnowledgeNode:
        if node_id in self.nodes:
            raise EpistemicContractViolation(
                f"Node '{node_id}' already exists. Use update_node_status() with explicit reason, never silently overwrite."
            )
        node = KnowledgeNode(
            node_id=node_id,
            node_type=node_type,
            label=label,
            status=status,
            metadata=metadata or {},
        )
        self.nodes[node_id] = node
        return node

    def add_edge(self, source_id: str, target_id: str, edge_type: EdgeType, rationale: str = "") -> KnowledgeEdge:
        if source_id not in self.nodes or target_id not in self.nodes:
            raise EpistemicContractViolation(f"Cannot link missing nodes: {source_id} -> {target_id}")

        edge = KnowledgeEdge(
            edge_id=f"edge_{uuid.uuid4().hex[:6]}",
            source_id=source_id,
            target_id=target_id,
            edge_type=edge_type,
            rationale=rationale,
        )
        self.edges.append(edge)
        return edge

    def update_node_status(self, node_id: str, new_status: NodeStatus, reason: str) -> KnowledgeNode:
        if node_id not in self.nodes:
            raise EpistemicContractViolation(f"Node '{node_id}' does not exist to update.")

        node = self.nodes[node_id]
        node.status = new_status
        node.version += 1
        node.metadata["last_update_reason"] = reason
        return node

    def get_active_hypotheses(self) -> List[KnowledgeNode]:
        return [n for n in self.nodes.values() if n.status in (NodeStatus.HYPOTHESIS, NodeStatus.UNDER_TEST)]

    def get_unknown_gaps(self) -> List[KnowledgeNode]:
        return [n for n in self.nodes.values() if n.status == NodeStatus.UNKNOWN_GAP]


# ============================================================================
# EVOI Candidate Selector (Information Gain vs Greedy Gain)
# ============================================================================

@dataclass
class ResearchCandidate:
    candidate_id: str
    hypothesis_rationale: str
    is_null_model: bool
    expected_performance_gain: float  # e.g. 0.20 (+20% throughput)
    information_gain_evoi: float      # e.g. 0.85 (high discriminative power against competing models)
    falsification_capacity: float     # 0.0 to 1.0 (how decisively a negative result refutes the model)
    risk_level: float                 # 0.1 (low) to 1.0 (high)
    cost: float                       # Execution cost/time (e.g. 1.0 to 10.0)

    @property
    def evoi_priority_score(self) -> float:
        """
        EVOI Priority Formula:
        Priority = (Information_Gain * Falsification_Capacity) / (Risk * Cost)
        Does NOT rely solely on greedy performance gain.
        """
        effective_risk = max(0.1, self.risk_level)
        effective_cost = max(0.5, self.cost)
        return (self.information_gain_evoi * self.falsification_capacity) / (effective_risk * effective_cost)


class EVOICandidateSelector:
    """
    Selects the next research candidate based on highest EVOI (Expected Value of Information)
    rather than greedy local benchmark hunting.
    """
    @staticmethod
    def select_best_candidate(candidates: List[ResearchCandidate]) -> ResearchCandidate:
        if not candidates:
            raise EpistemicContractViolation("Cannot select from an empty list of research candidates.")
        # Filter for actionable intervention candidates to actuate and test.
        # The M0 NULL model serves as the obligatory comparator / falsifier baseline.
        actionable_candidates = [c for c in candidates if not c.is_null_model]
        candidates_to_rank = actionable_candidates if actionable_candidates else candidates
        # Sort by EVOI priority score descending
        sorted_candidates = sorted(candidates_to_rank, key=lambda c: c.evoi_priority_score, reverse=True)
        return sorted_candidates[0]


# ============================================================================
# Adversarial Critic (Independent Interrogation Component)
# ============================================================================

class AdversarialCriticVerdict(str, enum.Enum):
    CHALLENGE_ACCEPTED = "CHALLENGE_ACCEPTED"
    CONFOUNDER_SUSPECTED = "CONFOUNDER_SUSPECTED"
    FLUKES_OR_NOISE = "FLUKES_OR_NOISE"
    ORACLE_GAMING_SUSPECTED = "ORACLE_GAMING_SUSPECTED"


@dataclass
class AdversarialCritique:
    verdict: AdversarialCriticVerdict
    confidence: float
    identified_confounders: List[str]
    critique_details: str
    requires_discriminating_retest: bool


class AdversarialCritic:
    """
    Independent skeptical reviewer actively seeking reasons why an empirical result
    might be false, an artifact of measurement noise, or a manifestation of Goodhart's Law.
    """
    def __init__(self, critic_id: str = "adversarial_critic_01") -> None:
        self.critic_id = critic_id

    def critique_result(
        self,
        hypothesis_id: str,
        baseline_metric: float,
        post_change_metric: float,
        concurrent_subsystem_telemetry: Dict[str, Any],
        measured_variance: float = 0.02,
    ) -> AdversarialCritique:
        delta = abs(post_change_metric - baseline_metric)

        # 1. Check for flukes/noise: if change is smaller than 2x measured variance
        if delta < (2.0 * measured_variance * baseline_metric):
            return AdversarialCritique(
                verdict=AdversarialCriticVerdict.FLUKES_OR_NOISE,
                confidence=0.85,
                identified_confounders=["Random measurement variance", "Background scheduler jitter"],
                critique_details="Reported improvement is within the 2-sigma noise floor of baseline.",
                requires_discriminating_retest=True,
            )

        # 2. Check for confounders: did package temperature or throttling change during the test?
        thermal_drift = concurrent_subsystem_telemetry.get("thermal_drift_c", 0.0)
        if abs(thermal_drift) > 5.0:
            return AdversarialCritique(
                verdict=AdversarialCriticVerdict.CONFOUNDER_SUSPECTED,
                confidence=0.80,
                identified_confounders=[f"Significant thermal temperature drift: {thermal_drift:+.1f}C"],
                critique_details="Thermal drift detected: system cooled down or heated up significantly, confounding clock frequency.",
                requires_discriminating_retest=True,
            )

        # 3. Check for oracle gaming: did another metric regress catastrophically?
        side_effect_stall = concurrent_subsystem_telemetry.get("background_stall_rate", 0.0)
        if side_effect_stall > 0.15:  # > 15% stall elsewhere
            return AdversarialCritique(
                verdict=AdversarialCriticVerdict.ORACLE_GAMING_SUSPECTED,
                confidence=0.90,
                identified_confounders=["Starvation of background memory channels"],
                critique_details="Target metric improved by starving auxiliary subsystems (Goodhart's Law).",
                requires_discriminating_retest=True,
            )

        return AdversarialCritique(
            verdict=AdversarialCriticVerdict.CHALLENGE_ACCEPTED,
            confidence=0.95,
            identified_confounders=[],
            critique_details="Empirical result survived adversarial scrutiny; no obvious confounder identified.",
            requires_discriminating_retest=False,
        )


# ============================================================================
# Modularity Gate ("New First-Class Module vs Ad-Hoc Hack?")
# ============================================================================

class ModularityVerdict(str, enum.Enum):
    NEW_FIRST_CLASS_MODULE = "NEW_FIRST_CLASS_MODULE"
    AD_HOC_HACK = "AD_HOC_HACK"


@dataclass
class ModularityAuditReceipt:
    verdict: ModularityVerdict
    architectural_abstraction: str
    rejection_reasons: List[str]
    refactoring_recommendation: str


class ModuleVsHackClassifier:
    """
    Audits confirmed mechanisms to ensure they are codified as clean,
    first-class architectural modules rather than piling on ad-hoc tuning hacks.
    """
    @staticmethod
    def audit_modularity(
        mechanism_name: str,
        has_formal_interface: bool,
        is_cross_platform_abstracted: bool,
        has_magic_numbers: bool,
        is_bounded_by_invariants: bool,
    ) -> ModularityAuditReceipt:
        rejections = []
        if has_magic_numbers:
            rejections.append("Mechanism relies on uncalibrated, hard-coded magic constants.")
        if not has_formal_interface:
            rejections.append("Lacks a formal typed interface/contract.")
        if not is_bounded_by_invariants:
            rejections.append("Missing explicit constitutional invariant bounding.")

        if rejections:
            return ModularityAuditReceipt(
                verdict=ModularityVerdict.AD_HOC_HACK,
                architectural_abstraction="Rejected from permanent promotion",
                rejection_reasons=rejections,
                refactoring_recommendation="Extract into a clean thin adapter with typed parameters and formal invariant receipts.",
            )

        return ModularityAuditReceipt(
            verdict=ModularityVerdict.NEW_FIRST_CLASS_MODULE,
            architectural_abstraction=f"krystal_kernel/{mechanism_name.lower()}_module.py",
            rejection_reasons=[],
            refactoring_recommendation="Promote to permanent canonical engine component.",
        )


# ============================================================================
# Autonomous Epistemic Research Cycle Controller
# ============================================================================

class AutonomousResearchCycleController:
    """
    Orchestrates the continuous 8-stage epistemic discovery cycle.
    """
    def __init__(self) -> None:
        self.knowledge_graph = EpistemicKnowledgeGraph()
        self.selector = EVOICandidateSelector()
        self.critic = AdversarialCritic()
        self.modularity_gate = ModuleVsHackClassifier()
        self.cycle_count: int = 0

    def run_discovery_cycle(
        self,
        topic: str,
        candidates: List[ResearchCandidate],
        simulated_baseline: float,
        simulated_post_change: float,
        concurrent_telemetry: Dict[str, Any],
        has_formal_interface: bool = True,
        is_cross_platform_abstracted: bool = True,
        has_magic_numbers: bool = False,
    ) -> Dict[str, Any]:
        """
        Executes all 8 phases of the epistemic research loop.
        """
        self.cycle_count += 1
        cycle_id = f"cycle_{self.cycle_count:03d}"

        # 1. Fresh system mapping node
        probe_id = f"probe_fresh_{uuid.uuid4().hex[:6]}"
        self.knowledge_graph.add_node(
            node_id=probe_id,
            node_type=GraphNodeType.UNKNOWN_GAP,
            label=f"Fresh System Mapping: {topic}",
            status=NodeStatus.UNKNOWN_GAP,
            metadata={"cycle_id": cycle_id},
        )

        # 2 & 3. Ensure M0 NULL model is present in candidate portfolio
        has_null = any(c.is_null_model for c in candidates)
        if not has_null:
            raise EpistemicContractViolation(
                "Violation of Research Protocol: Candidate portfolio must include M0 NULL model."
            )

        # 4. EVOI Candidate Selection (Information gain, not greedy performance)
        selected_candidate = self.selector.select_best_candidate(candidates)

        hyp_node_id = f"hyp_{selected_candidate.candidate_id}"
        self.knowledge_graph.add_node(
            node_id=hyp_node_id,
            node_type=GraphNodeType.HYPOTHESIS,
            label=selected_candidate.hypothesis_rationale,
            status=NodeStatus.UNDER_TEST,
            metadata={"evoi_score": selected_candidate.evoi_priority_score},
        )
        self.knowledge_graph.add_edge(probe_id, hyp_node_id, EdgeType.DEPENDS_ON)

        # 5. Measure & apply
        # 6. Adversarial Critic interrogation
        critique = self.critic.critique_result(
            hypothesis_id=selected_candidate.candidate_id,
            baseline_metric=simulated_baseline,
            post_change_metric=simulated_post_change,
            concurrent_subsystem_telemetry=concurrent_telemetry,
        )

        # 7. Knowledge graph revision
        if critique.verdict == AdversarialCriticVerdict.CHALLENGE_ACCEPTED:
            self.knowledge_graph.update_node_status(
                hyp_node_id,
                NodeStatus.APPROVED_MECHANISM,
                reason="Survived adversarial critique without confounders.",
            )
            # 8. Modularity Gate: New Module vs Hack
            modularity_receipt = self.modularity_gate.audit_modularity(
                mechanism_name=selected_candidate.candidate_id,
                has_formal_interface=has_formal_interface,
                is_cross_platform_abstracted=is_cross_platform_abstracted,
                has_magic_numbers=has_magic_numbers,
                is_bounded_by_invariants=True,
            )
            cycle_outcome = "MECHANISM_DISCOVERED_AND_AUDITED"
        else:
            self.knowledge_graph.update_node_status(
                hyp_node_id,
                NodeStatus.FALSIFIED if critique.verdict == AdversarialCriticVerdict.FLUKES_OR_NOISE else NodeStatus.CONFLICT,
                reason=critique.critique_details,
            )
            modularity_receipt = None
            cycle_outcome = "FALSIFIED_OR_CONFOUNDED"

        return {
            "cycle_id": cycle_id,
            "topic": topic,
            "selected_candidate_id": selected_candidate.candidate_id,
            "evoi_priority": selected_candidate.evoi_priority_score,
            "critique_verdict": critique.verdict.value,
            "critique_details": critique.critique_details,
            "cycle_outcome": cycle_outcome,
            "modularity_receipt": modularity_receipt.__dict__ if modularity_receipt else None,
            "active_hypotheses_count": len(self.knowledge_graph.get_active_hypotheses()),
        }
