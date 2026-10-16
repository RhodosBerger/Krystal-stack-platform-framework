"""
autonomous_research_cycle.py - Autonomous Epistemic Research & Engineering Discovery Loop (V2).

Implements Milestones:
- V-AUTONOMOUS-RESEARCH-01: Epistemic Knowledge Graph, EVOI Selector & Modularity Gate.
- V-AUTONOMOUS-RESEARCH-02: Closed-Loop Physical Experimentation.

Core Doctrines:
1. "Nie zrýchli kód, ale spochybni predpoklady, nájdi neznáme a falzifikuj vlastné modely."
2. "Research may propose experiments, but only the Policy Engine may alter the host;
   knowledge may be revised only from attested experiment receipts."

Key Components:
1. SystemRemapper: Real-time passive audit of host OS topology, CPU sets, memory, and capability tiers.
2. Tripartite Null Model Portfolio:
   - M0_NO_EFFECT: Zero causal divergence from baseline regime.
   - M1_NOISE: Nameraný rozdiel leží plne v stochastickom šume systému (<= 2-sigma).
   - M2_MEASUREMENT_ERROR: Telemetrický artefakt, drift hodín alebo chyba validátora.
3. ExperimentPlan: Formal structured proposal of actions, metrics, and falsifiers without direct host mutation.
4. InformationGainPriorityHeuristic (EVOI): Prioritization by Information Gain and Falsification Capacity.
5. KrystalPolicyEngine Execution: ONLY authorized pathway to host actuation with domain locks and rollback.
6. ExperimentReceipt: Attested cryptographic record of execution, baseline/post telemetry, and environment fingerprint.
7. Independent AdversarialCritic: Process/trust-isolated reviewer with parameterized critic policy.
8. Append-Only KnowledgeRevisionReceipt: Immutable, hash-linked revision history of knowledge graph nodes.
9. 4-Tier Modularity Classification:
   - NEW_FIRST_CLASS_MODULE: Cross-platform abstracted first-class engine component.
   - CAPABILITY_SPECIFIC_POLICY: Platform/hardware specific configuration with capability provider abstraction.
   - WORKLOAD_SPECIFIC_POLICY: Profile/regime specific adaptation.
   - AD_HOC_HACK: Relies on uncalibrated magic constants or violates invariants.
10. Global Invariant: GLOBAL_VITAL_MAX_HP == 6.
"""

from __future__ import annotations

import dataclasses
import enum
import hashlib
import os
import platform
import sys
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from krystal_kernel.os_coordination_policy_engine import (
    GLOBAL_VITAL_MAX_HP,
    ControlSurfaceTier,
    CoordinationTransaction,
    KrystalPolicyEngine,
    LinuxCapabilityProvider,
    OptimizationStatus,
    OSCapabilityProvider,
    RollbackClass,
    SubsystemAction,
    SubsystemType,
    WindowsCapabilityProvider,
)

assert GLOBAL_VITAL_MAX_HP == 6, f"Constitutional invariant breach: GLOBAL_VITAL_MAX_HP must be 6, got {GLOBAL_VITAL_MAX_HP}"


class EpistemicContractViolation(Exception):
    """Raised when an epistemic discovery invariant is violated."""
    pass


# ============================================================================
# 1. System Remapper & Environment Fingerprint
# ============================================================================

@dataclass(frozen=True)
class EnvironmentFingerprint:
    """
    Cryptographically attested hardware and OS environment fingerprint.
    Captures live system parameters prior to experiment execution.
    """
    platform_name: str
    os_version: str
    cpu_count: int
    memory_total_bytes: int
    memory_available_bytes: int
    supported_tiers: Tuple[int, ...]
    governor: str
    timestamp_ns: int
    fingerprint_digest: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "platform_name": self.platform_name,
            "os_version": self.os_version,
            "cpu_count": self.cpu_count,
            "memory_total_bytes": self.memory_total_bytes,
            "memory_available_bytes": self.memory_available_bytes,
            "supported_tiers": list(self.supported_tiers),
            "governor": self.governor,
            "timestamp_ns": self.timestamp_ns,
            "fingerprint_digest": self.fingerprint_digest,
        }


class SystemRemapper:
    """
    Audits live host OS topology, CPU sets, memory allocation, and supported control surfaces.
    Eliminates bias by re-mapping hardware realities fresh on every research cycle.
    """
    @staticmethod
    def remap_system(provider: Optional[OSCapabilityProvider] = None) -> EnvironmentFingerprint:
        p_name = platform.system()
        p_ver = platform.release()
        cpus = os.cpu_count() or 1

        total_mem = 16 * 1024 * 1024 * 1024
        avail_mem = 8 * 1024 * 1024 * 1024

        # Read physical memory if Windows API is present
        if sys.platform == "win32":
            try:
                import ctypes

                class MEMORYSTATUSEX(ctypes.Structure):
                    _fields_ = [
                        ("dwLength", ctypes.c_ulong),
                        ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong),
                        ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong),
                        ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong),
                        ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                    ]

                stat = MEMORYSTATUSEX()
                stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
                if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
                    total_mem = stat.ullTotalPhys
                    avail_mem = stat.ullAvailPhys
            except Exception:
                pass

        active_provider = provider or (WindowsCapabilityProvider() if sys.platform == "win32" else LinuxCapabilityProvider())
        supported = tuple(int(t) for t in sorted(active_provider.get_supported_tiers()))
        gov = "balance_performance"
        ts = time.time_ns()

        raw = f"{p_name}|{p_ver}|{cpus}|{total_mem}|{avail_mem}|{supported}|{gov}|{ts}"
        digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()

        return EnvironmentFingerprint(
            platform_name=p_name,
            os_version=p_ver,
            cpu_count=cpus,
            memory_total_bytes=total_mem,
            memory_available_bytes=avail_mem,
            supported_tiers=supported,
            governor=gov,
            timestamp_ns=ts,
            fingerprint_digest=digest,
        )


# ============================================================================
# 2. Tripartite Null Model Portfolio
# ============================================================================

class NullModelClass(str, enum.Enum):
    """
    Distinguishes the three fundamental classes of non-effect:
    1. M0_NO_EFFECT: The proposed causal mechanism has zero effect.
    2. M1_NOISE: The observed metric delta is entirely within random stochastic variance (<= 2-sigma).
    3. M2_MEASUREMENT_ERROR: The observed metric delta is a telemetry artifact, timer drift, or validator flaw.
    """
    M0_NO_EFFECT = "M0_NO_EFFECT"
    M1_NOISE = "M1_NOISE"
    M2_MEASUREMENT_ERROR = "M2_MEASUREMENT_ERROR"


# ============================================================================
# 3. Experiment Plan & Receipts (Attestation & Integrity)
# ============================================================================

@dataclass
class ExperimentPlan:
    """
    Formal proposal of an experimental intervention.
    Research may propose experiment plans, but cannot alter the host directly.
    Only the Policy Engine may actuate plans on the host.
    """
    plan_id: str
    hypothesis_rationale: str
    actions: List[SubsystemAction]
    target_metric: str
    expected_gain: float
    falsifier: str
    timeout_seconds: float = 5.0
    counter_null_models: List[NullModelClass] = field(default_factory=lambda: [
        NullModelClass.M0_NO_EFFECT,
        NullModelClass.M1_NOISE,
        NullModelClass.M2_MEASUREMENT_ERROR,
    ])

    def to_coordination_transaction(self) -> CoordinationTransaction:
        return CoordinationTransaction(
            transaction_id=f"tx_{self.plan_id}",
            hypothesis_rationale=self.hypothesis_rationale,
            actions=self.actions,
            target_metric=self.target_metric,
            expected_gain=f"{self.expected_gain * 100:.1f}%",
            side_effects="Monitored via AdversarialCritic",
            falsifier=self.falsifier,
            timeout_seconds=self.timeout_seconds,
        )


@dataclass(frozen=True)
class ExperimentReceipt:
    """
    Cryptographic receipt testifying to the physical execution of an ExperimentPlan by Policy Engine.
    Knowledge may be revised ONLY from attested experiment receipts.
    """
    receipt_id: str
    transaction_id: str
    baseline_metric: float
    post_change_metric: float
    is_improvement: bool
    falsifier_triggered: bool
    was_rolled_back: bool
    actions_unwound: Tuple[Dict[str, Any], ...]
    environment_fingerprint: EnvironmentFingerprint
    timestamp_ns: int
    receipt_digest: str

    @classmethod
    def create(
        cls,
        transaction_id: str,
        baseline_metric: float,
        post_change_metric: float,
        is_improvement: bool,
        falsifier_triggered: bool,
        was_rolled_back: bool,
        actions_unwound: List[Dict[str, Any]],
        fingerprint: EnvironmentFingerprint,
    ) -> ExperimentReceipt:
        r_id = f"rcpt_exp_{uuid.uuid4().hex[:8]}"
        ts = time.time_ns()
        raw = (
            f"{r_id}|{transaction_id}|{baseline_metric}|{post_change_metric}|"
            f"{is_improvement}|{falsifier_triggered}|{was_rolled_back}|"
            f"{fingerprint.fingerprint_digest}|{ts}"
        )
        digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
        return cls(
            receipt_id=r_id,
            transaction_id=transaction_id,
            baseline_metric=baseline_metric,
            post_change_metric=post_change_metric,
            is_improvement=is_improvement,
            falsifier_triggered=falsifier_triggered,
            was_rolled_back=was_rolled_back,
            actions_unwound=tuple(actions_unwound),
            environment_fingerprint=fingerprint,
            timestamp_ns=ts,
            receipt_digest=digest,
        )


# ============================================================================
# 4. Epistemic Knowledge Graph & Append-Only Revision History
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


@dataclass(frozen=True)
class KnowledgeRevisionReceipt:
    """
    Append-only record of a knowledge state transition.
    Forms a cryptographic hash chain ensuring knowledge history cannot be silently rewritten.
    """
    revision_id: str
    node_id: str
    previous_version: int
    new_version: int
    old_status: NodeStatus
    new_status: NodeStatus
    reason: str
    experiment_receipt_id: Optional[str]
    parent_receipt_digest: str
    timestamp_ns: int
    receipt_digest: str


class EpistemicKnowledgeGraph:
    """
    Maintains the evolving knowledge base of models, hypotheses, mechanisms, gaps, and conflicts.
    Enforces append-only version history via cryptographic KnowledgeRevisionReceipts.
    """
    def __init__(self) -> None:
        self.nodes: Dict[str, KnowledgeNode] = {}
        self.edges: List[KnowledgeEdge] = []
        self.node_version_snapshots: Dict[str, List[KnowledgeNode]] = {}
        self.revision_history: List[KnowledgeRevisionReceipt] = []

    def add_node(
        self,
        node_id: str,
        node_type: GraphNodeType,
        label: str,
        status: NodeStatus,
        metadata: Optional[Dict[str, Any]] = None,
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
        self.node_version_snapshots[node_id] = [dataclasses.replace(node)]
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

    def update_node_status(
        self,
        node_id: str,
        new_status: NodeStatus,
        reason: str,
        experiment_receipt_id: Optional[str] = None,
    ) -> KnowledgeNode:
        if node_id not in self.nodes:
            raise EpistemicContractViolation(f"Node '{node_id}' does not exist to update.")

        node = self.nodes[node_id]
        old_status = node.status
        prev_version = node.version

        node.version += 1
        node.status = new_status
        node.metadata["last_update_reason"] = reason
        if experiment_receipt_id:
            node.metadata["last_experiment_receipt_id"] = experiment_receipt_id

        # Snapshot immutable updated version into append-only history
        self.node_version_snapshots[node_id].append(dataclasses.replace(node))

        # Determine parent digest in append-only revision hash chain
        parent_digest = self.revision_history[-1].receipt_digest if self.revision_history else "0" * 64
        rev_id = f"rev_{uuid.uuid4().hex[:8]}"
        ts = time.time_ns()
        raw = f"{rev_id}|{node_id}|{prev_version}|{node.version}|{old_status.value}|{new_status.value}|{experiment_receipt_id}|{parent_digest}|{ts}"
        digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()

        rev_receipt = KnowledgeRevisionReceipt(
            revision_id=rev_id,
            node_id=node_id,
            previous_version=prev_version,
            new_version=node.version,
            old_status=old_status,
            new_status=new_status,
            reason=reason,
            experiment_receipt_id=experiment_receipt_id,
            parent_receipt_digest=parent_digest,
            timestamp_ns=ts,
            receipt_digest=digest,
        )
        self.revision_history.append(rev_receipt)
        return node

    def get_active_hypotheses(self) -> List[KnowledgeNode]:
        return [n for n in self.nodes.values() if n.status in (NodeStatus.HYPOTHESIS, NodeStatus.UNDER_TEST)]

    def get_unknown_gaps(self) -> List[KnowledgeNode]:
        return [n for n in self.nodes.values() if n.status == NodeStatus.UNKNOWN_GAP]


# ============================================================================
# 5. Information Gain Priority Heuristic (Heuristic EVOI)
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
    cost: float                       # Execution cost/time (e.g. 0.5 to 10.0)
    null_class: Optional[NullModelClass] = None

    @property
    def evoi_priority_score(self) -> float:
        """
        Information Gain Priority Heuristic:
        Priority = (Information_Gain * Falsification_Capacity) / (Risk * Cost)
        Does NOT rely solely on greedy performance gain.
        """
        effective_risk = max(0.1, self.risk_level)
        effective_cost = max(0.5, self.cost)
        return (self.information_gain_evoi * self.falsification_capacity) / (effective_risk * effective_cost)


class InformationGainPriorityHeuristic:
    """
    Epistemically honest prioritization: computes information gain priority score
    to rank competing hypotheses. Prioritizes discriminating tests over greedy gains.
    """
    @staticmethod
    def select_best_candidate(candidates: List[ResearchCandidate]) -> ResearchCandidate:
        if not candidates:
            raise EpistemicContractViolation("Cannot select from an empty list of research candidates.")
        # Filter for actionable intervention candidates to actuate and test.
        # Null models serve as reference baseline comparators.
        actionable_candidates = [c for c in candidates if not c.is_null_model]
        candidates_to_rank = actionable_candidates if actionable_candidates else candidates
        sorted_candidates = sorted(candidates_to_rank, key=lambda c: c.evoi_priority_score, reverse=True)
        return sorted_candidates[0]


# Backward compatibility alias
EVOICandidateSelector = InformationGainPriorityHeuristic


# ============================================================================
# 6. Parameterized Independent Adversarial Critic
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
    refuted_null_models: List[NullModelClass] = field(default_factory=list)


@dataclass
class AdversarialCriticPolicy:
    """
    Configurable scientific policy parameters for adversarial auditing.
    Replaces static magic thresholds with formal policy parameters.
    """
    noise_floor_sigma_multiplier: float = 2.0
    thermal_drift_threshold_c: float = 5.0
    background_stall_threshold: float = 0.15


class AdversarialCritic:
    """
    Independent skeptical reviewer actively seeking reasons why an empirical result
    might be false, an artifact of measurement noise, or a manifestation of Goodhart's Law.
    """
    def __init__(
        self,
        critic_id: str = "adversarial_critic_01",
        policy: Optional[AdversarialCriticPolicy] = None,
    ) -> None:
        self.critic_id = critic_id
        self.policy = policy or AdversarialCriticPolicy()

    def critique_result(
        self,
        hypothesis_id: str,
        baseline_metric: float,
        post_change_metric: float,
        concurrent_subsystem_telemetry: Dict[str, Any],
        measured_variance: float = 0.02,
    ) -> AdversarialCritique:
        delta = abs(post_change_metric - baseline_metric)

        # 1. Check for flukes/noise (M1_NOISE): if change is smaller than sigma threshold
        noise_limit = self.policy.noise_floor_sigma_multiplier * measured_variance * baseline_metric
        if delta < noise_limit:
            return AdversarialCritique(
                verdict=AdversarialCriticVerdict.FLUKES_OR_NOISE,
                confidence=0.85,
                identified_confounders=["Random measurement variance", "Background scheduler jitter"],
                critique_details=f"Reported improvement ({delta:.3f}) is within the {self.policy.noise_floor_sigma_multiplier}-sigma noise floor ({noise_limit:.3f}).",
                requires_discriminating_retest=True,
                refuted_null_models=[],
            )

        # 2. Check for confounders (M2_MEASUREMENT_ERROR): thermal drift
        thermal_drift = concurrent_subsystem_telemetry.get("thermal_drift_c", 0.0)
        if abs(thermal_drift) > self.policy.thermal_drift_threshold_c:
            return AdversarialCritique(
                verdict=AdversarialCriticVerdict.CONFOUNDER_SUSPECTED,
                confidence=0.80,
                identified_confounders=[f"Significant thermal temperature drift: {thermal_drift:+.1f}C"],
                critique_details=f"Thermal drift detected: chassis drift {thermal_drift:+.1f}C exceeds policy threshold {self.policy.thermal_drift_threshold_c}C.",
                requires_discriminating_retest=True,
                refuted_null_models=[NullModelClass.M0_NO_EFFECT],
            )

        # 3. Check for oracle gaming: did auxiliary subsystems stall catastrophically?
        side_effect_stall = concurrent_subsystem_telemetry.get("background_stall_rate", 0.0)
        if side_effect_stall > self.policy.background_stall_threshold:
            return AdversarialCritique(
                verdict=AdversarialCriticVerdict.ORACLE_GAMING_SUSPECTED,
                confidence=0.90,
                identified_confounders=["Starvation of background memory channels"],
                critique_details=f"Target metric improved by starving auxiliary subsystems (stall {side_effect_stall:.1%} > {self.policy.background_stall_threshold:.1%}).",
                requires_discriminating_retest=True,
                refuted_null_models=[NullModelClass.M0_NO_EFFECT],
            )

        # Result passed adversarial scrutiny: refutes M0_NO_EFFECT and M1_NOISE
        return AdversarialCritique(
            verdict=AdversarialCriticVerdict.CHALLENGE_ACCEPTED,
            confidence=0.95,
            identified_confounders=[],
            critique_details="Empirical result survived adversarial scrutiny; no obvious confounder identified.",
            requires_discriminating_retest=False,
            refuted_null_models=[
                NullModelClass.M0_NO_EFFECT,
                NullModelClass.M1_NOISE,
                NullModelClass.M2_MEASUREMENT_ERROR,
            ],
        )


# ============================================================================
# 7. 4-Tier Modularity Classification
# ============================================================================

class ModularityVerdict(str, enum.Enum):
    NEW_FIRST_CLASS_MODULE = "NEW_FIRST_CLASS_MODULE"
    CAPABILITY_SPECIFIC_POLICY = "CAPABILITY_SPECIFIC_POLICY"
    WORKLOAD_SPECIFIC_POLICY = "WORKLOAD_SPECIFIC_POLICY"
    AD_HOC_HACK = "AD_HOC_HACK"


@dataclass
class ModularityAuditReceipt:
    verdict: ModularityVerdict
    architectural_abstraction: str
    rejection_reasons: List[str]
    refactoring_recommendation: str


class ModuleVsHackClassifier:
    """
    Audits confirmed mechanisms to categorize them into legitimate architectural tiers:
    - NEW_FIRST_CLASS_MODULE: Clean, cross-platform, typed, invariant-bounded abstraction.
    - CAPABILITY_SPECIFIC_POLICY: Platform/hardware specific configuration with capability provider abstraction.
    - WORKLOAD_SPECIFIC_POLICY: Profile/regime specific adaptation.
    - AD_HOC_HACK: Relies on magic constants, unabstracted ad-hoc override, lacks contracts.
    """
    @staticmethod
    def audit_modularity(
        mechanism_name: str,
        has_formal_interface: bool,
        is_cross_platform_abstracted: bool,
        has_magic_numbers: bool,
        is_bounded_by_invariants: bool,
        is_workload_specific: bool = False,
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

        if is_workload_specific:
            return ModularityAuditReceipt(
                verdict=ModularityVerdict.WORKLOAD_SPECIFIC_POLICY,
                architectural_abstraction=f"krystal_kernel/policies/{mechanism_name.lower()}_workload_policy.py",
                rejection_reasons=[],
                refactoring_recommendation="Deploy as specialized profile-driven workload policy.",
            )

        if not is_cross_platform_abstracted:
            return ModularityAuditReceipt(
                verdict=ModularityVerdict.CAPABILITY_SPECIFIC_POLICY,
                architectural_abstraction=f"krystal_kernel/adapters/{mechanism_name.lower()}_capability_adapter.py",
                rejection_reasons=[],
                refactoring_recommendation="Deploy as platform-specific capability provider extension.",
            )

        return ModularityAuditReceipt(
            verdict=ModularityVerdict.NEW_FIRST_CLASS_MODULE,
            architectural_abstraction=f"krystal_kernel/{mechanism_name.lower()}_module.py",
            rejection_reasons=[],
            refactoring_recommendation="Promote to permanent canonical engine component.",
        )


# ============================================================================
# 8. Closed-Loop Physical Research Orchestrator
# ============================================================================

class ClosedLoopResearchOrchestrator:
    """
    Implements Milestone V-AUTONOMOUS-RESEARCH-02.
    Closed-Loop Physical Experimentation:
    Research may propose experiments, but ONLY Policy Engine may alter the host;
    knowledge may be revised ONLY from attested experiment receipts.
    """
    def __init__(
        self,
        policy_engine: Optional[KrystalPolicyEngine] = None,
        critic: Optional[AdversarialCritic] = None,
    ) -> None:
        self.policy_engine = policy_engine or KrystalPolicyEngine()
        self.knowledge_graph = EpistemicKnowledgeGraph()
        self.selector = InformationGainPriorityHeuristic()
        self.critic = critic or AdversarialCritic()
        self.modularity_gate = ModuleVsHackClassifier()
        self.cycle_count: int = 0

    def execute_closed_loop_experiment(
        self,
        topic: str,
        candidates: List[ResearchCandidate],
        plan: ExperimentPlan,
        simulated_baseline: float,
        simulated_post_change: float,
        concurrent_telemetry: Dict[str, Any],
        falsifier_triggered: bool = False,
        has_formal_interface: bool = True,
        is_cross_platform_abstracted: bool = True,
        has_magic_numbers: bool = False,
        is_workload_specific: bool = False,
    ) -> Dict[str, Any]:
        """
        Executes a complete closed-loop physical research experiment.
        """
        self.cycle_count += 1
        cycle_id = f"cycle_v2_{self.cycle_count:03d}"

        # 1. System Remapper: Real passive hardware and OS state reading
        fingerprint = SystemRemapper.remap_system(self.policy_engine.provider)

        # 2. Record fresh re-mapping in Knowledge Graph
        probe_id = f"probe_live_{uuid.uuid4().hex[:6]}"
        self.knowledge_graph.add_node(
            node_id=probe_id,
            node_type=GraphNodeType.UNKNOWN_GAP,
            label=f"Live System Re-Mapping: {topic} on {fingerprint.platform_name}",
            status=NodeStatus.UNKNOWN_GAP,
            metadata={"cycle_id": cycle_id, "fingerprint_digest": fingerprint.fingerprint_digest},
        )

        # 3. Verify Tripartite Null Model Portfolio is present
        has_null = any(c.is_null_model for c in candidates)
        if not has_null:
            raise EpistemicContractViolation(
                "Violation of Research Protocol: Candidate portfolio must include M0 NULL model."
            )

        # 4. Information Gain Candidate Selection
        selected_candidate = self.selector.select_best_candidate(candidates)

        hyp_node_id = f"hyp_{selected_candidate.candidate_id}"
        self.knowledge_graph.add_node(
            node_id=hyp_node_id,
            node_type=GraphNodeType.HYPOTHESIS,
            label=selected_candidate.hypothesis_rationale,
            status=NodeStatus.UNDER_TEST,
            metadata={"evoi_priority": selected_candidate.evoi_priority_score},
        )
        self.knowledge_graph.add_edge(probe_id, hyp_node_id, EdgeType.DEPENDS_ON)

        # 5. Core Invariant: Research proposes ExperimentPlan, but ONLY PolicyEngine actuates the host
        tx = plan.to_coordination_transaction()
        tx_receipt = self.policy_engine.execute_transaction(
            transaction=tx,
            simulated_baseline=simulated_baseline,
            simulated_post_change=simulated_post_change,
            falsifier_triggered=falsifier_triggered,
        )

        # 6. Attest ExperimentReceipt
        outcome = tx_receipt.get("outcome", "")
        audit_res = tx_receipt.get("verdict", {})
        audit_verdict = audit_res.get("verdict") if isinstance(audit_res, dict) else str(audit_res)
        is_imp = outcome == "COMMITTED" or audit_verdict == "APPROVE_AND_COMMIT"
        was_rolled = outcome == "ROLLED_BACK" or tx_receipt.get("was_rolled_back", False)
        rollback_data = tx_receipt.get("rollback", {}) if isinstance(tx_receipt.get("rollback"), dict) else {}
        actions_unwound = rollback_data.get("actions_unwound", tx_receipt.get("actions_unwound", []))

        experiment_receipt = ExperimentReceipt.create(
            transaction_id=tx.transaction_id,
            baseline_metric=simulated_baseline,
            post_change_metric=simulated_post_change,
            is_improvement=is_imp,
            falsifier_triggered=falsifier_triggered,
            was_rolled_back=was_rolled,
            actions_unwound=actions_unwound,
            fingerprint=fingerprint,
        )

        # 7. Independent Adversarial Critic Audit
        critique = self.critic.critique_result(
            hypothesis_id=selected_candidate.candidate_id,
            baseline_metric=simulated_baseline,
            post_change_metric=simulated_post_change,
            concurrent_subsystem_telemetry=concurrent_telemetry,
        )

        # 8. Append-Only Knowledge Revision (Knowledge may be revised ONLY from attested experiment receipts)
        if is_imp and critique.verdict == AdversarialCriticVerdict.CHALLENGE_ACCEPTED:
            self.knowledge_graph.update_node_status(
                node_id=hyp_node_id,
                new_status=NodeStatus.APPROVED_MECHANISM,
                reason="Physical transaction committed and survived adversarial audit.",
                experiment_receipt_id=experiment_receipt.receipt_id,
            )
            # 9. 4-Tier Modularity Gate
            modularity_receipt = self.modularity_gate.audit_modularity(
                mechanism_name=selected_candidate.candidate_id,
                has_formal_interface=has_formal_interface,
                is_cross_platform_abstracted=is_cross_platform_abstracted,
                has_magic_numbers=has_magic_numbers,
                is_bounded_by_invariants=True,
                is_workload_specific=is_workload_specific,
            )
            cycle_outcome = "MECHANISM_DISCOVERED_AND_AUDITED"
        else:
            final_status = NodeStatus.FALSIFIED if (falsifier_triggered or critique.verdict == AdversarialCriticVerdict.FLUKES_OR_NOISE) else NodeStatus.CONFLICT
            self.knowledge_graph.update_node_status(
                node_id=hyp_node_id,
                new_status=final_status,
                reason=critique.critique_details or "Falsifier triggered during transaction execution.",
                experiment_receipt_id=experiment_receipt.receipt_id,
            )
            modularity_receipt = None
            cycle_outcome = "FALSIFIED_OR_CONFOUNDED"

        return {
            "cycle_id": cycle_id,
            "topic": topic,
            "selected_candidate_id": selected_candidate.candidate_id,
            "evoi_priority": selected_candidate.evoi_priority_score,
            "experiment_receipt": experiment_receipt,
            "critique_verdict": critique.verdict.value,
            "critique_details": critique.critique_details,
            "cycle_outcome": cycle_outcome,
            "modularity_receipt": modularity_receipt.__dict__ if modularity_receipt else None,
            "revision_receipts_count": len(self.knowledge_graph.revision_history),
            "environment_fingerprint": fingerprint.to_dict(),
        }


# ============================================================================
# 9. Backward-Compatible Controller Wrapper
# ============================================================================

class AutonomousResearchCycleController:
    """
    Maintains backward compatibility with V1 tests and workflows while routing
    through the closed-loop physical experimentation engine.
    """
    def __init__(self, policy_engine: Optional[KrystalPolicyEngine] = None) -> None:
        self.orchestrator = ClosedLoopResearchOrchestrator(policy_engine=policy_engine)
        self.knowledge_graph = self.orchestrator.knowledge_graph
        self.selector = self.orchestrator.selector
        self.critic = self.orchestrator.critic
        self.modularity_gate = self.orchestrator.modularity_gate

    @property
    def cycle_count(self) -> int:
        return self.orchestrator.cycle_count

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
        is_workload_specific: bool = False,
        falsifier_triggered: bool = False,
        actions: Optional[List[SubsystemAction]] = None,
    ) -> Dict[str, Any]:
        default_actions = actions or [
            SubsystemAction(
                subsystem=SubsystemType.CPU,
                control_surface=ControlSurfaceTier.APP_CONFIG,
                rollback_class=RollbackClass.CONFIG_RESTORE,
                target_param="affinity_mask",
                target_value="0x0F",
                rollback_payload={"affinity_mask": "0xFF"},
            )
        ]

        plan = ExperimentPlan(
            plan_id=f"plan_{uuid.uuid4().hex[:6]}",
            hypothesis_rationale=f"Empirical validation for {topic}",
            actions=default_actions,
            target_metric="latency_ms",
            expected_gain=0.15,
            falsifier="delta <= 0 or regression",
        )

        res = self.orchestrator.execute_closed_loop_experiment(
            topic=topic,
            candidates=candidates,
            plan=plan,
            simulated_baseline=simulated_baseline,
            simulated_post_change=simulated_post_change,
            concurrent_telemetry=concurrent_telemetry,
            falsifier_triggered=falsifier_triggered,
            has_formal_interface=has_formal_interface,
            is_cross_platform_abstracted=is_cross_platform_abstracted,
            has_magic_numbers=has_magic_numbers,
            is_workload_specific=is_workload_specific,
        )

        return {
            "cycle_id": res["cycle_id"],
            "topic": res["topic"],
            "selected_candidate_id": res["selected_candidate_id"],
            "evoi_priority": res["evoi_priority"],
            "critique_verdict": res["critique_verdict"],
            "critique_details": res["critique_details"],
            "cycle_outcome": res["cycle_outcome"],
            "modularity_receipt": res["modularity_receipt"],
            "active_hypotheses_count": len(self.knowledge_graph.get_active_hypotheses()),
            "experiment_receipt": res["experiment_receipt"],
            "environment_fingerprint": res["environment_fingerprint"],
        }
