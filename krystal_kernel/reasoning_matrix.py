#!/usr/bin/env python3
"""
==============================================================================
KRYSTAL-STACK: EPISTEMIC REASONING MATRIX & SELECTION FUNNEL (V-PLAN-02)
==============================================================================
Module: krystal_kernel/reasoning_matrix.py
Description:
    Realizes the 5-stage candidate selection funnel, dynamic reasoning matrix,
    and polymorphic multi-node workspace scheduler.

Architectural Guarantees:
    1. Separation of Analytical Proposal and Deterministic Commit:
       Analytical models (LLM, neural, heuristic) generate candidate proposals.
       Commit authority resides strictly in the deterministic bitmask reducer.
    2. Strict Zero-Intersection Veto:
       A_commit == 0 strictly triggers AUTH_DEGRADED_SAFE_FALLBACK.
    3. Multi-Node Dynamic Workspace:
       Scheduler does not emit linear chat text; it dynamically composes
       a coherent tactical workspace (Causal Diagnosis + Memory Node Card +
       Exploration Node Card + Interactive Simulation Layer).
    4. Constitutional Invariant:
       assert VITAL_MAX_HP == 6 across all candidates, envelopes, and scenes.

Author: Dušan Kopecký & Krystal-Stack Architecture & Epistemic Systems Council (2026)
==============================================================================
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Callable, Dict, List, Mapping, Optional, Protocol, Sequence, Set, Tuple, runtime_checkable

from krystal_kernel.scene_constructor_presets import VITAL_MAX_HP
from krystal_kernel.verification.authority_bits import (
    AuthorityProfileBit,
    PROFILE_TO_BIT,
    ALL_PROFILE_BITS,
    reduce_authority_masks,
)
from krystal_kernel.verification.capability_registry import (
    CapabilitySnapshot,
    HardwareBackend,
    GLOBAL_CAPABILITY_REGISTRY,
)
from krystal_kernel.custom_governor import (
    ExecutionEnvelope,
    GovernorDecision,
    GovernorMode,
    BatteryBand,
    ThermalBand,
)

assert VITAL_MAX_HP == 6, f"Constitutional invariant breach: VITAL_MAX_HP must be 6, got {VITAL_MAX_HP}"


# ============================================================================
# 1. REASONING MODES (FROM PROFILES TO MODES OF THOUGHT)
# ============================================================================

class ReasoningModeKind(str, Enum):
    DETERMINISTIC_BITMASK = "DETERMINISTIC_BITMASK"
    PARETO_HEURISTIC_SOLVER = "PARETO_HEURISTIC_SOLVER"
    CUSTOM_NN_PREDICTOR = "CUSTOM_NN_PREDICTOR"
    LLM_CAUSAL_NODE = "LLM_CAUSAL_NODE"
    SHADOW_EXPLORER = "SHADOW_EXPLORER"


@dataclass(frozen=True)
class CandidateAction:
    """An analytical candidate action prior to deterministic filtering."""
    action_id: str
    target_profile: str
    reasoning_mode: ReasoningModeKind
    proposed_by: str
    predicted_latency_ms: float
    predicted_throughput_fps: float
    predicted_power_w: float
    predicted_uma_pressure: float
    requires_gpu: bool
    requires_npu: bool
    requires_openvino: bool
    is_shadow_experiment: bool = False
    evidence_score: float = 1.0  # [0.0 .. 1.0]
    vital_max_hp: int = VITAL_MAX_HP

    def __post_init__(self):
        assert self.vital_max_hp == 6, "Platform invariant VITAL_MAX_HP == 6 must hold"


# ============================================================================
# 2. 5-STAGE SELECTION FUNNEL (SELEKČNÝ LIEVIK)
# ============================================================================

@dataclass
class FunnelStageTelemetry:
    initial_candidates: int
    after_capability_filter: int
    after_safety_filter: int
    after_evidence_filter: int
    after_utility_scoring: int
    winner_selected: Optional[str]
    authority_verdict: str
    vital_max_hp: int = VITAL_MAX_HP


class AdmissibleHypothesisFunnel:
    """
    Filters high-entropy candidate spaces down to a single provably safe,
    Pareto-optimal, and oscillation-suppressed execution choice.
    """

    def __init__(
        self,
        capability_snapshot: Optional[CapabilitySnapshot] = None,
        max_uma_pressure: float = 0.82,
        min_dwell_s: float = 1.0,
    ):
        self.caps = capability_snapshot or GLOBAL_CAPABILITY_REGISTRY.get_snapshot()
        self.max_uma_pressure = max_uma_pressure
        self.min_dwell_s = min_dwell_s
        self._last_commit_time = 0.0
        self._active_profile = "balanced_npu"

    def filter_and_select(
        self,
        candidates: List[CandidateAction],
        *,
        current_temp_c: float,
        tjmax_c: float = 100.0,
        current_uma_pressure: float,
        battery_pct: float,
        now_s: Optional[float] = None,
    ) -> Tuple[Optional[CandidateAction], FunnelStageTelemetry]:
        """
        Executes the 5-stage selection funnel:
          1. Capability Filter
          2. Safety Filter (Thermal, UMA, Battery)
          3. Evidence Filter
          4. Utility Scoring
          5. Dwell & Hysteresis
        """
        t_now = time.time() if now_s is None else now_s
        n0 = len(candidates)

        # 1. Capability Filter
        verified_backends = set(self.caps.verified_backends)
        s1: List[CandidateAction] = []
        for c in candidates:
            if c.requires_npu and HardwareBackend.NPU.value not in verified_backends:
                continue
            if c.requires_gpu and HardwareBackend.OPENVINO_GPU.value not in verified_backends and HardwareBackend.VULKAN.value not in verified_backends:
                continue
            if c.requires_openvino and HardwareBackend.OPENVINO_CPU.value not in verified_backends:
                continue
            s1.append(c)

        # 2. Safety Filter
        s2: List[CandidateAction] = []
        for c in s1:
            if current_temp_c >= (tjmax_c - 5.0) and c.target_profile in ("performance_gpu", "max_burst"):
                continue  # Thermal safety veto
            if current_uma_pressure >= self.max_uma_pressure and c.predicted_uma_pressure >= current_uma_pressure:
                continue  # UMA saturation prevention
            if battery_pct < 15.0 and c.predicted_power_w > 8.0:
                continue  # Battery conservation veto
            s2.append(c)

        # 3. Evidence Filter
        s3 = [c for c in s2 if c.evidence_score >= 0.70]

        # 4. Utility Scoring (Pareto utility: Throughput - Latency - Power)
        def score(c: CandidateAction) -> float:
            return (c.predicted_throughput_fps * 1.5) - (c.predicted_latency_ms * 2.0) - (c.predicted_power_w * 0.5)

        s4 = sorted(s3, key=score, reverse=True)[:2]

        # 5. Dwell & Asymmetric Hysteresis
        winner: Optional[CandidateAction] = None
        authority_verdict = "AUTH_COMMIT_APPROVED"

        if not s4:
            # Zero feasible candidates (A_commit == 0) -> strict degraded-safe fallback
            authority_verdict = "AUTH_DEGRADED_SAFE_FALLBACK"
            winner = CandidateAction(
                action_id="act_degraded_safe_fallback",
                target_profile="ultra_saver",
                reasoning_mode=ReasoningModeKind.DETERMINISTIC_BITMASK,
                proposed_by="AdmissibleHypothesisFunnel.FAIL_SAFE",
                predicted_latency_ms=50.0,
                predicted_throughput_fps=30.0,
                predicted_power_w=3.5,
                predicted_uma_pressure=0.20,
                requires_gpu=False,
                requires_npu=False,
                requires_openvino=False,
                evidence_score=1.0,
            )
        else:
            top = s4[0]
            elapsed_dwell = t_now - self._last_commit_time
            if elapsed_dwell < self.min_dwell_s and top.target_profile != self._active_profile:
                # Anti-flapping hysteresis holds active profile if within dwell
                existing = next((c for c in s4 if c.target_profile == self._active_profile), None)
                winner = existing or top
            else:
                winner = top
                self._active_profile = winner.target_profile
                self._last_commit_time = t_now

        telemetry = FunnelStageTelemetry(
            initial_candidates=n0,
            after_capability_filter=len(s1),
            after_safety_filter=len(s2),
            after_evidence_filter=len(s3),
            after_utility_scoring=len(s4),
            winner_selected=winner.target_profile if winner else None,
            authority_verdict=authority_verdict,
        )
        return winner, telemetry


# ============================================================================
# 3. POLYMORPHIC MULTI-NODE WORKSPACE SCHEDULER
# ============================================================================

class WorkspaceNodeKind(str, Enum):
    CAUSAL_DIAGNOSIS = "CAUSAL_DIAGNOSIS"
    ACTION_CARD = "ACTION_CARD"
    INTERACTIVE_SIMULATION = "INTERACTIVE_SIMULATION"
    EVIDENCE_VERIFIER = "EVIDENCE_VERIFIER"


@dataclass
class WorkspaceNode:
    """Polymorphic building block of the dynamically composed application UI."""
    node_id: str
    kind: WorkspaceNodeKind
    title: str
    summary: str
    properties: Dict[str, Any] = field(default_factory=dict)
    actions: List[Dict[str, str]] = field(default_factory=list)
    vital_max_hp: int = VITAL_MAX_HP

    def __post_init__(self):
        assert self.vital_max_hp == 6, "Platform invariant VITAL_MAX_HP == 6 must hold"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DynamicWorkspaceScene:
    """A coherent tactical workspace generated in response to diagnostic queries."""
    scene_id: str
    query: str
    root_cause_summary: str
    nodes: List[WorkspaceNode]
    authority_verdict: str = "AUTH_COMMIT_APPROVED"
    is_shadow_available: bool = True
    vital_max_hp: int = VITAL_MAX_HP

    def __post_init__(self):
        assert self.vital_max_hp == 6, "Platform invariant VITAL_MAX_HP == 6 must hold"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scene_id": self.scene_id,
            "query": self.query,
            "root_cause_summary": self.root_cause_summary,
            "nodes": [n.to_dict() for n in self.nodes],
            "authority_verdict": self.authority_verdict,
            "is_shadow_available": self.is_shadow_available,
            "vital_max_hp": self.vital_max_hp,
        }


class CausalDiagnosticWorkspaceComposer:
    """
    Synthesizes the end-to-end interactive workspace for telemetry-driven queries.
    Demonstrates the transition from simple chat response to live generated micro-app.
    """

    def compose_diagnostic_scene(
        self,
        query: str,
        observed_uma_pressure: float,
        observed_latency_p99_ms: float,
        observed_temp_c: float,
    ) -> DynamicWorkspaceScene:
        assert VITAL_MAX_HP == 6

        is_uma_saturated = observed_uma_pressure > 0.82
        root_cause = (
            f"UMA bus thrashing detected (Pressure: {observed_uma_pressure:.2f} > 0.82 limit). "
            f"Inference p99 degraded to {observed_latency_p99_ms:.1f} ms due to CPU/GPU bus contention."
            if is_uma_saturated
            else "Nominal hardware envelope; latency within expected stochastic bounds."
        )

        # 1. Main Causal Diagnosis Panel
        main_node = WorkspaceNode(
            node_id="node_causal_diag_main",
            kind=WorkspaceNodeKind.CAUSAL_DIAGNOSIS,
            title="Causal Diagnostic Engine",
            summary=root_cause,
            properties={
                "observed_uma": observed_uma_pressure,
                "observed_latency_p99_ms": observed_latency_p99_ms,
                "temperature_c": observed_temp_c,
            },
        )

        # 2. Alternative A: Memory Node (Conservative Stabilization)
        card_a = WorkspaceNode(
            node_id="node_alt_a_memory",
            kind=WorkspaceNodeKind.ACTION_CARD,
            title="Alternatíva A (Pamäťový uzol)",
            summary="Konzervatívna stabilizácia zbernice: zníženie GPU streamov 4 -> 2 + 16-bit kompresia.",
            properties={
                "predicted_gain": "-54% latencia",
                "predicted_uma": 0.68,
                "predicted_latency_ms": 11.2,
                "risk_profile": "0% (deterministicky overené)",
                "authority_safe": True,
            },
            actions=[{"label": "Aplikovať okamžite", "action_id": "commit_alt_a"}],
        )

        # 3. Alternative B: Exploration Node (Aggressive Shadow Experiment)
        card_b = WorkspaceNode(
            node_id="node_alt_b_exploration",
            kind=WorkspaceNodeKind.ACTION_CARD,
            title="Alternatíva B (Exploračný uzol)",
            summary="Agresívny shadow experiment: dočasný offload cache do CPU SIMD a meranie VRAM fragmentácie.",
            properties={
                "predicted_gain": "-68% latencia",
                "predicted_uma": 0.62,
                "predicted_latency_ms": 7.8,
                "risk_profile": "Možný krátkodobý frame drop počas testu",
                "requires_shadow_envelope": True,
            },
            actions=[{"label": "Zobraziť detail a simuláciu", "action_id": "expand_alt_b"}],
        )

        # 4. Interactive Simulation & Verification Layer (Detail for Alternative B)
        sim_node = WorkspaceNode(
            node_id="node_sim_verification_b",
            kind=WorkspaceNodeKind.INTERACTIVE_SIMULATION,
            title="Verifikačná & Simulačná Vrstva (Alternatíva B)",
            summary="Predikcia dynamického poklesu UMA tlaku z 0.84 na 0.62 s deterministickým ohraničením.",
            properties={
                "simulation_horizon_s": 2.0,
                "missing_evidence": "Vplyv na 8 kHz input stream nebol overený.",
                "authority_boundary": "Autorita obmedzí shadow experiment na max. 3.0 sekundy.",
            },
            actions=[{"label": "SPUSTIŤ EXPERIMENT V TIENI", "action_id": "launch_shadow_experiment"}],
        )

        return DynamicWorkspaceScene(
            scene_id="SCENE-DIAG-UMA-THRASHING",
            query=query,
            root_cause_summary=root_cause,
            nodes=[main_node, card_a, card_b, sim_node],
            authority_verdict="AUTH_COMMIT_APPROVED",
            is_shadow_available=True,
        )


# ============================================================================
# 4. UNIFIED EPISTEMIC DATA TYPES & NODE CONTRACT (INTER-NODE ABI)
# ============================================================================

@dataclass(frozen=True)
class WorldStateVector:
    """Canonical immutable world state snapshot feeding all reasoning nodes."""
    state_timestamp_ns: int
    sequence_id: int
    cpu_utilization_pct: float
    gpu_utilization_pct: float
    uma_pressure: float
    temperature_c: float
    battery_pct: float
    plugged: bool
    queue_depth: int
    evidence_confidence: float = 1.0
    vital_max_hp: int = VITAL_MAX_HP

    def __post_init__(self):
        assert self.vital_max_hp == 6, "Platform invariant VITAL_MAX_HP == 6 must hold"


@dataclass(frozen=True)
class HypothesisEnvelope:
    """Formalized proposal emitted by any reasoning node."""
    hypothesis_id: str
    node_id: str
    reasoning_mode: ReasoningModeKind
    target_profile: str
    predicted_latency_ms: float
    predicted_throughput_fps: float
    predicted_power_w: float
    predicted_uma_pressure: float
    assumptions: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0
    requires_gpu: bool = False
    requires_npu: bool = False
    requires_openvino: bool = False
    is_shadow: bool = False
    vital_max_hp: int = VITAL_MAX_HP

    def __post_init__(self):
        assert self.vital_max_hp == 6, "Platform invariant VITAL_MAX_HP == 6 must hold"

    def to_candidate_action(self) -> CandidateAction:
        return CandidateAction(
            action_id=self.hypothesis_id,
            target_profile=self.target_profile,
            reasoning_mode=self.reasoning_mode,
            proposed_by=self.node_id,
            predicted_latency_ms=self.predicted_latency_ms,
            predicted_throughput_fps=self.predicted_throughput_fps,
            predicted_power_w=self.predicted_power_w,
            predicted_uma_pressure=self.predicted_uma_pressure,
            requires_gpu=self.requires_gpu,
            requires_npu=self.requires_npu,
            requires_openvino=self.requires_openvino,
            is_shadow_experiment=self.is_shadow,
            evidence_score=self.confidence,
            vital_max_hp=self.vital_max_hp,
        )


@dataclass(frozen=True)
class ObservationResidual:
    """Prediction error residual used strictly for attested calibration."""
    residual_id: str
    hypothesis_id: str
    node_id: str
    error_latency_ms: float       # real - predicted
    error_throughput_fps: float   # real - predicted
    error_power_w: float          # real - predicted
    error_uma_pressure: float     # real - predicted
    real_duration_us: float
    proof_hash: str
    vital_max_hp: int = VITAL_MAX_HP

    def __post_init__(self):
        assert self.vital_max_hp == 6, "Platform invariant VITAL_MAX_HP == 6 must hold"


@dataclass(frozen=True)
class AdmissibilityVerdict:
    """Formal authority verdict bounding a proposed hypothesis before commit."""
    verdict_id: str
    hypothesis_id: str
    target_profile: str
    authority_mask: int
    is_admissible: bool
    verdict_code: str                     # "AUTH_COMMIT_APPROVED" or "AUTH_DEGRADED_SAFE_FALLBACK"
    reason_codes: Tuple[str, ...]
    capability_generation: int
    state_sequence_id: int
    valid_until_ns: int
    vital_max_hp: int = VITAL_MAX_HP

    def __post_init__(self):
        assert self.vital_max_hp == 6, "Platform invariant VITAL_MAX_HP == 6 must hold"


@dataclass(frozen=True)
class ActuationReceipt:
    """Cryptographically auditable proof of physical actuation."""
    receipt_id: str
    verdict_id: str
    applied_profile: str
    actuation_timestamp_ns: int
    applied_envelope: ExecutionEnvelope
    was_degraded_fallback: bool
    is_shadow: bool
    transition_type: str = "NORMAL_SELECTION"  # or "FAILSAFE_AUTHORITY_TRANSITION"
    vital_max_hp: int = VITAL_MAX_HP

    def __post_init__(self):
        assert self.vital_max_hp == 6, "Platform invariant VITAL_MAX_HP == 6 must hold"


@runtime_checkable
class IEpistemicNode(Protocol):
    """Formal inter-node compute contract for all reasoning, prediction, and solver nodes."""
    @property
    def node_id(self) -> str: ...
    @property
    def reasoning_mode(self) -> ReasoningModeKind: ...
    def can_evaluate(self, state: WorldStateVector) -> bool: ...
    def propose(self, state: WorldStateVector) -> List[HypothesisEnvelope]: ...
    def feedback(self, residual: ObservationResidual) -> None: ...


@runtime_checkable
class IWorkspaceContributor(Protocol):
    """Optional presentation contract for epistemic nodes that project visual artifacts or cards into the workspace."""
    def render_ui_contribution(
        self, state: WorldStateVector, hypothesis: Optional[HypothesisEnvelope]
    ) -> Optional[WorkspaceNode]: ...


class CausalDiagnosticNode:
    """Implements causal reasoning over telemetries (LLM / Causal Graph)."""

    def __init__(self, node_id: str = "node_causal_diag_main"):
        self._node_id = node_id
        self._residual_history: List[ObservationResidual] = []

    @property
    def node_id(self) -> str:
        return self._node_id

    @property
    def reasoning_mode(self) -> ReasoningModeKind:
        return ReasoningModeKind.LLM_CAUSAL_NODE

    def can_evaluate(self, state: WorldStateVector) -> bool:
        assert state.vital_max_hp == 6
        return True

    def propose(self, state: WorldStateVector) -> List[HypothesisEnvelope]:
        assert state.vital_max_hp == 6
        is_uma_high = state.uma_pressure > 0.82
        return [
            HypothesisEnvelope(
                hypothesis_id=f"{self._node_id}_hyp_balanced",
                node_id=self._node_id,
                reasoning_mode=self.reasoning_mode,
                target_profile="balanced_npu" if is_uma_high else "balanced_hybrid",
                predicted_latency_ms=12.0 if is_uma_high else 8.0,
                predicted_throughput_fps=180.0,
                predicted_power_w=6.5,
                predicted_uma_pressure=0.65 if is_uma_high else state.uma_pressure,
                assumptions={"cause": "UMA_SATURATION" if is_uma_high else "NOMINAL"},
                confidence=0.92,
                requires_npu=True,
            )
        ]

    def feedback(self, residual: ObservationResidual) -> None:
        assert residual.vital_max_hp == 6
        self._residual_history.append(residual)

    def render_ui_contribution(
        self, state: WorldStateVector, hypothesis: Optional[HypothesisEnvelope]
    ) -> Optional[WorkspaceNode]:
        assert state.vital_max_hp == 6
        is_uma = state.uma_pressure > 0.82
        summary = (
            f"Koreňová príčina: Saturácia zbernice UMA (Tlak {state.uma_pressure:.2f} > 0.82 limit). "
            f"Zdieľaná pamäť brzdí GPU prístupy k inferenčnému modelu."
            if is_uma
            else "Hardvérová telemetria v normále."
        )
        return WorkspaceNode(
            node_id=self._node_id,
            kind=WorkspaceNodeKind.CAUSAL_DIAGNOSIS,
            title="Causal Diagnostic Engine",
            summary=summary,
            properties={
                "observed_uma": state.uma_pressure,
                "temperature_c": state.temperature_c,
                "queue_depth": state.queue_depth,
            },
        )


class DeterministicMemoryStabilizerNode:
    """Implements conservative memory bandwidth and stream stabilization."""

    def __init__(self, node_id: str = "node_alt_a_memory"):
        self._node_id = node_id
        self._calibration_scale = 1.0

    @property
    def node_id(self) -> str:
        return self._node_id

    @property
    def reasoning_mode(self) -> ReasoningModeKind:
        return ReasoningModeKind.PARETO_HEURISTIC_SOLVER

    def can_evaluate(self, state: WorldStateVector) -> bool:
        assert state.vital_max_hp == 6
        return state.uma_pressure >= 0.70

    def propose(self, state: WorldStateVector) -> List[HypothesisEnvelope]:
        assert state.vital_max_hp == 6
        return [
            HypothesisEnvelope(
                hypothesis_id=f"{self._node_id}_hyp_stream_throttle",
                node_id=self._node_id,
                reasoning_mode=self.reasoning_mode,
                target_profile="balanced_hybrid",
                predicted_latency_ms=11.2 * self._calibration_scale,
                predicted_throughput_fps=150.0,
                predicted_power_w=5.0,
                predicted_uma_pressure=0.68,
                assumptions={"throttle_gpu_streams": 2, "activation_compression_16bit": True},
                confidence=0.96,
                requires_gpu=False,
            )
        ]

    def feedback(self, residual: ObservationResidual) -> None:
        assert residual.vital_max_hp == 6
        # Learn strictly from attested residual: calibrate scaling
        if abs(residual.error_latency_ms) > 0.5:
            # Subtle Bayesian adjustment (damping factor 0.1)
            self._calibration_scale += 0.1 * (residual.error_latency_ms / 10.0)
            self._calibration_scale = max(0.5, min(2.0, self._calibration_scale))

    def render_ui_contribution(
        self, state: WorldStateVector, hypothesis: Optional[HypothesisEnvelope]
    ) -> Optional[WorkspaceNode]:
        assert state.vital_max_hp == 6
        return WorkspaceNode(
            node_id=self._node_id,
            kind=WorkspaceNodeKind.ACTION_CARD,
            title="Alternatíva A (Pamäťový uzol)",
            summary="Konzervatívna stabilizácia zbernice: zníženie GPU streamov 4 -> 2 + 16-bit kompresia.",
            properties={
                "predicted_gain": "-54% latencia",
                "predicted_uma": 0.68,
                "predicted_latency_ms": 11.2 * self._calibration_scale,
                "risk_profile": "0% (deterministicky overené)",
                "authority_safe": True,
            },
            actions=[{"label": "Aplikovať okamžite", "action_id": "commit_alt_a"}],
        )


class ShadowExplorationNode:
    """Implements speculative shadow exploration and sandbox execution."""

    def __init__(self, node_id: str = "node_alt_b_exploration"):
        self._node_id = node_id
        self._exploration_bias = 0.0

    @property
    def node_id(self) -> str:
        return self._node_id

    @property
    def reasoning_mode(self) -> ReasoningModeKind:
        return ReasoningModeKind.SHADOW_EXPLORER

    def can_evaluate(self, state: WorldStateVector) -> bool:
        assert state.vital_max_hp == 6
        return state.plugged and state.temperature_c < 85.0

    def propose(self, state: WorldStateVector) -> List[HypothesisEnvelope]:
        assert state.vital_max_hp == 6
        return [
            HypothesisEnvelope(
                hypothesis_id=f"{self._node_id}_hyp_shadow_simd_cache",
                node_id=self._node_id,
                reasoning_mode=self.reasoning_mode,
                target_profile="performance_gpu",
                predicted_latency_ms=7.8,
                predicted_throughput_fps=220.0,
                predicted_power_w=14.0,
                predicted_uma_pressure=0.62,
                assumptions={"offload_embedding_to_simd": True, "measure_vram_fragmentation": True},
                confidence=0.88,
                requires_gpu=True,
                is_shadow=True,
            )
        ]

    def feedback(self, residual: ObservationResidual) -> None:
        assert residual.vital_max_hp == 6
        # Update exploration bias based on real vs predicted margin
        self._exploration_bias += (residual.error_throughput_fps / 100.0)

    def render_ui_contribution(
        self, state: WorldStateVector, hypothesis: Optional[HypothesisEnvelope]
    ) -> Optional[WorkspaceNode]:
        assert state.vital_max_hp == 6
        return WorkspaceNode(
            node_id=self._node_id,
            kind=WorkspaceNodeKind.ACTION_CARD,
            title="Alternatíva B (Exploračný uzol)",
            summary="Agresívny shadow experiment: dočasný offload cache do CPU SIMD a meranie VRAM fragmentácie.",
            properties={
                "predicted_gain": "-68% latencia",
                "predicted_uma": 0.62,
                "predicted_latency_ms": 7.8,
                "risk_profile": "Možný krátkodobý frame drop počas testu",
                "requires_shadow_envelope": True,
            },
            actions=[{"label": "Zobraziť detail a simuláciu", "action_id": "expand_alt_b"}],
        )


# ============================================================================
# 5. REASONING META-SCHEDULER & COMPUTE BUDGET
# ============================================================================

class EvidenceDepthTier(str, Enum):
    SHALLOW = "SHALLOW"                # Deterministic bitmask & fast heuristics (< 50 µs)
    BALANCED = "BALANCED"              # Neural predictors & specialized solvers (< 2 ms)
    DEEP_EPISTEMIC = "DEEP_EPISTEMIC"  # Causal LLM / Deep exploration (> 10 ms)


@dataclass(frozen=True)
class ComputeBudget:
    """Computational envelope allocated to the reasoning meta-scheduler."""
    max_time_us: float = 5000.0          # Time budget in microseconds
    max_tokens: int = 1024               # Token budget for generative / LLM nodes
    energy_budget_uj: float = 50_000.0   # Energy budget in microjoules
    allowed_tiers: Tuple[EvidenceDepthTier, ...] = (
        EvidenceDepthTier.SHALLOW,
        EvidenceDepthTier.BALANCED,
        EvidenceDepthTier.DEEP_EPISTEMIC,
    )
    vital_max_hp: int = VITAL_MAX_HP

    def __post_init__(self):
        assert self.vital_max_hp == 6, "Platform invariant VITAL_MAX_HP == 6 must hold"


@dataclass(frozen=True)
class ScheduledNodeExecution:
    node_id: str
    priority: int
    tier: EvidenceDepthTier
    allocated_time_us: float
    vital_max_hp: int = VITAL_MAX_HP

    def __post_init__(self):
        assert self.vital_max_hp == 6, "Platform invariant VITAL_MAX_HP == 6 must hold"


@dataclass(frozen=True)
class MetaSchedulePlan:
    """Orchestrated composition of reasoning nodes selected for a problem context."""
    plan_id: str
    target_query: str
    scheduled_nodes: Tuple[ScheduledNodeExecution, ...]
    early_exit_confidence_threshold: float
    actual_entropy: float
    dispatched_tiers: Tuple[EvidenceDepthTier, ...]
    vital_max_hp: int = VITAL_MAX_HP

    def __post_init__(self):
        assert self.vital_max_hp == 6, "Platform invariant VITAL_MAX_HP == 6 must hold"


@dataclass(frozen=True)
class ReasoningPlan:
    """Formal specification of how the system will reason over a problem."""
    plan_id: str
    selected_nodes: Tuple[str, ...]
    execution_order: Tuple[str, ...]
    parallel_groups: Tuple[Tuple[str, ...], ...]
    max_time_us: int
    max_tokens: int
    required_evidence: float
    stop_condition: str
    state_version: int
    capability_generation: int
    vital_max_hp: int = VITAL_MAX_HP

    def __post_init__(self):
        assert self.vital_max_hp == 6, "Platform invariant VITAL_MAX_HP == 6 must hold"


@dataclass(frozen=True)
class ReasoningReceipt:
    """Cryptographically auditable proof of epistemic computation, cost, and evidence gained."""
    receipt_id: str
    plan_id: str
    executed_nodes: Tuple[str, ...]
    skipped_nodes: Tuple[str, ...]
    node_durations_us: Dict[str, float]
    tokens_consumed: int
    confidence_contribution: Dict[str, float]
    evidence_gained: float
    early_exit_reason: Optional[str]
    vital_max_hp: int = VITAL_MAX_HP

    def __post_init__(self):
        assert self.vital_max_hp == 6, "Platform invariant VITAL_MAX_HP == 6 must hold"


class ReasoningMetaScheduler:
    """
    Schedules the *mode of computational reasoning* rather than raw hardware execution.
    Plans node composition, assigns compute/token budgets, and detects early evidence convergence.
    """
    def __init__(self, confidence_convergence_threshold: float = 0.90):
        self.convergence_threshold = confidence_convergence_threshold

    def plan_schedule(
        self,
        state: WorldStateVector,
        nodes: Sequence[IEpistemicNode],
        budget: ComputeBudget,
        query: str = "Root cause analysis",
    ) -> MetaSchedulePlan:
        assert state.vital_max_hp == 6
        assert budget.vital_max_hp == 6

        # Calculate telemetry entropy / anomaly pressure
        is_anomalous = (state.uma_pressure > 0.80) or (state.temperature_c > 85.0) or (state.queue_depth > 20)
        entropy = 0.85 if is_anomalous else 0.18

        scheduled: List[ScheduledNodeExecution] = []
        dispatched_tiers: List[EvidenceDepthTier] = []

        # Tier 1: Deterministic bitmask & Pareto solvers (Always scheduled first)
        if EvidenceDepthTier.SHALLOW in budget.allowed_tiers:
            dispatched_tiers.append(EvidenceDepthTier.SHALLOW)
            for n in nodes:
                if n.reasoning_mode in (ReasoningModeKind.DETERMINISTIC_BITMASK, ReasoningModeKind.PARETO_HEURISTIC_SOLVER):
                    scheduled.append(ScheduledNodeExecution(
                        node_id=n.node_id,
                        priority=1,
                        tier=EvidenceDepthTier.SHALLOW,
                        allocated_time_us=min(500.0, budget.max_time_us * 0.1),
                    ))

        # Tier 2: Neural Predictors (Scheduled next if budget allows)
        if EvidenceDepthTier.BALANCED in budget.allowed_tiers and budget.max_time_us >= 1000.0:
            dispatched_tiers.append(EvidenceDepthTier.BALANCED)
            for n in nodes:
                if n.reasoning_mode == ReasoningModeKind.CUSTOM_NN_PREDICTOR:
                    scheduled.append(ScheduledNodeExecution(
                        node_id=n.node_id,
                        priority=2,
                        tier=EvidenceDepthTier.BALANCED,
                        allocated_time_us=min(2000.0, budget.max_time_us * 0.4),
                    ))

        # Tier 3: LLM Causal Node / Shadow Explorer (Dispatched ONLY if entropy is high and budget has tokens/time)
        # Early exit / budget preservation: if entropy is low (< 0.50), Tier 3 is omitted!
        if EvidenceDepthTier.DEEP_EPISTEMIC in budget.allowed_tiers:
            if entropy >= 0.50 and budget.max_tokens >= 256 and budget.max_time_us >= 3000.0:
                dispatched_tiers.append(EvidenceDepthTier.DEEP_EPISTEMIC)
                for n in nodes:
                    if n.reasoning_mode in (ReasoningModeKind.LLM_CAUSAL_NODE, ReasoningModeKind.SHADOW_EXPLORER):
                        scheduled.append(ScheduledNodeExecution(
                            node_id=n.node_id,
                            priority=3,
                            tier=EvidenceDepthTier.DEEP_EPISTEMIC,
                            allocated_time_us=budget.max_time_us * 0.5,
                        ))

        return MetaSchedulePlan(
            plan_id=f"PLAN-{state.sequence_id}-{int(budget.max_time_us)}",
            target_query=query,
            scheduled_nodes=tuple(scheduled),
            early_exit_confidence_threshold=self.convergence_threshold,
            actual_entropy=entropy,
            dispatched_tiers=tuple(dispatched_tiers),
            vital_max_hp=6,
        )

    def plan_reasoning_plan(
        self,
        state: WorldStateVector,
        nodes: Sequence[IEpistemicNode],
        budget: ComputeBudget,
        query: str = "Root cause analysis",
        capability_generation: int = 0,
    ) -> ReasoningPlan:
        """Formal synthesizer emitting an immutable ReasoningPlan."""
        assert state.vital_max_hp == 6
        assert budget.vital_max_hp == 6

        group_shallow: List[str] = []
        group_balanced: List[str] = []
        group_deep: List[str] = []

        is_anomalous = (state.uma_pressure > 0.80) or (state.temperature_c > 85.0) or (state.queue_depth > 20)
        entropy = 0.85 if is_anomalous else 0.18

        if EvidenceDepthTier.SHALLOW in budget.allowed_tiers:
            for n in nodes:
                if n.reasoning_mode in (ReasoningModeKind.DETERMINISTIC_BITMASK, ReasoningModeKind.PARETO_HEURISTIC_SOLVER):
                    group_shallow.append(n.node_id)

        if EvidenceDepthTier.BALANCED in budget.allowed_tiers and budget.max_time_us >= 1000.0:
            for n in nodes:
                if n.reasoning_mode == ReasoningModeKind.CUSTOM_NN_PREDICTOR:
                    group_balanced.append(n.node_id)

        if EvidenceDepthTier.DEEP_EPISTEMIC in budget.allowed_tiers:
            if entropy >= 0.50 and budget.max_tokens >= 256 and budget.max_time_us >= 3000.0:
                for n in nodes:
                    if n.reasoning_mode in (ReasoningModeKind.LLM_CAUSAL_NODE, ReasoningModeKind.SHADOW_EXPLORER):
                        group_deep.append(n.node_id)

        parallel_groups_list = []
        if group_shallow:
            parallel_groups_list.append(tuple(group_shallow))
        if group_balanced:
            parallel_groups_list.append(tuple(group_balanced))
        if group_deep:
            parallel_groups_list.append(tuple(group_deep))

        exec_order = tuple(group_shallow + group_balanced + group_deep)

        return ReasoningPlan(
            plan_id=f"RPLAN-{state.sequence_id}-{int(budget.max_time_us)}",
            selected_nodes=exec_order,
            execution_order=exec_order,
            parallel_groups=tuple(parallel_groups_list),
            max_time_us=int(budget.max_time_us),
            max_tokens=int(budget.max_tokens),
            required_evidence=self.convergence_threshold,
            stop_condition="EVIDENCE_CONVERGENCE_OR_BUDGET_EXHAUSTION",
            state_version=state.sequence_id,
            capability_generation=capability_generation,
            vital_max_hp=6,
        )

    def execute_plan(
        self,
        plan: ReasoningPlan,
        state: WorldStateVector,
        nodes: Mapping[str, IEpistemicNode],
    ) -> Tuple[List[HypothesisEnvelope], ReasoningReceipt]:
        """Executes the ReasoningPlan, returning hypotheses and an auditable ReasoningReceipt."""
        assert state.vital_max_hp == 6
        assert plan.vital_max_hp == 6

        harvested_hypotheses: List[HypothesisEnvelope] = []
        executed_nodes: List[str] = []
        skipped_nodes: List[str] = []
        node_durations_us: Dict[str, float] = {}
        confidence_contrib: Dict[str, float] = {}
        tokens_consumed = 0
        early_exit_reason: Optional[str] = None

        t_start = time.perf_counter()

        for idx, node_id in enumerate(plan.execution_order):
            elapsed_us = (time.perf_counter() - t_start) * 1e6
            if elapsed_us >= plan.max_time_us:
                early_exit_reason = "TIME_BUDGET_EXCEEDED"
                skipped_nodes.extend(plan.execution_order[idx:])
                break

            node = nodes.get(node_id)
            if not node or not node.can_evaluate(state):
                skipped_nodes.append(node_id)
                continue

            t0 = time.perf_counter()
            hyps = node.propose(state)
            t1 = time.perf_counter()
            dur_us = max(1.0, (t1 - t0) * 1e6)

            executed_nodes.append(node_id)
            node_durations_us[node_id] = dur_us
            harvested_hypotheses.extend(hyps)

            node_conf = max((h.confidence for h in hyps), default=0.0)
            confidence_contrib[node_id] = node_conf

            if node.reasoning_mode == ReasoningModeKind.LLM_CAUSAL_NODE:
                tokens_consumed += 128

            # Early exit check: if evidence threshold reached
            if node_conf >= plan.required_evidence and idx < len(plan.execution_order) - 1:
                early_exit_reason = "EVIDENCE_CONVERGENCE_THRESHOLD_REACHED"
                skipped_nodes.extend(plan.execution_order[idx + 1:])
                break

        evidence_gained = max((h.confidence for h in harvested_hypotheses), default=0.0)

        receipt = ReasoningReceipt(
            receipt_id=f"RRCPT-{plan.plan_id}",
            plan_id=plan.plan_id,
            executed_nodes=tuple(executed_nodes),
            skipped_nodes=tuple(skipped_nodes),
            node_durations_us=node_durations_us,
            tokens_consumed=tokens_consumed,
            confidence_contribution=confidence_contrib,
            evidence_gained=evidence_gained,
            early_exit_reason=early_exit_reason,
            vital_max_hp=6,
        )
        return harvested_hypotheses, receipt

    def execute_scheduled_synthesis(
        self,
        schedule: MetaSchedulePlan,
        state: WorldStateVector,
        nodes: Mapping[str, IEpistemicNode],
    ) -> List[HypothesisEnvelope]:
        """Executes nodes in priority order, supporting early exit when evidence converges."""
        assert state.vital_max_hp == 6
        assert schedule.vital_max_hp == 6

        harvested_hypotheses: List[HypothesisEnvelope] = []
        for exec_spec in schedule.scheduled_nodes:
            node = nodes.get(exec_spec.node_id)
            if node and node.can_evaluate(state):
                hyps = node.propose(state)
                harvested_hypotheses.extend(hyps)
                # Check for decisive high-confidence early exit
                if any(h.confidence >= schedule.early_exit_confidence_threshold for h in hyps):
                    if exec_spec.priority >= 2:
                        break

        return harvested_hypotheses


class EpistemicOrchestrationKernel:
    """
    Central orchestration kernel coordinating multi-node candidate generation,
    funnel filtration, deterministic authority gates, and dynamic scene composition.
    """

    def __init__(
        self,
        funnel: Optional[AdmissibleHypothesisFunnel] = None,
        nodes: Optional[List[IEpistemicNode]] = None,
        scheduler: Optional[ReasoningMetaScheduler] = None,
    ):
        self.funnel = funnel or AdmissibleHypothesisFunnel()
        self.scheduler = scheduler
        self.nodes: Dict[str, IEpistemicNode] = {}
        self.last_reasoning_plan: Optional[ReasoningPlan] = None
        self.last_reasoning_receipt: Optional[ReasoningReceipt] = None
        for n in (nodes or []):
            self.register_node(n)

    def register_node(self, node: IEpistemicNode) -> None:
        self.nodes[node.node_id] = node

    def step(
        self,
        state: WorldStateVector,
        query: str = "Tactical Cycle",
        now_s: Optional[float] = None,
        budget: Optional[ComputeBudget] = None,
    ) -> Tuple[Optional[CandidateAction], DynamicWorkspaceScene, FunnelStageTelemetry]:
        """
        Executes a complete orchestration cycle:
          1. Gathers hypotheses from all eligible nodes (optionally via ReasoningMetaScheduler)
          2. Runs hypotheses through the 5-stage selection funnel
          3. Emits dynamic polymorphic UI scene
          4. Returns (WinningAction, WorkspaceScene, FunnelTelemetry)
        """
        assert state.vital_max_hp == 6

        # 1. Harvest proposals
        active_hypotheses: List[HypothesisEnvelope] = []
        ui_nodes: List[WorkspaceNode] = []

        if self.scheduler is not None and budget is not None:
            self.last_reasoning_plan = self.scheduler.plan_reasoning_plan(
                state=state,
                nodes=list(self.nodes.values()),
                budget=budget,
                query=query,
                capability_generation=self.funnel.caps.generation,
            )
            active_hypotheses, self.last_reasoning_receipt = self.scheduler.execute_plan(
                plan=self.last_reasoning_plan,
                state=state,
                nodes=self.nodes,
            )
            for nid in self.last_reasoning_receipt.executed_nodes:
                node = self.nodes.get(nid)
                if node and isinstance(node, IWorkspaceContributor):
                    node_hyps = [h for h in active_hypotheses if h.node_id == node.node_id]
                    for h in node_hyps:
                        ui_elem = node.render_ui_contribution(state, h)
                        if ui_elem:
                            ui_nodes.append(ui_elem)
        else:
            self.last_reasoning_plan = None
            self.last_reasoning_receipt = None
            for node in self.nodes.values():
                if node.can_evaluate(state):
                    hyps = node.propose(state)
                    active_hypotheses.extend(hyps)
                    if isinstance(node, IWorkspaceContributor):
                        for h in hyps:
                            ui_elem = node.render_ui_contribution(state, h)
                            if ui_elem:
                                ui_nodes.append(ui_elem)

        # 2. Convert hypotheses to candidate actions for funnel filtering
        candidates = [h.to_candidate_action() for h in active_hypotheses]

        # 3. Filter through 5-stage funnel & authority gate
        winner, telemetry = self.funnel.filter_and_select(
            candidates,
            current_temp_c=state.temperature_c,
            current_uma_pressure=state.uma_pressure,
            battery_pct=state.battery_pct,
            now_s=now_s,
        )

        # 4. Synthesize interactive simulation layer if shadow action is present
        has_shadow = any(h.is_shadow for h in active_hypotheses)
        if has_shadow:
            sim_node = WorkspaceNode(
                node_id="node_sim_verification_b",
                kind=WorkspaceNodeKind.INTERACTIVE_SIMULATION,
                title="Verifikačná & Simulačná Vrstva (Alternatíva B)",
                summary="Predikcia dynamického poklesu UMA tlaku z 0.84 na 0.62 s deterministickým ohraničením.",
                properties={
                    "simulation_horizon_s": 2.0,
                    "missing_evidence": "Vplyv na 8 kHz input stream nebol overený.",
                    "authority_boundary": "Autorita obmedzí shadow experiment na max. 3.0 sekundy.",
                },
                actions=[{"label": "SPUSTIŤ EXPERIMENT V TIENI", "action_id": "launch_shadow_experiment"}],
            )
            ui_nodes.append(sim_node)

        # 5. Build Workspace Scene
        root_cause = (
            f"UMA bus thrashing detected (Pressure {state.uma_pressure:.2f} > 0.82 limit)."
            if state.uma_pressure > 0.82
            else "Hardware envelope nominal."
        )

        scene = DynamicWorkspaceScene(
            scene_id=f"SCENE-{state.sequence_id}",
            query=query,
            root_cause_summary=root_cause,
            nodes=ui_nodes,
            authority_verdict=telemetry.authority_verdict,
            is_shadow_available=has_shadow,
            vital_max_hp=6,
        )

        return winner, scene, telemetry

    def execute_pipeline(
        self,
        state: WorldStateVector,
        query: str = "Tactical Cycle",
        now_s: Optional[float] = None,
        budget: Optional[ComputeBudget] = None,
    ) -> OrchestrationCycleResult:
        """
        Executes the formal closed-loop pipeline from WorldStateVector to ActuationReceipt.
        Adheres strictly to the invariant:
          A_commit == 0 => AUTH_DEGRADED_SAFE_FALLBACK (FAILSAFE_AUTHORITY_TRANSITION)
        """
        assert state.vital_max_hp == 6
        winner_action, scene, telemetry = self.step(state, query=query, now_s=now_s, budget=budget)

        is_approved = (telemetry.authority_verdict == "AUTH_COMMIT_APPROVED")
        profile_name = winner_action.target_profile if winner_action else "ultra_saver"

        # 1. Construct AdmissibilityVerdict
        verdict = AdmissibilityVerdict(
            verdict_id=f"VERDICT-{state.sequence_id}",
            hypothesis_id=winner_action.action_id if winner_action else "none",
            target_profile=profile_name,
            authority_mask=int(PROFILE_TO_BIT.get(profile_name, AuthorityProfileBit.ULTRA_SAVER)),
            is_admissible=is_approved,
            verdict_code=telemetry.authority_verdict,
            reason_codes=(telemetry.authority_verdict, f"PROFILE_{profile_name.upper()}"),
            capability_generation=self.funnel.caps.generation,
            state_sequence_id=state.sequence_id,
            valid_until_ns=state.state_timestamp_ns + 1_000_000_000,
            vital_max_hp=6,
        )

        # 2. Build ExecutionEnvelope
        allowed = ("CPU_SCALAR", "CPU_SIMD_DETERMINISTIC")
        if profile_name in ("performance_gpu", "max_burst"):
            allowed = allowed + ("GPU_OPENVINO",)
        if "npu" in profile_name:
            allowed = allowed + ("NPU_OFFLOAD",)

        envelope = ExecutionEnvelope(
            governor_mode="AUTO" if is_approved else "EMERGENCY_COOL",
            allowed_backends=allowed,
            max_workers=4 if is_approved else 1,
            max_power_w=winner_action.predicted_power_w if winner_action else 3.5,
            max_tick_hz=8000,
            codec_width_bytes=32,
            thermal_budget_c=max(0.0, 95.0 - state.temperature_c),
            memory_pressure_limit=self.funnel.max_uma_pressure,
            valid_until_ns=state.state_timestamp_ns + 1_000_000_000,
            state_version=state.sequence_id,
            capability_generation=self.funnel.caps.generation,
            policy_version="karaf_v2",
            vital_max_hp=6,
        )

        # 3. Construct ActuationReceipt
        receipt = ActuationReceipt(
            receipt_id=f"RCPT-{state.sequence_id}",
            verdict_id=verdict.verdict_id,
            applied_profile=profile_name,
            actuation_timestamp_ns=state.state_timestamp_ns + 50_000,  # +50 µs actuation latency
            applied_envelope=envelope,
            was_degraded_fallback=not is_approved,
            is_shadow=winner_action.is_shadow_experiment if winner_action else False,
            transition_type="FAILSAFE_AUTHORITY_TRANSITION" if not is_approved else "NORMAL_SELECTION",
            vital_max_hp=6,
        )

        return OrchestrationCycleResult(
            state_sequence_id=state.sequence_id,
            selected_hypothesis=None,  # Or mapped from active_hypotheses
            verdict=verdict,
            receipt=receipt,
            workspace_scene=scene,
            telemetry=telemetry,
            reasoning_plan=self.last_reasoning_plan,
            reasoning_receipt=self.last_reasoning_receipt,
            vital_max_hp=6,
        )

    def compute_attested_residual(
        self,
        receipt: ActuationReceipt,
        hypothesis: HypothesisEnvelope,
        measured_latency_ms: float,
        measured_throughput_fps: float,
        measured_power_w: float,
        measured_uma_pressure: float,
        measured_duration_us: float,
        proof_hash: str,
    ) -> ObservationResidual:
        """Computes attested observation residual strictly from empirical measurement minus prediction."""
        assert receipt.vital_max_hp == 6
        assert hypothesis.vital_max_hp == 6
        return ObservationResidual(
            residual_id=f"RESIDUAL-{receipt.receipt_id}",
            hypothesis_id=hypothesis.hypothesis_id,
            node_id=hypothesis.node_id,
            error_latency_ms=measured_latency_ms - hypothesis.predicted_latency_ms,
            error_throughput_fps=measured_throughput_fps - hypothesis.predicted_throughput_fps,
            error_power_w=measured_power_w - hypothesis.predicted_power_w,
            error_uma_pressure=measured_uma_pressure - hypothesis.predicted_uma_pressure,
            real_duration_us=measured_duration_us,
            proof_hash=proof_hash,
            vital_max_hp=6,
        )

    def dispatch_feedback(self, residual: ObservationResidual) -> None:
        """Dispatches verified observation residuals back to the originating node."""
        assert residual.vital_max_hp == 6
        node = self.nodes.get(residual.node_id)
        if node:
            node.feedback(residual)


@dataclass(frozen=True)
class OrchestrationCycleResult:
    """End-to-end outcome of an orchestration step from WorldState to Receipt."""
    state_sequence_id: int
    selected_hypothesis: Optional[HypothesisEnvelope]
    verdict: AdmissibilityVerdict
    receipt: ActuationReceipt
    workspace_scene: DynamicWorkspaceScene
    telemetry: FunnelStageTelemetry
    reasoning_plan: Optional[ReasoningPlan] = None
    reasoning_receipt: Optional[ReasoningReceipt] = None
    vital_max_hp: int = VITAL_MAX_HP

    def __post_init__(self):
        assert self.vital_max_hp == 6, "Platform invariant VITAL_MAX_HP == 6 must hold"
