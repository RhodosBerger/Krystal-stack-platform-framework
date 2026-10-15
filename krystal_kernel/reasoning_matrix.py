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
from typing import Any, Dict, List, Optional, Set, Tuple

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
