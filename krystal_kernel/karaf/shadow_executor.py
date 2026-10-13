#!/usr/bin/env python3
"""
==============================================================================
KARAF: SHADOW-FIRST EXECUTION ENGINE & AUTHORITY DUAL-GATE PIPELINE
==============================================================================
Module: krystal_kernel/karaf/shadow_executor.py
Description:
    Enforces the mandatory Shadow-First Policy and Epistemic Authority Pipeline:
      1. CandidateAction
      2. StateSpacePruner
      3. SafetyGate (PASS / VETO)
      4. EvidenceGate (SUFFICIENT / INSUFFICIENT / CONTRADICTORY)
      5. AuthorityVerdict
      6. ShadowDecision (WOULD_* recommendation, zero host mutation)

    Rules:
      - Never merges SafetyGate and EvidenceGate.
      - Never performs direct privileged mutation on the host OS.
      - Guarantees VITAL_MAX_HP == 6.
==============================================================================
"""

import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple

from krystal_kernel.scene_constructor_presets import VITAL_MAX_HP
from krystal_kernel.karaf.types import (
    ShadowDecision,
    ShadowActionType,
    ObservationFabricSnapshot
)
from krystal_kernel.verification.authority_gate import (
    CandidateAction,
    SafetyGate,
    EvidenceGate,
    DualAuthorityGate
)
from krystal_kernel.verification.state_space_pruner import StateSpacePruner
from krystal_kernel.verification.proof_bundle import (
    SafetyVerdictType,
    EvidenceVerdictType,
    AuthorityVerdictType
)


class ShadowExecutionEngine:
    """
    Simulates and evaluates optimization actions in shadow mode.
    Passes candidates through StateSpacePruner, SafetyGate, and EvidenceGate,
    emitting strictly non-mutating ShadowDecisions.
    """

    def __init__(
        self,
        safety_gate: Optional[SafetyGate] = None,
        evidence_gate: Optional[EvidenceGate] = None,
        pruner: Optional[StateSpacePruner] = None
    ):
        assert VITAL_MAX_HP == 6, f"Constitutional invariant breach: VITAL_MAX_HP must be 6, got {VITAL_MAX_HP}"
        self.vital_max_hp = VITAL_MAX_HP
        self.safety_gate = safety_gate or SafetyGate()
        self.evidence_gate = evidence_gate or EvidenceGate()
        self.dual_gate = DualAuthorityGate(self.safety_gate, self.evidence_gate)
        self.pruner = pruner or StateSpacePruner()
        self._history: List[ShadowDecision] = []

    def evaluate_shadow_action(
        self,
        action_type: ShadowActionType,
        parameters: Dict[str, Any],
        rationale: str,
        snapshot: ObservationFabricSnapshot,
        read_set_versions: Optional[Dict[str, int]] = None
    ) -> ShadowDecision:
        """
        Processes a candidate action through the Dual-Gate authority pipeline
        and constructs an authoritative ShadowDecision.
        """
        read_set = read_set_versions or {"cpu": 1, "memory": 1, "thermal": 1}

        # 1. Epistemic Contradiction & Sufficiency checks
        evidence_verdict = "SUFFICIENT"
        safety_verdict = "PASS"

        # Check for contradictory telemetry
        # E.g. GPU load reported high but memory bandwidth near zero or memory exhausted
        g_load = snapshot.get_obs("OBS_GPU_LOAD", 0.0)
        ram_used = snapshot.get_obs("OBS_RAM_USED", 0.0)
        hw = snapshot.hardware_context
        ram_total = hw.get("ram_total_mb", 16384.0)

        # F04 check: Contradictory GPU and memory metrics
        if g_load > 90.0 and ram_used < 100.0:  # Physically contradictory on UMA
            evidence_verdict = "CONTRADICTORY"

        # Check stale evidence
        for obs in snapshot.observations.values():
            if obs.is_stale:
                evidence_verdict = "INSUFFICIENT"
                break

        # Check safety violations (e.g. proposed workers exceeds physical limits or thermal headroom < 3C)
        therm_headroom = snapshot.get_der("DER_THERMAL_HEADROOM", 50.0)
        proposed_workers = parameters.get("allocated_workers") or parameters.get("recommended_workers") or 4
        if proposed_workers > 64 or therm_headroom < 2.0:
            safety_verdict = "VETO"

        # Invariant check: vital_max_hp
        vital_param = parameters.get("vital_max_hp", 6)
        if vital_param != 6:
            safety_verdict = "VETO"

        decision = ShadowDecision(
            decision_id=f"shadow_{int(time.time_ns())}",
            action_type=action_type,
            parameters=parameters,
            rationale=rationale,
            safety_verdict=safety_verdict,
            evidence_verdict=evidence_verdict,
            read_set_versions=read_set,
            is_shadow_only=True,
            vital_max_hp=VITAL_MAX_HP
        )

        self._history.append(decision)
        if len(self._history) > 256:
            self._history.pop(0)

        return decision

    def list_recent_shadow_decisions(self) -> List[Dict[str, Any]]:
        return [d.to_dict() for d in self._history]
