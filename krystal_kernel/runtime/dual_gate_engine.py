#!/usr/bin/env python3
"""
==============================================================================
KRYSTAL-STACK: DUAL-GATE GOVERNANCE ENGINE (SAFETYGATE + EVIDENCEGATE)
==============================================================================
Module: krystal_kernel/runtime/dual_gate_engine.py
Description:
    Decomposes authoritative governance into two strictly orthogonal gates:
      1. SafetyGate: Verifies constitutional invariants (VITAL_MAX_HP == 6),
         thermal headroom, power envelopes, and observer overhead.
      2. EvidenceGate: Verifies empirical ground truth, confidence thresholds,
         sample sizes, and temporal TTL leases.

    Authority Function:
      f(SafetyGate, EvidenceGate) -> AuthorityState
        - (Safe, Evidenced)   -> AUTH_COMMIT_APPROVED (Permit actuation)
        - (Safe, Unevidenced) -> AUTH_DEFERRED        (Trigger micro-probes)
        - (Safe, Conflicted)  -> AUTH_QUARANTINED     (Freeze and recalibrate)
        - (Unsafe, *)         -> AUTH_VETOED          (Strict invariant block)

Constitutional Invariant:
    assert VITAL_MAX_HP == 6
==============================================================================
"""

import time
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Optional, Tuple

from krystal_kernel.scene_constructor_presets import VITAL_MAX_HP
from krystal_kernel.verification.semantic_protocol import (
    AuthorityState,
    SafetyState,
    EpistemicState,
    TemporalState,
    to_wire_token
)
from krystal_kernel.runtime.epistemic_type_system import EpistemicValue, LatticeCertaintyLevel

assert VITAL_MAX_HP == 6, f"Constitutional invariant breach: VITAL_MAX_HP must be 6, got {VITAL_MAX_HP}"


@dataclass
class SafetyGateResult:
    """Outcome of SafetyGate verification."""
    is_safe: bool
    safety_code: SafetyState
    reason: str
    thermal_celsius: float
    power_watts: float
    observer_overhead_pct: float
    vital_max_hp: int = VITAL_MAX_HP

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_safe": self.is_safe,
            "safety_code": self.safety_code.name,
            "safety_token": to_wire_token(self.safety_code),
            "reason": self.reason,
            "thermal_celsius": self.thermal_celsius,
            "power_watts": self.power_watts,
            "observer_overhead_pct": self.observer_overhead_pct,
            "vital_max_hp": self.vital_max_hp
        }


@dataclass
class EvidenceGateResult:
    """Outcome of EvidenceGate verification."""
    is_evidenced: bool
    epistemic_state: EpistemicState
    belief_degree: float
    evidence_tokens: List[str]
    is_quarantined: bool
    is_contradictory: bool
    reason: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_evidenced": self.is_evidenced,
            "epistemic_state": self.epistemic_state.name,
            "epistemic_token": to_wire_token(self.epistemic_state),
            "belief_degree": round(self.belief_degree, 4),
            "evidence_tokens": self.evidence_tokens,
            "is_quarantined": self.is_quarantined,
            "is_contradictory": self.is_contradictory,
            "reason": self.reason
        }


@dataclass
class DualGateVerdict:
    """Final authoritative decision produced by DualGateEvaluator."""
    authority_state: AuthorityState
    safety: SafetyGateResult
    evidence: EvidenceGateResult
    verdict_message: str
    evaluated_at_ns: int = field(default_factory=time.perf_counter_ns)
    vital_max_hp: int = VITAL_MAX_HP

    @property
    def can_actuate(self) -> bool:
        return self.authority_state == AuthorityState.AUTH_COMMIT_APPROVED

    def to_dict(self) -> Dict[str, Any]:
        return {
            "authority_state": self.authority_state.name,
            "authority_token": to_wire_token(self.authority_state),
            "can_actuate": self.can_actuate,
            "safety": self.safety.to_dict(),
            "evidence": self.evidence.to_dict(),
            "verdict_message": self.verdict_message,
            "evaluated_at_ns": self.evaluated_at_ns,
            "vital_max_hp": self.vital_max_hp
        }


class SafetyGate:
    """
    Evaluates physical, silicon, and invariant boundaries:
      - Strictly enforces VITAL_MAX_HP == 6
      - Max thermal threshold: 85.0 °C
      - Max power budget: 28.0 W
      - Max observer overhead: 2.5%
    """

    def __init__(
        self,
        max_thermal_celsius: float = 85.0,
        max_power_watts: float = 28.0,
        max_observer_overhead_pct: float = 2.5
    ):
        self.max_thermal = max_thermal_celsius
        self.max_power = max_power_watts
        self.max_observer_overhead = max_observer_overhead_pct

    def evaluate(
        self,
        vital_hp: int = VITAL_MAX_HP,
        thermal_celsius: float = 52.0,
        power_watts: float = 18.0,
        observer_overhead_pct: float = 1.8
    ) -> SafetyGateResult:
        # 1. Constitutional Invariant Check
        if vital_hp != VITAL_MAX_HP:
            return SafetyGateResult(
                is_safe=False,
                safety_code=SafetyState.SAFE_BREACH,
                reason=f"Constitutional invariant breach: VITAL_MAX_HP {vital_hp} != {VITAL_MAX_HP}",
                thermal_celsius=thermal_celsius,
                power_watts=power_watts,
                observer_overhead_pct=observer_overhead_pct
            )

        # 2. Thermal Headroom Check
        if thermal_celsius > self.max_thermal:
            return SafetyGateResult(
                is_safe=False,
                safety_code=SafetyState.SAFE_BREACH,
                reason=f"Thermal threshold exceeded: {thermal_celsius:.1f}°C > {self.max_thermal}°C",
                thermal_celsius=thermal_celsius,
                power_watts=power_watts,
                observer_overhead_pct=observer_overhead_pct
            )

        # 3. Power Budget Check
        if power_watts > self.max_power:
            return SafetyGateResult(
                is_safe=False,
                safety_code=SafetyState.SAFE_BREACH,
                reason=f"Power budget exceeded: {power_watts:.1f}W > {self.max_power}W",
                thermal_celsius=thermal_celsius,
                power_watts=power_watts,
                observer_overhead_pct=observer_overhead_pct
            )

        # 4. Observer Overhead Check
        if observer_overhead_pct > self.max_observer_overhead:
            return SafetyGateResult(
                is_safe=False,
                safety_code=SafetyState.SAFE_BREACH,
                reason=f"Observer overhead exceeded limit: {observer_overhead_pct:.2f}% > {self.max_observer_overhead}%",
                thermal_celsius=thermal_celsius,
                power_watts=power_watts,
                observer_overhead_pct=observer_overhead_pct
            )

        return SafetyGateResult(
            is_safe=True,
            safety_code=SafetyState.SAFE_PASS,
            reason="All physical invariants and safety boundaries verified.",
            thermal_celsius=thermal_celsius,
            power_watts=power_watts,
            observer_overhead_pct=observer_overhead_pct
        )


class EvidenceGate:
    """
    Evaluates epistemic ground truth, certainty thresholds, and empirical confidence:
      - Minimum belief degree: 0.70
      - Quarantine check: stream must not be quarantined
      - Contradiction check: no opposing sensor claims
    """

    def __init__(self, min_belief_degree: float = 0.70):
        self.min_belief_degree = min_belief_degree

    def evaluate(self, epistemic_val: Optional[EpistemicValue[Any]] = None) -> EvidenceGateResult:
        if epistemic_val is None:
            return EvidenceGateResult(
                is_evidenced=False,
                epistemic_state=EpistemicState.EPI_UNKNOWN,
                belief_degree=0.0,
                evidence_tokens=[],
                is_quarantined=False,
                is_contradictory=False,
                reason="Missing epistemic value (unobserved context)"
            )

        # Check quarantine
        if epistemic_val.epistemic_state == EpistemicState.EPI_QUARANTINED:
            return EvidenceGateResult(
                is_evidenced=False,
                epistemic_state=EpistemicState.EPI_QUARANTINED,
                belief_degree=epistemic_val.belief_degree,
                evidence_tokens=epistemic_val.evidence_tokens,
                is_quarantined=True,
                is_contradictory=False,
                reason="Sensor stream quarantined due to fault or suspicious divergence"
            )

        # Check contradiction
        if epistemic_val.epistemic_state == EpistemicState.EPI_CONTRADICTORY:
            return EvidenceGateResult(
                is_evidenced=False,
                epistemic_state=EpistemicState.EPI_CONTRADICTORY,
                belief_degree=epistemic_val.belief_degree,
                evidence_tokens=epistemic_val.evidence_tokens,
                is_quarantined=False,
                is_contradictory=True,
                reason="Contradictory evidence detected across probe channels"
            )

        # Check falsification
        if epistemic_val.is_falsified:
            return EvidenceGateResult(
                is_evidenced=False,
                epistemic_state=EpistemicState.EPI_FALSIFIED,
                belief_degree=epistemic_val.belief_degree,
                evidence_tokens=epistemic_val.evidence_tokens,
                is_quarantined=False,
                is_contradictory=False,
                reason="Empirical hypothesis falsified by evidence probes"
            )

        # Check belief degree
        if epistemic_val.belief_degree < self.min_belief_degree:
            return EvidenceGateResult(
                is_evidenced=False,
                epistemic_state=EpistemicState.EPI_CALIBRATED,
                belief_degree=epistemic_val.belief_degree,
                evidence_tokens=epistemic_val.evidence_tokens,
                is_quarantined=False,
                is_contradictory=False,
                reason=f"Insufficient belief degree: {epistemic_val.belief_degree:.2f} < {self.min_belief_degree:.2f}"
            )

        return EvidenceGateResult(
            is_evidenced=True,
            epistemic_state=epistemic_val.epistemic_state,
            belief_degree=epistemic_val.belief_degree,
            evidence_tokens=epistemic_val.evidence_tokens,
            is_quarantined=False,
            is_contradictory=False,
            reason="Sufficient empirical evidence corroborated"
        )


class DualGateEvaluator:
    """
    Synthesizes SafetyGate and EvidenceGate into an authoritative verdict:
      f(SafetyGate, EvidenceGate) -> AuthorityState
    """

    def __init__(self, safety_gate: Optional[SafetyGate] = None, evidence_gate: Optional[EvidenceGate] = None):
        self.safety_gate = safety_gate or SafetyGate()
        self.evidence_gate = evidence_gate or EvidenceGate()

    def evaluate(
        self,
        epistemic_val: Optional[EpistemicValue[Any]] = None,
        vital_hp: int = VITAL_MAX_HP,
        thermal_celsius: float = 52.0,
        power_watts: float = 18.0,
        observer_overhead_pct: float = 1.8
    ) -> DualGateVerdict:
        safety_res = self.safety_gate.evaluate(
            vital_hp=vital_hp,
            thermal_celsius=thermal_celsius,
            power_watts=power_watts,
            observer_overhead_pct=observer_overhead_pct
        )
        evidence_res = self.evidence_gate.evaluate(epistemic_val)

        # Rule 1: Any safety failure is an absolute VETO
        if not safety_res.is_safe:
            return DualGateVerdict(
                authority_state=AuthorityState.AUTH_VETOED,
                safety=safety_res,
                evidence=evidence_res,
                verdict_message=f"AUTH_VETOED: SafetyGate rejection ({safety_res.reason})"
            )

        # Rule 2: Safe, but quarantined or contradictory -> QUARANTINE
        if evidence_res.is_quarantined or evidence_res.is_contradictory:
            return DualGateVerdict(
                authority_state=AuthorityState.AUTH_QUARANTINED,
                safety=safety_res,
                evidence=evidence_res,
                verdict_message=f"AUTH_QUARANTINED: Stream anomaly detected ({evidence_res.reason})"
            )

        # Rule 3: Safe, but insufficient evidence -> DEFER (triggers micro-probes)
        if not evidence_res.is_evidenced:
            return DualGateVerdict(
                authority_state=AuthorityState.AUTH_DEFERRED,
                safety=safety_res,
                evidence=evidence_res,
                verdict_message=f"AUTH_DEFERRED: Insufficient empirical evidence ({evidence_res.reason})"
            )

        # Rule 4: Safe AND Evidenced -> COMMIT APPROVED
        return DualGateVerdict(
            authority_state=AuthorityState.AUTH_COMMIT_APPROVED,
            safety=safety_res,
            evidence=evidence_res,
            verdict_message="AUTH_COMMIT_APPROVED: Both SafetyGate and EvidenceGate cleared."
        )


# Global Singleton Evaluator
GLOBAL_DUAL_GATE_EVALUATOR = DualGateEvaluator()
