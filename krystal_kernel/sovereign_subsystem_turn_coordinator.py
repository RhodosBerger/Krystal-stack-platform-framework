#!/usr/bin/env python3
"""
================================================================================
KRYSTAL STACK PLATFORM FRAMEWORK - SOVEREIGN SUBSYSTEM TURN COORDINATOR
================================================================================
Module:     krystal_kernel/sovereign_subsystem_turn_coordinator.py
Tenets:     KSPP v1 (P1-P10) | V-KSPP-PRINCIPLES-01
Components:
  1. ProceduralAgileDraftingInterpolator (Adaptive EMA & Procedural Drafting)
  2. SafeOperationSelectionPack (Multi-Gate Operational Vetting & Scored Vetting)
  3. InstructionalDependencyCoordinator & SubsystemTurnLogicBehavior
Author:     Antigravity Cognitive Invariants Group
Invariants: VITAL_MAX_HP == 6 | Proposer != Evaluator | Delta S == 0 on breach
================================================================================
"""

from __future__ import annotations

import enum
import hashlib
import hmac
import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from .constants import (
    VITAL_MAX_HP,
    UMA_SATURATION_THRESHOLD,
    HYSTERESIS_DWELL_FRAMES,
    ARRHENIUS_MAX_TEMP_C,
    CapabilityScope,
    ContractViolation,
)
from .sovereign_runtime import SubsystemNodeId


# =============================================================================
# ENUMS & STATUS CODES
# =============================================================================

class TurnState(str, enum.Enum):
    """Authoritative states of a subsystem in the turn-taking lifecycle."""
    IDLE = "IDLE"
    AWAITING_DEPENDENCY = "AWAITING_DEPENDENCY"
    REQUESTING_TURN = "REQUESTING_TURN"
    TURN_GRANTED = "TURN_GRANTED"
    EXECUTING_SLICE = "EXECUTING_SLICE"
    TURN_YIELDED = "TURN_YIELDED"
    TURN_EXPIRED_ROLLBACK = "TURN_EXPIRED_ROLLBACK"


class OperationSafetyVerdict(str, enum.Enum):
    """Evaluation verdict emitted by SafeOperationSelectionPack."""
    ACCEPTED_AND_DRAFTED = "ACCEPTED_AND_DRAFTED"
    REJECTED_CONSTITUTIONAL_BREACH = "REJECTED_CONSTITUTIONAL_BREACH"
    REJECTED_AUTHORITY_SEPARATION = "REJECTED_AUTHORITY_SEPARATION"
    REJECTED_CAPABILITY_DENIED = "REJECTED_CAPABILITY_DENIED"
    REJECTED_THERMAL_HEADROOM = "REJECTED_THERMAL_HEADROOM"
    REJECTED_BUS_SATURATED = "REJECTED_BUS_SATURATED"
    REJECTED_DEPENDENCY_UNRESOLVED = "REJECTED_DEPENDENCY_UNRESOLVED"
    REJECTED_DOMAIN_CONFLICT = "REJECTED_DOMAIN_CONFLICT"


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass(frozen=True)
class AgileDraftedEnvelope:
    """Procedurally drafted execution envelope for a prospective operation."""
    allocated_quantum_us: float
    max_energy_budget_uj: float
    thermal_headroom_c: float
    uma_bus_ceiling: float
    hysteresis_dwell_required: int
    draft_confidence: float
    vital_max_hp: int = VITAL_MAX_HP

    def __post_init__(self) -> None:
        if self.vital_max_hp != VITAL_MAX_HP:
            raise ContractViolation(f"Constitutional HP breach in AgileDraftedEnvelope: {self.vital_max_hp} != {VITAL_MAX_HP}")


@dataclass
class CandidateOperation:
    """Represents a prospective operation submitted to the selection pack."""
    operation_id: str
    title: str
    subsystem: SubsystemNodeId
    proposer_id: str
    evaluator_id: str
    required_capability: CapabilityScope
    estimated_latency_us: float
    estimated_energy_uj: float
    priority: int = 1  # 1 (Normal) to 5 (Critical)
    depends_on: List[str] = field(default_factory=list)
    resource_domains: Set[str] = field(default_factory=set)
    is_reversible: bool = True
    vital_max_hp: int = VITAL_MAX_HP

    def __post_init__(self) -> None:
        if self.vital_max_hp != VITAL_MAX_HP:
            raise ContractViolation(f"Constitutional HP breach in CandidateOperation: {self.vital_max_hp} != {VITAL_MAX_HP}")


@dataclass(frozen=True)
class SelectionEvaluationResult:
    """Formal decision returned by the SafeOperationSelectionPack."""
    operation_id: str
    verdict: OperationSafetyVerdict
    is_authorized: bool
    drafted_envelope: Optional[AgileDraftedEnvelope]
    rejection_reason: str = ""
    evaluation_latency_us: float = 0.0
    hmac_digest: str = ""
    vital_max_hp: int = VITAL_MAX_HP

    def __post_init__(self) -> None:
        if self.vital_max_hp != VITAL_MAX_HP:
            raise ContractViolation(f"Constitutional HP breach in SelectionEvaluationResult: {self.vital_max_hp} != {VITAL_MAX_HP}")


@dataclass(frozen=True)
class TurnToken:
    """Authoritative bus-mastering token granting single-subsystem write capability."""
    token_id: str
    subsystem: SubsystemNodeId
    operation_id: str
    allocated_quantum_us: float
    granted_at_ns: int
    expires_at_ns: int
    hmac_signature: str
    vital_max_hp: int = VITAL_MAX_HP

    def __post_init__(self) -> None:
        if self.vital_max_hp != VITAL_MAX_HP:
            raise ContractViolation(f"Constitutional HP breach in TurnToken: {self.vital_max_hp} != {VITAL_MAX_HP}")


@dataclass(frozen=True)
class InstructionReceipt:
    """Cryptographic execution receipt of a completed instruction in the dependency graph."""
    instruction_id: str
    subsystem: SubsystemNodeId
    status: str
    latency_us: float
    energy_uj: float
    delta_s: int  # 0 on rollback, 1 on state commitment
    timestamp_ns: int
    hmac_signature: str
    vital_max_hp: int = VITAL_MAX_HP


# =============================================================================
# COMPONENT 1: PROCEDURAL AGILE DRAFTING INTERPOLATOR
# =============================================================================

class ProceduralAgileDraftingInterpolator:
    """
    Interpolates moving averages and procedural agile drafting envelopes.
    Uses adaptive exponential moving averages (EMA) combined with running variance
    and cubic Hermite smoothstep drafting to synthesize dynamic operational envelopes.
    """

    def __init__(
        self,
        base_alpha: float = 0.20,
        volatility_weight: float = 0.85,
        vital_max_hp: int = VITAL_MAX_HP
    ) -> None:
        if vital_max_hp != VITAL_MAX_HP:
            raise ContractViolation(f"Constitutional HP breach: {vital_max_hp} != {VITAL_MAX_HP}")
        self.vital_max_hp: int = vital_max_hp
        self.base_alpha: float = base_alpha
        self.volatility_weight: float = volatility_weight

        # Telemetry History: metric -> (ema, variance, sample_count)
        self._telemetry_emas: Dict[str, float] = {}
        self._telemetry_vars: Dict[str, float] = {}
        self._sample_counts: Dict[str, int] = {}

        # Dwell hysteretic state tracker
        self._consecutive_dwell_counts: Dict[str, int] = {}
        self._active_corridors: Dict[str, Tuple[float, float]] = {}

    def update_telemetry_metric(self, metric_name: str, sample_value: float) -> None:
        """Updates streaming exponential moving average and variance using adaptive filtering."""
        if metric_name not in self._telemetry_emas:
            self._telemetry_emas[metric_name] = sample_value
            self._telemetry_vars[metric_name] = 0.0
            self._sample_counts[metric_name] = 1
            return

        old_ema = self._telemetry_emas[metric_name]
        old_var = self._telemetry_vars[metric_name]
        count = self._sample_counts[metric_name] + 1
        self._sample_counts[metric_name] = count

        # Adaptive alpha: volatile inputs increase responsiveness
        diff = abs(sample_value - old_ema)
        relative_diff = diff / max(0.001, abs(old_ema))
        adaptive_alpha = min(0.65, max(0.05, self.base_alpha * (1.0 + relative_diff)))

        # Update EMA: \bar{x}_k = \alpha x_k + (1 - \alpha) \bar{x}_{k-1}
        new_ema = adaptive_alpha * sample_value + (1.0 - adaptive_alpha) * old_ema
        self._telemetry_emas[metric_name] = new_ema

        # Update running variance: \sigma_k^2 = (1 - \beta)\sigma_{k-1}^2 + \beta(x_k - \bar{x}_k)^2
        new_var = (1.0 - adaptive_alpha) * old_var + adaptive_alpha * ((sample_value - new_ema) ** 2)
        self._telemetry_vars[metric_name] = new_var

    def get_ema(self, metric_name: str, fallback: float = 0.0) -> float:
        """Returns current exponential moving average."""
        return self._telemetry_emas.get(metric_name, fallback)

    def get_std_dev(self, metric_name: str, fallback: float = 0.0) -> float:
        """Returns current standard deviation."""
        var = self._telemetry_vars.get(metric_name, fallback ** 2)
        return math.sqrt(max(0.0, var))

    def cubic_smoothstep(self, t: float) -> float:
        """Cubic Hermite interpolation s(t) = 3t^2 - 2t^3 clamped to [0.0, 1.0]."""
        clamped_t = max(0.0, min(1.0, t))
        return clamped_t * clamped_t * (3.0 - 2.0 * clamped_t)

    def draft_quantum_slice_us(
        self,
        subsystem: SubsystemNodeId,
        priority: int,
        current_uma: float,
        base_quantum_us: float = 500.0,
        min_quantum_us: float = 50.0,
        max_quantum_us: float = 3500.0
    ) -> float:
        """
        Procedurally drafts an allocated time slice based on historical latency EMA,
        subsystem priority, and current UMA bus pressure.
        """
        metric_key = f"{subsystem.value}_latency_us"
        latency_ema = self.get_ema(metric_key, fallback=base_quantum_us)

        # UMA scaling factor: as UMA approaches UMA_SATURATION_THRESHOLD (0.78), quantum shrinks
        uma_headroom = max(0.0, (UMA_SATURATION_THRESHOLD - current_uma) / UMA_SATURATION_THRESHOLD)
        uma_factor = self.cubic_smoothstep(uma_headroom)

        # Priority multiplier: 1 -> 1.0, 5 -> 1.8
        p_multiplier = 1.0 + 0.20 * (max(1, min(5, priority)) - 1)

        # Blend base quantum with latency EMA
        blended_target = (0.35 * base_quantum_us + 0.65 * latency_ema) * p_multiplier * uma_factor
        return round(max(min_quantum_us, min(max_quantum_us, blended_target)), 2)

    def draft_thermal_headroom_c(self, current_temp_c: float) -> float:
        """Drafts safe thermal margin accounting for running temperature volatility."""
        std_temp = self.get_std_dev("system_temperature_c", fallback=1.5)
        safe_margin = 3.0 * std_temp
        headroom = max(0.0, ARRHENIUS_MAX_TEMP_C - current_temp_c - safe_margin)
        return round(headroom, 2)

    def draft_execution_envelope(
        self,
        candidate: CandidateOperation,
        current_uma: float,
        current_temp_c: float
    ) -> AgileDraftedEnvelope:
        """
        Synthesizes a complete procedural agile drafting envelope for a candidate operation.
        """
        quantum_us = self.draft_quantum_slice_us(
            subsystem=candidate.subsystem,
            priority=candidate.priority,
            current_uma=current_uma,
            base_quantum_us=candidate.estimated_latency_us * 1.5
        )

        energy_metric = f"{candidate.subsystem.value}_energy_uj"
        energy_ema = self.get_ema(energy_metric, fallback=candidate.estimated_energy_uj)
        budget_energy = max(candidate.estimated_energy_uj, energy_ema * 1.25)

        thermal_headroom = self.draft_thermal_headroom_c(current_temp_c)

        # Confidence decays if variance is high relative to the mean
        lat_std = self.get_std_dev(f"{candidate.subsystem.value}_latency_us", fallback=10.0)
        confidence = max(0.20, min(0.99, 1.0 - (lat_std / max(1.0, quantum_us))))

        return AgileDraftedEnvelope(
            allocated_quantum_us=quantum_us,
            max_energy_budget_uj=round(budget_energy, 2),
            thermal_headroom_c=thermal_headroom,
            uma_bus_ceiling=UMA_SATURATION_THRESHOLD,
            hysteresis_dwell_required=HYSTERESIS_DWELL_FRAMES,
            draft_confidence=round(confidence, 3),
            vital_max_hp=self.vital_max_hp
        )


# =============================================================================
# COMPONENT 2: SAFE OPERATION SELECTION PACK
# =============================================================================

class SafeOperationSelectionPack:
    """
    Evaluates and filters candidate operations against strict safety gates:
    Constitutional HP, Authority Separation, Capability Token, Thermal & Bus limits,
    Instructional Dependencies, and Resource Domain Non-Interference.
    """

    def __init__(
        self,
        interpolator: ProceduralAgileDraftingInterpolator,
        secret_key: bytes = b"krystal_safe_operation_pack_secret_128b",
        vital_max_hp: int = VITAL_MAX_HP
    ) -> None:
        if vital_max_hp != VITAL_MAX_HP:
            raise ContractViolation(f"Constitutional HP breach: {vital_max_hp} != {VITAL_MAX_HP}")
        self.vital_max_hp: int = vital_max_hp
        self.interpolator = interpolator
        self.secret_key = secret_key

    def evaluate_candidate(
        self,
        candidate: CandidateOperation,
        current_uma: float,
        current_temp_c: float,
        resolved_dependencies: Set[str],
        active_locked_domains: Set[str]
    ) -> SelectionEvaluationResult:
        """
        Executes multi-gate safety vetting of a candidate operation.
        Emits an authorized evaluation result with an agile drafted envelope or rejects with Delta S == 0.
        """
        t0 = time.perf_counter_ns()

        # Gate 1: Constitutional Invariant Check (VITAL_MAX_HP == 6)
        if candidate.vital_max_hp != VITAL_MAX_HP:
            return self._build_result(
                candidate.operation_id,
                OperationSafetyVerdict.REJECTED_CONSTITUTIONAL_BREACH,
                False,
                None,
                f"Candidate violated constitutional invariant: {candidate.vital_max_hp} != {VITAL_MAX_HP}",
                t0
            )

        # Gate 2: Authority Separation (Proposer != Evaluator)
        if candidate.proposer_id == candidate.evaluator_id:
            return self._build_result(
                candidate.operation_id,
                OperationSafetyVerdict.REJECTED_AUTHORITY_SEPARATION,
                False,
                None,
                f"Authority violation: Proposer '{candidate.proposer_id}' cannot evaluate own operation",
                t0
            )

        # Gate 3: Capability Scope Authorization
        if candidate.required_capability == CapabilityScope.NONE:
            return self._build_result(
                candidate.operation_id,
                OperationSafetyVerdict.REJECTED_CAPABILITY_DENIED,
                False,
                None,
                "Operation possesses empty CapabilityScope. Permission denied.",
                t0
            )

        # Gate 4: Thermal Arrhenius Headroom Gate
        if current_temp_c >= ARRHENIUS_MAX_TEMP_C:
            return self._build_result(
                candidate.operation_id,
                OperationSafetyVerdict.REJECTED_THERMAL_HEADROOM,
                False,
                None,
                f"Current temperature {current_temp_c}°C exceeds Arrhenius threshold {ARRHENIUS_MAX_TEMP_C}°C",
                t0
            )

        # Gate 5: UMA Bus Saturation Gate
        if current_uma >= UMA_SATURATION_THRESHOLD:
            return self._build_result(
                candidate.operation_id,
                OperationSafetyVerdict.REJECTED_BUS_SATURATED,
                False,
                None,
                f"UMA bus pressure {current_uma:.3f} >= threshold {UMA_SATURATION_THRESHOLD}. Compute rejected.",
                t0
            )

        # Gate 6: Instructional Dependencies Check
        unresolved = [dep for dep in candidate.depends_on if dep not in resolved_dependencies]
        if unresolved:
            return self._build_result(
                candidate.operation_id,
                OperationSafetyVerdict.REJECTED_DEPENDENCY_UNRESOLVED,
                False,
                None,
                f"Unresolved dependencies: {', '.join(unresolved)}",
                t0
            )

        # Gate 7: Resource Domain Conflict Isolation
        conflicts = candidate.resource_domains.intersection(active_locked_domains)
        if conflicts:
            return self._build_result(
                candidate.operation_id,
                OperationSafetyVerdict.REJECTED_DOMAIN_CONFLICT,
                False,
                None,
                f"Resource domain conflict on active locks: {', '.join(conflicts)}",
                t0
            )

        # All Gates Passed: Synthesize Agile Drafted Envelope
        envelope = self.interpolator.draft_execution_envelope(
            candidate=candidate,
            current_uma=current_uma,
            current_temp_c=current_temp_c
        )

        return self._build_result(
            candidate.operation_id,
            OperationSafetyVerdict.ACCEPTED_AND_DRAFTED,
            True,
            envelope,
            "All safety gates passed. Operation authorized.",
            t0
        )

    def _build_result(
        self,
        op_id: str,
        verdict: OperationSafetyVerdict,
        authorized: bool,
        envelope: Optional[AgileDraftedEnvelope],
        reason: str,
        start_ns: int
    ) -> SelectionEvaluationResult:
        eval_us = (time.perf_counter_ns() - start_ns) / 1000.0
        sig_data = f"{op_id}:{verdict.value}:{authorized}:{eval_us}:{self.vital_max_hp}".encode()
        digest = hmac.new(self.secret_key, sig_data, hashlib.sha256).hexdigest()

        return SelectionEvaluationResult(
            operation_id=op_id,
            verdict=verdict,
            is_authorized=authorized,
            drafted_envelope=envelope,
            rejection_reason=reason,
            evaluation_latency_us=round(eval_us, 2),
            hmac_digest=digest,
            vital_max_hp=self.vital_max_hp
        )


# =============================================================================
# COMPONENT 3: INSTRUCTIONAL DEPENDENCY COORDINATOR & TURN LOGIC BEHAVIOR
# =============================================================================

class InstructionalDependencyCoordinator:
    """
    Coordinates turn logic behavior and instructional dependencies across subsystems.
    Enforces that:
    1. Only ONE subsystem holds active bus-mastering write authority at any moment.
    2. Subsystems execute in cooperative time-slices governed by the Agile Drafted Envelope.
    3. Instructional dependency receipts must be cryptographically posted before a dependent turn starts.
    4. Slices exceeding quantum or breaching invariants forcefully rollback with Delta S == 0.
    """

    def __init__(
        self,
        interpolator: Optional[ProceduralAgileDraftingInterpolator] = None,
        selection_pack: Optional[SafeOperationSelectionPack] = None,
        secret_key: bytes = b"krystal_turn_coordinator_hmac_secret_2026",
        vital_max_hp: int = VITAL_MAX_HP
    ) -> None:
        if vital_max_hp != VITAL_MAX_HP:
            raise ContractViolation(f"Constitutional HP breach: {vital_max_hp} != {VITAL_MAX_HP}")
        self.vital_max_hp: int = vital_max_hp
        self.secret_key = secret_key

        self.interpolator = interpolator or ProceduralAgileDraftingInterpolator(vital_max_hp=self.vital_max_hp)
        self.selection_pack = selection_pack or SafeOperationSelectionPack(
            interpolator=self.interpolator,
            vital_max_hp=self.vital_max_hp
        )

        # Subsystem tracking: SubsystemNodeId -> TurnState
        self.subsystem_states: Dict[SubsystemNodeId, TurnState] = {
            node: TurnState.IDLE for node in SubsystemNodeId
        }

        # Active bus-mastering token (Only one subsystem may hold this)
        self.active_turn_token: Optional[TurnToken] = None
        self.active_locked_domains: Set[str] = set()

        # Instructional Dependency Resolution Graph
        self.completed_receipts: Dict[str, InstructionReceipt] = {}
        self.resolved_instruction_ids: Set[str] = set()

        # Turn sequence metrics
        self.total_turns_granted: int = 0
        self.total_turns_yielded: int = 0
        self.total_turns_rolled_back: int = 0

    def get_subsystem_state(self, subsystem: SubsystemNodeId) -> TurnState:
        """Returns the current turn state of a given subsystem."""
        return self.subsystem_states.get(subsystem, TurnState.IDLE)

    def is_bus_master_active(self) -> bool:
        """True if any subsystem currently holds the active bus-mastering write turn."""
        return self.active_turn_token is not None

    def post_instruction_receipt(self, receipt: InstructionReceipt) -> None:
        """Registers a completed instruction receipt into the dependency tracker."""
        if receipt.vital_max_hp != VITAL_MAX_HP:
            raise ContractViolation(f"Constitutional HP breach in InstructionReceipt: {receipt.vital_max_hp} != {VITAL_MAX_HP}")

        self.completed_receipts[receipt.instruction_id] = receipt
        if receipt.delta_s == 1:
            self.resolved_instruction_ids.add(receipt.instruction_id)

    def request_and_grant_turn(
        self,
        candidate: CandidateOperation,
        current_uma: float = 0.40,
        current_temp_c: float = 45.0
    ) -> Tuple[bool, Optional[TurnToken], SelectionEvaluationResult]:
        """
        Coordinates the turn request lifecycle:
        1. Checks bus-master mutual exclusion.
        2. Evaluates candidate operation via SafeOperationSelectionPack.
        3. If approved, mints a cryptographic TurnToken and updates state to TURN_GRANTED.
        """
        # Invariant: If a turn is already executing, reject concurrent grant
        if self.is_bus_master_active():
            eval_res = self.selection_pack._build_result(
                candidate.operation_id,
                OperationSafetyVerdict.REJECTED_BUS_SATURATED,
                False,
                None,
                f"Bus-mastering turn already held by subsystem '{self.active_turn_token.subsystem.value}'",  # type: ignore
                time.perf_counter_ns()
            )
            self.subsystem_states[candidate.subsystem] = TurnState.REQUESTING_TURN
            return False, None, eval_res

        # Evaluate safety gates
        eval_result = self.selection_pack.evaluate_candidate(
            candidate=candidate,
            current_uma=current_uma,
            current_temp_c=current_temp_c,
            resolved_dependencies=self.resolved_instruction_ids,
            active_locked_domains=self.active_locked_domains
        )

        if not eval_result.is_authorized or eval_result.drafted_envelope is None:
            self.subsystem_states[candidate.subsystem] = (
                TurnState.AWAITING_DEPENDENCY
                if eval_result.verdict == OperationSafetyVerdict.REJECTED_DEPENDENCY_UNRESOLVED
                else TurnState.IDLE
            )
            return False, None, eval_result

        # Mint cryptographic TurnToken
        now_ns = time.perf_counter_ns()
        quantum_ns = int(eval_result.drafted_envelope.allocated_quantum_us * 1000)
        expires_ns = now_ns + quantum_ns
        token_id = f"TOKEN-{candidate.subsystem.value[:4]}-{now_ns & 0xFFFFFF}"

        sig_data = f"{token_id}:{candidate.subsystem.value}:{candidate.operation_id}:{now_ns}:{expires_ns}:{self.vital_max_hp}".encode()
        token_sig = hmac.new(self.secret_key, sig_data, hashlib.sha256).hexdigest()

        token = TurnToken(
            token_id=token_id,
            subsystem=candidate.subsystem,
            operation_id=candidate.operation_id,
            allocated_quantum_us=eval_result.drafted_envelope.allocated_quantum_us,
            granted_at_ns=now_ns,
            expires_at_ns=expires_ns,
            hmac_signature=token_sig,
            vital_max_hp=self.vital_max_hp
        )

        # Grant turn
        self.active_turn_token = token
        self.active_locked_domains.update(candidate.resource_domains)
        self.subsystem_states[candidate.subsystem] = TurnState.TURN_GRANTED
        self.total_turns_granted += 1

        return True, token, eval_result

    def begin_execution_slice(self, token: TurnToken) -> bool:
        """Transitions the subsystem into EXECUTING_SLICE."""
        if self.active_turn_token != token:
            return False
        self.subsystem_states[token.subsystem] = TurnState.EXECUTING_SLICE
        return True

    def yield_turn_commit(
        self,
        token: TurnToken,
        latency_us: float,
        energy_uj: float,
        output_payload_hash: str
    ) -> InstructionReceipt:
        """
        Gracefully yields the turn upon successful task completion.
        Updates telemetry moving averages, releases locks, and emits an InstructionReceipt with Delta S == 1.
        """
        if self.active_turn_token != token:
            raise ContractViolation("Cannot yield a turn without holding the active TurnToken!")

        # Update telemetry in the agile drafting interpolator
        self.interpolator.update_telemetry_metric(f"{token.subsystem.value}_latency_us", latency_us)
        self.interpolator.update_telemetry_metric(f"{token.subsystem.value}_energy_uj", energy_uj)

        now_ns = time.perf_counter_ns()
        sig_data = f"{token.operation_id}:{token.subsystem.value}:COMMITTED:{latency_us}:{energy_uj}:{output_payload_hash}:{now_ns}:{self.vital_max_hp}".encode()
        receipt_sig = hmac.new(self.secret_key, sig_data, hashlib.sha256).hexdigest()

        receipt = InstructionReceipt(
            instruction_id=token.operation_id,
            subsystem=token.subsystem,
            status="COMMITTED",
            latency_us=latency_us,
            energy_uj=energy_uj,
            delta_s=1,
            timestamp_ns=now_ns,
            hmac_signature=receipt_sig,
            vital_max_hp=self.vital_max_hp
        )

        # Record receipt
        self.post_instruction_receipt(receipt)

        # Release token and locks
        self.active_turn_token = None
        self.active_locked_domains.clear()
        self.subsystem_states[token.subsystem] = TurnState.TURN_YIELDED
        self.total_turns_yielded += 1

        return receipt

    def abort_turn_rollback(self, token: TurnToken, reason: str) -> InstructionReceipt:
        """
        Forcefully aborts and rolls back an active turn (Quantum expiration, error, invariant breach).
        Ensures constitutional invariant Delta S == 0.
        """
        if self.active_turn_token != token:
            raise ContractViolation("Cannot abort a turn without matching the active TurnToken!")

        now_ns = time.perf_counter_ns()
        sig_data = f"{token.operation_id}:{token.subsystem.value}:ROLLED_BACK:{reason}:{now_ns}:{self.vital_max_hp}".encode()
        receipt_sig = hmac.new(self.secret_key, sig_data, hashlib.sha256).hexdigest()

        receipt = InstructionReceipt(
            instruction_id=token.operation_id,
            subsystem=token.subsystem,
            status=f"ROLLED_BACK_{reason}",
            latency_us=token.allocated_quantum_us,
            energy_uj=0.0,
            delta_s=0,  # Constitutional Invariant: Delta S == 0 on rollback
            timestamp_ns=now_ns,
            hmac_signature=receipt_sig,
            vital_max_hp=self.vital_max_hp
        )

        self.post_instruction_receipt(receipt)

        # Release token and locks
        self.active_turn_token = None
        self.active_locked_domains.clear()
        self.subsystem_states[token.subsystem] = TurnState.TURN_EXPIRED_ROLLBACK
        self.total_turns_rolled_back += 1

        return receipt

    def check_and_enforce_quantum_expiry(self, current_time_ns: Optional[int] = None) -> bool:
        """
        Periodically audited watchdog check. If the active turn exceeded its allocated quantum,
        it is forcefully aborted with Delta S == 0.
        """
        if self.active_turn_token is None:
            return False

        now_ns = current_time_ns if current_time_ns is not None else time.perf_counter_ns()
        if now_ns > self.active_turn_token.expires_at_ns:
            self.abort_turn_rollback(self.active_turn_token, reason="QUANTUM_EXPIRED")
            return True
        return False

    def get_coordinator_telemetry(self) -> Dict[str, Any]:
        """Provides high-level coordination and turn logic metrics."""
        return {
            "vital_max_hp": self.vital_max_hp,
            "constitutional_invariant_intact": (self.vital_max_hp == VITAL_MAX_HP),
            "is_bus_master_active": self.is_bus_master_active(),
            "active_bus_master_subsystem": self.active_turn_token.subsystem.value if self.active_turn_token else None,
            "total_turns_granted": self.total_turns_granted,
            "total_turns_yielded": self.total_turns_yielded,
            "total_turns_rolled_back": self.total_turns_rolled_back,
            "resolved_dependencies_count": len(self.resolved_instruction_ids),
            "active_locked_domains": list(self.active_locked_domains),
            "subsystem_states": {node.value: state.value for node, state in self.subsystem_states.items()},
        }


# Global singleton instance
GLOBAL_SUBSYSTEM_TURN_COORDINATOR = InstructionalDependencyCoordinator()
