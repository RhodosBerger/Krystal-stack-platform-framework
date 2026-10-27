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
    REJECTED_PRECOMMIT_VALIDATION_REQUIRED = "REJECTED_PRECOMMIT_VALIDATION_REQUIRED"


class OperationReversibilityClass(str, enum.Enum):
    """
    Physical reversibility classification of operations.
    Enforces that Delta S == 0 is physically honest, not just an in-memory discard:
    - REVERSIBLE: Pure software state / memory buffer mutation that rolls back fully.
    - COMPENSATABLE: Side effect with a formal inverse compensating action.
    - IRREVERSIBLE: Physical side effect (GPIO write, network packet, CNC step, raw flash write).
      Mandates PreCommitValidationRequired before execution token issuance!
    """
    REVERSIBLE = "REVERSIBLE"
    COMPENSATABLE = "COMPENSATABLE"
    IRREVERSIBLE = "IRREVERSIBLE"


# Physical Constants for Reliability
BOLTZMANN_CONSTANT_EV_PER_K: float = 8.617333262145e-5  # eV / K
KELVIN_OFFSET: float = 273.15


# =============================================================================
# HARDWARE ARCHITECTURE PROFILE
# =============================================================================

@dataclass(frozen=True)
class ArchitectureProfile:
    """
    Decouples hardware-specific thermal, bus pressure, and timing parameters
    from constitutional invariants. Invariants mandate that T < configured_safe
    and UMA < configured_limit, allowing portable execution across heterogeneous silicon.
    """
    profile_name: str = "GENERIC_CHRONO_SOC"
    configured_thermal_limit_c: float = ARRHENIUS_MAX_TEMP_C  # default 75.0 C
    configured_uma_limit: float = UMA_SATURATION_THRESHOLD      # default 0.78
    arrhenius_activation_energy_ev: float = 0.70                 # Silicon semiconductor Ea ~ 0.7 eV
    base_quantum_us: float = 500.0
    min_quantum_us: float = 50.0
    max_quantum_us: float = 3500.0
    vital_max_hp: int = VITAL_MAX_HP

    def __post_init__(self) -> None:
        if self.vital_max_hp != VITAL_MAX_HP:
            raise ContractViolation(f"Constitutional HP breach in ArchitectureProfile: {self.vital_max_hp} != {VITAL_MAX_HP}")


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
    reversibility: OperationReversibilityClass = OperationReversibilityClass.REVERSIBLE
    pre_commit_validated: bool = False
    compensating_action: Optional[str] = None
    vital_max_hp: int = VITAL_MAX_HP

    def __post_init__(self) -> None:
        if self.vital_max_hp != VITAL_MAX_HP:
            raise ContractViolation(f"Constitutional HP breach in CandidateOperation: {self.vital_max_hp} != {VITAL_MAX_HP}")
        if self.reversibility == OperationReversibilityClass.IRREVERSIBLE:
            self.is_reversible = False


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
    resource_domains: Tuple[str, ...] = ()
    vital_max_hp: int = VITAL_MAX_HP

    def __post_init__(self) -> None:
        if self.vital_max_hp != VITAL_MAX_HP:
            raise ContractViolation(f"Constitutional HP breach in TurnToken: {self.vital_max_hp} != {VITAL_MAX_HP}")


# =============================================================================
# ARRHENIUS RELIABILITY & STATISTICAL THERMAL MODEL
# =============================================================================

class ArrheniusReliabilityModel:
    """
    Physical Arrhenius Reliability and Acceleration Factor Model.
    AF = exp[ (Ea / kB) * (1 / T_use - 1 / T_stress) ]
    where temperatures are strictly evaluated in Kelvins.
    Ea is the activation energy in eV (typically 0.70 eV for silicon semiconductor degradation).
    kB is Boltzmann's constant in eV/K (8.617333262e-5 eV/K).
    """

    @staticmethod
    def to_kelvin(celsius: float) -> float:
        """Converts Celsius to absolute temperature in Kelvin."""
        return celsius + KELVIN_OFFSET

    @classmethod
    def calculate_acceleration_factor(
        cls,
        t_use_c: float,
        t_stress_c: float,
        activation_energy_ev: float = 0.70
    ) -> float:
        """
        Calculates the thermal acceleration factor AF according to the Arrhenius relation.
        AF > 1.0 indicates accelerated wearout/aging at elevated stress temperatures.
        """
        t_use_k = cls.to_kelvin(t_use_c)
        t_stress_k = cls.to_kelvin(t_stress_c)

        if t_use_k <= 0.0 or t_stress_k <= 0.0:
            raise ContractViolation("Absolute temperature in Kelvin must be strictly positive!")

        exponent = (activation_energy_ev / BOLTZMANN_CONSTANT_EV_PER_K) * (
            (1.0 / t_use_k) - (1.0 / t_stress_k)
        )
        clipped_exp = max(-50.0, min(50.0, exponent))
        return round(math.exp(clipped_exp), 6)

    @classmethod
    def calculate_relative_failure_rate(
        cls,
        operating_temp_c: float,
        reference_temp_c: float = 40.0,
        activation_energy_ev: float = 0.70
    ) -> float:
        """
        Calculates relative failure rate multiplier lambda(T) / lambda(T_ref).
        """
        return cls.calculate_acceleration_factor(
            t_use_c=reference_temp_c,
            t_stress_c=operating_temp_c,
            activation_energy_ev=activation_energy_ev
        )


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
        profile: Optional[ArchitectureProfile] = None,
        vital_max_hp: int = VITAL_MAX_HP
    ) -> None:
        if vital_max_hp != VITAL_MAX_HP:
            raise ContractViolation(f"Constitutional HP breach: {vital_max_hp} != {VITAL_MAX_HP}")
        self.vital_max_hp: int = vital_max_hp
        self.base_alpha: float = base_alpha
        self.volatility_weight: float = volatility_weight
        self.profile: ArchitectureProfile = profile or ArchitectureProfile(vital_max_hp=vital_max_hp)

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
        base_quantum_us: Optional[float] = None,
        min_quantum_us: Optional[float] = None,
        max_quantum_us: Optional[float] = None
    ) -> float:
        """
        Procedurally drafts an allocated time slice based on historical latency EMA,
        subsystem priority, and current UMA bus pressure against ArchitectureProfile limits.
        """
        base_q = base_quantum_us if base_quantum_us is not None else self.profile.base_quantum_us
        min_q = min_quantum_us if min_quantum_us is not None else self.profile.min_quantum_us
        max_q = max_quantum_us if max_quantum_us is not None else self.profile.max_quantum_us
        uma_limit = self.profile.configured_uma_limit

        metric_key = f"{subsystem.value}_latency_us"
        latency_ema = self.get_ema(metric_key, fallback=base_q)

        # UMA scaling factor: as UMA approaches configured ceiling, quantum shrinks smoothly
        uma_headroom = max(0.0, (uma_limit - current_uma) / max(0.01, uma_limit))
        uma_factor = self.cubic_smoothstep(uma_headroom)

        # Priority multiplier: 1 -> 1.0, 5 -> 1.8
        p_multiplier = 1.0 + 0.20 * (max(1, min(5, priority)) - 1)

        # Blend base quantum with latency EMA
        blended_target = (0.35 * base_q + 0.65 * latency_ema) * p_multiplier * uma_factor
        return round(max(min_q, min(max_q, blended_target)), 2)

    def draft_statistical_thermal_margin_c(self, current_temp_c: float) -> Tuple[float, float]:
        """
        Drafts the 3-sigma statistical thermal safety margin and remaining headroom.
        Margin = 3 * std_dev(system_temperature_c)
        Headroom = max(0.0, configured_thermal_limit_c - current_temp_c - Margin)
        Returns (margin_c, headroom_c).
        """
        std_temp = self.get_std_dev("system_temperature_c", fallback=1.5)
        safe_margin = 3.0 * std_temp
        headroom = max(0.0, self.profile.configured_thermal_limit_c - current_temp_c - safe_margin)
        return round(safe_margin, 2), round(headroom, 2)

    def draft_thermal_headroom_c(self, current_temp_c: float) -> float:
        """Drafts safe thermal headroom accounting for 3-sigma thermal volatility envelope."""
        _, headroom = self.draft_statistical_thermal_margin_c(current_temp_c)
        return headroom

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
            uma_bus_ceiling=self.profile.configured_uma_limit,
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
    1. Constitutional Invariant (VITAL_MAX_HP == 6)
    2. Authority Separation (Proposer != Evaluator)
    3. Capability Scope Authorization
    4. Pre-Commit Physical Reversibility Gate (Irreversible ops require pre-commit validation)
    5. Configured Thermal Headroom Gate
    6. Configured UMA Bus Saturation Gate
    7. Instructional Dependency Graph Resolution
    8. Resource Domain Conflict Isolation
    """

    def __init__(
        self,
        interpolator: ProceduralAgileDraftingInterpolator,
        profile: Optional[ArchitectureProfile] = None,
        secret_key: bytes = b"krystal_safe_operation_pack_secret_128b",
        vital_max_hp: int = VITAL_MAX_HP
    ) -> None:
        if vital_max_hp != VITAL_MAX_HP:
            raise ContractViolation(f"Constitutional HP breach: {vital_max_hp} != {VITAL_MAX_HP}")
        self.vital_max_hp: int = vital_max_hp
        self.interpolator = interpolator
        self.profile: ArchitectureProfile = profile or interpolator.profile
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

        # Gate 4: Physical Reversibility & Pre-Commit Validation Gate
        if candidate.reversibility == OperationReversibilityClass.IRREVERSIBLE and not candidate.pre_commit_validated:
            return self._build_result(
                candidate.operation_id,
                OperationSafetyVerdict.REJECTED_PRECOMMIT_VALIDATION_REQUIRED,
                False,
                None,
                "Irreversible physical operation requires formal pre-commit validation before turn issuance. Delta S cannot be rolled back after physical actuation.",
                t0
            )

        # Gate 5: Thermal Headroom Gate against Configured ArchitectureProfile
        if current_temp_c >= self.profile.configured_thermal_limit_c:
            return self._build_result(
                candidate.operation_id,
                OperationSafetyVerdict.REJECTED_THERMAL_HEADROOM,
                False,
                None,
                f"Current temperature {current_temp_c}°C exceeds configured threshold {self.profile.configured_thermal_limit_c}°C ({self.profile.profile_name})",
                t0
            )

        # Gate 6: UMA Bus Saturation Gate against Configured ArchitectureProfile
        if current_uma >= self.profile.configured_uma_limit:
            return self._build_result(
                candidate.operation_id,
                OperationSafetyVerdict.REJECTED_BUS_SATURATED,
                False,
                None,
                f"UMA bus pressure {current_uma:.3f} >= configured threshold {self.profile.configured_uma_limit:.3f}. Compute rejected.",
                t0
            )

        # Gate 7: Instructional Dependencies Check
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

        # Gate 8: Resource Domain Conflict Isolation
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
    Enforces:
    1. Conflict-Aware Parallel Turn Scheduling: Subsystems with disjoint resource domains
       can execute slices concurrently without violating determinism.
    2. Subsystems execute in cooperative time-slices governed by the Agile Drafted Envelope.
    3. Instructional dependency receipts must be cryptographically posted before dependent turns.
    4. Slices exceeding quantum or breaching invariants forcefully rollback with Delta S == 0.
    5. Real-time telemetry tracking: Turn Efficiency, Wait times (P95), Overruns, and Lock conflicts.
    """

    def __init__(
        self,
        interpolator: Optional[ProceduralAgileDraftingInterpolator] = None,
        selection_pack: Optional[SafeOperationSelectionPack] = None,
        profile: Optional[ArchitectureProfile] = None,
        secret_key: bytes = b"krystal_turn_coordinator_hmac_secret_2026",
        vital_max_hp: int = VITAL_MAX_HP
    ) -> None:
        if vital_max_hp != VITAL_MAX_HP:
            raise ContractViolation(f"Constitutional HP breach: {vital_max_hp} != {VITAL_MAX_HP}")
        self.vital_max_hp: int = vital_max_hp
        self.secret_key = secret_key
        self.profile: ArchitectureProfile = profile or (interpolator.profile if interpolator else ArchitectureProfile(vital_max_hp=vital_max_hp))

        self.interpolator = interpolator or ProceduralAgileDraftingInterpolator(
            profile=self.profile,
            vital_max_hp=self.vital_max_hp
        )
        self.selection_pack = selection_pack or SafeOperationSelectionPack(
            interpolator=self.interpolator,
            profile=self.profile,
            vital_max_hp=self.vital_max_hp
        )

        # Subsystem tracking: SubsystemNodeId -> TurnState
        self.subsystem_states: Dict[SubsystemNodeId, TurnState] = {
            node: TurnState.IDLE for node in SubsystemNodeId
        }

        # Conflict-Aware Parallel Turn Scheduling Token Pools
        self.active_turn_tokens: Dict[str, TurnToken] = {}  # token_id -> TurnToken
        self.domain_locks: Dict[str, str] = {}               # domain_name -> token_id
        self.token_to_domains: Dict[str, Set[str]] = {}      # token_id -> Set[domain_name]

        # Instructional Dependency Resolution Graph
        self.completed_receipts: Dict[str, InstructionReceipt] = {}
        self.resolved_instruction_ids: Set[str] = set()

        # Turn sequence metrics
        self.total_turns_granted: int = 0
        self.total_turns_yielded: int = 0
        self.total_turns_rolled_back: int = 0

        # Runtime engineering telemetry
        self.turn_efficiencies: List[float] = []
        self.turn_wait_times_us: List[float] = []
        self.quantum_overrun_count: int = 0
        self.lock_conflict_count: int = 0
        self.dependency_stall_count: int = 0
        self.precommit_reject_count: int = 0
        self.total_evaluations_count: int = 0

    @property
    def active_turn_token(self) -> Optional[TurnToken]:
        """Backwards-compatible accessor returning primary active bus-mastering token."""
        if not self.active_turn_tokens:
            return None
        return next(iter(self.active_turn_tokens.values()))

    @active_turn_token.setter
    def active_turn_token(self, token: Optional[TurnToken]) -> None:
        """Setter to support legacy assignment if needed."""
        if token is None:
            self.active_turn_tokens.clear()
            self.domain_locks.clear()
            self.token_to_domains.clear()
        else:
            self.active_turn_tokens[token.token_id] = token
            domains = set(token.resource_domains) if token.resource_domains else {"GLOBAL_BUS_MASTER"}
            self.token_to_domains[token.token_id] = domains
            for d in domains:
                self.domain_locks[d] = token.token_id

    @property
    def active_locked_domains(self) -> Set[str]:
        """Returns the set of currently locked resource domains."""
        return set(self.domain_locks.keys())

    @property
    def average_turn_efficiency(self) -> float:
        """TE = UsefulExecutionTime / GrantedQuantum across all completed turns."""
        if not self.turn_efficiencies:
            return 1.0
        return round(sum(self.turn_efficiencies) / len(self.turn_efficiencies), 4)

    @property
    def quantum_overrun_rate(self) -> float:
        """Ratio of turns exceeding their granted quantum slice."""
        return round(self.quantum_overrun_count / max(1, self.total_turns_granted), 4)

    @property
    def rollback_rate(self) -> float:
        """Ratio of rolled back turns over all initiated turns."""
        total = self.total_turns_granted + self.total_turns_rolled_back
        return round(self.total_turns_rolled_back / max(1, total), 4)

    @property
    def lock_conflict_rate(self) -> float:
        """Ratio of evaluation requests rejected due to domain lock contention."""
        return round(self.lock_conflict_count / max(1, self.total_evaluations_count), 4)

    @property
    def dependency_stall_rate(self) -> float:
        """Ratio of evaluation requests stalled by unresolved dependencies."""
        return round(self.dependency_stall_count / max(1, self.total_evaluations_count), 4)

    @property
    def turn_wait_p95_us(self) -> float:
        """95th percentile waiting time from request to turn grant in microseconds."""
        if not self.turn_wait_times_us:
            return 0.0
        sorted_waits = sorted(self.turn_wait_times_us)
        idx = int(0.95 * (len(sorted_waits) - 1))
        return round(sorted_waits[idx], 2)

    def get_subsystem_state(self, subsystem: SubsystemNodeId) -> TurnState:
        """Returns the current turn state of a given subsystem."""
        return self.subsystem_states.get(subsystem, TurnState.IDLE)

    def is_bus_master_active(self) -> bool:
        """True if any subsystem currently holds an active turn token."""
        return len(self.active_turn_tokens) > 0

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
        current_temp_c: float = 45.0,
        request_time_ns: Optional[int] = None
    ) -> Tuple[bool, Optional[TurnToken], SelectionEvaluationResult]:
        """
        Coordinates the turn request lifecycle:
        1. Checks conflict-aware resource domain isolation.
        2. Evaluates candidate operation via SafeOperationSelectionPack (8 gates).
        3. If approved, mints a cryptographic TurnToken and locks the requested domains.
        """
        now_ns = request_time_ns if request_time_ns is not None else time.perf_counter_ns()
        self.total_evaluations_count += 1

        # Determine requested domains. Default to GLOBAL_BUS_MASTER if unspecified
        requested_domains = set(candidate.resource_domains) if candidate.resource_domains else {"GLOBAL_BUS_MASTER"}

        # Conflict-Aware Parallel Turn Check:
        conflicting_domains = requested_domains.intersection(self.domain_locks.keys())
        if conflicting_domains:
            self.lock_conflict_count += 1
            conflict_holders = [self.domain_locks[d] for d in conflicting_domains if d in self.domain_locks]
            verdict = (
                OperationSafetyVerdict.REJECTED_BUS_SATURATED
                if "GLOBAL_BUS_MASTER" in conflicting_domains
                else OperationSafetyVerdict.REJECTED_DOMAIN_CONFLICT
            )
            reason = (
                f"Bus-mastering turn already held on global bus by: {', '.join(set(conflict_holders))}"
                if verdict == OperationSafetyVerdict.REJECTED_BUS_SATURATED
                else f"Resource domain conflict on active locks: {', '.join(sorted(conflicting_domains))} (held by: {', '.join(set(conflict_holders))})"
            )
            eval_res = self.selection_pack._build_result(
                candidate.operation_id,
                verdict,
                False,
                None,
                reason,
                now_ns
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
            if eval_result.verdict == OperationSafetyVerdict.REJECTED_DEPENDENCY_UNRESOLVED:
                self.dependency_stall_count += 1
                self.subsystem_states[candidate.subsystem] = TurnState.AWAITING_DEPENDENCY
            elif eval_result.verdict == OperationSafetyVerdict.REJECTED_PRECOMMIT_VALIDATION_REQUIRED:
                self.precommit_reject_count += 1
                self.subsystem_states[candidate.subsystem] = TurnState.IDLE
            else:
                self.subsystem_states[candidate.subsystem] = TurnState.IDLE
            return False, None, eval_result

        # Mint cryptographic TurnToken
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
            resource_domains=tuple(sorted(requested_domains)),
            vital_max_hp=self.vital_max_hp
        )

        # Register active token and acquire domain locks
        self.active_turn_tokens[token.token_id] = token
        self.token_to_domains[token.token_id] = requested_domains
        for domain in requested_domains:
            self.domain_locks[domain] = token.token_id

        self.subsystem_states[candidate.subsystem] = TurnState.TURN_GRANTED
        self.total_turns_granted += 1
        self.turn_wait_times_us.append(0.0)

        return True, token, eval_result

    def begin_execution_slice(self, token: TurnToken) -> bool:
        """Transitions the subsystem into EXECUTING_SLICE."""
        if token.token_id not in self.active_turn_tokens:
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
        if token.token_id not in self.active_turn_tokens:
            raise ContractViolation("Cannot yield a turn without holding an active TurnToken!")

        # Calculate Turn Efficiency: TE = UsefulExecutionTime / GrantedQuantum
        te = latency_us / max(1.0, token.allocated_quantum_us)
        self.turn_efficiencies.append(te)
        if latency_us > token.allocated_quantum_us:
            self.quantum_overrun_count += 1

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

        # Release locks and token
        domains = self.token_to_domains.pop(token.token_id, set())
        for d in domains:
            self.domain_locks.pop(d, None)
        self.active_turn_tokens.pop(token.token_id, None)

        self.subsystem_states[token.subsystem] = TurnState.TURN_YIELDED
        self.total_turns_yielded += 1

        return receipt

    def abort_turn_rollback(self, token: TurnToken, reason: str) -> InstructionReceipt:
        """
        Forcefully aborts and rolls back an active turn (Quantum expiration, error, invariant breach).
        Ensures constitutional invariant Delta S == 0.
        """
        if token.token_id not in self.active_turn_tokens:
            raise ContractViolation("Cannot abort a turn without matching an active TurnToken!")

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

        # Release locks and token
        domains = self.token_to_domains.pop(token.token_id, set())
        for d in domains:
            self.domain_locks.pop(d, None)
        self.active_turn_tokens.pop(token.token_id, None)

        self.subsystem_states[token.subsystem] = TurnState.TURN_EXPIRED_ROLLBACK
        self.total_turns_rolled_back += 1

        return receipt

    def check_and_enforce_quantum_expiry(self, current_time_ns: Optional[int] = None) -> bool:
        """
        Periodically audited watchdog check across all active turn tokens.
        If any active turn exceeded its allocated quantum, it is forcefully aborted with Delta S == 0.
        """
        if not self.active_turn_tokens:
            return False

        now_ns = current_time_ns if current_time_ns is not None else time.perf_counter_ns()
        expired_any = False
        for token in list(self.active_turn_tokens.values()):
            if now_ns > token.expires_at_ns:
                self.abort_turn_rollback(token, reason="QUANTUM_EXPIRED")
                expired_any = True
        return expired_any

    def get_coordinator_telemetry(self) -> Dict[str, Any]:
        """Provides high-level coordination and turn logic metrics."""
        return {
            "vital_max_hp": self.vital_max_hp,
            "constitutional_invariant_intact": (self.vital_max_hp == VITAL_MAX_HP),
            "is_bus_master_active": self.is_bus_master_active(),
            "active_turns_count": len(self.active_turn_tokens),
            "active_bus_master_subsystem": self.active_turn_token.subsystem.value if self.active_turn_token else None,
            "total_turns_granted": self.total_turns_granted,
            "total_turns_yielded": self.total_turns_yielded,
            "total_turns_rolled_back": self.total_turns_rolled_back,
            "average_turn_efficiency": self.average_turn_efficiency,
            "quantum_overrun_rate": self.quantum_overrun_rate,
            "rollback_rate": self.rollback_rate,
            "lock_conflict_rate": self.lock_conflict_rate,
            "dependency_stall_rate": self.dependency_stall_rate,
            "turn_wait_p95_us": self.turn_wait_p95_us,
            "resolved_dependencies_count": len(self.resolved_instruction_ids),
            "active_locked_domains": list(self.active_locked_domains),
            "subsystem_states": {node.value: state.value for node, state in self.subsystem_states.items()},
        }


# =============================================================================
# BENCHMARK HARNESS: STATIC QUANTUM VS AGILE DRAFTED QUANTUM
# =============================================================================

def run_static_vs_agile_drafting_benchmark(
    dag: Optional[List[CandidateOperation]] = None,
    profile: Optional[ArchitectureProfile] = None,
    repetitions: int = 5
) -> Dict[str, Any]:
    """
    Executes a formal engineering benchmark comparing STATIC QUANTUM vs AGILE DRAFTED QUANTUM
    on an identical DAG of operations.
    Measures:
    - Total execution latency (us)
    - Throughput (ops/ms)
    - P95 turn wait time (us)
    - Quantum overruns count and rate
    - Rollbacks count and rate
    - Average turn efficiency (TE)
    - DraftingGain = (Utility_agile / Utility_static) - 1.0
    Guarantees constitutional invariants: VITAL_MAX_HP == 6, Delta S == 0 on breach.
    """
    prof = profile or ArchitectureProfile()

    # Build canonical DAG if not provided
    if dag is None:
        dag = [
            # Disjoint domain operations
            CandidateOperation(
                operation_id="OP-GPU-01",
                title="GPU Compute Dispatch",
                subsystem=SubsystemNodeId.CHRONO_SOC,
                proposer_id="PROP_GPU",
                evaluator_id="EVAL_AUDIT",
                required_capability=CapabilityScope.CAN_EXECUTE,
                estimated_latency_us=220.0,
                estimated_energy_uj=11.0,
                priority=3,
                resource_domains={"GPU_QUEUE_0"}
            ),
            CandidateOperation(
                operation_id="OP-NET-01",
                title="Network Packet TX",
                subsystem=SubsystemNodeId.WSL_BRIDGE,
                proposer_id="PROP_NET",
                evaluator_id="EVAL_AUDIT",
                required_capability=CapabilityScope.CAN_EXECUTE,
                estimated_latency_us=180.0,
                estimated_energy_uj=8.0,
                priority=2,
                resource_domains={"NETWORK_TX"}
            ),
            CandidateOperation(
                operation_id="OP-TERRAIN-01",
                title="Terrain Chunk Mesh Build",
                subsystem=SubsystemNodeId.TERRAIN_MANIFOLD,
                proposer_id="PROP_TERR",
                evaluator_id="EVAL_AUDIT",
                required_capability=CapabilityScope.CAN_EXECUTE,
                estimated_latency_us=450.0,
                estimated_energy_uj=22.0,
                priority=4,
                resource_domains={"TERRAIN_MESH_BUFFER"}
            ),
            # Dependent operation
            CandidateOperation(
                operation_id="OP-LLM-01",
                title="LLM Cascade Infer Step",
                subsystem=SubsystemNodeId.SLOVAK_LLM_CASCADE,
                proposer_id="PROP_LLM",
                evaluator_id="EVAL_AUDIT",
                required_capability=CapabilityScope.CAN_EXECUTE,
                estimated_latency_us=650.0,
                estimated_energy_uj=35.0,
                priority=5,
                depends_on=["OP-GPU-01"],
                resource_domains={"LLM_KV_CACHE"}
            ),
            # Irreversible operation with pre-commit validation
            CandidateOperation(
                operation_id="OP-PHYS-01",
                title="Physical Actuation Strobe",
                subsystem=SubsystemNodeId.VPP_GRID_BROKER,
                proposer_id="PROP_PHYS",
                evaluator_id="EVAL_AUDIT",
                required_capability=CapabilityScope.CAN_EXECUTE,
                estimated_latency_us=120.0,
                estimated_energy_uj=5.0,
                priority=2,
                reversibility=OperationReversibilityClass.IRREVERSIBLE,
                pre_commit_validated=True,
                resource_domains={"GPIO_STROBE_PIN"}
            ),
        ]

    # -------------------------------------------------------------------------
    # RUN 1: STATIC SCHEDULER
    # - Fixed quantum = 500 us
    # - Monolithic single bus master (all ops contend on GLOBAL_BUS_MASTER)
    # -------------------------------------------------------------------------
    static_coordinator = InstructionalDependencyCoordinator(profile=prof)
    static_start_ns = time.perf_counter_ns()
    static_completed = 0
    static_overruns = 0
    static_rollbacks = 0
    static_te_list: List[float] = []
    static_wait_list: List[float] = []

    for _ in range(repetitions):
        # In static mode, serialize sequentially as a global mutex
        for op in dag:
            wait_t0 = time.perf_counter_ns()
            # Force empty domains to trigger global bus master serialization
            static_op = CandidateOperation(
                operation_id=f"STATIC-{op.operation_id}",
                title=op.title,
                subsystem=op.subsystem,
                proposer_id=op.proposer_id,
                evaluator_id=op.evaluator_id,
                required_capability=op.required_capability,
                estimated_latency_us=op.estimated_latency_us,
                estimated_energy_uj=op.estimated_energy_uj,
                priority=op.priority,
                depends_on=[f"STATIC-{d}" for d in op.depends_on],
                resource_domains=set(),  # Contends on GLOBAL_BUS_MASTER
                reversibility=op.reversibility,
                pre_commit_validated=op.pre_commit_validated
            )
            ok, token, _ = static_coordinator.request_and_grant_turn(static_op)
            wait_us = (time.perf_counter_ns() - wait_t0) / 1000.0
            static_wait_list.append(wait_us)

            if ok and token:
                # With static quantum of 500us, if task is > 500us it overruns
                actual_lat = op.estimated_latency_us
                te = actual_lat / prof.base_quantum_us
                static_te_list.append(te)
                if actual_lat > prof.base_quantum_us:
                    static_overruns += 1

                static_coordinator.yield_turn_commit(
                    token=token,
                    latency_us=actual_lat,
                    energy_uj=op.estimated_energy_uj,
                    output_payload_hash="static_hash_abc"
                )
                static_completed += 1

    static_duration_ms = max(0.001, (time.perf_counter_ns() - static_start_ns) / 1_000_000.0)
    static_throughput = static_completed / static_duration_ms
    static_avg_te = sum(static_te_list) / max(1, len(static_te_list))
    static_overrun_rate = static_overruns / max(1, static_completed)
    static_rollback_rate = static_rollbacks / max(1, static_completed)
    static_utility = (static_throughput * static_avg_te) / (
        (1.0 + static_overrun_rate) * (1.0 + static_rollback_rate)
    )

    # -------------------------------------------------------------------------
    # RUN 2: AGILE DRAFTED SCHEDULER
    # - Procedurally drafted quantum based on latency EMA, priority, UMA headroom
    # - Conflict-aware parallel turn scheduling (disjoint domains execute in parallel)
    # -------------------------------------------------------------------------
    agile_coordinator = InstructionalDependencyCoordinator(profile=prof)
    agile_start_ns = time.perf_counter_ns()
    agile_completed = 0
    agile_overruns = 0
    agile_rollbacks = 0
    agile_te_list: List[float] = []
    agile_wait_list: List[float] = []

    for _ in range(repetitions):
        active_tokens: List[TurnToken] = []
        for op in dag:
            wait_t0 = time.perf_counter_ns()
            agile_op = CandidateOperation(
                operation_id=f"AGILE-{op.operation_id}",
                title=op.title,
                subsystem=op.subsystem,
                proposer_id=op.proposer_id,
                evaluator_id=op.evaluator_id,
                required_capability=op.required_capability,
                estimated_latency_us=op.estimated_latency_us,
                estimated_energy_uj=op.estimated_energy_uj,
                priority=op.priority,
                depends_on=[f"AGILE-{d}" for d in op.depends_on],
                resource_domains=set(op.resource_domains),  # Parallel disjoint locks
                reversibility=op.reversibility,
                pre_commit_validated=op.pre_commit_validated
            )
            ok, token, _ = agile_coordinator.request_and_grant_turn(agile_op)
            wait_us = (time.perf_counter_ns() - wait_t0) / 1000.0
            agile_wait_list.append(wait_us)

            if ok and token:
                actual_lat = op.estimated_latency_us
                te = actual_lat / max(1.0, token.allocated_quantum_us)
                agile_te_list.append(te)
                if actual_lat > token.allocated_quantum_us:
                    agile_overruns += 1

                # Yield turn immediately in this test batch
                agile_coordinator.yield_turn_commit(
                    token=token,
                    latency_us=actual_lat,
                    energy_uj=op.estimated_energy_uj,
                    output_payload_hash="agile_hash_xyz"
                )
                agile_completed += 1

    agile_duration_ms = max(0.001, (time.perf_counter_ns() - agile_start_ns) / 1_000_000.0)
    agile_throughput = agile_completed / agile_duration_ms
    agile_avg_te = sum(agile_te_list) / max(1, len(agile_te_list))
    agile_overrun_rate = agile_overruns / max(1, agile_completed)
    agile_rollback_rate = agile_rollbacks / max(1, agile_completed)
    agile_utility = (agile_throughput * agile_avg_te) / (
        (1.0 + agile_overrun_rate) * (1.0 + agile_rollback_rate)
    )

    drafting_gain = round((agile_utility / max(0.001, static_utility)) - 1.0, 4)

    return {
        "constitutional_invariant_verified": (prof.vital_max_hp == VITAL_MAX_HP),
        "total_operations_per_run": len(dag) * repetitions,
        "static_quantum_metrics": {
            "duration_ms": round(static_duration_ms, 2),
            "throughput_ops_per_ms": round(static_throughput, 3),
            "average_turn_efficiency": round(static_avg_te, 4),
            "quantum_overrun_rate": round(static_overrun_rate, 4),
            "rollback_rate": round(static_rollback_rate, 4),
            "utility": round(static_utility, 3)
        },
        "agile_drafted_metrics": {
            "duration_ms": round(agile_duration_ms, 2),
            "throughput_ops_per_ms": round(agile_throughput, 3),
            "average_turn_efficiency": round(agile_avg_te, 4),
            "quantum_overrun_rate": round(agile_overrun_rate, 4),
            "rollback_rate": round(agile_rollback_rate, 4),
            "utility": round(agile_utility, 3)
        },
        "drafting_gain": drafting_gain,
        "conclusion": "AGILE_DRAFTED_SUPERIOR" if drafting_gain > 0.0 else "STATIC_COMPARABLE"
    }


# Global singleton instance
GLOBAL_SUBSYSTEM_TURN_COORDINATOR = InstructionalDependencyCoordinator()

