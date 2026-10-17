"""
Krystal Stack Platform Framework - Phase-Locked Epistemic Action (PLEA) Engine
=============================================================================
Module: krystal_kernel/plea_engine.py
Milestone: V-PLEA-ENGINE-01 (Phase-Locked Epistemic Action & Meta-Theory Axioms)
Constitutional Invariant: VITAL_MAX_HP == 6

Implements the unified PLEA subsystem synthesized from the Meta-Theory Axiom Pack (MTAP v1):
  T1: Thermodynamics of Epistemic Work (Landauer-Krystal Principle) -> EpistemicCalorimeter
  T2: Topological Separation of Conflicts (Homotopy of Disagreements) -> BifurcatedConflictRing
  T3: Chronometric Phase Duality (Phase Shadow Theory) -> VBlankExperimentInjector
  T4: Cryptographic Causal Inertia (Merkle-Momentum) -> CausalInertiaDamper
  T5: Invariant Symmetry of Survival (Noether-Krystal Theorem) -> VITAL_MAX_HP == 6 Gate

Epistemic Action Integral:
  S[pi] = integral (alpha * UMA_Cost + beta * TopoConflict - gamma * CausalInertia) dt
  subject to HP(pi) == 6
"""

from __future__ import annotations

import dataclasses
import enum
import hashlib
import hmac
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from krystal_kernel.scene_constructor_presets import VITAL_MAX_HP
from krystal_kernel.contracts import (
    ContractHeader,
    ContractViolation,
    ReceiptContract,
    compute_canonical_digest,
    canonical_serialize,
)
from krystal_kernel.zero_copy_contracts import (
    BinaryContractType,
    KRYSTAL_MAGIC,
    KRYSTAL_PROTOCOL_VERSION,
    ZERO_COPY_FRAME_SIZE,
    ZeroCopyContractFrame,
)
from krystal_kernel.epistemic_enlg_compiler import (
    EnlgRole,
    EnlgEvidenceClass,
    EnlgVerdict,
    ArtifactGate,
    QuarantinedArtifact,
)

if VITAL_MAX_HP != 6:
    raise ContractViolation(
        f"Constitutional invariant breach: VITAL_MAX_HP must be 6, got {VITAL_MAX_HP}",
        violation_code="ERR_CONSTITUTIONAL_INVARIANT_BREACH",
    )


# ============================================================================
# CONSTANTS & BIT FLAGS (MTAP v1 / PLEA)
# ============================================================================

VBLANK_WINDOW_MS: float = 2.2           # Sub-frame blanking budget on 120Hz/60Hz display
PL1_BASE_POWER_WATTS: float = 28.0      # Default package power ceiling (Tiger Lake / Iris Xe)
DEFAULT_GAMMA_DECAY: float = 0.85       # Merkle-Momentum exponential decay factor
PARASITIC_ENERGY_THRESHOLD_J: float = 5.0 # Max Joules permitted before speculation is marked parasitic
PARASITIC_UMA_THRESHOLD_MB: float = 100.0 # Max UMA bandwidth before speculation is severed

# Frame bit flags
FLAG_PLEA_ROLE_MASK: int = 0x0F
FLAG_PLEA_VBLANK_INJECTED: int = 0x20
FLAG_PLEA_CONFLICT_BIFURCATION: int = 0x40
FLAG_PLEA_PARASITIC_DECOUPLED: int = 0x80

# Evidence Weight for Merkle-Momentum (T4)
EVIDENCE_MOMENTUM_WEIGHT: Dict[EnlgEvidenceClass, float] = {
    EnlgEvidenceClass.UNKNOWN: 0.0,
    EnlgEvidenceClass.SIMULATED: 0.2,
    EnlgEvidenceClass.DERIVED: 0.25,
    EnlgEvidenceClass.CODE_PRESENT: 0.3,
    EnlgEvidenceClass.LOCAL_MEASURED: 0.65,
    EnlgEvidenceClass.REMOTE_CI: 0.9,
    EnlgEvidenceClass.PHYSICAL_MEASURED: 1.0,
    EnlgEvidenceClass.EXTERNAL_REPLICATED: 1.0,
}


# ============================================================================
# 1. EPISTEMIC CALORIMETER (T1 + T3)
# ============================================================================

@dataclass
class HypothesisThermodynamics:
    """Tracks thermodynamic energy and UMA consumption for an individual hypothesis."""
    hypothesis_id: str
    start_timestamp_ns: int
    cumulative_energy_joules: float = 0.0
    cumulative_uma_bytes: int = 0
    entropy_reduction_delta: float = 0.0
    is_parasitic: bool = False
    vital_max_hp: int = VITAL_MAX_HP

    def __post_init__(self):
        if self.vital_max_hp != 6:
            raise ContractViolation("VITAL_MAX_HP must be 6 in HypothesisThermodynamics")


class EpistemicCalorimeter:
    """
    Thermodynamic accountant of hypothesis work (T1 + T3).
    Meters electrical energy (Joules) and UMA memory bandwidth (MB) consumed
    by speculative reasoning paths. Shuts down parasitic speculations.
    """

    def __init__(
        self,
        max_energy_joules: float = PARASITIC_ENERGY_THRESHOLD_J,
        max_uma_mb: float = PARASITIC_UMA_THRESHOLD_MB,
        power_watts: float = PL1_BASE_POWER_WATTS,
    ):
        if VITAL_MAX_HP != 6:
            raise ContractViolation("VITAL_MAX_HP must be 6 in EpistemicCalorimeter")
        self.max_energy_joules = max_energy_joules
        self.max_uma_bytes = int(max_uma_mb * 1024 * 1024)
        self.power_watts = power_watts
        self._hypotheses: Dict[str, HypothesisThermodynamics] = {}

    def register_hypothesis(self, hypothesis_id: str, timestamp_ns: Optional[int] = None) -> HypothesisThermodynamics:
        """Initializes thermodynamic tracking for a new hypothesis."""
        if not hypothesis_id or not hypothesis_id.strip():
            raise ContractViolation("Hypothesis ID must be non-empty", violation_code="ERR_CALORIMETER_INVALID_ID")
        t_ns = timestamp_ns if timestamp_ns is not None else time.time_ns()
        entry = HypothesisThermodynamics(hypothesis_id=hypothesis_id, start_timestamp_ns=t_ns)
        self._hypotheses[hypothesis_id] = entry
        return entry

    def record_work(
        self,
        hypothesis_id: str,
        duration_ms: float,
        uma_bytes_transferred: int,
        entropy_reduction: float = 0.0,
    ) -> HypothesisThermodynamics:
        """
        Records work done by a hypothesis.
        Energy = Power (Watts) * Time (seconds).
        """
        if hypothesis_id not in self._hypotheses:
            self.register_hypothesis(hypothesis_id)

        entry = self._hypotheses[hypothesis_id]
        if entry.is_parasitic:
            return entry

        energy_j = (self.power_watts * (duration_ms / 1000.0))
        entry.cumulative_energy_joules += energy_j
        entry.cumulative_uma_bytes += uma_bytes_transferred
        entry.entropy_reduction_delta += entropy_reduction

        # Parasitic trip check
        if (
            entry.cumulative_energy_joules >= self.max_energy_joules
            or entry.cumulative_uma_bytes >= self.max_uma_bytes
        ):
            if entropy_reduction <= 0.0 or entry.entropy_reduction_delta <= 0.0:
                entry.is_parasitic = True

        return entry

    def is_parasitic(self, hypothesis_id: str) -> bool:
        """Checks if the hypothesis has devolved into thermal noise."""
        entry = self._hypotheses.get(hypothesis_id)
        if entry is None:
            return False
        return entry.is_parasitic

    def compute_thermodynamic_cost(self, hypothesis_id: str) -> float:
        """Normalized UMA & energy cost metric for the Action Integral S[pi]."""
        entry = self._hypotheses.get(hypothesis_id)
        if entry is None:
            return 0.0
        energy_norm = entry.cumulative_energy_joules / max(1.0, self.max_energy_joules)
        uma_norm = entry.cumulative_uma_bytes / max(1.0, self.max_uma_bytes)
        return float(energy_norm + uma_norm)


# ============================================================================
# 2. BIFURCATED CONFLICT RING (T2 + T4)
# ============================================================================

@dataclass(frozen=True)
class BifurcatedBranch:
    """Parallel SPSC branch representing an unresolved conflict node."""
    branch_id: str
    generation_id: int
    fork_predecessor_hash: bytes
    frame: ZeroCopyContractFrame
    evidence_tag: EnlgEvidenceClass
    vital_max_hp: int = VITAL_MAX_HP

    def __post_init__(self):
        if self.vital_max_hp != 6:
            raise ContractViolation("VITAL_MAX_HP must be 6 in BifurcatedBranch")
        if len(self.fork_predecessor_hash) != 32:
            raise ContractViolation("fork_predecessor_hash must be exactly 32 bytes")


class BifurcatedConflictRing:
    """
    Topological separation of conflicts (T2 + T4).
    Strictly forbids convex averaging of truth:
      E_A ⊥ E_B => (E_A + E_B)/2 ∉ M_valid.
    Forks the SPSC ring into Branch Alpha and Branch Beta with identical
    predecessor_hash until collapsed by an empirical discriminating test (CDT).
    """

    def __init__(self):
        if VITAL_MAX_HP != 6:
            raise ContractViolation("VITAL_MAX_HP must be 6 in BifurcatedConflictRing")
        self._active_forks: Dict[str, Tuple[BifurcatedBranch, BifurcatedBranch]] = {}

    def attempt_convex_averaging(self, val_a: float, val_b: float) -> float:
        """Explicitly rejects any attempt to linearly blend contradictory facts."""
        raise ContractViolation(
            f"Homotopy breach (T2): Convex averaging of conflicting evidence ({val_a} ⊥ {val_b}) is forbidden. "
            "Truth is not the convex mean of contradictory measurements.",
            violation_code="ERR_HOMOTOPY_CONVEX_AVERAGING_FORBIDDEN",
        )

    def fork_conflict(
        self,
        conflict_id: str,
        parent_frame: ZeroCopyContractFrame,
        claim_a: str,
        evidence_a: EnlgEvidenceClass,
        metric_a: float,
        claim_b: str,
        evidence_b: EnlgEvidenceClass,
        metric_b: float,
    ) -> Tuple[BifurcatedBranch, BifurcatedBranch]:
        """
        Creates two orthogonal branches Alpha and Beta sharing parent_frame digest.
        """
        if parent_frame.vital_hp != 6:
            raise ContractViolation("Parent frame vital_hp must be 6")

        parent_digest = parent_frame.compute_binary_digest()

        # Build Branch Alpha
        frame_alpha = ZeroCopyContractFrame(
            magic=KRYSTAL_MAGIC,
            version=KRYSTAL_PROTOCOL_VERSION,
            vital_hp=6,
            contract_type=BinaryContractType.EVIDENCE,
            payload_size=32,
            flags=parent_frame.flags | FLAG_PLEA_CONFLICT_BIFURCATION,
            sequence_id=parent_frame.sequence_id + 1,
            generation_id=1,  # Branch Alpha ID
            timestamp_ns=time.time_ns(),
            capability_scope=parent_frame.capability_scope,
            nonce=101,
            predecessor_hash_raw=parent_digest,
            profile_id=parent_frame.profile_id,
            verdict_id=int(EnlgVerdict.CONFLICT.value == "CONFLICT"),
            approved_authority_mask=parent_frame.approved_authority_mask,
            metric1=metric_a,
            metric2=0.0,
            signature_or_digest_raw=b"\x00" * 32,
        )
        sig_alpha = frame_alpha.compute_binary_digest()
        frame_alpha = dataclasses.replace(frame_alpha, signature_or_digest_raw=sig_alpha)
        branch_alpha = BifurcatedBranch(
            branch_id=f"{conflict_id}_alpha",
            generation_id=1,
            fork_predecessor_hash=parent_digest,
            frame=frame_alpha,
            evidence_tag=evidence_a,
        )

        # Build Branch Beta
        frame_beta = ZeroCopyContractFrame(
            magic=KRYSTAL_MAGIC,
            version=KRYSTAL_PROTOCOL_VERSION,
            vital_hp=6,
            contract_type=BinaryContractType.EVIDENCE,
            payload_size=32,
            flags=parent_frame.flags | FLAG_PLEA_CONFLICT_BIFURCATION,
            sequence_id=parent_frame.sequence_id + 1,
            generation_id=2,  # Branch Beta ID
            timestamp_ns=time.time_ns(),
            capability_scope=parent_frame.capability_scope,
            nonce=102,
            predecessor_hash_raw=parent_digest,
            profile_id=parent_frame.profile_id,
            verdict_id=int(EnlgVerdict.CONFLICT.value == "CONFLICT"),
            approved_authority_mask=parent_frame.approved_authority_mask,
            metric1=metric_b,
            metric2=0.0,
            signature_or_digest_raw=b"\x00" * 32,
        )
        sig_beta = frame_beta.compute_binary_digest()
        frame_beta = dataclasses.replace(frame_beta, signature_or_digest_raw=sig_beta)
        branch_beta = BifurcatedBranch(
            branch_id=f"{conflict_id}_beta",
            generation_id=2,
            fork_predecessor_hash=parent_digest,
            frame=frame_beta,
            evidence_tag=evidence_b,
        )

        self._active_forks[conflict_id] = (branch_alpha, branch_beta)
        return branch_alpha, branch_beta

    def collapse_branch(
        self,
        conflict_id: str,
        winning_branch_generation: int,
        cdt_receipt: ReceiptContract,
    ) -> ZeroCopyContractFrame:
        """
        Collapses the bifurcated superposition once the Cheapest Discriminating Test (CDT) completes.
        Returns the surviving authoritative 128B frame.
        """
        if conflict_id not in self._active_forks:
            raise ContractViolation(f"Unknown conflict fork: {conflict_id}")

        if winning_branch_generation not in (1, 2):
            raise ContractViolation("winning_branch_generation must be 1 (Alpha) or 2 (Beta)")

        branch_alpha, branch_beta = self._active_forks[conflict_id]
        chosen_branch = branch_alpha if winning_branch_generation == 1 else branch_beta

        if cdt_receipt.header.vital_max_hp != 6:
            raise ContractViolation("CDT receipt vital_max_hp must be 6")

        # Create collapsed trunk frame
        collapsed_frame = ZeroCopyContractFrame(
            magic=KRYSTAL_MAGIC,
            version=KRYSTAL_PROTOCOL_VERSION,
            vital_hp=6,
            contract_type=BinaryContractType.RECEIPT,
            payload_size=32,
            flags=chosen_branch.frame.flags & ~FLAG_PLEA_CONFLICT_BIFURCATION,
            sequence_id=chosen_branch.frame.sequence_id + 1,
            generation_id=0,  # Merged to unified trunk
            timestamp_ns=time.time_ns(),
            capability_scope=chosen_branch.frame.capability_scope,
            nonce=201,
            predecessor_hash_raw=chosen_branch.frame.signature_or_digest_raw,
            profile_id=chosen_branch.frame.profile_id,
            verdict_id=int(EnlgVerdict.MEASURED_WITH_SCOPE.value == "MEASURED_WITH_SCOPE"),
            approved_authority_mask=chosen_branch.frame.approved_authority_mask,
            metric1=chosen_branch.frame.metric1,
            metric2=chosen_branch.frame.metric2,
            signature_or_digest_raw=b"\x00" * 32,
        )
        sig = collapsed_frame.compute_binary_digest()
        collapsed_frame = dataclasses.replace(collapsed_frame, signature_or_digest_raw=sig)

        del self._active_forks[conflict_id]
        return collapsed_frame


# ============================================================================
# 3. VBLANK EXPERIMENT INJECTOR (T3 + T1)
# ============================================================================

class VBlankExperimentInjector:
    """
    Sub-millisecond shadow phase executor (T3 + T1).
    Ensures:
      Phi_Actuation(t) * Phi_Epistemic(t) == 0.
    Invasive epistemic probing is strictly confined within the 2.2 ms VBlank window.
    If an experiment threatens to overrun, it is suspended/yielded to prevent frame stutter.
    """

    def __init__(self, vblank_budget_ms: float = VBLANK_WINDOW_MS):
        if VITAL_MAX_HP != 6:
            raise ContractViolation("VITAL_MAX_HP must be 6 in VBlankExperimentInjector")
        self.vblank_budget_ms = vblank_budget_ms

    def can_inject(self, estimated_duration_ms: float, elapsed_vblank_ms: float) -> bool:
        """Determines if the experiment fits into the remaining phase shadow."""
        remaining_ms = self.vblank_budget_ms - elapsed_vblank_ms
        return estimated_duration_ms <= remaining_ms

    def execute_in_shadow(
        self,
        experiment_fn: Callable[[], Any],
        estimated_duration_ms: float,
        elapsed_vblank_ms: float,
    ) -> Tuple[bool, Any, float]:
        """
        Executes an experiment if within the shadow budget; otherwise yields execution.
        Returns: (success, result_or_reason, actual_duration_ms)
        """
        if not self.can_inject(estimated_duration_ms, elapsed_vblank_ms):
            return False, "ERR_VBLANK_BUDGET_EXCEEDED_YIELD_BEFORE_SCANOUT", 0.0

        t0 = time.perf_counter()
        result = experiment_fn()
        t1 = time.perf_counter()
        actual_ms = (t1 - t0) * 1000.0

        return True, result, actual_ms


# ============================================================================
# 4. CAUSAL INERTIA DAMPER (T4 + T5)
# ============================================================================

class CausalInertiaDamper:
    """
    Cryptographic Causal Momentum filter (T4 + T5).
    Weights incoming proposals by historical Merkle-chain momentum:
      M(Chain_k) = sum gamma^(k-i) * Weight(Evidence_i).
    Speculative proposals from PROPOSER cannot mutate authoritative system state
    if their evidence momentum is inferior to trunk inertia.
    Guarantees VITAL_MAX_HP == 6 is invariant across all mutations.
    """

    def __init__(self, gamma: float = DEFAULT_GAMMA_DECAY):
        if VITAL_MAX_HP != 6:
            raise ContractViolation("VITAL_MAX_HP must be 6 in CausalInertiaDamper")
        self.gamma = gamma
        self._current_momentum: float = 0.0
        self._last_frame_digest: Optional[bytes] = None
        self._step_count: int = 0

    @property
    def current_momentum(self) -> float:
        return self._current_momentum

    def reset_recurrent_state(self, initial_momentum: float = 0.0) -> None:
        self._current_momentum = float(initial_momentum)
        self._last_frame_digest = None
        self._step_count = 0

    def step_recurrent(
        self,
        evidence: EnlgEvidenceClass,
        frame: Optional[ZeroCopyContractFrame] = None,
        session_key: Optional[bytes] = None,
    ) -> float:
        """
        O(1) Recurrent Merkle-Momentum step:
          M_k = gamma * M_{k-1} + Weight(Evidence_k)
        Executed in O(1) constant time with zero allocations.
        Optionally validates cryptographic continuity against previous frame's digest.
        """
        if frame is not None:
            if frame.vital_hp != 6:
                raise ContractViolation(
                    "Noether-Krystal invariant breach: Frame vital_hp must be 6 in Merkle-Momentum step",
                    violation_code="ERR_PLEA_MOMENTUM_HP_BREACH",
                )
            if not frame.verify_frame_integrity(session_key):
                raise ContractViolation(
                    "Frame integrity verification failed during Merkle-Momentum computation",
                    violation_code="ERR_PLEA_MOMENTUM_INTEGRITY_FAILED",
                )
            if self._last_frame_digest is not None:
                if frame.predecessor_hash_raw != self._last_frame_digest:
                    raise ContractViolation(
                        "Cryptographic causal break: predecessor_hash_raw does not match previous frame digest",
                        violation_code="ERR_PLEA_MOMENTUM_CHAIN_BROKEN",
                    )
            self._last_frame_digest = frame.signature_or_digest_raw

        weight = EVIDENCE_MOMENTUM_WEIGHT.get(evidence, 0.0)
        self._current_momentum = float((self.gamma * self._current_momentum) + weight)
        self._step_count += 1
        return self._current_momentum

    def compute_momentum(self, evidence_stream: List[EnlgEvidenceClass]) -> float:
        """Computes exponential causal momentum of a verified evidence sequence."""
        k = len(evidence_stream)
        if k == 0:
            return 0.0

        momentum = 0.0
        for i, ev in enumerate(evidence_stream):
            weight = EVIDENCE_MOMENTUM_WEIGHT.get(ev, 0.0)
            factor = self.gamma ** (k - 1 - i)
            momentum += factor * weight
        return float(momentum)

    def evaluate_proposal_admissibility(
        self,
        trunk_evidence: Optional[List[EnlgEvidenceClass]],
        proposal_evidence: EnlgEvidenceClass,
        proposal_role: EnlgRole,
        is_hp_valid: bool = True,
    ) -> Tuple[bool, str]:
        """
        Determines whether a proposal has sufficient momentum to influence authoritative state.
        Supports both static history list and O(1) streaming recurrent momentum.
        """
        if not is_hp_valid:
            raise ContractViolation("Noether-Krystal invariant breach: HP must be 6")

        if trunk_evidence:
            trunk_momentum = self.compute_momentum(trunk_evidence)
        else:
            trunk_momentum = self._current_momentum

        prop_weight = EVIDENCE_MOMENTUM_WEIGHT.get(proposal_evidence, 0.0)

        # Proposer with unverified evidence cannot override heavy trunk
        if proposal_role == EnlgRole.PROPOSER:
            if proposal_evidence in (EnlgEvidenceClass.UNKNOWN, EnlgEvidenceClass.SIMULATED, EnlgEvidenceClass.DERIVED):
                if trunk_momentum > 1.5:
                    return False, f"PROPOSAL_DAMPENED: Trunk momentum {trunk_momentum:.2f} > Speculative weight {prop_weight:.2f}"

        return True, "PROPOSAL_ADMISSIBLE"


# ============================================================================
# 5. UNIFIED PLEA ENGINE
# ============================================================================

@dataclass(frozen=True)
class PleaTrajectoryVerdict:
    """Actionable result of the Epistemic Action Integral evaluation."""
    admissible: bool
    action_integral_value: float
    thermodynamic_cost: float
    has_conflict: bool
    causal_momentum: float
    status_code: str
    vital_max_hp: int = VITAL_MAX_HP

    def __post_init__(self):
        if self.vital_max_hp != 6:
            raise ContractViolation("VITAL_MAX_HP must be 6 in PleaTrajectoryVerdict")


class PleaEngine:
    """
    Unified engine for Phase-Locked Epistemic Action (PLEA).
    Synthesizes Calorimeter, ConflictRing, VBlankInjector, CausalDamper, and ArtifactGate
    into the Epistemic Action Integral:
      S[pi] = integral (alpha * UMA_Cost + beta * TopoConflict - gamma * CausalInertia) dt
      subject to VITAL_MAX_HP == 6
    """

    def __init__(
        self,
        calorimeter: Optional[EpistemicCalorimeter] = None,
        conflict_ring: Optional[BifurcatedConflictRing] = None,
        vblank_injector: Optional[VBlankExperimentInjector] = None,
        inertia_damper: Optional[CausalInertiaDamper] = None,
        artifact_gate: Optional[ArtifactGate] = None,
    ):
        if VITAL_MAX_HP != 6:
            raise ContractViolation("VITAL_MAX_HP must be 6 in PleaEngine")

        self.calorimeter = calorimeter or EpistemicCalorimeter()
        self.conflict_ring = conflict_ring or BifurcatedConflictRing()
        self.vblank_injector = vblank_injector or VBlankExperimentInjector()
        self.inertia_damper = inertia_damper or CausalInertiaDamper()
        self.artifact_gate = artifact_gate or ArtifactGate()

    def evaluate_action_integral(
        self,
        uma_cost: float,
        has_topological_conflict: bool,
        causal_inertia: float,
        alpha: float = 1.0,
        beta: float = 2.0,
        gamma: float = 0.5,
    ) -> float:
        """
        Computes S[pi] = alpha * UMA_Cost + beta * TopoConflict - gamma * CausalInertia.
        Lower action represents a more stable, physically grounded epistemic trajectory.
        """
        conflict_term = 1.0 if has_topological_conflict else 0.0
        s_val = (alpha * uma_cost) + (beta * conflict_term) - (gamma * causal_inertia)
        return float(s_val)

    def process_proposal(
        self,
        hypothesis_id: str,
        proposal_role: EnlgRole,
        proposal_evidence: EnlgEvidenceClass,
        trunk_history: List[EnlgEvidenceClass],
        estimated_vblank_ms: float = 0.5,
        elapsed_vblank_ms: float = 0.0,
        uma_bytes: int = 1024,
    ) -> PleaTrajectoryVerdict:
        """
        Evaluates a candidate proposal against the thermodynamic, homotopy,
        vblank, and momentum barriers.
        """
        if VITAL_MAX_HP != 6:
            raise ContractViolation("VITAL_MAX_HP must be 6 in process_proposal")

        # 1. Calorimeter check (T1)
        if self.calorimeter.is_parasitic(hypothesis_id):
            return PleaTrajectoryVerdict(
                admissible=False,
                action_integral_value=999.0,
                thermodynamic_cost=self.calorimeter.compute_thermodynamic_cost(hypothesis_id),
                has_conflict=False,
                causal_momentum=0.0,
                status_code="REJECTED_THERMAL_PARASITE",
            )

        # Record work
        self.calorimeter.record_work(
            hypothesis_id=hypothesis_id,
            duration_ms=estimated_vblank_ms,
            uma_bytes_transferred=uma_bytes,
        )
        thermo_cost = self.calorimeter.compute_thermodynamic_cost(hypothesis_id)

        # 2. VBlank shadow check (T3)
        if not self.vblank_injector.can_inject(estimated_vblank_ms, elapsed_vblank_ms):
            return PleaTrajectoryVerdict(
                admissible=False,
                action_integral_value=500.0,
                thermodynamic_cost=thermo_cost,
                has_conflict=False,
                causal_momentum=0.0,
                status_code="YIELDED_VBLANK_WINDOW_EXCEEDED",
            )

        # 3. Causal inertia check (T4 + T5)
        admissible, reason = self.inertia_damper.evaluate_proposal_admissibility(
            trunk_evidence=trunk_history,
            proposal_evidence=proposal_evidence,
            proposal_role=proposal_role,
            is_hp_valid=True,
        )
        trunk_momentum = self.inertia_damper.compute_momentum(trunk_history)

        action_integral = self.evaluate_action_integral(
            uma_cost=thermo_cost,
            has_topological_conflict=False,
            causal_inertia=trunk_momentum,
        )

        return PleaTrajectoryVerdict(
            admissible=admissible,
            action_integral_value=action_integral,
            thermodynamic_cost=thermo_cost,
            has_conflict=False,
            causal_momentum=trunk_momentum,
            status_code=reason if admissible else "DAMPENED_INSUFFICIENT_MOMENTUM",
        )
