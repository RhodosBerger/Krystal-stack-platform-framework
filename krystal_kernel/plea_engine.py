"""
Krystal Stack Platform Framework - Phase-Locked Epistemic Action (PLEA Engine)
=============================================================================
Module: krystal_kernel/plea_engine.py
Milestones: V-PLEA-ENGINE-01 / KSPP v1 (Principle 3, 5, 6, 8)
Constitutional Invariant: VITAL_MAX_HP == 6

Executes the Meta-Theory Axiom Pack (MTAP v1):
  T1 (Calorimeter): Landauer work metering & parasitic hypothesis decoupling.
  T2 (Homotopy): Bifurcated SPSC conflict ring, prohibits convex averaging of truth.
  T3 (Chronometrics): Sub-millisecond VBlank experiment injection (Phi_Scanout * Phi_Epistemic == 0).
  T4 & T5 (Inertia): O(1) recurrent Merkle-Momentum damping and action integral minimization.
"""

from __future__ import annotations

import dataclasses
import hashlib
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Final

from krystal_kernel.constants import (
    VITAL_MAX_HP,
    UMA_SATURATION_THRESHOLD,
    HYSTERESIS_DWELL_FRAMES,
    FrameFlags,
    ContractViolation,
)
from krystal_kernel.zero_copy_contracts import (
    BinaryContractType,
    KRYSTAL_MAGIC,
    KRYSTAL_PROTOCOL_VERSION,
    ZeroCopyContractFrame,
    canonicalize_float32,
)
from krystal_kernel.epistemic_enlg_compiler import (
    EnlgRole,
    EnlgEvidenceClass,
    EnlgVerdict,
    ArtifactGate,
)
from krystal_kernel.contracts import ReceiptContract

if VITAL_MAX_HP != 6:
    raise ContractViolation(
        f"Constitutional invariant breach: VITAL_MAX_HP must be 6, got {VITAL_MAX_HP}",
        violation_code="ERR_CONSTITUTIONAL_INVARIANT_BREACH",
    )

# Hardware & Engine Defaults
VBLANK_WINDOW_MS: Final[float] = 2.20
DEFAULT_GAMMA_DECAY: Final[float] = 0.85
PARASITIC_ENERGY_THRESHOLD_J: Final[float] = 5.0
PARASITIC_UMA_THRESHOLD_MB: Final[float] = 100.0
PL1_BASE_POWER_WATTS: Final[float] = 28.0

# Bit flag for bifurcated conflict in FrameFlags
FLAG_PLEA_CONFLICT_BIFURCATION: Final[int] = int(FrameFlags.BIFURCATED_SUPERPOSITION)

# Specific Violation Codes
ERR_PLEA_THERMAL_BUDGET_EXCEEDED: Final[str] = "ERR_PLEA_THERMAL_BUDGET_EXCEEDED"
ERR_PLEA_HOMOTOPY_VIOLATION: Final[str] = "ERR_PLEA_HOMOTOPY_VIOLATION"
ERR_PLEA_VBLANK_OVERRUN: Final[str] = "ERR_PLEA_VBLANK_OVERRUN"
ERR_PLEA_MOMENTUM_CHAIN_BROKEN: Final[str] = "ERR_PLEA_MOMENTUM_CHAIN_BROKEN"

EVIDENCE_MOMENTUM_WEIGHT: Final[Dict[EnlgEvidenceClass, float]] = {
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
class HypothesisMeter:
    """Blueprint hypothesis thermodynamic meter."""
    hypothesis_id: str
    joules_consumed: float = 0.0
    uma_megabytes_transferred: float = 0.0
    entropy_reduced: bool = False
    is_decoupled_parasite: bool = False
    start_timestamp_ns: int = 0
    entropy_reduction_delta: float = 0.0
    vital_max_hp: int = VITAL_MAX_HP

    def __post_init__(self):
        if self.vital_max_hp != 6:
            raise ContractViolation("VITAL_MAX_HP must be 6 in HypothesisMeter")

    @property
    def cumulative_energy_joules(self) -> float:
        return self.joules_consumed

    @property
    def cumulative_uma_bytes(self) -> int:
        return int(round(self.uma_megabytes_transferred * 1024 * 1024))

    @property
    def is_parasitic(self) -> bool:
        return self.is_decoupled_parasite


# Alias for compatibility
HypothesisThermodynamics = HypothesisMeter


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
        max_joules: Optional[float] = None,
    ) -> None:
        if VITAL_MAX_HP != 6:
            raise ContractViolation("VITAL_MAX_HP must be 6 in EpistemicCalorimeter")
        self.max_joules = max_joules if max_joules is not None else max_energy_joules
        self.max_uma_mb = max_uma_mb
        self.power_watts = power_watts
        self.meters: Dict[str, HypothesisMeter] = {}

    def register_hypothesis(self, hypothesis_id: str, timestamp_ns: Optional[int] = None) -> HypothesisMeter:
        """Initializes thermodynamic tracking for a new hypothesis."""
        if not hypothesis_id or not hypothesis_id.strip():
            raise ContractViolation("Hypothesis ID must be non-empty", violation_code="ERR_CALORIMETER_INVALID_ID")
        entry = HypothesisMeter(hypothesis_id=hypothesis_id)
        self.meters[hypothesis_id] = entry
        return entry

    def record_work(
        self,
        hypothesis_id: str,
        delta_joules_or_duration_ms: Optional[float] = None,
        delta_mb_or_uma_bytes: Optional[float] = None,
        promoted_or_entropy_reduction: Any = None,
        *,
        delta_joules: Optional[float] = None,
        delta_mb: Optional[float] = None,
        promoted: Optional[bool] = None,
        duration_ms: Optional[float] = None,
        uma_bytes_transferred: Optional[int | float] = None,
        entropy_reduction: Optional[float] = None,
    ) -> HypothesisMeter:
        """
        Polymorphic work recorder:
        Form 1 (Blueprint): record_work(hypothesis_id, delta_joules=..., delta_mb=..., promoted=...)
        Form 2 (Trace):     record_work(hypothesis_id, duration_ms=..., uma_bytes_transferred=..., entropy_reduction=...)
        """
        m = self.meters.setdefault(hypothesis_id, HypothesisMeter(hypothesis_id))
        if m.is_decoupled_parasite:
            raise ContractViolation(f"Hypothesis {hypothesis_id} already decoupled as thermal parasite", violation_code=ERR_PLEA_THERMAL_BUDGET_EXCEEDED)

        # Check keyword arguments for Form 1
        if delta_joules is not None or delta_mb is not None or promoted is not None:
            dj = float(delta_joules or 0.0)
            dmb = float(delta_mb or 0.0)
            p = bool(promoted)
            m.joules_consumed += dj
            m.uma_megabytes_transferred += dmb
            if p:
                m.entropy_reduced = True
                m.entropy_reduction_delta += 1.0
            if not m.entropy_reduced and (m.joules_consumed > self.max_joules or m.uma_megabytes_transferred > self.max_uma_mb):
                m.is_decoupled_parasite = True
            return m

        # Check keyword arguments for Form 2
        if duration_ms is not None or uma_bytes_transferred is not None or entropy_reduction is not None:
            dur = float(duration_ms or 0.0)
            ub = float(uma_bytes_transferred or 0.0)
            er = float(entropy_reduction or 0.0)
            energy_j = (self.power_watts * (dur / 1000.0))
            uma_mb = ub / (1024.0 * 1024.0)
            m.joules_consumed += energy_j
            m.uma_megabytes_transferred += uma_mb
            m.entropy_reduction_delta += er
            if er > 0.0:
                m.entropy_reduced = True
            if (m.joules_consumed >= self.max_joules or m.uma_megabytes_transferred >= self.max_uma_mb):
                if er <= 0.0 or not m.entropy_reduced:
                    m.is_decoupled_parasite = True
            return m

        # Positional arguments dispatch
        if isinstance(promoted_or_entropy_reduction, bool):
            delta_j = float(delta_joules_or_duration_ms or 0.0)
            delta_mb = float(delta_mb_or_uma_bytes or 0.0)
            p = promoted_or_entropy_reduction
            m.joules_consumed += delta_j
            m.uma_megabytes_transferred += delta_mb
            if p:
                m.entropy_reduced = True
                m.entropy_reduction_delta += 1.0
            if (m.joules_consumed > self.max_joules or m.uma_megabytes_transferred > self.max_uma_mb):
                if not p:
                    m.is_decoupled_parasite = True
        else:
            dur = float(delta_joules_or_duration_ms or 0.0)
            ub = float(delta_mb_or_uma_bytes or 0.0)
            er = float(promoted_or_entropy_reduction or 0.0)
            energy_j = (self.power_watts * (dur / 1000.0))
            uma_mb = ub / (1024.0 * 1024.0)
            m.joules_consumed += energy_j
            m.uma_megabytes_transferred += uma_mb
            m.entropy_reduction_delta += er
            if er > 0.0:
                m.entropy_reduced = True
            if (m.joules_consumed >= self.max_joules or m.uma_megabytes_transferred >= self.max_uma_mb):
                if er <= 0.0 or not m.entropy_reduced:
                    m.is_decoupled_parasite = True

        return m

    # Compatibility properties and helpers
    @property
    def cumulative_energy_joules(self) -> float:
        return sum(m.joules_consumed for m in self.meters.values())

    def is_parasitic(self, hypothesis_id: str) -> bool:
        m = self.meters.get(hypothesis_id)
        return m.is_decoupled_parasite if m else False

    def compute_thermodynamic_cost(self, hypothesis_id: str) -> float:
        m = self.meters.get(hypothesis_id)
        if not m:
            return 0.0
        return (m.joules_consumed / max(self.max_joules, 1e-6)) + (m.uma_megabytes_transferred / max(self.max_uma_mb, 1e-6))


# ============================================================================
# 2. BIFURCATED CONFLICT RING (T2 + T4)
# ============================================================================

@dataclass
class BifurcatedBranch:
    """Explicit branch representation within a bifurcated conflict fork."""
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
    """

    def __init__(self) -> None:
        if VITAL_MAX_HP != 6:
            raise ContractViolation("VITAL_MAX_HP must be 6 in BifurcatedConflictRing")
        self.branches: Dict[int, ZeroCopyContractFrame] = {}
        self._active_forks: Dict[str, Tuple[BifurcatedBranch, BifurcatedBranch]] = {}
        self.resolved_winner: Optional[ZeroCopyContractFrame] = None

    @staticmethod
    def attempt_convex_averaging(val_a: float, val_b: float) -> float:
        raise ContractViolation(
            f"Homotopic Conflict Law: Convex averaging of orthogonal evidence ({val_a} ⊥ {val_b}) is forbidden. "
            "Truth is not the convex mean of contradictory measurements.",
            violation_code="ERR_HOMOTOPY_CONVEX_AVERAGING_FORBIDDEN",
        )

    def fork_superposition(
        self,
        common_predecessor_hash: bytes,
        frame_alpha: ZeroCopyContractFrame,
        frame_beta: ZeroCopyContractFrame,
    ) -> Tuple[ZeroCopyContractFrame, ZeroCopyContractFrame]:
        """Blueprint fork protocol: sets generation=1 (Alpha) and generation=2 (Beta)."""
        frame_alpha.predecessor_hash = common_predecessor_hash
        frame_alpha.generation = 1
        frame_alpha.flags |= int(FrameFlags.BIFURCATED_SUPERPOSITION)

        frame_beta.predecessor_hash = common_predecessor_hash
        frame_beta.generation = 2
        frame_beta.flags |= int(FrameFlags.BIFURCATED_SUPERPOSITION)

        self.branches[1] = frame_alpha
        self.branches[2] = frame_beta
        return frame_alpha, frame_beta

    def collapse_with_cdt_receipt(
        self, winning_generation: int, cdt_receipt_valid: bool
    ) -> ZeroCopyContractFrame:
        """Blueprint collapse protocol: collapses branches into authoritative winner."""
        if not cdt_receipt_valid or winning_generation not in self.branches:
            raise ContractViolation("Cannot collapse bifurcated ring without valid CDT Receipt", violation_code="ERR_PLEA_HOMOTOPY_VIOLATION")
        winner = self.branches[winning_generation]
        winner.flags &= ~int(FrameFlags.BIFURCATED_SUPERPOSITION)
        self.resolved_winner = winner
        self.branches.clear()
        return winner

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
        """Creates two orthogonal branches Alpha and Beta sharing parent_frame digest."""
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
            sequence=parent_frame.sequence + 1,
            generation=1,
            timestamp_ns=time.time_ns(),
            cap_scope=parent_frame.cap_scope,
            nonce=101,
            predecessor_hash=parent_digest,
            profile_id=parent_frame.profile_id,
            verdict_id=int(EnlgVerdict.CONFLICT),
            mask=parent_frame.mask,
            gpu_render_ms=metric_a,
            drs_scale_factor=0.0,
            frame_hmac=b"\x00" * 32,
        )
        frame_alpha.frame_hmac = frame_alpha.compute_binary_digest()
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
            sequence=parent_frame.sequence + 1,
            generation=2,
            timestamp_ns=time.time_ns(),
            cap_scope=parent_frame.cap_scope,
            nonce=102,
            predecessor_hash=parent_digest,
            profile_id=parent_frame.profile_id,
            verdict_id=int(EnlgVerdict.CONFLICT),
            mask=parent_frame.mask,
            gpu_render_ms=metric_b,
            drs_scale_factor=0.0,
            frame_hmac=b"\x00" * 32,
        )
        frame_beta.frame_hmac = frame_beta.compute_binary_digest()
        branch_beta = BifurcatedBranch(
            branch_id=f"{conflict_id}_beta",
            generation_id=2,
            fork_predecessor_hash=parent_digest,
            frame=frame_beta,
            evidence_tag=evidence_b,
        )

        self._active_forks[conflict_id] = (branch_alpha, branch_beta)
        self.branches[1] = frame_alpha
        self.branches[2] = frame_beta
        return branch_alpha, branch_beta

    def collapse_branch(
        self,
        conflict_id: str,
        winning_branch_generation: int,
        cdt_receipt: ReceiptContract,
    ) -> ZeroCopyContractFrame:
        """Collapses branch with ReceiptContract."""
        if conflict_id not in self._active_forks:
            raise ContractViolation(f"Unknown conflict fork: {conflict_id}")

        if winning_branch_generation not in (1, 2):
            raise ContractViolation("winning_branch_generation must be 1 (Alpha) or 2 (Beta)")

        branch_alpha, branch_beta = self._active_forks[conflict_id]
        chosen_branch = branch_alpha if winning_branch_generation == 1 else branch_beta

        if cdt_receipt.header.vital_max_hp != 6:
            raise ContractViolation("CDT receipt vital_max_hp must be 6")

        collapsed_frame = ZeroCopyContractFrame(
            magic=KRYSTAL_MAGIC,
            version=KRYSTAL_PROTOCOL_VERSION,
            vital_hp=6,
            contract_type=BinaryContractType.RECEIPT,
            payload_size=32,
            flags=chosen_branch.frame.flags & ~FLAG_PLEA_CONFLICT_BIFURCATION,
            sequence=chosen_branch.frame.sequence + 1,
            generation=0,
            timestamp_ns=time.time_ns(),
            cap_scope=chosen_branch.frame.cap_scope,
            nonce=201,
            predecessor_hash=chosen_branch.frame.frame_hmac,
            profile_id=chosen_branch.frame.profile_id,
            verdict_id=int(EnlgVerdict.MEASURED_WITH_SCOPE),
            mask=chosen_branch.frame.mask,
            gpu_render_ms=chosen_branch.frame.gpu_render_ms,
            drs_scale_factor=chosen_branch.frame.drs_scale_factor,
            frame_hmac=b"\x00" * 32,
        )
        collapsed_frame.frame_hmac = collapsed_frame.compute_binary_digest()

        del self._active_forks[conflict_id]
        if winning_branch_generation in self.branches:
            self.branches.clear()
        return collapsed_frame


# ============================================================================
# 3. VBLANK EXPERIMENT INJECTOR (T3 + T1)
# ============================================================================

class VBlankExperimentInjector:
    """
    Sub-millisecond shadow phase executor (T3 + T1).
    Ensures Phi_Actuation(t) * Phi_Epistemic(t) == 0.
    Invasive epistemic probing is strictly confined within the 2.2 ms VBlank window.
    """

    def __init__(self, vblank_budget_ms: float = VBLANK_WINDOW_MS, jitter_guard_ms: float = 0.05) -> None:
        if VITAL_MAX_HP != 6:
            raise ContractViolation("VITAL_MAX_HP must be 6 in VBlankExperimentInjector")
        self.vblank_budget_ms = vblank_budget_ms
        self.jitter_guard_ms = jitter_guard_ms

    def can_inject(self, estimated_duration_ms: float, elapsed_vblank_ms: float) -> bool:
        remaining_ms = self.vblank_budget_ms - elapsed_vblank_ms
        return estimated_duration_ms <= (remaining_ms - self.jitter_guard_ms + 1e-9)

    def execute_in_shadow(
        self,
        experiment_fn_or_scanout: Any = None,
        estimated_duration_ms: float = 0.0,
        task_or_elapsed: Any = 0.0,
        *,
        scanout_active: Optional[bool] = None,
        task: Optional[Callable[[], bool]] = None,
        experiment_fn: Optional[Callable[[], Any]] = None,
        elapsed_vblank_ms: Optional[float] = None,
    ) -> Any:
        """
        Polymorphic invocation:
        Form 1 (Blueprint): execute_in_shadow(scanout_active=..., estimated_duration_ms=..., task=...) -> Tuple[bool, str]
        Form 2 (Trace):     execute_in_shadow(experiment_fn=..., estimated_duration_ms=..., elapsed_vblank_ms=0.0) -> Tuple[bool, Any, float]
        """
        # Form 1: Blueprint invocation
        is_blueprint = (scanout_active is not None) or (task is not None) or isinstance(experiment_fn_or_scanout, bool)
        if is_blueprint:
            s_active = scanout_active if scanout_active is not None else bool(experiment_fn_or_scanout)
            t_fn = task if task is not None else task_or_elapsed
            if s_active:
                return False, "DEFERRED_SCANOUT_ACTIVE"
            usable_window = self.vblank_budget_ms - self.jitter_guard_ms
            if estimated_duration_ms > usable_window:
                return False, "YIELDED_BEFORE_DEADLINE"
            return t_fn(), "EXECUTED_IN_VBLANK_SHADOW"

        # Form 2: execute_in_shadow(experiment_fn: Callable, estimated_duration_ms: float, elapsed_vblank_ms: float = 0.0)
        actual_experiment_fn: Callable[[], Any] = experiment_fn if experiment_fn is not None else experiment_fn_or_scanout
        actual_elapsed_ms: float = elapsed_vblank_ms if elapsed_vblank_ms is not None else float(task_or_elapsed or 0.0)

        remaining_ms = self.vblank_budget_ms - actual_elapsed_ms
        if estimated_duration_ms > remaining_ms:
            return False, "ERR_VBLANK_BUDGET_EXCEEDED_YIELD_BEFORE_SCANOUT", 0.0

        t0 = time.perf_counter()
        result = actual_experiment_fn()
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
    """

    EVIDENCE_WEIGHTS: Final[Dict[EnlgEvidenceClass, float]] = {
        EnlgEvidenceClass.UNKNOWN: 0.0,
        EnlgEvidenceClass.DERIVED: 0.1,
        EnlgEvidenceClass.CODE_PRESENT: 0.2,
        EnlgEvidenceClass.SIMULATED: 0.3,
        EnlgEvidenceClass.LOCAL_MEASURED: 1.0,
        EnlgEvidenceClass.REMOTE_CI: 1.5,
        EnlgEvidenceClass.PHYSICAL_MEASURED: 2.5,
        EnlgEvidenceClass.EXTERNAL_REPLICATED: 3.0,
    }

    def __init__(self, gamma: float = DEFAULT_GAMMA_DECAY, threshold: float = 1.0) -> None:
        if VITAL_MAX_HP != 6:
            raise ContractViolation("VITAL_MAX_HP must be 6 in CausalInertiaDamper")
        self.gamma = gamma
        self.threshold = threshold
        self.momentum: float = 0.0
        self.last_frame_hash: bytes = b"\x00" * 32
        self._current_momentum: float = 0.0
        self._last_frame_digest: Optional[bytes] = None
        self._step_count: int = 0

    @property
    def current_momentum(self) -> float:
        return self.momentum or self._current_momentum

    def reset_recurrent_state(self, initial_momentum: float = 0.0) -> None:
        self.momentum = float(initial_momentum)
        self._current_momentum = float(initial_momentum)
        self.last_frame_hash = b"\x00" * 32
        self._last_frame_digest = None
        self._step_count = 0

    def push_verified_frame(self, frame: ZeroCopyContractFrame, evidence_class: EnlgEvidenceClass) -> float:
        if frame.vital_hp != VITAL_MAX_HP:
            raise ContractViolation("Noether-Krystal survival symmetry violated: vital_hp != 6")
        if self.last_frame_hash != (b"\x00" * 32) and frame.predecessor_hash != self.last_frame_hash:
            raise ContractViolation("Causal chain broken: predecessor_hash mismatch", violation_code=ERR_PLEA_MOMENTUM_CHAIN_BROKEN)

        weight = self.EVIDENCE_WEIGHTS.get(evidence_class, 0.0)
        self.momentum = float((self.gamma * self.momentum) + weight)
        self.last_frame_hash = hashlib.sha256(frame.frame_hmac).digest()
        self._current_momentum = self.momentum
        return self.momentum

    def can_override_trunk(self, proposal_weight: float, min_ratio: float = 0.60) -> bool:
        return proposal_weight >= (self.current_momentum * min_ratio)

    def step_recurrent(
        self,
        evidence: Optional[EnlgEvidenceClass] = None,
        frame: Optional[ZeroCopyContractFrame] = None,
        session_key: Optional[bytes] = None,
        proposal_delta: Optional[float] = None,
        evidence_weight: Optional[float] = None,
        predecessor_hash: Optional[bytes] = None,
    ) -> Any:
        """O(1) Recurrent Merkle-Momentum step."""
        if frame is not None:
            if frame.vital_hp != 6:
                raise ContractViolation("Noether-Krystal invariant breach: Frame vital_hp must be 6 in Merkle-Momentum step", violation_code="ERR_PLEA_MOMENTUM_HP_BREACH")
            if not frame.verify_frame_integrity(session_key):
                raise ContractViolation("Frame integrity verification failed during Merkle-Momentum computation", violation_code="ERR_PLEA_MOMENTUM_INTEGRITY_FAILED")
            if self._last_frame_digest is not None:
                pred = getattr(frame, "predecessor_hash_raw", None) or getattr(frame, "predecessor_hash", None)
                if pred != self._last_frame_digest:
                    raise ContractViolation("Cryptographic causal break: predecessor_hash_raw does not match previous frame digest", violation_code=ERR_PLEA_MOMENTUM_CHAIN_BROKEN)
            self._last_frame_digest = getattr(frame, "signature_or_digest_raw", None) or getattr(frame, "frame_hmac", None)

        if evidence is not None:
            w = EVIDENCE_MOMENTUM_WEIGHT.get(evidence, 0.0)
        elif evidence_weight is not None:
            w = float(evidence_weight)
        else:
            w = 0.0

        self.momentum = float((self.gamma * self.momentum) + w)
        self._current_momentum = self.momentum
        self._step_count += 1
        return self.momentum

    def compute_momentum(self, evidence_stream: List[EnlgEvidenceClass]) -> float:
        k = len(evidence_stream)
        if k == 0:
            return 0.0
        m = 0.0
        for i, ev in enumerate(evidence_stream):
            w = EVIDENCE_MOMENTUM_WEIGHT.get(ev, 0.0)
            m += (self.gamma ** (k - 1 - i)) * w
        self.momentum = float(m)
        self._current_momentum = float(m)
        return float(m)

    def evaluate_proposal_admissibility(
        self,
        trunk_evidence: Optional[List[EnlgEvidenceClass]],
        proposal_evidence: EnlgEvidenceClass,
        proposal_role: EnlgRole,
        is_hp_valid: bool = True,
    ) -> Tuple[bool, str]:
        if not is_hp_valid:
            raise ContractViolation("Noether-Krystal invariant breach: HP must be 6")

        trunk_momentum = self.compute_momentum(trunk_evidence) if trunk_evidence else self.current_momentum
        prop_weight = EVIDENCE_MOMENTUM_WEIGHT.get(proposal_evidence, 0.0)

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
    """Hlavný regulátor vyhodnocujúci Epistemický Akčný Integrál S[pi]."""

    def __init__(
        self,
        calorimeter: Optional[EpistemicCalorimeter] = None,
        conflict_ring: Optional[BifurcatedConflictRing] = None,
        vblank_injector: Optional[VBlankExperimentInjector] = None,
        inertia_damper: Optional[CausalInertiaDamper] = None,
        artifact_gate: Optional[ArtifactGate] = None,
        alpha: float = 1.0,
        beta: float = 2.0,
        gamma_weight: float = 0.5,
    ) -> None:
        if VITAL_MAX_HP != 6:
            raise ContractViolation("VITAL_MAX_HP must be 6 in PleaEngine")

        self.alpha = alpha
        self.beta = beta
        self.gamma_weight = gamma_weight
        self.vital_hp = VITAL_MAX_HP
        self.calorimeter = calorimeter or EpistemicCalorimeter()
        self.conflict_ring = conflict_ring or BifurcatedConflictRing()
        self.vblank_injector = vblank_injector or VBlankExperimentInjector()
        self.inertia_damper = inertia_damper or CausalInertiaDamper()
        self.artifact_gate = artifact_gate or ArtifactGate()

    def evaluate_action_integral(
        self,
        uma_cost: float,
        topo_conflict: float | bool = 0.0,
        causal_inertia: float = 0.0,
        vital_hp: int = VITAL_MAX_HP,
        alpha: Optional[float] = None,
        beta: Optional[float] = None,
        gamma: Optional[float] = None,
        *,
        has_topological_conflict: Optional[bool] = None,
    ) -> float:
        """Computes S[pi] = alpha * UMA_Cost + beta * TopoConflict - gamma * CausalInertia."""
        if vital_hp != VITAL_MAX_HP:
            raise ContractViolation("Noether-Krystal invariant violated in PleaEngine")

        a = alpha if alpha is not None else self.alpha
        b = beta if beta is not None else self.beta
        g = gamma if gamma is not None else self.gamma_weight

        tc = has_topological_conflict if has_topological_conflict is not None else topo_conflict
        conflict_term = 1.0 if isinstance(tc, bool) and tc else float(tc or 0.0)
        s_val = (a * uma_cost) + (b * conflict_term) - (g * causal_inertia)
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
                status_code="REJECTED_PARASITIC_THERMAL_DECOUPLING",
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

    def apply_transition_guarded(self, current_state: Dict[str, Any], candidate_action: Dict[str, Any]) -> Dict[str, Any]:
        """Ústavná zmena stavu: not Valid(Action) => S_{t+1} == S_t (Delta S == 0)."""
        if not candidate_action.get("cap_token") or candidate_action.get("status") == "UNAUTHORIZED":
            return current_state
        new_state = dict(current_state)
        new_state.update(candidate_action.get("delta", {}))
        return new_state
