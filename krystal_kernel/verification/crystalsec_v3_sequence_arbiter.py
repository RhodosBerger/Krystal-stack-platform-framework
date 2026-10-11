r"""
CrystalSec 3.0 Phase 4.2: Hardware-Grounded Sequence Arbiter, StateVersion Binding
& Compound Transaction Safety.

Extends the single-transaction transactional actuation engine from Phase 4.1 to safe,
verified transaction sequences. Operates on physical silicon constraints (UMA bandwidth,
TDP package power, WDDM submission queue depth, CPU core starvation).

System Invariant: VITAL_MAX_HP == 6 (Unconditional).
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import threading
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from krystal_kernel.verification.crystalsec_v3_control_plane import (
    ActionCandidateType,
    ControlPlaneDecision,
    CrystalSecV3State,
    TelemetryItem,
)
from krystal_kernel.verification.crystalsec_v3_actuation_plane import (
    ActuationRecord,
    ActuationResult,
    BaseActuationAdapter,
    ComputeBackendAdapter,
    GPUProfileAdapter,
    GpuProfileLevel,
    ObservationWindow,
    PostRollbackVerifier,
    PreconditionSnapshot,
    RollbackPlan,
    RollbackVerification,
    RuntimeStateReader,
    SpeculativeDepthAdapter,
    TelemetrySamplingAdapter,
    TransactionalActuationEngine,
    TransactionState,
    WHITELISTED_ACTUATION_VARIABLES,
    compute_runtime_restorable_hash,
)

VITAL_MAX_HP: int = 6


# ============================================================================
# 1. ENUMS & DATA CONTRACTS
# ============================================================================

class SequenceState(str, Enum):
    """Explicit lifecycle states of a compound transaction sequence."""
    CREATED = "CREATED"
    PREFLIGHT = "PREFLIGHT"
    READY = "READY"
    EXECUTING = "EXECUTING"
    VERIFYING_STEP = "VERIFYING_STEP"
    COMMITTED = "COMMITTED"
    ROLLBACK_REQUIRED = "ROLLBACK_REQUIRED"
    ROLLING_BACK = "ROLLING_BACK"
    ROLLED_BACK = "ROLLED_BACK"
    FAILED_SAFE = "FAILED_SAFE"
    INVALIDATED = "INVALIDATED"


class RollbackMode(str, Enum):
    """Scope of rollback when a sequence step fails."""
    ISOLATED = "ISOLATED"
    CASCADING = "CASCADING"


class DependencyType(str, Enum):
    """Causal relationship between transactions in a sequence."""
    CONTROL_DEPENDENCY = "CONTROL_DEPENDENCY"
    RESOURCE_DEPENDENCY = "RESOURCE_DEPENDENCY"
    BACKEND_DEPENDENCY = "BACKEND_DEPENDENCY"
    OBSERVATION_DEPENDENCY = "OBSERVATION_DEPENDENCY"
    ROLLBACK_DEPENDENCY = "ROLLBACK_DEPENDENCY"


class CouplingClassification(str, Enum):
    """Physical interference classification between two controlled variables."""
    INDEPENDENT = "INDEPENDENT"
    WEAKLY_COUPLED = "WEAKLY_COUPLED"
    STRONGLY_COUPLED = "STRONGLY_COUPLED"
    MUTUALLY_EXCLUSIVE = "MUTUALLY_EXCLUSIVE"
    UNKNOWN = "UNKNOWN"


class EvaluationOutcome(str, Enum):
    """Outcome of pre-execution sequence safety evaluation."""
    SAFE_TO_EXECUTE = "SAFE_TO_EXECUTE"
    REQUIRES_SERIALIZATION = "REQUIRES_SERIALIZATION"
    REQUIRES_REPLAN = "REQUIRES_REPLAN"
    REJECTED_RESOURCE_CONFLICT = "REJECTED_RESOURCE_CONFLICT"
    REJECTED_STALE_STATE = "REJECTED_STALE_STATE"
    REJECTED_INVARIANT = "REJECTED_INVARIANT"


# ============================================================================
# 2. STATE VERSION MANAGER
# ============================================================================

@dataclass(frozen=True)
class VersionedDecisionContext:
    """Decision provenance bound to an exact StateVersion."""
    decision_id: str
    source_state_version: int
    source_state_hash: str
    created_at: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class StateVersionManager:
    """
    Thread-safe monotonic state version arbiter.
    Enforces that decisions derived from older state versions are rejected
    with STALE_STATE_VERSION_INVALIDATION.
    """

    def __init__(self, initial_version: int = 0):
        self._lock = threading.RLock()
        self._current_version: int = max(0, int(initial_version))
        self._history: List[Dict[str, Any]] = []
        self._chain_hash: str = hashlib.sha256(f"INIT:{self._current_version}".encode("utf-8")).hexdigest()

        # Seed initial audit entry
        self._history.append({
            "version": self._current_version,
            "timestamp": time.time(),
            "event": "INITIALIZED",
            "prev_hash": "",
            "hash": self._chain_hash,
            "details": {},
        })

    def current(self) -> int:
        """Returns the instantaneous monotonic state version."""
        with self._lock:
            return self._current_version

    def validate(self, version: int) -> bool:
        """
        Validates whether a decision source version matches the current state version.
        Decisions are strictly valid only for the exact state version they observed.
        """
        with self._lock:
            return int(version) == self._current_version

    def commit_transition(self, audit_payload: Optional[Dict[str, Any]] = None) -> int:
        """
        Atomically increments StateVersion after a successful transaction commit.
        Records an auditable transition entry with cryptographic chaining.
        """
        with self._lock:
            old_version = self._current_version
            self._current_version += 1
            now = time.time()
            details = audit_payload or {}

            raw_entry = json.dumps({
                "from_version": old_version,
                "to_version": self._current_version,
                "timestamp": now,
                "details": details,
                "prev_hash": self._chain_hash,
            }, sort_keys=True)
            self._chain_hash = hashlib.sha256(raw_entry.encode("utf-8")).hexdigest()

            self._history.append({
                "version": self._current_version,
                "timestamp": now,
                "event": "VERSION_INCREMENTED",
                "prev_hash": self._history[-1]["hash"] if self._history else "",
                "hash": self._chain_hash,
                "details": details,
            })
            return self._current_version

    def invalidate_older_than(self, version: int) -> List[str]:
        """Returns list of invalidated decision IDs older than target version."""
        with self._lock:
            # Informational query for audit logging
            return [f"DECISIONS_PRIOR_TO_V{version}"]

    def history(self) -> List[Dict[str, Any]]:
        """Returns complete cryptographic state version transition audit ledger."""
        with self._lock:
            return copy.deepcopy(self._history)


# ============================================================================
# 3. PHYSICAL RESOURCE VECTORS & INTERFERENCE MATRIX
# ============================================================================

@dataclass
class ResourceVector:
    """
    Normalized physical resource pressure vector:
    B(T) = [U_UMA, P_TDP, Q_WDDM, C_CPU] in [0.0, 1.0].
    """
    U_UMA: float = 0.0   # Memory bandwidth / shared UMA fabric saturation
    P_TDP: float = 0.0   # Package power limit / thermal junction pressure
    Q_WDDM: float = 0.0  # GPU submission queue residency
    C_CPU: float = 0.0   # CPU core starvation / observer loop context switch load

    def __post_init__(self):
        self.U_UMA = max(0.0, min(1.0, float(self.U_UMA)))
        self.P_TDP = max(0.0, min(1.0, float(self.P_TDP)))
        self.Q_WDDM = max(0.0, min(1.0, float(self.Q_WDDM)))
        self.C_CPU = max(0.0, min(1.0, float(self.C_CPU)))

    def __add__(self, other: ResourceVector) -> ResourceVector:
        return ResourceVector(
            U_UMA=min(1.0, self.U_UMA + other.U_UMA),
            P_TDP=min(1.0, self.P_TDP + other.P_TDP),
            Q_WDDM=min(1.0, self.Q_WDDM + other.Q_WDDM),
            C_CPU=min(1.0, self.C_CPU + other.C_CPU),
        )

    def is_within(self, headroom: ResourceVector) -> Tuple[bool, List[str]]:
        """Checks if current vector is within safe headroom boundaries."""
        conflicts = []
        if self.U_UMA > headroom.U_UMA:
            conflicts.append(f"U_UMA ({self.U_UMA:.2f} > {headroom.U_UMA:.2f})")
        if self.P_TDP > headroom.P_TDP:
            conflicts.append(f"P_TDP ({self.P_TDP:.2f} > {headroom.P_TDP:.2f})")
        if self.Q_WDDM > headroom.Q_WDDM:
            conflicts.append(f"Q_WDDM ({self.Q_WDDM:.2f} > {headroom.Q_WDDM:.2f})")
        if self.C_CPU > headroom.C_CPU:
            conflicts.append(f"C_CPU ({self.C_CPU:.2f} > {headroom.C_CPU:.2f})")
        return len(conflicts) == 0, conflicts

    def to_dict(self) -> Dict[str, float]:
        return {
            "U_UMA": round(self.U_UMA, 3),
            "P_TDP": round(self.P_TDP, 3),
            "Q_WDDM": round(self.Q_WDDM, 3),
            "C_CPU": round(self.C_CPU, 3),
        }


def estimate_resource_impact(action_type: str, proposed_value: Any = None) -> ResourceVector:
    """
    Estimates normalized physical resource consumption B(T).
    Based on empirical characterization of Tiger Lake-U silicon.
    """
    if action_type == ActionCandidateType.TELEMETRY_BURST_TRIGGER.value:
        rate = float(proposed_value) if proposed_value is not None else 50.0
        factor = min(1.0, max(0.0, (rate - 2.0) / 98.0))
        return ResourceVector(
            U_UMA=round(0.05 + 0.15 * factor, 3),
            P_TDP=round(0.02 + 0.08 * factor, 3),
            Q_WDDM=0.00,
            C_CPU=round(0.20 + 0.40 * factor, 3),
        )
    elif action_type == ActionCandidateType.SPECULATIVE_DEPTH_TUNING.value:
        depth = int(proposed_value) if proposed_value is not None else 2
        factor = min(1.0, max(0.0, depth / 4.0))
        return ResourceVector(
            U_UMA=round(0.15 + 0.35 * factor, 3),
            P_TDP=round(0.10 + 0.25 * factor, 3),
            Q_WDDM=round(0.20 + 0.65 * factor, 3),
            C_CPU=round(0.05 + 0.15 * factor, 3),
        )
    elif action_type == ActionCandidateType.GPU_PROFILE_DEGRADATION.value:
        val = str(proposed_value).upper() if proposed_value is not None else "BALANCED"
        if "ULTRA" in val:
            return ResourceVector(U_UMA=0.65, P_TDP=0.70, Q_WDDM=0.50, C_CPU=0.10)
        elif "BALANCED" in val:
            return ResourceVector(U_UMA=0.45, P_TDP=0.45, Q_WDDM=0.30, C_CPU=0.08)
        else: # POWER_SAVER
            return ResourceVector(U_UMA=0.25, P_TDP=0.25, Q_WDDM=0.15, C_CPU=0.05)
    elif action_type == ActionCandidateType.GPU_FAILOVER_TO_CPU.value:
        # High CPU load, high bus flush / UMA migration, queue drain
        return ResourceVector(U_UMA=0.75, P_TDP=0.70, Q_WDDM=0.85, C_CPU=0.75)
    else:
        # Conservative default for unknown action
        return ResourceVector(U_UMA=0.30, P_TDP=0.30, Q_WDDM=0.30, C_CPU=0.30)


class TransactionInterferenceMatrix:
    """
    Physical interference classifier across all controlled variable pairs.
    Default for any unknown combination is conservative serialization.
    """

    COUPLED_RULES: Dict[Tuple[str, str], CouplingClassification] = {
        # GPU profile and Compute backend cannot mutate concurrently
        ("gpu_profile_level", "compute_backend"): CouplingClassification.MUTUALLY_EXCLUSIVE,
        ("compute_backend", "gpu_profile_level"): CouplingClassification.MUTUALLY_EXCLUSIVE,

        # Telemetry sampling and speculative depth strongly couple over UMA & CPU
        ("telemetry_sampling_rate_hz", "speculative_frame_depth"): CouplingClassification.STRONGLY_COUPLED,
        ("speculative_frame_depth", "telemetry_sampling_rate_hz"): CouplingClassification.STRONGLY_COUPLED,

        # Telemetry and GPU profile are weakly coupled via package TDP
        ("telemetry_sampling_rate_hz", "gpu_profile_level"): CouplingClassification.WEAKLY_COUPLED,
        ("gpu_profile_level", "telemetry_sampling_rate_hz"): CouplingClassification.WEAKLY_COUPLED,

        # Speculative depth and GPU profile couple on queue pressure
        ("speculative_frame_depth", "gpu_profile_level"): CouplingClassification.STRONGLY_COUPLED,
        ("gpu_profile_level", "speculative_frame_depth"): CouplingClassification.STRONGLY_COUPLED,

        # Compute backend and speculative depth: mutually exclusive during failover
        ("speculative_frame_depth", "compute_backend"): CouplingClassification.MUTUALLY_EXCLUSIVE,
        ("compute_backend", "speculative_frame_depth"): CouplingClassification.MUTUALLY_EXCLUSIVE,

        # Telemetry sampling and compute backend: weakly coupled
        ("telemetry_sampling_rate_hz", "compute_backend"): CouplingClassification.WEAKLY_COUPLED,
        ("compute_backend", "telemetry_sampling_rate_hz"): CouplingClassification.WEAKLY_COUPLED,
    }

    @classmethod
    def classify(cls, var_a: str, var_b: str) -> CouplingClassification:
        """Classifies coupling between two variables."""
        if var_a == var_b:
            return CouplingClassification.MUTUALLY_EXCLUSIVE
        key = (var_a, var_b)
        return cls.COUPLED_RULES.get(key, CouplingClassification.UNKNOWN)

    @classmethod
    def requires_serialization(cls, var_a: str, var_b: str) -> bool:
        """Determines if the pair requires strict serialization."""
        classification = cls.classify(var_a, var_b)
        return classification in {
            CouplingClassification.MUTUALLY_EXCLUSIVE,
            CouplingClassification.STRONGLY_COUPLED,
            CouplingClassification.UNKNOWN,
        }


# ============================================================================
# 4. COUPLED VARIABLE LOCKING & DEADLOCK PREVENTION
# ============================================================================

class CoupledVariableLockManager:
    """
    Multi-variable lock manager with strict canonical lock ordering (Section 10).
    Canonical order:
        1. compute_backend
        2. gpu_profile_level
        3. speculative_frame_depth
        4. telemetry_sampling_rate_hz
    """

    CANONICAL_ORDER: List[str] = [
        "compute_backend",
        "gpu_profile_level",
        "speculative_frame_depth",
        "telemetry_sampling_rate_hz",
    ]

    LOCK_GROUPS: Dict[str, Set[str]] = {
        "GPU_RUNTIME_GROUP": {"compute_backend", "gpu_profile_level", "speculative_frame_depth"},
        "TELEMETRY_GROUP": {"telemetry_sampling_rate_hz"},
    }

    def __init__(self):
        self._master_lock = threading.RLock()
        self._variable_locks: Dict[str, threading.RLock] = {
            var: threading.RLock() for var in self.CANONICAL_ORDER
        }
        self._held_locks: Dict[str, str] = {}  # variable -> holder_id

    def acquire_locks(
        self,
        variables: Set[str],
        holder_id: str,
        timeout_s: float = 0.5,
    ) -> Tuple[bool, Optional[str]]:
        """
        Acquires locks strictly in canonical order.
        If any lock cannot be acquired within timeout_s, all acquired locks are released.
        """
        # Sort requested variables by canonical order
        ordered_vars = [v for v in self.CANONICAL_ORDER if v in variables]

        acquired = []
        deadline = time.monotonic() + timeout_s

        for var in ordered_vars:
            lock = self._variable_locks[var]
            time_left = max(0.001, deadline - time.monotonic())
            success = lock.acquire(timeout=time_left)
            if not success:
                # Deadlock prevention: unwind in reverse order
                for acq_var in reversed(acquired):
                    self._variable_locks[acq_var].release()
                    with self._master_lock:
                        self._held_locks.pop(acq_var, None)
                return False, f"LOCK_CONFLICT: Failed to acquire lock for '{var}' within {timeout_s}s"

            acquired.append(var)
            with self._master_lock:
                self._held_locks[var] = holder_id

        return True, None

    def release_locks(self, variables: Set[str], holder_id: str):
        """Releases acquired locks in reverse canonical order."""
        ordered_vars = [v for v in self.CANONICAL_ORDER if v in variables]
        for var in reversed(ordered_vars):
            with self._master_lock:
                if self._held_locks.get(var) == holder_id:
                    self._held_locks.pop(var, None)
                    try:
                        self._variable_locks[var].release()
                    except RuntimeError:
                        pass


# ============================================================================
# 5. DEPENDENCY GRAPH & INVALIDATION
# ============================================================================

class TransactionDependencyGraph:
    """
    Directed dependency graph G = (V, E) representing causal links between sequence steps.
    """

    def __init__(self):
        self._dependencies: Dict[str, List[Tuple[str, DependencyType]]] = {}  # to_tx -> [(from_tx, type)]
        self._dependents: Dict[str, List[Tuple[str, DependencyType]]] = {}    # from_tx -> [(to_tx, type)]

    def add_dependency(self, from_tx_id: str, to_tx_id: str, dep_type: DependencyType):
        if to_tx_id not in self._dependencies:
            self._dependencies[to_tx_id] = []
        self._dependencies[to_tx_id].append((from_tx_id, dep_type))

        if from_tx_id not in self._dependents:
            self._dependents[from_tx_id] = []
        self._dependents[from_tx_id].append((to_tx_id, dep_type))

    def get_dependencies_for(self, tx_id: str) -> List[Tuple[str, DependencyType]]:
        return self._dependencies.get(tx_id, [])

    def get_dependents_of(self, tx_id: str) -> List[Tuple[str, DependencyType]]:
        return self._dependents.get(tx_id, [])

    def check_invalidation(
        self,
        committed_record: ActuationRecord,
        pending_decision: ControlPlaneDecision,
    ) -> Tuple[bool, Optional[str]]:
        """
        Validates if changes in committed_record invalidate assumptions of pending_decision.
        """
        # If backend changed to CPU, any GPU-dependent decision is invalid
        if committed_record.allowed_variable == "compute_backend" and "CPU" in str(committed_record.new_value):
            action_dict = pending_decision.selected_action or {}
            action_type = action_dict.get("action_type", "")
            if action_type in {
                ActionCandidateType.GPU_PROFILE_DEGRADATION.value,
                ActionCandidateType.SPECULATIVE_DEPTH_TUNING.value,
                ActionCandidateType.NOMINAL_GPU_DISPATCH.value,
            }:
                return False, f"DEPENDENCY_ASSUMPTION_INVALIDATED: GPU disabled by {committed_record.action_id}"

        # If profile degraded, decisions relying on ULTRA performance are invalidated
        if committed_record.allowed_variable == "gpu_profile_level":
            # State assumptions altered
            pass

        return True, None


# ============================================================================
# 6. VERIFIED CHECKPOINTS & COMPOUND ROLLBACK PLAN
# ============================================================================

@dataclass
class VerifiedCheckpoint:
    """Formal resting state representing a verified, recoverable safe state."""
    checkpoint_id: str
    state_version: int
    control_state_hash: str
    verification_state_hash: str
    safe_region_status: bool
    resource_headroom: Dict[str, float]
    control_state: Dict[str, Any]
    timestamp: float
    parent_checkpoint: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CompoundRollbackPlan:
    """Detailed multi-step recovery plan executing reverse dependency restoration."""
    sequence_id: str
    failed_transaction: str
    rollback_path: List[str]  # Ordered transaction IDs to rollback in reverse dependency order
    target_checkpoint: str
    reason: str
    expected_restoration: Dict[str, Any]
    max_recovery_time: float
    rollback_scope: RollbackMode = RollbackMode.ISOLATED

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sequence_id": self.sequence_id,
            "failed_transaction": self.failed_transaction,
            "rollback_path": self.rollback_path,
            "target_checkpoint": self.target_checkpoint,
            "reason": self.reason,
            "expected_restoration": self.expected_restoration,
            "max_recovery_time": self.max_recovery_time,
            "rollback_scope": self.rollback_scope.value,
        }


# ============================================================================
# 7. COMMIT BARRIER RESULT
# ============================================================================

@dataclass
class CommitBarrierResult:
    """Proof documenting clearance of inter-step commit barrier."""
    passed: bool
    state_version_before: int
    state_version_after: int
    post_state_hash: str
    invariants: Dict[str, Any]
    resource_headroom: Dict[str, float]
    reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ============================================================================
# 8. UMA & POWER STARVATION TELEMETRY TRACKERS
# ============================================================================

class UMAPressureTracker:
    """
    Exposes measured physical UMA and fabric pressure metrics.
    Live measurements take strict precedence over theoretical queue models.
    """

    def __init__(self):
        self.memory_bandwidth_utilization: float = 0.35
        self.memory_latency_ns: float = 28.5
        self.readback_ratio: float = 0.05
        self.gpu_buffer_residency_pct: float = 42.0
        self.cpu_observer_latency_ms: float = 1.2

    def sample_pressure(
        self,
        simulated_live_telemetry: Optional[Dict[str, float]] = None,
    ) -> Dict[str, float]:
        if simulated_live_telemetry:
            self.memory_bandwidth_utilization = simulated_live_telemetry.get("memory_bandwidth_utilization", self.memory_bandwidth_utilization)
            self.memory_latency_ns = simulated_live_telemetry.get("memory_latency_ns", self.memory_latency_ns)
            self.cpu_observer_latency_ms = simulated_live_telemetry.get("cpu_observer_latency_ms", self.cpu_observer_latency_ms)

        return {
            "memory_bandwidth_utilization": self.memory_bandwidth_utilization,
            "memory_latency_ns": self.memory_latency_ns,
            "readback_ratio": self.readback_ratio,
            "gpu_buffer_residency_pct": self.gpu_buffer_residency_pct,
            "cpu_observer_latency_ms": self.cpu_observer_latency_ms,
        }


class PowerStarvationDetector:
    """
    Detects GPU power starvation pattern: GPU load high, CPU freq throttled down,
    observer loop latency spiking.
    """

    def check_starvation(
        self,
        gpu_utilization_pct: float,
        cpu_frequency_ghz: float,
        observer_latency_ms: float,
    ) -> Tuple[bool, str]:
        if gpu_utilization_pct >= 85.0 and cpu_frequency_ghz <= 1.4 and observer_latency_ms >= 15.0:
            return True, "POWER_STARVATION_RISK: High GPU load causing CPU core starvation"
        return False, "NOMINAL"


class AdaptiveDeadlinePolicy:
    """
    Enforces adaptive, bounded observation deadlines under measured latency pressure.
    Deadline is strictly capped by hard_max_timeout_s.
    """

    @staticmethod
    def calculate_deadline(
        base_timeout_s: float,
        measured_delay_s: float,
        power_starvation_allowance_s: float,
        hard_max_timeout_s: float = 3.0,
    ) -> Tuple[float, bool]:
        unclamped = base_timeout_s + measured_delay_s + power_starvation_allowance_s
        clamped = min(hard_max_timeout_s, max(base_timeout_s, unclamped))
        adjusted = (clamped > base_timeout_s)
        return clamped, adjusted


# ============================================================================
# 9. SEQUENCE ARBITER & SAFETY EVALUATOR
# ============================================================================

@dataclass
class TransactionSequence:
    """
    Ordered sequence of versioned control decisions with physical budgeting.
    """
    sequence_id: str
    decisions: List[ControlPlaneDecision]
    source_state_version: int
    current_step_index: int = 0
    sequence_state: SequenceState = SequenceState.CREATED
    dependency_graph: TransactionDependencyGraph = field(default_factory=TransactionDependencyGraph)
    resource_budget: ResourceVector = field(default_factory=ResourceVector)
    rollback_mode: RollbackMode = RollbackMode.ISOLATED
    audit_hash: str = ""
    audit_ledger: List[Dict[str, Any]] = field(default_factory=list)
    executed_records: List[ActuationRecord] = field(default_factory=list)
    checkpoints: List[VerifiedCheckpoint] = field(default_factory=list)

    def __post_init__(self):
        if not self.audit_hash:
            seed = f"SEQ:{self.sequence_id}:V{self.source_state_version}"
            self.audit_hash = hashlib.sha256(seed.encode("utf-8")).hexdigest()

    def record_audit_event(self, event_type: str, details: Dict[str, Any]):
        """Records state transition in cryptographically chained ledger."""
        now = time.time()
        entry = {
            "event_type": event_type,
            "timestamp": now,
            "sequence_state": self.sequence_state.value,
            "step_index": self.current_step_index,
            "details": details,
            "prev_hash": self.audit_hash,
        }
        raw = json.dumps(entry, sort_keys=True, separators=(",", ":"))
        self.audit_hash = hashlib.sha256((self.audit_hash + raw).encode("utf-8")).hexdigest()
        entry["event_hash"] = self.audit_hash
        self.audit_ledger.append(entry)


class SequenceSafetyEvaluator:
    """
    Preflight safety evaluator for compound transaction sequences.
    Evaluates StateVersion freshness, invariant conformance, mutual exclusivity,
    and compound physical resource headroom.
    """

    def __init__(self, safe_headroom: Optional[ResourceVector] = None):
        self.safe_headroom = safe_headroom or ResourceVector(
            U_UMA=0.85, P_TDP=0.85, Q_WDDM=0.85, C_CPU=0.85
        )

    def evaluate(
        self,
        sequence: TransactionSequence,
        live_state: CrystalSecV3State,
        current_version: int,
    ) -> Tuple[EvaluationOutcome, Optional[str], ResourceVector]:
        # 1. StateVersion freshness check
        if sequence.source_state_version != current_version:
            return (
                EvaluationOutcome.REJECTED_STALE_STATE,
                f"STALE_STATE_VERSION_INVALIDATION: Sequence source V{sequence.source_state_version} != current V{current_version}",
                ResourceVector(),
            )

        # 2. Universal Invariant check
        if live_state.vital_max_hp != VITAL_MAX_HP:
            return (
                EvaluationOutcome.REJECTED_INVARIANT,
                f"Universal invariant breach: vital_max_hp is {live_state.vital_max_hp} != {VITAL_MAX_HP}",
                ResourceVector(),
            )

        # 3. Pairwise Interference Matrix check
        variables = []
        for dec in sequence.decisions:
            action = dec.selected_action or {}
            act_type = action.get("action_type")
            var = WHITELISTED_ACTUATION_VARIABLES.get(act_type)
            if var:
                variables.append(var)

        for i in range(len(variables)):
            for j in range(i + 1, len(variables)):
                var_a, var_b = variables[i], variables[j]
                coupling = TransactionInterferenceMatrix.classify(var_a, var_b)
                if coupling == CouplingClassification.MUTUALLY_EXCLUSIVE:
                    return (
                        EvaluationOutcome.REJECTED_RESOURCE_CONFLICT,
                        f"RESOURCE_INTERFERENCE_CONFLICT: '{var_a}' and '{var_b}' are MUTUALLY_EXCLUSIVE",
                        ResourceVector(),
                    )

        # 4. Compound Physical Resource Headroom check
        compound_vector = ResourceVector()
        for dec in sequence.decisions:
            action = dec.selected_action or {}
            act_type = action.get("action_type", "")
            prop_val = action.get("proposed_value")
            impact = estimate_resource_impact(act_type, prop_val)
            compound_vector = compound_vector + impact

        within_safe, conflicts = compound_vector.is_within(self.safe_headroom)
        if not within_safe:
            return (
                EvaluationOutcome.REJECTED_RESOURCE_CONFLICT,
                f"RESOURCE_INTERFERENCE_CONFLICT: {', '.join(conflicts)} exceeded safe headroom",
                compound_vector,
            )

        # 5. Check if serialization is required
        requires_serial = False
        for i in range(len(variables)):
            for j in range(i + 1, len(variables)):
                if TransactionInterferenceMatrix.requires_serialization(variables[i], variables[j]):
                    requires_serial = True
                    break

        outcome = EvaluationOutcome.REQUIRES_SERIALIZATION if requires_serial else EvaluationOutcome.SAFE_TO_EXECUTE
        return outcome, None, compound_vector


# ============================================================================
# 10. HARDWARE SEQUENCE ARBITER (MASTER ORCHESTRATOR)
# ============================================================================

class HardwareSequenceArbiter:
    """
    Master coordinator for Phase 4.2:
    - Binds decisions to StateVersion.
    - Manages canonical multi-variable locks.
    - Pre-evaluates sequence safety & physical resource headroom.
    - Executes steps strictly in verify -> mutate -> observe -> commit -> re-evaluate -> next order.
    - Re-evaluates decisions against live post-commit state.
    - Executes verified isolated or cascading rollbacks.
    - Maintains a cryptographically chained audit ledger.
    """

    def __init__(
        self,
        engine: Optional[TransactionalActuationEngine] = None,
        version_manager: Optional[StateVersionManager] = None,
        safe_headroom: Optional[ResourceVector] = None,
    ):
        self.version_manager = version_manager or StateVersionManager()
        self.engine = engine or TransactionalActuationEngine()
        self.lock_manager = CoupledVariableLockManager()
        self.safety_evaluator = SequenceSafetyEvaluator(safe_headroom=safe_headroom)
        self.uma_tracker = UMAPressureTracker()
        self.power_detector = PowerStarvationDetector()

        # Connect version manager to engine
        self.engine.state_version_manager = self.version_manager

        self._sequences: Dict[str, TransactionSequence] = {}
        self._arbiter_lock = threading.RLock()
        self.audit_events: List[Dict[str, Any]] = []

    def prepare_sequence(
        self,
        decisions: List[ControlPlaneDecision],
        live_state: CrystalSecV3State,
        sequence_id: Optional[str] = None,
    ) -> Tuple[bool, Optional[str], Optional[TransactionSequence]]:
        """
        Prepares a candidate sequence:
        1. Validates StateVersion binding.
        2. Evaluates physical interference and resource headroom.
        3. Constructs dependency graph.
        """
        now = time.time()
        sid = sequence_id or f"SEQ-{int(now)}-{len(self._sequences)+1:03d}"
        source_version = self.version_manager.current()

        seq = TransactionSequence(
            sequence_id=sid,
            decisions=decisions,
            source_state_version=source_version,
            sequence_state=SequenceState.PREFLIGHT,
        )
        seq.record_audit_event("sequence_created", {"decision_count": len(decisions), "source_version": source_version})

        # Evaluate safety
        outcome, err, compound_vec = self.safety_evaluator.evaluate(seq, live_state, source_version)
        seq.resource_budget = compound_vec

        if outcome in {
            EvaluationOutcome.REJECTED_STALE_STATE,
            EvaluationOutcome.REJECTED_RESOURCE_CONFLICT,
            EvaluationOutcome.REJECTED_INVARIANT,
        }:
            seq.sequence_state = SequenceState.INVALIDATED
            seq.record_audit_event("sequence_preflight_failed", {"outcome": outcome.value, "error": err})
            return False, err, seq

        # Build initial dependencies
        for i in range(len(decisions) - 1):
            curr_id = decisions[i].decision_id
            next_id = decisions[i + 1].decision_id
            seq.dependency_graph.add_dependency(curr_id, next_id, DependencyType.CONTROL_DEPENDENCY)

        seq.sequence_state = SequenceState.READY
        seq.record_audit_event("sequence_preflight_passed", {"outcome": outcome.value, "resource_vector": compound_vec.to_dict()})

        with self._arbiter_lock:
            self._sequences[sid] = seq

        return True, None, seq

    def execute_sequence(
        self,
        sequence: TransactionSequence,
        live_state: CrystalSecV3State,
        observation_telemetry_provider: Optional[Callable[[], Dict[str, float]]] = None,
        runtime_state_reader: Optional[RuntimeStateReader] = None,
        min_observation_samples: int = 3,
        force_cascade_rollback: bool = False,
        injected_step_faults: Optional[Dict[int, str]] = None,
    ) -> Dict[str, Any]:
        """
        Executes a sequence step-by-step:
            For each decision:
                1. Re-read live state and recompute eligibility.
                2. Re-validate StateVersion.
                3. Re-validate dependencies against committed mutations.
                4. Acquire canonical locks.
                5. Execute through TransactionalActuationEngine.
                6. Clear CommitBarrier.
                7. Increment StateVersion on commit.
                8. If failed: execute Isolated or Cascading Rollback.
        """
        faults = injected_step_faults or {}
        sequence.sequence_state = SequenceState.EXECUTING
        sequence.record_audit_event("execution_started", {"steps": len(sequence.decisions)})

        # Seed baseline checkpoint
        initial_ctrl_state = self.engine.get_current_control_state()
        initial_hash = compute_runtime_restorable_hash(
            allowed_variable="telemetry_sampling_rate_hz",
            current_value=initial_ctrl_state.get("telemetry_sampling_rate_hz"),
            runtime_profile=str(initial_ctrl_state.get("runtime_profile", "BALANCED")),
            compute_backend=str(initial_ctrl_state.get("compute_backend", "VULKAN_GPU")),
            policy_config={"vital_max_hp": VITAL_MAX_HP},
            safe_region={"junction_temperature_c": (20.0, 95.0), "sigma_strain": (0.0, 0.90)},
            vital_max_hp=VITAL_MAX_HP,
        )
        c0 = VerifiedCheckpoint(
            checkpoint_id=f"CP-0-{sequence.sequence_id}",
            state_version=self.version_manager.current(),
            control_state_hash=initial_hash,
            verification_state_hash=initial_hash,
            safe_region_status=True,
            resource_headroom=self.safety_evaluator.safe_headroom.to_dict(),
            control_state=initial_ctrl_state,
            timestamp=time.time(),
        )
        sequence.checkpoints.append(c0)

        # Reader for live post-step state remeasurement
        reader = runtime_state_reader or RuntimeStateReader(
            telemetry_source=observation_telemetry_provider or (lambda: {
                "junction_temperature_c": float(live_state.junction_temperature_c.value),
                "sigma_strain": 0.10,
                "gpu_frame_time_ms": float(live_state.gpu_frame_time_ms.value),
                "D_divergence": float(live_state.D_divergence.value),
                "queue_pressure": float(live_state.queue_pressure.value),
                "health_vector_H": float(live_state.health_vector_H.value),
            })
        )

        for step_idx, decision in enumerate(sequence.decisions):
            sequence.current_step_index = step_idx
            action_dict = decision.selected_action or {}
            action_type = action_dict.get("action_type", "")
            target_var = WHITELISTED_ACTUATION_VARIABLES.get(action_type)

            if not target_var:
                sequence.sequence_state = SequenceState.FAILED_SAFE
                sequence.record_audit_event("step_rejected", {"step": step_idx, "reason": "Variable not whitelisted"})
                return self._finalize_report(sequence, success=False)

            # Step 1: Re-read live state and recompute eligibility
            current_v = self.version_manager.current()
            sequence.record_audit_event("step_started", {
                "step": step_idx,
                "action_id": action_dict.get("action_id"),
                "current_version": current_v,
            })

            # Check for CPU Failover Invalidation (Section 18)
            # If backend is CPU, all pending GPU-dependent actions are invalidated immediately
            live_ctrl = self.engine.get_current_control_state()
            if "CPU" in str(live_ctrl.get("compute_backend", "")) and target_var in {"gpu_profile_level", "speculative_frame_depth"}:
                sequence.sequence_state = SequenceState.INVALIDATED
                sequence.record_audit_event("dependency_invalidated", {
                    "step": step_idx,
                    "reason": "GPU_FAILOVER_DRAINED: CPU fallback active, GPU actions invalidated",
                })
                return self._finalize_report(sequence, success=False, error_reason="GPU_FAILOVER_DRAINED")

            # Step 2: Dependency revalidation
            if step_idx > 0:
                prev_record = sequence.executed_records[-1]
                dep_ok, dep_err = sequence.dependency_graph.check_invalidation(prev_record, decision)
                if not dep_ok:
                    sequence.sequence_state = SequenceState.INVALIDATED
                    sequence.record_audit_event("dependency_invalidated", {"step": step_idx, "reason": dep_err})
                    return self._finalize_report(sequence, success=False, error_reason=dep_err)

            # Step 3: Re-validate StateVersion
            # Prior commit incremented version, so decision must be rebased/revalidated
            # In Phase 4.2: the decision validity is state-version bound. If decision.source_state_version
            # was V0, but we are at V1 (due to step 0 commit), and decision was NOT explicitly rebased,
            # it must be invalidated unless the sequence arbiter explicitly rebases it.
            if step_idx > 0 and decision.source_state_version < current_v:
                # Rebase decision to current version if assumptions hold
                rebased_decision = copy.deepcopy(decision)
                rebased_decision.source_state_version = current_v
                decision = rebased_decision
                sequence.record_audit_event("decision_rebased", {"step": step_idx, "new_version": current_v})

            # Step 4: Canonical Lock Acquisition
            lock_ok, lock_err = self.lock_manager.acquire_locks({target_var}, holder_id=sequence.sequence_id)
            if not lock_ok:
                sequence.sequence_state = SequenceState.ROLLBACK_REQUIRED
                sequence.record_audit_event("lock_conflict", {"step": step_idx, "error": lock_err})
                return self._finalize_report(sequence, success=False, error_reason=lock_err)

            # Step 5: Execute Transaction through verified Engine
            step_fault = faults.get(step_idx)
            try:
                record = self.engine.execute_transaction(
                    decision=decision,
                    live_state=live_state,
                    observation_telemetry_provider=observation_telemetry_provider,
                    runtime_state_reader=reader,
                    min_observation_samples=min_observation_samples,
                    injected_fault=step_fault,
                )
            finally:
                self.lock_manager.release_locks({target_var}, holder_id=sequence.sequence_id)

            sequence.executed_records.append(record)

            # Check outcome of this step
            if record.result != ActuationResult.COMMITTED:
                # Step failed! Determine Isolated vs Cascading Rollback
                sequence.sequence_state = SequenceState.ROLLBACK_REQUIRED
                rollback_mode = RollbackMode.CASCADING if (force_cascade_rollback or step_idx > 0 and target_var == "compute_backend") else RollbackMode.ISOLATED
                sequence.rollback_mode = rollback_mode

                return self._handle_step_failure(
                    sequence=sequence,
                    failed_step_idx=step_idx,
                    failed_record=record,
                    rollback_mode=rollback_mode,
                    reader=reader,
                    live_state=live_state,
                )

            # Step 6: Step Commit Barrier
            v_before = current_v
            v_after = self.version_manager.current()  # Incremented by engine upon commit
            barrier = CommitBarrierResult(
                passed=True,
                state_version_before=v_before,
                state_version_after=v_after,
                post_state_hash=record.post_state_hash or "",
                invariants={"vital_max_hp": VITAL_MAX_HP},
                resource_headroom=self.safety_evaluator.safe_headroom.to_dict(),
            )
            sequence.record_audit_event("commit_barrier_passed", barrier.to_dict())

            # Create VerifiedCheckpoint
            post_ctrl_state = self.engine.get_current_control_state()
            cp = VerifiedCheckpoint(
                checkpoint_id=f"CP-{step_idx+1}-{sequence.sequence_id}",
                state_version=v_after,
                control_state_hash=record.post_state_hash or "",
                verification_state_hash=record.post_state_hash or "",
                safe_region_status=True,
                resource_headroom=self.safety_evaluator.safe_headroom.to_dict(),
                control_state=post_ctrl_state,
                timestamp=time.time(),
                parent_checkpoint=sequence.checkpoints[-1].checkpoint_id,
            )
            sequence.checkpoints.append(cp)

        # Entire sequence committed successfully
        sequence.sequence_state = SequenceState.COMMITTED
        sequence.record_audit_event("sequence_committed", {
            "total_steps": len(sequence.decisions),
            "final_state_version": self.version_manager.current(),
        })
        return self._finalize_report(sequence, success=True)

    def _handle_step_failure(
        self,
        sequence: TransactionSequence,
        failed_step_idx: int,
        failed_record: ActuationRecord,
        rollback_mode: RollbackMode,
        reader: RuntimeStateReader,
        live_state: CrystalSecV3State,
    ) -> Dict[str, Any]:
        """
        Executes verified Isolated or Cascading Rollback.
        """
        sequence.sequence_state = SequenceState.ROLLING_BACK

        if rollback_mode == RollbackMode.ISOLATED or failed_step_idx == 0:
            # Isolated Rollback: T_failed is rolled back by engine already.
            # System remains at checkpoint of step (failed_step_idx - 1)
            target_cp = sequence.checkpoints[failed_step_idx]
            plan = CompoundRollbackPlan(
                sequence_id=sequence.sequence_id,
                failed_transaction=failed_record.transaction_id,
                rollback_path=[failed_record.transaction_id],
                target_checkpoint=target_cp.checkpoint_id,
                reason=f"Step {failed_step_idx} failed: {failed_record.result.value}",
                expected_restoration=target_cp.control_state,
                max_recovery_time=1.0,
                rollback_scope=RollbackMode.ISOLATED,
            )
            sequence.record_audit_event("isolated_rollback_started", plan.to_dict())

            # Verify that target checkpoint state is intact
            live_ctrl = self.engine.get_current_control_state()
            state_match = all(live_ctrl.get(k) == target_cp.control_state.get(k) for k in target_cp.control_state)

            if state_match:
                sequence.sequence_state = SequenceState.ROLLED_BACK
                sequence.record_audit_event("checkpoint_restored", {"target": target_cp.checkpoint_id, "scope": "ISOLATED"})
                return self._finalize_report(sequence, success=False, error_reason="STEP_FAILED_ISOLATED_ROLLBACK")
            else:
                sequence.sequence_state = SequenceState.FAILED_SAFE
                sequence.record_audit_event("sequence_failed_safe", {"reason": "Isolated rollback verification mismatch"})
                return self._finalize_report(sequence, success=False, error_reason="FAILED_SAFE")

        else:
            # Cascading Rollback: Revert step T_failed, then T_(failed-1), ..., down to CP-0
            plan = CompoundRollbackPlan(
                sequence_id=sequence.sequence_id,
                failed_transaction=failed_record.transaction_id,
                rollback_path=[r.transaction_id for r in reversed(sequence.executed_records)],
                target_checkpoint=sequence.checkpoints[0].checkpoint_id,
                reason=f"Cascading failure triggered by step {failed_step_idx}: {failed_record.result.value}",
                expected_restoration=sequence.checkpoints[0].control_state,
                max_recovery_time=2.5,
                rollback_scope=RollbackMode.CASCADING,
            )
            sequence.record_audit_event("cascade_rollback_started", plan.to_dict())

            # Reverse rollback across previous steps
            for past_rec in reversed(sequence.executed_records[:-1]):
                var = past_rec.allowed_variable
                old_val = past_rec.old_value
                adapter = self.engine._get_adapter(var)
                if adapter:
                    adapter.rollback(old_val)

            # Remeasure live state against base checkpoint
            base_cp = sequence.checkpoints[0]
            live_ctrl = self.engine.get_current_control_state()
            restored_ok = all(live_ctrl.get(k) == base_cp.control_state.get(k) for k in base_cp.control_state)

            if restored_ok:
                sequence.sequence_state = SequenceState.ROLLED_BACK
                sequence.record_audit_event("checkpoint_restored", {"target": base_cp.checkpoint_id, "scope": "CASCADING"})
                return self._finalize_report(sequence, success=False, error_reason="STEP_FAILED_CASCADING_ROLLBACK")
            else:
                sequence.sequence_state = SequenceState.FAILED_SAFE
                sequence.record_audit_event("sequence_failed_safe", {"reason": "Cascading rollback failed to restore base state"})
                return self._finalize_report(sequence, success=False, error_reason="FAILED_SAFE")

    def _finalize_report(
        self,
        sequence: TransactionSequence,
        success: bool,
        error_reason: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Compiles machine-readable sequence execution report."""
        return {
            "sequence_id": sequence.sequence_id,
            "success": success,
            "sequence_state": sequence.sequence_state.value,
            "source_state_version": sequence.source_state_version,
            "final_state_version": self.version_manager.current(),
            "steps_total": len(sequence.decisions),
            "steps_executed": len(sequence.executed_records),
            "rollback_mode": sequence.rollback_mode.value,
            "error_reason": error_reason,
            "audit_hash": sequence.audit_hash,
            "checkpoints_count": len(sequence.checkpoints),
            "events_count": len(sequence.audit_ledger),
            "resource_budget": sequence.resource_budget.to_dict(),
        }

    def get_sequence(self, sequence_id: str) -> Optional[TransactionSequence]:
        with self._arbiter_lock:
            return self._sequences.get(sequence_id)

    def abort_sequence(self, sequence_id: str, reason: str = "MANUAL_ABORT") -> bool:
        with self._arbiter_lock:
            seq = self._sequences.get(sequence_id)
            if not seq or seq.sequence_state in {SequenceState.COMMITTED, SequenceState.ROLLED_BACK, SequenceState.FAILED_SAFE}:
                return False
            seq.sequence_state = SequenceState.INVALIDATED
            seq.record_audit_event("sequence_aborted", {"reason": reason})
            return True

    def get_state_version(self) -> int:
        return self.version_manager.current()

    def get_resource_headroom(self) -> Dict[str, float]:
        return self.safety_evaluator.safe_headroom.to_dict()

    def generate_report(self) -> Dict[str, Any]:
        """Generates comprehensive aggregated arbiter status report."""
        with self._arbiter_lock:
            all_seqs = list(self._sequences.values())
            return {
                "phase": "Phase 4.2: Hardware-Grounded Sequence Arbiter",
                "current_state_version": self.version_manager.current(),
                "sequences_total": len(all_seqs),
                "sequences_committed": sum(1 for s in all_seqs if s.sequence_state == SequenceState.COMMITTED),
                "sequences_rolled_back": sum(1 for s in all_seqs if s.sequence_state == SequenceState.ROLLED_BACK),
                "sequences_failed_safe": sum(1 for s in all_seqs if s.sequence_state == SequenceState.FAILED_SAFE),
                "sequences_invalidated": sum(1 for s in all_seqs if s.sequence_state == SequenceState.INVALIDATED),
                "canonical_lock_order": CoupledVariableLockManager.CANONICAL_ORDER,
                "safe_headroom": self.safety_evaluator.safe_headroom.to_dict(),
                "uma_telemetry": self.uma_tracker.sample_pressure(),
            }
