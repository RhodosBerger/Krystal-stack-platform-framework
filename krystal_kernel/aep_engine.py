"""
KRYSTAL-STACK // ADAPTIVE EXECUTION PROCESSOR (AEP) ENGINE
=============================================================================
Normative Architecture: docs/architecture/AEP_CANONICAL_SPEC.md
Classification: [IMPLEMENTED]

Core Implementation:
  1. AepReceipt - Universal observability & audit contract
  2. AepTransactionalEngine - L8 Candidate State Machine (S' = T(St, A))
  3. InvariantRule hierarchy (L2):
     - VitalHpInvariantRule (0 <= HP <= 6)
     - DoubleEntryConservationRule (sum(Debits) == sum(Credits))
     - NonNegativeBalanceRule (Balance >= 0)
  4. Workload dispatching & Telemetry recording
"""

from __future__ import annotations

import dataclasses
import enum
import hashlib
import hmac
import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar

# Constitutional Invariants
VITAL_MAX_HP: int = 6


class WorkloadType(str, enum.Enum):
    """Supported workload domains running over AEP."""
    GAME = "GAME"
    ECONOMY = "ECONOMY"
    INFERENCE = "INFERENCE"
    RENDER = "RENDER"
    SYSTEM = "SYSTEM"


class ExecutionLane(str, enum.Enum):
    """L5 Scheduler priority execution lanes."""
    REALTIME = "REALTIME"          # Low-latency, VSync locked
    COMPUTE = "COMPUTE"            # Batch, background LLM/shader
    BACKGROUND = "BACKGROUND"      # Telemetry, audits, exports


class AepVerdict(str, enum.Enum):
    """L8 Transactional commit verdict."""
    COMMITTED = "COMMITTED"
    DISCARDED_INVARIANT_VIOLATION = "DISCARDED_INVARIANT_VIOLATION"
    DISCARDED_BACKPRESSURE_REJECTED = "DISCARDED_BACKPRESSURE_REJECTED"
    DISCARDED_CONTRACT_INVALID = "DISCARDED_CONTRACT_INVALID"


@dataclass(frozen=True)
class AepReceipt:
    """
    Standardized observability and audit contract emitted by AEP (Layer 8).
    Shared across Game, Economy, AI Inference, and Render workloads.
    """
    operation_id: str
    runtime_id: str
    workload: WorkloadType
    action: str
    state_before: str          # SHA256 digest of St
    state_after: str           # SHA256 digest of St+1
    policy_id: str
    policy_revision: int
    execution_lane: ExecutionLane
    backend: str
    device: str
    execution_us: int
    bus_pressure: float        # Normalized pressure [0.0, 1.0]
    deadline_slack_us: int
    verdict: AepVerdict
    timestamp_ns: int = field(default_factory=time.time_ns)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "operation_id": self.operation_id,
            "runtime_id": self.runtime_id,
            "workload": self.workload.value,
            "action": self.action,
            "state_before": self.state_before,
            "state_after": self.state_after,
            "policy": {
                "id": self.policy_id,
                "revision": self.policy_revision,
            },
            "execution": {
                "lane": self.execution_lane.value,
                "backend": self.backend,
                "device": self.device,
            },
            "telemetry": {
                "execution_us": self.execution_us,
                "bus_pressure": round(self.bus_pressure, 4),
                "deadline_slack_us": self.deadline_slack_us,
            },
            "verdict": self.verdict.value,
            "timestamp_ns": self.timestamp_ns,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)


def compute_state_digest(state_data: Any, session_key: str = "krystal-aep-key") -> str:
    """Computes deterministic canonical HMAC-SHA256 digest over state."""
    if isinstance(state_data, dict):
        canonical_bytes = json.dumps(state_data, sort_keys=True).encode("utf-8")
    elif hasattr(state_data, "to_dict"):
        canonical_bytes = json.dumps(state_data.to_dict(), sort_keys=True).encode("utf-8")
    else:
        canonical_bytes = str(state_data).encode("utf-8")
    
    mac = hmac.new(session_key.encode("utf-8"), canonical_bytes, hashlib.sha256)
    return f"sha256:{mac.hexdigest()}"


class InvariantRule:
    """Base class for L2 Invariant Predicates."""
    def evaluate(self, candidate_state: Any) -> Tuple[bool, str]:
        raise NotImplementedError


class VitalHpInvariantRule(InvariantRule):
    """
    L2 Invariant: Combatant HP must strictly remain within [0, 6].
    """
    def evaluate(self, candidate_state: Any) -> Tuple[bool, str]:
        hp_vals = []
        if isinstance(candidate_state, dict):
            if "player_hp" in candidate_state:
                hp_vals.append(candidate_state["player_hp"])
            if "opponent_hp" in candidate_state:
                hp_vals.append(candidate_state["opponent_hp"])
            if "hp" in candidate_state:
                hp_vals.append(candidate_state["hp"])
        elif hasattr(candidate_state, "player_hp"):
            hp_vals.append(getattr(candidate_state, "player_hp"))
            if hasattr(candidate_state, "opponent_hp"):
                hp_vals.append(getattr(candidate_state, "opponent_hp"))
        
        for val in hp_vals:
            if not isinstance(val, (int, float)):
                return False, f"HP value must be numeric, got {type(val)}"
            if val < 0 or val > VITAL_MAX_HP:
                return False, f"HP bound invariant violated: {val} not in [0, {VITAL_MAX_HP}]"
        
        return True, "HP_INVARIANT_SATISFIED"


class DoubleEntryConservationRule(InvariantRule):
    """
    L2 Invariant: Financial/Resource balance: sum(Debits) == sum(Credits).
    Guarantees no assets are created or destroyed ex nihilo.
    """
    def evaluate(self, candidate_state: Any) -> Tuple[bool, str]:
        if not isinstance(candidate_state, dict):
            if hasattr(candidate_state, "to_dict"):
                candidate_state = candidate_state.to_dict()
            else:
                return False, "Candidate state must be mapping for double-entry evaluation"

        debits = candidate_state.get("debits", 0.0)
        credits = candidate_state.get("credits", 0.0)
        
        # Exact numerical balance within epsilon
        if abs(float(debits) - float(credits)) > 1e-9:
            return False, f"Double-entry violation: debits ({debits}) != credits ({credits})"
            
        return True, "DOUBLE_ENTRY_CONSERVED"


class NonNegativeBalanceRule(InvariantRule):
    """
    L2 Invariant: Account or inventory quantities must never drop below 0.
    """
    def evaluate(self, candidate_state: Any) -> Tuple[bool, str]:
        if not isinstance(candidate_state, dict):
            if hasattr(candidate_state, "to_dict"):
                candidate_state = candidate_state.to_dict()
            else:
                return False, "Candidate state must be mapping for balance evaluation"

        balances = candidate_state.get("balances", {})
        if isinstance(balances, dict):
            for res_id, qty in balances.items():
                if qty < 0:
                    return False, f"Negative balance invariant violated for {res_id}: {qty} < 0"

        return True, "NON_NEGATIVE_BALANCES_SATISFIED"


@dataclass
class AepTransactionResult:
    """Result of an AEP transactional transition."""
    committed: bool
    state: Any
    receipt: AepReceipt
    message: str


class AepTransactionalEngine:
    """
    Layer 8 Transactional State Machine implementing candidate transition semantics:
      1. St (Authoritative current state)
      2. S' = T(St, A) (Candidate transition)
      3. Invariant gate check: I(S')
      4. Backpressure gate check
      5. Commit: St+1 = S' if valid else St (Discard S', Delta S == 0)
    """

    def __init__(
        self,
        invariants: Optional[List[InvariantRule]] = None,
        max_pressure_threshold: float = 0.85,
    ) -> None:
        self.invariants: List[InvariantRule] = invariants or [VitalHpInvariantRule()]
        self.max_pressure_threshold = max_pressure_threshold
        self.receipt_ledger: List[AepReceipt] = []

    def execute_transition(
        self,
        runtime_id: str,
        workload: WorkloadType,
        action: str,
        current_state: Any,
        transition_fn: Callable[[Any, Dict[str, Any]], Any],
        action_payload: Dict[str, Any],
        policy_id: str = "default-adaptive-pacing",
        policy_revision: int = 1,
        execution_lane: ExecutionLane = ExecutionLane.REALTIME,
        backend: str = "python-reference",
        device: str = "CPU",
        current_bus_pressure: float = 0.10,
        deadline_budget_us: int = 16666,
    ) -> AepTransactionResult:
        """
        Executes zero-dirty candidate mutation under AEP contract.
        """
        t_start = time.perf_counter_ns()
        op_id = f"01J{uuid.uuid4().hex[:20].upper()}"
        state_before_digest = compute_state_digest(current_state)

        # 1. Backpressure Pre-Check (L7)
        if current_bus_pressure > self.max_pressure_threshold:
            t_exec_us = max(1, (time.perf_counter_ns() - t_start) // 1000)
            receipt = AepReceipt(
                operation_id=op_id,
                runtime_id=runtime_id,
                workload=workload,
                action=action,
                state_before=state_before_digest,
                state_after=state_before_digest,
                policy_id=policy_id,
                policy_revision=policy_revision,
                execution_lane=execution_lane,
                backend=backend,
                device=device,
                execution_us=t_exec_us,
                bus_pressure=current_bus_pressure,
                deadline_slack_us=max(0, deadline_budget_us - t_exec_us),
                verdict=AepVerdict.DISCARDED_BACKPRESSURE_REJECTED,
            )
            self.receipt_ledger.append(receipt)
            return AepTransactionResult(
                committed=False,
                state=current_state,
                receipt=receipt,
                message=f"Transition rejected due to excessive bus pressure: {current_bus_pressure:.2f} > {self.max_pressure_threshold:.2f}",
            )

        # 2. Candidate Generation (L6): S' = T(St, A)
        try:
            candidate_state = transition_fn(current_state, action_payload)
        except Exception as e:
            t_exec_us = max(1, (time.perf_counter_ns() - t_start) // 1000)
            receipt = AepReceipt(
                operation_id=op_id,
                runtime_id=runtime_id,
                workload=workload,
                action=action,
                state_before=state_before_digest,
                state_after=state_before_digest,
                policy_id=policy_id,
                policy_revision=policy_revision,
                execution_lane=execution_lane,
                backend=backend,
                device=device,
                execution_us=t_exec_us,
                bus_pressure=current_bus_pressure,
                deadline_slack_us=max(0, deadline_budget_us - t_exec_us),
                verdict=AepVerdict.DISCARDED_CONTRACT_INVALID,
            )
            self.receipt_ledger.append(receipt)
            return AepTransactionResult(
                committed=False,
                state=current_state,
                receipt=receipt,
                message=f"Transition function exception: {str(e)}",
            )

        # 3. Invariant Evaluation (L2): I(S')
        for rule in self.invariants:
            is_valid, reason = rule.evaluate(candidate_state)
            if not is_valid:
                # S' discarded; St remains completely untouched (Delta S == 0)
                t_exec_us = max(1, (time.perf_counter_ns() - t_start) // 1000)
                receipt = AepReceipt(
                    operation_id=op_id,
                    runtime_id=runtime_id,
                    workload=workload,
                    action=action,
                    state_before=state_before_digest,
                    state_after=state_before_digest,
                    policy_id=policy_id,
                    policy_revision=policy_revision,
                    execution_lane=execution_lane,
                    backend=backend,
                    device=device,
                    execution_us=t_exec_us,
                    bus_pressure=current_bus_pressure,
                    deadline_slack_us=max(0, deadline_budget_us - t_exec_us),
                    verdict=AepVerdict.DISCARDED_INVARIANT_VIOLATION,
                )
                self.receipt_ledger.append(receipt)
                return AepTransactionResult(
                    committed=False,
                    state=current_state,
                    receipt=receipt,
                    message=f"Candidate state discarded due to invariant failure: {reason}",
                )

        # 4. Commit Gate (L8): St+1 = S'
        state_after_digest = compute_state_digest(candidate_state)
        t_exec_us = max(1, (time.perf_counter_ns() - t_start) // 1000)
        receipt = AepReceipt(
            operation_id=op_id,
            runtime_id=runtime_id,
            workload=workload,
            action=action,
            state_before=state_before_digest,
            state_after=state_after_digest,
            policy_id=policy_id,
            policy_revision=policy_revision,
            execution_lane=execution_lane,
            backend=backend,
            device=device,
            execution_us=t_exec_us,
            bus_pressure=current_bus_pressure,
            deadline_slack_us=max(0, deadline_budget_us - t_exec_us),
            verdict=AepVerdict.COMMITTED,
        )
        self.receipt_ledger.append(receipt)

        return AepTransactionResult(
            committed=True,
            state=candidate_state,
            receipt=receipt,
            message="State transition successfully committed",
        )
