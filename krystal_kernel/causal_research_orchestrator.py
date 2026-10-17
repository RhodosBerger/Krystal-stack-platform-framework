"""
Krystal Stack Platform Framework - Closed-Loop Causal Research Orchestrator
===========================================================================
Module: krystal_kernel/causal_research_orchestrator.py
Milestone: V-KSPP-PRINCIPLES-01 (Principle 7: Physical-First Actuation)
Reference: docs/architecture/KRYSTAL_SOVEREIGN_PRINCIPLE_PACK.md

Guarantees:
  Commit(Delta_S) => State == AUDITED and not RegressionObserved(M_post, M_base)
Enforces a 2-phase physical actuation state machine with canonical IEEE-754
hexadecimal serialization and rollback on regression (Delta_S == 0).
"""

from __future__ import annotations

import json
import hmac
import hashlib
from enum import IntEnum
from dataclasses import dataclass
from typing import Callable, Any, Final

from krystal_kernel.constants import VITAL_MAX_HP, ContractViolation
from krystal_kernel.zero_copy_contracts import canonicalize_float32


class ExperimentState(IntEnum):
    PLANNED            = 1
    BASELINE_MEASURED  = 2
    ACTUATED           = 3
    POST_MEASURED      = 4
    AUDITED            = 5
    COMMITTED          = 6
    ROLLED_BACK        = 7
    RECEIPT_FINALIZED  = 8
    KNOWLEDGE_ELIGIBLE = 9


class MetricObjective(IntEnum):
    MINIMIZE     = 1
    MAXIMIZE     = 2
    TARGET_RANGE = 3


def canonical_float_repr(val: float) -> str:
    """Matematicky injektívna IEEE-754 hexadecimálna bijekcia (so zjednotenou +0.0)."""
    norm = canonicalize_float32(val)
    return float(norm).hex()


def is_objective_satisfied(baseline: float, post: float, objective: MetricObjective, min_delta: float = 1e-6) -> bool:
    if objective == MetricObjective.MINIMIZE:
        return (baseline - post) >= min_delta
    if objective == MetricObjective.MAXIMIZE:
        return (post - baseline) >= min_delta
    return abs(post - baseline) <= min_delta


@dataclass(frozen=True)
class CausalTransactionReceipt:
    """Signed audit trail confirming physical telemetry and directional validation."""
    experiment_id: str
    state: ExperimentState
    baseline_hex: str
    post_hex: str
    committed: bool
    origin_hmac: str
    vital_hp: int = VITAL_MAX_HP

    def __post_init__(self):
        if self.vital_hp != VITAL_MAX_HP:
            raise ContractViolation("VITAL_MAX_HP must be 6 in CausalTransactionReceipt")

    @classmethod
    def create_signed(
        cls, exp_id: str, state: ExperimentState, baseline: float, post: float, committed: bool, key: bytes
    ) -> "CausalTransactionReceipt":
        b_hex = canonical_float_repr(baseline)
        p_hex = canonical_float_repr(post)
        payload = json.dumps(
            {"committed": committed, "exp_id": exp_id, "b": b_hex, "p": p_hex, "s": int(state), "hp": VITAL_MAX_HP},
            sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        sig = hmac.new(key, payload, hashlib.sha256).hexdigest()
        return cls(exp_id, state, b_hex, p_hex, committed, sig, VITAL_MAX_HP)

    def verify_signature(self, key: bytes) -> bool:
        payload = json.dumps(
            {"committed": self.committed, "exp_id": self.experiment_id, "b": self.baseline_hex, "p": self.post_hex, "s": int(self.state), "hp": self.vital_hp},
            sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        expected_sig = hmac.new(key, payload, hashlib.sha256).hexdigest()
        return hmac.compare_digest(self.origin_hmac, expected_sig)


class ClosedLoopCausalOrchestrator:
    """Garantuje, že žiaden commit neprebehne pred fyzickým POST_MEASURED a smerovým auditom."""

    def __init__(self, secret_key: bytes) -> None:
        self.secret_key = secret_key

    def execute_grounded_transaction(
        self,
        exp_id: str,
        objective: MetricObjective,
        sampler: Callable[[], float],
        actuate_fn: Callable[[], None],
        rollback_fn: Callable[[], None]
    ) -> CausalTransactionReceipt:
        state = ExperimentState.PLANNED
        baseline_val = sampler()
        state = ExperimentState.BASELINE_MEASURED

        actuate_fn()
        state = ExperimentState.ACTUATED

        # Fyzické meranie počas aktívnej zmeny na hostiteľskom OS
        post_val = sampler()
        state = ExperimentState.POST_MEASURED

        passed = is_objective_satisfied(baseline_val, post_val, objective)
        state = ExperimentState.AUDITED

        if passed:
            state = ExperimentState.COMMITTED
            committed = True
            final_state = ExperimentState.KNOWLEDGE_ELIGIBLE
        else:
            rollback_fn()
            state = ExperimentState.ROLLED_BACK
            committed = False
            final_state = ExperimentState.RECEIPT_FINALIZED

        return CausalTransactionReceipt.create_signed(
            exp_id, final_state, baseline_val, post_val, committed, self.secret_key
        )
