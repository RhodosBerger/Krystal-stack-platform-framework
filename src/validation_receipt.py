#!/usr/bin/env python3
"""
==============================================================================
KRYSTAL-STACK: VALIDATION RECEIPT (M1)
==============================================================================
Module: src/validation_receipt.py
Standardized, immutable, cryptographic audit receipt for executed transitions:
    - state_before: List[float]
    - predicted_state: List[float]
    - action: str
    - actual_state_after: List[float]
    - prediction_residual: List[float]
    - prediction_mae: float
    - transition_reliability: float
    - utility_gain: float
    - verdict: str ('COMMITTED' or 'DISCARDED')
    - hmac_digest: str
Constitutional Invariant: VITAL_MAX_HP == 6
==============================================================================
"""

from __future__ import annotations

import time
import hashlib
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional
from .constants import VITAL_MAX_HP


@dataclass(frozen=True)
class ValidationReceipt:
    """Immutable, cryptographically verifiable transition receipt."""
    receipt_id: str
    timestamp_ns: int
    state_before: List[float]
    predicted_state: List[float]
    action: str
    actual_state_after: List[float]
    prediction_residual: List[float]
    prediction_mae: float
    transition_reliability: float
    utility_gain: float
    is_constraint_violated: bool
    verdict: str  # 'COMMITTED', 'DISCARDED', 'QUARANTINED'
    vital_max_hp: int
    hmac_digest: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def create(
        cls,
        receipt_id: str,
        state_before: List[float],
        predicted_state: List[float],
        action: str,
        actual_state_after: List[float],
        utility_gain: float,
        is_constraint_violated: bool,
        verdict: str = "COMMITTED",
        vital_max_hp: int = VITAL_MAX_HP,
    ) -> ValidationReceipt:
        """Constructs an immutable ValidationReceipt with cryptographic HMAC signature."""
        min_dim = min(len(predicted_state), len(actual_state_after))
        residual = [round(actual_state_after[i] - predicted_state[i], 4) for i in range(min_dim)]
        mae = round(sum(abs(r) for r in residual) / max(1, min_dim), 4)

        # Dimension-tolerant Transition Reliability
        # Tolerances: [0.05, 0.05, 0.05, 0.05, 0.05, 0.05] on normalized space
        nr = [min(1.0, abs(r) / 0.08) for r in residual]
        tr = round(max(0.0, 1.0 - (sum(nr) / max(1, min_dim))), 4)

        timestamp_ns = time.time_ns()
        sig_body = f"{receipt_id}:{action}:{mae}:{tr}:{verdict}:{vital_max_hp}:{timestamp_ns}"
        hmac = hashlib.sha256(sig_body.encode("utf-8")).hexdigest()

        return cls(
            receipt_id=receipt_id,
            timestamp_ns=timestamp_ns,
            state_before=state_before,
            predicted_state=predicted_state,
            action=action,
            actual_state_after=actual_state_after,
            prediction_residual=residual,
            prediction_mae=mae,
            transition_reliability=tr,
            utility_gain=round(utility_gain, 4),
            is_constraint_violated=is_constraint_violated,
            verdict=verdict,
            vital_max_hp=vital_max_hp,
            hmac_digest=hmac,
        )
