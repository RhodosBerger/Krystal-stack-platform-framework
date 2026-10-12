#!/usr/bin/env python3
"""
==============================================================================
KRYSTAL-STACK: PREDICTION ERROR LEDGER & CONTEXTUAL RISK REGISTRY
==============================================================================
Module: krystal_kernel/runtime/prediction_error_ledger.py
Description:
    Implements shadow prediction error evaluation, calibration feedback,
    and contextual risk modeling.

    Rules:
      1. VETO must NOT create global permanent blacklisting!
         Model: Risk(action | context) where context includes:
         (action_signature, host_state_class, workload_class, pressure_region, policy_version).
      2. DEFER produces targeted micro-probe sampling feedback.
      3. QUARANTINE freezes decisions for contradictory streams and recalibrates.
      4. Tracks expected vs actual future deltas and updates model calibration error.
==============================================================================
"""

import time
import math
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple

from krystal_kernel.verification.resource_evidence_envelope import ExpectedDelta


@dataclass
class ContextualRiskKey:
    action_signature: str
    host_state_class: str
    workload_class: str
    pressure_region: str
    policy_version_id: str
    hardware_class: str = "generic_x86_uma"

    def as_key(self) -> str:
        return f"{self.action_signature}::{self.host_state_class}::{self.workload_class}::{self.pressure_region}::{self.policy_version_id}::{self.hardware_class}"


class ContextualRiskRegistry:
    """
    Context-sensitive risk registry.
    Prevents blanket global vetoes from poisoning unrelated workload contexts.
    """

    def __init__(self):
        self._context_risk_scores: Dict[str, float] = {}       # Key -> Risk score [0.0 - 1.0]
        self._context_observations: Dict[str, int] = {}
        self._quarantined_streams: Dict[str, Dict[str, Any]] = {}
        self._sampling_boost_targets: Dict[str, int] = {}

    def get_risk(self, context: ContextualRiskKey) -> float:
        """Returns contextual risk score (default 0.1 for unobserved contexts)."""
        return self._context_risk_scores.get(context.as_key(), 0.1)

    def record_veto(self, context: ContextualRiskKey, severity: float = 0.25):
        """
        Records a contextual veto. Increases risk ONLY for the specific matching context.
        Does NOT globally blacklist the action across other workloads or hardware states.
        """
        k = context.as_key()
        cur = self._context_risk_scores.get(k, 0.1)
        self._context_risk_scores[k] = min(1.0, cur + severity)
        self._context_observations[k] = self._context_observations.get(k, 0) + 1

    def record_defer(self, target_domain: str):
        """DEFER feedback: triggers increased targeted micro-probe sampling."""
        self._sampling_boost_targets[target_domain] = self._sampling_boost_targets.get(target_domain, 0) + 1

    def record_quarantine(self, stream_id: str, reason: str, details: Dict[str, Any]):
        """QUARANTINE feedback: freezes decisions and flags sensor for recalibration."""
        self._quarantined_streams[stream_id] = {
            "quarantined_at": time.time(),
            "reason": reason,
            "details": details
        }

    def is_stream_quarantined(self, stream_id: str) -> bool:
        return stream_id in self._quarantined_streams

    def lift_quarantine(self, stream_id: str):
        self._quarantined_streams.pop(stream_id, None)


@dataclass
class PredictionErrorRecord:
    record_id: str
    action_id: str
    expected_delta: ExpectedDelta
    actual_delta: Dict[str, float]
    mean_squared_error: float
    max_absolute_error: float
    evaluated_at_ns: int
    calibration_update_applied: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "record_id": self.record_id,
            "action_id": self.action_id,
            "expected_delta": self.expected_delta.to_dict(),
            "actual_delta": {k: round(v, 4) for k, v in self.actual_delta.items()},
            "mean_squared_error": round(self.mean_squared_error, 4),
            "max_absolute_error": round(self.max_absolute_error, 4),
            "evaluated_at_ns": self.evaluated_at_ns,
            "calibration_update_applied": round(self.calibration_update_applied, 4)
        }


class PredictionErrorLedger:
    """
    Shadow prediction ledger comparing expected counterfactual deltas
    against observed post-window states to adapt empirical calibration error.
    """

    def __init__(self, history_limit: int = 500):
        self._history_limit = history_limit
        self._records: List[PredictionErrorRecord] = []
        self._running_calibration_error: float = 0.05

    @property
    def running_calibration_error(self) -> float:
        return self._running_calibration_error

    def record_comparison(
        self,
        action_id: str,
        expected: ExpectedDelta,
        actual_delta: Dict[str, float],
        now_ns: Optional[int] = None
    ) -> PredictionErrorRecord:
        eval_time = now_ns or int(time.time() * 1e9)
        common_keys = set(expected.metric_deltas.keys()).intersection(actual_delta.keys())

        if not common_keys:
            mse = 1.0
            max_err = 1.0
        else:
            diffs = [expected.metric_deltas[k] - actual_delta[k] for k in common_keys]
            mse = sum(d ** 2 for d in diffs) / len(diffs)
            max_err = max(abs(d) for d in diffs)

        # Exponential moving average calibration adjustment
        alpha = 0.1
        self._running_calibration_error = (1 - alpha) * self._running_calibration_error + alpha * min(0.5, mse)

        rec = PredictionErrorRecord(
            record_id=f"ERR-{len(self._records)+1:06d}",
            action_id=action_id,
            expected_delta=expected,
            actual_delta=actual_delta,
            mean_squared_error=mse,
            max_absolute_error=max_err,
            evaluated_at_ns=eval_time,
            calibration_update_applied=self._running_calibration_error
        )
        self._records.append(rec)
        if len(self._records) > self._history_limit:
            self._records.pop(0)

        return rec

    def get_recent_records(self, limit: int = 10) -> List[Dict[str, Any]]:
        return [r.to_dict() for r in self._records[-limit:]]
