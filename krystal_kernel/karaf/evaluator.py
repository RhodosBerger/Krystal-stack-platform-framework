#!/usr/bin/env python3
"""
==============================================================================
KARAF: CLOSED-LOOP EVALUATOR & DEPENDENCY-SCOPED STATE VALIDATOR
==============================================================================
Module: krystal_kernel/karaf/evaluator.py
Description:
    Evaluates expected vs observed telemetry outcomes, logs prediction errors,
    and enforces fine-grained dependency-scoped state validity:
      - Validates read sets per domain (cpu, memory, thermal, gpu)
      - Prevents global monolithic state version livelock (F14)
      - Feeds discrepancy signals into the HypothesisMiner
==============================================================================
"""

import time
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Set, Tuple

from krystal_kernel.scene_constructor_presets import VITAL_MAX_HP
from krystal_kernel.karaf.types import (
    ObservationFabricSnapshot,
    RuntimeHypothesis
)
from krystal_kernel.runtime.prediction_error_ledger import PredictionErrorLedger


@dataclass
class ScopedStateVersion:
    """Dependency-scoped version vector preventing global state invalidation livelocks."""
    domain_versions: Dict[str, int] = field(default_factory=dict)
    hardware_snapshot_id: str = ""
    generated_at_ns: int = field(default_factory=time.time_ns)
    valid_until_ns: int = field(default_factory=lambda: time.time_ns() + 5_000_000_000)
    vital_max_hp: int = VITAL_MAX_HP

    def __post_init__(self):
        assert self.vital_max_hp == 6, "VITAL_MAX_HP must equal 6"

    def is_domain_valid(self, domain: str, current_domain_version: int) -> bool:
        """Verifies if the specific domain has remained stable."""
        expected = self.domain_versions.get(domain)
        if expected is None:
            return True
        return expected == current_domain_version

    def is_expired(self, current_time_ns: Optional[int] = None) -> bool:
        now = current_time_ns or time.time_ns()
        return now > self.valid_until_ns


@dataclass
class EvaluationRecord:
    metric_name: str
    expected_value: float
    observed_value: float
    absolute_error: float
    percentage_error: float
    context_tags: Dict[str, Any]
    timestamp_ns: int = field(default_factory=time.time_ns)
    vital_max_hp: int = VITAL_MAX_HP

    def to_dict(self) -> Dict[str, Any]:
        return {
            "metric_name": self.metric_name,
            "expected_value": round(self.expected_value, 4),
            "observed_value": round(self.observed_value, 4),
            "absolute_error": round(self.absolute_error, 4),
            "percentage_error": round(self.percentage_error, 2),
            "context_tags": self.context_tags,
            "timestamp_ns": self.timestamp_ns,
            "vital_max_hp": self.vital_max_hp
        }


class ClosedLoopEvaluator:
    """
    Evaluator connecting expected predictions with real telemetry,
    driving error ledgers and hypothesis falsification.
    """

    def __init__(self, error_ledger: Optional[PredictionErrorLedger] = None):
        assert VITAL_MAX_HP == 6, f"Constitutional invariant breach: VITAL_MAX_HP must be 6, got {VITAL_MAX_HP}"
        self.vital_max_hp = VITAL_MAX_HP
        self.error_ledger = error_ledger or PredictionErrorLedger()
        self._current_domain_versions: Dict[str, int] = {
            "cpu": 1,
            "memory": 1,
            "thermal": 1,
            "gpu": 1,
            "network": 1
        }

    def bump_domain_version(self, domain: str):
        """Bumps only the specific domain version without invalidating other domains."""
        self._current_domain_versions[domain] = self._current_domain_versions.get(domain, 1) + 1

    def create_scoped_version(self, domains: List[str], snapshot_id: str) -> ScopedStateVersion:
        """Captures read-set domain versions."""
        d_vers = {d: self._current_domain_versions.get(d, 1) for d in domains}
        return ScopedStateVersion(
            domain_versions=d_vers,
            hardware_snapshot_id=snapshot_id,
            generated_at_ns=time.time_ns(),
            valid_until_ns=time.time_ns() + 5_000_000_000,
            vital_max_hp=VITAL_MAX_HP
        )

    def verify_scoped_validity(self, scoped_version: ScopedStateVersion, required_domains: List[str]) -> Tuple[bool, Optional[str]]:
        """
        Verifies that none of the required domains have been invalidated.
        Unrelated domain changes (e.g. network burst) will NOT invalidate CPU decisions!
        """
        if scoped_version.is_expired():
            return False, "Scoped state version has expired (TTL elapsed)"

        for d in required_domains:
            curr_v = self._current_domain_versions.get(d, 1)
            if not scoped_version.is_domain_valid(d, curr_v):
                return False, f"Domain '{d}' state changed from v{scoped_version.domain_versions.get(d)} to v{curr_v}"

        return True, None

    def compare_expected_vs_observed(
        self,
        expected_metrics: Dict[str, float],
        snapshot: ObservationFabricSnapshot,
        context_tags: Optional[Dict[str, Any]] = None
    ) -> List[EvaluationRecord]:
        """
        Compares expected predictions against observed telemetry in snapshot.
        """
        records: List[EvaluationRecord] = []
        tags = context_tags or {}

        for m_name, exp_val in expected_metrics.items():
            obs_val = snapshot.get_obs(m_name)
            if obs_val is None:
                obs_val = snapshot.get_der(m_name, 0.0)
            obs_val = float(obs_val)

            abs_err = abs(obs_val - exp_val)
            pct_err = (abs_err / max(1e-4, abs(exp_val))) * 100.0

            rec = EvaluationRecord(
                metric_name=m_name,
                expected_value=exp_val,
                observed_value=obs_val,
                absolute_error=abs_err,
                percentage_error=pct_err,
                context_tags=tags,
                timestamp_ns=snapshot.timestamp_ns,
                vital_max_hp=VITAL_MAX_HP
            )
            records.append(rec)

            # Record in kernel error ledger
            self.error_ledger.record_error(
                metric_name=m_name,
                predicted=exp_val,
                actual=obs_val,
                context=tags
            )

        return records
