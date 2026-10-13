#!/usr/bin/env python3
"""
==============================================================================
KARAF: ALGORITHM H — EPISTEMIC EVIDENCE DECAY
==============================================================================
Module: krystal_kernel/karaf/evidence_decay.py
Description:
    Computes time-dependent confidence decay for empirical evidence:
        C(t) = C_0 * exp(-lambda * dt)

    Rules:
      - Domain-specific decay rates:
          * Volatile (CPU load, GPU load, queue depth): high lambda (~1.2 / sec)
          * Semi-volatile (thermal, memory pressure): medium lambda (~0.25 / sec)
          * Static (hardware topology, device capabilities): zero or near-zero lambda
      - Emits status flags:
          * NOMINAL (fresh)
          * TMP_STALE (confidence dropped below threshold)
          * TMP_EXPIRED (age exceeded hard TTL)
      - Old data remains historical; NEVER silently masquerades as current evidence.
==============================================================================
"""

import math
import time
from enum import Enum, unique
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple

from krystal_kernel.scene_constructor_presets import VITAL_MAX_HP
from krystal_kernel.karaf.types import ObservationMetric


@unique
class EvidenceFreshnessStatus(str, Enum):
    NOMINAL     = "NOMINAL"
    TMP_STALE   = "TMP_STALE"
    TMP_EXPIRED = "TMP_EXPIRED"


@dataclass
class DecayedEvidenceReport:
    metric_name: str
    original_confidence: float
    current_confidence: float
    age_seconds: float
    decay_lambda: float
    status: EvidenceFreshnessStatus
    is_usable_as_current: bool
    vital_max_hp: int = VITAL_MAX_HP

    def to_dict(self) -> Dict[str, Any]:
        return {
            "metric_name": self.metric_name,
            "original_confidence": round(self.original_confidence, 4),
            "current_confidence": round(self.current_confidence, 4),
            "age_seconds": round(self.age_seconds, 4),
            "decay_lambda": self.decay_lambda,
            "status": self.status.value,
            "is_usable_as_current": self.is_usable_as_current,
            "vital_max_hp": self.vital_max_hp
        }


class EvidenceDecayEngine:
    """
    Manages exponential decay models for empirical evidence across domains.
    """

    # Domain / Metric lambda decay coefficients (per second)
    DEFAULT_LAMBDAS = {
        # Highly volatile
        "OBS_CPU_UTILIZATION": 1.5,
        "OBS_GPU_LOAD": 1.2,
        "OBS_CONTEXT_SWITCH_RATE": 1.8,
        "OBS_NETWORK_RX": 1.5,
        "queue_depth": 2.0,

        # Semi-volatile
        "OBS_PACKAGE_POWER": 0.4,
        "OBS_CORE_TEMP": 0.25,
        "OBS_RAM_USED": 0.15,
        "OBS_GPU_UMA_ALLOC": 0.20,
        "OBS_STORAGE_LATENCY": 0.50,

        # Slow-changing / invariant
        "hardware_topology": 0.0001,
        "device_capability": 0.0001,
        "ram_total_mb": 0.0,
        "OBS_VITAL_MAX_HP": 0.0
    }

    def __init__(
        self,
        stale_confidence_threshold: float = 0.50,
        hard_ttl_multiplier: float = 3.0
    ):
        assert VITAL_MAX_HP == 6, f"Constitutional invariant breach: VITAL_MAX_HP must be 6, got {VITAL_MAX_HP}"
        self.vital_max_hp = VITAL_MAX_HP
        self.stale_threshold = stale_confidence_threshold
        self.ttl_multiplier = hard_ttl_multiplier

    def get_lambda_for_metric(self, metric_name: str) -> float:
        if metric_name in self.DEFAULT_LAMBDAS:
            return self.DEFAULT_LAMBDAS[metric_name]
        if "CPU" in metric_name or "LOAD" in metric_name or "SWITCH" in metric_name:
            return 1.2
        if "TEMP" in metric_name or "POWER" in metric_name or "MEMORY" in metric_name:
            return 0.3
        return 0.1

    def evaluate_metric(
        self,
        metric: ObservationMetric,
        current_time_ns: Optional[int] = None,
        c0: float = 1.0
    ) -> DecayedEvidenceReport:
        """
        Calculates C(t) = C_0 * exp(-lambda * dt).
        """
        now_ns = current_time_ns or time.time_ns()
        dt_sec = max(0.0, (now_ns - metric.timestamp_ns) / 1e9)
        lmb = self.get_lambda_for_metric(metric.name)

        # Decay formula
        c_t = c0 * math.exp(-lmb * dt_sec) if lmb > 0.0 else c0
        c_t = max(0.0, min(1.0, c_t))

        # Hard TTL check
        hard_ttl_sec = (metric.ttl_ns / 1e9) * self.ttl_multiplier

        if dt_sec > hard_ttl_sec or c_t < 0.10:
            status = EvidenceFreshnessStatus.TMP_EXPIRED
            usable = False
        elif c_t < self.stale_threshold or metric.is_stale:
            status = EvidenceFreshnessStatus.TMP_STALE
            usable = False
        else:
            status = EvidenceFreshnessStatus.NOMINAL
            usable = True

        return DecayedEvidenceReport(
            metric_name=metric.name,
            original_confidence=c0,
            current_confidence=c_t,
            age_seconds=dt_sec,
            decay_lambda=lmb,
            status=status,
            is_usable_as_current=usable,
            vital_max_hp=VITAL_MAX_HP
        )

    def filter_fresh_evidence(
        self,
        metrics: Dict[str, ObservationMetric]
    ) -> Tuple[Dict[str, ObservationMetric], List[DecayedEvidenceReport]]:
        """
        Partitions an evidence dictionary into fresh usable evidence and stale/expired audit logs.
        """
        fresh: Dict[str, ObservationMetric] = {}
        reports: List[DecayedEvidenceReport] = []

        now_ns = time.time_ns()
        for k, m in metrics.items():
            rep = self.evaluate_metric(m, current_time_ns=now_ns)
            reports.append(rep)
            if rep.is_usable_as_current:
                fresh[k] = m

        return fresh, reports
