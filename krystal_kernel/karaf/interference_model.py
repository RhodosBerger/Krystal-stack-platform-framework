#!/usr/bin/env python3
"""
==============================================================================
KARAF: ALGORITHM G — RUNTIME INTERFERENCE MATRIX
==============================================================================
Module: krystal_kernel/karaf/interference_model.py
Description:
    Computes empirical pairwise interference tensor I_ij(t) among 8 hardware
    and runtime domains:
      CPU, MEMORY, UMA, GPU, MODEL_CASCADE, POWER, THERMAL, STORAGE, NETWORK

    Epistemic Law:
      "Correlation != Causality."
      Distinguishes raw correlation, lagged cross-correlation, and common-cause
      confounders. Emits testable RuntimeHypotheses instead of asserting causation.
==============================================================================
"""

import math
import time
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple

from krystal_kernel.scene_constructor_presets import VITAL_MAX_HP
from krystal_kernel.karaf.types import (
    RuntimeHypothesis,
    HypothesisStatus,
    ObservationFabricSnapshot
)


@dataclass
class InterferenceEdge:
    domain_i: str
    domain_j: str
    metric_i: str
    metric_j: str
    pearson_r: float
    lagged_r: float
    lag_steps: int
    sample_count: int
    confidence: float
    is_causal_candidate: bool
    confounder_risk: float
    vital_max_hp: int = VITAL_MAX_HP

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pair": f"{self.domain_i} -> {self.domain_j}",
            "metrics": f"{self.metric_i} vs {self.metric_j}",
            "pearson_r": round(self.pearson_r, 4),
            "lagged_r": round(self.lagged_r, 4),
            "lag_steps": self.lag_steps,
            "sample_count": self.sample_count,
            "confidence": round(self.confidence, 3),
            "is_causal_candidate": self.is_causal_candidate,
            "confounder_risk": round(self.confounder_risk, 3),
            "vital_max_hp": self.vital_max_hp
        }


class RuntimeInterferenceMatrix:
    """
    Maintains online sliding window correlation and Granger-lagged causality
    tracking among telemetry channels.
    """

    DOMAINS = [
        "CPU", "MEMORY", "UMA", "GPU", "MODEL_CASCADE",
        "POWER", "THERMAL", "STORAGE", "NETWORK"
    ]

    def __init__(self, history_capacity: int = 128):
        assert VITAL_MAX_HP == 6, f"Constitutional invariant breach: VITAL_MAX_HP must be 6, got {VITAL_MAX_HP}"
        self.vital_max_hp = VITAL_MAX_HP
        self.capacity = history_capacity
        self._series: Dict[str, List[float]] = {}

    def ingest_snapshot(self, snapshot: ObservationFabricSnapshot):
        """Records time-series points from snapshot metrics."""
        mapping = {
            "CPU:util": float(snapshot.get_obs("OBS_CPU_UTILIZATION", 0.0)),
            "CPU:ctx": float(snapshot.get_obs("OBS_CONTEXT_SWITCH_RATE", 0.0)),
            "MEMORY:ram": float(snapshot.get_obs("OBS_RAM_USED", 0.0)),
            "UMA:alloc": float(snapshot.get_obs("OBS_GPU_UMA_ALLOC", 0.0)),
            "GPU:load": float(snapshot.get_obs("OBS_GPU_LOAD", 0.0)),
            "POWER:watts": float(snapshot.get_obs("OBS_PACKAGE_POWER", 0.0)),
            "THERMAL:temp": float(snapshot.get_obs("OBS_CORE_TEMP", 0.0)),
            "STORAGE:lat": float(snapshot.get_obs("OBS_STORAGE_LATENCY", 0.0)),
            "NETWORK:rx": float(snapshot.get_obs("OBS_NETWORK_RX", 0.0)),
        }

        for k, v in mapping.items():
            if k not in self._series:
                self._series[k] = []
            self._series[k].append(v)
            if len(self._series[k]) > self.capacity:
                self._series[k].pop(0)

    def _calc_pearson(self, x: List[float], y: List[float]) -> float:
        n = min(len(x), len(y))
        if n < 3:
            return 0.0
        mx = sum(x[:n]) / n
        my = sum(y[:n]) / n
        var_x = sum((val - mx)**2 for val in x[:n])
        var_y = sum((val - my)**2 for val in y[:n])
        if var_x < 1e-9 or var_y < 1e-9:
            return 0.0
        cov = sum((x[i] - mx) * (y[i] - my) for i in range(n))
        return cov / math.sqrt(var_x * var_y)

    def compute_interference_matrix(self) -> List[InterferenceEdge]:
        """
        Builds the empirical interference edges I_ij(t).
        """
        edges: List[InterferenceEdge] = []
        keys = list(self._series.keys())

        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                k1, k2 = keys[i], keys[j]
                s1, s2 = self._series[k1], self._series[k2]
                n = min(len(s1), len(s2))
                if n < 4:
                    continue

                r = self._calc_pearson(s1, s2)

                # Lagged cross-correlation (k1 leads k2 by 1 step)
                r_lag = self._calc_pearson(s1[:-1], s2[1:]) if n > 4 else r

                # Causality heuristic: lagged correlation must exceed instantaneous correlation
                # and sample count must be substantial to avoid spurious noise
                is_candidate = (abs(r_lag) > 0.65) and (abs(r_lag) >= abs(r) * 0.9) and (n >= 10)
                confounder_risk = 0.45 if ("POWER" in k1 or "CPU" in k1) and ("THERMAL" in k2 or "GPU" in k2) else 0.15

                d1 = k1.split(":")[0]
                d2 = k2.split(":")[0]

                edge = InterferenceEdge(
                    domain_i=d1,
                    domain_j=d2,
                    metric_i=k1,
                    metric_j=k2,
                    pearson_r=r,
                    lagged_r=r_lag,
                    lag_steps=1,
                    sample_count=n,
                    confidence=min(1.0, 0.4 + (n / 30.0) * 0.6),
                    is_causal_candidate=is_candidate,
                    confounder_risk=confounder_risk,
                    vital_max_hp=VITAL_MAX_HP
                )
                edges.append(edge)

        return edges

    def generate_interference_hypotheses(self) -> List[RuntimeHypothesis]:
        """
        Translates strong interference edges into falsifiable RuntimeHypotheses.
        """
        edges = self.compute_interference_matrix()
        hypotheses: List[RuntimeHypothesis] = []

        for edge in edges:
            if abs(edge.pearson_r) > 0.70:
                h_id = f"hyp_interf_{edge.domain_i}_{edge.domain_j}_{int(time.time())}"
                hyp = (
                    f"Coupled interference observed between {edge.metric_i} and {edge.metric_j} "
                    f"(Pearson r={round(edge.pearson_r, 3)}, lagged_r={round(edge.lagged_r, 3)}). "
                    f"Confounder risk: {round(edge.confounder_risk, 2)}."
                )
                pred = (
                    f"Perturbing {edge.metric_i} will induce a correlated change in {edge.metric_j} "
                    f"if and only if the relation is causal and not confounded by third-party governor."
                )
                falsif = (
                    f"Controlled perturbation experiment: hold {edge.domain_i} constant while injecting "
                    f"synthetic load to isolate confounder."
                )

                rh = RuntimeHypothesis(
                    hypothesis_id=h_id,
                    hypothesis=hyp,
                    prediction=pred,
                    falsification_test=falsif,
                    observations=[edge.metric_i, edge.metric_j],
                    derived_metrics=["DER_WORKER_CONTENTION", "DER_THERMAL_HEADROOM"],
                    sample_count=edge.sample_count,
                    effect_size=round(abs(edge.pearson_r), 3),
                    confidence=edge.confidence,
                    status=HypothesisStatus.HYP_PROPOSED,
                    vital_max_hp=VITAL_MAX_HP
                )
                hypotheses.append(rh)

        return hypotheses
