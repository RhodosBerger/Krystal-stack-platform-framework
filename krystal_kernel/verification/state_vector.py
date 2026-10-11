#!/usr/bin/env python3
"""
KRYSTAL-STACK // UNIFIED STATE VECTOR & MULTI-DIMENSIONAL GEOMETRY CONTRACT
=============================================================================
Section 1 & 2 Implementation:
Provides a domain-independent, bounded mathematical state vector and geometry
evaluator capable of representing and comparing disparate domains:
  - Game simulation (OBT, KON, POS, DOS)
  - LLM runtime (LAT, TPS, MEM, ROB)
  - Linux kernel (resource_pressure, scheduler_stability, memory_integrity, flow)
  - Cyber-physical CNC (thermal_gradient, rpm_stability, cutting_load, collapse_margin)
  - Fault-injection & recovery workloads

Core Invariant:
  VITAL_MAX_HP == 6 strictly preserved.
  No configuration is classified as "better" using Utility U alone.
=============================================================================
"""

import os
import sys
import math
import time
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple
from enum import Enum

WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

from krystal_kernel.scene_constructor_presets import VITAL_MAX_HP


class TradeoffClassification(str, Enum):
    PARETO_STRICT_IMPROVEMENT = "PARETO_STRICT_IMPROVEMENT" # All dimensions >= old, at least one >
    PARETO_STRICT_REGRESSION = "PARETO_STRICT_REGRESSION"   # All dimensions <= old, at least one <
    TRADEOFF = "TRADEOFF"                                   # Some improved, some degraded (e.g. +TPS, -ROB)
    NEUTRAL_EQUIVALENCE = "NEUTRAL_EQUIVALENCE"             # Changes within noise margin (|delta| < epsilon)


class ParetoStatus(str, Enum):
    NON_DOMINATED = "NON_DOMINATED"
    DOMINATED = "DOMINATED"
    BALANCED_FRONTIER = "BALANCED_FRONTIER"


@dataclass
class StateGeometry:
    composite_utility_u: float
    weakest_axis_code: str
    weakest_axis_score: float
    weighted_dispersion_d: float
    radar_polygon_svg: str
    is_balanced: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "composite_utility_u": round(self.composite_utility_u, 3),
            "weakest_axis_code": self.weakest_axis_code,
            "weakest_axis_score": round(self.weakest_axis_score, 2),
            "weighted_dispersion_d": round(self.weighted_dispersion_d, 3),
            "radar_polygon_svg": self.radar_polygon_svg,
            "is_balanced": self.is_balanced
        }


@dataclass
class StateVectorDelta:
    delta_vector: Dict[str, float]
    delta_utility_u: float
    delta_weakest_axis: float
    tradeoff_classification: TradeoffClassification
    pareto_status: ParetoStatus
    tradeoff_summary: str
    improved_dimensions: List[str]
    degraded_dimensions: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "delta_vector": {k: round(v, 3) for k, v in self.delta_vector.items()},
            "delta_utility_u": round(self.delta_utility_u, 3),
            "delta_weakest_axis": round(self.delta_weakest_axis, 3),
            "tradeoff_classification": self.tradeoff_classification.value,
            "pareto_status": self.pareto_status.value,
            "tradeoff_summary": self.tradeoff_summary,
            "improved_dimensions": self.improved_dimensions,
            "degraded_dimensions": self.degraded_dimensions
        }


@dataclass
class StateVector:
    """Universal, domain-independent bounded state vector."""
    domain_id: str
    subject_id: str
    measurement_dimensions: List[str]
    normalized_scores: List[float]       # Clamped [0.0 - 5.0]
    raw_values: List[float]
    units: List[str]
    weights: List[float]                 # Must sum to 1.0 (within epsilon)
    confidence: List[float]              # [0.0 - 1.0] per dimension
    uncertainty: List[float]             # [0.0 - 1.0] per dimension
    invariants: List[str]
    configuration_snapshot: Dict[str, Any]
    runtime_context: Dict[str, Any]
    timestamp: float = field(default_factory=time.time)
    vital_hp: int = VITAL_MAX_HP

    def __post_init__(self):
        # Validate lengths
        n = len(self.measurement_dimensions)
        if len(self.normalized_scores) != n or len(self.weights) != n:
            raise ValueError(f"Mismatched dimensions: dims={n}, scores={len(self.normalized_scores)}, weights={len(self.weights)}")

        # Enforce score clamping [0.0 - 5.0]
        self.normalized_scores = [max(0.0, min(5.0, float(s))) for s in self.normalized_scores]

        # Enforce weight normalization
        total_w = sum(self.weights)
        if total_w > 0 and abs(total_w - 1.0) > 0.001:
            self.weights = [round(w / total_w, 4) for w in self.weights]
        elif total_w == 0:
            self.weights = [1.0 / n] * n

        # Invariant check
        if self.vital_hp != VITAL_MAX_HP:
            self.vital_hp = VITAL_MAX_HP

    def compute_geometry(self, radius: float = 100.0, center_x: float = 120.0, center_y: float = 120.0) -> StateGeometry:
        """
        Calculates:
          U = sum(w_i * s_i)
          S_min = min(s_i)
          D = sqrt( sum( w_i * (s_i - U)^2 ) )
        and 2D polygon SVG coords.
        """
        n = len(self.measurement_dimensions)
        u = sum(w * s for w, s in zip(self.weights, self.normalized_scores))

        # Weakest axis
        min_idx = min(range(n), key=lambda i: self.normalized_scores[i])
        s_min = self.normalized_scores[min_idx]
        s_min_code = self.measurement_dimensions[min_idx]

        # Dispersion D
        dispersion_var = sum(w * ((s - u) ** 2) for w, s in zip(self.weights, self.normalized_scores))
        dispersion_d = math.sqrt(max(0.0, dispersion_var))

        # 2D SVG Polygon
        coords = []
        angle_step = (2.0 * math.pi) / n
        base_angle = -math.pi / 2.0
        for i, s in enumerate(self.normalized_scores):
            ang = base_angle + i * angle_step
            r = (s / 5.0) * radius
            x = center_x + r * math.cos(ang)
            y = center_y + r * math.sin(ang)
            coords.append(f"{round(x, 1)},{round(y, 1)}")

        is_balanced = dispersion_d <= 0.75 and s_min >= 2.5

        return StateGeometry(
            composite_utility_u=u,
            weakest_axis_code=s_min_code,
            weakest_axis_score=s_min,
            weighted_dispersion_d=dispersion_d,
            radar_polygon_svg=" ".join(coords),
            is_balanced=is_balanced
        )

    def compare_with_baseline(self, baseline: "StateVector", epsilon: float = 0.05) -> StateVectorDelta:
        """
        Compares self against baseline:
          Delta S = S(self) - S(baseline)
        Explicitly classifies TRADEOFF vs STRICT_IMPROVEMENT vs REGRESSION.
        Never uses U alone.
        """
        if self.domain_id != baseline.domain_id or self.measurement_dimensions != baseline.measurement_dimensions:
            raise ValueError(f"Cannot directly diff disparate dimensions: {self.measurement_dimensions} vs {baseline.measurement_dimensions}")

        geo_self = self.compute_geometry()
        geo_base = baseline.compute_geometry()

        delta_u = geo_self.composite_utility_u - geo_base.composite_utility_u
        delta_s_min = geo_self.weakest_axis_score - geo_base.weakest_axis_score

        delta_vec = {}
        improved = []
        degraded = []

        for code, s_curr, s_prev in zip(self.measurement_dimensions, self.normalized_scores, baseline.normalized_scores):
            diff = s_curr - s_prev
            delta_vec[code] = diff
            if diff > epsilon:
                improved.append(code)
            elif diff < -epsilon:
                degraded.append(code)

        # Classification logic
        if improved and not degraded:
            classification = TradeoffClassification.PARETO_STRICT_IMPROVEMENT
            pareto = ParetoStatus.NON_DOMINATED
            summary = f"Strict Pareto improvement: enhanced {', '.join(improved)} with zero regressions."
        elif degraded and not improved:
            classification = TradeoffClassification.PARETO_STRICT_REGRESSION
            pareto = ParetoStatus.DOMINATED
            summary = f"Strict regression: degraded {', '.join(degraded)} without compensations."
        elif improved and degraded:
            classification = TradeoffClassification.TRADEOFF
            pareto = ParetoStatus.BALANCED_FRONTIER
            summary = (
                f"Tradeoff observed: enhanced {', '.join(improved)} at the cost of {', '.join(degraded)}. "
                f"Composite utility delta: {delta_u:+.2f}."
            )
        else:
            classification = TradeoffClassification.NEUTRAL_EQUIVALENCE
            pareto = ParetoStatus.NON_DOMINATED
            summary = "Neutral equivalence: all state changes fall within noise margin."

        return StateVectorDelta(
            delta_vector=delta_vec,
            delta_utility_u=delta_u,
            delta_weakest_axis=delta_s_min,
            tradeoff_classification=classification,
            pareto_status=pareto,
            tradeoff_summary=summary,
            improved_dimensions=improved,
            degraded_dimensions=degraded
        )

    def to_dict(self) -> Dict[str, Any]:
        geo = self.compute_geometry()
        return {
            "domain_id": self.domain_id,
            "subject_id": self.subject_id,
            "timestamp": self.timestamp,
            "measurement_dimensions": self.measurement_dimensions,
            "normalized_scores": self.normalized_scores,
            "raw_values": self.raw_values,
            "units": self.units,
            "weights": self.weights,
            "confidence": self.confidence,
            "uncertainty": self.uncertainty,
            "invariants": self.invariants,
            "configuration_snapshot": self.configuration_snapshot,
            "runtime_context": self.runtime_context,
            "geometry": geo.to_dict(),
            "vital_hp": self.vital_hp,
            "conduct_badge": "[ACT] [!LCK] [CNF] [!HAZ] [VIT 6HP]"
        }
