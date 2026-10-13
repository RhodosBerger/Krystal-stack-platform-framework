#!/usr/bin/env python3
"""
==============================================================================
KARAF: ALGORITHM D — PREDICTIVE THERMAL GOVERNOR
==============================================================================
Module: krystal_kernel/karaf/thermal_predictor.py
Description:
    Predictive thermal modeling predicting temperature T(t + tau) across
    multi-horizon timeframes (micro: 100ms, meso: 500ms, macro: 2000ms).

    Computes:
      - PRED_THERMAL_RISK
      - PRED_THROTTLE_PROBABILITY
    
    Proposes read-only shadow mitigations (WOULD_REDUCE_POWER, WOULD_SCALE_WORKERS).
    Strictly avoids naive single-threshold binary reactions.
==============================================================================
"""

import math
import time
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple

from krystal_kernel.scene_constructor_presets import VITAL_MAX_HP
from krystal_kernel.karaf.types import (
    ObservationFabricSnapshot,
    ShadowDecision,
    ShadowActionType
)


@dataclass
class ThermalHorizonPrediction:
    horizon_name: str
    horizon_ms: float
    predicted_temp_c: float
    throttle_probability: float
    thermal_risk: float  # 0.0 (safe) to 1.0 (imminent thermal throttle)


@dataclass
class ThermalGovernorReport:
    current_temp_c: float
    tj_max_c: float
    package_power_watts: float
    horizons: Dict[str, ThermalHorizonPrediction]
    pred_thermal_risk: float
    pred_throttle_probability: float
    shadow_decision: ShadowDecision
    vital_max_hp: int = VITAL_MAX_HP

    def __post_init__(self):
        assert self.vital_max_hp == 6, "VITAL_MAX_HP must equal 6"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "current_temp_c": self.current_temp_c,
            "tj_max_c": self.tj_max_c,
            "package_power_watts": self.package_power_watts,
            "pred_thermal_risk": self.pred_thermal_risk,
            "pred_throttle_probability": self.pred_throttle_probability,
            "horizons": {
                k: {
                    "horizon_ms": v.horizon_ms,
                    "predicted_temp_c": v.predicted_temp_c,
                    "throttle_probability": v.throttle_probability,
                    "thermal_risk": v.thermal_risk
                } for k, v in self.horizons.items()
            },
            "shadow_decision": self.shadow_decision.to_dict(),
            "vital_max_hp": self.vital_max_hp
        }


class PredictiveThermalGovernor:
    """
    Multi-horizon Newton cooling / heat dissipation predictor.
    Estimates thermal inertia and power injection to anticipate throttling
    before TjMax violation occurs.
    """

    def __init__(
        self,
        horizons_ms: Optional[Dict[str, float]] = None,
        thermal_resistance_c_per_w: float = 1.15,
        thermal_capacitance_j_per_c: float = 24.0
    ):
        assert VITAL_MAX_HP == 6, f"Constitutional invariant breach: VITAL_MAX_HP must be 6, got {VITAL_MAX_HP}"
        self.vital_max_hp = VITAL_MAX_HP
        self.horizons_ms = horizons_ms or {
            "micro": 100.0,
            "meso": 500.0,
            "macro": 2000.0
        }
        self.r_th = thermal_resistance_c_per_w
        self.c_th = thermal_capacitance_j_per_c
        self._temp_history: List[Tuple[int, float]] = []

    def record_temperature(self, temp_c: float, timestamp_ns: Optional[int] = None):
        t_ns = timestamp_ns or time.time_ns()
        self._temp_history.append((t_ns, temp_c))
        if len(self._temp_history) > 64:
            self._temp_history.pop(0)

    def predict(
        self,
        snapshot: ObservationFabricSnapshot,
        worker_count: int = 4
    ) -> ThermalGovernorReport:
        """
        Calculates multi-horizon predictions T(t + tau).
        """
        curr_temp = snapshot.get_obs("OBS_CORE_TEMP", 50.0)
        power_w = snapshot.get_obs("OBS_PACKAGE_POWER", 20.0)
        gpu_load = snapshot.get_obs("OBS_GPU_LOAD", 10.0)
        cpu_util = snapshot.get_obs("OBS_CPU_UTILIZATION", 35.0)

        hw_ctx = snapshot.hardware_context
        tj_max = hw_ctx.get("tj_max_c", 100.0)

        self.record_temperature(curr_temp, snapshot.timestamp_ns)

        # Estimate temperature trend (dT/dt) from history if available
        trend_c_per_sec = 0.0
        if len(self._temp_history) >= 2:
            t0, temp0 = self._temp_history[0]
            t1, temp1 = self._temp_history[-1]
            dt_sec = max(1e-4, (t1 - t0) / 1e9)
            trend_c_per_sec = (temp1 - temp0) / dt_sec

        # Effective thermal power equilibrium: T_steady = T_ambient + Power * R_th
        t_ambient = 32.0
        effective_power = power_w + (gpu_load * 0.08) + (worker_count * 0.5)
        t_steady = t_ambient + (effective_power * self.r_th)

        # Thermal time constant tau_th = R_th * C_th
        tau_th_sec = max(0.5, self.r_th * self.c_th * 0.1)

        horizon_preds: Dict[str, ThermalHorizonPrediction] = {}
        max_risk = 0.0
        max_throttle_prob = 0.0

        for h_name, h_ms in self.horizons_ms.items():
            dt = h_ms / 1000.0
            # First order step response: T(t+dt) = T_steady + (T_curr - T_steady) * exp(-dt / tau_th)
            pred_temp = t_steady + (curr_temp - t_steady) * math.exp(-dt / tau_th_sec)
            pred_temp += trend_c_per_sec * dt * 0.5
            pred_temp = round(pred_temp, 2)

            # Throttle probability based on proximity to TjMax
            headroom = max(0.0, tj_max - pred_temp)
            if headroom <= 0.0:
                throttle_prob = 1.0
                risk = 1.0
            elif headroom < 5.0:
                throttle_prob = 0.85
                risk = 0.90
            elif headroom < 12.0:
                throttle_prob = 0.40
                risk = 0.60
            elif headroom < 20.0:
                throttle_prob = 0.10
                risk = 0.25
            else:
                throttle_prob = 0.01
                risk = 0.05

            max_risk = max(max_risk, risk)
            max_throttle_prob = max(max_throttle_prob, throttle_prob)

            horizon_preds[h_name] = ThermalHorizonPrediction(
                horizon_name=h_name,
                horizon_ms=h_ms,
                predicted_temp_c=pred_temp,
                throttle_probability=round(throttle_prob, 3),
                thermal_risk=round(risk, 3)
            )

        # Synthesize shadow decision
        if max_risk > 0.75:
            action = ShadowActionType.WOULD_REDUCE_POWER
            rationale = (
                f"Elevated thermal risk {round(max_risk, 2)}: Macro predicted temp "
                f"{horizon_preds['macro'].predicted_temp_c}C approaches TjMax ({tj_max}C). "
                f"Proposing PL1 power clamp reduction."
            )
            safety = "PASS"
        elif max_risk > 0.50:
            action = ShadowActionType.WOULD_SCALE_WORKERS
            rationale = (
                f"Moderate thermal accumulation risk {round(max_risk, 2)}. "
                f"Proposing worker contraction to relieve core temperature."
            )
            safety = "PASS"
        else:
            action = ShadowActionType.WOULD_CHANGE_PROFILE
            rationale = (
                f"Thermal status nominal: risk={round(max_risk, 2)}, "
                f"macro temp={horizon_preds['macro'].predicted_temp_c}C (TjMax={tj_max}C)."
            )
            safety = "PASS"

        shadow_dec = ShadowDecision(
            action_type=action,
            parameters={
                "pred_thermal_risk": max_risk,
                "pred_throttle_probability": max_throttle_prob,
                "macro_predicted_temp_c": horizon_preds["macro"].predicted_temp_c,
                "current_temp_c": curr_temp,
                "tj_max_c": tj_max
            },
            rationale=rationale,
            safety_verdict=safety,
            evidence_verdict="SUFFICIENT",
            read_set_versions={"thermal": 1, "power": 1},
            is_shadow_only=True
        )

        return ThermalGovernorReport(
            current_temp_c=curr_temp,
            tj_max_c=tj_max,
            package_power_watts=power_w,
            horizons=horizon_preds,
            pred_thermal_risk=round(max_risk, 3),
            pred_throttle_probability=round(max_throttle_prob, 3),
            shadow_decision=shadow_dec,
            vital_max_hp=VITAL_MAX_HP
        )
