#!/usr/bin/env python3
"""
==============================================================================
KARAF: ALGORITHM E — UMA CONTENTION PREDICTOR
==============================================================================
Module: krystal_kernel/karaf/uma_contention.py
Description:
    Models coupled Host RAM and integrated GPU shared memory (Unified Memory
    Architecture / Intel Iris Xe UMA).

    Estimates:
      - DER_UMA_PRESSURE
      - PRED_UMA_STALL_RISK = f(RAM_used, GPU_shared, dirty_pages, swap, page_faults)

    Generates shadow mitigation proposals:
      - WOULD_UNLOAD_MODEL
      - WOULD_MOVE_MODEL
      - WOULD_SCALE_WORKERS
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
class UmaContentionReport:
    ram_used_mb: float
    ram_total_mb: float
    gpu_shared_used_mb: float
    swap_activity_mb: float
    der_uma_pressure: float
    pred_uma_stall_risk: float  # 0.0 to 1.0
    recommended_mitigation: str
    shadow_decision: ShadowDecision
    vital_max_hp: int = VITAL_MAX_HP

    def __post_init__(self):
        assert self.vital_max_hp == 6, "VITAL_MAX_HP must equal 6"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ram_used_mb": self.ram_used_mb,
            "ram_total_mb": self.ram_total_mb,
            "gpu_shared_used_mb": self.gpu_shared_used_mb,
            "swap_activity_mb": self.swap_activity_mb,
            "der_uma_pressure": self.der_uma_pressure,
            "pred_uma_stall_risk": self.pred_uma_stall_risk,
            "recommended_mitigation": self.recommended_mitigation,
            "shadow_decision": self.shadow_decision.to_dict(),
            "vital_max_hp": self.vital_max_hp
        }


class UmaContentionPredictor:
    """
    Coupled memory model for Unified Memory Architectures (UMA).
    Detects memory bus saturation, GPU eviction page faults, and swap churn.
    """

    def __init__(self, stall_risk_threshold: float = 0.70):
        assert VITAL_MAX_HP == 6, f"Constitutional invariant breach: VITAL_MAX_HP must be 6, got {VITAL_MAX_HP}"
        self.vital_max_hp = VITAL_MAX_HP
        self.stall_risk_threshold = stall_risk_threshold

    def evaluate_contention(
        self,
        snapshot: ObservationFabricSnapshot,
        loaded_cascade_models_mb: float = 0.0
    ) -> UmaContentionReport:
        """
        Calculates UMA pressure and predicts memory bus stall risk.
        """
        ram_used = snapshot.get_obs("OBS_RAM_USED", 8192.0)
        gpu_uma = snapshot.get_obs("OBS_GPU_UMA_ALLOC", 1024.0)
        swap_used = snapshot.get_obs("OBS_SWAP_ACTIVITY", 0.0)
        cascade_node_mem = snapshot.get_obs("OBS_CASCADE_NODE_MEMORY", loaded_cascade_models_mb)

        hw_ctx = snapshot.hardware_context
        ram_total = max(1.0, float(hw_ctx.get("ram_total_mb", 16384.0)))

        # 1. Coupled UMA Pressure: (Host RAM used + GPU UMA allocated) / Total RAM
        total_coupled = ram_used + gpu_uma
        uma_pressure = min(1.0, total_coupled / ram_total)

        # 2. Stall risk calculation
        # Factors: high UMA pressure (>75%), presence of swap activity, cascade model footprint
        swap_factor = min(1.0, swap_used / 1024.0) * 0.4
        footprint_factor = min(0.3, (cascade_node_mem / ram_total) * 0.5)

        stall_risk = 0.0
        if uma_pressure > 0.85:
            stall_risk = 0.75 + (uma_pressure - 0.85) * 1.5 + swap_factor
        elif uma_pressure > 0.70:
            stall_risk = 0.35 + (uma_pressure - 0.70) * 2.0 + swap_factor
        else:
            stall_risk = (uma_pressure * 0.3) + swap_factor

        stall_risk = round(min(1.0, max(0.0, stall_risk + footprint_factor)), 4)

        # 3. Mitigation proposal synthesis
        if stall_risk >= 0.80:
            mitigation = "UNLOAD_IDLE_CASCADE_MODELS"
            action = ShadowActionType.WOULD_UNLOAD_MODEL
            rationale = (
                f"Severe UMA stall risk ({stall_risk} >= 0.80) with UMA pressure {round(uma_pressure, 3)}. "
                f"Proposing unloading idle model nodes to prevent kernel memory thrashing."
            )
            safety = "PASS"
        elif stall_risk >= 0.60:
            mitigation = "MOVE_MODEL_PLACEMENT_TO_CPU_RAM"
            action = ShadowActionType.WOULD_MOVE_MODEL
            rationale = (
                f"Moderate UMA stall risk ({stall_risk}). "
                f"Proposing moving non-critical GPU models to CPU_SIMD."
            )
            safety = "PASS"
        elif uma_pressure > 0.80:
            mitigation = "REDUCE_WORKER_CONCURRENCY"
            action = ShadowActionType.WOULD_SCALE_WORKERS
            rationale = (
                f"High UMA footprint ({round(uma_pressure, 3)}). "
                f"Proposing reducing worker thread count to decrease active stack/buffer allocations."
            )
            safety = "PASS"
        else:
            mitigation = "NOMINAL_KEEP_ALLOCATION"
            action = ShadowActionType.WOULD_CHANGE_PROFILE
            rationale = f"UMA status nominal: pressure={round(uma_pressure, 3)}, stall_risk={stall_risk}."
            safety = "PASS"

        shadow_dec = ShadowDecision(
            action_type=action,
            parameters={
                "uma_pressure": round(uma_pressure, 4),
                "pred_uma_stall_risk": stall_risk,
                "mitigation": mitigation,
                "ram_used_mb": ram_used,
                "gpu_uma_mb": gpu_uma
            },
            rationale=rationale,
            safety_verdict=safety,
            evidence_verdict="SUFFICIENT",
            read_set_versions={"memory": 1, "gpu": 1},
            is_shadow_only=True
        )

        return UmaContentionReport(
            ram_used_mb=ram_used,
            ram_total_mb=ram_total,
            gpu_shared_used_mb=gpu_uma,
            swap_activity_mb=swap_used,
            der_uma_pressure=round(uma_pressure, 4),
            pred_uma_stall_risk=stall_risk,
            recommended_mitigation=mitigation,
            shadow_decision=shadow_dec,
            vital_max_hp=VITAL_MAX_HP
        )
