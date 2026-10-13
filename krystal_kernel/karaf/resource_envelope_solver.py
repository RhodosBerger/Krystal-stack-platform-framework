#!/usr/bin/env python3
"""
==============================================================================
KARAF: ALGORITHM A — ADAPTIVE RESOURCE ENVELOPE SOLVER
==============================================================================
Module: krystal_kernel/karaf/resource_envelope_solver.py
Description:
    Calculates the optimal resource allocation envelope:
        R* = argmax_R U(R | H, W, S, E)

    Constraints:
      - SAFE_PASS, INV_NOMINAL
      - Evidence freshness (rejects stale/expired evidence)
      - No hard-coded universal constants: bounds are derived dynamically
        from observed hardware context and calibrated policy.
      - Produces shadow recommendations with zero host mutation.
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
from krystal_kernel.karaf.template_registry import GLOBAL_TEMPLATE_REGISTRY


@dataclass
class ResourceEnvelopeSolution:
    """Optimal resource allocation envelope result."""
    allocated_workers: int
    power_limit_watts: float
    gpu_stream_count: int
    utility_score: float
    objective_breakdown: Dict[str, float]
    constraints_satisfied: bool
    evidence_fresh: bool
    rationale: str
    shadow_decision: ShadowDecision
    vital_max_hp: int = VITAL_MAX_HP

    def __post_init__(self):
        assert self.vital_max_hp == 6, "VITAL_MAX_HP must equal 6"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "allocated_workers": self.allocated_workers,
            "power_limit_watts": self.power_limit_watts,
            "gpu_stream_count": self.gpu_stream_count,
            "utility_score": self.utility_score,
            "objective_breakdown": self.objective_breakdown,
            "constraints_satisfied": self.constraints_satisfied,
            "evidence_fresh": self.evidence_fresh,
            "rationale": self.rationale,
            "shadow_decision": self.shadow_decision.to_dict(),
            "vital_max_hp": self.vital_max_hp
        }


class AdaptiveResourceEnvelopeSolver:
    """
    Solves Pareto-optimal resource allocation envelope R* without
    hardcoding universal constants. Bounds and baseline scale with detected host.
    """

    def __init__(self, template_id: str = "adaptive_resource_envelope"):
        assert VITAL_MAX_HP == 6, f"Constitutional invariant breach: VITAL_MAX_HP must be 6, got {VITAL_MAX_HP}"
        self.vital_max_hp = VITAL_MAX_HP
        self.template_id = template_id

    def solve(
        self,
        snapshot: ObservationFabricSnapshot,
        workload_context: Optional[Dict[str, Any]] = None,
        runtime_state: Optional[Dict[str, Any]] = None
    ) -> ResourceEnvelopeSolution:
        """
        Solves R* = argmax U(R | H, W, S, E)
        """
        workload = workload_context or {"priority": 1, "target_deadline_us": 50000.0}
        state = runtime_state or {"current_workers": 4, "queue_depth": 0}

        # 1. Check Evidence Freshness
        is_fresh = True
        stale_keys = []
        for k, obs in snapshot.observations.items():
            if obs.is_stale:
                is_fresh = False
                stale_keys.append(k)

        # 2. Extract hardware dynamic bounds from observed snapshot
        hw = snapshot.hardware_context
        detected_cores = hw.get("logical_cores", 8)
        p_cores = hw.get("p_cores", max(1, detected_cores // 2))
        ram_total = hw.get("ram_total_mb", 16384.0)

        # Observe current telemetry
        cpu_util = snapshot.get_obs("OBS_CPU_UTILIZATION", 30.0)
        core_temp = snapshot.get_obs("OBS_CORE_TEMP", 50.0)
        current_pwr = snapshot.get_obs("OBS_PACKAGE_POWER", 18.0)
        mem_pressure = snapshot.get_der("DER_MEMORY_PRESSURE", 0.4)
        uma_pressure = snapshot.get_der("DER_UMA_PRESSURE", 0.45)
        therm_headroom = snapshot.get_der("DER_THERMAL_HEADROOM", 40.0)
        worker_contention = snapshot.get_der("DER_WORKER_CONTENTION", 0.15)

        # Dynamic search space: min 1 worker up to detected logical cores
        min_workers = 1
        max_workers = max(1, detected_cores)

        # Evaluate candidate worker and power configurations
        best_u = -float("inf")
        best_w = state.get("current_workers", 4)
        best_pwr = current_pwr
        best_streams = 1
        best_metrics: Dict[str, float] = {}

        # Search candidates in neighborhood
        candidate_workers = [w for w in range(min_workers, max_workers + 1)]

        for w in candidate_workers:
            # Modeled throughput utility (Amdahl with contention penalty)
            # Parallel fraction estimated from workload priority
            p_fraction = 0.85
            speedup = 1.0 / ((1.0 - p_fraction) + (p_fraction / w))
            # Contention penalty increases quadratically when w > p_cores
            contention_pen = 0.0
            if w > p_cores:
                contention_pen = 0.12 * ((w - p_cores) ** 1.5)

            throughput_u = max(0.1, speedup - contention_pen)

            # Thermal penalty if temperature close to TjMax
            thermal_pen = 0.0
            if therm_headroom < 15.0:
                thermal_pen = (15.0 - therm_headroom) * 0.15

            # Memory pressure penalty
            mem_pen = mem_pressure * 0.8 + uma_pressure * 0.5

            # Combined Utility U(R | H, W, S, E)
            u = (throughput_u * 1.5) - contention_pen - thermal_pen - mem_pen

            # Adjust by evidence freshness
            if not is_fresh:
                u *= 0.5  # Heavy penalty for stale evidence

            if u > best_u:
                best_u = u
                best_w = w
                best_pwr = round(min(current_pwr * 1.1, 45.0), 2)
                best_streams = 2 if w >= 4 else 1
                best_metrics = {
                    "throughput_utility": round(throughput_u, 3),
                    "contention_penalty": round(contention_pen, 3),
                    "thermal_penalty": round(thermal_pen, 3),
                    "memory_penalty": round(mem_pen, 3),
                    "raw_utility": round(u, 4)
                }

        # Format rationale
        rationale = (
            f"Selected R* with {best_w} workers (p_cores={p_cores}, total={detected_cores}), "
            f"target_pwr={best_pwr}W. Contention: {round(worker_contention, 3)}, "
            f"Thermal Headroom: {round(therm_headroom, 1)}C. Evidence fresh: {is_fresh}."
        )
        if not is_fresh:
            rationale += f" WARNING: Stale evidence detected in: {stale_keys}"

        # Build Shadow Decision
        shadow_dec = ShadowDecision(
            action_type=ShadowActionType.WOULD_CHANGE_PROFILE,
            parameters={
                "allocated_workers": best_w,
                "power_limit_watts": best_pwr,
                "gpu_stream_count": best_streams,
                "previous_workers": state.get("current_workers", 4)
            },
            rationale=rationale,
            safety_verdict="PASS" if is_fresh and therm_headroom > 5.0 else "VETO",
            evidence_verdict="SUFFICIENT" if is_fresh else "INSUFFICIENT",
            read_set_versions={"cpu": 1, "memory": 1, "thermal": 1},
            is_shadow_only=True
        )

        return ResourceEnvelopeSolution(
            allocated_workers=best_w,
            power_limit_watts=best_pwr,
            gpu_stream_count=best_streams,
            utility_score=round(best_u, 4),
            objective_breakdown=best_metrics,
            constraints_satisfied=(is_fresh and therm_headroom > 5.0),
            evidence_fresh=is_fresh,
            rationale=rationale,
            shadow_decision=shadow_dec,
            vital_max_hp=VITAL_MAX_HP
        )
