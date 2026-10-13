#!/usr/bin/env python3
"""
==============================================================================
KARAF: ALGORITHM B — WORKER ELASTICITY GRADIENT
==============================================================================
Module: krystal_kernel/karaf/worker_elasticity.py
Description:
    Estimates the marginal utility gradient:
        G_worker = dUtility / dWorkers

    Evaluates candidate neighborhood {-2, -1, 0, +1, +2} relative to current
    worker topology and classifies state into:
      - SCALE_UP_SUPPORTED
      - SATURATION_REGION
      - CONTENTION_REGION

    Context-bound to hardware_class, workload_class, and policy_version.
    Strictly read-only / shadow execution.
==============================================================================
"""

import math
import time
from typing import Dict, Any, List, Optional, Tuple

from krystal_kernel.scene_constructor_presets import VITAL_MAX_HP
from krystal_kernel.karaf.types import (
    WorkerTopology,
    ElasticityClassification,
    WorkerElasticityResult,
    ObservationFabricSnapshot,
    ShadowDecision,
    ShadowActionType
)


class WorkerElasticityGradient:
    """
    Evaluates worker elasticity gradient G_worker over local neighborhood.
    Prevents thread oversubscription and detects lock/cache contention inflection points.
    """

    def __init__(self, noise_floor: float = 0.035, epsilon: float = 0.08):
        assert VITAL_MAX_HP == 6, f"Constitutional invariant breach: VITAL_MAX_HP must be 6, got {VITAL_MAX_HP}"
        self.vital_max_hp = VITAL_MAX_HP
        self.noise_floor = noise_floor
        self.epsilon = epsilon

    def evaluate_elasticity(
        self,
        current_topology: WorkerTopology,
        snapshot: ObservationFabricSnapshot,
        workload_class: str = "COMPUTE_BOUND_CONVOLUTION",
        neighborhood: Optional[List[int]] = None
    ) -> WorkerElasticityResult:
        """
        Calculates G_worker = dUtility / dWorkers across neighborhood.
        """
        curr_w = current_topology.worker_count
        hw = snapshot.hardware_context
        total_cores = hw.get("logical_cores", 8)
        p_cores = hw.get("p_cores", max(1, total_cores // 2))
        hw_class = f"x86_hybrid_{p_cores}P_{total_cores - p_cores}E"

        deltas = neighborhood if neighborhood is not None else [-2, -1, 0, 1, 2]

        # Extract current observation values
        c_util = snapshot.get_obs("OBS_CPU_UTILIZATION", 40.0)
        c_ctx = snapshot.get_obs("OBS_CONTEXT_SWITCH_RATE", 1200.0)
        mem_pressure = snapshot.get_der("DER_MEMORY_PRESSURE", 0.3)
        contention_der = snapshot.get_der("DER_WORKER_CONTENTION", 0.1)

        # Baseline utility at curr_w
        base_speedup = curr_w / (1.0 + (curr_w - 1) * 0.12)
        base_contention = (c_ctx / 5000.0) + (max(0, curr_w - p_cores) * 0.25)
        base_utility = max(0.1, base_speedup - base_contention)

        best_cand = curr_w
        best_grad = 0.0
        gradient_map: Dict[int, float] = {}

        for d in deltas:
            cand_w = curr_w + d
            if cand_w < 1 or cand_w > (total_cores * 2):
                continue
            if cand_w == curr_w:
                gradient_map[cand_w] = 0.0
                continue

            # Model candidate performance under hardware physics
            cand_speedup = cand_w / (1.0 + (cand_w - 1) * 0.12)

            # High contention penalty if candidate workers exceed physical/P-core limit
            cand_contention = (c_ctx / 5000.0) * (cand_w / float(curr_w))
            if cand_w > p_cores:
                # Cache line bouncing, lock preemption, memory bus saturation
                cand_contention += 0.35 * ((cand_w - p_cores) ** 1.8)

            cand_utility = max(0.01, cand_speedup - cand_contention)

            # Gradient Gw = (U_cand - U_base) / (W_cand - W_base)
            gw = (cand_utility - base_utility) / float(cand_w - curr_w)
            gradient_map[cand_w] = round(gw, 4)

        # Choose the gradient for the immediate step (+1 or -1 if +1 is worse)
        target_step = 1 if (curr_w + 1) in gradient_map else -1
        effective_gw = gradient_map.get(curr_w + target_step, 0.0)

        # Classify region
        if effective_gw > self.epsilon:
            classification = ElasticityClassification.SCALE_UP_SUPPORTED
            target_workers = curr_w + 1
        elif abs(effective_gw) <= self.noise_floor:
            classification = ElasticityClassification.SATURATION_REGION
            target_workers = curr_w
        else:
            classification = ElasticityClassification.CONTENTION_REGION
            # If in contention region, recommended action is to scale DOWN
            target_workers = max(1, curr_w - 1)

        # Metrics delta
        metrics_delta = {
            "effective_gradient_gw": effective_gw,
            "target_workers": target_workers,
            "current_workers": curr_w,
            "delta_workers": target_workers - curr_w,
            "modeled_contention": round(base_contention, 3),
            "p_core_threshold": p_cores,
            "logical_core_ceiling": total_cores
        }

        confidence = 0.94 if snapshot.get_obs("OBS_CPU_UTILIZATION") is not None else 0.60

        return WorkerElasticityResult(
            current_workers=curr_w,
            candidate_workers=target_workers,
            gradient=effective_gw,
            classification=classification,
            metrics_delta=metrics_delta,
            confidence=confidence,
            hardware_class=hw_class,
            workload_class=workload_class,
            vital_max_hp=VITAL_MAX_HP
        )

    def create_shadow_decision(self, result: WorkerElasticityResult) -> ShadowDecision:
        """Translates elasticity result into a non-mutating ShadowDecision."""
        if result.classification == ElasticityClassification.SCALE_UP_SUPPORTED:
            action = ShadowActionType.WOULD_SCALE_WORKERS
            rationale = (
                f"Scale up supported: G_worker={result.gradient} > {self.epsilon}. "
                f"Proposing worker expansion from {result.current_workers} to {result.candidate_workers}."
            )
            safety = "PASS"
        elif result.classification == ElasticityClassification.CONTENTION_REGION:
            action = ShadowActionType.WOULD_SCALE_WORKERS
            rationale = (
                f"Contention region detected: G_worker={result.gradient} < -{self.epsilon}. "
                f"Proposing contraction from {result.current_workers} to {result.candidate_workers} to relieve LLC misses."
            )
            safety = "PASS"
        else:
            action = ShadowActionType.WOULD_SCALE_WORKERS
            rationale = (
                f"Saturation region: |G_worker|={abs(result.gradient)} <= {self.noise_floor}. "
                f"Maintaining worker count at {result.current_workers}."
            )
            safety = "PASS"

        return ShadowDecision(
            action_type=action,
            parameters={
                "current_workers": result.current_workers,
                "recommended_workers": result.candidate_workers,
                "classification": result.classification.value,
                "gradient": result.gradient,
                "workload_class": result.workload_class,
                "hardware_class": result.hardware_class
            },
            rationale=rationale,
            safety_verdict=safety,
            evidence_verdict="SUFFICIENT" if result.confidence >= 0.7 else "INSUFFICIENT",
            read_set_versions={"cpu": 1, "scheduler": 1},
            is_shadow_only=True
        )
