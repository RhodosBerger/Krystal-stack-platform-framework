#!/usr/bin/env python3
"""
==============================================================================
KARAF: SKILL-DRIVEN EXPERIMENT GENERATOR
==============================================================================
Module: krystal_kernel/karaf/skill_experiment_generator.py
Description:
    Formulates scientific, reproducible, controlled ExperimentPlans designed
    to test and falsify RuntimeHypotheses using Antigravity Skills as
    engineering procedures:
      - Maps hypotheses to applicable skill procedures and algorithm templates.
      - Isolates independent, dependent, and controlled variables.
      - Sets explicit falsification and safety abort conditions.
      - Enforces zero direct host mutation (read-only shadow testing).
==============================================================================
"""

import time
import uuid
from typing import Dict, Any, List, Optional

from krystal_kernel.scene_constructor_presets import VITAL_MAX_HP
from krystal_kernel.karaf.types import (
    RuntimeHypothesis,
    ExperimentPlan,
    ObservationFabricSnapshot
)
from krystal_kernel.karaf.template_registry import GLOBAL_TEMPLATE_REGISTRY


class SkillDrivenExperimentGenerator:
    """
    Generates structured, non-mutating experiment plans to test and falsify
    runtime hypotheses.
    """

    CANONICAL_SKILLS_MAP = {
        "WORKER_SATURATION": ["krystal-kernel-and-bot-operations", "krystal-sprint-debugging-and-elemental-runtime"],
        "THERMAL_THROTTLE": ["krystal-kernel-and-bot-operations"],
        "UMA_CONTENTION": ["neural-ascii-engine-development", "krystal-sprint-debugging-and-elemental-runtime"],
        "CASCADE_BOTTLENECK": ["krystal-sprint-debugging-and-elemental-runtime", "procedural-generation-pipelines"]
    }

    def __init__(self):
        assert VITAL_MAX_HP == 6, f"Constitutional invariant breach: VITAL_MAX_HP must be 6, got {VITAL_MAX_HP}"
        self.vital_max_hp = VITAL_MAX_HP

    def generate_experiment(
        self,
        hypothesis: RuntimeHypothesis,
        snapshot: ObservationFabricSnapshot,
        template_id: Optional[str] = None
    ) -> ExperimentPlan:
        """
        Synthesizes a controlled ExperimentPlan to falsify the hypothesis.
        """
        h_id = hypothesis.hypothesis_id
        exp_id = f"exp_{h_id}_{int(time.time())}"

        # Match applicable skills
        skills = ["krystal-sprint-debugging-and-elemental-runtime"]
        if "worker" in hypothesis.hypothesis.lower():
            skills.append("krystal-kernel-and-bot-operations")
            tpl_id = template_id or "worker_elasticity"
            indep = {"worker_count": [hypothesis.runtime_context.get("current_workers", 4), hypothesis.runtime_context.get("candidate_workers", 6)]}
            dep = ["throughput_ops_sec", "p99_latency_ms", "context_switches_sec", "llc_miss_rate"]
            ctrl = {"matrix_dimension": 256, "governor": snapshot.hardware_context.get("governor", "powersave")}
            falsif = "throughput_delta < 0.02 and p99_latency_increase > 0.15"
        elif "thermal" in hypothesis.hypothesis.lower():
            skills.append("krystal-kernel-and-bot-operations")
            tpl_id = template_id or "predictive_thermal_governor"
            indep = {"target_pl1_watts": [15.0, 28.0]}
            dep = ["OBS_CORE_TEMP", "predicted_temp_c", "throttle_flag"]
            ctrl = {"worker_count": 4}
            falsif = "macro_predicted_temp_c - observed_temp_c > 8.0"
        elif "uma" in hypothesis.hypothesis.lower() or "memory" in hypothesis.hypothesis.lower():
            skills.append("neural-ascii-engine-development")
            tpl_id = template_id or "uma_contention"
            indep = {"gpu_shared_buffer_mb": [256.0, 1024.0]}
            dep = ["DER_UMA_PRESSURE", "page_fault_rate", "stall_cycles"]
            ctrl = {"ram_total_mb": snapshot.hardware_context.get("ram_total_mb", 16384.0)}
            falsif = "DER_UMA_PRESSURE > 0.85 without stall_cycles elevation"
        else:
            tpl_id = template_id or "adaptive_resource_envelope"
            indep = {"profile": ["CONSERVATIVE", "AGGRESSIVE"]}
            dep = ["utility_score", "throughput", "thermal_headroom"]
            ctrl = {"vital_max_hp": 6}
            falsif = "measured_utility < baseline_utility"

        baseline = {
            "worker_count": hypothesis.runtime_context.get("current_workers", 4),
            "timestamp_ns": snapshot.timestamp_ns,
            "observations": {k: v.value for k, v in snapshot.observations.items()}
        }

        candidate = {
            "worker_count": hypothesis.runtime_context.get("candidate_workers", 6),
            "mode": "SHADOW_SIMULATION"
        }

        return ExperimentPlan(
            experiment_id=exp_id,
            hypothesis_id=h_id,
            template_id=tpl_id,
            active_skills=skills,
            baseline=baseline,
            candidate=candidate,
            controlled_variables=ctrl,
            independent_variables=indep,
            dependent_variables=dep,
            sample_plan={"trials": 10, "sampling_interval_ms": 50},
            warmup_plan={"warmup_iterations": 3, "cooldown_sec": 1.0},
            falsification_condition=falsif,
            abort_condition="OBS_CORE_TEMP > 92.0 or vital_max_hp != 6",
            expected_output={"utility_gain_pct": 12.0, "confidence_target": 0.90},
            vital_max_hp=VITAL_MAX_HP
        )
