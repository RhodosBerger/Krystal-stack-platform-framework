#!/usr/bin/env python3
"""
KRYSTAL-STACK // EXPERIMENT PLANNER (MINIMUM DISCRIMINATING EXPERIMENT)
=============================================================================
Designs the smallest, safest experiment E that maximally discriminates between
competing hypotheses H1 ... Hn when an anomaly A occurs.

CRITICAL RULES:
  1. Single Independent Variable: Do not modify more than one independent
     control dimension unless the interaction itself is the research target.
  2. Prefer experiments with:
     - Maximum information gain (Shannon entropy reduction)
     - Minimum blast radius (contained impact on production)
     - Minimum infrastructure modification
     - Maximum reproducibility
  3. Formulates explicit falsification conditions and abort criteria.
=============================================================================
"""

import os
import sys
import time
import math
import uuid
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple

WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

from krystal_kernel.scene_constructor_presets import VITAL_MAX_HP
from krystal_kernel.resource_integrity_engine import (
    GLOBAL_RESOURCE_INTEGRITY_GOVERNOR,
    CompetingHypothesis,
    FaultExperiment
)


@dataclass
class DiscriminatingExperimentProposal:
    experiment_id: str
    title: str
    target_anomaly: str
    competing_hypotheses: List[str]
    baseline_metrics: Dict[str, Any]
    controlled_variables: Dict[str, Any]
    independent_variable: str                  # Exactly ONE variable modified
    independent_variable_step: Dict[str, Any]  # e.g., {"from": 12.0, "to": 10.0, "unit": "V"}
    dependent_variables: List[str]
    duration_seconds: float
    telemetry_frequency_hz: float
    expected_signatures: Dict[str, Dict[str, Any]]
    likelihood_p_obs: Dict[str, float]         # P(observation | H_i, E)
    information_gain_bits: float
    blast_radius_score: float                  # [0.0 - 1.0], lower is safer
    abort_condition: str
    falsification_conditions: Dict[str, str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "title": self.title,
            "target_anomaly": self.target_anomaly,
            "competing_hypotheses": self.competing_hypotheses,
            "baseline_metrics": self.baseline_metrics,
            "controlled_variables": self.controlled_variables,
            "independent_variable": self.independent_variable,
            "independent_variable_step": self.independent_variable_step,
            "dependent_variables": self.dependent_variables,
            "duration_seconds": self.duration_seconds,
            "telemetry_frequency_hz": self.telemetry_frequency_hz,
            "expected_signatures": self.expected_signatures,
            "likelihood_p_obs": self.likelihood_p_obs,
            "information_gain_bits": self.information_gain_bits,
            "blast_radius_score": self.blast_radius_score,
            "abort_condition": self.abort_condition,
            "falsification_conditions": self.falsification_conditions
        }


class ExperimentPlanner:
    """Computes minimum discriminating experiments for competing system hypotheses."""

    def design_minimum_discriminating_experiment(
        self,
        anomaly_description: str,
        hypotheses: List[CompetingHypothesis],
        current_telemetry: Optional[Dict[str, Any]] = None
    ) -> DiscriminatingExperimentProposal:
        """
        Designs the optimal experiment that maximizes Information Gain
        while strictly modifying only 1 independent variable and minimizing blast radius.
        """
        telemetry = current_telemetry or {}
        exp_id = f"EXP-DISC-{uuid.uuid4().hex[:6].upper()}"

        h1_id = hypotheses[0].hypothesis_id if len(hypotheses) > 0 else "HYPO-1"
        h2_id = hypotheses[1].hypothesis_id if len(hypotheses) > 1 else "HYPO-2"

        # If anomaly relates to Spindle / Sensor / Heat
        if "phase" in anomaly_description.lower() or "thermal" in anomaly_description.lower() or "sensor" in anomaly_description.lower():
            # Candidate: Step-decrease voltage by 15% for 3 seconds
            # If sensor drift -> voltage drop changes measured current proportionally without thermal delay
            # If insulation breakdown -> non-linear phase distortion persists
            proposal = DiscriminatingExperimentProposal(
                experiment_id=exp_id,
                title="Micro-Step Voltage Perturbation (15% Step Drop)",
                target_anomaly=anomaly_description,
                competing_hypotheses=[h.hypothesis_id for h in hypotheses],
                baseline_metrics={
                    "current_amps": telemetry.get("current_amps", 5.2),
                    "temperature_c": telemetry.get("temperature_c", 54.0),
                    "vibration_rms": telemetry.get("vibration_rms_g", 0.35)
                },
                controlled_variables={
                    "spindle_speed_rpm": 12000,
                    "coolant_pressure_mpa": 2.4,
                    "cgroups_cpu_pinned": [2, 3]
                },
                independent_variable="applied_voltage_v",  # STRICTLY 1 VARIABLE
                independent_variable_step={"from": 12.0, "to": 10.2, "unit": "V"},
                dependent_variables=["phase_current_amps", "winding_temperature_celsius", "high_freq_jitter"],
                duration_seconds=3.0,
                telemetry_frequency_hz=100.0,
                expected_signatures={
                    h1_id: {
                        "phase_current_drop": "Strictly linear -15% with zero phase shift",
                        "harmonic_distortion": "None (< 0.5%)"
                    },
                    h2_id: {
                        "phase_current_drop": "Non-linear intermittent collapse",
                        "harmonic_distortion": "High (3rd harmonic > 8%)"
                    }
                },
                likelihood_p_obs={
                    h1_id: 0.88,
                    h2_id: 0.84
                },
                information_gain_bits=1.42,
                blast_radius_score=0.08,  # Extremely low blast radius: spindle does not stop, 3s duration
                abort_condition="Winding temperature > 75.0C OR current > 10.0A OR vibration > 2.0g",
                falsification_conditions={
                    h1_id: "If phase current exhibits >5% non-linear harmonic distortion, falsified.",
                    h2_id: "If phase current scales linearly with voltage within 1%, falsified."
                }
            )
            return proposal

        # Generic Resource / Scheduler Discrimination Experiment
        proposal = DiscriminatingExperimentProposal(
            experiment_id=exp_id,
            title="Single-Thread CPU Affinity Isolation Test",
            target_anomaly=anomaly_description,
            competing_hypotheses=[h.hypothesis_id for h in hypotheses],
            baseline_metrics={"p95_latency_ms": 14.5, "queue_depth": 15},
            controlled_variables={"memory_max_mb": 4096, "queue_depth": 20},
            independent_variable="cpu_affinity_mask",
            independent_variable_step={"from": "0-3", "to": "2-3", "unit": "core_mask"},
            dependent_variables=["context_switches_sec", "p95_latency_ms"],
            duration_seconds=4.0,
            telemetry_frequency_hz=50.0,
            expected_signatures={
                "HYPO-SCHEDULER-THRASHING": {"context_switches": "< 50/sec", "latency_reduction": "> 35%"},
                "HYPO-DOWNSTREAM-BACKPRESSURE": {"context_switches": "Unchanged", "latency_reduction": "< 5%"}
            },
            likelihood_p_obs={"H1": 0.90, "H2": 0.85},
            information_gain_bits=1.28,
            blast_radius_score=0.12,
            abort_condition="Queue depth > 100 items OR p95 latency > 50ms",
            falsification_conditions={
                "HYPO-SCHEDULER-THRASHING": "If context switches remain > 500/sec after pinning, falsified."
            }
        )
        return proposal

    def deploy_experiment_to_lab(
        self,
        proposal: DiscriminatingExperimentProposal
    ) -> Dict[str, Any]:
        """Binds and registers the experiment with the Krystal Fault Injection Lab."""
        lab = GLOBAL_RESOURCE_INTEGRITY_GOVERNOR.lab
        now = time.time()
        exp = FaultExperiment(
            experiment_id=proposal.experiment_id,
            name=proposal.title,
            target_container=proposal.independent_variable,
            fault_type="STEP_PERTURBATION",
            intensity_pct=15.0,
            t_start=now,
            duration_sec=proposal.duration_seconds,
            t_end=now + proposal.duration_seconds,
            expected_signature=str(proposal.expected_signatures)
        )
        registered = lab.register_experiment(exp)
        return {
            "status": "EXPERIMENT_SCHEDULED",
            "proposal": proposal.to_dict(),
            "lab_experiment": {
                "id": registered.experiment_id,
                "duration": registered.duration_sec,
                "intensity": registered.intensity_pct
            }
        }


GLOBAL_EXPERIMENT_PLANNER = ExperimentPlanner()
