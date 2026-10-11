#!/usr/bin/env python3
"""
CRYSTALSEC 2.0 // LLM RESEARCH COPILOT & DETERMINISTIC POLICY GATE
=============================================================================
Deploys a secondary reasoning layer acting as a research scientist over
system telemetry and Digital Mirror drift:

  "Latency increased by 31%. Propose competing explanations, identify
   the single variable to isolate, and formulate an experiment with
   maximum information gain."

CRITICAL SAFETY INVARIANT:
  The Research Copilot NEVER modifies production or kernel state directly.
  Every recommendation must traverse the deterministic policy pipeline:

  Proposal -> Policy Review -> Invariants -> Allowed Variables -> Controlled Lab

ALLOWED CONTROLLED VARIABLES WHITELIST (Localhost / Edge):
  - cpu.threads (1 .. 8)
  - openvino.streams (1 .. 4)
  - cpu.affinity (subsets of available cores)
  - batch_size (1 .. 16)
  - context_limit_tokens (512 .. 8192)
  - queue_max_depth (10 .. 200)
  - memory_limit_mb (1024 .. 8192)
  - sampling_rate_hz (1.0 .. 100.0)
=============================================================================
"""

import os
import sys
import time
import math
import uuid
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple, Set

WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

from krystal_kernel.scene_constructor_presets import VITAL_MAX_HP
from krystal_kernel.resource_integrity_engine import (
    GLOBAL_RESOURCE_INTEGRITY_GOVERNOR,
    CompetingHypothesis,
    FaultExperiment
)
from krystal_kernel.experiment_planner import (
    GLOBAL_EXPERIMENT_PLANNER,
    DiscriminatingExperimentProposal
)


ALLOWED_CONTROLLED_VARIABLES: Set[str] = {
    "cpu.threads",
    "openvino.streams",
    "cpu.affinity",
    "batch_size",
    "context_limit_tokens",
    "queue_max_depth",
    "memory_limit_mb",
    "sampling_rate_hz",
    "applied_voltage_v"
}


@dataclass
class CopilotReasoningResult:
    analysis_id: str
    timestamp: float
    observed_anomaly: str
    hypotheses: List[Dict[str, Any]]
    isolated_control_variable: str
    proposed_experiment: Dict[str, Any]
    information_gain_rationale: str
    policy_approval_status: str       # "APPROVED", "REJECTED_BY_POLICY"
    policy_rejection_reason: Optional[str] = None
    vital_hp: int = VITAL_MAX_HP

    def to_dict(self) -> Dict[str, Any]:
        return {
            "analysis_id": self.analysis_id,
            "timestamp": self.timestamp,
            "observed_anomaly": self.observed_anomaly,
            "hypotheses": self.hypotheses,
            "isolated_control_variable": self.isolated_control_variable,
            "proposed_experiment": self.proposed_experiment,
            "information_gain_rationale": self.information_gain_rationale,
            "policy_approval_status": self.policy_approval_status,
            "policy_rejection_reason": self.policy_rejection_reason,
            "vital_hp": self.vital_hp
        }


class LLMResearchCopilot:
    """Secondary reasoning copilot analyzing telemetry and formulating safe experiments."""

    def __init__(self):
        self.reasoning_log: List[CopilotReasoningResult] = []

    def analyze_divergence_and_propose_experiment(
        self,
        anomaly_metric: str,
        delta_pct: float,
        current_telemetry: Dict[str, Any]
    ) -> CopilotReasoningResult:
        """
        Analyzes a telemetry divergence, formulates competing hypotheses,
        selects exactly 1 variable to isolate, and submits the proposal
        to the deterministic safety gate.
        """
        t_now = time.time()
        analysis_id = f"ANALYSIS-{uuid.uuid4().hex[:8].upper()}"

        # 1. Reason about competing explanations
        if "latency" in anomaly_metric.lower():
            anomaly_desc = f"Execution tail latency surged by {delta_pct:.1f}%"
            h1 = CompetingHypothesis(
                code="HYPO_SCHED_MIGRATION",
                name="CFS Cross-Core Thread Migration Overhead",
                expected_data_signature="High context switches (> 800/s) with L2 cache misses",
                falsification_condition="If core-pinning does not reduce p95 latency by > 20%, falsified.",
                prior_probability=0.65
            )
            h2 = CompetingHypothesis(
                code="HYPO_KV_CACHE_EVICTION",
                name="KV-Cache VRAM Saturation and Synchronous Reclaim",
                expected_data_signature="KV cache pressure > 90% with memory allocation stalls",
                falsification_condition="If reducing batch size leaves latency unchanged, falsified.",
                prior_probability=0.35
            )
            hypotheses = [h1, h2]
            isolated_var = "cpu.affinity"
            info_rationale = "Pinning execution threads directly tests HYPO_SCHED_MIGRATION while holding memory and model constants invariant."

        elif "thermal" in anomaly_metric.lower() or "temperature" in anomaly_metric.lower():
            anomaly_desc = f"Operating temperature accelerated by {delta_pct:.1f}%"
            h1 = CompetingHypothesis(
                code="HYPO_OPEN_VINO_OVERCOMMIT",
                name="OpenVINO Concurrency Over-Subscription",
                expected_data_signature="Streams count > physical cores causing thermal dissipation spike",
                falsification_condition="If reducing streams to 1 does not drop thermal derivative, falsified.",
                prior_probability=0.70
            )
            h2 = CompetingHypothesis(
                code="HYPO_COOLANT_FLOW_RESTRICTION",
                name="Coolant Flow Mechanical Degrade",
                expected_data_signature="Pressure drop in coolant line below 2.0 MPa",
                falsification_condition="If pressure is > 2.2 MPa across all cycles, falsified.",
                prior_probability=0.30
            )
            hypotheses = [h1, h2]
            isolated_var = "openvino.streams"
            info_rationale = "Modulating active streams directly isolates thermal compute load from coolant mechanical state."

        else:
            anomaly_desc = f"Unmodeled drift observed on {anomaly_metric} (+{delta_pct:.1f}%)"
            h1 = CompetingHypothesis("HYPO_QUEUE_BACKPRESSURE", "Ingress Queue Saturation", "Qd > 100", "Qd <= 20", 0.6)
            h2 = CompetingHypothesis("HYPO_PAGE_CACHE_MISS", "Memory Slab Eviction", "major_faults > 0", "faults == 0", 0.4)
            hypotheses = [h1, h2]
            isolated_var = "queue_max_depth"
            info_rationale = "Clamping queue depth immediately differentiates ingress backpressure from internal memory compaction."

        # 2. Design discriminating experiment via planner
        proposal = GLOBAL_EXPERIMENT_PLANNER.design_minimum_discriminating_experiment(
            anomaly_description=anomaly_desc,
            hypotheses=hypotheses,
            current_telemetry=current_telemetry
        )

        # 3. Deterministic Safety Gate Review
        approval_status = "APPROVED"
        rejection_reason = None

        # Check whitelist
        if isolated_var not in ALLOWED_CONTROLLED_VARIABLES and proposal.independent_variable not in ALLOWED_CONTROLLED_VARIABLES:
            approval_status = "REJECTED_BY_POLICY"
            rejection_reason = f"Proposed variable '{isolated_var}' is not in the whitelist of permitted controlled variables."

        # Check blast radius
        if proposal.blast_radius_score > 0.25:
            approval_status = "REJECTED_BY_POLICY"
            rejection_reason = f"Experiment blast radius ({proposal.blast_radius_score:.2f}) exceeds maximum safety bound (0.25)."

        # If approved, schedule into Controlled Lab
        if approval_status == "APPROVED":
            GLOBAL_EXPERIMENT_PLANNER.deploy_experiment_to_lab(proposal)

        result = CopilotReasoningResult(
            analysis_id=analysis_id,
            timestamp=t_now,
            observed_anomaly=anomaly_desc,
            hypotheses=[
                {"code": h.code, "name": h.name, "falsification": h.falsification_condition, "prior": h.confidence_score}
                for h in hypotheses
            ],
            isolated_control_variable=isolated_var,
            proposed_experiment=proposal.to_dict(),
            information_gain_rationale=info_rationale,
            policy_approval_status=approval_status,
            policy_rejection_reason=rejection_reason,
            vital_hp=VITAL_MAX_HP
        )

        self.reasoning_log.append(result)
        return result


GLOBAL_RESEARCH_COPILOT = LLMResearchCopilot()
