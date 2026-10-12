#!/usr/bin/env python3
"""
KRYSTAL-STACK // ANTIGRAVITY CANDIDATE GENERATOR
=============================================================================
Section 7 & 8 Architecture:
Antigravity is NOT a generator of raw actions or direct mutations.
It is a GENERATOR OF TYPED CANDIDATES, EVIDENCE REQUIREMENTS, AND
FALSIFICATION EXPERIMENTS.

The executive authority remains small, deterministic, and auditable in the runtime.
Formal Inclusion Lattice:
  A_possible >= A_safe >= A_evidence_supported >= A_committable
=============================================================================
"""

import os
import sys
import time
import uuid
from typing import Dict, Any, List, Optional, Tuple

WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

from krystal_kernel.scene_constructor_presets import VITAL_MAX_HP
from krystal_kernel.verification.paravirtual_envelope import (
    CandidateAction,
    EvidenceRequirements,
    FalsificationExperiment,
    ReversibilityClass,
    ContextualRiskRegistry,
    ResourceEvidenceEnvelope,
)


class AntigravityCandidateGenerator:
    """
    Antigravity Generator of Typed Candidates, Evidence Requirements,
    and Falsification Experiments.
    """
    def __init__(self, risk_registry: Optional[ContextualRiskRegistry] = None):
        self.risk_registry = risk_registry or ContextualRiskRegistry()
        self.generated_candidates_count: int = 0

    def generate_candidate(
        self,
        tenant_id: str,
        action_type: str,
        current_state_version: int,
        desired_effect: str,
        resource_request: Dict[str, float],
        active_envelopes: List[ResourceEvidenceEnvelope],
        context: Dict[str, str]
    ) -> CandidateAction:
        """
        Formulates a typed CandidateAction with explicit evidence requirements
        and a falsification experiment.
        """
        self.generated_candidates_count += 1
        action_id = f"CAND-{uuid.uuid4().hex[:8].upper()}"

        # Select evidence references from active fresh envelopes
        evidence_ids = [env.envelope_id for env in active_envelopes if env.guest_id == tenant_id]
        if not evidence_ids and active_envelopes:
            evidence_ids = [active_envelopes[0].envelope_id]

        # Determine reversibility class
        if action_type in ("PIN_VCPU_AFFINITY", "SET_SCHED_NICE"):
            reversibility = ReversibilityClass.REVERSIBLE
        elif action_type in ("RESIZE_BALLOON", "SET_CGROUP_MAX"):
            reversibility = ReversibilityClass.COMPENSABLE
        else:
            reversibility = ReversibilityClass.IRREVERSIBLE

        # Derive Evidence Requirements based on contextual risk
        action_sig = f"{action_type}:{tenant_id}:{sorted(resource_request.items())}"
        contextual_risk = self.risk_registry.get_contextual_risk(
            action_signature=action_sig,
            host_state_class=context.get("host_state_class", "NOMINAL"),
            workload_class=context.get("workload_class", "STANDARD"),
            pressure_region=context.get("pressure_region", "NOMINAL"),
            policy_version=context.get("policy_version", "v1.0.0")
        )

        # High risk requires stricter freshness (< 500ms) and additional mandatory dimensions
        max_age = 0.5 if contextual_risk > 0.3 else 1.5
        mandatory_dims = ["cpu_usage_pct", "memory_rss_bytes"]
        if contextual_risk > 0.3:
            mandatory_dims.extend(["io_iops", "temperature_celsius"])

        evidence_reqs = EvidenceRequirements(
            required_evidence_level="RUNTIME_MEASURED",
            max_acceptable_age_seconds=max_age,
            mandatory_dimensions=mandatory_dims
        )

        # Formulate Falsification Experiment for this candidate
        falsification = self._formulate_falsification_experiment(
            tenant_id=tenant_id,
            action_type=action_type,
            resource_request=resource_request
        )

        return CandidateAction(
            action_id=action_id,
            state_version=current_state_version,
            action_type=action_type,
            target_tenant_id=tenant_id,
            expected_delta={"delta_urgency": -0.15, "delta_contention": -0.20},
            required_resources=resource_request,
            evidence_refs=evidence_ids,
            reversibility=reversibility,
            evidence_requirements=evidence_reqs,
            falsification_experiment=falsification,
            vital_hp=VITAL_MAX_HP
        )

    def _formulate_falsification_experiment(
        self,
        tenant_id: str,
        action_type: str,
        resource_request: Dict[str, float]
    ) -> FalsificationExperiment:
        """
        Synthesizes a concrete falsification test to refute the underlying hypothesis
        during shadow-mode observation before live commit.
        """
        hyp_id = f"HYP-{uuid.uuid4().hex[:6].upper()}"

        if action_type == "RESIZE_BALLOON":
            return FalsificationExperiment(
                hypothesis_id=hyp_id,
                statement=f"Resizing balloon on {tenant_id} will not cause swap thrashing or drop free memory below 512MB",
                metric_key="guest_swap_bytes",
                falsification_operator=">",
                falsification_threshold=0.0,
                shadow_mode_runs=3,
                calibration_tag="[CALIBRATION_HYPOTHESIS]"
            )
        elif action_type == "SET_CGROUP_MAX":
            return FalsificationExperiment(
                hypothesis_id=hyp_id,
                statement=f"Limiting cgroup quota for {tenant_id} will not degrade P99 latency beyond 20ms",
                metric_key="p99_latency_ms",
                falsification_operator=">",
                falsification_threshold=20.0,
                shadow_mode_runs=3,
                calibration_tag="[CALIBRATION_HYPOTHESIS]"
            )
        else: # Generic affinity or allocation
            return FalsificationExperiment(
                hypothesis_id=hyp_id,
                statement=f"Reallocating resources on {tenant_id} will maintain CPU throttle count at 0",
                metric_key="nr_throttled",
                falsification_operator=">",
                falsification_threshold=0.0,
                shadow_mode_runs=3,
                calibration_tag="[CALIBRATION_HYPOTHESIS]"
            )
