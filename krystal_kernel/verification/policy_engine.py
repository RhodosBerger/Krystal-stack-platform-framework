#!/usr/bin/env python3
"""
KRYSTAL-STACK // VERSIONED POLICY & TRANSFORMATION ENGINE
=============================================================================
Section 3 Implementation:
Generic transformation engine:
  StateVector + Configuration + Input + Policy -> ProposedAction

Rules:
  1. Every policy is explicitly versioned (e.g., "policy-v2.1.0").
  2. Actions require explicit controlled variables, risk scores, and blast radii.
  3. No action may be executed directly by an LLM without passing through
     the deterministic safety gate.
=============================================================================
"""

import os
import sys
import uuid
import time
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple

WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

from krystal_kernel.verification.state_vector import StateVector
from krystal_kernel.scene_constructor_presets import VITAL_MAX_HP


@dataclass
class ProposedAction:
    action_id: str
    action_type: str                   # e.g. "THROTTLE_CONCURRENCY", "ISOLATE_CORE", "PIN_BANDWIDTH"
    domain: str
    policy_version: str
    controlled_variables: Dict[str, Any]
    expected_effect: str
    expected_state_delta: Dict[str, float]
    risk_score: float                  # [0.0 - 1.0]
    blast_radius: float                # [0.0 - 1.0]
    falsification_condition: str
    is_safety_gate_approved: bool = False
    rejection_reason: Optional[str] = None
    vital_hp: int = VITAL_MAX_HP

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action_id": self.action_id,
            "action_type": self.action_type,
            "domain": self.domain,
            "policy_version": self.policy_version,
            "controlled_variables": self.controlled_variables,
            "expected_effect": self.expected_effect,
            "expected_state_delta": self.expected_state_delta,
            "risk_score": round(self.risk_score, 3),
            "blast_radius": round(self.blast_radius, 3),
            "falsification_condition": self.falsification_condition,
            "is_safety_gate_approved": self.is_safety_gate_approved,
            "rejection_reason": self.rejection_reason,
            "vital_hp": self.vital_hp
        }


@dataclass
class VerificationPolicy:
    policy_id: str
    version: str
    max_allowed_risk: float = 0.40
    max_allowed_blast_radius: float = 0.25
    allowed_variables: List[str] = field(default_factory=lambda: [
        "cpu.threads", "openvino.streams", "cpu.affinity",
        "batch_size", "queue_max_depth", "context_limit_tokens",
        "memory_limit_mb", "sampling_rate_hz"
    ])


class PolicyTransformationEngine:
    """Evaluates state vectors and derives safely bounded intervention proposals."""

    def __init__(self, active_policy: Optional[VerificationPolicy] = None):
        self.active_policy = active_policy or VerificationPolicy(
            policy_id="POL-DEFAULT-STABILITY",
            version="policy-v2.1.0"
        )
        self.action_history: List[ProposedAction] = []

    def evaluate_and_propose_action(
        self,
        current_state: StateVector,
        input_signal: Dict[str, Any],
        proposed_intervention_type: str = "TUNE_REACTIVE"
    ) -> ProposedAction:
        """
        StateVector + Configuration + Input + Policy -> ProposedAction
        """
        act_id = f"ACT-{uuid.uuid4().hex[:8].upper()}"
        geo = current_state.compute_geometry()

        # Identify weakest axis to intervene upon
        weakest = geo.weakest_axis_code
        weakest_val = geo.weakest_axis_score

        controlled_vars = {}
        expected_delta = {}
        risk = 0.15
        blast_radius = 0.10
        expected_effect = ""
        falsification = ""

        if current_state.domain_id == "LLM":
            if weakest == "LAT":
                controlled_vars = {"cpu.threads": 4, "openvino.streams": 2}
                expected_delta = {"LAT": +0.8, "ROB": -0.2}
                risk = 0.20
                blast_radius = 0.15
                expected_effect = "Reduce TTFT/ITL tail latency by binding concurrency to physical cores."
                falsification = "If TTFT does not reduce by >= 15%, falsified."
            elif weakest == "MEM":
                controlled_vars = {"queue_max_depth": 50, "context_limit_tokens": 2048}
                expected_delta = {"MEM": +1.0, "TPS": -0.3}
                expected_effect = "Clamp queue depth and context limit to arrest KV-cache growth."
                falsification = "If KV-cache pressure remains > 80%, falsified."
            else:
                controlled_vars = {"openvino.streams": 1}
                expected_delta = {weakest: +0.5}
                expected_effect = f"Stabilize LLM execution plane around {weakest}."
                falsification = f"If {weakest} improves by < 5%, falsified."

        elif current_state.domain_id == "CNC":
            controlled_vars = {"rpm_differential_limit": 1500, "feed_reduction_pct": 20}
            expected_delta = {"thermal_gradient": +1.2, "collapse_margin": +1.0}
            risk = 0.25
            blast_radius = 0.18
            expected_effect = "Reduce spindle RPM in proportion to thermal rise rate to preserve collapse margin."
            falsification = "If bearing temperature rate dT/dt does not decrease by > 20%, falsified."

        elif current_state.domain_id == "GAME":
            controlled_vars = {"hand_card_cost_threshold": 2}
            expected_delta = {"KON": +0.8, "POŠ": -0.2}
            risk = 0.10
            blast_radius = 0.05
            expected_effect = "Prioritize crowd-control action cards to stabilize combat profile."
            falsification = "If control duration is < 2 rounds, falsified."

        else: # KERNEL / FAULT
            controlled_vars = {"cpu.affinity": [0, 1], "cgroups.memory.high": 2048}
            expected_delta = {weakest: +0.6}
            risk = 0.18
            blast_radius = 0.12
            expected_effect = f"Isolate kernel execution slices to protect {weakest}."
            falsification = f"If context switches do not decrease by > 200/s, falsified."

        # Safety Gate Validation
        approved = True
        rejection_reason = None

        if risk > self.active_policy.max_allowed_risk:
            approved = False
            rejection_reason = f"Risk score ({risk:.2f}) exceeds policy maximum ({self.active_policy.max_allowed_risk:.2f})."

        if blast_radius > self.active_policy.max_allowed_blast_radius:
            approved = False
            rejection_reason = f"Blast radius ({blast_radius:.2f}) exceeds policy ceiling ({self.active_policy.max_allowed_blast_radius:.2f})."

        for v in controlled_vars.keys():
            if v not in self.active_policy.allowed_variables and not any(v.startswith(p) for p in ("rpm_", "feed_", "hand_")):
                approved = False
                rejection_reason = f"Controlled variable '{v}' not permitted in active policy {self.active_policy.version}."
                break

        action = ProposedAction(
            action_id=act_id,
            action_type=proposed_intervention_type,
            domain=current_state.domain_id,
            policy_version=self.active_policy.version,
            controlled_variables=controlled_vars,
            expected_effect=expected_effect,
            expected_state_delta=expected_delta,
            risk_score=risk,
            blast_radius=blast_radius,
            falsification_condition=falsification,
            is_safety_gate_approved=approved,
            rejection_reason=rejection_reason,
            vital_hp=VITAL_MAX_HP
        )

        self.action_history.append(action)
        return action


GLOBAL_POLICY_ENGINE = PolicyTransformationEngine()
