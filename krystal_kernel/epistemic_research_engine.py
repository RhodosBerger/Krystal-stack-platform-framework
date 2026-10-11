#!/usr/bin/env python3
"""
KRYSTAL-STACK // EPISTEMIC RESEARCH ARCHITECTURE CORE
=============================================================================
Defines the foundational epistemic data contracts and rules for the scientific
research subsystem in Krystal-Stack:

  PRIMARY PRINCIPLE:
    Observe -> Model -> Explain -> Predict -> Falsify -> Validate -> Intervene

  EPISTEMIC CLASSES:
    OBSERVED   : Directly measured telemetry
    DERIVED    : Mathematically derived from observations
    HYPOTHESIS : Proposed explanation
    PREDICTION : Expected future observation if hypothesis holds
    FALSIFIED  : Contradicted by experiment
    SUPPORTED  : Survived defined falsification tests
    UNKNOWN    : Insufficient evidence

  MULTI-OBJECTIVE HEALTH VECTOR H:
    [latency, throughput, memory_pressure, cpu_pressure, io_pressure,
     queue_pressure, thermal_state, energy_cost, stability, recovery_time]

  SAFETY CONTRACT:
    Research output produces InterventionProposal only. Execution requires
    a separate deterministic policy layer.
=============================================================================
"""

import os
import sys
import time
import math
import uuid
import json
from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple, Set

WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

from krystal_kernel.scene_constructor_presets import VITAL_MAX_HP


class EpistemicStatus(str, Enum):
    OBSERVED = "OBSERVED"
    DERIVED = "DERIVED"
    HYPOTHESIS = "HYPOTHESIS"
    PREDICTION = "PREDICTION"
    FALSIFIED = "FALSIFIED"
    SUPPORTED = "SUPPORTED"
    UNKNOWN = "UNKNOWN"


class DriftType(str, Enum):
    EXPECTED_VARIANCE = "EXPECTED_VARIANCE"
    PERFORMANCE_DRIFT = "PERFORMANCE_DRIFT"
    RESOURCE_DRIFT = "RESOURCE_DRIFT"
    CONFIGURATION_DRIFT = "CONFIGURATION_DRIFT"
    TOPOLOGY_DRIFT = "TOPOLOGY_DRIFT"
    SECURITY_DRIFT = "SECURITY_DRIFT"
    UNKNOWN_DRIFT = "UNKNOWN_DRIFT"


@dataclass
class ResearchStateVectorH:
    """Complete 10-dimensional state vector for multi-objective optimization."""
    latency_ms: float = 12.0
    throughput_ops_sec: float = 1250.0
    memory_pressure: float = 0.22      # [0.0 - 1.0]
    cpu_pressure: float = 0.35         # [0.0 - 1.0]
    io_pressure: float = 0.10          # [0.0 - 1.0]
    queue_pressure: float = 0.08       # [0.0 - 1.0] (queue_depth / max_sla)
    thermal_state_celsius: float = 48.0
    energy_cost_watts: float = 45.0
    stability_score: float = 0.99      # [0.0 - 1.0] (1.0 = zero jitter)
    recovery_time_ms: float = 8.5

    def to_dict(self) -> Dict[str, float]:
        return {
            "latency_ms": self.latency_ms,
            "throughput_ops_sec": self.throughput_ops_sec,
            "memory_pressure": self.memory_pressure,
            "cpu_pressure": self.cpu_pressure,
            "io_pressure": self.io_pressure,
            "queue_pressure": self.queue_pressure,
            "thermal_state_celsius": self.thermal_state_celsius,
            "energy_cost_watts": self.energy_cost_watts,
            "stability_score": self.stability_score,
            "recovery_time_ms": self.recovery_time_ms
        }

    def compute_composite_cost(self, weights: Optional[Dict[str, float]] = None) -> float:
        """Calculates multi-objective cost J(C). Lower is better."""
        w = weights or {
            "w_latency": 0.25,
            "w_resource": 0.20,
            "w_instability": 0.20,
            "w_energy": 0.15,
            "w_recovery": 0.10,
            "w_thermal": 0.10
        }
        # Normalized components
        norm_lat = min(1.0, self.latency_ms / 100.0)
        norm_res = max(self.memory_pressure, self.cpu_pressure, self.io_pressure, self.queue_pressure)
        norm_instab = 1.0 - self.stability_score
        norm_energy = min(1.0, self.energy_cost_watts / 250.0)
        norm_rec = min(1.0, self.recovery_time_ms / 200.0)
        norm_therm = min(1.0, max(0.0, (self.thermal_state_celsius - 30.0) / 50.0))

        cost = (
            w.get("w_latency", 0.25) * norm_lat +
            w.get("w_resource", 0.20) * norm_res +
            w.get("w_instability", 0.20) * norm_instab +
            w.get("w_energy", 0.15) * norm_energy +
            w.get("w_recovery", 0.10) * norm_rec +
            w.get("w_thermal", 0.10) * norm_therm
        )
        return round(cost, 4)


@dataclass
class EpistemicClaim:
    """A statement in the system explicitly bound to its epistemic classification."""
    claim_id: str
    subject: str
    assertion: str
    status: EpistemicStatus
    evidence_source: Optional[str] = None
    confidence: float = 1.0
    falsification_condition: Optional[str] = None
    created_at: float = field(default_factory=time.time)

    def promote_to_supported(self, experiment_id: str):
        if self.status != EpistemicStatus.HYPOTHESIS and self.status != EpistemicStatus.PREDICTION:
            raise ValueError(f"Cannot promote claim from status {self.status} to SUPPORTED directly")
        self.status = EpistemicStatus.SUPPORTED
        self.evidence_source = f"Experiment:{experiment_id}"

    def falsify(self, experiment_id: str, contradicting_data: str):
        self.status = EpistemicStatus.FALSIFIED
        self.evidence_source = f"FalsifiedBy:{experiment_id}:{contradicting_data}"


@dataclass
class DriftEvent:
    """Represents a measured distance between C_desired and C_observed."""
    drift_id: str
    timestamp: float
    component: str
    c_desired: Dict[str, Any]
    c_observed: Dict[str, Any]
    drift_distance: float
    drift_type: DriftType
    explanation: Optional[str] = None
    requires_freeze: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "drift_id": self.drift_id,
            "timestamp": self.timestamp,
            "component": self.component,
            "c_desired": self.c_desired,
            "c_observed": self.c_observed,
            "drift_distance": self.drift_distance,
            "drift_type": self.drift_type.value,
            "explanation": self.explanation,
            "requires_freeze": self.requires_freeze
        }


@dataclass
class InterventionProposal:
    """Strictly passive recommendation produced by research models. Never modifies production directly."""
    proposal_id: str
    target_component: str
    proposed_configuration: Dict[str, Any]
    predicted_state_delta: Dict[str, Any]
    supporting_evidence: List[str]
    falsification_risk: float
    safety_invariant_verified: bool
    vital_hp: int = VITAL_MAX_HP
    policy_approval_status: str = "PENDING_SAFETY_BOARD"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "proposal_id": self.proposal_id,
            "target_component": self.target_component,
            "proposed_configuration": self.proposed_configuration,
            "predicted_state_delta": self.predicted_state_delta,
            "supporting_evidence": self.supporting_evidence,
            "falsification_risk": self.falsification_risk,
            "safety_invariant_verified": self.safety_invariant_verified,
            "vital_hp": self.vital_hp,
            "policy_approval_status": self.policy_approval_status
        }
