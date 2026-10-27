#!/usr/bin/env python3
"""
==============================================================================
KRYSTAL-STACK: UNIFIED TRANSITION ENVELOPE & ARCHITECTURAL MODELS (MILESTONE M2)
==============================================================================
Module: krystal_kernel/transition/models.py
Normative Reference:
    - docs/architecture/MASTER_BLUEPRINT_KRYSTAL_STACK_SOVEREIGN_OMNI_SYSTEM.md
    - docs/architecture/AEP_CANONICAL_SPEC.md
Constitutional Invariant:
    - VITAL_MAX_HP == 6
    - Invariant != Policy
    - Zero-Dirty State Mutation Delta S == 0 on breach
==============================================================================
"""

from __future__ import annotations

import time
import math
import uuid
import hashlib
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Dict, List, Tuple, Optional, Any, Set

from krystal_kernel.constants import VITAL_MAX_HP, ContractViolation


class DomainType(str, Enum):
    """The four domain poles of the Krystal Stack platform."""
    CNC_MANUFACTURING     = "CNC_MANUFACTURING"
    SILICON_COMPUTE       = "SILICON_COMPUTE"
    SOVEREIGN_ENGINE      = "SOVEREIGN_ENGINE"
    FARM_MARKET_SCENARIO  = "FARM_MARKET_SCENARIO"


class TransitionVerdict(str, Enum):
    """Authoritative verdict emitted for a transition envelope."""
    APPROVED_COMMITTED    = "APPROVED_COMMITTED"
    DISCARDED_OVERLOAD    = "DISCARDED_OVERLOAD"
    INVALID_INVARIANT     = "INVALID_INVARIANT"
    QUARANTINED           = "QUARANTINED"


@dataclass
class DomainStateVector:
    """Normalized multi-dimensional state representation."""
    metrics: Dict[str, float]          # Raw or normalized domain metrics [0.0 to 1.0+]
    normalized_vector: List[float]     # Canonical float32 vector for mathematical projection
    timestamp_ns: int
    dimension_count: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ExecutionCandidate:
    """A proposed candidate action to be evaluated by the transition gate."""
    candidate_id: str
    action_name: str
    parameters: Dict[str, Any]
    expected_utility: float            # Expected performance/efficiency improvement
    risk_score: float                  # [0.0 to 1.0]
    is_reversible: bool = True
    rollback_payload: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ConstraintSet:
    """Hard boundaries that must NOT be breached during transition."""
    thermal_limit_celsius: float
    max_power_watts: float
    max_queue_depth: int
    latency_deadline_ms: float
    vital_max_hp: int = VITAL_MAX_HP
    custom_rules: Dict[str, float] = field(default_factory=dict)


@dataclass
class KrystalTransitionEnvelope:
    """
    Authoritative Unified Transition Envelope (Milestone M2 Contract).
    The single common interface binding all four operational poles.
    """
    runtime_id: str
    domain: str

    state_before: DomainStateVector
    candidate: ExecutionCandidate
    predicted_state: DomainStateVector

    constraints: ConstraintSet

    state_after: Optional[DomainStateVector] = None

    prediction_residual: Optional[float] = None
    utility_gain: Optional[float] = None

    verdict: str = TransitionVerdict.APPROVED_COMMITTED.value
    receipt_hash: Optional[str] = None
    vital_max_hp: int = VITAL_MAX_HP

    def __post_init__(self):
        if self.vital_max_hp != 6:
            raise ContractViolation(f"Constitutional breach: VITAL_MAX_HP == 6 required, got {self.vital_max_hp}")

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "runtime_id": self.runtime_id,
            "domain": self.domain,
            "state_before": self.state_before.to_dict(),
            "candidate": asdict(self.candidate),
            "predicted_state": self.predicted_state.to_dict(),
            "constraints": asdict(self.constraints),
            "state_after": self.state_after.to_dict() if self.state_after else None,
            "prediction_residual": self.prediction_residual,
            "utility_gain": self.utility_gain,
            "verdict": self.verdict,
            "receipt_hash": self.receipt_hash,
            "vital_max_hp": self.vital_max_hp,
        }
        return d
