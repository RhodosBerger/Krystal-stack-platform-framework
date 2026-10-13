#!/usr/bin/env python3
"""
==============================================================================
KRYSTAL-STACK: EPISTEMIC TYPE SYSTEM & TRUTH LATTICE (RUNTIME CORE)
==============================================================================
Module: krystal_kernel/runtime/epistemic_type_system.py
Description:
    Strongly-typed epistemic representations with continuous belief degrees,
    uncertainty intervals, truth lattices, and formal lattice operations
    (meet, join, contradiction detection).

Constitutional Invariant:
    assert VITAL_MAX_HP == 6
==============================================================================
"""

import math
import time
import hashlib
from enum import Enum, unique
from typing import Dict, Any, List, Optional, Tuple, TypeVar, Generic
from dataclasses import dataclass, field, asdict

from krystal_kernel.scene_constructor_presets import VITAL_MAX_HP
from krystal_kernel.verification.semantic_protocol import (
    EpistemicState,
    AuthorityState,
    TemporalState,
    to_wire_token
)

assert VITAL_MAX_HP == 6, f"Constitutional invariant breach: VITAL_MAX_HP must be 6, got {VITAL_MAX_HP}"

T = TypeVar("T")


@unique
class LatticeCertaintyLevel(str, Enum):
    """Discrete levels of epistemic certainty on the truth lattice."""
    BOTTOM_UNKNOWN = "BOTTOM_UNKNOWN"          # Uncalibrated or missing origin
    HYPOTHETICAL = "HYPOTHETICAL"              # Formulated by model or generator; ungrounded
    EVIDENCED_WEAK = "EVIDENCED_WEAK"          # Single observation or noisy probe
    EVIDENCED_STRONG = "EVIDENCED_STRONG"      # Corroborated across multiple independent probes
    TOP_AXIOMATIC = "TOP_AXIOMATIC"            # Inviolable platform law (e.g. VITAL_MAX_HP == 6)


@dataclass(frozen=True)
class ConfidenceInterval:
    """Bounded uncertainty interval [lower_bound, upper_bound] at specified sigma confidence."""
    lower_bound: float
    upper_bound: float
    sigma: float = 2.0  # 95.4% normal distribution coverage

    @property
    def midpoint(self) -> float:
        return (self.lower_bound + self.upper_bound) / 2.0

    @property
    def spread(self) -> float:
        return max(0.0, self.upper_bound - self.lower_bound)

    def contains(self, value: float) -> bool:
        return self.lower_bound <= value <= self.upper_bound

    def overlaps(self, other: "ConfidenceInterval") -> bool:
        return max(self.lower_bound, other.lower_bound) <= min(self.upper_bound, other.upper_bound)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "lower_bound": round(self.lower_bound, 4),
            "upper_bound": round(self.upper_bound, 4),
            "midpoint": round(self.midpoint, 4),
            "spread": round(self.spread, 4),
            "sigma": self.sigma
        }


@dataclass
class EpistemicValue(Generic[T]):
    """
    Core Epistemic Value Container.
    Encapsulates a payload with its epistemic state, continuous belief degree,
    uncertainty bounds, evidence pedigree, and timestamp.
    """
    value: T
    epistemic_state: EpistemicState = EpistemicState.EPI_OBSERVED
    belief_degree: float = 1.0  # Continuous belief degree in [0.0, 1.0]
    certainty_level: LatticeCertaintyLevel = LatticeCertaintyLevel.EVIDENCED_STRONG
    confidence_interval: Optional[ConfidenceInterval] = None
    evidence_tokens: List[str] = field(default_factory=list)
    source_probe: str = "core_sensor"
    timestamp_ns: int = field(default_factory=time.perf_counter_ns)
    monotonic_revision: int = 1
    vital_max_hp: int = VITAL_MAX_HP

    def __post_init__(self):
        assert self.vital_max_hp == 6, "EpistemicValue requires VITAL_MAX_HP == 6"
        # Clamp belief degree
        if self.belief_degree < 0.0:
            object.__setattr__(self, "belief_degree", 0.0)
        elif self.belief_degree > 1.0:
            object.__setattr__(self, "belief_degree", 1.0)

    @property
    def wire_token(self) -> str:
        return to_wire_token(self.epistemic_state)

    @property
    def is_falsified(self) -> bool:
        return self.epistemic_state == EpistemicState.EPI_FALSIFIED or self.belief_degree < 0.20

    @property
    def is_strongly_evidenced(self) -> bool:
        return (
            self.belief_degree >= 0.85
            and self.certainty_level in (LatticeCertaintyLevel.EVIDENCED_STRONG, LatticeCertaintyLevel.TOP_AXIOMATIC)
            and self.epistemic_state not in (EpistemicState.EPI_FALSIFIED, EpistemicState.EPI_CONTRADICTORY, EpistemicState.EPI_QUARANTINED)
        )

    def compute_evidence_hash(self) -> str:
        """Computes deterministic SHA-256 evidence fingerprint."""
        raw = f"{self.value}:{self.epistemic_state.value}:{self.belief_degree:.4f}:{self.source_probe}:{self.monotonic_revision}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "value": self.value,
            "epistemic_state": self.epistemic_state.name,
            "epistemic_state_code": hex(self.epistemic_state.value),
            "belief_degree": round(self.belief_degree, 4),
            "certainty_level": self.certainty_level.value,
            "confidence_interval": self.confidence_interval.to_dict() if self.confidence_interval else None,
            "evidence_tokens": self.evidence_tokens,
            "evidence_hash": self.compute_evidence_hash(),
            "source_probe": self.source_probe,
            "timestamp_ns": self.timestamp_ns,
            "monotonic_revision": self.monotonic_revision,
            "wire_token": self.wire_token,
            "vital_max_hp": self.vital_max_hp
        }


class EpistemicTruthLattice:
    """
    Formal algebraic operations on the Epistemic Truth Lattice.
    Implements Meet (GLB - greatest lower bound), Join (LUB - least upper bound),
    and contradiction detection across parallel observations.
    """

    @staticmethod
    def meet(v1: EpistemicValue[T], v2: EpistemicValue[T]) -> EpistemicValue[T]:
        """
        Lattice Meet (∧): Conservative lower bound.
        Yields the minimum belief degree and combines required evidence tokens.
        """
        assert v1.vital_max_hp == 6 and v2.vital_max_hp == 6
        new_belief = min(v1.belief_degree, v2.belief_degree)

        # Lower bound state
        if v1.epistemic_state == EpistemicState.EPI_FALSIFIED or v2.epistemic_state == EpistemicState.EPI_FALSIFIED:
            state = EpistemicState.EPI_FALSIFIED
            level = LatticeCertaintyLevel.BOTTOM_UNKNOWN
        elif v1.epistemic_state == EpistemicState.EPI_QUARANTINED or v2.epistemic_state == EpistemicState.EPI_QUARANTINED:
            state = EpistemicState.EPI_QUARANTINED
            level = LatticeCertaintyLevel.BOTTOM_UNKNOWN
        elif v1.epistemic_state == EpistemicState.EPI_CONTRADICTORY or v2.epistemic_state == EpistemicState.EPI_CONTRADICTORY:
            state = EpistemicState.EPI_CONTRADICTORY
            level = LatticeCertaintyLevel.BOTTOM_UNKNOWN
        else:
            state = EpistemicState.EPI_VALIDATED if new_belief >= 0.8 else EpistemicState.EPI_CALIBRATED
            level = LatticeCertaintyLevel.EVIDENCED_WEAK if new_belief < 0.8 else LatticeCertaintyLevel.EVIDENCED_STRONG

        combined_tokens = sorted(list(set(v1.evidence_tokens + v2.evidence_tokens + ["OP_MEET"])))
        return EpistemicValue(
            value=v1.value,
            epistemic_state=state,
            belief_degree=new_belief,
            certainty_level=level,
            evidence_tokens=combined_tokens,
            source_probe=f"meet({v1.source_probe},{v2.source_probe})",
            timestamp_ns=max(v1.timestamp_ns, v2.timestamp_ns),
            monotonic_revision=max(v1.monotonic_revision, v2.monotonic_revision) + 1,
            vital_max_hp=VITAL_MAX_HP
        )

    @staticmethod
    def join(v1: EpistemicValue[T], v2: EpistemicValue[T]) -> EpistemicValue[T]:
        """
        Lattice Join (∨): Optimistic upper bound.
        Corroborates evidence to boost belief degree when observations agree.
        """
        assert v1.vital_max_hp == 6 and v2.vital_max_hp == 6
        # Bayesian / independent evidence combination: P(A ∪ B) = 1 - (1 - P1)*(1 - P2)
        combined_belief = 1.0 - ((1.0 - v1.belief_degree) * (1.0 - v2.belief_degree))
        combined_belief = min(0.999, max(0.0, combined_belief))

        combined_tokens = sorted(list(set(v1.evidence_tokens + v2.evidence_tokens + ["OP_JOIN_CORROBORATED"])))
        return EpistemicValue(
            value=v1.value,
            epistemic_state=EpistemicState.EPI_VALIDATED if combined_belief >= 0.85 else EpistemicState.EPI_OBSERVED,
            belief_degree=combined_belief,
            certainty_level=LatticeCertaintyLevel.EVIDENCED_STRONG if combined_belief >= 0.85 else LatticeCertaintyLevel.EVIDENCED_WEAK,
            evidence_tokens=combined_tokens,
            source_probe=f"join({v1.source_probe},{v2.source_probe})",
            timestamp_ns=max(v1.timestamp_ns, v2.timestamp_ns),
            monotonic_revision=max(v1.monotonic_revision, v2.monotonic_revision) + 1,
            vital_max_hp=VITAL_MAX_HP
        )

    @staticmethod
    def detect_contradiction(
        v1: EpistemicValue[float],
        v2: EpistemicValue[float],
        tolerance_abs: float = 0.15
    ) -> Tuple[bool, float]:
        """
        Checks whether two numerical epistemic observations contradict each other.
        Returns (is_contradictory, divergence_magnitude).
        """
        divergence = abs(float(v1.value) - float(v2.value))
        if v1.confidence_interval and v2.confidence_interval:
            # If confidence intervals do not overlap, contradiction is detected
            overlaps = v1.confidence_interval.overlaps(v2.confidence_interval)
            return (not overlaps and divergence > tolerance_abs), divergence
        return (divergence > tolerance_abs), divergence
