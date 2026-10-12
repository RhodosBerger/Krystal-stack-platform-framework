#!/usr/bin/env python3
"""
==============================================================================
KRYSTAL-STACK: TYPED EPISTEMIC VALUE MODEL
==============================================================================
Module: krystal_kernel/verification/epistemic_value.py
Description:
    Wraps raw telemetry and inferences with rigorous epistemic metadata:
      - Provenance, evidence level, uncertainty bounds.
      - Nanosecond validity windows (valid_from_ns, valid_to_ns).
      - StateVersion attribution and physical clock domain binding.
      - Discrete validity lifecycle: FRESH, STALE, EXPIRED, CONTRADICTORY, UNKNOWN.

    Critical Invariant:
      Raw float/int telemetry must NEVER participate directly in authority decisions
      without epistemic metadata after normalization.
      Expired data becomes historical evidence and MUST NOT silently participate
      in current decisions.
==============================================================================
"""

import time
from enum import Enum
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Generic, TypeVar

from krystal_kernel.verification.state_version import StateVersion

T = TypeVar("T")


class ValidityState(str, Enum):
    FRESH = "FRESH"                    # Within active sampling interval and uncertainty bounds
    STALE = "STALE"                    # Surpassed sampling interval but within grace period
    EXPIRED = "EXPIRED"                # Surpassed valid_to_ns; historical evidence ONLY
    CONTRADICTORY = "CONTRADICTORY"    # Diverges beyond physical/statistical calibration error
    UNKNOWN = "UNKNOWN"                # Future clock drift or uncalibrated domain


class EvidenceKind(str, Enum):
    HARDWARE_MEASURED = "HARDWARE_MEASURED"            # Directly sampled from physical silicon/sensors
    RUNTIME_MEASURED = "RUNTIME_MEASURED"              # OS/Hypervisor process counters (cgroups, WMI)
    INTEGRATION_VERIFIED = "INTEGRATION_VERIFIED"      # Inter-module contract verified
    DERIVED = "DERIVED"                                # Mathematically computed from hardware/runtime signals
    INFERRED = "INFERRED"                              # Counterfactual / model prediction


@dataclass
class EpistemicValue(Generic[T]):
    """
    Epistemic container for measured or inferred values.
    Carries complete provenance, uncertainty, validity window, and clock domain.
    """
    value: T
    provenance: str
    evidence_kind: EvidenceKind
    uncertainty: float                       # Fraction [0.0 - 1.0] representing uncertainty margin
    valid_from_ns: int                       # Nanosecond timestamp when measurement became valid
    valid_to_ns: int                         # Nanosecond timestamp after which value expires
    domain: str                              # Metric domain (e.g. "cpu.thermal", "memory.rss")
    state_version: StateVersion              # State version at time of measurement
    source_adapter: str                      # Adapter identifier (e.g. "KvmQemuAdapter")
    clock_domain_id: str = "host_clock_0"
    calibration_error: float = 0.0           # Known instrument/sensor calibration error

    def get_validity_state(
        self,
        now_ns: int,
        max_stale_ns: Optional[int] = None,
        clock_skew_tolerance_ns: int = 100_000_000
    ) -> ValidityState:
        """
        Determines current epistemic validity relative to evaluation timestamp now_ns.
        """
        # 1. Clock skew or future timestamp detection
        if now_ns < self.valid_from_ns:
            if (self.valid_from_ns - now_ns) <= clock_skew_tolerance_ns:
                return ValidityState.FRESH
            return ValidityState.UNKNOWN

        # 2. Hard expiration check: strictly becomes historical evidence
        if now_ns > self.valid_to_ns:
            return ValidityState.EXPIRED

        # 3. Staleness threshold check
        stale_limit = max_stale_ns if max_stale_ns is not None else int((self.valid_to_ns - self.valid_from_ns) * 0.7)
        if (now_ns - self.valid_from_ns) > stale_limit:
            return ValidityState.STALE

        return ValidityState.FRESH

    def is_usable_for_decision(self, now_ns: int, max_stale_ns: Optional[int] = None) -> bool:
        """
        Only FRESH values are authorized to participate in current authority decisions.
        STALE, EXPIRED, CONTRADICTORY, or UNKNOWN values must be rejected or deferred.
        """
        return self.get_validity_state(now_ns, max_stale_ns) == ValidityState.FRESH

    def to_dict(self) -> Dict[str, Any]:
        return {
            "value": self.value,
            "provenance": self.provenance,
            "evidence_kind": self.evidence_kind.value,
            "uncertainty": round(self.uncertainty, 4),
            "valid_from_ns": self.valid_from_ns,
            "valid_to_ns": self.valid_to_ns,
            "domain": self.domain,
            "state_version": self.state_version.to_dict(),
            "source_adapter": self.source_adapter,
            "clock_domain_id": self.clock_domain_id,
            "calibration_error": round(self.calibration_error, 4)
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'EpistemicValue':
        return cls(
            value=d["value"],
            provenance=str(d["provenance"]),
            evidence_kind=EvidenceKind(d["evidence_kind"]),
            uncertainty=float(d["uncertainty"]),
            valid_from_ns=int(d["valid_from_ns"]),
            valid_to_ns=int(d["valid_to_ns"]),
            domain=str(d["domain"]),
            state_version=StateVersion.from_dict(d["state_version"]),
            source_adapter=str(d["source_adapter"]),
            clock_domain_id=str(d.get("clock_domain_id", "host_clock_0")),
            calibration_error=float(d.get("calibration_error", 0.0))
        )
