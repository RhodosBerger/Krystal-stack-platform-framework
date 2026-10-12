#!/usr/bin/env python3
"""
==============================================================================
KRYSTAL-STACK: STATE VERSION & CAUSAL CLOCK DOMAIN MODEL
==============================================================================
Module: krystal_kernel/verification/state_version.py
Description:
    Implements formal StateVersion and distributed causal ordering across
    independent host clock domains.

    Rules:
      1. Monotonic clocks are NOT globally comparable across distinct physical hosts.
      2. Every state change advances a discrete sequence number with causal parent linkage.
      3. A CandidateAction becomes strictly invalid if CurrentStateVersion != source_state_version.
==============================================================================
"""

import time
import hashlib
from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional


@dataclass(frozen=True)
class StateVersion:
    """
    Causally-versioned state identifier with discrete sequence numbering
    and physical clock domain attribution.
    """
    major: int
    minor: int
    sequence: int
    wall_timestamp_utc: float
    clock_domain_id: str = "host_clock_0"
    causal_parent: Optional[str] = None

    def next_sequence(
        self,
        new_minor: Optional[int] = None,
        causal_parent_hash: Optional[str] = None
    ) -> 'StateVersion':
        """Advances sequence number monotonically with causal parent chain."""
        return StateVersion(
            major=self.major,
            minor=new_minor if new_minor is not None else self.minor,
            sequence=self.sequence + 1,
            wall_timestamp_utc=time.time(),
            clock_domain_id=self.clock_domain_id,
            causal_parent=causal_parent_hash or self.version_digest()
        )

    def is_compatible_with(self, other: 'StateVersion') -> bool:
        """Verifies if two state versions belong to the exact same continuous state lineage."""
        return (
            self.major == other.major and
            self.minor == other.minor and
            self.sequence == other.sequence and
            self.clock_domain_id == other.clock_domain_id
        )

    def version_digest(self) -> str:
        """Deterministic cryptographic fingerprint of the state version."""
        raw = f"{self.major}:{self.minor}:{self.sequence}:{self.clock_domain_id}:{self.wall_timestamp_utc}:{self.causal_parent}"
        return hashlib.sha256(raw.encode('utf-8')).hexdigest()[:16]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "major": self.major,
            "minor": self.minor,
            "sequence": self.sequence,
            "wall_timestamp_utc": self.wall_timestamp_utc,
            "clock_domain_id": self.clock_domain_id,
            "causal_parent": self.causal_parent,
            "version_digest": self.version_digest()
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'StateVersion':
        return cls(
            major=int(d["major"]),
            minor=int(d["minor"]),
            sequence=int(d["sequence"]),
            wall_timestamp_utc=float(d["wall_timestamp_utc"]),
            clock_domain_id=str(d.get("clock_domain_id", "host_clock_0")),
            causal_parent=d.get("causal_parent")
        )

    def __str__(self) -> str:
        return f"v{self.major}.{self.minor}.{self.sequence}@{self.clock_domain_id}"
