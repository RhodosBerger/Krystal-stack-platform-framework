"""
==============================================================================
KSL CLAIMS LATTICE & CRYPTOGRAPHIC DIGEST ENGINE
==============================================================================
Formal 4-Dimensional Epistemic Lattice:
    L = < Implementation, Verification, Formal, Measurement >
And RFC 8785 Canonical Claim Digest Computation.
==============================================================================
"""

import hashlib
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Any, Tuple


# Poset partial orders along each orthogonal epistemic axis
IMPLEMENTATION_ORDER = ["UNIMPLEMENTED", "STUB", "IMPLEMENTED", "FROZEN"]
VERIFICATION_ORDER = ["UNTESTED", "UNIT_TESTED", "INTEGRATION_TESTED", "REGRESSION_TESTED"]
FORMAL_ORDER = ["NONE", "CONTRACT_BOUND", "MODEL_CHECKED", "FORMAL_BMC", "FORMALLY_PROVEN"]
MEASUREMENT_ORDER = ["PENDING", "SIMULATED", "HOST_BENCHMARKED", "FPGA_BOARD_MEASURED"]

AXIS_RANKS = {
    "implementation": {val: idx for idx, val in enumerate(IMPLEMENTATION_ORDER)},
    "verification": {val: idx for idx, val in enumerate(VERIFICATION_ORDER)},
    "formal": {val: idx for idx, val in enumerate(FORMAL_ORDER)},
    "measurement": {val: idx for idx, val in enumerate(MEASUREMENT_ORDER)},
}


@dataclass(frozen=True)
class EvidenceLatticePoint:
    """A point in the 4-dimensional epistemic evidence lattice."""
    implementation: str
    verification: str
    formal: str
    measurement: str

    def to_dict(self) -> Dict[str, str]:
        return {
            "implementation": self.implementation,
            "verification": self.verification,
            "formal": self.formal,
            "measurement": self.measurement
        }

    @classmethod
    def from_dict(cls, data: Dict[str, str]) -> "EvidenceLatticePoint":
        return cls(
            implementation=data["implementation"],
            verification=data["verification"],
            formal=data["formal"],
            measurement=data["measurement"]
        )

    def is_less_than_or_equal_to(self, other: "EvidenceLatticePoint") -> bool:
        """
        True iff every coordinate of self is <= other in its respective poset.
        """
        for axis, rank_map in AXIS_RANKS.items():
            s_val = getattr(self, axis)
            o_val = getattr(other, axis)
            if rank_map[s_val] > rank_map[o_val]:
                return False
        return True


def canonical_jcs_serialize(data: Any) -> bytes:
    """RFC 8785 Canonical JSON Serialization."""
    if isinstance(data, dict):
        items = sorted(data.items(), key=lambda x: x[0].encode("utf-8"))
        return b"{" + b",".join(
            json.dumps(k).encode("utf-8") + b":" + canonical_jcs_serialize(v)
            for k, v in items
        ) + b"}"
    elif isinstance(data, list):
        return b"[" + b",".join(canonical_jcs_serialize(x) for x in data) + b"]"
    elif isinstance(data, str):
        return json.dumps(data, ensure_ascii=False).encode("utf-8")
    elif isinstance(data, bool):
        return b"true" if data else b"false"
    elif data is None:
        return b"null"
    elif isinstance(data, (int, float)):
        return json.dumps(data).encode("utf-8")
    else:
        raise ValueError(f"Unserializable type: {type(data)}")


def compute_claim_semantic_digest(claim_data: Dict[str, Any]) -> str:
    """
    Computes SHA-256 over RFC 8785 canonical serialization of the Claim Semantic Core:
      claim_id, statement, scope, constitutional_invariant, required_authority, falsifier.
    """
    semantic_core = {
        "claim_id": str(claim_data["claim_id"]),
        "statement": str(claim_data["statement"]),
        "scope": str(claim_data["scope"]),
        "constitutional_invariant": str(claim_data["constitutional_invariant"]),
        "required_authority": str(claim_data.get("authority", claim_data.get("required_authority", ""))),
        "falsifier": str(claim_data["falsifier"])
    }
    canonical_bytes = canonical_jcs_serialize(semantic_core)
    return hashlib.sha256(canonical_bytes).hexdigest()
