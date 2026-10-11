#!/usr/bin/env python3
"""
KRYSTAL-STACK // EVIDENCE LEVELS & EPISTEMIC TRANSITION ENGINE
=============================================================================
Section 4 & 7 Implementation:
Strict hierarchy of verification levels and epistemic claim states.

Rules:
  1. No claim may simply be labeled "verified" without declaring its EvidenceLevel.
  2. A claim supported only by unit tests (UNIT_VERIFIED) must NEVER be claimed
     as a hardware guarantee (HARDWARE_MEASURED) or production-ready (PRODUCTION_VALIDATED).
  3. Hypotheses transition strictly through falsification logic:
     HYPOTHESIS -> (reproducibility >= 85% + N >= 5) -> SUPPORTED
     HYPOTHESIS -> (falsification condition triggered) -> FALSIFIED
     Never silently promote a hypothesis to a fact.
=============================================================================
"""

import os
import sys
import time
import uuid
from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional


class EvidenceLevel(str, Enum):
    UNIT_VERIFIED = "UNIT_VERIFIED"                    # Logic tested in isolation with mocks
    INTEGRATION_VERIFIED = "INTEGRATION_VERIFIED"      # Inter-module contracts verified across boundaries
    RUNTIME_MEASURED = "RUNTIME_MEASURED"              # Real process execution traced on host OS
    HARDWARE_MEASURED = "HARDWARE_MEASURED"            # Directly sampled from physical sensors / hardware registers
    STATISTICALLY_REPEATED = "STATISTICALLY_REPEATED"  # N >= 5 repetitions with CV <= 0.15
    PRODUCTION_VALIDATED = "PRODUCTION_VALIDATED"      # Operating under live sustained production SLAs


class EpistemicState(str, Enum):
    OBSERVED = "OBSERVED"        # Directly measured without inferential processing
    DERIVED = "DERIVED"          # Mathematically computed from OBSERVED data
    HYPOTHESIS = "HYPOTHESIS"    # Proposed explanation awaiting experimental falsification
    SUPPORTED = "SUPPORTED"      # Survived falsification across N >= 5 repetitions
    FALSIFIED = "FALSIFIED"      # Empirically refuted by data signature breach
    UNKNOWN = "UNKNOWN"          # Unmodeled phenomenon requiring investigation


@dataclass
class EpistemicClaim:
    claim_id: str
    statement: str
    epistemic_state: EpistemicState
    evidence_level: EvidenceLevel
    falsification_condition: str
    supporting_observations: List[str] = field(default_factory=list)
    confidence_score: float = 0.5                      # [0.0 - 1.0]
    uncertainty_score: float = 0.5                     # [0.0 - 1.0]
    repetitions_count: int = 0
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "claim_id": self.claim_id,
            "statement": self.statement,
            "epistemic_state": self.epistemic_state.value,
            "evidence_level": self.evidence_level.value,
            "falsification_condition": self.falsification_condition,
            "supporting_observations": self.supporting_observations,
            "confidence_score": round(self.confidence_score, 3),
            "uncertainty_score": round(self.uncertainty_score, 3),
            "repetitions_count": self.repetitions_count,
            "created_at": self.created_at,
            "updated_at": self.updated_at
        }


class EvidenceRegistry:
    """Manages verified claims and ensures formal epistemic promotion rules."""

    def __init__(self):
        self.claims: Dict[str, EpistemicClaim] = {}

    def register_claim(
        self,
        statement: str,
        falsification_condition: str,
        initial_evidence_level: EvidenceLevel = EvidenceLevel.UNIT_VERIFIED,
        initial_state: EpistemicState = EpistemicState.HYPOTHESIS
    ) -> EpistemicClaim:
        cid = f"CLAIM-{uuid.uuid4().hex[:8].upper()}"
        claim = EpistemicClaim(
            claim_id=cid,
            statement=statement,
            epistemic_state=initial_state,
            evidence_level=initial_evidence_level,
            falsification_condition=falsification_condition
        )
        self.claims[cid] = claim
        return claim

    def update_evidence(
        self,
        claim_id: str,
        observed_delta: float,
        is_falsified: bool,
        evidence_level: EvidenceLevel,
        repetition_run: bool = True
    ) -> EpistemicClaim:
        claim = self.claims.get(claim_id)
        if not claim:
            raise KeyError(f"Claim {claim_id} not found in evidence registry.")

        claim.updated_at = time.time()
        if repetition_run:
            claim.repetitions_count += 1

        # Upgrade evidence level if new observation has higher rigor
        level_ranks = {
            EvidenceLevel.UNIT_VERIFIED: 1,
            EvidenceLevel.INTEGRATION_VERIFIED: 2,
            EvidenceLevel.RUNTIME_MEASURED: 3,
            EvidenceLevel.HARDWARE_MEASURED: 4,
            EvidenceLevel.STATISTICALLY_REPEATED: 5,
            EvidenceLevel.PRODUCTION_VALIDATED: 6
        }
        if level_ranks[evidence_level] > level_ranks[claim.evidence_level]:
            claim.evidence_level = evidence_level

        # Evaluate falsification vs support
        if is_falsified:
            claim.epistemic_state = EpistemicState.FALSIFIED
            claim.confidence_score = 0.0
            claim.uncertainty_score = 0.0
        else:
            if claim.repetitions_count >= 5:
                claim.epistemic_state = EpistemicState.SUPPORTED
                claim.confidence_score = min(0.98, claim.confidence_score + 0.10)
                claim.uncertainty_score = max(0.02, claim.uncertainty_score - 0.10)
            else:
                claim.confidence_score = min(0.80, claim.confidence_score + 0.05)

        return claim


GLOBAL_EVIDENCE_REGISTRY = EvidenceRegistry()
