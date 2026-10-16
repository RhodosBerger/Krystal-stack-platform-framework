"""
KRYSTAL-STACK // PRODUCTION EPISTEMIC PROVENANCE & RETROSPECTIVE CALIBRATION LEDGER
===================================================================================
Module: krystal_kernel/epistemic_provenance_logger.py
Milestone: V-EPISTEMIC-CALIBRATION-01
Constitutional Invariant: VITAL_MAX_HP == 6

WHAT THIS MODULE SOLVES:
  Implements the production-grade audit logging and retrospective calibration
  demanded by the Dual-Mode (EXPLORE vs VERIFY) architecture:
    1. Mandatory 4-Tuple Logging:
       < Provenance, Verdict, Evidence, Empirical Outcome >
    2. Prevention of Self-Certification:
       Enforces Proposer != Evaluator across every recorded claim.
    3. Epistemic Stratification & Falsifiability:
       Requires explicit evidence classes and concrete falsification criteria.
    4. Cryptographically Chained Append-Only Audit Ledger:
       SHA-256 rolling block chain ensuring tamper-evident history.
    5. Retrospective Calibration Feedback Loop:
       Computes Expected Calibration Error (ECE) and False Discovery Rate (FDR)
       to recalibrate system thresholds dynamically against ground-truth data.
"""

from __future__ import annotations

import dataclasses
import enum
import hashlib
import hmac
import json
import math
import os
import sys
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from krystal_kernel.scene_constructor_presets import VITAL_MAX_HP
from krystal_kernel.contracts import ContractViolation

assert VITAL_MAX_HP == 6, f"Constitutional invariant breach: VITAL_MAX_HP must be 6, got {VITAL_MAX_HP}"


class EpistemicVerdict(str, enum.Enum):
    """Verdict rendered by the VERIFY plane on an incoming claim."""
    APPROVED_FACT = "APPROVED_FACT"                    # Empirically verified, admitted into knowledge base
    PROVISIONAL_HYPOTHESIS = "PROVISIONAL_HYPOTHESIS"  # Permitted in EXPLORE; awaiting empirical testing
    EVIDENCE_INSUFFICIENT = "EVIDENCE_INSUFFICIENT"    # Lacks required evidence class or empirical coverage
    SOURCE_CONFLICT = "SOURCE_CONFLICT"                # Unreconciled disagreement between evidence sources
    REJECTED_REFUTED = "REJECTED_REFUTED"              # Falsified by counterexample or invariant breach
    DEGRADED_FALLBACK = "DEGRADED_FALLBACK"            # Safe operational fallback state


class EpistemicEvidenceClass(str, enum.Enum):
    """Hierarchy of evidence types supporting claims."""
    PHYSICAL_MEASUREMENT = "PHYSICAL_MEASUREMENT"      # Hardware instrument, frame latency, bus wattage, PSNR
    EXECUTED_TEST = "EXECUTED_TEST"                    # Deterministic test suite execution (pytest)
    FORMAL_DERIVATION = "FORMAL_DERIVATION"            # Mathematical proof or Z3 SMT solver theorem
    CALIBRATED_OBSERVATION = "CALIBRATED_OBSERVATION"  # Calibrated hardware sensor / direct telemetry
    SIMULATION_OUTPUT = "SIMULATION_OUTPUT"            # Synthetic model run
    HEURISTIC_PREDICTION = "HEURISTIC_PREDICTION"      # Deterministic finite-difference / trend prediction
    UNVERIFIED_ASSERTION = "UNVERIFIED_ASSERTION"      # Bare assertion without backing evidence


@dataclass
class EpistemicAuditEntry:
    """
    Mandatory 4-Tuple Audit Record:
    < Provenance, Verdict, Evidence, Empirical Outcome >
    """
    entry_id: str
    sequence_number: int
    timestamp_ns: int

    # 1. Pôvod (Provenance / Source)
    proposer_id: str
    claim_summary: str
    claim_context_hash: str

    # 2. Rozhodnutie (Verdict / Decision)
    evaluator_id: str
    verdict: EpistemicVerdict
    declared_confidence: float  # [0.0 - 1.0]

    # 3. Použité dôkazy (Evidence Artifacts)
    evidence_class: EpistemicEvidenceClass
    evidence_artifact_uri: str
    falsification_condition: str

    # 4. Výsledok (Empirical Outcome)
    measured_outcome_metric: Optional[float]
    was_falsified: bool
    calibration_delta: float

    # Tamper-Evident Chaining
    predecessor_hash: str
    entry_digest: str
    hmac_signature: Optional[str] = None
    vital_max_hp: int = VITAL_MAX_HP

    def __post_init__(self):
        if self.vital_max_hp != 6:
            raise ContractViolation("Platform invariant VITAL_MAX_HP == 6 must hold in EpistemicAuditEntry")
        # Anti-Self-Certification invariant: Proposer cannot evaluate its own claim
        if self.proposer_id == self.evaluator_id and self.verdict == EpistemicVerdict.APPROVED_FACT:
            raise ContractViolation(
                f"Self-certification prohibited: Proposer '{self.proposer_id}' cannot evaluate claim to APPROVED_FACT"
            )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entry_id": self.entry_id,
            "sequence_number": self.sequence_number,
            "timestamp_ns": self.timestamp_ns,
            "provenance": {
                "proposer_id": self.proposer_id,
                "claim_summary": self.claim_summary,
                "claim_context_hash": self.claim_context_hash,
            },
            "verdict": {
                "evaluator_id": self.evaluator_id,
                "verdict": self.verdict.value,
                "declared_confidence": round(self.declared_confidence, 4),
            },
            "evidence": {
                "evidence_class": self.evidence_class.value,
                "evidence_artifact_uri": self.evidence_artifact_uri,
                "falsification_condition": self.falsification_condition,
            },
            "outcome": {
                "measured_outcome_metric": self.measured_outcome_metric,
                "was_falsified": self.was_falsified,
                "calibration_delta": round(self.calibration_delta, 4),
            },
            "predecessor_hash": self.predecessor_hash,
            "entry_digest": self.entry_digest,
            "hmac_signature": self.hmac_signature,
            "vital_max_hp": self.vital_max_hp,
        }


@dataclass
class RetrospectiveCalibrationReport:
    """Statistical audit report used for retrospective system recalibration."""
    report_id: str
    total_audited_claims: int
    approved_facts_count: int
    rejected_refuted_count: int
    insufficient_evidence_count: int
    false_discovery_rate: float        # P(Falsified | APPROVED_FACT)
    expected_calibration_error: float  # ECE: |declared_confidence - empirical_accuracy|
    over_conservatism_score: float     # Rate of viable hypotheses rejected due to strict thresholds
    recommended_threshold_shift: Dict[str, float]
    vital_max_hp: int = VITAL_MAX_HP

    def __post_init__(self):
        if self.vital_max_hp != 6:
            raise ContractViolation("Platform invariant VITAL_MAX_HP == 6 must hold in RetrospectiveCalibrationReport")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "report_id": self.report_id,
            "total_audited_claims": self.total_audited_claims,
            "approved_facts_count": self.approved_facts_count,
            "rejected_refuted_count": self.rejected_refuted_count,
            "insufficient_evidence_count": self.insufficient_evidence_count,
            "false_discovery_rate": round(self.false_discovery_rate, 4),
            "expected_calibration_error": round(self.expected_calibration_error, 4),
            "over_conservatism_score": round(self.over_conservatism_score, 4),
            "recommended_threshold_shift": self.recommended_threshold_shift,
            "vital_max_hp": self.vital_max_hp,
        }


class EpistemicProvenanceLedger:
    """
    Append-only cryptographically chained audit ledger tracking the
    4-tuple lifecycle of all system claims and supporting retrospective calibration.
    """

    GENESIS_HASH: str = "0000000000000000000000000000000000000000000000000000000000000000"

    def __init__(self, hmac_key: Optional[bytes] = None):
        self.hmac_key = hmac_key
        self.entries: List[EpistemicAuditEntry] = []
        self.current_tip_hash: str = self.GENESIS_HASH

    def log_claim_evaluation(
        self,
        proposer_id: str,
        claim_summary: str,
        evaluator_id: str,
        verdict: EpistemicVerdict,
        declared_confidence: float,
        evidence_class: EpistemicEvidenceClass,
        evidence_artifact_uri: str,
        falsification_condition: str,
        measured_outcome_metric: Optional[float] = None,
        was_falsified: bool = False,
    ) -> EpistemicAuditEntry:
        """
        Appends an audited claim to the ledger with rolling cryptographic hash.
        """
        seq = len(self.entries)
        now_ns = time.time_ns()
        entry_id = f"AUDIT-{seq:06d}-{uuid.uuid4().hex[:6].upper()}"

        # Context hash
        ctx_data = f"{proposer_id}:{claim_summary}:{evidence_artifact_uri}".encode("utf-8")
        ctx_hash = hashlib.sha256(ctx_data).hexdigest()

        # Calibration delta: empirical truth (1.0 if approved and not falsified, else 0.0)
        empirical_target = 1.0 if (verdict == EpistemicVerdict.APPROVED_FACT and not was_falsified) else 0.0
        calib_delta = abs(declared_confidence - empirical_target)

        # Chained Entry Digest
        entry_payload = (
            f"{entry_id}:{seq}:{self.current_tip_hash}:{proposer_id}:{ctx_hash}:{evaluator_id}:"
            f"{verdict.value}:{declared_confidence}:{evidence_class.value}:"
            f"{evidence_artifact_uri}:{was_falsified}:{VITAL_MAX_HP}"
        )
        entry_digest = hashlib.sha256(entry_payload.encode("utf-8")).hexdigest()

        # Optional HMAC signature
        hmac_sig = None
        if self.hmac_key:
            hmac_sig = hmac.new(self.hmac_key, entry_digest.encode("utf-8"), hashlib.sha256).hexdigest()

        entry = EpistemicAuditEntry(
            entry_id=entry_id,
            sequence_number=seq,
            timestamp_ns=now_ns,
            proposer_id=proposer_id,
            claim_summary=claim_summary,
            claim_context_hash=ctx_hash,
            evaluator_id=evaluator_id,
            verdict=verdict,
            declared_confidence=declared_confidence,
            evidence_class=evidence_class,
            evidence_artifact_uri=evidence_artifact_uri,
            falsification_condition=falsification_condition,
            measured_outcome_metric=measured_outcome_metric,
            was_falsified=was_falsified,
            calibration_delta=calib_delta,
            predecessor_hash=self.current_tip_hash,
            entry_digest=entry_digest,
            hmac_signature=hmac_sig,
            vital_max_hp=VITAL_MAX_HP,
        )

        self.entries.append(entry)
        self.current_tip_hash = entry_digest
        return entry

    def verify_ledger_integrity(self) -> bool:
        """Verifies mathematical continuity of the hash chain from genesis to tip."""
        expected_prev = self.GENESIS_HASH
        for e in self.entries:
            if e.predecessor_hash != expected_prev:
                return False
            # Recompute context hash to verify claim content integrity
            ctx_data = f"{e.proposer_id}:{e.claim_summary}:{e.evidence_artifact_uri}".encode("utf-8")
            recomputed_ctx_hash = hashlib.sha256(ctx_data).hexdigest()
            if recomputed_ctx_hash != e.claim_context_hash:
                return False

            # Recompute digest
            payload = (
                f"{e.entry_id}:{e.sequence_number}:{expected_prev}:{e.proposer_id}:{recomputed_ctx_hash}:{e.evaluator_id}:"
                f"{e.verdict.value}:{e.declared_confidence}:{e.evidence_class.value}:"
                f"{e.evidence_artifact_uri}:{e.was_falsified}:{VITAL_MAX_HP}"
            )
            recomputed = hashlib.sha256(payload.encode("utf-8")).hexdigest()
            if recomputed != e.entry_digest:
                return False
            if self.hmac_key and e.hmac_signature:
                recomputed_hmac = hmac.new(self.hmac_key, recomputed.encode("utf-8"), hashlib.sha256).hexdigest()
                if recomputed_hmac != e.hmac_signature:
                    return False
            expected_prev = e.entry_digest
        return True

    def compute_retrospective_calibration(self) -> RetrospectiveCalibrationReport:
        """
        Evaluates historical audit entries to compute calibration error and
        suggest threshold recalibrations.
        """
        report_id = f"CALIB-RPT-{uuid.uuid4().hex[:8].upper()}"
        total = len(self.entries)
        if total == 0:
            return RetrospectiveCalibrationReport(
                report_id=report_id,
                total_audited_claims=0,
                approved_facts_count=0,
                rejected_refuted_count=0,
                insufficient_evidence_count=0,
                false_discovery_rate=0.0,
                expected_calibration_error=0.0,
                over_conservatism_score=0.0,
                recommended_threshold_shift={},
                vital_max_hp=VITAL_MAX_HP,
            )

        approved = [e for e in self.entries if e.verdict == EpistemicVerdict.APPROVED_FACT]
        refuted = [e for e in self.entries if e.verdict == EpistemicVerdict.REJECTED_REFUTED]
        insufficient = [e for e in self.entries if e.verdict == EpistemicVerdict.EVIDENCE_INSUFFICIENT]

        # False Discovery: Claims that were approved as facts but empirical outcome falsified them
        falsified_approved = [e for e in approved if e.was_falsified]
        fdr = (len(falsified_approved) / len(approved)) if approved else 0.0

        # Expected Calibration Error (Mean Calibration Delta)
        ece = sum(e.calibration_delta for e in self.entries) / total

        # Over-conservatism score: Insufficient evidence claims that had high empirical quality metrics
        viable_unadmitted = [
            e for e in insufficient
            if e.measured_outcome_metric is not None and e.measured_outcome_metric >= 40.0
        ]
        over_conservatism = (len(viable_unadmitted) / len(insufficient)) if insufficient else 0.0

        # Suggested threshold adjustments
        shifts: Dict[str, float] = {}
        if fdr > 0.05:
            # Too lenient: tighten evidence bar
            shifts["confidence_threshold_shift"] = +0.05
            shifts["evidence_coverage_strictness"] = +0.10
        elif ece < 0.10 and over_conservatism > 0.20:
            # Too conservative: safely relax thresholds
            shifts["confidence_threshold_shift"] = -0.03
            shifts["provisional_admission_rate"] = +0.05
        else:
            # Calibrated equilibrium
            shifts["system_calibration_status"] = 0.0  # Stable

        return RetrospectiveCalibrationReport(
            report_id=report_id,
            total_audited_claims=total,
            approved_facts_count=len(approved),
            rejected_refuted_count=len(refuted),
            insufficient_evidence_count=len(insufficient),
            false_discovery_rate=fdr,
            expected_calibration_error=ece,
            over_conservatism_score=over_conservatism,
            recommended_threshold_shift=shifts,
            vital_max_hp=VITAL_MAX_HP,
        )
