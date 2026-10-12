#!/usr/bin/env python3
"""
==============================================================================
KRYSTAL-STACK: AUTHORITY PROOF BUNDLE & VERIFICATION TAXONOMY
==============================================================================
Module: krystal_kernel/verification/proof_bundle.py
Description:
    Implements machine-readable cryptographic and causal proof bundles for
    deterministic execution verdicts.

    Core Invariant:
      AuthorityVerdict must NEVER be a naked enum.
      Every verdict (CommitApproved, DeferInsufficientEvidence,
      QuarantineContradictoryEvidence, VetoSafetyBreach) must emit a structured,
      self-contained proof bundle consumable by UI and audit logs directly
      without requiring causal reconstruction after the fact.
==============================================================================
"""

import time
import hashlib
from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional

from krystal_kernel.verification.state_version import StateVersion
from krystal_kernel.verification.policy_registry import PolicyVersion


class ReversibilityClass(str, Enum):
    REVERSIBLE = "REVERSIBLE"          # In-memory undoable (e.g. CPU nice priority)
    COMPENSABLE = "COMPENSABLE"        # Requires inverse transaction (e.g. balloon resize)
    IRREVERSIBLE = "IRREVERSIBLE"      # Permanent hardware state modification


class SafetyTier(str, Enum):
    MEMORY_SAFETY = "MEMORY_SAFETY"          # Ownership, lifetime, pointer bounds, UAF prevention
    RESOURCE_SAFETY = "RESOURCE_SAFETY"      # OS cgroup limits, balloon headroom, OOM risk
    PHYSICAL_SAFETY = "PHYSICAL_SAFETY"      # Thermal policy, power ceiling, hardware fault states
    SYSTEM_INVARIANTS = "SYSTEM_INVARIANTS"  # StateVersion continuity, VITAL_MAX_HP == 6, zero mutation


class SafetyVerdictType(str, Enum):
    PASS = "PASS"
    VETO = "VETO"


class EvidenceVerdictType(str, Enum):
    SUFFICIENT = "SUFFICIENT"
    INSUFFICIENT = "INSUFFICIENT"
    CONTRADICTORY = "CONTRADICTORY"


class AuthorityVerdictType(str, Enum):
    COMMIT_APPROVED = "COMMIT_APPROVED"                                  # In Sprint A: SHADOW_COMMIT_APPROVED
    DEFER_INSUFFICIENT_EVIDENCE = "DEFER_INSUFFICIENT_EVIDENCE"          # Increase targeted sampling
    QUARANTINE_CONTRADICTORY_EVIDENCE = "QUARANTINE_CONTRADICTORY_EVIDENCE" # Freeze stream & trigger recalibration
    VETO_SAFETY_BREACH = "VETO_SAFETY_BREACH"                            # Pre-actuation rejection -> REJECT_NOOP


class FailureLifecycleStage(str, Enum):
    PRE_ACTUATION_VETO = "PRE_ACTUATION_VETO"        # Pre-actuation veto -> REJECT_NOOP
    POST_PREPARE_FAILURE = "POST_PREPARE_FAILURE"    # Staging failure -> ABORT_STAGED_STATE
    POST_COMMIT_DIVERGENCE = "POST_COMMIT_DIVERGENCE" # Commit divergence -> COMPENSATE


@dataclass
class SafetyViolationRecord:
    tier: SafetyTier
    violation_code: str
    details: str
    measured_value: Any
    threshold_value: Any
    timestamp_ns: int = field(default_factory=lambda: int(time.time() * 1e9))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tier": self.tier.value,
            "violation_code": self.violation_code,
            "details": self.details,
            "measured_value": self.measured_value,
            "threshold_value": self.threshold_value,
            "timestamp_ns": self.timestamp_ns
        }


@dataclass
class SafetyProof:
    """
    Exhaustive formal verification proof emitted by SafetyGate across all 4 safety domains.
    """
    verdict: SafetyVerdictType
    memory_safety_check: Dict[str, Any]
    resource_safety_check: Dict[str, Any]
    physical_safety_check: Dict[str, Any]
    system_invariants_check: Dict[str, Any]
    policy_version: PolicyVersion
    evaluated_at_ns: int
    violations: List[SafetyViolationRecord] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "verdict": self.verdict.value,
            "memory_safety_check": self.memory_safety_check,
            "resource_safety_check": self.resource_safety_check,
            "physical_safety_check": self.physical_safety_check,
            "system_invariants_check": self.system_invariants_check,
            "policy_version": self.policy_version.to_dict(),
            "evaluated_at_ns": self.evaluated_at_ns,
            "violations": [v.to_dict() for v in self.violations]
        }


@dataclass
class EvidenceProof:
    """
    Exhaustive epistemic lattice verification proof emitted by EvidenceGate.
    """
    verdict: EvidenceVerdictType
    freshness_eval: Dict[str, Any]
    provenance_eval: Dict[str, Any]
    cross_sensor_consistency: Dict[str, Any]
    sample_count_eval: Dict[str, Any]
    calibration_error_eval: Dict[str, Any]
    domain_eval: Dict[str, Any]
    state_version_eval: Dict[str, Any]
    clock_validity_eval: Dict[str, Any]
    evaluated_at_ns: int
    contradictions: List[Dict[str, Any]] = field(default_factory=list)
    gaps: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "verdict": self.verdict.value,
            "freshness_eval": self.freshness_eval,
            "provenance_eval": self.provenance_eval,
            "cross_sensor_consistency": self.cross_sensor_consistency,
            "sample_count_eval": self.sample_count_eval,
            "calibration_error_eval": self.calibration_error_eval,
            "domain_eval": self.domain_eval,
            "state_version_eval": self.state_version_eval,
            "clock_validity_eval": self.clock_validity_eval,
            "evaluated_at_ns": self.evaluated_at_ns,
            "contradictions": self.contradictions,
            "gaps": self.gaps
        }


@dataclass
class AuthorityProofBundle:
    """
    Comprehensive, machine-readable proof bundle generated for every decision.
    Unifies safety proofs, epistemic proofs, causal lineages, and lifecycle directives.
    """
    verdict: AuthorityVerdictType
    safety_proof: SafetyProof
    evidence_proof: EvidenceProof
    policy_version: PolicyVersion
    state_version: StateVersion
    evaluated_at_ns: int
    evidence_refs: List[str]
    action_id: str
    lifecycle_action: str                      # "REJECT_NOOP" | "ABORT_STAGED_STATE" | "COMPENSATE" | "WOULD_COMMIT" | "DEFER_SAMPLE" | "QUARANTINE_RECALIBRATE"
    audit_hash: str = ""

    def __post_init__(self):
        if not self.audit_hash:
            self.audit_hash = self.compute_audit_hash()

    def compute_audit_hash(self) -> str:
        payload = f"{self.verdict.value}:{self.action_id}:{self.state_version.version_digest()}:{self.evaluated_at_ns}:{self.safety_proof.verdict.value}:{self.evidence_proof.verdict.value}"
        return hashlib.sha256(payload.encode('utf-8')).hexdigest()[:24]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "verdict": self.verdict.value,
            "safety_proof": self.safety_proof.to_dict(),
            "evidence_proof": self.evidence_proof.to_dict(),
            "policy_version": self.policy_version.to_dict(),
            "state_version": self.state_version.to_dict(),
            "evaluated_at_ns": self.evaluated_at_ns,
            "evidence_refs": self.evidence_refs,
            "action_id": self.action_id,
            "lifecycle_action": self.lifecycle_action,
            "audit_hash": self.audit_hash
        }
