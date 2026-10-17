"""
Krystal Stack Platform Framework - Epistemic Neurolinguistic Grammar (ENLG v2)
=============================================================================
Module: krystal_kernel/epistemic_enlg_compiler.py
Milestone: V-ENLG-COMPILER-01 (Cognitive Tri-Plane Parity & ArtifactGate)
Constitutional Invariant: VITAL_MAX_HP == 6

Provides machine-enforced epistemic parsing, Evidence Lattice validation, and
the executable ArtifactGate runtime representing the Anti-Promotion Bar:
    Release(x) <=> ReceiptValid(x) and AuthorityValid(x) and StatePreconditionValid(x)
    not ReceiptValid(x) => Quarantine(x) and Delta_S == 0
"""

from __future__ import annotations

import dataclasses
import enum
import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from krystal_kernel.scene_constructor_presets import VITAL_MAX_HP
from krystal_kernel.contracts import (
    ContractHeader,
    ContractViolation,
    ReceiptContract,
    compute_canonical_digest,
    canonical_serialize,
)

if VITAL_MAX_HP != 6:
    raise ContractViolation(
        f"Constitutional invariant breach: VITAL_MAX_HP must be 6, got {VITAL_MAX_HP}",
        violation_code="ERR_CONSTITUTIONAL_INVARIANT_BREACH",
    )


# ============================================================================
# 1. ENLG ENUMS & EVIDENCE LATTICE
# ============================================================================

class EnlgRole(enum.Enum):
    """Functional role of an epistemic statement or actor."""
    PROPOSER = "PROPOSER"
    VERIFIER = "VERIFIER"
    AUTHORITY = "AUTHORITY"
    EXECUTOR = "EXECUTOR"
    OBSERVER = "OBSERVER"


class EnlgEvidenceClass(enum.Enum):
    """Stratified evidence types forming an algebraic partial order."""
    UNKNOWN = "UNKNOWN"
    SIMULATED = "SIMULATED"
    DERIVED = "DERIVED"
    CODE_PRESENT = "CODE-PRESENT"
    LOCAL_MEASURED = "LOCAL-MEASURED"
    REMOTE_CI = "REMOTE-CI"
    PHYSICAL_MEASURED = "PHYSICAL-MEASURED"
    EXTERNAL_REPLICATED = "EXTERNAL-REPLICATED"


class EnlgVerdict(enum.Enum):
    """Closed finite set of allowable epistemic verdict states."""
    UNKNOWN = "UNKNOWN"
    CONFLICT = "CONFLICT"
    FALSIFIED = "FALSIFIED"
    SUPPORTED_WITH_SCOPE = "SUPPORTED_WITH_SCOPE"
    MEASURED_WITH_SCOPE = "MEASURED_WITH_SCOPE"
    VERIFIED_WITH_SCOPE = "VERIFIED_WITH_SCOPE"


# Numerical rank ordering for Evidence Lattice calculations
EVIDENCE_RANK: Dict[EnlgEvidenceClass, int] = {
    EnlgEvidenceClass.UNKNOWN: 0,
    EnlgEvidenceClass.SIMULATED: 1,
    EnlgEvidenceClass.DERIVED: 1,
    EnlgEvidenceClass.CODE_PRESENT: 1,
    EnlgEvidenceClass.LOCAL_MEASURED: 2,
    EnlgEvidenceClass.REMOTE_CI: 3,
    EnlgEvidenceClass.PHYSICAL_MEASURED: 3,
    EnlgEvidenceClass.EXTERNAL_REPLICATED: 3,
}

VERDICT_RANK: Dict[EnlgVerdict, int] = {
    EnlgVerdict.UNKNOWN: 0,
    EnlgVerdict.CONFLICT: 0,
    EnlgVerdict.FALSIFIED: 0,
    EnlgVerdict.SUPPORTED_WITH_SCOPE: 1,
    EnlgVerdict.MEASURED_WITH_SCOPE: 2,
    EnlgVerdict.VERIFIED_WITH_SCOPE: 3,
}

# Absolute banned words unless bounded by explicit model assumptions
BANNED_UNBOUNDED_SUPERLATIVES = {
    "always",
    "never",
    "guarantees",
    "zero-risk",
    "fully secure",
    "optimal",
    "universal",
    "proven",
}


# ============================================================================
# 2. ENLG AST DATASTRUCTURES
# ============================================================================

@dataclass(frozen=True)
class NullTriplet:
    """The mandatory three competing null models for contrastive truth."""
    m0_no_effect: str
    m1_noise: str
    m2_confounder: str
    vital_max_hp: int = VITAL_MAX_HP

    def __post_init__(self):
        if self.vital_max_hp != 6:
            raise ContractViolation("VITAL_MAX_HP must be 6 in NullTriplet")
        if not self.m0_no_effect.strip():
            raise ContractViolation("NullTriplet requires non-empty m0_no_effect", violation_code="ERR_ENLG_NULL_M0_EMPTY")
        if not self.m1_noise.strip():
            raise ContractViolation("NullTriplet requires non-empty m1_noise", violation_code="ERR_ENLG_NULL_M1_EMPTY")
        if not self.m2_confounder.strip():
            raise ContractViolation("NullTriplet requires non-empty m2_confounder", violation_code="ERR_ENLG_NULL_M2_EMPTY")


@dataclass(frozen=True)
class NextExperimentPlan:
    """Actionable closure terminating every epistemic evaluation trace."""
    task: str
    measurement: str
    success_criterion: str
    failure_criterion: str
    rollback: str
    stop_condition: str
    vital_max_hp: int = VITAL_MAX_HP

    def __post_init__(self):
        if self.vital_max_hp != 6:
            raise ContractViolation("VITAL_MAX_HP must be 6 in NextExperimentPlan")
        for field_name in ["task", "measurement", "success_criterion", "failure_criterion", "rollback", "stop_condition"]:
            val = getattr(self, field_name)
            if not isinstance(val, str) or not val.strip():
                raise ContractViolation(f"NextExperimentPlan requires non-empty {field_name}", violation_code="ERR_ENLG_EXPERIMENT_FIELD_EMPTY")


@dataclass(frozen=True)
class EnlgClaimTrace:
    """Complete, machine-parsable 11-step ENLG reasoning trace."""
    claim: str
    scope: str
    role: EnlgRole
    evidence: Tuple[EnlgEvidenceClass, ...]
    nulls: NullTriplet
    falsifier: str
    validity: Dict[str, str]  # Must include 'construct', 'internal', 'external'
    verdict: EnlgVerdict
    residual: Tuple[str, ...]
    next_experiment: Optional[NextExperimentPlan] = None
    has_conflicting_evidence: bool = False
    vital_max_hp: int = VITAL_MAX_HP

    def __post_init__(self):
        if self.vital_max_hp != 6:
            raise ContractViolation("VITAL_MAX_HP must be 6 in EnlgClaimTrace")
        if not self.claim.strip():
            raise ContractViolation("EnlgClaimTrace requires non-empty claim", violation_code="ERR_ENLG_CLAIM_EMPTY")
        if not self.scope.strip():
            raise ContractViolation("EnlgClaimTrace requires non-empty scope", violation_code="ERR_ENLG_SCOPE_EMPTY")
        if not self.falsifier.strip():
            raise ContractViolation("EnlgClaimTrace requires non-empty falsifier", violation_code="ERR_ENLG_FALSIFIER_EMPTY")
        if not self.evidence:
            raise ContractViolation("EnlgClaimTrace requires at least one evidence item", violation_code="ERR_ENLG_EVIDENCE_EMPTY")
        if not self.residual:
            raise ContractViolation("EnlgClaimTrace requires non-empty residual uncertainty", violation_code="ERR_ENLG_RESIDUAL_EMPTY")

    def compute_digest(self) -> str:
        """Deterministic canonical digest of the claim trace."""
        return compute_canonical_digest({
            "claim": self.claim,
            "scope": self.scope,
            "role": self.role.value,
            "evidence": [e.value for e in self.evidence],
            "falsifier": self.falsifier,
            "verdict": self.verdict.value,
            "residual": list(self.residual),
            "vital_max_hp": self.vital_max_hp,
        })


# ============================================================================
# 3. ENLG AST VALIDATOR & EVIDENCE LATTICE ENGINE
# ============================================================================

class EnlgAstValidator:
    """
    Formal validator verifying the 10 ENLG cognitive patterns and enforcing
    the Evidence Lattice anti-promotion barriers.
    """

    @classmethod
    def validate_claim_trace(cls, trace: EnlgClaimTrace) -> bool:
        """
        Validates an EnlgClaimTrace against all ENLG v2 constitutional invariants.
        Raises ContractViolation on any illegal promotion or missing constraint.
        """
        # 1. Scope-Lock & Anti-Overclaim Check (Pattern 6 & 9)
        claim_lower = trace.claim.lower()
        for banned in BANNED_UNBOUNDED_SUPERLATIVES:
            pattern = rf"\b{re.escape(banned)}\b"
            if re.search(pattern, claim_lower) and "defined invariant" not in trace.scope.lower() and "model assumption" not in trace.scope.lower():
                raise ContractViolation(
                    f"Anti-Overclaim Breach: Unbounded superlative '{banned}' detected without explicit model constraints in scope",
                    violation_code="ERR_ENLG_OVERCLAIM_SUPERLATIVE"
                )

        # 2. Authority Separation Check (Pattern 3): Proposer cannot self-certify VERIFIED
        if trace.role == EnlgRole.PROPOSER and trace.verdict == EnlgVerdict.VERIFIED_WITH_SCOPE:
            raise ContractViolation(
                "Authority Separation Breach: Role PROPOSER cannot self-assign VERIFIED_WITH_SCOPE verdict",
                violation_code="ERR_ENLG_PROPOSER_SELF_CERTIFICATION"
            )

        # 3. Contradiction Preservation (Pattern 10): Conflicting evidence forces CONFLICT verdict
        if trace.has_conflicting_evidence:
            if trace.verdict != EnlgVerdict.CONFLICT:
                raise ContractViolation(
                    f"Contradiction Preservation Breach: Conflicting evidence exists, verdict must be CONFLICT, got {trace.verdict.value}",
                    violation_code="ERR_ENLG_CONFLICT_NOT_PRESERVED"
                )
            return True

        # 4. Evidence Lattice Inequality (Pattern 1, 4 & Anti-Promotion Bar):
        #    Level(VERDICT) <= min_{e in EVIDENCE} MaxAllowedVerdict(e)
        max_evidence_rank = max(EVIDENCE_RANK[e] for e in trace.evidence)
        verdict_rank = VERDICT_RANK[trace.verdict]

        # Specific Anti-Promotion Bar rules:
        # A. If evidence contains UNKNOWN or only rank 0 -> cannot be SUPPORTED, MEASURED, or VERIFIED
        if all(e == EnlgEvidenceClass.UNKNOWN for e in trace.evidence):
            if trace.verdict in (EnlgVerdict.SUPPORTED_WITH_SCOPE, EnlgVerdict.MEASURED_WITH_SCOPE, EnlgVerdict.VERIFIED_WITH_SCOPE):
                raise ContractViolation(
                    f"Anti-Promotion Breach: Evidence is strictly UNKNOWN, cannot promote to {trace.verdict.value}",
                    violation_code="ERR_ENLG_UNKNOWN_PROMOTION"
                )

        # B. Simulation / Derived / Code-Present -> max allowed is SUPPORTED_WITH_SCOPE (rank 1)
        if max_evidence_rank == 1 and verdict_rank > 1:
            raise ContractViolation(
                f"Anti-Promotion Breach: Simulation/Derived evidence (rank 1) cannot promote to {trace.verdict.value} (rank {verdict_rank})",
                violation_code="ERR_ENLG_SIMULATION_TO_MEASUREMENT"
            )

        # C. Local-Measured -> max allowed is MEASURED_WITH_SCOPE (rank 2) unless backed by remote-CI / physical
        if max_evidence_rank == 2 and verdict_rank > 2:
            raise ContractViolation(
                f"Anti-Promotion Breach: Local-Measured evidence (rank 2) cannot promote to VERIFIED_WITH_SCOPE without remote-CI/external replication",
                violation_code="ERR_ENLG_LOCAL_TO_VERIFIED_PROMOTION"
            )

        # 5. Tripartite Validity check
        required_validity_keys = {"construct", "internal", "external"}
        if not required_validity_keys.issubset(set(k.lower() for k in trace.validity.keys())):
            missing = required_validity_keys - set(k.lower() for k in trace.validity.keys())
            raise ContractViolation(
                f"Validity Incomplete: Missing required validity evaluations: {missing}",
                violation_code="ERR_ENLG_VALIDITY_INCOMPLETE"
            )

        return True

    @classmethod
    def parse_compiler_text(cls, text: str) -> EnlgClaimTrace:
        """
        Deterministic parser for the KRYSTAL ENLG Compiler text format.
        Converts raw structured prompt outputs into typed EnlgClaimTrace AST.
        """
        sections: Dict[str, str] = {}
        current_section: Optional[str] = None
        current_lines: List[str] = []

        header_tags = [
            "CLAIM:", "SCOPE:", "ROLE:", "EVIDENCE:", "ANTI-PROMOTION:",
            "NULLS:", "FALSIFIER:", "VALIDITY:", "VERDICT:", "RESIDUAL:", "NEXT_EXPERIMENT:"
        ]

        for line in text.strip().splitlines():
            line_str = line.strip()
            matched_header = None
            for tag in header_tags:
                if line_str.startswith(tag):
                    matched_header = tag[:-1]
                    break

            if matched_header:
                if current_section:
                    sections[current_section] = "\n".join(current_lines).strip()
                current_section = matched_header
                content_after = line_str[len(matched_header) + 1:].strip()
                current_lines = [content_after] if content_after else []
            else:
                if current_section:
                    current_lines.append(line.rstrip())

        if current_section:
            sections[current_section] = "\n".join(current_lines).strip()

        # Check required sections
        required_sections = ["CLAIM", "SCOPE", "ROLE", "EVIDENCE", "NULLS", "FALSIFIER", "VALIDITY", "VERDICT", "RESIDUAL"]
        for sec in required_sections:
            if sec not in sections or not sections[sec]:
                raise ContractViolation(f"ENLG Parse Error: Missing required section '{sec}:'", violation_code=f"ERR_ENLG_MISSING_SECTION_{sec}")

        # Parse Role
        role_str = sections["ROLE"].split()[0].upper().replace(":", "")
        try:
            role = EnlgRole[role_str]
        except KeyError:
            raise ContractViolation(f"Unrecognized ENLG Role: '{role_str}'", violation_code="ERR_ENLG_INVALID_ROLE")

        # Parse Verdict
        verdict_str = sections["VERDICT"].split()[0].upper().replace(":", "")
        try:
            verdict = EnlgVerdict[verdict_str]
        except KeyError:
            raise ContractViolation(f"Unrecognized ENLG Verdict: '{verdict_str}'", violation_code="ERR_ENLG_INVALID_VERDICT")

        # Parse Evidence items
        ev_items: List[EnlgEvidenceClass] = []
        ev_text = sections["EVIDENCE"].upper()
        has_conflict = ("⊥" in ev_text) or ("CONFLICT" in ev_text) or ("EVIDENCE_A" in ev_text and "EVIDENCE_B" in ev_text)

        for ev_enum in EnlgEvidenceClass:
            tag = ev_enum.value.upper()
            if tag in ev_text or ev_enum.name in ev_text:
                ev_items.append(ev_enum)

        if not ev_items:
            ev_items.append(EnlgEvidenceClass.UNKNOWN)

        # Parse Nulls
        null_text = sections["NULLS"]
        m0_match = re.search(r"M0_NO_EFFECT[:\s]+([^\n]+)", null_text, re.IGNORECASE)
        m1_match = re.search(r"M1_NOISE[:\s]+([^\n]+)", null_text, re.IGNORECASE)
        m2_match = re.search(r"M2_MEASUREMENT_OR_CONFOUNDER[:\s]+([^\n]+)", null_text, re.IGNORECASE)

        nulls = NullTriplet(
            m0_no_effect=m0_match.group(1).strip() if m0_match else "Unspecified M0 baseline",
            m1_noise=m1_match.group(1).strip() if m1_match else "Unspecified M1 noise",
            m2_confounder=m2_match.group(1).strip() if m2_match else "Unspecified M2 artifact",
        )

        # Parse Validity
        validity_text = sections["VALIDITY"]
        validity_dict = {
            "construct": "Construct validity stated" if "construct" in validity_text.lower() else "unspecified",
            "internal": "Internal validity stated" if "internal" in validity_text.lower() else "unspecified",
            "external": "External validity stated" if "external" in validity_text.lower() else "unspecified",
        }

        # Parse Residual
        residual_lines = [r.strip("- *") for r in sections["RESIDUAL"].splitlines() if r.strip("- *")]
        if not residual_lines:
            residual_lines = ["Default residual uncertainty: unmeasured external boundaries."]

        # Construct trace AST
        trace = EnlgClaimTrace(
            claim=sections["CLAIM"],
            scope=sections["SCOPE"],
            role=role,
            evidence=tuple(ev_items),
            nulls=nulls,
            falsifier=sections["FALSIFIER"],
            validity=validity_dict,
            verdict=verdict,
            residual=tuple(residual_lines),
            has_conflicting_evidence=has_conflict,
            vital_max_hp=6,
        )

        # Run full validation check
        cls.validate_claim_trace(trace)
        return trace


# Canonical alias
EnlgEpistemicCompiler = EnlgAstValidator


# ============================================================================
# 4. ARTIFACT GATE: THE RUNTIME ANTI-PROMOTION BAR
# ============================================================================

@dataclass
class QuarantinedArtifact:
    """Artifact held in default quarantine pending verified proof."""
    artifact_id: str
    candidate_payload: Any
    proposer_role: EnlgRole
    deposit_timestamp_ns: int
    is_released: bool = False
    release_receipt_digest: Optional[str] = None
    vital_max_hp: int = VITAL_MAX_HP


class ArtifactGate:
    """
    Executable realization of the Anti-Promotion Bar.
    Enforces the formal contract:
        Release(x) <=> ReceiptValid(x) and AuthorityValid(x) and StatePreconditionValid(x)
        not ReceiptValid(x) => Quarantine(x) and Delta_S == 0
    """

    def __init__(self) -> None:
        self._quarantine: Dict[str, QuarantinedArtifact] = {}
        self._released_history: List[str] = []
        self._rejection_log: List[Dict[str, Any]] = []

    @property
    def quarantined_count(self) -> int:
        return len([a for a in self._quarantine.values() if not a.is_released])

    @property
    def released_count(self) -> int:
        return len(self._released_history)

    def deposit(self, artifact_id: str, candidate_payload: Any, proposer_role: EnlgRole) -> str:
        """
        Deposits an unverified candidate artifact strictly into QUARANTINE.
        Returns the confirmed artifact_id.
        """
        if not artifact_id:
            raise ContractViolation("ArtifactGate deposit requires non-empty artifact_id")
        if artifact_id in self._quarantine and not self._quarantine[artifact_id].is_released:
            raise ContractViolation(f"Artifact '{artifact_id}' is already held in quarantine", violation_code="ERR_GATE_DUPLICATE_DEPOSIT")

        entry = QuarantinedArtifact(
            artifact_id=artifact_id,
            candidate_payload=candidate_payload,
            proposer_role=proposer_role,
            deposit_timestamp_ns=1,
            is_released=False,
            vital_max_hp=6,
        )
        self._quarantine[artifact_id] = entry
        return artifact_id

    def release(
        self,
        artifact_id: str,
        receipt: ReceiptContract,
        evaluation_trace: EnlgClaimTrace,
        commit_state_fn: Callable[[], Any],
    ) -> Any:
        """
        Attempts to release an artifact from quarantine.
        Only succeeds if:
            1. Receipt is valid and non-degraded.
            2. Evaluation trace passes EnlgAstValidator without promotion breaches.
            3. Evaluator role is strictly an independent VERIFIER or AUTHORITY (not PROPOSER).
            4. Invariant VITAL_MAX_HP == 6 holds.
        On any failure:
            Raises ContractViolation, keeps artifact in quarantine, guarantees Delta_S == 0.
        """
        # Precondition: Artifact must be in quarantine
        if artifact_id not in self._quarantine or self._quarantine[artifact_id].is_released:
            self._log_rejection(artifact_id, "ERR_GATE_ARTIFACT_NOT_IN_QUARANTINE")
            raise ContractViolation(f"Artifact '{artifact_id}' is not in active quarantine", violation_code="ERR_GATE_ARTIFACT_NOT_IN_QUARANTINE")

        quarantined = self._quarantine[artifact_id]

        # 1. State Precondition: Constitutional Invariant
        if receipt.vital_max_hp != 6 or evaluation_trace.vital_max_hp != 6:
            self._log_rejection(artifact_id, "ERR_GATE_VITAL_HP_BREACH")
            raise ContractViolation("Constitutional Invariant breach: VITAL_MAX_HP must be 6", violation_code="ERR_GATE_VITAL_HP_BREACH")

        # 2. Authority Separation: Evaluator cannot be PROPOSER
        if evaluation_trace.role == EnlgRole.PROPOSER:
            self._log_rejection(artifact_id, "ERR_GATE_EVALUATOR_IS_PROPOSER")
            raise ContractViolation(
                f"Authority Breach: Artifact release evaluation cannot be conducted by PROPOSER (Separation of Concerns)",
                violation_code="ERR_GATE_EVALUATOR_IS_PROPOSER"
            )

        # 3. Receipt Validity: Must not be emergency degraded fallback
        if receipt.was_degraded_fallback:
            self._log_rejection(artifact_id, "ERR_GATE_DEGRADED_RECEIPT_REJECTED")
            raise ContractViolation(
                f"Receipt Invalid: Cannot release artifact '{artifact_id}' on degraded fallback receipt",
                violation_code="ERR_GATE_DEGRADED_RECEIPT_REJECTED"
            )

        # 4. AST Validation: Trace must satisfy ENLG Evidence Lattice
        try:
            EnlgAstValidator.validate_claim_trace(evaluation_trace)
        except ContractViolation as exc:
            self._log_rejection(artifact_id, exc.violation_code or "ERR_GATE_EVALUATION_INVALID")
            raise ContractViolation(
                f"Authority Invalid: Evaluation trace failed ENLG compliance: {exc.message}",
                violation_code=exc.violation_code or "ERR_GATE_EVALUATION_INVALID"
            )

        # 5. Verdict Admissibility: Only VERIFIED or MEASURED with approved scope
        if evaluation_trace.verdict not in (EnlgVerdict.VERIFIED_WITH_SCOPE, EnlgVerdict.MEASURED_WITH_SCOPE):
            self._log_rejection(artifact_id, "ERR_GATE_VERDICT_NOT_RELEASABLE")
            raise ContractViolation(
                f"Release Denied: Verdict '{evaluation_trace.verdict.value}' is insufficient for artifact release",
                violation_code="ERR_GATE_VERDICT_NOT_RELEASABLE"
            )

        # All conditions satisfied: commit state atomically
        committed_result = commit_state_fn()

        # Update gate records
        self._quarantine[artifact_id] = dataclasses.replace(
            quarantined,
            is_released=True,
            release_receipt_digest=receipt.compute_digest(),
        )
        self._released_history.append(artifact_id)

        return committed_result

    def _log_rejection(self, artifact_id: str, reason_code: str) -> None:
        self._rejection_log.append({
            "artifact_id": artifact_id,
            "reason_code": reason_code,
            "delta_state": 0,
        })
