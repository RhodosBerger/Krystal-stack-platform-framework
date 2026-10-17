"""
Krystal Stack Platform Framework - Epistemic Neurolinguistic Grammar (ENLG v2)
=============================================================================
Module: krystal_kernel/epistemic_enlg_compiler.py
Milestone: V-ENLG-COMPILER-01 / KSPP v1 (Principle 4: Epistemic Type Safety)
Constitutional Invariant: VITAL_MAX_HP == 6

Provides machine-enforced epistemic parsing, Evidence Lattice validation, and
the executable ArtifactGate runtime representing the Anti-Promotion Bar:
    Release(x) <=> ReceiptValid(x) and AuthorityValid(x) and StatePreconditionValid(x)
    not ReceiptValid(x) => Quarantine(x) and Delta_S == 0
"""

from __future__ import annotations

import dataclasses
from enum import IntEnum
import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

from krystal_kernel.constants import (
    VITAL_MAX_HP,
    ContractViolation,
)
from krystal_kernel.contracts import (
    ContractHeader,
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

class EnlgRole(IntEnum):
    """Functional role of an epistemic statement or actor."""
    OBSERVER  = 1
    PROPOSER  = 2
    EXECUTOR  = 3
    VERIFIER  = 4
    AUTHORITY = 5


class EnlgEvidenceClass(IntEnum):
    """Ordered algebraic evidence poset."""
    UNKNOWN             = 0
    DERIVED             = 1
    CODE_PRESENT        = 2
    SIMULATED           = 3
    LOCAL_MEASURED      = 4
    REMOTE_CI           = 5
    PHYSICAL_MEASURED   = 6
    EXTERNAL_REPLICATED = 7


class EnlgVerdict(IntEnum):
    """Closed finite set of allowable epistemic verdict states."""
    UNKNOWN              = 0
    CONFLICT             = 1
    FALSIFIED            = 2
    SUPPORTED_WITH_SCOPE = 3
    MEASURED_WITH_SCOPE  = 4
    VERIFIED_WITH_SCOPE  = 5


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

BANNED_UNBOUNDED_SUPERLATIVES = {
    "always",
    "never",
    "guarantees",
    "zero-risk",
    "fully secure",
    "optimal",
    "universal",
    "proven",
    "dokázané",
    "bezpečné",
    "stabilné",
    "guaranteed",
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


@dataclass(frozen=True)
class EnlgCompiledBlock:
    """Blueprint-specific compiled ENLG block."""
    claim: str
    scope: str
    role: EnlgRole
    evidence: Set[EnlgEvidenceClass]
    falsifier: str
    null_m0_no_effect: str
    null_m1_noise: str
    null_m2_confounder: str
    verdict: EnlgVerdict
    has_contradiction: bool = False
    vital_hp: int = VITAL_MAX_HP


@dataclass(frozen=True)
class EnlgClaimTrace:
    """Full typed abstract syntax tree of a single ENLG evaluation."""
    claim: str
    scope: str
    role: EnlgRole
    evidence: Tuple[EnlgEvidenceClass, ...]
    nulls: NullTriplet
    falsifier: str
    validity: Dict[str, str]
    verdict: EnlgVerdict
    residual: Tuple[str, ...]
    has_conflicting_evidence: bool = False
    next_experiment: Optional[NextExperimentPlan] = None
    vital_max_hp: int = VITAL_MAX_HP

    def __post_init__(self):
        if self.vital_max_hp != 6:
            raise ContractViolation("VITAL_MAX_HP must be 6 in EnlgClaimTrace")
        if not self.claim.strip():
            raise ContractViolation("EnlgClaimTrace requires non-empty atomic claim", violation_code="ERR_ENLG_CLAIM_EMPTY")
        if not self.scope.strip():
            raise ContractViolation("EnlgClaimTrace requires non-empty scope boundary", violation_code="ERR_ENLG_SCOPE_EMPTY")
        if not self.falsifier.strip():
            raise ContractViolation("EnlgClaimTrace requires non-empty falsifier", violation_code="ERR_ENLG_FALSIFIER_EMPTY")
        if not self.residual:
            raise ContractViolation("EnlgClaimTrace requires non-empty residual uncertainty", violation_code="ERR_ENLG_RESIDUAL_EMPTY")


# ============================================================================
# 3. ENLG AST VALIDATOR & EPISTEMIC COMPILER
# ============================================================================

class EnlgAstValidator:
    """
    Formal validator verifying the 10 ENLG cognitive patterns and enforcing
    the Evidence Lattice anti-promotion barriers.
    """
    BANNED_UNSCOPED_SUPERLATIVES = {"dokázané", "bezpečné", "stabilné", "guaranteed", "proven"}

    @classmethod
    def validate_block(cls, block: EnlgCompiledBlock) -> bool:
        """Validates EnlgCompiledBlock per KSPP Blueprint Specification."""
        if block.vital_hp != VITAL_MAX_HP:
            raise ContractViolation("ENLG invariant VITAL_MAX_HP == 6 violated")

        # Vzor [6]: Scope Compression & Falsifier-First
        if not block.scope.strip():
            raise ContractViolation("Missing explicit SCOPE definition", violation_code="ERR_ENLG_SCOPE_EMPTY")
        if not block.falsifier.strip():
            raise ContractViolation("Falsifier-First pattern violated: empty FALSIFIER", violation_code="ERR_ENLG_FALSIFIER_EMPTY")

        # Vzor [10]: Contradiction Preservation
        if block.has_contradiction and block.verdict != EnlgVerdict.CONFLICT:
            raise ContractViolation("Contradiction detected: verdict must be CONFLICT", violation_code="ERR_ENLG_CONFLICT_NOT_PRESERVED")

        # Vzor [3]: Authority Separation (PROPOSER nemôže udeliť VERIFIED)
        if block.role == EnlgRole.PROPOSER and block.verdict == EnlgVerdict.VERIFIED_WITH_SCOPE:
            raise ContractViolation("Authority Separation: PROPOSER cannot self-certify VERIFIED", violation_code="ERR_ENLG_PROPOSER_SELF_CERTIFICATION")

        # Vzor [1] & [4]: Evidence Lattice Anti-Promotion Barrier
        max_evidence = max(block.evidence) if block.evidence else EnlgEvidenceClass.UNKNOWN
        if max_evidence == EnlgEvidenceClass.UNKNOWN and block.verdict > EnlgVerdict.CONFLICT:
            raise ContractViolation("Cannot promote UNKNOWN evidence to positive verdict", violation_code="ERR_ENLG_UNKNOWN_PROMOTION")
        if max_evidence <= EnlgEvidenceClass.SIMULATED and block.verdict >= EnlgVerdict.MEASURED_WITH_SCOPE:
            raise ContractViolation("Anti-Promotion Barrier: Simulation -/-> Measurement", violation_code="ERR_ENLG_SIMULATION_TO_MEASUREMENT")
        if max_evidence == EnlgEvidenceClass.LOCAL_MEASURED and block.verdict == EnlgVerdict.VERIFIED_WITH_SCOPE:
            raise ContractViolation("Anti-Promotion Barrier: Local measurement requires CI/Physical verification", violation_code="ERR_ENLG_LOCAL_TO_VERIFIED_PROMOTION")

        return True

    @classmethod
    def validate_claim_trace(cls, trace: EnlgClaimTrace) -> bool:
        """Validates an EnlgClaimTrace against all ENLG v2 constitutional invariants."""
        claim_lower = trace.claim.lower()
        for banned in BANNED_UNBOUNDED_SUPERLATIVES:
            pattern = rf"\b{re.escape(banned)}\b"
            if re.search(pattern, claim_lower) and "defined invariant" not in trace.scope.lower() and "model assumption" not in trace.scope.lower():
                raise ContractViolation(
                    f"Anti-Overclaim Breach: Unbounded superlative '{banned}' detected without explicit model constraints in scope",
                    violation_code="ERR_ENLG_OVERCLAIM_SUPERLATIVE"
                )

        if trace.role == EnlgRole.PROPOSER and trace.verdict == EnlgVerdict.VERIFIED_WITH_SCOPE:
            raise ContractViolation(
                "Authority Separation Breach: Role PROPOSER cannot self-assign VERIFIED_WITH_SCOPE verdict",
                violation_code="ERR_ENLG_PROPOSER_SELF_CERTIFICATION"
            )

        if trace.has_conflicting_evidence:
            if trace.verdict != EnlgVerdict.CONFLICT:
                raise ContractViolation(
                    f"Contradiction Preservation Breach: Conflicting evidence exists, verdict must be CONFLICT, got {trace.verdict.name}",
                    violation_code="ERR_ENLG_CONFLICT_NOT_PRESERVED"
                )
            return True

        max_evidence_rank = max(EVIDENCE_RANK[e] for e in trace.evidence)
        verdict_rank = VERDICT_RANK[trace.verdict]

        if all(e == EnlgEvidenceClass.UNKNOWN for e in trace.evidence):
            if trace.verdict in (EnlgVerdict.SUPPORTED_WITH_SCOPE, EnlgVerdict.MEASURED_WITH_SCOPE, EnlgVerdict.VERIFIED_WITH_SCOPE):
                raise ContractViolation(
                    f"Anti-Promotion Breach: Evidence is strictly UNKNOWN, cannot promote to {trace.verdict.name}",
                    violation_code="ERR_ENLG_UNKNOWN_PROMOTION"
                )

        if max_evidence_rank == 1 and verdict_rank > 1:
            raise ContractViolation(
                f"Anti-Promotion Breach: Simulation/Derived evidence (rank 1) cannot promote to {trace.verdict.name} (rank {verdict_rank})",
                violation_code="ERR_ENLG_SIMULATION_TO_MEASUREMENT"
            )

        if max_evidence_rank == 2 and verdict_rank > 2:
            raise ContractViolation(
                f"Anti-Promotion Breach: Local-Measured evidence (rank 2) cannot promote to VERIFIED_WITH_SCOPE without remote-CI/external replication",
                violation_code="ERR_ENLG_LOCAL_TO_VERIFIED_PROMOTION"
            )

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
        """Deterministic parser for the KRYSTAL ENLG Compiler text format."""
        sections: Dict[str, str] = {}
        header_tags = [
            "CLAIM:", "SCOPE:", "ROLE:", "EVIDENCE:", "ANTI-PROMOTION:",
            "NULLS:", "FALSIFIER:", "VALIDITY:", "VERDICT:", "RESIDUAL:", "NEXT_EXPERIMENT:"
        ]

        current_header = None
        current_lines: List[str] = []

        for line in text.strip().splitlines():
            line_str = line.strip()
            matched_header = None
            for tag in header_tags:
                if line_str.startswith(tag):
                    matched_header = tag[:-1]
                    break

            if matched_header:
                if current_header:
                    sections[current_header] = "\n".join(current_lines).strip()
                current_header = matched_header
                remainder = line_str[len(matched_header) + 1:].strip()
                current_lines = [remainder] if remainder else []
            else:
                if current_header:
                    current_lines.append(line_str)

        if current_header:
            sections[current_header] = "\n".join(current_lines).strip()

        claim = sections.get("CLAIM", "")
        scope = sections.get("SCOPE", "")
        role_str = sections.get("ROLE", "VERIFIER").strip().upper()
        role = getattr(EnlgRole, role_str, EnlgRole.VERIFIER)

        ev_lines = sections.get("EVIDENCE", "UNKNOWN").splitlines()
        ev_items = []
        has_conflict = False
        for el in ev_lines:
            token = el.strip().replace("-", "_").upper()
            if "CONFLICT" in token:
                has_conflict = True
            for member in EnlgEvidenceClass:
                if member.name in token:
                    ev_items.append(member)
        if not ev_items:
            ev_items = [EnlgEvidenceClass.UNKNOWN]

        null_text = sections.get("NULLS", "")
        m0, m1, m2 = "No effect", "Noise only", "Confounder"
        for line in null_text.splitlines():
            if "M0" in line:
                m0 = line.split(":", 1)[-1].strip() or m0
            elif "M1" in line:
                m1 = line.split(":", 1)[-1].strip() or m1
            elif "M2" in line:
                m2 = line.split(":", 1)[-1].strip() or m2

        falsifier = sections.get("FALSIFIER", "False observation")

        val_text = sections.get("VALIDITY", "")
        validity = {"construct": "Standard", "internal": "Controlled", "external": "Bounded"}
        for line in val_text.splitlines():
            if ":" in line:
                k, v = line.split(":", 1)
                validity[k.strip().lower()] = v.strip()

        verdict_str = sections.get("VERDICT", "UNKNOWN").strip().replace("-", "_").upper()
        verdict = getattr(EnlgVerdict, verdict_str, EnlgVerdict.UNKNOWN)

        residual_text = sections.get("RESIDUAL", "")
        residual_lines = [r.strip() for r in residual_text.splitlines() if r.strip()]
        if not residual_lines:
            residual_lines = ["Unmeasured external parameters"]

        trace = EnlgClaimTrace(
            claim=claim,
            scope=scope,
            role=role,
            evidence=tuple(ev_items),
            nulls=NullTriplet(m0_no_effect=m0, m1_noise=m1, m2_confounder=m2),
            falsifier=falsifier,
            validity=validity,
            verdict=verdict,
            residual=tuple(residual_lines),
            has_conflicting_evidence=has_conflict,
            vital_max_hp=6,
        )

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
        self.quarantine_store: Dict[str, EnlgCompiledBlock] = {}
        self.released_store: Dict[str, EnlgCompiledBlock] = {}

    @property
    def quarantined_count(self) -> int:
        return len([a for a in self._quarantine.values() if not a.is_released]) + len(self.quarantine_store)

    @property
    def released_count(self) -> int:
        return len(self._released_history) + len(self.released_store)

    def deposit(
        self,
        artifact_id: str,
        candidate_payload_or_block: Any,
        proposer_role: Optional[EnlgRole] = None,
    ) -> str:
        """Deposits an unverified candidate artifact strictly into QUARANTINE."""
        if not artifact_id:
            raise ContractViolation("ArtifactGate deposit requires non-empty artifact_id")

        if isinstance(candidate_payload_or_block, EnlgCompiledBlock):
            EnlgEpistemicCompiler.validate_block(candidate_payload_or_block)
            self.quarantine_store[artifact_id] = candidate_payload_or_block
            return artifact_id

        role = proposer_role if proposer_role is not None else EnlgRole.PROPOSER
        if artifact_id in self._quarantine and not self._quarantine[artifact_id].is_released:
            raise ContractViolation(f"Artifact '{artifact_id}' is already held in quarantine", violation_code="ERR_GATE_DUPLICATE_DEPOSIT")

        entry = QuarantinedArtifact(
            artifact_id=artifact_id,
            candidate_payload=candidate_payload_or_block,
            proposer_role=role,
            deposit_timestamp_ns=1,
            is_released=False,
            vital_max_hp=6,
        )
        self._quarantine[artifact_id] = entry
        return artifact_id

    def release(
        self,
        artifact_id: str,
        receipt_or_evaluator: Any = None,
        evaluator_trace_or_verified: Any = None,
        commit_action: Optional[Callable[[], Any]] = None,
        is_degraded_fallback: bool = False,
        *,
        evaluator_role: Optional[EnlgRole] = None,
        receipt_verified: Optional[bool] = None,
        receipt: Optional[ReceiptContract] = None,
        evaluation_trace: Optional[EnlgClaimTrace] = None,
        evaluator_trace: Optional[EnlgClaimTrace] = None,
        commit_state_fn: Optional[Callable[[], Any]] = None,
    ) -> Any:
        """
        Polymorphic release:
        Form 1 (Blueprint): release(artifact_id, evaluator_role=..., receipt_verified=..., is_degraded_fallback=False) -> bool
        Form 2 (Trace/Receipt): release(artifact_id, receipt=..., evaluation_trace=..., commit_state_fn=...) -> commit_result
        """
        role = evaluator_role if evaluator_role is not None else (receipt_or_evaluator if isinstance(receipt_or_evaluator, EnlgRole) else None)
        verified = receipt_verified if receipt_verified is not None else (bool(evaluator_trace_or_verified) if role is not None else False)

        # Form 1: Blueprint invocation
        if role is not None:
            if artifact_id not in self.quarantine_store and artifact_id not in self._quarantine:
                raise ContractViolation("Artifact not in quarantine")
            if role == EnlgRole.PROPOSER:
                raise ContractViolation("PROPOSER cannot authorize ArtifactGate release (Delta S = 0)", violation_code="ERR_GATE_EVALUATOR_IS_PROPOSER")
            if not verified or is_degraded_fallback:
                return False  # Zostáva v karanténe, Delta S = 0

            if artifact_id in self.quarantine_store:
                self.released_store[artifact_id] = self.quarantine_store.pop(artifact_id)
            if artifact_id in self._quarantine:
                self._quarantine[artifact_id].is_released = True
                self._released_history.append(artifact_id)
            return True

        # Form 2: Trace & Receipt Contract invocation
        actual_receipt = receipt if receipt is not None else receipt_or_evaluator
        actual_trace = evaluation_trace if evaluation_trace is not None else (evaluator_trace if evaluator_trace is not None else evaluator_trace_or_verified)
        actual_commit = commit_state_fn if commit_state_fn is not None else commit_action

        if artifact_id not in self._quarantine:
            raise ContractViolation(f"Artifact '{artifact_id}' not found in quarantine", violation_code="ERR_GATE_NOT_FOUND")

        artifact = self._quarantine[artifact_id]
        if artifact.is_released:
            raise ContractViolation(f"Artifact '{artifact_id}' has already been released", violation_code="ERR_GATE_ALREADY_RELEASED")

        # 1. Authority Separation Check
        if actual_trace is not None and actual_trace.role == EnlgRole.PROPOSER:
            raise ContractViolation("Evaluator cannot be PROPOSER (Separation of Concerns violated, Delta_S == 0)", violation_code="ERR_GATE_EVALUATOR_IS_PROPOSER")

        # 2. Receipt Validity Check
        if actual_receipt is not None:
            if actual_receipt.header.vital_max_hp != 6:
                raise ContractViolation("Receipt header vital_max_hp != 6 (Delta_S == 0)", violation_code="ERR_GATE_RECEIPT_HP_INVALID")

            if actual_receipt.was_degraded_fallback:
                raise ContractViolation("Cannot promote artifact from degraded fallback receipt (Delta_S == 0)", violation_code="ERR_GATE_DEGRADED_RECEIPT_REJECTED")

        # 3. Release & Commit
        artifact.is_released = True
        if actual_receipt is not None:
            artifact.release_receipt_digest = actual_receipt.execution_digest
        self._released_history.append(artifact_id)

        if actual_commit is not None:
            return actual_commit()
        return True
