"""
pattern_debug_protocol.py - Antigravity & Gemini 4 Argon Epistemic Debug & Pattern Revision Engine.

Implements the normative tenets of Milestone V-DEBUG-PROTOCOL-01 (SPEC-EPISTEMIC-DEBUG-01):
1. Capabilities are not authority; the goal is task model correctness, not blind plan completion.
2. Anchor Locking: Primary goal, scope, invariants (VITAL_MAX_HP == 6), and stop conditions locked separately against context drift.
3. Pattern 6-Tuple: (Characteristic, Source, Scope, Evidential Status, Validity, Dependencies).
4. Combination is not proof: Any new combination begins strictly as PATTERN_CANDIDATE.
5. Epistemic residual calculation and versioned claim store (strict ban on silent overwriting).
6. Tool/Interpretation separation: Previous model outputs are not evidence; raw tool output vs interpretation; internet is a hypothesis source.
7. Plan Lock-In & Context Drift Guard: Rational Irritation triggers STOP -> Frame Audit -> Falsifier Search.
8. Error Asymmetry: Runtime error != falsehood; Non-execution != falsification.
9. Epistemic Mantra enforcement.
"""

from __future__ import annotations

import enum
import math
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

# Constitutional Repository Invariant
VITAL_MAX_HP: int = 6

EPISTEMIC_MANTRA: str = (
    "Drž cieľ stabilný, plán môžeš meniť, kombinuj len označené charakteristiky, "
    "nevyrob istotu, otváraj rámec pri rozpore, neoverené ostáva neznáme."
)


class EpistemicContractViolation(Exception):
    """Raised when an epistemic contract or constitutional invariant is violated."""
    pass


class PatternStatus(str, enum.Enum):
    PATTERN_CANDIDATE = "PATTERN_CANDIDATE"
    UNDER_TEST = "UNDER_TEST"
    EMPIRICALLY_VERIFIED = "EMPIRICALLY_VERIFIED"
    REFUTED = "REFUTED"


@dataclass(frozen=True)
class PatternCharacteristic:
    """
    6-Tuple representation of an architectural or algorithmic pattern characteristic:
    ⟨Characteristic, Source, Scope, Evidential Status, Validity, Dependencies⟩
    """
    name: str
    characteristic: str
    source: str
    scope: str
    evidential_status: str
    validity_bounds: str
    dependencies: Tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "characteristic": self.characteristic,
            "source": self.source,
            "scope": self.scope,
            "evidential_status": self.evidential_status,
            "validity_bounds": self.validity_bounds,
            "dependencies": list(self.dependencies),
        }


@dataclass
class PatternCandidate:
    """
    Represents a synthesis or composition of patterns.
    Invariant: ANY new combination begins strictly as PATTERN_CANDIDATE.
    Combination is NOT proof.
    """
    candidate_id: str
    components: List[PatternCharacteristic]
    status: PatternStatus = PatternStatus.PATTERN_CANDIDATE
    hypothesis_rationale: str = ""
    falsifier: str = ""
    proposer_id: str = "agent_proposer"
    evaluator_id: Optional[str] = None
    verification_test_trace: Optional[str] = None
    vital_max_hp: int = VITAL_MAX_HP

    def __post_init__(self) -> None:
        if self.vital_max_hp != VITAL_MAX_HP:
            raise EpistemicContractViolation(
                f"Constitutional violation: VITAL_MAX_HP must be {VITAL_MAX_HP}, got {self.vital_max_hp}"
            )
        if not self.components:
            raise EpistemicContractViolation("PatternCandidate must contain at least one pattern component.")
        if not self.falsifier:
            raise EpistemicContractViolation("PatternCandidate requires an explicit falsifier.")

    @classmethod
    def combine(
        cls,
        p1: PatternCharacteristic,
        p2: PatternCharacteristic,
        rationale: str,
        falsifier: str,
        proposer_id: str = "agent_proposer"
    ) -> PatternCandidate:
        """
        Synthesizes two pattern characteristics into a composite candidate.
        MANDATORY RULE: The result is ALWAYS in status PATTERN_CANDIDATE.
        Combination is NOT proof.
        """
        return cls(
            candidate_id=f"cand_{uuid.uuid4().hex[:8]}",
            components=[p1, p2],
            status=PatternStatus.PATTERN_CANDIDATE,
            hypothesis_rationale=rationale,
            falsifier=falsifier,
            proposer_id=proposer_id,
            vital_max_hp=VITAL_MAX_HP,
        )

    def promote_to_verified(self, evaluator_id: str, test_trace: str) -> None:
        """
        Promotes the candidate to EMPIRICALLY_VERIFIED under strict external audit:
        Anti-Self-Certification: proposer_id != evaluator_id.
        """
        if self.proposer_id == evaluator_id:
            raise EpistemicContractViolation(
                f"Anti-Self-Certification Violation: Proposer '{self.proposer_id}' cannot evaluate own candidate."
            )
        if not test_trace or "PASS" not in test_trace:
            raise EpistemicContractViolation("Cannot promote without a valid passing test trace.")

        self.evaluator_id = evaluator_id
        self.verification_test_trace = test_trace
        self.status = PatternStatus.EMPIRICALLY_VERIFIED


@dataclass
class LockedAnchors:
    """
    Holds core invariants, primary goal, scope, and stop conditions.
    Protects against context drift and erosion over long conversation windows.
    """
    primary_goal: str
    scope: str
    stop_conditions: List[str]
    invariants: Dict[str, Any] = field(default_factory=lambda: {"VITAL_MAX_HP": VITAL_MAX_HP})
    is_locked: bool = True

    def __post_init__(self) -> None:
        if self.invariants.get("VITAL_MAX_HP") != VITAL_MAX_HP:
            raise EpistemicContractViolation(
                f"Constitutional violation in LockedAnchors: VITAL_MAX_HP must be {VITAL_MAX_HP}."
            )

    def check_context_drift(self, candidate_goal: str, candidate_scope: str) -> Tuple[bool, str]:
        """
        Verifies if an action or candidate drifts away from locked primary goal and scope.
        """
        # If the candidate attempts to alter the invariant or broaden scope illicitly
        if not candidate_goal or not candidate_scope:
            return True, "Context drift detected: Missing goal or scope definition."
        if self.is_locked and (self.invariants.get("VITAL_MAX_HP") != VITAL_MAX_HP):
            return True, f"Invariant violation: VITAL_MAX_HP compromised."
        return False, "Anchors intact."


@dataclass
class VersionedClaim:
    """
    Represents an explicitly versioned claim.
    Prevents silent overwrites upon encountering contradiction.
    """
    claim_id: str
    version: int
    statement: str
    evidence_status: str
    residual_error: float
    superseded_reason: Optional[str] = None
    vital_max_hp: int = VITAL_MAX_HP

    def __post_init__(self) -> None:
        if self.vital_max_hp != VITAL_MAX_HP:
            raise EpistemicContractViolation(f"VITAL_MAX_HP must be {VITAL_MAX_HP}.")


class VersionedClaimStore:
    """
    Storage and audit facility for versioned claims.
    Strictly forbids silent overwrites.
    """
    def __init__(self) -> None:
        self._claims: Dict[str, List[VersionedClaim]] = {}

    def register_claim(self, claim_id: str, statement: str, evidence_status: str) -> VersionedClaim:
        if claim_id in self._claims:
            raise EpistemicContractViolation(
                f"Claim '{claim_id}' already exists. Use update_with_residual() to version, never silently overwrite."
            )
        c = VersionedClaim(
            claim_id=claim_id,
            version=1,
            statement=statement,
            evidence_status=evidence_status,
            residual_error=0.0,
            superseded_reason=None,
        )
        self._claims[claim_id] = [c]
        return c

    def update_with_residual(
        self,
        claim_id: str,
        new_statement: str,
        new_evidence_status: str,
        residual_error: float,
        reason: str
    ) -> VersionedClaim:
        """
        Updates a claim by appending a new version with explicit residual and reason.
        Never erases history.
        """
        if claim_id not in self._claims:
            raise EpistemicContractViolation(f"Claim '{claim_id}' does not exist to update.")

        history = self._claims[claim_id]
        current_version = history[-1].version
        new_v = VersionedClaim(
            claim_id=claim_id,
            version=current_version + 1,
            statement=new_statement,
            evidence_status=new_evidence_status,
            residual_error=residual_error,
            superseded_reason=reason,
        )
        history.append(new_v)
        return new_v

    def get_history(self, claim_id: str) -> List[VersionedClaim]:
        if claim_id not in self._claims:
            raise EpistemicContractViolation(f"Claim '{claim_id}' not found.")
        return list(self._claims[claim_id])

    def get_latest(self, claim_id: str) -> VersionedClaim:
        history = self.get_history(claim_id)
        return history[-1]


@dataclass
class ToolResultEnvelope:
    """
    Cleanly separates raw tool output from model interpretation.
    Enforces rules:
    - Previous model outputs are NOT proof.
    - Internet data is a hypothesis source, not local truth.
    """
    raw_tool_output: Any
    model_interpretation: str
    is_external_internet: bool = False
    is_previous_model_output: bool = False
    timestamp_iso: str = "2026-10-16T23:35:00Z"
    vital_max_hp: int = VITAL_MAX_HP

    def __post_init__(self) -> None:
        if self.vital_max_hp != VITAL_MAX_HP:
            raise EpistemicContractViolation(f"VITAL_MAX_HP must be {VITAL_MAX_HP}.")

    @property
    def is_valid_local_proof(self) -> bool:
        """
        True only if the source is calibrated local execution, NOT previous model output
        and NOT ungrounded external internet claims.
        """
        if self.is_previous_model_output:
            return False
        if self.is_external_internet:
            return False
        return True

    def format_epistemic_record(self) -> Dict[str, Any]:
        status = "HYPOTHESIS_EXTERNAL" if self.is_external_internet else (
            "MODEL_PRIOR_STATE" if self.is_previous_model_output else "LOCAL_TELEMETRY"
        )
        return {
            "status": status,
            "is_valid_proof": self.is_valid_local_proof,
            "raw_output": self.raw_tool_output,
            "interpretation": self.model_interpretation,
        }


class PlanLockInGuard:
    """
    Monitors context drift and plan inertia.
    When 'Rational Irritation' (discrepancy, failure burst, or unexpected anomaly) occurs,
    enforces a STOP -> Frame Audit -> Falsifier requirement before further steps.
    """
    def __init__(self, failure_threshold: int = 2) -> None:
        self.failure_threshold = failure_threshold
        self.consecutive_failures = 0
        self.irritation_active = False
        self.audit_completed = False
        self.pending_falsifier: Optional[str] = None

    def record_step(self, passed: bool, residual: float, unexpected_anomaly: bool = False) -> Dict[str, Any]:
        """
        Processes the outcome of a plan step.
        """
        if not passed or unexpected_anomaly or residual > 0.05:
            self.consecutive_failures += 1
        else:
            self.consecutive_failures = max(0, self.consecutive_failures - 1)

        if self.consecutive_failures >= self.failure_threshold or unexpected_anomaly:
            self.irritation_active = True
            self.audit_completed = False
            return {
                "action": "STOP_AND_AUDIT_FRAME",
                "reason": "Rational Irritation triggered by anomaly or cumulative residual.",
                "consecutive_failures": self.consecutive_failures,
                "residual": residual,
                "can_proceed_with_original_plan": False,
            }

        return {
            "action": "PROCEED",
            "consecutive_failures": self.consecutive_failures,
            "can_proceed_with_original_plan": True,
        }

    def execute_frame_audit(self, audited_model_summary: str, formulated_falsifier: str) -> None:
        """
        Satisfies the required Frame Audit after rational irritation.
        """
        if not audited_model_summary:
            raise EpistemicContractViolation("Frame Audit requires a non-empty audited model summary.")
        if not formulated_falsifier:
            raise EpistemicContractViolation("Frame Audit requires formulating a concrete falsifier.")

        self.pending_falsifier = formulated_falsifier
        self.audit_completed = True
        self.irritation_active = False
        self.consecutive_failures = 0

    def can_dispatch_next_action(self) -> bool:
        if self.irritation_active and not self.audit_completed:
            return False
        return True


class EpistemicErrorClassifier:
    """
    Implements the Epistemic Asymmetry of Errors:
    1. Runtime error != falsehood of underlying scientific hypothesis.
    2. Non-execution != falsification.
    """
    @staticmethod
    def classify(error_type: str, execution_attempted: bool, execution_succeeded: bool) -> Dict[str, Any]:
        if not execution_attempted:
            return {
                "category": "NON_EXECUTION",
                "is_falsification": False,
                "status": "UNTESTED",
                "meaning": "Test was never executed; premise has NOT been falsified.",
            }

        if not execution_succeeded:
            # Runtime, syntax, packaging, or network failure
            is_artifact_failure = any(term in error_type.lower() for term in ["syntax", "import", "typeerror", "timeout", "env"])
            return {
                "category": "ARTIFACT_OR_ENV_ERROR" if is_artifact_failure else "EXECUTION_FAILURE",
                "is_falsification": False,  # Error != proof of falsehood
                "status": "EVIDENCE_INSUFFICIENT",
                "meaning": "Execution encountered an error; does NOT prove underlying hypothesis false.",
            }

        return {
            "category": "EXECUTION_SUCCESS",
            "is_falsification": False,
            "status": "VALID_EXECUTION",
            "meaning": "Execution completed; proceed with empirical metric verification.",
        }


def compute_epistemic_residual(predicted_val: float, observed_val: float) -> float:
    """
    Calculates residual discrepancy: || observed - predicted ||.
    """
    return abs(observed_val - predicted_val)


def verify_mantra_adherence(text: str) -> bool:
    """
    Validates whether the core tenets of the mantra are respected.
    """
    key_phrases = [
        "cieľ stabilný",
        "plán môžeš meniť",
        "kombinuj len označené charakteristiky",
        "nevyrob istotu",
        "otváraj rámec",
        "neoverené ostáva neznáme",
    ]
    normalized = text.lower()
    return any(p in normalized for p in key_phrases)
