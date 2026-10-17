"""
scientific_evidence_map.py - Scientific Evidence Protocol & Evidence Mapping Engine.

Implements the normative tenets of Milestone V-SCIENCE-EVIDENCE-01 (SPEC-SCIENCE-EVIDENCE-01):
1. Truth is never assigned to authority, popularity, citation count, or consensus.
2. 5-Tuple Claim Audit: Provenance, Methodology, Study Type, Funding, Conflicts of Interest (COI).
3. Conflict of Interest increases bias risk, but is NOT proof of manipulation.
4. Methodological validity verification: internal/external validity, sample size (N),
   representativeness, statistical power, publication bias, pre-registration vs HARKing,
   and independent replications.
5. Consensus = Prior, not proof.
6. Ban on averaging study discrepancies: generate ScientificConflictNode with missing confounders.
7. Competing models including M0 (NULL Hypothesis) with explicit weakening, supporting, and falsifying conditions.
8. Internet is discovery, not verdict: local telemetry strictly cannot be overwritten by external consensus.
9. Promotion to local knowledge requires author/independent audit (Anti-Self-Certification).
10. Strict distinctions: Review != Direct proof, Repetition/Echo != Independent replication, Unknown != Refuted.
11. Constitutional Invariant: VITAL_MAX_HP == 6.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

VITAL_MAX_HP: int = 6


class EpistemicContractViolation(Exception):
    """Raised when a scientific protocol invariant is violated."""
    pass


class StudyType(str, enum.Enum):
    RCT = "RCT"
    OBSERVATIONAL_COHORT = "OBSERVATIONAL_COHORT"
    META_ANALYSIS = "META_ANALYSIS"
    IN_SILICO_SIMULATION = "IN_SILICO_SIMULATION"
    FORMAL_PROOF = "FORMAL_PROOF"
    NARRATIVE_REVIEW = "NARRATIVE_REVIEW"


@dataclass
class StudyValidityMetrics:
    """
    Evaluates methodological rigor and biases of a scientific study.
    """
    internal_validity_score: float  # 0.0 to 1.0
    external_validity_score: float  # 0.0 to 1.0
    sample_size: int
    is_representative: bool
    is_preregistered: bool
    is_independent_replication: bool
    publication_bias_risk: float  # 0.0 (low) to 1.0 (high)
    statistical_power: float  # 0.0 to 1.0

    def evaluate_methodological_soundness(self) -> Dict[str, Any]:
        flags = []
        if not self.is_preregistered:
            flags.append("RISK_OF_HARKING_OR_P_HACKING")
        if self.sample_size < 30:
            flags.append("UNDERPOWERED_SMALL_SAMPLE")
        if not self.is_representative:
            flags.append("SELECTION_BIAS_EXTERNAL_VALIDITY_COMPROMISED")
        if self.publication_bias_risk > 0.6:
            flags.append("HIGH_PUBLICATION_BIAS_FILE_DRAWER_RISK")

        is_sound = (
            self.internal_validity_score >= 0.7
            and self.sample_size >= 30
            and self.statistical_power >= 0.8
        )
        return {
            "is_sound": is_sound,
            "flags": flags,
            "internal_validity": self.internal_validity_score,
            "statistical_power": self.statistical_power,
        }


@dataclass
class ScientificClaimAudit:
    """
    5-Tuple Scientific Claim Audit:
    ⟨Provenance, Methodology, Study Type, Funding, Conflicts of Interest⟩
    """
    claim_id: str
    statement: str
    provenance: str
    methodology: str
    study_type: StudyType
    funding: str
    conflicts_of_interest: List[str]
    validity: StudyValidityMetrics
    falsifier: str
    vital_max_hp: int = VITAL_MAX_HP

    def __post_init__(self) -> None:
        if self.vital_max_hp != VITAL_MAX_HP:
            raise EpistemicContractViolation(
                f"Constitutional violation in ScientificClaimAudit: VITAL_MAX_HP must be {VITAL_MAX_HP}."
            )
        if not self.falsifier:
            raise EpistemicContractViolation("Every scientific claim requires a concrete falsifier.")

    def evaluate_coi_risk(self) -> Dict[str, Any]:
        """
        Normative rule: COI increases bias risk, but is NOT proof of manipulation.
        """
        has_coi = len(self.conflicts_of_interest) > 0
        return {
            "has_coi": has_coi,
            "coi_list": list(self.conflicts_of_interest),
            "bias_risk": "ELEVATED" if has_coi else "BASELINE",
            "is_proof_of_manipulation": False,  # Strict prohibition: COI != proof of fraud
            "recommendation": "Require independent replication and micro-data audit" if has_coi else "Standard audit",
        }

    @property
    def is_direct_proof(self) -> bool:
        """
        Normative rule: Review is NOT direct proof.
        """
        if self.study_type == StudyType.NARRATIVE_REVIEW:
            return False
        return self.validity.internal_validity_score >= 0.75


@dataclass
class CompetingModel:
    """
    Competing hypothesis/model in the Evidence Map.
    Must define what weakens it, what supports it, and what falsifies it.
    """
    model_id: str
    name: str
    mechanism: str
    supporting_evidence: List[str] = field(default_factory=list)
    weakening_evidence: List[str] = field(default_factory=list)
    falsifier: str = ""
    is_null_model: bool = False

    def __post_init__(self) -> None:
        if not self.falsifier:
            raise EpistemicContractViolation(f"Model '{self.model_id}' must define a decisive falsifier.")


@dataclass
class ScientificConflictNode:
    """
    Node preserving study discrepancies without synthetic averaging.
    Identifies missing confounders and telemetry gaps.
    """
    conflict_id: str
    topic: str
    study_a_id: str
    study_a_finding: float
    study_b_id: str
    study_b_finding: float
    missing_confounders: List[str]
    is_averaged: bool = False
    vital_max_hp: int = VITAL_MAX_HP

    def __post_init__(self) -> None:
        if self.vital_max_hp != VITAL_MAX_HP:
            raise EpistemicContractViolation(f"VITAL_MAX_HP must be {VITAL_MAX_HP}.")
        if self.is_averaged:
            raise EpistemicContractViolation(
                "Epistemic violation: Averaging conflicting scientific studies is strictly forbidden."
            )

    def attempt_average(self) -> float:
        """
        Strict violation trigger: attempting to average study findings raises an error.
        """
        raise EpistemicContractViolation(
            "Forbidden action: Cannot average conflicting studies. Retain conflict and isolate missing confounders."
        )


class EvidenceMap:
    """
    Evidence Map orchestrator:
    - Maintains competing models including M0 NULL.
    - Tracks conflicting studies without averaging.
    - Prevents external consensus from overwriting local telemetry.
    - Enforces Anti-Self-Certification for promotion.
    """
    def __init__(self, topic: str) -> None:
        self.topic = topic
        self.claims: Dict[str, ScientificClaimAudit] = {}
        self.models: Dict[str, CompetingModel] = {}
        self.conflicts: List[ScientificConflictNode] = []
        self.local_approved_knowledge: Set[str] = set()

        # Enforce presence of M0 NULL model
        self.models["M0_NULL"] = CompetingModel(
            model_id="M0_NULL",
            name="Null Hypothesis (Noise / Random / Inefficacy)",
            mechanism="Observed effect is an artifact of measurement noise, random drift, or sampling error.",
            falsifier="Statistically significant effect replicated independently across p < 0.001 with N >= 1000.",
            is_null_model=True,
        )

    def register_claim(self, claim: ScientificClaimAudit) -> None:
        self.claims[claim.claim_id] = claim

    def add_competing_model(self, model: CompetingModel) -> None:
        self.models[model.model_id] = model

    def register_conflict(self, conflict: ScientificConflictNode) -> None:
        self.conflicts.append(conflict)

    def verify_m0_null_present(self) -> bool:
        return "M0_NULL" in self.models and self.models["M0_NULL"].is_null_model

    def resolve_external_consensus_vs_local_telemetry(
        self,
        local_measurement: float,
        external_consensus_claim: float,
        external_citation_count: int,
    ) -> Dict[str, Any]:
        """
        Normative rule: Internet is discovery, not verdict.
        Local audited telemetry is NOT overwritten by external consensus.
        Citation count does not dictate truth.
        """
        delta = abs(local_measurement - external_consensus_claim)
        return {
            "retained_ground_truth": local_measurement,
            "external_consensus": external_consensus_claim,
            "citation_count": external_citation_count,
            "consensus_treated_as": "BAYESIAN_PRIOR_FOR_INVESTIGATION",
            "local_overwritten": False,  # Strictly forbidden
            "discrepancy_delta": delta,
            "status": "LOCAL_EMPIRICAL_HOLDS_SUPREMACY",
        }

    def promote_to_local_knowledge(
        self,
        claim_id: str,
        proposer_id: str,
        auditor_id: str,
        test_trace: str,
    ) -> None:
        """
        Anti-Self-Certification: proposer_id != auditor_id.
        """
        if claim_id not in self.claims:
            raise EpistemicContractViolation(f"Claim '{claim_id}' not found in EvidenceMap.")
        if proposer_id == auditor_id:
            raise EpistemicContractViolation(
                f"Anti-Self-Certification violation: Proposer '{proposer_id}' cannot audit own claim."
            )
        if "PASS" not in test_trace:
            raise EpistemicContractViolation("Test trace must contain a passing verification.")

        claim = self.claims[claim_id]
        if not claim.is_direct_proof:
            raise EpistemicContractViolation("Cannot promote a narrative review or flawed study to verified knowledge.")

        self.local_approved_knowledge.add(claim_id)


def evaluate_replication_vs_repetition(
    primary_study_id: str,
    citing_studies: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Distinguishes genuine independent replications from mere citations/echoes.
    """
    independent_replications = 0
    mere_echoes = 0

    for s in citing_studies:
        has_new_data = s.get("collected_new_data", False)
        independent_authors = s.get("independent_authors", False)
        if has_new_data and independent_authors:
            independent_replications += 1
        else:
            mere_echoes += 1

    return {
        "primary_study_id": primary_study_id,
        "independent_replications": independent_replications,
        "citation_echoes": mere_echoes,
        "verdict": "REPLICATED" if independent_replications >= 2 else "UNREPLICATED_ECHO",
    }
