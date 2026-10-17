"""
Deep Research Epistemic Planner & Hypothesis Portfolio Protocol
===============================================================
Milestone: V-DEEP-RESEARCH-01
Constitutional Invariant: VITAL_MAX_HP == 6

Core Invariants:
  1. Priority dictates what to investigate, not what to believe.
  2. Deep Research originates from ResearchPlan:
     Local empirical evidence strictly precedes external lookups.
     External sources admitted ONLY if Expected Value of Information (EVOI) > Cost.
     Local ground truth is never overwritten by external consensus.
  3. Hypotheses Portfolio Prioritization:
     Priority = (GoalRelevance * ExpectedInfoGain * FalsificationValue) / Cost.
     Prohibited: Prioritizing by prior probability or popular consensus.
  4. Side Findings Stash:
     Incidental findings are queued/stashed to avoid cognitive tunneling and scope drift.
  5. Premise Refutation Escape Hatch:
     Counter-evidence challenging a foundational premise has an immediate escape hatch,
     escalating to Reframing or Validator Audit, spawning a NEW_BRANCH.
  6. Output Triad:
     Every research deliverable is strictly RESULT, UNKNOWN, or NEW_BRANCH with an unalterable evidence trace.
"""

from __future__ import annotations

import enum
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from krystal_kernel.contracts import ContractViolation
from krystal_kernel.scene_constructor_presets import VITAL_MAX_HP
from krystal_kernel.epistemic_provenance_logger import (
    EpistemicVerdict,
    EpistemicEvidenceClass,
    EpistemicProvenanceLedger,
)


class EvidenceScope(enum.Enum):
    """Scope of evidence origin."""
    LOCAL_WORKSPACE = "LOCAL_WORKSPACE"      # Local system telemetry, hardware probes, verified test traces
    EXTERNAL_CONSENSUS = "EXTERNAL_CONSENSUS" # External papers, benchmark corpus, web articles


class EpistemicOutputKind(enum.Enum):
    """Output Triad Contract: Every research deliverable must match one of these 3 states."""
    RESULT = "RESULT"            # Verified empirical fact with local evidence trace
    UNKNOWN = "UNKNOWN"          # Formal Unknown state with documented missing evidence fields
    NEW_BRANCH = "NEW_BRANCH"    # Escalated new research branch triggered by premise refutation


@dataclass
class ResearchHypothesis:
    """
    Hypothesis specification within a deep research portfolio.
    Constitutional Invariant: vital_max_hp == 6.
    """
    hyp_id: str
    premise: str
    mechanism: str
    goal_relevance: float         # [0.0, 1.0]
    expected_info_gain: float     # [0.0, 1.0]
    falsification_value: float    # [0.0, 1.0]
    cost_ms: float                # Strictly positive
    scope: EvidenceScope = EvidenceScope.LOCAL_WORKSPACE
    prior_probability: float = 0.5  # Tracked for Bayesian updates, NEVER used for priority ranking!
    popularity_index: float = 0.0   # Tracked for social bias detection, NEVER used for priority!
    status: str = "PROPOSED"
    vital_max_hp: int = 6

    def __post_init__(self) -> None:
        if self.vital_max_hp != VITAL_MAX_HP:
            raise ContractViolation(f"VITAL_MAX_HP must be 6 in ResearchHypothesis, got {self.vital_max_hp}")
        if self.cost_ms <= 0.0:
            raise ContractViolation("cost_ms must be strictly positive")
        if not (0.0 <= self.goal_relevance <= 1.0):
            raise ContractViolation("goal_relevance must be in [0.0, 1.0]")
        if not (0.0 <= self.expected_info_gain <= 1.0):
            raise ContractViolation("expected_info_gain must be in [0.0, 1.0]")
        if not (0.0 <= self.falsification_value <= 1.0):
            raise ContractViolation("falsification_value must be in [0.0, 1.0]")

    def priority_score(self) -> float:
        """
        Priority = (GoalRelevance * ExpectedInfoGain * FalsificationValue) / Cost
        Governs WHAT TO INVESTIGATE, not what to believe.
        """
        numerator = self.goal_relevance * self.expected_info_gain * self.falsification_value
        denom = max(self.cost_ms, 1e-6)
        return float(numerator / denom)


@dataclass
class SideFinding:
    """
    Prompt / Research park for side discoveries.
    Prevents scope drift and cognitive tunneling without discarding unexpected findings.
    Constitutional Invariant: vital_max_hp == 6.
    """
    finding_id: str
    description: str
    source_context: str
    discovered_at: float = field(default_factory=time.time)
    stashed: bool = True
    vital_max_hp: int = 6

    def __post_init__(self) -> None:
        if self.vital_max_hp != VITAL_MAX_HP:
            raise ContractViolation(f"VITAL_MAX_HP must be 6 in SideFinding")


@dataclass
class PremiseRefutationReceipt:
    """
    Receipt generated when counter-evidence successfully triggers the Premise Escape Hatch.
    Constitutional Invariant: vital_max_hp == 6.
    """
    receipt_id: str
    refuted_premise: str
    counter_evidence_summary: str
    measured_anomaly: float
    escalated_action: str  # REFRAME_PROBLEM or AUDIT_VALIDATOR
    new_branch_id: str
    timestamp: float = field(default_factory=time.time)
    vital_max_hp: int = 6

    def __post_init__(self) -> None:
        if self.vital_max_hp != VITAL_MAX_HP:
            raise ContractViolation(f"VITAL_MAX_HP must be 6 in PremiseRefutationReceipt")


@dataclass
class EpistemicOutputTriad:
    """
    Structured deliverable conforming to the Output Triad: RESULT, UNKNOWN, or NEW_BRANCH.
    Constitutional Invariant: vital_max_hp == 6.
    """
    output_id: str
    kind: EpistemicOutputKind
    title: str
    evidence_trace: List[str]
    missing_evidence: List[str] = field(default_factory=list)
    escalation_receipt: Optional[PremiseRefutationReceipt] = None
    vital_max_hp: int = 6

    def __post_init__(self) -> None:
        if self.vital_max_hp != VITAL_MAX_HP:
            raise ContractViolation(f"VITAL_MAX_HP must be 6 in EpistemicOutputTriad")
        if not self.evidence_trace:
            raise ContractViolation("EpistemicOutputTriad must contain an evidence_trace")


class DeepResearchPlan:
    """
    Orchestrates deep research across competing hypotheses.
    Enforces local-evidence-first, prioritization without popularity bias,
    side-findings stashing, and the premise refutation escape hatch.
    """

    def __init__(
        self,
        plan_id: str,
        goal_description: str,
        ledger: Optional[EpistemicProvenanceLedger] = None,
        vital_max_hp: int = 6,
    ) -> None:
        if vital_max_hp != VITAL_MAX_HP:
            raise ContractViolation(f"VITAL_MAX_HP must be 6 in DeepResearchPlan")
        self.plan_id = plan_id
        self.goal_description = goal_description
        self.vital_max_hp = vital_max_hp
        self.ledger = ledger or EpistemicProvenanceLedger()

        self.portfolio: Dict[str, ResearchHypothesis] = {}
        self.side_findings: List[SideFinding] = []
        self.refutation_escalations: List[PremiseRefutationReceipt] = []

    def register_hypothesis(self, hyp: ResearchHypothesis) -> None:
        """Adds a hypothesis to the research portfolio."""
        if hyp.vital_max_hp != self.vital_max_hp:
            raise ContractViolation("VITAL_MAX_HP invariant breach in register_hypothesis")
        self.portfolio[hyp.hyp_id] = hyp

    def rank_portfolio(self) -> List[Tuple[ResearchHypothesis, float]]:
        """
        Ranks hypotheses strictly by priority_score() = (Relevance * InfoGain * Falsification) / Cost.
        Strictly prohibits sorting by prior_probability or popularity_index.
        """
        ranked = [
            (h, h.priority_score())
            for h in self.portfolio.values()
            if h.status in ("PROPOSED", "INVESTIGATING")
        ]
        ranked.sort(key=lambda item: item[1], reverse=True)
        return ranked

    def check_scope_admissibility(
        self,
        scope: EvidenceScope,
        local_evidence_available: bool,
        expected_evoi: float,
        external_investigation_cost: float,
    ) -> Tuple[bool, str]:
        """
        Enforces evidence hierarchy:
        Local evidence strictly precedes external evidence.
        External lookups are blocked if EVOI <= external_investigation_cost or if local evidence is unexamined.
        """
        if scope == EvidenceScope.LOCAL_WORKSPACE:
            return True, "Local workspace evidence is unconditionally admissible as primary ground truth."

        # External consensus scope
        if not local_evidence_available:
            return False, "Local evidence has not been examined; local evidence strictly precedes external lookups."

        if expected_evoi <= external_investigation_cost:
            return False, f"External lookup rejected: EVOI ({expected_evoi:.2f}) does not exceed Cost ({external_investigation_cost:.2f})."

        return True, "External lookup admissible: EVOI exceeds investigation overhead."

    def stash_side_finding(self, description: str, source_context: str) -> SideFinding:
        """
        Stashes unexpected or orthogonal observations to avoid cognitive tunneling
        while preserving information for future research plans.
        """
        fid = f"SIDE-{len(self.side_findings) + 1:04d}"
        finding = SideFinding(
            finding_id=fid,
            description=description,
            source_context=source_context,
            vital_max_hp=6,
        )
        self.side_findings.append(finding)
        return finding

    def evaluate_premise_escape_hatch(
        self,
        hyp_id: str,
        counter_evidence_metric: float,
        refutation_threshold: float,
        symptom_description: str,
        escalate_to_validator_audit: bool = False,
        refutes_when_less_than: bool = False,
    ) -> Optional[PremiseRefutationReceipt]:
        """
        Escape Hatch for Premise Refutation:
        If counter-evidence exceeds the refutation threshold (or falls below if refutes_when_less_than),
        the premise is falsified. Rather than ignoring or burying the counter-evidence, it immediately
        triggers an escalation to Reframing or Validator Audit and creates a NEW_BRANCH.
        """
        hyp = self.portfolio.get(hyp_id)
        if not hyp:
            raise ContractViolation(f"Hypothesis '{hyp_id}' not found in portfolio")

        # Refutation check
        is_refuted = (
            counter_evidence_metric <= refutation_threshold
            if refutes_when_less_than
            else counter_evidence_metric >= refutation_threshold
        )
        if is_refuted:
            hyp.status = "PREMISE_REFUTED"
            receipt_id = f"REFUT-{hyp_id}-{int(time.time())}"
            new_branch_id = f"BRANCH-REFRAME-{hyp_id}"
            action = "AUDIT_VALIDATOR" if escalate_to_validator_audit else "REFRAME_PROBLEM"

            receipt = PremiseRefutationReceipt(
                receipt_id=receipt_id,
                refuted_premise=hyp.premise,
                counter_evidence_summary=symptom_description,
                measured_anomaly=counter_evidence_metric,
                escalated_action=action,
                new_branch_id=new_branch_id,
                vital_max_hp=6,
            )
            self.refutation_escalations.append(receipt)
            return receipt

        return None

    def construct_output(
        self,
        output_kind: EpistemicOutputKind,
        title: str,
        evidence_trace: List[str],
        missing_evidence: Optional[List[str]] = None,
        receipt: Optional[PremiseRefutationReceipt] = None,
    ) -> EpistemicOutputTriad:
        """
        Produces a validated output strictly conforming to the Output Triad
        (RESULT, UNKNOWN, or NEW_BRANCH) with an indelible evidence trace.
        """
        oid = f"OUT-{self.plan_id}-{output_kind.value}-{int(time.time())}"
        return EpistemicOutputTriad(
            output_id=oid,
            kind=output_kind,
            title=title,
            evidence_trace=evidence_trace,
            missing_evidence=missing_evidence or [],
            escalation_receipt=receipt,
            vital_max_hp=6,
        )
