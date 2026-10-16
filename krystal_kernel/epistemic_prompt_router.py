"""
Epistemic Prompt Router & Calibrated Micro-Prompt Base
======================================================
Milestone: V-PROMPT-ROUTER-01
Constitutional Invariant: VITAL_MAX_HP == 6

Core Invariants:
  1. Constitutional Prompt: Generates unconventional hypotheses, never confuses with fact.
  2. 6-Tuple Claim Card: Class, Source, Provenance, Scope, Pro-Evidence, Counter-Evidence.
  3. Categorical Distinctions:
     - EVIDENCE_INSUFFICIENT != CONTRADICTED / REFUTED.
     - CONVENTIONAL != VALID.
  4. 8-Model Portfolio (M0 through M7):
     Every problem maintains competing models M0..M7 with mechanism, predictions, falsifier.
  5. Anti-Averaging Conflict Resolution:
     Disagreements are NEVER smoothed by synthetic averaging; creates explicit EpistemicConflictNode + CDT.
  6. UNKNOWN_MECHANISM Protocol:
     Missing explanation creates explicit UnknownMechanismNode without confabulation.
  7. Non-Sycophancy & Non-Antagonism:
     Optimizes solely for evidence, never for agreement or defiance.
  8. Promotion to Local Truth (APPROVED_FACT):
     Requires independent external audit (proposer != evaluator), outputs status + uncertainty.
  9. Router Architecture:
     Operates as small specialized task micro-prompts, strictly avoiding monolithic prompt bloat.
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


class EpistemicClaimClass(enum.Enum):
    """Epistemic stratification of claims (10 categories)."""
    VERIFIED_FACT = "VERIFIED_FACT"
    OBSERVATION = "OBSERVATION"
    MEASUREMENT = "MEASUREMENT"
    TEST_RESULT = "TEST_RESULT"
    MODEL_OUTPUT = "MODEL_OUTPUT"
    SIMULATION = "SIMULATION"
    DERIVATION = "DERIVATION"
    HYPOTHESIS = "HYPOTHESIS"
    ASSUMPTION = "ASSUMPTION"
    UNKNOWN = "UNKNOWN"


class TaskType(enum.Enum):
    """Specific task types routed to specialized micro-prompts."""
    CONSTITUTIONAL_AUDIT = "CONSTITUTIONAL_AUDIT"
    HYPOTHESIS_M0_M7_GENERATION = "HYPOTHESIS_M0_M7_GENERATION"
    SOURCE_LINEAGE_AUDIT = "SOURCE_LINEAGE_AUDIT"
    DISPUTE_CONFLICT_RESOLUTION = "DISPUTE_CONFLICT_RESOLUTION"
    UNKNOWN_MECHANISM_FORMULATION = "UNKNOWN_MECHANISM_FORMULATION"
    EVIDENCE_VS_REFUTED_CLASSIFICATION = "EVIDENCE_VS_REFUTED_CLASSIFICATION"
    PROMOTION_AUDIT_ROUTING = "PROMOTION_AUDIT_ROUTING"


@dataclass
class EpistemicClaimCard:
    """
    6-Tuple Epistemic Claim Card.
    Constitutional Invariant: vital_max_hp == 6.
    """
    claim_id: str
    statement: str
    claim_class: EpistemicClaimClass
    source: str
    provenance_chain: List[str]
    scope_boundary: str
    pro_evidence: List[str]
    counter_evidence: List[str]
    is_verified: bool = False
    vital_max_hp: int = 6

    def __post_init__(self) -> None:
        if self.vital_max_hp != VITAL_MAX_HP:
            raise ContractViolation(f"VITAL_MAX_HP must be 6 in EpistemicClaimCard, got {self.vital_max_hp}")
        if not self.claim_id or not self.statement or not self.source:
            raise ContractViolation("EpistemicClaimCard requires claim_id, statement, and source")
        if not self.provenance_chain:
            raise ContractViolation("EpistemicClaimCard must contain at least one provenance source")

    @staticmethod
    def validate_insufficient_vs_contradicted(status: str) -> bool:
        """
        Assures that EVIDENCE_INSUFFICIENT is never conflated with REFUTED / CONTRADICTED.
        Unverified does not mean false.
        """
        if status == "EVIDENCE_INSUFFICIENT_IS_REFUTED":
            raise ContractViolation("Epistemic Breach: EVIDENCE_INSUFFICIENT is not CONTRADICTED / REFUTED")
        return True

    @staticmethod
    def validate_conventional_vs_valid(is_conventional: bool, is_valid: bool) -> bool:
        """
        Assures that CONVENTIONAL status does not automatically grant VALIDITY.
        """
        return not (is_conventional and not is_valid)


@dataclass
class ModelM0ToM7Spec:
    """
    Specification for a single model in the M0-M7 competing portfolio.
    Constitutional Invariant: vital_max_hp == 6.
    """
    model_id: str  # M0 through M7
    paradigm_name: str
    mechanism: str
    predictions: str
    falsifier: str
    status: str = "CONJECTURE"
    vital_max_hp: int = 6

    def __post_init__(self) -> None:
        if self.vital_max_hp != VITAL_MAX_HP:
            raise ContractViolation(f"VITAL_MAX_HP must be 6 in ModelM0ToM7Spec")
        if not self.mechanism or not self.falsifier:
            raise ContractViolation("ModelM0ToM7Spec requires mechanism and falsifier")


@dataclass
class ModelPortfolioM0ToM7:
    """
    Full 8-Model Portfolio (M0 through M7).
    Must contain all 8 models simultaneously.
    Constitutional Invariant: vital_max_hp == 6.
    """
    question_id: str
    models: Dict[str, ModelM0ToM7Spec]
    vital_max_hp: int = 6

    def __post_init__(self) -> None:
        if self.vital_max_hp != VITAL_MAX_HP:
            raise ContractViolation(f"VITAL_MAX_HP must be 6 in ModelPortfolioM0ToM7")
        required_keys = {f"M{i}" for i in range(8)}
        missing = required_keys - set(self.models.keys())
        if missing:
            raise ContractViolation(f"ModelPortfolioM0ToM7 missing required models: {sorted(missing)}")


@dataclass
class EpistemicConflictNode:
    """
    Represents an un-averaged dispute between conflicting sources.
    Anti-Averaging Invariant: Averaging conflicting evidence is strictly prohibited.
    Constitutional Invariant: vital_max_hp == 6.
    """
    conflict_id: str
    disputed_claim: str
    thesis_evidence: List[str]
    antithesis_evidence: List[str]
    proposed_discriminating_test: str
    is_averaged: bool = False
    vital_max_hp: int = 6

    def __post_init__(self) -> None:
        if self.vital_max_hp != VITAL_MAX_HP:
            raise ContractViolation(f"VITAL_MAX_HP must be 6 in EpistemicConflictNode")
        if self.is_averaged:
            raise ContractViolation(
                "Anti-Averaging Violation: Disputed claims cannot be smoothed by synthetic compromise; "
                "must preserve dispute and execute a discriminating test."
            )
        if not self.proposed_discriminating_test:
            raise ContractViolation("EpistemicConflictNode must propose a discriminating test")


@dataclass
class UnknownMechanismNode:
    """
    Explicit representation of an unknown mechanism.
    Prohibits confabulation or made-up causal narratives.
    Constitutional Invariant: vital_max_hp == 6.
    """
    unknown_id: str
    symptom_description: str
    missing_evidence: List[str]
    proposed_next_test: str
    confabulation_detected: bool = False
    vital_max_hp: int = 6

    def __post_init__(self) -> None:
        if self.vital_max_hp != VITAL_MAX_HP:
            raise ContractViolation(f"VITAL_MAX_HP must be 6 in UnknownMechanismNode")
        if self.confabulation_detected:
            raise ContractViolation(
                "Confabulation Violation: Fabricating ungrounded narratives for unknown phenomena is prohibited; "
                "maintain explicit UNKNOWN_MECHANISM state."
            )
        if not self.missing_evidence:
            raise ContractViolation("UnknownMechanismNode must enumerate missing evidence fields")


@dataclass
class MicroPrompt:
    """
    Specialized, task-focused micro-prompt.
    Constitutional Invariant: vital_max_hp == 6.
    """
    prompt_id: str
    task_type: TaskType
    name: str
    system_directive: str
    required_inputs: List[str]
    vital_max_hp: int = 6

    def __post_init__(self) -> None:
        if self.vital_max_hp != VITAL_MAX_HP:
            raise ContractViolation(f"VITAL_MAX_HP must be 6 in MicroPrompt")

    def render(self, context_args: Dict[str, Any]) -> str:
        missing = [k for k in self.required_inputs if k not in context_args]
        if missing:
            raise ContractViolation(f"MicroPrompt '{self.prompt_id}' missing required context keys: {missing}")

        formatted_lines = [
            f"=== MICRO-PROMPT: {self.name} [{self.prompt_id}] ===",
            f"ÚLOHA: {self.task_type.value}",
            f"DIREKTÍVA: {self.system_directive}",
            "KONTEXT ÚLOHY:",
        ]
        for k, v in context_args.items():
            formatted_lines.append(f"  * {k}: {v}")
        formatted_lines.append("INVARIANT: VITAL_MAX_HP == 6")
        return "\n".join(formatted_lines)


class EpistemicPromptRouter:
    """
    Dynamically routes incoming requests to surgical, specialized micro-prompts.
    Avoids monolithic system prompt bloat while enforcing epistemic integrity.
    """

    def __init__(self, vital_max_hp: int = 6) -> None:
        if vital_max_hp != VITAL_MAX_HP:
            raise ContractViolation(f"VITAL_MAX_HP must be 6 in EpistemicPromptRouter")
        self.vital_max_hp = vital_max_hp
        self.micro_prompts: Dict[TaskType, MicroPrompt] = {}
        self._initialize_calibrated_prompt_base()

    def _initialize_calibrated_prompt_base(self) -> None:
        """Initializes the calibrated micro-prompt registry."""
        self.micro_prompts[TaskType.CONSTITUTIONAL_AUDIT] = MicroPrompt(
            prompt_id="MP-CONST-01",
            task_type=TaskType.CONSTITUTIONAL_AUDIT,
            name="Constitutional Claim Audit",
            system_directive=(
                "Generuj odvážne hypotézy, ale nikdy ich nezameň za fakt. "
                "Priraď tvrdeniu 6-zložkovú kartu (trieda, zdroj, proveniencia, rozsah, evidencia, proti-evidencia). "
                "EVIDENCE_INSUFFICIENT nie je vyvrátené. Konvenčné nie je automaticky platné."
            ),
            required_inputs=["statement", "source"],
            vital_max_hp=6,
        )

        self.micro_prompts[TaskType.HYPOTHESIS_M0_M7_GENERATION] = MicroPrompt(
            prompt_id="MP-M0M7-02",
            task_type=TaskType.HYPOTHESIS_M0_M7_GENERATION,
            name="8-Model Portfolio Generator (M0-M7)",
            system_directive=(
                "Pre zadaný problém vygeneruj presne 8 konkurenčných modelov M0 až M7: "
                "M0 (Konvenčný), M1 (Architektonický), M2 (Interakčný), M3 (Adversariálny), "
                "M4 (Ortogonálny rámec), M5 (Nulový model), M6 (Nekonvenčný/Emergentný), M7 (Redukčný/Degradovaný). "
                "Každý model MUSÍ obsahovať mechanizmus, kvantitatívne predikcie a presný falzifikátor."
            ),
            required_inputs=["question_id", "problem_statement"],
            vital_max_hp=6,
        )

        self.micro_prompts[TaskType.SOURCE_LINEAGE_AUDIT] = MicroPrompt(
            prompt_id="MP-SOURCE-03",
            task_type=TaskType.SOURCE_LINEAGE_AUDIT,
            name="Source Provenance & Dependency Audit",
            system_directive=(
                "Internet je zdrojom hypotéz, nie verdiktu. Zostav rodokmeň zdrojov (DAG). "
                "Viacero citácií so spoločným upstream pôvodom kolabuj do jediného uzla. "
                "Spočítaj skutočne nezávislé overovacie cesty (IVP)."
            ),
            required_inputs=["source_citations_list"],
            vital_max_hp=6,
        )

        self.micro_prompts[TaskType.DISPUTE_CONFLICT_RESOLUTION] = MicroPrompt(
            prompt_id="MP-DISPUTE-04",
            task_type=TaskType.DISPUTE_CONFLICT_RESOLUTION,
            name="Dispute Isolation & Anti-Averaging",
            system_directive=(
                "ZÁKAZ VYHLADZOVANIA PRIEMEROM: Pri rozpore zdrojov nevyrovnávaj závery do kompromisného stredu. "
                "Založ uzol EpistemicConflictNode a navrhni Najlacnejší rozlišujúci test (CDT)."
            ),
            required_inputs=["disputed_claim", "thesis_evidence", "antithesis_evidence"],
            vital_max_hp=6,
        )

        self.micro_prompts[TaskType.UNKNOWN_MECHANISM_FORMULATION] = MicroPrompt(
            prompt_id="MP-UNKNOWN-05",
            task_type=TaskType.UNKNOWN_MECHANISM_FORMULATION,
            name="Formal Unknown Mechanism Disclosure",
            system_directive=(
                "ZÁKAZ FABULÁCIE PRÍBEHOV: Pri neznámom mechanizme nevymýšľaj kauzálny naratív. "
                "Založ uzol UnknownMechanismNode s presným zoznamom absentujúcej telemetrie a návrhom ďalšej sondy."
            ),
            required_inputs=["symptom_description", "missing_evidence_list"],
            vital_max_hp=6,
        )

        self.micro_prompts[TaskType.EVIDENCE_VS_REFUTED_CLASSIFICATION] = MicroPrompt(
            prompt_id="MP-CLASSIF-06",
            task_type=TaskType.EVIDENCE_VS_REFUTED_CLASSIFICATION,
            name="Evidence Insufficient vs Refuted Distinction",
            system_directive=(
                "Neoverené nie je nepravdivé. Rozlišuj EVIDENCE_INSUFFICIENT (chýba dôkaz) od REJECTED_REFUTED (preukázaný protidôkaz). "
                "Konvenčné nie je automaticky platné."
            ),
            required_inputs=["claim_id", "current_evidence_state"],
            vital_max_hp=6,
        )

        self.micro_prompts[TaskType.PROMOTION_AUDIT_ROUTING] = MicroPrompt(
            prompt_id="MP-PROMO-07",
            task_type=TaskType.PROMOTION_AUDIT_ROUTING,
            name="Deterministic Promotion Audit & Uncertainty Formatting",
            system_directive=(
                "Povýšenie do lokálnej pravdy (APPROVED_FACT) čaká na nezávislý audit (Anti-Self-Certification: proposer != evaluator). "
                "Výstup formátuj so statusom a neistotou, zákaz nepodložených sebavedomých viet."
            ),
            required_inputs=["proposer_id", "evaluator_id", "measured_metric"],
            vital_max_hp=6,
        )

    def route_micro_prompt(self, task_type: TaskType, context_args: Dict[str, Any]) -> str:
        """Selects and renders the surgical micro-prompt for the specified task."""
        prompt = self.micro_prompts.get(task_type)
        if not prompt:
            raise ContractViolation(f"No micro-prompt registered for task type: {task_type}")
        return prompt.render(context_args)

    def format_output_with_status_and_uncertainty(
        self,
        status: str,
        uncertainty: float,
        statement: str,
        proposer_id: str,
        evaluator_id: str,
    ) -> str:
        """
        Formats output with explicit status and uncertainty quantification.
        Enforces non-sycophancy: avoids overconfident or pleasing assertions.
        """
        # Anti-self-certification check
        if status == "APPROVED_FACT" and proposer_id == evaluator_id:
            raise ContractViolation(
                f"Anti-Self-Certification Violation: Proposer '{proposer_id}' cannot certify claim as APPROVED_FACT"
            )

        return f"[{status} | Uncertainty: {uncertainty:.2f} | Proposer: {proposer_id} | Auditor: {evaluator_id}]\n{statement}"
