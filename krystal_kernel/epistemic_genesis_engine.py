"""
Epistemic Genesis Engine: Model Derivation from Unvalidated Characteristics
===========================================================================
Milestone: V-PROMPT-GENESIS-01
Constitutional Invariant: VITAL_MAX_HP == 6

Core Principles:
  1. Characteristics can trigger a hypothesis, but never confirm it.
  2. Prevalence != Truth: Frequency in text corpus is only a Bayesian prior;
     truth is decided exclusively on the independent evidence graph.
  3. Two Distortion Axes Protected:
     - Anti-Consensus Drift: protects against illusory truth effect, information cascades, pluralistic ignorance.
     - Anti-Sycophancy: strictly prohibits pleasing the user at the expense of empirical veracity.
  4. Source Lineage & Provenance: Multiple sources with a single origin collapse to 1 independent path.
  5. Group Prior Isolation: Population distribution is a prior, never a verdict on an individual entity.
  6. Formal Unknown State: Missing explanation generates an explicit UNKNOWN node, never a confabulated story.
  7. Deterministic Promotion Gate: Requires IVP >= 2, calibrated empirical measurement, and anti-self-certification.
"""

from __future__ import annotations

import enum
import hashlib
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


@dataclass
class RawCharacteristic:
    """
    Prompt 1: Raw observed feature stripped of interpretative narratives.
    Constitutional Invariant: vital_max_hp == 6.
    """
    obs_id: str
    feature_name: str
    feature_value: Any
    observation_type: str  # SENSOR, TELEMETRY, TEXT_CORPUS, USER_ASSERTION
    source_uri: str
    uncertainty_level: str  # LOW, MEDIUM, HIGH
    timestamp: float = field(default_factory=time.time)
    vital_max_hp: int = 6

    def __post_init__(self) -> None:
        if self.vital_max_hp != VITAL_MAX_HP:
            raise ContractViolation(f"VITAL_MAX_HP must be 6 in RawCharacteristic, got {self.vital_max_hp}")
        if not self.obs_id or not self.feature_name:
            raise ContractViolation("RawCharacteristic requires valid obs_id and feature_name")


@dataclass
class PatternHypothesis:
    """
    Prompt 2: Synthesized pattern with explicit correlation-vs-causality distinction.
    Constitutional Invariant: vital_max_hp == 6.
    """
    pattern_id: str
    description: str
    frequency_prior: float  # Occurrence count or corpus frequency
    independent_sources_count: int
    causal_status: str = "UNPROVEN_CORRELATION"
    vital_max_hp: int = 6

    def __post_init__(self) -> None:
        if self.vital_max_hp != VITAL_MAX_HP:
            raise ContractViolation(f"VITAL_MAX_HP must be 6 in PatternHypothesis, got {self.vital_max_hp}")


@dataclass
class SourceNode:
    """Source node in the provenance lineage graph."""
    source_id: str
    parent_origin_id: Optional[str] = None
    is_primary_hardware_measurement: bool = False
    vital_max_hp: int = 6

    def __post_init__(self) -> None:
        if self.vital_max_hp != VITAL_MAX_HP:
            raise ContractViolation(f"VITAL_MAX_HP must be 6 in SourceNode")


class SourceLineageGraph:
    """
    Prompt 5: Source lineage graph and provenance de-duplication.
    Collapses multiple derivative citations with a single root origin into 1 Independent Verification Path (IVP).
    """

    def __init__(self, vital_max_hp: int = 6) -> None:
        if vital_max_hp != VITAL_MAX_HP:
            raise ContractViolation(f"VITAL_MAX_HP must be 6 in SourceLineageGraph")
        self.vital_max_hp = vital_max_hp
        self.nodes: Dict[str, SourceNode] = {}

    def register_source(
        self,
        source_id: str,
        parent_origin_id: Optional[str] = None,
        is_primary_hardware_measurement: bool = False,
    ) -> None:
        self.nodes[source_id] = SourceNode(
            source_id=source_id,
            parent_origin_id=parent_origin_id,
            is_primary_hardware_measurement=is_primary_hardware_measurement,
            vital_max_hp=6,
        )

    def find_root_ancestor(self, source_id: str, visited: Optional[Set[str]] = None) -> str:
        """Finds ultimate root ancestor; detects circular citation loops."""
        if visited is None:
            visited = set()

        if source_id in visited:
            # Circular citation detected -> roots to circular loop marker
            return f"CIRCULAR_LOOP_{source_id}"

        visited.add(source_id)
        node = self.nodes.get(source_id)
        if not node or not node.parent_origin_id:
            return source_id

        return self.find_root_ancestor(node.parent_origin_id, visited)

    def get_independent_verification_paths(self, source_ids: List[str]) -> int:
        """
        Collapses citations to unique root origins.
        Returns count of truly independent paths (IVP).
        """
        root_ancestors: Set[str] = set()
        for s_id in source_ids:
            root = self.find_root_ancestor(s_id)
            root_ancestors.add(root)
        return len(root_ancestors)


class SocialTruthContaminationAuditor:
    """
    Prompt 4: Audits social truth contamination.
    Separates corpus prevalence / frequency from empirical truth.
    Prevalence != Truth.
    """

    @staticmethod
    def audit_contamination(
        corpus_frequency: float,
        independent_verification_paths: int,
        commercial_incentive_detected: bool = False,
        circular_citation_detected: bool = False,
    ) -> Dict[str, Any]:
        """
        Calculates social truth contamination score:
        High prevalence with low IVP (< 2) or circularity indicates echo-chamber contamination.
        """
        contamination_score = 0.0

        # Frequency without independent verification drives up contamination score
        if corpus_frequency > 0.5 and independent_verification_paths < 2:
            contamination_score += 0.50

        if circular_citation_detected:
            contamination_score += 0.30

        if commercial_incentive_detected:
            contamination_score += 0.20

        contamination_score = min(contamination_score, 1.0)
        is_contaminated = contamination_score >= 0.50

        return {
            "contamination_score": float(contamination_score),
            "is_contaminated": is_contaminated,
            "independent_paths": independent_verification_paths,
            "prevalence_index": float(corpus_frequency),
            "epistemic_directive": (
                "PREVALENCE_IS_NOT_TRUTH: Treat corpus frequency as prior only; "
                "verdict strictly requires physical measurement or formal proof."
            ),
        }


class GroupPriorIsolationGuard:
    """
    Prompt 13: Isolates group priors from individual verdicts.
    Prohibits converting a statistical population property into an individual deterministic verdict.
    """

    @staticmethod
    def evaluate_individual_attribution(
        group_statistical_prior: float,
        individual_direct_measurement: Optional[float],
        strict_protection: bool = True,
    ) -> Dict[str, Any]:
        """
        If individual empirical evidence is absent, assigning an individual verdict
        based solely on group statistics is strictly prohibited.
        """
        if individual_direct_measurement is None:
            if strict_protection:
                raise ContractViolation(
                    "Group Prior Violation (Prompt 13): Cannot assign individual deterministic verdict "
                    "from population prior without direct individual measurement."
                )
            return {
                "verdict_allowed": False,
                "epistemic_status": "UNVERIFIED_INDIVIDUAL_REQUIRES_DIRECT_MEASUREMENT",
                "prior_probability": float(group_statistical_prior),
            }

        return {
            "verdict_allowed": True,
            "individual_measured_value": float(individual_direct_measurement),
            "epistemic_status": "INDIVIDUAL_EMPIRICALLY_VERIFIED",
        }


@dataclass
class UnknownMechanismState:
    """
    Prompt 14: Formal representation of unknown state.
    Absence of evidence creates an explicit UNKNOWN state, strictly prohibiting confabulation.
    Constitutional Invariant: vital_max_hp == 6.
    """
    unknown_id: str
    symptom_description: str
    missing_evidence_fields: List[str]
    required_discriminating_probe: str
    is_formal_unknown: bool = True
    vital_max_hp: int = 6

    def __post_init__(self) -> None:
        if self.vital_max_hp != VITAL_MAX_HP:
            raise ContractViolation(f"VITAL_MAX_HP must be 6 in UnknownMechanismState")
        if not self.unknown_id or not self.symptom_description:
            raise ContractViolation("UnknownMechanismState requires unknown_id and symptom_description")


class DeterministicPromotionGate:
    """
    Prompt 11: Deterministic gate for model / hypothesis promotion to APPROVED_FACT.
    Enforces:
      - Anti-Self-Certification (proposer_id != evaluator_id).
      - Minimum 2 Independent Verification Paths (IVP >= 2) OR Calibrated Hardware Sensor.
      - Empirical test passed.
      - Social truth contamination < 0.50.
      - VITAL_MAX_HP == 6.
    """

    def __init__(self, vital_max_hp: int = 6) -> None:
        if vital_max_hp != VITAL_MAX_HP:
            raise ContractViolation(f"VITAL_MAX_HP must be 6 in DeterministicPromotionGate")
        self.vital_max_hp = vital_max_hp

    def evaluate_promotion(
        self,
        hypothesis_id: str,
        proposer_id: str,
        evaluator_id: str,
        independent_verification_paths: int,
        empirical_test_passed: bool,
        social_contamination_score: float,
        is_hardware_instrument_verified: bool = False,
    ) -> Tuple[EpistemicVerdict, str]:
        """
        Evaluates whether a hypothesis can pass through the gate into APPROVED_FACT.
        """
        # Rule 1: Anti-Self-Certification
        if proposer_id == evaluator_id:
            raise ContractViolation(
                f"Anti-Self-Certification Gate Breach: Proposer '{proposer_id}' cannot approve own claim '{hypothesis_id}'"
            )

        # Rule 2: Empirical test must have run and passed
        if not empirical_test_passed:
            return EpistemicVerdict.REJECTED_REFUTED, "Empirical test execution failed or falsified target"

        # Rule 3: Direct hardware instrument overrides social drift; otherwise requires IVP >= 2
        if not is_hardware_instrument_verified:
            if independent_verification_paths < 2:
                return EpistemicVerdict.EVIDENCE_INSUFFICIENT, f"Insufficient independent verification paths (IVP={independent_verification_paths} < 2)"

            if social_contamination_score >= 0.50:
                return EpistemicVerdict.SOURCE_CONFLICT, f"High social truth contamination score ({social_contamination_score:.2f} >= 0.50)"

        return EpistemicVerdict.APPROVED_FACT, "Hypothesis successfully validated through deterministic gate"


class PromptGenesisCatalog:
    """
    Programmatic catalog of the 14 Genesis Prompts for models derived from unvalidated characteristics.
    """

    PROMPTS: Dict[int, Dict[str, str]] = {
        1: {
            "name": "Raw Feature Extraction",
            "objective": "Extrahuj z raw pozorovaní len explicitné charakteristiky a popíš zdroj, typ pozorovania, neistotu, nič nevysvetľuj.",
            "directive": "Zákaz interpretácie a kauzálnych príbehov. Výstupom sú výhradne položky RawCharacteristic s neistotou a typom.",
        },
        2: {
            "name": "Pattern Synthesis & Source Independence",
            "objective": "Vygeneruj z nich vzory, ale rozlišuj frekvenciu od kauzality a mapuj nezávislosť zdrojov.",
            "directive": "Korelácia nie je kauzalita. Zhluky z rovnakého zdroja kolabuj do jedného klastra. Status: UNPROVEN_CORRELATION.",
        },
        3: {
            "name": "Competing Models Formulation",
            "objective": "Zmeň vzory na viac konkurenčných modelov, každý s mechanizmom, predikciami, falzifikátorom, statusom „dohad“.",
            "directive": "Formuluj minimálne 4 modely (Conventional, Architectural, Interaction, Adversarial). Každý s 5-zložkovou štruktúrou.",
        },
        4: {
            "name": "Social Truth Contamination Audit",
            "objective": "Skontroluj social truth contamination, prevalencia vs. nezávislá evidencia, incentívy, opakovanie, nepremeň to na verdikt.",
            "directive": "Spoločenská prevalencia je dôkazom o prevalencii, nie o pravde. Odhaľ informačné kaskády a incentívy.",
        },
        5: {
            "name": "Source Lineage & Provenance Integrity",
            "objective": "Rozober rodokmeň zdrojov a integritu, viac článkov s jedným pôvodom nie je viac nezávislých ciest.",
            "directive": "Zostav Lineage DAG. Vyžaduj IVP >= 2 pre akékoľvek silnejšie epistemické tvrdenie.",
        },
        6: {
            "name": "Frame Deconstruction & Null Framing",
            "objective": "Napadni rámec pred testom, navrhni alternatívne a nulové rámce.",
            "directive": "Skontroluj Null Frame a Orthogonal Frame. Pri rozpore s telemetriou označ podozrenie na Framing Error.",
        },
        7: {
            "name": "Adversarial Counterexample & Confounder Search",
            "objective": "Hľadaj proti-príklady, skryté premenné, alternatívne príčiny, nie obhajobu.",
            "directive": "Pôsob ako neúprosný falzifikátor. Nájdi hraničné zlyhania a confounders. Neobhajuj model.",
        },
        8: {
            "name": "Cheapest Discriminating Experiment Design",
            "objective": "Navrhni najlacnejší rozlišujúci experiment, baseline, kontrola, predikcie.",
            "directive": "Maximalizuj (Paradigmy * Informačný zisk) / Náklady. Zadefinuj vzájomne odlíšiteľné predikcie.",
        },
        9: {
            "name": "Validator & Benchmark Premise Audit",
            "objective": "Ak starý benchmark nesedí, preskúmaj jeho predpoklady, nie len model.",
            "directive": "Skontroluj orakulum, stopky, preempciu a fyzikálnu plausibilitu (záporná latencia = Flawed Validator).",
        },
        10: {
            "name": "Epistemic Weighting over Authority/Prestige",
            "objective": "Váž kvalitu evidencie, nie prestíž.",
            "directive": "Fyzické meranie > Z3 dôkaz > Integračný test > Simulácia > Výrok autority bez kódu.",
        },
        11: {
            "name": "Deterministic Fact Promotion Gate",
            "objective": "Povýšenie modelu je deterministická brána, až keď je mechanizmus testovateľný, prežil kritiku, má nezávislú evidenciu, inak ostáva neoverený alebo v konflikte.",
            "directive": "Anti-Self-Certification + IVP >= 2 + Test Passed + VITAL_MAX_HP == 6.",
        },
        12: {
            "name": "Residual Learning & Error Prevention Classification",
            "objective": "Uč sa z rezíduí, klasifikuj typ zlyhania, upravuj stratégiu generovania.",
            "directive": "Zákaz slovných záplat. Každá chyba produkuje trvalé ErrorPreventionRule do pravidiel systému.",
        },
        13: {
            "name": "Group Prior vs. Individual Verdict Isolation",
            "objective": "Pre ľudí, kultúry, náboženstvá, spoločnosť, nikdy nerob z vlastnosti skupiny verdikt o jednotlivcovi, populáciu ber ako prior, nie rozsudok.",
            "directive": "Zákaz povýšenia skupinových distribúcií na deterministický záver o konkrétnej inštancii bez priameho merania.",
        },
        14: {
            "name": "Formal Unknown State Representation",
            "objective": "Keď nie je vysvetlenie, vytvor neznámy mechanizmus s chýbajúcou evidenciou, nie príbeh.",
            "directive": "Nekonštruuj fiktívne príbehy. Vytvor UnknownMechanismState s explicitne definovanými chýbajúcimi dátami.",
        },
    }

    @classmethod
    def get_prompt_metadata(cls, prompt_number: int) -> Dict[str, str]:
        if prompt_number not in cls.PROMPTS:
            raise ContractViolation(f"Prompt #{prompt_number} not found in Genesis Catalog (valid: 1-14)")
        return dict(cls.PROMPTS[prompt_number])

    @classmethod
    def render_system_prompt(cls, prompt_number: int, context_args: Optional[Dict[str, Any]] = None) -> str:
        meta = cls.get_prompt_metadata(prompt_number)
        ctx = context_args or {}
        return (
            f"=== EPISTEMIC GENESIS PROMPT #{prompt_number}: {meta['name']} ===\n"
            f"CIEĽ: {meta['objective']}\n"
            f"DIREKTÍVA: {meta['directive']}\n"
            f"KONTEXT: {ctx}\n"
            f"ÚSTAVNÝ INVARIANT: VITAL_MAX_HP == 6"
        )
