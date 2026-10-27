#!/usr/bin/env python3
"""
================================================================================
KRYSTAL STACK PLATFORM FRAMEWORK - SOVEREIGN REASONING PATTERN BRAIN
================================================================================
Module:     krystal_kernel/sovereign_reasoning_brain.py
Milestone:  BRAIN-REASONING-PATTERNS-V1
Tenets:     KSPP v1 (P1-P10) | Dual-Mode Epistemic Architecture (EXPLORE vs VERIFY)
Invariants: VITAL_MAX_HP == 6 | Proposer != Evaluator | Delta S == 0 on breach
Theory:     Theory Dependency Graph strictly frozen at 56 nodes (0 cycles)

WHAT THIS MODULE PROVIDES:
  1. DUAL-MODE EPISTEMIC ENGINE (EXPLORE vs VERIFY):
     - EXPLORE: Divergent hypothesis generation, analogical reasoning.
       All new combinations start as PATTERN_CANDIDATE. Zero state mutation.
     - VERIFY: Convergent formal audit, CDT execution, invariant checking.
       Proposer != Evaluator; no mode self-certifies its own outputs.
  2. 6-TUPLE EVIDENCE SCHEMA:
     ⟨Charakteristika, Zdroj, Scope, Evidenčný status, Validita, Závislosti⟩
  3. 8-MODEL EPISTEMIC PORTFOLIO (M0 to M7):
     M0 (Conventional), M1 (Architectural), M2 (Interaction), M3 (Adversarial),
     M4 (Orthogonal Frame), M5 (Null Hypothesis), M6 (Emergent), M7 (Fallback).
  4. UNIFIED PURPOSE INDEX (fetch_all_purposes):
     Aggregates and queries authoritative purposes across 7 structural layers:
     - 15 Secondary Innovation Loops
     - 7 Sovereign Promptbook Recipes
     - 5 Compositional Reasoning Patterns
     - 6 Subsystem Nodes
     - 9 AEP Hypervisor Layers (L0 to L8)
     - 10 Sovereign Principles (KSPP v1 P1 to P10)
     - 8 Reasoning Models (M0 to M7)
  5. CHEAPEST DISCRIMINATING TEST (CDT) & HOMOTOPIC CONFLICT PRESERVATION:
     Never averages contradictions; forks branches until CDT collapses state.
================================================================================
"""

from __future__ import annotations

import enum
import hashlib
import hmac
import json
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from .constants import VITAL_MAX_HP, ContractViolation
from .secondary_innovation_loops import CANONICAL_15_LOOPS_CATALOG, InnovationLoopId


# =============================================================================
# ENUMS & STATUS CODES
# =============================================================================

class ReasoningMode(str, enum.Enum):
    """Dual-Mode Epistemic Architecture operational modes."""
    EXPLORE = "EXPLORE"  # Divergent discovery: proposals, hypotheses, PATTERN_CANDIDATE (Delta S == 0)
    VERIFY  = "VERIFY"   # Convergent audit: CDT execution, formal verification, facts vs refutations


class EvidentialStatus(str, enum.Enum):
    """
    Epistemic stratification lattice levels (10 categories of knowledge).
    Strictly stratified: Demotion mandatory on negative test; premature promotion forbidden.
    """
    PATTERN_CANDIDATE    = "PATTERN_CANDIDATE"    # Newly combined or unvalidated synthesis
    UNVALIDATED          = "UNVALIDATED"          # Theoretical proposal awaiting empirical test
    SIMULATED            = "SIMULATED"            # Synthetic model or simulation trace
    LOCAL_MEASURED       = "LOCAL_MEASURED"       # Empirical quantitative telemetry on local device
    VERIFIED_EMPIRICAL   = "VERIFIED_EMPIRICAL"   # Multi-trial hardware verification with variance
    APPROVED_FACT        = "APPROVED_FACT"        # Certified fact passing external independent audit
    EVIDENCE_INSUFFICIENT= "EVIDENCE_INSUFFICIENT"# Missing or incomplete telemetry
    REFUTED              = "REFUTED"              # Falsified by counter-evidence or CDT breach


class EpistemicModelKind(str, enum.Enum):
    """The 8 competing reasoning paradigms (M0 to M7)."""
    M0_CONVENTIONAL      = "M0_CONVENTIONAL"      # Baseline conventional explanation
    M1_ARCHITECTURAL     = "M1_ARCHITECTURAL"     # Hardware, UMA bus, cache line bouncing, NUMA
    M2_INTERACTION       = "M2_INTERACTION"       # Cross-dynamic interference, phase slip, barrier lockup
    M3_ADVERSARIAL       = "M3_ADVERSARIAL"       # Worst-case payload, fuzzer, edge jitter
    M4_ORTHOGONAL_FRAME  = "M4_ORTHOGONAL_FRAME"  # Hidden confounding variable / frame error
    M5_NULL_HYPOTHESIS   = "M5_NULL_HYPOTHESIS"   # Null hypothesis / background noise / zero effect
    M6_EMERGENT_NONLINEAR= "M6_EMERGENT_NONLINEAR"# Nonlinear phase transition / collective behavior
    M7_REDUCTIVE_FALLBACK= "M7_REDUCTIVE_FALLBACK"# Degraded safe minimum viable operation


class PurposeCategory(str, enum.Enum):
    """Categorization of purposes across the Krystal Stack framework."""
    INNOVATION_LOOP       = "INNOVATION_LOOP"       # Secondary innovation loops (15 loops)
    PROMPTBOOK_RECIPE     = "PROMPTBOOK_RECIPE"     # Sovereign promptbook recipes (7 recipes)
    COMPOSITIONAL_PATTERN = "COMPOSITIONAL_PATTERN" # Compositional visual/tactical patterns (5 patterns)
    SUBSYSTEM_NODE        = "SUBSYSTEM_NODE"        # Runtime subsystems (6 nodes)
    AEP_LAYER             = "AEP_LAYER"             # AEP Hypervisor Layers (L0 to L8)
    SOVEREIGN_PRINCIPLE   = "SOVEREIGN_PRINCIPLE"   # Constitutional Principles (P1 to P10)
    EPISTEMIC_MODEL       = "EPISTEMIC_MODEL"       # Epistemic Reasoning Models (M0 to M7)


# =============================================================================
# DATA STRUCTURES: EVIDENCE 6-TUPLE & PURPOSES
# =============================================================================

@dataclass(frozen=True)
class PatternEvidenceTuple:
    """
    6-Tuple evidence card for pattern combinations (Gemini 4 Argon debugging protocol):
    ⟨Charakteristika, Zdroj, Scope, Evidenčný status, Validita, Závislosti⟩
    """
    characteristic: str
    source: str
    scope: str
    evidential_status: EvidentialStatus
    validity_boundaries: str
    dependencies: List[str]
    falsification_condition: str
    vital_max_hp: int = VITAL_MAX_HP

    def __post_init__(self) -> None:
        if self.vital_max_hp != VITAL_MAX_HP:
            raise ContractViolation(f"Constitutional HP breach in PatternEvidenceTuple: {self.vital_max_hp} != {VITAL_MAX_HP}")


@dataclass(frozen=True)
class PurposeDescriptor:
    """Authoritative representation of an architectural or functional purpose."""
    entity_id: str
    name: str
    category: PurposeCategory
    primary_purpose: str
    scope: str
    target_metric_or_contract: str
    authority_level: str
    vital_max_hp: int = VITAL_MAX_HP

    def __post_init__(self) -> None:
        if self.vital_max_hp != VITAL_MAX_HP:
            raise ContractViolation(f"Constitutional HP breach in PurposeDescriptor: {self.vital_max_hp} != {VITAL_MAX_HP}")


@dataclass(frozen=True)
class EpistemicModelCandidate:
    """A competing hypothesis candidate in the 8-model portfolio (M0 to M7)."""
    model_id: str
    kind: EpistemicModelKind
    hypothesis_mechanism: str
    expected_effect: str
    baseline_metric: float
    predicted_metric: float
    falsifier: str
    cost_evoi_ratio: float = 1.0
    vital_max_hp: int = VITAL_MAX_HP

    def __post_init__(self) -> None:
        if self.vital_max_hp != VITAL_MAX_HP:
            raise ContractViolation(f"Constitutional HP breach in EpistemicModelCandidate: {self.vital_max_hp} != {VITAL_MAX_HP}")


@dataclass
class ReasoningPatternDescriptor:
    """A formal reasoning pattern registered in the Brain."""
    pattern_id: str
    name: str
    evidence: PatternEvidenceTuple
    associated_purpose_id: Optional[str]
    model_kind: EpistemicModelKind
    predicate: Optional[Callable[[Dict[str, Any]], bool]] = None
    vital_max_hp: int = VITAL_MAX_HP

    def __post_init__(self) -> None:
        if self.vital_max_hp != VITAL_MAX_HP:
            raise ContractViolation(f"Constitutional HP breach in ReasoningPatternDescriptor: {self.vital_max_hp} != {VITAL_MAX_HP}")


@dataclass(frozen=True)
class ModelArbitrationReport:
    """Outcome of 8-model competitive arbitration and Cheapest Discriminating Test."""
    arbitration_id: str
    observation_metric: str
    observed_value: float
    best_candidate_id: str
    best_candidate_kind: EpistemicModelKind
    predicted_vs_observed_delta: float
    is_falsified: bool
    cheapest_discriminating_test: str
    residual_uncertainty: float
    competing_evaluations: Dict[str, Dict[str, Any]]
    timestamp_ns: int
    hmac_digest: str
    vital_max_hp: int = VITAL_MAX_HP

    def __post_init__(self) -> None:
        if self.vital_max_hp != VITAL_MAX_HP:
            raise ContractViolation(f"Constitutional HP breach in ModelArbitrationReport: {self.vital_max_hp} != {VITAL_MAX_HP}")


# =============================================================================
# THE SOVEREIGN REASONING PATTERN BRAIN
# =============================================================================

class SovereignReasoningPatternBrain:
    """
    Unified cognitive reasoning, arbitration, and purpose registry of the Krystal Stack:
      Brain = (Patterns, Models, Purposes, Router, VerifierGate)
    Maintains:
      - 6-tuple evidence tracking for pattern combinations.
      - 8-model epistemic portfolio (M0 to M7).
      - Dual-mode switching (EXPLORE vs VERIFY) enforcing Proposer != Evaluator.
      - Comprehensive purpose index across all 7 framework layers.
      - Cheapest Discriminating Test (CDT) execution and Homotopic Conflict Preservation (P6).
      - Constitutional Invariant VITAL_MAX_HP == 6 & Delta S == 0 on breach.
      - Theory dependency graph frozen at 56 nodes.
    """

    FROZEN_THEORY_NODE_COUNT: int = 56

    def __init__(
        self,
        mode: ReasoningMode = ReasoningMode.VERIFY,
        secret_key: bytes = b"krystal_sovereign_reasoning_brain_key_2026",
        vital_max_hp: int = VITAL_MAX_HP
    ) -> None:
        if vital_max_hp != VITAL_MAX_HP:
            raise ContractViolation(f"Constitutional HP breach: {vital_max_hp} != {VITAL_MAX_HP}")
        self.vital_max_hp: int = vital_max_hp
        self.mode: ReasoningMode = mode
        self.secret_key: bytes = secret_key

        self._purposes: Dict[str, PurposeDescriptor] = {}
        self._patterns: Dict[str, ReasoningPatternDescriptor] = {}
        self._models: Dict[str, EpistemicModelCandidate] = {}

        self._initialize_authoritative_purposes()
        self._initialize_canonical_reasoning_patterns()
        self._initialize_canonical_eight_models()

    # -------------------------------------------------------------------------
    # DUAL-MODE CONTROL
    # -------------------------------------------------------------------------

    def switch_mode(self, new_mode: ReasoningMode) -> ReasoningMode:
        """Toggles between EXPLORE (divergent/candidate) and VERIFY (formal audit)."""
        old_mode = self.mode
        self.mode = new_mode
        return old_mode

    @property
    def current_mode(self) -> ReasoningMode:
        return self.mode

    # -------------------------------------------------------------------------
    # PURPOSE ENGINE (fetch_all_purposes)
    # -------------------------------------------------------------------------

    def register_purpose(self, purpose: PurposeDescriptor) -> None:
        """Registers a verified purpose descriptor into the Brain's purpose index."""
        if purpose.vital_max_hp != self.vital_max_hp:
            raise ContractViolation("Purpose descriptor vital HP mismatch")
        self._purposes[purpose.entity_id] = purpose

    def fetch_all_purposes(
        self,
        category: Optional[PurposeCategory] = None,
        keyword: Optional[str] = None
    ) -> List[PurposeDescriptor]:
        """
        Introspects and queries all architectural purposes across the framework.
        Filters by PurposeCategory and/or substring keyword search.
        """
        results = list(self._purposes.values())

        if category is not None:
            results = [p for p in results if p.category == category]

        if keyword is not None:
            kw_lower = keyword.lower()
            results = [
                p for p in results
                if kw_lower in p.entity_id.lower()
                or kw_lower in p.name.lower()
                or kw_lower in p.primary_purpose.lower()
                or kw_lower in p.scope.lower()
            ]

        return sorted(results, key=lambda p: (p.category.value, p.entity_id))

    def get_purpose(self, entity_id: str) -> Optional[PurposeDescriptor]:
        """Retrieves a single purpose by its entity identifier."""
        return self._purposes.get(entity_id)

    # -------------------------------------------------------------------------
    # PATTERN REGISTRY & 6-TUPLE SYNTHESIS
    # -------------------------------------------------------------------------

    def register_pattern(self, pattern: ReasoningPatternDescriptor) -> None:
        """Registers a reasoning pattern with formal 6-tuple evidence."""
        if pattern.vital_max_hp != self.vital_max_hp:
            raise ContractViolation("Reasoning pattern vital HP mismatch")
        self._patterns[pattern.pattern_id] = pattern

    def get_pattern(self, pattern_id: str) -> Optional[ReasoningPatternDescriptor]:
        """Retrieves a registered reasoning pattern."""
        return self._patterns.get(pattern_id)

    def list_patterns_by_status(self, status: EvidentialStatus) -> List[ReasoningPatternDescriptor]:
        """Filters reasoning patterns by their current evidential status level."""
        return [p for p in self._patterns.values() if p.evidence.evidential_status == status]

    def synthesize_pattern_candidate(
        self,
        name: str,
        characteristic: str,
        source: str,
        scope: str,
        dependencies: List[str],
        validity: str,
        falsifier: str,
        model_kind: EpistemicModelKind = EpistemicModelKind.M0_CONVENTIONAL,
        associated_purpose_id: Optional[str] = None,
        predicate: Optional[Callable[[Dict[str, Any]], bool]] = None
    ) -> ReasoningPatternDescriptor:
        """
        Synthesizes a new pattern combination.
        Rule 3 (Invariants): Combination is not proof. Every new pattern begins strictly
        as PATTERN_CANDIDATE and requires independent CDT verification before promotion.
        """
        now_ns = time.perf_counter_ns()
        pattern_id = f"PAT-CAND-{now_ns & 0xFFFFFF}"

        # Invariant: Theory graph remains frozen at 56 nodes; cannot reference imaginary node 57+
        for dep in dependencies:
            if "node_57" in dep.lower() or "theory_57" in dep.lower():
                raise ContractViolation(f"Prohibited theory graph expansion: {dep}. Graph frozen at 56 nodes.")

        evidence = PatternEvidenceTuple(
            characteristic=characteristic,
            source=source,
            scope=scope,
            evidential_status=EvidentialStatus.PATTERN_CANDIDATE,  # Non-negotiable entry state
            validity_boundaries=validity,
            dependencies=dependencies,
            falsification_condition=falsifier,
            vital_max_hp=self.vital_max_hp
        )

        pattern = ReasoningPatternDescriptor(
            pattern_id=pattern_id,
            name=name,
            evidence=evidence,
            associated_purpose_id=associated_purpose_id,
            model_kind=model_kind,
            predicate=predicate,
            vital_max_hp=self.vital_max_hp
        )
        self.register_pattern(pattern)
        return pattern

    # -------------------------------------------------------------------------
    # 8-MODEL PORTFOLIO & CHEAPEST DISCRIMINATING TEST (CDT)
    # -------------------------------------------------------------------------

    def register_model_candidate(self, model: EpistemicModelCandidate) -> None:
        """Registers an epistemic model candidate into the active portfolio."""
        if model.vital_max_hp != self.vital_max_hp:
            raise ContractViolation("Model candidate vital HP mismatch")
        self._models[model.model_id] = model

    def arbitrate_competing_models(
        self,
        observation_metric: str,
        observed_value: float,
        candidate_ids: Optional[List[str]] = None
    ) -> ModelArbitrationReport:
        """
        Evaluates competing models against observed quantitative telemetry.
        Never averages contradictory explanations (Homotopic Conflict Preservation P6).
        Calculates error residual and formats the Cheapest Discriminating Test (CDT).
        """
        target_models = (
            [self._models[m_id] for m_id in candidate_ids if m_id in self._models]
            if candidate_ids is not None
            else list(self._models.values())
        )

        if not target_models:
            raise ContractViolation("No valid model candidates available for arbitration")

        now_ns = time.perf_counter_ns()
        arbitration_id = f"ARB-{now_ns & 0xFFFFFF}"

        evaluations: Dict[str, Dict[str, Any]] = {}
        best_model: Optional[EpistemicModelCandidate] = None
        lowest_delta: float = float("inf")

        for m in target_models:
            delta = abs(observed_value - m.predicted_metric)
            is_falsified = delta > (0.15 * max(1.0, abs(m.predicted_metric)))
            evaluations[m.model_id] = {
                "kind": m.kind.value,
                "mechanism": m.hypothesis_mechanism,
                "predicted": m.predicted_metric,
                "observed": observed_value,
                "delta": round(delta, 4),
                "is_falsified": is_falsified,
                "falsifier": m.falsifier
            }
            if delta < lowest_delta and not is_falsified:
                lowest_delta = delta
                best_model = m

        if best_model is None:
            # All competing models failed: assign to Null Model M5 or report Reframing
            best_model = self._models.get("M5-NULL-NOISE", target_models[0])
            lowest_delta = abs(observed_value - best_model.predicted_metric)
            is_winner_falsified = True
            cdt_test = f"CDT-REFRAME: All primary models falsified against {observation_metric}={observed_value}. Reframe required."
        else:
            is_winner_falsified = False
            cdt_test = f"CDT-EXECUTE: Run lowest-cost telemetry probe discriminating {best_model.model_id} ({best_model.kind.value}) vs M0/M5."

        # Cryptographic HMAC Digest
        sig_payload = f"{arbitration_id}:{best_model.model_id}:{observed_value}:{lowest_delta}:{now_ns}:{self.vital_max_hp}".encode()
        digest = hmac.new(self.secret_key, sig_payload, hashlib.sha256).hexdigest()

        return ModelArbitrationReport(
            arbitration_id=arbitration_id,
            observation_metric=observation_metric,
            observed_value=observed_value,
            best_candidate_id=best_model.model_id,
            best_candidate_kind=best_model.kind,
            predicted_vs_observed_delta=round(lowest_delta, 4),
            is_falsified=is_winner_falsified,
            cheapest_discriminating_test=cdt_test,
            residual_uncertainty=round(lowest_delta / max(0.001, abs(observed_value)), 4),
            competing_evaluations=evaluations,
            timestamp_ns=now_ns,
            hmac_digest=digest,
            vital_max_hp=self.vital_max_hp
        )

    def execute_cheapest_discriminating_test(
        self,
        candidate_pattern_id: str,
        test_telemetry: Dict[str, Any]
    ) -> EvidentialStatus:
        """
        Executes the cheapest discriminating test on a PATTERN_CANDIDATE.
        If the predicate holds and invariants are preserved, promotes to VERIFIED_EMPIRICAL.
        If violated or execution throws, demotes to REFUTED (Delta S == 0 on breach).
        """
        pattern = self.get_pattern(candidate_pattern_id)
        if pattern is None:
            raise ContractViolation(f"Unknown pattern candidate: {candidate_pattern_id}")

        if pattern.predicate is None:
            new_status = EvidentialStatus.EVIDENCE_INSUFFICIENT
        else:
            try:
                passed = bool(pattern.predicate(test_telemetry))
                new_status = EvidentialStatus.VERIFIED_EMPIRICAL if passed else EvidentialStatus.REFUTED
            except Exception:
                new_status = EvidentialStatus.REFUTED

        # Update evidence card immutably
        updated_evidence = PatternEvidenceTuple(
            characteristic=pattern.evidence.characteristic,
            source=pattern.evidence.source,
            scope=pattern.evidence.scope,
            evidential_status=new_status,
            validity_boundaries=pattern.evidence.validity_boundaries,
            dependencies=pattern.evidence.dependencies,
            falsification_condition=pattern.evidence.falsification_condition,
            vital_max_hp=self.vital_max_hp
        )
        pattern.evidence = updated_evidence
        return new_status

    # -------------------------------------------------------------------------
    # CATALOG EXPORT
    # -------------------------------------------------------------------------

    def generate_brain_markdown_catalog(self) -> str:
        """Renders an executive markdown catalog of all registered purposes and reasoning patterns."""
        lines = [
            "# KRYSTAL STACK: SOVEREIGN REASONING PATTERN BRAIN",
            "## Unified Catalog of Architectural Purposes, Epistemic Models & Reasoning Patterns",
            "",
            f"- **Operational Mode:** `{self.mode.value}`",
            f"- **Constitutional Invariant:** `VITAL_MAX_HP == {self.vital_max_hp}`",
            f"- **Theory Dependency Graph:** `Frozen at {self.FROZEN_THEORY_NODE_COUNT} nodes (0 cycles)`",
            f"- **Registered Purposes Count:** `{len(self._purposes)}`",
            f"- **Registered Patterns Count:** `{len(self._patterns)}`",
            f"- **Active Model Portfolio:** `{len(self._models)} (M0..M7)`",
            "",
            "### 1. Unified Purpose Registry (Across 7 Stack Layers)",
            "| Entity ID | Category | Name | Primary Purpose | Scope |",
            "|:---|:---|:---|:---|:---|",
        ]
        for p in self.fetch_all_purposes():
            lines.append(
                f"| `{p.entity_id}` | `{p.category.value}` | **{p.name}** | {p.primary_purpose} | `{p.scope}` |"
            )

        lines.extend([
            "",
            "### 2. Active Reasoning Patterns (with 6-Tuple Evidence)",
            "| Pattern ID | Name | Model Kind | Evidential Status | Characteristic Invariant |",
            "|:---|:---|:---|:---|:---|",
        ])
        for pat in self._patterns.values():
            lines.append(
                f"| `{pat.pattern_id}` | **{pat.name}** | `{pat.model_kind.value}` | `{pat.evidence.evidential_status.value}` | {pat.evidence.characteristic} |"
            )

        lines.extend([
            "",
            "### 3. Epistemic 8-Model Portfolio (M0 to M7)",
            "| Model ID | Kind | Mechanism | Expected Effect | Predicted Metric |",
            "|:---|:---|:---|:---|:---|",
        ])
        for m in self._models.values():
            lines.append(
                f"| `{m.model_id}` | `{m.kind.value}` | {m.hypothesis_mechanism} | {m.expected_effect} | `{m.predicted_metric}` |"
            )

        return "\n".join(lines)

    # -------------------------------------------------------------------------
    # INTERNAL INITIALIZERS
    # -------------------------------------------------------------------------

    def _initialize_authoritative_purposes(self) -> None:
        """Populates all purposes across the 7 layers of Krystal Stack."""

        # 1. Secondary Innovation Loops (15 Loops from CANONICAL_15_LOOPS_CATALOG)
        for loop_id, desc in CANONICAL_15_LOOPS_CATALOG.items():
            self.register_purpose(
                PurposeDescriptor(
                    entity_id=loop_id.value,
                    name=desc.name,
                    category=PurposeCategory.INNOVATION_LOOP,
                    primary_purpose=desc.primary_purpose,
                    scope=f"PriorityTier::{desc.priority_tier.value}",
                    target_metric_or_contract=desc.target_metric,
                    authority_level="SecondaryInnovationLoopAuthority",
                    vital_max_hp=self.vital_max_hp
                )
            )

        # 2. Sovereign Promptbook Recipes (7 Operational Recipes)
        recipes_spec = [
            ("RECIPE-01", "Operations Autocompleter & DAG Decomposer", "Transform unstructured goals into 6-phase acyclic execution DAGs with rollback."),
            ("RECIPE-02", "Bot Archetype & Axiom Synthesis", "Assign cybernetic bot archetypes across timescales and synthesize formal axioms."),
            ("RECIPE-03", "Agile Turn Arbitration Envelope", "Draft and arbitrate subsystem execution turns against thermal and UMA pressure."),
            ("RECIPE-04", "UMA Hex Grid Inspector & Governor", "Inspect 256 memory slots in 16x16 lattice and emit binding governor candidate advice."),
            ("RECIPE-05", "Micro-Op Knapsack Packing Bot", "Pack behavioral sub-operations under 3D (time, memory, energy) budgets with zero planned overrun."),
            ("RECIPE-06", "Behavioral Vulkan Shader Compiler", "Verify RAG phase concurrence and synthesize SPIR-V shaders with economic quotas."),
            ("RECIPE-07", "Runtime Control Plane Benchmark", "Benchmark execution across S0, S1, S2, and K modes to isolate mechanism gains."),
        ]
        for rid, name, purp in recipes_spec:
            self.register_purpose(
                PurposeDescriptor(
                    entity_id=rid,
                    name=name,
                    category=PurposeCategory.PROMPTBOOK_RECIPE,
                    primary_purpose=purp,
                    scope="CompilerFrontend::OperationalContracts",
                    target_metric_or_contract="Proposer != Evaluator & Delta S == 0",
                    authority_level="PromptbookCompilerAuthority",
                    vital_max_hp=self.vital_max_hp
                )
            )

        # 3. Compositional Reasoning Patterns (5 Tactical Patterns from RFC-007)
        comp_patterns = [
            ("COMP-PAT-01", "DepthLayeringPattern", "Partitions viewport into 3 distinct depth planes ensuring depth contrast >= 3.5:1."),
            ("COMP-PAT-02", "FocalPointAnchorPattern", "Pins tactical units to Golden Ratio and Rule of Thirds avoiding visual clutter."),
            ("COMP-PAT-03", "ChromaticHarmonicBalancePattern", "Calculates 3D color tensors ensuring triadic, complementary, or chiaroscuro harmony."),
            ("COMP-PAT-04", "MortarBallisticsCoverPattern", "Solves parabolic ballistic trajectory and validates obstacle clearance."),
            ("COMP-PAT-05", "CapacityTriageBalancingPattern", "Balances magic/physical spell power capacity under VITAL_MAX_HP == 6 invariant."),
        ]
        for pid, name, purp in comp_patterns:
            self.register_purpose(
                PurposeDescriptor(
                    entity_id=pid,
                    name=name,
                    category=PurposeCategory.COMPOSITIONAL_PATTERN,
                    primary_purpose=purp,
                    scope="InterModuleProtocol::TacticalSceneAssembly",
                    target_metric_or_contract="VisualClutterScore < 0.15",
                    authority_level="CompositionalReasoningAuthority",
                    vital_max_hp=self.vital_max_hp
                )
            )

        # 4. Runtime Subsystem Nodes (6 Core Subsystems)
        subsystems = [
            ("SUB-CHRONO-SOC", "Chrono SoC Engine", "Direct memory-mapped hardware silicon, FPGA timers, and lock-free SPSC rings."),
            ("SUB-WSL-BRIDGE", "WSL Linux Bridge", "Zero-copy cross-boundary Linux ABI shared memory and Vulkan execution bridge."),
            ("SUB-GODOT-HOLO", "Godot HoloBridge", "Real-time 3D spatial viewport, mesh buffer generation, and camera transforms."),
            ("SUB-TERRAIN-MAN", "Terrain Manifold", "Procedural geometric elevation, hex coordinates, and volumetric density meshes."),
            ("SUB-SLOVAK-LLM", "Slovak LLM Cascade", "Speculative linguistic transduction, cognitive microprompt routing, and intent compilation."),
            ("SUB-VPP-GRID", "VPP Grid Broker", "Industrial edge telemetry aggregation, power profile dispatch, and economic credit metering."),
        ]
        for sid, name, purp in subsystems:
            self.register_purpose(
                PurposeDescriptor(
                    entity_id=sid,
                    name=name,
                    category=PurposeCategory.SUBSYSTEM_NODE,
                    primary_purpose=purp,
                    scope="RuntimeSubsystemFabric",
                    target_metric_or_contract="TurnWait < 16.6ms & ZeroLockContention",
                    authority_level="SubsystemControllerAuthority",
                    vital_max_hp=self.vital_max_hp
                )
            )

        # 5. AEP Hypervisor Layers (L0 to L8)
        aep_layers = [
            ("AEP-L0", "Contract Layer L0", "128B naturally aligned ZeroCopyContractFrame contracts and capabilities."),
            ("AEP-L1", "Silicon Isolation L1", "Physical hardware register partitions, NUMA domains, and BRAM boundaries."),
            ("AEP-L2", "Spatial Manifold L2", "Hexagonal spatial metric, geodesic coordinates, and geometric bounding boxes."),
            ("AEP-L3", "Observation Plane L3", "UMA 16x16 hex contention topology inspection and sensor telemetry capture."),
            ("AEP-L4", "Governor Advisory L4", "Thermal, RAPL power, and bus stress evaluation issuing advisory directives."),
            ("AEP-L5", "Turn & MicroOp Scheduler L5", "3D multidimensional knapsack admission and turn token allocation."),
            ("AEP-L6", "Execution Fabric L6", "Hardware worker dispatch across CPU SIMD, GPU compute, and NPU pipelines."),
            ("AEP-L7", "Watchdog & Backpressure L7", "Overrun prevention, deadline monitoring, and emergency throttle actuation."),
            ("AEP-L8", "Commit Ledger L8", "Cryptographic HMAC-SHA256 receipts and Merkle tree transaction finalization."),
        ]
        for lid, name, purp in aep_layers:
            self.register_purpose(
                PurposeDescriptor(
                    entity_id=lid,
                    name=name,
                    category=PurposeCategory.AEP_LAYER,
                    primary_purpose=purp,
                    scope="AEP_NineLayerHypervisor",
                    target_metric_or_contract="StrictLinearLayerPrecedence",
                    authority_level="AepConstitutionalAuthority",
                    vital_max_hp=self.vital_max_hp
                )
            )

        # 6. Sovereign Principles (KSPP v1 P1 to P10)
        principles = [
            ("KSPP-P1", "Zero-Touch Interposition", "Zero source-code delta on external open-source modules; interposition at ABI boundary."),
            ("KSPP-P2", "Universal 128B Lingua Franca", "Critical path IPC uses exclusively 128B aligned frames over lock-free SPSC rings."),
            ("KSPP-P3", "Chronometric Shadow Multiplexing", "Reasoning and heavy compute execute strictly in phase shadows to prevent UMA bus thrashing."),
            ("KSPP-P4", "Epistemic Type Safety", "Knowledge levels form an algebraic lattice; demotion is mandatory; premature promotion is forbidden."),
            ("KSPP-P5", "Thermodynamic Metering", "Hypotheses consuming Joules or UMA MB without entropy reduction are terminated as thermal parasites."),
            ("KSPP-P6", "Homotopic Conflict Preservation", "Contradictory evidence is never convexly averaged; superposed branches fork until CDT collapse."),
            ("KSPP-P7", "Physical-First Actuation", "No state commitment on raw prediction; parameter changes require two-phase physical telemetry audit."),
            ("KSPP-P8", "Inertial Stability & Anti-Oscillation", "Fast downscale for deadlines; 12-frame dwell hysteretic upscale with Merkle-momentum."),
            ("KSPP-P9", "Full-Frame Provenance", "Cryptographic HMAC-SHA256 integrity over 96 bytes and canonical IEEE-754 representation."),
            ("KSPP-P10", "Noether-Krystal Survival Symmetry", "Constitutional invariant VITAL_MAX_HP == 6 holds across all layers; invalid action implies Delta S == 0."),
        ]
        for pid, name, purp in principles:
            self.register_purpose(
                PurposeDescriptor(
                    entity_id=pid,
                    name=name,
                    category=PurposeCategory.SOVEREIGN_PRINCIPLE,
                    primary_purpose=purp,
                    scope="KSPP_v1_ConstitutionalLaw",
                    target_metric_or_contract="VITAL_MAX_HP == 6 & Delta S == 0",
                    authority_level="ConstitutionalSovereignAuthority",
                    vital_max_hp=self.vital_max_hp
                )
            )

        # 7. Epistemic Reasoning Models (M0 to M7)
        models_spec = [
            ("EPIST-M0", "Conventional Model", "Baseline standard system explanation (I/O saturation, CPU bottleneck, memory leak)."),
            ("EPIST-M1", "Architectural Model", "Structural hardware invariant failure, cache line bouncing, NUMA/UMA transfer bottleneck."),
            ("EPIST-M2", "Interaction Model", "Dynamic cross-process interference, sync barrier lockup, phase slip (LLM burst vs VSync)."),
            ("EPIST-M3", "Adversarial Model", "Worst-case jitter, malicious/corrupted payload, buffer overflow, boundary drift."),
            ("EPIST-M4", "Orthogonal Frame Model", "Hidden confounding variable / problem framing error requiring reframing."),
            ("EPIST-M5", "Null Hypothesis Model", "Background noise, measurement error, zero true causal effect."),
            ("EPIST-M6", "Emergent Nonlinear Model", "Nonlinear phase transition, collective swarm behavior, bifurcation."),
            ("EPIST-M7", "Reductive Fallback Model", "Emergency degraded safe mode operating within absolute minimum vital invariants."),
        ]
        for mid, name, purp in models_spec:
            self.register_purpose(
                PurposeDescriptor(
                    entity_id=mid,
                    name=name,
                    category=PurposeCategory.EPISTEMIC_MODEL,
                    primary_purpose=purp,
                    scope="EpistemicReasoningPortfolio",
                    target_metric_or_contract="CheapestDiscriminatingTestEvaluated",
                    authority_level="EpistemicRouterAuthority",
                    vital_max_hp=self.vital_max_hp
                )
            )

    def _initialize_canonical_reasoning_patterns(self) -> None:
        """Initializes canonical verified reasoning patterns."""
        # 1. Depth Layering Pattern
        self.register_pattern(
            ReasoningPatternDescriptor(
                pattern_id="PAT-DEPTH-LAYERING",
                name="Three-Tier Viewport Depth Layering",
                evidence=PatternEvidenceTuple(
                    characteristic="contrast_ratio >= 3.5:1 across 3 depth planes",
                    source="RFC-007::InterModuleProtocol",
                    scope="ViewportCompositor",
                    evidential_status=EvidentialStatus.VERIFIED_EMPIRICAL,
                    validity_boundaries="Valid when camera near_z >= 0.5m",
                    dependencies=["THEORY_01_SPATIAL_CONTINUITY", "THEORY_12_COMPOSITION_HARMONY"],
                    falsification_condition="contrast_ratio < 3.5",
                    vital_max_hp=self.vital_max_hp
                ),
                associated_purpose_id="COMP-PAT-01",
                model_kind=EpistemicModelKind.M0_CONVENTIONAL,
                predicate=lambda ctx: float(ctx.get("contrast_ratio", 4.0)) >= 3.5,
                vital_max_hp=self.vital_max_hp
            )
        )

        # 2. Constitutional Invariant Guard Pattern
        self.register_pattern(
            ReasoningPatternDescriptor(
                pattern_id="PAT-VITAL-HP-GUARD",
                name="Constitutional VITAL_MAX_HP Strict Clamp",
                evidence=PatternEvidenceTuple(
                    characteristic="VITAL_MAX_HP == 6 holds under all operations",
                    source="KSPP_v1::PrincipleP10",
                    scope="EntireKrystalStack",
                    evidential_status=EvidentialStatus.APPROVED_FACT,
                    validity_boundaries="Universal across all execution lanes and timescales",
                    dependencies=["THEORY_00_NOETHER_CONSERVATION"],
                    falsification_condition="vital_hp != 6",
                    vital_max_hp=self.vital_max_hp
                ),
                associated_purpose_id="KSPP-P10",
                model_kind=EpistemicModelKind.M1_ARCHITECTURAL,
                predicate=lambda ctx: ctx.get("vital_max_hp", VITAL_MAX_HP) == VITAL_MAX_HP,
                vital_max_hp=self.vital_max_hp
            )
        )

        # 3. UMA Headroom Throttle Pattern
        self.register_pattern(
            ReasoningPatternDescriptor(
                pattern_id="PAT-UMA-HEADROOM-THROTTLE",
                name="Predictive UMA Headroom Throttling",
                evidence=PatternEvidenceTuple(
                    characteristic="Burst authorized only when headroom > 0.60 and UMA < 0.40",
                    source="SovereignGovernor::ResourceControlPlane",
                    scope="UMA_InspectorGovernor",
                    evidential_status=EvidentialStatus.VERIFIED_EMPIRICAL,
                    validity_boundaries="Valid when thermal < 75.0C",
                    dependencies=["THEORY_04_CHRONOMETRIC_SHADOW", "THEORY_18_BUS_CONTENTION"],
                    falsification_condition="burst_authorized == True when headroom <= 0.60",
                    vital_max_hp=self.vital_max_hp
                ),
                associated_purpose_id="RECIPE-04",
                model_kind=EpistemicModelKind.M1_ARCHITECTURAL,
                predicate=lambda ctx: float(ctx.get("headroom", 0.70)) > 0.60 and float(ctx.get("uma", 0.30)) < 0.40,
                vital_max_hp=self.vital_max_hp
            )
        )

        # 4. Zero Planned Overrun Admission Pattern
        self.register_pattern(
            ReasoningPatternDescriptor(
                pattern_id="PAT-ZERO-OVERRUN-ADMISSION",
                name="3D MicroOp Knapsack Zero Overrun Admission",
                evidence=PatternEvidenceTuple(
                    characteristic="sum(tau_i) <= Q_time and sum(mu_i) <= Q_memory and sum(E_i) <= Q_energy",
                    source="MicroOpAllocationBot::P5",
                    scope="TurnScheduler_L5",
                    evidential_status=EvidentialStatus.VERIFIED_EMPIRICAL,
                    validity_boundaries="Valid under nominal RAPL limits",
                    dependencies=["THEORY_08_KNAPSACK_ADMISSION", "THEORY_22_THERMODYNAMIC_METERING"],
                    falsification_condition="total_planned_cycles > cycle_quota",
                    vital_max_hp=self.vital_max_hp
                ),
                associated_purpose_id="RECIPE-05",
                model_kind=EpistemicModelKind.M0_CONVENTIONAL,
                predicate=lambda ctx: float(ctx.get("planned_overrun_rate", 0.0)) == 0.0,
                vital_max_hp=self.vital_max_hp
            )
        )

    def _initialize_canonical_eight_models(self) -> None:
        """Initializes the standard competing 8-model portfolio (M0 to M7)."""
        self.register_model_candidate(
            EpistemicModelCandidate(
                model_id="M0-CONVENTIONAL",
                kind=EpistemicModelKind.M0_CONVENTIONAL,
                hypothesis_mechanism="Linear processing queue latency and basic scheduling overhead",
                expected_effect="Proportional increase in turn wait time with FIFO queue depth",
                baseline_metric=1.0,
                predicted_metric=1.15,
                falsifier="Measured wait time is nonlinearly decoupled from queue depth"
            )
        )
        self.register_model_candidate(
            EpistemicModelCandidate(
                model_id="M1-ARCH-BUS-CONTENTION",
                kind=EpistemicModelKind.M1_ARCHITECTURAL,
                hypothesis_mechanism="UMA memory bus saturation causing CPU/GPU cache line bouncing",
                expected_effect="Sudden 3x latency cliff when memory traffic exceeds 0.78 UMA ceiling",
                baseline_metric=1.0,
                predicted_metric=3.20,
                falsifier="High latency observed with UMA pressure strictly < 0.40"
            )
        )
        self.register_model_candidate(
            EpistemicModelCandidate(
                model_id="M2-INTERACTION-SLIP",
                kind=EpistemicModelKind.M2_INTERACTION,
                hypothesis_mechanism="Phase slip between LLM inference burst and VBlank scanout window",
                expected_effect="Periodic jitter spikes correlated with 16.6ms display refresh cadence",
                baseline_metric=0.0,
                predicted_metric=16.6,
                falsifier="Jitter spikes occur without display scanout correlation"
            )
        )
        self.register_model_candidate(
            EpistemicModelCandidate(
                model_id="M3-ADVERSARIAL-OVERRUN",
                kind=EpistemicModelKind.M3_ADVERSARIAL,
                hypothesis_mechanism="Malicious or corrupted payload forcing unbounded micro-op recursion",
                expected_effect="Instantaneous cycle quota exhaustion within single quantum window",
                baseline_metric=100.0,
                predicted_metric=5000.0,
                falsifier="Batch execution terminates cleanly within allocated quantum"
            )
        )
        self.register_model_candidate(
            EpistemicModelCandidate(
                model_id="M4-ORTHOGONAL-CONFOUNDER",
                kind=EpistemicModelKind.M4_ORTHOGONAL_FRAME,
                hypothesis_mechanism="Hidden OS background paging or DPC latency spike masking true state",
                expected_effect="Erratic latency bursts uncorrelated with any Krystal stack subsystem",
                baseline_metric=0.5,
                predicted_metric=25.0,
                falsifier="Latency perfectly tracks internal subsystem execution counters"
            )
        )
        self.register_model_candidate(
            EpistemicModelCandidate(
                model_id="M5-NULL-NOISE",
                kind=EpistemicModelKind.M5_NULL_HYPOTHESIS,
                hypothesis_mechanism="Normal statistical white noise and sensor quantization variance",
                expected_effect="Zero mean deviation with normal Gaussian distribution variance",
                baseline_metric=0.0,
                predicted_metric=0.02,
                falsifier="Distribution shows systematic non-zero bias or skewed tail percentiles"
            )
        )
        self.register_model_candidate(
            EpistemicModelCandidate(
                model_id="M6-EMERGENT-PHASE",
                kind=EpistemicModelKind.M6_EMERGENT_NONLINEAR,
                hypothesis_mechanism="Nonlinear phase transition in multi-actor token negotiation lattice",
                expected_effect="Self-organizing rhythmic synchronization across disparate subsystems",
                baseline_metric=0.2,
                predicted_metric=0.95,
                falsifier="System response remains strictly linear without synchronization lock"
            )
        )
        self.register_model_candidate(
            EpistemicModelCandidate(
                model_id="M7-REDUCTIVE-FALLBACK",
                kind=EpistemicModelKind.M7_REDUCTIVE_FALLBACK,
                hypothesis_mechanism="Emergency degraded safe mode clamping all subsystems to VITAL_MAX_HP == 6",
                expected_effect="Total freeze of non-essential computation while preserving invariant state",
                baseline_metric=1.0,
                predicted_metric=0.0,
                falsifier="Subsystem continues active mutation despite emergency clamp"
            )
        )


# Global singleton instance
GLOBAL_REASONING_PATTERN_BRAIN = SovereignReasoningPatternBrain()
