#!/usr/bin/env python3
"""
================================================================================
UNIT TESTS: SOVEREIGN REASONING PATTERN BRAIN
================================================================================
Module:     tests/test_sovereign_reasoning_brain.py
Author:     Antigravity Cognitive Invariants Group (2026)
Invariants: VITAL_MAX_HP == 6 | Proposer != Evaluator | Delta S == 0 on breach
Theory:     Theory Dependency Graph frozen at 56 nodes (0 cycles)
================================================================================
"""

import unittest
from krystal_kernel.constants import VITAL_MAX_HP, ContractViolation
from krystal_kernel.sovereign_reasoning_brain import (
    ReasoningMode,
    EvidentialStatus,
    EpistemicModelKind,
    PurposeCategory,
    PatternEvidenceTuple,
    PurposeDescriptor,
    EpistemicModelCandidate,
    ReasoningPatternDescriptor,
    ModelArbitrationReport,
    SovereignReasoningPatternBrain,
    GLOBAL_REASONING_PATTERN_BRAIN,
)


class TestSovereignReasoningPatternBrain(unittest.TestCase):
    """Verifies Brain initialization, purpose indexing, 6-tuple evidence, and 8-model arbitration."""

    def setUp(self) -> None:
        self.brain = SovereignReasoningPatternBrain(vital_max_hp=VITAL_MAX_HP)

    def test_brain_initialization_and_vital_hp(self) -> None:
        """Brain must initialize with VITAL_MAX_HP == 6 in VERIFY mode."""
        self.assertEqual(self.brain.vital_max_hp, VITAL_MAX_HP)
        self.assertEqual(self.brain.current_mode, ReasoningMode.VERIFY)
        self.assertEqual(self.brain.FROZEN_THEORY_NODE_COUNT, 56)

        # Constitutional breach check
        with self.assertRaises(ContractViolation):
            SovereignReasoningPatternBrain(vital_max_hp=5)

    def test_fetch_all_purposes_across_all_seven_layers(self) -> None:
        """fetch_all_purposes gathers all authoritative purposes across 7 stack layers."""
        all_purposes = self.brain.fetch_all_purposes()
        self.assertGreaterEqual(len(all_purposes), 50)

        # 1. 15 Secondary Innovation Loops
        loops = self.brain.fetch_all_purposes(category=PurposeCategory.INNOVATION_LOOP)
        self.assertEqual(len(loops), 15)

        # 2. 7 Sovereign Promptbook Recipes
        recipes = self.brain.fetch_all_purposes(category=PurposeCategory.PROMPTBOOK_RECIPE)
        self.assertEqual(len(recipes), 7)

        # 3. 5 Compositional Reasoning Patterns
        comp_pats = self.brain.fetch_all_purposes(category=PurposeCategory.COMPOSITIONAL_PATTERN)
        self.assertEqual(len(comp_pats), 5)

        # 4. 6 Runtime Subsystems
        subsystems = self.brain.fetch_all_purposes(category=PurposeCategory.SUBSYSTEM_NODE)
        self.assertEqual(len(subsystems), 6)

        # 5. 9 AEP Hypervisor Layers (L0 to L8)
        aep_layers = self.brain.fetch_all_purposes(category=PurposeCategory.AEP_LAYER)
        self.assertEqual(len(aep_layers), 9)

        # 6. 10 Sovereign Principles (KSPP v1 P1 to P10)
        principles = self.brain.fetch_all_purposes(category=PurposeCategory.SOVEREIGN_PRINCIPLE)
        self.assertEqual(len(principles), 10)

        # 7. 8 Epistemic Models (M0 to M7)
        models = self.brain.fetch_all_purposes(category=PurposeCategory.EPISTEMIC_MODEL)
        self.assertEqual(len(models), 8)

    def test_purpose_keyword_search(self) -> None:
        """fetch_all_purposes supports substring keyword search across names, purposes, scopes."""
        uma_purposes = self.brain.fetch_all_purposes(keyword="UMA")
        self.assertGreater(len(uma_purposes), 0)
        for p in uma_purposes:
            matches = (
                "uma" in p.entity_id.lower()
                or "uma" in p.name.lower()
                or "uma" in p.primary_purpose.lower()
                or "uma" in p.scope.lower()
            )
            self.assertTrue(matches)

    def test_dual_mode_switching(self) -> None:
        """Mode switching toggles between EXPLORE and VERIFY."""
        self.assertEqual(self.brain.current_mode, ReasoningMode.VERIFY)
        old = self.brain.switch_mode(ReasoningMode.EXPLORE)
        self.assertEqual(old, ReasoningMode.VERIFY)
        self.assertEqual(self.brain.current_mode, ReasoningMode.EXPLORE)
        self.brain.switch_mode(ReasoningMode.VERIFY)
        self.assertEqual(self.brain.current_mode, ReasoningMode.VERIFY)

    def test_synthesize_pattern_candidate_strict_entry_state(self) -> None:
        """Synthesized patterns must enter strictly as PATTERN_CANDIDATE (Rule 3)."""
        pat = self.brain.synthesize_pattern_candidate(
            name="Thermal-Aware Burst Pacing Candidate",
            characteristic="Allow 1.25x burst only when temp < 55C and headroom > 0.70",
            source="Gemini4Argon::ExplorationBranch",
            scope="UMA_Governor",
            dependencies=["THEORY_04_CHRONOMETRIC_SHADOW"],
            validity="Valid in scanout window",
            falsifier="temp >= 55C under burst",
            model_kind=EpistemicModelKind.M1_ARCHITECTURAL,
            predicate=lambda ctx: ctx.get("temp", 50.0) < 55.0 and ctx.get("headroom", 0.80) > 0.70
        )
        self.assertIsNotNone(pat)
        self.assertEqual(pat.evidence.evidential_status, EvidentialStatus.PATTERN_CANDIDATE)
        self.assertEqual(pat.evidence.vital_max_hp, VITAL_MAX_HP)

    def test_theory_graph_frozen_invariant_enforcement(self) -> None:
        """Referencing prohibited node 57+ must raise ContractViolation."""
        with self.assertRaises(ContractViolation) as cm:
            self.brain.synthesize_pattern_candidate(
                name="Prohibited Theory 57 Extension",
                characteristic="Hypothetical node 57",
                source="IllegalSource",
                scope="Global",
                dependencies=["THEORY_57_UNAUTHORIZED_NODE"],
                validity="Never",
                falsifier="Always"
            )
        self.assertIn("Prohibited theory graph expansion", str(cm.exception))

    def test_eight_model_arbitration_and_conflict_preservation(self) -> None:
        """arbitrate_competing_models selects best matching model without averaging contradictions."""
        # Telemetry observation: 3.18 (closely matches M1 Architectural Bus Contention predicted at 3.20)
        report = self.brain.arbitrate_competing_models(
            observation_metric="latency_cliff_factor",
            observed_value=3.18
        )
        self.assertEqual(report.best_candidate_id, "M1-ARCH-BUS-CONTENTION")
        self.assertEqual(report.best_candidate_kind, EpistemicModelKind.M1_ARCHITECTURAL)
        self.assertFalse(report.is_falsified)
        self.assertLess(report.predicted_vs_observed_delta, 0.05)
        self.assertIn("CDT-EXECUTE", report.cheapest_discriminating_test)
        self.assertEqual(len(report.hmac_digest), 64)

    def test_cheapest_discriminating_test_promotion_and_demotion(self) -> None:
        """CDT execution promotes passing candidate and demotes failing candidate (Delta S == 0)."""
        candidate = self.brain.synthesize_pattern_candidate(
            name="VBLANK Bus Headroom CDT Candidate",
            characteristic="Headroom must exceed 0.60 for burst",
            source="GovernorExperiment",
            scope="BusScheduler",
            dependencies=["THEORY_18_BUS_CONTENTION"],
            validity="VBlank active",
            falsifier="headroom <= 0.60",
            predicate=lambda ctx: float(ctx.get("headroom", 0.0)) > 0.60
        )
        self.assertEqual(candidate.evidence.evidential_status, EvidentialStatus.PATTERN_CANDIDATE)

        # 1. Successful CDT -> Promotes to VERIFIED_EMPIRICAL
        status_pass = self.brain.execute_cheapest_discriminating_test(
            candidate_pattern_id=candidate.pattern_id,
            test_telemetry={"headroom": 0.75}
        )
        self.assertEqual(status_pass, EvidentialStatus.VERIFIED_EMPIRICAL)
        self.assertEqual(candidate.evidence.evidential_status, EvidentialStatus.VERIFIED_EMPIRICAL)

        # 2. Failing CDT -> Demotes to REFUTED
        status_fail = self.brain.execute_cheapest_discriminating_test(
            candidate_pattern_id=candidate.pattern_id,
            test_telemetry={"headroom": 0.40}
        )
        self.assertEqual(status_fail, EvidentialStatus.REFUTED)
        self.assertEqual(candidate.evidence.evidential_status, EvidentialStatus.REFUTED)

    def test_brain_markdown_catalog_generation(self) -> None:
        """generate_brain_markdown_catalog exports comprehensive summary report."""
        md = self.brain.generate_brain_markdown_catalog()
        self.assertIn("# KRYSTAL STACK: SOVEREIGN REASONING PATTERN BRAIN", md)
        self.assertIn("VITAL_MAX_HP == 6", md)
        self.assertIn("Frozen at 56 nodes", md)
        self.assertIn("RECIPE-01", md)
        self.assertIn("KSPP-P10", md)
        self.assertIn("M0-CONVENTIONAL", md)
        self.assertIn("M1-ARCH-BUS-CONTENTION", md)


if __name__ == "__main__":
    unittest.main()
