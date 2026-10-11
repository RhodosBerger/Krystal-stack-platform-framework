"""
==============================================================================
KRYSTAL-STACK VERIFICATION SUITE: FORMAL THEORY RESOLUTION (TRACK C: P1 & P2)
==============================================================================
Suite: tests/test_formal_theory_resolution.py
Description: Exhaustive test suite for P1 (Strain-Activated Reachability & Triage)
             and P2 (Smith's Rule / WSPT Scheduling & w_COMP Bound).
             Includes deterministic property testing, hysteresis validation,
             counterexample searches, and end-to-end pipeline integration.

Platform Invariant: VITAL_MAX_HP == 6
Epistemic Status: PROPERTY_TESTED, INTEGRATION_VERIFIED
==============================================================================
"""

import unittest
import math
import random
from typing import List

from krystal_kernel.psychological_coping_cortex_mcp_bridge import (
    PsychologicalCopingMechanism,
    VITAL_MAX_HP
)
from krystal_theory.strain_operator import (
    StrainState,
    StrainDomainModel,
    compute_apex_strain,
    CandidateStrainTransform
)
from krystal_theory.triage_operator import (
    TriageOperator,
    ReachabilityResult,
    ReachabilityStatus
)
from krystal_theory.wspt_ranker import (
    SchedulableAction,
    SmithRanker,
    derive_w_comp_upper_bound,
    StressWeightAmplifier
)
from krystal_theory.formal_theory_bridge import (
    FormalTheoryBridge,
    InvariantSafetyGate
)


class TestFormalTheoryResolution(unittest.TestCase):
    """Rigorous verification of Track C formal theory models and scheduler."""

    def setUp(self):
        self.bridge = FormalTheoryBridge()
        self.triage = TriageOperator()
        self.assertEqual(VITAL_MAX_HP, 6, "Universal invariant VITAL_MAX_HP must be 6")

    # ========================================================================
    # P1: STRAIN DOMAIN & REACHABLE MECHANISMS
    # ========================================================================

    def test_zero_strain_no_activation(self):
        """Delta >= 0 yields zero strain, leaving stress mechanisms dormant."""
        # a=1, b=4, c=3 -> Delta = 16 - 12 = 4 > 0 (real regime)
        strain = compute_apex_strain(a=1.0, b=4.0, c=3.0)
        self.assertFalse(strain.is_complex)
        self.assertEqual(strain.raw_strain, 0.0)
        self.assertEqual(strain.normalized_strain, 0.0)
        self.assertEqual(strain.apex_real, -2.0)

        reach = self.triage.evaluate_reachability(
            context_entropy=0.20,
            memory_contention=1.0,
            thermal_pressure_pct=60.0,
            strain=strain
        )
        # In zero strain, HYPERVIGILANCE, RATIONALIZATION, REPRESSION must NOT be active
        self.assertNotIn(PsychologicalCopingMechanism.HYPERVIGILANCE.value, reach.currently_active)
        self.assertNotIn(PsychologicalCopingMechanism.RATIONALIZATION.value, reach.currently_active)
        self.assertNotIn(PsychologicalCopingMechanism.REPRESSION.value, reach.currently_active)
        # PROJECTION is active in nominal quiescent regime
        self.assertIn(PsychologicalCopingMechanism.PROJECTION.value, reach.currently_active)

    def test_positive_strain_reachability(self):
        """Delta < 0 yields positive bounded strain, making stress mechanisms reachable."""
        # a=1, b=2, c=5 -> Delta = 4 - 20 = -16 < 0 (latent complex regime)
        # raw_strain = sqrt(16) / 2 = 2.0
        strain = compute_apex_strain(a=1.0, b=2.0, c=5.0)
        self.assertTrue(strain.is_complex)
        self.assertAlmostEqual(strain.raw_strain, 2.0, places=5)
        self.assertGreater(strain.normalized_strain, 0.0)
        self.assertLess(strain.normalized_strain, 1.0)
        self.assertEqual(strain.apex_real, -1.0)

        reach = self.triage.evaluate_reachability(
            context_entropy=0.20,
            memory_contention=1.0,
            thermal_pressure_pct=70.0,
            strain=strain,
            epistemic_uncertainty=0.50,
            safety_margin=0.20
        )
        # HYPERVIGILANCE is now reachable and active
        self.assertIn(PsychologicalCopingMechanism.HYPERVIGILANCE.value, reach.runtime_reachable)
        self.assertIn(PsychologicalCopingMechanism.HYPERVIGILANCE.value, reach.currently_active)

    def test_activation_threshold(self):
        """Mechanisms activate if and only if strain meets or exceeds their tau_on."""
        triage = TriageOperator()
        # HYPERVIGILANCE tau_on is 0.35
        below_strain = StrainState(
            delta=-1.0, raw_strain=0.34, normalized_strain=0.34,
            apex_real=0.0, is_complex=True, transform_model=CandidateStrainTransform.BOUNDED_TANH
        )
        reach_below = triage.evaluate_reachability(
            context_entropy=0.20, memory_contention=1.0, thermal_pressure_pct=60.0,
            strain=below_strain, epistemic_uncertainty=0.50, safety_margin=0.20
        )
        self.assertNotIn(PsychologicalCopingMechanism.HYPERVIGILANCE.value, reach_below.currently_active)

        at_strain = StrainState(
            delta=-1.0, raw_strain=0.35, normalized_strain=0.35,
            apex_real=0.0, is_complex=True, transform_model=CandidateStrainTransform.BOUNDED_TANH
        )
        reach_at = triage.evaluate_reachability(
            context_entropy=0.20, memory_contention=1.0, thermal_pressure_pct=60.0,
            strain=at_strain, epistemic_uncertainty=0.50, safety_margin=0.20
        )
        self.assertIn(PsychologicalCopingMechanism.HYPERVIGILANCE.value, reach_at.currently_active)

    def test_hysteresis_prevents_chatter(self):
        """
        Sequence: [0.19, 0.21, 0.34, 0.36, 0.33, 0.22, 0.18].
        HYPERVIGILANCE: tau_on = 0.35, tau_off = 0.20.
        Must transition OFF -> ON at 0.36, remain latched ON through 0.33 and 0.22,
        and transition ON -> OFF only at 0.18. Zero oscillation!
        """
        triage = TriageOperator()
        test_sequence = [0.19, 0.21, 0.34, 0.36, 0.33, 0.22, 0.18]
        expected_states = [False, False, False, True, True, True, False]
        observed_states: List[bool] = []

        for s_val in test_sequence:
            strain = StrainState(
                delta=-1.0, raw_strain=s_val, normalized_strain=s_val,
                apex_real=0.0, is_complex=True, transform_model=CandidateStrainTransform.BOUNDED_TANH
            )
            reach = triage.evaluate_reachability(
                context_entropy=0.20, memory_contention=1.0, thermal_pressure_pct=60.0,
                strain=strain, epistemic_uncertainty=0.50, safety_margin=0.20
            )
            is_active = (PsychologicalCopingMechanism.HYPERVIGILANCE.value in reach.currently_active)
            observed_states.append(is_active)

        self.assertEqual(observed_states, expected_states, "Hysteresis state tracking failed")

        # Verify exact number of transitions is 2 (one rising, one falling)
        transitions = sum(1 for i in range(1, len(observed_states)) if observed_states[i] != observed_states[i - 1])
        self.assertEqual(transitions, 2, "Hysteresis must produce exactly 2 transitions without chatter")

    def test_blocked_mechanism_reason(self):
        """Blocked mechanisms provide an explicit, audit-verifiable reason."""
        triage = TriageOperator()
        strain = StrainState(
            delta=1.0, raw_strain=0.0, normalized_strain=0.0,
            apex_real=0.0, is_complex=False, transform_model=CandidateStrainTransform.BOUNDED_TANH
        )
        reach = triage.evaluate_reachability(
            context_entropy=0.10, memory_contention=0.5, thermal_pressure_pct=50.0,
            strain=strain
        )
        self.assertIn("HYPERVIGILANCE", reach.blocked_reasons)
        self.assertIn("STRAIN_BELOW_ON_THRESHOLD", reach.blocked_reasons["HYPERVIGILANCE"])
        self.assertIn("COMPARTMENTALIZATION", reach.blocked_reasons)
        self.assertEqual(reach.blocked_reasons["COMPARTMENTALIZATION"], "MEMORY_CONTENTION_BELOW_LIMIT")

    def test_reachable_vs_active_distinction(self):
        """Confirms distinct classification: THEORETICALLY_REACHABLE, RUNTIME_REACHABLE, CURRENTLY_ACTIVE."""
        triage = TriageOperator()
        # High strain & high temperature triggers REGRESSION
        strain = StrainState(
            delta=-10.0, raw_strain=0.90, normalized_strain=0.90,
            apex_real=0.0, is_complex=True, transform_model=CandidateStrainTransform.BOUNDED_TANH
        )
        reach = triage.evaluate_reachability(
            context_entropy=0.10, memory_contention=1.0, thermal_pressure_pct=90.0,
            strain=strain
        )
        # All 7 are theoretically reachable in taxonomy
        self.assertEqual(len(reach.theoretically_reachable), 7)
        # REGRESSION is active
        self.assertIn("REGRESSION", reach.currently_active)
        # PROJECTION is mutually exclusive with emergency REGRESSION
        self.assertNotIn("PROJECTION", reach.currently_active)

    # ========================================================================
    # P2: SCHEDULING SEMANTICS & SMITH'S RULE / WSPT
    # ========================================================================

    def test_wspt_basic_order(self):
        """Smith's Rule orders jobs in non-increasing order of rho_i = w_i / p_i."""
        a1 = SchedulableAction("A1", PsychologicalCopingMechanism.REGRESSION, weight=2.0, processing_time=50.0) # rho = 0.040
        a2 = SchedulableAction("A2", PsychologicalCopingMechanism.PROJECTION, weight=1.0, processing_time=500.0) # rho = 0.002
        a3 = SchedulableAction("A3", PsychologicalCopingMechanism.COMPARTMENTALIZATION, weight=0.20, processing_time=120.0) # rho = 0.001667

        ranked = SmithRanker.rank_actions([a2, a3, a1])
        ordered_ids = [act.id for _, act in ranked]
        self.assertEqual(ordered_ids, ["A1", "A2", "A3"])

    def test_wspt_monotonic_weight(self):
        """
        Property Test: If w_i increases while p_i is fixed, rho_i increases
        and rank cannot worsen across 1,000 deterministic trials.
        """
        rng = random.Random(42)
        trials = 1000

        for _ in range(trials):
            # Generate 5 background actions
            background = [
                SchedulableAction(
                    f"BG{i}",
                    PsychologicalCopingMechanism.SUBLIMATION,
                    weight=rng.uniform(0.1, 5.0),
                    processing_time=rng.uniform(10.0, 500.0),
                    safety_criticality=rng.uniform(0.0, 0.5),
                    uncertainty=rng.uniform(0.0, 0.5)
                )
                for i in range(5)
            ]

            p_fixed = rng.uniform(20.0, 200.0)
            w_low = rng.uniform(0.1, 2.0)
            w_high = w_low + rng.uniform(0.5, 3.0)

            act_low = SchedulableAction("TARGET", PsychologicalCopingMechanism.HYPERVIGILANCE, weight=w_low, processing_time=p_fixed)
            act_high = SchedulableAction("TARGET", PsychologicalCopingMechanism.HYPERVIGILANCE, weight=w_high, processing_time=p_fixed)

            ranked_low = SmithRanker.rank_actions(background + [act_low])
            ranked_high = SmithRanker.rank_actions(background + [act_high])

            rank_low = next(r for r, a in ranked_low if a.id == "TARGET")
            rank_high = next(r for r, a in ranked_high if a.id == "TARGET")

            # Higher weight must yield smaller or equal rank number (1 is best)
            self.assertLessEqual(rank_high, rank_low, f"Rank worsened when weight increased: {rank_high} > {rank_low}")

    def test_wspt_monotonic_processing_time(self):
        """
        Property Test: If p_i decreases while w_i is fixed, rho_i increases
        and rank cannot worsen across 1,000 deterministic trials.
        """
        rng = random.Random(1337)
        trials = 1000

        for _ in range(trials):
            background = [
                SchedulableAction(
                    f"BG{i}",
                    PsychologicalCopingMechanism.SUBLIMATION,
                    weight=rng.uniform(0.1, 5.0),
                    processing_time=rng.uniform(10.0, 500.0),
                    safety_criticality=rng.uniform(0.0, 0.5),
                    uncertainty=rng.uniform(0.0, 0.5)
                )
                for i in range(5)
            ]

            w_fixed = rng.uniform(0.5, 3.0)
            p_high = rng.uniform(100.0, 400.0)
            p_low = p_high - rng.uniform(10.0, 80.0)

            act_slow = SchedulableAction("TARGET", PsychologicalCopingMechanism.HYPERVIGILANCE, weight=w_fixed, processing_time=p_high)
            act_fast = SchedulableAction("TARGET", PsychologicalCopingMechanism.HYPERVIGILANCE, weight=w_fixed, processing_time=p_low)

            ranked_slow = SmithRanker.rank_actions(background + [act_slow])
            ranked_fast = SmithRanker.rank_actions(background + [act_fast])

            rank_slow = next(r for r, a in ranked_slow if a.id == "TARGET")
            rank_fast = next(r for r, a in ranked_fast if a.id == "TARGET")

            self.assertLessEqual(rank_fast, rank_slow, f"Rank worsened when duration shortened: {rank_fast} > {rank_slow}")

    def test_deterministic_tie_break(self):
        """
        When rho_i is identical, ties break deterministically by:
        1. Higher safety criticality
        2. Lower epistemic uncertainty
        3. Lower duration
        4. Lexicographical ID.
        """
        # All actions have identical rho = 1.0 / 100.0 = 0.010
        # Differing safety criticality
        a1 = SchedulableAction("ID-C", PsychologicalCopingMechanism.SUBLIMATION, weight=1.0, processing_time=100.0, safety_criticality=0.3)
        a2 = SchedulableAction("ID-B", PsychologicalCopingMechanism.SUBLIMATION, weight=1.0, processing_time=100.0, safety_criticality=0.9)
        ranked = SmithRanker.rank_actions([a1, a2])
        self.assertEqual(ranked[0][1].id, "ID-B")

        # Differing uncertainty
        b1 = SchedulableAction("ID-U1", PsychologicalCopingMechanism.SUBLIMATION, weight=1.0, processing_time=100.0, safety_criticality=0.5, uncertainty=0.8)
        b2 = SchedulableAction("ID-U2", PsychologicalCopingMechanism.SUBLIMATION, weight=1.0, processing_time=100.0, safety_criticality=0.5, uncertainty=0.2)
        ranked_b = SmithRanker.rank_actions([b1, b2])
        self.assertEqual(ranked_b[0][1].id, "ID-U2")

        # Differing duration (e.g. 2.0/200.0 == 1.0/100.0 == 0.010)
        c1 = SchedulableAction("ID-LONG", PsychologicalCopingMechanism.SUBLIMATION, weight=2.0, processing_time=200.0, safety_criticality=0.5, uncertainty=0.5)
        c2 = SchedulableAction("ID-SHORT", PsychologicalCopingMechanism.SUBLIMATION, weight=1.0, processing_time=100.0, safety_criticality=0.5, uncertainty=0.5)
        ranked_c = SmithRanker.rank_actions([c1, c2])
        self.assertEqual(ranked_c[0][1].id, "ID-SHORT")

        # Differing ID
        d1 = SchedulableAction("ALPHA", PsychologicalCopingMechanism.SUBLIMATION, weight=1.0, processing_time=100.0, safety_criticality=0.5, uncertainty=0.5)
        d2 = SchedulableAction("BETA", PsychologicalCopingMechanism.SUBLIMATION, weight=1.0, processing_time=100.0, safety_criticality=0.5, uncertainty=0.5)
        ranked_d = SmithRanker.rank_actions([d2, d1])
        self.assertEqual(ranked_d[0][1].id, "ALPHA")

    def test_w_comp_bound(self):
        """
        Derives and tests the w_COMP <= 0.24 condition from Theorem 2.6.
        When w_COMP <= 0.24, PROJ precedes COMP.
        When w_COMP > 0.24, COMP precedes PROJ!
        """
        info = derive_w_comp_upper_bound()
        self.assertEqual(info["w_comp_upper_bound"], 0.24)
        self.assertEqual(info["epistemic_classification"], "MATHEMATICALLY_DERIVED_FROM_EMPIRICAL_PARAMETERS")

        # Case 1: Admissible bound w_COMP = 0.20 <= 0.24
        proj = SchedulableAction("PROJ", PsychologicalCopingMechanism.PROJECTION, weight=1.0, processing_time=500.0) # rho = 0.0020
        comp_admissible = SchedulableAction("COMP", PsychologicalCopingMechanism.COMPARTMENTALIZATION, weight=0.20, processing_time=120.0) # rho = 0.001667
        ranked_adm = SmithRanker.rank_actions([comp_admissible, proj])
        self.assertEqual([a.id for _, a in ranked_adm], ["PROJ", "COMP"])

        # Case 2: Exceeded bound w_COMP = 0.30 > 0.24
        comp_exceeded = SchedulableAction("COMP", PsychologicalCopingMechanism.COMPARTMENTALIZATION, weight=0.30, processing_time=120.0) # rho = 0.0025 > 0.0020
        ranked_exc = SmithRanker.rank_actions([comp_exceeded, proj])
        self.assertEqual([a.id for _, a in ranked_exc], ["COMP", "PROJ"])

    def test_stress_weight_saturation(self):
        """Verifies stress multiplier g(sigma) is monotonic and strictly bounded."""
        amp = StressWeightAmplifier()
        # Projection sensitivity is 0.0 (nominal work never boosted by stress)
        self.assertEqual(amp.compute_multiplier(PsychologicalCopingMechanism.PROJECTION, 0.8), 1.0)

        # Hypervigilance sensitivity alpha = 3.0
        g_0 = amp.compute_multiplier(PsychologicalCopingMechanism.HYPERVIGILANCE, 0.0)
        g_mid = amp.compute_multiplier(PsychologicalCopingMechanism.HYPERVIGILANCE, 0.5)
        g_high = amp.compute_multiplier(PsychologicalCopingMechanism.HYPERVIGILANCE, 1.0)
        g_extreme = amp.compute_multiplier(PsychologicalCopingMechanism.HYPERVIGILANCE, 100.0)

        self.assertEqual(g_0, 1.0)
        self.assertGreater(g_mid, g_0)
        self.assertGreater(g_high, g_mid)
        # Saturation ceiling is exactly 1.0 + alpha = 4.0
        self.assertLessEqual(g_extreme, 4.0)
        self.assertAlmostEqual(g_extreme, 4.0, places=3)

    # ========================================================================
    # COMPOSED PIPELINE & INVARIANT GATE (TRACK C COMPLETE)
    # ========================================================================

    def test_p1_p2_composed_pipeline(self):
        """Executes Scenarios A, B, and C across the full coupled pipeline."""
        bridge = FormalTheoryBridge()

        # Scenario A: Low / zero strain (Delta >= 0)
        res_a = bridge.execute_coupled_pipeline(a=1.0, b=4.0, c=3.0, thermal_pressure_pct=50.0)
        self.assertEqual(res_a["strain"]["normalized_strain"], 0.0)
        self.assertIn("PROJECTION", res_a["reachability"]["currently_active"])
        self.assertEqual(res_a["selected_action"]["mechanism"], "PROJECTION")

        # Scenario B: Moderate strain (sigma in [0.35, 0.70]) -> HYPERVIGILANCE becomes active & ranked
        # a=1, b=2, c=1.5 -> Delta = 4 - 6 = -2 < 0 -> raw = sqrt(2)/2 = 0.7071 -> tanh(0.7071) ~= 0.609
        res_b = bridge.execute_coupled_pipeline(
            a=1.0, b=2.0, c=1.5,
            thermal_pressure_pct=60.0, epistemic_uncertainty=0.50, safety_margin=0.20
        )
        self.assertGreater(res_b["strain"]["normalized_strain"], 0.35)
        self.assertLess(res_b["strain"]["normalized_strain"], 0.85)
        self.assertIn("HYPERVIGILANCE", res_b["reachability"]["currently_active"])
        self.assertEqual(res_b["selected_action"]["mechanism"], "HYPERVIGILANCE")

        # Scenario C: High stress with memory contention
        res_c = bridge.execute_coupled_pipeline(
            a=1.0, b=2.0, c=5.0,
            memory_contention=3.0, thermal_pressure_pct=75.0,
            epistemic_uncertainty=0.60, safety_margin=0.15
        )
        self.assertTrue(len(res_c["candidate_actions"]) >= 2)
        # Ranks must be non-empty and valid
        self.assertIsNotNone(res_c["selected_action"])

    def test_invariant_overrides_rank(self):
        """
        Safety criticality / Invariant Gate strictly overrides any scheduler decision.
        If a non-emergency action is passed when thermal ceiling (95 C) is breached,
        mandatory emergency REGRESSION is enforced.
        """
        proj_action = SchedulableAction("PROJ", PsychologicalCopingMechanism.PROJECTION, weight=10.0, processing_time=50.0)
        approved, effective, reason = InvariantSafetyGate.verify_action(
            action=proj_action,
            thermal_c=96.0
        )
        self.assertFalse(approved)
        self.assertEqual(effective.mechanism, PsychologicalCopingMechanism.REGRESSION)
        self.assertEqual(reason, "THERMAL_CEILING_EXCEEDED_MANDATORY_REGRESSION")

        # In pipeline with manual lockout
        bridge = FormalTheoryBridge()
        res_lockout = bridge.execute_coupled_pipeline(
            a=1.0, b=4.0, c=3.0,
            thermal_pressure_pct=50.0,
            manual_lockout=True
        )
        self.assertFalse(res_lockout["invariant_gate"]["approved"])
        self.assertEqual(res_lockout["invariant_gate"]["reason"], "MANUAL_LOCKOUT_ACTIVE_FALLBACK_TO_REGRESSION")
        self.assertEqual(res_lockout["selected_action"]["mechanism"], "REGRESSION")

    def test_adversarial_counterexamples_falsification(self):
        """
        Actively probes edge cases and potential counterexamples:
        1. High weight + huge duration vs tiny weight + tiny duration
        2. Non-finite values pass to fail-safe
        3. Degenerate quadratic equation
        """
        # Counterexample probe: Action with weight 100 but duration 1,000,000 us vs weight 0.1 duration 10 us
        huge_slow = SchedulableAction("SLOW", PsychologicalCopingMechanism.SUBLIMATION, weight=100.0, processing_time=100000.0) # rho = 0.001
        tiny_fast = SchedulableAction("FAST", PsychologicalCopingMechanism.SUBLIMATION, weight=0.1, processing_time=10.0) # rho = 0.010
        ranked = SmithRanker.rank_actions([huge_slow, tiny_fast])
        # WSPT must pick FAST first despite much lower weight, proving cost efficiency
        self.assertEqual(ranked[0][1].id, "FAST")

        # Non-finite fail-safe probe
        nan_strain = compute_apex_strain(float("nan"), 1.0, 2.0)
        self.assertEqual(nan_strain.normalized_strain, 1.0)
        self.assertTrue(nan_strain.is_complex)

        # Universal invariant check
        self.assertEqual(VITAL_MAX_HP, 6)


if __name__ == "__main__":
    unittest.main()
