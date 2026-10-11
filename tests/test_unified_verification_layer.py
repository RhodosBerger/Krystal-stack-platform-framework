#!/usr/bin/env python3
"""
UNIT & INTEGRATION TESTS: KRYSTAL UNIFIED SIMULATION & VERIFICATION LAYER
=============================================================================
Tests all core architectural tenets:
  1. Multi-dimensional StateVector math, U, S_min, dispersion D, and SVG radar.
  2. Strict Pareto tradeoff rule (NEVER classify as better using U alone).
  3. Universal invariant enforcement (VITAL_MAX_HP == 6, zero swap, temp ceiling).
  4. Versioned policy engine with deterministic safety gate.
  5. Outcome evaluator with prediction RMSE, unexpected side effects, and repeatability.
  6. Epistemic claim promotion rules (N >= 5, CV <= 0.15) and falsification triggers.
  7. Cryptographic ReplayPackage determinism and divergence bounds.
  8. Cross-domain comparator across all 5 verification domains.
  9. Full 8-phase epistemic execution pipeline via UnifiedVerificationFacade.
=============================================================================
"""

import os
import sys
import unittest
import math

WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

from krystal_kernel.scene_constructor_presets import VITAL_MAX_HP
from krystal_kernel.verification.state_vector import (
    StateVector,
    StateGeometry,
    TradeoffClassification,
    ParetoStatus,
)
from krystal_kernel.verification.invariants import (
    UniversalInvariantLayer,
    InvariantSeverity,
)
from krystal_kernel.verification.evidence import (
    EvidenceLevel,
    EpistemicState,
    EvidenceRegistry,
)
from krystal_kernel.verification.policy_engine import (
    VerificationPolicy,
    PolicyTransformationEngine,
    ProposedAction,
)
from krystal_kernel.verification.outcome_evaluator import OutcomeEvaluator
from krystal_kernel.verification.replay_engine import ReplayEngine, ReplayPackage
from krystal_kernel.verification.cross_domain_comparator import (
    CrossDomainComparator,
    DOMAIN_ARCHETYPES,
)
from krystal_kernel.verification.unified_facade import (
    UnifiedVerificationFacade,
    GLOBAL_VERIFICATION_FACADE,
)


class TestStateVectorAndGeometry(unittest.TestCase):
    def test_state_vector_initialization_and_clamping(self):
        sv = StateVector(
            domain_id="game_simulation",
            subject_id="crystal_colony",
            measurement_dimensions=["OBT", "KON", "POS", "DOS"],
            normalized_scores=[6.2, -1.0, 3.5, 4.0],  # Clamped to [0.0, 5.0]
            raw_values=[10, 20, 30, 40],
            units=["score", "pct", "dpm", "rad"],
            weights=[1.0, 1.0, 1.0, 1.0],  # Normalized to 0.25 each
            confidence=[0.9] * 4,
            uncertainty=[0.1] * 4,
            invariants=["INV_VITAL_MAX_HP"],
            configuration_snapshot={},
            runtime_context={"vital_hp": VITAL_MAX_HP}
        )
        self.assertEqual(sv.normalized_scores[0], 5.0)
        self.assertEqual(sv.normalized_scores[1], 0.0)
        self.assertEqual(sv.vital_hp, 6)
        self.assertAlmostEqual(sum(sv.weights), 1.0, places=3)

    def test_compute_geometry_utility_weakest_and_dispersion(self):
        sv = StateVector(
            domain_id="llm_runtime",
            subject_id="crystalsec_openvino",
            measurement_dimensions=["LAT", "TPS", "MEM", "ROB"],
            normalized_scores=[4.0, 3.0, 2.0, 1.0],
            raw_values=[15.0, 50.0, 4.0, 0.9],
            units=["ms", "tps", "gb", "ratio"],
            weights=[0.25, 0.25, 0.25, 0.25],
            confidence=[0.9] * 4,
            uncertainty=[0.1] * 4,
            invariants=[],
            configuration_snapshot={},
            runtime_context={}
        )
        geo = sv.compute_geometry()
        # U = 0.25 * (4 + 3 + 2 + 1) = 2.5
        self.assertAlmostEqual(geo.composite_utility_u, 2.5, places=2)
        # Weakest axis: ROB (1.0)
        self.assertEqual(geo.weakest_axis_code, "ROB")
        self.assertEqual(geo.weakest_axis_score, 1.0)
        # Dispersion D = sqrt( 0.25 * ((4-2.5)^2 + (3-2.5)^2 + (2-2.5)^2 + (1-2.5)^2) )
        # = sqrt( 0.25 * (2.25 + 0.25 + 0.25 + 2.25) ) = sqrt( 0.25 * 5.0 ) = sqrt(1.25) ~= 1.118
        self.assertAlmostEqual(geo.weighted_dispersion_d, math.sqrt(1.25), places=2)
        self.assertTrue(len(geo.radar_polygon_svg) > 0)

    def test_tradeoff_rule_never_judge_by_utility_alone(self):
        """
        Critical Rule: If TPS increases but ROB decreases, it is a TRADEOFF,
        even if composite utility U improves!
        """
        baseline = StateVector(
            domain_id="llm_runtime",
            subject_id="crystalsec_openvino",
            measurement_dimensions=["LAT", "TPS", "MEM", "ROB"],
            normalized_scores=[3.0, 3.0, 3.0, 3.0],
            raw_values=[20.0, 40.0, 4.0, 0.8],
            units=["ms", "tps", "gb", "ratio"],
            weights=[0.25, 0.25, 0.25, 0.25],
            confidence=[0.9] * 4,
            uncertainty=[0.1] * 4,
            invariants=[],
            configuration_snapshot={},
            runtime_context={}
        )
        # Higher TPS (+1.5), but lower ROB (-0.5) -> U goes from 3.0 to 3.25 (+0.25)
        tradeoff_candidate = StateVector(
            domain_id="llm_runtime",
            subject_id="crystalsec_openvino",
            measurement_dimensions=["LAT", "TPS", "MEM", "ROB"],
            normalized_scores=[3.0, 4.5, 3.0, 2.5],
            raw_values=[20.0, 60.0, 4.0, 0.65],
            units=["ms", "tps", "gb", "ratio"],
            weights=[0.25, 0.25, 0.25, 0.25],
            confidence=[0.9] * 4,
            uncertainty=[0.1] * 4,
            invariants=[],
            configuration_snapshot={},
            runtime_context={}
        )
        delta = tradeoff_candidate.compare_with_baseline(baseline)
        self.assertTrue(delta.delta_utility_u > 0, "Composite utility U actually improved")
        self.assertEqual(
            delta.tradeoff_classification,
            TradeoffClassification.TRADEOFF,
            "Must be classified as TRADEOFF, never Pareto strict improvement"
        )
        self.assertIn("TPS", delta.improved_dimensions)
        self.assertIn("ROB", delta.degraded_dimensions)


class TestUniversalInvariantLayer(unittest.TestCase):
    def setUp(self):
        self.invariants = UniversalInvariantLayer(temperature_ceiling_celsius=75.0)

    def test_vital_max_hp_invariant_holds(self):
        ctx = {"vital_hp": 6, "domain_id": "game"}
        passed, violations = self.invariants.verify_all(ctx)
        self.assertTrue(passed)
        self.assertEqual(len(violations), 0)

    def test_vital_max_hp_breach_halts(self):
        ctx = {"vital_hp": 7, "domain_id": "game"}
        passed, violations = self.invariants.verify_all(ctx)
        self.assertFalse(passed)
        self.assertEqual(len(violations), 1)
        self.assertEqual(violations[0].invariant_code, "INV_VITAL_MAX_HP")
        self.assertEqual(violations[0].severity, InvariantSeverity.CRITICAL_SAFETY)
        self.assertTrue(len(violations[0].audit_hash) == 64)

    def test_linux_swap_and_throttling_invariants(self):
        # Swap breach
        ctx_swap = {"vital_hp": 6, "memory_swap_current": 1024, "nr_throttled": 0}
        passed, violations = self.invariants.verify_all(ctx_swap)
        self.assertFalse(passed)
        self.assertEqual(violations[0].invariant_code, "INV_ZERO_SWAP")

        # Throttling breach
        ctx_throt = {"vital_hp": 6, "memory_swap_current": 0, "nr_throttled": 12}
        passed, violations = self.invariants.verify_all(ctx_throt)
        self.assertFalse(passed)
        self.assertEqual(violations[0].invariant_code, "INV_ZERO_THROTTLING")

    def test_temperature_ceiling_breach(self):
        ctx_temp = {"vital_hp": 6, "temperature_celsius": 84.5}
        passed, violations = self.invariants.verify_all(ctx_temp)
        self.assertFalse(passed)
        self.assertEqual(violations[0].invariant_code, "INV_TEMP_CEILING")


class TestPolicyEngineAndSafetyGate(unittest.TestCase):
    def setUp(self):
        self.policy = VerificationPolicy(
            policy_id="POL-TEST",
            version="policy-v2.1.0",
            max_allowed_risk=0.40,
            max_allowed_blast_radius=0.25
        )
        self.engine = PolicyTransformationEngine(active_policy=self.policy)

    def test_safe_action_approval(self):
        sv = StateVector(
            domain_id="game_simulation",
            subject_id="crystal_colony",
            measurement_dimensions=["OBT", "KON", "POS", "DOS"],
            normalized_scores=[3.0, 2.0, 3.5, 4.0],
            raw_values=[3.0, 40.0, 3.5, 30.0],
            units=["score", "pct", "dpm", "rad"],
            weights=[0.25, 0.25, 0.25, 0.25],
            confidence=[0.9] * 4,
            uncertainty=[0.1] * 4,
            invariants=[],
            configuration_snapshot={},
            runtime_context={"vital_hp": VITAL_MAX_HP}
        )
        act = self.engine.evaluate_and_propose_action(sv, {"intensity": 1.0})
        self.assertTrue(act.is_safety_gate_approved)
        self.assertLessEqual(act.risk_score, 0.40)
        self.assertLessEqual(act.blast_radius, 0.25)
        self.assertEqual(act.vital_hp, 6)

    def test_unsafe_action_rejection(self):
        unsafe_action = ProposedAction(
            action_id="ACT-UNSAFE",
            action_type="OVERCLOCK_UNCHECKED",
            domain="cyber_physical_cnc",
            policy_version="policy-v2.1.0",
            controlled_variables={"spindle_rpm": 25000},
            expected_effect="Aggressive machining",
            expected_state_delta={"THG": -1.5},
            risk_score=0.85,  # Exceeds 0.40
            blast_radius=0.60, # Exceeds 0.25
            falsification_condition="temp > 75C"
        )
        approved, reason = self.engine.verify_safety_gate(unsafe_action)
        self.assertFalse(approved)
        self.assertIn("Risk score", reason)


class TestOutcomeEvaluatorAndEvidencePromotion(unittest.TestCase):
    def test_prediction_error_and_side_effects(self):
        reg = EvidenceRegistry()
        evaluator = OutcomeEvaluator(tolerance_epsilon=0.35, evidence_registry=reg)

        claim = reg.register_claim(
            statement="Thermal breathing stabilizes CNC spindle temperature.",
            falsification_condition="thermal_degrade == True",
            initial_evidence_level=EvidenceLevel.INTEGRATION_VERIFIED,
            initial_state=EpistemicState.HYPOTHESIS
        )

        base_state = StateVector(
            domain_id="cyber_physical_cnc",
            subject_id="adaptive_spindle_rpm",
            measurement_dimensions=["THG", "RPM", "CUT", "COL"],
            normalized_scores=[2.5, 3.0, 3.0, 3.0],
            raw_values=[50.0, 2.0, 300.0, 1.0],
            units=["C", "rpm", "N", "mm/s"],
            weights=[0.25, 0.25, 0.25, 0.25],
            confidence=[0.9] * 4,
            uncertainty=[0.1] * 4,
            invariants=[],
            configuration_snapshot={},
            runtime_context={}
        )

        action = ProposedAction(
            action_id="ACT-01",
            action_type="THERMAL_BREATHING",
            domain="cyber_physical_cnc",
            policy_version="policy-v2.1.0",
            controlled_variables={"sampling_rate_hz": 200},
            expected_effect="Improve THG",
            expected_state_delta={"THG": 1.0}, # Expect THG: 2.5 + 1.0 = 3.5
            risk_score=0.2,
            blast_radius=0.1,
            falsification_condition="thermal_degrade"
        )

        obs_state = StateVector(
            domain_id="cyber_physical_cnc",
            subject_id="adaptive_spindle_rpm",
            measurement_dimensions=["THG", "RPM", "CUT", "COL"],
            normalized_scores=[3.45, 3.05, 2.95, 3.0], # Close to predicted
            raw_values=[40.0, 1.8, 305.0, 1.0],
            units=["C", "rpm", "N", "mm/s"],
            weights=[0.25, 0.25, 0.25, 0.25],
            confidence=[0.9] * 4,
            uncertainty=[0.1] * 4,
            invariants=[],
            configuration_snapshot={},
            runtime_context={}
        )

        # Run 5 times to trigger promotion
        for i in range(5):
            report = evaluator.evaluate_outcome(base_state, action, obs_state, associated_claim_id=claim.claim_id)

        self.assertTrue(report.is_prediction_accurate)
        self.assertEqual(len(report.unexpected_side_effects), 0)
        # Verify promotion to SUPPORTED and STATISTICALLY_REPEATED
        updated_claim = reg.claims[claim.claim_id]
        self.assertEqual(updated_claim.epistemic_state, EpistemicState.SUPPORTED)
        self.assertEqual(updated_claim.evidence_level, EvidenceLevel.STATISTICALLY_REPEATED)
        self.assertGreaterEqual(updated_claim.repetitions_count, 5)


class TestReplayEngine(unittest.TestCase):
    def test_deterministic_replay_and_hash(self):
        engine = ReplayEngine(divergence_threshold=0.05)
        pkg = ReplayPackage(
            package_id="PKG-TEST-001",
            domain_id="game_simulation",
            subject_id="crystal_colony",
            seed=1337,
            policy_version="policy-v2.1.0",
            runtime_version="krystal-v2.6",
            telemetry_profile={"hz": 100},
            configuration_snapshot={"seed": 1337},
            initial_state_vector={
                "measurement_dimensions": ["OBT", "KON", "POS", "DOS"],
                "normalized_scores": [3.0, 3.0, 3.0, 3.0]
            },
            recorded_inputs_and_events=[{"type": "TICK", "value": 1.0}],
            expected_outcome_vector={
                "measurement_dimensions": ["OBT", "KON", "POS", "DOS"],
                "normalized_scores": [3.05, 3.05, 3.05, 3.05]
            },
            vital_hp=VITAL_MAX_HP
        )
        self.assertTrue(len(pkg.package_sha256) == 64)

        # Execute replay
        res = engine.execute_replay(pkg)
        self.assertTrue(res.is_deterministic)
        self.assertLessEqual(res.divergence_score, 0.05)
        self.assertTrue(res.invariants_passed)
        self.assertEqual(res.vital_hp, 6)


class TestCrossDomainComparator(unittest.TestCase):
    def test_all_five_domains_normalization_and_comparison(self):
        comp = CrossDomainComparator()
        domains = [
            ("game_simulation", "crystal_colony", [4.2, 88.0, 3.8, 32.0]),
            ("llm_runtime", "crystalsec_openvino", [18.5, 45.2, 5.8, 0.92]),
            ("cyber_physical_cnc", "adaptive_spindle_rpm", [52.4, 2.8, 310.0, 1.45]),
            ("linux_kernel_resource", "cgroup_v2_isolator", [0.8, 45.0, 0.0, 0.0]),
            ("fault_injection_recovery", "core_thermal_shock", [95.0, 12.0, 1.4, 1.0])
        ]
        vectors = []
        for d_id, s_id, raw in domains:
            vec = comp.normalize_domain_metrics(d_id, s_id, raw)
            self.assertEqual(len(vec.normalized_scores), 4)
            for s in vec.normalized_scores:
                self.assertGreaterEqual(s, 0.0)
                self.assertLessEqual(s, 5.0)
            vectors.append(vec)

        summary = comp.compare_multiple_domains(vectors)
        self.assertEqual(len(summary.ranking_by_utility), 5)
        self.assertEqual(len(summary.ranking_by_weakest_axis), 5)
        self.assertEqual(len(summary.ranking_by_dispersion), 5)
        self.assertEqual(summary.vital_hp, 6)


class TestUnifiedVerificationFacade(unittest.TestCase):
    def test_full_eight_phase_epistemic_cycle(self):
        facade = UnifiedVerificationFacade()
        result = facade.run_verification_cycle(
            domain_id="cyber_physical_cnc",
            subject_id="adaptive_spindle_rpm",
            input_signal={"intensity": 1.0, "mode": "STABILIZE"},
            seed=42
        )
        self.assertEqual(result.domain_id, "cyber_physical_cnc")
        self.assertEqual(result.vital_hp, 6)
        self.assertTrue(result.invariant_verification["invariants_passed"])
        self.assertIn("replay_package_id", result.to_dict())
        self.assertTrue(result.cycle_duration_ms > 0)


if __name__ == "__main__":
    unittest.main()
