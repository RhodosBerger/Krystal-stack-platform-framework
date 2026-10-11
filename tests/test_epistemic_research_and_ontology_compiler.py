#!/usr/bin/env python3
"""
Unit and Integration Tests for Krystal-Stack Research Architecture
=============================================================================
Verifies:
  1. Epistemic research contracts (OBSERVED, DERIVED, HYPOTHESIS, FALSIFIED, etc.)
  2. Multi-objective research state vector H (10-dimensional evaluation)
  3. Ontology Compiler operational verification & circular explanation rejection
  4. Mathematical Composition Engine Pareto candidate generation (C1, C2, C3)
  5. Digital Mirror G_desired vs G_observed synchronization & UNKNOWN_DRIFT freeze
  6. Experiment Planner minimum discriminating experiment with 1 independent variable
=============================================================================
"""

import os
import sys
import unittest
import time
import json

WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

from krystal_kernel.scene_constructor_presets import VITAL_MAX_HP
from krystal_kernel.epistemic_research_engine import (
    EpistemicStatus,
    DriftType,
    ResearchStateVectorH,
    EpistemicClaim,
    InterventionProposal
)
from krystal_kernel.ontology_compiler import (
    OntologyCompiler,
    OntologyEntity,
    OntologyRelation,
    CircularDependencyError,
    build_default_krystal_kernel_ontology
)
from krystal_kernel.mathematical_composition_engine import (
    MathematicalCompositionEngine,
    CandidateConfiguration,
    GLOBAL_COMPOSITION_ENGINE
)
from krystal_kernel.digital_mirror_synchronizer import (
    DigitalMirrorSynchronizer,
    GLOBAL_DIGITAL_MIRROR
)
from krystal_kernel.experiment_planner import (
    ExperimentPlanner,
    DiscriminatingExperimentProposal,
    GLOBAL_EXPERIMENT_PLANNER
)
from krystal_kernel.resource_integrity_engine import CompetingHypothesis


class TestEpistemicResearchCore(unittest.TestCase):
    """Verifies epistemic state transitions and 10D multi-objective evaluation."""

    def test_epistemic_claim_lifecycle(self):
        claim = EpistemicClaim(
            claim_id="CLAIM-01",
            subject="Spindle Thermal Lag",
            assertion="Peltier cooling reduces thermal accumulation by 25%",
            status=EpistemicStatus.HYPOTHESIS,
            falsification_condition="If temp delta < 5C over 10s, falsify."
        )

        self.assertEqual(claim.status, EpistemicStatus.HYPOTHESIS)
        claim.promote_to_supported("EXP-101")
        self.assertEqual(claim.status, EpistemicStatus.SUPPORTED)
        self.assertIn("EXP-101", claim.evidence_source)

        # Falsification
        claim.falsify("EXP-102", "Temp delta was only 1.2C")
        self.assertEqual(claim.status, EpistemicStatus.FALSIFIED)

    def test_multi_objective_state_vector_h(self):
        h = ResearchStateVectorH(
            latency_ms=10.5,
            throughput_ops_sec=2200.0,
            memory_pressure=0.25,
            cpu_pressure=0.40,
            io_pressure=0.15,
            queue_pressure=0.10,
            thermal_state_celsius=52.0,
            energy_cost_watts=48.0,
            stability_score=0.99,
            recovery_time_ms=7.0
        )
        cost = h.compute_composite_cost()
        self.assertIsInstance(cost, float)
        self.assertGreater(cost, 0.0)
        self.assertLess(cost, 1.0)

    def test_intervention_proposal_safety_invariant(self):
        proposal = InterventionProposal(
            proposal_id="PROP-01",
            target_component="cgroups.slice",
            proposed_configuration={"cores": [2, 3]},
            predicted_state_delta={"latency_reduction_pct": 20.0},
            supporting_evidence=["EXP-101"],
            falsification_risk=0.05,
            safety_invariant_verified=True,
            vital_hp=VITAL_MAX_HP
        )
        self.assertEqual(proposal.vital_hp, 6)
        self.assertEqual(proposal.policy_approval_status, "PENDING_SAFETY_BOARD")
        self.assertTrue(proposal.safety_invariant_verified)


class TestOntologyCompiler(unittest.TestCase):
    """Verifies ontology generation and rejection of tautological circular explanations."""

    def test_canonical_krystal_kernel_ontology(self):
        compiler = build_default_krystal_kernel_ontology()
        bundle = compiler.compile()

        self.assertIn("ontology.json", bundle)
        self.assertIn("dependency_graph.json", bundle)
        self.assertIn("invariants.json", bundle)
        self.assertIn("unknowns.json", bundle)

        entities = bundle["ontology.json"]["entities"]
        self.assertIn("E_CGROUPS_CPU", entities)
        self.assertIn("E_SPINDLE_THERMAL", entities)
        self.assertTrue(entities["E_CGROUPS_CPU"]["is_operational"])

    def test_rejection_of_unobservable_circular_dependency(self):
        """A circular explanation with zero observable variables must be rejected."""
        compiler = OntologyCompiler()

        # Entity A with no observable variables
        compiler.register_entity(OntologyEntity(
            entity_id="ENT_A",
            name="Unobserved Entity A",
            description="Purely conceptual entity without sensors",
            observable_variables=[],
            control_variables=[],
            invariant="A == B"
        ))

        # Entity B with no observable variables
        compiler.register_entity(OntologyEntity(
            entity_id="ENT_B",
            name="Unobserved Entity B",
            description="Tautological twin without sensors",
            observable_variables=[],
            control_variables=[],
            invariant="B == A"
        ))

        # Create circular cycle A -> B -> A
        compiler.add_relation(OntologyRelation(
            source_entity_id="ENT_A",
            target_entity_id="ENT_B",
            relation_type="CAUSAL",
            mathematical_expression="B = A",
            confidence=0.5,
            measurement_source="NONE",
            sampling_frequency_hz=0.0,
            expected_delay_ms=0.0,
            falsification_test="NONE"
        ))
        compiler.add_relation(OntologyRelation(
            source_entity_id="ENT_B",
            target_entity_id="ENT_A",
            relation_type="CAUSAL",
            mathematical_expression="A = B",
            confidence=0.5,
            measurement_source="NONE",
            sampling_frequency_hz=0.0,
            expected_delay_ms=0.0,
            falsification_test="NONE"
        ))

        with self.assertRaises(CircularDependencyError):
            compiler.compile()


class TestMathematicalCompositionEngine(unittest.TestCase):
    """Verifies Pareto-optimal configuration search across multi-objective trade-offs."""

    def test_pareto_search_generates_three_archetypes(self):
        engine = GLOBAL_COMPOSITION_ENGINE
        candidates = engine.search_pareto_configurations()

        self.assertEqual(len(candidates), 3)
        c1, c2, c3 = candidates

        self.assertEqual(c1.archetype, "C1_DETERMINISTIC_HARDENED")
        self.assertEqual(c2.archetype, "C2_BALANCED_ADAPTIVE")
        self.assertEqual(c3.archetype, "C3_THROUGHPUT_BURST")

        # Invariant checks
        for c in candidates:
            self.assertTrue(c.safety_invariants_satisfied)
            self.assertEqual(c.vital_hp, 6)
            self.assertGreater(len(c.assumptions), 0)
            self.assertIn("context_switches", str(c.expected_telemetry_signature) + str(c.possible_failure_modes))
            self.assertTrue(c.experiment_required_for_validation.startswith("E_"))

        # Trade-off validation: C1 must have lower latency, C3 must have higher throughput
        self.assertLess(c1.predicted_state.latency_ms, c3.predicted_state.latency_ms)
        self.assertGreater(c3.predicted_state.throughput_ops_sec, c1.predicted_state.throughput_ops_sec)


class TestDigitalMirrorSynchronizer(unittest.TestCase):
    """Verifies dual state graphs and freeze policy on UNKNOWN_DRIFT."""

    def setUp(self):
        self.mirror = DigitalMirrorSynchronizer()

    def test_nominal_telemetry_produces_variance_without_freeze(self):
        telemetry = {
            "nr_throttled": 0,
            "temperature_celsius": 52.0,
            "swap_bytes": 0
        }
        diffs = self.mirror.update_observed_state(telemetry)
        self.assertFalse(self.mirror.is_auto_optimization_frozen)

    def test_unknown_drift_triggers_immediate_freeze(self):
        """An unmodeled waveform anomaly must freeze auto-optimization and preserve window."""
        anomaly_telemetry = {
            "nr_throttled": 0,
            "temperature_celsius": 50.0,
            "unmodeled_anomaly_signal": True,
            "anomaly_magnitude": 14.8
        }
        diffs = self.mirror.update_observed_state(anomaly_telemetry)

        self.assertTrue(self.mirror.is_auto_optimization_frozen)
        self.assertIn("Unmodeled phase-shift", self.mirror.freeze_reason)
        self.assertIsNotNone(self.mirror.frozen_telemetry_snapshot)

        # Confirm unfreeze after experiment
        self.mirror.unfreeze_optimization_after_experiment("EXP-TEST", "Sensor calibrated")
        self.assertFalse(self.mirror.is_auto_optimization_frozen)


class TestExperimentPlanner(unittest.TestCase):
    """Verifies minimum discriminating experiment design with strictly 1 independent variable."""

    def test_minimum_discriminating_experiment_design(self):
        planner = GLOBAL_EXPERIMENT_PLANNER
        hypotheses = [
            CompetingHypothesis("H1", "Sensor Drift", "Ground reference offset", {}, "falsify_1", 0.6),
            CompetingHypothesis("H2", "Coil Insulation Breakdown", "Partial discharge", {}, "falsify_2", 0.4)
        ]

        exp = planner.design_minimum_discriminating_experiment(
            anomaly_description="Phase current variance and thermal lag",
            hypotheses=hypotheses,
            current_telemetry={"temperature_c": 54.0, "current_amps": 5.2}
        )

        # Single independent variable rule
        self.assertEqual(exp.independent_variable, "applied_voltage_v")
        self.assertIn("from", exp.independent_variable_step)
        self.assertIn("to", exp.independent_variable_step)

        # Information gain & blast radius
        self.assertGreater(exp.information_gain_bits, 1.0)
        self.assertLessEqual(exp.blast_radius_score, 0.20)  # Contained blast radius
        self.assertIn("H1", exp.falsification_conditions or exp.expected_signatures)

        # Registration with Fault Injection Lab
        lab_res = planner.deploy_experiment_to_lab(exp)
        self.assertEqual(lab_res["status"], "EXPERIMENT_SCHEDULED")


if __name__ == "__main__":
    unittest.main()
