#!/usr/bin/env python3
"""
==============================================================================
UNIT TESTS: KRYSTAL RUNTIME CORE & RUNTIME EXTENSIONS FABRIC
==============================================================================
Tests:
  1. Epistemic Type System & Truth Lattice Operations (meet, join, contradiction)
  2. State / Policy / Evidence Versioning Triple <S_v, P_v, E_v>
  3. SafetyGate + EvidenceGate Dual-Gate Authoritative Evaluation
  4. 8 Runtime Extension Branches (Predictor, Planner, Worker, Telemetry, Policy, HW, Hypervisor, Experiment)
  5. Master KrystalRuntimeCore Orchestration & Architectural Tree
  6. Strict Preservation of Constitutional Invariant VITAL_MAX_HP == 6
==============================================================================
"""

import unittest
import time
from typing import Dict, Any

from krystal_kernel.scene_constructor_presets import VITAL_MAX_HP
from krystal_kernel.verification.semantic_protocol import (
    EpistemicState,
    AuthorityState,
    SafetyState
)
from krystal_kernel.runtime import (
    EpistemicValue,
    EpistemicTruthLattice,
    LatticeCertaintyLevel,
    ConfidenceInterval,
    StatePolicyEvidenceVersion,
    StatePolicyEvidenceVersioningEngine,
    SafetyGate,
    EvidenceGate,
    DualGateEvaluator,
    DualGateVerdict,
    ExtensionCategory,
    RuntimeExtensionRegistry,
    GLOBAL_RUNTIME_EXTENSION_REGISTRY,
    CoreExecutionReport,
    KrystalRuntimeCore,
    GLOBAL_KRYSTAL_RUNTIME_CORE
)


class TestKrystalRuntimeCoreAndExtensions(unittest.TestCase):

    def setUp(self):
        self.assertEqual(VITAL_MAX_HP, 6, "Invariant VITAL_MAX_HP == 6 must hold")

    def test_01_epistemic_type_system_and_lattice(self):
        """Validates EpistemicValue container and algebraic lattice operations."""
        ci1 = ConfidenceInterval(lower_bound=0.80, upper_bound=0.95)
        v1 = EpistemicValue(
            value=42.0,
            epistemic_state=EpistemicState.EPI_OBSERVED,
            belief_degree=0.90,
            confidence_interval=ci1,
            source_probe="probe_a"
        )
        self.assertEqual(v1.vital_max_hp, 6)
        self.assertTrue(v1.is_strongly_evidenced)
        self.assertFalse(v1.is_falsified)

        ci2 = ConfidenceInterval(lower_bound=0.75, upper_bound=0.88)
        v2 = EpistemicValue(
            value=42.2,
            epistemic_state=EpistemicState.EPI_CALIBRATED,
            belief_degree=0.82,
            confidence_interval=ci2,
            source_probe="probe_b"
        )

        # Lattice Meet (∧) - conservative lower bound
        v_meet = EpistemicTruthLattice.meet(v1, v2)
        self.assertAlmostEqual(v_meet.belief_degree, 0.82, places=2)
        self.assertIn("OP_MEET", v_meet.evidence_tokens)

        # Lattice Join (∨) - optimistic corroboration
        v_join = EpistemicTruthLattice.join(v1, v2)
        self.assertGreater(v_join.belief_degree, 0.90)
        self.assertIn("OP_JOIN_CORROBORATED", v_join.evidence_tokens)

        # Contradiction Detection
        v3 = EpistemicValue(
            value=999.0,
            confidence_interval=ConfidenceInterval(lower_bound=990.0, upper_bound=1000.0)
        )
        is_contra, div = EpistemicTruthLattice.detect_contradiction(v1, v3)
        self.assertTrue(is_contra)
        self.assertGreater(div, 900.0)

    def test_02_versioning_triple_evolution(self):
        """Validates S_v, P_v, E_v monotonic evolution, causality, and snapshots."""
        engine = StatePolicyEvidenceVersioningEngine(
            initial_policy_version="POL-2.0.0",
            initial_evidence_root="EV_GENESIS_ROOT"
        )
        v0 = engine.current_version
        self.assertEqual(v0.state_version, 1)
        self.assertEqual(v0.policy_version, "POL-2.0.0")
        self.assertEqual(v0.vital_max_hp, 6)

        # Advance state
        v1 = engine.advance_state({"entity": "SPATIAL_CLUSTER"}, mutation_cause="MUTATION_TEST")
        self.assertEqual(v1.state_version, 2)
        self.assertIn("S:2", v1.vector_string)

        # Update policy
        v2 = engine.update_policy("POL-2.1.0", reason="SECURITY_UPGRADE")
        self.assertEqual(v2.state_version, 3)
        self.assertEqual(v2.policy_version, "POL-2.1.0")

        # Commit evidence batch
        v3 = engine.commit_evidence_batch(["HASH_A1", "HASH_B2"])
        self.assertEqual(v3.state_version, 4)
        self.assertNotEqual(v3.evidence_version, "EV_GENESIS_ROOT")

        # Causal dependency verification
        ok, msg = engine.verify_causal_dependency(v3)
        self.assertTrue(ok)

        # Future horizon anomaly detection
        future_ver = StatePolicyEvidenceVersion(state_version=999, policy_version="POL-2.1.0", evidence_version="ROOT")
        bad_ok, bad_msg = engine.verify_causal_dependency(future_ver)
        self.assertFalse(bad_ok)
        self.assertIn("Causal horizon anomaly", bad_msg)

    def test_03_dual_gate_governance(self):
        """Validates SafetyGate + EvidenceGate -> AuthorityState decisions."""
        dual_gate = DualGateEvaluator()

        ev_good = EpistemicValue(
            value="TEST_ACTION",
            epistemic_state=EpistemicState.EPI_VALIDATED,
            belief_degree=0.92
        )

        # 1. Safe & Evidenced -> AUTH_COMMIT_APPROVED
        res_ok = dual_gate.evaluate(epistemic_val=ev_good, vital_hp=6, thermal_celsius=50.0)
        self.assertEqual(res_ok.authority_state, AuthorityState.AUTH_COMMIT_APPROVED)
        self.assertTrue(res_ok.can_actuate)

        # 2. Constitutional Invariant Breach -> AUTH_VETOED
        res_veto = dual_gate.evaluate(epistemic_val=ev_good, vital_hp=5)
        self.assertEqual(res_veto.authority_state, AuthorityState.AUTH_VETOED)
        self.assertFalse(res_veto.can_actuate)

        # 3. Thermal Overheat -> AUTH_VETOED
        res_thermal = dual_gate.evaluate(epistemic_val=ev_good, vital_hp=6, thermal_celsius=92.0)
        self.assertEqual(res_thermal.authority_state, AuthorityState.AUTH_VETOED)

        # 4. Observer Overhead Breach -> AUTH_VETOED
        res_overhead = dual_gate.evaluate(epistemic_val=ev_good, vital_hp=6, observer_overhead_pct=3.8)
        self.assertEqual(res_overhead.authority_state, AuthorityState.AUTH_VETOED)

        # 5. Low belief degree -> AUTH_DEFERRED (Micro-probe trigger)
        ev_weak = EpistemicValue(value="ACTION", belief_degree=0.45)
        res_defer = dual_gate.evaluate(epistemic_val=ev_weak, vital_hp=6)
        self.assertEqual(res_defer.authority_state, AuthorityState.AUTH_DEFERRED)

        # 6. Quarantined stream -> AUTH_QUARANTINED
        ev_quar = EpistemicValue(value="ACTION", epistemic_state=EpistemicState.EPI_QUARANTINED)
        res_quar = dual_gate.evaluate(epistemic_val=ev_quar, vital_hp=6)
        self.assertEqual(res_quar.authority_state, AuthorityState.AUTH_QUARANTINED)

    def test_04_runtime_extensions_registry_all_categories(self):
        """Validates all 8 Runtime Extension branches in the registry."""
        registry = RuntimeExtensionRegistry()
        tree = registry.get_extension_tree_summary()

        expected_categories = [
            "PREDICTOR", "PLANNER", "WORKER_STRATEGY", "TELEMETRY_PROBE",
            "POLICY", "HARDWARE_ADAPTER", "HYPERVISOR_ADAPTER", "EXPERIMENT_MODULE"
        ]
        for cat in expected_categories:
            self.assertIn(cat, tree)
            self.assertGreaterEqual(tree[cat]["total_available"], 1, f"Category {cat} must have at least 1 plugin")
            self.assertIsNotNone(tree[cat]["active_plugin_id"], f"Category {cat} must have an active plugin")

        # Test active plugin switching
        self.assertTrue(registry.set_active_plugin(ExtensionCategory.PLANNER, "PLANNER_CONSERVATIVE_THERMAL"))
        active_plan = registry.get_active_plugin(ExtensionCategory.PLANNER)
        self.assertEqual(active_plan.plugin_id, "PLANNER_CONSERVATIVE_THERMAL")

    def test_05_master_runtime_core_execution(self):
        """Validates end-to-end execution of a workload via KrystalRuntimeCore."""
        core = KrystalRuntimeCore()
        tree = core.get_runtime_architecture_tree()
        self.assertEqual(tree["root"], "Krystal Runtime Core")
        self.assertEqual(tree["vital_max_hp"], 6)

        # Execute workload
        report = core.execute_workload(
            workload_id="WL_CORE_001",
            workload_category="COMPUTE_BOUND_CONVOLUTION",
            target_deadline_us=40000.0,
            thermal_celsius=54.0,
            power_watts=19.0,
            observer_overhead_pct=1.4
        )
        self.assertEqual(report.vital_max_hp, 6)
        self.assertEqual(report.dual_gate_verdict["authority_state"], "AUTH_COMMIT_APPROVED")
        self.assertIn("S:", report.version["vector_string"])
        self.assertGreater(report.cycle_duration_us, 0.0)
        self.assertGreaterEqual(len(report.probes_sampled), 1)

    def test_06_experiment_module_ab_run(self):
        """Validates experiment execution and hypothesis feedback into Hypothesis Ledger."""
        core = KrystalRuntimeCore()
        exp_res = core.run_experiment({"trials": 6, "experiment_id": "EXP_TEST_RUN"})
        self.assertEqual(exp_res.vital_max_hp, 6)
        self.assertEqual(exp_res.trials_run, 6)
        self.assertGreater(exp_res.mean_baseline_latency, 0.0)
        self.assertGreater(exp_res.mean_perturbed_latency, 0.0)


if __name__ == "__main__":
    unittest.main()
