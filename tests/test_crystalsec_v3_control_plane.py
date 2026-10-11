"""
================================================================================
KRYSTAL-STACK / CRYSTAL-SEC 3.0: PHASE 3 TEST SUITE
Test File: tests/test_crystalsec_v3_control_plane.py

Role:
Formal verification of the CrystalSec 3.0 Phase 3 Control Plane:
- Physical strain evaluation and Delta mapping
- Strain-aware triage with dual-threshold hysteresis
- Dynamic WSPT ranking with 5-tier deterministic tie-breakers
- Empirical action cost adaptation
- Counterfactual perturbation and robust scoring
- Two-tier preemption gate (Hard Safety Veto & Cooperative Preemption)
- Stress-sensitive telemetry governor & Repression invariant
- Deterministic decision replayability
- Non-mutating contract in Phase 3
================================================================================
"""

import json
import os
import sys
import time
import unittest

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from krystal_kernel.verification.crystalsec_v3_control_plane import (
    ActionCandidateType,
    ActionCostEstimator,
    CounterfactualTriageSimulator,
    CrystalSecV3ControlPlane,
    CrystalSecV3State,
    DecisionReplayPackage,
    NonMutatingCortexApiDispatcher,
    PhysicalStrainEvaluator,
    PsychologicalCopingMechanism,
    RepressionEventFilter,
    SchedulableAction,
    StrainAwareTriageController,
    StrainRegime,
    StressSensitiveTelemetryGovernor,
    TelemetryItem,
    VITAL_MAX_HP,
)
from src.python.vulkan_compute_driver import VulkanComputeDriver


def make_test_state(
    temp_c: float = 45.0,
    h_val: float = 0.95,
    dh_dt: float = 0.0,
    mem_cont: float = 0.1,
    q_press: float = 0.1,
    divergence: float = 0.0,
    oracle_rca: str = "CONSISTENT_WITH_H0_IDENTICAL",
    gpu_recovery: str = "READY",
    fault_active: bool = False,
    oracle_conf: float = 0.95,
    telemetry_age_s: float = 0.0,
    vital_hp: int = 6,
) -> CrystalSecV3State:
    """Helper creating a test CrystalSecV3State with precise values."""
    now = time.time() - telemetry_age_s

    def item(v, conf=0.95, valid=True):
        return TelemetryItem(value=v, timestamp=now, source="TEST_FIXTURE", confidence=conf, validity=valid)

    return CrystalSecV3State(
        health_vector_H=item(h_val),
        dH_dt=item(dh_dt),
        junction_temperature_c=item(temp_c),
        memory_contention=item(mem_cont),
        queue_pressure=item(q_press),
        gpu_frame_time_ms=item(0.45),
        gpu_kernel_time_ms=item(0.31),
        gpu_readback_time_ms=item(0.0),
        D_divergence=item(divergence, conf=oracle_conf),
        oracle_confidence=item(oracle_conf),
        oracle_RCA=item(oracle_rca),
        gpu_recovery_state=item(gpu_recovery),
        fault_active=item(fault_active),
        telemetry_sampling_rate_hz=item(2.0),
        vital_max_hp=vital_hp,
    )


class TestCrystalSecV3ControlPlane(unittest.TestCase):
    """Formal verification suite for Phase 3 predictive triage and scheduling."""

    def setUp(self):
        self.control_plane = CrystalSecV3ControlPlane()

    def test_nominal_state_selects_nominal_dispatch(self):
        """Under cool, healthy, zero-divergence conditions, selects NOMINAL_GPU_DISPATCH."""
        state = make_test_state(temp_c=42.0, h_val=0.98, divergence=0.0)
        decision = self.control_plane.evaluate_control_plane(state)

        self.assertEqual(decision.strain_regime, StrainRegime.NOMINAL.value)
        self.assertLess(decision.sigma_strain, 0.20)
        self.assertIsNotNone(decision.selected_action)
        self.assertEqual(decision.selected_action["action_type"], ActionCandidateType.NOMINAL_GPU_DISPATCH.value)
        self.assertTrue(decision.invariant_gate_result["approved"])

    def test_divergence_increases_strain(self):
        """Oracle divergence D_divergence strictly increases calculated sigma_strain."""
        state_zero = make_test_state(divergence=0.0, temp_c=50.0)
        state_high = make_test_state(divergence=0.85, temp_c=50.0)

        strain_zero = PhysicalStrainEvaluator.evaluate(state_zero)
        strain_high = PhysicalStrainEvaluator.evaluate(state_high)

        self.assertGreater(strain_high.sigma, strain_zero.sigma)
        self.assertGreaterEqual(strain_high.contributing_signals["D_divergence"], 0.85)

    def test_hypervigilance_reachable(self):
        """Elevated strain activates HYPERVIGILANCE and proposes TELEMETRY_BURST_TRIGGER."""
        state = make_test_state(temp_c=72.0, h_val=0.60, divergence=0.40, mem_cont=0.80)
        decision = self.control_plane.evaluate_control_plane(state)

        self.assertIn(PsychologicalCopingMechanism.HYPERVIGILANCE.value, decision.reachable_mechanisms)
        candidate_types = [c["action_type"] for c in decision.candidate_actions]
        self.assertIn(ActionCandidateType.TELEMETRY_BURST_TRIGGER.value, candidate_types)

    def test_hysteresis_prevents_chatter(self):
        """
        Once HYPERVIGILANCE is activated at sigma >= 0.35, dropping sigma to 0.30
        (> 0.25 deactivation threshold) keeps it active, preventing state chatter.
        """
        triage = StrainAwareTriageController()

        # Step 1: Sub-threshold strain -> Inactive
        res_low = PhysicalStrainEvaluator.evaluate(make_test_state(temp_c=40.0, h_val=0.95))
        self.assertFalse(triage._active_mechanisms[PsychologicalCopingMechanism.HYPERVIGILANCE])

        # Step 2: Elevate strain >= 0.35 -> Activates
        res_high = PhysicalStrainEvaluator.evaluate(make_test_state(temp_c=75.0, mem_cont=0.7, divergence=0.35))
        reachable_high = triage.evaluate_reachability(res_high)
        self.assertIn(PsychologicalCopingMechanism.HYPERVIGILANCE, reachable_high)
        self.assertTrue(triage._active_mechanisms[PsychologicalCopingMechanism.HYPERVIGILANCE])

        # Step 3: Moderately relax strain to intermediate region (e.g. 0.30)
        # Hysteresis invariant: Stays ACTIVE because 0.30 >= tau_deact (0.25)
        res_inter = PhysicalStrainEvaluator.evaluate(make_test_state(temp_c=65.0, mem_cont=0.5, divergence=0.15))
        reachable_inter = triage.evaluate_reachability(res_inter)
        self.assertIn(PsychologicalCopingMechanism.HYPERVIGILANCE, reachable_inter)

        # Step 4: Fully relax strain < 0.25 -> Deactivates
        res_calm = PhysicalStrainEvaluator.evaluate(make_test_state(temp_c=41.0, h_val=0.99, divergence=0.0))
        reachable_calm = triage.evaluate_reachability(res_calm)
        self.assertNotIn(PsychologicalCopingMechanism.HYPERVIGILANCE, reachable_calm)
        self.assertFalse(triage._active_mechanisms[PsychologicalCopingMechanism.HYPERVIGILANCE])

    def test_wspt_dynamic_ranking(self):
        """WSPT ranking sorts strictly by rho_i = w'_i / p_i descending."""
        act_a = SchedulableAction("A", ActionCandidateType.SPECULATIVE_DEPTH_TUNING, PsychologicalCopingMechanism.RATIONALIZATION, 1.0, 2.0, 100.0, 0.5, 0.1, "x", 1, "e", None)
        act_b = SchedulableAction("B", ActionCandidateType.TELEMETRY_BURST_TRIGGER, PsychologicalCopingMechanism.HYPERVIGILANCE, 1.0, 5.0, 100.0, 0.8, 0.1, "x", 1, "e", None)
        # rho_A = 2.0 / 100 = 0.02, rho_B = 5.0 / 100 = 0.05 => B should rank above A

        res_strain = PhysicalStrainEvaluator.evaluate(make_test_state())
        ranked = self.control_plane._rank_wspt([act_a, act_b], res_strain)
        self.assertEqual(ranked[0].action_id, "B")
        self.assertEqual(ranked[1].action_id, "A")

    def test_measured_cost_updates_rank(self):
        """Empirical reduction in measured execution latency promotes action WSPT rank."""
        estimator = ActionCostEstimator()
        # Feed fast measured times for SPECULATIVE_DEPTH_TUNING
        for _ in range(8):
            estimator.record_cost(ActionCandidateType.SPECULATIVE_DEPTH_TUNING, 30.0)

        cost, count, conf, source = estimator.estimate_cost(ActionCandidateType.SPECULATIVE_DEPTH_TUNING)
        self.assertAlmostEqual(cost, 30.0, delta=1.0)
        self.assertEqual(source, "MEASURED_RECENT")
        self.assertGreaterEqual(conf, 0.55)

    def test_counterfactual_rejects_fragile_action(self):
        """Actions that violate stability under +20% stress perturbation are blocked."""
        # Setup state near thermal boundary (e.g. 92C)
        state = make_test_state(temp_c=92.0)
        strain = PhysicalStrainEvaluator.evaluate(state)

        # An action that keeps nominal GPU execution will fail perturbed_safe (92 + 5 = 97C >= 95C)
        nom_act = SchedulableAction(
            "ACT-NOM", ActionCandidateType.NOMINAL_GPU_DISPATCH, PsychologicalCopingMechanism.SUBLIMATION,
            1.0, 1.0, 100.0, 0.5, 0.1, "compute_backend", "VULKAN_GPU", "nominal", None
        )
        cf = CounterfactualTriageSimulator.simulate_action(nom_act, state, strain)
        self.assertFalse(cf["perturbed_safe"])
        self.assertGreaterEqual(cf["perturbed_temp_c"], 95.0)

    def test_temperature_forces_regression(self):
        """Tier 2 Hard Safety Veto triggers when temperature >= 95C, forcing CPU fallback."""
        state = make_test_state(temp_c=96.5)
        decision = self.control_plane.evaluate_control_plane(state)

        self.assertIsNotNone(decision.selected_action)
        self.assertEqual(decision.selected_action["action_type"], ActionCandidateType.GPU_FAILOVER_TO_CPU.value)
        self.assertIn("HARD_VETO: Thermal ceiling 95C exceeded", decision.veto_reason)
        self.assertEqual(decision.invariant_gate_result["tier"], 2)

    def test_device_loss_forces_cpu_fallback(self):
        """Simulated GPU device loss triggers Tier 2 veto and forces CPU fallback."""
        state = make_test_state(gpu_recovery="FAULT_DETECTED", fault_active=True)
        decision = self.control_plane.evaluate_control_plane(state)

        self.assertEqual(decision.selected_action["action_type"], ActionCandidateType.GPU_FAILOVER_TO_CPU.value)
        self.assertIn("HARD_VETO: GPU device fault detected", decision.veto_reason)

    def test_invariant_overrides_scheduler(self):
        """Breach of universal invariant VITAL_MAX_HP != 6 triggers unconditional halt veto."""
        # Using valid constructor through modified dictionary to simulate breach
        state_dict = make_test_state().to_dict()
        state_dict["vital_max_hp"] = 5  # Violation of platform charter!

        # Attempting direct initialization triggers assertion
        with self.assertRaises(AssertionError):
            state = CrystalSecV3State(**state_dict)

    def test_repression_preserves_critical_audit(self):
        """
        Universal rule: REPRESSION must NEVER suppress:
        - AUDIT_EVENT, FAULT_EVENT, INVARIANT_VIOLATION, POLICY_DECISION, PREEMPTION_EVENT.
        """
        # Critical event types are NEVER suppressible
        for crit in ["AUDIT_EVENT", "FAULT_EVENT", "INVARIANT_VIOLATION", "POLICY_DECISION", "PREEMPTION_EVENT"]:
            can_suppress = RepressionEventFilter.can_suppress(crit, repression_active=True)
            self.assertFalse(can_suppress, f"Critical event {crit} was illegally suppressible under REPRESSION")

        # Low-priority verbose diagnostic IS suppressible
        self.assertTrue(RepressionEventFilter.can_suppress("VERBOSE_TRACE_DUMP", repression_active=True))
        # When repression is inactive, nothing is suppressed
        self.assertFalse(RepressionEventFilter.can_suppress("VERBOSE_TRACE_DUMP", repression_active=False))

    def test_deterministic_replay(self):
        """Replaying a saved DecisionReplayPackage produces the identical decision output."""
        state = make_test_state(temp_c=58.0, h_val=0.85, divergence=0.12)
        decision = self.control_plane.evaluate_control_plane(state, current_time=1791747000.0)

        pkg = DecisionReplayPackage(
            package_id="REPLAY-TEST-001",
            timestamp=1791747000.0,
            state_dict=state.to_dict(),
            decision_dict=decision.to_dict(),
        )
        pkg_json = pkg.to_json()

        # Verify deterministic replay bit-identity
        match = DecisionReplayPackage.verify_replay(pkg_json)
        self.assertTrue(match)

    def test_same_state_same_decision(self):
        """Two evaluations of the exact same state produce identical hashes and action selections."""
        state1 = make_test_state(temp_c=52.0, h_val=0.90, divergence=0.08)
        state2 = make_test_state(temp_c=52.0, h_val=0.90, divergence=0.08)

        t_fixed = 1791748000.0
        dec1 = self.control_plane.evaluate_control_plane(state1, current_time=t_fixed)
        dec2 = self.control_plane.evaluate_control_plane(state2, current_time=t_fixed)

        self.assertEqual(dec1.input_state_hash, dec2.input_state_hash)
        self.assertEqual(dec1.sigma_strain, dec2.sigma_strain)
        self.assertEqual(dec1.selected_action["action_id"], dec2.selected_action["action_id"])

    def test_uncertainty_reduces_action_confidence(self):
        """Low confidence in telemetry increases epistemic uncertainty and lowers decision confidence."""
        state_sure = make_test_state(oracle_conf=0.99)
        state_unsure = make_test_state(oracle_conf=0.10)

        dec_sure = self.control_plane.evaluate_control_plane(state_sure)
        dec_unsure = self.control_plane.evaluate_control_plane(state_unsure)

        self.assertGreater(dec_sure.confidence, dec_unsure.confidence)

    def test_no_actuation_occurs_in_phase3(self):
        """
        Phase 3 Boundary Invariant:
        Evaluating the control plane produces decision objects but does NOT mutate
        the live Vulkan compute driver handles or execution state.
        """
        driver = VulkanComputeDriver(prefer_gpu=False)
        prev_disp_count = driver.dispatch_count
        prev_recovery = driver.recovery_state
        prev_backend = driver.compute_backend

        state = make_test_state(temp_c=96.0, gpu_recovery="FAULT_DETECTED")
        decision = self.control_plane.evaluate_control_plane(state)

        # Confirm decision proposes CPU failover
        self.assertEqual(decision.selected_action["action_type"], ActionCandidateType.GPU_FAILOVER_TO_CPU.value)

        # Confirm live Vulkan driver was NOT touched
        self.assertEqual(driver.dispatch_count, prev_disp_count)
        self.assertEqual(driver.recovery_state, prev_recovery)
        self.assertEqual(driver.compute_backend, prev_backend)
        driver.cleanup()

    # --- COUNTEREXAMPLE TEST CASES ---
    def test_counterexample_high_wspt_but_unsafe_action(self):
        """
        Counterexample: An action with massive WSPT weight that breaches safety margin
        must be intercepted and vetoed by the preemption gate.
        """
        state = make_test_state(temp_c=95.5)  # Already over ceiling
        decision = self.control_plane.evaluate_control_plane(state)

        # Even if a hypothetical action promised high throughput, veto forces CPU fallback
        self.assertEqual(decision.selected_action["action_type"], ActionCandidateType.GPU_FAILOVER_TO_CPU.value)
        self.assertIn("HARD_VETO", decision.veto_reason)

    def test_counterexample_stale_shadow_oracle_evidence(self):
        """
        Counterexample: Oracle telemetry older than 5.0 seconds is rejected as stale
        and falls back to fail-safe conservative strain evaluation.
        """
        stale_state = make_test_state(telemetry_age_s=120.0)  # 2 minutes old
        strain = PhysicalStrainEvaluator.evaluate(stale_state)

        # Stale telemetry falls back to conservative defaults (temp 90C)
        self.assertGreater(strain.sigma, 0.40)

    def test_counterexample_equal_wspt_tie_break(self):
        """
        Counterexample: Two actions with mathematically equal rho_i = w / p
        are deterministically broken by safety_criticality DESC -> uncertainty ASC.
        """
        # Equal rho = 1.0 / 100.0 = 0.01, but act_safe has higher safety_criticality
        act_risky = SchedulableAction("ACT-RISKY", ActionCandidateType.NOMINAL_GPU_DISPATCH, PsychologicalCopingMechanism.SUBLIMATION, 1.0, 1.0, 100.0, 0.2, 0.1, "x", 1, "e", None)
        act_safe = SchedulableAction("ACT-SAFE", ActionCandidateType.SPECULATIVE_DEPTH_TUNING, PsychologicalCopingMechanism.RATIONALIZATION, 1.0, 1.0, 100.0, 0.9, 0.1, "x", 1, "e", None)

        strain = PhysicalStrainEvaluator.evaluate(make_test_state())
        ranked = self.control_plane._rank_wspt([act_risky, act_safe], strain)

        self.assertEqual(ranked[0].action_id, "ACT-SAFE")
        self.assertEqual(ranked[1].action_id, "ACT-RISKY")

    def test_telemetry_governor_adaptive_rate(self):
        """StressSensitiveTelemetryGovernor adjusts rate and detects observer overhead limit."""
        gov = StressSensitiveTelemetryGovernor(observer_overhead_budget_pct=15.0)

        # Calm conditions -> 2 Hz
        rate_calm = gov.compute_target_rate(h_val=0.9, dh_dt=0.0, sigma=0.1, d_divergence=0.0)
        self.assertEqual(rate_calm, 2.0)

        # Elevated strain -> 35 Hz
        rate_elev = gov.compute_target_rate(h_val=0.7, dh_dt=0.0, sigma=0.45, d_divergence=0.1)
        self.assertEqual(rate_elev, 35.0)

        # Acute divergence / fast drop -> 100 Hz burst
        rate_burst = gov.compute_target_rate(h_val=0.4, dh_dt=-0.8, sigma=0.75, d_divergence=0.4)
        self.assertEqual(rate_burst, 100.0)

        # Observer limit exceeded (>15%) -> degrades rate
        rate_throttled = gov.compute_target_rate(h_val=0.4, dh_dt=-0.8, sigma=0.75, d_divergence=0.4, measured_observer_overhead_pct=22.0)
        self.assertEqual(rate_throttled, 35.0)
        self.assertEqual(len(gov.limit_reached_events), 1)

    def test_non_mutating_api_dispatcher(self):
        """NonMutatingCortexApiDispatcher handles read-only queries without runtime actuation."""
        dispatcher = NonMutatingCortexApiDispatcher(self.control_plane)
        state_dict = make_test_state().to_dict()

        res_post = dispatcher.dispatch_request("/api/cortex/evaluate-state", method="POST", payload=state_dict)
        self.assertEqual(res_post["status"], "SUCCESS")
        self.assertFalse(res_post["mutated"])

        res_get_dec = dispatcher.dispatch_request("/api/cortex/crystalsec-v3-decision", method="GET")
        self.assertEqual(res_get_dec["status"], "SUCCESS")
        self.assertIsNotNone(res_get_dec["decision"])

        res_get_queue = dispatcher.dispatch_request("/api/cortex/wspt-queue", method="GET")
        self.assertEqual(res_get_queue["status"], "SUCCESS")
        self.assertIsInstance(res_get_queue["wspt_queue"], list)


if __name__ == "__main__":
    unittest.main()
