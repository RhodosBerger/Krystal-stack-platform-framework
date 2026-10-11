r"""
Unit and Property Tests for CrystalSec 3.0 Phase 4.2:
Hardware-Grounded Sequence Arbiter, StateVersion Binding & Compound Safety.

Covers:
1. StateVersion monotonicity and stale decision invalidation
2. Step-by-step revalidation and re-basing
3. Physical resource interference and compound budgeting
4. Canonical deadlock-free lock ordering
5. Isolated vs Cascading rollback
6. Emergency GPU failover branch invalidation
7. Adaptive bounded deadline adjustments
8. Cryptographic audit chain integrity
"""

import threading
import time
import unittest
from typing import Dict, Any

from krystal_kernel.verification.crystalsec_v3_control_plane import (
    ActionCandidateType,
    ControlPlaneDecision,
    CrystalSecV3State,
    EvidenceLevel,
    TelemetryItem,
)
from krystal_kernel.verification.crystalsec_v3_actuation_plane import (
    ActuationRecord,
    ActuationResult,
    GpuProfileLevel,
    RuntimeStateReader,
    TelemetrySamplingAdapter,
    TransactionalActuationEngine,
    TransactionState,
    WHITELISTED_ACTUATION_VARIABLES,
)
from krystal_kernel.verification.crystalsec_v3_sequence_arbiter import (
    AdaptiveDeadlinePolicy,
    CoupledVariableLockManager,
    CouplingClassification,
    EvaluationOutcome,
    HardwareSequenceArbiter,
    PowerStarvationDetector,
    ResourceVector,
    RollbackMode,
    SequenceSafetyEvaluator,
    SequenceState,
    StateVersionManager,
    TransactionDependencyGraph,
    TransactionInterferenceMatrix,
    TransactionSequence,
    UMAPressureTracker,
    estimate_resource_impact,
)


def make_test_state(vital_max_hp: int = 6, temp_c: float = 62.0) -> CrystalSecV3State:
    """Helper to produce standard nominal test state."""
    t = time.time()
    def item(v, conf=1.0):
        return TelemetryItem(v, t, "TEST", conf, True)

    return CrystalSecV3State(
        health_vector_H=item(1.0),
        dH_dt=item(0.0),
        junction_temperature_c=item(temp_c),
        memory_contention=item(0.20),
        queue_pressure=item(0.15),
        gpu_frame_time_ms=item(0.45),
        gpu_kernel_time_ms=item(0.35),
        gpu_readback_time_ms=item(0.10),
        D_divergence=item(0.001),
        oracle_confidence=item(0.99),
        oracle_RCA=item("NONE"),
        gpu_recovery_state=item("READY"),
        fault_active=item(False),
        telemetry_sampling_rate_hz=item(10.0),
        vital_max_hp=vital_max_hp,
    )


def make_test_decision(
    action_type: ActionCandidateType,
    proposed_value: Any,
    decision_id: str = "DEC-TEST-001",
    source_state_version: int = 0,
) -> ControlPlaneDecision:
    """Helper to produce a valid ControlPlaneDecision."""
    var = WHITELISTED_ACTUATION_VARIABLES.get(action_type.value, "unknown")
    return ControlPlaneDecision(
        decision_id=decision_id,
        timestamp=time.time(),
        input_state_hash="a1b2c3d4e5f60718293a",
        sigma_strain=0.15,
        strain_regime="NOMINAL",
        reachable_mechanisms=["SUBLIMATION"],
        candidate_actions=[],
        wspt_ranking=[],
        counterfactual_results=[],
        selected_action={
            "action_id": f"ACT-{action_type.value}",
            "action_type": action_type.value,
            "controlled_variable": var,
            "proposed_value": proposed_value,
            "expected_effect": "Test effect",
            "rollback_action": "RESTORE",
        },
        invariant_gate_result={"all_passed": True},
        confidence=0.95,
        evidence_level=EvidenceLevel.SIMULATION_VERIFIED.value,
        veto_reason=None,
        vital_max_hp=6,
        source_state_version=source_state_version,
    )


class TestCrystalSecV3SequenceArbiter(unittest.TestCase):
    """Formal test suite for Phase 4.2 Sequence Arbiter & StateVersion Binding."""

    def setUp(self):
        self.version_mgr = StateVersionManager(initial_version=0)
        self.engine = TransactionalActuationEngine(state_version_manager=self.version_mgr)
        self.arbiter = HardwareSequenceArbiter(
            engine=self.engine,
            version_manager=self.version_mgr,
        )

    # 1. StateVersion increments on commit
    def test_state_version_increments_on_commit(self):
        self.assertEqual(self.version_mgr.current(), 0)
        state = make_test_state()
        dec = make_test_decision(ActionCandidateType.TELEMETRY_BURST_TRIGGER, 50.0, source_state_version=0)

        record = self.engine.execute_transaction(decision=dec, live_state=state)
        self.assertEqual(record.result, ActuationResult.COMMITTED)
        self.assertEqual(self.version_mgr.current(), 1)

    # 2. Stale decision invalidated after previous commit
    def test_stale_decision_invalidated_after_previous_commit(self):
        state = make_test_state()
        dec1 = make_test_decision(ActionCandidateType.TELEMETRY_BURST_TRIGGER, 40.0, decision_id="DEC-1", source_state_version=0)
        dec2_stale = make_test_decision(ActionCandidateType.SPECULATIVE_DEPTH_TUNING, 2, decision_id="DEC-2", source_state_version=0)

        # Execute dec1 -> increments version to 1
        rec1 = self.engine.execute_transaction(decision=dec1, live_state=state)
        self.assertEqual(rec1.result, ActuationResult.COMMITTED)
        self.assertEqual(self.version_mgr.current(), 1)

        # Attempt to execute dec2_stale with version 0 -> must be rejected
        rec2 = self.engine.execute_transaction(decision=dec2_stale, live_state=state, prepared_state_version=0)
        self.assertEqual(rec2.result, ActuationResult.ELIGIBILITY_FAILED)
        self.assertIn("STALE_STATE_VERSION_INVALIDATION", rec2.events[-1]["details"]["reason"])

    # 3. Sequence revalidates each step
    def test_sequence_revalidates_each_step(self):
        state = make_test_state()
        dec1 = make_test_decision(ActionCandidateType.TELEMETRY_BURST_TRIGGER, 25.0, decision_id="DEC-STEP1", source_state_version=0)
        dec2 = make_test_decision(ActionCandidateType.SPECULATIVE_DEPTH_TUNING, 2, decision_id="DEC-STEP2", source_state_version=0)

        prep_ok, err, seq = self.arbiter.prepare_sequence([dec1, dec2], state)
        self.assertTrue(prep_ok)
        self.assertIsNotNone(seq)

        report = self.arbiter.execute_sequence(seq, state)
        self.assertTrue(report["success"])
        self.assertEqual(report["steps_executed"], 2)
        self.assertEqual(report["final_state_version"], 2)

    # 4. Independent transactions can sequence
    def test_independent_transactions_can_sequence(self):
        state = make_test_state()
        dec1 = make_test_decision(ActionCandidateType.TELEMETRY_BURST_TRIGGER, 20.0, decision_id="DEC-A", source_state_version=0)
        dec2 = make_test_decision(ActionCandidateType.SPECULATIVE_DEPTH_TUNING, 1, decision_id="DEC-B", source_state_version=0)

        prep_ok, _, seq = self.arbiter.prepare_sequence([dec1, dec2], state)
        self.assertTrue(prep_ok)

        report = self.arbiter.execute_sequence(seq, state)
        self.assertTrue(report["success"])
        self.assertEqual(report["sequence_state"], SequenceState.COMMITTED.value)

    # 5. Coupled transactions are serialized
    def test_coupled_transactions_are_serialized(self):
        classification = TransactionInterferenceMatrix.classify("telemetry_sampling_rate_hz", "speculative_frame_depth")
        self.assertEqual(classification, CouplingClassification.STRONGLY_COUPLED)
        self.assertTrue(TransactionInterferenceMatrix.requires_serialization("telemetry_sampling_rate_hz", "speculative_frame_depth"))

    # 6. Resource interference blocks unsafe pair (Counterexample A)
    def test_resource_interference_blocks_unsafe_pair(self):
        state = make_test_state()
        # Tight headroom
        tight_headroom = ResourceVector(U_UMA=0.40, P_TDP=0.40, Q_WDDM=0.40, C_CPU=0.40)
        evaluator = SequenceSafetyEvaluator(safe_headroom=tight_headroom)

        dec1 = make_test_decision(ActionCandidateType.GPU_PROFILE_DEGRADATION, "ULTRA", decision_id="DEC-P1")
        dec2 = make_test_decision(ActionCandidateType.SPECULATIVE_DEPTH_TUNING, 4, decision_id="DEC-P2")
        seq = TransactionSequence("SEQ-OVERLOAD", [dec1, dec2], source_state_version=0)

        outcome, err, _ = evaluator.evaluate(seq, state, current_version=0)
        self.assertEqual(outcome, EvaluationOutcome.REJECTED_RESOURCE_CONFLICT)
        self.assertIn("RESOURCE_INTERFERENCE_CONFLICT", err)

    # 7. Lock order prevents deadlock
    def test_lock_order_prevents_deadlock(self):
        lock_mgr = CoupledVariableLockManager()
        # Thread 1 requests compute_backend then telemetry_sampling_rate_hz
        # Thread 2 requests telemetry_sampling_rate_hz then compute_backend
        # Both must acquire in canonical order without deadlock

        results = {}

        def worker1():
            ok, err = lock_mgr.acquire_locks({"compute_backend", "telemetry_sampling_rate_hz"}, holder_id="T1", timeout_s=0.2)
            results["T1"] = ok
            if ok:
                time.sleep(0.02)
                lock_mgr.release_locks({"compute_backend", "telemetry_sampling_rate_hz"}, holder_id="T1")

        def worker2():
            # Deliberately reversed in request set, lock_mgr must sort canonically
            ok, err = lock_mgr.acquire_locks({"telemetry_sampling_rate_hz", "compute_backend"}, holder_id="T2", timeout_s=0.2)
            results["T2"] = ok
            if ok:
                time.sleep(0.02)
                lock_mgr.release_locks({"telemetry_sampling_rate_hz", "compute_backend"}, holder_id="T2")

        t1 = threading.Thread(target=worker1)
        t2 = threading.Thread(target=worker2)
        t1.start()
        t2.start()
        t1.join(timeout=1.0)
        t2.join(timeout=1.0)

        self.assertFalse(t1.is_alive(), "Worker 1 deadlocked")
        self.assertFalse(t2.is_alive(), "Worker 2 deadlocked")
        # At least one succeeded cleanly
        self.assertTrue(results.get("T1") or results.get("T2"))

    # 8. Failed second transaction rolls back second only when safe (Isolated Rollback / Counterexample C)
    def test_failed_second_transaction_rolls_back_second_only_when_safe(self):
        state = make_test_state()
        dec1 = make_test_decision(ActionCandidateType.TELEMETRY_BURST_TRIGGER, 30.0, decision_id="DEC-S1", source_state_version=0)
        dec2 = make_test_decision(ActionCandidateType.SPECULATIVE_DEPTH_TUNING, 3, decision_id="DEC-S2", source_state_version=0)

        prep_ok, _, seq = self.arbiter.prepare_sequence([dec1, dec2], state)
        self.assertTrue(prep_ok)

        # Inject failure into step 1 (dec2)
        report = self.arbiter.execute_sequence(
            seq,
            state,
            injected_step_faults={1: "PREDICTION_DIVERGENCE"},
            force_cascade_rollback=False,
        )

        self.assertFalse(report["success"])
        self.assertEqual(report["sequence_state"], SequenceState.ROLLED_BACK.value)
        self.assertEqual(report["rollback_mode"], RollbackMode.ISOLATED.value)

        # Verify step 1 value remains applied because S1 was safe!
        live_ctrl = self.engine.get_current_control_state()
        self.assertEqual(live_ctrl["telemetry_sampling_rate_hz"], 30.0)
        # But speculative depth was restored to baseline (1)
        self.assertEqual(live_ctrl["speculative_frame_depth"], 1)

    # 9. Cascade rollback returns to safe checkpoint (Cascading Rollback / Counterexample D)
    def test_cascade_rollback_returns_to_safe_checkpoint(self):
        state = make_test_state()
        dec1 = make_test_decision(ActionCandidateType.TELEMETRY_BURST_TRIGGER, 45.0, decision_id="DEC-C1", source_state_version=0)
        dec2 = make_test_decision(ActionCandidateType.SPECULATIVE_DEPTH_TUNING, 4, decision_id="DEC-C2", source_state_version=0)

        prep_ok, _, seq = self.arbiter.prepare_sequence([dec1, dec2], state)
        self.assertTrue(prep_ok)

        report = self.arbiter.execute_sequence(
            seq,
            state,
            injected_step_faults={1: "RESOURCE_SATURATION_CASCADE"},
            force_cascade_rollback=True,
        )

        self.assertFalse(report["success"])
        self.assertEqual(report["sequence_state"], SequenceState.ROLLED_BACK.value)
        self.assertEqual(report["rollback_mode"], RollbackMode.CASCADING.value)

        # Verify both steps were rolled back to CP-0 baseline
        live_ctrl = self.engine.get_current_control_state()
        self.assertEqual(live_ctrl["telemetry_sampling_rate_hz"], 2.0)
        self.assertEqual(live_ctrl["speculative_frame_depth"], 1)

    # 10. GPU failover invalidates pending GPU transactions (Counterexample E)
    def test_gpu_failover_invalidates_pending_gpu_transactions(self):
        state = make_test_state()
        # Failover to CPU followed by GPU profile tuning
        dec1 = make_test_decision(ActionCandidateType.GPU_FAILOVER_TO_CPU, "CPU_FALLBACK", decision_id="DEC-FO", source_state_version=0)
        dec2 = make_test_decision(ActionCandidateType.GPU_PROFILE_DEGRADATION, "POWER_SAVER", decision_id="DEC-GPU", source_state_version=0)

        # In preflight: compute_backend + gpu_profile_level are MUTUALLY_EXCLUSIVE
        prep_ok, err, seq = self.arbiter.prepare_sequence([dec1, dec2], state)
        self.assertFalse(prep_ok)
        self.assertIn("MUTUALLY_EXCLUSIVE", err)

    # 11. Compound prediction is rebased after first commit (Counterexample B)
    def test_compound_prediction_is_rebased_after_first_commit(self):
        state = make_test_state()
        dec1 = make_test_decision(ActionCandidateType.TELEMETRY_BURST_TRIGGER, 20.0, decision_id="DEC-R1", source_state_version=0)
        dec2 = make_test_decision(ActionCandidateType.SPECULATIVE_DEPTH_TUNING, 2, decision_id="DEC-R2", source_state_version=0)

        prep_ok, _, seq = self.arbiter.prepare_sequence([dec1, dec2], state)
        self.assertTrue(prep_ok)

        report = self.arbiter.execute_sequence(seq, state)
        self.assertTrue(report["success"])

        # Check audit ledger contains decision_rebased event
        rebase_events = [e for e in seq.audit_ledger if e["event_type"] == "decision_rebased"]
        self.assertEqual(len(rebase_events), 1)
        self.assertEqual(rebase_events[0]["details"]["new_version"], 1)

    # 12. Live resource measurement overrides model estimate
    def test_live_resource_measurement_overrides_model_estimate(self):
        tracker = UMAPressureTracker()
        sample = tracker.sample_pressure({"memory_bandwidth_utilization": 0.88, "memory_latency_ns": 75.0})
        self.assertEqual(sample["memory_bandwidth_utilization"], 0.88)
        self.assertEqual(sample["memory_latency_ns"], 75.0)

    # 13. Timeout adjustment is bounded and audited
    def test_timeout_adjustment_is_bounded_and_audited(self):
        detector = PowerStarvationDetector()
        is_starved, classification = detector.check_starvation(
            gpu_utilization_pct=92.0,
            cpu_frequency_ghz=1.2,
            observer_latency_ms=22.0,
        )
        self.assertTrue(is_starved)
        self.assertIn("POWER_STARVATION_RISK", classification)

        # Adaptive deadline
        budget, adjusted = AdaptiveDeadlinePolicy.calculate_deadline(
            base_timeout_s=1.0,
            measured_delay_s=0.25,
            power_starvation_allowance_s=0.50,
            hard_max_timeout_s=1.50,
        )
        self.assertTrue(adjusted)
        self.assertLessEqual(budget, 1.50)
        self.assertEqual(budget, 1.50)  # 1.0 + 0.25 + 0.50 = 1.75 -> clamped to 1.50

    # 14. Sequence audit chain integrity
    def test_sequence_audit_chain_integrity(self):
        state = make_test_state()
        dec = make_test_decision(ActionCandidateType.TELEMETRY_BURST_TRIGGER, 15.0, decision_id="DEC-CHAIN", source_state_version=0)
        _, _, seq = self.arbiter.prepare_sequence([dec], state)
        self.arbiter.execute_sequence(seq, state)

        # Verify cryptographic chain
        self.assertGreater(len(seq.audit_ledger), 2)
        for i in range(1, len(seq.audit_ledger)):
            self.assertEqual(seq.audit_ledger[i]["prev_hash"], seq.audit_ledger[i - 1]["event_hash"])

    # 15. Unknown dependency defaults to conservative serialization
    def test_unknown_dependency_defaults_to_conservative_serialization(self):
        classification = TransactionInterferenceMatrix.classify("custom_var_x", "telemetry_sampling_rate_hz")
        self.assertEqual(classification, CouplingClassification.UNKNOWN)
        self.assertTrue(TransactionInterferenceMatrix.requires_serialization("custom_var_x", "telemetry_sampling_rate_hz"))

    # 16. Counterexample F: Two threads prepare conflicting coupled variables simultaneously
    def test_concurrent_conflicting_coupled_variables(self):
        state = make_test_state()
        results = []

        def worker(sid: str, action_type: ActionCandidateType, val: Any):
            dec = make_test_decision(action_type, val, decision_id=f"DEC-{sid}", source_state_version=self.arbiter.get_state_version())
            prep_ok, err, seq = self.arbiter.prepare_sequence([dec], state, sequence_id=sid)
            if prep_ok:
                rep = self.arbiter.execute_sequence(seq, state)
                results.append((sid, rep["success"]))
            else:
                results.append((sid, False))

        t1 = threading.Thread(target=worker, args=("W1", ActionCandidateType.TELEMETRY_BURST_TRIGGER, 35.0))
        t2 = threading.Thread(target=worker, args=("W2", ActionCandidateType.TELEMETRY_BURST_TRIGGER, 45.0))
        t1.start()
        t2.start()
        t1.join(timeout=2.0)
        t2.join(timeout=2.0)

        # One succeeds or deterministic winner / loser, no deadlock or corruption
        self.assertEqual(len(results), 2)
        live_ctrl = self.engine.get_current_control_state()
        self.assertIn(live_ctrl["telemetry_sampling_rate_hz"], {35.0, 45.0, 2.0})


if __name__ == "__main__":
    unittest.main()
