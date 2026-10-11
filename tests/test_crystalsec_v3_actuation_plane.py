"""
Formal Verification Test Suite for CrystalSec 3.0 Phase 4:
Transactional Actuation, Predictive Rollback & Runtime Commit Protocol.

Verifies:
1. Bounded mutation whitelist enforcement
2. Two-layer PreconditionSnapshot (Control State + Verification State)
3. Prediction error calculation and commit threshold
4. Deterministic rollback on divergence, timeout, and observer limit
5. Rollback verification (control state restored and S in R_safe)
6. FAILED_SAFE fallback on failed rollback
7. Idempotency and concurrency conflict locking
8. GPU failover safe lifecycle and restriction on immediate GPU re-enablement
"""

import concurrent.futures
import copy
import time
import unittest

from krystal_kernel.verification.crystalsec_v3_actuation_plane import (
    ActuationRecord,
    ActuationResult,
    ComputeBackendAdapter,
    GPUProfileAdapter,
    GpuProfileLevel,
    ObservationWindow,
    PostRollbackVerifier,
    PreconditionSnapshot,
    RollbackPlan,
    RollbackVerification,
    RuntimeStateReader,
    SpeculativeDepthAdapter,
    TelemetrySamplingAdapter,
    TransactionalActuationApiDispatcher,
    TransactionalActuationEngine,
    TransactionState,
    compute_runtime_restorable_hash,
)
from krystal_kernel.verification.crystalsec_v3_control_plane import (
    ActionCandidateType,
    ControlPlaneDecision,
    CrystalSecV3ControlPlane,
    CrystalSecV3State,
    PsychologicalCopingMechanism,
    SchedulableAction,
    StrainRegime,
    StrainResult,
    TelemetryItem,
)


def make_test_state(
    temp_c: float = 45.0,
    h_val: float = 0.95,
    dh_dt: float = 0.0,
    mem_cont: float = 0.10,
    q_press: float = 0.10,
    divergence: float = 0.0,
    oracle_conf: float = 0.95,
    oracle_rca: str = "NONE",
    gpu_recovery: str = "READY",
    fault_active: bool = False,
    vital_hp: int = 6,
    telemetry_age_s: float = 0.0,
) -> CrystalSecV3State:
    t = time.time() - telemetry_age_s

    def item(v, conf=1.0):
        return TelemetryItem(v, t, "TEST", conf, True)

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


class TestCrystalSecV3ActuationPlane(unittest.TestCase):
    """Test suite certifying Phase 4 Transactional Actuation and Rollback."""

    def setUp(self):
        self.control_plane = CrystalSecV3ControlPlane()
        self.telemetry_adapter = TelemetrySamplingAdapter(initial_rate_hz=2.0)
        self.speculative_adapter = SpeculativeDepthAdapter(initial_depth=1)
        self.gpu_profile_adapter = GPUProfileAdapter(initial_profile=GpuProfileLevel.BALANCED)
        self.compute_backend_adapter = ComputeBackendAdapter(initial_backend="VULKAN_GPU")

        self.engine = TransactionalActuationEngine(
            telemetry_adapter=self.telemetry_adapter,
            speculative_adapter=self.speculative_adapter,
            gpu_profile_adapter=self.gpu_profile_adapter,
            compute_backend_adapter=self.compute_backend_adapter,
        )

    def test_bounded_mutation_applies_correctly(self):
        """Verifies that a whitelisted action applies its bounded mutation cleanly."""
        state = make_test_state(temp_c=72.0, h_val=0.65, mem_cont=0.75, divergence=0.35)
        decision = self.control_plane.evaluate_control_plane(state)

        # Force selected action to be SPECULATIVE_DEPTH_TUNING for clean unit testing
        decision.selected_action = {
            "action_id": "ACT-SPEC-001",
            "action_type": ActionCandidateType.SPECULATIVE_DEPTH_TUNING.value,
            "proposed_value": 0,
            "controlled_variable": "speculative_frame_depth",
            "allowed": True,
        }
        decision.counterfactual_results = [{
            "action_id": "ACT-SPEC-001",
            "predicted_sigma": 0.35,
            "predicted_temperature": 70.0,
            "predicted_latency": 0.40,
            "predicted_queue_pressure": 0.05,
            "predicted_divergence": 0.30,
        }]

        record = self.engine.execute_transaction(decision, state)
        self.assertEqual(record.result, ActuationResult.COMMITTED)
        self.assertEqual(record.transaction_state, TransactionState.COMMITTED)
        self.assertEqual(self.speculative_adapter.read_current(), 0)

    def test_prediction_match_commits_transaction(self):
        """When observed delta matches counterfactual prediction within tolerance, commits."""
        state = make_test_state(temp_c=68.0, h_val=0.70)
        decision = self.control_plane.evaluate_control_plane(state)

        decision.selected_action = {
            "action_id": "ACT-TEL-001",
            "action_type": ActionCandidateType.TELEMETRY_BURST_TRIGGER.value,
            "proposed_value": 50.0,
            "controlled_variable": "telemetry_sampling_rate_hz",
            "allowed": True,
        }
        decision.counterfactual_results = [{
            "action_id": "ACT-TEL-001",
            "predicted_sigma": 0.25,
            "predicted_temperature": 68.0,
            "predicted_latency": 0.45,
            "predicted_queue_pressure": 0.10,
            "predicted_divergence": 0.0,
        }]

        # Provider matches predicted delta closely
        def good_telemetry():
            return {
                "junction_temperature_c": 68.0,
                "sigma_strain": 0.25,
                "gpu_frame_time_ms": 0.45,
                "queue_pressure": 0.10,
                "D_divergence": 0.0,
                "health_vector_H": 0.70,
            }

        record = self.engine.execute_transaction(decision, state, observation_telemetry_provider=good_telemetry)
        self.assertEqual(record.result, ActuationResult.COMMITTED)
        self.assertLessEqual(record.prediction_error, record.max_allowed_deviation)
        self.assertEqual(self.telemetry_adapter.read_current(), 50.0)

    def test_prediction_divergence_triggers_rollback(self):
        """When observed behavior diverges beyond max_allowed_deviation, triggers rollback."""
        state = make_test_state(temp_c=65.0, mem_cont=0.6)
        decision = self.control_plane.evaluate_control_plane(state)

        decision.selected_action = {
            "action_id": "ACT-PROF-001",
            "action_type": ActionCandidateType.GPU_PROFILE_DEGRADATION.value,
            "proposed_value": GpuProfileLevel.POWER_SAVER.value,
            "controlled_variable": "gpu_profile_level",
            "allowed": True,
        }
        decision.counterfactual_results = [{
            "action_id": "ACT-PROF-001",
            "predicted_sigma": 0.20,
            "predicted_temperature": 55.0,
            "predicted_latency": 0.30,
            "predicted_queue_pressure": 0.05,
            "predicted_divergence": 0.0,
        }]

        # Injected anomaly: Temperature actually spikes to 93C and latency surges to 15.0ms!
        def divergent_telemetry():
            return {
                "junction_temperature_c": 93.0,
                "sigma_strain": 0.85,
                "gpu_frame_time_ms": 15.0,
                "queue_pressure": 0.90,
                "D_divergence": 0.70,
                "health_vector_H": 0.40,
            }

        record = self.engine.execute_transaction(decision, state, observation_telemetry_provider=divergent_telemetry)
        self.assertEqual(record.result, ActuationResult.ROLLED_BACK)
        self.assertEqual(record.transaction_state, TransactionState.ROLLED_BACK)
        # Profile must be restored to BALANCED
        self.assertEqual(self.gpu_profile_adapter.read_current(), GpuProfileLevel.BALANCED.value)

    def test_timeout_triggers_deterministic_rollback(self):
        """Observation timeout triggers deterministic rollback to precondition snapshot."""
        state = make_test_state()
        decision = self.control_plane.evaluate_control_plane(state)

        decision.selected_action = {
            "action_id": "ACT-TO-001",
            "action_type": ActionCandidateType.SPECULATIVE_DEPTH_TUNING.value,
            "proposed_value": 3,
            "controlled_variable": "speculative_frame_depth",
            "allowed": True,
        }

        record = self.engine.execute_transaction(decision, state, injected_fault="TIMEOUT")
        self.assertEqual(record.result, ActuationResult.ROLLED_BACK)
        self.assertEqual(self.speculative_adapter.read_current(), 1)  # Restored initial value

    def test_stale_decision_fails_eligibility_check(self):
        """Decisions older than MAX_DECISION_AGE_S are rejected at eligibility check."""
        state = make_test_state()
        decision = self.control_plane.evaluate_control_plane(state, current_time=time.time() - 10.0)

        record = self.engine.execute_transaction(decision, state, current_time=time.time())
        self.assertEqual(record.result, ActuationResult.ELIGIBILITY_FAILED)
        self.assertEqual(record.transaction_state, TransactionState.REJECTED)

    def test_unwhitelisted_variable_is_rejected(self):
        """UMA_ISOLATION_ACTUATION or arbitrary unwhitelisted variables are rejected."""
        state = make_test_state()
        decision = self.control_plane.evaluate_control_plane(state)

        decision.selected_action = {
            "action_id": "ACT-UMA-001",
            "action_type": ActionCandidateType.UMA_ISOLATION_ACTUATION.value,
            "proposed_value": "ISOLATE_PAGES_0_TO_1024",
            "allowed": True,
        }

        record = self.engine.execute_transaction(decision, state)
        self.assertEqual(record.result, ActuationResult.ELIGIBILITY_FAILED)
        self.assertIn("not in the actuation whitelist", record.events[0]["details"]["reason"])

    def test_cpu_failover_transaction_lifecycle(self):
        """GPU_FAILOVER_TO_CPU smoothly drains GPU work and transitions to CPU_FALLBACK."""
        state = make_test_state(temp_c=96.0, gpu_recovery="FAULT_DETECTED")
        decision = self.control_plane.evaluate_control_plane(state)

        self.assertEqual(decision.selected_action["action_type"], ActionCandidateType.GPU_FAILOVER_TO_CPU.value)

        record = self.engine.execute_transaction(decision, state)
        self.assertEqual(record.result, ActuationResult.COMMITTED)
        self.assertEqual(self.compute_backend_adapter.read_current(), "CPU_FALLBACK")

    def test_restored_control_state_matches_snapshot(self):
        """Verifies PreconditionSnapshot equality after a successful rollback."""
        state = make_test_state()
        decision = self.control_plane.evaluate_control_plane(state)

        decision.selected_action = {
            "action_id": "ACT-SNAP-001",
            "action_type": ActionCandidateType.TELEMETRY_BURST_TRIGGER.value,
            "proposed_value": 75.0,
            "allowed": True,
        }

        # Incur rollback via divergence
        def bad_telemetry():
            return {"junction_temperature_c": 94.0, "sigma_strain": 0.95}

        record = self.engine.execute_transaction(decision, state, observation_telemetry_provider=bad_telemetry)
        self.assertEqual(record.result, ActuationResult.ROLLED_BACK)

        snapshot = self.engine._snapshots[record.transaction_id]
        current_state = self.engine.get_current_control_state()
        self.assertTrue(snapshot.matches_control_state(current_state))
        self.assertEqual(self.telemetry_adapter.read_current(), 2.0)

    def test_rollback_reenters_safe_region(self):
        """Rollback verification proves runtime state re-enters declared safe region bounds."""
        state = make_test_state()
        decision = self.control_plane.evaluate_control_plane(state)

        decision.selected_action = {
            "action_id": "ACT-SAFE-001",
            "action_type": ActionCandidateType.SPECULATIVE_DEPTH_TUNING.value,
            "proposed_value": 4,
            "allowed": True,
        }

        record = self.engine.execute_transaction(decision, state, injected_fault="TIMEOUT")
        last_event = record.events[-1]
        self.assertEqual(last_event["event_type"], "rollback_verified")
        self.assertTrue(last_event["details"]["safe_region_reentered"])
        self.assertTrue(last_event["details"]["verified"])

    def test_failed_rollback_enters_failed_safe(self):
        """When an adapter fails to execute rollback, transitions unconditionally to FAILED_SAFE."""
        state = make_test_state()
        decision = self.control_plane.evaluate_control_plane(state)

        decision.selected_action = {
            "action_id": "ACT-FAIL-001",
            "action_type": ActionCandidateType.SPECULATIVE_DEPTH_TUNING.value,
            "proposed_value": 2,
            "allowed": True,
        }

        record = self.engine.execute_transaction(
            decision,
            state,
            injected_fault="TIMEOUT",
            force_fail_rollback=True,  # Simulate hardware or driver adapter failure
        )
        self.assertEqual(record.result, ActuationResult.FAILED_SAFE)
        self.assertEqual(record.transaction_state, TransactionState.FAILED_SAFE)
        # Confirms safe fallbacks enforced
        self.assertEqual(self.compute_backend_adapter.read_current(), "CPU_FALLBACK")
        self.assertEqual(self.gpu_profile_adapter.read_current(), GpuProfileLevel.POWER_SAVER.value)

    def test_duplicate_transaction_is_idempotent(self):
        """Resubmitting the same transaction ID returns existing terminal result without re-mutating."""
        state = make_test_state()
        decision = self.control_plane.evaluate_control_plane(state)

        decision.selected_action = {
            "action_id": "ACT-IDEM-001",
            "action_type": ActionCandidateType.SPECULATIVE_DEPTH_TUNING.value,
            "proposed_value": 0,
            "allowed": True,
        }

        rec1 = self.engine.execute_transaction(decision, state)
        self.assertEqual(rec1.result, ActuationResult.COMMITTED)

        # Mutate adapter manually to test if second execution touches it
        self.speculative_adapter._depth = 2

        rec2 = self.engine.execute_transaction(decision, state)
        self.assertEqual(rec1.transaction_id, rec2.transaction_id)
        # Did not re-execute or alter adapter
        self.assertEqual(self.speculative_adapter._depth, 2)

    def test_conflicting_transaction_is_rejected(self):
        """Submitting a transaction on a variable currently locked by another transaction is rejected."""
        # Artificially lock variable
        self.engine._active_variable_locks["speculative_frame_depth"] = "TX-ACTIVE-001"

        state = make_test_state()
        decision = self.control_plane.evaluate_control_plane(state)
        decision.selected_action = {
            "action_id": "ACT-CONFLICT-001",
            "action_type": ActionCandidateType.SPECULATIVE_DEPTH_TUNING.value,
            "proposed_value": 0,
            "allowed": True,
        }

        record = self.engine.execute_transaction(decision, state)
        self.assertEqual(record.result, ActuationResult.ELIGIBILITY_FAILED)
        self.assertIn("currently locked", record.events[0]["details"]["reason"])

    def test_invariant_violation_preempts_transaction(self):
        """Breach of universal invariant VITAL_MAX_HP != 6 preempts eligibility."""
        state_breached = make_test_state()
        # Direct attribute breach bypass
        object.__setattr__(state_breached, "vital_max_hp", 5)

        decision = self.control_plane.evaluate_control_plane(make_test_state())
        decision.selected_action = {
            "action_id": "ACT-INV-001",
            "action_type": ActionCandidateType.SPECULATIVE_DEPTH_TUNING.value,
            "proposed_value": 0,
            "allowed": True,
        }

        record = self.engine.execute_transaction(decision, state_breached)
        self.assertEqual(record.result, ActuationResult.ELIGIBILITY_FAILED)
        self.assertIn("Universal invariant breached", record.events[0]["details"]["reason"])

    def test_observer_overhead_rolls_back_telemetry_burst(self):
        """Telemetry burst that exceeds observer overhead budget (>15%) rolls back safely."""
        state = make_test_state()
        decision = self.control_plane.evaluate_control_plane(state)
        decision.selected_action = {
            "action_id": "ACT-BURST-001",
            "action_type": ActionCandidateType.TELEMETRY_BURST_TRIGGER.value,
            "proposed_value": 100.0,
            "allowed": True,
        }

        record = self.engine.execute_transaction(decision, state, injected_fault="OBSERVER_OVERHEAD_SPIKE")
        self.assertEqual(record.result, ActuationResult.ROLLED_BACK)
        self.assertEqual(self.telemetry_adapter.read_current(), 2.0)

    def test_gpu_restore_requires_new_decision(self):
        """
        Critical Rule: In ComputeBackendAdapter, rollbacks cannot blindly resurrect
        a faulted GPU pipeline into VULKAN_GPU in the same transaction.
        """
        adapter = ComputeBackendAdapter(initial_backend="VULKAN_GPU")
        adapter.apply("CPU_FALLBACK")
        self.assertEqual(adapter.read_current(), "CPU_FALLBACK")

        # Attempt rollback to VULKAN_GPU
        ok = adapter.rollback("VULKAN_GPU")
        self.assertFalse(ok)
        self.assertEqual(adapter.read_current(), "CPU_FALLBACK")

    def test_machine_readable_report_and_api_dispatcher(self):
        """Verifies report generation contract and API dispatcher endpoints."""
        report = self.engine.generate_report()
        self.assertEqual(report["phase"], "CrystalSec3_Transactional_Actuation")
        self.assertIn("telemetry_sampling_rate_hz", report["actuation_surface"])
        self.assertFalse(report["uma_isolation_enabled"])

        # Test API Dispatcher
        dispatcher = TransactionalActuationApiDispatcher(self.engine, self.control_plane)
        state = make_test_state()
        decision = self.control_plane.evaluate_control_plane(state)
        decision.selected_action = {
            "action_id": "ACT-API-001",
            "action_type": ActionCandidateType.SPECULATIVE_DEPTH_TUNING.value,
            "proposed_value": 0,
            "allowed": True,
        }

        res = dispatcher.dispatch_request(
            "/api/cortex/actuation/submit",
            method="POST",
            payload={"decision": decision.to_dict(), "state": state.to_dict()},
        )
        self.assertEqual(res["status"], "SUCCESS")
        tx_id = res["record"]["transaction_id"]

        # Query transaction
        res_get = dispatcher.dispatch_request(f"/api/cortex/actuation/{tx_id}", method="GET")
        self.assertEqual(res_get["status"], "SUCCESS")
        self.assertEqual(res_get["record"]["transaction_id"], tx_id)

    # ========================================================================
    # PHASE 4.1: RUNTIME TRUTH VERIFICATION & FAULTED RECOVERY TESTS
    # ========================================================================

    def test_post_rollback_hash_uses_live_runtime_state(self):
        """
        Anti-Tautology Proof: If an adapter fails to restore the live variable,
        the post-rollback hash computed from the live state diverges from the
        precondition snapshot hash, failing verification.
        """
        state = make_test_state()
        decision = self.control_plane.evaluate_control_plane(state)
        decision.selected_action = {
            "action_id": "ACT-HASH-001",
            "action_type": ActionCandidateType.SPECULATIVE_DEPTH_TUNING.value,
            "proposed_value": 3,
            "allowed": True,
        }

        # Initial depth is 1
        broken_adapter = SpeculativeDepthAdapter(initial_depth=1)
        engine = TransactionalActuationEngine(speculative_adapter=broken_adapter)
        # Capture snapshot while initial depth is 1
        snapshot = engine.capture_precondition_snapshot(decision, state, "speculative_frame_depth")
        self.assertEqual(snapshot.old_value, 1)

        # Mutate to 3
        broken_adapter.apply(3)
        # Corrupt rollback to be a no-op that leaves value at 3
        broken_adapter.rollback = lambda old_val: True  # claims success, but leaves value at 3!

        # Verify live hash calculation
        current_control_state = engine.get_current_control_state()
        self.assertEqual(current_control_state["speculative_frame_depth"], 3)  # Still corrupted

        verification = PostRollbackVerifier.verify(
            snapshot=snapshot,
            current_control_state=current_control_state,
            live_telemetry=snapshot.telemetry_baseline,
            invariant_evaluation={"vital_max_hp": True, "no_device_fault": True},
        )

        self.assertFalse(verification.matches_precondition_hash)
        self.assertFalse(verification.control_state_restored)
        self.assertFalse(verification.verified)

    def test_fake_snapshot_hash_cannot_pass_verification(self):
        """If a precondition snapshot possesses a tampered hash, rollback verification fails."""
        state = make_test_state()
        decision = self.control_plane.evaluate_control_plane(state)
        snapshot = self.engine.capture_precondition_snapshot(decision, state, "speculative_frame_depth")
        # Tamper snapshot hash
        snapshot.canonical_restorable_hash = "0" * 64

        current_control_state = self.engine.get_current_control_state()
        verification = PostRollbackVerifier.verify(
            snapshot=snapshot,
            current_control_state=current_control_state,
            live_telemetry=snapshot.telemetry_baseline,
            invariant_evaluation={"vital_max_hp": True, "no_device_fault": True},
        )
        self.assertFalse(verification.matches_precondition_hash)
        self.assertFalse(verification.verified)

    def test_invariant_restoration_is_remeasured(self):
        """Invariant restoration must be evaluated from actual live measurements."""
        state = make_test_state()
        decision = self.control_plane.evaluate_control_plane(state)
        snapshot = self.engine.capture_precondition_snapshot(decision, state, "speculative_frame_depth")

        # Invariant breached in live state (e.g. vital_max_hp != 6)
        live_invariants = {"vital_max_hp": False, "no_device_fault": True}
        verification = PostRollbackVerifier.verify(
            snapshot=snapshot,
            current_control_state=self.engine.get_current_control_state(),
            live_telemetry=snapshot.telemetry_baseline,
            invariant_evaluation=live_invariants,
        )
        self.assertFalse(verification.invariant_state_restored)
        self.assertFalse(verification.verified)

    def test_safe_region_uses_fresh_telemetry(self):
        """Safe region verification re-evaluates fresh measured telemetry, not static baseline."""
        state = make_test_state()
        decision = self.control_plane.evaluate_control_plane(state)
        snapshot = self.engine.capture_precondition_snapshot(decision, state, "speculative_frame_depth")

        # Fresh telemetry exceeds thermal ceiling (99C > 95C limit)
        hot_telemetry = copy.deepcopy(snapshot.telemetry_baseline)
        hot_telemetry["junction_temperature_c"] = 99.0

        verification = PostRollbackVerifier.verify(
            snapshot=snapshot,
            current_control_state=self.engine.get_current_control_state(),
            live_telemetry=hot_telemetry,
            invariant_evaluation={"vital_max_hp": True, "no_device_fault": True},
        )
        self.assertFalse(verification.safe_region_reentered)
        self.assertFalse(verification.verified)

    def test_real_monotonic_timeout_rolls_back(self):
        """ObservationWindow enforces real monotonic timeout and triggers rollback."""
        state = make_test_state()
        decision = self.control_plane.evaluate_control_plane(state)
        decision.selected_action = {
            "action_id": "ACT-TIME-REAL",
            "action_type": ActionCandidateType.SPECULATIVE_DEPTH_TUNING.value,
            "proposed_value": 3,
            "allowed": True,
        }

        # Simulated slow provider that takes 0.05s per sample
        def slow_provider():
            time.sleep(0.04)
            return {"gpu_frame_time_ms": 0.45}

        # Timeout set to 0.02s (< 0.04s sample latency)
        record = self.engine.execute_transaction(
            decision,
            state,
            observation_telemetry_provider=slow_provider,
            min_observation_samples=3,
            override_timeout_s=0.02,
        )

        self.assertEqual(record.transaction_state, TransactionState.ROLLED_BACK)
        self.assertEqual(record.result, ActuationResult.ROLLED_BACK)
        # Confirms 'timed_out' event was recorded
        event_types = [e["event_type"] for e in record.events]
        self.assertIn("timed_out", event_types)

    def test_observation_requires_minimum_samples(self):
        """ObservationWindow will not evaluate early until min_samples is achieved."""
        obs = ObservationWindow(min_samples=4, sample_interval_s=0.001, timeout_s=0.02)
        call_count = [0]

        def limited_provider():
            call_count[0] += 1
            if call_count[0] <= 2:
                return {"gpu_frame_time_ms": 0.45}
            time.sleep(0.03)  # Block remaining
            return {}

        success = obs.collect(limited_provider)
        self.assertFalse(success)
        self.assertTrue(obs.timed_out)
        self.assertLess(len(obs.samples), 4)

    def test_single_good_sample_cannot_mask_bad_window(self):
        """
        A single low-latency sample cannot mask a degraded observation window;
        the median reflects the prevailing degraded state and triggers rollback.
        """
        state = make_test_state()
        decision = self.control_plane.evaluate_control_plane(state)
        decision.selected_action = {
            "action_id": "ACT-BURST-MEDIAN",
            "action_type": ActionCandidateType.SPECULATIVE_DEPTH_TUNING.value,
            "proposed_value": 4,
            "allowed": True,
        }
        decision.counterfactual_results = [{
            "action_id": "ACT-BURST-MEDIAN",
            "predicted_latency": 0.45,
            "predicted_sigma": 0.10,
        }]

        sample_iter = iter([
            {"gpu_frame_time_ms": 0.45, "sigma_strain": 0.10},  # 1 lucky sample
            {"gpu_frame_time_ms": 25.0, "sigma_strain": 0.90},  # 4 heavily degraded samples
            {"gpu_frame_time_ms": 26.0, "sigma_strain": 0.92},
            {"gpu_frame_time_ms": 24.0, "sigma_strain": 0.88},
            {"gpu_frame_time_ms": 27.0, "sigma_strain": 0.95},
        ])

        def burst_provider():
            try:
                return next(sample_iter)
            except StopIteration:
                return {"gpu_frame_time_ms": 25.0, "sigma_strain": 0.90}

        record = self.engine.execute_transaction(
            decision,
            state,
            observation_telemetry_provider=burst_provider,
            min_observation_samples=5,
        )

        self.assertEqual(record.result, ActuationResult.ROLLED_BACK)
        # Median latency was ~25.0ms, far exceeding prediction 0.45ms
        self.assertGreater(record.observed_delta["gpu_frame_time_ms"], 20.0)

    def test_concurrent_same_variable_submission_is_atomic(self):
        """Multiple threads submitting transactions on the same variable are serialized atomically."""
        engine = TransactionalActuationEngine()
        state = make_test_state()
        decision = self.control_plane.evaluate_control_plane(state)
        decision.selected_action = {
            "action_id": "ACT-THREAD-SAFE",
            "action_type": ActionCandidateType.SPECULATIVE_DEPTH_TUNING.value,
            "proposed_value": 0,
            "allowed": True,
        }

        results = []

        def worker(thread_idx: int):
            dec_copy = copy.deepcopy(decision)
            dec_copy.decision_id = f"DEC-THREAD-{thread_idx}"
            # Slow observation window to guarantee overlap
            def delay_provider():
                time.sleep(0.01)
                return {"gpu_frame_time_ms": 0.45}

            rec = engine.execute_transaction(
                dec_copy,
                state,
                observation_telemetry_provider=delay_provider,
                min_observation_samples=2,
            )
            return rec.result

        with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
            futures = [executor.submit(worker, i) for i in range(6)]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]

        # Exactly 1 transaction should acquire the variable and commit
        # The remaining 5 should be rejected as concurrent conflicting locks
        committed_count = results.count(ActuationResult.COMMITTED)
        rejected_count = results.count(ActuationResult.ELIGIBILITY_FAILED)

        self.assertEqual(committed_count, 1)
        self.assertEqual(rejected_count, 5)

    def test_failed_safe_report_is_not_marked_verified(self):
        """When any transaction enters FAILED_SAFE, generate_report() reports verified = False."""
        state = make_test_state()
        decision = self.control_plane.evaluate_control_plane(state)
        decision.selected_action = {
            "action_id": "ACT-FS-REPORT",
            "action_type": ActionCandidateType.SPECULATIVE_DEPTH_TUNING.value,
            "proposed_value": 2,
            "allowed": True,
        }

        # Force rollback failure
        self.engine.execute_transaction(
            decision,
            state,
            injected_fault="TIMEOUT",
            force_fail_rollback=True,
        )

        report = self.engine.generate_report()
        self.assertFalse(report["rollback"]["verified"])
        self.assertGreaterEqual(report["transactions"]["failed_safe"], 1)

    def test_committed_transaction_uses_observed_not_synthetic_delta(self):
        """Verifies that observed_delta in a committed transaction accurately matches empirical median."""
        state = make_test_state()
        decision = self.control_plane.evaluate_control_plane(state)
        decision.selected_action = {
            "action_id": "ACT-EMPIRICAL-001",
            "action_type": ActionCandidateType.SPECULATIVE_DEPTH_TUNING.value,
            "proposed_value": 0,
            "allowed": True,
        }
        decision.counterfactual_results = [{
            "action_id": "ACT-EMPIRICAL-001",
            "predicted_latency": 0.40,
            "predicted_sigma": 0.15,
        }]

        # Provider emits explicit samples with latency 0.38ms (baseline is 0.45ms -> delta = -0.07ms)
        def empirical_provider():
            return {
                "gpu_frame_time_ms": 0.38,
                "sigma_strain": 0.15,
                "junction_temperature_c": 45.0,
                "queue_pressure": 0.10,
                "D_divergence": 0.0,
                "health_vector_H": 0.95,
            }

        record = self.engine.execute_transaction(
            decision,
            state,
            observation_telemetry_provider=empirical_provider,
            min_observation_samples=3,
        )

        self.assertEqual(record.result, ActuationResult.COMMITTED)
        self.assertAlmostEqual(record.observed_delta["gpu_frame_time_ms"], -0.07, places=3)



if __name__ == "__main__":
    unittest.main()
