r"""
Unit and Property Tests for CrystalSec 3.0 Phase 5:
Heterogeneous Execution Triad, Runtime Placement Engine & Cross-Tier Verification.

Covers all 20 required formal specifications:
1. Truthful hardware discovery (Tier 3 PHYSICALLY_UNAVAILABLE on Tiger Lake)
2. DirectSim mathematical correctness oracle
3. CPU AVX-512/VNNI and Iris Xe GPU capabilities
4. Multi-objective placement utility and hard constraint filtering
5. Migration cost, hysteresis, and oscillation prevention
6. StateVersion binding and stale placement invalidation
7. Transactional migration and rollback on divergence
8. Dynamic shadow oracle verification ramp-up
9. Thermal and UMA contention constraints
10. Deterministic placement repeatability
"""

import time
import unittest
from typing import Dict, Any

from krystal_kernel.verification.crystalsec_v3_control_plane import (
    ActionCandidateType,
    CrystalSecV3State,
    TelemetryItem,
)
from krystal_kernel.verification.crystalsec_v3_sequence_arbiter import StateVersionManager
from krystal_kernel.verification.crystalsec_v3_heterogeneous_execution import (
    CpuVnniAdapter,
    DedicatedNpuAdapter,
    DirectSimAdapter,
    ExecutionMode,
    ExecutionPlacementDecision,
    ExecutionTierID,
    HardwareCapabilityMap,
    HardwareDiscoveryEngine,
    HeterogeneousTriadController,
    IrisXeVulkanAdapter,
    RuntimePlacementEngine,
    TierAvailabilityStatus,
    TierOscillationGuard,
    TransactionalMigrationArbiter,
)
from src.python.cpu_vnni_backend import CpuVnniBackend


def make_test_state(
    temp_c: float = 60.0,
    mem_cont: float = 0.20,
    vital_max_hp: int = 6,
) -> CrystalSecV3State:
    t = time.time()
    def item(v, conf=1.0):
        return TelemetryItem(v, t, "TEST", conf, True)

    return CrystalSecV3State(
        health_vector_H=item(1.0),
        dH_dt=item(0.0),
        junction_temperature_c=item(temp_c),
        memory_contention=item(mem_cont),
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


class TestCrystalSecV3HeterogeneousExecution(unittest.TestCase):
    """Formal test suite for Phase 5 Heterogeneous Execution Triad."""

    def setUp(self):
        self.version_mgr = StateVersionManager(initial_version=0)
        self.controller = HeterogeneousTriadController(version_manager=self.version_mgr)
        self.placement_engine = self.controller.placement_engine
        self.migration_arbiter = self.controller.migration_arbiter

    # 1. Tier 0 is always available
    def test_tier0_always_available(self):
        t0 = self.controller.directsim
        self.assertTrue(t0.available())
        cap = t0.detect()
        self.assertTrue(cap.physically_present)
        self.assertTrue(cap.runtime_available)
        self.assertEqual(cap.status, TierAvailabilityStatus.READY)

    # 2. Tiger Lake reports no dedicated NPU
    def test_tiger_lake_reports_no_dedicated_npu(self):
        hw = self.controller.hardware_map
        self.assertFalse(hw.dedicated_npu_present)

        npu_adapter = self.controller.dedicated_npu
        cap = npu_adapter.detect()
        self.assertFalse(cap.physically_present)
        self.assertFalse(cap.runtime_available)
        self.assertEqual(cap.status, TierAvailabilityStatus.PHYSICALLY_UNAVAILABLE)
        self.assertEqual(cap.epistemic_label, "PHYSICALLY_UNAVAILABLE")

    # 3. VNNI capability detection
    def test_vnni_capability_detection(self):
        cpu_cap = self.controller.cpu_vnni.detect()
        self.assertTrue(cpu_cap.physically_present)
        self.assertTrue(cpu_cap.runtime_available)
        self.assertIn("INT8", cpu_cap.supported_precisions)
        self.assertTrue(self.controller.cpu_vnni.backend.capability.has_vnni)

    # 4. Vulkan capability detection
    def test_vulkan_capability_detection(self):
        gpu_cap = self.controller.iris_xe.detect()
        self.assertTrue(gpu_cap.physically_present)
        self.assertTrue(gpu_cap.runtime_available)
        self.assertIn("DP4A_INT8", gpu_cap.supported_precisions)

    # 5. Unavailable tier never selected
    def test_unavailable_tier_never_selected(self):
        state = make_test_state()
        action = {"action_id": "ACT-1", "action_type": ActionCandidateType.NOMINAL_GPU_DISPATCH.value}
        decision = self.placement_engine.evaluate_placement("W1", action, state)
        self.assertNotEqual(decision.selected_tier, ExecutionTierID.TIER_3_DEDICATED_NPU.value)
        self.assertNotIn(ExecutionTierID.TIER_3_DEDICATED_NPU.value, decision.candidate_tiers)

    # 6. GPU selected when faster and safe
    def test_gpu_selected_when_faster_and_safe(self):
        state = make_test_state(temp_c=50.0, mem_cont=0.20)
        action = {"action_id": "ACT-FAST", "action_type": ActionCandidateType.NOMINAL_GPU_DISPATCH.value}
        decision = self.placement_engine.evaluate_placement("W-THROUGHPUT", action, state, mode=ExecutionMode.THROUGHPUT_MODE)
        self.assertEqual(decision.selected_tier, ExecutionTierID.TIER_2_IRIS_XE.value)

    # 7. CPU selected when GPU UMA is saturated
    def test_cpu_selected_when_gpu_uma_saturated(self):
        # Memory contention >= 0.80 excludes GPU
        state = make_test_state(temp_c=55.0, mem_cont=0.88)
        action = {"action_id": "ACT-UMA", "action_type": ActionCandidateType.NOMINAL_GPU_DISPATCH.value}
        decision = self.placement_engine.evaluate_placement("W-UMA", action, state, mode=ExecutionMode.BALANCED_MODE)
        self.assertEqual(decision.selected_tier, ExecutionTierID.TIER_1_CPU_VNNI.value)

    # 8. DirectSim selected as last safe fallback
    def test_directsim_selected_as_last_safe_fallback(self):
        state = make_test_state()
        action = {"action_id": "ACT-FALLBACK"}

        # Simulate faulting both GPU and CPU
        self.controller.iris_xe.is_healthy = False
        self.controller.cpu_vnni.is_healthy = False

        decision = self.placement_engine.evaluate_placement("W-FALLBACK", action, state)
        self.assertEqual(decision.selected_tier, ExecutionTierID.TIER_0_DIRECTSIM.value)

    # 9. Migration cost prevents marginal switch
    def test_migration_cost_prevents_marginal_switch(self):
        state = make_test_state()
        action = {"action_id": "ACT-MARGINAL"}

        # Set current tier to CPU VNNI
        self.placement_engine.current_tier = ExecutionTierID.TIER_1_CPU_VNNI
        self.placement_engine.migration_threshold = 0.50  # Artificially high threshold

        decision = self.placement_engine.evaluate_placement("W-MARGINAL", action, state, mode=ExecutionMode.BALANCED_MODE)
        # Should stay on current tier because gain does not exceed threshold
        self.assertEqual(decision.selected_tier, ExecutionTierID.TIER_1_CPU_VNNI.value)
        self.assertIn("Maintained", decision.selection_reason)

    # 10. Tier hysteresis prevents oscillation
    def test_tier_hysteresis_prevents_oscillation(self):
        guard = self.placement_engine.oscillation_guard
        now = time.time()

        # Simulate 3 rapid reversals: CPU -> GPU -> CPU -> GPU -> CPU
        guard.record_transition("TIER_1_CPU_VNNI", "TIER_2_IRIS_XE")
        guard.record_transition("TIER_2_IRIS_XE", "TIER_1_CPU_VNNI")
        guard.record_transition("TIER_1_CPU_VNNI", "TIER_2_IRIS_XE")
        can_fourth = guard.record_transition("TIER_2_IRIS_XE", "TIER_1_CPU_VNNI")

        self.assertFalse(can_fourth)
        self.assertTrue(guard.is_locked)

        can_mig, err = guard.can_migrate("TIER_2_IRIS_XE", "TIER_1_CPU_VNNI", last_migration_time=0.0)
        self.assertFalse(can_mig)
        self.assertIn("TIER_OSCILLATION_DETECTED", err)

    # 11. StateVersion invalidates stale placement
    def test_state_version_invalidates_stale_placement(self):
        state = make_test_state()
        action = {"action_id": "ACT-VERSION"}

        # Current version is 0
        self.assertEqual(self.version_mgr.current(), 0)

        # Increment version to 1
        self.version_mgr.commit_transition()
        self.assertEqual(self.version_mgr.current(), 1)

        # Attempt placement evaluation bound to V0 -> must raise STALE_STATE_VERSION_INVALIDATION
        with self.assertRaises(ValueError) as ctx:
            self.placement_engine.evaluate_placement("W-STALE", action, state, source_state_version=0)
        self.assertIn("STALE_STATE_VERSION_INVALIDATION", str(ctx.exception))

    # 12. GPU fault falls back to CPU
    def test_gpu_fault_falls_back_to_cpu(self):
        state = make_test_state()
        action = {"action_id": "ACT-GPU-FAULT"}

        # Simulate GPU device loss
        self.controller.iris_xe.device_loss_active = True
        decision = self.placement_engine.evaluate_placement("W-GPU-LOSS", action, state)
        self.assertEqual(decision.selected_tier, ExecutionTierID.TIER_1_CPU_VNNI.value)

    # 13. CPU fault falls back to DirectSim
    def test_cpu_fault_falls_back_to_directsim(self):
        state = make_test_state()
        action = {"action_id": "ACT-CPU-FAULT"}

        # Simulate both GPU unavailable and CPU degraded
        self.controller.iris_xe.device_loss_active = True
        self.controller.cpu_vnni.is_healthy = False

        decision = self.placement_engine.evaluate_placement("W-CPU-LOSS", action, state)
        self.assertEqual(decision.selected_tier, ExecutionTierID.TIER_0_DIRECTSIM.value)

    # 14. Cross-tier output matches reference
    def test_cross_tier_output_matches_reference(self):
        workload = {"input": [1.0, 2.0, 3.0, 4.0, 5.0], "weights": [0.5, 0.5, 0.5, 0.5, 0.5]}

        t0_out, _ = self.controller.directsim.execute(workload)
        t1_out, _ = self.controller.cpu_vnni.execute(workload)
        t2_out, _ = self.controller.iris_xe.execute(workload)

        # Compare T1 vs T0
        ok_t1, diff_t1 = self.controller.cpu_vnni.verify(t1_out, t0_out)
        self.assertTrue(ok_t1)
        self.assertLessEqual(diff_t1, 1e-4)

        # Compare T2 vs T0
        ok_t2, diff_t2 = self.controller.iris_xe.verify(t2_out, t0_out)
        self.assertTrue(ok_t2)
        self.assertLessEqual(diff_t2, 1e-4)

    # 15. Migration is transactional
    def test_migration_is_transactional(self):
        state = make_test_state()
        workload = {"input": [2.0, 4.0, 6.0], "weights": [0.5, 0.5, 0.5]}
        action = {"action_id": "ACT-TX-MIG"}

        self.placement_engine.current_tier = ExecutionTierID.TIER_0_DIRECTSIM
        decision = self.placement_engine.evaluate_placement("W-TX", action, state, mode=ExecutionMode.THROUGHPUT_MODE)

        ok, msg, rec = self.migration_arbiter.execute_migration_transaction(decision, workload, state)
        self.assertTrue(ok)
        self.assertEqual(self.placement_engine.current_tier.value, decision.selected_tier)
        self.assertEqual(self.version_mgr.current(), 1)  # Version incremented on commit

    # 16. Failed migration rolls back
    def test_failed_migration_rolls_back(self):
        state = make_test_state()
        workload = {"input": [2.0, 4.0, 6.0], "weights": [0.5, 0.5, 0.5]}
        action = {"action_id": "ACT-FAIL-MIG"}

        self.placement_engine.current_tier = ExecutionTierID.TIER_0_DIRECTSIM
        decision = self.placement_engine.evaluate_placement("W-FAIL", action, state, mode=ExecutionMode.THROUGHPUT_MODE)

        # Inject output divergence fault during migration verification
        ok, msg, _ = self.migration_arbiter.execute_migration_transaction(
            decision, workload, state, injected_fault="OUTPUT_DIVERGENCE"
        )
        self.assertFalse(ok)
        self.assertIn("MIGRATION_ROLLED_BACK", msg)
        # Verify routing remained at TIER_0_DIRECTSIM
        self.assertEqual(self.placement_engine.current_tier, ExecutionTierID.TIER_0_DIRECTSIM)

    # 17. Shadow sampling increases after migration
    def test_shadow_sampling_increases_after_migration(self):
        state = make_test_state()
        workload = {"input": [1.0, 1.0], "weights": [0.5, 0.5]}
        action = {"action_id": "ACT-SHADOW"}

        self.placement_engine.current_tier = ExecutionTierID.TIER_0_DIRECTSIM
        decision = self.placement_engine.evaluate_placement("W-SHADOW", action, state, mode=ExecutionMode.THROUGHPUT_MODE)

        self.assertEqual(self.migration_arbiter.shadow_sample_interval, 64)
        ok, _, _ = self.migration_arbiter.execute_migration_transaction(decision, workload, state)
        self.assertTrue(ok)
        # Sample interval decreased to 1:8 (increased sampling frequency)
        self.assertEqual(self.migration_arbiter.shadow_sample_interval, 8)

    # 18. Resource arbiter can override fastest tier
    def test_resource_arbiter_can_override_fastest_tier(self):
        # Even in THROUGHPUT_MODE where GPU is fastest, UMA contention overrides GPU and selects CPU
        state = make_test_state(temp_c=55.0, mem_cont=0.85)
        action = {"action_id": "ACT-OVERRIDE"}

        decision = self.placement_engine.evaluate_placement("W-OVERRIDE", action, state, mode=ExecutionMode.THROUGHPUT_MODE)
        self.assertEqual(decision.selected_tier, ExecutionTierID.TIER_1_CPU_VNNI.value)

    # 19. Thermal limit blocks GPU placement
    def test_thermal_limit_blocks_gpu_placement(self):
        # Junction temperature >= 82 C hard blocks GPU
        state = make_test_state(temp_c=83.5)
        action = {"action_id": "ACT-HOT"}

        decision = self.placement_engine.evaluate_placement("W-HOT", action, state, mode=ExecutionMode.THROUGHPUT_MODE)
        self.assertNotEqual(decision.selected_tier, ExecutionTierID.TIER_2_IRIS_XE.value)
        self.assertEqual(decision.selected_tier, ExecutionTierID.TIER_1_CPU_VNNI.value)

    # 20. Same state produces same placement (Determinism)
    def test_same_state_same_placement(self):
        state = make_test_state(temp_c=55.0, mem_cont=0.25)
        action = {"action_id": "ACT-DET"}

        dec1 = self.placement_engine.evaluate_placement("W-DET-1", action, state, mode=ExecutionMode.BALANCED_MODE)
        dec2 = self.placement_engine.evaluate_placement("W-DET-2", action, state, mode=ExecutionMode.BALANCED_MODE)

        self.assertEqual(dec1.selected_tier, dec2.selected_tier)
        self.assertEqual(dec1.predicted_latency_ms, dec2.predicted_latency_ms)
        self.assertEqual(dec1.predicted_throughput_ops, dec2.predicted_throughput_ops)


if __name__ == "__main__":
    unittest.main()
