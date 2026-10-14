#!/usr/bin/env python3
"""
==============================================================================
KRYSTAL-STACK TEST SUITE: CUSTOM GOVERNOR & WORKLOAD PLANNER (V2)
==============================================================================
Validates:
  1. Strict constitutional invariant VITAL_MAX_HP == 6 across all state classes.
  2. Three-tier battery bands (CRITICAL < 15%, CONSERVATION 15-20%, NORMAL >= 20%, PLUGGED).
  3. Asymmetric thermal hysteresis (enter 88°C / exit 82°C, emergency enter 95°C / exit 85°C).
  4. Proof-carrying GovernorDecision and immutable ExecutionEnvelope contract.
  5. Capability-gated backend dispatch with deterministic verification fallback.
  6. Strict shadow task isolation (shadow_only=True, zero side-effects on primary).
  7. Two-layer execution feedback loop (residuals -> EWMA variance -> contextual risk).
  8. End-to-end vertical control convergence scenario.
==============================================================================
"""

import os
import sys
import unittest
import time

# Ensure repository root is on sys.path
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from krystal_kernel.scene_constructor_presets import VITAL_MAX_HP
from krystal_kernel.custom_governor import (
    CustomHardwareGovernor,
    GovernorMode,
    GovernorLimits,
    GovernorDecision,
    ExecutionEnvelope,
    BatteryBand,
    ThermalBand,
)
from krystal_kernel.custom_planner import (
    CustomWorkloadPlanner,
    WorkloadTask,
    TaskCategory,
    WorkloadPlan,
    ScheduledTaskAssignment,
    ExecutionFeedbackRecord,
)
from krystal_kernel.verification.capability_registry import (
    CapabilityRegistry,
    HardwareBackend,
    CapabilityState,
)
from krystal_kernel.verification.config_catalog import ConfigProfile


class TestCustomGovernorAndPlanner(unittest.TestCase):

    def setUp(self):
        self.caps = CapabilityRegistry(auto_detect=True)
        # Ensure deterministic and NPU backends are verified for unit testing
        self.caps.set_state(HardwareBackend.CPU_SIMD_DETERMINISTIC, CapabilityState.VERIFIED)
        self.caps.set_state(HardwareBackend.NPU, CapabilityState.VERIFIED)
        self.caps.set_state(HardwareBackend.OPENVINO_GPU, CapabilityState.VERIFIED)

        self.governor = CustomHardwareGovernor(caps=self.caps)
        self.planner = CustomWorkloadPlanner(governor=self.governor, caps=self.caps)

    def test_01_vital_max_hp_invariant(self):
        """Verifies VITAL_MAX_HP == 6 invariant across all components."""
        self.assertEqual(VITAL_MAX_HP, 6)
        self.assertEqual(self.governor.vital_max_hp, 6)
        self.assertEqual(self.planner.vital_max_hp, 6)

        decision = self.governor.evaluate_telemetry(battery_pct=80.0, plugged=False, temperature_c=50.0)
        self.assertEqual(decision.vital_max_hp, 6)
        self.assertEqual(decision.envelope.vital_max_hp, 6)

        task = WorkloadTask("t1", "test_task", TaskCategory.COMPUTE, 10.0)
        self.assertEqual(task.vital_max_hp, 6)

        plan = self.planner.generate_plan([task], envelope=decision.envelope)
        self.assertEqual(plan.vital_max_hp, 6)

    def test_02_three_tier_battery_bands_and_plugged(self):
        """Validates transitions across CRITICAL (<15%), CONSERVATION (15-20%), NORMAL (>=20%), and PLUGGED."""
        cur_t = [1000.0]
        fake_clock = lambda: cur_t[0]
        gov = CustomHardwareGovernor(caps=self.caps, clock=fake_clock)

        # 1. Critical Battery (< 15%) -> immediate ultra_saver, bypasses dwell
        d_crit = gov.evaluate_telemetry(battery_pct=12.0, plugged=False, temperature_c=50.0)
        self.assertEqual(d_crit.battery_state, BatteryBand.CRITICAL)
        self.assertEqual(d_crit.selected_profile.name, "ultra_saver")
        self.assertTrue(d_crit.hysteresis_bypassed)

        # 2. Conservation Battery (15% - 20%) -> constrained battery pool
        cur_t[0] += 5.0  # past dwell window
        d_cons = gov.evaluate_telemetry(battery_pct=18.0, plugged=False, temperature_c=50.0)
        self.assertEqual(d_cons.battery_state, BatteryBand.CONSERVATION)
        self.assertEqual(d_cons.selected_profile.category, "battery")

        # 3. Normal Battery (>= 20%) -> balanced pool (e.g. balanced_npu)
        cur_t[0] += 5.0
        d_norm = gov.evaluate_telemetry(battery_pct=80.0, plugged=False, temperature_c=50.0)
        self.assertEqual(d_norm.battery_state, BatteryBand.NORMAL)
        self.assertEqual(d_norm.selected_profile.name, "balanced_npu")

        # 4. Plugged AC with high compute demand -> performance_gpu
        cur_t[0] += 5.0
        d_plugged = gov.evaluate_telemetry(battery_pct=50.0, plugged=True, temperature_c=55.0, cpu_utilization_pct=85.0)
        self.assertEqual(d_plugged.battery_state, BatteryBand.PLUGGED)
        self.assertEqual(d_plugged.selected_profile.name, "performance_gpu")
        self.assertTrue(d_plugged.selected_profile.gpu_enabled)

    def test_03_asymmetric_thermal_hysteresis_and_emergency(self):
        """Validates enter/exit asymmetric thresholds (enter 88°C / exit 82°C, emergency 95°C / 85°C)."""
        cur_t = [2000.0]
        fake_clock = lambda: cur_t[0]
        gov = CustomHardwareGovernor(caps=self.caps, clock=fake_clock)

        # 1. Temp reaches 89°C (>= 88°C enter) -> enters THROTTLED
        cur_t[0] += 5.0
        d1 = gov.evaluate_telemetry(battery_pct=90.0, plugged=True, temperature_c=89.0)
        self.assertEqual(d1.thermal_state, ThermalBand.THROTTLED)

        # 2. Temp drops to 85°C (still >= 82°C exit) -> stays THROTTLED (asymmetry)
        cur_t[0] += 5.0
        d2 = gov.evaluate_telemetry(battery_pct=90.0, plugged=True, temperature_c=85.0)
        self.assertEqual(d2.thermal_state, ThermalBand.THROTTLED)

        # 3. Temp drops to 80°C (< 82°C exit) -> returns to NOMINAL
        cur_t[0] += 5.0
        d3 = gov.evaluate_telemetry(battery_pct=90.0, plugged=True, temperature_c=80.0)
        self.assertEqual(d3.thermal_state, ThermalBand.NOMINAL)

        # 4. Emergency spike to 96°C (>= 95°C) -> enters EMERGENCY, forces ultra_saver, bypasses dwell
        d_emerg = gov.evaluate_telemetry(battery_pct=90.0, plugged=True, temperature_c=96.0)
        self.assertEqual(d_emerg.thermal_state, ThermalBand.EMERGENCY)
        self.assertEqual(d_emerg.governor_mode, GovernorMode.EMERGENCY_COOL)
        self.assertEqual(d_emerg.selected_profile.name, "ultra_saver")
        self.assertTrue(d_emerg.hysteresis_bypassed)

    def test_04_proof_carrying_governor_decision(self):
        """Verifies proof-carrying decision attributes, criteria results, and derived inputs."""
        dec = self.governor.evaluate_telemetry(
            battery_pct=75.0, plugged=False, temperature_c=65.0, uma_pressure=0.35
        )
        self.assertIn("OBS_TEMP_C", dec.observed_inputs)
        self.assertIn("DER_THERMAL_HEADROOM", dec.derived_inputs)
        self.assertIn("thermal_safety", dec.criteria_results)
        self.assertTrue(dec.criteria_results["thermal_safety"]["satisfied"])
        self.assertIsInstance(dec.envelope, ExecutionEnvelope)
        self.assertGreater(dec.envelope.max_workers, 0)
        self.assertIn(dec.envelope.codec_width_bytes, (16, 32, 64))

    def test_05_execution_envelope_to_planner(self):
        """Verifies that Planner strictly operates through an immutable ExecutionEnvelope."""
        envelope = ExecutionEnvelope(
            governor_mode="BALANCED",
            allowed_backends=("CPU_SCALAR", "CPU_SIMD_DETERMINISTIC", "NPU_OFFLOAD"),
            max_workers=4,
            max_power_w=8.0,
            max_tick_hz=30,
            codec_width_bytes=32,
            thermal_budget_c=20.0,
            memory_pressure_limit=0.80,
            valid_until_ns=time.monotonic_ns() + 1_000_000_000,
            state_version=42,
            policy_version="karaf_v2"
        )
        tasks = [
            WorkloadTask("t1", "task_one", TaskCategory.COMPUTE, 15.0),
            WorkloadTask("t2", "task_two", TaskCategory.INFERENCE, 30.0, dependencies=["t1"]),
        ]
        plan = self.planner.generate_plan(tasks, envelope=envelope)
        self.assertEqual(plan.envelope_state_version, 42)
        self.assertEqual(plan.concurrency_limit, 4)
        self.assertEqual(plan.envelope_summary["codec_width_bytes"], 32)
        self.assertEqual(plan.critical_path, ["t1", "t2"])

    def test_06_capability_gated_dispatch_and_verification_contract(self):
        """Verifies abstract verification contract and capability-gated fallback."""
        tasks = [
            WorkloadTask("v1", "verify_roots", TaskCategory.VERIFICATION, 10.0),
            WorkloadTask("inf1", "embed_query", TaskCategory.INFERENCE, 20.0),
        ]
        decision = self.governor.evaluate_telemetry(battery_pct=80.0, plugged=False, temperature_c=50.0)
        plan = self.planner.generate_plan(tasks, envelope=decision.envelope, include_shadow_experiment=False)

        assign_map = {a.task_id: a.target_backend for a in plan.assignments}
        # Verification MUST resolve to CPU_SIMD_DETERMINISTIC
        self.assertEqual(assign_map["v1"], "CPU_SIMD_DETERMINISTIC")
        # Inference resolves to NPU because NPU is verified
        self.assertEqual(assign_map["inf1"], "NPU")

        # Now simulate NPU becoming UNAVAILABLE (e.g. driver fault)
        self.caps.set_state(HardwareBackend.NPU, CapabilityState.UNAVAILABLE)
        plan_fallback = self.planner.generate_plan(tasks, envelope=decision.envelope, include_shadow_experiment=False)
        assign_map2 = {a.task_id: a.target_backend for a in plan_fallback.assignments}
        # Must gracefully fallback to CPU_SIMD_DETERMINISTIC
        self.assertEqual(assign_map2["inf1"], "CPU_SIMD_DETERMINISTIC")

    def test_07_isolated_shadow_tasks(self):
        """Validates that shadow candidate tasks are strictly quarantined (shadow_only=True)."""
        tasks = [
            WorkloadTask("main1", "pipeline_compute", TaskCategory.COMPUTE, 40.0),
            WorkloadTask("main2", "pipeline_transform", TaskCategory.COMPUTE, 20.0),
        ]
        decision = self.governor.evaluate_telemetry(battery_pct=80.0, plugged=False, temperature_c=50.0)
        plan = self.planner.generate_plan(tasks, envelope=decision.envelope, include_shadow_experiment=True)

        shadow_assignments = [a for a in plan.assignments if a.is_shadow_candidate]
        self.assertEqual(len(shadow_assignments), 1)
        shadow_task = shadow_assignments[0]
        self.assertTrue(shadow_task.shadow_only)
        self.assertFalse(shadow_task.is_critical_path)
        self.assertTrue(shadow_task.task_id.startswith("shadow_"))

    def test_08_two_layer_execution_feedback(self):
        """Verifies statistical execution feedback, EWMA error, and variance tracking."""
        # Initial feedback
        rec = self.planner.record_execution_feedback(
            task_id="t1",
            backend="NPU",
            predicted_latency_ms=10.0,
            actual_latency_ms=12.0,
            workload_class="INFERENCE"
        )
        self.assertEqual(rec.residual_ms, 2.0)
        self.assertAlmostEqual(rec.relative_error, 0.20, places=2)

        # Repeated high error to trigger statistical drift
        for _ in range(5):
            self.planner.record_execution_feedback(
                task_id="t_err",
                backend="NPU",
                predicted_latency_ms=10.0,
                actual_latency_ms=35.0,
                workload_class="INFERENCE"
            )
        status = self.planner.get_status()
        self.assertGreater(status["ewma_error_ms"], 5.0)
        self.assertGreater(status["error_variance_ms"], 20.0)

    def test_09_end_to_end_vertical_control_scenario(self):
        """
        End-to-End Vertical Control Convergence Scenario:
        1. Battery 82%, Temp 61°C -> balanced_npu envelope
        2. Battery drops to 13% -> ultra_saver envelope, max_workers reduced, 16B codec
        3. Temp jumps to 97°C -> EMERGENCY_COOL envelope, GPU disallowed
        """
        cur_t = [5000.0]
        fake_clock = lambda: cur_t[0]
        gov = CustomHardwareGovernor(caps=self.caps, clock=fake_clock)
        planner = CustomWorkloadPlanner(governor=gov, caps=self.caps)

        # Step 1: Normal operational state (Battery 82%, Temp 61°C)
        d1 = gov.evaluate_telemetry(battery_pct=82.0, plugged=False, temperature_c=61.0)
        self.assertEqual(d1.selected_profile.name, "balanced_npu")
        self.assertEqual(d1.envelope.codec_width_bytes, 32)
        tasks = [
            WorkloadTask("t1", "load", TaskCategory.COMPUTE, 10.0),
            WorkloadTask("t2", "infer", TaskCategory.INFERENCE, 20.0, dependencies=["t1"]),
            WorkloadTask("t3", "verif", TaskCategory.VERIFICATION, 5.0, dependencies=["t2"]),
        ]
        plan1 = planner.generate_plan(tasks, envelope=d1.envelope)
        self.assertEqual(plan1.governor_mode, "AUTO")
        self.assertEqual(plan1.concurrency_limit, d1.envelope.max_workers)

        # Step 2: Battery drops to 13% (Critical Battery)
        cur_t[0] += 5.0
        d2 = gov.evaluate_telemetry(battery_pct=13.0, plugged=False, temperature_c=62.0)
        self.assertEqual(d2.selected_profile.name, "ultra_saver")
        self.assertEqual(d2.envelope.codec_width_bytes, 16)
        self.assertEqual(d2.envelope.max_workers, 2)  # 25% CPU cap -> 2 workers
        plan2 = planner.generate_plan(tasks, envelope=d2.envelope)
        self.assertEqual(plan2.envelope_summary["codec_width_bytes"], 16)
        self.assertEqual(plan2.concurrency_limit, 2)

        # Step 3: Temperature jumps to 97°C (Thermal Emergency)
        d3 = gov.evaluate_telemetry(battery_pct=13.0, plugged=False, temperature_c=97.0)
        self.assertEqual(d3.thermal_state, ThermalBand.EMERGENCY)
        self.assertEqual(d3.governor_mode, GovernorMode.EMERGENCY_COOL)
        self.assertNotIn("GPU_OPENVINO", d3.envelope.allowed_backends)
        self.assertNotIn("GPU_VULKAN", d3.envelope.allowed_backends)


if __name__ == "__main__":
    unittest.main()
