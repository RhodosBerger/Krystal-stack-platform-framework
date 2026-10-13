#!/usr/bin/env python3
"""
==============================================================================
KRYSTAL-STACK: UNIT TEST SUITE FOR HARDWARE LAYERS, TIMERS & COORDINATION
==============================================================================
File: tests/test_hardware_layers_and_timer_coordination.py
Description:
    Validates:
      1. Constitutional Invariant: VITAL_MAX_HP == 6 across all state containers.
      2. Windows QPC monotonic clock accuracy, frequency, and elapsed latency.
      3. Windows Scheduler Resolution (winmm timeBeginPeriod/timeEndPeriod scope).
      4. High-Resolution Waitable Timer precision and jitter analysis.
      5. Multi-domain Hardware Management Layers (CPU, GPU, NPU, Power/Thermal).
      6. Win32 CPU affinity mask clamping and process priority class validation.
      7. Estimation Task DAG generation, stage dependencies, and microsecond timings.
      8. Coordination Template System dispatch, MMCSS profile, and shadow execution.
==============================================================================
"""

import sys
import unittest

from krystal_kernel.scene_constructor_presets import VITAL_MAX_HP
from krystal_kernel.verification.semantic_protocol import (
    ActuationState,
    ObservationState,
    DerivedState,
    PredictionState,
    SafetyState,
    AuthorityState,
    TransactionState,
    InvariantState,
    HardwareState,
    to_wire_token
)
from krystal_kernel.windows_timer_subsystem import (
    WindowsQpcClock,
    WindowsSchedulerResolution,
    WindowsMmcssCoordinator,
    WindowsHighResolutionWaitableTimer,
    TimerJitterTelemetryAnalyzer,
    WindowsTimerSubsystem,
    GLOBAL_WINDOWS_TIMER_SUBSYSTEM
)
from krystal_kernel.hardware_layers import (
    HardwareDomainKind,
    CpuHardwareLayer,
    GpuHardwareLayer,
    NpuHardwareLayer,
    ThermalPowerHardwareLayer,
    HardwareSubsystemCoordinator,
    GLOBAL_HARDWARE_COORDINATOR
)
from krystal_kernel.estimation_task_planner import (
    TaskStageKind,
    EstimationTask,
    EstimationTaskPlan,
    build_canonical_estimation_dag
)
from krystal_kernel.coordination_templates import (
    CoordinationTemplateId,
    CoordinationTemplate,
    CANONICAL_TEMPLATES,
    CoordinationTemplateEngine,
    GLOBAL_COORDINATION_TEMPLATE_ENGINE
)


class TestHardwareLayersAndTimerCoordination(unittest.TestCase):
    """Test suite for Hardware Layers, Windows API Timers, and Coordination Templates."""

    def test_01_constitutional_invariant_vital_6hp(self):
        """Invariant: VITAL_MAX_HP must strictly equal 6 across all components."""
        self.assertEqual(VITAL_MAX_HP, 6)

        hw_status = GLOBAL_HARDWARE_COORDINATOR.get_full_status()
        self.assertEqual(hw_status["vital_max_hp"], 6)

        timer_telemetry = GLOBAL_WINDOWS_TIMER_SUBSYSTEM.get_telemetry()
        self.assertEqual(timer_telemetry["vital_max_hp"], 6)

        template = CANONICAL_TEMPLATES[CoordinationTemplateId.TEMPLATE_BALANCED_ADAPTIVE_CADENCE]
        self.assertEqual(template.vital_max_hp, 6)

    def test_02_windows_qpc_clock(self):
        """Windows QPC Clock must report positive frequency and monotonic elapsed time."""
        clock = WindowsQpcClock()
        self.assertGreater(clock.frequency_hz, 0)

        t0_ticks = clock.now_ticks()
        t0_ns = clock.now_ns()
        self.assertGreater(t0_ticks, 0)
        self.assertGreater(t0_ns, 0)

        # Busy wait 500 microseconds
        target_ns = clock.now_ns() + 500_000
        while clock.now_ns() < target_ns:
            pass

        elapsed_us = clock.elapsed_us(t0_ticks)
        self.assertGreaterEqual(elapsed_us, 400.0)

    def test_03_windows_scheduler_resolution(self):
        """Scheduler resolution must report min/max capabilities and support scoped periods."""
        sched = WindowsSchedulerResolution()
        caps = sched.get_capabilities()
        self.assertGreaterEqual(caps.min_period_ms, 1)
        self.assertLessEqual(caps.min_period_ms, caps.max_period_ms)

        with sched.scoped_period(1) as active_ms:
            self.assertEqual(active_ms, 1)
            self.assertEqual(sched.active_period_ms, 1)

        self.assertIsNone(sched.active_period_ms)

    def test_04_windows_waitable_timer_and_jitter(self):
        """Waitable timer must pause accurately and compute jitter statistics."""
        timer_subsystem = WindowsTimerSubsystem()
        report = timer_subsystem.benchmark_jitter(target_ms=0.5, count=4)

        self.assertEqual(report.sample_count, 4)
        self.assertAlmostEqual(report.target_period_us, 500.0, places=1)
        self.assertGreater(report.mean_period_us, 0.0)
        self.assertGreaterEqual(report.jitter_std_dev_us, 0.0)
        self.assertEqual(report.vital_max_hp, 6)
        self.assertTrue(report.observation_status.startswith("obs:"))

    def test_05_cpu_hardware_layer_affinity_and_priority(self):
        """CPU layer must clamp affinity mask and set valid priority classes."""
        cpu = CpuHardwareLayer()
        self.assertGreater(cpu.total_cores, 0)
        self.assertGreater(cpu.p_core_mask, 0)

        # Setting 0 mask must safely default to Core 0 (0x0001)
        cpu.set_affinity_mask(0)
        self.assertEqual(cpu.active_affinity_mask, 1)

        # Setting valid mask (e.g. 0x000F)
        mask = min(0x000F, (1 << cpu.total_cores) - 1)
        ok = cpu.set_affinity_mask(mask)
        self.assertTrue(ok)
        self.assertEqual(cpu.active_affinity_mask, mask)

        # Setting process priority class
        ok_pri = cpu.set_priority_class("HIGH_PRIORITY_CLASS")
        self.assertTrue(ok_pri)
        self.assertEqual(cpu.active_priority_class, "HIGH_PRIORITY_CLASS")

        status = cpu.get_status()
        self.assertEqual(status.hardware_status, to_wire_token(HardwareState.HW_NOMINAL))

    def test_06_gpu_and_npu_hardware_layers(self):
        """GPU and NPU layers must report status and handle stream configurations."""
        gpu = GpuHardwareLayer()
        self.assertTrue(gpu.allocate_vram_slab(256.0))
        self.assertEqual(gpu.vram_allocated_mb, 768.0)
        self.assertTrue(gpu.release_vram_slab(256.0))
        self.assertEqual(gpu.vram_allocated_mb, 512.0)

        npu = NpuHardwareLayer()
        self.assertTrue(npu.configure_streams(stream_count=4, dp4a_enabled=True))
        self.assertEqual(npu.active_streams, 4)
        self.assertTrue(npu.dp4a_vector_mode)

        npu_status = npu.get_status()
        self.assertEqual(npu_status.active_streams, 4)

    def test_07_thermal_power_hardware_layer(self):
        """Thermal/Power layer must derive thermal headroom and clamp PL1."""
        power = ThermalPowerHardwareLayer()
        power.set_pl1_clamp(28.0)
        self.assertEqual(power.pl1_clamp_w, 28.0)

        power.update_telemetry(temp_c=72.0, pkg_watts=22.5)
        status = power.get_status()
        self.assertEqual(status.temperature_c, 72.0)
        self.assertEqual(status.thermal_headroom_c, 28.0)
        self.assertFalse(status.thermal_throttling_engaged)

        # Trigger emergency throttle test
        power.update_telemetry(temp_c=92.0, pkg_watts=45.0)
        status_hot = power.get_status()
        self.assertTrue(status_hot.thermal_throttling_engaged)
        self.assertEqual(status_hot.hardware_status, to_wire_token(HardwareState.HW_THROTTLED_THERMAL))

    def test_08_estimation_task_dag_execution(self):
        """Estimation task DAG must execute 6 stages in order with sub-millisecond tracking."""
        dag = build_canonical_estimation_dag(
            plan_id="TEST_DAG_01",
            template_id="TEMPLATE_BALANCED_ADAPTIVE_CADENCE",
            plan_bias=0.45,
            affinity_mask=0x00FF,
            priority_class="NORMAL_PRIORITY_CLASS",
            pl1_watts=24.0,
            npu_streams=2,
            is_shadow_only=False
        )

        report = dag.execute()
        self.assertEqual(report.total_tasks, 6)
        self.assertEqual(report.executed_tasks, 6)
        self.assertGreater(report.total_latency_us, 0.0)
        self.assertEqual(report.safety_verdict, to_wire_token(SafetyState.SAFE_PASS))
        self.assertEqual(report.authority_verdict, to_wire_token(AuthorityState.AUTH_COMMIT_APPROVED))
        self.assertEqual(report.transaction_state, to_wire_token(TransactionState.TXN_COMMITTED))

        # Check each task has executed and measured latency
        for task in report.tasks:
            self.assertEqual(task["state"], to_wire_token(ActuationState.ACT_EXECUTED))
            self.assertGreaterEqual(task["actual_latency_us"], 0.0)

    def test_09_coordination_template_engine_dispatch(self):
        """CoordinationTemplateEngine must dispatch templates, configure timers & HW, and run DAG."""
        engine = CoordinationTemplateEngine()
        templates = engine.list_templates()
        self.assertEqual(len(templates), 5)

        # Dispatch TEMPLATE_ULTRA_LOW_JITTER_REALTIME
        report_rt = engine.dispatch_template(
            CoordinationTemplateId.TEMPLATE_ULTRA_LOW_JITTER_REALTIME,
            cadence_ticks=2
        )
        self.assertEqual(report_rt.vital_max_hp, 6)
        self.assertEqual(report_rt.template_id, CoordinationTemplateId.TEMPLATE_ULTRA_LOW_JITTER_REALTIME.value)
        self.assertGreater(report_rt.execution_duration_us, 0.0)
        self.assertEqual(report_rt.task_dag_report["executed_tasks"], 6)

        # Dispatch TEMPLATE_COUNTERFACTUAL_SHADOW_BURST
        report_shadow = engine.dispatch_template(
            CoordinationTemplateId.TEMPLATE_COUNTERFACTUAL_SHADOW_BURST,
            cadence_ticks=2
        )
        self.assertEqual(report_shadow.task_dag_report["authority_verdict"], to_wire_token(AuthorityState.AUTH_SHADOW_ONLY))
        self.assertEqual(report_shadow.task_dag_report["transaction_state"], to_wire_token(TransactionState.TXN_ABORTED))


if __name__ == "__main__":
    unittest.main()
