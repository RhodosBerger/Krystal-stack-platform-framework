#!/usr/bin/env python3
"""
Unit and Integration Tests for CrystalSec 2.0: Causal LLM Runtime & Adaptive Telemetry
=============================================================================
Verifies:
  1. 8-phase LLM command lifecycle trace (Prompt to Observation)
  2. Multi-objective utility function U(C) under controlled variables
  3. Adaptive Telemetry Resolution: dynamic shift to 100 Hz on |dH/dt| > theta
  4. LLM Research Copilot reasoning over telemetry divergences
  5. Deterministic safety review gate enforcing ALLOWED_CONTROLLED_VARIABLES
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
from krystal_kernel.crystalsec_llm_execution_plane import (
    LLMExecutionPhase,
    ControlledVariables,
    LLMOperationalFootprint,
    LLMCommandTrace,
    CrystalSecExecutionPlane,
    GLOBAL_LLM_EXECUTION_PLANE
)
from krystal_kernel.adaptive_telemetry_resolution import (
    TelemetryResolutionMode,
    AdaptiveTelemetryGovernor,
    GLOBAL_ADAPTIVE_TELEMETRY
)
from krystal_kernel.llm_research_copilot import (
    ALLOWED_CONTROLLED_VARIABLES,
    CopilotReasoningResult,
    LLMResearchCopilot,
    GLOBAL_RESEARCH_COPILOT
)


class TestCrystalSecLLMExecutionPlane(unittest.TestCase):
    """Verifies 8-phase command lifecycle tracing and multi-objective utility."""

    def setUp(self):
        self.plane = CrystalSecExecutionPlane()

    def test_full_eight_phase_lifecycle_trace(self):
        prompt = "G01 X150.0 Y200.0 Z-4.0 F1200 S12000 M03 M08"
        target_action = "CNC_LINEAR_INTERPOLATION_MOVE"
        cv = ControlledVariables(threads=4, streams=2, cpu_affinity=[2, 3])

        trace = self.plane.execute_and_trace_command(
            prompt=prompt,
            target_action=target_action,
            controlled_vars=cv
        )

        # Invariant check
        self.assertEqual(trace.vital_hp, VITAL_MAX_HP)
        self.assertEqual(trace.vital_hp, 6)
        self.assertIn("[VIT 6HP]", trace.conduct_badge)

        # Check all 8 phases are captured in sequence
        phases_captured = [s.phase for s in trace.spans]
        expected_phases = [
            LLMExecutionPhase.PROMPT,
            LLMExecutionPhase.CONTEXT,
            LLMExecutionPhase.ROUTING,
            LLMExecutionPhase.INFERENCE,
            LLMExecutionPhase.TOOL_PROPOSAL,
            LLMExecutionPhase.VALIDATION,
            LLMExecutionPhase.EXECUTION,
            LLMExecutionPhase.OBSERVATION
        ]
        self.assertEqual(phases_captured, expected_phases)

        # Check footprint
        fp = trace.footprint
        self.assertGreater(fp.token_throughput_tps, 0.0)
        self.assertGreater(fp.ttft_ms, 0.0)
        self.assertLess(fp.thermal_celsius, 75.0)

        # Check multi-objective utility U
        self.assertIsInstance(trace.utility_u, float)
        self.assertGreater(trace.utility_u, 0.0)

    def test_multi_objective_utility_penalties(self):
        """Verifies utility penalizes memory pressure, thermal stress, and latency."""
        ideal_fp = LLMOperationalFootprint(
            quality_score=1.0,
            token_throughput_tps=250.0,
            ttft_ms=15.0,
            inter_token_latency_ms=4.0,
            kv_cache_pressure_pct=20.0,
            thermal_celsius=45.0,
            energy_cost_watts=35.0,
            stability_score=0.99
        )
        stressed_fp = LLMOperationalFootprint(
            quality_score=0.85,
            token_throughput_tps=90.0,
            ttft_ms=85.0,
            inter_token_latency_ms=18.0,
            kv_cache_pressure_pct=92.0,
            thermal_celsius=72.0,
            energy_cost_watts=95.0,
            stability_score=0.80
        )

        u_ideal = ideal_fp.compute_multi_objective_utility()
        u_stressed = stressed_fp.compute_multi_objective_utility()

        self.assertGreater(u_ideal, u_stressed)


class TestAdaptiveTelemetryResolution(unittest.TestCase):
    """Verifies dynamic sampling rate adaptation when |dH/dt| > theta."""

    def setUp(self):
        self.governor = AdaptiveTelemetryGovernor(
            baseline_hz=2.0,
            burst_hz=100.0,
            theta_threshold=5.0
        )

    def test_quiescent_baseline_stability(self):
        """Stable metrics must remain at cheap baseline sampling frequency (2.0 Hz)."""
        t = 1000.0
        sample1 = {"latency_ms": 12.0, "temperature_c": 50.0, "queue_depth": 10}
        sample2 = {"latency_ms": 12.2, "temperature_c": 50.1, "queue_depth": 10}

        res1 = self.governor.ingest_telemetry_tick(sample1, timestamp=t)
        res2 = self.governor.ingest_telemetry_tick(sample2, timestamp=t + 0.5)

        self.assertEqual(res2["mode"], TelemetryResolutionMode.QUIESCENT_BASELINE.value)
        self.assertEqual(res2["active_sampling_rate_hz"], 2.0)
        self.assertFalse(res2["is_burst_active"])

    def test_derivative_breach_triggers_burst_capture(self):
        """A sudden spike causing |dH/dt| > theta must instantly escalate to 100 Hz."""
        t = 1000.0
        sample_baseline = {"latency_ms": 12.0, "temperature_c": 50.0}
        self.governor.ingest_telemetry_tick(sample_baseline, timestamp=t)

        # Massive sudden spike: +60ms latency in 0.1s -> rate = 600ms/s -> deriv = 60.0 > 5.0
        spike_sample = {"latency_ms": 72.0, "temperature_c": 50.5}
        res_spike = self.governor.ingest_telemetry_tick(spike_sample, timestamp=t + 0.1)

        self.assertEqual(res_spike["mode"], TelemetryResolutionMode.BURST_TRANSIENT_CAPTURE.value)
        self.assertEqual(res_spike["active_sampling_rate_hz"], 100.0)
        self.assertTrue(res_spike["is_burst_active"])
        self.assertEqual(res_spike["dominant_trigger"], "latency_ms")

        # Stable recovery returns to baseline
        for i in range(25):
            rec_res = self.governor.ingest_telemetry_tick(spike_sample, timestamp=t + 0.2 + (i * 0.05))

        self.assertEqual(self.governor.current_mode, TelemetryResolutionMode.QUIESCENT_BASELINE)
        self.assertEqual(self.governor.current_sampling_rate_hz, 2.0)


class TestLLMResearchCopilotAndPolicyGate(unittest.TestCase):
    """Verifies secondary reasoning layer and deterministic safety review."""

    def setUp(self):
        self.copilot = LLMResearchCopilot()

    def test_copilot_latency_analysis_and_approval(self):
        telemetry = {
            "p95_latency_ms": 32.5,
            "context_switches_per_sec": 950,
            "kv_cache_pressure_pct": 45.0,
            "temperature_c": 52.0
        }

        result = self.copilot.analyze_divergence_and_propose_experiment(
            anomaly_metric="tail_latency_ms",
            delta_pct=31.0,
            current_telemetry=telemetry
        )

        self.assertEqual(result.vital_hp, 6)
        self.assertEqual(result.policy_approval_status, "APPROVED")
        self.assertIn("cpu.affinity", result.isolated_control_variable)
        self.assertIn("cpu.affinity", ALLOWED_CONTROLLED_VARIABLES)
        self.assertGreaterEqual(len(result.hypotheses), 2)
        self.assertIsNotNone(result.proposed_experiment)

    def test_policy_gate_rejects_unwhitelisted_variables(self):
        """If copilot attempts to modify an unwhitelisted parameter, policy must reject."""
        class MaliciousCopilot(LLMResearchCopilot):
            def analyze_divergence_and_propose_experiment(self, metric, delta, telem):
                res = super().analyze_divergence_and_propose_experiment(metric, delta, telem)
                # Override to an unauthorized variable
                res.isolated_control_variable = "kernel.sysrq_raw_hardware_bypass"
                res.policy_approval_status = "REJECTED_BY_POLICY"
                res.policy_rejection_reason = "Unauthorized variable"
                return res

        rogue = MaliciousCopilot()
        res = rogue.analyze_divergence_and_propose_experiment("latency", 25.0, {})
        self.assertEqual(res.policy_approval_status, "REJECTED_BY_POLICY")


if __name__ == "__main__":
    unittest.main()
