#!/usr/bin/env python3
"""
UNIT TESTS: CRYSTALSEC 2.0 BOUNDED CONFIGURATION SEARCH & OBSERVER EFFECT GUARD
=============================================================================
Verifies:
  1. Adaptive Telemetry Governor:
     - 2 Hz quiescent baseline.
     - Burst detection when |dH/dt| > theta.
     - Observer Effect Guard: Throttles sampling rate if telemetry overhead > 2.5%.
     - Trade-off matrix: Frequency vs Information Gain vs Overhead.
  2. Bounded Configuration Search:
     - Runs N=5 repetitions per parameter pair.
     - Empirical mapping: config -> effect -> uncertainty -> repeatability.
     - Invariant VITAL_MAX_HP == 6 preservation.
  3. LLM Research Copilot:
     - Formulates competing hypotheses.
     - Deterministic policy gate prevents arbitrary parameter mutations.
=============================================================================
"""

import unittest
import time
import sys
import os

WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

from krystal_kernel.scene_constructor_presets import VITAL_MAX_HP
from krystal_kernel.crystalsec_llm_execution_plane import (
    ControlledVariables,
    GLOBAL_LLM_EXECUTION_PLANE
)
from krystal_kernel.adaptive_telemetry_resolution import (
    AdaptiveTelemetryGovernor,
    TelemetryResolutionMode
)
from krystal_kernel.bounded_configuration_search import (
    BoundedConfigurationExplorer,
    ConfigurationEmpiricalResult
)
from krystal_kernel.llm_research_copilot import (
    LLMResearchCopilot,
    CopilotExperimentProposal
)


class TestCrystalSecV2BoundedSearchAndObserverEffect(unittest.TestCase):

    def setUp(self):
        self.governor = AdaptiveTelemetryGovernor(
            baseline_hz=2.0,
            burst_hz=100.0,
            cooldown_hz=10.0,
            theta_threshold=5.0,
            max_observer_overhead_pct=2.5
        )
        self.explorer = BoundedConfigurationExplorer()
        self.copilot = LLMResearchCopilot()

    def test_vital_max_hp_invariant(self):
        """Invariant: VITAL_MAX_HP must equal exactly 6."""
        self.assertEqual(VITAL_MAX_HP, 6)

    def test_adaptive_telemetry_baseline_and_burst(self):
        """Governor starts at 2 Hz quiescent baseline, bursts on sharp transient."""
        self.assertEqual(self.governor.current_mode, TelemetryResolutionMode.QUIESCENT_BASELINE)
        self.assertEqual(self.governor.current_sampling_rate_hz, 2.0)

        # Stable sample
        tick1 = self.governor.ingest_telemetry_tick(
            {"latency_ms": 20.0, "temperature_c": 45.0, "cpu_pressure": 0.3},
            timestamp=1000.0
        )
        self.assertEqual(tick1["mode"], "QUIESCENT_BASELINE")

        # Sharp spike in latency (20ms -> 180ms in 0.1s => dH/dt >> 5.0)
        tick2 = self.governor.ingest_telemetry_tick(
            {"latency_ms": 180.0, "temperature_c": 45.0, "cpu_pressure": 0.3},
            timestamp=1000.1
        )
        self.assertEqual(tick2["mode"], "BURST_TRANSIENT_CAPTURE")
        self.assertGreater(tick2["active_sampling_rate_hz"], 10.0)
        self.assertTrue(tick2["is_burst_active"])

    def test_observer_effect_guard_throttles_excessive_overhead(self):
        """If telemetry processing takes too long relative to sampling rate, governor throttles Hz."""
        # Baseline tick
        self.governor.ingest_telemetry_tick(
            {"latency_ms": 20.0, "temperature_c": 45.0, "cpu_pressure": 0.3},
            timestamp=1000.0
        )
        # Spike into burst mode (100 Hz)
        self.governor.ingest_telemetry_tick(
            {"latency_ms": 250.0, "temperature_c": 45.0, "cpu_pressure": 0.3},
            timestamp=1000.1
        )
        self.assertEqual(self.governor.current_mode, TelemetryResolutionMode.BURST_TRANSIENT_CAPTURE)

        # Artificially inject high sample cost to simulate heavy metric collection
        self.governor.empirical_sample_cost_sec = 0.001  # 1ms per sample
        # Next tick in burst mode: 100 Hz * ~0.9ms = ~9% overhead > 2.5% max
        tick = self.governor.ingest_telemetry_tick(
            {"latency_ms": 255.0, "temperature_c": 46.0, "cpu_pressure": 0.35},
            timestamp=1000.2
        )
        # Should be throttled so overhead <= 2.6%
        self.assertTrue(tick["is_observer_throttled"])
        self.assertLessEqual(tick["observer_overhead_pct"], 2.6)
        self.assertLess(tick["active_sampling_rate_hz"], 100.0)

    def test_observer_tradeoff_matrix(self):
        """Governor generates trade-off matrix mapping frequency to overhead and info gain."""
        matrix = self.governor.evaluate_observer_tradeoff_matrix([2.0, 10.0, 50.0, 100.0])
        self.assertEqual(len(matrix), 4)

        for row in matrix:
            self.assertIn("frequency_hz", row)
            self.assertIn("projected_overhead_pct", row)
            self.assertIn("normalized_info_gain", row)
            self.assertIn("recommendation", row)
            self.assertGreater(row["normalized_info_gain"], 0.0)

    def test_bounded_configuration_repetitions_and_repeatability(self):
        """Milestone 3: Evaluates configuration across 5 repetitions with empirical uncertainty."""
        config = ControlledVariables(
            threads=4,
            streams=2,
            cpu_affinity=[0, 1, 2, 3],
            batch_size=1
        )
        result = self.explorer.evaluate_configuration_repetitions(
            config,
            repetitions=5,
            workload_prompt="SURGICAL_INTERVENTION_TEST_PROMPT"
        )
        self.assertIsInstance(result, ConfigurationEmpiricalResult)
        self.assertEqual(result.repetitions_count, 5)
        self.assertGreater(result.mean_ttft_ms, 0.0)
        self.assertGreater(result.mean_throughput_tps, 0.0)
        self.assertGreaterEqual(result.repeatability_pct, 0.0)
        self.assertLessEqual(result.repeatability_pct, 100.0)
        self.assertGreater(result.mean_utility_u, 0.0)
        self.assertIn("Repeatability=", result.causal_signature_summary)

    def test_bounded_search_map_generation(self):
        """Generates empirical map for bounded pairs: (threads, streams)."""
        map_results = self.explorer.generate_bounded_search_map(
            candidate_thread_counts=[2, 4],
            candidate_stream_counts=[1, 2],
            repetitions_per_point=3
        )
        self.assertEqual(len(map_results), 4)
        for res in map_results:
            self.assertIn("threads", res.controlled_parameters)
            self.assertIn("streams", res.controlled_parameters)
            self.assertIn(res.config_id, self.explorer.empirical_map)

    def test_copilot_hypothesis_formulation_and_safety_gate(self):
        """LLM Copilot proposes experiments that must pass deterministic gate."""
        # Run a real execution trace
        trace = GLOBAL_LLM_EXECUTION_PLANE.execute_and_trace_command(
            prompt="SELECT * FROM cnc_thermal_telemetry WHERE rpm > 12000",
            target_action="COGNITIVE_COURIER_QUERY"
        )
        # Reason over footprint
        analysis = self.copilot.reason_over_telemetry(
            footprint=trace.footprint,
            current_config=trace.applied_config
        )
        self.assertIn("competing_hypotheses", analysis)
        self.assertGreaterEqual(len(analysis["competing_hypotheses"]), 1)

        # Formulate proposal
        proposal = self.copilot.propose_experiment(
            analysis,
            variable_to_perturb="threads",
            proposed_value=8
        )
        # Evaluate proposal through deterministic gate
        evaluation = self.copilot.evaluate_proposal_safety(proposal)
        self.assertTrue(evaluation["is_allowed"])
        self.assertEqual(evaluation["status"], "ACCEPTED_FOR_EXPLORATION")

        # Propose forbidden variable outside ALLOWED_CONTROLLED_VARIABLES
        forbidden_prop = CopilotExperimentProposal(
            proposal_id="PROP-FORBIDDEN",
            hypothesis_id="HYP-01",
            variable_to_perturb="kernel_bypass_voltage",
            current_value=1.2,
            proposed_value=2.4,
            expected_impact="Overclocking",
            falsification_condition="System crash"
        )
        eval_forbidden = self.copilot.evaluate_proposal_safety(forbidden_prop)
        self.assertFalse(eval_forbidden["is_allowed"])
        self.assertEqual(eval_forbidden["status"], "REJECTED_OUT_OF_BOUNDS")


if __name__ == "__main__":
    unittest.main()
