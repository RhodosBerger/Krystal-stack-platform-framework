#!/usr/bin/env python3
# ==============================================================================
# TESTS FOR FORMAL SILICON PIPELINE ORCHESTRATOR
# SYSTEM MODEL: (S, A, T, I, O, E, H, W)
# ==============================================================================

import unittest
from krystal_kernel.silicon_pipeline_orchestrator import (
    SiliconPipelineOrchestrator,
    SiliconAction,
    SiliconState,
    VITAL_MAX_HP,
    ContractViolation
)
from krystal_kernel.silicon_telemetry_lexicon import PulseSequenceLogger


class TestSiliconPipelineOrchestrator(unittest.TestCase):
    """Verifies formal state machine, Delta S=0 rejection, and pipeline integration."""

    def setUp(self):
        self.logger = PulseSequenceLogger(capacity=1024)
        self.orchestrator = SiliconPipelineOrchestrator(logger=self.logger)

    def test_01_invariant_holds(self):
        """Constitutional invariant must strictly equal 6."""
        self.assertEqual(VITAL_MAX_HP, 6)
        self.assertEqual(self.orchestrator.current_state.vital_max_hp, 6)

    def test_02_action_validation_and_delta_s_zero(self):
        """Invalid actions must be rejected without mutating state (Delta S = 0)."""
        initial_state = self.orchestrator.current_state
        initial_hash = initial_state.provenance_hash

        # Invalid action: negative prompt tokens
        invalid_action = SiliconAction(
            action_type="DISPATCH_INFERENCE",
            prompt_tokens=-5,
            generation_tokens=1
        )
        res = self.orchestrator.step_pipeline(invalid_action)
        self.assertEqual(res["status"], "ACTION_REJECTED")
        self.assertEqual(res["delta_s"], 0)
        self.assertEqual(self.orchestrator.current_state.step, 0)
        self.assertEqual(self.orchestrator.current_state.provenance_hash, initial_hash)

        # Invalid action: PL1 out of bounds
        invalid_pl1 = SiliconAction(
            action_type="ARBITRATE_POLICY",
            target_pl1_watts=95.0,
            prompt_tokens=32
        )
        res2 = self.orchestrator.step_pipeline(invalid_pl1)
        self.assertEqual(res2["status"], "ACTION_REJECTED")
        self.assertEqual(res2["delta_s"], 0)

    def test_03_valid_step_transition_and_provenance(self):
        """Valid actions commit state transitions S_t -> S_{t+1} with chained provenance hashes."""
        valid_action = SiliconAction(
            action_type="DISPATCH_INFERENCE",
            prompt_tokens=32,
            generation_tokens=1
        )
        res = self.orchestrator.step_pipeline(valid_action, simulated_time_s=1.0, load_power_w=20.0)
        self.assertEqual(res["status"], "TRANSITION_COMMITTED")
        self.assertEqual(res["delta_s"], 1)
        self.assertEqual(self.orchestrator.current_state.step, 1)
        self.assertNotEqual(self.orchestrator.current_state.provenance_hash, "S0_INITIAL_SILICON_FABRIC")
        self.assertEqual(res["vital_max_hp"], 6)

    def test_04_multi_step_thermal_stress_policy_shift(self):
        """Heavy sustained load must elevate temperature and trigger policy arbitration."""
        action = SiliconAction(action_type="DISPATCH_INFERENCE", prompt_tokens=64)
        
        # Step 1: Short low-power pulse (nominal)
        res1 = self.orchestrator.step_pipeline(action, simulated_time_s=0.5, load_power_w=15.0)
        self.assertEqual(res1["arbitration"]["selected_policy"], "HYBRID_PREFILL_DECODE")

        # Step 2: High sustained power over long time window (causing thermal rise)
        res2 = self.orchestrator.step_pipeline(action, simulated_time_s=180.0, load_power_w=45.0)
        self.assertGreater(self.orchestrator.current_state.temperature_c, 80.0)
        self.assertEqual(res2["arbitration"]["selected_policy"], "GPU_ONLY")

    def test_05_observables_projection(self):
        """Observables projection O must yield structured state and pulse timeline."""
        action = SiliconAction(action_type="DISPATCH_INFERENCE", prompt_tokens=16)
        self.orchestrator.step_pipeline(action)

        obs = self.orchestrator.get_observables(pulse_diagram_limit=8)
        self.assertIn("current_state", obs)
        self.assertIn("pulse_diagram", obs)
        self.assertIn("[PULSE_SEQUENCE_TIMELINE]", obs["pulse_diagram"])
        self.assertGreater(obs["total_logged_events"], 0)
        self.assertEqual(obs["vital_max_hp"], 6)


if __name__ == "__main__":
    unittest.main()
