# ==============================================================================
# KRYSTAL-STACK: UNIT & INTEGRATION TESTS FOR TECH STACK METABOLIC GOVERNOR
# ==============================================================================
# Verifies Perception #1 (Industrial Metabolism) & Perception #5 (Brainwaves):
#   - 4 Cognitive Brainwave Phases (ALPHA, BETA, GAMMA, OMEGA)
#   - Thermodynamic Backpressure and Adaptive Fidelity Profiles
#   - Strict enforcement of VITAL_MAX_HP = 6
# ==============================================================================

import unittest
import os
import sys

# Ensure repository root is on sys.path
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from krystal_kernel.metabolic_governor import (
    VITAL_MAX_HP,
    BrainwavePhase,
    MetabolicFidelityProfile,
    SystemMetabolismState,
    MetabolicGovernor,
    PHASE_PROFILES
)


class TestTechStackMetabolismGovernor(unittest.TestCase):

    def setUp(self):
        self.governor = MetabolicGovernor(initial_phase=BrainwavePhase.BETA)

    def test_vital_hp_invariant_clamping(self):
        """Verifies VITAL_MAX_HP = 6 is strictly enforced under any input."""
        state_high = self.governor.evaluate_telemetry(0.5, 0.5, 0.5, 50.0, vital_hp=100)
        self.assertEqual(state_high.vital_hp, VITAL_MAX_HP)

        state_neg = self.governor.evaluate_telemetry(0.5, 0.5, 0.5, 50.0, vital_hp=-5)
        self.assertEqual(state_neg.vital_hp, 0)

        state_valid = self.governor.evaluate_telemetry(0.5, 0.5, 0.5, 50.0, vital_hp=4)
        self.assertEqual(state_valid.vital_hp, 4)

    def test_alpha_idle_consolidation_transition(self):
        """Low visual entropy and low CPU usage transition to ALPHA phase."""
        state = self.governor.evaluate_telemetry(
            cpu_utilization=0.10,
            vram_pressure=0.20,
            visual_entropy=0.25,
            thermal_temp_c=42.0,
            vital_hp=6
        )
        self.assertEqual(state.active_phase, BrainwavePhase.ALPHA)
        self.assertEqual(state.fidelity.symbol, "≈")
        self.assertEqual(state.fidelity.raymarch_steps, 0)
        self.assertEqual(state.fidelity.terrain_octaves, 0)
        self.assertEqual(state.fidelity.glyph_mode, "quiescent")
        self.assertFalse(state.is_throttled)

    def test_beta_active_flow_transition(self):
        """Moderate load and entropy maintain BETA phase (60 FPS standard)."""
        state = self.governor.evaluate_telemetry(
            cpu_utilization=0.45,
            vram_pressure=0.50,
            visual_entropy=0.45,
            thermal_temp_c=55.0,
            vital_hp=6
        )
        self.assertEqual(state.active_phase, BrainwavePhase.BETA)
        self.assertEqual(state.fidelity.symbol, "::")
        self.assertEqual(state.fidelity.target_fps, 60)
        self.assertEqual(state.fidelity.raymarch_steps, 20)
        self.assertEqual(state.fidelity.terrain_octaves, 4)
        self.assertEqual(state.fidelity.glyph_mode, "standard")

    def test_gamma_hyper_focus_transition(self):
        """High visual entropy (> 0.55) or VRAM pressure (> 0.75) shifts to GAMMA phase."""
        state = self.governor.evaluate_telemetry(
            cpu_utilization=0.75,
            vram_pressure=0.82,
            visual_entropy=0.62,
            thermal_temp_c=68.0,
            vital_hp=6
        )
        self.assertEqual(state.active_phase, BrainwavePhase.GAMMA)
        self.assertEqual(state.fidelity.symbol, "⚡")
        self.assertEqual(state.fidelity.target_fps, 120)
        self.assertEqual(state.fidelity.raymarch_steps, 32)
        self.assertEqual(state.fidelity.terrain_octaves, 6)
        self.assertEqual(state.fidelity.glyph_mode, "full_blocks")

    def test_omega_emergency_backpressure_on_thermal_spike(self):
        """Thermal temp >= 85°C triggers immediate OMEGA phase and fidelity throttling."""
        state = self.governor.evaluate_telemetry(
            cpu_utilization=0.90,
            vram_pressure=0.90,
            visual_entropy=0.60,
            thermal_temp_c=87.5, # Exceeds 85°C limit
            vital_hp=6
        )
        self.assertEqual(state.active_phase, BrainwavePhase.OMEGA)
        self.assertEqual(state.fidelity.symbol, "Ω")
        self.assertTrue(state.is_throttled)
        # Verify fidelity was stepped down to prevent thermal shutdown
        self.assertEqual(state.fidelity.raymarch_steps, 10)
        self.assertEqual(state.fidelity.terrain_octaves, 2)
        self.assertEqual(state.fidelity.glyph_mode, "simple_wireframe")

    def test_omega_emergency_backpressure_on_entropy_spike(self):
        """Critical visual entropy (> 0.70) triggers OMEGA phase."""
        state = self.governor.evaluate_telemetry(
            cpu_utilization=0.60,
            vram_pressure=0.60,
            visual_entropy=0.78, # Exceeds 0.70 limit
            thermal_temp_c=60.0,
            vital_hp=6
        )
        self.assertEqual(state.active_phase, BrainwavePhase.OMEGA)
        self.assertTrue(state.is_throttled)

    def test_transition_count_and_history_buffering(self):
        """Verifies state transition telemetry and ring buffer capping."""
        gov = MetabolicGovernor(initial_phase=BrainwavePhase.ALPHA)
        self.assertEqual(gov.transition_count, 0)

        # Force transition ALPHA -> GAMMA
        gov.evaluate_telemetry(0.8, 0.8, 0.65, 50.0, 6)
        self.assertEqual(gov.active_phase, BrainwavePhase.GAMMA)
        self.assertEqual(gov.transition_count, 1)

        # Force transition GAMMA -> OMEGA
        gov.evaluate_telemetry(0.8, 0.8, 0.85, 90.0, 6)
        self.assertEqual(gov.active_phase, BrainwavePhase.OMEGA)
        self.assertEqual(gov.transition_count, 2)

        # Ensure history retains states
        self.assertEqual(len(gov.history), 2)
        d = gov.history[-1].to_dict()
        self.assertEqual(d["active_phase"], "OMEGA")
        self.assertEqual(d["vital_hp"], 6)


if __name__ == "__main__":
    unittest.main()
