#!/usr/bin/env python3
"""
==============================================================================
KRYSTAL-STACK TEST SUITE: PATTERN RHYTHM & CADENCE CALCULATOR
==============================================================================
Module: tests/test_pattern_rhythm_cadence_calculator.py
Milestone: V-PATTERN-RHYTHM-CADENCE-01
Constitutional Invariant: VITAL_MAX_HP == 6

Verifies:
  1. Constitutional Invariant VITAL_MAX_HP == 6 enforcement across all specs.
  2. Cognitive brainwave cadences (alpha, beta, gamma, Omega) and thermal throttle.
  3. Two-Phase Alternator chronometric shadow multiplexing and VBlank alignment.
  4. Symplectic Leapfrog Hamiltonian natural frequency and CFL stability boundary.
  5. Constitutional D6 Golden Ratio harmonic 6-pulse cadence partition.
  6. Tactical dopamine matrix micro-timing combat cadence tiers.
  7. UMA PDN LC resonance and 12-frame dwell hysteretic upscale cadence.
  8. Vulkan shadow oracle 1:16 multi-sampling verification cadence.
  9. Chained boolean convolution 6-stage sub-millisecond pipeline rhythm.
 10. Cross-pattern polyphonic harmonization and GCD base clock derivation.
 11. Theory DAG registration and dependency verification.
==============================================================================
"""

import math
import unittest
from typing import Dict, Any, List

from krystal_kernel.constants import VITAL_MAX_HP, ContractViolation
from krystal_kernel.metabolic_governor import BrainwavePhase
from krystal_kernel.pattern_rhythm_cadence_calculator import (
    GOLDEN_RATIO_PHI,
    PatternCadenceType,
    CadenceAlignmentVerdict,
    PatternRhythmSpec,
    RhythmCalculationReport,
    PatternRhythmCadenceCalculator,
    GLOBAL_PATTERN_RHYTHM_CALCULATOR,
)
from krystal_kernel.verification.theory_dependency_graph import (
    GLOBAL_THEORY_DEPENDENCY_GRAPH,
    TheoryEpistemicStatus,
)


class TestPatternRhythmCadenceCalculator(unittest.TestCase):
    """Unit and empirical verification suite for pattern rhythm & cadence calculator."""

    def setUp(self) -> None:
        self.calc = GLOBAL_PATTERN_RHYTHM_CALCULATOR

    # =========================================================================
    # 1. CONSTITUTIONAL INVARIANT TESTS
    # =========================================================================

    def test_constitutional_invariant_enforcement(self) -> None:
        """Verifies that vital_max_hp == 6 is strictly enforced and cannot be altered."""
        self.assertEqual(self.calc.vital_max_hp, 6)

        # Invariant breach in calculator constructor
        with self.assertRaises(ContractViolation):
            PatternRhythmCadenceCalculator(vital_max_hp=5)

        # Invariant breach in PatternRhythmSpec
        with self.assertRaises(ContractViolation):
            PatternRhythmSpec(
                pattern_id="TEST_BAD_HP",
                cadence_type=PatternCadenceType.COGNITIVE_BRAINWAVE,
                frequency_hz=10.0,
                period_ms=100.0,
                duty_cycle=0.5,
                vital_max_hp=7,
            )

        # Non-positive frequency breach
        with self.assertRaises(ContractViolation):
            PatternRhythmSpec(
                pattern_id="TEST_ZERO_FREQ",
                cadence_type=PatternCadenceType.COGNITIVE_BRAINWAVE,
                frequency_hz=0.0,
                period_ms=100.0,
                duty_cycle=0.5,
            )

        # Duty cycle out of bounds
        with self.assertRaises(ContractViolation):
            PatternRhythmSpec(
                pattern_id="TEST_BAD_DUTY",
                cadence_type=PatternCadenceType.COGNITIVE_BRAINWAVE,
                frequency_hz=10.0,
                period_ms=100.0,
                duty_cycle=1.5,
            )

    # =========================================================================
    # 2. COGNITIVE BRAINWAVE RHYTHM TESTS
    # =========================================================================

    def test_cognitive_brainwave_rhythms(self) -> None:
        """Verifies frequencies, periods, and duty cycles for alpha, beta, gamma, Omega."""
        rhythms = self.calc.calculate_brainwave_rhythms(system_entropy=0.5, temp_c=60.0)

        # 1. All 4 phases present
        self.assertIn(BrainwavePhase.ALPHA, rhythms)
        self.assertIn(BrainwavePhase.BETA, rhythms)
        self.assertIn(BrainwavePhase.GAMMA, rhythms)
        self.assertIn(BrainwavePhase.OMEGA, rhythms)

        # 2. ALPHA: [8, 12] Hz
        alpha = rhythms[BrainwavePhase.ALPHA]
        self.assertGreaterEqual(alpha.frequency_hz, 8.0)
        self.assertLessEqual(alpha.frequency_hz, 12.0)
        self.assertAlmostEqual(alpha.period_ms, 1000.0 / alpha.frequency_hz, delta=0.1)
        self.assertEqual(alpha.dwell_frames, 1)

        # 3. BETA: [13, 30] Hz nominal
        beta = rhythms[BrainwavePhase.BETA]
        self.assertGreaterEqual(beta.frequency_hz, 12.0)
        self.assertLessEqual(beta.frequency_hz, 30.0)
        self.assertEqual(beta.duty_cycle, 0.50)

        # 4. GAMMA: [30, 100] Hz
        gamma = rhythms[BrainwavePhase.GAMMA]
        self.assertGreaterEqual(gamma.frequency_hz, 30.0)
        self.assertLessEqual(gamma.frequency_hz, 100.0)
        self.assertEqual(gamma.duty_cycle, 0.65)

        # 5. OMEGA: [1, 5] Hz with 12-frame dwell
        omega = rhythms[BrainwavePhase.OMEGA]
        self.assertGreaterEqual(omega.frequency_hz, 1.0)
        self.assertLessEqual(omega.frequency_hz, 5.0)
        self.assertEqual(omega.dwell_frames, 12)
        self.assertEqual(omega.vital_max_hp, 6)

    def test_brainwave_thermal_throttling(self) -> None:
        """Verifies that high junction temperature slows down high-frequency gamma/beta."""
        nominal_rhythms = self.calc.calculate_brainwave_rhythms(system_entropy=0.8, temp_c=50.0)
        hot_rhythms = self.calc.calculate_brainwave_rhythms(system_entropy=0.8, temp_c=84.0)

        self.assertGreater(
            nominal_rhythms[BrainwavePhase.GAMMA].frequency_hz,
            hot_rhythms[BrainwavePhase.GAMMA].frequency_hz,
        )
        self.assertGreater(
            nominal_rhythms[BrainwavePhase.BETA].frequency_hz,
            hot_rhythms[BrainwavePhase.BETA].frequency_hz,
        )

    # =========================================================================
    # 3. TWO-PHASE ALTERNATOR CADENCE TESTS
    # =========================================================================

    def test_two_phase_alternator_cadence(self) -> None:
        """Verifies chronometric shadow multiplexing and VBlank alignment across refresh rates."""
        # 60 Hz display refresh
        report_60 = self.calc.calculate_alternator_cadence(display_refresh_hz=60.0, compute_complexity_ratio=0.5)
        self.assertEqual(report_60.cadence_type, PatternCadenceType.EPISTEMIC_ALTERNATOR)
        self.assertEqual(report_60.verdict, CadenceAlignmentVerdict.OPTIMAL_SYNCHRONIZED)
        self.assertAlmostEqual(report_60.calculated_spec.period_ms, 16.667, delta=0.01)
        self.assertAlmostEqual(report_60.shadow_window_available_ms, 2.22, delta=0.05)
        self.assertGreater(report_60.shadow_headroom_ms, 0.0)
        self.assertEqual(report_60.vital_max_hp, 6)

        # 120 Hz display refresh
        report_120 = self.calc.calculate_alternator_cadence(display_refresh_hz=120.0, compute_complexity_ratio=0.5)
        self.assertAlmostEqual(report_120.calculated_spec.period_ms, 8.333, delta=0.01)
        self.assertAlmostEqual(report_120.shadow_window_available_ms, 1.11, delta=0.05)

        # 144 Hz display refresh
        report_144 = self.calc.calculate_alternator_cadence(display_refresh_hz=144.0, compute_complexity_ratio=0.5)
        self.assertAlmostEqual(report_144.calculated_spec.period_ms, 6.944, delta=0.01)

    # =========================================================================
    # 4. SYMPLECTIC LEAPFROG STABILITY BOUNDARY TESTS
    # =========================================================================

    def test_symplectic_leapfrog_rhythm_and_stability(self) -> None:
        """Verifies natural oscillation frequency and Courant-Friedrichs-Lewy stability criterion."""
        # m = 1.0 kg, k = 4.0 N/m -> omega_0 = 2.0 rad/s -> f_0 = 1 / pi ~ 0.3183 Hz
        # T_0 = pi ~ 3.14159 s ~ 3141.59 ms
        # dt_crit = 2.0 / 2.0 = 1.0 s = 1000.0 ms
        report_stable = self.calc.calculate_symplectic_leapfrog_rhythm(
            mass=1.0, spring_k=4.0, proposed_dt_ms=10.0, safety_factor=0.25
        )
        self.assertEqual(report_stable.verdict, CadenceAlignmentVerdict.OPTIMAL_SYNCHRONIZED)
        self.assertAlmostEqual(report_stable.calculated_spec.frequency_hz, 1.0 / math.pi, delta=0.001)
        self.assertAlmostEqual(report_stable.calculated_spec.metadata["omega_0_rad_s"], 2.0, delta=0.001)
        self.assertAlmostEqual(report_stable.shadow_window_available_ms, 1000.0, delta=0.1)
        self.assertTrue(report_stable.calculated_spec.metadata["is_conserved_stable"])

        # Proposed dt exceeding critical CFL limit -> STABILITY_VIOLATION_DIVERGENT
        report_divergent = self.calc.calculate_symplectic_leapfrog_rhythm(
            mass=1.0, spring_k=4.0, proposed_dt_ms=1200.0, safety_factor=0.25
        )
        self.assertEqual(report_divergent.verdict, CadenceAlignmentVerdict.STABILITY_VIOLATION_DIVERGENT)
        self.assertFalse(report_divergent.calculated_spec.metadata["is_conserved_stable"])

    # =========================================================================
    # 5. CONSTITUTIONAL D6 GOLDEN RATIO HARMONIC RHYTHM TESTS
    # =========================================================================

    def test_d6_golden_ratio_harmonic_cadence(self) -> None:
        """Verifies dihedral 6-pulse harmonic cadence under Golden Ratio Phi."""
        specs = self.calc.calculate_d6_harmonic_cadence(base_frequency_hz=60.0)

        # 1. Exactly 6 sub-rhythms matching VITAL_MAX_HP == 6
        self.assertEqual(len(specs), 6)

        expected_sectors = ["0x10", "0x20", "0x30", "0x40", "0x50", "0x60"]
        for k, spec in enumerate(specs):
            self.assertEqual(spec.cadence_type, PatternCadenceType.CONSTITUTIONAL_D6_HARMONIC)
            self.assertEqual(spec.vital_max_hp, 6)
            self.assertEqual(spec.metadata["phase_index"], k)
            self.assertEqual(spec.metadata["sector_code"], expected_sectors[k])

            # 2. Phase offset theta_k = k * 2*pi / 6 = k*pi / 3
            expected_theta = (k * 2.0 * math.pi) / 6.0
            self.assertAlmostEqual(spec.phase_offset_rad, expected_theta, delta=0.001)

            # 3. Frequency scaled by Golden Ratio progression
            phi_exponent = (k - 2.5) / 2.0
            expected_freq = 60.0 * (GOLDEN_RATIO_PHI ** phi_exponent)
            self.assertAlmostEqual(spec.frequency_hz, expected_freq, delta=0.01)
            self.assertAlmostEqual(spec.duty_cycle, 1.0 / 6.0, delta=0.001)

    # =========================================================================
    # 6. TACTICAL DOPAMINE CADENCE TESTS
    # =========================================================================

    def test_tactical_dopamine_cadence_tiers(self) -> None:
        """Verifies tactical combat micro-timing cadence tiers against golden window tau_0 = 850 ms."""
        # Perfect parry tier (100 ms)
        rep_parry = self.calc.calculate_tactical_dopamine_cadence(target_tier="PERFECT_PARRY", bpm=120.0)
        self.assertEqual(rep_parry.verdict, CadenceAlignmentVerdict.OPTIMAL_SYNCHRONIZED)
        self.assertEqual(rep_parry.calculated_spec.metadata["combo_multiplier"], 2.50)
        self.assertEqual(rep_parry.calculated_spec.frequency_hz, 10.0)

        # Flow state tier (250 ms)
        rep_flow = self.calc.calculate_tactical_dopamine_cadence(target_tier="FLOW_STATE", bpm=120.0)
        self.assertEqual(rep_flow.calculated_spec.metadata["combo_multiplier"], 1.80)
        self.assertEqual(rep_flow.calculated_spec.period_ms, 250.0)

        # Mortar salvo burst tier (375 ms / 160 RPM)
        rep_mortar = self.calc.calculate_tactical_dopamine_cadence(target_tier="MORTAR_SALVO_BURST", bpm=120.0)
        self.assertEqual(rep_mortar.calculated_spec.metadata["combo_multiplier"], 1.60)
        self.assertAlmostEqual(rep_mortar.calculated_spec.frequency_hz, 2.667, delta=0.01)

        # Rapid tempo tier (600 ms)
        rep_rapid = self.calc.calculate_tactical_dopamine_cadence(target_tier="RAPID_TEMPO", bpm=120.0)
        self.assertEqual(rep_rapid.calculated_spec.metadata["combo_multiplier"], 1.30)
        self.assertGreater(rep_rapid.shadow_headroom_ms, 0.0)

    # =========================================================================
    # 7. UMA PDN RESONANCE & DWELL CADENCE TESTS
    # =========================================================================

    def test_uma_pdn_resonance_and_dwell_cadence(self) -> None:
        """Verifies LC power delivery network resonant frequency and 12-frame dwell time."""
        # L = 1.0 nH, C = 220 uF -> f_res ~ 339.3 kHz, T_res ~ 2.947 us
        rep_pdn = self.calc.calculate_uma_pdn_resonance_cadence(l_eff_nh=1.0, c_eff_uf=220.0)
        self.assertEqual(rep_pdn.cadence_type, PatternCadenceType.UMA_INDUCTIVE_PDN_RESONANCE)
        self.assertAlmostEqual(rep_pdn.calculated_spec.metadata["f_res_khz"], 339.32, delta=1.0)
        self.assertAlmostEqual(rep_pdn.calculated_spec.metadata["t_dwell_ms"], 200.0, delta=0.1)
        self.assertEqual(rep_pdn.calculated_spec.dwell_frames, 12)
        self.assertEqual(rep_pdn.calculated_spec.metadata["max_droop_clamp_mv"], 50.0)

    # =========================================================================
    # 8. VULKAN ORACLE MULTI-SAMPLING CADENCE TESTS
    # =========================================================================

    def test_vulkan_oracle_multisampling_cadence(self) -> None:
        """Verifies 1:16 asynchronous CPU oracle correctness sampling cadence."""
        rep_oracle_60 = self.calc.calculate_vulkan_oracle_multisampling_cadence(render_fps=60.0, sampling_ratio=16)
        self.assertEqual(rep_oracle_60.cadence_type, PatternCadenceType.VULKAN_ORACLE_MULTISAMPLING)
        self.assertAlmostEqual(rep_oracle_60.calculated_spec.frequency_hz, 3.75, delta=0.01)
        self.assertAlmostEqual(rep_oracle_60.calculated_spec.period_ms, 266.667, delta=0.1)

        rep_oracle_120 = self.calc.calculate_vulkan_oracle_multisampling_cadence(render_fps=120.0, sampling_ratio=16)
        self.assertAlmostEqual(rep_oracle_120.calculated_spec.frequency_hz, 7.50, delta=0.01)
        self.assertAlmostEqual(rep_oracle_120.calculated_spec.period_ms, 133.333, delta=0.1)

    # =========================================================================
    # 9. CHAINED BOOLEAN CONVOLUTION WAVE TESTS
    # =========================================================================

    def test_chained_boolean_convolution_wave_cadence(self) -> None:
        """Verifies 6-stage boolean convolution pipeline cycle period < 1.0 ms."""
        rep_conv = self.calc.calculate_boolean_convolution_wave_cadence(
            matrix_dim=64, stages=6, time_per_stage_us=150.0
        )
        self.assertEqual(rep_conv.cadence_type, PatternCadenceType.BOOLEAN_CONVOLUTIONAL_WAVE)
        self.assertEqual(rep_conv.verdict, CadenceAlignmentVerdict.OPTIMAL_SYNCHRONIZED)
        # 6 * 150 us = 900 us = 0.90 ms
        self.assertAlmostEqual(rep_conv.calculated_spec.period_ms, 0.90, delta=0.01)
        self.assertAlmostEqual(rep_conv.shadow_headroom_ms, 0.10, delta=0.01)
        self.assertEqual(rep_conv.vital_max_hp, 6)

        # Stage count mutating invariant 6 raises ContractViolation
        with self.assertRaises(ContractViolation):
            self.calc.calculate_boolean_convolution_wave_cadence(stages=5)

    # =========================================================================
    # 10. UNIFIED CROSS-PATTERN POLYPHONY TESTS
    # =========================================================================

    def test_unified_polyphony_evaluation(self) -> None:
        """Verifies polyphonic harmonization and GCD base clock across concurrent pattern specs."""
        brainwaves = self.calc.calculate_brainwave_rhythms()
        d6_specs = self.calc.calculate_d6_harmonic_cadence(base_frequency_hz=60.0)
        alt_report = self.calc.calculate_alternator_cadence(display_refresh_hz=60.0)

        concurrent_specs = [
            brainwaves[BrainwavePhase.BETA],
            alt_report.calculated_spec,
            d6_specs[0],
            d6_specs[3],
        ]

        polyphony = self.calc.evaluate_unified_pattern_polyphony(concurrent_specs)
        self.assertEqual(polyphony["patterns_harmonized_count"], 4)
        self.assertTrue(polyphony["all_constitutional_valid"])
        self.assertEqual(polyphony["vital_max_hp"], 6)
        self.assertGreater(polyphony["gcd_base_clock_hz"], 0)
        self.assertGreater(polyphony["total_energy_cost_uj"], 0.0)

    # =========================================================================
    # 11. THEORY DAG REGISTRATION TESTS
    # =========================================================================

    def test_theory_dag_registration(self) -> None:
        """Verifies that THEORY-PATTERN-RHYTHM-CADENCE-COUPLING is registered and valid in Theory DAG."""
        dag = GLOBAL_THEORY_DEPENDENCY_GRAPH
        node = dag.get_theory("THEORY-PATTERN-RHYTHM-CADENCE-COUPLING")
        self.assertIsNotNone(node)
        self.assertEqual(node.epistemic_status, TheoryEpistemicStatus.DOKAZANE)
        self.assertIn("dt < 2.0 / omega_0", node.assumptions)
        self.assertIn("THEORY-P3-CHRONOMETRIC-SHADOW", node.parent_theories)
        self.assertIn("THEORY-SYMPLECTIC-HAMILTONIAN-CONSERVATION", node.parent_theories)
        self.assertIn("THEORY-GOLDEN-RATIO-D6-TENSOR", node.parent_theories)


if __name__ == "__main__":
    unittest.main()
