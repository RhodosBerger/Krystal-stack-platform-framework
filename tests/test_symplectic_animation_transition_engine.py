"""
tests/test_symplectic_animation_transition_engine.py

Unit and invariant verification suite for SymplecticAnimationTransitionEngine.
Constitutional Invariant: VITAL_MAX_HP == 6
"""

import unittest
from krystal_kernel.constants import VITAL_MAX_HP, ContractViolation
from krystal_kernel.symplectic_animation_transition_engine import (
    TransitionCurvePreset,
    PhaseSpacePoint,
    SymplecticTransitionTrajectory,
    HexHslColorKeyframe,
    SymplecticAnimationTransitionEngine,
    GLOBAL_SYMPLECTIC_TRANSITION_ENGINE
)


class TestSymplecticAnimationTransitionEngine(unittest.TestCase):

    def setUp(self) -> None:
        self.engine = SymplecticAnimationTransitionEngine()

    def test_01_engine_initialization_and_invariant(self) -> None:
        """Verifies engine initialization and constitutional invariant VITAL_MAX_HP == 6."""
        self.assertEqual(self.engine.vital_max_hp, VITAL_MAX_HP)

    def test_02_spring_trajectory_leapfrog_integration(self) -> None:
        """Verifies symplectic spring trajectory properties and determinant det(J) == 1.0."""
        traj = self.engine.compute_spring_trajectory(
            start_val=0.0,
            target_val=1.0,
            stiffness=180.0,
            damping=14.0,
            duration_sec=0.5,
            sample_rate_hz=60.0
        )
        self.assertTrue(traj.jacobian_determinant_verified)
        self.assertEqual(traj.vital_max_hp, VITAL_MAX_HP)
        self.assertGreater(traj.total_samples, 20)
        self.assertTrue(traj.css_cubic_bezier.startswith("cubic-bezier("))

        # Check endpoints
        p_start = traj.points[0]
        p_end = traj.points[-1]
        self.assertEqual(p_start.q_position, 0.0)
        self.assertAlmostEqual(p_end.q_position, 1.0, delta=0.05)

    def test_03_hex_hsl_chromatic_interpolation(self) -> None:
        """Verifies D6 orbital sweep across Hex-HSL cylindrical manifold."""
        keyframes = self.engine.compute_hex_hsl_transition(
            start_hex_sector=0x10,
            end_hex_sector=0x60,
            steps=12
        )
        self.assertEqual(len(keyframes), 13)
        self.assertEqual(keyframes[0].d6_sector_hex, "0x10")
        self.assertEqual(keyframes[-1].d6_sector_hex, "0x60")
        for kf in keyframes:
            self.assertTrue(kf.css_color.startswith("hsl("))
            self.assertGreaterEqual(kf.hue_degrees, 0.0)
            self.assertLessEqual(kf.hue_degrees, 360.0)

    def test_04_alpha_harmonic_pulse_frequency(self) -> None:
        """Verifies 10 Hz alpha harmonic cognitive pulse oscillation."""
        p0 = self.engine.compute_alpha_harmonic_pulse(0.0, frequency_hz=10.0)
        self.assertAlmostEqual(p0, 0.5, delta=1e-5)

        # Peak at 1/4 of cycle: T = 0.1s -> t = 0.025s
        p_peak = self.engine.compute_alpha_harmonic_pulse(0.025, frequency_hz=10.0)
        self.assertAlmostEqual(p_peak, 1.0, delta=1e-4)

        # Trough at 3/4 of cycle: t = 0.075s
        p_trough = self.engine.compute_alpha_harmonic_pulse(0.075, frequency_hz=10.0)
        self.assertAlmostEqual(p_trough, 0.0, delta=1e-4)

    def test_05_constitutional_invariant_breach_prevention(self) -> None:
        """Verifies ContractViolation is raised if vital_max_hp != 6 is attempted."""
        with self.assertRaises(ContractViolation):
            SymplecticAnimationTransitionEngine(vital_max_hp=7)

        with self.assertRaises(ContractViolation):
            SymplecticTransitionTrajectory(
                preset=TransitionCurvePreset.SYMPLECTIC_SPRING_RESPONSIVE,
                start_value=0.0,
                target_value=1.0,
                duration_sec=0.5,
                sample_rate_hz=60.0,
                total_samples=30,
                jacobian_determinant_verified=True,
                max_energy_drift=0.0001,
                css_cubic_bezier="cubic-bezier(0.16, 1, 0.3, 1)",
                points=(),
                vital_max_hp=5
            )


if __name__ == "__main__":
    unittest.main()
