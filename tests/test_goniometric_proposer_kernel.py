#!/usr/bin/env python3
"""
==============================================================================
UNIT TESTS: SOVEREIGN GONIOMETRIC PROPOSER KERNEL & COMPUTING BACKEND
==============================================================================
Module: tests/test_goniometric_proposer_kernel.py
Description:
    Comprehensive verification for GoniometricComputingBackend and
    GoniometricProposerKernel under strict dual-mode epistemic rules:
      - Separation of Concerns: Proposer != Evaluator.
      - State Safety Invariant: not Valid(Action) => Delta S == 0.
      - Survival Invariant: VITAL_MAX_HP == 6.
      - Symplectic Invariant: det(J) == 1.0, |Delta H| < 1e-3.
      - Camera Vertigo Invariant: d * tan(FOV / 2) == const.
==============================================================================
"""

import math
import unittest
from typing import Dict, Any

from krystal_kernel.constants import VITAL_MAX_HP, ContractViolation
from krystal_kernel.goniometric_proposer_kernel import (
    GoniometricProposalCategory,
    GoniometricRadiusProfile,
    GoniometricActionProposal,
    GoniometricComputingBackend,
    GoniometricProposerKernel,
    get_goniometric_proposer_kernel,
    reset_goniometric_proposer_kernel
)
from krystal_kernel.sovereign_layer_optimizer import OptimizedConstitutionalSafetyActuator
from krystal_janet.sovereign_janet_api_bridge import (
    SovereignJanetSystemBridge,
    JanetCommandKind
)


class TestGoniometricProposerKernel(unittest.TestCase):

    def setUp(self) -> None:
        self.backend = GoniometricComputingBackend()
        self.kernel = reset_goniometric_proposer_kernel()
        self.actuator = OptimizedConstitutionalSafetyActuator(vital_max_hp=VITAL_MAX_HP)
        self.janet_bridge = SovereignJanetSystemBridge(vital_max_hp=VITAL_MAX_HP)

    # -----------------------------------------------------------------
    # 1. Goniometric Computing Backend Verification
    # -----------------------------------------------------------------

    def test_01_unit_circle_s1_projection_and_so2_rotation(self) -> None:
        """Verifies S^1 projection on the unit circle and SO(2) orthogonal rotation."""
        for deg in [0.0, 30.0, 45.0, 60.0, 90.0, 180.0, 270.0]:
            rad = math.radians(deg)
            x, y = self.backend.s1_projection(rad)
            r_sq = x * x + y * y
            self.assertAlmostEqual(r_sq, 1.0, places=6, msg=f"S^1 invariant violated at {deg} deg")

        # SO(2) rotation: (1, 0) rotated by 90 deg -> (0, 1)
        rx, ry = self.backend.so2_rotate(1.0, 0.0, math.pi * 0.5)
        self.assertAlmostEqual(rx, 0.0, places=6)
        self.assertAlmostEqual(ry, 1.0, places=6)

    def test_02_radial_derivatives_and_polar_curvature(self) -> None:
        """Verifies radial profile: r(theta), first and second derivatives, and curvature kappa."""
        harmonics = [
            (1, 10.0, 4.0, 0.0),       # 4-lobe harmonic order 1
            (2, 5.0, 4.0, math.pi / 4) # order 2
        ]
        profile = self.backend.evaluate_radial_profile(
            base_radius=100.0,
            harmonics=harmonics,
            theta_rad=0.0
        )
        self.assertIsInstance(profile, GoniometricRadiusProfile)
        self.assertGreater(profile.radius, 0.0)
        self.assertAlmostEqual(profile.radius, 100.0 + 10.0 * math.cos(0.0) + 5.0 * math.cos(math.pi / 4), places=4)
        self.assertGreater(profile.curvature_kappa, 0.0)
        self.assertFalse(profile.is_undercut)

    def test_03_camera_vertigo_dolly_zoom_invariant(self) -> None:
        """Verifies Hitchcock Vertigo Dolly Zoom invariant: d * tan(FOV/2) == const."""
        d1 = 10.0
        fov1_deg = 60.0
        target_d2 = 5.0

        d2, fov2_deg, k_const = self.backend.solve_camera_dolly_zoom(
            d1=d1,
            fov1_deg=fov1_deg,
            target_d2=target_d2
        )
        self.assertEqual(d2, 5.0)
        self.assertGreater(fov2_deg, fov1_deg, "Moving closer must widen FOV to preserve subject size")

        # Check invariant holds for both states
        k1 = d1 * math.tan(math.radians(fov1_deg * 0.5))
        k2 = d2 * math.tan(math.radians(fov2_deg * 0.5))
        self.assertAlmostEqual(k1, k2, places=6)
        self.assertAlmostEqual(k_const, k1, places=6)

    def test_04_ballistic_firing_angles_low_and_high_arc(self) -> None:
        """Verifies direct fire (low arc) vs mortar (high arc) goniometric solutions."""
        sol = self.backend.solve_ballistic_firing_angles(
            range_m=100.0,
            muzzle_velocity_mps=40.0,
            gravity_mps2=9.80665
        )
        self.assertTrue(sol["in_range"])
        low_deg = sol["low_arc"]["elevation_deg"]
        high_deg = sol["high_arc_mortar"]["elevation_deg"]

        self.assertLessEqual(low_deg, 45.0)
        self.assertGreaterEqual(high_deg, 45.0)
        # Complementarity: low + high == 90 deg
        self.assertAlmostEqual(low_deg + high_deg, 90.0, places=4)
        # High arc mortar has higher apex and longer flight time
        self.assertGreater(sol["high_arc_mortar"]["apex_height_m"], sol["low_arc"]["apex_height_m"])
        self.assertGreater(sol["high_arc_mortar"]["flight_time_s"], sol["low_arc"]["flight_time_s"])

    def test_05_symplectic_harmonic_oscillator_liouville(self) -> None:
        """Verifies det(J) == 1.000000 and energy conservation in harmonic potential."""
        res = self.backend.evaluate_symplectic_harmonic_oscillator(
            q0=1.0,
            p0=0.0,
            omega=2.0,
            mass=1.0,
            dt=0.005,
            steps=1000
        )
        self.assertEqual(res["det_j"], 1.000000)
        self.assertTrue(res["is_conservative"])
        self.assertLess(res["max_energy_drift"], 1e-3)

    # -----------------------------------------------------------------
    # 2. Goniometric Proposer Kernel & Epistemic Separation
    # -----------------------------------------------------------------

    def test_06_propose_trajectory_motion(self) -> None:
        """Verifies trajectory proposal generation with full 5-tuple provenance."""
        prop = self.kernel.propose_trajectory_motion(
            current_pos=(10.0, 0.0, 0.0),
            target_angle_rad=math.pi * 0.5,
            angular_speed_rad_per_s=1.0,
            dt_s=0.016667
        )
        self.assertEqual(prop.category, GoniometricProposalCategory.TRAJECTORY_MOTION)
        self.assertEqual(prop.proposer_role, "GONIOMETRIC_PROPOSER_KERNEL")
        self.assertEqual(prop.vital_max_hp, 6)
        self.assertIn("det_j", prop.mathematical_metrics)
        self.assertEqual(prop.mathematical_metrics["det_j"], 1.0)
        self.assertTrue(len(prop.compute_sha256()) == 64)

    def test_07_propose_camera_dolly_zoom(self) -> None:
        """Verifies camera Dolly Zoom proposal adhering to Vertigo invariant."""
        prop = self.kernel.propose_camera_dolly_zoom(
            current_dist=12.0,
            current_fov_deg=50.0,
            target_dist=6.0
        )
        self.assertEqual(prop.category, GoniometricProposalCategory.CAMERA_VERTIGO_ZOOM)
        self.assertGreater(prop.proposed_value, 50.0)
        self.assertIn("vertigo_invariant_k", prop.mathematical_metrics)

    def test_08_propose_ballistic_solution(self) -> None:
        """Verifies ballistic artillery and mortar elevation proposals."""
        prop = self.kernel.propose_ballistic_solution(
            target_range_m=120.0,
            muzzle_velocity_mps=45.0,
            prefer_mortar_arc=True
        )
        self.assertEqual(prop.category, GoniometricProposalCategory.BALLISTIC_ELEVATION)
        self.assertGreaterEqual(prop.proposed_value, 45.0, "Mortar fire must propose >= 45 deg")

    def test_09_propose_pdn_frequency_phase_shift(self) -> None:
        """Verifies silicon PDN anti-phase proposal for transient damping."""
        prop = self.kernel.propose_pdn_frequency_phase_shift(
            base_clock_mhz=1200.0,
            pdn_load_phase_rad=0.785398  # pi / 4
        )
        self.assertEqual(prop.category, GoniometricProposalCategory.PDN_FREQUENCY_PHASE)
        self.assertIn("counter_phase_rad", prop.mathematical_metrics)
        self.assertIn("damping_factor", prop.mathematical_metrics)

    def test_10_separation_of_concerns_proposer_cannot_self_certify(self) -> None:
        """
        Constitutional law: Proposer != Evaluator.
        Submitting a proposal where evaluator matches proposer must raise ContractViolation.
        """
        prop = self.kernel.propose_state_intervention(
            element_id="ELEM_CORE_VOLTAGE",
            current_value=0.85,
            harmonic_order=1,
            base_delta=0.02,
            rationale="Test self-certification block"
        )
        with self.assertRaises(ContractViolation) as ctx:
            self.kernel.submit_proposal_to_evaluator(
                proposal=prop,
                evaluator_actuator=self.actuator,
                evaluator_role="GONIOMETRIC_PROPOSER_KERNEL"  # FORBIDDEN: self-certification
            )
        self.assertIn("cannot self-certify", str(ctx.exception))

    def test_11_evaluator_approval_and_rejection_with_delta_s_zero(self) -> None:
        """
        Submitting safe proposal commits state;
        submitting unsafe proposal (voltage > 1.00V) rejects with Delta S == 0.
        """
        # 1. Safe proposal (+0.03V on 0.85V -> 0.88V)
        safe_prop = self.kernel.propose_state_intervention(
            element_id="ELEM_CORE_VOLTAGE",
            current_value=0.85,
            harmonic_order=0,  # cos(0) = 1.0
            base_delta=0.03,
            rationale="Safe goniometric DVFS boost"
        )
        safe_receipt = self.kernel.submit_proposal_to_evaluator(
            proposal=safe_prop,
            evaluator_actuator=self.actuator,
            evaluator_role="CONSTITUTIONAL_GATEKEEPER"
        )
        self.assertTrue(safe_receipt.authorized)
        self.assertEqual(safe_receipt.status, "COMMITTED_SAFE")
        self.assertAlmostEqual(safe_receipt.committed_value, 0.88, places=4)

        # 2. Unsafe proposal (+0.25V pushing voltage to 1.13V > 1.00V)
        unsafe_prop = self.kernel.propose_state_intervention(
            element_id="ELEM_CORE_VOLTAGE",
            current_value=0.88,
            harmonic_order=0,
            base_delta=0.25,
            rationale="Unsafe voltage breach"
        )
        unsafe_receipt = self.kernel.submit_proposal_to_evaluator(
            proposal=unsafe_prop,
            evaluator_actuator=self.actuator,
            evaluator_role="CONSTITUTIONAL_GATEKEEPER"
        )
        self.assertFalse(unsafe_receipt.authorized)
        self.assertEqual(unsafe_receipt.status, "QUARANTINED_SAFETY_BREACH")
        # Strict state rule: Delta S == 0
        self.assertEqual(unsafe_receipt.committed_value, 0.88)
        self.assertEqual(unsafe_receipt.mutation_delta_bytes, 0)

    # -----------------------------------------------------------------
    # 3. Janet API Bridge Goniometric Dispatch Verification
    # -----------------------------------------------------------------

    def test_12_janet_bridge_goniometric_dispatch(self) -> None:
        """Verifies Janet bridge dispatches goniometric commands and auto-submit."""
        # 1. Trajectory dispatch
        rec_traj = self.janet_bridge.dispatch_janet_command(
            command="propose-goniometric-trajectory",
            args={"current_x": 1.0, "current_y": 0.0, "current_z": 0.0, "target_angle_rad": 1.57}
        )
        self.assertTrue(rec_traj.is_success)
        self.assertEqual(rec_traj.status_code, "GONIOMETRIC_TRAJECTORY_PROPOSED")
        self.assertEqual(rec_traj.output_payload["proposer_role"], "GONIOMETRIC_PROPOSER_KERNEL")

        # 2. Camera zoom dispatch
        rec_zoom = self.janet_bridge.dispatch_janet_command(
            command="propose-goniometric-camera-zoom",
            args={"current_dist": 10.0, "current_fov_deg": 60.0, "target_dist": 5.0}
        )
        self.assertTrue(rec_zoom.is_success)
        self.assertEqual(rec_zoom.status_code, "GONIOMETRIC_CAMERA_ZOOM_PROPOSED")

        # 3. Ballistic dispatch
        rec_ball = self.janet_bridge.dispatch_janet_command(
            command="propose-goniometric-ballistic",
            args={"range_m": 100.0, "muzzle_velocity_mps": 40.0, "prefer_mortar_arc": True}
        )
        self.assertTrue(rec_ball.is_success)
        self.assertEqual(rec_ball.status_code, "GONIOMETRIC_BALLISTIC_PROPOSED")

        # 4. Status query
        rec_stat = self.janet_bridge.dispatch_janet_command(
            command="query-goniometric-proposer-status",
            args={}
        )
        self.assertTrue(rec_stat.is_success)
        self.assertEqual(rec_stat.output_payload["kernel_name"], "GoniometricProposerKernel")
        self.assertEqual(rec_stat.output_payload["epistemic_role"], "PROPOSER")

        # 5. S-expression dispatch
        sexp_res = self.janet_bridge.eval_sexp_string("(query-goniometric-proposer-status)")
        self.assertTrue(sexp_res.is_success)


if __name__ == "__main__":
    unittest.main()
