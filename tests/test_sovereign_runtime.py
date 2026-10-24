"""
tests/test_sovereign_runtime.py

Unit test suite verifying the Krystal Unified Sovereign Runtime & Adaptive Coordination Interpolators.
Strictly enforces VITAL_MAX_HP == 6, 128B frame alignment, det(J) == 1.0, and Delta S == 0 on violation.
"""

import unittest
from krystal_kernel.constants import VITAL_MAX_HP, ContractViolation
from krystal_kernel.zero_copy_contracts import BinaryContractType
from krystal_kernel.pattern_rhythm_cadence_calculator import PatternCadenceType
from krystal_kernel.sovereign_runtime import (
    SubsystemNodeId,
    InterpolationMode,
    SpatialHexHslInterpolationResult,
    CadenceHarmonicInterpolationResult,
    SymplecticVerletState,
    VoltageImpedanceState,
    CoordinationChannel,
    RuntimeExecutionCycleReceipt,
    SpatialHexHslInterpolator,
    CadenceHarmonicInterpolator,
    SymplecticPhaseSpaceInterpolator,
    VoltageImpedanceInterpolator,
    KrystalSovereignRuntime,
    GLOBAL_SOVEREIGN_RUNTIME
)


class TestKrystalSovereignRuntime(unittest.TestCase):
    """Test suite for KrystalSovereignRuntime and its adapted interpolators."""

    def setUp(self) -> None:
        self.runtime = KrystalSovereignRuntime()

    def test_runtime_initialization_and_channels(self) -> None:
        """Verifies initialization of the 6 canonical coordination channels."""
        self.assertEqual(len(self.runtime.channels), 6)
        self.assertEqual(self.runtime.vital_max_hp, VITAL_MAX_HP)

        expected_channels = {
            "WSL_TO_CHRONO_SOC",
            "GODOT_TO_CHRONO_SOC",
            "TERRAIN_TO_GODOT",
            "CHRONO_SOC_TO_CASCADE",
            "CASCADE_TO_VPP",
            "VPP_TO_CHRONO_SOC"
        }
        self.assertEqual(set(self.runtime.channels.keys()), expected_channels)

        for ch in self.runtime.channels.values():
            self.assertEqual(ch.vital_max_hp, VITAL_MAX_HP)
            self.assertEqual(ch.capacity, 64)
            self.assertEqual(len(ch.frame_ring), 0)

    def test_spatial_hex_hsl_interpolator(self) -> None:
        """Verifies spatial position and shortest-path Hue interpolation into 32-bit Hex-HSL."""
        interp = self.runtime.spatial_interpolator
        p1 = (10.0, 5.0, 0.0)
        p2 = (-10.0, 5.0, 10.0)

        # Test midpoint interpolation t = 0.5
        res = interp.interpolate(p1, p2, t=0.5)
        self.assertIsInstance(res, SpatialHexHslInterpolationResult)
        self.assertEqual(res.x, 0.0)
        self.assertEqual(res.y, 5.0)
        self.assertEqual(res.z, 5.0)
        self.assertEqual(res.smooth_factor, 0.5)  # s(0.5) = 3(0.25) - 2(0.125) = 0.5
        self.assertIn(res.sector_code, [0x10, 0x20, 0x30, 0x40, 0x50, 0x60])
        self.assertGreater(res.hex_hsla_u32, 0)
        self.assertEqual(res.vital_max_hp, VITAL_MAX_HP)

        # Test boundary clamping t = 1.5 -> clamped to 1.0
        res_end = interp.interpolate(p1, p2, t=1.5)
        self.assertEqual(res_end.x, -10.0)
        self.assertEqual(res_end.z, 10.0)
        self.assertEqual(res_end.smooth_factor, 1.0)

    def test_cadence_harmonic_interpolator_dwell_hysteresis(self) -> None:
        """Verifies Principle P8: 12-frame dwell hysteresis before promoting cadence change."""
        interp = self.runtime.cadence_interpolator
        self.assertEqual(interp.active_cadence, PatternCadenceType.COGNITIVE_BRAINWAVE)

        # Request change to BOOLEAN_CONVOLUTIONAL_WAVE (12.5 ms)
        for i in range(1, 12):
            res = interp.request_cadence(PatternCadenceType.BOOLEAN_CONVOLUTIONAL_WAVE)
            self.assertFalse(res.is_promoted)
            self.assertEqual(res.dwell_counter, i)
            self.assertEqual(interp.active_cadence, PatternCadenceType.COGNITIVE_BRAINWAVE)

        # 12th consecutive request triggers promotion
        res_promo = interp.request_cadence(PatternCadenceType.BOOLEAN_CONVOLUTIONAL_WAVE)
        self.assertTrue(res_promo.is_promoted)
        self.assertEqual(res_promo.dwell_counter, 0)
        self.assertEqual(interp.active_cadence, PatternCadenceType.BOOLEAN_CONVOLUTIONAL_WAVE)

    def test_symplectic_phase_space_interpolator_volume_preservation(self) -> None:
        """Verifies 2nd-order Symplectic Verlet step guarantees det(J) == 1.0 and bounds energy."""
        interp = self.runtime.symplectic_interpolator
        q0 = 1.0
        p0 = 0.0
        dt = 0.016  # 16 ms step

        state1 = interp.step(q=q0, p=p0, dt_sec=dt)
        self.assertIsInstance(state1, SymplecticVerletState)
        self.assertEqual(state1.jacobian_determinant, 1.0)
        self.assertTrue(state1.energy_conserved)
        self.assertEqual(state1.vital_max_hp, VITAL_MAX_HP)

        # Step 10 times and verify Hamiltonian energy remains bounded
        current_state = state1
        for _ in range(10):
            current_state = interp.step(q=current_state.q, p=current_state.p, dt_sec=dt)
            self.assertEqual(current_state.jacobian_determinant, 1.0)
            self.assertAlmostEqual(current_state.hamiltonian_energy, 0.5, delta=0.01)

    def test_voltage_impedance_interpolator_rlc_and_clamping(self) -> None:
        """Verifies exponential RLC voltage settling and inductive back-EMF clamping."""
        interp = self.runtime.voltage_interpolator
        self.assertEqual(interp.bus_impedance_ohm, 50.0)

        # Normal small current change: should settle towards 0.95V without clamp
        st1 = interp.step_voltage(target_voltage=0.95, dt_ns=10.0, delta_current_a=0.01)
        self.assertFalse(st1.is_voltage_clamped)
        self.assertLess(st1.current_voltage_v, 1.05)
        self.assertEqual(st1.vital_max_hp, VITAL_MAX_HP)

        # Massive current spike (e.g. 5.0A in 0.1 ns): should trigger inductive clamp (+-150 mV)
        st2 = interp.step_voltage(target_voltage=0.95, dt_ns=0.1, delta_current_a=5.0)
        self.assertTrue(st2.is_voltage_clamped)
        self.assertAlmostEqual(st2.current_voltage_v, 0.95 - 0.15, places=3)

    def test_transmit_128b_zero_copy_contract_frame(self) -> None:
        """Verifies 128-byte contract transmission over coordination channel."""
        channel_id = "WSL_TO_CHRONO_SOC"
        frame = self.runtime.transmit_contract_frame(
            channel_id=channel_id,
            contract_type=BinaryContractType.EVIDENCE,
            source_id=0x10,
            target_id=0x20,
            payload_u64=123456789
        )
        self.assertIsNotNone(frame)
        self.assertEqual(frame.vital_hp, VITAL_MAX_HP)
        self.assertEqual(frame.contract_type, int(BinaryContractType.EVIDENCE))

        ch = self.runtime.channels[channel_id]
        self.assertEqual(ch.frames_transmitted, 1)
        self.assertEqual(ch.frames_rejected, 0)
        self.assertEqual(len(ch.frame_ring), 1)
        self.assertEqual(len(ch.frame_ring[0]), 128)

    def test_execute_shadow_cycle_commit_and_rejection(self) -> None:
        """Verifies 2-Phase Shadow Cycle execution under budget vs over budget."""
        # 1. Nominal cycle within 3.5 ms shadow window -> State Committed
        receipt_pass = self.runtime.execute_shadow_cycle(simulated_compute_ms=2.5, uma_bus_pressure=0.50)
        self.assertTrue(receipt_pass.is_shadow_budget_met)
        self.assertTrue(receipt_pass.state_committed)
        self.assertEqual(receipt_pass.vital_max_hp, VITAL_MAX_HP)
        self.assertGreater(len(receipt_pass.hmac_signature), 0)

        # 2. Exceeded shadow window (4.0 ms > 3.5 ms) -> Delta S == 0 (Rejected)
        receipt_fail = self.runtime.execute_shadow_cycle(simulated_compute_ms=4.0, uma_bus_pressure=0.50)
        self.assertFalse(receipt_fail.is_shadow_budget_met)
        self.assertFalse(receipt_fail.state_committed)

        # 3. High UMA bus pressure (0.85 >= 0.78) -> Delta S == 0 (Rejected)
        receipt_press = self.runtime.execute_shadow_cycle(simulated_compute_ms=2.0, uma_bus_pressure=0.85)
        self.assertFalse(receipt_press.is_shadow_budget_met)
        self.assertFalse(receipt_press.state_committed)

    def test_constitutional_invariant_strictness(self) -> None:
        """Ensures any invariant breach triggers ContractViolation."""
        with self.assertRaises(ContractViolation):
            KrystalSovereignRuntime(vital_max_hp=4)

        with self.assertRaises(ContractViolation):
            SpatialHexHslInterpolator(vital_max_hp=5)

        with self.assertRaises(ContractViolation):
            SymplecticPhaseSpaceInterpolator(vital_max_hp=7)


if __name__ == "__main__":
    unittest.main()
