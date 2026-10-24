"""
tests/test_textbook_inferable_patterns.py

Unit test suite verifying the 8 textbook canonical inferable patterns in Krystal Stack.
Strictly enforces VITAL_MAX_HP == 6, det(J) == 1.0, CFL limits, and Delta S == 0 on gate rejection.
"""

import unittest
import math
from krystal_kernel.constants import VITAL_MAX_HP, ContractViolation
from krystal_kernel.textbook_inferable_patterns import (
    InferablePatternId,
    PatternEvidenceTuple,
    ReasoningTrace11Tuple,
    PhaseCrossoverInferenceResult,
    ChronometricShadowInferenceResult,
    SymplecticLeapfrogInferenceResult,
    DwellHysteresisInferenceResult,
    HexHslBijectionInferenceResult,
    InductiveVoltageInferenceResult,
    PulseTelemetryInferenceResult,
    ArtifactGateInferenceResult,
    TextbookPatternDescriptor,
    TextbookInferablePatternsEngine,
    infer_asymmetric_phase_crossover,
    infer_chronometric_vblank_shadow,
    infer_symplectic_leapfrog_step,
    infer_12frame_dwell_hysteresis,
    infer_hex_hsl_discrete_bijection,
    infer_uma_inductive_voltage_clamping,
    infer_sub50ns_pulse_telemetry,
    infer_artifact_gate_commit,
    GLOBAL_TEXTBOOK_PATTERNS_ENGINE
)


class TestTextbookInferablePatterns(unittest.TestCase):
    """Test suite validating canonical inferable patterns, their mathematics, and invariants."""

    def setUp(self) -> None:
        self.engine = TextbookInferablePatternsEngine()

    def test_p1_asymmetric_phase_crossover(self) -> None:
        """Verifies analytical crossover calculation between GPU prefill and CPU decode."""
        # Large prompt (512 tokens) -> Should route to GPU parallel GEMM
        res_gpu = infer_asymmetric_phase_crossover(prompt_tokens=512)
        self.assertEqual(res_gpu.optimal_execution_target, "GPU_LEVEL_ZERO")
        self.assertGreater(res_gpu.energy_advantage_ratio, 1.0)
        self.assertEqual(res_gpu.vital_max_hp, VITAL_MAX_HP)
        self.assertLessEqual(res_gpu.usm_handoff_latency_us, 0.748)

        # Single token decode step (1 token) -> Should route to CPU host dispatch
        res_cpu = infer_asymmetric_phase_crossover(prompt_tokens=1)
        self.assertEqual(res_cpu.optimal_execution_target, "CPU_NPU_HOST")
        self.assertGreaterEqual(res_cpu.crossover_token_threshold, 1)

        # Negative test: Constitutional invariant breach
        with self.assertRaises(ContractViolation):
            infer_asymmetric_phase_crossover(prompt_tokens=100, vital_max_hp=5)

    def test_p2_chronometric_vblank_shadow(self) -> None:
        """Verifies contention-free VBlank shadow budget reservation."""
        # Nominal 60 Hz with 42% UMA load -> Should allocate >= 3.50 ms shadow window
        res_nom = infer_chronometric_vblank_shadow(refresh_rate_hz=60.0, uma_utilization=0.42)
        self.assertTrue(res_nom.shadow_window_authorized)
        self.assertTrue(res_nom.uma_contention_free)
        self.assertGreaterEqual(res_nom.allocated_shadow_ms, 3.50)
        self.assertEqual(res_nom.vital_max_hp, VITAL_MAX_HP)

        # High UMA load (82%) -> Exceeds 0.78 threshold -> Must inhibit compute shadow
        res_throttled = infer_chronometric_vblank_shadow(refresh_rate_hz=60.0, uma_utilization=0.82)
        self.assertFalse(res_throttled.shadow_window_authorized)
        self.assertFalse(res_throttled.uma_contention_free)
        self.assertEqual(res_throttled.allocated_shadow_ms, 0.0)

    def test_p3_symplectic_leapfrog_conservation(self) -> None:
        """Verifies exact phase-space conservation det(J) == 1.0 without secular drift."""
        q0 = 1.0
        p0 = 0.0
        dt = 0.02
        omega = 1.5

        res = infer_symplectic_leapfrog_step(q0=q0, p0=p0, dt=dt, omega_0=omega)
        self.assertTrue(res.is_symplectic)
        self.assertAlmostEqual(res.jacobian_determinant, 1.0, places=10)
        self.assertEqual(res.vital_max_hp, VITAL_MAX_HP)

        # Verify stability over 100 continuous steps
        curr_q, curr_p = q0, p0
        initial_energy = 0.5 * (omega ** 2) * (q0 ** 2)
        for _ in range(100):
            step = infer_symplectic_leapfrog_step(q0=curr_q, p0=curr_p, dt=dt, omega_0=omega)
            curr_q, curr_p = step.q_final, step.p_final

        final_energy = 0.5 * (curr_p ** 2) + 0.5 * (omega ** 2) * (curr_q ** 2)
        # Bounded oscillation without numerical divergence
        self.assertLess(abs(final_energy - initial_energy), 0.05)

        # Negative test: CFL criterion violation (dt >= 2 / omega)
        cfl_dt = (2.0 / omega) + 0.1
        with self.assertRaises(ContractViolation):
            infer_symplectic_leapfrog_step(q0=1.0, p0=0.0, dt=cfl_dt, omega_0=omega)

    def test_p4_12frame_dwell_hysteresis(self) -> None:
        """Verifies P8 anti-oscillation dwell counter and fast downscale."""
        # 1. High frequency oscillation (alternating requests) -> Must suppress transitions
        noisy_stream = [1.0, 1.5, 1.0, 1.5, 1.0, 1.5, 1.0, 1.5]
        res_noisy = infer_12frame_dwell_hysteresis(initial_state=1.0, target_stream=noisy_stream)
        self.assertEqual(res_noisy.total_transitions_committed, 0)
        self.assertEqual(res_noisy.final_committed_state, 1.0)

        # 2. Consistent sustained upscale request for 15 frames -> Commits after 12 frames
        sustained_stream = [2.0] * 15
        res_sustained = infer_12frame_dwell_hysteresis(initial_state=1.0, target_stream=sustained_stream)
        self.assertEqual(res_sustained.total_transitions_committed, 1)
        self.assertGreater(res_sustained.final_committed_state, 1.5)

        # 3. Emergency critical downscale (-0.5 delta) -> Instant commit
        emergency_stream = [0.5]
        res_downscale = infer_12frame_dwell_hysteresis(initial_state=1.0, target_stream=emergency_stream)
        self.assertEqual(res_downscale.total_transitions_committed, 1)
        self.assertEqual(res_downscale.final_committed_state, 0.5)

    def test_p5_hex_hsl_discrete_bijection(self) -> None:
        """Verifies 3D continuous Cartesian to 6-sector discrete Hex-HSL bijection."""
        # Sector 0x10: (1.0, 0.2, 0.0) -> angle ~ 11 deg -> sector 0x10
        r1 = infer_hex_hsl_discrete_bijection(x=1.0, y=0.2, z=0.0)
        self.assertEqual(r1.discrete_sector_id, 0x10)
        self.assertTrue(r1.packed_color_hex.startswith("0x"))
        self.assertEqual(len(r1.packed_color_hex), 10)  # "0xHHSSLLAA"

        # Sector 0x20: (0.0, 1.0, 0.5) -> angle 90 deg -> sector 0x20
        r2 = infer_hex_hsl_discrete_bijection(x=0.0, y=1.0, z=0.5)
        self.assertEqual(r2.discrete_sector_id, 0x20)

        # Sector 0x40: (-1.0, 0.0, -0.5) -> angle 180 deg -> sector 0x40
        r4 = infer_hex_hsl_discrete_bijection(x=-1.0, y=0.0, z=-0.5)
        self.assertEqual(r4.discrete_sector_id, 0x40)

        # Hermite smoothstep value must be in [0, 1]
        self.assertGreaterEqual(r1.hermite_smoothstep, 0.0)
        self.assertLessEqual(r1.hermite_smoothstep, 1.0)

    def test_p6_inductive_voltage_back_emf_clamping(self) -> None:
        """Verifies Faraday-Lenz back-EMF calculation and active impedance clamping."""
        res = infer_uma_inductive_voltage_clamping(
            delta_current_amps=15.0,
            transition_time_ps=200.0,
            loop_inductance_nh=2.5,
            nominal_voltage_volts=0.85,
            damping_impedance_ohms=50.0
        )
        # Raw V_ind = 2.5e-9 * (15 / 200e-12) = 187.5 V
        self.assertAlmostEqual(res.raw_induced_voltage_volts, 187.5, places=1)
        # Active clamp limits to nominal + 0.15 V = 1.00 V
        self.assertLessEqual(res.clamped_voltage_volts, 1.00)
        self.assertTrue(res.overshoot_within_silicon_limit)
        # Settling time = 5 * (2.5e-9 / 50) = 250 ps
        self.assertAlmostEqual(res.settling_time_ps, 250.0, places=1)

    def test_p7_sub50ns_pulse_telemetry(self) -> None:
        """Verifies packed 8-byte pulse serialization and sub-50ns latency tracking."""
        res = infer_sub50ns_pulse_telemetry(
            channel_id=0x42,
            status_code=0x01,
            sequence_id=0x09,
            delta_tsc_ticks=48,
            tsc_freq_ghz=2.4
        )
        # 48 ticks at 2.4 GHz = 20.0 ns
        self.assertEqual(res.inferred_latency_ns, 20.0)
        self.assertEqual(len(res.packed_hex), 18)  # "0x" + 16 hex digits = 8 bytes
        self.assertTrue(res.is_non_temporal_aligned)
        self.assertLess(res.observer_overhead_ratio, 0.0015)  # < 0.15%

    def test_p8_dual_mode_artifact_gate(self) -> None:
        """Verifies dual-mode epistemic authorization gate (ArtifactGate)."""
        # 1. Valid authorization: Independent proposer and evaluator, valid receipt, vital_hp=6
        res_ok = infer_artifact_gate_commit(
            action_name="COMMIT_VALID_BLOCK",
            proposer_role="EXPLORE_AGENT",
            evaluator_role="VERIFY_ENGINE",
            receipt_valid=True,
            authority_valid=True,
            falsifier_passed=True,
            vital_max_hp=VITAL_MAX_HP
        )
        self.assertTrue(res_ok.state_committed)
        self.assertEqual(res_ok.delta_s, "COMMITTED")
        self.assertTrue(len(res_ok.audit_hash) == 64)

        # 2. Self-certification violation (proposer == evaluator) -> Forbidden!
        res_self = infer_artifact_gate_commit(
            action_name="SELF_CERTIFIED_BLOCK",
            proposer_role="EXPLORE_AGENT",
            evaluator_role="EXPLORE_AGENT",
            receipt_valid=True,
            authority_valid=True,
            falsifier_passed=True,
            vital_max_hp=VITAL_MAX_HP
        )
        self.assertFalse(res_self.state_committed)
        self.assertEqual(res_self.delta_s, "DELTA_S_ZERO")
        self.assertFalse(res_self.separation_of_concerns_valid)

        # 3. Invalid receipt -> Forbidden!
        res_bad_rcpt = infer_artifact_gate_commit(
            action_name="TAMPERED_RECEIPT",
            proposer_role="EXPLORE_AGENT",
            evaluator_role="VERIFY_ENGINE",
            receipt_valid=False,
            authority_valid=True,
            falsifier_passed=True,
            vital_max_hp=VITAL_MAX_HP
        )
        self.assertFalse(res_bad_rcpt.state_committed)
        self.assertEqual(res_bad_rcpt.delta_s, "DELTA_S_ZERO")

        # 4. Constitutional HP breach -> Forbidden!
        res_bad_hp = infer_artifact_gate_commit(
            action_name="HP_BREACH",
            proposer_role="EXPLORE_AGENT",
            evaluator_role="VERIFY_ENGINE",
            receipt_valid=True,
            authority_valid=True,
            falsifier_passed=True,
            vital_max_hp=5
        )
        self.assertFalse(res_bad_hp.state_committed)
        self.assertEqual(res_bad_hp.delta_s, "DELTA_S_ZERO")

    def test_textbook_engine_descriptors_and_suite(self) -> None:
        """Verifies complete pattern engine descriptor catalogue and full execution suite."""
        descriptors = self.engine.list_all_descriptors()
        self.assertEqual(len(descriptors), 8)

        # Verify all 8 have proper 6-tuple evidence and 11-step reasoning trace
        for desc in descriptors:
            self.assertIsInstance(desc.evidence_6tuple, PatternEvidenceTuple)
            self.assertIsInstance(desc.reasoning_trace, ReasoningTrace11Tuple)
            self.assertEqual(desc.reasoning_trace.verdict, "APPROVED_FACT")
            self.assertIn("VITAL_MAX_HP == 6", desc.reasoning_trace.invariant)

        # Execute full validation suite
        report = self.engine.execute_all_patterns_suite()
        self.assertEqual(report["total_patterns_executed"], 8)
        self.assertTrue(report["all_patterns_verified"])
        self.assertEqual(report["vital_max_hp"], VITAL_MAX_HP)
        self.assertTrue(len(report["suite_receipt_hash"]) == 64)


if __name__ == "__main__":
    unittest.main()
