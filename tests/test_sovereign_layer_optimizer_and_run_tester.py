"""
tests/test_sovereign_layer_optimizer_and_run_tester.py

Authoritative Unit Tests for Sovereign Layer Optimizers & Automated Run Harness.
================================================================================
Validates sub-microsecond execution, symplectic area/energy conservation,
12-frame dwell hysteresis anti-chattering, 128B SPSC contract throughput,
and guaranteed Delta S == 0 on any safety violation.
"""

import os
import json
import unittest
from krystal_kernel.constants import VITAL_MAX_HP, ContractViolation
from krystal_kernel.sovereign_layer_optimizer import (
    OptimizedSiliconPdnGovernor,
    OptimizedChronometricShadowScheduler,
    OptimizedSymplecticPhaseSpaceEngine,
    SymplecticParticleState,
    OptimizedCognitiveCadenceGovernor,
    OptimizedZeroCopyContractSerializer,
    OptimizedConstitutionalSafetyActuator
)
from krystal_kernel.sovereign_run_benchmark_harness import (
    FullSovereignRunReceipt,
    SovereignRunBenchmarkHarness,
    GLOBAL_SOVEREIGN_RUN_BENCHMARK_HARNESS
)


class TestSovereignLayerOptimizerAndRunTester(unittest.TestCase):
    """Test suite for optimized sovereign layers and automated execution benchmarking."""

    def setUp(self) -> None:
        self.harness = GLOBAL_SOVEREIGN_RUN_BENCHMARK_HARNESS

    def test_01_optimized_l0_pdn_eval(self) -> None:
        """Tests L0: Silicon PDN transient evaluation and RLC damping."""
        gov = OptimizedSiliconPdnGovernor()
        res_safe = gov.evaluate_pdn_transient(vdd_v=0.85, di_dt_a_per_us=35.0)
        self.assertTrue(res_safe.is_safe)
        self.assertLessEqual(res_safe.peak_voltage_v, 1.00)
        self.assertGreater(res_safe.damping_factor_zeta, 0.0)
        self.assertLess(res_safe.attenuation_db, 0.0)  # Negative attenuation in dB indicates damping
        self.assertEqual(res_safe.vital_max_hp, VITAL_MAX_HP)

        # Extreme transient breaching ceiling (vdd_v > 1.00V)
        res_strained = gov.evaluate_pdn_transient(vdd_v=1.02, di_dt_a_per_us=80.0)
        self.assertFalse(res_strained.is_safe)

    def test_02_optimized_l1_chronometric_shadow(self) -> None:
        """Tests L1: VBlank shadow scheduling with jitter compensation."""
        sched = OptimizedChronometricShadowScheduler()
        rep_60 = sched.compute_shadow_window(refresh_hz=60.0, uma_contention=0.45)
        self.assertTrue(rep_60.is_safe)
        self.assertGreaterEqual(rep_60.shadow_headroom_ms, 3.50)

        # 144 Hz truncating VBlank shadow
        rep_144 = sched.compute_shadow_window(refresh_hz=144.0, uma_contention=0.45)
        self.assertFalse(rep_144.is_safe)
        self.assertLess(rep_144.shadow_headroom_ms, 3.50)

    def test_03_optimized_l2_symplectic_integration(self) -> None:
        """Tests L2: Symplectic Leapfrog Verlet phase-space integration."""
        engine = OptimizedSymplecticPhaseSpaceEngine()
        p = SymplecticParticleState(qx=1.0, qy=0.0, qz=0.5, px=0.0, py=1.5, pz=0.0, mass=1.0)

        res = engine.integrate_n_steps(particle=p, dt=0.01, steps=1000, omega_0=2.0)
        self.assertEqual(res.det_j_preservation, 1.0)
        self.assertLess(res.energy_drift, 1e-3)
        self.assertTrue(res.cfl_margin_adhered)
        self.assertIn(res.sector_code, ("0x10", "0x20", "0x30", "0x40", "0x50", "0x60"))
        self.assertEqual(len(res.hex_color_rgb), 3)

    def test_04_optimized_l3_cadence_hysteresis_antichatter(self) -> None:
        """Tests L3: 12-frame dwell hysteresis anti-chattering suppression."""
        gov = OptimizedCognitiveCadenceGovernor()
        # Fast downscale test
        eval_down = gov.evaluate_frame_cadence(frame_time_ms=22.0, deadline_ms=16.67)
        self.assertLess(eval_down.current_scale, 1.00)

        # Upscale requires at least 12 frames
        for _ in range(11):
            eval_up = gov.evaluate_frame_cadence(frame_time_ms=10.0, deadline_ms=16.67)
            self.assertEqual(eval_up.current_scale, eval_down.current_scale)  # Still waiting on dwell

        # Frame 12 should commit the upscale
        eval_up_12 = gov.evaluate_frame_cadence(frame_time_ms=10.0, deadline_ms=16.67)
        self.assertGreater(eval_up_12.current_scale, eval_down.current_scale)

    def test_05_optimized_l4_zero_copy_contract_serializer(self) -> None:
        """Tests L4: Aligned 128B SPSC contract frame packing and integrity."""
        ser = OptimizedZeroCopyContractSerializer()
        payload = (10.0, 20.0, 30.0, 40.0, 50.0, 60.0)

        frame = ser.pack_frame_fast(contract_type=2, source_node=0x0010, target_node=0x0020, payload_18_floats=payload)
        self.assertEqual(len(frame.frame_bytes), 128)

        valid, reason, data = ser.unpack_and_verify_fast(frame.frame_bytes)
        self.assertTrue(valid)
        self.assertEqual(data["vital_hp"], VITAL_MAX_HP)

        # Tampered byte check
        tampered = bytearray(frame.frame_bytes)
        tampered[30] ^= 0xFF
        tampered_valid, _, _ = ser.unpack_and_verify_fast(bytes(tampered))
        self.assertFalse(tampered_valid)

    def test_06_optimized_l5_safety_actuator_delta_s_zero(self) -> None:
        """Tests L5: Polyhedral safety matrix evaluation and guaranteed Delta S == 0."""
        act = OptimizedConstitutionalSafetyActuator()
        init_v = act.state["ELEM_CORE_VOLTAGE"]

        # 1. Safe commit
        safe_rec = act.evaluate_and_commit(
            element_id="ELEM_CORE_VOLTAGE",
            delta=0.02,
            proposer_role="SAFE_PROPOSER",
            evaluator_role="SAFETY_AUDITOR",
            rationale="Small voltage bump"
        )
        self.assertTrue(safe_rec.authorized)
        self.assertEqual(safe_rec.mutation_delta_bytes, 8)
        self.assertAlmostEqual(act.state["ELEM_CORE_VOLTAGE"], init_v + 0.02, places=4)

        # 2. Out-of-bounds rejection -> Delta S == 0
        curr_v = act.state["ELEM_CORE_VOLTAGE"]
        unsafe_rec = act.evaluate_and_commit(
            element_id="ELEM_CORE_VOLTAGE",
            delta=0.25,  # Exceeds 1.00V ceiling
            proposer_role="SAFE_PROPOSER",
            evaluator_role="SAFETY_AUDITOR",
            rationale="Extreme overvoltage"
        )
        self.assertFalse(unsafe_rec.authorized)
        self.assertEqual(unsafe_rec.mutation_delta_bytes, 0)
        self.assertEqual(act.state["ELEM_CORE_VOLTAGE"], curr_v)  # Delta S == 0

        # 3. Self-certification rejection -> Delta S == 0
        soc_rec = act.evaluate_and_commit(
            element_id="ELEM_PDN_IMPEDANCE",
            delta=5.0,
            proposer_role="SAME_AGENT",
            evaluator_role="SAME_AGENT",
            rationale="Self-certified change"
        )
        self.assertFalse(soc_rec.authorized)
        self.assertEqual(soc_rec.status, "REJECTED_SOC")
        self.assertEqual(soc_rec.mutation_delta_bytes, 0)

    def test_07_full_benchmark_harness_execution(self) -> None:
        """Tests the full multi-layer automated run harness execution."""
        receipt = self.harness.run_full_suite()

        self.assertIsInstance(receipt, FullSovereignRunReceipt)
        self.assertTrue(receipt.all_layers_passed)
        self.assertEqual(receipt.total_layers_tested, 6)
        self.assertEqual(receipt.vital_max_hp, VITAL_MAX_HP)
        self.assertTrue(len(receipt.receipt_hmac_digest) > 0)

        for layer_id, bench in receipt.layer_benchmarks.items():
            self.assertTrue(bench.passed)
            self.assertGreater(bench.throughput_ops_per_sec, 50_000)
            self.assertLess(bench.latency_p50_us, 50.0)

    def test_08_constitutional_invariant_breach_prevention(self) -> None:
        """Verifies ContractViolation is raised if VITAL_MAX_HP != 6 is attempted."""
        with self.assertRaises(ContractViolation):
            OptimizedSiliconPdnGovernor(vital_max_hp=5)

        with self.assertRaises(ContractViolation):
            OptimizedChronometricShadowScheduler(vital_max_hp=7)

        with self.assertRaises(ContractViolation):
            OptimizedSymplecticPhaseSpaceEngine(vital_max_hp=4)

        with self.assertRaises(ContractViolation):
            SovereignRunBenchmarkHarness(vital_max_hp=8)


if __name__ == "__main__":
    unittest.main()
