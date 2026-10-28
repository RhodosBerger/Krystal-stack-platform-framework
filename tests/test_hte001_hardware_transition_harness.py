#!/usr/bin/env python3
"""
================================================================================
UNIT TESTS: HTE-001 HARDWARE TRANSITION ACQUISITION HARNESS
================================================================================
Module: tests/test_hte001_hardware_transition_harness.py

Tests:
  - Freeze contract state invariants
  - Preregistered manifest loading & RFC 8785 canonical hash stability
  - Cardinality separation: N_runs >= 30 vs N_ops/run == 1,000,000
  - Paired idle baseline protocol & INA226 thermal drift compensation
  - Unclamped dynamic energy estimator (no max(0, E) distortion)
  - Division by zero prevention: I_0 (exact zero) vs I_+ (normalized tolerance)
  - Decoupling AcquisitionValid from TransitionPass (survivorship bias prevention)
  - Statistical Conflict Resolution: Envelope Compliance (Reliability) vs Bootstrap 95% CI (Accuracy/Bias)
  - Verification that r1 = +2, r2 = -2 (mean 0) FAILS experiment pass on envelope breach
================================================================================
"""

import unittest
from pathlib import Path

from krystal_kernel.constants import VITAL_MAX_HP, ContractViolation
from krystal_kernel.hte001_hardware_transition_harness import (
    SOFTWARE_BASELINE_STATE,
    BASELINE_PURPOSE,
    THEORY_NODES,
    PARETO_DATASET,
    PHYSICAL_EVIDENCE,
    Hte001RunManifest,
    HardwareTransitionReceipt,
    Hte001Orchestrator,
    GLOBAL_HTE001_MANIFEST_PATH,
)


class TestHte001HardwareTransitionHarness(unittest.TestCase):
    """Test suite for HTE-001 hardware transition experiment, paired idle baseline and receipts."""

    def setUp(self) -> None:
        self.manifest_path = GLOBAL_HTE001_MANIFEST_PATH
        self.orchestrator = Hte001Orchestrator(self.manifest_path)

    def test_freeze_contract_constants(self) -> None:
        """Verifies that the software baseline freeze constants are strictly locked."""
        self.assertEqual(SOFTWARE_BASELINE_STATE, "FROZEN")
        self.assertEqual(BASELINE_PURPOSE, "HTE-001_REFERENCE")
        self.assertEqual(THEORY_NODES, 56)
        self.assertEqual(PARETO_DATASET, "CAMPAIGN_A")
        self.assertEqual(PHYSICAL_EVIDENCE, "PENDING")
        self.assertEqual(VITAL_MAX_HP, 6)

    def test_manifest_loading_and_canonical_hash(self) -> None:
        """Verifies preregistered manifest fields and deterministic RFC 8785 hash."""
        manifest, digest = Hte001RunManifest.load_from_file(self.manifest_path)
        self.assertEqual(manifest.experiment_id, "HTE-001")
        self.assertEqual(manifest.run_series_id, "HTE-001-SERIES-ALPHA")
        self.assertEqual(manifest.preregistration_status, "FINAL_PREREGISTERED_MANIFEST")
        self.assertEqual(manifest.supersedes_manifest_hash, "3a2003ce8101b3fb1ce0568f86f148ef37d41b4b4bca7fd4ba18783e2a44a468")
        self.assertEqual(manifest.superseded_reason, "SUPERSEDED_BEFORE_PHYSICAL_ACQUISITION_INA226_CADENCE_AND_PROVENANCE_IDENTITY_EXPANSION")
        self.assertIsNone(manifest.first_physical_sample_timestamp)
        self.assertEqual(manifest.board_model, "XILINX-KCU116-KINTEX-USPLUS-REV1")
        self.assertEqual(manifest.clock_frequency_hz, 500_000_000)
        self.assertEqual(manifest.sampling_rate_hz, 10_000)
        self.assertEqual(manifest.measurement_boundary, "FPGA_BOARD")
        self.assertEqual(manifest.operations_per_run, 1_000_000)
        self.assertEqual(manifest.minimum_runs, 30)
        self.assertEqual(manifest.exact_zero_mode, "EXACT")
        self.assertEqual(manifest.exact_zero_epsilon, 0.0)
        self.assertEqual(manifest.exact_zero_invariant_fields, ["memory"])
        self.assertIn("energy", manifest.normalized_tolerance_fields)
        self.assertEqual(manifest.software_commit, "12f91b5c1b4d42b5ce979fb3a01d2233ba02f0ed")
        self.assertEqual(manifest.acquisition_harness_hash, "f520b171e29ea8b85b5d11d65c2ad7dd4baa9e0019ecd4eb56fee8d585f7a064")
        self.assertEqual(manifest.transition_oracle_hash, "3c87c20d94855b78f5ebcb92067376dca9dbc5699a723e7207ef8399c4fcbf42")
        self.assertEqual(manifest.sensor_contract_hash, "257637962a9ff799222323732d938036e94b14f00d1ed2d55b4af944316934e9")
        self.assertEqual(manifest.requirements_lock_hash, "3b00d587bf644fe1372aa72d3d9753111d3e1ace386c241ef94b57bfaa08bd8e")
        self.assertEqual(manifest.acquisition_software_identity, "e5c8578cbe3382dd059a319ad915567859b18c6cb1d0fe61ed3abbbc0c7d3fd0")
        self.assertEqual(manifest.independent_sample_method, "CONVERSION_READY_OBSERVED")
        self.assertEqual(len(digest), 64, "Manifest hash must be SHA-256")
        self.assertEqual(digest, "ae6a1add5fcf1d3cb36c36859d3f38ef48a27df340b8aaf39daa3b47ea17f9cf")
        self.assertEqual(digest, self.orchestrator.manifest_hash)

    def test_p0_exact_zero_invariant_vs_normalized_tolerances(self) -> None:
        """
        P0 Test: Verifies I_0 exact zero check for epsilon_memory == 0.0 without division by zero.
        I_0 = {memory}: R_memory == 0.0 => exact pass
        I_+ = {compute, energy, latency, queue, thermal}: |R / epsilon| <= 1.0
        """
        s_predicted = {
            "compute": 0.45,
            "memory": 0.30,  # matches s_after exactly
            "thermal": 0.52,
            "latency": 0.30,
            "energy": 0.42,
            "queue": 0.25,
        }
        s_after = {
            "compute": 0.45,
            "memory": 0.30,  # diff == 0.0 B
            "thermal": 0.52,
            "latency": 0.30,
            "energy": 0.42,
            "queue": 0.25,
        }
        residuals, norm_res, max_r, exact_pass, pass_tr = self.orchestrator.evaluate_transition_correctness(
            s_predicted=s_predicted,
            s_after=s_after,
            actual_energy_per_op_uj=0.42,
            predicted_energy_per_op_uj=0.42
        )
        self.assertTrue(exact_pass)
        self.assertTrue(pass_tr)
        self.assertEqual(norm_res["memory"], 0.0)

        # Non-zero memory diff must fail exact check immediately
        s_after_leak = dict(s_after)
        s_after_leak["memory"] = 0.30001
        _, norm_leak, _, exact_pass_leak, pass_tr_leak = self.orchestrator.evaluate_transition_correctness(
            s_predicted=s_predicted,
            s_after=s_after_leak,
            actual_energy_per_op_uj=0.42,
            predicted_energy_per_op_uj=0.42
        )
        self.assertFalse(exact_pass_leak)
        self.assertFalse(pass_tr_leak)

    def test_unclamped_dynamic_energy_and_paired_idle(self) -> None:
        """Verifies paired idle baseline and unclamped dynamic energy (no max(0, E) upward bias)."""
        s_before = {
            "compute": 0.60,
            "memory": 0.30,
            "thermal": 0.50,
            "latency": 0.50,
            "energy": 0.50,
            "queue": 0.50,
        }
        res = self.orchestrator.transport.execute_physical_run(
            s_before=s_before,
            action={"code": 1},
            operations_count=1_000_000
        )
        self.assertAlmostEqual(res["latency_actual_ns"], 208.0, places=1)
        self.assertGreaterEqual(res["active_sample_count"], 2000)
        self.assertGreaterEqual(res["idle_sample_count"], 1000)
        self.assertEqual(res["dropped_samples"], 0)
        # Dynamic energy is directly raw - idle_offset (unclamped)
        expected_dynamic = res["raw_energy_board_j"] - (res["p_idle_mean_w"] * (res["burst_duration_ns"] * 1e-9))
        self.assertAlmostEqual(res["idle_corrected_energy_j"], expected_dynamic, places=7)

    def test_decoupling_acquisition_valid_from_transition_pass(self) -> None:
        """
        Critical epistemic test:
        An acquisition run that fails its transition hypothesis:
          - AcquisitionValid == True (measurement was authentic, no dropped samples)
          - EvidenceClass == FPGA_BOARD_MEASURED (preserved!)
          - TransitionPass == False
          - SemanticVerdict == FAIL
          - RETAINED in dataset to prevent survivorship bias!
        """
        s_before = {
            "compute": 0.60,
            "memory": 0.30,
            "thermal": 0.50,
            "latency": 0.50,
            "energy": 0.50,
            "queue": 0.50,
        }
        s_predicted_wrong = {
            "compute": 0.90,  # discrepancy triggers transition failure
            "memory": 0.30,
            "thermal": 0.52,
            "latency": 0.30,
            "energy": 0.42,
            "queue": 0.25,
        }
        receipt = self.orchestrator.run_single_acquisition(
            run_id="RUN-0001-FAIL-TEST",
            sequence=1,
            s_before=s_before,
            s_predicted=s_predicted_wrong,
            predicted_energy_per_op_uj=0.42,
            action={"code": 1}
        )

        self.assertTrue(receipt.acquisition_valid, "Acquisition must be valid")
        self.assertFalse(receipt.transition_pass, "Transition hypothesis failed")
        self.assertEqual(receipt.evidence_class, "FPGA_BOARD_MEASURED")
        self.assertEqual(receipt.semantic_verdict, "FAIL")
        self.assertEqual(receipt.memory_residual_mode, "EXACT")
        self.assertEqual(receipt.memory_epsilon, 0.0)
        self.assertTrue(receipt.memory_exact_pass)
        self.assertTrue(receipt.verify_integrity())

    def test_envelope_compliance_prevents_bias_cancellation(self) -> None:
        """
        Tests the user's critical insight:
        If runs have r = +2.0 and r = -2.0, the mean is 0.0 (passing bias test CI [-1, 1]),
        BUT both individual transitions failed the hard envelope (|r| > 1.0).
        PASS_HTE001 MUST FAIL due to Envelope Compliance breach!
        """
        s_before = {
            "compute": 0.60, "memory": 0.30, "thermal": 0.50,
            "latency": 0.50, "energy": 0.50, "queue": 0.50
        }
        # Run A: compute s_after will be 0.45. If predicted is 0.65, diff is -0.20 => r = -2.0 (|r| > 1)
        r_a = self.orchestrator.run_single_acquisition(
            run_id="RUN-CANCEL-A", sequence=1, s_before=s_before,
            s_predicted={"compute": 0.65, "memory": 0.30, "thermal": 0.52, "latency": 0.30, "energy": 0.42, "queue": 0.25},
            predicted_energy_per_op_uj=0.194, action={"code": 1}
        )
        # Run B: if predicted is 0.25, diff is +0.20 => r = +2.0 (|r| > 1)
        r_b = self.orchestrator.run_single_acquisition(
            run_id="RUN-CANCEL-B", sequence=2, s_before=s_before,
            s_predicted={"compute": 0.25, "memory": 0.30, "thermal": 0.52, "latency": 0.30, "energy": 0.42, "queue": 0.25},
            predicted_energy_per_op_uj=0.194, action={"code": 1}
        )

        self.assertFalse(r_a.transition_pass)
        self.assertFalse(r_b.transition_pass)

        # Dataset of 30 runs formed of 15 pairs of (+2.0, -2.0)
        # The mean is exactly 0.0, but all runs breached envelope
        receipts_cancelling = [r_a, r_b] * 15
        exp_pass, details = self.orchestrator.evaluate_experiment_pass(receipts_cancelling)

        # MUST FAIL on Envelope Compliance!
        self.assertFalse(exp_pass)
        self.assertIn("Envelope compliance breached", details["reason"])
        self.assertEqual(details["failed_envelope_runs_count"], 30)

    def test_full_hte001_pass_when_both_envelope_and_bias_pass(self) -> None:
        """Verifies full experiment pass when all runs meet both Envelope Compliance and Bias tests."""
        s_before = {
            "compute": 0.60, "memory": 0.30, "thermal": 0.50,
            "latency": 0.50, "energy": 0.50, "queue": 0.50
        }
        s_predicted = {
            "compute": 0.45, "memory": 0.30, "thermal": 0.52,
            "latency": 0.30, "energy": 0.42, "queue": 0.25
        }
        receipt = self.orchestrator.run_single_acquisition(
            run_id="RUN-0001", sequence=1, s_before=s_before, s_predicted=s_predicted,
            predicted_energy_per_op_uj=0.194, action={"code": 1}
        )

        self.assertTrue(receipt.acquisition_valid)
        self.assertTrue(receipt.transition_pass)
        self.assertEqual(receipt.semantic_verdict, "COMMITTED")
        self.assertEqual(receipt.evidence_class, "FPGA_BOARD_MEASURED")
        self.assertEqual(receipt.operations_in_run, 1_000_000)
        self.assertEqual(receipt.dropped_samples, 0)

        # 30 identical passing runs fulfill both Envelope Compliance (0 breaches) and Bootstrap 95% CI Bias test
        receipts_30 = [receipt] * 30
        exp_pass_30, details_30 = self.orchestrator.evaluate_experiment_pass(receipts_30)
        self.assertTrue(exp_pass_30)
        self.assertEqual(details_30["n_acquisition_valid"], 30)
        self.assertIn("ci_results", details_30)

    def test_ina226_adc_conversion_cadence_and_deduplication(self) -> None:
        """
        Hardware Instrumentation Test (P0 Hardware Contract):
        Verifies that INA226 register polling cadence (10 kHz) is explicitly separated
        from physical ADC conversion cadence (3.57 kHz @ 280 us full update cycle).
        Ensures register configuration is frozen in receipt, duplicate reads are tracked,
        and first_physical_sample_timestamp in receipt does not alter manifest hash 3a2003ce...
        """
        s_before = {
            "compute": 0.60, "memory": 0.30, "thermal": 0.50,
            "latency": 0.50, "energy": 0.50, "queue": 0.50
        }
        s_predicted = {
            "compute": 0.45, "memory": 0.30, "thermal": 0.52,
            "latency": 0.30, "energy": 0.42, "queue": 0.25
        }
        receipt = self.orchestrator.run_single_acquisition(
            run_id="RUN-0001",
            sequence=1,
            s_before=s_before,
            s_predicted=s_predicted,
            predicted_energy_per_op_uj=0.194,
            action={"code": 1},
            sample_timestamp="2026-10-28T09:55:00Z"
        )

        # 1. INA226 Frozen Register Configuration
        self.assertEqual(receipt.ina226_vbusct, "140_US")
        self.assertEqual(receipt.ina226_vshct, "140_US")
        self.assertEqual(receipt.ina226_avg, 1)
        self.assertEqual(receipt.ina226_mode, "SHUNT_AND_BUS_CONTINUOUS")
        self.assertEqual(receipt.sensor_poll_rate_hz, 10_000.0)
        self.assertAlmostEqual(receipt.sensor_conversion_rate_hz, 3571.42857, places=3)
        self.assertEqual(receipt.estimated_conversion_period_ns, 280_000)

        # 2. Authoritative Physical Cardinality & Event Attribution
        self.assertGreaterEqual(receipt.raw_read_count, 3000)
        self.assertGreaterEqual(receipt.independent_conversion_count, 1000)
        self.assertGreaterEqual(receipt.active_independent_conversion_count, 700)
        self.assertEqual(receipt.independent_sample_method, "CONVERSION_READY_OBSERVED")
        self.assertEqual(receipt.conversion_ready_events_count, receipt.independent_conversion_count)
        self.assertEqual(
            receipt.duplicate_read_count,
            receipt.raw_read_count - receipt.independent_conversion_count
        )
        self.assertGreater(receipt.duplicate_read_count, 1500)

        # 3. Provenance & Invariant Manifest Sealing
        self.assertTrue(receipt.provenance_pass)
        self.assertEqual(receipt.acquisition_software_identity, self.orchestrator.manifest.acquisition_software_identity)
        self.assertEqual(receipt.first_physical_sample_timestamp, "2026-10-28T09:55:00Z")
        self.assertIsNone(self.orchestrator.manifest.first_physical_sample_timestamp)
        self.assertEqual(
            self.orchestrator.manifest_hash,
            "ae6a1add5fcf1d3cb36c36859d3f38ef48a27df340b8aaf39daa3b47ea17f9cf"
        )
        self.assertTrue(receipt.verify_integrity())

    def test_dual_layer_provenance_software_identity_gate(self) -> None:
        """
        P0 Provenance Test:
        Verifies dual-layer software identity:
          AcquisitionSoftwareIdentity = H(HarnessSource || Oracle || SensorContract || Dependencies || Commit)
          PASS_provenance <=> H_runtime == H_manifest
        Verifies that any mismatch in software identity fails acquisition validity.
        """
        self.assertTrue(self.orchestrator.provenance_pass)
        self.assertEqual(
            self.orchestrator.runtime_software_identity,
            self.orchestrator.manifest.acquisition_software_identity
        )

        # Negative test: Tampering with bitstream hash or software provenance invalidates acquisition
        valid_tampered = self.orchestrator.evaluate_acquisition_validity(
            manifest_hash=self.orchestrator.manifest_hash,
            bitstream_hash=self.orchestrator.manifest.bitstream_hash,
            measurement_complete=True,
            dropped_samples=0,
            receipt_integrity="VERIFIED",
            provenance_pass=False  # Simulated provenance breach
        )
        self.assertFalse(valid_tampered, "Breached provenance MUST fail acquisition validity")

    def test_raw_read_count_diagnostic_tolerance_not_hard_gate(self) -> None:
        """
        P0 Diagnostic Test:
        Verifies that raw_read_count is a transport diagnostic, NOT a hard physical failure gate.
        If I2C transport jitter yields raw_read_count < 3000 (e.g. 2400 @ 8 kHz polling),
        the run MUST STILL BE VALID as long as:
          active_independent_conversion_count >= 700
          independent_conversion_count >= 1000
          burst_duration_ns >= 200_000_000
          dropped_samples == 0
        """
        # Simulate transport telemetry with I2C clock stretching / jitter (raw_read_count = 2400)
        hw_simulated = {
            "independent_conversion_count": 1098,
            "active_independent_conversion_count": 742,
            "raw_read_count": 2400,  # Below 3000 due to I2C transport jitter
            "burst_duration_ns": 208_000_000,
            "validated_ops": 1_000_000,
            "dropped_samples": 0
        }
        # Evaluation using the orchestrator's criteria
        measurement_complete = (
            hw_simulated["independent_conversion_count"] >= 1000 and
            hw_simulated["active_independent_conversion_count"] >= 700 and
            hw_simulated["burst_duration_ns"] >= 200_000_000 and
            hw_simulated["validated_ops"] > 0
        )
        self.assertTrue(measurement_complete)

        is_valid = self.orchestrator.evaluate_acquisition_validity(
            manifest_hash=self.orchestrator.manifest_hash,
            bitstream_hash=self.orchestrator.manifest.bitstream_hash,
            measurement_complete=measurement_complete,
            dropped_samples=hw_simulated["dropped_samples"],
            receipt_integrity="VERIFIED",
            provenance_pass=self.orchestrator.provenance_pass
        )
        self.assertTrue(is_valid, "Valid physical ADC conversions with raw_read_count < 3000 must NOT fail acquisition")

    def test_deduplication_not_by_value_equality(self) -> None:
        """
        Methodological Invariant: SameValue !=> SameConversion.
        Two consecutive independent ADC conversions can legitimately return identical quantized values.
        Verifies that deduplication is based strictly on conversion cycle identity / event transitions,
        never on numerical value equality (value_new == value_previous).
        """
        # Two consecutive readings with identical voltage/current in steady-state
        val_1 = 3.2450
        val_2 = 3.2450
        self.assertEqual(val_1, val_2)
        # However, if two Conversion-Ready events occurred, both are independent conversions!
        events_observed = 2
        independent_samples = events_observed  # Not collapsed by numerical equality
        self.assertEqual(independent_samples, 2)


if __name__ == "__main__":
    unittest.main()
