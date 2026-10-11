"""
CrystalSec 3.0 Phase 5.2 - Evidence Integrity regression tests.

These tests exist because Phase 5 / 5.1 shipped with:
  - T1 / T2 adapters labelled HARDWARE_MEASURED while executing pure Python on the host CPU,
  - `real_dp4a_execution` reporting True exactly when the GPU path was emulated,
  - per-tier latency constants added to "measured" wall clock,
  - a regret tracker that emitted 'OPTIMAL' (even with zero decisions),
  - calibration predictions that already contained the observation they were scored against.

Each test pins one of those failure modes so it cannot silently return.
"""

import statistics
import unittest

from krystal_kernel.verification.crystalsec_v3_heterogeneous_execution import (
    EvidenceLabel,
    ExecutionMode,
    ExecutionTierID,
    HeterogeneousTriadController,
    create_default_live_state,
)
from krystal_kernel.verification.crystalsec_v3_placement_calibration import (
    CalibrationErrorTracker,
    HardwareEvidenceProbe,
    PlacementCalibrationController,
    PlacementQuality,
    PlacementRegretTracker,
    TaxonomyExecutionEngine,
    VITAL_MAX_HP,
    WorkloadClass,
    WorkloadSynthesizer,
)


class _FakeDriver:
    """Minimal stand-in for VulkanComputeDriver (no GPU required)."""

    def __init__(self, real_gpu: bool):
        self._real = real_gpu
        self.device_name = "FakeDevice"
        self.real_gpu_dispatch = False
        self.compute_backend = "CPU_EMULATION"
        self.cleaned = False

    def execute_raymarch(self, w, h, t, readback=True):
        self.real_gpu_dispatch = self._real
        self.compute_backend = "VULKAN_GPU" if self._real else "CPU_EMULATION"
        return [1] * (w * h), [0] * (w * h)

    def execute_raymarch_cpu(self, w, h, t):
        self.real_gpu_dispatch = False
        self.compute_backend = "CPU_EMULATION"
        return [1] * (w * h), [0] * (w * h)

    def cleanup(self):
        self.cleaned = True


class TestEvidenceLabels(unittest.TestCase):
    def setUp(self):
        self.triad = HeterogeneousTriadController()

    def test_vital_max_hp_invariant(self):
        self.assertEqual(VITAL_MAX_HP, 6)

    def test_no_python_tier_claims_hardware_measured(self):
        for tier_id, adapter in self.triad.adapters.items():
            cap = adapter.detect()
            self.assertNotEqual(
                cap.epistemic_label, EvidenceLabel.HARDWARE_MEASURED.value,
                f"{tier_id.value} executes on the host CPU in Python and must not claim HARDWARE_MEASURED",
            )

    def test_capability_metrics_are_labelled_modeled_prior(self):
        for adapter in self.triad.adapters.values():
            self.assertEqual(adapter.detect().metrics_evidence, EvidenceLabel.MODELED_PRIOR.value)

    def test_iris_xe_does_not_report_real_dp4a_when_emulating(self):
        iris = self.triad.adapters[ExecutionTierID.TIER_2_IRIS_XE]
        telem = iris.get_telemetry()
        self.assertFalse(telem["real_dp4a_execution"])
        self.assertEqual(telem["dispatch_backend"], EvidenceLabel.HOST_CPU_EMULATED.value)

    def test_vnni_capability_is_separate_from_execution(self):
        cap = self.triad.adapters[ExecutionTierID.TIER_1_CPU_VNNI].backend.capability
        self.assertFalse(cap.real_vnni_execution)
        self.assertEqual(cap.vnni_capable, cap.has_vnni)
        self.assertEqual(cap.execution_evidence, EvidenceLabel.HOST_CPU_EMULATED.value)

    def test_npu_remains_physically_unavailable(self):
        cap = self.triad.adapters[ExecutionTierID.TIER_3_DEDICATED_NPU].detect()
        self.assertFalse(cap.physically_present)
        self.assertEqual(cap.epistemic_label, "PHYSICALLY_UNAVAILABLE")


class TestMeasurementIsWallClockOnly(unittest.TestCase):
    def setUp(self):
        self.adapters = HeterogeneousTriadController().adapters

    def test_no_injected_gpu_branch_penalty(self):
        """The removed constant alone was +0.120 ms; a 1-element loop must be far below it."""
        wl = WorkloadSynthesizer.generate(WorkloadClass.BRANCH_HEAVY_CONTROL, size=1)
        lats = [
            TaxonomyExecutionEngine.execute_workload(wl, ExecutionTierID.TIER_2_IRIS_XE, self.adapters)[1]
            for _ in range(7)
        ]
        self.assertLess(statistics.median(lats), 0.110)

    def test_placement_engine_uses_empirical_latency_not_hardcoded_branch_values(self):
        ctrl = PlacementCalibrationController()
        for tier in (ExecutionTierID.TIER_0_DIRECTSIM, ExecutionTierID.TIER_1_CPU_VNNI, ExecutionTierID.TIER_2_IRIS_XE):
            for _ in range(20):
                ctrl.cost_model.record_observation(WorkloadClass.BRANCH_HEAVY_CONTROL, tier, 0.5)
        decision = ctrl.triad_controller.placement_engine.evaluate_placement(
            workload_id="W-BR",
            action={"action_id": "ACT-BR", "type": WorkloadClass.BRANCH_HEAVY_CONTROL.value},
            live_state=create_default_live_state(),
            mode=ExecutionMode.BALANCED_MODE,
        )
        self.assertAlmostEqual(decision.predicted_latency_ms, 0.5, places=6)


class TestRegretVerdicts(unittest.TestCase):
    T0 = ExecutionTierID.TIER_0_DIRECTSIM
    T1 = ExecutionTierID.TIER_1_CPU_VNNI

    def test_empty_history_is_insufficient_evidence_not_optimal(self):
        m = PlacementRegretTracker().get_regret_metrics()
        self.assertEqual(m["placement_quality"], PlacementQuality.INSUFFICIENT_EVIDENCE.value)
        self.assertNotEqual(m["model_optimality"], "OPTIMAL")

    def test_few_decisions_are_insufficient_evidence(self):
        tr = PlacementRegretTracker(min_decisions_for_verdict=10)
        for i in range(5):
            tr.record_decision(f"w{i}", WorkloadClass.INT8_DOT_PRODUCT, self.T0, 1.0, self.T0, 1.0)
        self.assertEqual(tr.get_regret_metrics()["placement_quality"], PlacementQuality.INSUFFICIENT_EVIDENCE.value)

    def test_zero_regret_is_scoped_and_single_tier_flagged(self):
        tr = PlacementRegretTracker()
        for i in range(12):
            tr.record_decision(f"w{i}", WorkloadClass.INT8_DOT_PRODUCT, self.T0, 1.0, self.T0, 1.0,
                               evaluated_tiers=[self.T0, self.T1])
        m = tr.get_regret_metrics()
        self.assertEqual(m["placement_quality"], PlacementQuality.ZERO_REGRET_ON_EVALUATED_SET.value)
        self.assertTrue(m["single_tier_selection"])
        self.assertEqual(m["selected_tier_distribution"], {self.T0.value: 12})
        self.assertEqual(m["evaluated_tiers"], sorted([self.T0.value, self.T1.value]))

    def test_mutation_deliberately_wrong_placement_yields_nonzero_regret(self):
        """If a placement policy is wrong, the tracker must notice (guards against structural R=0)."""
        tr = PlacementRegretTracker()
        for i in range(12):
            # Selected tier is 2x slower than the best observed tier: U_sel = 0.5
            tr.record_decision(f"w{i}", WorkloadClass.BRANCH_HEAVY_CONTROL, self.T1, 0.5, self.T0, 1.0,
                               selected_latency_ms=0.2, best_latency_ms=0.1)
        m = tr.get_regret_metrics()
        self.assertGreater(m["average_regret_rate"], 0.0)
        self.assertEqual(m["placement_quality"], PlacementQuality.PLACEMENT_RECALIBRATION_REQUIRED.value)
        self.assertAlmostEqual(m["mean_latency_regret_ms"], 0.1, places=6)

    def test_gap_within_measured_noise_is_not_a_misplacement(self):
        tr = PlacementRegretTracker()
        obs_noise = tr.record_decision("n1", WorkloadClass.INT8_DOT_PRODUCT, self.T0, 0.90, self.T1, 1.0,
                                       selected_latency_ms=0.0200, best_latency_ms=0.0180, noise_floor_ms=0.0030)
        obs_real = tr.record_decision("n2", WorkloadClass.INT8_DOT_PRODUCT, self.T0, 0.50, self.T1, 1.0,
                                      selected_latency_ms=0.0400, best_latency_ms=0.0200, noise_floor_ms=0.0030)
        self.assertTrue(obs_noise.within_measurement_noise)
        self.assertFalse(obs_noise.is_misplacement)
        self.assertFalse(obs_real.within_measurement_noise)
        self.assertTrue(obs_real.is_misplacement)

    def test_optimal_label_never_emitted(self):
        for q in PlacementQuality:
            self.assertNotEqual(q.value, "OPTIMAL")


class TestCalibrationEvidence(unittest.TestCase):
    def test_modeled_secondary_metrics_excluded_from_composite(self):
        tr = CalibrationErrorTracker()
        s = tr.record_prediction("c1", WorkloadClass.INT8_DOT_PRODUCT, ExecutionTierID.TIER_0_DIRECTSIM,
                                 pred_lat=0.12, obs_lat=0.10, pred_uma=0.9, obs_uma=0.1,
                                 secondary_evidence=EvidenceLabel.MODELED_PRIOR.value)
        self.assertAlmostEqual(s.composite_error, s.latency_error, places=9)
        self.assertFalse(tr.get_calibration_metrics()["secondary_metrics_included"])

    def test_empty_tracker_is_not_well_calibrated(self):
        m = CalibrationErrorTracker().get_calibration_metrics()
        self.assertFalse(m["is_well_calibrated"])
        self.assertEqual(m["calibration_status"], "INSUFFICIENT_EVIDENCE")


class TestCalibrationSuiteReport(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ctrl = PlacementCalibrationController()
        cls.report = cls.ctrl.run_calibration_suite(iterations_per_class=3, repeats_per_measurement=3)

    def test_report_never_contains_optimal(self):
        self.assertNotIn('"OPTIMAL"', repr(self.report).replace("'", '"'))

    def test_report_declares_evidence_provenance(self):
        r = self.report
        self.assertEqual(r["evidence_revision"], "PHASE_5_2_EVIDENCE_INTEGRITY")
        self.assertEqual(r["telemetry_evidence"], "MODELED_PRIOR")
        self.assertIn("HOST_CPU_EMULATED", r["latency_evidence_counts"])
        self.assertNotIn("HARDWARE_MEASURED", r["latency_evidence_counts"])
        self.assertTrue(r["unsupported_claims"])
        self.assertIn("selected_tier_distribution", r["regret_metrics"])

    def test_router_is_actually_exercised(self):
        self.assertGreater(self.report["confidence_router_probes"], 0)

    def test_first_prediction_uses_prior_not_its_own_observation(self):
        first = self.ctrl.calibration_tracker.samples[0]
        prior = self.ctrl.cost_model.priors[(first.workload_class, first.tier)]
        self.assertAlmostEqual(first.predicted_latency_ms, prior, places=9)


class TestHardwareEvidenceProbe(unittest.TestCase):
    def test_real_vulkan_rows_are_hardware_measured(self):
        drv = _FakeDriver(real_gpu=True)
        res = HardwareEvidenceProbe(driver_factory=lambda: drv).run_sdf_raymarch_sweep(((4, 4),), repeats=2)
        self.assertEqual(res["status"], "HARDWARE_MEASURED")
        self.assertEqual(res["rows"][0]["t2_evidence"], "HARDWARE_MEASURED")
        self.assertEqual(res["rows"][0]["glyph_match_ratio"], 1.0)
        self.assertTrue(drv.cleaned)

    def test_cpu_fallback_rows_are_not_hardware_measured(self):
        res = HardwareEvidenceProbe(driver_factory=lambda: _FakeDriver(real_gpu=False)).run_sdf_raymarch_sweep(((4, 4),), repeats=2)
        self.assertEqual(res["rows"][0]["t2_evidence"], "HOST_CPU_EMULATED")
        self.assertEqual(res["status"], "HOST_CPU_EMULATED")
        self.assertIsNone(res["smallest_evaluated_size_where_t2_faster_px"])

    def test_driver_init_failure_is_unavailable(self):
        def boom():
            raise OSError("vulkan-1.dll not found")
        res = HardwareEvidenceProbe(driver_factory=boom).run_sdf_raymarch_sweep()
        self.assertEqual(res["status"], "UNAVAILABLE")
        self.assertEqual(res["rows"], [])


if __name__ == "__main__":
    unittest.main()
