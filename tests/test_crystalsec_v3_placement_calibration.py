"""
Unit & Integration Test Suite for CrystalSec 3.0 Phase 5.1:
Hardware Evidence Calibration, Placement Regret & Cross-Tier Model Validation.
"""

import unittest
import time
from typing import Dict, List, Any

from krystal_kernel.verification.crystalsec_v3_placement_calibration import (
    WorkloadClass,
    SyntheticWorkload,
    WorkloadSynthesizer,
    TaxonomyExecutionEngine,
    EmpiricalCostModel,
    EmpiricalDistributionStats,
    CalibrationErrorTracker,
    PlacementRegretTracker,
    ConfidenceAwareRouter,
    PlacementCalibrationController,
    GLOBAL_PLACEMENT_CALIBRATION_CONTROLLER,
    VITAL_MAX_HP,
)
from krystal_kernel.verification.crystalsec_v3_heterogeneous_execution import (
    ExecutionTierID,
    ExecutionMode,
    DirectSimAdapter,
    CpuVnniAdapter,
    IrisXeVulkanAdapter,
)


class TestCrystalSecV3PlacementCalibration(unittest.TestCase):
    def setUp(self):
        self.triad_ctrl = GLOBAL_PLACEMENT_CALIBRATION_CONTROLLER.triad_controller
        self.adapters = self.triad_ctrl.adapters
        self.cost_model = EmpiricalCostModel(window_size=50)
        self.calib_tracker = CalibrationErrorTracker(max_samples=100)
        self.regret_tracker = PlacementRegretTracker(max_history=100, misplacement_threshold=0.05)
        self.router = ConfidenceAwareRouter(
            cost_model=self.cost_model,
            adapters=self.adapters,
            uncertainty_threshold=0.35,
            min_probe_samples=3,
        )

    def test_vital_max_hp_universal_invariant(self):
        """Universal invariant VITAL_MAX_HP == 6 must hold unconditionally."""
        self.assertEqual(VITAL_MAX_HP, 6)

    def test_workload_taxonomy_generation(self):
        """Validates synthetic workload generation for all 5 taxonomy classes."""
        classes = [
            WorkloadClass.INT8_DOT_PRODUCT,
            WorkloadClass.FP32_VECTOR_MATH,
            WorkloadClass.MEMORY_BOUND_STREAM,
            WorkloadClass.BRANCH_HEAVY_CONTROL,
            WorkloadClass.MIXED_COMPUTE_MEMORY,
        ]
        for wc in classes:
            wl = WorkloadSynthesizer.generate(wc, size=64)
            self.assertEqual(wl.workload_class, wc)
            self.assertGreater(wl.elements_count, 0)
            self.assertGreater(wl.working_set_bytes, 0)
            self.assertGreater(wl.expected_intensity_flop_per_byte, 0.0)

    def test_taxonomy_execution_across_tiers(self):
        """Executes INT8 dot product on T0, T1, and T2, verifying outputs match."""
        wl = WorkloadSynthesizer.generate(WorkloadClass.INT8_DOT_PRODUCT, size=64)
        out_t0, lat_t0, telem_t0 = TaxonomyExecutionEngine.execute_workload(wl, ExecutionTierID.TIER_0_DIRECTSIM, self.adapters)
        out_t1, lat_t1, telem_t1 = TaxonomyExecutionEngine.execute_workload(wl, ExecutionTierID.TIER_1_CPU_VNNI, self.adapters)
        out_t2, lat_t2, telem_t2 = TaxonomyExecutionEngine.execute_workload(wl, ExecutionTierID.TIER_2_IRIS_XE, self.adapters)

        self.assertEqual(len(out_t0), len(out_t1))
        self.assertEqual(len(out_t0), len(out_t2))
        self.assertAlmostEqual(out_t0[0], out_t1[0], places=3)
        self.assertAlmostEqual(out_t0[0], out_t2[0], places=3)
        self.assertGreater(lat_t0, 0.0)
        self.assertGreater(lat_t1, 0.0)
        self.assertGreater(lat_t2, 0.0)

    def test_branch_heavy_workload_favors_cpu(self):
        """
        Microarchitectural invariant: on branch-heavy workloads with irregular
        control flow, CPU out-of-order execution handles branching more efficiently
        than GPU SIMT warp divergence.
        """
        wl = WorkloadSynthesizer.generate(WorkloadClass.BRANCH_HEAVY_CONTROL, size=128)
        _, lat_cpu, _ = TaxonomyExecutionEngine.execute_workload(wl, ExecutionTierID.TIER_1_CPU_VNNI, self.adapters)
        _, lat_gpu, _ = TaxonomyExecutionEngine.execute_workload(wl, ExecutionTierID.TIER_2_IRIS_XE, self.adapters)

        # CPU latency is lower than GPU due to branch divergence penalty on GPU
        self.assertLess(lat_cpu, lat_gpu)

    def test_memory_bound_workload_uma_pressure(self):
        """
        Verifies that MEMORY_BOUND_STREAM produces elevated UMA bus contention
        on Iris Xe integrated graphics due to shared memory traffic.
        """
        wl = WorkloadSynthesizer.generate(WorkloadClass.MEMORY_BOUND_STREAM, size=512)
        _, _, telem_gpu = TaxonomyExecutionEngine.execute_workload(wl, ExecutionTierID.TIER_2_IRIS_XE, self.adapters)
        
        self.assertGreaterEqual(telem_gpu["uma_pressure"], 0.70)

    def test_empirical_cost_distribution_statistics(self):
        """Tests rolling distribution moments (mean, median, p95, p99, std, cv, iqr)."""
        wc = WorkloadClass.FP32_VECTOR_MATH
        tier = ExecutionTierID.TIER_1_CPU_VNNI

        # Record 20 known observations
        for i in range(1, 21):
            self.cost_model.record_observation(wc, tier, float(i) * 0.01)

        stats = self.cost_model.get_distribution_stats(wc, tier)
        self.assertEqual(stats.sample_count, 20)
        self.assertAlmostEqual(stats.mean_val, 0.105, places=3)
        self.assertAlmostEqual(stats.median_val, 0.105, places=3)
        self.assertGreater(stats.p95_val, stats.median_val)
        self.assertGreater(stats.p99_val, stats.p95_val)
        self.assertGreater(stats.std_val, 0.0)
        self.assertGreater(stats.iqr_val, 0.0)

    def test_uncertainty_score_decreases_with_sample_count(self):
        """Validates that statistical uncertainty decreases monotonically as observations accumulate."""
        wc = WorkloadClass.INT8_DOT_PRODUCT
        tier = ExecutionTierID.TIER_2_IRIS_XE

        u_initial = self.cost_model.get_distribution_stats(wc, tier).uncertainty_score
        self.assertEqual(u_initial, 1.0)  # Prior uncertainty

        for _ in range(5):
            self.cost_model.record_observation(wc, tier, 0.035)
        u_5 = self.cost_model.get_distribution_stats(wc, tier).uncertainty_score

        for _ in range(25):
            self.cost_model.record_observation(wc, tier, 0.035)
        u_30 = self.cost_model.get_distribution_stats(wc, tier).uncertainty_score

        self.assertLess(u_5, u_initial)
        self.assertLess(u_30, u_5)

    def test_calibration_error_calculation(self):
        """Verifies E_latency = |L_pred - L_obs| / max(L_obs, eps) and MACE calculation."""
        sample = self.calib_tracker.record_prediction(
            workload_id="test_calib_1",
            workload_class=WorkloadClass.INT8_DOT_PRODUCT,
            tier=ExecutionTierID.TIER_1_CPU_VNNI,
            pred_lat=0.10,
            obs_lat=0.08,
            pred_uma=0.20,
            obs_uma=0.20,
            pred_temp=0.15,
            obs_temp=0.15,
        )
        expected_lat_err = abs(0.10 - 0.08) / 0.08  # 0.25 (25%)
        self.assertAlmostEqual(sample.latency_error, expected_lat_err, places=3)
        self.assertAlmostEqual(sample.bias_latency, 0.02, places=3)

        metrics = self.calib_tracker.get_calibration_metrics()
        self.assertEqual(metrics["sample_count"], 1)
        self.assertTrue(metrics["is_well_calibrated"])

    def test_prediction_bias_tracks_directionality(self):
        """Verifies that over-prediction yields positive bias and under-prediction yields negative bias."""
        # Over-prediction
        self.calib_tracker.record_prediction("s1", WorkloadClass.FP32_VECTOR_MATH, ExecutionTierID.TIER_1_CPU_VNNI, 0.20, 0.10)
        self.assertGreater(self.calib_tracker.get_calibration_metrics()["prediction_bias"], 0.0)

        # Under-prediction
        tracker2 = CalibrationErrorTracker()
        tracker2.record_prediction("s2", WorkloadClass.FP32_VECTOR_MATH, ExecutionTierID.TIER_1_CPU_VNNI, 0.10, 0.20)
        self.assertLess(tracker2.get_calibration_metrics()["prediction_bias"], 0.0)

    def test_placement_regret_calculation(self):
        """Verifies R_placement = max(0, U_best - U_selected)."""
        # Case A: Selected tier was optimal (U_sel == U_best) -> Regret = 0.0
        obs1 = self.regret_tracker.record_decision(
            workload_id="dec_1",
            workload_class=WorkloadClass.INT8_DOT_PRODUCT,
            selected_tier=ExecutionTierID.TIER_2_IRIS_XE,
            selected_utility=0.85,
            best_tier=ExecutionTierID.TIER_2_IRIS_XE,
            best_utility=0.85,
        )
        self.assertEqual(obs1.regret, 0.0)
        self.assertFalse(obs1.is_misplacement)

        # Case B: Suboptimal choice (selected utility 0.60 vs best 0.80) -> Regret = 0.20
        obs2 = self.regret_tracker.record_decision(
            workload_id="dec_2",
            workload_class=WorkloadClass.BRANCH_HEAVY_CONTROL,
            selected_tier=ExecutionTierID.TIER_2_IRIS_XE,
            selected_utility=0.60,
            best_tier=ExecutionTierID.TIER_1_CPU_VNNI,
            best_utility=0.80,
        )
        self.assertAlmostEqual(obs2.regret, 0.20, places=3)
        self.assertTrue(obs2.is_misplacement)

    def test_cumulative_regret_and_rate(self):
        """Verifies cumulative regret summation and average regret rate."""
        self.regret_tracker.record_decision("w1", WorkloadClass.INT8_DOT_PRODUCT, ExecutionTierID.TIER_1_CPU_VNNI, 0.70, ExecutionTierID.TIER_2_IRIS_XE, 0.80)  # R=0.10
        self.regret_tracker.record_decision("w2", WorkloadClass.INT8_DOT_PRODUCT, ExecutionTierID.TIER_2_IRIS_XE, 0.85, ExecutionTierID.TIER_2_IRIS_XE, 0.85)  # R=0.00
        self.regret_tracker.record_decision("w3", WorkloadClass.FP32_VECTOR_MATH, ExecutionTierID.TIER_1_CPU_VNNI, 0.75, ExecutionTierID.TIER_2_IRIS_XE, 0.85)  # R=0.10

        metrics = self.regret_tracker.get_regret_metrics()
        self.assertEqual(metrics["decision_count"], 3)
        self.assertAlmostEqual(metrics["cumulative_regret"], 0.20, places=3)
        self.assertAlmostEqual(metrics["average_regret_rate"], 0.20 / 3.0, places=3)
        self.assertEqual(metrics["misplacement_count"], 2)

    def test_confidence_aware_routing_triggers_micro_probe(self):
        """
        When empirical data is sparse (N < 3), ConfidenceAwareRouter triggers
        a micro-probe execution before routing.
        """
        wl = WorkloadSynthesizer.generate(WorkloadClass.INT8_DOT_PRODUCT, size=64)
        target, probed, reason = self.router.evaluate_routing(wl, ExecutionTierID.TIER_2_IRIS_XE)

        self.assertTrue(probed)
        self.assertEqual(target, ExecutionTierID.TIER_2_IRIS_XE)
        self.assertIn("Probed successfully", reason)
        self.assertEqual(self.router.probe_executions_count, 1)

    def test_confidence_aware_routing_direct_when_calibrated(self):
        """
        When empirical confidence is high (N >= 3 and u <= 0.35), routing proceeds
        directly without probe overhead.
        """
        wc = WorkloadClass.INT8_DOT_PRODUCT
        tier = ExecutionTierID.TIER_1_CPU_VNNI

        # Seed with 10 observations to build statistical confidence
        for _ in range(10):
            self.cost_model.record_observation(wc, tier, 0.12)

        wl = WorkloadSynthesizer.generate(wc, size=64)
        target, probed, reason = self.router.evaluate_routing(wl, tier)

        self.assertFalse(probed)
        self.assertEqual(target, tier)
        self.assertIn("Confident routing", reason)

    def test_calibration_controller_full_suite_run(self):
        """Runs the complete calibration suite across all 5 classes and validates output."""
        controller = PlacementCalibrationController()
        report = controller.run_calibration_suite(iterations_per_class=3)

        self.assertEqual(report["phase"], "CrystalSec3_Phase5_1_Calibration")
        self.assertEqual(len(report["classes_evaluated"]), 5)
        self.assertIn("INT8_DOT_PRODUCT", report["empirical_distributions"])
        self.assertIn("BRANCH_HEAVY_CONTROL", report["empirical_distributions"])
        self.assertIn("calibration_metrics", report)
        self.assertIn("regret_metrics", report)
    def test_probe_failure_falls_back_to_safe_tier(self):
        """If micro-probe fails on a candidate tier, routing safely falls back to CPU or DirectSim."""
        class FaultyAdapter(IrisXeVulkanAdapter):
            def execute(self, workload):
                raise RuntimeError("DEVICE_REMOVED_SIMULATION")

        faulty_adapters = dict(self.adapters)
        faulty_adapters[ExecutionTierID.TIER_2_IRIS_XE] = FaultyAdapter()
        router = ConfidenceAwareRouter(
            cost_model=EmpiricalCostModel(),
            adapters=faulty_adapters,
            uncertainty_threshold=0.35,
            min_probe_samples=3,
        )
        wl = WorkloadSynthesizer.generate(WorkloadClass.INT8_DOT_PRODUCT, size=64)
        target, probed, reason = router.evaluate_routing(wl, ExecutionTierID.TIER_2_IRIS_XE)

        self.assertTrue(probed)
        self.assertIn(target, (ExecutionTierID.TIER_1_CPU_VNNI, ExecutionTierID.TIER_0_DIRECTSIM))
        self.assertIn("Probe failed/degraded", reason)

    def test_online_bayesian_cost_prior_updating(self):
        """Tests that empirical distribution updates predict accurately as true latency drifts."""
        wc = WorkloadClass.MEMORY_BOUND_STREAM
        tier = ExecutionTierID.TIER_1_CPU_VNNI

        # Initial prediction is based on prior
        p_init = self.cost_model.predict_latency(wc, tier)

        # Ingest observed stream latencies
        for _ in range(15):
            self.cost_model.record_observation(wc, tier, 0.045)

        p_updated = self.cost_model.predict_latency(wc, tier)
        self.assertAlmostEqual(p_updated, 0.045, places=3)
        self.assertNotEqual(p_init, p_updated)


if __name__ == "__main__":
    unittest.main()
