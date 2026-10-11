#!/usr/bin/env python3
"""
==============================================================================
UNIT & ENTERPRISE COMPLIANCE TEST SUITE: INTEL ENTERPRISE PERFORMANCE SUITE
==============================================================================
Module: tests/test_intel_enterprise_performance_suite.py
Description: Validates universal Intel microarchitecture acceleration,
             Arrhenius longevity guardrails, UMA zero-copy allocations,
             Thread Director core affinity, and enterprise telemetry.

Platform Invariant: VITAL_MAX_HP = 6
Author: Dušan Kopecký & Krystal-Stack Architecture Council (2026)
==============================================================================
"""

import sys
import unittest
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from krystal_kernel.intel_enterprise_performance_suite import (
    IntelMicroarchitecture,
    EnterprisePerformancePolicy,
    AccelerationExecutionState,
    HardwareTopologyProbe,
    ArrheniusLongevityGuardrail,
    IntelHardwareProber,
    TigerLakeStrategy,
    AlderRaptorLakeStrategy,
    MeteorLakeStrategy,
    LunarLakeStrategy,
    GenericIntelFallbackStrategy,
    IntelEnterprisePerformanceSuite,
    GLOBAL_INTEL_ENTERPRISE_SUITE,
    VITAL_MAX_HP
)


class TestIntelEnterprisePerformanceSuite(unittest.TestCase):
    """Test suite for validating universal Intel hardware acceleration."""

    def test_01_invariant_constant(self):
        """Ensures the sacred system invariant is strictly 6."""
        self.assertEqual(VITAL_MAX_HP, 6)

    def test_02_arrhenius_guardrail_nominal(self):
        """Tests that nominal post-undervolt temperature (65C, 0.95V) passes Arrhenius AF <= 1.05."""
        status = ArrheniusLongevityGuardrail.evaluate(temp_c=65.0, voltage_v=0.950)
        self.assertTrue(status.is_within_safety_limits)
        self.assertLessEqual(status.wear_acceleration_factor, 1.05)
        self.assertGreaterEqual(status.projected_silicon_lifespan_years, 10.0)
        self.assertIn("ENTERPRISE-CERTIFIED", status.compliance_certificate)
        self.assertEqual(status.vital_max_hp, 6)

    def test_03_arrhenius_guardrail_overtemp_protection(self):
        """Tests that severe thermal stress (>85C) triggers safety alerts."""
        status = ArrheniusLongevityGuardrail.evaluate(temp_c=92.0, voltage_v=1.150)
        self.assertFalse(status.is_within_safety_limits)
        self.assertGreater(status.wear_acceleration_factor, 1.05)
        self.assertIn("SAFETY-ALERT", status.compliance_certificate)

    def test_04_hardware_topology_prober(self):
        """Tests dynamic hardware topology probing on current system."""
        probe = IntelHardwareProber.probe()
        self.assertIsInstance(probe, HardwareTopologyProbe)
        self.assertGreater(probe.logical_cores, 0)
        self.assertGreater(probe.physical_cores, 0)
        self.assertGreater(probe.total_physical_ram_gb, 0.0)
        self.assertEqual(probe.vital_max_hp, 6)

    def test_05_tiger_lake_strategy(self):
        """Tests Tiger Lake 11th Gen tuning profile."""
        strategy = TigerLakeStrategy()
        mock_topo = HardwareTopologyProbe(
            cpu_model_string="Intel Core i5-1135G7",
            architecture_family=IntelMicroarchitecture.TIGER_LAKE,
            physical_cores=4,
            logical_cores=8,
            p_core_count=4,
            e_core_count=0,
            has_hybrid_topology=False,
            avx2_supported=True,
            avx512_supported=True,
            avx_vnni_supported=True,
            has_amx=False,
            has_intel_gpu=True,
            gpu_execution_units=96,
            gpu_architecture_generation="Intel Iris Xe Graphics",
            npu_present=False,
            npu_rated_tops=0.0,
            l1_cache_kb=192,
            l2_cache_kb=5120,
            l3_cache_kb=12288,
            total_physical_ram_gb=16.0,
            os_environment="win32",
            is_wsl=False,
            vital_max_hp=6
        )
        plan = strategy.calculate_power_plan(EnterprisePerformancePolicy.HIGH_THROUGHPUT_ENTERPRISE, mock_topo)
        self.assertEqual(plan.pl1_sustained_watts, 32.0)
        self.assertEqual(plan.pl2_burst_watts, 54.0)
        self.assertEqual(plan.voltage_offset_mv, -60.0)
        self.assertTrue(plan.bypassed_oem_throttle)

        ov = strategy.configure_openvino(EnterprisePerformancePolicy.HIGH_THROUGHPUT_ENTERPRISE, mock_topo)
        self.assertEqual(ov["inference_precision_hint"], "i8")
        self.assertEqual(ov["gpu_throughput_streams"], 4)
        self.assertIn("AVX-512", ov["isa_vectorization"])

        uma = strategy.get_uma_allocation(EnterprisePerformancePolicy.HIGH_THROUGHPUT_ENTERPRISE, mock_topo)
        self.assertTrue(uma.zero_copy_active)
        self.assertTrue(uma.aperture_clamp_bypassed)

    def test_06_alder_raptor_lake_strategy(self):
        """Tests Alder/Raptor Lake hybrid P/E core affinity routing."""
        strategy = AlderRaptorLakeStrategy()
        mock_topo = HardwareTopologyProbe(
            cpu_model_string="Intel Core i7-13700H",
            architecture_family=IntelMicroarchitecture.RAPTOR_LAKE,
            physical_cores=14,
            logical_cores=20,
            p_core_count=6,
            e_core_count=8,
            has_hybrid_topology=True,
            avx2_supported=True,
            avx512_supported=False,
            avx_vnni_supported=True,
            has_amx=False,
            has_intel_gpu=True,
            gpu_execution_units=96,
            gpu_architecture_generation="Intel Iris Xe Enhanced",
            npu_present=False,
            npu_rated_tops=0.0,
            l1_cache_kb=640,
            l2_cache_kb=16384,
            l3_cache_kb=24576,
            total_physical_ram_gb=32.0,
            os_environment="win32",
            is_wsl=False,
            vital_max_hp=6
        )
        plan = strategy.calculate_power_plan(EnterprisePerformancePolicy.HIGH_THROUGHPUT_ENTERPRISE, mock_topo)
        self.assertEqual(plan.pl1_sustained_watts, 55.0)

        affinity = strategy.get_core_affinity_mapping(mock_topo)
        self.assertIn("tensor_compute_cores", affinity)
        self.assertIn("background_telemetry_cores", affinity)
        # Compute cores mapped to P-cores
        self.assertEqual(affinity["tensor_compute_cores"][:4], [0, 1, 2, 3])

    def test_07_meteor_and_lunar_lake_npu_strategies(self):
        """Tests Meteor Lake and Lunar Lake with NPU offloading."""
        meteor = MeteorLakeStrategy()
        lunar = LunarLakeStrategy()

        mock_lunar = HardwareTopologyProbe(
            cpu_model_string="Intel Core Ultra 7 258V",
            architecture_family=IntelMicroarchitecture.LUNAR_LAKE,
            physical_cores=8,
            logical_cores=8,
            p_core_count=4,
            e_core_count=4,
            has_hybrid_topology=True,
            avx2_supported=True,
            avx512_supported=False,
            avx_vnni_supported=True,
            has_amx=True,
            has_intel_gpu=True,
            gpu_execution_units=128,
            gpu_architecture_generation="Intel Arc Xe2-LPG",
            npu_present=True,
            npu_rated_tops=48.0,
            l1_cache_kb=512,
            l2_cache_kb=10240,
            l3_cache_kb=16384,
            total_physical_ram_gb=32.0,
            os_environment="win32",
            is_wsl=False,
            vital_max_hp=6
        )
        lunar_ov = lunar.configure_openvino(EnterprisePerformancePolicy.HIGH_THROUGHPUT_ENTERPRISE, mock_lunar)
        self.assertIn("NPU", lunar_ov["device"])
        self.assertIn("48 TOPS", lunar_ov["isa_vectorization"])
        self.assertGreater(lunar_ov["predicted_tokens_sec"], 40.0)

    def test_08_enterprise_suite_end_to_end(self):
        """Tests end-to-end optimization report and singleton invocation."""
        suite = GLOBAL_INTEL_ENTERPRISE_SUITE
        report = suite.apply_optimizations(current_temp_c=65.0, voltage_v=0.950)

        self.assertIsNotNone(report.session_id)
        self.assertTrue(report.system_invariant_verified)
        self.assertEqual(report.vital_max_hp, 6)
        self.assertGreater(report.overall_performance_boost_pct, 50.0)
        self.assertEqual(report.rendering_latency_target_ms, 0.38)

    def test_09_enterprise_benchmark(self):
        """Tests the comparative enterprise benchmark."""
        suite = GLOBAL_INTEL_ENTERPRISE_SUITE
        bench = suite.run_enterprise_benchmark()

        self.assertIn("benchmark_id", bench)
        self.assertGreater(bench["unlocked_tokens_per_sec"], bench["stock_tokens_per_sec"])
        self.assertGreater(bench["token_throughput_gain_pct"], 0.0)
        self.assertEqual(bench["vital_max_hp"], 6)

    def test_10_prometheus_metrics_export(self):
        """Tests Prometheus / OpenTelemetry text format generation."""
        suite = GLOBAL_INTEL_ENTERPRISE_SUITE
        metrics = suite.export_prometheus_metrics()

        self.assertIn("intel_pl1_limit_watts", metrics)
        self.assertIn("intel_arrhenius_wear_factor", metrics)
        self.assertIn("intel_uma_pool_mb", metrics)
        self.assertIn("intel_vital_max_hp 6", metrics)


if __name__ == "__main__":
    unittest.main()
