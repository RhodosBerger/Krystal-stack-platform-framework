#!/usr/bin/env python3
"""
==============================================================================
KRYSTAL-STACK: UNIT TESTS FOR ISA PERFORMANCE TAXONOMIES & PROFILER
==============================================================================
File: tests/test_isa_performance_taxonomies_and_profiler.py
Covers:
  - Epistemic taxonomy dimensions (Family, Vector Width, Workload, Cache, Thermal)
  - Canonical ISA catalog (AVX-512, AVX2, AMX, DP4A, NEON, SVE2, RVV, SSE4.2, Scalar)
  - Williams Roofline analytical bounds & operational regimes
  - Profiling metrics (GFLOPS, Latency, Bandwidth, GFLOPS/W, Thermal Delta)
  - Profile-Guided ISA Governor & Pareto-optimal ranking
  - Constitutional Invariant: VITAL_MAX_HP == 6
==============================================================================
"""

import unittest
from krystal_kernel.scene_constructor_presets import VITAL_MAX_HP
from krystal_kernel.isa_performance_taxonomies_and_profiler import (
    IsaArchitectureFamily,
    VectorRegisterWidth,
    WorkloadArithmeticTaxon,
    CacheLocalityTaxon,
    ThermalPowerTaxon,
    IsaInstructionDescriptor,
    CANONICAL_ISA_CATALOG,
    IsaPerformanceProfile,
    IsaRecommendationReport,
    IsaPerformanceProfiler,
    ProfileGuidedIsaGovernor,
    GLOBAL_ISA_PERFORMANCE_PROFILER,
    GLOBAL_ISA_GOVERNOR
)


class TestIsaPerformanceTaxonomiesAndProfiler(unittest.TestCase):
    """Test suite for ISA performance taxonomies, profiler, and governor."""

    def setUp(self):
        self.profiler = IsaPerformanceProfiler()
        self.governor = ProfileGuidedIsaGovernor(self.profiler)

    def test_constitutional_invariant(self):
        """Verify the constitutional invariant VITAL_MAX_HP == 6 holds strictly."""
        self.assertEqual(VITAL_MAX_HP, 6)
        profile = self.profiler.profile_workload("AVX2_FMA", WorkloadArithmeticTaxon.FP32_CONVOLUTION)
        self.assertTrue(profile.passed_invariant_check)
        self.assertEqual(profile.vital_max_hp, 6)

    def test_taxonomy_dimensions_integrity(self):
        """Verify all taxonomy enumerations are intact and well-formed."""
        self.assertEqual(len(IsaArchitectureFamily), 5)
        self.assertIn(IsaArchitectureFamily.X86_64, IsaArchitectureFamily)
        self.assertIn(IsaArchitectureFamily.ARM64, IsaArchitectureFamily)
        self.assertIn(IsaArchitectureFamily.RISCV, IsaArchitectureFamily)
        self.assertIn(IsaArchitectureFamily.NPU_ACCELERATOR, IsaArchitectureFamily)
        self.assertIn(IsaArchitectureFamily.UNIVERSAL_SCALAR, IsaArchitectureFamily)

        self.assertEqual(len(VectorRegisterWidth), 7)
        self.assertEqual(len(WorkloadArithmeticTaxon), 6)
        self.assertEqual(len(CacheLocalityTaxon), 4)
        self.assertEqual(len(ThermalPowerTaxon), 4)

    def test_canonical_isa_catalog_completeness(self):
        """Verify the canonical catalog includes all mandatory instruction sets."""
        expected_isas = [
            "AVX512_VNNI", "AVX2_FMA", "SSE4_2_STREAMING", "INTEL_AMX_TILE",
            "INTEL_DP4A_TENSOR", "ARM_NEON_ASIMD", "ARM_SVE2_DYNAMIC",
            "RISCV_VECTOR_V1", "SCALAR_PORTABLE_C"
        ]
        for isa in expected_isas:
            self.assertIn(isa, CANONICAL_ISA_CATALOG)
            desc = CANONICAL_ISA_CATALOG[isa]
            self.assertGreater(desc.theoretical_peak_gflops_per_core, 0.0)
            self.assertGreater(desc.theoretical_bandwidth_gbps, 0.0)
            self.assertIsInstance(desc.canonical_opcodes, list)
            self.assertGreater(len(desc.canonical_opcodes), 0)

            # Dictionary serialization check
            desc_dict = desc.to_dict()
            self.assertEqual(desc_dict["vital_max_hp"], 6)

    def test_roofline_profiler_compute_vs_memory_bound(self):
        """Verify the Williams Roofline analytical bound identifies regimes correctly."""
        # 1. High arithmetic intensity (INT8 Tensor) -> Compute-bound on AVX-512
        prof_compute = self.profiler.profile_workload(
            isa_id="AVX512_VNNI",
            workload_taxon=WorkloadArithmeticTaxon.INT8_TENSOR_DOT,
            data_size_bytes=1024 * 1024
        )
        self.assertIsInstance(prof_compute, IsaPerformanceProfile)
        self.assertEqual(prof_compute.operational_regime, "COMPUTE_BOUND")
        self.assertGreater(prof_compute.measured_gflops, 50.0)
        self.assertGreater(prof_compute.speedup_vs_scalar, 3.0)

        # 2. Low arithmetic intensity (Streaming Memory) -> Memory-bound on AVX2
        prof_memory = self.profiler.profile_workload(
            isa_id="AVX2_FMA",
            workload_taxon=WorkloadArithmeticTaxon.STREAMING_MEMORY_BOUND,
            data_size_bytes=4 * 1024 * 1024
        )
        self.assertEqual(prof_memory.operational_regime, "MEMORY_BOUND")
        self.assertLess(prof_memory.arithmetic_intensity_flops_per_byte, 1.0)

    def test_profile_all_isas_sorting(self):
        """Verify profiling across all ISAs returns a ranked list."""
        profiles = self.profiler.profile_all_isas(
            workload_taxon=WorkloadArithmeticTaxon.INT8_TENSOR_DOT,
            data_size_bytes=512 * 1024
        )
        self.assertEqual(len(profiles), len(CANONICAL_ISA_CATALOG))
        # Ensure profiles are sorted descending by measured_gflops
        for i in range(len(profiles) - 1):
            self.assertGreaterEqual(profiles[i].measured_gflops, profiles[i + 1].measured_gflops)

    def test_profile_guided_governor_recommendation(self):
        """Verify the governor recommends the optimal ISA given workload and constraints."""
        report = self.governor.recommend_optimal_isa(
            workload_taxon=WorkloadArithmeticTaxon.INT8_TENSOR_DOT,
            data_size_bytes=1024 * 1024,
            current_thermal_celsius=50.0,
            power_budget_watts=45.0
        )

        self.assertIsInstance(report, IsaRecommendationReport)
        self.assertIn(report.selected_isa_id, ["INTEL_AMX_TILE", "INTEL_DP4A_TENSOR", "AVX512_VNNI"])
        self.assertGreater(report.projected_speedup, 1.0)
        self.assertLessEqual(report.projected_thermal_celsius, 85.0)
        self.assertEqual(report.vital_max_hp, 6)
        self.assertGreater(len(report.candidate_rankings), 0)

        report_dict = report.to_dict()
        self.assertEqual(report_dict["vital_max_hp"], 6)
        self.assertIn("selected_isa_id", report_dict)
        self.assertIn("rationale", report_dict)

    def test_thermal_downclock_guardrail_penalty(self):
        """Verify that high baseline temperatures penalize hot ISAs."""
        # Under extreme thermal strain (79°C), high-power AVX512 should be heavily penalized
        report_hot = self.governor.recommend_optimal_isa(
            workload_taxon=WorkloadArithmeticTaxon.FP32_CONVOLUTION,
            current_thermal_celsius=79.5,
            power_budget_watts=15.0  # Constrained power budget
        )
        # Should select more power-efficient or cooler ISA (e.g. AVX2 or NEON or DP4A)
        self.assertNotEqual(report_hot.selected_isa_id, "AVX512_VNNI")

    def test_global_singletons(self):
        """Verify the module global singletons are active and operational."""
        self.assertIsNotNone(GLOBAL_ISA_PERFORMANCE_PROFILER)
        self.assertIsNotNone(GLOBAL_ISA_GOVERNOR)
        rep = GLOBAL_ISA_GOVERNOR.recommend_optimal_isa(WorkloadArithmeticTaxon.SCALAR_CONTROL_LOOP)
        self.assertEqual(rep.vital_max_hp, 6)


if __name__ == "__main__":
    unittest.main()
