#!/usr/bin/env python3
"""
==============================================================================
KRYSTAL-STACK: UNIT TEST SUITE
POLYMORPHIC SHAPE & TELEMETRY ACCELERATED BENCHMARK ENGINE
==============================================================================
Tests:
  1. Universal Invariants: VITAL_MAX_HP == 6, GOLDEN_RATIO == 1.61803398875.
  2. All 6 Polymorphic Shape Archetype Mesh Generation.
  3. 8-Octant Spatial Partitioning & Centroid Calculation.
  4. Benchmark Execution across Acceleration Tiers (Scalar vs NPU vs Full Accelerated).
  5. Speedup and Throughput Evaluation (GRADE_S_SUPREME >= 4.0x).
  6. Hardware Telemetry Integration (Ohmic Watts, GPU Temp, L2 Cache).
  7. Report Serialization & History Ledger Tracking.
==============================================================================
"""

import unittest
from typing import Dict, Any

from krystal_kernel.polymorphic_telemetry_benchmark_engine import (
    VITAL_MAX_HP,
    GOLDEN_RATIO,
    AccelerationTier,
    BenchmarkGrade,
    OctantDistribution,
    ArchetypeBenchmarkMetric,
    ComprehensiveBenchmarkReport,
    PolymorphicTelemetryBenchmarkEngine,
    GLOBAL_POLYMORPHIC_BENCHMARK_ENGINE
)


class TestPolymorphicTelemetryBenchmark(unittest.TestCase):

    def setUp(self):
        self.engine = PolymorphicTelemetryBenchmarkEngine()

    def test_universal_invariants(self):
        """Verify strict adherence to platform invariant VITAL_MAX_HP == 6."""
        self.assertEqual(VITAL_MAX_HP, 6, "VITAL_MAX_HP must strictly be 6 across all benchmarks")
        self.assertAlmostEqual(GOLDEN_RATIO, 1.61803398875, places=7)

    def test_all_six_archetypes_generation(self):
        """Verify mesh generation for all 6 canonical polymorphic archetypes."""
        self.assertEqual(len(self.engine.ARCHETYPES), 6)
        for arch in self.engine.ARCHETYPES:
            verts = self.engine.generate_mesh_vertices(arch, density_multiplier=1)
            self.assertGreater(len(verts), 20, f"Archetype {arch} must generate valid vertices")
            for v in verts:
                self.assertEqual(len(v), 3)

    def test_eight_octants_spatial_partitioning(self):
        """Verify 8-octant spatial partitioning logic, counts, and percentages."""
        # Cube spanning all 8 octants
        test_cube = [
            (1.0, 1.0, 1.0),   # oct 7
            (-1.0, 1.0, 1.0),  # oct 6
            (1.0, -1.0, 1.0),  # oct 5
            (-1.0, -1.0, 1.0), # oct 4
            (1.0, 1.0, -1.0),  # oct 3
            (-1.0, 1.0, -1.0), # oct 2
            (1.0, -1.0, -1.0), # oct 1
            (-1.0, -1.0, -1.0) # oct 0
        ]
        dist = self.engine.evaluate_8_octants(test_cube)
        self.assertEqual(len(dist.octant_counts), 8)
        self.assertEqual(len(dist.octant_percentages), 8)
        self.assertAlmostEqual(sum(dist.octant_percentages), 100.0, places=2)
        self.assertEqual(dist.centroid, (0.0, 0.0, 0.0))
        self.assertGreater(dist.bounding_box_volume, 0.0)

    def test_benchmark_acceleration_tiers(self):
        """Verify speedup scaling across different software acceleration tiers."""
        # 1. Standard Scalar Baseline
        rep_baseline = self.engine.execute_benchmark(
            acceleration=AccelerationTier.STANDARD_SCALAR_BASELINE,
            density_multiplier=1,
            iterations=5
        )
        self.assertAlmostEqual(rep_baseline.average_speedup, 1.0, places=1)

        # 2. Krystal Full Accelerated Suite
        rep_accel = self.engine.execute_benchmark(
            acceleration=AccelerationTier.KRYSTAL_FULL_ACCELERATED,
            density_multiplier=1,
            iterations=5
        )
        self.assertGreaterEqual(rep_accel.average_speedup, 4.0, "Full suite must yield >= 4.0x speedup")
        self.assertEqual(rep_accel.overall_grade, BenchmarkGrade.GRADE_S_SUPREME)
        self.assertEqual(rep_accel.vital_max_hp_rule, 6)

    def test_hardware_telemetry_integration(self):
        """Verify hardware telemetry is embedded into benchmark report."""
        rep = self.engine.execute_benchmark(density_multiplier=1, iterations=5)
        telem = rep.hardware_telemetry
        self.assertIn("gpu_temperature_c", telem)
        self.assertIn("electrical_power_watts", telem)
        self.assertIn("gpu_l2_cache_mb", telem)
        self.assertTrue(telem["fan_cooling_discount_active"])
        self.assertEqual(telem["vital_max_hp_rule"], 6)

    def test_report_serialization_and_history(self):
        """Verify JSON export and history ledger tracking."""
        initial_history_len = len(self.engine.benchmark_history)
        rep = self.engine.execute_benchmark(density_multiplier=1, iterations=5)
        self.assertEqual(len(self.engine.benchmark_history), initial_history_len + 1)

        d = rep.to_dict()
        self.assertIn("report_id", d)
        self.assertIn("archetype_results", d)
        self.assertEqual(len(d["archetype_results"]), 6)
        self.assertEqual(d["vital_max_hp_rule"], 6)

        status = self.engine.get_engine_status()
        self.assertEqual(status["status"], "ONLINE")
        self.assertIsNotNone(status["last_benchmark"])


if __name__ == "__main__":
    unittest.main()
