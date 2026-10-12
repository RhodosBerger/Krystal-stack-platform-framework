#!/usr/bin/env python3
"""
==============================================================================
KRYSTAL-STACK // TEST SUITE: PROCEDURAL LAND BENCHMARK & CI/CD THROUGHPUT
==============================================================================
Tests:
  1. Universal Invariants: VITAL_MAX_HP == 6 across all structures.
  2. Continuous Terrain & Coupled Erosion: Talus slope & laplacian erosion dampening.
  3. SDF Raymarching Latency: Frame budget verification (< 15.0 ms).
  4. Deterministic Surface Hashing: Consistency across identical seeds.
  5. Canonical Seed Baseline Verification: Seed 42 & 1337 ground truth matching.
  6. Throughput Thresholds: Verification of >= 45.0 chunks/sec.
  7. Controlled Baseline Update: Validates update_baseline_with_result().
  8. Economic Engine Re-export Bridge: Verification of bridge exports.
==============================================================================
"""

import unittest
import os
import sys
import tempfile
import json

# Ensure repository root is on sys.path
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from krystal_kernel.procedural_land_benchmark_engine import (
    VITAL_MAX_HP,
    BenchmarkStatus,
    TerrainSurfaceSample,
    ProceduralLandBenchmarkResult,
    ProceduralLandBenchmarkEngine,
    GLOBAL_PROCEDURAL_LAND_BENCHMARK_ENGINE
)
from krystal_web_hub.economic_engine.procedural_land_benchmark_bridge import (
    GLOBAL_PROCEDURAL_LAND_BENCHMARK_ENGINE as BRIDGE_ENGINE,
    ProceduralLandBenchmarkEngine as BridgeEngineClass,
    ProceduralLandBenchmarkResult as BridgeResultClass
)


class TestProceduralLandBenchmark(unittest.TestCase):

    def setUp(self):
        self.engine = ProceduralLandBenchmarkEngine()

    def test_universal_invariants(self):
        """Verifies VITAL_MAX_HP == 6 invariant in all models and telemetry."""
        self.assertEqual(VITAL_MAX_HP, 6)
        res = self.engine.run_benchmark(chunk_count=10, grid_size=16, seed=42)
        self.assertEqual(res.vital_max_hp_rule, 6)
        self.assertEqual(res.surface_sample.vital_max_hp_rule, 6)

        status = self.engine.get_status()
        self.assertEqual(status["vital_max_hp_rule"], 6)

    def test_continuous_elevation_and_erosion(self):
        """Validates continuous mathematics and coupled erosion slope dampening."""
        elev_crystal = self.engine.compute_continuous_elevation(10.0, 10.0, 42, "Crystal_Severni_Stity")
        elev_druid = self.engine.compute_continuous_elevation(10.0, 10.0, 42, "Druid_Pradavny_Les")
        self.assertGreater(elev_crystal, 0.0)
        self.assertGreater(elev_druid, 0.0)

        # Create synthetic steep grid and verify erosion smooths extreme slope
        steep_grid = [[0.0 for _ in range(16)] for _ in range(16)]
        steep_grid[8][8] = 25.0  # Extreme peak
        eroded = self.engine.simulate_coupled_erosion(steep_grid, passes=3)
        self.assertLess(eroded[8][8], 25.0)
        self.assertGreater(eroded[8][7], 0.0)  # Talus slope accumulation

    def test_sdf_raymarching_latency_budget(self):
        """Verifies SDF sphere-tracing latency is within the 15 ms target budget."""
        grid = [[2.0 + 0.1 * i for j in range(32)] for i in range(32)]
        latency_ms = self.engine.simulate_sdf_raymarching(grid, steps=32)
        self.assertLess(latency_ms, 15.0)

    def test_deterministic_surface_hashing(self):
        """Validates that identical seeds yield bit-for-bit identical SHA-256 surface hashes."""
        res_a = self.engine.run_benchmark(chunk_count=10, grid_size=16, seed=77, biome="Crystal_Severni_Stity")
        res_b = self.engine.run_benchmark(chunk_count=10, grid_size=16, seed=77, biome="Crystal_Severni_Stity")
        self.assertEqual(res_a.surface_sample.surface_hash, res_b.surface_sample.surface_hash)
        self.assertEqual(res_a.surface_sample.mean_elevation, res_b.surface_sample.mean_elevation)

        # Different seed must yield different hash
        res_diff = self.engine.run_benchmark(chunk_count=10, grid_size=16, seed=88, biome="Crystal_Severni_Stity")
        self.assertNotEqual(res_a.surface_sample.surface_hash, res_diff.surface_sample.surface_hash)

    def test_canonical_seed_baseline_verification(self):
        """Validates canonical seeds (42 and 1337) against stored baseline expectations."""
        res_42 = self.engine.run_benchmark(chunk_count=20, grid_size=32, seed=42, biome="Crystal_Severni_Stity")
        self.assertTrue(res_42.passed_determinism, f"Seed 42 failed determinism: {res_42.surface_sample.surface_hash}")
        self.assertAlmostEqual(res_42.surface_sample.mean_elevation, 2.791, places=3)

        res_1337 = self.engine.run_benchmark(chunk_count=20, grid_size=32, seed=1337, biome="Druid_Pradavny_Les")
        self.assertTrue(res_1337.passed_determinism, f"Seed 1337 failed determinism: {res_1337.surface_sample.surface_hash}")
        self.assertAlmostEqual(res_1337.surface_sample.mean_elevation, 2.3377, places=3)

    def test_throughput_threshold_evaluation(self):
        """Validates chunk throughput meets or exceeds 45 chunks/sec threshold."""
        res = self.engine.run_benchmark(chunk_count=50, grid_size=32, seed=42)
        self.assertTrue(res.passed_throughput)
        self.assertGreaterEqual(res.chunk_throughput_chunks_per_sec, 45.0)
        self.assertLessEqual(res.raymarch_latency_ms, 15.0)

    def test_controlled_baseline_update(self):
        """Verifies updating baseline metrics in an isolated temp baseline file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            temp_path = f.name
            json.dump(self.engine.baseline_data, f)

        try:
            custom_engine = ProceduralLandBenchmarkEngine(baseline_path=temp_path)
            res = custom_engine.run_benchmark(chunk_count=25, grid_size=32, seed=42)
            update_res = custom_engine.update_baseline_with_result(res)
            self.assertTrue(update_res["success"])
            self.assertEqual(update_res["vital_max_hp_rule"], 6)

            with open(temp_path, "r", encoding="utf-8") as f:
                reloaded = json.load(f)
            self.assertEqual(
                reloaded["baseline_metrics"]["measured_chunk_throughput"],
                res.chunk_throughput_chunks_per_sec
            )
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def test_economic_engine_bridge_reexport(self):
        """Verifies that the economic engine bridge re-exports identical instances and classes."""
        self.assertIs(BRIDGE_ENGINE, GLOBAL_PROCEDURAL_LAND_BENCHMARK_ENGINE)
        self.assertIs(BridgeEngineClass, ProceduralLandBenchmarkEngine)
        self.assertIs(BridgeResultClass, ProceduralLandBenchmarkResult)


if __name__ == "__main__":
    unittest.main()
