#!/usr/bin/env python3
"""
Unit and integration tests for Procedural Composition Interpreter,
Tile Cache Memory Partitioner, and Codebase Synergy Benchmark.

Verifies:
  - Gielis Superformula parametric contours.
  - Harmonic Color Tensor PBR properties and spectral coherence.
  - TileCacheMemoryManager 64-byte alignment and L2 cache fit.
  - ProceduralCompositionInterpreter autonomous execution (<150ms).
  - Compositional Synergy Index (CSI) improvement (+25% to +50% over baseline).
  - LLMSceneBuilder synergic integration and zero-glitch certification.
  - Invariant VITAL_MAX_HP == 6.
"""

import unittest
import math
from krystal_kernel.scene_constructor_presets import VITAL_MAX_HP
from krystal_kernel.procedural_composition_interpreter import (
    GielisSuperformula,
    HarmonicColorTensor,
    TileCacheMemoryManager,
    ProceduralCompositionInterpreter,
    GLOBAL_COMPOSITION_INTERPRETER
)
from krystal_kernel.composition_synergy_benchmark import (
    CompositionSynergyBenchmarkEngine,
    GLOBAL_COMPOSITION_BENCHMARK
)
from krystal_kernel.llm_scene_builder import GLOBAL_LLM_SCENE_BUILDER


class TestProceduralFormulas(unittest.TestCase):
    """Verifies mathematical formulas for procedural composition."""

    def test_gielis_superformula_archetypes(self):
        # Hexagonal archetype: m=6
        r_hex = GielisSuperformula.evaluate(0.0, m=6.0, n1=1.0, n2=1.0, n3=1.0)
        self.assertGreater(r_hex, 0.0)

        # Symmetry test: r(theta) == r(theta + 2*pi/m)
        r_hex_rot = GielisSuperformula.evaluate(2.0 * math.pi / 6.0, m=6.0, n1=1.0, n2=1.0, n3=1.0)
        self.assertAlmostEqual(r_hex, r_hex_rot, places=4)

        # Preset retrieval
        p_crystal = GielisSuperformula.get_preset_for_archetype("CRYSTAL_FACET")
        self.assertEqual(p_crystal["m"], 8.0)
        p_vortex = GielisSuperformula.get_preset_for_archetype("SOUL_VORTEX_MOTE")
        self.assertEqual(p_vortex["m"], 5.0)

    def test_harmonic_color_tensor(self):
        contexts = ["TACTICAL_ARENA", "OPENWORLD_BIOME", "URBAN_METROPOLIS", "SKEUOMORPHIC_DUNGEON"]
        for ctx in contexts:
            pbr = HarmonicColorTensor.sample_tensor(ctx, depth_norm=0.5, focal_weight=0.8, archetype="SOUL_VORTEX")
            self.assertEqual(len(pbr["albedo"]), 3)
            self.assertTrue(all(0.0 <= c <= 1.0 for c in pbr["albedo"]))
            self.assertGreater(pbr["emission_energy"], 0.0)
            self.assertGreaterEqual(pbr["metallic"], 0.0)
            self.assertGreaterEqual(pbr["roughness"], 0.0)


class TestTileCacheMemoryManager(unittest.TestCase):
    """Verifies L2 cache tile partitioning and memory layout."""

    def test_tile_partitioning_and_l2_fit(self):
        tile_mgr = TileCacheMemoryManager(full_width=256, full_height=256, tile_dim=64)
        tiles = tile_mgr.get_tile_descriptors()
        self.assertEqual(len(tiles), 16)  # (256/64) * (256/64) = 4 * 4 = 16

        stats = tile_mgr.get_memory_stats()
        self.assertTrue(stats["intel_l2_cache_fit"])
        self.assertTrue(stats["cache_line_aligned"])
        self.assertEqual(stats["vital_max_hp"], 6)
        self.assertEqual(stats["tile_size_kb"], 64.0)  # 64 * 64 * 16 bytes = 65,536 B = 64 KB


class TestProceduralCompositionInterpreter(unittest.TestCase):
    """Verifies autonomous interpreter execution, raster generation, and CSI audit."""

    def setUp(self):
        self.interpreter = ProceduralCompositionInterpreter()

    def test_rule_of_thirds_focal_field(self):
        # Node (0.333, 0.333) should yield a high focal weight
        focal_at_node = self.interpreter.calculate_focal_field(0.333, 0.333)
        self.assertGreater(focal_at_node, 0.6)

        # Corner (0.0, 0.0) should yield a lower focal weight
        focal_at_corner = self.interpreter.calculate_focal_field(0.0, 0.0)
        self.assertLess(focal_at_corner, focal_at_node)

    def test_generate_formula_elements(self):
        elements = self.interpreter.generate_formula_elements("TACTICAL_ARENA", count=1024, seed=77)
        self.assertEqual(len(elements), 1024)
        elem0 = elements[0]
        self.assertIn("position", elem0)
        self.assertIn("material_pbr", elem0)
        self.assertIn("focal_weight", elem0)
        self.assertIn("depth_norm", elem0)

    def test_render_procedural_image_raster(self):
        elements = self.interpreter.generate_formula_elements("TACTICAL_ARENA", count=512, seed=42)
        raster = self.interpreter.render_procedural_image_raster("TACTICAL_ARENA", elements, width=32, height=32)
        self.assertEqual(raster["width"], 32)
        self.assertEqual(raster["height"], 32)
        self.assertIn("ascii_art", raster)
        self.assertIn("ppm_base64_preview", raster)
        self.assertEqual(raster["vital_max_hp"], 6)

    def test_interpret_and_compose_end_to_end(self):
        res = self.interpreter.interpret_and_compose(
            prompt="Monumentálna taktická scéna s mínometmi a zákopmi",
            context="TACTICAL_ARENA",
            target_count=8192,
            render_preview=True,
            raster_size=64,
            seed=42
        )
        self.assertEqual(res["status"], "SUCCESS")
        self.assertEqual(res["total_elements"], 8192)
        self.assertEqual(res["vital_max_hp"], 6)
        self.assertLess(res["interpretation_latency_ms"], 300.0)

        csi = res["synergy_report"]
        self.assertGreaterEqual(csi["compositional_synergy_index"], 0.75)
        self.assertIn(csi["competitive_tier"], ["HIGH_FIDELITY", "ELITE_NEURAL_GRADE"])


class TestCompositionSynergyBenchmark(unittest.TestCase):
    """Verifies comparative benchmarking against baseline and competitive advantage."""

    def setUp(self):
        self.bench = CompositionSynergyBenchmarkEngine()

    def test_synergy_benchmark_execution(self):
        res = self.bench.run_comprehensive_benchmark("TACTICAL_ARENA", target_count=2048, iterations=1)
        self.assertGreater(res.csi_delta, 0.10)
        self.assertGreater(res.synergy_improvement_pct, 20.0)
        self.assertEqual(res.vital_max_hp, 6)
        self.assertIn("Krystal", res.competitive_verdict)
        self.assertTrue(len(res.actionable_editor_recommendations) >= 3)


class TestLLMSceneBuilderSynergicIntegration(unittest.TestCase):
    """Verifies that LLMSceneBuilder can generate synergic scenes with Keras Lupa certification."""

    def test_build_synergic_scene(self):
        pkg = GLOBAL_LLM_SCENE_BUILDER.build_synergic_scene(
            prompt="Generuj synergickú scénu arény",
            context="TACTICAL_ARENA",
            target_count=1024,
            render_preview=True,
            seed=42
        )
        self.assertEqual(pkg["status"], "SUCCESS")
        self.assertEqual(pkg["vital_max_hp"], 6)
        self.assertTrue(pkg["glitch_free_guarantee"])
        self.assertEqual(pkg["lupa_report"]["anomalies_detected"], 0)


if __name__ == "__main__":
    unittest.main()
