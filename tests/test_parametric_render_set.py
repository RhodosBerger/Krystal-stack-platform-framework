"""
Tests for Parametric Render Set Engine and AI Studio Specifications.
Validates:
- Loading of 5 canonical render sets
- Mathematical topology: r(theta), radius derivative dr/dtheta, overhang metric xi(theta)
- Invariant VITAL_MAX_HP == 6 across all configs
- JSON serialization and manifest export
- AI Studio markdown prompt formatting
"""

import unittest
import math
import json
from krystal_kernel.parametric_render_set import (
    ParametricRenderCatalog,
    ParametricRenderSetId,
    ParametricRenderSetConfig,
    HarmonicCosineTopology,
    HarmonicHarmonicComponent,
    VITAL_MAX_HP,
    GOLDEN_RATIO
)


class TestParametricRenderSet(unittest.TestCase):

    def test_catalog_contains_all_five_sets(self):
        """Verifies that all 5 canonical render sets are present in the catalog."""
        sets = ParametricRenderCatalog.get_all_sets()
        self.assertEqual(len(sets), 5)
        for expected_id in ParametricRenderSetId:
            self.assertIn(expected_id, sets, f"Missing set {expected_id}")
            cfg = sets[expected_id]
            self.assertIsInstance(cfg, ParametricRenderSetConfig)
            self.assertEqual(cfg.constraints.vital_max_hp, 6)

    def test_harmonic_cosine_topology_math(self):
        """Validates mathematical correctness of r(theta) and dr/dtheta."""
        comp = [HarmonicHarmonicComponent(1, 20.0, 1.0, 0.0)]
        topo = HarmonicCosineTopology(base_radius_r0=100.0, fundamental_lobes_omega=4, components=comp)

        # At theta = 0: r = 100 + 20*cos(0) = 120.0
        self.assertAlmostEqual(topo.evaluate_radius(0.0), 120.0, places=4)

        # At theta = 0: dr/dtheta = -20 * (1*4) * sin(0) = 0.0 (stationary peak)
        self.assertAlmostEqual(topo.evaluate_radius_derivative(0.0), 0.0, places=4)

        # At theta = pi/8: freq*theta = 4 * (pi/8) = pi/2
        # r = 100 + 20*cos(pi/2) = 100.0
        # dr/dtheta = -20 * 4 * sin(pi/2) = -80.0 (maximum negative slope)
        self.assertAlmostEqual(topo.evaluate_radius(math.pi / 8.0), 100.0, places=4)
        self.assertAlmostEqual(topo.evaluate_radius_derivative(math.pi / 8.0), -80.0, places=4)

    def test_overhang_detection_metric(self):
        """Validates overhang classification metric xi(theta) = |dr/dtheta| / r(theta)."""
        # Smooth topography
        smooth_comp = [HarmonicHarmonicComponent(1, 5.0, 1.0, 0.0)]
        smooth_topo = HarmonicCosineTopology(base_radius_r0=100.0, fundamental_lobes_omega=2, components=smooth_comp)
        analysis_smooth = smooth_topo.analyze_profile(sample_count=36)
        self.assertFalse(analysis_smooth["has_overhangs"])
        self.assertLessEqual(analysis_smooth["xi_max"], 1.0)

        # Steep cliff / cavern topography with deliberate xi > 1.0
        steep_comp = [HarmonicHarmonicComponent(1, 50.0, 1.0, 0.0)]
        steep_topo = HarmonicCosineTopology(base_radius_r0=60.0, fundamental_lobes_omega=4, components=steep_comp)
        analysis_steep = steep_topo.analyze_profile(sample_count=36)
        self.assertTrue(analysis_steep["has_overhangs"])
        self.assertGreater(analysis_steep["xi_max"], 1.0)

    def test_set_04_cavern_has_overhangs(self):
        """Verifies SET_04_OVERHANG_BIOME_CAVERN_8K is correctly configured with overhangs."""
        cavern_set = ParametricRenderCatalog.get_set(ParametricRenderSetId.OVERHANG_BIOME_CAVERN_8K)
        self.assertIsNotNone(cavern_set)
        profile = cavern_set.harmonic_topology.analyze_profile(sample_count=36)
        self.assertTrue(profile["has_overhangs"])
        self.assertGreater(profile["xi_max"], 1.0)

    def test_hardware_envelopes(self):
        """Verifies Intel Iris Xe UMA and Z-fighting constraints across all sets."""
        for sid, s in ParametricRenderCatalog.get_all_sets().items():
            self.assertGreaterEqual(s.constraints.min_separation_epsilon_m, 0.012, f"{sid} Z-epsilon too small")
            self.assertLessEqual(s.constraints.intel_xe_uma_mb_ceiling, 256.0, f"{sid} UMA memory exceeds 256MB")
            self.assertEqual(s.constraints.target_fps, 60, f"{sid} must target 60 FPS")
            self.assertEqual(s.constraints.vital_max_hp, 6, f"{sid} invariant breach")

    def test_ai_studio_prompt_block_formatting(self):
        """Verifies markdown prompt text formatting for AI Studio."""
        s = ParametricRenderCatalog.get_set(ParametricRenderSetId.CRYSTAL_POLYHEDRA_D6_D12)
        prompt_block = s.format_ai_studio_prompt_block()
        self.assertIn("RENDER SET SPECIFICATION", prompt_block)
        self.assertIn("SET_01_CRYSTAL_POLYHEDRA_D6_D12", prompt_block)
        self.assertIn("PLATONIC_OCTAHEDRON_GYROID", prompt_block)
        self.assertIn("assert VITAL_MAX_HP == 6", prompt_block)

    def test_manifest_export_and_json_serializability(self):
        """Verifies manifest export can be converted to JSON and loaded back."""
        manifest = ParametricRenderCatalog.export_full_catalog_manifest()
        self.assertEqual(manifest["sets_count"], 5)
        self.assertEqual(manifest["vital_max_hp_invariant"], 6)

        # JSON encode and decode test
        encoded = json.dumps(manifest)
        decoded = json.loads(encoded)
        self.assertIn("render_sets", decoded)
        self.assertEqual(len(decoded["render_sets"]), 5)


if __name__ == "__main__":
    unittest.main()
