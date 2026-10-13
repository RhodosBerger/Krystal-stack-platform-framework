#!/usr/bin/env python3
"""
==============================================================================
KRYSTAL-STACK: UNIT TEST SUITE
CYCLES & V-RAY HYBRID REVERSE-ENGINEERED RENDER ENGINE
==============================================================================
Tests:
  1. Universal Platform Invariants: VITAL_MAX_HP == 6, GOLDEN_RATIO.
  2. Sampling & Ray Equivalence: Samples == Subdivs^2 bi-directional mapping.
  3. PBR Material Translation: Roughness <-> Glossiness reciprocity (G = 1 - R).
  4. Canonical Elemental Materials: Crystal Ice, Toxic Slime, Magma, Void Glass.
  5. Multi-Disciplinary Measurement Telemetry (Thermal Ohmic Joules, L2 Cache).
  6. Ray Tracing Budget & Memory Working Set Estimator.
  7. Standalone Blender Cycles and V-Ray Automation Script Generation.
==============================================================================
"""

import unittest
from typing import Dict, Any

from krystal_kernel.cycles_vray_hybrid_render_engine import (
    VITAL_MAX_HP,
    GOLDEN_RATIO,
    RenderTargetEngine,
    GiMethod,
    DenoiseEngine,
    CyclesRenderParameters,
    VRayRenderParameters,
    ReverseEngineeredPbrMaterial,
    HardwareMeasurementTelemetry,
    CyclesVRayHybridRenderEngine,
    GLOBAL_CYCLES_VRAY_ENGINE
)


class TestCyclesVRayHybridRenderEngine(unittest.TestCase):

    def setUp(self):
        self.engine = CyclesVRayHybridRenderEngine()

    def test_universal_invariants(self):
        """Verify strict conformance to platform invariant VITAL_MAX_HP == 6."""
        self.assertEqual(VITAL_MAX_HP, 6, "VITAL_MAX_HP must strictly be 6 across render engines")
        self.assertAlmostEqual(GOLDEN_RATIO, 1.61803398875, places=7)

        # Check all material presets enforce VITAL_MAX_HP == 6
        for m in self.engine.materials.values():
            self.assertEqual(m.vital_max_hp, 6)

    def test_sampling_equivalence_bidirectional(self):
        """Verify reverse-engineered mathematical equivalence: Samples == Subdivs^2."""
        # 1. Cycles to V-Ray
        cycles_p = CyclesRenderParameters(samples=400, min_samples=16, adaptive_threshold=0.015, tile_size=64)
        vray_p = CyclesVRayHybridRenderEngine.cycles_to_vray_sampling(cycles_p)
        self.assertEqual(vray_p.max_subdivs, 20, "sqrt(400) == 20 subdivs")
        self.assertEqual(vray_p.min_subdivs, 4, "sqrt(16) == 4 subdivs")
        self.assertEqual(vray_p.color_threshold, 0.015)
        self.assertEqual(vray_p.bucket_width, 32)

        # 2. V-Ray to Cycles
        vray_in = VRayRenderParameters(max_subdivs=24, min_subdivs=3, color_threshold=0.008, bucket_width=32)
        cycles_out = CyclesVRayHybridRenderEngine.vray_to_cycles_sampling(vray_in)
        self.assertEqual(cycles_out.samples, 576, "24^2 == 576 samples")
        self.assertEqual(cycles_out.min_samples, 9, "3^2 == 9 samples")
        self.assertEqual(cycles_out.adaptive_threshold, 0.008)
        self.assertEqual(cycles_out.tile_size, 64)

    def test_roughness_glossiness_reciprocity(self):
        """Verify exact reciprocity: Glossiness = 1.0 - Roughness."""
        r = 0.35
        g = CyclesVRayHybridRenderEngine.convert_roughness_to_glossiness(r)
        self.assertAlmostEqual(g, 0.65, places=4)

        r_back = CyclesVRayHybridRenderEngine.convert_glossiness_to_roughness(g)
        self.assertAlmostEqual(r_back, 0.35, places=4)

        # Boundary clamping tests
        self.assertEqual(CyclesVRayHybridRenderEngine.convert_roughness_to_glossiness(1.5), 0.0)
        self.assertEqual(CyclesVRayHybridRenderEngine.convert_roughness_to_glossiness(-0.5), 1.0)
        self.assertEqual(CyclesVRayHybridRenderEngine.convert_glossiness_to_roughness(1.2), 0.0)
        self.assertEqual(CyclesVRayHybridRenderEngine.convert_glossiness_to_roughness(-0.3), 1.0)

    def test_canonical_elemental_materials(self):
        """Verify presence and validity of canonical elemental PBR materials."""
        mats = self.engine.materials
        self.assertIn("crystal_ice_prism", mats)
        self.assertIn("toxic_caustic_slime", mats)
        self.assertIn("basalt_magma_forge", mats)
        self.assertIn("cyber_gold_aether", mats)
        self.assertIn("void_singularity_glass", mats)

        # Check crystal ice IOR
        ice = mats["crystal_ice_prism"]
        self.assertAlmostEqual(ice.ior, 1.31, places=2)
        self.assertAlmostEqual(ice.roughness_cycles + ice.glossiness_vray, 1.0, places=3)

        # Check toxic slime SSS
        slime = mats["toxic_caustic_slime"]
        self.assertGreater(slime.subsurface_weight, 0.5)
        self.assertAlmostEqual(slime.roughness_cycles + slime.glossiness_vray, 1.0, places=3)

    def test_hardware_measurement_ingestion(self):
        """Verify multi-disciplinary measurement telemetry affects render dispatch."""
        # Cool state: 50 C, fan discount active, tile size 64
        cool_telem = self.engine.ingest_hardware_measurements(gpu_temp=50.0, power_watts=28.0)
        self.assertTrue(cool_telem.thermal_gas_discount_active)
        self.assertEqual(cool_telem.recommended_tile_size, 64)
        self.assertEqual(cool_telem.vital_max_hp, 6)

        # Hot throttled state: 80 C, tile size throttled to 32
        hot_telem = self.engine.ingest_hardware_measurements(gpu_temp=80.0, power_watts=35.0)
        self.assertFalse(hot_telem.thermal_gas_discount_active)
        self.assertEqual(hot_telem.recommended_tile_size, 32)
        self.assertLess(hot_telem.estimated_samples_per_sec, cool_telem.estimated_samples_per_sec * 1.5)

    def test_render_budget_estimation(self):
        """Verify ray count and memory footprint calculation."""
        p = CyclesRenderParameters(samples=256, min_samples=32, diffuse_bounces=4, glossy_bounces=4, tile_size=64)
        budget = self.engine.estimate_render_budget(width=1280, height=720, cycles_params=p)

        self.assertEqual(budget["resolution"], "1280x720")
        self.assertEqual(budget["total_pixels"], 1280 * 720)
        self.assertGreater(budget["total_primary_rays"], 1000000)
        self.assertGreater(budget["total_bounced_rays"], budget["total_primary_rays"])
        self.assertGreater(budget["estimated_render_time_sec"], 0.0)
        self.assertTrue(budget["fits_in_gpu_l2_cache"])
        self.assertEqual(budget["vital_max_hp_rule"], 6)

    def test_standalone_script_generation(self):
        """Verify generated Python and V-Ray scripts."""
        cycles_code = self.engine.generate_blender_cycles_script()
        self.assertIn("import bpy", cycles_code)
        self.assertIn("cycles.samples = 256", cycles_code)
        self.assertIn("OPENIMAGEDENOISE", cycles_code)
        self.assertIn("VITAL_MAX_HP == 6", cycles_code)

        vray_code = self.engine.generate_vray_automation_script()
        self.assertIn("SettingsDMC", vray_code)
        self.assertIn("SettingsImageSampler", vray_code)
        self.assertIn("VRayDenoiser", vray_code)
        self.assertIn("VITAL_MAX_HP == 6", vray_code)


if __name__ == "__main__":
    unittest.main()
