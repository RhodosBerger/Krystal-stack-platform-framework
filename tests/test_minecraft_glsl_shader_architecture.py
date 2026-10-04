"""
Unit tests for Minecraft GLSL Shader Architecture and Godot 4.x Graphics Engine.
Verifies LabPBR 1.3 decoding, deferred pipeline simulation, canonical packs,
and the 6 Max HP vital invariant.
"""

import os
import unittest
from krystal_web_hub.economic_engine.minecraft_glsl_shader_architecture import (
    GLOBAL_MINECRAFT_SHADER_ENGINE,
    CANONICAL_SHADER_PACKS,
    CANONICAL_DEFERRED_STAGES,
    CANONICAL_GODOT_PLUGINS,
    VITAL_MAX_HP,
    GOLDEN_RATIO,
)


class TestMinecraftGLSLShaderArchitecture(unittest.TestCase):
    def setUp(self):
        self.engine = GLOBAL_MINECRAFT_SHADER_ENGINE

    def test_canonical_shader_packs_integrity(self):
        self.assertGreaterEqual(len(CANONICAL_SHADER_PACKS), 5)
        pack_ids = [p.id for p in CANONICAL_SHADER_PACKS]
        self.assertIn("bsl_v8", pack_ids)
        self.assertIn("complementary_reimagined", pack_ids)
        self.assertIn("seus_ptgi_hrr", pack_ids)
        self.assertIn("continuum_cinematic", pack_ids)
        self.assertIn("iris_vanilla_plus", pack_ids)

        for pack in CANONICAL_SHADER_PACKS:
            self.assertEqual(pack.vital_max_hp, VITAL_MAX_HP)
            self.assertLessEqual(pack.hp_cost, VITAL_MAX_HP)
            self.assertGreater(len(pack.features), 3)
            self.assertGreater(pack.target_fps_iris_xe, 20)

    def test_canonical_deferred_stages(self):
        self.assertGreaterEqual(len(CANONICAL_DEFERRED_STAGES), 6)
        stage_ids = [s.id for s in CANONICAL_DEFERRED_STAGES]
        self.assertIn("gbuffers_terrain", stage_ids)
        self.assertIn("gbuffers_water", stage_ids)
        self.assertIn("shadow_cascades", stage_ids)
        self.assertIn("voxel_gi_dda", stage_ids)
        self.assertIn("volumetric_fog_godrays", stage_ids)
        self.assertIn("post_process_composite", stage_ids)

        for stage in CANONICAL_DEFERRED_STAGES:
            self.assertEqual(stage.vital_max_hp, VITAL_MAX_HP)
            self.assertGreater(stage.vram_cost_mb, 0.0)
            self.assertGreater(stage.gpu_time_ms_iris_xe, 0.0)

    def test_canonical_godot_plugins(self):
        self.assertGreaterEqual(len(CANONICAL_GODOT_PLUGINS), 4)
        plugin_ids = [p.id for p in CANONICAL_GODOT_PLUGINS]
        self.assertIn("terrain_3d", plugin_ids)
        self.assertIn("phantom_camera", plugin_ids)
        self.assertIn("zylann_voxel", plugin_ids)
        self.assertIn("limbo_ai", plugin_ids)

    def test_labpbr_decoding_dielectric_smooth(self):
        # 80% smooth, low F0 (dielectric stone), no emission
        decoded = self.engine.decode_labpbr_pixel(
            specular_r=0.8,
            specular_g=0.04,
            specular_b=0.2,
            specular_a=1.0, # 255 = no emission
            normal_r=0.5,
            normal_g=0.5,
            normal_b=1.0,
            normal_a=0.3,
        )
        self.assertAlmostEqual(decoded["smoothness"], 0.8, places=2)
        # Roughness = (1.0 - 0.8)^2 = 0.04
        self.assertAlmostEqual(decoded["linear_roughness"], 0.04, places=2)
        self.assertFalse(decoded["is_metal"])
        self.assertFalse(decoded["is_emissive"])
        self.assertEqual(decoded["vital_max_hp"], VITAL_MAX_HP)

    def test_labpbr_decoding_metal_emissive(self):
        # High F0 (metal > 229/255 = 0.898), glowing redstone/crystal
        decoded = self.engine.decode_labpbr_pixel(
            specular_r=0.9,
            specular_g=0.95, # > 229/255 -> Metal!
            specular_b=0.0,
            specular_a=0.5, # glowing emission
            normal_r=0.5,
            normal_g=0.5,
            normal_b=0.8,
            normal_a=0.5,
        )
        self.assertTrue(decoded["is_metal"])
        self.assertEqual(decoded["f0_reflectance"], 1.0)
        self.assertTrue(decoded["is_emissive"])
        self.assertGreater(decoded["emissive_intensity"], 2.0)

    def test_pipeline_simulation_metrics(self):
        sim = self.engine.simulate_pipeline_run(
            profile_id="complementary_reimagined",
            resolution_width=1920,
            resolution_height=1080,
        )
        self.assertIn("profile", sim)
        self.assertIn("projected_fps", sim)
        self.assertGreater(sim["projected_fps"], 30)
        self.assertGreater(sim["total_vram_mb"], 100.0)
        self.assertEqual(sim["vital_max_hp_rule"], VITAL_MAX_HP)

    def test_godot_shader_files_on_disk(self):
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        terrain_shader = os.path.join(base_dir, "godot_project", "shaders", "minecraft_pbr_terrain.gdshader")
        water_shader = os.path.join(base_dir, "godot_project", "shaders", "minecraft_water_ocean.gdshader")
        sky_shader = os.path.join(base_dir, "godot_project", "shaders", "minecraft_volumetric_fog_and_sky.gdshader")
        controller_script = os.path.join(base_dir, "godot_project", "scripts", "MinecraftAtmosphereController.gd")

        self.assertTrue(os.path.isfile(terrain_shader), f"Missing {terrain_shader}")
        self.assertTrue(os.path.isfile(water_shader), f"Missing {water_shader}")
        self.assertTrue(os.path.isfile(sky_shader), f"Missing {sky_shader}")
        self.assertTrue(os.path.isfile(controller_script), f"Missing {controller_script}")

        with open(terrain_shader, "r", encoding="utf-8") as f:
            content = f.read()
            self.assertIn("LabPBR", content)
            self.assertIn("calculate_pom_uv", content)
            self.assertIn("VITAL_MAX_HP", content)

        with open(water_shader, "r", encoding="utf-8") as f:
            content = f.read()
            self.assertIn("evaluate_gerstner_wave", content)
            self.assertIn("absorption_coefficients", content)


if __name__ == "__main__":
    unittest.main()
