"""
Unit and Integration Tests for Krystal-Stack Open-World Engine
=============================================================
Tests:
1. Natural language compilation (Slovak / English)
2. Mathematical parameter derivation & biome classification
3. Terrain manifold height & coupled erosion sampling
4. Code synthesis for Python, Godot 4.x Shaders, and Janet DSL
5. ASCII camera raymarching and artifact placement
"""

import unittest
import os
import sys

# Ensure repository root is on path
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from openworld_engine.terrain_manifold import (
    TerrainManifold, BiomePhaseSpace, fbm_2d, ridged_multifractal_2d
)
from openworld_engine.world_semantic_compiler import WorldSemanticCompiler
from openworld_engine.openworld_renderer import OpenWorldRenderer

class TestOpenWorldEngine(unittest.TestCase):

    def setUp(self):
        self.compiler = WorldSemanticCompiler()

    def test_01_slovak_prompt_compilation(self):
        prompt = "vytvor nekonečný vulkanický svet s lávovými kaňonmi, hustým sírnym dymom a 6 rekurzií"
        spec = self.compiler.compile_natural_prompt(prompt)

        self.assertIn("dominant_biome", spec)
        self.assertEqual(spec["topography_type"], "CANYON_TRENCHES")
        self.assertEqual(spec["atmosphere_type"], "SULFUR_SMOG")
        self.assertEqual(spec["mathematical_parameters"]["octaves"], 6)
        self.assertGreater(spec["mathematical_parameters"]["height_scale"], 3.0)
        self.assertGreater(spec["mathematical_parameters"]["ridge_weight"], 0.7)

    def test_02_english_prompt_compilation(self):
        prompt = "create rolling dunes desert with ancient monoliths, clear crystal aether and smooth gentle hills"
        spec = self.compiler.compile_natural_prompt(prompt)

        self.assertEqual(spec["topography_type"], "ROLLING_DUNES_PLAINS")
        self.assertEqual(spec["atmosphere_type"], "CRYSTALLINE_AETHER")
        self.assertLess(spec["mathematical_parameters"]["height_scale"], 2.0)
        # Verify artifact scatter rule detected monoliths
        recipes = [r["recipe_id"] for r in spec["artifact_scatter_rules"]]
        self.assertIn("ANCIENT_OBELISK_MONOLITH", recipes)

    def test_03_terrain_manifold_evaluation(self):
        manifold = TerrainManifold(
            base_elevation=0.0,
            height_scale=3.0,
            frequency=0.05,
            octaves=4,
            ridge_weight=0.5,
            erosion_strength=0.4,
            seed=42
        )

        h_center, slope, sediment = manifold.sample_height_and_erosion(0.0, 0.0)
        self.assertIsInstance(h_center, float)
        self.assertIsInstance(slope, float)
        self.assertIsInstance(sediment, float)

        # Check normal vector is normalized
        nx, ny, nz = manifold.sample_normal(0.0, 0.0)
        mag = (nx**2 + ny**2 + nz**2)**0.5
        self.assertAlmostEqual(mag, 1.0, places=3)
        self.assertGreater(ny, 0.0)  # Upward pointing

        # Check SDF distance calculation
        sdf_val = manifold.evaluate_world_point_sdf(0.0, h_center + 5.0, 0.0)
        self.assertGreater(sdf_val, 0.0)  # Above surface -> positive

    def test_04_code_synthesizers(self):
        prompt = "vytvor kybernetickú pustatinu s neonovými vežami a silnou eróziou"
        spec = self.compiler.compile_natural_prompt(prompt)

        # 1. Python Code Synthesis
        py_code = self.compiler.to_procedural_python(spec)
        self.assertIn("class ProceduralWorld:", py_code)
        self.assertIn("def evaluate_height", py_code)

        # 2. Godot Shader Synthesis
        godot_code = self.compiler.to_godot_shader(spec)
        self.assertIn("shader_type spatial;", godot_code)
        self.assertIn("float fbm_terrain", godot_code)

        # 3. Janet DSL Synthesis
        janet_code = self.compiler.to_janet_dsl(spec)
        self.assertIn("(def world-spec", janet_code)
        self.assertIn("(defn sample-terrain-height [x z]", janet_code)

    def test_05_ascii_rendering_pipeline(self):
        prompt = "vulkanické hory s obrannými vežičkami a popolom"
        spec = self.compiler.compile_natural_prompt(prompt)
        renderer = OpenWorldRenderer(spec["manifold"], spec)

        # Check placed actors
        self.assertIsInstance(renderer.placed_actors, list)

        # Render ASCII frame
        frame = renderer.render_ascii_frame(width=48, height=12, cam_pos=(0.0, 5.0, -8.0))
        lines = frame.splitlines()
        self.assertEqual(len(lines), 12)
        self.assertEqual(len(lines[0]), 48)
        self.assertTrue(any(c in frame for c in ":-+*#%@·"))


if __name__ == "__main__":
    unittest.main()
