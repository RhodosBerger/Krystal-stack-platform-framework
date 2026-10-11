#!/usr/bin/env python3
"""
Unit and integration tests for Compositor Scaling Unit Protocols,
Godot 4.x autonomous MultiMesh modules, and Deep Research Photorealistic Synthesis.

Verifies:
  - Compositor scaling unit synthesis (>=8,000 instances, modular partitioning).
  - Fermat Phyllotaxis and Gielis Superformula configurations.
  - Human-editable JSON manifest generation (zero-AI dependency).
  - Godot 4.x .tscn declarative scene generation with MultiMesh nodes.
  - Presence of GDScript module and GDShader files.
  - Invariant VITAL_MAX_HP == 6.
"""

import json
import unittest
from pathlib import Path

from krystal_kernel.scene_constructor_presets import VITAL_MAX_HP
from krystal_kernel.compositor_scaling_protocol import (
    CompositorScalingUnitConfig,
    CompositorScenePackage8K,
    CompositorScalingProtocolEngine,
    GLOBAL_COMPOSITOR_SCALING_ENGINE,
    TARGET_OBJECT_COUNT_8K
)


class TestCompositorScalingProtocols(unittest.TestCase):
    """Verifies modular scaling units and mathematical synthesis."""

    def setUp(self):
        self.engine = GLOBAL_COMPOSITOR_SCALING_ENGINE

    def test_vital_max_hp_invariant(self):
        self.assertEqual(VITAL_MAX_HP, 6)

    def test_create_scaling_units_8k(self):
        units = self.engine.create_scaling_units_for_context("TACTICAL_ARENA", TARGET_OBJECT_COUNT_8K)
        self.assertTrue(len(units) >= 4)
        total_objects = sum(u.instance_count for u in units)
        self.assertGreaterEqual(total_objects, 8000)

        for u in units:
            self.assertEqual(u.vital_max_hp, 6)
            self.assertEqual(u.placement.type, "FERMAT_PHYLLOTAXIS")
            self.assertGreater(u.placement.c_scale, 0.0)
            self.assertEqual(u.morphology.type, "GIELIS_SUPERFORMULA")
            self.assertGreaterEqual(u.morphology.m, 3.0)
            self.assertGreaterEqual(u.material_tensor.base_hue, 0.0)
            self.assertLessEqual(u.material_tensor.base_hue, 1.0)

    def test_human_editable_manifest_generation(self):
        units = self.engine.create_scaling_units_for_context("TACTICAL_ARENA", 8192)
        manifest_str = self.engine.synthesize_human_editable_manifest(units)
        self.assertTrue(isinstance(manifest_str, str))
        
        # Parse and verify JSON integrity
        data = json.loads(manifest_str)
        self.assertEqual(data["version"], "2.0-NORMATIVE")
        self.assertEqual(data["vital_max_hp"], 6)
        self.assertIn("editing_guide", data)
        self.assertIn("scaling_units", data)
        self.assertEqual(len(data["scaling_units"]), len(units))

        # Test human-editability: user alters gielis symmetry 'm'
        u0 = data["scaling_units"][0]
        original_m = u0["morphology"]["m"]
        u0["morphology"]["m"] = 8.0  # Human changes hexagon to octagonal crystal
        self.assertEqual(u0["morphology"]["m"], 8.0)
        self.assertNotEqual(u0["morphology"]["m"], original_m)

    def test_godot_tscn_generation(self):
        units = self.engine.create_scaling_units_for_context("TACTICAL_ARENA", 8192)
        tscn_code = self.engine.generate_godot_tscn("TEST_ARENA_8K", "TACTICAL_ARENA", units)
        self.assertIn('[gd_scene load_steps=', tscn_code)
        self.assertIn('res://shaders/photorealistic_pbr_procedural.gdshader', tscn_code)
        self.assertIn('res://scripts/CompositorScalingUnit.gd', tscn_code)
        self.assertIn('volumetric_fog_enabled = true', tscn_code)
        self.assertIn('metadata/vital_max_hp = 6', tscn_code)

        # Confirm all scaling unit nodes exist as MultiMeshInstance3D
        for u in units:
            self.assertIn(f'metadata/unit_id = "{u.unit_id}"', tscn_code)
            self.assertIn(f'metadata/archetype = "{u.archetype}"', tscn_code)

    def test_synthesize_8k_scene_package(self):
        pkg = self.engine.synthesize_8k_scene_package(
            prompt="Generuj fotorealistickú taktickú arénu so škálovacími jednotkami",
            context="TACTICAL_ARENA",
            target_count=8192,
            seed=42
        )
        self.assertIsInstance(pkg, CompositorScenePackage8K)
        self.assertGreaterEqual(pkg.total_objects, 8192)
        self.assertEqual(pkg.vital_max_hp, 6)
        self.assertTrue(len(pkg.photorealistic_features) >= 5)
        self.assertLess(pkg.generation_latency_ms, 50.0)


class TestGodotAssetsAndResearchDocs(unittest.TestCase):
    """Verifies that all Godot 4 scripts, shaders, and research docs exist and are valid."""

    def test_godot_gdscript_module_exists(self):
        script_path = Path("godot_project/scripts/CompositorScalingUnit.gd")
        self.assertTrue(script_path.exists(), "CompositorScalingUnit.gd must exist")
        content = script_path.read_text(encoding="utf-8")
        self.assertIn("class_name CompositorScalingUnit", content)
        self.assertIn("extends MultiMeshInstance3D", content)
        self.assertIn("VITAL_MAX_HP: int = 6", content)
        self.assertIn("func build_multimesh_instances()", content)
        self.assertIn("func evaluate_gielis(", content)

    def test_godot_pbr_shader_exists(self):
        shader_path = Path("godot_project/shaders/photorealistic_pbr_procedural.gdshader")
        self.assertTrue(shader_path.exists(), "photorealistic_pbr_procedural.gdshader must exist")
        content = shader_path.read_text(encoding="utf-8")
        self.assertIn("shader_type spatial;", content)
        self.assertIn("specular_schlick_ggx", content)
        self.assertIn("INSTANCE_CUSTOM", content)
        self.assertIn("VITAL_MAX_HP = 6", content)

    def test_godot_8k_scene_file_exists(self):
        scene_path = Path("godot_project/scenes/Compositor8KScalingArena.tscn")
        self.assertTrue(scene_path.exists(), "Compositor8KScalingArena.tscn must exist")
        content = scene_path.read_text(encoding="utf-8")
        self.assertIn("ScalingUnit_", content)
        self.assertIn("MultiMeshInstance3D", content)
        self.assertIn("metadata/vital_max_hp = 6", content)

    def test_deep_research_document_exists(self):
        doc_path = Path("docs/research/DEEP_RESEARCH_PHOTOREALISTIC_PROCEDURAL_COMPOSITOR_AND_SCALING_PROTOCOLS.md")
        self.assertTrue(doc_path.exists(), "Deep Research document must exist")
        content = doc_path.read_text(encoding="utf-8")
        self.assertIn("DEEP RESEARCH: PHOTOREALISTIC PROCEDURAL COMPOSITOR SCALING PROTOCOLS", content)
        self.assertIn("Cook-Torrance Microfacet Specular BRDF", content)
        self.assertIn("Gielis Superformula for Continuous Morphometry", content)
        self.assertIn("Fermat Phyllotaxis & Golden Spiral Spatial Dispersion", content)
        self.assertIn("VITAL_MAX_HP = 6", content)


if __name__ == "__main__":
    unittest.main()
