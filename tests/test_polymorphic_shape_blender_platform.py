#!/usr/bin/env python3
"""
==============================================================================
KRYSTAL-STACK: UNIT TEST SUITE
POLYMORPHIC SHAPE BLENDER PLATFORM, MULTILINGUAL LOOKUP & CARTESIAN ENGINE
==============================================================================
Tests:
  1. Universal Invariants: VITAL_MAX_HP == 6, GOLDEN_RATIO == 1.61803398875.
  2. Multilingual Lookup Registry across SK, EN, CZ, DE.
  3. Coordinate System Conventions & 4x4 Homogeneous Transformation Matrices.
  4. Descriptive Mesh & Cartesian Coordinate Descriptor (Centroid, Bounding Box, 8 Octants, Spherical).
  5. Canonical Shape Template Factory for all 6 archetypes.
  6. Shape Economic Observer & HP Clamping [1, 6].
  7. Standalone Blender 4.x Script (`bpy`) with Metric System & Coordinate Gizmos.
  8. Wavefront .OBJ 3D Mesh Topology Generation.
  9. Command Shape Platform Execution & History Tracking.
  10. Package Re-exports in krystal_web_hub.economic_engine.
==============================================================================
"""

import math
import unittest
from typing import Dict, Any

from krystal_web_hub.economic_engine.polymorphic_shape_blender_platform import (
    VITAL_MAX_HP,
    GOLDEN_RATIO,
    LanguageMode,
    CoordinateSystemType,
    ShapeArchetype,
    ModifierType,
    ShapeCommandType,
    ShapeModifierSpec,
    ShapeEconomicPlan,
    PolymorphicShapeInstance,
    MultilingualLookupRegistry,
    CoordinateSystemConverter,
    MeshGeometryDescriptor,
    CartesianCoordinateDescriptor,
    ShapeTemplateFactory,
    ShapeEconomicObserver,
    BlenderScriptGenerator,
    CommandShapePlatform,
    GLOBAL_SHAPE_PLATFORM
)


class TestPolymorphicShapeBlenderPlatform(unittest.TestCase):

    def setUp(self):
        self.platform = CommandShapePlatform(default_lang=LanguageMode.SK)

    def test_universal_invariants(self):
        """Verify strict adherence to core platform invariants."""
        self.assertEqual(VITAL_MAX_HP, 6, "VITAL_MAX_HP must strictly be 6 across all engines")
        self.assertAlmostEqual(GOLDEN_RATIO, 1.61803398875, places=7)

        # Check starter shapes all obey 6 HP rule
        for s in self.platform._library.values():
            self.assertEqual(s.vital_max_hp_rule, 6)
            self.assertGreaterEqual(s.economic_plan.structural_hp, 1)
            self.assertLessEqual(s.economic_plan.structural_hp, 6)

    def test_multilingual_lookup_registry(self):
        """Verify multilingual translations across SK, EN, CZ, DE."""
        for lang in LanguageMode:
            # Platform title
            title = MultilingualLookupRegistry.get_term("platform_title", lang)
            self.assertTrue(len(title) > 0, f"Title missing for language {lang}")

            # Archetype terms
            for arch in ShapeArchetype:
                info = MultilingualLookupRegistry.get_archetype_info(arch, lang)
                self.assertTrue(len(info["name"]) > 0, f"Archetype {arch} name missing in {lang}")
                self.assertTrue(len(info["description"]) > 0, f"Archetype {arch} desc missing in {lang}")

            # Economic & Coordinate terms
            for key in ["crystal_cost", "mana_cost", "gold_cost", "build_time", "centroid", "bounding_box", "axis_x", "axis_z"]:
                term = MultilingualLookupRegistry.get_term(key, lang)
                self.assertTrue(len(term) > 0, f"Term {key} missing in {lang}")

            # 8 Octants terms
            for i in range(1, 9):
                oct_term = MultilingualLookupRegistry.get_term(f"octant_{i}", lang)
                self.assertTrue(len(oct_term) > 0, f"Octant {i} term missing in {lang}")

    def test_coordinate_system_conversions(self):
        """Verify 4x4 matrices and coordinate transformations from Blender Cartesian."""
        pt = (2.0, 3.0, 5.0)

        # Blender Z-Up (Identity)
        v_blender = CoordinateSystemConverter.transform_vertex(pt, CoordinateSystemType.BLENDER_Z_UP)
        self.assertEqual(v_blender, (2.0, 3.0, 5.0))

        # WebGL / OpenGL Y-Up: (x, z, -y)
        v_webgl = CoordinateSystemConverter.transform_vertex(pt, CoordinateSystemType.OPENGL_WEBGL_Y_UP)
        self.assertEqual(v_webgl, (2.0, 5.0, -3.0))

        # Unreal Left-Handed: (y, x, z)
        v_unreal = CoordinateSystemConverter.transform_vertex(pt, CoordinateSystemType.UNREAL_Z_UP_LH)
        self.assertEqual(v_unreal, (3.0, 2.0, 5.0))

        # Matrices
        m_blender = CoordinateSystemConverter.get_matrix_for_system(CoordinateSystemType.BLENDER_Z_UP)
        self.assertEqual(m_blender[0][0], 1.0)
        self.assertEqual(m_blender[1][1], 1.0)
        self.assertEqual(m_blender[2][2], 1.0)

        m_webgl = CoordinateSystemConverter.get_matrix_for_system(CoordinateSystemType.OPENGL_WEBGL_Y_UP)
        self.assertEqual(m_webgl[1][2], 1.0)
        self.assertEqual(m_webgl[2][1], -1.0)

    def test_cartesian_coordinate_descriptor_generation(self):
        """Verify mesh geometry descriptor creates valid centroid, bounds, octants, and spherical mappings."""
        desc = MeshGeometryDescriptor.build_coordinate_descriptor(
            ShapeArchetype.TACTICAL_OBELISK,
            {"height": 5.0, "base_radius": 1.5, "taper_ratio": 0.78},
            CoordinateSystemType.BLENDER_Z_UP,
            LanguageMode.SK
        )

        self.assertIsInstance(desc, CartesianCoordinateDescriptor)
        self.assertEqual(desc.origin, (0.0, 0.0, 0.0))
        self.assertEqual(desc.coordinate_system, CoordinateSystemType.BLENDER_Z_UP)
        self.assertEqual(desc.handedness, "RIGHT_HANDED")

        # Bounding box & dimensions
        self.assertAlmostEqual(desc.bounding_box["min_x"], -1.5, places=1)
        self.assertAlmostEqual(desc.bounding_box["max_x"], 1.5, places=1)
        self.assertGreaterEqual(desc.dimensions["width_x"], 2.8)
        self.assertGreaterEqual(desc.dimensions["height_z"], 5.0)

        # Centroid
        cx, cy, cz = desc.centroid
        self.assertAlmostEqual(cx, 0.0, places=1)
        self.assertAlmostEqual(cy, 0.0, places=1)
        self.assertGreater(cz, 1.0)

        # 8-Octant distribution sum
        total_pct = sum(oct_info["percentage"] for oct_info in desc.octant_distribution.values())
        self.assertAlmostEqual(total_pct, 100.0, places=0)

        # Spherical & Cylindrical
        self.assertGreater(desc.spherical_centroid["radius_r"], 0.0)
        self.assertGreaterEqual(desc.spherical_centroid["theta_deg"], 0.0)
        self.assertLessEqual(desc.spherical_centroid["theta_deg"], 180.0)

    def test_canonical_template_factory(self):
        """Verify template factory provides complete configurations for all 6 archetypes."""
        templates = ShapeTemplateFactory.get_canonical_templates(LanguageMode.EN)
        self.assertEqual(len(templates), 6)

        for arch in ShapeArchetype:
            self.assertIn(arch, templates)
            tpl = templates[arch]
            self.assertIn("name", tpl)
            self.assertIn("description", tpl)
            self.assertIn("default_params", tpl)
            self.assertIn("modifiers", tpl)
            self.assertIsInstance(tpl["modifiers"], list)
            for m in tpl["modifiers"]:
                self.assertIsInstance(m, ShapeModifierSpec)

    def test_economic_observer_and_hp_bounding(self):
        """Verify structural HP is strictly clamped between 1 and 6 HP."""
        # Ultra tiny shape -> HP should be clamped to at least 1
        plan_tiny = ShapeEconomicObserver.calculate_plan(
            "test_tiny", ShapeArchetype.TACTICAL_OBELISK, {"height": 0.1, "base_radius": 0.05}, []
        )
        self.assertEqual(plan_tiny.structural_hp, 1)
        self.assertEqual(plan_tiny.vital_max_hp_rule, 6)

        # Huge mega citadel -> HP must be clamped to VITAL_MAX_HP == 6
        plan_huge = ShapeEconomicObserver.calculate_plan(
            "test_huge", ShapeArchetype.CITADEL_SPIRE, {"height": 100.0, "base_radius": 50.0}, []
        )
        self.assertEqual(plan_huge.structural_hp, 6)
        self.assertEqual(plan_huge.vital_max_hp_rule, 6)

        # Economic values must be positive
        self.assertGreater(plan_huge.crystal_cost, 0)
        self.assertGreater(plan_huge.mana_cost, 0)
        self.assertGreater(plan_huge.gold_cost, 0)
        self.assertGreater(plan_huge.build_time_seconds, 0)
        self.assertGreater(plan_huge.complexity_score, 0)

    def test_blender_script_generation(self):
        """Verify Blender 4.x Python script output with metric Cartesian setup and coordinate gizmo."""
        shape = self.platform.create_shape(
            archetype=ShapeArchetype.TACTICAL_OBELISK,
            parameters={"height": 4.5, "base_radius": 1.2}
        )

        # Generate in German (DE)
        script_de = BlenderScriptGenerator.generate_blender_script(shape, lang=LanguageMode.DE)
        self.assertIn("import bpy", script_de)
        self.assertIn("bpy.context.scene.unit_settings.system = \"METRIC\"", script_de)
        self.assertIn("Krystal_Cartesian_Gizmo", script_de)
        self.assertIn(f"krystal_structural_hp\"] = {shape.economic_plan.structural_hp}", script_de)
        self.assertIn("krystal_vital_max_hp\"] = 6", script_de)
        self.assertIn("Koordinatensystem", script_de)

        # Generate in Slovak (SK)
        script_sk = BlenderScriptGenerator.generate_blender_script(shape, lang=LanguageMode.SK)
        self.assertIn("Nastavenie Súradnicového Systému", script_sk)
        self.assertIn("Krystal_Material", script_sk)

    def test_obj_mesh_generation(self):
        """Verify Wavefront OBJ string generation."""
        shape = self.platform.create_shape(
            archetype=ShapeArchetype.CITADEL_SPIRE,
            parameters={"height": 6.0, "base_radius": 2.0}
        )
        obj_text = BlenderScriptGenerator.generate_obj_mesh(shape)
        self.assertIn("# Krystal-Stack Wavefront OBJ", obj_text)
        self.assertIn("v ", obj_text)
        self.assertIn("vn ", obj_text)
        self.assertIn("f ", obj_text)
        self.assertIn("Max HP: 6", obj_text)

    def test_command_shape_platform_lifecycle(self):
        """Verify Command Shape Platform commands (create, morph, modifier, lang, coord analysis)."""
        plat = CommandShapePlatform()

        # 1. Create Shape
        shape = plat.create_shape(
            archetype=ShapeArchetype.ETHEREAL_CRYSTAL,
            name="Test Crystal",
            parameters={"scale": 2.0}
        )
        self.assertIsNotNone(shape)
        self.assertEqual(shape.archetype, ShapeArchetype.ETHEREAL_CRYSTAL)
        self.assertIn(shape.shape_id, plat._library)

        # 2. Morph Parameters
        morphed = plat.morph_shape(shape.shape_id, {"scale": 3.5})
        self.assertIsNotNone(morphed)
        self.assertEqual(morphed.parameters["scale"], 3.5)

        # 3. Apply Modifier
        mod = ShapeModifierSpec(modifier_type=ModifierType.BEVEL, parameters={"width": 0.05, "segments": 2})
        mod_instance = plat.apply_modifier(shape.shape_id, mod)
        self.assertIsNotNone(mod_instance)
        self.assertTrue(any(m.modifier_type == ModifierType.BEVEL for m in mod_instance.modifiers))

        # 4. Coordinate Analysis
        coord_analysis = plat.get_coordinate_analysis(shape.shape_id, target_system=CoordinateSystemType.OPENGL_WEBGL_Y_UP)
        self.assertIsNotNone(coord_analysis)
        self.assertEqual(coord_analysis["coordinate_system"], CoordinateSystemType.OPENGL_WEBGL_Y_UP.value)

        # 5. Language Mode Switching
        plat.set_language_mode(LanguageMode.CZ)
        self.assertEqual(plat.get_language_mode(), LanguageMode.CZ)
        loc_lib = plat.get_library()
        self.assertTrue(len(loc_lib) > 0)
        self.assertIn("localized_name", loc_lib[0])

        # 6. Status Report
        status = plat.get_status()
        self.assertEqual(status["active_language"], "CZ")
        self.assertEqual(status["vital_max_hp_rule"], 6)
        self.assertGreater(status["registered_shapes_count"], 0)

    def test_economic_engine_module_reexport(self):
        """Verify krystal_web_hub.economic_engine re-exports all essential classes and singleton."""
        import krystal_web_hub.economic_engine as ee
        self.assertTrue(hasattr(ee, "LanguageMode"))
        self.assertTrue(hasattr(ee, "CoordinateSystemType"))
        self.assertTrue(hasattr(ee, "CartesianCoordinateDescriptor"))
        self.assertTrue(hasattr(ee, "CoordinateSystemConverter"))
        self.assertTrue(hasattr(ee, "MultilingualLookupRegistry"))
        self.assertTrue(hasattr(ee, "GLOBAL_SHAPE_PLATFORM"))
        self.assertEqual(ee.GLOBAL_SHAPE_PLATFORM.get_language_mode(), LanguageMode.SK)


if __name__ == '__main__':
    unittest.main()
