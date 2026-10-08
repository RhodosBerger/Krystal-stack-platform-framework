"""
Unit test suite for Skeuomorphic Procedural Synthesis Engine.
Verifies real-world item/character/room synthesis, Golden Ratio canon,
material substrates, multi-substrate code generation, and the VITAL_MAX_HP = 6 invariant.
"""

import unittest
from krystal_web_hub.economic_engine.skeuomorphic_procedural_engine import (
    MaterialSubstrate,
    MATERIAL_SUBSTRATES,
    SkeuomorphicItemType,
    CharacterArchetype,
    RoomArchetype,
    SkeuomorphicProceduralEngine,
    GLOBAL_SKEUOMORPHIC_ENGINE,
    VITAL_MAX_HP,
    GOLDEN_RATIO
)

class TestSkeuomorphicProceduralEngine(unittest.TestCase):

    def setUp(self):
        self.engine = SkeuomorphicProceduralEngine()

    def test_material_substrates_properties(self):
        """Verify all 7 physical substrates have valid physical and PBR properties."""
        self.assertEqual(len(MATERIAL_SUBSTRATES), 7)
        for sub_id, mat in MATERIAL_SUBSTRATES.items():
            self.assertGreater(mat.density_g_cm3, 0.0)
            self.assertGreaterEqual(mat.roughness, 0.0)
            self.assertLessEqual(mat.roughness, 1.0)
            self.assertGreaterEqual(mat.metallic, 0.0)
            self.assertLessEqual(mat.metallic, 1.0)
            self.assertGreater(mat.normal_map_depth, 0.0)
            self.assertTrue(mat.color_hex.startswith("#"))
            self.assertTrue(len(mat.tactile_grain_texture) > 0)

    def test_item_synthesis_and_invariant(self):
        """Verify that items reflect real-world physics, components, and VITAL_MAX_HP = 6."""
        for item_type in SkeuomorphicItemType:
            item = self.engine.synthesize_item(item_type, seed=42)
            self.assertEqual(item.vital_max_hp, VITAL_MAX_HP)
            self.assertEqual(item.vital_max_hp, 6)
            self.assertGreater(item.mass_weight_kg, 0.0)
            self.assertGreater(len(item.components), 0)
            self.assertTrue(len(item.functional_joinery) > 0)
            self.assertEqual(len(item.bounding_dimensions_cm), 3)
            self.assertGreater(item.bounding_dimensions_cm[0], 0.0)

    def test_character_vitruvian_canon(self):
        """Verify characters follow 8-head Vitruvian proportion and garment layering."""
        for arch in CharacterArchetype:
            ch = self.engine.synthesize_character(arch, seed=108)
            self.assertEqual(ch.vital_max_hp, VITAL_MAX_HP)
            self.assertEqual(ch.vital_max_hp, 6)
            self.assertEqual(ch.vitruvian_head_ratio, 8.0)
            self.assertAlmostEqual(ch.total_height_cm / ch.head_height_cm, 8.0, places=2)
            self.assertGreaterEqual(len(ch.garment_layers), 3)
            self.assertIn(arch.value, ch.archetype.value)
            self.assertTrue(len(ch.tactile_ascii_silhouette) > 0)

    def test_room_architecture_golden_ratio(self):
        """Verify architectural rooms follow Golden Ratio proportioning."""
        for room_type in RoomArchetype:
            room = self.engine.synthesize_room(room_type, seed=256)
            self.assertAlmostEqual(room.length_m / room.width_m, GOLDEN_RATIO, places=2)
            self.assertGreaterEqual(room.golden_ratio_adherence, 0.95)
            self.assertGreaterEqual(len(room.architectural_features), 4)
            self.assertTrue(len(room.elevation_ascii) > 0)

    def test_deterministic_reproducibility(self):
        """Verify identical seed yields identical mass, wear, and dimensions."""
        seed = 777
        item_a = self.engine.synthesize_item(SkeuomorphicItemType.FORGED_DAMASCUS_DAGGER, seed=seed)
        item_b = self.engine.synthesize_item(SkeuomorphicItemType.FORGED_DAMASCUS_DAGGER, seed=seed)
        self.assertEqual(item_a.mass_weight_kg, item_b.mass_weight_kg)
        self.assertEqual(item_a.wear_patina_factor, item_b.wear_patina_factor)
        self.assertEqual(item_a.bounding_dimensions_cm, item_b.bounding_dimensions_cm)

    def test_multisubstrate_code_generators(self):
        """Verify Godot 4 Forward+ .tscn, Java 21 Records, and Janet DSL exports."""
        item = self.engine.synthesize_item(SkeuomorphicItemType.ALCHEMIST_LEATHER_GRIMOIRE, seed=42)
        
        # Godot .tscn
        tscn = self.engine.export_to_godot_tscn(item)
        self.assertIn("[gd_scene", tscn)
        self.assertIn("StandardMaterial3D", tscn)
        self.assertIn("metadata/vital_max_hp = 6", tscn)

        # Java 21 Records
        java_code = self.engine.export_to_java_records(item)
        self.assertIn("public record SkeuomorphicItemRecord", java_code)
        self.assertIn("public static final int VITAL_MAX_HP = 6;", java_code)

        # Janet DSL
        janet_code = self.engine.export_to_janet_dsl(item)
        self.assertIn("(def VITAL-MAX-HP 6)", janet_code)
        self.assertIn(":alchemist_leather_grimoire", janet_code)

    def test_svg_rendering(self):
        """Verify vector SVG rendering produces well-formed XML with gradients."""
        item = self.engine.synthesize_item(SkeuomorphicItemType.BRASS_ASTROLABE_SEXTANT, seed=42)
        svg = self.engine.render_item_svg(item)
        self.assertTrue(svg.startswith("<svg"))
        self.assertTrue(svg.endswith("</svg>"))
        self.assertIn("<defs>", svg)
        self.assertIn("linearGradient", svg)

if __name__ == '__main__':
    unittest.main()
