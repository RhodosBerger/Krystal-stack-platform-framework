"""
Unit tests for Procedural City Composition Engine (krystal_web_hub.economic_engine)
Validates multi-asset canvas painting, Golden Ratio focal positioning,
deterministic reproduction, dual ASCII visual rendering, and VITAL_MAX_HP invariant.
"""

import unittest
from krystal_web_hub.economic_engine import (
    ProceduralCityCompositionEngine,
    GLOBAL_CITY_COMPOSITION_ENGINE,
    VITAL_MAX_HP,
    AssetCategory,
    ASSET_BRUSH_CATALOG,
)


class TestProceduralCityComposition(unittest.TestCase):

    def setUp(self) -> None:
        self.engine = ProceduralCityCompositionEngine()

    def test_global_engine_singleton(self) -> None:
        self.assertIsNotNone(GLOBAL_CITY_COMPOSITION_ENGINE)
        self.assertIsInstance(GLOBAL_CITY_COMPOSITION_ENGINE, ProceduralCityCompositionEngine)

    def test_asset_brush_catalog_completeness(self) -> None:
        required_brushes = [
            "CYBER_SPIRE_MONOLITH",
            "ALCHEMICAL_CLOCKTOWER",
            "DATA_CITADEL_ZIGGURAT",
            "MODULAR_TENEMENT_BLOCK",
            "COMMERCIAL_ARCADE_PLINTH",
            "STEPPED_TERRACE_RESIDENCE",
            "GRAND_BOULEVARD_CONDUIT",
            "CANAL_BASIN_WATERWAY",
            "AVENUE_LINDEN_TREE",
            "URBAN_PLAZA_FOUNTAIN",
            "ORNATE_STREET_LAMP",
            "NEON_CYBER_BILLBOARD",
            "PANTHER_CRUISER_2D",
        ]
        for b_id in required_brushes:
            self.assertIn(b_id, ASSET_BRUSH_CATALOG)
            brush = ASSET_BRUSH_CATALOG[b_id]
            self.assertLessEqual(brush.vital_hp, VITAL_MAX_HP)
            self.assertTrue(brush.width_m > 0)
            self.assertTrue(brush.height_m >= 0)

    def test_deterministic_seed_reproduction(self) -> None:
        seed = 10101
        comp_a = self.engine.paint_city_composition(seed=seed, city_name="Test City Alpha")
        comp_b = self.engine.paint_city_composition(seed=seed, city_name="Test City Alpha")

        self.assertEqual(comp_a.total_assets_count, comp_b.total_assets_count)
        self.assertAlmostEqual(comp_a.focal_point_x, comp_b.focal_point_x, places=6)
        self.assertAlmostEqual(comp_a.focal_point_z, comp_b.focal_point_z, places=6)
        self.assertEqual(comp_a.ascii_skyline_view, comp_b.ascii_skyline_view)
        self.assertEqual(comp_a.ascii_plan_view, comp_b.ascii_plan_view)

        for layer_idx in comp_a.layers:
            instances_a = comp_a.layers[layer_idx].instances
            instances_b = comp_b.layers[layer_idx].instances
            self.assertEqual(len(instances_a), len(instances_b))
            for ia, ib in zip(instances_a, instances_b):
                self.assertEqual(ia.instance_id, ib.instance_id)
                self.assertAlmostEqual(ia.pos_x, ib.pos_x, places=5)
                self.assertAlmostEqual(ia.pos_z, ib.pos_z, places=5)
                self.assertEqual(ia.vital_hp, ib.vital_hp)

    def test_vital_max_hp_strict_invariant(self) -> None:
        seeds = [1, 42, 999, 12345]
        for s in seeds:
            comp = self.engine.paint_city_composition(seed=s)
            self.assertTrue(comp.vital_max_hp_invariant_verified)
            for layer in comp.layers.values():
                for inst in layer.instances:
                    self.assertLessEqual(inst.vital_hp, VITAL_MAX_HP)

    def test_golden_ratio_focal_positioning(self) -> None:
        comp = self.engine.paint_city_composition(seed=777, canvas_width_m=200.0)
        self.assertGreaterEqual(comp.golden_ratio_adherence_score, 0.95)
        # Check that focal anchor instance exists in Layer 0
        layer_0 = comp.layers[0]
        focal_instances = [inst for inst in layer_0.instances if inst.is_focal_anchor]
        self.assertEqual(len(focal_instances), 1)
        self.assertEqual(focal_instances[0].brush_id, "CYBER_SPIRE_MONOLITH")

    def test_dual_ascii_visual_canvases_generated(self) -> None:
        comp = self.engine.paint_city_composition(seed=42)
        # Verify skyline silhouette
        self.assertIn("LEGENDA SILUETY", comp.ascii_skyline_view)
        self.assertIn("▲", comp.ascii_skyline_view)  # Spire needle
        self.assertIn("═", comp.ascii_skyline_view)  # Ground line
        self.assertIn("Cruiser (HP=6)", comp.ascii_skyline_view)

        # Verify plan view
        self.assertIn("PÔDORYS KOMPOZÍCIE", comp.ascii_plan_view)
        self.assertIn("║", comp.ascii_plan_view)  # Boulevard
        self.assertIn("♣", comp.ascii_plan_view)  # Linden trees
        self.assertIn("Cruiser (HP=6)", comp.ascii_plan_view)

    def test_godot_tscn_export(self) -> None:
        comp = self.engine.paint_city_composition(seed=42)
        tscn_text = self.engine.export_to_godot_tscn(comp)

        self.assertIn('[gd_scene load_steps=6 format=3', tscn_text)
        self.assertIn('[node name="CityCompositionRoot" type="Node3D"]', tscn_text)
        self.assertIn('[node name="WorldEnvironment" type="WorldEnvironment"', tscn_text)
        self.assertIn('[node name="DirectionalSunLight" type="DirectionalLight3D"', tscn_text)
        self.assertIn('[node name="CinematicCamera3D" type="Camera3D"', tscn_text)
        self.assertIn('metadata/vital_hp = 6', tscn_text)
        self.assertIn('metadata/is_focal_anchor = true', tscn_text)

    def test_java_records_export(self) -> None:
        comp = self.engine.paint_city_composition(seed=42)
        java_text = self.engine.export_to_java_records(comp)

        self.assertIn("public final class CityCompositionPipeline", java_text)
        self.assertIn("public record CityAssetInstance", java_text)
        self.assertIn("public record CityComposition", java_text)
        self.assertIn("public static final int VITAL_MAX_HP = 6;", java_text)
        self.assertIn("switch (asset.layerIndex())", java_text)

    def test_janet_dsl_export(self) -> None:
        comp = self.engine.paint_city_composition(seed=42)
        janet_text = self.engine.export_to_janet_dsl(comp)

        self.assertIn("(def VITAL-MAX-HP 6)", janet_text)
        self.assertIn("(def CITY-COMPOSITION", janet_text)
        self.assertIn(":vital-invariant-verified true", janet_text)
        self.assertIn(":brush \"CYBER_SPIRE_MONOLITH\"", janet_text)


if __name__ == "__main__":
    unittest.main()
