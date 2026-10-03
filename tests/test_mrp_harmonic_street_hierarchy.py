"""
Unit Tests for MRP Harmonic Street Hierarchy & Pink Panther Aesthetic Engine
============================================================================
Validates the 5-tier MRP procedural engine hierarchy, Golden Ratio phi color axioms,
Pink Panther aesthetic tints, 12 canonical world regions, procedural street maps,
top-down 2D vehicle cruisers, and the strict 6 Max HP vital invariant.
"""

import unittest
import os
from krystal_web_hub.economic_engine.mrp_harmonic_street_hierarchy import (
    GOLDEN_RATIO,
    INV_GOLDEN_RATIO,
    VITAL_MAX_HP,
    PINK_PANTHER_PALETTE,
    HarmonicColorTint,
    MRPHierarchyLevel,
    WorldRegionMapSpec,
    VehicleCruiser2D,
    MRPHarmonicStreetEngine,
    GLOBAL_MRP_HARMONIC_STREET_ENGINE
)


class TestMRPHarmonicStreetHierarchy(unittest.TestCase):

    def setUp(self):
        self.engine = MRPHarmonicStreetEngine()

    def test_vital_max_hp_constant_is_strictly_six(self):
        self.assertEqual(VITAL_MAX_HP, 6)

    def test_golden_ratio_constants(self):
        self.assertAlmostEqual(GOLDEN_RATIO, 1.61803398875, places=5)
        self.assertAlmostEqual(INV_GOLDEN_RATIO, 1.0 / 1.61803398875, places=5)

    def test_mrp_hierarchy_5_levels(self):
        levels = self.engine.get_hierarchy_levels()
        self.assertEqual(len(levels), 5)
        level_ids = [lvl["level_id"] for lvl in levels]
        self.assertEqual(level_ids, [0, 1, 2, 3, 4])

        # Level 0 is Axiomatic Root
        self.assertEqual(levels[0]["level_id"], 0)
        self.assertIn("Harmonic Color Tints", levels[0]["name"])
        self.assertEqual(levels[0]["dependencies"], [])

        # Level 4 is Execution & 6 Max HP rule
        self.assertEqual(levels[4]["level_id"], 4)
        self.assertIn("6 Max HP", levels[4]["name"])
        self.assertIn("6", levels[4]["axiomatic_invariant"])

    def test_pink_panther_palette_integrity(self):
        palette = self.engine.get_pink_panther_palette()
        self.assertIn("panther_pink", palette)
        self.assertIn("hot_magenta", palette)
        self.assertIn("blush_pastel", palette)
        self.assertIn("champagne_cream", palette)
        self.assertIn("noir_charcoal", palette)
        self.assertIn("gotham_slate", palette)
        self.assertIn("sin_city_crimson", palette)
        self.assertIn("vegas_gold", palette)

        # Verify RGB tuples and hex codes
        for key, color in palette.items():
            self.assertTrue(color["hex"].startswith("#"))
            self.assertEqual(len(color["rgb"]), 3)
            for ch in color["rgb"]:
                self.assertTrue(0 <= ch <= 255)

    def test_harmonic_color_tint_computation(self):
        tint_base = self.engine.compute_harmonic_color_tint(
            base_color=(236, 72, 153),
            step=1,
            weight=1.0,
            style="pink_panther_chic"
        )
        self.assertIsInstance(tint_base, HarmonicColorTint)
        self.assertEqual(tint_base.step, 1)
        self.assertTrue(0 <= tint_base.r <= 255)
        self.assertTrue(0 <= tint_base.g <= 255)
        self.assertTrue(0 <= tint_base.b <= 255)
        self.assertTrue(tint_base.hex_code.startswith("#"))
        self.assertEqual(len(tint_base.hex_code), 7)

        # Test step clamping [0, 4]
        tint_over = self.engine.compute_harmonic_color_tint(step=10)
        self.assertEqual(tint_over.step, 4)
        tint_neg = self.engine.compute_harmonic_color_tint(step=-5)
        self.assertEqual(tint_neg.step, 0)

        # Test noir style
        tint_noir = self.engine.compute_harmonic_color_tint(style="noir_monochrome")
        self.assertTrue(0 <= tint_noir.r <= 255)

    def test_twelve_canonical_world_regions(self):
        regions = self.engine.get_canonical_regions()
        self.assertEqual(len(regions), 12)

        expected_regions = [
            "gotham_city", "arkham_city", "sin_city", "las_vegas",
            "alabama", "ohio", "florida", "australia",
            "canary_islands", "bolivia", "ecuador", "peru"
        ]
        for r_id in expected_regions:
            self.assertIn(r_id, regions, f"Missing expected canonical region: {r_id}")
            reg = regions[r_id]
            self.assertIn("osm_bounding_box", reg)
            self.assertEqual(len(reg["osm_bounding_box"]), 4)
            self.assertGreater(reg["default_speed_limit_kmh"], 0)

    def test_procedural_street_network_generation(self):
        # Gotham City (grid topology)
        gotham_net = self.engine.generate_procedural_street_network("gotham_city", seed=101)
        self.assertGreaterEqual(gotham_net["nodes_count"], 15)
        self.assertGreaterEqual(gotham_net["edges_count"], 20)
        self.assertGreaterEqual(gotham_net["parcels_count"], 5)
        self.assertIn("osm_metadata", gotham_net)

        # Bolivia (mountain serpentine ribbon topology)
        bolivia_net = self.engine.generate_procedural_street_network("bolivia", seed=202)
        self.assertGreaterEqual(bolivia_net["nodes_count"], 10)
        self.assertGreaterEqual(bolivia_net["edges_count"], 9)
        self.assertIn("Pass Segment", bolivia_net["edges"][0]["name"])

        # Las Vegas (strip boulevard topology)
        vegas_net = self.engine.generate_procedural_street_network("las_vegas", seed=303)
        self.assertGreaterEqual(vegas_net["nodes_count"], 10)
        self.assertTrue(any("Strip" in e["name"] for e in vegas_net["edges"]))

    def test_vehicle_simulation_and_kinematics(self):
        v = self.engine.get_vehicle("panther_coupe_01")
        self.assertIsNotNone(v)
        start_x = v["x"]
        start_y = v["y"]

        # Forward tick
        updated = self.engine.simulate_vehicle_tick(
            vehicle_id="panther_coupe_01",
            dt_seconds=0.1,
            throttle=1.0,
            steering=0.0
        )
        self.assertEqual(updated["vehicle_id"], "panther_coupe_01")
        self.assertGreater(updated["velocity_mps"], 15.0)

        # Steering & drift tick
        steered = self.engine.simulate_vehicle_tick(
            vehicle_id="panther_coupe_01",
            dt_seconds=0.1,
            throttle=0.8,
            steering=0.7,
            drift_boost=True
        )
        self.assertTrue(steered["is_drifting"])

    def test_strict_six_max_hp_vital_invariant(self):
        # Register cruiser with higher HP -> must clamp to 6
        rogue_cruiser = VehicleCruiser2D(
            vehicle_id="test_over_hp",
            name="Overclocked Test Car",
            cruiser_type="panther_coupe",
            region_id="gotham_city",
            x=100.0,
            y=100.0,
            heading_deg=0.0,
            velocity_mps=10.0,
            max_velocity_mps=40.0,
            acceleration_mps2=5.0,
            braking_mps2=10.0,
            turn_rate_degps=90.0,
            drift_factor=0.5,
            hp=999  # Violation!
        )
        reg_dict = self.engine.register_vehicle(rogue_cruiser)
        self.assertEqual(reg_dict["hp"], 6)
        self.assertEqual(reg_dict["max_hp"], 6)

        # Damage vehicle
        damaged = self.engine.apply_damage_to_vehicle("test_over_hp", damage_amount=3)
        self.assertLess(damaged["hp"], 6)
        self.assertGreaterEqual(damaged["hp"], 0)

        # Repair vehicle -> cannot exceed 6
        repaired = self.engine.repair_vehicle("test_over_hp", repair_amount=10)
        self.assertEqual(repaired["hp"], 6)
        self.assertEqual(repaired["status"], "operational")

        # Destroy vehicle
        destroyed = self.engine.apply_damage_to_vehicle("test_over_hp", damage_amount=50)
        self.assertEqual(destroyed["hp"], 0)
        self.assertEqual(destroyed["status"], "destroyed")
        self.assertEqual(destroyed["velocity_mps"], 0.0)

        # Re-repairing destroyed vehicle rebuilds it to 1 HP
        rebuilt = self.engine.repair_vehicle("test_over_hp", repair_amount=1)
        self.assertEqual(rebuilt["hp"], 1)
        self.assertEqual(rebuilt["status"], "critical")

    def test_engine_manifesto(self):
        manifesto = self.engine.get_engine_manifesto()
        self.assertEqual(manifesto["vital_max_hp_invariant"], 6)
        self.assertEqual(manifesto["total_mrp_hierarchy_levels"], 5)
        self.assertEqual(manifesto["total_canonical_world_regions"], 12)
        self.assertEqual(manifesto["status"], "active_and_invariant_compliant")

    def test_janet_dsl_definition_file_exists(self):
        janet_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "krystal_janet",
            "mrp_harmonic_street_engine.janet"
        )
        self.assertTrue(os.path.exists(janet_path), f"File {janet_path} must exist")
        with open(janet_path, "r", encoding="utf-8") as f:
            content = f.read()
        self.assertIn("(def VITAL-MAX-HP 6)", content)
        self.assertIn("(def GOLDEN-RATIO 1.6180339887", content)
        self.assertIn("PINK-PANTHER-COLOR-AXIOMS", content)
        self.assertIn("CANONICAL-WORLD-REGIONS", content)
        self.assertIn("simulate-2d-vehicle-vector", content)


if __name__ == "__main__":
    unittest.main()
