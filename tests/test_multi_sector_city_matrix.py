"""
Unit tests for Multi-Sector Metropolis Matrix Engine (krystal_web_hub.economic_engine)
Validates multi-sector grid synthesis, 5 district biomes, kinetic traffic fleet,
boundary stitching, and strict VITAL_MAX_HP = 6 enforcement across all entities.
"""

import unittest
from krystal_web_hub.economic_engine import (
    MultiSectorMetropolisEngine,
    GLOBAL_METROPOLIS_ENGINE,
    DistrictBiome,
    DISTRICT_SPECS,
    VITAL_MAX_HP,
    GOLDEN_RATIO,
)


class TestMultiSectorMetropolisMatrix(unittest.TestCase):

    def setUp(self) -> None:
        self.engine = MultiSectorMetropolisEngine()

    def test_global_singleton(self) -> None:
        self.assertIsNotNone(GLOBAL_METROPOLIS_ENGINE)
        self.assertIsInstance(GLOBAL_METROPOLIS_ENGINE, MultiSectorMetropolisEngine)

    def test_district_biomes_completeness(self) -> None:
        expected = [
            DistrictBiome.DOWNTOWN_CYBER_SPIRES,
            DistrictBiome.HISTORIC_GOTHIC_QUARTER,
            DistrictBiome.INDUSTRIAL_DOCKS_CANAL,
            DistrictBiome.RESIDENTIAL_GARDEN_TERRACES,
            DistrictBiome.BIOPHILIC_CENTRAL_PARK
        ]
        for biome in expected:
            self.assertIn(biome, DISTRICT_SPECS)
            spec = DISTRICT_SPECS[biome]
            self.assertTrue(len(spec.name_sk) > 0)
            self.assertTrue(len(spec.primary_dominant_brush) > 0)

    def test_metropolis_3x3_synthesis_and_vital_hp(self) -> None:
        metro = self.engine.build_metropolis(seed=101, grid_cols=3, grid_rows=3, sector_size_m=240.0)

        self.assertEqual(metro.grid_dim, (3, 3))
        self.assertEqual(len(metro.sectors), 9)
        self.assertEqual(metro.total_world_width_m, 720.0)
        self.assertEqual(metro.total_world_depth_m, 720.0)
        self.assertGreater(metro.total_assets_count, 400)
        self.assertGreater(len(metro.kinetic_traffic_fleet), 30)

        # Verify Immutable Invariant: VITAL_MAX_HP = 6
        self.assertTrue(metro.vital_max_hp_verified)
        for agent in metro.kinetic_traffic_fleet:
            self.assertLessEqual(agent.vital_hp, VITAL_MAX_HP)

        for sec in metro.sectors.values():
            self.assertTrue(sec.composition.vital_max_hp_invariant_verified)
            for layer in sec.composition.layers.values():
                for inst in layer.instances:
                    self.assertLessEqual(inst.vital_hp, VITAL_MAX_HP)

    def test_deterministic_seed_reproduction(self) -> None:
        seed = 5555
        metro_a = self.engine.build_metropolis(seed=seed)
        metro_b = self.engine.build_metropolis(seed=seed)

        self.assertEqual(metro_a.total_assets_count, metro_b.total_assets_count)
        self.assertEqual(len(metro_a.kinetic_traffic_fleet), len(metro_b.kinetic_traffic_fleet))
        self.assertEqual(metro_a.metropolis_ascii_map, metro_b.metropolis_ascii_map)

        for coord in metro_a.sectors:
            sec_a = metro_a.sectors[coord]
            sec_b = metro_b.sectors[coord]
            self.assertEqual(sec_a.district_biome, sec_b.district_biome)
            self.assertEqual(sec_a.world_origin_x, sec_b.world_origin_x)
            self.assertEqual(sec_a.world_origin_z, sec_b.world_origin_z)
            self.assertEqual(sec_a.composition.total_assets_count, sec_b.composition.total_assets_count)

    def test_ascii_metropolis_map_generation(self) -> None:
        metro = self.engine.build_metropolis(seed=42)
        ascii_map = metro.metropolis_ascii_map

        self.assertIn("KRYSTAL METROPOLIS MULTI-SECTOR TACTICAL OVERVIEW MAP", ascii_map)
        self.assertIn("DOWNTOWN", ascii_map)
        self.assertIn("ZÓNOVANIE", ascii_map)

    def test_godot_metropolis_tscn_export(self) -> None:
        metro = self.engine.build_metropolis(seed=42)
        tscn_text = self.engine.export_metropolis_to_godot_tscn(metro)

        self.assertIn('[gd_scene load_steps=5 format=3', tscn_text)
        self.assertIn('[node name="MetropolisRoot" type="Node3D"]', tscn_text)
        self.assertIn('[node name="WorldEnvironment" type="WorldEnvironment"', tscn_text)
        self.assertIn('[node name="CinematicOverviewCamera" type="Camera3D"', tscn_text)
        self.assertIn('metadata/district_biome =', tscn_text)
        self.assertIn('metadata/vital_hp = 6', tscn_text)


if __name__ == "__main__":
    unittest.main()
