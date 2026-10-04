"""
Unit Tests for Godot Interior Surface Node Engine
=============================================================================
Tests:
- Interior Material Catalog & PBR Properties
- 6 Max HP Vital Invariant Preservation
- Node Graph Topology & Default Connections
- Procedural World Pre-generation (Trees, Buildings, Superblocks, Soil Horizons)
- Godot 4.x .TSCN Scene & GDScript Process Exporter
=============================================================================
"""

import unittest
from krystal_web_hub.economic_engine.godot_interior_surface_node_engine import (
    GodotInteriorSurfaceNodeEngine,
    GLOBAL_GODOT_INTERIOR_NODE_ENGINE,
    INTERIOR_MATERIALS_CATALOG,
    VITAL_MAX_HP,
    GOLDEN_RATIO
)


class TestGodotInteriorSurfaceNodeEngine(unittest.TestCase):
    def setUp(self):
        self.engine = GodotInteriorSurfaceNodeEngine(seed=42)

    def test_vital_max_hp_rule(self):
        self.assertEqual(VITAL_MAX_HP, 6)
        self.assertAlmostEqual(GOLDEN_RATIO, 1.61803398875, places=6)

    def test_material_catalog(self):
        catalog = self.engine.get_node_catalog()
        self.assertIn("materials_catalog", catalog)
        self.assertGreaterEqual(len(catalog["materials_catalog"]), 7)

        mat_ids = {m["material_id"] for m in catalog["materials_catalog"]}
        self.assertIn("oak_herringbone_parquet", mat_ids)
        self.assertIn("biophilic_vertical_moss", mat_ids)
        self.assertIn("roman_travertine_stone", mat_ids)
        self.assertIn("venetian_terrazzo", mat_ids)

        moss = INTERIOR_MATERIALS_CATALOG["biophilic_vertical_moss"]
        self.assertGreater(moss.biophilic_resonance, 0.9)
        self.assertGreater(moss.acoustic_absorption, 0.8)

    def test_graph_topology(self):
        graph = self.engine.get_current_graph()
        self.assertEqual(graph["vital_max_hp_rule"], 6)
        self.assertGreaterEqual(graph["nodes_count"], 7)
        self.assertGreaterEqual(graph["connections_count"], 8)

        node_types = {n["node_type"] for n in graph["nodes"]}
        self.assertIn("InteriorMaterialNode", node_types)
        self.assertIn("BiophilicVegetationPreGenNode", node_types)
        self.assertIn("ArchitecturalBuildingNode", node_types)
        self.assertIn("UrbanCitySuperblockNode", node_types)
        self.assertIn("SoilClimateSubstrateNode", node_types)
        self.assertIn("GodotMultiMeshExporterNode", node_types)

    def test_procedural_world_pregeneration(self):
        res = self.engine.evaluate_graph({"seed": 777})
        self.assertEqual(res["status"], "SUCCESS_WORLD_PREGENERATED")
        self.assertEqual(res["vital_max_hp_rule"], 6)

        metrics = res["metrics"]
        self.assertGreater(metrics["total_trees_instanced"], 100)
        self.assertGreater(metrics["total_buildings"], 10)
        self.assertGreater(metrics["total_superblocks"], 2)
        self.assertEqual(metrics["soil_horizons_count"], 4)
        self.assertLessEqual(metrics["multimesh_instance_count"], 15000)

        # Invariant checks on sample entities
        for tree in res["trees_sample"]:
            self.assertEqual(tree["vital_hp"], 6)
            self.assertGreater(tree["scale"], 0)

        for bldg in res["buildings_sample"]:
            self.assertEqual(bldg["vital_hp"], 6)
            self.assertGreater(bldg["height"], 0)

        for sb in res["superblocks"]:
            self.assertEqual(sb["vital_hp"], 6)
            self.assertGreater(sb["green_percent"], 0)

        soil_codes = [s["horizon"] for s in res["soil_horizons"]]
        self.assertEqual(soil_codes, ["A", "B", "C", "R"])

    def test_godot_scene_tscn_export(self):
        tscn = self.engine.generate_godot_scene_tscn()
        self.assertIn('[gd_scene format=3 uid="uid://krystal_interior_surface_world_pregen"]', tscn)
        self.assertIn('[sub_resource type="StandardMaterial3D" id="Mat_OakHerringbone"]', tscn)
        self.assertIn('[sub_resource type="MultiMesh" id="MultiMesh_Trees"]', tscn)
        self.assertIn('[node name="InteriorWorldStage" type="Node3D"]', tscn)
        self.assertIn('[node name="VegetationMultiMesh" type="MultiMeshInstance3D"', tscn)

    def test_godot_gdscript_export(self):
        gd = self.engine.generate_godot_gdscript()
        self.assertIn('extends Node3D', gd)
        self.assertIn('const VITAL_MAX_HP: int = 6', gd)
        self.assertIn('func _process(delta: float) -> void:', gd)
        self.assertIn('func _physics_process(delta: float) -> void:', gd)
        self.assertIn('func _populate_tree_transforms() -> void:', gd)


if __name__ == '__main__':
    unittest.main()
