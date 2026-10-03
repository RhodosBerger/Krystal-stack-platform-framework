# ==============================================================================
# KRYSTAL-STACK: UNIT TESTS FOR PROCEDURAL RENDER PARAMETERS & GODOT 4 SCENES
# ==============================================================================

import unittest
from krystal_web_hub.economic_engine import (
    PROCEDURAL_PBR_PRESETS,
    PROCEDURAL_MESH_PRESETS,
    PROCEDURAL_PARTICLE_PRESETS,
    PROCEDURAL_ENVIRONMENT_CONFIG,
    Godot4ProceduralSceneBuilder,
    generate_procedural_godot_arena
)

class TestProceduralRenderParameters(unittest.TestCase):

    def test_pbr_presets_integrity(self):
        expected_keys = [
            "crystal_facet_pbr", "toxic_slime_organic", "druid_ancient_bark",
            "druid_mossy_granite", "basalt_fortress_stone", "aether_projectile_energy"
        ]
        for key in expected_keys:
            self.assertIn(key, PROCEDURAL_PBR_PRESETS)
            mat = PROCEDURAL_PBR_PRESETS[key]
            self.assertEqual(mat["type"], "StandardMaterial3D")
            self.assertIn("albedo_color", mat)
            self.assertIn("metallic", mat)
            self.assertIn("roughness", mat)
            self.assertEqual(len(mat["albedo_color"]), 4)

    def test_procedural_mesh_presets(self):
        self.assertIn("hex_column_mesh", PROCEDURAL_MESH_PRESETS)
        hex_mesh = PROCEDURAL_MESH_PRESETS["hex_column_mesh"]
        self.assertEqual(hex_mesh["type"], "CylinderMesh")
        self.assertEqual(hex_mesh["radial_segments"], 6) # Pointy-topped regular hex
        self.assertEqual(hex_mesh["top_radius"], 1.5)

        self.assertIn("crystal_shard_mesh", PROCEDURAL_MESH_PRESETS)
        self.assertIn("slime_blob_mesh", PROCEDURAL_MESH_PRESETS)
        self.assertIn("shockwave_ring_mesh", PROCEDURAL_MESH_PRESETS)

    def test_gpu_particle_presets(self):
        self.assertIn("mana_dust_particles", PROCEDURAL_PARTICLE_PRESETS)
        mana = PROCEDURAL_PARTICLE_PRESETS["mana_dust_particles"]
        self.assertEqual(mana["type"], "ParticleProcessMaterial")
        self.assertIn("initial_velocity_min", mana)
        self.assertIn("initial_velocity_max", mana)
        self.assertIn("color", mana)

    def test_procedural_scene_builder(self):
        builder = Godot4ProceduralSceneBuilder("test_scene")
        mat_id = builder.register_sub_resource(PROCEDURAL_PBR_PRESETS["crystal_facet_pbr"])
        mesh_id = builder.register_sub_resource(PROCEDURAL_MESH_PRESETS["hex_column_mesh"])

        builder.add_node("Root", "Node3D", parent=".")
        builder.add_node(
            "Tile0",
            "MeshInstance3D",
            parent="Root",
            transform=[1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
            mesh_ref=mesh_id,
            material_ref=mat_id
        )

        tscn = builder.serialize_to_tscn()
        self.assertIn('[gd_scene format=3 uid="uid://test_scene"]', tscn)
        self.assertIn('[sub_resource type="StandardMaterial3D" id="StandardMaterial3D_cryst"]', tscn)
        self.assertIn('[sub_resource type="CylinderMesh" id="CylinderMesh_hex"]', tscn)
        self.assertIn('[node name="Root" type="Node3D"]', tscn)
        self.assertIn('mesh = SubResource("CylinderMesh_hex")', tscn)
        self.assertIn('surface_material_override/0 = SubResource("StandardMaterial3D_cryst")', tscn)

    def test_generate_procedural_godot_arena(self):
        arena_tscn = generate_procedural_godot_arena("ArenaSanctum", radius=2, include_particles=True)
        self.assertIn('[gd_scene format=3 uid="uid://krystal_arenasanctum"]', arena_tscn)
        self.assertIn('[node name="HexGrid" type="Node3D"]', arena_tscn)
        self.assertIn('Hex_0_0', arena_tscn)
        self.assertIn('[node name="SpireShard" type="MeshInstance3D"', arena_tscn)
        self.assertIn('[node name="SpireLight" type="OmniLight3D"', arena_tscn)
        self.assertIn('[node name="SpireAuraParticles" type="GPUParticles3D"', arena_tscn)
        self.assertIn('process_material = SubResource("ParticleProcessMaterial_mana")', arena_tscn)

if __name__ == '__main__':
    unittest.main()
