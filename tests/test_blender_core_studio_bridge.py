"""
Unit Test Suite for Blender Core Studio Bridge & Modifier Compositor
====================================================================
Verifies:
  1. ModifierStack (Array, Mirror, Bevel, Displace, Deform, Solidify)
  2. Mimicry Recipe Catalog (13 high-fidelity recipes)
  3. Game Scene Catalog (9 composite scenes)
  4. 3D Wavefront OBJ export generation
  5. Blender 4.x Python script generation (bpy)
  6. Godot 4.x scene generation (.tscn)
  7. AI Studio NPRS v3.0 prompt block generation
  8. MCP Tool dispatch integration
  9. Invariant: VITAL_MAX_HP == 6
"""

import unittest
from mimicry_engine.primitives import sdf_box
from krystal_kernel.blender_core_studio_bridge import (
    BlenderModifierStack,
    ArrayModifier,
    MirrorModifier,
    BevelModifier,
    DisplaceModifier,
    DeformModifier,
    SolidifyModifier,
    GLOBAL_BLENDER_CORE_STUDIO_BRIDGE,
    VITAL_MAX_HP
)
from krystal_web_hub.hybrid_mcp_antigravity_connector import GLOBAL_HYBRID_MCP_CONNECTOR


class TestBlenderCoreStudioBridge(unittest.TestCase):

    def test_vital_max_hp_invariant(self):
        """Ensure vital invariant VITAL_MAX_HP == 6 is strictly preserved."""
        self.assertEqual(VITAL_MAX_HP, 6)

    def test_modifier_stack_eval(self):
        """Test modifier stack SDF evaluation chaining."""
        base_box = lambda p: sdf_box(p, (0.5, 0.5, 0.5))
        stack = BlenderModifierStack(base_sdf=base_box)
        stack.add_modifier(ArrayModifier(name="LinearArray", count=3, offset=(1.5, 0.0, 0.0)))
        stack.add_modifier(MirrorModifier(name="SymmX", use_x=True, use_y=False, use_z=False))
        stack.add_modifier(BevelModifier(name="BevelEdge", radius=0.05))
        stack.add_modifier(SolidifyModifier(name="Shell", thickness=0.08))

        # Evaluate at origin (inside the box)
        d_center = stack.evaluate((0.0, 0.0, 0.0))
        self.assertIsInstance(d_center, float)

        # Check serialization
        serialized = stack.to_dict()
        self.assertEqual(len(serialized), 4)
        self.assertEqual(serialized[0]["type"], "ArrayModifier")
        self.assertEqual(serialized[1]["type"], "MirrorModifier")
        self.assertEqual(serialized[2]["type"], "BevelModifier")
        self.assertEqual(serialized[3]["type"], "SolidifyModifier")

    def test_catalog_inventory(self):
        """Verify that all 13 recipes and at least 9 scenes are cataloged."""
        catalog = GLOBAL_BLENDER_CORE_STUDIO_BRIDGE.get_catalog()
        self.assertEqual(catalog["total_recipes"], 13)
        self.assertGreaterEqual(catalog["total_scenes"], 9)
        self.assertEqual(len(catalog["recipes"]), 13)
        self.assertGreaterEqual(len(catalog["scenes"]), 9)
        self.assertEqual(catalog["vital_max_hp"], 6)

    def test_recipe_import_bundle(self):
        """Test importing a recipe and generating export artifacts."""
        bundle = GLOBAL_BLENDER_CORE_STUDIO_BRIDGE.import_recipe_to_studio("CYBER_TURRET_MK4")
        self.assertEqual(bundle.recipe_id, "CYBER_TURRET_MK4")
        self.assertGreater(bundle.vertex_count, 0)
        self.assertGreater(bundle.face_count, 0)
        self.assertGreater(bundle.parts_count, 0)

        # Check Wavefront OBJ
        self.assertIn("# Krystal-Stack Studio 3D Mesh Export", bundle.obj_data)
        self.assertIn("v ", bundle.obj_data)
        self.assertIn("f ", bundle.obj_data)

        # Check Blender bpy script
        self.assertIn("import bpy", bundle.blender_bpy_script)
        self.assertIn("def build_krystal_asset():", bundle.blender_bpy_script)
        self.assertIn("modifiers.new", bundle.blender_bpy_script)

        # Check Godot .tscn
        self.assertIn("[gd_scene format=3", bundle.godot_tscn)
        self.assertIn('type="Node3D"', bundle.godot_tscn)

        # Check AI Studio Prompt
        self.assertIn("AI STUDIO // NPRS v3.0", bundle.ai_studio_prompt)

    def test_scene_import_bundle(self):
        """Test importing an entire composite 3D scene."""
        bundle = GLOBAL_BLENDER_CORE_STUDIO_BRIDGE.import_scene_to_studio("SCENE_CYBERPUNK_DISTRICT")
        self.assertEqual(bundle.scene_id, "SCENE_CYBERPUNK_DISTRICT")
        self.assertEqual(bundle.actors_count, 4)
        self.assertIn("import bpy", bundle.blender_bpy_script)
        self.assertIn("CYBERPUNK_DISTRICT", bundle.blender_bpy_script)
        self.assertEqual(bundle.vital_max_hp, 6)

    def test_mcp_tool_execution(self):
        """Test hybrid MCP tool 'krystal-blender-core-studio-import'."""
        # 1. Catalog action
        res_cat = GLOBAL_HYBRID_MCP_CONNECTOR.execute_tool(
            "krystal-blender-core-studio-import",
            {"action": "catalog"}
        )
        self.assertEqual(res_cat["status"], "OK")
        self.assertEqual(res_cat["total_recipes"], 13)

        # 2. Import recipe action
        res_rec = GLOBAL_HYBRID_MCP_CONNECTOR.execute_tool(
            "krystal-blender-core-studio-import",
            {"action": "import_recipe", "recipe_id": "MECH_WALKER_TITAN"}
        )
        self.assertEqual(res_rec["status"], "OK")
        self.assertEqual(res_rec["bundle"]["recipe_id"], "MECH_WALKER_TITAN")

        # 3. Export OBJ action
        res_obj = GLOBAL_HYBRID_MCP_CONNECTOR.execute_tool(
            "krystal-blender-core-studio-import",
            {"action": "export_obj", "recipe_id": "CYBER_TURRET_MK4"}
        )
        self.assertEqual(res_obj["status"], "OK")
        self.assertIn("v ", res_obj["obj_data"])

        # 4. Generate BPY action
        res_bpy = GLOBAL_HYBRID_MCP_CONNECTOR.execute_tool(
            "krystal-blender-core-studio-import",
            {"action": "generate_bpy", "recipe_id": "ANCIENT_OBELISK_MONOLITH"}
        )
        self.assertEqual(res_bpy["status"], "OK")
        self.assertIn("import bpy", res_bpy["blender_bpy_script"])

        # 5. Import Scene action
        res_scn = GLOBAL_HYBRID_MCP_CONNECTOR.execute_tool(
            "krystal-blender-core-studio-import",
            {"action": "import_scene", "scene_id": "SCENE_OLD_TOWN_PRAGUE_SQUARE"}
        )
        self.assertEqual(res_scn["status"], "OK")
        self.assertEqual(res_scn["bundle"]["scene_id"], "SCENE_OLD_TOWN_PRAGUE_SQUARE")


if __name__ == "__main__":
    unittest.main()
