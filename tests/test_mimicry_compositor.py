"""
Krystal-Stack Automated Verification: Blender Modifier Core & Mimicry Compositor
================================================================================
Converted to standard unittest.TestCase for full discoverability and CI reliability.
Preserves all original verification logic:
1. Primitive Signed Distance Functions & Polynomial Smooth CSG (smin / smax)
2. Blender Modifier Stack (Array, Mirror, Boolean, Bevel, Displace, Deform, Solidify)
3. 13 Real-World Composite Object Recipes (Turret, Mech, Obelisk, Xenodrone, Spire, Explorer, etc.)
4. 9 Game Scene Outlines & Godot 4.x .tscn Generation & Export
5. 8 Strict Urban Spatial Composition Rules Engine
6. Live HTTP Endpoints on Localhost Hub (when available)
"""

import sys
import os
import json
import urllib.request
import urllib.error
import unittest

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)


class TestMimicryCompositor(unittest.TestCase):
    """
    Standard discoverable test suite for Blender-style modifier stack,
    SDF primitives, CSG operations, composite recipes, and urban spatial scenes.
    """

    def setUp(self):
        self.root_dir = ROOT_DIR

    def tearDown(self):
        # Clean transient export artifacts
        transient_tscn = os.path.join(self.root_dir, "godot_project", "scenes", "SCENE_PARISIAN_HAUSSMANN_BOULEVARD.tscn")
        if os.path.exists(transient_tscn):
            try:
                os.remove(transient_tscn)
            except OSError:
                pass

    def test_primitives_and_csg(self):
        """Verifies Signed Distance Functions and polynomial smooth CSG operators."""
        from mimicry_engine.primitives import (
            sdf_box, sdf_sphere, sdf_cylinder, sdf_torus, sdf_hex_prism,
            smin, smax, smooth_difference
        )
        p = (0.0, 0.0, 0.0)
        self.assertEqual(sdf_sphere(p, r=1.0), -1.0)
        self.assertEqual(sdf_box(p, b=(0.5, 0.5, 0.5)), -0.5)
        self.assertEqual(sdf_cylinder(p, r=0.5, h=1.0), -0.5)
        self.assertLessEqual(sdf_hex_prism(p, h=0.8, r=0.8), 0.0)

        # Smooth min between 0.5 and 0.5 with k=0.2 should be 0.5 - 0.2/4 = 0.45
        s_val = smin(0.5, 0.5, k=0.2)
        self.assertAlmostEqual(s_val, 0.45, places=4)

        # Smooth max between 0.5 and 0.5 with k=0.2
        s_max_val = smax(0.5, 0.5, k=0.2)
        self.assertAlmostEqual(s_max_val, 0.55, places=4)

        # Smooth difference between 0.2 and 0.5 with k=0.1
        s_diff = smooth_difference(0.2, 0.5, k=0.1)
        self.assertIsInstance(s_diff, float)

    def test_modifier_stack(self):
        """Verifies Blender Modifier Stack (Array, Mirror, Bevel, Displace, Deform)."""
        from mimicry_engine.primitives import sdf_box
        from mimicry_engine.modifiers import (
            ModifierStack, ArrayModifier, MirrorModifier, BevelModifier,
            DisplaceModifier, DeformModifier, SolidifyModifier
        )
        stack = ModifierStack(lambda pt: sdf_box(pt, (0.5, 0.5, 0.5)))
        stack.add_modifier(ArrayModifier("Array_2", count=2, offset=(1.2, 0, 0)))
        stack.add_modifier(MirrorModifier("Mirror_X", use_x=True))
        stack.add_modifier(BevelModifier("Bevel_Edge", radius=0.05))
        stack.add_modifier(DisplaceModifier("Displace_Noise", strength=0.02, frequency=4.0))
        stack.add_modifier(DeformModifier("Twist_Y", deform_type="TWIST", factor=0.2))

        val_center = stack.evaluate((0.0, 0.0, 0.0))
        self.assertIsInstance(val_center, float)
        self.assertEqual(len(stack.modifiers), 5)
        serialized = stack.to_dict()
        self.assertEqual(len(serialized), 5)

    def test_composite_recipes(self):
        """Verifies 13 Pre-Assembled Real-World Mimicry Recipes & VITAL_MAX_HP == 6 invariant."""
        from mimicry_engine.mimic_recipes import list_recipes, get_recipe
        recipes = list_recipes()
        self.assertEqual(len(recipes), 13, f"Expected 13 recipes, got {len(recipes)}")

        for r in recipes:
            obj = get_recipe(r["id"])
            self.assertIsNotNone(obj, f"Recipe '{r['id']}' could not be loaded")
            self.assertGreaterEqual(
                len(obj.parts), 3,
                f"Recipe '{obj.name}' has fewer than 3 parts: {len(obj.parts)}"
            )
            # Critical invariant check
            self.assertEqual(
                obj.vital_max_hp, 6,
                f"Invariant VITAL_MAX_HP violation on {obj.object_id}: {obj.vital_max_hp}"
            )
            # Raymarch small preview
            ascii_frame = obj.render_ascii_projection(width=40, height=8, t=0.2)
            lines = ascii_frame.split('\n')
            self.assertEqual(len(lines), 8, f"Expected 8 lines for recipe {obj.name}, got {len(lines)}")

    def test_scene_outlines_and_godot_export(self):
        """Verifies 9 Game Scene Outlines and Godot 4 .tscn generator."""
        from mimicry_engine.scene_composer import list_scenes, get_scene
        scenes = list_scenes()
        self.assertEqual(len(scenes), 9, f"Expected 9 scenes, got {len(scenes)}")

        for sc_info in scenes:
            scene = get_scene(sc_info["id"])
            self.assertIsNotNone(scene, f"Scene '{sc_info['id']}' could not be loaded")
            self.assertGreaterEqual(
                len(scene.actors), 3,
                f"Scene '{scene.name}' has fewer than 3 actors: {len(scene.actors)}"
            )
            for actor in scene.actors:
                self.assertLessEqual(
                    actor.vital_hp, 6,
                    f"Actor vital_hp violation in {scene.name}: {actor.vital_hp}"
                )
            tscn_text = scene.export_godot_tscn()
            self.assertIn("[node name=", tscn_text)
            self.assertIn("WorldEnvironment", tscn_text)
            self.assertIn("metadata/vital_max_hp = 6", tscn_text)
            for actor in scene.actors:
                self.assertIn(actor.actor_id, tscn_text)

            ascii_view = scene.render_ascii_view(width=40, height=8, t=0.5)
            self.assertEqual(len(ascii_view.split('\n')), 8)

    def test_urban_spatial_composition_rules(self):
        """Verifies 8 Strict Urban Spatial Composition Rules on real-world scenes."""
        from mimicry_engine.scene_composer import get_scene, UrbanSpatialCompositionRules
        real_world_scene_ids = [
            "SCENE_OLD_TOWN_PRAGUE_SQUARE",
            "SCENE_PARISIAN_HAUSSMANN_BOULEVARD",
            "SCENE_MEDITERRANEAN_COASTAL_PORT",
            "SCENE_ALPINE_TIMBER_TOWNSHIP",
            "SCENE_INDUSTRIAL_CANAL_WATERFRONT"
        ]
        for sid in real_world_scene_ids:
            sc = get_scene(sid)
            self.assertIsNotNone(sc, f"Scene '{sid}' not found")
            val_result = UrbanSpatialCompositionRules.validate_scene(sc)
            self.assertTrue(
                val_result.passed,
                f"Scene {sid} failed rules: {val_result.failed_rules}"
            )
            self.assertGreaterEqual(
                val_result.compliance_score, 0.85,
                f"Scene {sid} score too low: {val_result.compliance_score}"
            )

    def test_live_hub_endpoints(self):
        """Verifies Live Localhost Hub HTTP Endpoints (skipped if hub is offline)."""
        base_url = "http://127.0.0.1:8080"
        try:
            req = urllib.request.Request(f"{base_url}/api/health")
            with urllib.request.urlopen(req, timeout=1.5) as resp:
                health_data = json.loads(resp.read().decode("utf-8"))
                self.assertEqual(health_data.get("status"), "HEALTHY")
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            self.skipTest(f"Localhost hub at {base_url} is offline or unreachable: {e}")

        # If hub is reachable, test remaining endpoints
        # 1. Mimicry Objects List
        req = urllib.request.Request(f"{base_url}/api/mimicry/objects")
        with urllib.request.urlopen(req, timeout=3) as resp:
            objs_data = json.loads(resp.read().decode("utf-8"))
            self.assertEqual(objs_data.get("status"), "OK")
            self.assertEqual(len(objs_data.get("recipes", [])), 13)

        # 2. Mimicry Game Scenes List
        req = urllib.request.Request(f"{base_url}/api/mimicry/scenes")
        with urllib.request.urlopen(req, timeout=3) as resp:
            scs_data = json.loads(resp.read().decode("utf-8"))
            self.assertEqual(scs_data.get("status"), "OK")
            self.assertEqual(len(scs_data.get("scenes", [])), 9)

        # 3. POST select object
        select_obj_payload = json.dumps({"type": "object", "id": "HISTORIC_TENEMENT_FACADE"}).encode("utf-8")
        req = urllib.request.Request(f"{base_url}/api/mimicry/select", data=select_obj_payload, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=3) as resp:
            sel_resp = json.loads(resp.read().decode("utf-8"))
            self.assertEqual(sel_resp.get("status"), "SUCCESS")
            self.assertEqual(sel_resp["object"]["object_id"], "HISTORIC_TENEMENT_FACADE")
            self.assertEqual(sel_resp["object"]["vital_max_hp"], 6)

        # 4. POST select scene
        select_sc_payload = json.dumps({"type": "scene", "id": "SCENE_OLD_TOWN_PRAGUE_SQUARE"}).encode("utf-8")
        req = urllib.request.Request(f"{base_url}/api/mimicry/select", data=select_sc_payload, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=3) as resp:
            sel_sc_resp = json.loads(resp.read().decode("utf-8"))
            self.assertEqual(sel_sc_resp.get("status"), "SUCCESS")
            self.assertEqual(sel_sc_resp["scene"]["scene_id"], "SCENE_OLD_TOWN_PRAGUE_SQUARE")

        # 5. GET validate-active
        val_req = urllib.request.Request(f"{base_url}/api/mimicry/validate-active")
        with urllib.request.urlopen(val_req, timeout=3) as resp:
            val_api_resp = json.loads(resp.read().decode("utf-8"))
            self.assertEqual(val_api_resp.get("status"), "OK")
            self.assertTrue(val_api_resp["validation"]["passed"])

        # 6. POST paint-real-world
        paint_payload = json.dumps({"archetype": "haussmann_boulevard", "seed": 77}).encode("utf-8")
        paint_req = urllib.request.Request(f"{base_url}/api/mimicry/paint-real-world", data=paint_payload, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(paint_req, timeout=3) as resp:
            paint_resp = json.loads(resp.read().decode("utf-8"))
            self.assertEqual(paint_resp.get("status"), "SUCCESS")
            self.assertTrue(paint_resp["validation"]["passed"])

        # 7. POST export-godot
        export_req = urllib.request.Request(f"{base_url}/api/mimicry/export-godot", data=b"{}", headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(export_req, timeout=3) as resp:
            exp_resp = json.loads(resp.read().decode("utf-8"))
            self.assertEqual(exp_resp.get("status"), "SUCCESS")
            exported_file = os.path.join(ROOT_DIR, exp_resp["exported_file"])
            self.assertTrue(os.path.exists(exported_file), f"Exported file not found: {exported_file}")

        # 8. GET status
        status_req = urllib.request.Request(f"{base_url}/api/status")
        with urllib.request.urlopen(status_req, timeout=3) as resp:
            st_resp = json.loads(resp.read().decode("utf-8"))
            self.assertEqual(st_resp.get("mode"), "GAME_SCENE")


if __name__ == "__main__":
    unittest.main()
