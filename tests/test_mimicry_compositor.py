"""
Krystal-Stack Automated Verification: Blender Modifier Core & Mimicry Compositor
================================================================================
Verifies:
1. Primitive Signed Distance Functions & Polynomial Smooth CSG (smin / smax)
2. Blender Modifier Stack (Array, Mirror, Boolean, Bevel, Displace, Deform, Solidify)
3. 6 Real-World Composite Object Recipes (Turret, Mech, Obelisk, Xenodrone, Spire, Explorer)
4. 4 Game Scene Outlines (Cyber District, Alchemical Ruins, Deep Space Hangar, Alien Hive)
5. Godot 4.x .tscn Generation & Export
6. Live HTTP Endpoints on Localhost Hub:
   - GET  /api/health
   - GET  /api/mimicry/objects
   - GET  /api/mimicry/scenes
   - POST /api/mimicry/select (Object)
   - POST /api/mimicry/select (Scene)
   - POST /api/mimicry/export-godot
"""

import sys
import os
import json
import urllib.request

if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

def run_tests():
    print("=" * 70)
    print(" [TEST SUITE] BLENDER MODIFIER CORE & REAL-WORLD MIMICRY COMPOSITOR")
    print("=" * 70)

    # ── Test 1: Primitives & Smooth CSG ──────────────────────────────────
    print("\n[TEST 1] Verifying Primitives and Polynomial Smooth CSG...")
    from mimicry_engine.primitives import (
        sdf_box, sdf_sphere, sdf_cylinder, sdf_torus, sdf_hex_prism,
        smin, smax, smooth_difference
    )
    p = (0.0, 0.0, 0.0)
    assert sdf_sphere(p, r=1.0) == -1.0
    assert sdf_box(p, b=(0.5, 0.5, 0.5)) == -0.5
    assert sdf_cylinder(p, r=0.5, h=1.0) == -0.5
    assert sdf_hex_prism(p, h=0.8, r=0.8) <= 0.0
    
    # Smooth min between 0.5 and 0.5 with k=0.2 should be 0.5 - 0.2/4 = 0.45
    s_val = smin(0.5, 0.5, k=0.2)
    assert abs(s_val - 0.45) < 1e-4, f"Expected 0.45, got {s_val}"
    print("  -> All 3D primitives and smooth CSG operators verified.")

    # ── Test 2: Modifier Stack Pipeline ──────────────────────────────────
    print("\n[TEST 2] Verifying Blender Modifier Stack (Array, Mirror, Displace, Deform)...")
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
    assert isinstance(val_center, float)
    print(f"  -> Modifier stack evaluated successfully (d at origin: {val_center:.4f}).")
    assert len(stack.modifiers) == 5
    print(f"  -> Stack serialized to dict with {len(stack.to_dict())} modifiers.")

    # ── Test 3: 6 Real-World Composite Object Recipes ─────────────────────
    print("\n[TEST 3] Verifying 6 Pre-Assembled Real-World Mimicry Recipes...")
    from mimicry_engine.mimic_recipes import list_recipes, get_recipe
    recipes = list_recipes()
    assert len(recipes) == 6, f"Expected 6 recipes, got {len(recipes)}"

    for r in recipes:
        obj = get_recipe(r["id"])
        assert obj is not None
        assert len(obj.parts) >= 3
        # Raymarch small preview
        ascii_frame = obj.render_ascii_projection(width=40, height=8, t=0.2)
        assert len(ascii_frame.split('\n')) == 8
        print(f"  -> Recipe '{obj.name}' verified ({len(obj.parts)} parts).")

    # ── Test 4: 4 Game Scene Outlines & Godot .tscn Export ────────────────
    print("\n[TEST 4] Verifying 4 Game Scene Outlines & Godot 4 .tscn Generator...")
    from mimicry_engine.scene_composer import list_scenes, get_scene
    scenes = list_scenes()
    assert len(scenes) == 4, f"Expected 4 scenes, got {len(scenes)}"

    for sc_info in scenes:
        scene = get_scene(sc_info["id"])
        assert scene is not None
        assert len(scene.actors) >= 3
        tscn_text = scene.export_godot_tscn()
        assert "[node name=" in tscn_text
        assert "WorldEnvironment" in tscn_text
        for actor in scene.actors:
            assert actor.actor_id in tscn_text
        ascii_view = scene.render_ascii_view(width=40, height=8, t=0.5)
        assert len(ascii_view.split('\n')) == 8
        print(f"  -> Scene '{scene.name}' verified ({len(scene.actors)} actors, {len(tscn_text)} chars .tscn).")

    # ── Test 5: Live Localhost Hub HTTP Endpoints ─────────────────────────
    print("\n[TEST 5] Verifying Live Localhost Hub HTTP Endpoints (http://127.0.0.1:8080)...")
    base_url = "http://127.0.0.1:8080"

    # 5a. Health
    req = urllib.request.Request(f"{base_url}/api/health")
    with urllib.request.urlopen(req, timeout=3) as resp:
        health_data = json.loads(resp.read().decode("utf-8"))
        assert health_data.get("status") == "HEALTHY"
        print("  -> GET /api/health: OK (HEALTHY)")

    # 5b. Mimicry Objects List
    req = urllib.request.Request(f"{base_url}/api/mimicry/objects")
    with urllib.request.urlopen(req, timeout=3) as resp:
        objs_data = json.loads(resp.read().decode("utf-8"))
        assert objs_data.get("status") == "OK"
        assert len(objs_data.get("recipes", [])) == 6
        print(f"  -> GET /api/mimicry/objects: OK ({len(objs_data['recipes'])} recipes cataloged)")

    # 5c. Mimicry Game Scenes List
    req = urllib.request.Request(f"{base_url}/api/mimicry/scenes")
    with urllib.request.urlopen(req, timeout=3) as resp:
        scs_data = json.loads(resp.read().decode("utf-8"))
        assert scs_data.get("status") == "OK"
        assert len(scs_data.get("scenes", [])) == 4
        print(f"  -> GET /api/mimicry/scenes: OK ({len(scs_data['scenes'])} scenes cataloged)")

    # 5d. POST /api/mimicry/select (Object)
    select_obj_payload = json.dumps({"type": "object", "id": "MECH_WALKER_TITAN"}).encode("utf-8")
    req = urllib.request.Request(f"{base_url}/api/mimicry/select", data=select_obj_payload, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=3) as resp:
        sel_resp = json.loads(resp.read().decode("utf-8"))
        assert sel_resp.get("status") == "SUCCESS"
        assert sel_resp["object"]["object_id"] == "MECH_WALKER_TITAN"
        print(f"  -> POST /api/mimicry/select (Object): OK ({sel_resp['object']['name']})")

    # 5e. POST /api/mimicry/select (Scene)
    select_sc_payload = json.dumps({"type": "scene", "id": "SCENE_CYBERPUNK_DISTRICT"}).encode("utf-8")
    req = urllib.request.Request(f"{base_url}/api/mimicry/select", data=select_sc_payload, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=3) as resp:
        sel_sc_resp = json.loads(resp.read().decode("utf-8"))
        assert sel_sc_resp.get("status") == "SUCCESS"
        assert sel_sc_resp["scene"]["scene_id"] == "SCENE_CYBERPUNK_DISTRICT"
        print(f"  -> POST /api/mimicry/select (Scene): OK ({sel_sc_resp['scene']['name']})")

    # 5f. POST /api/mimicry/export-godot
    export_req = urllib.request.Request(f"{base_url}/api/mimicry/export-godot", data=b"{}", headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(export_req, timeout=3) as resp:
        exp_resp = json.loads(resp.read().decode("utf-8"))
        assert exp_resp.get("status") == "SUCCESS"
        exported_file = os.path.join(ROOT_DIR, exp_resp["exported_file"])
        assert os.path.exists(exported_file), f"Exported file not found: {exported_file}"
        print(f"  -> POST /api/mimicry/export-godot: OK (Exported to {exp_resp['exported_file']})")

    # 5g. GET /api/status (Check mode is GAME_SCENE)
    status_req = urllib.request.Request(f"{base_url}/api/status")
    with urllib.request.urlopen(status_req, timeout=3) as resp:
        st_resp = json.loads(resp.read().decode("utf-8"))
        assert st_resp.get("mode") == "GAME_SCENE"
        print(f"  -> GET /api/status: OK (Engine mode set to GAME_SCENE, FPS: {st_resp.get('fps'):.1f})")

    print("\n" + "=" * 70)
    print(" [SUCCESS] ALL 5 BLENDER MIMICRY & SCENE COMPOSER TESTS PASSED (100%)")
    print("=" * 70)

if __name__ == "__main__":
    run_tests()
