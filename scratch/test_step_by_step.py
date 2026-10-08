import sys
import os
import time

WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

log_path = os.path.join(WORKSPACE_ROOT, "scratch", "step_results.txt")

def log(msg):
    print(msg, flush=True)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(msg + "\n")
        f.flush()

# Reset log
with open(log_path, "w", encoding="utf-8") as f:
    f.write("=== RUNNING MIMICRY COMPOSITOR STEP-BY-STEP ===\n")

try:
    log("\n[STEP 1] Primitives & Smooth CSG...")
    t0 = time.time()
    from mimicry_engine.primitives import (
        sdf_box, sdf_sphere, sdf_cylinder, sdf_torus, sdf_hex_prism,
        smin, smax, smooth_difference
    )
    p = (0.0, 0.0, 0.0)
    assert sdf_sphere(p, r=1.0) == -1.0
    assert sdf_box(p, b=(0.5, 0.5, 0.5)) == -0.5
    assert sdf_cylinder(p, r=0.5, h=1.0) == -0.5
    assert sdf_hex_prism(p, h=0.8, r=0.8) <= 0.0
    s_val = smin(0.5, 0.5, k=0.2)
    assert abs(s_val - 0.45) < 1e-4
    log(f"  -> STEP 1 OK ({time.time() - t0:.3f}s)")

    log("\n[STEP 2] Blender Modifier Stack...")
    t0 = time.time()
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
    assert len(stack.modifiers) == 5
    log(f"  -> STEP 2 OK ({time.time() - t0:.3f}s)")

    log("\n[STEP 3] 13 Mimicry Recipes & HP=6 Invariant...")
    t0 = time.time()
    from mimicry_engine.mimic_recipes import list_recipes, get_recipe
    recipes = list_recipes()
    log(f"  Found {len(recipes)} recipes:")
    assert len(recipes) == 13, f"Expected 13 recipes, got {len(recipes)}"
    for r in recipes:
        t_sub = time.time()
        obj = get_recipe(r["id"])
        assert obj is not None
        assert len(obj.parts) >= 3
        assert obj.vital_max_hp == 6
        # Small raymarch preview
        ascii_frame = obj.render_ascii_projection(width=20, height=5, t=0.1)
        log(f"    - {obj.object_id} ({obj.name}): {len(obj.parts)} parts, HP={obj.vital_max_hp} ({time.time() - t_sub:.3f}s)")
    log(f"  -> STEP 3 OK ({time.time() - t0:.3f}s)")

    log("\n[STEP 4] 9 Game Scenes & Godot .tscn Export...")
    t0 = time.time()
    from mimicry_engine.scene_composer import list_scenes, get_scene
    scenes = list_scenes()
    log(f"  Found {len(scenes)} scenes:")
    assert len(scenes) == 9, f"Expected 9 scenes, got {len(scenes)}"
    for sc_info in scenes:
        t_sub = time.time()
        scene = get_scene(sc_info["id"])
        assert scene is not None
        assert len(scene.actors) >= 3
        for actor in scene.actors:
            assert actor.vital_hp <= 6
        tscn_text = scene.export_godot_tscn()
        assert "[node name=" in tscn_text
        assert "WorldEnvironment" in tscn_text
        assert "metadata/vital_max_hp = 6" in tscn_text
        ascii_view = scene.render_ascii_view(width=20, height=5, t=0.1)
        log(f"    - {scene.scene_id} ({scene.name}): {len(scene.actors)} actors, {len(tscn_text)} chars .tscn ({time.time() - t_sub:.3f}s)")
    log(f"  -> STEP 4 OK ({time.time() - t0:.3f}s)")

    log("\n[STEP 5] 8 Strict Urban Spatial Composition Rules Engine...")
    t0 = time.time()
    from mimicry_engine.scene_composer import UrbanSpatialCompositionRules
    real_world_scene_ids = [
        "SCENE_OLD_TOWN_PRAGUE_SQUARE",
        "SCENE_PARISIAN_HAUSSMANN_BOULEVARD",
        "SCENE_MEDITERRANEAN_COASTAL_PORT",
        "SCENE_ALPINE_TIMBER_TOWNSHIP",
        "SCENE_INDUSTRIAL_CANAL_WATERFRONT"
    ]
    for sid in real_world_scene_ids:
        sc = get_scene(sid)
        assert sc is not None
        val_result = UrbanSpatialCompositionRules.validate_scene(sc)
        assert val_result.passed is True, f"Failed rules: {val_result.failed_rules}"
        assert val_result.compliance_score >= 0.85
        log(f"    - {sc.name}: PASSED (Score: {val_result.compliance_score*100:.1f}%, 0 failed rules)")
    log(f"  -> STEP 5 OK ({time.time() - t0:.3f}s)")

    log("\n[STEP 6] Deterministic Seed Synthesizer paint_real_world_scene...")
    t0 = time.time()
    from mimicry_engine.scene_composer import paint_real_world_scene
    for seed in [1, 42, 999]:
        sc_seeded = paint_real_world_scene("haussmann_boulevard", seed=seed)
        assert sc_seeded.seed == seed
        res = UrbanSpatialCompositionRules.validate_scene(sc_seeded)
        assert res.passed is True
        log(f"    - Seed {seed}: Generated '{sc_seeded.name}' (Score: {res.compliance_score*100:.1f}%)")
    log(f"  -> STEP 6 OK ({time.time() - t0:.3f}s)")

    log("\n=== ALL TEST STEPS PASSED SUCCESSFULLY ===")
except Exception as e:
    import traceback
    log(f"\n[ERROR] Exception during execution: {e}")
    log(traceback.format_exc())
