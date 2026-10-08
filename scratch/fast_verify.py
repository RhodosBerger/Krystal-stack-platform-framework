import sys
import os
import time

WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

out_path = os.path.join(WORKSPACE_ROOT, "scratch", "fast_verify_out.txt")

with open(out_path, "w", encoding="utf-8") as f:
    f.write("=== FAST VERIFY INITIALIZED ===\n")

def record(msg):
    print(msg, flush=True)
    with open(out_path, "a", encoding="utf-8") as f:
        f.write(msg + "\n")
        f.flush()

def main():
    record("=" * 72)
    record(" [KRYSTAL-STACK] FAST OBSERVABLE VERIFICATION: URBAN COMPOSITOR RULES")
    record("=" * 72)

    # 1. Primitives & Smooth CSG
    t0 = time.time()
    from mimicry_engine.primitives import (
        sdf_box, sdf_sphere, sdf_cylinder, sdf_capped_cone, sdf_hex_prism,
        smin, smax, smooth_difference
    )
    p = (0.0, 0.0, 0.0)
    assert sdf_sphere(p, r=1.0) == -1.0
    assert sdf_box(p, b=(0.5, 0.5, 0.5)) == -0.5
    assert sdf_cylinder(p, r=0.5, h=1.0) == -0.5
    assert sdf_hex_prism(p, h=0.8, r=0.8) <= 0.0
    assert abs(smin(0.5, 0.5, k=0.2) - 0.45) < 1e-4
    record(f"[VERIFY 1] Primitives & Polynomial Smooth CSG: OK ({time.time()-t0:.4f}s)")

    # 2. Modifier Stack
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
    val = stack.evaluate((0.0, 0.0, 0.0))
    assert isinstance(val, float)
    assert len(stack.modifiers) == 5
    record(f"[VERIFY 2] Blender Modifier Stack Pipeline: OK ({time.time()-t0:.4f}s)")

    # 3. 13 Mimicry Recipes & HP=6 Invariant
    t0 = time.time()
    from mimicry_engine.mimic_recipes import list_recipes, get_recipe
    recipes = list_recipes()
    record(f"\n[VERIFY 3] Mimicry Recipes Catalog (Count: {len(recipes)}):")
    assert len(recipes) == 13, f"Expected 13 recipes, got {len(recipes)}"
    for r in recipes:
        obj = get_recipe(r["id"])
        assert obj is not None
        assert len(obj.parts) >= 3
        assert obj.vital_max_hp == 6
        record(f"  * {obj.object_id:30s} | {obj.category:20s} | Parts: {len(obj.parts)} | Vital Max HP: {obj.vital_max_hp}")
    record(f"-> All 13 Recipes Verified with VITAL_MAX_HP=6 ({time.time()-t0:.4f}s)")

    # 4. 9 Game Scenes & Godot .tscn Export
    t0 = time.time()
    from mimicry_engine.scene_composer import list_scenes, get_scene
    scenes = list_scenes()
    record(f"\n[VERIFY 4] Game Scenes Catalog (Count: {len(scenes)}):")
    assert len(scenes) == 9, f"Expected 9 scenes, got {len(scenes)}"
    for sc_info in scenes:
        sc = get_scene(sc_info["id"])
        assert sc is not None
        assert len(sc.actors) >= 3
        for a in sc.actors:
            assert a.vital_hp <= 6
        tscn = sc.export_godot_tscn()
        assert "[node name=" in tscn
        assert "WorldEnvironment" in tscn
        assert "metadata/vital_max_hp = 6" in tscn
        record(f"  * {sc.scene_id:35s} | Genre: {sc.genre:25s} | Actors: {len(sc.actors)} | .tscn chars: {len(tscn)}")
    record(f"-> All 9 Game Scenes Verified with Godot 4 .tscn Export ({time.time()-t0:.4f}s)")

    # 5. Strict Urban Spatial Composition Rules Engine
    t0 = time.time()
    from mimicry_engine.scene_composer import UrbanSpatialCompositionRules
    real_world_scene_ids = [
        "SCENE_OLD_TOWN_PRAGUE_SQUARE",
        "SCENE_PARISIAN_HAUSSMANN_BOULEVARD",
        "SCENE_MEDITERRANEAN_COASTAL_PORT",
        "SCENE_ALPINE_TIMBER_TOWNSHIP",
        "SCENE_INDUSTRIAL_CANAL_WATERFRONT"
    ]
    record("\n[VERIFY 5] 8 Strict Urban Spatial Composition Rules Validation:")
    for sid in real_world_scene_ids:
        sc = get_scene(sid)
        res = UrbanSpatialCompositionRules.validate_scene(sc)
        record(f"  * {sc.name:45s} | Passed: {str(res.passed):5s} | Score: {res.compliance_score*100:5.1f}% | Failed: {res.failed_rules}")
        record(f"    - Macro/Meso/Micro Actors : {res.metrics.get('macro_count')} macro, {res.metrics.get('meso_count')} meso, {res.metrics.get('micro_count')} micro")
        record(f"    - Enclosure Ratio (D/H)   : {res.metrics.get('enclosure_ratio'):.3f} (target phi=1.618)")
        record(f"    - Vista Offset X          : {res.metrics.get('vista_anchor_offset_x'):.3f} m")
        record(f"    - Max Foundation Elevation: {res.metrics.get('max_foundation_elevation'):.3f} m")
        assert res.passed is True
        assert res.compliance_score >= 0.85
    record(f"-> All 5 Real-World Scenes Passed Strict Composition Rules ({time.time()-t0:.4f}s)")

    # 6. Seed Determinism & Procedural Synthesis
    t0 = time.time()
    from mimicry_engine.scene_composer import paint_real_world_scene
    sc_seed42_a = paint_real_world_scene("haussmann_boulevard", seed=42)
    sc_seed42_b = paint_real_world_scene("haussmann_boulevard", seed=42)
    sc_seed99   = paint_real_world_scene("haussmann_boulevard", seed=99)

    assert sc_seed42_a.actors[0].position == sc_seed42_b.actors[0].position
    assert sc_seed42_a.actors[0].position != sc_seed99.actors[0].position
    record(f"\n[VERIFY 6] Deterministic Seeded Reproducibility Verified: OK ({time.time()-t0:.4f}s)")

    # 7. Visual ASCII Render of Historic Tenement & Old Town Prague Square (fast preview)
    record("\n" + "=" * 72)
    record(" [VISUAL PROJECTION] HISTORIC_TENEMENT_FACADE (Ascii Raymarch)")
    record("=" * 72)
    facade = get_recipe("HISTORIC_TENEMENT_FACADE")
    ascii_facade = facade.render_ascii_projection(width=50, height=10, t=0.0)
    record(ascii_facade)

    record("\n" + "=" * 72)
    record(" [VISUAL PROJECTION] SCENE_OLD_TOWN_PRAGUE_SQUARE (Ascii Raymarch)")
    record("=" * 72)
    prague_sq = get_scene("SCENE_OLD_TOWN_PRAGUE_SQUARE")
    ascii_sq = prague_sq.render_ascii_view(width=50, height=10, t=0.0)
    record(ascii_sq)

    record("\n" + "=" * 72)
    record(" [SUCCESS] ALL REAL-WORLD URBAN COMPOSITOR RULES EXPLICITLY VERIFIED (100%)")
    record("=" * 72)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        record(f"\n[FATAL ERROR] {e}")
        record(traceback.format_exc())
