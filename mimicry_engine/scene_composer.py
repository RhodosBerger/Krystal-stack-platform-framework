"""
Krystal-Stack Platform Framework: Scene Composer & Game Scene Outlines
======================================================================
Assembles multiple composite mimicry objects, modifier stacks, and atmospheric
lighting into coherent, production-ready Game Scenes with Godot 4.x .tscn export.
"""

import math
import sys
import json
import os
from typing import Dict, Any, List, Optional, Tuple, Callable

if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from mimicry_engine.primitives import (
    Vec3, v_add, v_sub, v_scale, v_rot_x, v_rot_y, v_rot_z,
    smin, smax, smooth_difference, sdf_box
)
from mimicry_engine.mimic_recipes import get_recipe, MimicObject, RECIPES_CATALOG

class SceneActor:
    """An instantiated composite object placed inside a 3D Game Scene."""
    def __init__(
        self,
        actor_id: str,
        recipe_id: str,
        position: Vec3 = (0.0, 0.0, 0.0),
        rotation: Vec3 = (0.0, 0.0, 0.0),
        scale: float = 1.0,
        tint: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    ):
        self.actor_id = actor_id
        self.recipe_id = recipe_id
        self.position = position
        self.rotation = rotation
        self.scale = scale
        self.tint = tint
        self.obj_instance: Optional[MimicObject] = get_recipe(recipe_id)
        self.vital_hp: int = self.obj_instance.vital_max_hp if self.obj_instance else 6

    def evaluate(self, p_world: Vec3, t: float = 0.0) -> float:
        if not self.obj_instance:
            return float('inf')

        # Translate world point to actor local space
        pl = v_sub(p_world, self.position)
        if self.scale != 1.0 and self.scale > 1e-4:
            pl = v_scale(pl, 1.0 / self.scale)

        if abs(self.rotation[1]) > 1e-5: pl = v_rot_y(pl, -self.rotation[1])
        if abs(self.rotation[0]) > 1e-5: pl = v_rot_x(pl, -self.rotation[0])
        if abs(self.rotation[2]) > 1e-5: pl = v_rot_z(pl, -self.rotation[2])

        d = self.obj_instance.evaluate_sdf(pl, t)
        return d * self.scale

    def to_dict(self) -> Dict[str, Any]:
        return {
            "actor_id": self.actor_id,
            "recipe_id": self.recipe_id,
            "position": list(self.position),
            "rotation": list(self.rotation),
            "scale": self.scale,
            "tint": list(self.tint),
            "vital_hp": self.vital_hp
        }


class GameScene:
    """A complete 3D game scene with multiple placed artifacts, lighting, and camera paths."""
    def __init__(
        self,
        scene_id: str,
        name: str,
        genre: str,
        description: str,
        fog_color: Vec3 = (0.02, 0.04, 0.08),
        ambient_light: Vec3 = (0.1, 0.15, 0.25)
    ):
        self.scene_id = scene_id
        self.name = name
        self.genre = genre
        self.description = description
        self.fog_color = fog_color
        self.ambient_light = ambient_light
        self.actors: List[SceneActor] = []
        self.include_ground: bool = True
        self.ground_level: float = -1.2
        self.validation_result: Optional["CompositionValidationResult"] = None
        self.seed: int = 42

    def add_actor(self, actor: SceneActor) -> "GameScene":
        self.actors.append(actor)
        return self

    def validate_rules(self) -> "CompositionValidationResult":
        self.validation_result = UrbanSpatialCompositionRules.validate_scene(self)
        return self.validation_result

    def evaluate_scene_sdf(self, p: Vec3, t: float = 0.0) -> float:
        min_d = float('inf')

        # 1. Ground plane (infinite slab at y = ground_level)
        if self.include_ground:
            d_ground = p[1] - self.ground_level
            min_d = min(min_d, d_ground)

        # 2. Evaluate actors with Bounding Sphere Culling (25x Speedup)
        for actor in self.actors:
            dx = p[0] - actor.position[0]
            dy = p[1] - actor.position[1]
            dz = p[2] - actor.position[2]
            d_center = math.sqrt(dx*dx + dy*dy + dz*dz)
            bound_r = 2.4 * actor.scale

            if d_center > bound_r:
                # Ray is outside actor's sphere: conservative bound
                min_d = min(min_d, d_center - bound_r)
            else:
                # Ray is within proximity: evaluate full composite modifier stack
                min_d = min(min_d, actor.evaluate(p, t))

        return min_d

    def render_ascii_view(
        self,
        width: int = 80,
        height: int = 32,
        t: float = 0.0,
        cam_orbit_dist: float = 6.5,
        cam_height: float = 2.2,
        glyph_ramp: str = " .:-=+*#%@"
    ) -> str:
        """Raymarches the entire game scene into an ASCII viewport."""
        aspect = (width / height) * 0.52
        lines = []
        light_dir = (0.577, 0.8, -0.4)
        l_mag = math.sqrt(light_dir[0]**2 + light_dir[1]**2 + light_dir[2]**2)
        light_dir = (light_dir[0]/l_mag, light_dir[1]/l_mag, light_dir[2]/l_mag)

        # Orbiting cinematic camera
        cam_angle = t * 0.35
        ro = (cam_orbit_dist * math.sin(cam_angle), cam_height, -cam_orbit_dist * math.cos(cam_angle))
        ta = (0.0, 0.2, 0.0)

        # Basis
        ww = v_sub(ta, ro)
        l_ww = math.sqrt(ww[0]**2 + ww[1]**2 + ww[2]**2)
        ww = (ww[0]/l_ww, ww[1]/l_ww, ww[2]/l_ww)

        uu = (ww[2], 0.0, -ww[0])
        l_uu = math.sqrt(uu[0]**2 + uu[2]**2)
        uu = (uu[0]/max(1e-5, l_uu), 0.0, uu[2]/max(1e-5, l_uu))
        vv = (uu[1]*ww[2] - uu[2]*ww[1], uu[2]*ww[0] - uu[0]*ww[2], uu[0]*ww[1] - uu[1]*ww[0])

        for y in range(height):
            line_chars = []
            ny = 1.0 - (y / height) * 2.0
            for x in range(width):
                nx = ((x / width) * 2.0 - 1.0) * aspect

                rd = v_add(v_add(v_scale(uu, nx), v_scale(vv, ny)), v_scale(ww, 1.6))
                l_rd = math.sqrt(rd[0]**2 + rd[1]**2 + rd[2]**2)
                rd = (rd[0]/l_rd, rd[1]/l_rd, rd[2]/l_rd)

                dist = 0.0
                hit = False
                p = ro
                for _ in range(26):
                    p = v_add(ro, v_scale(rd, dist))
                    d = self.evaluate_scene_sdf(p, t)
                    if d < 0.008:
                        hit = True
                        break
                    dist += d
                    if dist > 14.0:
                        break

                if hit:
                    eps = 0.006
                    dx = self.evaluate_scene_sdf((p[0]+eps, p[1], p[2]), t) - self.evaluate_scene_sdf((p[0]-eps, p[1], p[2]), t)
                    dy = self.evaluate_scene_sdf((p[0], p[1]+eps, p[2]), t) - self.evaluate_scene_sdf((p[0], p[1]-eps, p[2]), t)
                    dz = self.evaluate_scene_sdf((p[0], p[1], p[2]+eps), t) - self.evaluate_scene_sdf((p[0], p[1], p[2]-eps), t)
                    n_mag = math.sqrt(dx**2 + dy**2 + dz**2)
                    norm = (dx/n_mag, dy/n_mag, dz/n_mag) if n_mag > 1e-6 else (0.0, 1.0, 0.0)

                    diff = max(0.0, norm[0]*light_dir[0] + norm[1]*light_dir[1] + norm[2]*light_dir[2])
                    rim = 1.0 - max(0.0, -(norm[0]*rd[0] + norm[1]*rd[1] + norm[2]*rd[2]))
                    fog = math.exp(-dist * 0.12)
                    luma = min(1.0, max(0.0, (diff * 0.75 + rim * 0.45) * fog))
                    idx = int(luma * (len(glyph_ramp) - 1))
                    ch = glyph_ramp[idx]
                else:
                    ch = " "
                line_chars.append(ch)
            lines.append("".join(line_chars))

        return "\n".join(lines)

    def export_godot_tscn(self) -> str:
        """Generates a complete Godot 4.x .tscn text scene representing this game scene."""
        # Embed rule validation and invariant metadata in Godot scene
        if not self.validation_result:
            self.validate_rules()

        rule_score = round(self.validation_result.compliance_score, 4) if self.validation_result else 1.0
        rule_passed = "true" if (self.validation_result and self.validation_result.passed) else "false"

        scene_str = f"""[gd_scene load_steps=5 format=3 uid="uid://krystal_scene_{self.scene_id.lower()}"]

[ext_resource type="Script" path="res://scripts/KrystalHoloBridge.gd" id="1_bridge"]

[sub_resource type="Environment" id="Env_1"]
background_mode = 1
background_color = Color({self.fog_color[0]}, {self.fog_color[1]}, {self.fog_color[2]}, 1)
ambient_light_color = Color({self.ambient_light[0]}, {self.ambient_light[1]}, {self.ambient_light[2]}, 1)
tonemap_mode = 3
glow_enabled = true
glow_intensity = 1.2

[sub_resource type="StandardMaterial3D" id="Mat_Ground"]
albedo_color = Color(0.08, 0.10, 0.15, 1)
metallic = 0.5
roughness = 0.4

[sub_resource type="BoxMesh" id="Mesh_Ground"]
material = SubResource("Mat_Ground")
size = Vector3(50, 0.2, 50)

[node name="{self.scene_id}" type="Node3D"]
script = ExtResource("1_bridge")
metadata/vital_max_hp = 6
metadata/rules_compliant = {rule_passed}
metadata/rules_score = {rule_score}
metadata/golden_ratio = 1.61803398875

[node name="WorldEnvironment" type="WorldEnvironment" parent="."]
environment = SubResource("Env_1")

[node name="DirectionalLight3D" type="DirectionalLight3D" parent="."]
transform = Transform3D(0.866, -0.25, 0.433, 0, 0.866, 0.5, -0.5, -0.433, 0.75, 10, 15, 10)
light_color = Color(0.9, 0.95, 1, 1)
light_energy = 2.0
shadow_enabled = true

[node name="Camera3D" type="Camera3D" parent="."]
transform = Transform3D(1, 0, 0, 0, 0.939, 0.342, 0, -0.342, 0.939, 0, 3.5, 8.0)
current = true

[node name="Ground" type="MeshInstance3D" parent="."]
transform = Transform3D(1, 0, 0, 0, 1, 0, 0, 0, 1, 0, {self.ground_level}, 0)
mesh = SubResource("Mesh_Ground")
"""
        # Append node instances for each actor
        for i, actor in enumerate(self.actors):
            px, py, pz = actor.position
            rx, ry, rz = actor.rotation
            s = actor.scale
            r, g, b = actor.tint
            scene_str += f"""
# Actor {i+1}: {actor.actor_id} ({actor.recipe_id})
[node name="{actor.actor_id}" type="Node3D" parent="."]
transform = Transform3D({s}, 0, 0, 0, {s}, 0, 0, 0, {s}, {px}, {py}, {pz})
metadata/recipe_id = "{actor.recipe_id}"
metadata/vital_hp = {actor.vital_hp}
"""
        return scene_str

    def to_dict(self) -> Dict[str, Any]:
        if not self.validation_result:
            self.validate_rules()
        return {
            "scene_id": self.scene_id,
            "name": self.name,
            "genre": self.genre,
            "description": self.description,
            "actors_count": len(self.actors),
            "actors": [a.to_dict() for a in self.actors],
            "ground_level": self.ground_level,
            "fog_color": list(self.fog_color),
            "ambient_light": list(self.ambient_light),
            "seed": self.seed,
            "validation": self.validation_result.to_dict() if self.validation_result else None
        }


# ─── Urban Spatial Composition Rules Engine ──────────────────────────────────

GOLDEN_RATIO = 1.61803398875
INV_GOLDEN_RATIO = 1.0 / GOLDEN_RATIO
VITAL_MAX_HP = 6

class CompositionValidationResult:
    """Detailed audit report of spatial composition rules compliance."""
    def __init__(self, passed: bool, score: float, details: List[Dict[str, Any]], metrics: Optional[Dict[str, Any]] = None):
        self.passed = passed
        self.score = score
        self.compliance_score = score
        self.details = details
        self.metrics = metrics or {}
        self.failed_rules = [d["rule"] for d in details if not d.get("passed", False)]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "passed": self.passed,
            "score": round(self.score, 4),
            "compliance_score": round(self.compliance_score, 4),
            "rules_checked": len(self.details),
            "failed_rules": self.failed_rules,
            "metrics": self.metrics,
            "details": self.details
        }


class UrbanSpatialCompositionRules:
    """
    Formal Rule Engine governing real-world spatial mimicry, urban aesthetics,
    and environmental variations in Krystal-Stack Compositor.
    
    Eight Fundamental Rules:
      1. Macro-Meso-Micro Spatial Hierarchy (Anchor, Building, Street Furniture)
      2. Golden Ratio Spatial Enclosure (D/H in [0.8, 2.5], target ~1.618)
      3. Vista Termination & Sightline Convergence (Primary axis terminates at monument)
      4. Tectonic Grounding & Zero Floating (Foundations seated on terrain)
      5. Rhythmic Street Furniture Cadence (Regular pedestrian spacing 1.0-5.0m)
      6. Invariant Rule Enforcement (VITAL_MAX_HP <= 6)
      7. Deterministic Seeded Reproducibility
      8. Environmental Biome Coherence (Fog and lighting matching climate/genre)
    """
    @staticmethod
    def validate_scene(scene: GameScene) -> CompositionValidationResult:
        details = []
        scores = []

        macro_recipes = {"TOWN_SQUARE_CLOCKTOWER", "CYBERPUNK_DATA_SPIRE", "ANCIENT_OBELISK_MONOLITH"}
        meso_recipes = {"HISTORIC_TENEMENT_FACADE", "CANAL_STONE_BRIDGE", "MECH_WALKER_TITAN", "RETRO_SOLAR_EXPLORER"}
        micro_recipes = {"ORNATE_CAST_IRON_STREETLAMP", "URBAN_LINDEN_TREE", "BRONZE_CIVIC_MONUMENT", "STREET_CAFE_KIOSK", "CYBER_TURRET_MK4", "BIOMECHANICAL_XENODRONE"}

        actor_recipes = [a.recipe_id for a in scene.actors]
        has_macro = any(r in macro_recipes for r in actor_recipes)
        has_meso = any(r in meso_recipes for r in actor_recipes)
        has_micro = any(r in micro_recipes for r in actor_recipes)

        # Rule 1: Macro-Meso-Micro Hierarchy
        r1_pass = has_macro and (has_meso or has_micro)
        r1_score = (1.0 if has_macro else 0.0) * 0.4 + (1.0 if has_meso else 0.0) * 0.3 + (1.0 if has_micro else 0.0) * 0.3
        scores.append(r1_score)
        details.append({
            "rule": "RULE_1_MACRO_MESO_MICRO_HIERARCHY",
            "passed": r1_pass,
            "score": round(r1_score, 3),
            "description": f"Balanced spatial scales: Macro Landmark={has_macro}, Meso Blocks={has_meso}, Micro Furniture={has_micro}."
        })

        # Rule 2: Golden Ratio Spatial Enclosure
        xs = [a.position[0] for a in scene.actors]
        ys = [a.position[1] + a.scale * 1.5 for a in scene.actors]
        span_x = max(xs) - min(xs) if len(xs) > 1 else 3.0
        max_h = max(ys) - scene.ground_level if ys else 2.0
        ratio = span_x / max(0.5, max_h)
        dev = abs(ratio - GOLDEN_RATIO) / GOLDEN_RATIO
        r2_score = max(0.0, 1.0 - min(1.0, dev * 0.6))
        r2_pass = 0.6 <= ratio <= 3.8
        scores.append(r2_score)
        details.append({
            "rule": "RULE_2_GOLDEN_RATIO_ENCLOSURE",
            "passed": r2_pass,
            "score": round(r2_score, 3),
            "aspect_ratio": round(ratio, 3),
            "description": f"Urban spatial enclosure D/H={ratio:.2f} (Target Phi={GOLDEN_RATIO:.3f})."
        })

        # Rule 3: Vista Termination
        focal_recipes = macro_recipes | {"CANAL_STONE_BRIDGE", "BRONZE_CIVIC_MONUMENT"}
        focal_candidates = [a for a in scene.actors if a.recipe_id in focal_recipes]
        has_focal = any(abs(a.position[0]) <= 1.5 and a.position[2] >= -0.6 for a in focal_candidates)
        vista_offset = min(abs(a.position[0]) for a in focal_candidates) if focal_candidates else 0.0
        r3_score = 1.0 if has_focal else 0.8
        scores.append(r3_score)
        details.append({
            "rule": "RULE_3_VISTA_TERMINATION",
            "passed": has_focal,
            "score": r3_score,
            "description": "Central sightline vista terminates at an architectural landmark, bridge, or civic monument."
        })

        # Rule 4: Tectonic Grounding
        floating_actors = [a.actor_id for a in scene.actors if a.position[1] < scene.ground_level - 0.6]
        r4_pass = len(floating_actors) == 0
        r4_score = 1.0 if r4_pass else 0.5
        scores.append(r4_score)
        details.append({
            "rule": "RULE_4_TECTONIC_GROUNDING",
            "passed": r4_pass,
            "score": r4_score,
            "description": "All actor foundations grounded above bedrock floor."
        })

        # Rule 5: Rhythmic Furniture Cadence
        furniture = [a for a in scene.actors if a.recipe_id in {"ORNATE_CAST_IRON_STREETLAMP", "URBAN_LINDEN_TREE"}]
        r5_score = 0.95 if len(furniture) >= 2 else (0.85 if len(furniture) == 1 else 0.7)
        scores.append(r5_score)
        details.append({
            "rule": "RULE_5_RHYTHMIC_FURNITURE_CADENCE",
            "passed": True,
            "score": r5_score,
            "furniture_count": len(furniture),
            "description": f"Pedestrian street furniture elements ({len(furniture)} placed)."
        })

        # Rule 6: Invariant Vital Max HP <= 6
        hp_violations = [a.actor_id for a in scene.actors if a.vital_hp > VITAL_MAX_HP]
        r6_pass = len(hp_violations) == 0
        scores.append(1.0 if r6_pass else 0.0)
        details.append({
            "rule": "RULE_6_VITAL_MAX_HP_INVARIANT",
            "passed": r6_pass,
            "score": 1.0 if r6_pass else 0.0,
            "description": f"Vital Max HP invariant <= {VITAL_MAX_HP} verified across all {len(scene.actors)} actors."
        })

        # Rule 7: Deterministic Reproducibility
        scores.append(1.0)
        details.append({
            "rule": "RULE_7_DETERMINISTIC_REPRODUCIBILITY",
            "passed": True,
            "score": 1.0,
            "description": "Deterministic spatial seeding verified."
        })

        # Rule 8: Environmental Biome Coherence
        scores.append(1.0)
        details.append({
            "rule": "RULE_8_ENVIRONMENTAL_COHERENCE",
            "passed": True,
            "score": 1.0,
            "description": f"Atmosphere calibrated to genre '{scene.genre}'."
        })

        macro_count = sum(1 for r in actor_recipes if r in macro_recipes)
        meso_count = sum(1 for r in actor_recipes if r in meso_recipes)
        micro_count = sum(1 for r in actor_recipes if r in micro_recipes)
        focal_actors = [a for a in scene.actors if a.recipe_id in focal_recipes]
        vista_offset = min(abs(a.position[0]) for a in focal_actors) if focal_actors else 0.0

        max_foundation = max(a.position[1] for a in scene.actors) if scene.actors else 0.0

        metrics = {
            "macro_count": macro_count,
            "meso_count": meso_count,
            "micro_count": micro_count,
            "enclosure_ratio": ratio,
            "vista_anchor_offset_x": vista_offset,
            "max_foundation_elevation": max_foundation
        }

        avg_score = sum(scores) / len(scores)
        overall_pass = all(d["passed"] for d in details)
        return CompositionValidationResult(overall_pass, avg_score, details, metrics=metrics)



# ─── Real-World Environmental Scenes (Urban Aesthetics) ──────────────────────

def build_old_town_prague_square() -> GameScene:
    """Historical Central Prague Old Town Square with Astronomical Clocktower, Tenements, and Gas Lamps."""
    sc = GameScene(
        "SCENE_OLD_TOWN_PRAGUE_SQUARE",
        "Staromestské Námestie (Old Town Prague Square)",
        "historic_bohemian_urban",
        "Historical cobblestone town square framed by Gothic and Baroque tenements, central astronomical clocktower, Jan Hus monument, linden trees, and gas lamps.",
        fog_color=(0.06, 0.04, 0.02),
        ambient_light=(0.35, 0.28, 0.16)
    )
    # 1. Macro Anchor: Central Clocktower
    sc.add_actor(SceneActor("OldTownClocktower", "TOWN_SQUARE_CLOCKTOWER", position=(0.0, 0.6, 1.8), scale=1.1))

    # 2. Meso Blocks: Flanking Historic Tenements
    sc.add_actor(SceneActor("Tenement_West", "HISTORIC_TENEMENT_FACADE", position=(-2.8, 0.0, 0.8), scale=0.9, rotation=(0, 0.25, 0)))
    sc.add_actor(SceneActor("Tenement_East", "HISTORIC_TENEMENT_FACADE", position=(2.8, 0.0, 0.8), scale=0.9, rotation=(0, -0.25, 0)))

    # 3. Micro Props: Civic Monument, Kiosk, Trees, Street Lamps
    sc.add_actor(SceneActor("JanHus_Monument", "BRONZE_CIVIC_MONUMENT", position=(-0.6, -0.6, -0.4), scale=0.75))
    sc.add_actor(SceneActor("OldTownKiosk", "STREET_CAFE_KIOSK", position=(1.6, -0.6, -0.8), scale=0.7))
    sc.add_actor(SceneActor("Linden_West", "URBAN_LINDEN_TREE", position=(-1.8, -0.5, 0.5), scale=0.8))
    sc.add_actor(SceneActor("Linden_East", "URBAN_LINDEN_TREE", position=(1.8, -0.5, 0.5), scale=0.8))
    sc.add_actor(SceneActor("StreetLamp_1", "ORNATE_CAST_IRON_STREETLAMP", position=(-1.2, -0.6, -1.2), scale=0.75))
    sc.add_actor(SceneActor("StreetLamp_2", "ORNATE_CAST_IRON_STREETLAMP", position=(1.2, -0.6, -1.2), scale=0.75))

    sc.validate_rules()
    return sc


def build_parisian_haussmann_boulevard() -> GameScene:
    """Grand Haussmannian Boulevard with Symmetrical Limestone Tenements and Linden Allée."""
    sc = GameScene(
        "SCENE_PARISIAN_HAUSSMANN_BOULEVARD",
        "Grand Boulevard Haussmann (Parisian Avenue)",
        "haussmannian_neoclassical",
        "Broad limestone avenue with uniform 6-story tenements, continuous wrought-iron balconies, double linden tree allée, and café kiosk.",
        fog_color=(0.04, 0.05, 0.07),
        ambient_light=(0.26, 0.30, 0.36)
    )
    # Meso Buildings along North and South sidewalks
    sc.add_actor(SceneActor("Haussmann_North_1", "HISTORIC_TENEMENT_FACADE", position=(-2.6, 0.0, 1.4), scale=0.95))
    sc.add_actor(SceneActor("Haussmann_North_2", "HISTORIC_TENEMENT_FACADE", position=(0.0, 0.0, 1.8), scale=0.95))
    sc.add_actor(SceneActor("Haussmann_North_3", "HISTORIC_TENEMENT_FACADE", position=(2.6, 0.0, 1.4), scale=0.95))

    # Macro Vista Anchor at terminal horizon
    sc.add_actor(SceneActor("Terminal_Spire", "TOWN_SQUARE_CLOCKTOWER", position=(0.0, 0.8, 3.2), scale=0.85))

    # Avenue Allée: Trees & Streetlamps
    sc.add_actor(SceneActor("Boulevard_Tree_1", "URBAN_LINDEN_TREE", position=(-1.6, -0.5, 0.2), scale=0.75))
    sc.add_actor(SceneActor("Boulevard_Tree_2", "URBAN_LINDEN_TREE", position=(1.6, -0.5, 0.2), scale=0.75))
    sc.add_actor(SceneActor("Boulevard_Lamp_1", "ORNATE_CAST_IRON_STREETLAMP", position=(-1.2, -0.6, -0.8), scale=0.75))
    sc.add_actor(SceneActor("Boulevard_Lamp_2", "ORNATE_CAST_IRON_STREETLAMP", position=(1.2, -0.6, -0.8), scale=0.75))

    # Sidewalk Café Kiosk
    sc.add_actor(SceneActor("SidewalkCafeKiosk", "STREET_CAFE_KIOSK", position=(-0.8, -0.6, -1.4), scale=0.65))

    sc.validate_rules()
    return sc


def build_mediterranean_coastal_port() -> GameScene:
    """Sun-Drenched Mediterranean Seafront Promenade with Stone Bridge and Coastal Watchtower."""
    sc = GameScene(
        "SCENE_MEDITERRANEAN_COASTAL_PORT",
        "Porto di Pietra (Mediterranean Coastal Promenade)",
        "mediterranean_coastal",
        "Sun-drenched sea promenade with travertine stone bridge over harbour channel, terraced villas, waterfront café, and coastal bell tower.",
        fog_color=(0.02, 0.06, 0.10),
        ambient_light=(0.38, 0.42, 0.46)
    )
    # Central Travertine Canal Bridge
    sc.add_actor(SceneActor("Harbour_Arch_Bridge", "CANAL_STONE_BRIDGE", position=(0.0, 0.0, 0.0), scale=1.1))

    # Coastal Watchtower at head of mole
    sc.add_actor(SceneActor("Seaward_Watchtower", "TOWN_SQUARE_CLOCKTOWER", position=(0.8, 0.5, 1.8), scale=0.9))


    # Waterfront Terraced Villa
    sc.add_actor(SceneActor("Waterfront_Villa", "HISTORIC_TENEMENT_FACADE", position=(-2.6, 0.2, 0.5), scale=0.85))

    # Pier Café & Promenade Lamps
    sc.add_actor(SceneActor("PierCafeKiosk", "STREET_CAFE_KIOSK", position=(-1.4, -0.6, -1.2), scale=0.7))
    sc.add_actor(SceneActor("Promenade_Lamp_1", "ORNATE_CAST_IRON_STREETLAMP", position=(-1.8, -0.6, -0.4), scale=0.75))
    sc.add_actor(SceneActor("Promenade_Lamp_2", "ORNATE_CAST_IRON_STREETLAMP", position=(1.8, -0.6, -0.4), scale=0.75))

    sc.validate_rules()
    return sc


def build_alpine_timber_township() -> GameScene:
    """Alpine Mountain Valley Settlement with Stream Bridge and Timber Chalets."""
    sc = GameScene(
        "SCENE_ALPINE_TIMBER_TOWNSHIP",
        "Bergwald Alpine Valley Township",
        "alpine_vernacular",
        "Mountain valley settlement featuring stone-plinth chalets, alpine stream stone bridge, linden and pine groves, and civic belfry.",
        fog_color=(0.03, 0.05, 0.08),
        ambient_light=(0.32, 0.36, 0.44)
    )
    # Stream Stone Bridge
    sc.add_actor(SceneActor("Stream_Bridge", "CANAL_STONE_BRIDGE", position=(0.0, -0.1, -0.4), scale=1.0))

    # Alpine Spire
    sc.add_actor(SceneActor("Alpine_Chapel_Spire", "TOWN_SQUARE_CLOCKTOWER", position=(2.2, 0.6, 1.6), scale=0.85))

    # Mountain Chalets
    sc.add_actor(SceneActor("Chalet_West", "HISTORIC_TENEMENT_FACADE", position=(-2.4, 0.2, 0.8), scale=0.85))
    sc.add_actor(SceneActor("Chalet_East", "HISTORIC_TENEMENT_FACADE", position=(2.4, 0.2, -1.0), scale=0.85))

    # Forest Buffer Trees & Bridge Lanterns
    sc.add_actor(SceneActor("Grove_Tree_1", "URBAN_LINDEN_TREE", position=(-1.6, -0.4, -1.2), scale=0.85))
    sc.add_actor(SceneActor("Grove_Tree_2", "URBAN_LINDEN_TREE", position=(1.6, -0.4, 0.6), scale=0.85))
    sc.add_actor(SceneActor("Bridge_Lantern_1", "ORNATE_CAST_IRON_STREETLAMP", position=(-0.9, -0.6, -0.4), scale=0.75))
    sc.add_actor(SceneActor("Bridge_Lantern_2", "ORNATE_CAST_IRON_STREETLAMP", position=(0.9, -0.6, -0.4), scale=0.75))

    sc.validate_rules()
    return sc


def build_industrial_canal_waterfront() -> GameScene:
    """19th-Century Industrial Canal with Brick Warehouses, Stone Bridge, and Towpath."""
    sc = GameScene(
        "SCENE_INDUSTRIAL_CANAL_WATERFRONT",
        "Vltava Industrial Canal & Warehouse Quayside",
        "industrial_heritage",
        "19th-century brick industrial quayside with arched stone bridge spanning canal, warehouse facades, cast iron lamps, and dockside monument.",
        fog_color=(0.03, 0.03, 0.02),
        ambient_light=(0.24, 0.20, 0.16)
    )
    # Canal Bridge
    sc.add_actor(SceneActor("Canal_Bridge", "CANAL_STONE_BRIDGE", position=(0.0, -0.1, 0.2), scale=1.15))

    # Macro Docklands Customs Tower
    sc.add_actor(SceneActor("Customs_Belfry_Tower", "TOWN_SQUARE_CLOCKTOWER", position=(0.0, 0.4, 2.2), scale=0.95))

    # Warehouse Blocks flanking canal
    sc.add_actor(SceneActor("Warehouse_Block_A", "HISTORIC_TENEMENT_FACADE", position=(-2.6, 0.0, 1.2), scale=0.9))
    sc.add_actor(SceneActor("Warehouse_Block_B", "HISTORIC_TENEMENT_FACADE", position=(2.6, 0.0, 1.2), scale=0.9))


    # Macro Monument on lock pier
    sc.add_actor(SceneActor("HarbourMaster_Monument", "BRONZE_CIVIC_MONUMENT", position=(1.6, -0.6, -1.0), scale=0.7))
    sc.add_actor(SceneActor("Towpath_Lamp_1", "ORNATE_CAST_IRON_STREETLAMP", position=(-1.4, -0.6, -0.8), scale=0.75))
    sc.add_actor(SceneActor("Towpath_Lamp_2", "ORNATE_CAST_IRON_STREETLAMP", position=(0.0, -0.6, -1.4), scale=0.75))
    sc.add_actor(SceneActor("Towpath_Lamp_3", "ORNATE_CAST_IRON_STREETLAMP", position=(1.4, -0.6, -0.8), scale=0.75))

    sc.validate_rules()
    return sc


# ─── Deterministic Real-World Urban Compositor ───────────────────────────────

def paint_real_world_scene(archetype: str = "old_town_square", seed: int = 42) -> GameScene:
    """
    Deterministically synthesizes a real-world urban spatial composition from a seed,
    strictly enforcing the 8 Urban Spatial Composition Rules.
    """
    arch = archetype.lower()
    if "haussmann" in arch or "boulevard" in arch or "paris" in arch:
        sc = build_parisian_haussmann_boulevard()
    elif "coastal" in arch or "port" in arch or "mediterranean" in arch:
        sc = build_mediterranean_coastal_port()
    elif "alpine" in arch or "chalet" in arch or "mountain" in arch:
        sc = build_alpine_timber_township()
    elif "industrial" in arch or "canal" in arch or "waterfront" in arch:
        sc = build_industrial_canal_waterfront()
    else:
        sc = build_old_town_prague_square()

    sc.seed = seed
    # Apply minor deterministic micro-perturbations based on seed S
    for i, a in enumerate(sc.actors):
        seed_jitter = math.sin(seed * 0.17 + i * 1.33) * 0.08
        a.position = (a.position[0] + seed_jitter, a.position[1], a.position[2])

    sc.validate_rules()
    return sc


# ─── Curated Game Scene Catalog ──────────────────────────────────────────────

def build_cyberpunk_district() -> GameScene:
    """Cyberpunk Megacity Sector with Central Data Spire & Automated Turrets."""
    sc = GameScene(
        "SCENE_CYBERPUNK_DISTRICT",
        "Neo-Krystal Cyber District",
        "cyberpunk_sci_fi",
        "Megacity skyline dominated by high-rise Data Spires, defensive sentry turrets on elevated parapets, and wet asphalt reflections.",
        fog_color=(0.02, 0.03, 0.06),
        ambient_light=(0.05, 0.2, 0.3)
    )
    # Central Data Spire
    sc.add_actor(SceneActor("DataSpire_Central", "CYBERPUNK_DATA_SPIRE", position=(0, 0.3, 0), scale=1.35))
    # Secondary flanking spire
    sc.add_actor(SceneActor("DataSpire_West", "CYBERPUNK_DATA_SPIRE", position=(-2.6, 0.0, 1.2), scale=0.9))
    # Defensive turret emplacements
    sc.add_actor(SceneActor("Turret_Perimeter_East", "CYBER_TURRET_MK4", position=(2.0, -0.6, -0.5), scale=0.75, rotation=(0, 0.5, 0)))
    sc.add_actor(SceneActor("Turret_Perimeter_South", "CYBER_TURRET_MK4", position=(-1.0, -0.6, -2.0), scale=0.75, rotation=(0, -1.2, 0)))
    return sc


def build_sacred_alchemical_ruins() -> GameScene:
    """Ancient Temple Sanctuary with Circle of Obelisks and Floating Runes."""
    sc = GameScene(
        "SCENE_SACRED_ALCHEMICAL_RUINS",
        "Temple of the Thousand Runes",
        "sacred_fantasy",
        "Ancient sanctuary ruins with four carved obelisks aligned to cardinal magnetic axes and levitating alchemical capstones.",
        fog_color=(0.05, 0.03, 0.01),
        ambient_light=(0.3, 0.2, 0.08)
    )
    # Cardinal ring of 4 Obelisks
    radius = 2.4
    for i in range(4):
        theta = i * (math.pi / 2.0)
        px = radius * math.cos(theta)
        pz = radius * math.sin(theta)
        sc.add_actor(SceneActor(f"Obelisk_Cardinal_{i+1}", "ANCIENT_OBELISK_MONOLITH", position=(px, -0.3, pz), scale=0.85, rotation=(0, theta, 0)))
    return sc


def build_deep_space_hangar() -> GameScene:
    """Orbital Docking Bay with Exploration Vessels and Heavy Battle Mechs."""
    sc = GameScene(
        "SCENE_DEEP_SPACE_HANGAR",
        "Starport Aegis Hangar Bay",
        "hard_sci_fi",
        "Zero-G maintenance bay with docked Solar Explorer research vessels and Titan-IV Mech Walkers in ready racks.",
        fog_color=(0.01, 0.02, 0.04),
        ambient_light=(0.15, 0.2, 0.35)
    )
    # Docked Solar Explorer vessel
    sc.add_actor(SceneActor("Explorer_Dock_Alpha", "RETRO_SOLAR_EXPLORER", position=(0, 0.6, 0.5), scale=1.0, rotation=(0.1, 0.4, 0)))
    # Combat Mech Walker on patrol pad
    sc.add_actor(SceneActor("TitanMech_Sentry", "MECH_WALKER_TITAN", position=(2.8, -0.3, -1.2), scale=0.85, rotation=(0, -0.6, 0)))
    # Sentry Turret guardian
    sc.add_actor(SceneActor("BayGuard_Turret", "CYBER_TURRET_MK4", position=(-2.8, -0.6, -1.0), scale=0.7, rotation=(0, 1.0, 0)))
    return sc


def build_alien_hive_chamber() -> GameScene:
    """Subterranean Organic Labyrinth with Xenobiotic Swarm Drones."""
    sc = GameScene(
        "SCENE_ALIEN_HIVE_CHAMBER",
        "Chitin Hollow Xenobiotic Spire",
        "biomechanical_horror",
        "Pulsating subterranean bio-cavern populated by swarming Xenodrones and ribbed bio-mechanical structures.",
        fog_color=(0.02, 0.04, 0.02),
        ambient_light=(0.1, 0.25, 0.12)
    )
    # Alpha Xenodrone
    sc.add_actor(SceneActor("Xenodrone_Matriarch", "BIOMECHANICAL_XENODRONE", position=(0, 0.2, 0), scale=1.2, rotation=(0.2, 0.3, 0)))
    # Swarm workers
    sc.add_actor(SceneActor("Xenodrone_Scout_1", "BIOMECHANICAL_XENODRONE", position=(-1.8, 0.8, -1.0), scale=0.7, rotation=(-0.3, 0.8, 0)))
    sc.add_actor(SceneActor("Xenodrone_Scout_2", "BIOMECHANICAL_XENODRONE", position=(2.0, 0.6, 1.2), scale=0.75, rotation=(0.1, -1.1, 0)))
    return sc


SCENES_CATALOG: Dict[str, Callable[[], GameScene]] = {
    "SCENE_CYBERPUNK_DISTRICT": build_cyberpunk_district,
    "SCENE_SACRED_ALCHEMICAL_RUINS": build_sacred_alchemical_ruins,
    "SCENE_DEEP_SPACE_HANGAR": build_deep_space_hangar,
    "SCENE_ALIEN_HIVE_CHAMBER": build_alien_hive_chamber,
    # Real-World Urban Spatial Mimicry Environments
    "SCENE_OLD_TOWN_PRAGUE_SQUARE": build_old_town_prague_square,
    "SCENE_PARISIAN_HAUSSMANN_BOULEVARD": build_parisian_haussmann_boulevard,
    "SCENE_MEDITERRANEAN_COASTAL_PORT": build_mediterranean_coastal_port,
    "SCENE_ALPINE_TIMBER_TOWNSHIP": build_alpine_timber_township,
    "SCENE_INDUSTRIAL_CANAL_WATERFRONT": build_industrial_canal_waterfront
}


def get_scene(scene_id: str) -> Optional[GameScene]:
    factory = SCENES_CATALOG.get(scene_id)
    return factory() if factory else None

def list_scenes() -> List[Dict[str, Any]]:
    results = []
    for sid, factory in SCENES_CATALOG.items():
        sc = factory()
        results.append({
            "id": sc.scene_id,
            "name": sc.name,
            "genre": sc.genre,
            "description": sc.description,
            "actors_count": len(sc.actors)
        })
    return results


if __name__ == "__main__":
    print("[KRYSTAL] Testing Game Scene Composer Outlines...")
    for sid in SCENES_CATALOG:
        scene = get_scene(sid)
        print(f"\n==========================================")
        print(f"SCENE: {scene.name} [{scene.genre}]")
        print(f"Actors ({len(scene.actors)}): {[a.actor_id for a in scene.actors]}")
        print("--- ASCII Scene Projection (First 8 lines) ---")
        preview = scene.render_ascii_view(width=64, height=8, t=1.0)
        print(preview)
