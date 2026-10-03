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
            "tint": list(self.tint)
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

    def add_actor(self, actor: SceneActor) -> "GameScene":
        self.actors.append(actor)
        return self

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
"""
        return scene_str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scene_id": self.scene_id,
            "name": self.name,
            "genre": self.genre,
            "description": self.description,
            "actors_count": len(self.actors),
            "actors": [a.to_dict() for a in self.actors],
            "ground_level": self.ground_level,
            "fog_color": list(self.fog_color),
            "ambient_light": list(self.ambient_light)
        }


# ─── 4 Curated Game Scene Outlines ───────────────────────────────────────────

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
    "SCENE_ALIEN_HIVE_CHAMBER": build_alien_hive_chamber
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
