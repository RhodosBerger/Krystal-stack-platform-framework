"""
Krystal-Stack Platform Framework: Mimicry Object Recipes
=========================================================
Procedural recipes that mimic complex real-world objects using combinations
of geometric primitives and Blender-style modifier stacks.
"""

import math
import sys
from typing import Dict, Any, List, Optional, Callable, Tuple

if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
from mimicry_engine.primitives import (
    Vec3, v_add, v_sub, v_scale, v_rot_x, v_rot_y, v_rot_z,
    sdf_sphere, sdf_box, sdf_round_box, sdf_cylinder, sdf_capped_cone,
    sdf_torus, sdf_capsule, sdf_hex_prism, sdf_octahedron,
    smin, smax, smooth_difference
)
from mimicry_engine.modifiers import (
    Modifier, ArrayModifier, MirrorModifier, BooleanModifier,
    BevelModifier, DisplaceModifier, DeformModifier, SolidifyModifier,
    ModifierStack
)

class MimicPart:
    """A single sub-part of a composite mimic object with local transform and modifier stack."""
    def __init__(
        self,
        name: str,
        primitive_func: Callable[[Vec3], float],
        offset: Vec3 = (0.0, 0.0, 0.0),
        rotation: Vec3 = (0.0, 0.0, 0.0),
        blend_k: float = 0.15,
        is_subtraction: bool = False
    ):
        self.name = name
        self.primitive_func = primitive_func
        self.offset = offset
        self.rotation = rotation
        self.blend_k = blend_k
        self.is_subtraction = is_subtraction
        self.modifier_stack = ModifierStack(self._eval_primitive_local)

    def add_modifier(self, mod: Modifier) -> "MimicPart":
        self.modifier_stack.add_modifier(mod)
        return self

    def _eval_primitive_local(self, p: Vec3) -> float:
        return self.primitive_func(p)

    def evaluate(self, p_world: Vec3) -> float:
        # Inverse transform to local space
        pl = v_sub(p_world, self.offset)
        if abs(self.rotation[1]) > 1e-5: pl = v_rot_y(pl, -self.rotation[1])
        if abs(self.rotation[0]) > 1e-5: pl = v_rot_x(pl, -self.rotation[0])
        if abs(self.rotation[2]) > 1e-5: pl = v_rot_z(pl, -self.rotation[2])
        return self.modifier_stack.evaluate(pl)


class MimicObject:
    """A full real-world composite object composed of multiple modifier-driven parts."""
    def __init__(
        self,
        object_id: str,
        name: str,
        category: str,
        description: str,
        preferred_glyphs: str = " ░▒▓█"
    ):
        self.object_id = object_id
        self.name = name
        self.category = category
        self.description = description
        self.preferred_glyphs = preferred_glyphs
        self.parts: List[MimicPart] = []

    def add_part(self, part: MimicPart) -> "MimicObject":
        self.parts.append(part)
        return self

    def evaluate_sdf(self, p: Vec3, t: float = 0.0) -> float:
        if not self.parts:
            return float('inf')

        total_d = self.parts[0].evaluate(p)
        for part in self.parts[1:]:
            d_part = part.evaluate(p)
            if part.is_subtraction:
                total_d = smooth_difference(total_d, d_part, part.blend_k)
            else:
                total_d = smin(total_d, d_part, part.blend_k)

        return total_d

    def render_ascii_projection(
        self,
        width: int = 64,
        height: int = 24,
        t: float = 0.0,
        cam_dist: float = 3.2,
        rot_speed: float = 0.6
    ) -> str:
        """Raymarches the composite object into a terminal ASCII frame."""
        aspect = (width / height) * 0.52
        lines = []
        ramp = self.preferred_glyphs
        light_dir = (0.577, 0.577, -0.577)

        for y in range(height):
            line_chars = []
            ny = 1.0 - (y / height) * 2.0
            for x in range(width):
                nx = ((x / width) * 2.0 - 1.0) * aspect

                # Camera setup: rotating orbit around object
                angle = t * rot_speed
                ro = (cam_dist * math.sin(angle), 0.8, -cam_dist * math.cos(angle))
                ta = (0.0, 0.0, 0.0)

                # Camera Ray basis
                ww = v_sub(ta, ro)
                l_ww = math.sqrt(ww[0]**2 + ww[1]**2 + ww[2]**2)
                ww = (ww[0]/l_ww, ww[1]/l_ww, ww[2]/l_ww)

                uu = (ww[2], 0.0, -ww[0]) # cross with up=(0,1,0)
                l_uu = math.sqrt(uu[0]**2 + uu[2]**2)
                uu = (uu[0]/max(1e-5, l_uu), 0.0, uu[2]/max(1e-5, l_uu))
                vv = (uu[1]*ww[2] - uu[2]*ww[1], uu[2]*ww[0] - uu[0]*ww[2], uu[0]*ww[1] - uu[1]*ww[0])

                rd = v_add(v_add(v_scale(uu, nx), v_scale(vv, ny)), v_scale(ww, 1.8))
                l_rd = math.sqrt(rd[0]**2 + rd[1]**2 + rd[2]**2)
                rd = (rd[0]/l_rd, rd[1]/l_rd, rd[2]/l_rd)

                # Raymarch
                dist = 0.0
                hit = False
                p = ro
                for _ in range(24):
                    p = v_add(ro, v_scale(rd, dist))
                    d = self.evaluate_sdf(p, t)
                    if d < 0.006:
                        hit = True
                        break
                    dist += d
                    if dist > 6.0:
                        break

                if hit:
                    # Compute normal
                    eps = 0.005
                    nx_s = self.evaluate_sdf((p[0]+eps, p[1], p[2]), t) - self.evaluate_sdf((p[0]-eps, p[1], p[2]), t)
                    ny_s = self.evaluate_sdf((p[0], p[1]+eps, p[2]), t) - self.evaluate_sdf((p[0], p[1]-eps, p[2]), t)
                    nz_s = self.evaluate_sdf((p[0], p[1], p[2]+eps), t) - self.evaluate_sdf((p[0], p[1], p[2]-eps), t)
                    mag = math.sqrt(nx_s**2 + ny_s**2 + nz_s**2)
                    norm = (nx_s/mag, ny_s/mag, nz_s/mag) if mag > 1e-6 else (0.0, 1.0, 0.0)

                    diff = max(0.0, norm[0]*light_dir[0] + norm[1]*light_dir[1] + norm[2]*light_dir[2])
                    rim = 1.0 - max(0.0, -(norm[0]*rd[0] + norm[1]*rd[1] + norm[2]*rd[2]))
                    luma = min(1.0, max(0.0, diff * 0.7 + rim * 0.5))
                    idx = int(luma * (len(ramp) - 1))
                    ch = ramp[idx]
                else:
                    ch = " "
                line_chars.append(ch)
            lines.append("".join(line_chars))

        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "object_id": self.object_id,
            "name": self.name,
            "category": self.category,
            "description": self.description,
            "parts_count": len(self.parts),
            "parts": [
                {
                    "name": p.name,
                    "offset": list(p.offset),
                    "rotation": list(p.rotation),
                    "blend_k": p.blend_k,
                    "is_subtraction": p.is_subtraction,
                    "modifiers": p.modifier_stack.to_dict()
                }
                for p in self.parts
            ]
        }


# ─── 6 Pre-Assembled Real-World Composite Object Recipes ─────────────────────

def build_cyber_turret_mk4() -> MimicObject:
    """Autonomous Combat Turret with Dual Cannons & Armored Mantlet."""
    obj = MimicObject(
        "CYBER_TURRET_MK4",
        "Cyber Turret MK-4 Automated Sentry",
        "military_hardware",
        "Hexagonal armored pedestal with rotating spherical gimbal, dual railgun barrels, and flared ammo drums."
    )
    # 1. Base pedestal: Hexagonal Prism + Bevel
    p_base = MimicPart("PedestalBase", lambda p: sdf_hex_prism(p, h=0.25, r=1.0), offset=(0, -0.75, 0))
    p_base.add_modifier(BevelModifier("BaseChamfer", radius=0.06))
    obj.add_part(p_base)

    # 2. Swivel gimbal: Sphere
    p_gimbal = MimicPart("GimbalCore", lambda p: sdf_sphere(p, r=0.55), offset=(0, -0.3, 0), blend_k=0.1)
    obj.add_part(p_gimbal)

    # 3. Turret Head Mantlet: Beveled Box + Mirror
    p_head = MimicPart("TurretMantlet", lambda p: sdf_round_box(p, b=(0.4, 0.35, 0.6), r=0.08), offset=(0, 0.1, 0))
    obj.add_part(p_head)

    # 4. Dual Railgun Barrels: Cylinder + ArrayModifier (offset X)
    p_barrels = MimicPart("DualBarrels", lambda p: sdf_cylinder(v_rot_x(p, 1.5708), r=0.09, h=0.75), offset=(0, 0.1, 0.7))
    p_barrels.add_modifier(ArrayModifier("BarrelDuo", count=2, offset=(0.36, 0, 0)))
    obj.add_part(p_barrels)

    # 5. Ammo Drum Pods: Torus + Mirror X
    p_ammo = MimicPart("AmmoPods", lambda p: sdf_torus(v_rot_z(p, 1.5708), r1=0.28, r2=0.1), offset=(0.55, 0.05, -0.1))
    p_ammo.add_modifier(MirrorModifier("MirrorAmmo", use_x=True))
    obj.add_part(p_ammo)

    return obj


def build_mech_walker_titan() -> MimicObject:
    """Bipedal Heavy Walker Mech with Articulated Leg Pylons."""
    obj = MimicObject(
        "MECH_WALKER_TITAN",
        "Titan-IV Heavy Bipedal Mech",
        "robotics_and_vehicles",
        "Aggressive armored chassis with reinforced cockpit visor, inverted knee actuators, and hydraulic foot stabilizers."
    )
    # 1. Main Cockpit Hull: Tapered Box
    p_cockpit = MimicPart("CockpitPod", lambda p: sdf_round_box(p, b=(0.6, 0.5, 0.7), r=0.1), offset=(0, 0.6, 0))
    p_cockpit.add_modifier(DeformModifier("HullTaper", deform_type="TAPER", factor=0.35, axis="y"))
    obj.add_part(p_cockpit)

    # 2. Visor Sensor Slit: Subtraction Box
    p_visor = MimicPart("SensorVisorCut", lambda p: sdf_box(p, b=(0.5, 0.08, 0.3)), offset=(0, 0.65, 0.6), is_subtraction=True)
    obj.add_part(p_visor)

    # 3. Hip Crossbar: Cylinder
    p_hips = MimicPart("PelvisGirdle", lambda p: sdf_cylinder(v_rot_z(p, 1.5708), r=0.2, h=0.6), offset=(0, 0.0, 0))
    obj.add_part(p_hips)

    # 4. Upper Thigh & Knee Pylons: Capsules + Mirror X
    p_thigh = MimicPart("LegStruts", lambda p: sdf_capsule(p, (0, 0, 0), (0, -0.7, -0.2), r=0.14), offset=(0.65, -0.05, 0))
    p_thigh.add_modifier(MirrorModifier("MirrorLegs", use_x=True))
    obj.add_part(p_thigh)

    # 5. Lower Calves & Foot Pads: Boxes + Mirror X
    p_feet = MimicPart("TractionPads", lambda p: sdf_round_box(p, b=(0.28, 0.08, 0.45), r=0.04), offset=(0.65, -0.85, -0.1))
    p_feet.add_modifier(MirrorModifier("MirrorFeet", use_x=True))
    obj.add_part(p_feet)

    return obj


def build_ancient_obelisk_monolith() -> MimicObject:
    """Mystical Alchemical Monolith with Stepped Pedestal & Floating Crystal."""
    obj = MimicObject(
        "ANCIENT_OBELISK_MONOLITH",
        "Ancient Runic Obelisk Monolith",
        "ancient_monuments",
        "Stepped limestone plinth supporting a carved stone spire with procedural glyph displacement and a levitating crystal capstone."
    )
    # 1. Stepped Plinth: Array of 3 expanding stepped slabs
    p_plinth = MimicPart("SteppedPlinth", lambda p: sdf_box(p, b=(0.9, 0.08, 0.9)), offset=(0, -0.9, 0))
    p_plinth.add_modifier(ArrayModifier("PlinthSteps", count=3, offset=(0, 0.12, 0)))
    obj.add_part(p_plinth)

    # 2. Main Obelisk Column: Capped Cone + Displace Runic Relief
    p_column = MimicPart("ObeliskSpire", lambda p: sdf_capped_cone(p, h=1.0, r1=0.45, r2=0.25), offset=(0, 0.35, 0))
    p_column.add_modifier(DisplaceModifier("RunicRelief", strength=0.05, frequency=5.0))
    obj.add_part(p_column)

    # 3. Levitating Octahedron Capstone: Floating at top
    p_capstone = MimicPart("FloatingCapstone", lambda p: sdf_octahedron(p, s=0.32), offset=(0, 1.6, 0))
    p_capstone.add_modifier(BevelModifier("CrystalFacet", radius=0.02))
    obj.add_part(p_capstone)

    return obj


def build_biomechanical_xenodrone() -> MimicObject:
    """Giger-esque Organic Chitin Drone with Segmented Spine."""
    obj = MimicObject(
        "BIOMECHANICAL_XENODRONE",
        "Xenobiotic Swarm Drone",
        "biological_organisms",
        "Curved exoskeleton with ribbed spinal segmentation, sickle mandibles, and chitinous vein displacement."
    )
    # 1. Elongated Chitin Skull: Ellipsoid-like Box
    p_skull = MimicPart("ChitinCranium", lambda p: sdf_round_box(p, b=(0.35, 0.3, 0.65), r=0.15), offset=(0, 0.4, 0.2))
    p_skull.add_modifier(DisplaceModifier("BioVeins", strength=0.06, frequency=6.0))
    obj.add_part(p_skull)

    # 2. Segmented Spinal Ribs: Torus + ArrayModifier along Z
    p_spine = MimicPart("SpinalVertebrae", lambda p: sdf_torus(p, r1=0.32, r2=0.08), offset=(0, 0.15, -0.4))
    p_spine.add_modifier(ArrayModifier("RibArray", count=4, offset=(0, -0.15, -0.22)))
    obj.add_part(p_spine)

    # 3. Sickle Mandibles: Capped Cone + Mirror X
    p_mandibles = MimicPart("MandibleSickles", lambda p: sdf_capped_cone(v_rot_z(p, 0.4), h=0.3, r1=0.08, r2=0.02), offset=(0.3, 0.2, 0.75))
    p_mandibles.add_modifier(MirrorModifier("MirrorMandibles", use_x=True))
    obj.add_part(p_mandibles)

    return obj


def build_cyberpunk_data_spire() -> MimicObject:
    """Futuristic Megastructure Server Spire with Cantilevered Decks."""
    obj = MimicObject(
        "CYBERPUNK_DATA_SPIRE",
        "Megacity Central Data Spire",
        "urban_architecture",
        "Brutalist vertical core with cantilevered server pods, radiator heat sink arrays, and a top-mounted holographic ring."
    )
    # 1. Central Core Spine: Tall slender Box
    p_core = MimicPart("CentralCore", lambda p: sdf_round_box(p, b=(0.35, 1.4, 0.35), r=0.05), offset=(0, 0, 0))
    obj.add_part(p_core)

    # 2. Cantilever Server Decks: Array of horizontal slabs along Y
    p_decks = MimicPart("ServerDecks", lambda p: sdf_round_box(p, b=(0.75, 0.06, 0.75), r=0.02), offset=(0, -0.4, 0))
    p_decks.add_modifier(ArrayModifier("DeckLevels", count=5, offset=(0, 0.38, 0)))
    obj.add_part(p_decks)

    # 3. Hologram Broadcast Ring: Torus at zenith
    p_ring = MimicPart("HoloRingCrown", lambda p: sdf_torus(p, r1=0.7, r2=0.07), offset=(0, 1.4, 0))
    obj.add_part(p_ring)

    # 4. Radiator Heat Sink Fins: Thin slabs + Mirror X
    p_fins = MimicPart("CoolantFins", lambda p: sdf_box(p, b=(0.04, 0.5, 0.55)), offset=(0.45, 0.2, 0))
    p_fins.add_modifier(MirrorModifier("MirrorFins", use_x=True))
    obj.add_part(p_fins)

    return obj


def build_retro_solar_explorer() -> MimicObject:
    """Interplanetary Research Vessel with Photovoltaic Wings & Ion Thrusters."""
    obj = MimicObject(
        "RETRO_SOLAR_EXPLORER",
        "Solar Explorer Research Vessel",
        "spacecraft",
        "Pressurized spherical crew module, deployable solar wing arrays, high-gain dish, and quad ion thrusters."
    )
    # 1. Crew Habitat Sphere
    p_hab = MimicPart("CommandSphere", lambda p: sdf_sphere(p, r=0.65), offset=(0, 0, 0.2))
    obj.add_part(p_hab)

    # 2. Quad Ion Thrusters: ArrayModifier Radial (4 nozzles)
    p_thrusters = MimicPart("IonNozzles", lambda p: sdf_capped_cone(v_rot_x(p, 1.5708), h=0.25, r1=0.14, r2=0.08), offset=(0.25, 0.25, -0.6))
    p_thrusters.add_modifier(ArrayModifier("QuadRadial", count=4, radial=True, radial_axis="z"))
    obj.add_part(p_thrusters)

    # 3. Solar Photovoltaic Panels: Thin Box + Mirror X
    p_panels = MimicPart("SolarWings", lambda p: sdf_box(p, b=(0.9, 0.02, 0.45)), offset=(1.5, 0, 0.1))
    p_panels.add_modifier(MirrorModifier("MirrorWings", use_x=True))
    obj.add_part(p_panels)

    # 4. High-Gain Comm Dish: Capped Cone + Solidify
    p_dish = MimicPart("HighGainDish", lambda p: sdf_capped_cone(v_rot_x(p, -1.5708), h=0.15, r1=0.35, r2=0.08), offset=(0, 0.75, 0.4))
    p_dish.add_modifier(SolidifyModifier("DishWall", thickness=0.03))
    obj.add_part(p_dish)

    return obj


# Catalog of pre-assembled recipes
RECIPES_CATALOG: Dict[str, Callable[[], MimicObject]] = {
    "CYBER_TURRET_MK4": build_cyber_turret_mk4,
    "MECH_WALKER_TITAN": build_mech_walker_titan,
    "ANCIENT_OBELISK_MONOLITH": build_ancient_obelisk_monolith,
    "BIOMECHANICAL_XENODRONE": build_biomechanical_xenodrone,
    "CYBERPUNK_DATA_SPIRE": build_cyberpunk_data_spire,
    "RETRO_SOLAR_EXPLORER": build_retro_solar_explorer
}

def get_recipe(recipe_id: str) -> Optional[MimicObject]:
    factory = RECIPES_CATALOG.get(recipe_id)
    return factory() if factory else None

def list_recipes() -> List[Dict[str, Any]]:
    results = []
    for rid, factory in RECIPES_CATALOG.items():
        obj = factory()
        results.append({
            "id": obj.object_id,
            "name": obj.name,
            "category": obj.category,
            "description": obj.description,
            "parts_count": len(obj.parts)
        })
    return results


if __name__ == "__main__":
    print("[KRYSTAL] Testing Mimicry Object Recipes...")
    for rid in RECIPES_CATALOG:
        obj = get_recipe(rid)
        print(f"\n==========================================")
        print(f"OBJECT: {obj.name} [{obj.category}]")
        print(f"Parts: {[p.name for p in obj.parts]}")
        print("--- ASCII Projection Preview (First 8 lines) ---")
        preview = obj.render_ascii_projection(width=56, height=8, t=0.5)
        print(preview)
