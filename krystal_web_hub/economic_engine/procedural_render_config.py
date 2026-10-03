# ==============================================================================
# KRYSTAL-STACK: PROCEDURAL RENDER CONFIGURATION & GODOT 4 SCENE ENGINE
# ==============================================================================
# Defines exact procedural render parameters, PBR materials, shaders, procedural
# primitive meshes, particle systems, and sub-resource .TSCN serialization.
# ==============================================================================

from typing import Dict, List, Any, Optional, Tuple
import math

# ------------------------------------------------------------------------------
# 1. PBR STANDARD MATERIAL 3D PRESETS
# ------------------------------------------------------------------------------
PROCEDURAL_PBR_PRESETS: Dict[str, Dict[str, Any]] = {
    "crystal_facet_pbr": {
        "id": "StandardMaterial3D_cryst",
        "type": "StandardMaterial3D",
        "albedo_color": [0.40, 0.98, 0.94, 0.85],
        "metallic": 0.85,
        "roughness": 0.15,
        "emission_enabled": True,
        "emission": [0.0, 1.0, 1.0, 1.0],
        "emission_energy_multiplier": 2.4,
        "subsurface_scattering_enabled": True,
        "transparency": 1, # Alpha blend
        "refraction_enabled": True,
        "refraction_scale": 0.05
    },
    "toxic_slime_organic": {
        "id": "StandardMaterial3D_slime",
        "type": "StandardMaterial3D",
        "albedo_color": [0.22, 1.0, 0.08, 0.90],
        "metallic": 0.10,
        "roughness": 0.05,
        "emission_enabled": True,
        "emission": [0.22, 1.0, 0.08, 1.0],
        "emission_energy_multiplier": 1.6,
        "subsurface_scattering_enabled": True,
        "transparency": 1,
        "clearcoat_enabled": True,
        "clearcoat": 1.0,
        "clearcoat_roughness": 0.1
    },
    "druid_ancient_bark": {
        "id": "StandardMaterial3D_bark",
        "type": "StandardMaterial3D",
        "albedo_color": [0.55, 0.27, 0.07, 1.0],
        "metallic": 0.05,
        "roughness": 0.85,
        "emission_enabled": True,
        "emission": [1.0, 0.84, 0.0, 1.0],
        "emission_energy_multiplier": 0.8,
        "rim_enabled": True,
        "rim": 0.5,
        "rim_tint": 0.5
    },
    "druid_mossy_granite": {
        "id": "StandardMaterial3D_granite",
        "type": "StandardMaterial3D",
        "albedo_color": [0.33, 0.42, 0.18, 1.0],
        "metallic": 0.15,
        "roughness": 0.75,
        "emission_enabled": False,
        "detail_enabled": True
    },
    "basalt_fortress_stone": {
        "id": "StandardMaterial3D_basalt",
        "type": "StandardMaterial3D",
        "albedo_color": [0.12, 0.14, 0.18, 1.0],
        "metallic": 0.45,
        "roughness": 0.65,
        "emission_enabled": True,
        "emission": [0.4, 0.8, 1.0, 1.0],
        "emission_energy_multiplier": 1.2
    },
    "aether_projectile_energy": {
        "id": "StandardMaterial3D_projectile",
        "type": "StandardMaterial3D",
        "albedo_color": [1.0, 1.0, 1.0, 0.95],
        "metallic": 0.90,
        "roughness": 0.10,
        "emission_enabled": True,
        "emission": [0.0, 0.95, 1.0, 1.0],
        "emission_energy_multiplier": 4.5,
        "transparency": 1
    }
}

# ------------------------------------------------------------------------------
# 2. PROCEDURAL MESH PRESETS (GODOT 4 PRIMITIVES)
# ------------------------------------------------------------------------------
PROCEDURAL_MESH_PRESETS: Dict[str, Dict[str, Any]] = {
    "hex_column_mesh": {
        "id": "CylinderMesh_hex",
        "type": "CylinderMesh",
        "top_radius": 1.5,
        "bottom_radius": 1.5,
        "height": 0.4,
        "radial_segments": 6,
        "rings": 1
    },
    "crystal_shard_mesh": {
        "id": "PrismMesh_crystal",
        "type": "PrismMesh",
        "left_to_right": 0.5,
        "size": [1.2, 2.8, 1.2]
    },
    "slime_blob_mesh": {
        "id": "SphereMesh_slime",
        "type": "SphereMesh",
        "radius": 1.1,
        "height": 1.4,
        "radial_segments": 24,
        "rings": 12
    },
    "totem_column_mesh": {
        "id": "CylinderMesh_totem",
        "type": "CylinderMesh",
        "top_radius": 0.6,
        "bottom_radius": 0.8,
        "height": 2.2,
        "radial_segments": 12
    },
    "monolith_slab_mesh": {
        "id": "BoxMesh_monolith",
        "type": "BoxMesh",
        "size": [0.8, 2.4, 0.5]
    },
    "projectile_core_mesh": {
        "id": "SphereMesh_proj",
        "type": "SphereMesh",
        "radius": 0.25,
        "height": 0.5,
        "radial_segments": 16,
        "rings": 8
    },
    "shockwave_ring_mesh": {
        "id": "TorusMesh_shockwave",
        "type": "TorusMesh",
        "inner_radius": 0.8,
        "outer_radius": 1.0,
        "rings": 32,
        "ring_segments": 8
    }
}

# ------------------------------------------------------------------------------
# 3. GPU PARTICLE PROCESS MATERIAL PRESETS
# ------------------------------------------------------------------------------
PROCEDURAL_PARTICLE_PRESETS: Dict[str, Dict[str, Any]] = {
    "mana_dust_particles": {
        "id": "ParticleProcessMaterial_mana",
        "type": "ParticleProcessMaterial",
        "emission_shape": 1, # Sphere
        "emission_sphere_radius": 0.8,
        "direction": [0.0, 1.0, 0.0],
        "spread": 25.0,
        "initial_velocity_min": 1.5,
        "initial_velocity_max": 3.2,
        "gravity": [0.0, -1.0, 0.0],
        "scale_min": 0.05,
        "scale_max": 0.15,
        "color": [0.0, 1.0, 1.0, 0.8]
    },
    "poison_mist_particles": {
        "id": "ParticleProcessMaterial_poison",
        "type": "ParticleProcessMaterial",
        "emission_shape": 1,
        "emission_sphere_radius": 1.5,
        "direction": [0.0, 0.5, 0.0],
        "spread": 45.0,
        "initial_velocity_min": 0.5,
        "initial_velocity_max": 1.8,
        "gravity": [0.0, 0.2, 0.0], # Rising vapor
        "scale_min": 0.2,
        "scale_max": 0.6,
        "color": [0.22, 1.0, 0.08, 0.6]
    },
    "life_spark_particles": {
        "id": "ParticleProcessMaterial_sparks",
        "type": "ParticleProcessMaterial",
        "emission_shape": 2, # Box
        "emission_box_extents": [0.5, 0.1, 0.5],
        "direction": [0.0, 1.0, 0.0],
        "spread": 15.0,
        "initial_velocity_min": 2.0,
        "initial_velocity_max": 4.0,
        "gravity": [0.0, -0.5, 0.0],
        "scale_min": 0.08,
        "scale_max": 0.20,
        "color": [1.0, 0.84, 0.0, 0.9]
    }
}

# ------------------------------------------------------------------------------
# 4. LIGHTING & WORLD ENVIRONMENT PRESETS
# ------------------------------------------------------------------------------
PROCEDURAL_ENVIRONMENT_CONFIG: Dict[str, Any] = {
    "sun_light": {
        "color": [1.0, 0.98, 0.92, 1.0],
        "energy": 1.25,
        "transform": [0.866, -0.25, 0.433, 0.0, 0.866, 0.5, -0.5, -0.433, 0.75, 12.0, 24.0, 12.0],
        "shadow_enabled": True
    },
    "world_environment": {
        "background_mode": 1, # Color
        "background_color": [0.03, 0.04, 0.06, 1.0],
        "ambient_light_color": [0.15, 0.18, 0.25, 1.0],
        "ambient_light_energy": 0.6,
        "tonemap_mode": 3, # ACES
        "glow_enabled": True,
        "glow_intensity": 0.85,
        "glow_bloom": 0.30,
        "fog_enabled": True,
        "fog_light_color": [0.08, 0.12, 0.18, 1.0],
        "fog_density": 0.012
    }
}


# ------------------------------------------------------------------------------
# 5. PROCEDURAL SCENE SERIALIZER TO GODOT 4 .TSCN WITH SUB-RESOURCES
# ------------------------------------------------------------------------------
class Godot4ProceduralSceneBuilder:
    """Builds and serializes rich Godot 4 .tscn files with sub-resources."""

    def __init__(self, scene_id: str = "krystal_procedural_scene"):
        self.scene_id = scene_id
        self.sub_resources: List[Dict[str, Any]] = []
        self.nodes: List[Dict[str, Any]] = []
        self._registered_res_ids = set()

    def register_sub_resource(self, res_data: Dict[str, Any]) -> str:
        res_id = res_data.get("id", f"SubRes_{len(self.sub_resources)}")
        if res_id not in self._registered_res_ids:
            self.sub_resources.append(res_data)
            self._registered_res_ids.add(res_id)
        return res_id

    def add_node(
        self,
        name: str,
        node_type: str,
        parent: str = ".",
        transform: Optional[List[float]] = None,
        mesh_ref: Optional[str] = None,
        material_ref: Optional[str] = None,
        process_material_ref: Optional[str] = None,
        extra_props: Optional[Dict[str, Any]] = None
    ):
        node = {
            "name": name,
            "type": node_type,
            "parent": parent,
            "transform": transform,
            "mesh_ref": mesh_ref,
            "material_ref": material_ref,
            "process_material_ref": process_material_ref,
            "extra_props": extra_props or {}
        }
        self.nodes.append(node)

    def serialize_to_tscn(self) -> str:
        lines = [
            f'[gd_scene format=3 uid="uid://{self.scene_id}"]',
            '',
            '# =====================================================================',
            '# KRYSTAL-STACK: PROCEDURAL GODOT 4.X SCENE FILE',
            '# Generated by Procedural Render Engine (PBR, Shaders, Particles)',
            '# =====================================================================',
            ''
        ]

        # 1. Serialize Sub-Resources
        for res in self.sub_resources:
            r_type = res["type"]
            r_id = res["id"]
            lines.append(f'[sub_resource type="{r_type}" id="{r_id}"]')

            if r_type == "StandardMaterial3D":
                alb = res.get("albedo_color", [1, 1, 1, 1])
                lines.append(f'albedo_color = Color({alb[0]}, {alb[1]}, {alb[2]}, {alb[3]})')
                lines.append(f'metallic = {res.get("metallic", 0.0)}')
                lines.append(f'roughness = {res.get("roughness", 0.5)}')
                if res.get("emission_enabled"):
                    em = res.get("emission", [1, 1, 1, 1])
                    lines.append('emission_enabled = true')
                    lines.append(f'emission = Color({em[0]}, {em[1]}, {em[2]}, {em[3]})')
                    lines.append(f'emission_energy_multiplier = {res.get("emission_energy_multiplier", 1.0)}')
                if res.get("transparency", 0) > 0:
                    lines.append(f'transparency = {res.get("transparency", 1)}')
                if res.get("subsurface_scattering_enabled"):
                    lines.append('subsurface_scattering_enabled = true')

            elif r_type == "CylinderMesh":
                lines.append(f'top_radius = {res.get("top_radius", 1.0)}')
                lines.append(f'bottom_radius = {res.get("bottom_radius", 1.0)}')
                lines.append(f'height = {res.get("height", 0.5)}')
                lines.append(f'radial_segments = {res.get("radial_segments", 12)}')

            elif r_type == "PrismMesh":
                lines.append(f'left_to_right = {res.get("left_to_right", 0.5)}')
                sz = res.get("size", [1, 2, 1])
                lines.append(f'size = Vector3({sz[0]}, {sz[1]}, {sz[2]})')

            elif r_type == "SphereMesh":
                lines.append(f'radius = {res.get("radius", 0.5)}')
                lines.append(f'height = {res.get("height", 1.0)}')

            elif r_type == "ParticleProcessMaterial":
                lines.append(f'emission_shape = {res.get("emission_shape", 1)}')
                if "emission_sphere_radius" in res:
                    lines.append(f'emission_sphere_radius = {res["emission_sphere_radius"]}')
                lines.append(f'spread = {res.get("spread", 45.0)}')
                lines.append(f'initial_velocity_min = {res.get("initial_velocity_min", 1.0)}')
                lines.append(f'initial_velocity_max = {res.get("initial_velocity_max", 2.0)}')
                col = res.get("color", [1, 1, 1, 1])
                lines.append(f'color = Color({col[0]}, {col[1]}, {col[2]}, {col[3]})')

            lines.append('')

        # 2. Serialize Nodes
        for node in self.nodes:
            n_name = node["name"]
            n_type = node["type"]
            n_parent = node["parent"]

            if n_parent == ".":
                lines.append(f'[node name="{n_name}" type="{n_type}"]')
            else:
                lines.append(f'[node name="{n_name}" type="{n_type}" parent="{n_parent}"]')

            if node.get("transform"):
                t = node["transform"]
                lines.append(f'transform = Transform3D({t[0]}, {t[1]}, {t[2]}, {t[3]}, {t[4]}, {t[5]}, {t[6]}, {t[7]}, {t[8]}, {t[9]}, {t[10]}, {t[11]})')

            if node.get("mesh_ref"):
                lines.append(f'mesh = SubResource("{node["mesh_ref"]}")')
            if node.get("material_ref"):
                lines.append(f'surface_material_override/0 = SubResource("{node["material_ref"]}")')
            if node.get("process_material_ref"):
                lines.append(f'process_material = SubResource("{node["process_material_ref"]}")')

            for k, v in node.get("extra_props", {}).items():
                if isinstance(v, bool):
                    lines.append(f'{k} = {"true" if v else "false"}')
                elif isinstance(v, (int, float)):
                    lines.append(f'{k} = {v}')
                elif isinstance(v, list) and len(v) == 4:
                    lines.append(f'{k} = Color({v[0]}, {v[1]}, {v[2]}, {v[3]})')
                elif isinstance(v, list) and len(v) == 3:
                    lines.append(f'{k} = Vector3({v[0]}, {v[1]}, {v[2]})')
                else:
                    lines.append(f'{k} = "{v}"')

            lines.append('')

        return "\n".join(lines)


# ------------------------------------------------------------------------------
# 6. HIGH-LEVEL PROCEDURAL SCENE GENERATOR FOR KRYSTAL GAME ELEMENTS
# ------------------------------------------------------------------------------
def generate_procedural_godot_arena(
    arena_name: str = "KrystalTacticalArena",
    radius: int = 2,
    include_particles: bool = True
) -> str:
    """Generates a fully textured and lit Godot 4 arena with hex tiles and tribal bases."""
    builder = Godot4ProceduralSceneBuilder(f"krystal_{arena_name.lower()}")

    # 1. Register PBR Materials
    cryst_mat = builder.register_sub_resource(PROCEDURAL_PBR_PRESETS["crystal_facet_pbr"])
    slime_mat = builder.register_sub_resource(PROCEDURAL_PBR_PRESETS["toxic_slime_organic"])
    druid_mat = builder.register_sub_resource(PROCEDURAL_PBR_PRESETS["druid_ancient_bark"])
    basalt_mat = builder.register_sub_resource(PROCEDURAL_PBR_PRESETS["basalt_fortress_stone"])

    # 2. Register Procedural Meshes
    hex_mesh = builder.register_sub_resource(PROCEDURAL_MESH_PRESETS["hex_column_mesh"])
    shard_mesh = builder.register_sub_resource(PROCEDURAL_MESH_PRESETS["crystal_shard_mesh"])
    slime_mesh = builder.register_sub_resource(PROCEDURAL_MESH_PRESETS["slime_blob_mesh"])
    monolith_mesh = builder.register_sub_resource(PROCEDURAL_MESH_PRESETS["monolith_slab_mesh"])

    # 3. Register Particles
    if include_particles:
        mana_particles = builder.register_sub_resource(PROCEDURAL_PARTICLE_PRESETS["mana_dust_particles"])

    # 4. Root Scene Node
    builder.add_node(arena_name, "Node3D", parent=".")

    # 5. Sun and Environment
    sun_cfg = PROCEDURAL_ENVIRONMENT_CONFIG["sun_light"]
    builder.add_node(
        "TacticalSun",
        "DirectionalLight3D",
        parent=".",
        transform=sun_cfg["transform"],
        extra_props={"light_color": sun_cfg["color"], "light_energy": sun_cfg["energy"], "shadow_enabled": True}
    )

    # 6. Hex Grid Container
    builder.add_node("HexGrid", "Node3D", parent=".")

    # 7. Generate Hex Tiles
    hex_w = 1.732
    hex_h = 1.5
    for q in range(-radius, radius + 1):
        r1 = max(-radius, -q - radius)
        r2 = min(radius, -q + radius)
        for r in range(r1, r2 + 1):
            x = round(hex_w * (q + r * 0.5), 2)
            z = round(hex_h * r, 2)
            y = 0.0

            # Assign material based on sector quadrant
            tile_mat = basalt_mat
            if q > 0 and r >= 0:
                tile_mat = cryst_mat
            elif q < 0:
                tile_mat = slime_mat
            elif r < 0:
                tile_mat = druid_mat

            builder.add_node(
                f"Hex_{q}_{r}",
                "MeshInstance3D",
                parent="HexGrid",
                transform=[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, x, y, z],
                mesh_ref=hex_mesh,
                material_ref=tile_mat
            )

    # 8. Crystal Spire at North Hex (0, 2)
    builder.add_node(
        "CrystalSpireBase",
        "Node3D",
        parent=".",
        transform=[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.73, 0.4, 3.0]
    )
    builder.add_node(
        "SpireShard",
        "MeshInstance3D",
        parent="CrystalSpireBase",
        transform=[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.4, 0.0],
        mesh_ref=shard_mesh,
        material_ref=cryst_mat
    )
    builder.add_node(
        "SpireLight",
        "OmniLight3D",
        parent="CrystalSpireBase",
        transform=[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 2.5, 0.0],
        extra_props={"light_color": [0.0, 1.0, 1.0, 1.0], "light_energy": 2.5, "omni_range": 6.0}
    )

    if include_particles:
        builder.add_node(
            "SpireAuraParticles",
            "GPUParticles3D",
            parent="CrystalSpireBase",
            transform=[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.5, 0.0],
            process_material_ref=mana_particles,
            extra_props={"amount": 80, "lifetime": 1.5}
        )

    return builder.serialize_to_tscn()
