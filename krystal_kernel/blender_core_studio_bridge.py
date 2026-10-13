#!/usr/bin/env python3
"""
==============================================================================
KRYSTAL-STACK: BLENDER CORE STUDIO BRIDGE & ASSET IMPORTER
==============================================================================
Module: krystal_kernel/blender_core_studio_bridge.py
Description:
    Bridges the Blender Modifier Core Engine, Mimicry Compositor recipes,
    procedural 3D mesh synthesis, and Blender 4.x Python automation scripts (bpy)
    directly into the Krystal Studio ecosystem.

Core Capabilities:
    1. Import Blender Modifier Stacks (Array, Mirror, Bevel, Displace, Deform, Solidify)
       into Studio-compatible live SDF shaders and WebGL parameters.
    2. Import 13 Real-World Mimicry Recipes into complete Studio 3D Asset Bundles.
    3. Import 9 Environmental Game Scenes with camera trajectories and lighting.
    4. Generate executable Blender 4.x Python scripts (`bpy`) with modifiers,
       materials, and trajectory keyframes.
    5. Procedurally synthesize Wavefront .OBJ 3D polygonal meshes for Studio viewports.
    6. Export 10-block AI Studio Parametric Render Set (NPRS v3.0) prompts.

Constitutional Invariant:
    assert VITAL_MAX_HP == 6
==============================================================================
"""

import math
import sys
import os
import json
import time
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple, Callable

from krystal_kernel.scene_constructor_presets import VITAL_MAX_HP
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
from mimicry_engine.mimic_recipes import (
    RECIPES_CATALOG, MimicObject, MimicPart, get_recipe, list_recipes
)
from mimicry_engine.scene_composer import (
    SCENES_CATALOG, GameScene, SceneActor, CameraTrajectory,
    OrbitalTrajectory, FlythroughSplineTrajectory, MortarBallisticCameraTrajectory,
    UrbanSpatialCompositionRules, get_scene, list_scenes
)

# Alias ModifierStack for Blender Core branding
BlenderModifierStack = ModifierStack


@dataclass
class StudioAssetBundle:
    """
    Standardized 3D asset bundle representing an imported Blender Core object
    in Studio with mesh geometry, modifier stacks, shader code, and bpy scripts.
    """
    asset_id: str
    name: str
    category: str
    description: str
    parts_count: int
    parts: List[Dict[str, Any]]
    modifiers: List[Dict[str, Any]]
    obj_data: str
    vertex_count: int
    face_count: int
    blender_bpy_script: str
    godot_tscn: str
    ai_studio_prompt: str
    vital_max_hp: int = VITAL_MAX_HP
    timestamp: float = field(default_factory=time.time)

    @property
    def recipe_id(self) -> str:
        return self.asset_id

    def to_dict(self) -> Dict[str, Any]:
        return {
            "asset_id": self.asset_id,
            "recipe_id": self.asset_id,
            "name": self.name,
            "category": self.category,
            "description": self.description,
            "parts_count": self.parts_count,
            "parts": self.parts,
            "modifiers": self.modifiers,
            "vertex_count": self.vertex_count,
            "face_count": self.face_count,
            "obj_data": self.obj_data,
            "blender_bpy_script": self.blender_bpy_script,
            "godot_tscn": self.godot_tscn,
            "ai_studio_prompt": self.ai_studio_prompt,
            "vital_max_hp": self.vital_max_hp,
            "timestamp": self.timestamp,
        }


@dataclass
class StudioSceneBundle:
    """
    Standardized multi-actor scene bundle imported into Studio with lighting,
    camera trajectories, actors, and urban spatial verification metrics.
    """
    scene_id: str
    title: str
    genre: str
    description: str
    actors_count: int
    actors: List[Dict[str, Any]]
    camera_trajectory: Dict[str, Any]
    ambient_light: List[float]
    fog_color: List[float]
    composition_rules_pass: bool
    composition_score: float
    blender_bpy_script: str
    godot_tscn: str
    vital_max_hp: int = VITAL_MAX_HP

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scene_id": self.scene_id,
            "title": self.title,
            "genre": self.genre,
            "description": self.description,
            "actors_count": self.actors_count,
            "actors": self.actors,
            "camera_trajectory": self.camera_trajectory,
            "ambient_light": self.ambient_light,
            "fog_color": self.fog_color,
            "composition_rules_pass": self.composition_rules_pass,
            "composition_score": self.composition_score,
            "blender_bpy_script": self.blender_bpy_script,
            "godot_tscn": self.godot_tscn,
            "vital_max_hp": self.vital_max_hp,
        }


class BlenderCoreStudioBridge:
    """
    Unified Bridge & Importer bringing Blender Core into the Krystal Studio ecosystem.
    """

    def __init__(self):
        assert VITAL_MAX_HP == 6, f"Constitutional breach: VITAL_MAX_HP must be 6, got {VITAL_MAX_HP}"

    # ─── 1. CATALOGS ──────────────────────────────────────────────────────────

    def get_catalog(self) -> Dict[str, Any]:
        """Returns catalog of all 13 recipes, 9 scenes, and available modifiers."""
        recipes = list_recipes()
        scenes = list_scenes()
        available_modifiers = [
            {"type": "ArrayModifier", "description": "Linear and radial repetition with constant/relative offset."},
            {"type": "MirrorModifier", "description": "Bilateral and axial symmetry across X, Y, and Z axes."},
            {"type": "BooleanModifier", "description": "CSG union, difference, and intersection with polynomial smin."},
            {"type": "BevelModifier", "description": "Edge rounding, chamfering, and faceting."},
            {"type": "DisplaceModifier", "description": "Harmonic 3D noise for organic veins, surface texture, and relief."},
            {"type": "DeformModifier", "description": "Non-linear spatial deformation: Twist, Taper, and Bend."},
            {"type": "SolidifyModifier", "description": "Hollow shell generation with variable wall thickness."}
        ]
        return {
            "recipes_count": len(recipes),
            "scenes_count": len(scenes),
            "total_recipes": len(recipes),
            "total_scenes": len(scenes),
            "recipes": recipes,
            "scenes": scenes,
            "supported_modifiers": available_modifiers,
            "vital_max_hp_rule": VITAL_MAX_HP,
            "vital_max_hp": VITAL_MAX_HP,
            "blender_target_version": "4.x",
            "coordinate_system": "BLENDER_Z_UP"
        }

    # ─── 2. IMPORT RECIPE TO STUDIO ───────────────────────────────────────────

    def import_recipe_to_studio(self, recipe_id: str) -> StudioAssetBundle:
        """
        Imports a Blender Core mimic recipe into Studio format with 3D mesh OBJ,
        detailed part hierarchy, modifier stacks, and executable Blender 4.x Python script.
        """
        assert VITAL_MAX_HP == 6, f"Constitutional invariant breach: {VITAL_MAX_HP}"
        obj = get_recipe(recipe_id)
        if not obj:
            raise KeyError(f"Recipe '{recipe_id}' not found in RECIPES_CATALOG.")

        parts_summary = []
        all_modifiers = []

        for p in obj.parts:
            mods_serialized = p.modifier_stack.to_dict()
            parts_summary.append({
                "name": p.name,
                "offset": list(p.offset),
                "rotation": list(p.rotation),
                "blend_k": p.blend_k,
                "is_subtraction": p.is_subtraction,
                "modifiers": mods_serialized
            })
            for m in mods_serialized:
                m["part_name"] = p.name
                all_modifiers.append(m)

        # Procedural OBJ Mesh Generation
        obj_text, v_count, f_count = self._synthesize_recipe_obj_mesh(obj)

        # Blender 4.x Python Script Generation
        bpy_script = self.generate_blender_bpy_script(obj)

        # Godot 4.x .tscn Generation
        godot_tscn = self._generate_recipe_godot_tscn(obj)

        # AI Studio Prompt Block
        ai_studio_prompt = self._generate_ai_studio_prompt_block(obj)

        return StudioAssetBundle(
            asset_id=obj.object_id,
            name=obj.name,
            category=obj.category,
            description=obj.description,
            parts_count=len(obj.parts),
            parts=parts_summary,
            modifiers=all_modifiers,
            obj_data=obj_text,
            vertex_count=v_count,
            face_count=f_count,
            blender_bpy_script=bpy_script,
            godot_tscn=godot_tscn,
            ai_studio_prompt=ai_studio_prompt,
            vital_max_hp=VITAL_MAX_HP,
            timestamp=time.time()
        )

    # ─── 3. IMPORT SCENE TO STUDIO ────────────────────────────────────────────

    def import_scene_to_studio(self, scene_id: str) -> StudioSceneBundle:
        """
        Imports a complete Game Scene into Studio format with actors, lighting,
        camera trajectories, and urban composition validation scores.
        """
        assert VITAL_MAX_HP == 6, f"Constitutional invariant breach: {VITAL_MAX_HP}"
        sc = get_scene(scene_id)
        if not sc:
            raise KeyError(f"Scene '{scene_id}' not found in GAME_SCENES_CATALOG.")

        val_res = UrbanSpatialCompositionRules.validate_scene(sc)

        actors_data = []
        for a in sc.actors:
            actors_data.append({
                "actor_id": a.actor_id,
                "recipe_id": a.recipe_id,
                "position": list(a.position),
                "rotation": list(a.rotation),
                "scale": a.scale,
                "vital_hp": a.vital_hp
            })

        cam_data = sc.default_trajectory.to_dict() if sc.default_trajectory else {
            "type": "StaticCamera", "position": [0, 2.5, 7.5], "target": [0, 0, 0]
        }

        # Generate Scene BPY Script
        bpy_script = self._generate_scene_bpy_script(sc)

        # Export Godot Scene
        godot_tscn = sc.export_godot_tscn()

        comp_score = getattr(val_res, "compliance_score", getattr(val_res, "overall_score", 1.0))

        return StudioSceneBundle(
            scene_id=sc.scene_id,
            title=getattr(sc, "title", sc.name),
            genre=sc.genre,
            description=sc.description,
            actors_count=len(sc.actors),
            actors=actors_data,
            camera_trajectory=cam_data,
            ambient_light=list(sc.ambient_light),
            fog_color=list(sc.fog_color),
            composition_rules_pass=val_res.passed,
            composition_score=round(comp_score, 4),
            blender_bpy_script=bpy_script,
            godot_tscn=godot_tscn,
            vital_max_hp=VITAL_MAX_HP
        )

    # ─── 4. BLENDER 4.x PYTHON SCRIPT GENERATOR (BPY) ─────────────────────────

    def generate_blender_bpy_script(self, obj: MimicObject) -> str:
        """
        Generates production-grade Blender 4.x Python script using `bpy`
        to procedurally construct the object, parts, and modifier stacks in Blender.
        """
        lines = [
            f"# ==============================================================================",
            f"# KRYSTAL-STACK // BLENDER 4.x AUTOMATION SCRIPT",
            f"# Object: {obj.name} [{obj.object_id}]",
            f"# Category: {obj.category}",
            f"# Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
            f"# Constitutional Rule: assert VITAL_MAX_HP == {VITAL_MAX_HP}",
            f"# ==============================================================================",
            f"import bpy",
            f"import math",
            f"from mathutils import Vector, Euler",
            f"",
            f"def build_krystal_asset():",
            f"    # 1. Clean existing scene collections and default objects",
            f"    if bpy.context.object and bpy.context.object.mode != 'OBJECT':",
            f"        bpy.ops.object.mode_set(mode='OBJECT')",
            f"    bpy.ops.object.select_all(action='SELECT')",
            f"    bpy.ops.object.delete(use_global=False)",
            f"",
            f"    # 2. Configure Metric Units & World Settings",
            f"    scene = bpy.context.scene",
            f"    scene.unit_settings.system = 'METRIC'",
            f"    scene.unit_settings.scale_length = 1.0",
            f"    scene.unit_settings.length_unit = 'METERS'",
            f"",
            f"    # Create Root Master Collection",
            f"    coll_name = '{obj.object_id}_Collection'",
            f"    coll = bpy.data.collections.new(coll_name)",
            f"    scene.collection.children.link(coll)",
            f"",
            f"    # Root Empty Anchor",
            f"    bpy.ops.object.empty_add(type='PLAIN_AXES', location=(0, 0, 0))",
            f"    root_anchor = bpy.context.active_object",
            f"    root_anchor.name = '{obj.object_id}_Master'",
            f"    coll.objects.link(root_anchor)",
            f"    scene.collection.objects.unlink(root_anchor)",
            f""
        ]

        # Generate each part
        for idx, part in enumerate(obj.parts):
            p_name = f"Part_{idx}_{part.name}"
            ox, oy, oz = part.offset
            rx, ry, rz = part.rotation

            lines.append(f"    # ── Part: {part.name} ──")
            # Determine base primitive type from part name/hints
            name_lower = part.name.lower()
            if any(k in name_lower for k in ("sphere", "core", "dome", "visor")):
                lines.append(f"    bpy.ops.mesh.primitive_uv_sphere_add(radius=0.5, location=({ox}, {oz}, {oy}))")
            elif any(k in name_lower for k in ("cylinder", "barrel", "column", "post", "trunk")):
                lines.append(f"    bpy.ops.mesh.primitive_cylinder_add(radius=0.4, depth=1.0, location=({ox}, {oz}, {oy}))")
            elif any(k in name_lower for k in ("torus", "ring", "pod")):
                lines.append(f"    bpy.ops.mesh.primitive_torus_add(major_radius=0.6, minor_radius=0.15, location=({ox}, {oz}, {oy}))")
            elif any(k in name_lower for k in ("cone", "spire", "nozzle", "canopy", "dish")):
                lines.append(f"    bpy.ops.mesh.primitive_cone_add(radius1=0.6, radius2=0.1, depth=1.2, location=({ox}, {oz}, {oy}))")
            elif any(k in name_lower for k in ("hex", "pedestal", "kiosk")):
                lines.append(f"    bpy.ops.mesh.primitive_cylinder_add(vertices=6, radius=0.6, depth=0.8, location=({ox}, {oz}, {oy}))")
            else:
                lines.append(f"    bpy.ops.mesh.primitive_cube_add(size=1.0, location=({ox}, {oz}, {oy}))")

            lines.append(f"    part_obj = bpy.context.active_object")
            lines.append(f"    part_obj.name = '{p_name}'")
            lines.append(f"    part_obj.rotation_euler = Euler(({rx}, {rz}, {ry}))")
            lines.append(f"    part_obj.parent = root_anchor")
            lines.append(f"    coll.objects.link(part_obj)")
            lines.append(f"    scene.collection.objects.unlink(part_obj)")

            # Add Blender Modifiers from the part's modifier stack
            for mod in part.modifier_stack.modifiers:
                if not mod.enabled:
                    continue
                if isinstance(mod, ArrayModifier):
                    lines.append(f"    # Array Modifier: {mod.name}")
                    lines.append(f"    mod_arr = part_obj.modifiers.new(name='{mod.name}', type='ARRAY')")
                    lines.append(f"    mod_arr.count = {mod.count}")
                    if mod.radial:
                        lines.append(f"    # Radial Array simulated via Object Offset empty")
                        lines.append(f"    bpy.ops.object.empty_add(type='PLAIN_AXES', location=(0, 0, 0))")
                        lines.append(f"    emp_arr = bpy.context.active_object")
                        lines.append(f"    emp_arr.name = '{mod.name}_RotAnchor'")
                        lines.append(f"    emp_arr.rotation_euler.z = math.radians(360.0 / {mod.count})")
                        lines.append(f"    emp_arr.parent = part_obj")
                        lines.append(f"    mod_arr.use_relative_offset = False")
                        lines.append(f"    mod_arr.use_object_offset = True")
                        lines.append(f"    mod_arr.offset_object = emp_arr")
                    else:
                        lines.append(f"    mod_arr.use_relative_offset = True")
                        lines.append(f"    mod_arr.relative_offset_displace = ({mod.offset[0]}, {mod.offset[2]}, {mod.offset[1]})")

                elif isinstance(mod, MirrorModifier):
                    lines.append(f"    # Mirror Modifier: {mod.name}")
                    lines.append(f"    mod_mir = part_obj.modifiers.new(name='{mod.name}', type='MIRROR')")
                    lines.append(f"    mod_mir.use_axis[0] = {mod.use_x}")
                    lines.append(f"    mod_mir.use_axis[1] = {mod.use_z}")
                    lines.append(f"    mod_mir.use_axis[2] = {mod.use_y}")
                    lines.append(f"    mod_mir.mirror_object = root_anchor")

                elif isinstance(mod, BevelModifier):
                    lines.append(f"    # Bevel Modifier: {mod.name}")
                    lines.append(f"    mod_bev = part_obj.modifiers.new(name='{mod.name}', type='BEVEL')")
                    lines.append(f"    mod_bev.width = {mod.radius}")
                    lines.append(f"    mod_bev.segments = 3")

                elif isinstance(mod, DisplaceModifier):
                    lines.append(f"    # Displace Modifier: {mod.name}")
                    lines.append(f"    mod_disp = part_obj.modifiers.new(name='{mod.name}', type='DISPLACE')")
                    lines.append(f"    mod_disp.strength = {mod.strength}")
                    lines.append(f"    tex = bpy.data.textures.new('{mod.name}_Noise', type='CLOUDS')")
                    lines.append(f"    tex.noise_scale = {1.0 / max(0.1, mod.frequency)}")
                    lines.append(f"    mod_disp.texture = tex")

                elif isinstance(mod, SolidifyModifier):
                    lines.append(f"    # Solidify Modifier: {mod.name}")
                    lines.append(f"    mod_sol = part_obj.modifiers.new(name='{mod.name}', type='SOLIDIFY')")
                    lines.append(f"    mod_sol.thickness = {mod.thickness}")

                elif isinstance(mod, DeformModifier):
                    lines.append(f"    # SimpleDeform Modifier: {mod.name} ({mod.deform_type})")
                    lines.append(f"    mod_def = part_obj.modifiers.new(name='{mod.name}', type='SIMPLE_DEFORM')")
                    lines.append(f"    mod_def.deform_method = '{mod.deform_type}'")
                    lines.append(f"    mod_def.factor = {mod.factor}")

            lines.append(f"")

        # Materials & Lighting
        lines.extend([
            f"    # 3. Create Cyber-Physical PBR Material",
            f"    mat = bpy.data.materials.new(name='{obj.object_id}_Mat')",
            f"    mat.use_nodes = True",
            f"    bsdf = mat.node_tree.nodes.get('Principled BSDF')",
            f"    if bsdf:",
            f"        bsdf.inputs['Base Color'].default_value = (0.08, 0.45, 0.65, 1.0)",
            f"        bsdf.inputs['Metallic'].default_value = 0.85",
            f"        bsdf.inputs['Roughness'].default_value = 0.25",
            f"    for o in coll.objects:",
            f"        if o.type == 'MESH':",
            f"            o.data.materials.append(mat)",
            f"",
            f"    # 4. Cinematic Three-Point Camera & Lights",
            f"    bpy.ops.object.camera_add(location=(0, -6.5, 2.8), rotation=(math.radians(72), 0, 0))",
            f"    cam = bpy.context.active_object",
            f"    cam.name = '{obj.object_id}_Camera'",
            f"    scene.camera = cam",
            f"",
            f"    # Key Light",
            f"    bpy.ops.object.light_add(type='SUN', location=(4, -4, 6))",
            f"    light_sun = bpy.context.active_object",
            f"    light_sun.data.energy = 3.5",
            f"    light_sun.data.color = (1.0, 0.95, 0.85)",
            f"",
            f"    # Fill Rim Light",
            f"    bpy.ops.object.light_add(type='AREA', location=(-3, 3, 2))",
            f"    light_rim = bpy.context.active_object",
            f"    light_rim.data.energy = 150.0",
            f"    light_rim.data.color = (0.2, 0.8, 1.0)",
            f"",
            f"    print(f'[KRYSTAL] Successfully imported {obj.name} with {{len(coll.objects)}} objects.')",
            f"",
            f"if __name__ == '__main__':",
            f"    build_krystal_asset()"
        ])
        return "\n".join(lines)

    # ─── 5. SYNTHESIZE PROCEDURAL OBJ MESH ────────────────────────────────────

    def _synthesize_recipe_obj_mesh(self, obj: MimicObject) -> Tuple[str, int, int]:
        """
        Synthesizes a clean Wavefront .OBJ 3D polygonal mesh from the composite recipe
        for direct rendering inside the Studio 3D viewport.
        """
        vertices: List[Vec3] = []
        normals: List[Vec3] = []
        faces: List[List[int]] = []

        v_offset = 1  # 1-indexed for OBJ

        for p_idx, part in enumerate(obj.parts):
            px, py, pz = part.offset
            # Approximate each part with a bounding geometry box / cylinder
            name_l = part.name.lower()
            if any(k in name_l for k in ("sphere", "core", "dome")):
                sx, sy, sz = 0.45, 0.45, 0.45
            elif any(k in name_l for k in ("cylinder", "barrel", "post")):
                sx, sy, sz = 0.25, 0.8, 0.25
            elif any(k in name_l for k in ("cone", "spire", "awning")):
                sx, sy, sz = 0.5, 0.7, 0.5
            else:
                sx, sy, sz = 0.5, 0.5, 0.5

            # Apply array multiplier if present
            array_count = 1
            array_offset = (0.0, 0.0, 0.0)
            for m in part.modifier_stack.modifiers:
                if isinstance(m, ArrayModifier) and m.enabled:
                    array_count = min(4, m.count)
                    array_offset = m.offset

            for a_idx in range(array_count):
                ax = px + a_idx * array_offset[0]
                ay = py + a_idx * array_offset[1]
                az = pz + a_idx * array_offset[2]

                # 8 Box vertices
                sub_verts = [
                    (ax - sx, ay - sy, az - sz),
                    (ax + sx, ay - sy, az - sz),
                    (ax + sx, ay + sy, az - sz),
                    (ax - sx, ay + sy, az - sz),
                    (ax - sx, ay - sy, az + sz),
                    (ax + sx, ay - sy, az + sz),
                    (ax + sx, ay + sy, az + sz),
                    (ax - sx, ay + sy, az + sz),
                ]
                vertices.extend(sub_verts)

                # 6 Faces (quads: 1-based indices)
                sub_faces = [
                    [v_offset + 0, v_offset + 1, v_offset + 2, v_offset + 3],  # Back
                    [v_offset + 4, v_offset + 5, v_offset + 6, v_offset + 7],  # Front
                    [v_offset + 0, v_offset + 1, v_offset + 5, v_offset + 4],  # Bottom
                    [v_offset + 2, v_offset + 3, v_offset + 7, v_offset + 6],  # Top
                    [v_offset + 0, v_offset + 3, v_offset + 7, v_offset + 4],  # Left
                    [v_offset + 1, v_offset + 2, v_offset + 6, v_offset + 5],  # Right
                ]
                faces.extend(sub_faces)
                v_offset += 8

        # Generate OBJ lines
        obj_lines = [
            f"# Krystal-Stack Studio 3D Mesh Export",
            f"# Object: {obj.object_id} ({obj.name})",
            f"# VITAL_MAX_HP: {VITAL_MAX_HP}",
            f"o {obj.object_id}",
        ]
        for vx, vy, vz in vertices:
            # Export in Blender Z-Up coordinates: (x, z, y)
            obj_lines.append(f"v {round(vx, 4)} {round(vz, 4)} {round(vy, 4)}")

        for f in faces:
            obj_lines.append(f"f {f[0]} {f[1]} {f[2]} {f[3]}")

        return "\n".join(obj_lines), len(vertices), len(faces)

    # ─── 6. GODOT 4.x .TSCN GENERATOR ─────────────────────────────────────────

    def _generate_recipe_godot_tscn(self, obj: MimicObject) -> str:
        """Generates native Godot 4.x .tscn text format for the mimic object."""
        lines = [
            f'[gd_scene format=3 uid="uid://krystal_{obj.object_id.lower()}"]',
            f'',
            f'[node name="{obj.object_id}" type="Node3D"]',
            f'metadata/vital_max_hp = {VITAL_MAX_HP}',
            f'metadata/category = "{obj.category}"',
            f'metadata/parts_count = {len(obj.parts)}',
            f''
        ]
        for idx, part in enumerate(obj.parts):
            p_name = f"Part_{idx}_{part.name}"
            ox, oy, oz = part.offset
            rx, ry, rz = part.rotation
            lines.extend([
                f'[node name="{p_name}" type="MeshInstance3D" parent="."]',
                f'transform = Transform3D(1, 0, 0, 0, 1, 0, 0, 0, 1, {ox}, {oy}, {oz})',
                f'rotation = Vector3({rx}, {ry}, {rz})',
                f'metadata/blend_k = {part.blend_k}',
                f'metadata/is_subtraction = {str(part.is_subtraction).lower()}',
                f''
            ])
        return "\n".join(lines)

    # ─── 7. SCENE BPY SCRIPT GENERATOR ────────────────────────────────────────

    def _generate_scene_bpy_script(self, sc: GameScene) -> str:
        """Generates Blender 4.x script creating the full environmental scene."""
        lines = [
            f"# ==============================================================================",
            f"# KRYSTAL-STACK // BLENDER 4.x SCENE AUTOMATION SCRIPT",
            f"# Scene: {getattr(sc, 'title', sc.name)} [{sc.scene_id}]",
            f"# Genre: {sc.genre}",
            f"# ==============================================================================",
            f"import bpy",
            f"import math",
            f"",
            f"def build_krystal_scene():",
            f"    bpy.ops.object.select_all(action='SELECT')",
            f"    bpy.ops.object.delete()",
            f"    scene = bpy.context.scene",
            f"    scene.unit_settings.system = 'METRIC'",
            f"",
            f"    # Ground plane",
            f"    bpy.ops.mesh.primitive_plane_add(size=30, location=(0, 0, {sc.ground_level}))",
            f"    ground = bpy.context.active_object",
            f"    ground.name = 'GroundBedrock'",
            f""
        ]
        for actor in sc.actors:
            pos = actor.position
            rot = actor.rotation
            lines.append(f"    # Actor: {actor.actor_id} ({actor.recipe_id})")
            lines.append(f"    bpy.ops.mesh.primitive_cube_add(size=1.0, location=({pos[0]}, {pos[2]}, {pos[1]}))")
            lines.append(f"    act_obj = bpy.context.active_object")
            lines.append(f"    act_obj.name = '{actor.actor_id}'")
            lines.append(f"    act_obj.scale = ({actor.scale}, {actor.scale}, {actor.scale})")
            lines.append(f"    act_obj.rotation_euler = ({rot[0]}, {rot[2]}, {rot[1]})")

        lines.extend([
            f"",
            f"    # Camera Setup",
            f"    bpy.ops.object.camera_add(location=(0, -8.0, 3.5), rotation=(math.radians(70), 0, 0))",
            f"    cam = bpy.context.active_object",
            f"    scene.camera = cam",
            f"    print(f'[KRYSTAL] Scene {sc.scene_id} generated with {{len(scene.objects)}} elements.')",
            f"",
            f"if __name__ == '__main__':",
            f"    build_krystal_scene()"
        ])
        return "\n".join(lines)

    # ─── 8. AI STUDIO NPRS v3.0 PROMPT BLOCK ──────────────────────────────────

    def _generate_ai_studio_prompt_block(self, obj: MimicObject) -> str:
        """Generates structured 10-block AI Studio Parametric Render Set prompt."""
        return (
            f"================================================================================\n"
            f"[KRYSTAL AI STUDIO // NPRS v3.0 BLENDER CORE SPECIFICATION]\n"
            f"================================================================================\n"
            f"1. SceneIdentity: ID={obj.object_id} | Name={obj.name} | Category={obj.category}\n"
            f"2. WorldEnvironment: Subsystem=BlenderCore | CoordinateSystem=BLENDER_Z_UP | MetricUnits=1.0m\n"
            f"3. ProceduralStructure: ModifierStack=Array+Mirror+Bevel+Displace+Solidify | PartsCount={len(obj.parts)}\n"
            f"4. HarmonicTopology: CSG Polynomial Smooth Minimum smin(k=0.15) | Manifold=SDF_Lipschitz_1.0\n"
            f"5. TextureStack: PBR Metallic=0.85 | Roughness=0.25 | AmbientOcclusion=CavityMap\n"
            f"6. CinematicLighting: Key:Fill:Rim=(1.0:0.4:0.8) | VolumetricFog=0.035 | BlueAmberLuminescence\n"
            f"7. TypographicDynamics: OverlayMode=SINE_WAVE_FLOAT | Text='{obj.object_id}'\n"
            f"8. CompositionCamera: GoldenRatio_Phi=1.61803 | Elevation=25deg | FOV=60deg\n"
            f"9. HardwareLimits: Intel_Iris_Xe_UMA <= 256MB | Z-Fighting_Epsilon >= 0.012m | FPS >= 60\n"
            f"10. InvariantEnforcer: assert VITAL_MAX_HP == {VITAL_MAX_HP} (Constitutional Law)\n"
            f"================================================================================"
        )


# Global singleton instance
GLOBAL_BLENDER_CORE_STUDIO_BRIDGE = BlenderCoreStudioBridge()
