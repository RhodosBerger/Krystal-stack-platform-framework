"""
godot_asset_and_camera_pipeline.py
Krystal-Stack Economic & Holographic Engine Subsystem.
Provides automated package manager capabilities for Godot 4.x assets (3D models, animation clips),
camera perspective / orthographic projection templates, and projection matrix transformations.

Invariant: VITAL_MAX_HP = 6
"""

import os
import math
import json
import urllib.request
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Any, Optional

VITAL_MAX_HP: int = 6
GOLDEN_RATIO: float = 1.61803398875
INV_GOLDEN_RATIO: float = 0.61803398875

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
GODOT_DIR = os.path.join(BASE_DIR, "godot_project")
MODELS_DIR = os.path.join(GODOT_DIR, "assets", "models")
TEMPLATES_DIR = os.path.join(GODOT_DIR, "scenes", "templates")
SCRIPTS_DIR = os.path.join(GODOT_DIR, "scripts")

@dataclass
class GodotAnimatedModel:
    model_id: str
    name: str
    filename: str
    category: str
    file_size_bytes: int
    animations: List[str]
    description: str
    vital_max_hp: int = VITAL_MAX_HP
    res_path: str = ""

    def __post_init__(self):
        if self.vital_max_hp > VITAL_MAX_HP:
            self.vital_max_hp = VITAL_MAX_HP
        if not self.res_path:
            self.res_path = f"res://assets/models/{self.filename}"

@dataclass
class CameraPresetConfig:
    preset_id: str
    name: str
    projection_type: str  # PERSPECTIVE or ORTHOGRAPHIC
    pitch_deg: float
    yaw_deg: float
    distance: float
    fov_deg: Optional[float] = None
    ortho_size: Optional[float] = None
    script_path: str = ""
    template_tscn: str = ""
    vital_max_hp: int = VITAL_MAX_HP

    def __post_init__(self):
        if self.vital_max_hp > VITAL_MAX_HP:
            self.vital_max_hp = VITAL_MAX_HP

class GodotAssetAndCameraPipeline:
    """
    Core pipeline managing Godot 3D animated assets, camera presets,
    package manager downloads, and mathematical projection transitions.
    """

    def __init__(self):
        self._models: Dict[str, GodotAnimatedModel] = {}
        self._camera_presets: Dict[str, CameraPresetConfig] = {}
        self._init_camera_presets()
        self._load_local_models()

    def _init_camera_presets(self):
        self._camera_presets = {
            "perspective_action_follow": CameraPresetConfig(
                preset_id="perspective_action_follow",
                name="3rd Person Action Perspective (Orbital SpringArm)",
                projection_type="PERSPECTIVE",
                pitch_deg=-20.0,
                yaw_deg=0.0,
                distance=8.0,
                fov_deg=75.0,
                script_path="res://scripts/PerspectiveCameraController.gd",
                template_tscn="res://scenes/templates/CameraPerspectiveRig.tscn"
            ),
            "perspective_cinematic_wide": CameraPresetConfig(
                preset_id="perspective_cinematic_wide",
                name="Cinematic Wide-Angle Perspective",
                projection_type="PERSPECTIVE",
                pitch_deg=-12.0,
                yaw_deg=15.0,
                distance=10.5,
                fov_deg=85.0,
                script_path="res://scripts/PerspectiveCameraController.gd",
                template_tscn="res://scenes/templates/CameraPerspectiveRig.tscn"
            ),
            "orthographic_true_isometric": CameraPresetConfig(
                preset_id="orthographic_true_isometric",
                name="True Isometric Tactical Arena (35.264° / 45°)",
                projection_type="ORTHOGRAPHIC",
                pitch_deg=-35.264,  # atan(1/sqrt(2)) in degrees
                yaw_deg=45.0,
                distance=20.0,
                ortho_size=14.0,
                script_path="res://scripts/OrthographicCameraController.gd",
                template_tscn="res://scenes/templates/CameraOrthographicRig.tscn"
            ),
            "orthographic_dimetric_military": CameraPresetConfig(
                preset_id="orthographic_dimetric_military",
                name="Military Dimetric Projection (30° / 45°)",
                projection_type="ORTHOGRAPHIC",
                pitch_deg=-30.0,
                yaw_deg=45.0,
                distance=20.0,
                ortho_size=14.0,
                script_path="res://scripts/OrthographicCameraController.gd",
                template_tscn="res://scenes/templates/CameraOrthographicRig.tscn"
            ),
            "orthographic_topdown_tactical": CameraPresetConfig(
                preset_id="orthographic_topdown_tactical",
                name="Top-Down Tactical Hex Grid (89.9°)",
                projection_type="ORTHOGRAPHIC",
                pitch_deg=-89.9,
                yaw_deg=0.0,
                distance=25.0,
                ortho_size=18.0,
                script_path="res://scripts/OrthographicCameraController.gd",
                template_tscn="res://scenes/templates/CameraOrthographicRig.tscn"
            ),
            "dual_mode_rig": CameraPresetConfig(
                preset_id="dual_mode_rig",
                name="Unified DualCameraRig3D (Seamless Persp <-> Ortho)",
                projection_type="DUAL_HYBRID",
                pitch_deg=-25.0,
                yaw_deg=0.0,
                distance=8.0,
                fov_deg=75.0,
                ortho_size=14.0,
                script_path="res://scripts/DualCameraRig3D.gd",
                template_tscn="res://scenes/templates/DualCameraRig3D.tscn"
            )
        }

    def _load_local_models(self):
        # Register standard animated assets
        manifest_file = os.path.join(MODELS_DIR, "models_manifest.json")
        if os.path.exists(manifest_file):
            try:
                with open(manifest_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    for m in data.get("models", []):
                        self._models[m["id"]] = GodotAnimatedModel(
                            model_id=m["id"],
                            name=m["name"],
                            filename=m["filename"],
                            category=m["category"],
                            file_size_bytes=m["file_size_bytes"],
                            animations=m["animations"],
                            description=m["description"],
                            vital_max_hp=VITAL_MAX_HP,
                            res_path=m.get("local_path", f"res://assets/models/{m['filename']}")
                        )
            except Exception as e:
                print(f"[Godot Pipeline] Warning loading manifest: {e}")

        # Fallback if manifest not yet written
        if not self._models:
            self._models["robot_expressive"] = GodotAnimatedModel(
                model_id="robot_expressive",
                name="Krystal Cybernetic Robot Expressive",
                filename="RobotExpressive.glb",
                category="Character",
                file_size_bytes=463988,
                animations=["Idle", "Walking", "Running", "Dance", "Death", "Sitting", "Standing", "Jump", "Yes", "No", "Wave", "Punch", "ThumbsUp"],
                description="Rigged humanoid robot with 13 rich skeletal animation clips, ideal for Godot 4.x 3D character controllers."
            )
            self._models["cesium_man"] = GodotAnimatedModel(
                model_id="cesium_man",
                name="Cesium Humanoid Explorer",
                filename="CesiumMan.glb",
                category="Character",
                file_size_bytes=490956,
                animations=["Anim_0_Walk"],
                description="Rigged humanoid walker with standard bone hierarchy and walking cycle animation."
            )
            self._models["fox_quadruped"] = GodotAnimatedModel(
                model_id="fox_quadruped",
                name="Sovereign Fox Quadruped",
                filename="Fox.glb",
                category="Creature",
                file_size_bytes=162852,
                animations=["Survey", "Walk", "Run"],
                description="Rigged quadruped mammal with Survey (idle alert), Walk, and Run animation cycles."
            )

    def get_models_catalog(self) -> Dict[str, Any]:
        """Returns the list of installed and available 3D animated models."""
        return {
            "vital_max_hp_rule": VITAL_MAX_HP,
            "models_count": len(self._models),
            "models": [asdict(m) for m in self._models.values()]
        }

    def get_camera_templates(self) -> Dict[str, Any]:
        """Returns camera scripts and .tscn templates for perspective & orthographic projections."""
        return {
            "vital_max_hp_rule": VITAL_MAX_HP,
            "presets_count": len(self._camera_presets),
            "presets": [asdict(p) for p in self._camera_presets.values()],
            "scripts": [
                {
                    "name": "PerspectiveCameraController.gd",
                    "path": "res://scripts/PerspectiveCameraController.gd",
                    "description": "Orbital 3rd person follow camera with FOV zoom, camera trauma shake, and spring arm."
                },
                {
                    "name": "OrthographicCameraController.gd",
                    "path": "res://scripts/OrthographicCameraController.gd",
                    "description": "Tactical isometric & orthographic camera with size zoom, WASD / edge pan, and arena bounds clamping."
                },
                {
                    "name": "DualCameraRig3D.gd",
                    "path": "res://scripts/DualCameraRig3D.gd",
                    "description": "Unified rig that dynamically morphs between Perspective Action and Isometric Orthographic views."
                },
                {
                    "name": "GodotPackageManager.gd",
                    "path": "res://scripts/GodotPackageManager.gd",
                    "description": "In-engine asset manager to inspect installed models and instantiate scenes."
                }
            ],
            "scenes": [
                {
                    "name": "CameraPerspectiveRig.tscn",
                    "path": "res://scenes/templates/CameraPerspectiveRig.tscn",
                    "type": "Perspective Camera Rig Template"
                },
                {
                    "name": "CameraOrthographicRig.tscn",
                    "path": "res://scenes/templates/CameraOrthographicRig.tscn",
                    "type": "Isometric Orthographic Camera Rig Template"
                },
                {
                    "name": "DualCameraRig3D.tscn",
                    "path": "res://scenes/templates/DualCameraRig3D.tscn",
                    "type": "Unified Dual Mode Rig Template"
                },
                {
                    "name": "AnimatedModelShowcase.tscn",
                    "path": "res://scenes/AnimatedModelShowcase.tscn",
                    "type": "Master Showcase Scene (Robot, Fox, CesiumMan + Dual Camera Rig)"
                }
            ]
        }

    def calculate_equivalent_ortho_size(self, fov_deg: float, distance: float) -> float:
        """
        Calculates the equivalent orthographic size for a given perspective FOV and distance
        so that an object at `distance` has the identical screen height:
        size = 2 * distance * tan(fov / 2)
        """
        rad = math.radians(fov_deg)
        return round(2.0 * distance * math.tan(rad / 2.0), 4)

    def calculate_equivalent_perspective_fov(self, ortho_size: float, distance: float) -> float:
        """
        Calculates equivalent perspective FOV for an orthographic size at distance:
        fov = 2 * atan(size / (2 * distance))
        """
        if distance <= 0.001:
            return 75.0
        val = ortho_size / (2.0 * distance)
        return round(math.degrees(2.0 * math.atan(val)), 2)

    def compute_projection_matrices(self, fov_deg: float, ortho_size: float, aspect: float = 16.0 / 9.0, near: float = 0.1, far: float = 500.0) -> Dict[str, Any]:
        """
        Computes 4x4 projection matrices for both Perspective and Orthographic modes.
        """
        # Perspective matrix
        rad = math.radians(fov_deg)
        tan_half = math.tan(rad / 2.0)
        p_00 = 1.0 / (aspect * tan_half)
        p_11 = 1.0 / tan_half
        p_22 = -(far + near) / (far - near)
        p_23 = -(2.0 * far * near) / (far - near)
        p_32 = -1.0

        persp_matrix = [
            [round(p_00, 4), 0.0, 0.0, 0.0],
            [0.0, round(p_11, 4), 0.0, 0.0],
            [0.0, 0.0, round(p_22, 4), round(p_23, 4)],
            [0.0, 0.0, p_32, 0.0]
        ]

        # Orthographic matrix
        o_00 = 2.0 / (ortho_size * aspect)
        o_11 = 2.0 / ortho_size
        o_22 = -2.0 / (far - near)
        o_23 = -(far + near) / (far - near)

        ortho_matrix = [
            [round(o_00, 4), 0.0, 0.0, 0.0],
            [0.0, round(o_11, 4), 0.0, 0.0],
            [0.0, 0.0, round(o_22, 4), round(o_23, 4)],
            [0.0, 0.0, 0.0, 1.0]
        ]

        return {
            "vital_max_hp_rule": VITAL_MAX_HP,
            "aspect_ratio": aspect,
            "perspective_fov_deg": fov_deg,
            "orthographic_size": ortho_size,
            "perspective_matrix": persp_matrix,
            "orthographic_matrix": ortho_matrix,
            "near_plane": near,
            "far_plane": far
        }

    def query_godot_asset_lib(self, query: str = "camera", max_results: int = 5) -> Dict[str, Any]:
        """
        Queries official Godot Asset Library API for packages and templates.
        """
        url = f"https://godotengine.org/asset-library/api/asset?filter={urllib.request.quote(query)}&max_results={max_results}"
        req = urllib.request.Request(url, headers={"User-Agent": "Krystal-Godot-PackageManager/1.0"})
        try:
            with urllib.request.urlopen(req, timeout=5.0) as resp:
                data = json.loads(resp.read().decode('utf-8'))
                assets = []
                for item in data.get("result", []):
                    assets.append({
                        "asset_id": item.get("asset_id"),
                        "title": item.get("title"),
                        "author": item.get("author"),
                        "category": item.get("category"),
                        "version": item.get("version_string"),
                        "godot_version": item.get("godot_version"),
                        "download_url": item.get("download_url")
                    })
                return {
                    "success": True,
                    "query": query,
                    "count": len(assets),
                    "assets": assets,
                    "vital_max_hp_rule": VITAL_MAX_HP
                }
        except Exception as e:
            return {
                "success": False,
                "query": query,
                "error": str(e),
                "vital_max_hp_rule": VITAL_MAX_HP
            }

# Global singleton
GLOBAL_GODOT_ASSET_AND_CAMERA_PIPELINE = GodotAssetAndCameraPipeline()
