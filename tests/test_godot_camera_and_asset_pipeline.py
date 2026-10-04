import unittest
import os
import math
from krystal_web_hub.economic_engine.godot_asset_and_camera_pipeline import (
    GodotAssetAndCameraPipeline,
    GLOBAL_GODOT_ASSET_AND_CAMERA_PIPELINE,
    VITAL_MAX_HP,
    GOLDEN_RATIO
)

class TestGodotCameraAndAssetPipeline(unittest.TestCase):
    def setUp(self):
        self.pipeline = GLOBAL_GODOT_ASSET_AND_CAMERA_PIPELINE

    def test_catalog_models_and_vital_max_hp(self):
        catalog = self.pipeline.get_models_catalog()
        self.assertEqual(catalog["vital_max_hp_rule"], 6)
        self.assertGreaterEqual(catalog["models_count"], 3)
        
        model_ids = [m["model_id"] for m in catalog["models"]]
        self.assertIn("robot_expressive", model_ids)
        self.assertIn("cesium_man", model_ids)
        self.assertIn("fox_quadruped", model_ids)

        for m in catalog["models"]:
            self.assertEqual(m["vital_max_hp"], 6)
            self.assertTrue(len(m["animations"]) >= 1)
            self.assertTrue(m["file_size_bytes"] > 0)
            self.assertTrue(m["res_path"].startswith("res://assets/models/"))

    def test_camera_presets(self):
        templates = self.pipeline.get_camera_templates()
        self.assertEqual(templates["vital_max_hp_rule"], 6)
        self.assertGreaterEqual(templates["presets_count"], 5)

        preset_ids = [p["preset_id"] for p in templates["presets"]]
        self.assertIn("perspective_action_follow", preset_ids)
        self.assertIn("orthographic_true_isometric", preset_ids)
        self.assertIn("orthographic_topdown_tactical", preset_ids)
        self.assertIn("dual_mode_rig", preset_ids)

        # Check true isometric mathematical pitch
        iso = next(p for p in templates["presets"] if p["preset_id"] == "orthographic_true_isometric")
        self.assertAlmostEqual(iso["pitch_deg"], -35.264, places=2)
        self.assertEqual(iso["projection_type"], "ORTHOGRAPHIC")

    def test_fov_and_ortho_size_conversion(self):
        # Given distance 10.0 and FOV 60.0 degrees:
        # ortho_size = 2 * 10 * tan(30 deg) = 20 * (1/sqrt(3)) ≈ 11.547
        dist = 10.0
        fov = 60.0
        ortho_size = self.pipeline.calculate_equivalent_ortho_size(fov, dist)
        expected = 2.0 * dist * math.tan(math.radians(30.0))
        self.assertAlmostEqual(ortho_size, expected, places=3)

        # Roundtrip conversion
        calc_fov = self.pipeline.calculate_equivalent_perspective_fov(ortho_size, dist)
        self.assertAlmostEqual(calc_fov, fov, places=1)

    def test_projection_matrices(self):
        res = self.pipeline.compute_projection_matrices(fov_deg=75.0, ortho_size=14.0, aspect=16.0/9.0)
        self.assertEqual(res["vital_max_hp_rule"], 6)
        
        persp = res["perspective_matrix"]
        ortho = res["orthographic_matrix"]
        
        self.assertEqual(len(persp), 4)
        self.assertEqual(len(ortho), 4)
        self.assertGreater(persp[0][0], 0.0)
        self.assertGreater(persp[1][1], 0.0)
        self.assertEqual(persp[3][2], -1.0) # Standard perspective homogeneous coordinate

        self.assertGreater(ortho[0][0], 0.0)
        self.assertGreater(ortho[1][1], 0.0)
        self.assertEqual(ortho[3][3], 1.0) # Standard orthographic homogeneous coordinate

    def test_godot_scene_templates_and_scripts_on_disk(self):
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        godot_dir = os.path.join(base_dir, "godot_project")

        # Check scenes
        expected_scenes = [
            os.path.join(godot_dir, "scenes", "templates", "CameraPerspectiveRig.tscn"),
            os.path.join(godot_dir, "scenes", "templates", "CameraOrthographicRig.tscn"),
            os.path.join(godot_dir, "scenes", "templates", "DualCameraRig3D.tscn"),
            os.path.join(godot_dir, "scenes", "AnimatedModelShowcase.tscn")
        ]
        for scene_path in expected_scenes:
            self.assertTrue(os.path.exists(scene_path), f"Scene file should exist: {scene_path}")
            with open(scene_path, "r", encoding="utf-8") as f:
                content = f.read()
                self.assertIn("[gd_scene", content)

        # Check scripts
        expected_scripts = [
            os.path.join(godot_dir, "scripts", "PerspectiveCameraController.gd"),
            os.path.join(godot_dir, "scripts", "OrthographicCameraController.gd"),
            os.path.join(godot_dir, "scripts", "DualCameraRig3D.gd"),
            os.path.join(godot_dir, "scripts", "GodotPackageManager.gd")
        ]
        for script_path in expected_scripts:
            self.assertTrue(os.path.exists(script_path), f"Script file should exist: {script_path}")
            with open(script_path, "r", encoding="utf-8") as f:
                content = f.read()
                self.assertIn("VITAL_MAX_HP", content)

    def test_downloaded_3d_models_on_disk(self):
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        models_dir = os.path.join(base_dir, "godot_project", "assets", "models")
        
        expected_files = ["RobotExpressive.glb", "CesiumMan.glb", "Fox.glb", "models_manifest.json"]
        for fname in expected_files:
            p = os.path.join(models_dir, fname)
            self.assertTrue(os.path.exists(p), f"Asset should exist: {p}")
            self.assertGreater(os.path.getsize(p), 1000)

if __name__ == "__main__":
    unittest.main()
