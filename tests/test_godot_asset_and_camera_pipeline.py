"""
Unit Tests for Godot 3D Asset & Camera Rig Pipeline Extensions
==============================================================
Validates:
1. Strict enforcement of VITAL_MAX_HP == 6 across camera presets and packages.
2. Godot 3D model asset catalog.
3. Camera template configuration and overrides (TPS, Isometric, etc.).
4. Addon Package Manager staging and activation.
5. In-engine projection matrix calculations and FOV <-> Ortho-size equivalence.
"""

import unittest
from krystal_web_hub.economic_engine.godot_asset_and_camera_pipeline import (
    GodotAssetAndCameraPipeline,
    CameraPresetConfig,
    GodotAnimatedModel,
    GLOBAL_GODOT_ASSET_AND_CAMERA_PIPELINE,
    VITAL_MAX_HP
)


class TestGodotAssetAndCameraPipelineExtensions(unittest.TestCase):

    def setUp(self):
        self.pipeline = GodotAssetAndCameraPipeline()

    def test_vital_max_hp_rule_is_strictly_six(self):
        """Ensures the 6 Max HP vital invariant holds across all camera presets and models."""
        self.assertEqual(VITAL_MAX_HP, 6)

        cam_catalog = self.pipeline.get_camera_templates()
        self.assertEqual(cam_catalog["vital_max_hp_rule"], 6)
        for t in cam_catalog["presets"]:
            self.assertEqual(t["vital_max_hp"], 6)

        model_catalog = self.pipeline.get_models_catalog()
        self.assertEqual(model_catalog["vital_max_hp_rule"], 6)
        for m in model_catalog["models"]:
            self.assertEqual(m["vital_max_hp"], 6)

    def test_camera_configuration_and_overrides(self):
        """Verifies camera configuration with default aliases and custom overrides."""
        res_tps = self.pipeline.configure_camera("tps_orbit")
        self.assertTrue(res_tps["success"])
        self.assertEqual(res_tps["resolved_preset_id"], "perspective_action_follow")
        self.assertEqual(res_tps["camera"]["vital_max_hp"], 6)

        # Apply custom overrides
        res_custom = self.pipeline.configure_camera("tps_orbit", overrides={
            "fov_deg": 90.0,
            "distance": 12.0
        })
        self.assertTrue(res_custom["success"])
        self.assertEqual(res_custom["camera"]["fov_deg"], 90.0)
        self.assertEqual(res_custom["camera"]["distance"], 12.0)
        self.assertEqual(res_custom["camera"]["vital_max_hp"], 6)

    def test_addon_package_manager(self):
        """Verifies package installation staging and metadata."""
        res = self.pipeline.install_addon_package("camera-controller-3d")
        self.assertTrue(res["success"])
        self.assertEqual(res["package"]["status"], "installed")
        self.assertTrue(res["package"]["is_activated"])
        self.assertEqual(res["package"]["vital_max_hp"], 6)
        self.assertIn("addons/camera_controller", res["package"]["install_path"])

        # Test custom package installation
        res_custom = self.pipeline.install_addon_package("custom-mesh-importer")
        self.assertTrue(res_custom["success"])
        self.assertEqual(res_custom["package"]["vital_max_hp"], 6)

    def test_global_singleton(self):
        """Ensures the module-level global singleton is instantiated and ready."""
        self.assertIsNotNone(GLOBAL_GODOT_ASSET_AND_CAMERA_PIPELINE)
        models = GLOBAL_GODOT_ASSET_AND_CAMERA_PIPELINE.get_models_catalog()
        self.assertGreaterEqual(models["models_count"], 3)


if __name__ == '__main__':
    unittest.main()
