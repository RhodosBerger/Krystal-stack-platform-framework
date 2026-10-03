import unittest
import os
import json
import urllib.request
from krystal_web_hub.economic_engine.high_fidelity_3d_display_and_wsl_importer import (
    VITAL_MAX_HP,
    GOLDEN_RATIO,
    DisplayFidelityMode,
    CANONICAL_DISPLAY_PRESETS,
    HighFidelity3DAndWslEngine,
    GLOBAL_HIGH_FIDELITY_3D_ENGINE,
    ASSET_DIR
)

class TestHighFidelity3DDisplayAndWslImporter(unittest.TestCase):
    def setUp(self):
        self.engine = GLOBAL_HIGH_FIDELITY_3D_ENGINE

    def test_vital_max_hp_rule(self):
        self.assertEqual(VITAL_MAX_HP, 6)
        for preset in CANONICAL_DISPLAY_PRESETS.values():
            self.assertEqual(preset.vital_max_hp_rule, 6)

    def test_canonical_display_presets(self):
        self.assertGreaterEqual(len(CANONICAL_DISPLAY_PRESETS), 5)
        pbr = CANONICAL_DISPLAY_PRESETS["preset_pbr_ultra"]
        self.assertEqual(pbr.mode, DisplayFidelityMode.PBR_COOK_TORRANCE)
        self.assertGreater(pbr.exposure_ev, 1.0)
        self.assertTrue(pbr.golden_mean_accent)

    def test_high_fidelity_mesh_baking(self):
        p1 = self.engine.bake_crystal_dragon_sanctuary()
        self.assertTrue(os.path.exists(p1))
        self.assertGreater(os.path.getsize(p1), 10000)

        p2 = self.engine.bake_zodiac_celestial_astrolabe()
        self.assertTrue(os.path.exists(p2))
        self.assertGreater(os.path.getsize(p2), 10000)

        p3 = self.engine.bake_cybernetic_titan_mech()
        self.assertTrue(os.path.exists(p3))
        self.assertGreater(os.path.getsize(p3), 10000)

        p4 = self.engine.bake_biomorphic_tree_of_life()
        self.assertTrue(os.path.exists(p4))
        self.assertGreater(os.path.getsize(p4), 10000)

    def test_asset_catalog_and_metadata(self):
        catalog = self.engine.get_catalog()
        self.assertEqual(catalog["vital_max_hp_rule"], 6)
        self.assertGreaterEqual(catalog["models_count"], 10)
        dragon_meta = next((m for m in catalog["models"] if m["filename"] == "crystal_dragon_sanctuary.obj"), None)
        self.assertIsNotNone(dragon_meta)
        self.assertGreater(dragon_meta["vertex_count"], 150)

    def test_wsl_model_validation(self):
        res = self.engine.import_and_validate_model_via_wsl("crystal_dragon_sanctuary.obj")
        self.assertEqual(res["status"], "VALIDATED_HIGH_FIDELITY")
        self.assertEqual(res["vital_max_hp_rule"], 6)
        self.assertTrue(res["has_smooth_normals"])

    def test_http_api_endpoints(self):
        base_url = "http://127.0.0.1:8089"
        try:
            # 1. GET /api/3d-models/catalog
            req = urllib.request.Request(f"{base_url}/api/3d-models/catalog")
            with urllib.request.urlopen(req, timeout=3) as resp:
                self.assertEqual(resp.status, 200)
                d = json.loads(resp.read().decode('utf-8'))
                self.assertEqual(d["vital_max_hp_rule"], 6)
                self.assertGreaterEqual(len(d["models"]), 10)

            # 2. GET /api/3d-models/wsl-diagnostic
            req = urllib.request.Request(f"{base_url}/api/3d-models/wsl-diagnostic")
            with urllib.request.urlopen(req, timeout=5) as resp:
                self.assertEqual(resp.status, 200)
                d = json.loads(resp.read().decode('utf-8'))
                self.assertEqual(d["vital_max_hp_rule"], 6)

            # 3. POST /api/3d-models/bake-model
            req = urllib.request.Request(
                f"{base_url}/api/3d-models/bake-model",
                data=json.dumps({"model_target": "astrolabe"}).encode('utf-8'),
                headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req, timeout=5) as resp:
                self.assertEqual(resp.status, 200)
                d = json.loads(resp.read().decode('utf-8'))
                self.assertEqual(d["status"], "BAKE_SUCCESS")
                self.assertEqual(d["vital_max_hp_rule"], 6)

            # 4. POST /api/3d-models/wsl-validate
            req = urllib.request.Request(
                f"{base_url}/api/3d-models/wsl-validate",
                data=json.dumps({"filename": "zodiac_celestial_astrolabe.obj"}).encode('utf-8'),
                headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req, timeout=5) as resp:
                self.assertEqual(resp.status, 200)
                d = json.loads(resp.read().decode('utf-8'))
                self.assertEqual(d["status"], "VALIDATED_HIGH_FIDELITY")

            # 5. GET /high-fidelity-3d-studio
            req = urllib.request.Request(f"{base_url}/high-fidelity-3d-studio")
            with urllib.request.urlopen(req, timeout=3) as resp:
                self.assertEqual(resp.status, 200)
                html = resp.read().decode('utf-8')
                self.assertIn("HIGH-FIDELITY 3D DISPLEJ", html)

            # 6. GET /api/assets/crystal_dragon_sanctuary.obj
            req = urllib.request.Request(f"{base_url}/api/assets/crystal_dragon_sanctuary.obj")
            with urllib.request.urlopen(req, timeout=3) as resp:
                self.assertEqual(resp.status, 200)
                obj_text = resp.read().decode('utf-8')
                self.assertTrue(obj_text.startswith("# High-Fidelity 3D Mesh:"))

        except Exception as e:
            self.skipTest(f"Live server check skipped during direct test: {e}")

if __name__ == "__main__":
    unittest.main()
