#!/usr/bin/env python3
"""
Unit tests for Poslední Kmen Visual Fidelity Engine.
Verifies:
  1. Godot 4.x PBR Shaders (crystal_ice_prism, toxic_caustic_slime, aetheric_vortex_swirl).
  2. Godot 4.x Citadel Scene & Controller (FloatingCitadelWellOfSouls.tscn, FloatingCitadelController.gd).
  3. Interactive Web Visual Studio (posledni_kmen_visual_studio.html).
  4. Server HTTP endpoints (/visual-studio, /api/visual/tribes, /api/sensory/simulate_shockwave).
  5. Strict adherence to VITAL_MAX_HP == 6.
"""

import unittest
from pathlib import Path
from krystal_web_hub.economic_engine.game_service import VITAL_MAX_HP
from krystal_web_hub.economic_engine.sensory_shockwave_engine import GLOBAL_SENSORY_SHOCKWAVE_ENGINE

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent


class TestPosledniKmenVisualEngine(unittest.TestCase):

    def setUp(self):
        self.godot_shaders_dir = WORKSPACE_ROOT / "godot_project" / "shaders"
        self.godot_scenes_dir = WORKSPACE_ROOT / "godot_project" / "scenes"
        self.godot_scripts_dir = WORKSPACE_ROOT / "godot_project" / "scripts"
        self.static_dir = WORKSPACE_ROOT / "krystal_web_hub" / "static"

    def test_vital_max_hp_invariant(self):
        """Invariant: VITAL_MAX_HP must be strictly 6."""
        self.assertEqual(VITAL_MAX_HP, 6)

    def test_crystal_ice_prism_shader(self):
        """Verify Crystal biome ice prism shader exists and contains PBR uniforms."""
        shader_path = self.godot_shaders_dir / "crystal_ice_prism.gdshader"
        self.assertTrue(shader_path.exists(), f"Missing shader: {shader_path}")
        content = shader_path.read_text(encoding="utf-8")
        self.assertIn("shader_type spatial", content)
        self.assertIn("refraction_strength", content)
        self.assertIn("chromatic_dispersion", content)
        self.assertIn("facet_scale", content)
        self.assertIn("roughness", content)
        self.assertIn("ice_color", content)
        self.assertIn("aether_glow", content)

    def test_toxic_caustic_slime_shader(self):
        """Verify Toxic biome caustic slime shader exists and contains fluid wave uniforms."""
        shader_path = self.godot_shaders_dir / "toxic_caustic_slime.gdshader"
        self.assertTrue(shader_path.exists(), f"Missing shader: {shader_path}")
        content = shader_path.read_text(encoding="utf-8")
        self.assertIn("shader_type spatial", content)
        self.assertIn("wave_amplitude", content)
        self.assertIn("bubble_speed", content)
        self.assertIn("caustic_scale", content)
        self.assertIn("toxic_color", content)
        self.assertIn("mire_purple", content)

    def test_aetheric_vortex_shader(self):
        """Verify Studna Duší aetheric vortex swirl shader exists and has polar swirl."""
        shader_path = self.godot_shaders_dir / "aetheric_vortex_swirl.gdshader"
        self.assertTrue(shader_path.exists(), f"Missing shader: {shader_path}")
        content = shader_path.read_text(encoding="utf-8")
        self.assertIn("shader_type spatial", content)
        self.assertIn("swirl_speed", content)
        self.assertIn("spiral_arms", content)
        self.assertIn("core_color", content)
        self.assertIn("rim_color", content)
        self.assertIn("vortex_depth", content)

    def test_floating_citadel_scene_and_controller(self):
        """Verify 3D scene and camera controller script exist and link biomes."""
        controller_path = self.godot_scripts_dir / "FloatingCitadelController.gd"
        scene_path = self.godot_scenes_dir / "FloatingCitadelWellOfSouls.tscn"

        self.assertTrue(controller_path.exists(), f"Missing script: {controller_path}")
        self.assertTrue(scene_path.exists(), f"Missing scene: {scene_path}")

        ctrl_content = controller_path.read_text(encoding="utf-8")
        self.assertIn("switch_biome", ctrl_content)
        self.assertIn("well_vortex", ctrl_content)
        self.assertIn("crystal_spire_node", ctrl_content)
        self.assertIn("toxic_pool_node", ctrl_content)
        self.assertIn("druid_grove_node", ctrl_content)

        scene_content = scene_path.read_text(encoding="utf-8")
        self.assertIn("StudnaDusiVortex", scene_content)
        self.assertIn("NorthernCrystalBiome", scene_content)
        self.assertIn("SouthwestToxicBiome", scene_content)
        self.assertIn("SoutheastDruidBiome", scene_content)
        self.assertIn("VortexSoulParticles", scene_content)

    def test_posledni_kmen_visual_studio_html(self):
        """Verify Web Visual Studio contains canvas, radar chart, tribes and leaderboards."""
        html_path = self.static_dir / "posledni_kmen_visual_studio.html"
        self.assertTrue(html_path.exists(), f"Missing HTML: {html_path}")
        content = html_path.read_text(encoding="utf-8")
        self.assertIn("POSLEDNÍ KMEN", content)
        self.assertIn("Studna Duší", content)
        self.assertIn("islandCanvas", content)
        self.assertIn("radarPoly", content)
        self.assertIn("KRABO", content)
        self.assertIn("LUKY", content)
        self.assertIn("SLEEPING GUINEA PIG", content)
        self.assertIn("CRYSTAL", content)
        self.assertIn("TOXIC", content)
        self.assertIn("DRUID", content)
        self.assertIn("VITAL_MAX_HP = 6", content)

    def test_sensory_shockwave_engine_integration(self):
        """Verify sensory wave propagation calculates overpressure and trench dampening."""
        combatants = [
            {"unit_id": "c_open", "name": "Open Scout", "current_hex": [1, 0], "current_hp": 6},
            {"unit_id": "c_trenched", "name": "Trenched Sapper", "current_hex": [2, 0], "current_hp": 6}
        ]
        res = GLOBAL_SENSORY_SHOCKWAVE_ENGINE.trigger_tactical_detonation(
            epicenter_hex=[0, 0],
            combatants=combatants,
            cover_map={"1_0": 0.0, "2_0": 0.85},
            trenches=["2_0"],
            round_number=1,
            peak_intensity_kpa=125.0,
            blast_radius_m=9.0
        )
        self.assertEqual(res["vital_max_hp"], 6)
        self.assertEqual(res["shockwave_spec"]["epicenter_hex"], [0, 0])
        self.assertIn("evaluations", res)
        # Trenched unit should take significantly reduced shock / disruption
        trenched = next((c for c in res["evaluations"] if c["unit_id"] == "c_trenched"), None)
        open_unit = next((c for c in res["evaluations"] if c["unit_id"] == "c_open"), None)
        self.assertIsNotNone(trenched)
        self.assertIsNotNone(open_unit)
        self.assertTrue(trenched["wave_reception"]["is_sheltered_by_trench"])
        self.assertEqual(trenched["wave_reception"]["trench_dampening_pct"], 80.0)
        self.assertFalse(open_unit["wave_reception"]["is_sheltered_by_trench"])

    def test_server_visual_tribes_route(self):
        """Verify GET /api/visual/tribes route produces expected data and respects VITAL_MAX_HP == 6."""
        import json
        from io import BytesIO
        from krystal_web_hub.server import KrystalHubHandler

        class DummyServer:
            pass

        class MockHandler(KrystalHubHandler):
            def __init__(self, path):
                self.path = path
                self.wfile = BytesIO()
                self.headers = {}
                self.rfile = BytesIO()
                self.server = DummyServer()
                self.client_address = ('127.0.0.1', 8080)
                self.requestline = f'GET {path} HTTP/1.1'
                self.command = 'GET'
                self.close_connection = True
            def send_response(self, code, message=None):
                self._code = code
            def send_header(self, keyword, value):
                pass
            def end_headers(self):
                pass

        h = MockHandler('/api/visual/tribes')
        h.do_GET()
        payload = json.loads(h.wfile.getvalue().decode('utf-8'))
        self.assertEqual(payload["status"], "OK")
        self.assertEqual(payload["vital_max_hp"], 6)
        self.assertIn("crystal", payload["tribes"])
        self.assertIn("toxic", payload["tribes"])
        self.assertIn("druid", payload["tribes"])
        self.assertEqual(len(payload["leaderboards"]), 3)
        self.assertEqual(payload["leaderboards"][0]["player"], "KRABO")


if __name__ == "__main__":
    unittest.main()
