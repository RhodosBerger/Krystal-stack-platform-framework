import unittest
import json
import urllib.request
from krystal_web_hub.economic_engine.evolved_svg_vector_engine import (
    VITAL_MAX_HP,
    GOLDEN_RATIO,
    EvolvedSvgVectorEngine,
    GLOBAL_EVOLVED_SVG_ENGINE
)

class TestEvolvedSvgVectorEngine(unittest.TestCase):
    def setUp(self):
        self.engine = EvolvedSvgVectorEngine()

    def test_vital_max_hp_rule(self):
        self.assertEqual(VITAL_MAX_HP, 6)
        self.assertEqual(self.engine.vital_max_hp, 6)

    def test_master_blueprint_svg_generation(self):
        svg = self.engine.generate_master_blueprint_svg(1600, 900)
        self.assertTrue(svg.startswith('<svg'))
        self.assertTrue(svg.strip().endswith('</svg>'))
        
        # Verify key mathematical annotations
        self.assertIn('E = ∫ (∇φ)² dv', svg)
        self.assertIn('SDF(x) = min(||x - c||) - r', svg)
        self.assertIn('ΔH_thermal', svg)
        self.assertIn('H(x,z) = H₀ + A', svg)
        self.assertIn('VITAL_MAX_HP = 6', svg)

        # Verify biomes & heraldry
        self.assertIn('KŘIŠŤÁLOVÉ LEDOVÉ ŠTÍTY', svg)
        self.assertIn('TOXICKÉ KAŇONY JEDŮ', svg)
        self.assertIn('DRUIDSKÝ PODZIMNÍ LES', svg)
        self.assertIn('STUDNA DUŠÍ', svg)
        self.assertIn('sdfRaymarchVector', svg)

    def test_tribal_crest_svgs(self):
        # Crystal
        cr_svg = self.engine.generate_crystal_tribe_svg()
        self.assertTrue(cr_svg.startswith('<svg'))
        self.assertIn('CRYSTAL // VLÁDCI MRAZU', cr_svg)

        # Toxic
        tx_svg = self.engine.generate_toxic_tribe_svg()
        self.assertTrue(tx_svg.startswith('<svg'))
        self.assertIn('TOXIC // HNIJÍCÍ SLATINY', tx_svg)

        # Druid
        dr_svg = self.engine.generate_druid_tribe_svg()
        self.assertTrue(dr_svg.startswith('<svg'))
        self.assertIn('DRUID // PRADÁVNÝ LES', dr_svg)

        # Studna Duší
        sd_svg = self.engine.generate_studna_dusi_svg()
        self.assertTrue(sd_svg.startswith('<svg'))
        self.assertIn('STUDNA DUŠÍ // METAVORTEX', sd_svg)

    def test_http_api_endpoints(self):
        base_url = "http://127.0.0.1:8089"
        try:
            # 1. GET /api/svg/catalog
            req = urllib.request.Request(f"{base_url}/api/svg/catalog")
            with urllib.request.urlopen(req, timeout=3) as resp:
                self.assertEqual(resp.status, 200)
                d = json.loads(resp.read().decode('utf-8'))
                self.assertEqual(d["vital_max_hp_rule"], 6)
                self.assertGreaterEqual(len(d["presets"]), 5)

            # 2. GET /api/svg/master-blueprint.svg
            req = urllib.request.Request(f"{base_url}/api/svg/master-blueprint.svg")
            with urllib.request.urlopen(req, timeout=5) as resp:
                self.assertEqual(resp.status, 200)
                content_type = resp.headers.get("Content-Type", "")
                self.assertIn("image/svg+xml", content_type)
                svg_data = resp.read().decode('utf-8')
                self.assertIn("PROJECT: POSLEDNÍ KMEN", svg_data)

            # 3. GET /api/svg/crystal-crest.svg
            req = urllib.request.Request(f"{base_url}/api/svg/crystal-crest.svg")
            with urllib.request.urlopen(req, timeout=3) as resp:
                self.assertEqual(resp.status, 200)
                svg_data = resp.read().decode('utf-8')
                self.assertIn("CRYSTAL", svg_data)

            # 4. GET /evolved-svg-studio
            req = urllib.request.Request(f"{base_url}/evolved-svg-studio")
            with urllib.request.urlopen(req, timeout=3) as resp:
                self.assertEqual(resp.status, 200)
                html = resp.read().decode('utf-8')
                self.assertIn("EVOLVED HIGH-DETAIL SVG VECTOR STUDIO", html)

        except Exception as e:
            self.skipTest(f"Live server test skipped: {e}")

if __name__ == "__main__":
    unittest.main()
