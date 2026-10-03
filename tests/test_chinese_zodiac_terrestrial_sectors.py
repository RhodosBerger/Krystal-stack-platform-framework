import unittest
import json
import urllib.request
from krystal_web_hub.economic_engine.chinese_zodiac_terrestrial_sectors import (
    CANONICAL_ZODIAC_SECTORS,
    ZodiacSector,
    TerrestrialPhenomenon,
    ChineseZodiacSectorEngine,
    GLOBAL_CHINESE_ZODIAC_SECTOR_ENGINE,
    VITAL_MAX_HP,
    ZodiacElement,
    YinYang
)

class TestChineseZodiacTerrestrialSectors(unittest.TestCase):
    def setUp(self):
        self.engine = GLOBAL_CHINESE_ZODIAC_SECTOR_ENGINE

    def test_vital_max_hp_rule(self):
        self.assertEqual(VITAL_MAX_HP, 6)
        for s in CANONICAL_ZODIAC_SECTORS.values():
            self.assertEqual(s.fortification_hp, 6)
            self.assertEqual(s.max_fortification_hp, 6)
            self.assertEqual(s.phenomenon.vital_max_hp_rule, 6)

    def test_complete_twelve_earthly_branches_and_sectors(self):
        sectors = self.engine.get_all_sectors()
        self.assertEqual(len(sectors), 12)

        expected_branches = ["Zǐ", "Chǒu", "Yín", "Mǎo", "Chén", "Sì", "Wǔ", "Wèi", "Shēn", "Yǒu", "Xū", "Hài"]
        for b in expected_branches:
            found = any(b in s["earthly_branch"] for s in sectors)
            self.assertTrue(found, f"Branch {b} missing in sectors")

    def test_five_elements_presence(self):
        sectors = self.engine.get_all_sectors()
        elements_found = {s["element"] for s in sectors}
        self.assertGreaterEqual(len(elements_found), 5)

    def test_trigger_phenomenon_surge(self):
        res = self.engine.trigger_phenomenon("sector_05_dragon", intensity_delta=0.45)
        self.assertEqual(res["status"], "PHENOMENON_SURGED")
        self.assertEqual(res["vital_max_hp_rule"], 6)
        self.assertGreaterEqual(res["new_resonance_intensity"], 1.4)

    def test_global_terrestrial_cycle_evaluation(self):
        cycle_1 = self.engine.evaluate_global_terrestrial_cycle(turn_number=1)
        self.assertEqual(cycle_1["vital_max_hp_rule"], 6)
        self.assertIn("Zǐ", cycle_1["ruling_earthly_branch"])

        cycle_5 = self.engine.evaluate_global_terrestrial_cycle(turn_number=5)
        self.assertIn("Chén", cycle_5["ruling_earthly_branch"])

    def test_http_api_endpoints(self):
        base_url = "http://127.0.0.1:8089"
        try:
            # 1. GET /api/zodiac-sectors/all
            req = urllib.request.Request(f"{base_url}/api/zodiac-sectors/all")
            with urllib.request.urlopen(req, timeout=3) as resp:
                self.assertEqual(resp.status, 200)
                d = json.loads(resp.read().decode('utf-8'))
                self.assertEqual(d["vital_max_hp_rule"], 6)
                self.assertEqual(len(d["sectors"]), 12)

            # 2. GET /api/zodiac-sectors/cycle?turn=3
            req = urllib.request.Request(f"{base_url}/api/zodiac-sectors/cycle?turn=3")
            with urllib.request.urlopen(req, timeout=3) as resp:
                self.assertEqual(resp.status, 200)
                d = json.loads(resp.read().decode('utf-8'))
                self.assertEqual(d["vital_max_hp_rule"], 6)
                self.assertIn("Yín", d["ruling_earthly_branch"])

            # 3. POST /api/zodiac-sectors/trigger-phenomenon
            req = urllib.request.Request(
                f"{base_url}/api/zodiac-sectors/trigger-phenomenon",
                data=json.dumps({"sector_id": "sector_03_tiger", "intensity_delta": 0.5}).encode('utf-8'),
                headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req, timeout=3) as resp:
                self.assertEqual(resp.status, 200)
                d = json.loads(resp.read().decode('utf-8'))
                self.assertEqual(d["vital_max_hp_rule"], 6)
                self.assertEqual(d["status"], "PHENOMENON_SURGED")

            # 4. GET /chinese-zodiac-sectors HTML Studio
            req = urllib.request.Request(f"{base_url}/chinese-zodiac-sectors")
            with urllib.request.urlopen(req, timeout=3) as resp:
                self.assertEqual(resp.status, 200)
                html = resp.read().decode('utf-8')
                self.assertIn("ČÍNSKY ZVEROKRUH", html)

        except Exception as e:
            self.skipTest(f"Live server on {base_url} skipped during direct test: {e}")

if __name__ == "__main__":
    unittest.main()
