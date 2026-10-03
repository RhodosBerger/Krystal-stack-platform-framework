"""
Unit & Integration Test Suite for Krystal-Stack 3D Engine & Godot Builder
"""

import os
import unittest
import urllib.request
import json

BASE_URL = "http://127.0.0.1:8089"

class TestKrystalEngineCore(unittest.TestCase):
    def setUp(self):
        try:
            req = urllib.request.Request(
                f"{BASE_URL}/api/economy/reset",
                data=json.dumps({"player_tribe": "crystal", "enemy_tribe": "toxic"}).encode('utf-8'),
                headers={'Content-Type': 'application/json'}
            )
            urllib.request.urlopen(req)
        except Exception:
            pass

    def test_01_engine_status(self):
        url = f"{BASE_URL}/api/status"
        res = urllib.request.urlopen(url)
        self.assertEqual(res.status, 200)
        data = json.loads(res.read().decode('utf-8'))
        self.assertEqual(data["status"], "ONLINE")
        self.assertEqual(data["cards_count"], 9)
        self.assertIn("crystal_shard.obj", data["assets"])
        self.assertIn("hex_tile.obj", data["assets"])

    def test_02_cards_library(self):
        url = f"{BASE_URL}/api/cards"
        res = urllib.request.urlopen(url)
        self.assertEqual(res.status, 200)
        data = json.loads(res.read().decode('utf-8'))
        self.assertEqual(data["rules"]["max_hp"], 6)
        self.assertEqual(len(data["cards"]), 9)
        
        card_ids = [c["id"] for c in data["cards"]]
        self.assertIn("crystal_meteor", card_ids)
        self.assertIn("crystal_shield", card_ids)
        self.assertIn("acid_slime", card_ids)
        self.assertIn("toxic_cloud", card_ids)
        self.assertIn("earth_roots", card_ids)
        self.assertIn("nature_bless", card_ids)
        self.assertIn("druid_strike", card_ids)
        self.assertIn("decay_strike", card_ids)

    def test_03_generate_scene_ast(self):
        url = f"{BASE_URL}/api/bot/generate-scene"
        payload = json.dumps({"prompt": "Aréna pre Kryštálový Kmeň"}).encode('utf-8')
        req = urllib.request.Request(url, data=payload, headers={'Content-Type': 'application/json', 'Content-Length': str(len(payload))})
        res = urllib.request.urlopen(req)
        self.assertEqual(res.status, 200)
        data = json.loads(res.read().decode('utf-8'))
        self.assertIn("tree", data)
        self.assertEqual(data["tree"]["name"], "WorldRoot")
        self.assertIn("tscn", data)
        self.assertTrue(data["tscn"].startswith("[gd_scene"))

    def test_04_card_cast_and_ledger(self):
        url = f"{BASE_URL}/api/cards/cast"
        payload = json.dumps({"card_id": "nature_bless", "target_hex": [2, 0, 1]}).encode('utf-8')
        req = urllib.request.Request(url, data=payload, headers={'Content-Type': 'application/json', 'Content-Length': str(len(payload))})
        res = urllib.request.urlopen(req)
        self.assertEqual(res.status, 200)
        data = json.loads(res.read().decode('utf-8'))
        self.assertEqual(data["status"], "SUCCESS")
        self.assertIn("ledger_entry", data)
        self.assertIn("spawn_node", data)
        # HP must not exceed 6
        self.assertLessEqual(data["game_state"]["player_hp"], 6)

    def test_05_asset_serving(self):
        url = f"{BASE_URL}/api/assets/crystal_shard.obj"
        res = urllib.request.urlopen(url)
        self.assertEqual(res.status, 200)
        content = res.read().decode('utf-8')
        self.assertTrue(content.startswith("# Krystal-Stack"))
        self.assertIn("v ", content)
        self.assertIn("f ", content)

if __name__ == '__main__':
    unittest.main()
