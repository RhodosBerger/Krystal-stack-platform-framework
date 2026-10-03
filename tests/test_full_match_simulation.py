# ==============================================================================
# KRYSTAL-STACK: INTEGRATION TESTS FOR FULL MATCH SIMULATION (HTTP API)
# ==============================================================================

import unittest
import urllib.request
import urllib.error
import json
import time

BASE_URL = "http://127.0.0.1:8089"

def post_json(endpoint: str, data: dict) -> dict:
    url = f"{BASE_URL}{endpoint}"
    req = urllib.request.Request(
        url,
        data=json.dumps(data).encode('utf-8'),
        headers={'Content-Type': 'application/json'}
    )
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read().decode('utf-8'))

def get_json(endpoint: str) -> dict:
    url = f"{BASE_URL}{endpoint}"
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read().decode('utf-8'))

class TestFullMatchSimulation(unittest.TestCase):

    def setUp(self):
        # Reset match to pristine state
        try:
            post_json("/api/economy/reset", {})
        except Exception:
            pass

    def test_get_player_hand_and_deck(self):
        hand_data = get_json("/api/player/hand")
        self.assertIn("hand", hand_data)
        self.assertIn("draw_pile_count", hand_data)
        self.assertIn("discard_pile_count", hand_data)
        self.assertGreaterEqual(len(hand_data["hand"]), 1)
        self.assertEqual(hand_data["draw_pile_count"] + len(hand_data["hand"]) + hand_data["discard_pile_count"], 43)

    def test_match_status_endpoint(self):
        status = get_json("/api/match/status")
        self.assertIn("player", status)
        self.assertIn("enemy", status)
        self.assertEqual(status["player"]["hp"], 6)
        self.assertEqual(status["player"]["max_hp"], 6)
        self.assertEqual(status["enemy"]["hp"], 6)
        self.assertEqual(status["enemy"]["max_hp"], 6)

    def test_ai_turn_and_end_turn_flow(self):
        # 1. Player casts a ranged card at enemy (at distance 4)
        cast_resp = post_json("/api/cards/cast", {
            "card_id": "crystal_meteor",
            "source_hex": [0, -2],
            "target_hex": [0, 2]
        })
        self.assertTrue(cast_resp["success"])
        self.assertEqual(cast_resp["enemy_hp"], 4) # 6 - 2 = 4 HP

        # 2. Player ends turn (triggers mana allowance, hand replenish, and AI counter-attack)
        end_resp = post_json("/api/match/end-turn", {})
        self.assertTrue(end_resp["success"])
        self.assertIn("ai_action", end_resp)
        self.assertIn("turn", end_resp)
        self.assertGreaterEqual(end_resp["turn"], 2)

        # 3. Check that game state reflects combat actions
        state = get_json("/api/state")
        self.assertEqual(state["enemy_hp"], 4)
        self.assertLessEqual(state["player_hp"], 6)

if __name__ == '__main__':
    unittest.main()
