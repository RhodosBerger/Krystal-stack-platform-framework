# ==============================================================================
# KRYSTAL-STACK: UNIT & INTEGRATION TESTS FOR RANGE LIMITS & ANIMATION SYSTEM
# ==============================================================================
import unittest
import json
import urllib.request
import urllib.error
from krystal_web_hub.economic_engine import (
    Tribe, AttackType, AbilityType, AbilitySpec, ResourceCost,
    calculate_hex_distance, validate_target_range, ABILITY_REGISTRY,
    AbilityEngine, RoundController
)

class TestRangeAndAnimationSystem(unittest.TestCase):
    def setUp(self):
        self.match = RoundController.initialize_match("test_range_match", Tribe.CRYSTAL, Tribe.TOXIC)

    # --------------------------------------------------------------------------
    # 1. HEX DISTANCE ALGORITHMS
    # --------------------------------------------------------------------------
    def test_hex_distance_identical_tiles(self):
        dist = calculate_hex_distance([0, 0], [0, 0])
        self.assertEqual(dist, 0)
        dist2 = calculate_hex_distance([2, -3], [2, -3])
        self.assertEqual(dist2, 0)

    def test_hex_distance_adjacent_neighbors(self):
        # The 6 axial neighbors of (0, 0) are: (1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1)
        neighbors = [[1, 0], [1, -1], [0, -1], [-1, 0], [-1, 1], [0, 1]]
        for nb in neighbors:
            dist = calculate_hex_distance([0, 0], nb)
            self.assertEqual(dist, 1, f"Failed for neighbor {nb}")

    def test_hex_distance_multi_step(self):
        # Distance between player base at (0, -2) and center (0, 0) is 2
        dist = calculate_hex_distance([0, -2], [0, 0])
        self.assertEqual(dist, 2)

        # Distance between (0, -2) and (0, 2) is 4
        dist4 = calculate_hex_distance([0, -2], [0, 2])
        self.assertEqual(dist4, 4)

    # --------------------------------------------------------------------------
    # 2. RANGE BOUNDS & ATTACK TYPE VALIDATION
    # --------------------------------------------------------------------------
    def test_melee_attack_range_bounds(self):
        decay_touch = ABILITY_REGISTRY["decay_touch"]
        self.assertEqual(decay_touch.attack_type, AttackType.MELEE)
        self.assertEqual(decay_touch.min_range, 1)
        self.assertEqual(decay_touch.max_range, 1)

        # Adjacent (distance 1) -> Valid
        valid, dist, msg = validate_target_range([0, -2], [0, -1], decay_touch)
        self.assertTrue(valid)
        self.assertEqual(dist, 1)

        # Self (distance 0) -> Invalid for melee
        valid, dist, msg = validate_target_range([0, -2], [0, -2], decay_touch)
        self.assertFalse(valid)
        self.assertIn("príliš blízko", msg)

        # Distance 2 -> Out of melee range
        valid, dist, msg = validate_target_range([0, -2], [0, 0], decay_touch)
        self.assertFalse(valid)
        self.assertIn("mimo maximálneho dosahu", msg)

    def test_ranged_attack_range_bounds(self):
        meteor = ABILITY_REGISTRY["crystal_meteor"]
        self.assertEqual(meteor.attack_type, AttackType.RANGED)
        self.assertEqual(meteor.min_range, 1)
        self.assertEqual(meteor.max_range, 4)

        # In range (distance 1, 2, 3, 4)
        for target_hex in [[0, -1], [0, 0], [0, 1], [0, 2]]:
            valid, dist, msg = validate_target_range([0, -2], target_hex, meteor)
            self.assertTrue(valid, f"Failed for {target_hex}")

        # Out of range (distance 5: [0, 3])
        valid, dist, msg = validate_target_range([0, -2], [0, 3], meteor)
        self.assertFalse(valid)
        self.assertEqual(dist, 5)

    def test_self_buff_range_bounds(self):
        shield = ABILITY_REGISTRY["crystal_shield"]
        self.assertEqual(shield.attack_type, AttackType.SELF)
        valid, dist, msg = validate_target_range([0, -2], [0, -2], shield)
        self.assertTrue(valid)

    # --------------------------------------------------------------------------
    # 3. ABILITY ENGINE CAST WITH RANGE & ANIMATION PAYLOAD
    # --------------------------------------------------------------------------
    def test_ability_engine_cast_out_of_range(self):
        # Attempt melee decay_touch at distance 2
        success, details, msg = AbilityEngine.cast_ability(
            self.match.enemy, self.match.player, "decay_touch",
            self.match.escalation,
            source_hex=[0, -2], target_hex=[0, 0]
        )
        self.assertFalse(success)
        self.assertIn("mimo maximálneho dosahu", msg)

    def test_ability_engine_cast_in_range_with_animation_payload(self):
        # Set sufficient resources
        self.match.player.mana = 10
        self.match.player.aether_crystals = 5
        initial_enemy_hp = self.match.enemy.hp

        success, details, msg = AbilityEngine.cast_ability(
            self.match.player, self.match.enemy, "crystal_meteor",
            self.match.escalation,
            source_hex=[0, -2], target_hex=[0, 0]
        )
        self.assertTrue(success)
        self.assertEqual(self.match.enemy.hp, initial_enemy_hp - 2)

        # Verify animation metadata
        self.assertIn("animation", details)
        anim = details["animation"]
        self.assertEqual(anim["type"], "animated_arrow")
        self.assertEqual(anim["attack_type"], "ranged")
        self.assertEqual(anim["min_range"], 1)
        self.assertEqual(anim["max_range"], 4)
        self.assertEqual(anim["distance"], 2)
        self.assertEqual(anim["source_hex"], [0, -2])
        self.assertEqual(anim["target_hex"], [0, 0])
        self.assertGreater(anim["duration"], 0.4)

    # --------------------------------------------------------------------------
    # 4. HTTP API INTEGRATION TESTS (PORT 8089)
    # --------------------------------------------------------------------------
    def test_api_cards_library_has_ranges(self):
        req = urllib.request.urlopen("http://127.0.0.1:8089/api/cards")
        self.assertEqual(req.status, 200)
        data = json.loads(req.read().decode('utf-8'))
        cards = data.get("cards", [])
        self.assertGreaterEqual(len(cards), 9)

        card_map = {c["id"]: c for c in cards}
        self.assertIn("druid_strike", card_map)
        self.assertIn("decay_strike", card_map)
        self.assertIn("crystal_meteor", card_map)

        # Check melee specs
        ds = card_map["druid_strike"]
        self.assertEqual(ds["attack_type"], "melee")
        self.assertEqual(ds["min_range"], 1)
        self.assertEqual(ds["max_range"], 1)

        # Check ranged specs
        cm = card_map["crystal_meteor"]
        self.assertEqual(cm["attack_type"], "ranged")
        self.assertEqual(cm["min_range"], 1)
        self.assertEqual(cm["max_range"], 4)

    def test_api_targeting_validate_endpoint(self):
        # 1. Melee in-range test
        payload = json.dumps({
            "card_id": "druid_strike",
            "source_hex": [0, -2],
            "target_hex": [0, -1]
        }).encode('utf-8')
        req = urllib.request.Request("http://127.0.0.1:8089/api/targeting/validate", data=payload, headers={'Content-Type': 'application/json'})
        res = urllib.request.urlopen(req)
        data = json.loads(res.read().decode('utf-8'))
        self.assertTrue(data["valid"])
        self.assertEqual(data["distance"], 1)
        self.assertEqual(data["attack_type"], "melee")

        # 2. Melee out-of-range test (distance 2)
        payload_oor = json.dumps({
            "card_id": "druid_strike",
            "source_hex": [0, -2],
            "target_hex": [0, 0]
        }).encode('utf-8')
        req_oor = urllib.request.Request("http://127.0.0.1:8089/api/targeting/validate", data=payload_oor, headers={'Content-Type': 'application/json'})
        res_oor = urllib.request.urlopen(req_oor)
        data_oor = json.loads(res_oor.read().decode('utf-8'))
        self.assertFalse(data_oor["valid"])
        self.assertEqual(data_oor["distance"], 2)

    def test_api_cards_cast_range_enforcement(self):
        # 1. Cast out of range -> Expect 400
        payload_oor = json.dumps({
            "card_id": "druid_strike",
            "source_hex": [0, -2],
            "target_hex": [0, 0, 3.0] # distance 2
        }).encode('utf-8')
        req_oor = urllib.request.Request("http://127.0.0.1:8089/api/cards/cast", data=payload_oor, headers={'Content-Type': 'application/json'})
        try:
            urllib.request.urlopen(req_oor)
            self.fail("Expected HTTP 400 for out-of-range card cast")
        except urllib.error.HTTPError as e:
            self.assertEqual(e.code, 400)
            err_data = json.loads(e.read().decode('utf-8'))
            self.assertIn("mimo dosahu", err_data["error"])

        # 2. Cast in range -> Expect 200 with animation payload
        payload_ir = json.dumps({
            "card_id": "druid_strike",
            "source_hex": [0, -2],
            "target_hex": [0, 0, -1.5] # distance 1
        }).encode('utf-8')
        req_ir = urllib.request.Request("http://127.0.0.1:8089/api/cards/cast", data=payload_ir, headers={'Content-Type': 'application/json'})
        res_ir = urllib.request.urlopen(req_ir)
        self.assertEqual(res_ir.status, 200)
        data_ir = json.loads(res_ir.read().decode('utf-8'))
        self.assertEqual(data_ir["status"], "SUCCESS")
        self.assertIn("animation", data_ir)
        self.assertEqual(data_ir["animation"]["attack_type"], "melee")
        self.assertEqual(data_ir["animation"]["distance"], 1)

if __name__ == '__main__':
    unittest.main()
