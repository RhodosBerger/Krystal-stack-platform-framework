# ==============================================================================
# KRYSTAL-STACK: UNIT TESTS FOR TACTICAL AI & 43-CARD DECK MANAGER
# ==============================================================================

import unittest
from krystal_web_hub.economic_engine import (
    Tribe, MatchState, CombatantState, RoundController,
    TacticalAIEngine, generate_tribal_deck, PlayerDeckManager
)

class TestTacticalAIAndCards(unittest.TestCase):

    def test_deck_generation(self):
        # 43 cards per deck rule
        crystal_deck = generate_tribal_deck(Tribe.CRYSTAL, 43)
        toxic_deck = generate_tribal_deck(Tribe.TOXIC, 43)
        druid_deck = generate_tribal_deck(Tribe.DRUID, 43)

        self.assertEqual(len(crystal_deck), 43)
        self.assertEqual(len(toxic_deck), 43)
        self.assertEqual(len(druid_deck), 43)

        # Check tribal alignment and attributes
        for card in crystal_deck:
            self.assertEqual(card["tribe"], "crystal")
            self.assertIn("cost", card)
            self.assertIn("hp_delta", card)
            self.assertIn("attack_type", card)

    def test_player_deck_manager(self):
        mgr = PlayerDeckManager(Tribe.CRYSTAL, total_deck_size=43)
        self.assertEqual(len(mgr.draw_pile), 43)
        self.assertEqual(len(mgr.hand), 0)

        # Draw to full (5 cards)
        drawn = mgr.draw_to_full()
        self.assertEqual(len(drawn), 5)
        self.assertEqual(len(mgr.hand), 5)
        self.assertEqual(len(mgr.draw_pile), 38)

        # Play a card
        first_card_id = mgr.hand[0]["id"]
        played = mgr.play_card(first_card_id)
        self.assertIsNotNone(played)
        self.assertEqual(len(mgr.hand), 4)
        self.assertEqual(len(mgr.discard_pile), 1)

        # Replenish back to 5
        mgr.draw_to_full()
        self.assertEqual(len(mgr.hand), 5)
        self.assertEqual(len(mgr.draw_pile), 37)

    def test_tactical_ai_lethal_priority(self):
        match = RoundController.initialize_match("test_ai_match", Tribe.CRYSTAL, Tribe.TOXIC)
        # Enemy is toxic, target is player (player at 2 HP)
        match.player.hp = 2
        match.enemy.mana = 10

        lethal_cards = [
            {"id": "weak_poke", "name": "Weak Poke", "cost": 1, "hp_delta": -1, "attack_type": "ranged", "min_range": 1, "max_range": 4},
            {"id": "fatal_strike", "name": "Fatal Strike", "cost": 3, "hp_delta": -2, "attack_type": "ranged", "min_range": 1, "max_range": 4}
        ]

        result = TacticalAIEngine.evaluate_ai_turn(match, ai_side="enemy", available_cards=lethal_cards)
        self.assertTrue(result["executed"])
        self.assertEqual(result["card_id"], "fatal_strike")
        self.assertEqual(match.player.hp, 0)
        self.assertIn("Smrteľný úder", result["reasoning"])

    def test_tactical_ai_emergency_defense(self):
        match = RoundController.initialize_match("test_ai_match", Tribe.CRYSTAL, Tribe.TOXIC)
        match.enemy.hp = 2 # Critical HP
        match.enemy.mana = 10
        match.player.hp = 6

        defensive_cards = [
            {"id": "slime_cocoon", "name": "Slime Cocoon", "cost": 2, "hp_delta": 0, "armor_delta": 2, "attack_type": "self", "min_range": 0, "max_range": 0},
            {"id": "small_poke", "name": "Small Poke", "cost": 1, "hp_delta": -1, "armor_delta": 0, "attack_type": "ranged", "min_range": 1, "max_range": 4}
        ]

        result = TacticalAIEngine.evaluate_ai_turn(match, ai_side="enemy", available_cards=defensive_cards)
        self.assertTrue(result["executed"])
        self.assertEqual(result["card_id"], "slime_cocoon")
        self.assertEqual(match.enemy.armor, 2)
        self.assertIn("Kritický stav HP", result["reasoning"])

    def test_tactical_ai_range_constraint(self):
        match = RoundController.initialize_match("test_ai_match", Tribe.CRYSTAL, Tribe.TOXIC)
        # Distance between home hexes [0, 2] and [0, -2] is 4 hexes
        # Melee cards require strictly 1 hex
        melee_only_cards = [
            {"id": "decay_claw", "name": "Decay Claw", "cost": 1, "hp_delta": -2, "attack_type": "melee", "min_range": 1, "max_range": 1}
        ]

        result = TacticalAIEngine.evaluate_ai_turn(match, ai_side="enemy", available_cards=melee_only_cards)
        self.assertFalse(result["executed"])
        self.assertEqual(result["action_type"], "skip")
        self.assertIn("Žiadna akcia v dosahu", result["reasoning"])

if __name__ == '__main__':
    unittest.main()
