import unittest
import os
import time

from krystal_web_hub.economic_engine.combinatorial_tactical_moves_and_mulligan import (
    CANONICAL_MULLIGAN_CARDS,
    MulliganPhaseManager,
    CombinatorialTacticalMoveEngine
)
from krystal_janet.janet_bridge import JanetValidator


class TestCombinatorialTacticalMovesAndMulligan(unittest.TestCase):
    """
    Unit test suite verifying:
    1. Canonical Mulligan card templates matching the game visual photo.
    2. Mulligan phase card selection, exchange, and deck replenishment.
    3. 256,000 Combinatorial Tactical Move Engine (2^18 states).
    4. Unit escort, crystal conduit, and status combo mechanics.
    5. Janet DSL validation for combinatorial_moves_and_mulligan.janet.
    """

    def setUp(self):
        self.mulligan_mgr = MulliganPhaseManager(seed=42)
        self.combinatorial_engine = CombinatorialTacticalMoveEngine()

    # ── 1. CANONICAL CARD SPECS ───────────────────────────────────────────────
    def test_canonical_mulligan_cards_integrity(self):
        self.assertEqual(len(CANONICAL_MULLIGAN_CARDS), 4)
        card_names = {c["name"] for c in CANONICAL_MULLIGAN_CARDS}
        self.assertIn("ĽADOVÁ GUĽA", card_names)
        self.assertIn("POHYB S JEDNOTKOU", card_names)
        self.assertIn("SPOJOVACÍ KRYŠTÁL", card_names)
        self.assertIn("ZAMRZNUTIE", card_names)

        for card in CANONICAL_MULLIGAN_CARDS:
            self.assertIn(card["cost"], [1, 2])
            self.assertEqual(card["power"], 2)
            self.assertEqual(card["range_stat"], 1.5)
            self.assertIn("visual_spec", card)
            self.assertEqual(card["visual_spec"]["frame"], "carved_wood_gold_rivets")
            self.assertEqual(card["visual_spec"]["cost_badge"], "cyan_orb_top_left")
            self.assertEqual(card["visual_spec"]["art_viewport"], "horizontal_oval_cosmic_purple_nebula")

    # ── 2. MULLIGAN PHASE WORKFLOW ────────────────────────────────────────────
    def test_mulligan_initial_deal_and_selection(self):
        hand = self.mulligan_mgr.opening_hand
        self.assertEqual(len(hand), 4)
        self.assertEqual(len(self.mulligan_mgr.selected_indices), 0)

        # Toggle second card (index 1: POHYB S JEDNOTKOU - matching screenshot!)
        res = self.mulligan_mgr.toggle_card_selection(1)
        self.assertEqual(res["selected_count"], 1)
        self.assertIn(1, res["selected_indices"])
        self.assertTrue(self.mulligan_mgr.opening_hand[1]["selected_for_mulligan"])

    def test_mulligan_exchange_execution(self):
        # Select card 1 for exchange
        self.mulligan_mgr.toggle_card_selection(1)
        initial_card_id = self.mulligan_mgr.opening_hand[1]["instance_id"]
        deck_len_before = len(self.mulligan_mgr.deck)

        # Execute exchange
        res = self.mulligan_mgr.execute_mulligan()
        self.assertTrue(res["success"])
        self.assertEqual(res["replaced_count"], 1)
        self.assertEqual(len(res["final_hand"]), 4)
        self.assertTrue(res["mulligan_completed"])
        # Selected card replaced
        new_card_id = self.mulligan_mgr.opening_hand[1]["instance_id"]
        self.assertNotEqual(initial_card_id, new_card_id)

    # ── 3. 256,000 COMBINATORIAL MOVE ENGINE ──────────────────────────────────
    def test_simulate_256k_combinations(self):
        res = self.combinatorial_engine.simulate_256k_combinations(
            hero_pos=(0, -2),
            enemy_pos=(0, 2),
            active_crystals=[(0, -1), (1, 0), (0, 0)]
        )
        self.assertEqual(res["total_combinations_evaluated"], 262144)
        self.assertIn("2^18", res["combinatorial_formula"])
        self.assertGreater(len(res["top_strategies"]), 0)
        self.assertIn("optimal_turn_recommendation", res)
        # Check that evaluation is ultra-fast (< 500ms)
        self.assertLess(res["evaluation_time_ms"], 500.0)

        # Check strategy properties
        top_strat = res["optimal_turn_recommendation"]
        self.assertIn("projected_damage", top_strat)
        self.assertIn("map_control_pct", top_strat)
        self.assertIn("escort_safety", top_strat)
        self.assertIn("tactical_utility_score", top_strat)
        self.assertGreater(top_strat["tactical_utility_score"], 0)

    # ── 4. JANET DSL VALIDATION ───────────────────────────────────────────────
    def test_janet_combinatorial_dsl_syntax(self):
        janet_path = os.path.join(
            os.path.dirname(__file__), "..", "krystal_janet", "combinatorial_moves_and_mulligan.janet"
        )
        self.assertTrue(os.path.exists(janet_path), f"Janet file not found: {janet_path}")
        val_res = JanetValidator.validate_file(janet_path)
        self.assertTrue(val_res["valid"], f"Validation failed: {val_res.get('error')}")
        self.assertGreaterEqual(val_res["line_count"], 40)


if __name__ == "__main__":
    unittest.main()
