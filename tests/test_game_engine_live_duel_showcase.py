#!/usr/bin/env python3
"""
UNIT TESTS: POSLEDNÍ KMEN LIVE GAME ENGINE DUEL SHOWCASE
=============================================================================
Verifies:
  1. System Invariant: VITAL_MAX_HP == 6 strictly enforced.
  2. Fighter initialization: HP=6, starting hand of 4 cards, 25-card deck.
  3. Aspect linkage: Fighter links to multi-disciplinary radar profile (OBT, KON, POŠ, DOS).
  4. Combat simulation: Complete turn resolution, combat logs, and victory condition.
  5. Multi-tribe matchups: CRYSTAL vs. TOXIC, TOXIC vs. DRUID, DRUID vs. CRYSTAL.
  6. Damage & Healing clamping: HP strictly in range [0, 6].
=============================================================================
"""

import unittest
import os
import sys

WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

from krystal_kernel.scene_constructor_presets import VITAL_MAX_HP
from examples.posledni_kmen_live_duel_engine import (
    PosledniKmenLiveDuelEngine,
    DuelFighter
)


class TestGameEngineLiveDuelShowcase(unittest.TestCase):

    def setUp(self):
        self.engine = PosledniKmenLiveDuelEngine(seed=42)

    def test_vital_max_hp_invariant(self):
        """Invariant: VITAL_MAX_HP must strictly equal 6."""
        self.assertEqual(VITAL_MAX_HP, 6)

    def test_fighter_creation_and_stats(self):
        """Fighter must initialize with 6 HP, 4 cards in hand, and multi-axial radar profile."""
        fighter = self.engine.create_fighter("CRYSTAL", "Test Crystal")
        self.assertEqual(fighter.hp, 6)
        self.assertEqual(len(fighter.hand), 4)
        self.assertEqual(len(fighter.deck), 21) # 25 - 4 in hand
        self.assertEqual(fighter.tribe_code, "CRYSTAL")

        # Verify radar aspects presence
        aspects_codes = {a.code for a in fighter.profile.aspects}
        self.assertIn("OBT", aspects_codes)
        self.assertIn("KON", aspects_codes)
        self.assertIn("POŠ", aspects_codes)
        self.assertIn("DOS", aspects_codes)

    def test_damage_and_healing_bounds(self):
        """HP must strictly clamp between 0 and 6."""
        fighter = self.engine.create_fighter("TOXIC")
        self.assertEqual(fighter.hp, 6)

        # Healing when full does not exceed 6
        healed = fighter.heal(3)
        self.assertEqual(fighter.hp, 6)
        self.assertEqual(healed, 0)

        # Damage reduces HP
        dmg = fighter.apply_damage(2)
        self.assertEqual(fighter.hp, 4)
        self.assertEqual(dmg, 2)

        # Healing restores HP up to 6
        healed2 = fighter.heal(1)
        self.assertEqual(fighter.hp, 5)
        self.assertEqual(healed2, 1)

        # Overkill damage clamps to 0
        fighter.apply_damage(10)
        self.assertEqual(fighter.hp, 0)
        self.assertFalse(fighter.is_alive())

    def test_crystal_vs_toxic_duel_resolution(self):
        """Simulates full duel between Crystal and Toxic tribes."""
        f1 = self.engine.create_fighter("CRYSTAL", "Kráľovná")
        f2 = self.engine.create_fighter("TOXIC", "Šaman")

        match_res = self.engine.simulate_match(f1, f2, max_turns=20)
        self.assertEqual(match_res["status"], "COMPLETED")
        self.assertIn(match_res["winner"], [f1.name, f2.name])
        self.assertGreater(match_res["turns_count"], 0)
        self.assertTrue(match_res["vital_max_hp_preserved"])
        self.assertGreater(len(match_res["combat_log"]), 0)

    def test_all_tribal_matchups(self):
        """Simulates all pairings: Toxic vs Druid and Druid vs Crystal."""
        # Toxic vs Druid
        f_toxic = self.engine.create_fighter("TOXIC")
        f_druid = self.engine.create_fighter("DRUID")
        res1 = self.engine.simulate_match(f_toxic, f_druid, max_turns=20)
        self.assertEqual(res1["status"], "COMPLETED")
        self.assertTrue(res1["vital_max_hp_preserved"])

        # Druid vs Crystal
        f_druid2 = self.engine.create_fighter("DRUID")
        f_crystal2 = self.engine.create_fighter("CRYSTAL")
        res2 = self.engine.simulate_match(f_druid2, f_crystal2, max_turns=20)
        self.assertEqual(res2["status"], "COMPLETED")
        self.assertTrue(res2["vital_max_hp_preserved"])


if __name__ == "__main__":
    unittest.main()
