#!/usr/bin/env python3
"""
==============================================================================
KRYSTAL-STACK: UNIT TEST SUITE
POSLEDNÍ KMEN // FANTASY COMPOSITION & ELEMENTAL ARCHITECTURE ENGINE
==============================================================================
Tests:
  1. Universal Platform Invariants: VITAL_MAX_HP == 6, 129 Cards, 9 Tiles, 15 Flags.
  2. Canonical Tribe Registry & Metadata (Crystal, Toxic, Druid, Volcanic, Void).
  3. Card Stat Bounds Clamping: Damage <= 6, Defense <= 6, Range in [1, 6].
  4. Procedural Fantasy Composition Generation across Doctrinal Archetypes.
  5. Synergy, Threat, and Resilience Metric Bounds.
  6. Tactical 9-Tile / 19-Hex Formation Coordinate Mapping.
  7. Duel & Clash Simulation with Elemental Counter-Polarity.
  8. Catalog Summary Structure & Web Serialization.
==============================================================================
"""

import unittest
from typing import Dict, Any

from krystal_kernel.posledni_kmen_fantasy_composition_engine import (
    VITAL_MAX_HP,
    GOLDEN_RATIO,
    TOTAL_DECK_SIZE,
    TOTAL_CARDS_OFFICIAL,
    MODULAR_ARENA_TILES,
    FLAG_TOKENS_COUNT,
    TribeId,
    CardType,
    PlaystyleDoctrine,
    TribeAspectMetadata,
    FantasyCard,
    FantasyComposition,
    PosledniKmenFantasyCompositionEngine,
    GLOBAL_FANTASY_COMPOSITION_ENGINE
)


class TestPosledniKmenFantasyComposition(unittest.TestCase):

    def setUp(self):
        self.engine = PosledniKmenFantasyCompositionEngine()

    def test_universal_platform_invariants(self):
        """Verify strict conformance to Poslední Kmen and Krystal Stack invariants."""
        self.assertEqual(VITAL_MAX_HP, 6, "Platform invariant VITAL_MAX_HP must strictly be 6")
        self.assertEqual(TOTAL_DECK_SIZE, 43, "Each tribe must have a fixed 43-card deck")
        self.assertEqual(TOTAL_CARDS_OFFICIAL, 129, "Official game contains 129 total cards (3x 43)")
        self.assertEqual(MODULAR_ARENA_TILES, 9, "Modular arena must contain 9 tiles")
        self.assertEqual(FLAG_TOKENS_COUNT, 15, "Flag tokens count must be 15")

    def test_canonical_tribe_aspects(self):
        """Verify the 3 canonical tribes from https://www.poslednikmen.cz/ plus expansions."""
        tribes = self.engine.tribes
        self.assertIn(TribeId.CRYSTAL, tribes)
        self.assertIn(TribeId.TOXIC, tribes)
        self.assertIn(TribeId.DRUID, tribes)
        self.assertIn(TribeId.VOLCANIC, tribes)
        self.assertIn(TribeId.VOID, tribes)

        # Check Crystal aspects
        cry = tribes[TribeId.CRYSTAL]
        self.assertEqual(cry.territory_cz, "Severní štíty")
        self.assertEqual(cry.vital_max_hp, 6)
        self.assertTrue(any("Mráz" in m for m in cry.primary_mechanics))
        self.assertTrue(any("Meteory" in m for m in cry.primary_mechanics))

        # Check Toxic aspects
        tox = tribes[TribeId.TOXIC]
        self.assertEqual(tox.territory_cz, "Hnijící slatiny")
        self.assertEqual(tox.vital_max_hp, 6)
        self.assertTrue(any("jed" in m.lower() for m in tox.primary_mechanics))
        self.assertTrue(any("had" in m.lower() for m in tox.primary_mechanics))

        # Check Druid aspects
        dru = tribes[TribeId.DRUID]
        self.assertEqual(dru.territory_cz, "Pradávný les")
        self.assertEqual(dru.vital_max_hp, 6)
        self.assertTrue(any("zvíř" in m.lower() or "beast" in m.lower() for m in dru.primary_mechanics))
        self.assertTrue(any("strom" in m.lower() or "grove" in m.lower() for m in dru.primary_mechanics))

    def test_fantasy_card_stat_clamping(self):
        """Verify FantasyCard clamps stats strictly to [0, 6] HP/Damage/Defense and [1, 6] Range."""
        overpowered_card = FantasyCard(
            card_id="op_test",
            name="Mega Meteor",
            card_type=CardType.ATTACK,
            tribe=TribeId.CRYSTAL,
            damage=99,
            defense=50,
            range_hex=10
        )
        self.assertEqual(overpowered_card.damage, 6, "Damage must be clamped to VITAL_MAX_HP == 6")
        self.assertEqual(overpowered_card.defense, 6, "Defense must be clamped to VITAL_MAX_HP == 6")
        self.assertEqual(overpowered_card.range_hex, 6, "Range must be clamped to max 6 hexes")

        underpowered_card = FantasyCard(
            card_id="under_test",
            name="Blank Stone",
            card_type=CardType.DEFENSE,
            tribe=TribeId.DRUID,
            damage=-10,
            defense=-5,
            range_hex=0
        )
        self.assertEqual(underpowered_card.damage, 0)
        self.assertEqual(underpowered_card.defense, 0)
        self.assertEqual(underpowered_card.range_hex, 1)

    def test_fantasy_composition_generation(self):
        """Verify dynamic composition synthesis for multiple tribes and doctrines."""
        comp_aggro = self.engine.generate_fantasy_composition(
            tribe_id=TribeId.TOXIC,
            doctrine=PlaystyleDoctrine.HYPER_AGGRO,
            deck_size=10,
            seed=42
        )
        self.assertEqual(comp_aggro.tribe, TribeId.TOXIC)
        self.assertEqual(comp_aggro.doctrine, PlaystyleDoctrine.HYPER_AGGRO)
        self.assertEqual(len(comp_aggro.cards), 10)
        self.assertGreater(comp_aggro.synergy_score, 0.0)
        self.assertLessEqual(comp_aggro.synergy_score, 100.0)
        self.assertEqual(comp_aggro.vital_max_hp, 6)

        # Check all cards obey max 6
        for c in comp_aggro.cards:
            self.assertLessEqual(c.damage, 6)
            self.assertLessEqual(c.defense, 6)

    def test_modular_arena_hex_formations(self):
        """Verify formation axial coordinates (q, r) within 19-hex arena radius."""
        for doctrine in [PlaystyleDoctrine.HYPER_AGGRO, PlaystyleDoctrine.FORTRESS, PlaystyleDoctrine.CONTROL_ICE]:
            comp = self.engine.generate_fantasy_composition(
                tribe_id=TribeId.CRYSTAL,
                doctrine=doctrine,
                deck_size=6
            )
            coords = comp.formation_hex_coords
            self.assertGreater(len(coords), 0)
            for q, r in coords:
                # Within radius 2 of 19-hex arena
                s = -q - r
                self.assertLessEqual(max(abs(q), abs(r), abs(s)), 2)

    def test_duel_clash_simulation(self):
        """Verify elemental duel resolution between two fantasy compositions."""
        comp_crystal = self.engine.generate_fantasy_composition(
            tribe_id=TribeId.CRYSTAL,
            doctrine=PlaystyleDoctrine.CONTROL_ICE,
            deck_size=6,
            seed=101
        )
        comp_toxic = self.engine.generate_fantasy_composition(
            tribe_id=TribeId.TOXIC,
            doctrine=PlaystyleDoctrine.HYPER_AGGRO,
            deck_size=6,
            seed=202
        )

        duel_res = self.engine.simulate_duel_clash(comp_crystal, comp_toxic, rounds=5)
        self.assertIn("winner", duel_res)
        self.assertIn("combat_log", duel_res)
        self.assertGreater(duel_res["rounds_played"], 0)
        self.assertEqual(duel_res["vital_max_hp_rule"], 6)

        # Remaining health of players must be in [0, 6]
        hp_a = duel_res["final_hp"]["player_a"]
        hp_b = duel_res["final_hp"]["player_b"]
        self.assertGreaterEqual(hp_a, 0)
        self.assertLessEqual(hp_a, 6)
        self.assertGreaterEqual(hp_b, 0)
        self.assertLessEqual(hp_b, 6)

    def test_catalog_summary(self):
        """Verify web-facing catalog summary contains all official metadata."""
        cat = self.engine.get_catalog_summary()
        self.assertEqual(cat["official_source"], "https://www.poslednikmen.cz/")
        self.assertEqual(cat["total_official_cards"], 129)
        self.assertEqual(cat["cards_per_tribe"], 43)
        self.assertEqual(cat["vital_max_hp_rule"], 6)
        self.assertEqual(len(cat["tribes"]), 5)
        self.assertIn("crystal", cat["archetype_cards"])
        self.assertIn("toxic", cat["archetype_cards"])
        self.assertIn("druid", cat["archetype_cards"])


if __name__ == "__main__":
    unittest.main()
