# ==============================================================================
# TESTS FOR TACTICAL GRADIENTS, ORBITAL PHYSICS, THE WEST DUELS & 60 ROSTER
# ==============================================================================

import unittest
import math
from typing import Dict, List, Any

from krystal_web_hub.economic_engine.tactical_gradients_and_orbital_physics import (
    TacticalGradientEngine,
    PinnedOrbitalSpellcraftingEngine
)
from krystal_web_hub.economic_engine.the_west_duel_algebra import (
    DuelTargetZone,
    DuelDodgeStance,
    DuelWeaponCategory,
    TheWestDuelAlgebra
)
from krystal_web_hub.economic_engine.sixty_character_roster import (
    CRYSTAL_ROSTER,
    TOXIC_ROSTER,
    DRUID_ROSTER,
    ALL_SIXTY_CHARACTERS,
    SixtyCharacterRosterEngine
)


class TestTacticalGradientsAndOrbitalPhysics(unittest.TestCase):
    """Verifies elevation gradient calculations, vector arrows, and orbital celestial physics."""

    def test_terrain_height_gradient_and_slope(self):
        height_map = {
            "0_0": 2.0,
            "1_0": 5.0,  # +3m East
            "-1_0": 0.0, # -2m West
            "0_1": 4.0,  # +2m South
            "0_-1": 1.0  # -1m North
        }

        grad = TacticalGradientEngine.calculate_terrain_gradient(
            height_map=height_map,
            center_hex=(0, 0),
            hex_spacing=1.732
        )

        self.assertGreater(grad["gradient_magnitude"], 0.0)
        self.assertGreater(grad["slope_angle_deg"], 10.0)
        self.assertEqual(grad["center_height_m"], 2.0)

    def test_dynamic_weapon_scale_elevation_advantage(self):
        # Downhill fire: Attacker at 6m, target at 1m (dh = +5m)
        downhill = TacticalGradientEngine.calculate_dynamic_weapon_scale(
            base_range=10.0,
            base_damage=4,
            attacker_elevation=6.0,
            target_elevation=1.0,
            gradient_dot_fire_dir=0.5,
            is_mortar=True
        )

        self.assertGreater(downhill["effective_range"], 10.0)
        self.assertGreaterEqual(downhill["effective_damage"], 4)
        self.assertGreater(downhill["kinetic_multiplier"], 1.0)

        # Uphill fire: Attacker at 1m, target at 5m (dh = -4m)
        uphill = TacticalGradientEngine.calculate_dynamic_weapon_scale(
            base_range=10.0,
            base_damage=4,
            attacker_elevation=1.0,
            target_elevation=5.0,
            gradient_dot_fire_dir=-0.5,
            is_mortar=False
        )

        self.assertLess(uphill["effective_range"], 10.0)

    def test_movement_vector_arrow_spline(self):
        arrow = TacticalGradientEngine.generate_movement_vector_arrow(
            start_pos=(0.0, 0.0, 0.0),
            end_pos=(5.0, 3.0, 0.0), # Climbing 3m uphill
            gradient_engine_data={}
        )

        self.assertEqual(arrow["start_point"], (0.0, 0.0, 0.0))
        self.assertGreater(arrow["control_mid_point"][1], 3.0) # Upward apex curve
        self.assertGreater(arrow["stamina_cost_multiplier"], 1.0)

    def test_pinned_celestial_orbital_positions(self):
        orbits = PinnedOrbitalSpellcraftingEngine.calculate_celestial_orbit_positions(
            caster_pos=(10.0, 0.0, 10.0),
            t=1.5,
            orbit_radius_major=2.5,
            orbit_radius_minor=2.0,
            harmony_key="fibonacci_triad"
        )

        self.assertEqual(len(orbits), 3)
        for body in orbits:
            self.assertIn("position_3d", body)
            self.assertGreater(body["distance_to_caster"], 1.0)
            self.assertIn("visual_glow", body)

        # Evaluate aura resonance
        aura = PinnedOrbitalSpellcraftingEngine.evaluate_aura_resonance_field(
            orbiting_bodies=orbits,
            caster_toughness=25,
            caster_reflexes=20
        )
        self.assertGreater(aura["aura_ward_shield_points"], 0)
        self.assertGreater(aura["projectile_deflection_chance"], 0.20)


class TestTheWestDuelAlgebra(unittest.TestCase):
    """Verifies The West duel mechanics: toughness vs melee, reflexes vs ranged, zones, and 6 Max HP."""

    def test_melee_attack_mitigated_by_toughness(self):
        attacker = {"aim": 20, "appearance": 25}
        defender = {"toughness": 25, "reflexes": 10, "dodge": 15, "tactics": 18, "mobility": 12}

        res = TheWestDuelAlgebra.resolve_duel_round(
            attacker_stats=attacker,
            defender_stats=defender,
            attack_zone=DuelTargetZone.TORSO.value,
            defense_stance=DuelDodgeStance.STAND_FIRM.value,
            weapon_type=DuelWeaponCategory.COLD_MELEE,
            base_weapon_damage=12,
            defender_current_hp=6
        )

        self.assertTrue(res["hit"])
        self.assertEqual(res["mitigation_stat"], "Toughness (Húževnatosť)")
        self.assertGreater(res["mitigation_amount"], 8.0)
        self.assertLess(res["net_damage_dealt"], 12)
        self.assertLessEqual(res["defender_hp_after"], 6)
        self.assertEqual(res["max_hp_invariant"], 6)

    def test_ranged_attack_mitigated_by_reflexes(self):
        attacker = {"aim": 24, "appearance": 15}
        defender = {"toughness": 10, "reflexes": 28, "dodge": 20, "tactics": 15, "mobility": 20}

        res = TheWestDuelAlgebra.resolve_duel_round(
            attacker_stats=attacker,
            defender_stats=defender,
            attack_zone=DuelTargetZone.RIGHT_SHOULDER.value,
            defense_stance=DuelDodgeStance.STAND_FIRM.value,
            weapon_type=DuelWeaponCategory.RANGED_PROJECTILE,
            base_weapon_damage=14,
            defender_current_hp=6
        )

        self.assertTrue(res["hit"])
        self.assertEqual(res["mitigation_stat"], "Reflexes (Reflexy)")
        self.assertGreater(res["mitigation_amount"], 10.0)

    def test_duck_down_evades_head_shot_completely(self):
        attacker = {"aim": 35, "appearance": 40}
        defender = {"toughness": 10, "reflexes": 15, "dodge": 15, "tactics": 10, "mobility": 15}

        res = TheWestDuelAlgebra.resolve_duel_round(
            attacker_stats=attacker,
            defender_stats=defender,
            attack_zone=DuelTargetZone.HEAD.value,
            defense_stance=DuelDodgeStance.DUCK_DOWN.value,
            weapon_type=DuelWeaponCategory.RANGED_PROJECTILE,
            base_weapon_damage=20,
            defender_current_hp=6
        )

        self.assertTrue(res["evaded"])
        self.assertFalse(res["hit"])
        self.assertEqual(res["damage_dealt"], 0)
        self.assertEqual(res["defender_hp_after"], 6)

    def test_the_west_archetype_catalog(self):
        builds = TheWestDuelAlgebra.DUEL_BUILDS
        self.assertIn("odolavac_pure_resistance", builds)
        self.assertIn("vystupovac_intimidator", builds)
        self.assertIn("chladny_taktik", builds)
        self.assertIn("pohyblivy_ostrelovac", builds)

        odolavac = builds["odolavac_pure_resistance"]["stat_biases"]
        self.assertGreater(odolavac["toughness"], 25)
        self.assertGreater(odolavac["reflexes"], 25)


class TestSixtyCharacterRoster(unittest.TestCase):
    """Verifies that exactly 20 characters exist per race (60 total) across tiers 1-5 with HUD panels."""

    def test_roster_counts_and_race_partitioning(self):
        self.assertEqual(len(CRYSTAL_ROSTER), 20)
        self.assertEqual(len(TOXIC_ROSTER), 20)
        self.assertEqual(len(DRUID_ROSTER), 20)
        self.assertEqual(SixtyCharacterRosterEngine.get_total_roster_count(), 60)

    def test_hierarchy_tier_distribution(self):
        for race_name in ["crystal", "toxic", "druid"]:
            chars = SixtyCharacterRosterEngine.get_characters_by_race(race_name)
            self.assertEqual(len(chars), 20)

            # Each tier 1 through 5 must have characters
            for tier in [1, 2, 3, 4, 5]:
                tier_chars = [c for c in chars if c["tier"] == tier]
                self.assertGreater(len(tier_chars), 0)

    def test_character_stats_and_hud_panel_integrity(self):
        # Test Kryštálový Archón
        archon = SixtyCharacterRosterEngine.get_character("c_archon_17")
        self.assertIsNotNone(archon)
        self.assertEqual(archon["name"], "Kryštálový Archón")
        self.assertEqual(archon["tier"], 5)
        self.assertEqual(archon["warhammer_stats"]["W"], 6) # Max HP rule
        self.assertGreaterEqual(archon["duel_stats"]["appearance"], 35)
        self.assertIn("aim_reticle", archon["hud_panel"])
        self.assertIn("orbital_node", archon["hud_panel"])

        # Test Toxický Hnilobník
        defiler = SixtyCharacterRosterEngine.get_character("t_defiler_17")
        self.assertIsNotNone(defiler)
        self.assertEqual(defiler["race"], "toxic")
        self.assertEqual(defiler["warhammer_stats"]["W"], 6)

        # Test Prastarý Druid
        druid = SixtyCharacterRosterEngine.get_character("d_druid_elder_17")
        self.assertIsNotNone(druid)
        self.assertEqual(druid["race"], "druid")
        self.assertEqual(druid["warhammer_stats"]["W"], 6)


if __name__ == "__main__":
    unittest.main()
