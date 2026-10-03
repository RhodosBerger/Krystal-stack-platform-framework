# ==============================================================================
# KRYSTAL-STACK: UNIT TESTS FOR ENVIRONMENTAL MATRICES, TRIGONOMETRY & CARD ALGEBRA
# ==============================================================================

import unittest
import math
from krystal_web_hub.economic_engine import (
    Tribe,
    EnvironmentalHazardType,
    EnvironmentalCell,
    EnvironmentalMatrix,
    CombatSector,
    TacticalTrigonometry,
    MatrixTacticalAI,
    CardActionType,
    HypergeometricCardStatistics,
    CardCombatAlgebra
)

class TestEnvironmentalMatricesAndTacticalTrigonometry(unittest.TestCase):

    def setUp(self):
        self.matrix = EnvironmentalMatrix(radius=2)

    # --------------------------------------------------------------------------
    # 1. ENVIRONMENTAL MATRIX & TURN-BASED DECAY TIMERS
    # --------------------------------------------------------------------------
    def test_matrix_initialization(self):
        # Radius 2 pointy-topped axial grid has 3*R*(R+1) + 1 = 19 hexes
        self.assertEqual(len(self.matrix.cells), 19)
        center = self.matrix.get_cell(0, 0)
        self.assertIsNotNone(center)
        self.assertEqual(center.hazard, EnvironmentalHazardType.NONE)
        self.assertEqual(center.hazard_timer, 0)

    def test_apply_hazard_and_decay_ticks(self):
        # Apply toxic acid to center hex for 3 turns with potency 2
        res = self.matrix.apply_action_footprint(
            center_q=0, center_r=0,
            footprint_type="point",
            hazard=EnvironmentalHazardType.TOXIC_ACID,
            duration=3,
            potency=2
        )
        self.assertIn([0, 0], res["affected_cells"])

        cell = self.matrix.get_cell(0, 0)
        self.assertEqual(cell.hazard, EnvironmentalHazardType.TOXIC_ACID)
        self.assertEqual(cell.hazard_timer, 3)
        self.assertEqual(cell.hazard_potency, 2)
        self.assertEqual(cell.traversability_cost, 1.5)

        # Tick 1: timer becomes 2, unit on [0, 0] takes 2 tick damage
        tick1 = self.matrix.process_round_decay_and_ticks(unit_positions={"unit_a": (0, 0)})
        self.assertEqual(cell.hazard_timer, 2)
        self.assertEqual(len(tick1["unit_tick_events"]), 1)
        self.assertEqual(tick1["unit_tick_events"][0]["damage"], 2)
        self.assertEqual(tick1["unit_tick_events"][0]["status_applied"], "poisoned")

        # Tick 2: timer becomes 1
        self.matrix.process_round_decay_and_ticks()
        self.assertEqual(cell.hazard_timer, 1)

        # Tick 3: timer expires (reaches 0) -> hazard resets to NONE
        tick3 = self.matrix.process_round_decay_and_ticks()
        self.assertEqual(cell.hazard_timer, 0)
        self.assertEqual(cell.hazard, EnvironmentalHazardType.NONE)
        self.assertEqual(cell.hazard_potency, 0)
        self.assertEqual(cell.traversability_cost, 1.0)
        self.assertEqual(tick3["expired_cells_count"], 1)

    def test_elemental_chain_reaction_combustion(self):
        # Coat hex with Toxic Acid
        self.matrix.apply_action_footprint(
            center_q=1, center_r=-1,
            footprint_type="point",
            hazard=EnvironmentalHazardType.TOXIC_ACID,
            duration=3,
            potency=1
        )
        # Hit same hex with Burning Aether -> triggers combustion reaction!
        res = self.matrix.apply_action_footprint(
            center_q=1, center_r=-1,
            footprint_type="point",
            hazard=EnvironmentalHazardType.BURNING_AETHER,
            duration=2,
            potency=2,
            trigger_reaction=True
        )
        self.assertEqual(len(res["reactions"]), 1)
        self.assertEqual(res["reactions"][0]["type"], "combustion_shockwave")
        self.assertEqual(res["reactions"][0]["splash_damage"], 2)

        # Cell is cleared by the combustion explosion
        cell = self.matrix.get_cell(1, -1)
        self.assertEqual(cell.hazard, EnvironmentalHazardType.NONE)

    # --------------------------------------------------------------------------
    # 2. TRIGONOMETRIC KINEMATICS, SECTORS & FACING ANGLES
    # --------------------------------------------------------------------------
    def test_facing_angle_and_sectors(self):
        # Target at [0, 0] facing directly along positive X-axis (facing_angle = 0.0)
        target_pos = (0, 0)
        target_facing = 0.0

        # Attacker 1: directly in front along positive X -> FRONT sector
        att_front = (1, 0)
        sector, delta_deg = TacticalTrigonometry.determine_engagement_sector(att_front, target_pos, target_facing)
        self.assertEqual(sector, CombatSector.FRONT)
        self.assertAlmostEqual(delta_deg, 0.0, places=1)

        # Attacker 2: directly behind along negative X -> REAR sector (Backstab)
        att_rear = (-1, 0)
        sector_rear, delta_deg_rear = TacticalTrigonometry.determine_engagement_sector(att_rear, target_pos, target_facing)
        self.assertEqual(sector_rear, CombatSector.REAR)
        self.assertTrue(abs(delta_deg_rear) >= 135.0)

        # Attacker 3: to the flank
        att_flank = (0, 1)
        sector_flank, _ = TacticalTrigonometry.determine_engagement_sector(att_flank, target_pos, target_facing)
        self.assertIn(sector_flank, (CombatSector.FLANK_LEFT, CombatSector.FLANK_RIGHT))

    def test_elevation_pitch_angle(self):
        # High ground advantage: Attacker at 3.0m, target at 0.0m, dist 5.0m
        phi = TacticalTrigonometry.calculate_elevation_pitch_angle(h_attacker=3.0, h_target=0.0, horizontal_distance_m=5.0)
        self.assertGreater(phi, 0.50) # positive angle

        # Low ground: Attacker at 0.0m, target at 2.5m
        phi_low = TacticalTrigonometry.calculate_elevation_pitch_angle(h_attacker=0.0, h_target=2.5, horizontal_distance_m=5.0)
        self.assertLess(phi_low, 0.0)

    def test_line_of_sight_obstruction(self):
        # Line between [0, -2] and [0, 2] passes through [0, 0]
        # Place heavy cover on [0, 0]
        center_cell = self.matrix.get_cell(0, 0)
        center_cell.cover_density = 0.90 # Obstacle

        los = TacticalTrigonometry.check_line_of_sight((0, -2), (0, 2), self.matrix)
        self.assertFalse(los["clear"])
        self.assertEqual(los["blocked_by"], [0, 0])

        # Clear cover
        center_cell.cover_density = 0.20
        los_clear = TacticalTrigonometry.check_line_of_sight((0, -2), (0, 2), self.matrix)
        self.assertTrue(los_clear["clear"])

    # --------------------------------------------------------------------------
    # 3. HYPERGEOMETRIC COMBINATORIAL COUNTING MODEL
    # --------------------------------------------------------------------------
    def test_hypergeometric_probabilities(self):
        # 43-card deck, 5 legendary cards, draw 5 cards opening hand
        N = 43
        K = 5
        n = 5

        # P(X = 0): No legendary drawn
        p0 = HypergeometricCardStatistics.probability_draw_exact_k(N, K, n, 0)
        self.assertGreater(p0, 0.45)
        self.assertLess(p0, 0.60)

        # P(X >= 1): At least 1 legendary
        p_at_least_1 = HypergeometricCardStatistics.probability_draw_at_least_k(N, K, n, 1)
        self.assertAlmostEqual(p0 + p_at_least_1, 1.0, places=4)

        # Expected turns to draw
        e_turns = HypergeometricCardStatistics.expected_turn_to_draw(N, K, opening_hand=5)
        self.assertGreater(e_turns, 1.0)
        self.assertLess(e_turns, 5.0)

    def test_mana_curve_analysis(self):
        mock_deck = [
            {"name": "Card A", "cost": 1},
            {"name": "Card B", "cost": 2},
            {"name": "Card C", "cost": 2},
            {"name": "Card D", "cost": 3},
            {"name": "Card E", "cost": 4},
        ]
        curve = HypergeometricCardStatistics.analyze_mana_curve(mock_deck)
        self.assertEqual(curve["deck_size"], 5)
        self.assertEqual(curve["mean_cost"], 2.4)
        self.assertEqual(curve["cost_breakdown"], {1: 1, 2: 2, 3: 1, 4: 1})

    # --------------------------------------------------------------------------
    # 4. CARD COMBAT ALGEBRA & 6 MAX HP INVARIANT
    # --------------------------------------------------------------------------
    def test_card_combat_algebra_backstab_and_armor(self):
        card = {"name": "Shadow Dagger", "hp_delta": -2, "attack_type": "melee"}
        attacker = {"hp": 6, "skill": 3}
        defender = {"hp": 5, "armor": 2}

        # Frontal attack: 2 damage mitigated by 2 armor -> 0 net damage
        res_front = CardCombatAlgebra.resolve_card_action(
            card=card, attacker_stats=attacker, defender_stats=defender,
            sector=CombatSector.FRONT, d6_hit_roll=4
        )
        self.assertTrue(res_front["hit_success"])
        self.assertEqual(res_front["net_damage_dealt"], 0)
        self.assertEqual(res_front["defender_new_hp"], 5)

        # Rear attack (Backstab): Ignores armor + 1 crit bonus -> 3 net damage!
        res_rear = CardCombatAlgebra.resolve_card_action(
            card=card, attacker_stats=attacker, defender_stats=defender,
            sector=CombatSector.REAR, d6_hit_roll=4
        )
        self.assertTrue(res_rear["hit_success"])
        self.assertEqual(res_rear["net_damage_dealt"], 3)
        self.assertEqual(res_rear["defender_new_hp"], 2) # 5 - 3 = 2 HP

    def test_vital_6_max_hp_clamp(self):
        card_lethal = {"name": "Orbital Beam", "hp_delta": -10, "attack_type": "ranged"}
        attacker = {"hp": 6, "skill": 2}
        defender = {"hp": 3, "armor": 0}

        # Massive overkill damage clamps at 0 (never negative)
        res_overkill = CardCombatAlgebra.resolve_card_action(
            card=card_lethal, attacker_stats=attacker, defender_stats=defender,
            sector=CombatSector.FRONT, d6_hit_roll=5
        )
        self.assertEqual(res_overkill["defender_new_hp"], 0)

        # Massive heal clamps at 6 Max HP (never exceeds 6)
        card_mega_heal = {"name": "Full Gaia Rebirth", "hp_delta": 10, "attack_type": "self"}
        injured_attacker = {"hp": 4, "armor": 0}
        res_heal = CardCombatAlgebra.resolve_card_action(
            card=card_mega_heal, attacker_stats=injured_attacker, defender_stats=defender
        )
        self.assertEqual(res_heal["attacker_new_hp"], 6)
        self.assertEqual(res_heal["healed_amount"], 2) # only 2 gained before hit max 6 cap

    # --------------------------------------------------------------------------
    # 5. MATRIX TACTICAL AI DECISION ENGINE
    # --------------------------------------------------------------------------
    def test_matrix_tactical_ai_evaluation(self):
        ai = MatrixTacticalAI(tribe=Tribe.CRYSTAL, env_matrix=self.matrix)
        mock_cards = [
            {"id": "frost_shard", "name": "Mrazivý Črep", "cost": 1, "hp_delta": -1, "attack_type": "ranged", "min_range": 1, "max_range": 3},
            {"id": "glacial_lance", "name": "Ľadovcová Kopija", "cost": 3, "hp_delta": -3, "attack_type": "ranged", "min_range": 2, "max_range": 4}
        ]

        decision = ai.evaluate_tactical_position_and_action(
            npc_coord=(0, 2),
            npc_hp=6,
            npc_mana=5,
            target_coord=(0, 0),
            target_hp=3,
            target_facing_angle=0.0,
            available_cards=mock_cards
        )
        self.assertTrue(decision["action_executed"])
        self.assertIsNotNone(decision["selected_move"])
        self.assertIn("glacial_lance", decision["selected_card"]["id"])

if __name__ == '__main__':
    unittest.main()
