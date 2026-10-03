"""
Unit & Integration Test Suite for Krystal-Stack Economic Game Framework
"""

import unittest
import urllib.request
import json
import time

from krystal_web_hub.economic_engine import (
    Tribe, ResourceType, BuildingType, AbilityType, RoundPhase,
    EscalationStage, ResourceCost, BuildingSpec, AbilitySpec,
    CombatantState, MatchState,
    BUILDING_REGISTRY, EconomicLedger,
    ABILITY_REGISTRY, AbilityEngine,
    RoundController, SCENARIO_TEMPLATES, get_all_templates,
    generate_godot_tscn
)

BASE_URL = "http://127.0.0.1:8089"

class TestEconomicEngineModels(unittest.TestCase):
    def test_resource_cost(self):
        cost = ResourceCost(mana=5, aether_crystal=2)
        d = cost.to_dict()
        self.assertEqual(d["mana"], 5)
        self.assertEqual(d["aether_crystal"], 2)

    def test_building_registry(self):
        self.assertEqual(len(BUILDING_REGISTRY), 9)
        conduit = BUILDING_REGISTRY[BuildingType.AETHER_CONDUIT]
        self.assertEqual(conduit.tribe, Tribe.CRYSTAL)
        self.assertEqual(conduit.mana_generation, 3)

        tree = BUILDING_REGISTRY[BuildingType.WORLD_TREE]
        self.assertEqual(tree.tribe, Tribe.DRUID)
        self.assertEqual(tree.max_hp, 6)

    def test_ability_registry(self):
        self.assertGreaterEqual(len(ABILITY_REGISTRY), 12)
        meteor = ABILITY_REGISTRY["crystal_meteor"]
        self.assertEqual(meteor.damage, 2)
        
        combo = ABILITY_REGISTRY["corrosive_shatter_combo"]
        self.assertEqual(combo.combo_prerequisite, "rooted")
        self.assertEqual(combo.combo_multiplier, 1.5)

class TestEconomicRulesAndLedger(unittest.TestCase):
    def setUp(self):
        self.combatant = CombatantState(
            name="Test Hero",
            tribe=Tribe.CRYSTAL,
            hp=6,
            max_hp=6,
            armor=0,
            mana=10,
            max_mana=15,
            aether_crystals=4,
            toxic_slime=2,
            amber_runes=2
        )

    def test_affordability_and_deduction(self):
        cost = ResourceCost(mana=4, aether_crystal=2)
        can_afford, msg = EconomicLedger.can_afford(self.combatant, cost)
        self.assertTrue(can_afford)

        deltas = EconomicLedger.deduct_resources(self.combatant, cost)
        self.assertEqual(self.combatant.mana, 6)
        self.assertEqual(self.combatant.aether_crystals, 2)
        self.assertEqual(deltas["mana"], -4)

    def test_construction_and_passive_perks(self):
        # Build Aether Conduit
        b, deltas, status = EconomicLedger.construct_building(
            self.combatant, BuildingType.AETHER_CONDUIT, [1, 0], [1.73, 0, 0], EscalationStage.ROUND_1_SKIRMISH
        )
        self.assertIsNotNone(b)
        self.assertEqual(status, "SUCCESS")
        self.assertEqual(len(self.combatant.buildings), 1)

        # Build Capacitor Tower (extends max mana from 15 to 20)
        self.combatant.mana = 10
        self.combatant.aether_crystals = 5
        b2, deltas2, status2 = EconomicLedger.construct_building(
            self.combatant, BuildingType.CAPACITOR_TOWER, [-1, 0], [-1.73, 0, 0], EscalationStage.ROUND_1_SKIRMISH
        )
        self.assertIsNotNone(b2)
        self.assertEqual(self.combatant.max_mana, 20)

    def test_round_yield_with_escalation(self):
        # Add a building with 3 mana generation
        self.combatant.buildings.append(BUILDING_REGISTRY[BuildingType.AETHER_CONDUIT])
        self.combatant.mana = 0

        # Round 1: Skirmish (base 4 + 3 = 7 mana)
        y1 = EconomicLedger.calculate_round_yield(self.combatant, EscalationStage.ROUND_1_SKIRMISH)
        self.assertEqual(y1["mana"], 7)

        # Round 2: Surge (+50% yields: (4 + 3) * 1.5 = 10 mana)
        self.combatant.mana = 0
        y2 = EconomicLedger.calculate_round_yield(self.combatant, EscalationStage.ROUND_2_SURGE)
        self.assertEqual(y2["mana"], 10)

class TestAbilityCombosAndEscalation(unittest.TestCase):
    def setUp(self):
        self.caster = CombatantState("Player", Tribe.CRYSTAL, hp=6, max_hp=6, armor=0, mana=15, aether_crystals=5, toxic_slime=5, amber_runes=5)
        self.target = CombatantState("Enemy", Tribe.TOXIC, hp=6, max_hp=6, armor=2, mana=10)

    def test_combo_prerequisite_check(self):
        # Corrosive shatter requires rooted
        ok, res, msg = AbilityEngine.cast_ability(self.caster, self.target, "corrosive_shatter_combo", EscalationStage.ROUND_2_SURGE)
        self.assertFalse(ok)
        self.assertIn("Combo zlyhalo", msg)

        # Apply root first
        self.target.active_statuses["rooted"] = 1
        ok2, res2, msg2 = AbilityEngine.cast_ability(self.caster, self.target, "corrosive_shatter_combo", EscalationStage.ROUND_2_SURGE)
        self.assertTrue(ok2)
        self.assertTrue(res2["is_combo"])
        # Armor was 2, damage was 4 * 1.5 = 6. 2 absorbed by armor, 4 dealt to HP. HP becomes 6 - 4 = 2.
        self.assertEqual(self.target.hp, 2)
        self.assertEqual(self.target.armor, 0)

    def test_tier_3_ultimate_gating(self):
        # Ultimates should reject in Round 1 Skirmish
        ok, res, msg = AbilityEngine.cast_ability(self.caster, self.target, "supernova_cataclysm", EscalationStage.ROUND_1_SKIRMISH)
        self.assertFalse(ok)
        self.assertIn("odomknuté až od 3. kola", msg)

        # Ultimates should succeed in Round 3 Apex
        ok2, res2, msg2 = AbilityEngine.cast_ability(self.caster, self.target, "supernova_cataclysm", EscalationStage.ROUND_3_APEX)
        self.assertTrue(ok2)
        self.assertGreater(res2["damage_dealt"], 0)

class TestRoundStateController(unittest.TestCase):
    def test_phase_cyclical_progression(self):
        match = RoundController.initialize_match("test_cycle", Tribe.CRYSTAL, Tribe.TOXIC)
        self.assertEqual(match.round_number, 1)
        self.assertEqual(match.phase, RoundPhase.PHASE_1_ECONOMY)

        # Step 1: Economy -> Build
        s1 = RoundController.step_phase(match)
        self.assertEqual(s1["phase"], "build")

        # Step 2: Build -> Skirmish
        s2 = RoundController.step_phase(match)
        self.assertEqual(s2["phase"], "skirmish")

        # Step 3: Skirmish -> Escalate
        s3 = RoundController.step_phase(match)
        self.assertEqual(s3["phase"], "escalate")

        # Step 4: Escalate -> Round 2 Economy (Industrial Surge)
        s4 = RoundController.step_phase(match)
        self.assertEqual(s4["round_number"], 2)
        self.assertEqual(s4["escalation"], "industrial_surge")
        self.assertEqual(s4["phase"], "economy")

    def test_tscn_scene_export(self):
        match = RoundController.initialize_match("test_tscn", Tribe.CRYSTAL, Tribe.TOXIC)
        tscn = generate_godot_tscn(match)
        self.assertTrue(tscn.startswith('[gd_scene format=3'))
        self.assertIn('HexTile_0_0', tscn)
        self.assertIn('EconomicMatchRoot', tscn)

class TestEconomicHttpApi(unittest.TestCase):
    def test_api_state_and_step(self):
        res = urllib.request.urlopen(f"{BASE_URL}/api/economy/state")
        self.assertEqual(res.status, 200)
        data = json.loads(res.read().decode('utf-8'))
        self.assertIn("match_id", data)
        self.assertIn("player", data)
        self.assertIn("enemy", data)
        self.assertIn("ledger", data)

    def test_api_buildings_and_abilities(self):
        r_b = urllib.request.urlopen(f"{BASE_URL}/api/economy/buildings")
        d_b = json.loads(r_b.read().decode('utf-8'))
        self.assertEqual(len(d_b["buildings"]), 9)

        r_a = urllib.request.urlopen(f"{BASE_URL}/api/economy/abilities")
        d_a = json.loads(r_a.read().decode('utf-8'))
        self.assertGreaterEqual(len(d_a["abilities"]), 12)

    def test_api_templates(self):
        res = urllib.request.urlopen(f"{BASE_URL}/api/economy/templates")
        self.assertEqual(res.status, 200)
        data = json.loads(res.read().decode('utf-8'))
        self.assertIn("scenarios", data)
        self.assertIn("skirmish_severni_stity", data["scenarios"])

    def test_api_tscn_export(self):
        res = urllib.request.urlopen(f"{BASE_URL}/api/economy/tscn")
        self.assertEqual(res.status, 200)
        content = res.read().decode('utf-8')
        self.assertTrue(content.startswith('[gd_scene format=3'))

if __name__ == '__main__':
    unittest.main()
