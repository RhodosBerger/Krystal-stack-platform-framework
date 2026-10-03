# ==============================================================================
# KRYSTAL-STACK: UNIT & INTEGRATION TESTS FOR SECTOR CONQUEST & FIGHT SYSTEM
# ==============================================================================
import unittest
import json
import urllib.request
import urllib.error
from krystal_web_hub.economic_engine import (
    Tribe, RoundPhase, EscalationStage, MatchState, CombatantState,
    SectorContestStatus, GarrisonUnit, Sector,
    generate_default_sectors, SectorConquestEngine, RoundController
)

class TestSectorFightSystem(unittest.TestCase):
    def setUp(self):
        self.match = RoundController.initialize_match("test_sector_match", Tribe.CRYSTAL, Tribe.TOXIC)

    def test_default_sectors_generation(self):
        sectors = self.match.sectors
        self.assertEqual(len(sectors), 5)
        
        sector_ids = [s.id for s in sectors]
        self.assertIn("sector_north_crystal", sector_ids)
        self.assertIn("sector_south_toxic", sector_ids)
        self.assertIn("sector_east_druid", sector_ids)
        self.assertIn("sector_west_sulfur", sector_ids)
        self.assertIn("sector_center_citadel", sector_ids)

        north = next(s for s in sectors if s.id == "sector_north_crystal")
        self.assertEqual(north.owner, "player")
        self.assertEqual(north.fortification_hp, 8)
        self.assertEqual(len(north.garrison), 1)

        south = next(s for s in sectors if s.id == "sector_south_toxic")
        self.assertEqual(south.owner, "enemy")
        self.assertEqual(south.fortification_hp, 7)

        citadel = next(s for s in sectors if s.id == "sector_center_citadel")
        self.assertEqual(citadel.owner, "neutral")
        self.assertEqual(citadel.fortification_hp, 12)
        self.assertEqual(citadel.fortification_level, 3)

    def test_sector_assault_fortification_damage(self):
        initial_mana = self.match.player.mana
        success, details, msg = SectorConquestEngine.resolve_sector_assault(
            self.match, "sector_center_citadel", attacker_side="player", attack_power=4, mana_committed=2
        )
        self.assertTrue(success)
        self.assertEqual(details["fort_damage"], 4)
        self.assertEqual(details["fort_hp_left"], 8)
        self.assertEqual(self.match.player.mana, initial_mana - 2)

        citadel = SectorConquestEngine.get_sector_by_id(self.match, "sector_center_citadel")
        self.assertEqual(citadel.fortification_hp, 8)
        self.assertIn(citadel.status, [SectorContestStatus.UNDER_SIEGE, SectorContestStatus.CONTESTED])

    def test_sector_assault_cannot_attack_own_sector(self):
        success, details, msg = SectorConquestEngine.resolve_sector_assault(
            self.match, "sector_north_crystal", attacker_side="player", attack_power=3, mana_committed=2
        )
        self.assertFalse(success)
        self.assertIn("už vlastníte", msg)

    def test_sector_full_annexation_and_bounty(self):
        # Center citadel has 12 fort HP + garrison with 8 HP (total 20 HP)
        # Assault with attack power 25 to breach and conquer completely
        self.match.player.mana = 10
        prev_crystals = self.match.player.aether_crystals

        success, details, msg = SectorConquestEngine.resolve_sector_assault(
            self.match, "sector_center_citadel", attacker_side="player", attack_power=25, mana_committed=2
        )
        self.assertTrue(success)
        self.assertTrue(details["captured"])
        self.assertEqual(details["new_owner"], "player")
        self.assertEqual(details["garrison_alive"], 0)

        citadel = SectorConquestEngine.get_sector_by_id(self.match, "sector_center_citadel")
        self.assertEqual(citadel.owner, "player")
        self.assertEqual(citadel.capture_progress, 100)
        self.assertEqual(citadel.status, SectorContestStatus.CAPTURED)
        # Check conquest bounty
        self.assertEqual(self.match.player.aether_crystals, prev_crystals + 1)
        # Check ledger recorded conquest
        last_ledger = self.match.ledger[-1]
        self.assertEqual(last_ledger.event_type, "SECTOR_ANNEXED")

    def test_sector_fortification_upgrade(self):
        north = SectorConquestEngine.get_sector_by_id(self.match, "sector_north_crystal")
        prev_level = north.fortification_level
        prev_garrison_count = len(north.garrison)
        self.match.player.mana = 8

        success, details, msg = SectorConquestEngine.fortify_sector(
            self.match, "sector_north_crystal", side="player"
        )
        self.assertTrue(success)
        self.assertEqual(north.fortification_level, prev_level + 1)
        self.assertEqual(len(north.garrison), prev_garrison_count + 1)
        self.assertEqual(self.match.player.mana, 5) # 8 - 3

    def test_sector_fortify_fails_on_unowned(self):
        success, details, msg = SectorConquestEngine.fortify_sector(
            self.match, "sector_south_toxic", side="player"
        )
        self.assertFalse(success)
        self.assertIn("Môžete opevňovať iba sektory pod vašou kontrolou", msg)

    def test_sector_yields_integration_in_phase_1(self):
        # Round 1 starts in Phase 1
        p_mana_before = self.match.player.mana
        p_cryst_before = self.match.player.aether_crystals

        # Player owns sector_north_crystal (+2 Mana, +2 Crystals)
        yields = SectorConquestEngine.apply_sector_round_yields(self.match)
        self.assertEqual(yields["player"]["mana"], 2)
        self.assertEqual(yields["player"]["aether_crystal"], 2)
        self.assertEqual(self.match.player.aether_crystals, p_cryst_before + 2)

    def test_escalation_surge_and_apex_attack_multipliers(self):
        # Round 2 Surge -> +25% attack power
        self.match.escalation = EscalationStage.ROUND_2_SURGE
        citadel = SectorConquestEngine.get_sector_by_id(self.match, "sector_center_citadel")
        citadel.fortification_hp = 12

        success, details, _ = SectorConquestEngine.resolve_sector_assault(
            self.match, "sector_center_citadel", attacker_side="player", attack_power=8, mana_committed=2
        )
        self.assertTrue(success)
        # 8 * 1.25 = 10 effective power
        self.assertEqual(details["attack_power_used"], 10)
        self.assertEqual(details["fort_damage"], 10)

        # Round 3 Apex -> +50% attack power
        self.match.escalation = EscalationStage.ROUND_3_APEX
        citadel.fortification_hp = 12
        self.match.player.mana = 10
        success, details_apex, _ = SectorConquestEngine.resolve_sector_assault(
            self.match, "sector_center_citadel", attacker_side="player", attack_power=8, mana_committed=2
        )
        self.assertTrue(success)
        # 8 * 1.5 = 12 effective power
        self.assertEqual(details_apex["attack_power_used"], 12)

    def test_live_http_api_sectors(self):
        # Query HTTP API on port 8089
        try:
            req = urllib.request.urlopen("http://127.0.0.1:8089/api/sectors")
            data = json.loads(req.read().decode('utf-8'))
            self.assertIn("sectors", data)
            self.assertEqual(len(data["sectors"]), 5)
        except urllib.error.URLError:
            self.skipTest("Live engine server on port 8089 not reachable")

if __name__ == '__main__':
    unittest.main()
