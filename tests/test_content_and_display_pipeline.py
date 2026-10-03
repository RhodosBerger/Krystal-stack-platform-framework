import os
import unittest
import urllib.request
import json
from krystal_web_hub.economic_engine import (
    Tribe,
    generate_tribal_deck,
    PlayerDeckManager,
    get_full_tribal_card_catalog,
    UnitProfile,
    UnitClassification,
    SpecialRule,
    CANONICAL_HERO_PROFILES,
    CANONICAL_MINION_PROFILES,
    create_unit_instance,
    get_all_unit_archetypes,
    RACE_AND_SPECIALIZATION_REGISTRY,
    MAP_LEGEND_SPECIFICATION,
    get_race_and_specializations_catalog,
    get_map_legend_data
)


class TestGameContentAndDisplayPipeline(unittest.TestCase):
    """
    Validates the game content expansion (43-card decks per tribe, Warhammer statblocks),
    principles of display elements, and fetch dependency architecture.
    """

    def test_01_deck_composition_43_cards_per_tribe(self):
        """Verifies each tribe generates an exact 43-card deck with 25 T1, 13 T2, and 5 T3 cards."""
        for tribe in [Tribe.CRYSTAL, Tribe.TOXIC, Tribe.DRUID]:
            deck = generate_tribal_deck(tribe, total_cards=43)
            self.assertEqual(len(deck), 43, f"{tribe.value} deck must have exactly 43 cards.")

            t1_cards = [c for c in deck if c.get("tier") == 1]
            t2_cards = [c for c in deck if c.get("tier") == 2]
            t3_cards = [c for c in deck if c.get("tier") == 3]

            self.assertEqual(len(t1_cards), 25, f"{tribe.value} must have 25 Tier 1 cards.")
            self.assertEqual(len(t2_cards), 13, f"{tribe.value} must have 13 Tier 2 cards.")
            self.assertEqual(len(t3_cards), 5, f"{tribe.value} must have 5 Tier 3 cards.")

    def test_02_card_catalog_structure_and_tribal_diversity(self):
        """Verifies master card catalog covers all 3 tribes with distinct descriptions and costs."""
        catalog = get_full_tribal_card_catalog()
        self.assertIn("crystal", catalog)
        self.assertIn("toxic", catalog)
        self.assertIn("druid", catalog)

        for tribe_key, templates in catalog.items():
            self.assertGreaterEqual(len(templates), 14, f"{tribe_key} must have at least 14 archetypes.")
            for tmpl in templates:
                self.assertIn("id_base", tmpl)
                self.assertIn("name", tmpl)
                self.assertIn("cost", tmpl)
                self.assertIn("attack_type", tmpl)
                self.assertIn("desc", tmpl)
                self.assertGreater(len(tmpl["desc"]), 5)

    def test_03_hero_vital_invariant_max_6_hp(self):
        """Verifies all champions adhere to the axiomatic 6 Max HP vital invariant."""
        for hero_id, profile in CANONICAL_HERO_PROFILES.items():
            self.assertEqual(profile["wounds_max"], 6, f"Hero '{hero_id}' must have max 6 HP.")
            self.assertEqual(profile["current_wounds"], 6)
            self.assertGreaterEqual(profile["armor_save"], 2)
            self.assertIsNotNone(profile["invulnerable_save"])

    def test_04_minion_archetypes_and_special_rules(self):
        """Verifies minion profiles have valid Warhammer stats and special rules."""
        self.assertEqual(len(CANONICAL_MINION_PROFILES), 6)
        for minion_id, p in CANONICAL_MINION_PROFILES.items():
            self.assertIn(p["classification"], [UnitClassification.BEAST, UnitClassification.CONSTRUCT])
            self.assertGreater(p["move"], 0)
            self.assertGreater(p["attacks"], 0)
            self.assertTrue(len(p["special_rules"]) > 0)

    def test_05_unit_instance_combat_mutation(self):
        """Verifies taking damage and healing bounds on UnitProfile."""
        unit = create_unit_instance("crystal_archon", spawn_hex=[0, -2])
        self.assertEqual(unit.current_wounds, 6)
        self.assertEqual(unit.current_hex, [0, -2])

        # Take 3 damage
        dmg = unit.take_damage(3)
        self.assertEqual(dmg, 3)
        self.assertEqual(unit.current_wounds, 3)
        self.assertTrue(unit.is_alive)

        # Heal 5 (should cap at wounds_max 6)
        healed = unit.heal(5)
        self.assertEqual(healed, 3)
        self.assertEqual(unit.current_wounds, 6)

        # Lethal damage
        unit.take_damage(10)
        self.assertEqual(unit.current_wounds, 0)
        self.assertFalse(unit.is_alive)

    def test_06_player_deck_manager_reshuffle_cycle(self):
        """Verifies PlayerDeckManager hand drawing, discarding, and deck reshuffling on exhaustion."""
        mgr = PlayerDeckManager(Tribe.CRYSTAL, total_deck_size=10)
        self.assertEqual(len(mgr.draw_pile), 10)
        self.assertEqual(len(mgr.hand), 0)

        # Draw full hand
        drawn = mgr.draw_to_full()
        self.assertEqual(len(drawn), 5)
        self.assertEqual(len(mgr.hand), 5)
        self.assertEqual(len(mgr.draw_pile), 5)

        # Play all 5 cards
        card_ids = [c["id"] for c in list(mgr.hand)]
        for cid in card_ids:
            played = mgr.play_card(cid)
            self.assertIsNotNone(played)

        self.assertEqual(len(mgr.hand), 0)
        self.assertEqual(len(mgr.discard_pile), 5)

        # Draw 5 more (emptying draw_pile)
        mgr.draw_to_full()
        self.assertEqual(len(mgr.hand), 5)
        self.assertEqual(len(mgr.draw_pile), 0)

        # Discard 1 card from hand (discard pile now has 5 previous + 1 = 6 cards)
        to_discard = mgr.hand[0]["id"]
        mgr.play_card(to_discard)
        self.assertEqual(len(mgr.discard_pile), 6)
        self.assertEqual(len(mgr.hand), 4)

        # Draw to full (draw_pile is empty, so it reshuffles 6 discard cards into draw_pile, draws 1, leaving 5)
        mgr.draw_to_full()
        self.assertEqual(len(mgr.hand), 5)
        self.assertEqual(len(mgr.draw_pile), 5)
        self.assertEqual(len(mgr.discard_pile), 0)

    def test_07_local_vendor_assets_exist(self):
        """Verifies local vendor bundle files exist for air-gapped / offline resilience."""
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        three_path = os.path.join(base_dir, "krystal_web_hub", "static", "vendor", "three.min.js")
        orbit_path = os.path.join(base_dir, "krystal_web_hub", "static", "vendor", "OrbitControls.js")
        loader_path = os.path.join(base_dir, "krystal_web_hub", "static", "asset_loader.js")

        self.assertTrue(os.path.exists(three_path), "Local three.min.js must exist.")
        self.assertGreater(os.path.getsize(three_path), 100000)

        self.assertTrue(os.path.exists(orbit_path), "Local OrbitControls.js must exist.")
        self.assertGreater(os.path.getsize(orbit_path), 5000)

        self.assertTrue(os.path.exists(loader_path), "Asset loader script must exist.")

    def test_08_api_assets_manifest_endpoint(self):
        """Tests GET /api/assets/manifest returns healthy manifest structure."""
        url = "http://127.0.0.1:8089/api/assets/manifest"
        req = urllib.request.urlopen(url)
        self.assertEqual(req.status, 200)
        data = json.loads(req.read().decode("utf-8"))
        self.assertEqual(data.get("status"), "HEALTHY_OFFLINE_READY")
        self.assertIn("vendor_scripts", data)
        self.assertIn("procedural_meshes", data)

    def test_09_api_game_content_catalog_endpoint(self):
        """Tests GET /api/game/content-catalog returns full unified content database."""
        url = "http://127.0.0.1:8089/api/game/content-catalog"
        req = urllib.request.urlopen(url)
        self.assertEqual(req.status, 200)
        data = json.loads(req.read().decode("utf-8"))
        self.assertIn("tribal_cards", data)
        self.assertIn("unit_archetypes", data)
        self.assertIn("crafting_templates", data)
        self.assertIn("affixes", data)
        self.assertIn("rarity_limits", data)
        self.assertIn("checkpoints", data)
        self.assertGreaterEqual(len(data["unit_archetypes"]), 9)

    def test_10_api_game_match_state_snapshot_endpoint(self):
        """Tests GET /api/game/match-state returns complete single-request hydration payload."""
        url = "http://127.0.0.1:8089/api/game/match-state"
        req = urllib.request.urlopen(url)
        self.assertEqual(req.status, 200)
        data = json.loads(req.read().decode("utf-8"))
        self.assertIn("turn", data)
        self.assertIn("game_state", data)
        self.assertIn("units", data)
        self.assertIn("ward", data)
        self.assertIn("flags", data)
        self.assertIn("ledger", data)

    def test_11_race_and_specialization_matrix(self):
        """Verifies all 3 races have dual specializations with signature actions and passives."""
        catalog = get_race_and_specializations_catalog()
        self.assertEqual(len(catalog), 3)
        self.assertIn("crystal", catalog)
        self.assertIn("toxic", catalog)
        self.assertIn("druid", catalog)

        for race_key, race in catalog.items():
            self.assertIn("race_name", race)
            self.assertIn("racial_passive", race)
            self.assertIn("hero_class", race)
            hero = race["hero_class"]
            self.assertEqual(len(hero["specializations"]), 2, f"{race_key} must have exactly 2 specializations.")
            for spec in hero["specializations"]:
                self.assertIn("spec_id", spec)
                self.assertIn("name", spec)
                self.assertIn("role", spec)
                self.assertIn("passive", spec)
                self.assertIn("signature_action", spec)
                self.assertIn("weapon_affinity", spec)

    def test_12_map_legend_topology_and_contours(self):
        """Verifies map legend defines 19 hexes, 3 elevation levels, and tactical overlays."""
        legend = get_map_legend_data()
        self.assertEqual(legend["total_hexes"], 19)
        self.assertEqual(len(legend["elevation_contours"]), 3)
        self.assertGreaterEqual(len(legend["sectors"]), 7)
        self.assertGreaterEqual(len(legend["tactical_overlays"]), 3)

        sector_ids = [s["id"] for s in legend["sectors"]]
        self.assertIn("sector_nexus", sector_ids)
        self.assertIn("sector_player_base", sector_ids)
        self.assertIn("sector_enemy_base", sector_ids)

    def test_13_api_game_races_endpoint(self):
        """Tests GET /api/game/races returns full race and specialization data."""
        url = "http://127.0.0.1:8089/api/game/races"
        req = urllib.request.urlopen(url)
        self.assertEqual(req.status, 200)
        data = json.loads(req.read().decode("utf-8"))
        self.assertIn("crystal", data)
        self.assertIn("toxic", data)
        self.assertIn("druid", data)

    def test_14_api_game_legend_map_endpoint(self):
        """Tests GET /api/game/legend-map returns map topology, contours and sectors."""
        url = "http://127.0.0.1:8089/api/game/legend-map"
        req = urllib.request.urlopen(url)
        self.assertEqual(req.status, 200)
        data = json.loads(req.read().decode("utf-8"))
        self.assertIn("elevation_contours", data)
        self.assertIn("sectors", data)
        self.assertIn("tactical_overlays", data)

    def test_15_bidirectional_proxy_and_hub_routes(self):
        """Tests that both 8080 and 8089 serve /game, /manual, / and proxy API endpoints with zero 404s."""
        # Test Hub (8080)
        for path in ["/", "/game", "/manual", "/static/godot_builder_extension.html"]:
            res = urllib.request.urlopen(f"http://127.0.0.1:8080{path}", timeout=3)
            self.assertEqual(res.status, 200)
            self.assertEqual(res.headers.get_content_type(), "text/html")

        # Test Hub proxying game engine APIs to 8089
        for api_path in ["/api/game/races", "/api/game/legend-map", "/api/cards", "/api/sectors"]:
            res = urllib.request.urlopen(f"http://127.0.0.1:8080{api_path}", timeout=3)
            self.assertEqual(res.status, 200)
            self.assertEqual(res.headers.get_content_type(), "application/json")

        # Test Engine Core (8089) serving UI and static files
        for path in ["/", "/game", "/manual", "/static/godot_builder_extension.html"]:
            res = urllib.request.urlopen(f"http://127.0.0.1:8089{path}", timeout=3)
            self.assertEqual(res.status, 200)
            self.assertEqual(res.headers.get_content_type(), "text/html")

        # Test Engine Core proxying hub APIs to 8080
        res = urllib.request.urlopen("http://127.0.0.1:8089/api/health", timeout=3)
        self.assertEqual(res.status, 200)
        self.assertEqual(res.headers.get_content_type(), "application/json")


if __name__ == "__main__":
    unittest.main()
