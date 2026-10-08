"""
tests/test_game_language_api_fetcher.py
=======================================
Unit Tests for Game Language API Fetcher & Tactical Game Controller.

Validates:
1. Strict enforcement of VITAL_MAX_HP == 6 across all intents, results, and configs.
2. Multilingual parsing accuracy for Slovak and English commands.
3. Card, movement, camera, weather, artillery, citadel, and entity spawn dispatch.
4. Natural language state queries and battlefield briefings.
5. Audit history tracking and limit constraints.
6. Remote fetcher fallback and daemon lifecycle management.
"""

import unittest
from krystal_web_hub.economic_engine.game_language_api_fetcher import (
    GameLanguageApiFetcher,
    GameCommandType,
    LanguageCommandIntent,
    LanguageExecutionResult,
    LanguageFetcherConfig,
    GLOBAL_GAME_LANGUAGE_FETCHER,
    VITAL_MAX_HP
)


class TestGameLanguageApiFetcher(unittest.TestCase):

    def setUp(self):
        self.fetcher = GameLanguageApiFetcher(base_api_url="http://127.0.0.1:8089")
        self.fetcher.config.mock_mode = True

    def test_vital_max_hp_rule_is_strictly_six(self):
        """Ensures the 6 Max HP vital invariant holds across all components."""
        self.assertEqual(VITAL_MAX_HP, 6)

        cfg = LanguageFetcherConfig()
        self.assertEqual(cfg.vital_max_hp_rule, 6)

        intent = self.fetcher.parse_language_command("Zahraj kartu Kryštálový Štít na seba")
        self.assertEqual(intent.vital_max_hp_rule, 6)

        result = self.fetcher.execute_command("Presuň hrdinu na hex [1, 0]", execute_api=False)
        self.assertEqual(result.vital_max_hp_rule, 6)

        query = self.fetcher.query_game_state("Aký je stav hrdinu?")
        self.assertEqual(query["vital_max_hp_rule"], 6)
        self.assertLessEqual(query["hp"], 6)

    def test_slovak_command_parsing(self):
        """Validates semantic intent parsing for Slovak prompts."""
        # 1. Card Cast
        intent1 = self.fetcher.parse_language_command("Zahraj kartu Kryštálový Meteor na hex [1, -2]")
        self.assertEqual(intent1.command_type, GameCommandType.CAST_CARD)
        self.assertEqual(intent1.parameters.get("card_id"), "crystal_meteor")
        self.assertEqual(intent1.parameters.get("target_hex"), [1, -2])
        self.assertEqual(intent1.language, "sk")

        # 2. Tactical Move
        intent2 = self.fetcher.parse_language_command("Presuň hrdinu na hex [2, 0]")
        self.assertEqual(intent2.command_type, GameCommandType.TACTICAL_MOVE)
        self.assertEqual(intent2.parameters.get("destination_hex"), [2, 0])

        # 3. Weather & Midnight
        intent3 = self.fetcher.parse_language_command("Nastav počasie na búrku a čas na polnoc")
        self.assertEqual(intent3.command_type, GameCommandType.WEATHER_ATMOSPHERE)
        self.assertEqual(intent3.parameters.get("weather_state"), "Thunderstorm")
        self.assertEqual(intent3.parameters.get("time_of_day_hours"), 0.0)

        # 4. Camera Switch
        intent4 = self.fetcher.parse_language_command("Prepnúť kameru na izometrický pohľad")
        self.assertEqual(intent4.command_type, GameCommandType.CAMERA_CONTROL)
        self.assertEqual(intent4.parameters.get("camera_preset"), "orthographic_true_isometric")

        # 5. Artillery
        intent5 = self.fetcher.parse_language_command("Vystreľ z moždiara na sektor 3")
        self.assertEqual(intent5.command_type, GameCommandType.ARTILLERY_FIRE)
        self.assertEqual(intent5.parameters.get("weapon_type"), "mortar_indirect")

        # 6. Citadel
        intent6 = self.fetcher.parse_language_command("Postav slovanský mažiar na vežu 2")
        self.assertEqual(intent6.command_type, GameCommandType.CITADEL_BUILD)
        self.assertEqual(intent6.parameters.get("pylon_type"), "slavic_mortar")
        self.assertEqual(intent6.parameters.get("tower_slot"), 2)

        # 7. Spawn Entity
        intent7 = self.fetcher.parse_language_command("Vyvolaj CesiumMan na pozíciu [0, 1]")
        self.assertEqual(intent7.command_type, GameCommandType.SPAWN_ENTITY)
        self.assertEqual(intent7.parameters.get("model_id"), "CesiumMan")

        # 8. Match Escalation
        intent8 = self.fetcher.parse_language_command("Ukonči kolo a spusti ťah nepriateľa")
        self.assertEqual(intent8.command_type, GameCommandType.MATCH_ESCALATION)

    def test_english_command_parsing(self):
        """Validates semantic intent parsing for English prompts."""
        # 1. Cast Card
        intent1 = self.fetcher.parse_language_command("Cast Glacial Lance at hex [1, 2]")
        self.assertEqual(intent1.command_type, GameCommandType.CAST_CARD)
        self.assertEqual(intent1.parameters.get("card_id"), "glacial_lance")
        self.assertEqual(intent1.parameters.get("target_hex"), [1, 2])
        self.assertEqual(intent1.language, "en")

        # 2. Movement
        intent2 = self.fetcher.parse_language_command("Move hero to hex [0, -1]")
        self.assertEqual(intent2.command_type, GameCommandType.TACTICAL_MOVE)
        self.assertEqual(intent2.parameters.get("destination_hex"), [0, -1])

        # 3. Weather
        intent3 = self.fetcher.parse_language_command("Set weather to rain with dusk lighting")
        self.assertEqual(intent3.command_type, GameCommandType.WEATHER_ATMOSPHERE)
        self.assertEqual(intent3.parameters.get("weather_state"), "Rain")
        self.assertEqual(intent3.parameters.get("time_of_day_hours"), 18.5)

        # 4. Camera
        intent4 = self.fetcher.parse_language_command("Switch camera to cinematic wide")
        self.assertEqual(intent4.command_type, GameCommandType.CAMERA_CONTROL)
        self.assertEqual(intent4.parameters.get("camera_preset"), "perspective_cinematic_wide")

        # 5. Query
        intent5 = self.fetcher.parse_language_command("What is my current HP and mana?")
        self.assertEqual(intent5.command_type, GameCommandType.QUERY_STATUS)

    def test_execution_and_state_diffs(self):
        """Verifies state updates, execution feedback, and audit history."""
        # Execute Card Cast
        res = self.fetcher.execute_command("Zahraj kartu Kryštálový Štít na seba", execute_api=False)
        self.assertEqual(res.status, "SUCCESS")
        self.assertEqual(res.api_action_executed, "POST /api/cards/cast")
        self.assertIn("Zahraná karta", res.game_feedback)
        self.assertIn("mana_delta", res.state_diff)

        # Execute Tactical Move
        res2 = self.fetcher.execute_command("Presuň hrdinu na hex [2, -1]", execute_api=False)
        self.assertEqual(res2.status, "SUCCESS")
        self.assertEqual(res2.state_diff["new_pos"], [2, -1])

        # Check Audit History
        history = self.fetcher.get_history(limit=10)
        self.assertGreaterEqual(len(history), 2)
        self.assertEqual(history[0]["execution_id"], res2.execution_id)

    def test_query_game_state_briefing(self):
        """Verifies natural language state reporting."""
        sk_report = self.fetcher.query_game_state("Aký je stav zápasu?")
        self.assertIn("Stav Bojiska Poslední Kmen", sk_report["answer"])
        self.assertEqual(sk_report["max_hp"], 6)

        en_report = self.fetcher.query_game_state("What is the battlefield status?")
        self.assertIn("Poslední Kmen Battlefield Status", en_report["answer"])
        self.assertEqual(en_report["max_hp"], 6)

    def test_remote_fetch_and_daemon_lifecycle(self):
        """Verifies fallback remote execution and daemon start/stop."""
        # 1. Fetch remote with fallback simulation
        results = self.fetcher.fetch_and_execute_remote()
        self.assertGreaterEqual(len(results), 1)
        self.assertEqual(results[0].status, "SUCCESS")

        # 2. Start & Stop Daemon
        self.assertFalse(self.fetcher._daemon_running)
        self.fetcher.start_fetcher_daemon(poll_interval_s=1.0)
        self.assertTrue(self.fetcher._daemon_running)
        self.assertTrue(self.fetcher.config.enabled)

        # Stop
        self.fetcher.stop_fetcher_daemon()
        self.assertFalse(self.fetcher._daemon_running)
        self.assertFalse(self.fetcher.config.enabled)

    def test_capabilities_catalog(self):
        """Verifies vocabulary and capabilities catalog."""
        catalog = self.fetcher.get_capabilities_catalog()
        self.assertEqual(catalog["vital_max_hp_rule"], 6)
        self.assertIn("CAST_CARD", catalog["supported_commands"])
        self.assertIn("WEATHER_ATMOSPHERE", catalog["supported_commands"])
        self.assertGreaterEqual(catalog["known_cards_count"], 10)
        self.assertIn("sk", catalog["sample_prompts"])
        self.assertIn("en", catalog["sample_prompts"])

    def test_global_singleton_readiness(self):
        """Ensures module singleton is instantiated and functional."""
        self.assertIsNotNone(GLOBAL_GAME_LANGUAGE_FETCHER)
        cat = GLOBAL_GAME_LANGUAGE_FETCHER.get_capabilities_catalog()
        self.assertEqual(cat["vital_max_hp_rule"], 6)


if __name__ == '__main__':
    unittest.main()
