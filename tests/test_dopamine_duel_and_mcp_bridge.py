# ==============================================================================
# TESTS FOR DOPAMINE MATRIX COMBAT, GNOME DUEL WINDOWS, MORTAR DISPERSION,
# UBISOFT BULLET TIME & WORDPRESS MCP BRIDGE
# ==============================================================================

import unittest
import time
from typing import Dict, List, Any

from krystal_web_hub.economic_engine.dopamine_matrix_combat import (
    DopamineCombatState,
    DopamineCadenceEngine
)
from krystal_web_hub.economic_engine.mortar_dispersion_engine import (
    UncoveredTargetCalculator,
    MortarDispersionEngine
)
from krystal_web_hub.economic_engine.gnome_duel_window import (
    AspectRatio,
    DuelWindowMode,
    GnomeDuelWindowCompositor
)
from krystal_web_hub.economic_engine.ubisoft_bullet_time_compositor import (
    ActionType,
    ParallelActionTrack,
    UbisoftBulletTimeCompositor
)
from krystal_web_hub.economic_engine.community_builds_and_rewards import (
    EventThreatLevel,
    CommunityBuildRegistry,
    EventRewardDerivationEngine
)
from krystal_web_hub.economic_engine.wordpress_mcp_bridge import WordPressMcpBridge


class TestDopamineMatrixCombat(unittest.TestCase):
    """Verifies set-matrix operations and dopamine micro-timing cadence."""

    def test_dopamine_combat_state_partitions(self):
        state = DopamineCombatState(["hero_1", "hero_2", "enemy_1", "enemy_2", "enemy_3"])
        state.update_partitions(
            allies=["hero_1", "hero_2"],
            enemies=["enemy_1", "enemy_2", "enemy_3"],
            uncovered=["enemy_1", "hero_2"],
            crowd_controlled=["enemy_2"],
            warded=["enemy_3", "hero_1"],
            supernatural=["hero_1"]
        )

        self.assertIn("enemy_1", state.critical_vulnerable_units) # Enemy ∩ Uncovered
        self.assertIn("enemy_2", state.critical_vulnerable_units) # Enemy ∩ CC
        self.assertNotIn("enemy_3", state.critical_vulnerable_units)

        # Set difference: Uncovered enemies not CC'd
        diff = state.get_set_difference(state.enemy_units & state.uncovered_units, state.crowd_controlled_units)
        self.assertEqual(diff, {"enemy_1"})

    def test_dopamine_cadence_engine_overdrive(self):
        # 3 cards played rapidly within 0.50s intervals (<= 0.85s window)
        timestamps = [100.0, 100.45, 100.90]
        card_ids = ["CARD_STRIKE_1", "CARD_PARRY_2", "CARD_FINISHER_3"]

        res = DopamineCadenceEngine.evaluate_timing_chain(timestamps, card_ids)
        self.assertTrue(res["dopamine_overdrive"])
        self.assertEqual(res["cadence_rating"], "PERFECT_CADENCE_OVERDRIVE")
        self.assertGreaterEqual(res["mana_refund"], 2)
        self.assertGreater(res["damage_cascade_multiplier"], 1.0)
        self.assertIn("DOPAMINE_SURGE_CASCADE", res["sensory_feedback"])

    def test_matrix_shatter_combos(self):
        state = DopamineCombatState(["p1", "p2", "e1", "e2"])
        state.update_partitions(
            allies=["p1", "p2"],
            enemies=["e1", "e2"],
            uncovered=["e1"],
            crowd_controlled=[],
            warded=["e2"],
            supernatural=["p1"]
        )

        cadence_info = {"dopamine_overdrive": True, "damage_cascade_multiplier": 1.70}

        # 1. Aetheric cadence shatter
        shatter_res = DopamineCadenceEngine.execute_matrix_shatter_combo(
            state, "aetheric_cadence_shatter", cadence_info
        )
        self.assertTrue(shatter_res["success"])
        self.assertEqual(shatter_res["affected_set"], ["e1"])
        self.assertEqual(shatter_res["allied_ward_granted"], 3)

        # 2. Corrosive rupture cascade
        rupture_res = DopamineCadenceEngine.execute_matrix_shatter_combo(
            state, "corrosive_rupture_cascade", cadence_info
        )
        self.assertTrue(rupture_res["success"])
        self.assertEqual(rupture_res["affected_set"], ["e2"])
        self.assertTrue(rupture_res["ward_stripped"])

        # 3. Verdant avalanche cataclysm
        avalanche_res = DopamineCadenceEngine.execute_matrix_shatter_combo(
            state, "verdant_avalanche_cataclysm", cadence_info
        )
        self.assertTrue(avalanche_res["success"])
        self.assertEqual(set(avalanche_res["immobilized_set"]), {"e1", "e2"})


class TestMortarDispersionEngine(unittest.TestCase):
    """Verifies Line of Sight cover calculations, mortar CEP dispersion, and burst cadence."""

    def test_uncovered_targets_with_plunging_elevation(self):
        combatants = [
            {"unit_id": "unit_a", "name": "Scout", "current_hex": [3, 4]},
            {"unit_id": "unit_b", "name": "Bunker Guard", "current_hex": [5, 5]},
            {"unit_id": "unit_c", "name": "Low Wall Defender", "current_hex": [2, 1]}
        ]
        # Cover density: 0.0=none, 0.5=medium, 0.9=bunker
        cover_map = {
            "3_4": 0.10, # None
            "5_5": 0.90, # Full bunker
            "2_1": 0.45  # Low sandbags
        }
        # Elevation: low wall at 1.0m
        elevation_map = {
            "3_4": 0.0,
            "5_5": 3.0,
            "2_1": 1.0
        }

        results = UncoveredTargetCalculator.identify_uncovered_targets(
            combatants=combatants,
            cover_map=cover_map,
            elevation_map=elevation_map,
            attacker_hex=[0, 0],
            is_plunging_fire=True
        )

        eval_by_id = {u["unit_id"]: u for u in results}
        self.assertEqual(eval_by_id["unit_a"]["status"], "UNCOVERED")
        self.assertEqual(eval_by_id["unit_b"]["status"], "FULL_COVER")
        # Low wall is bypassed by plunging fire (elevation <= 1.2m and cover < 0.60)
        self.assertEqual(eval_by_id["unit_c"]["status"], "UNCOVERED")
        self.assertTrue(eval_by_id["unit_c"]["is_vulnerable_to_mortar"])

    def test_mortar_salvo_ballistics_and_cadence(self):
        salvo = MortarDispersionEngine.calculate_mortar_salvo(
            attacker_pos=(0.0, 0.0, 0.0),
            target_pos=(100.0, 0.0, 50.0),
            salvo_count=4,
            elevation_angle_deg=70.0,
            muzzle_velocity=50.0,
            cadence_interval_sec=0.35
        )

        self.assertEqual(salvo["salvo_count"], 4)
        self.assertGreater(salvo["flight_time_per_shell_sec"], 5.0)
        self.assertGreater(salvo["apex_height_meters"], 50.0)
        self.assertEqual(len(salvo["projectiles"]), 4)
        self.assertEqual(salvo["dispersion_footprint"]["dispersion_shape"], "elliptical_plunging")
        self.assertGreater(salvo["cadence_rounds_per_minute"], 150.0)


class TestGnomeDuelWindow(unittest.TestCase):
    """Verifies GNOME desktop windowing, aspect-ratio scaling, and close-up duel rendering."""

    def test_aspect_ratio_scaling_and_viewports(self):
        char_a = {"unit_id": "blade_master", "name": "Aetheric Duelist", "current_wounds": 5, "ward": 2, "weapon_type": "crystal_blade"}
        char_b = {"unit_id": "flail_zealot", "name": "Toxic Zealot", "current_wounds": 4, "ward": 0, "weapon_type": "toxic_censer_flail"}

        for ratio, expected_w in [
            (AspectRatio.CINEMATIC_16_9, 1280),
            (AspectRatio.ULTRAWIDE_21_9, 1680),
            (AspectRatio.CLASSIC_4_3, 960),
            (AspectRatio.SPLIT_DUEL_1_1, 720)
        ]:
            viewport = GnomeDuelWindowCompositor.compose_duel_viewport(
                character_left=char_a,
                character_right=char_b,
                aspect_ratio=ratio,
                window_mode=DuelWindowMode.SPLIT_OPPONENT,
                incoming_projectiles=[{"shell_index": 1}]
            )
            self.assertEqual(viewport["resolution"][0], expected_w)
            self.assertEqual(viewport["resolution"][1], 720)
            self.assertEqual(viewport["viewports"]["left_combatant"]["current_hp"], 5)
            self.assertEqual(len(viewport["active_projectiles"]), 1)
            self.assertIn("Adwaita", viewport["gnome_csd_header"]["theme"])


class TestUbisoftBulletTime(unittest.TestCase):
    """Verifies space-time collision intersection and Ubisoft bullet time choreography."""

    def test_trajectory_intersection_and_bullet_time_trigger(self):
        # Melee charge moving from (0,0,0) to (3.0,0,0) in 1.0s
        track_melee = ParallelActionTrack(
            action_id="melee_lunge",
            actor_id="duelist",
            action_type=ActionType.COLD_MELEE_STRIKE,
            start_pos=(0.0, 0.0, 0.0),
            end_pos=(3.0, 0.0, 0.0),
            start_time=0.0,
            duration=1.0,
            bounding_radius=0.65
        )

        # Mortar shell plunging down toward (2.5, 0.0, 0.0)
        track_mortar = ParallelActionTrack(
            action_id="mortar_plunge",
            actor_id="battery",
            action_type=ActionType.MORTAR_PROJECTILE,
            start_pos=(2.5, 4.5, 0.0),
            end_pos=(2.5, 0.0, 0.0),
            start_time=0.0,
            duration=1.0,
            bounding_radius=0.65
        )

        bt_result = UbisoftBulletTimeCompositor.compose_bullet_time_sequence(
            track_melee=track_melee,
            track_projectile=track_mortar,
            aspect_ratio="16:9"
        )

        self.assertTrue(bt_result["bullet_time_triggered"])
        self.assertAlmostEqual(bt_result["visual_shaders"]["time_dilation_factor"], 0.10, places=2)
        self.assertGreater(bt_result["visual_shaders"]["hit_stop_frames"], 0)
        self.assertIn("rule_of_thirds_anchor", bt_result["camera_specs"]["composition_grid"])
        self.assertEqual(len(bt_result["cinematic_keyframes"]), 4)


class TestCommunityBuildsAndRewards(unittest.TestCase):
    """Verifies community builds catalog and threat-level reward derivation math."""

    def test_community_build_catalog(self):
        builds = CommunityBuildRegistry.list_builds()
        self.assertGreaterEqual(len(builds), 4)
        mortar_build = CommunityBuildRegistry.get_build("mortar_siegebreaker")
        self.assertIsNotNone(mortar_build)
        self.assertIn("Twin-Barrel Heavy Mortar", mortar_build["primary_weapon"])
        self.assertGreater(mortar_build["stat_biases"]["blast_radius_mult"], 1.0)

    def test_threat_level_reward_derivation(self):
        # Threat 5 with low remaining HP (high bravery risk factor) and high cadence
        reward = EventRewardDerivationEngine.derive_event_rewards(
            threat_level=EventThreatLevel.THREAT_5_CATACLYSMIC_INCURSION,
            remaining_hp=1, # 1 out of 6 Max HP vital invariant = high risk bonus (+40%)
            dopamine_cadence_score=2.0,
            bullet_time_count=2,
            uncovered_targets_eliminated=3,
            player_account_id="champion_01"
        )

        self.assertEqual(reward["threat_level"], 5)
        self.assertEqual(reward["remaining_hp"], 1)
        self.assertEqual(reward["max_hp_invariant"], 6)
        self.assertEqual(reward["risk_bonus_percent"], 40)
        self.assertGreater(reward["rewards"]["gold"], 10000)
        self.assertGreater(reward["rewards"]["krystal_gems"], 1000)
        self.assertIn("BLUEPRINT_OBSIDIAN_BLADE_OF_OVERDRIVE", reward["rewards"]["blueprints"])

        # Check ledger audit spec
        ledger = reward["ledger_audit"]
        self.assertEqual(ledger["credit_account"], "champion_01")
        self.assertEqual(ledger["debit_account"], "treasury_event_rewards")


class TestWordPressMcpBridge(unittest.TestCase):
    """Verifies end-to-end MCP Bridge tool execution and opponent frame streaming."""

    def setUp(self):
        self.bridge = WordPressMcpBridge()

    def test_mcp_uncovered_and_mortar_tool(self):
        combatants = [
            {"unit_id": "enemy_sniper", "name": "Sniper", "current_hex": [4, 4]},
            {"unit_id": "bunker_heavy", "name": "Heavy", "current_hex": [8, 8]}
        ]
        cover_map = {"4_4": 0.15, "8_8": 0.95}
        elevation_map = {"4_4": 0.5, "8_8": 2.5}

        res = self.bridge.calculate_uncovered_and_mortar_cadence(
            combatants=combatants,
            cover_map=cover_map,
            elevation_map=elevation_map,
            attacker_pos=(0.0, 0.0, 0.0),
            target_pos=(40.0, 0.0, 40.0),
            salvo_count=3,
            elevation_angle_deg=65.0,
            cadence_interval_sec=0.40
        )

        self.assertTrue(res["success"])
        self.assertIn("enemy_sniper", res["uncovered_target_ids"])
        self.assertNotIn("bunker_heavy", res["uncovered_target_ids"])
        self.assertEqual(res["mortar_salvo"]["salvo_count"], 3)

    def test_mcp_stream_frame_to_opponent(self):
        char_left = {"unit_id": "hero_a", "name": "Aether Blade", "current_wounds": 6, "ward": 1, "weapon_type": "crystal_blade"}
        char_right = {"unit_id": "hero_b", "name": "Shadow Dagger", "current_wounds": 5, "ward": 0, "weapon_type": "dual_daggers"}

        frame = self.bridge.compose_gnome_duel_frame(
            character_left=char_left,
            character_right=char_right,
            aspect_ratio="16:9",
            window_mode="cinematic_zoom",
            incoming_projectiles=[{"shell_index": 1}]
        )
        self.assertIn("ascii_canvas_preview", frame)

        # Stream frame to opponent
        stream_res = self.bridge.stream_frame_to_opponent(
            duel_id="duel_room_77",
            sender_id="hero_a",
            recipient_id="hero_b",
            frame_payload=frame
        )
        self.assertTrue(stream_res["success"])
        self.assertEqual(stream_res["status"], "STREAMED_TO_OPPONENT_VIEWPORT")

        # Opponent polls frames
        polled = self.bridge.poll_opponent_frames(
            duel_id="duel_room_77",
            recipient_id="hero_b",
            since_timestamp=0.0
        )
        self.assertEqual(len(polled), 1)
        self.assertEqual(polled[0]["frame"]["aspect_ratio"], "16:9")

        # WordPress REST payload wrapper
        wp_payload = self.bridge.generate_wordpress_rest_payload("duel_room_77", frame)
        self.assertEqual(wp_payload["wp_channel"], "krystal_duel_duel_room_77")
        self.assertIn("--gnome-aspect-ratio", wp_payload["css_theme_vars"])


if __name__ == "__main__":
    unittest.main()
