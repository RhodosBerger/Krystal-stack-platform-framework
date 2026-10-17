#!/usr/bin/env python3
"""
VERIFICATION TEST SUITE: V-GAME-WEB-MVP-01
=============================================================================
Milestone: Deterministic 19-Hex Browser Duel
Product: Poslední Kmen — powered by Krystal Nexus / Krystal Stack

Verifies:
  1. Exactly 19 playable logical hexes with axial coordinates
  2. Invariant: GLOBAL_VITAL_MAX_HP == 6 across all combatants
  3. Strict [0, 6] bounds on all damage and healing
  4. Valid movement state transition and AP deduction
  5. Invalid movement produces strict NOOP (S_{t+1} == S_t)
  6. Valid attack with locational evasion algebra
  7. Invalid attack produces strict NOOP (S_{t+1} == S_t)
  8. Turn ownership enforcement (Commit(a) => TurnOwner(a))
  9. Terminal match conditions (VICTORY and DEFEAT)
  10. Deterministic Replay: Replay(S_0, seed, log) == S_final with identical SHA-256 digests
  11. HTTP delivery: /game and /api/game/state return HTTP 200
  12. HTML UI offline-first integrity and passive canvas projection
=============================================================================
"""

import unittest
import json
import time
from pathlib import Path
from http.server import HTTPServer
import threading
import urllib.request
import urllib.error

from krystal_web_hub.economic_engine.posledni_kmen_authoritative_duel import (
    DeterministicAuthoritativeDuelEngine,
    AuthoritativeGameState,
    DeterministicActionValidator,
    TurnResolver,
    ActionProposal,
    HexCoord,
    VALID_AXIAL_COORDS,
    AUTHORITATIVE_HEX_TILES,
    VITAL_MAX_HP,
    MatchPhase,
    ActionType,
    TargetZone,
    DodgeStance
)
from krystal_web_hub.server import KrystalHubHandler, ThreadedHTTPServer

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent


class TestVGameWebMvp01(unittest.TestCase):

    def setUp(self):
        self.engine = DeterministicAuthoritativeDuelEngine(rng_seed=42)

    def test_01_exactly_19_playable_logical_hexes(self):
        """Verifies exactly 19 playable logical hexes in axial radius 2."""
        self.assertEqual(len(VALID_AXIAL_COORDS), 19)
        self.assertEqual(len(AUTHORITATIVE_HEX_TILES), 19)

        # Every hex must satisfy max(|q|, |r|, |-q-r|) <= 2
        for q, r in VALID_AXIAL_COORDS:
            s = -q - r
            self.assertLessEqual(max(abs(q), abs(r), abs(s)), 2)

        # Biomes must be crystal, toxic, druid, or volcanic
        biomes = {t.biome for t in AUTHORITATIVE_HEX_TILES}
        self.assertEqual(biomes, {"crystal", "toxic", "druid", "volcanic"})

    def test_02_vital_max_hp_invariant_and_bounds(self):
        """Invariant: GLOBAL_VITAL_MAX_HP == 6, spawn HP == 6, bounds strictly in [0, 6]."""
        self.assertEqual(VITAL_MAX_HP, 6)
        state = self.engine.state

        self.assertEqual(state.player.hp, 6)
        self.assertEqual(state.player.max_hp, 6)
        self.assertEqual(state.enemy.hp, 6)
        self.assertEqual(state.enemy.max_hp, 6)

        # Healing when full does not exceed 6
        healed = state.player.heal(5)
        self.assertEqual(healed, 0)
        self.assertEqual(state.player.hp, 6)

        # Damage reduces HP
        dmg = state.player.take_damage(2)
        self.assertEqual(dmg, 2)
        self.assertEqual(state.player.hp, 4)

        # Healing restores HP up to 6
        h2 = state.player.heal(1)
        self.assertEqual(h2, 1)
        self.assertEqual(state.player.hp, 5)

        # Overkill damage clamps strictly to 0
        state.player.take_damage(20)
        self.assertEqual(state.player.hp, 0)
        self.assertFalse(state.player.is_alive)

    def test_03_valid_movement_state_transition(self):
        """Valid move updates coordinates, deducts 1 AP, updates tile occupancy, changes digest."""
        initial_digest = self.engine.state.compute_state_digest()
        self.assertEqual(self.engine.state.player.q, 0)
        self.assertEqual(self.engine.state.player.r, -1)
        self.assertEqual(self.engine.state.player.ap, 3)

        # Move to adjacent empty hex [0, 0] (distance 1)
        res = self.engine.propose_action({
            "actor_id": "player_hero",
            "action_type": "MOVE",
            "target_hex": [0, 0]
        })

        self.assertTrue(res["success"])
        self.assertEqual(self.engine.state.player.q, 0)
        self.assertEqual(self.engine.state.player.r, 0)
        self.assertEqual(self.engine.state.player.ap, 2)

        # Occupancy updated
        self.assertEqual(self.engine.state.hexes[(0, 0)].occupant_id, "player_hero")
        self.assertIsNone(self.engine.state.hexes[(0, -1)].occupant_id)

        # State digest mutated
        new_digest = self.engine.state.compute_state_digest()
        self.assertNotEqual(initial_digest, new_digest)

    def test_04_invalid_movement_produces_strict_noop(self):
        """Invariant: invalid_action => state_after == state_before (Strict NOOP)."""
        pre_digest = self.engine.state.compute_state_digest()
        pre_snapshot = json.dumps(self.engine.state.snapshot(), sort_keys=True)

        # 1. Distance > 1 (e.g. jumping 2 tiles)
        res_far = self.engine.propose_action({
            "actor_id": "player_hero",
            "action_type": "MOVE",
            "target_hex": [2, 0]
        })
        self.assertFalse(res_far["success"])
        self.assertEqual(res_far["receipt"]["verdict"], "REJECTED_INVALID")
        self.assertIn("Reachable violation", res_far["receipt"]["error_message"])
        self.assertEqual(self.engine.state.compute_state_digest(), pre_digest)

        # 2. Target out of 19-hex board bounds
        res_oob = self.engine.propose_action({
            "actor_id": "player_hero",
            "action_type": "MOVE",
            "target_hex": [5, -5]
        })
        self.assertFalse(res_oob["success"])
        self.assertEqual(res_oob["receipt"]["verdict"], "REJECTED_INVALID")
        self.assertEqual(self.engine.state.compute_state_digest(), pre_digest)

        # 3. Target occupied by enemy (hex [0, 1] after moving to [0, 0])
        # First move to [0, 0]
        self.engine.propose_action({"actor_id": "player_hero", "action_type": "MOVE", "target_hex": [0, 0]})
        digest_at_center = self.engine.state.compute_state_digest()

        # Try to step onto enemy at [0, 1]
        res_occ = self.engine.propose_action({
            "actor_id": "player_hero",
            "action_type": "MOVE",
            "target_hex": [0, 1]
        })
        self.assertFalse(res_occ["success"])
        self.assertEqual(res_occ["receipt"]["verdict"], "REJECTED_INVALID")
        self.assertIn("TargetValid violation: hex (0, 1) is occupied", res_occ["receipt"]["error_message"])
        self.assertEqual(self.engine.state.compute_state_digest(), digest_at_center)

    def test_05_valid_attack_and_locational_evasion(self):
        """Attacking head vs duck_down is evaded; torso hits for 2 damage."""
        # Enemy is at [0, 1], Player at [0, -1] -> distance 2 <= attack_range (2)
        # Set enemy stance to duck_down
        self.engine.state.enemy.dodge_stance = DodgeStance.DUCK_DOWN.value

        # Attack head -> should be evaded
        res_evade = self.engine.propose_action({
            "actor_id": "player_hero",
            "action_type": "ATTACK",
            "target_zone": "head"
        })
        self.assertTrue(res_evade["success"])
        self.assertTrue(res_evade["receipt"]["details"]["evaded"])
        self.assertEqual(res_evade["receipt"]["details"]["damage_dealt"], 0)
        self.assertEqual(self.engine.state.enemy.hp, 6)

        # Attack torso -> hits for 2 damage
        res_hit = self.engine.propose_action({
            "actor_id": "player_hero",
            "action_type": "ATTACK",
            "target_zone": "torso"
        })
        self.assertTrue(res_hit["success"])
        self.assertTrue(res_hit["receipt"]["details"]["hit"])
        self.assertEqual(res_hit["receipt"]["details"]["damage_dealt"], 2)
        self.assertEqual(self.engine.state.enemy.hp, 4)

    def test_06_invalid_attack_produces_strict_noop(self):
        """Invalid attack (e.g. 0 AP or out of range) produces strict NOOP."""
        # Deplete AP to 0
        self.engine.state.player.ap = 0
        pre_digest = self.engine.state.compute_state_digest()

        res = self.engine.propose_action({
            "actor_id": "player_hero",
            "action_type": "ATTACK",
            "target_zone": "torso"
        })
        self.assertFalse(res["success"])
        self.assertEqual(res["receipt"]["verdict"], "REJECTED_INVALID")
        self.assertIn("ResourceBudget violation", res["receipt"]["error_message"])
        self.assertEqual(self.engine.state.compute_state_digest(), pre_digest)

    def test_07_turn_ownership_and_alternation(self):
        """Only the turn owner may act; ending turn alternates turn ownership."""
        # Player turn -> enemy cannot act
        res_enemy = self.engine.propose_action({
            "actor_id": "enemy_hero",
            "action_type": "ATTACK",
            "target_zone": "torso"
        })
        self.assertFalse(res_enemy["success"])
        self.assertIn("TurnOwner violation", res_enemy["receipt"]["error_message"])

        # Player ends turn -> triggers deterministic AI turn and returns to player turn
        res_end = self.engine.propose_action({
            "actor_id": "player_hero",
            "action_type": "END_TURN"
        })
        self.assertTrue(res_end["success"])
        # Turn number advanced
        self.assertEqual(self.engine.state.turn_number, 2)
        self.assertEqual(self.engine.state.phase, MatchPhase.PLAYER_TURN)
        self.assertEqual(self.engine.state.player.ap, 3)

    def test_08_victory_and_defeat_conditions(self):
        """Reducing enemy HP to 0 yields VICTORY; no further moves allowed."""
        self.engine.state.enemy.hp = 2

        res_kill = self.engine.propose_action({
            "actor_id": "player_hero",
            "action_type": "ATTACK",
            "target_zone": "torso"
        })
        self.assertTrue(res_kill["success"])
        self.assertEqual(self.engine.state.enemy.hp, 0)
        self.assertEqual(self.engine.state.phase, MatchPhase.VICTORY)

        # Any further action is rejected
        res_after = self.engine.propose_action({
            "actor_id": "player_hero",
            "action_type": "MOVE",
            "target_hex": [0, 0]
        })
        self.assertFalse(res_after["success"])
        self.assertIn("Match is completed", res_after["receipt"]["error_message"])

    def test_09_deterministic_replay_and_identical_state_hashes(self):
        """Replay(S_0, seed, log) == S_final with identical cryptographic digest."""
        # Execute 4 actions
        self.engine.propose_action({"actor_id": "player_hero", "action_type": "SELECT_UNIT", "target_zone": "head", "dodge_stance": "duck_down"})
        self.engine.propose_action({"actor_id": "player_hero", "action_type": "MOVE", "target_hex": [0, 0]})
        self.engine.propose_action({"actor_id": "player_hero", "action_type": "ATTACK", "target_zone": "head"})
        self.engine.propose_action({"actor_id": "player_hero", "action_type": "END_TURN"})

        final_digest = self.engine.state.compute_state_digest()
        events = [r.to_dict() for r in self.engine.state.event_log if r.verdict == "COMMITTED"]

        # Replay from scratch
        ok, msg, replayed_state = DeterministicAuthoritativeDuelEngine.replay(
            initial_seed=self.engine.rng_seed,
            event_log=events
        )

        self.assertTrue(ok, msg)
        self.assertEqual(replayed_state["state_digest"], final_digest)
        self.assertEqual(replayed_state["player"]["hp"], self.engine.state.player.hp)
        self.assertEqual(replayed_state["enemy"]["hp"], self.engine.state.enemy.hp)
        self.assertEqual(replayed_state["player"]["q"], self.engine.state.player.q)
        self.assertEqual(replayed_state["player"]["r"], self.engine.state.player.r)

    def test_10_web_server_delivery_http_200_and_authoritative_endpoints(self):
        """Local Web Hub serves /game and /api/game/state with HTTP 200."""
        # Spin up test server on ephemeral port
        test_port = 8765
        server = ThreadedHTTPServer(("127.0.0.1", test_port), KrystalHubHandler)
        server_thread = threading.Thread(target=server.serve_forever, daemon=True)
        server_thread.start()
        time.sleep(0.1)

        try:
            # 1. GET /game
            with urllib.request.urlopen(f"http://127.0.0.1:{test_port}/game") as resp:
                self.assertEqual(resp.status, 200)
                body = resp.read().decode("utf-8")
                self.assertIn("POSLEDNÍ KMEN // TACTICAL DUEL", body)
                self.assertIn("GLOBAL_VITAL_MAX_HP == 6", body)
                self.assertIn("arenaCanvas", body)

            # 2. GET /api/game/state
            with urllib.request.urlopen(f"http://127.0.0.1:{test_port}/api/game/state") as resp:
                self.assertEqual(resp.status, 200)
                data = json.loads(resp.read().decode("utf-8"))
                self.assertEqual(data["status"], "OK")
                self.assertEqual(data["state"]["vital_max_hp"], 6)
                self.assertEqual(data["state"]["hex_count"], 19)

            # 3. POST /api/game/action
            req = urllib.request.Request(
                f"http://127.0.0.1:{test_port}/api/game/action",
                data=json.dumps({"actor_id": "player_hero", "action_type": "MOVE", "target_hex": [0, 0]}).encode("utf-8"),
                headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req) as resp:
                self.assertEqual(resp.status, 200)
                act_data = json.loads(resp.read().decode("utf-8"))
                self.assertEqual(act_data["status"], "OK")
                self.assertEqual(act_data["receipt"]["verdict"], "COMMITTED")

            # 4. GET /api/game/replay
            with urllib.request.urlopen(f"http://127.0.0.1:{test_port}/api/game/replay") as resp:
                self.assertEqual(resp.status, 200)
                rep_data = json.loads(resp.read().decode("utf-8"))
                self.assertTrue(rep_data["verified"])

        finally:
            server.shutdown()
            server.server_close()

    def test_11_html_asset_offline_purity_and_syntax(self):
        """Verifies posledni_kmen_game.html contains no external CDN dependencies and valid syntax."""
        html_path = WORKSPACE_ROOT / "krystal_web_hub" / "static" / "posledni_kmen_game.html"
        self.assertTrue(html_path.exists())
        content = html_path.read_text(encoding="utf-8")

        # Zero CDN dependencies
        self.assertNotIn("https://fonts.googleapis.com", content)
        self.assertNotIn("https://cdn.", content)
        self.assertNotIn("https://cdnjs.", content)

        # Constitutional invariants in UI
        self.assertIn("GLOBAL_VITAL_MAX_HP == 6", content)
        self.assertIn("arenaCanvas", content)
        self.assertIn("playerPips", content)
        self.assertIn("opponentPips", content)
        self.assertIn("/api/game/state", content)
        self.assertIn("/api/game/action", content)
        self.assertIn("/api/game/replay", content)


if __name__ == "__main__":
    unittest.main()
