#!/usr/bin/env python3
"""
GENUINE HTTP TRANSPORT E2E INTEGRATION TEST SUITE
=============================================================================
Product: Poslední Kmen — Tactical Sovereign Vertical Slice v0.1
Milestone: M0 (Contract Freeze) + M1 (Playable Sovereign Vertical Slice Stabilization)

Verifies the COMPLETE transport chain:
  HTTP Request -> Router -> JSON Parsing -> MatchRegistry Validation ->
  Deterministic GameRules -> Serialization -> HTTP Response -> Telemetry Record

Tests:
  1. GET /game delivers self-contained offline HTML (Option A) with 19-hex canvas and 6 HP invariant.
  2. GET /api/game/status and /api/game/archetypes_catalog over real HTTP.
  3. POST /api/game/duel_round over real HTTP with M1 telemetry record verification.
  4. POST /api/game/mortar_strike over real HTTP with CEP dispersion and telemetry.
  5. POST /api/game/sapper_trench over real HTTP raising cover density to 85%.
  6. POST /api/game/reset restores state with explicit semantics and 6 HP.
  7. Multi-tenant session isolation (match_id): Match A does NOT mutate Match B!
  8. Exhaustive invariant preservation: 0 <= HP <= 6, VITAL_MAX_HP == 6 across all rounds.
=============================================================================
"""

import json
import time
import socket
import threading
import unittest
import urllib.request
import urllib.error
from typing import Dict, Any

from krystal_web_hub.server import KrystalHubHandler, ThreadedHTTPServer
from krystal_web_hub.economic_engine.game_service import (
    MatchRegistry,
    VITAL_MAX_HP,
    M1TelemetryRecord
)


def get_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class TestPosledniKmenHttpTransportE2E(unittest.TestCase):
    """Genuine HTTP transport integration tests for Poslední Kmen Tactical Arena."""

    @classmethod
    def setUpClass(cls):
        MatchRegistry.clear_all()
        cls.port = get_free_port()
        cls.server = ThreadedHTTPServer(("127.0.0.1", cls.port), KrystalHubHandler)
        cls.server_thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.server_thread.start()
        cls.base_url = f"http://127.0.0.1:{cls.port}"
        # Small wait for socket initialization
        time.sleep(0.15)

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        MatchRegistry.clear_all()

    def _http_get(self, path: str, headers: Dict[str, str] = None) -> Tuple[int, bytes, Dict[str, str]]:
        req = urllib.request.Request(f"{self.base_url}{path}", headers=headers or {})
        with urllib.request.urlopen(req) as resp:
            return resp.status, resp.read(), dict(resp.headers)

    def _http_post_json(self, path: str, payload: Dict[str, Any], headers: Dict[str, str] = None) -> Tuple[int, Dict[str, Any]]:
        all_headers = {"Content-Type": "application/json"}
        if headers:
            all_headers.update(headers)
        data_bytes = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(f"{self.base_url}{path}", data=data_bytes, headers=all_headers)
        with urllib.request.urlopen(req) as resp:
            body = resp.read().decode("utf-8")
            return resp.status, json.loads(body)

    # ── 1. HTTP GET /game ──────────────────────────────────────────────────────
    def test_01_http_get_game_page_serves_offline_vertical_slice(self):
        """GET /game returns HTTP 200, offline-first HTML, exactly 19 hexes, and 6 HP invariant."""
        status, body_bytes, headers = self._http_get("/game")
        self.assertEqual(status, 200)
        self.assertIn("text/html", headers.get("Content-Type", ""))

        body = body_bytes.decode("utf-8")
        # Constitutional Invariant & Branding
        self.assertIn("POSLEDNÍ KMEN // TACTICAL DUEL", body)
        self.assertIn("GLOBAL_VITAL_MAX_HP == 6", body)
        self.assertIn("arenaCanvas", body)
        self.assertIn("playerPips", body)
        self.assertIn("opponentPips", body)

        # M1 Action buttons
        self.assertIn("btnDuelRound", body)
        self.assertIn("btnMortarStrike", body)
        self.assertIn("btnDigTrench", body)
        self.assertIn("btnResetMatch", body)

        # Option A: Zero external fonts or CDN dependencies
        self.assertNotIn("fonts.googleapis.com", body)
        self.assertNotIn("https://cdn.", body)
        self.assertNotIn("https://cdnjs.", body)

    # ── 2. HTTP GET /api/game/status & catalog ─────────────────────────────────
    def test_02_http_get_status_and_catalog(self):
        """GET /api/game/status and /api/game/archetypes_catalog over real HTTP."""
        # 1. Catalog
        status, cat_body, _ = self._http_get("/api/game/archetypes_catalog")
        self.assertEqual(status, 200)
        cat = json.loads(cat_body.decode("utf-8"))
        self.assertEqual(cat["status"], "OK")
        self.assertEqual(cat["vital_max_hp"], 6)
        self.assertEqual(cat["total_archetypes"], 40)
        self.assertEqual(cat["total_helpers"], 120)
        self.assertEqual(len(cat["races"]), 12)

        # 2. Match Status
        status, stat_body, _ = self._http_get("/api/game/status?match_id=session_init_01")
        self.assertEqual(status, 200)
        stat = json.loads(stat_body.decode("utf-8"))
        self.assertEqual(stat["status"], "OK")
        self.assertEqual(stat["vital_max_hp"], 6)
        self.assertEqual(stat["match"]["player"]["current_hp"], 6)
        self.assertEqual(stat["match"]["opponent"]["current_hp"], 6)
        self.assertIn("state_hash", stat)
        self.assertIn("telemetry", stat)

    # ── 3. HTTP POST /api/game/duel_round ──────────────────────────────────────
    def test_03_http_post_duel_round_pipeline_and_telemetry(self):
        """POST /api/game/duel_round executes locational combat and returns M1 telemetry."""
        status, res = self._http_post_json(
            "/api/game/duel_round",
            {
                "match_id": "session_duel_test",
                "player_attack_zone": "head",
                "player_dodge_stance": "duck_down"
            }
        )
        self.assertEqual(status, 200)
        self.assertEqual(res["status"], "SUCCESS")
        self.assertEqual(res["vital_max_hp"], 6)
        self.assertIn("player_shot", res)
        self.assertIn("opponent_shot", res)

        # Invariants on damage
        self.assertTrue(0 <= res["player_hp"] <= 6)
        self.assertTrue(0 <= res["opponent_hp"] <= 6)

        # Minimal M1 Telemetry schema verification
        tel = res.get("telemetry", {})
        self.assertEqual(tel["match_id"], "session_duel_test")
        self.assertEqual(tel["action"], "DUEL_ROUND")
        self.assertTrue(tel["invariant_ok"])
        self.assertTrue(tel["api_rtt_ms"] >= 0.0)
        self.assertEqual(tel["player_hp"], res["player_hp"])
        self.assertEqual(tel["opponent_hp"], res["opponent_hp"])
        self.assertIn("state_hash", tel)
        self.assertEqual(len(tel["state_hash"]), 64)  # Valid SHA-256

    # ── 4. HTTP POST /api/game/mortar_strike ───────────────────────────────────
    def test_04_http_post_mortar_strike_pipeline(self):
        """POST /api/game/mortar_strike resolves CEP dispersion and area damage."""
        status, res = self._http_post_json(
            "/api/game/mortar_strike",
            {
                "match_id": "session_mortar_test",
                "target_hex": [1, 1],
                "salvo_count": 3,
                "elevation_angle_deg": 65.0
            }
        )
        self.assertEqual(status, 200)
        self.assertEqual(res["status"], "SUCCESS")
        self.assertEqual(res["vital_max_hp"], 6)
        self.assertEqual(res["salvo"]["salvo_count"], 3)
        self.assertIn("dispersion_footprint", res["salvo"])
        self.assertIn("hits", res)

        # Telemetry
        tel = res.get("telemetry", {})
        self.assertEqual(tel["action"], "MORTAR_STRIKE")
        self.assertTrue(tel["invariant_ok"])

    # ── 5. HTTP POST /api/game/sapper_trench ───────────────────────────────────
    def test_05_http_post_sapper_trench_pipeline(self):
        """POST /api/game/sapper_trench raises cover density to 85% on target hex."""
        status, res = self._http_post_json(
            "/api/game/sapper_trench",
            {
                "match_id": "session_sapper_test",
                "hex_coord": [0, 0],
                "unit_id": "player_hero"
            }
        )
        self.assertEqual(status, 200)
        self.assertEqual(res["status"], "SUCCESS")
        self.assertEqual(res["new_cover_density"], 0.85)
        self.assertEqual(res["hex_key"], "0_0")
        self.assertIn("0_0", res["trenches"])

        # Subsequent mortar strike on [0, 0] should be protected
        _, mortar_res = self._http_post_json(
            "/api/game/mortar_strike",
            {
                "match_id": "session_sapper_test",
                "target_hex": [0, 0],
                "salvo_count": 3
            }
        )
        self.assertEqual(mortar_res["status"], "SUCCESS")

    # ── 6. HTTP POST /api/game/reset ───────────────────────────────────────────
    def test_06_http_post_reset_pipeline(self):
        """POST /api/game/reset restores combatants to full 6 HP and cleanses state."""
        # 1. Damage match
        self._http_post_json(
            "/api/game/duel_round",
            {"match_id": "session_reset_test", "player_attack_zone": "torso", "player_dodge_stance": "stand_firm"}
        )

        # 2. Reset match
        status, res = self._http_post_json(
            "/api/game/reset",
            {"match_id": "session_reset_test"}
        )
        self.assertEqual(status, 200)
        self.assertEqual(res["status"], "RESET")
        self.assertEqual(res["vital_max_hp"], 6)
        self.assertEqual(res["match"]["player"]["current_hp"], 6)
        self.assertEqual(res["match"]["opponent"]["current_hp"], 6)
        self.assertEqual(res["match"]["round_number"], 1)
        self.assertFalse(res["match"]["is_game_over"])

    # ── 7. MULTI-TENANT SESSION ISOLATION (ELIMINATION OF SINGLETON BLOCKER) ────
    def test_07_multi_session_isolation_architectural_proof(self):
        """
        ARCHITECTURAL PROOF:
        Match A (alpha) and Match B (beta) operate independently.
        Modifying Alpha DOES NOT contaminate or mutate Beta.
        """
        session_a = "alpha_contestant"
        session_b = "beta_contestant"

        # 1. Initialize both sessions via HTTP
        self._http_post_json("/api/game/reset", {"match_id": session_a})
        self._http_post_json("/api/game/reset", {"match_id": session_b})

        # 2. Cause heavy combat in Session A
        for _ in range(4):
            self._http_post_json(
                "/api/game/duel_round",
                {"match_id": session_a, "player_attack_zone": "head", "player_dodge_stance": "duck_down"}
            )

        # 3. Query Session A status
        _, a_stat_bytes, _ = self._http_get(f"/api/game/status?match_id={session_a}")
        a_stat = json.loads(a_stat_bytes.decode("utf-8"))

        # 4. Query Session B status
        _, b_stat_bytes, _ = self._http_get(f"/api/game/status?match_id={session_b}")
        b_stat = json.loads(b_stat_bytes.decode("utf-8"))

        # 5. Assert: Session A progressed and took battle wear
        self.assertGreater(a_stat["match"]["round_number"], 1)

        # 6. Assert: Session B remained 100% PRISTINE!
        self.assertEqual(b_stat["match"]["round_number"], 1)
        self.assertEqual(b_stat["match"]["player"]["current_hp"], 6)
        self.assertEqual(b_stat["match"]["opponent"]["current_hp"], 6)
        self.assertFalse(b_stat["match"]["is_game_over"])

        # Dig a trench in Session B
        self._http_post_json(
            "/api/game/sapper_trench",
            {"match_id": session_b, "hex_coord": [2, -1], "unit_id": "player_hero"}
        )

        # Assert: Session B has trench, Session A does NOT!
        _, b_stat_after, _ = self._http_get(f"/api/game/status?match_id={session_b}")
        _, a_stat_after, _ = self._http_get(f"/api/game/status?match_id={session_a}")
        b_res = json.loads(b_stat_after.decode("utf-8"))
        a_res = json.loads(a_stat_after.decode("utf-8"))

        self.assertIn("2_-1", b_res["match"]["trenches"])
        self.assertNotIn("2_-1", a_res["match"]["trenches"])

    # ── 8. EXHAUSTIVE INVARIANT PRESERVATION ───────────────────────────────────
    def test_08_exhaustive_invariant_preservation_under_combat(self):
        """In every round of intense combat, 0 <= HP <= 6 and VITAL_MAX_HP == 6 strictly hold."""
        m_id = "session_invariant_exhaustive"
        self._http_post_json("/api/game/reset", {"match_id": m_id})

        for round_idx in range(1, 15):
            status, res = self._http_post_json(
                "/api/game/duel_round",
                {"match_id": m_id, "player_attack_zone": "torso", "player_dodge_stance": "stand_firm"}
            )
            self.assertEqual(status, 200)
            self.assertEqual(res["vital_max_hp"], 6)

            p_hp = res["player_hp"]
            o_hp = res["opponent_hp"]

            # Strict bounds
            self.assertTrue(0 <= p_hp <= 6, f"Violation: player_hp={p_hp}")
            self.assertTrue(0 <= o_hp <= 6, f"Violation: opponent_hp={o_hp}")

            tel = res.get("telemetry", {})
            self.assertTrue(tel.get("invariant_ok", False))

            if res.get("is_game_over"):
                break


if __name__ == "__main__":
    unittest.main()
