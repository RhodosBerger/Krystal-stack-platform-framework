"""Tests for the Clash Battle Royale engine and HTTP mediator."""

from __future__ import annotations

import http.server
import json
import threading
import urllib.error
import urllib.request

import pytest

from krystal_kernel.constants import ContractViolation
from krystal_kernel.sovereign_openvino_oneapi_app import SovereignOpenVinoRequestHandler
from projects.trigonometric_hybrid_platform_framework.clash_battle_royale import (
    ROUND_BUDGET_FRAMES,
    ClashGame,
    board_cells,
)

VERDICTS = {f"C0{i}": {"verdict": v, "claim": f"check {i}"} for i, v in enumerate(
    ["VERIFIED_WITH_SCOPE", "SUPPORTED_WITH_SCOPE", "MEASURED_WITH_SCOPE", "VERIFIED_WITH_SCOPE",
     "CONFLICT", "MEASURED_WITH_SCOPE", "VERIFIED_WITH_SCOPE", "CONFLICT"], start=1)}


def test_board_and_invalid_action_leaves_state_unchanged() -> None:
    assert len(board_cells()) == 37
    g = ClashGame("SYNC", verdicts=VERDICTS)
    before = g.state_hash()
    with pytest.raises(ContractViolation):
        g.apply({"type": "move", "unit": "C01", "to": [9, 9]})          # off board
    with pytest.raises(ContractViolation):
        g.apply({"type": "move", "unit": "C03", "to": [0, 2]})          # not own unit
    with pytest.raises(ContractViolation):
        g.apply({"type": "card", "card": "FREIVALDS_AUDIT", "target": "C01"})  # C01 not in CONFLICT
    assert g.state_hash() == before


def test_move_costs_frame_and_receipt_has_128b_frame() -> None:
    g = ClashGame("SYNC", verdicts=VERDICTS)
    rec = g.apply({"type": "move", "unit": "C01", "to": [3, -1]})
    assert rec["frame_bytes"] == "128"
    assert g.state["budget"]["SYNC"] == ROUND_BUDGET_FRAMES - 1
    with pytest.raises(ContractViolation):
        g.apply({"type": "move", "unit": "C01", "to": [3, 0]})          # already acted


def test_full_game_terminates_with_hp_invariant() -> None:
    g = ClashGame("SYNC", verdicts=VERDICTS)
    for _ in range(20):
        if g.state["winner"]:
            break
        g.apply({"type": "end_turn"})
        assert all(0 <= u["hp"] <= 6 for u in g.state["units"].values())
    assert g.state["winner"] is not None


def test_http_mediator_roundtrip() -> None:
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), SovereignOpenVinoRequestHandler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_address[1]}"

    def post(path, body):
        req = urllib.request.Request(base + path, data=json.dumps(body).encode(), method="POST",
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.loads(r.read())

    try:
        new = post("/api/v1/runtime/clash/new", {"faction": "SYNC"})
        gid = new["game_id"]
        ok = post("/api/v1/runtime/clash/action", {"game_id": gid, "action": {"type": "move", "unit": "C01", "to": [3, -1]}})
        assert ok["pre_state_sha256"] != ok["state"]["state_sha256"]
        with pytest.raises(urllib.error.HTTPError) as exc:
            post("/api/v1/runtime/clash/action", {"game_id": gid, "action": {"type": "move", "unit": "C01", "to": [9, 9]}})
        assert exc.value.code == 400
        with urllib.request.urlopen(base + "/clash_arena", timeout=30) as r:
            assert r.status == 200 and b"Clash Battle Royale" in r.read()
    finally:
        server.shutdown()
        server.server_close()
