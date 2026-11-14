"""Tests for the Babylon.js simulation grid and production hardening of the HTTP server."""

from __future__ import annotations

import http.server
import json
import threading
import urllib.error
import urllib.request

import pytest

from krystal_kernel.sovereign_openvino_oneapi_app import SovereignOpenVinoRequestHandler
from projects.trigonometric_hybrid_platform_framework.simulation_grid_3d import (
    BABYLON_SRI,
    VENDOR_PATH,
    build_trace_dataset,
    ensure_babylon_vendor,
    hex_spiral_axial,
    render_simulation_grid_3d_html,
)


def test_hex_spiral_has_19_unique_cells_within_radius_2() -> None:
    cells = hex_spiral_axial(2)
    assert len(cells) == 19 and len(set(cells)) == 19
    assert all(max(abs(q), abs(r), abs(-q - r)) <= 2 for q, r in cells)


def test_trace_dataset_consistency() -> None:
    data = build_trace_dataset(ticks=200)
    for name, prof in data["profiles"].items():
        t = prof["trace"]
        assert len(t["kernel"]) == len(t["first_prediction"]) == len(t["final_client"]) == 200, name
        assert prof["metrics"]["desync_ticks"] == 0
        assert t["final_client"] == t["kernel"]  # after rollbacks client equals kernel on every tick
        assert t["corrected"] == [fp != fc for fp, fc in zip(t["first_prediction"], t["final_client"])]
        assert all(row[1] <= 6 and row[3] <= 6 for row in t["kernel"])
        assert all(0 <= row[0] < 19 and 0 <= row[2] < 19 for row in t["kernel"])


def test_html_references_pinned_babylon() -> None:
    html_doc = render_simulation_grid_3d_html(ticks=100)
    assert 'src="vendor/babylon.js"' in html_doc
    assert BABYLON_SRI in html_doc
    assert "box-shadow" not in html_doc


@pytest.mark.skipif(not VENDOR_PATH.exists(), reason="vendored Babylon.js not present (no network fetch in tests)")
def test_vendored_babylon_hash() -> None:
    info = ensure_babylon_vendor()
    assert info["version"] == "9.36.0"


def _serve():
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), SovereignOpenVinoRequestHandler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, server.server_address[1]


def test_healthz_and_production_prompt_edit_guard(monkeypatch: pytest.MonkeyPatch) -> None:
    server, port = _serve()
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/healthz", timeout=5) as r:
            assert r.status == 200 and json.loads(r.read())["status"] == "ok"

        monkeypatch.setenv("KRYSTAL_ENV", "production")
        monkeypatch.setenv("KRYSTAL_ADMIN_TOKEN", "secret-token")
        body = json.dumps({"prompt_edit": {"prompt_id": "MP_01_EPISTEMIC_DUAL_MODE",
                                           "updated_directive": "test"}}).encode()
        url = f"http://127.0.0.1:{port}/api/v1/runtime/trigonometric_hybrid_platform"

        with pytest.raises(urllib.error.HTTPError) as exc:
            urllib.request.urlopen(urllib.request.Request(url, data=body, method="POST"), timeout=10)
        assert exc.value.code == 403

        req = urllib.request.Request(url, data=body, method="POST", headers={"X-Krystal-Admin-Token": "secret-token"})
        with urllib.request.urlopen(req, timeout=30) as r:
            assert r.status == 200

        with urllib.request.urlopen(f"http://127.0.0.1:{port}/simulation_grid_3d", timeout=30) as r:
            assert r.status == 200 and b"Simulation Grid 3D" in r.read()
    finally:
        server.shutdown()
        server.server_close()
