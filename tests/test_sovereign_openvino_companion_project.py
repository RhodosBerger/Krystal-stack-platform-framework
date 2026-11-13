"""
Comprehensive tests for `projects/sovereign_openvino_companion` and full HTTP stack routes
(including GLOBAL_HYBRID_LORE_AGGREGATOR POST endpoints and OpenVINO CPU/GPU crossover).
"""

from __future__ import annotations

import http.server
import json
import math
import threading
import urllib.request
from pathlib import Path

import pytest

from krystal_kernel.constants import ContractViolation, HYSTERESIS_DWELL_FRAMES, VITAL_MAX_HP
from krystal_kernel.sovereign_openvino_oneapi_app import SovereignOpenVinoRequestHandler
from projects.sovereign_openvino_companion import (
    ComputationLogSystem,
    FROZEN_THEORY_NODES,
    SovereignOpenVinoCompanionProject,
)
from projects.sovereign_openvino_companion.cli import main as cli_main


def test_environment_and_constitutional_invariants(tmp_path: Path) -> None:
    proj = SovereignOpenVinoCompanionProject(log_dir=tmp_path, reset_ledger=True)
    env = proj.validate_environment()
    assert env["vital_max_hp"] == 6
    assert env["frozen_theory_nodes"] == 56
    assert env["hysteresis_dwell_frames"] == HYSTERESIS_DWELL_FRAMES
    assert "CPU" in env["available_devices"]


def test_crossover_and_12_frame_hysteresis(tmp_path: Path) -> None:
    proj = SovereignOpenVinoCompanionProject(log_dir=tmp_path, reset_ledger=True)

    # 1. Micro-tensor (< 16 KB) routes to CPU
    micro = proj.process_companion_frame(4.8, 0.22, payload_bytes=4096)
    assert micro["actual_device"] == "CPU"
    assert micro["k_speculative_slots"] == 8
    assert micro["vital_max_hp"] == 6

    # 2. Dense tensor (>= 16 KB) routes to GPU if GPU is available
    dense = proj.process_companion_frame(4.8, 0.25, payload_bytes=32768)
    expected_dense = "GPU" if "GPU" in proj.ov_runtime.available_devices else "CPU"
    assert dense["actual_device"] == expected_dense

    # 3. UMA pressure spike (>= 0.78) triggers fast downscale & 12-frame dwell
    spike = proj.process_companion_frame(4.8, 0.82, payload_bytes=32768)
    assert spike["actual_device"] == "CPU"
    assert spike["k_speculative_slots"] == 1
    assert spike["upscaling_active"] is True
    assert spike["hysteresis_dwell_counter"] == HYSTERESIS_DWELL_FRAMES

    # Next 12 frames remain pinned to CPU even at low UMA
    for _ in range(HYSTERESIS_DWELL_FRAMES):
        held = proj.process_companion_frame(4.8, 0.22, payload_bytes=32768)
        assert held["actual_device"] == "CPU"
        assert held["hysteresis_held"] is True
        assert held["k_speculative_slots"] == 1

    # Frame after dwell recovers dense GPU routing
    recovered = proj.process_companion_frame(4.8, 0.22, payload_bytes=32768)
    assert recovered["hysteresis_held"] is False
    assert recovered["k_speculative_slots"] == 8
    assert recovered["actual_device"] == expected_dense


def test_thermal_parasite_and_homotopic_conflict_quarantine(tmp_path: Path) -> None:
    proj = SovereignOpenVinoCompanionProject(log_dir=tmp_path, reset_ledger=True)

    # Epistemic entropy reduction <= 0 triggers P5 thermal parasite decoupling -> quarantined
    parasite = proj.process_companion_frame(
        4.8, 0.25, payload_bytes=4096, epistemic_entropy_nats=0.0, contradictory_evidence=False
    )
    assert parasite["gnosis_nats_per_mj"] == 0.0
    assert parasite["artifact_released"] is False

    # Contradictory evidence triggers homotopic conflict tag in ledger
    conflict = proj.process_companion_frame(
        4.8, 0.25, payload_bytes=4096, epistemic_entropy_nats=1.3689, contradictory_evidence=True
    )
    assert conflict["artifact_released"] is True
    ok, audit = proj.logger.verify_chain_integrity()
    assert ok is True
    assert audit["pattern_counts"].get("THERMAL_PARASITE_DECOUPLED", 0) >= 1
    assert audit["pattern_counts"].get("HOMOTOPIC_CONFLICT_FORK", 0) >= 1
    assert audit["pattern_counts"].get("AUTHORITY_SEPARATION_QUARANTINE", 0) >= 1


def test_computation_logger_tamper_detection_and_guards(tmp_path: Path) -> None:
    logger = ComputationLogSystem(log_dir=tmp_path, reset_ledger=True)
    logger.log_event("system", "BOOT", "S0", 0.12, {"uma_saturation": 0.80, "selected_device": "CPU"})
    logger.log_event("inference", "RUN", "S1", 0.45, {"payload_bytes": 4096, "selected_device": "CPU"})

    ok, report = logger.verify_chain_integrity()
    assert ok is True
    assert report["entries_checked"] == 2
    assert "UMA_BANDWIDTH_PRESSURE_SPIKE" in report["pattern_counts"]
    assert "MICRO_TENSOR_CPU_BYPASS" in report["pattern_counts"]

    # NaN float and invalid channel must raise ContractViolation
    with pytest.raises(ContractViolation):
        logger.log_event("invalid_channel", "BAD", "S0", 0.1, {})
    with pytest.raises(ContractViolation):
        logger.log_event("system", "BAD_FLOAT", "S0", 0.1, {"val": math.nan})

    # Tamper one character on disk and verify chain fails
    raw_lines = logger.ledger_path.read_text(encoding="utf-8").splitlines()
    tampered_rec = json.loads(raw_lines[0])
    tampered_rec["duration_ms"] = 999.0
    raw_lines[0] = json.dumps(tampered_rec, sort_keys=True)
    logger.ledger_path.write_text("\n".join(raw_lines) + "\n", encoding="utf-8")

    ok_after, fail_report = logger.verify_chain_integrity()
    assert ok_after is False
    assert "SHA256_TAMPER_DETECTED_AT_0" in fail_report["reason"]


def test_full_live_http_stack_routes() -> None:
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), SovereignOpenVinoRequestHandler)
    port = srv.server_address[1]
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()

    try:
        routes = [
            ("GET", "/api/v1/runtime/dusan_game_vulkan_power_plan", None),
            ("GET", "/api/v1/runtime/senior_prompt_lore_quest", None),
            ("GET", "/api/v1/runtime/adaptive_genetic_scenario", None),
            ("GET", "/api/v1/runtime/omniscience_core", None),
            ("GET", "/api/v1/runtime/ontologic_gnosis_strata?uma=0.35&conflict=0", None),
            ("GET", "/api/v1/runtime/ontologic_gnosis_strata?uma=0.75&conflict=1", None),
            ("GET", "/api/v1/runtime/ontologic_asset_objects", None),
            ("GET", "/api/v1/runtime/sovereign_openvino_companion", None),
            ("GET", "/studio", None),
            ("GET", "/static/audio/krystal_posledni_kmen_theme.wav", None),
            ("GET", "/static/img/krystal_hex_rubik_emblem.svg", None),
            ("POST", "/api/hybrid/fetch_objectives", {"semantic_lens": "LORE_NARRATIVE"}),
            ("POST", "/api/hybrid/marias_click", {"action_id": "ACT_MARIAS_TRUMP_MARRIAGE", "from_idx": 12, "to_idx": 28}),
        ]

        for method, path, body in routes:
            data = json.dumps(body).encode("utf-8") if body is not None else None
            headers = {"Content-Type": "application/json"} if body is not None else {}
            req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=data, headers=headers, method=method)
            with urllib.request.urlopen(req, timeout=10) as resp:
                assert resp.status == 200
                raw = resp.read()
                assert len(raw) > 0
                if path == "/api/hybrid/marias_click":
                    decoded = json.loads(raw.decode("utf-8"))
                    assert decoded["magic"] == "KMB1"
                    assert decoded["crc_verified"] is True
                    assert decoded["vital_max_hp"] == 6
    finally:
        srv.shutdown()
        srv.server_close()


def test_cli_runner_executes_cleanly(tmp_path: Path) -> None:
    rc = cli_main([
        "--validate-env",
        "--run-session",
        "--frames",
        "10",
        "--verify-logs",
        "--reset-ledger",
        "--log-dir",
        str(tmp_path),
    ])
    assert rc == 0
    assert VITAL_MAX_HP == 6
    assert FROZEN_THEORY_NODES == 56
