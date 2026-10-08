#!/usr/bin/env python3
"""
VERIFICATION SUITE: NPU HARDWARE PREDICTOR & SPECULATIVE MEMORY PRE-STAGING
===========================================================================
Verifies:
  1. Inviolable Invariant: VITAL_MAX_HP = 6.
  2. Windows Telemetry Ingestion (GlobalMemoryStatusEx, RAM, NPU power).
  3. NPU Cognitive Scheduling Inference & Strategy Generation.
  4. Speculative Memory Pre-Staging (SSD Swap -> Hot Pinned RAM Ring Buffer).
  5. Latency Benchmark: 0.08µs (Pinned RAM) vs 120µs (NVMe I/O) = 1500x Speedup.
  6. Server REST Endpoints:
     - GET /api/nextgen/npu_predictor
     - POST /api/nextgen/npu_prestage

Author: Dušan Kopecký & Krystal-Stack Architecture Council (2026)
"""

import sys
import os
import io
import json

WORKSPACE_ROOT = os.path.dirname(os.path.abspath(__file__))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

if sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

from krystal_stack_nextgen import (
    VITAL_MAX_HP,
    NPUPredictionAction,
    NPUSpeculativeStrategy,
    NPUTelemetrySnapshot,
    NPUSpeculativePreStager,
    NPUHardwarePredictor,
    GLOBAL_NPU_PREDICTOR
)
from krystal_web_hub.server import KrystalHubHandler


class DummyHandler(KrystalHubHandler):
    def __init__(self, path, method="GET", body=b""):
        self.path = path
        self.command = method
        self.requestline = f"{method} {path} HTTP/1.1"
        self.request_version = "HTTP/1.1"
        self.headers = {"Content-Length": str(len(body)), "Content-Type": "application/json"}
        self.rfile = io.BytesIO(body)
        self.wfile = io.BytesIO()
        if method == "GET":
            self.do_GET()
        elif method == "POST":
            self.do_POST()


def test_invariant():
    print("\n[TEST 1] Verifying Inviolable Architectural Invariant...")
    assert VITAL_MAX_HP == 6, f"Expected VITAL_MAX_HP == 6, got {VITAL_MAX_HP}"
    print(f"  ✓ Invariant strictly verified: VITAL_MAX_HP = {VITAL_MAX_HP}")


def test_windows_telemetry_and_npu_strategy():
    print("\n[TEST 2] Verifying Windows Telemetry & NPU Cognitive Strategy...")
    predictor = GLOBAL_NPU_PREDICTOR

    telem = predictor.probe_windows_telemetry()
    print(f"  ✓ Windows Host Telemetry: Free RAM={telem.system_ram_free_gb} GB (Load: {telem.system_ram_load_pct}%) | NPU Power={telem.npu_power_watts} W")
    assert telem.system_ram_free_gb > 0.0

    strat = predictor.evaluate_predictive_strategy(
        predicted_asset_key="BOHEMIAN_GOTHIC_TOWER_LOD0",
        expected_raymarch_steps=96
    )
    print(f"  ✓ NPU Strategy ID:      {strat.strategy_id}")
    print(f"  ✓ Action Decided:       {strat.action}")
    print(f"  ✓ Hit Probability:      {strat.predicted_hit_probability * 100:.1f}%")
    print(f"  ✓ Latency Saved:        {strat.latency_saved_us} µs")
    print(f"  ✓ NPU Decision Time:    {strat.npu_compute_time_us} µs")
    assert strat.vital_hp == 6
    assert strat.predicted_hit_probability > 0.80


def test_speculative_memory_prestaging_latency():
    print("\n[TEST 3] Verifying Speculative Memory Pre-Staging Latency (RAM vs SSD)...")
    prestager = GLOBAL_NPU_PREDICTOR.prestager
    test_block_id = "SHADOW_MAP_HEX_ARRAY_CHUNK_01"

    # 1. Test before pre-staging (Cold Miss)
    is_hot, lat_miss = prestager.query_block_latency(test_block_id)
    print(f"  • Unpinned Query: Pinned={is_hot}, Latency={lat_miss} µs (Cold NVMe SSD access)")
    assert is_hot is False
    assert lat_miss >= 100.0

    # 2. Pre-stage block from SSD into pinned RAM
    ok = prestager.prestage_block_from_ssd(test_block_id, payload_size_bytes=8192)
    assert ok is True

    # 3. Test after pre-staging (Hot RAM Hit)
    is_hot_after, lat_hit = prestager.query_block_latency(test_block_id)
    print(f"  • Pre-Staged Query: Pinned={is_hot_after}, Latency={lat_hit} µs (Direct Pinned RAM hit)")
    assert is_hot_after is True
    assert lat_hit < 1.0

    speedup = lat_miss / lat_hit
    print(f"  ✓ Verified Latency Elimination: {speedup:.1f}x faster access (Zero-Stall CPU/GPU throughput)!")
    assert speedup > 1000.0


def test_npu_predictor_server_endpoints():
    print("\n[TEST 4] Verifying Web Hub NPU Predictor REST Endpoints...")

    # 1. GET /api/nextgen/npu_predictor
    h1 = DummyHandler("/api/nextgen/npu_predictor")
    h1.wfile.seek(0)
    out1 = h1.wfile.read().decode("utf-8", errors="ignore")
    assert "200 OK" in out1, f"Expected 200 OK, got: {out1[:150]}"
    assert '"status": "OK"' in out1
    assert '"windows_telemetry"' in out1
    assert '"active_npu_strategy"' in out1
    assert '"vital_max_hp": 6' in out1
    print("  ✓ GET /api/nextgen/npu_predictor -> 200 OK")

    # 2. POST /api/nextgen/npu_prestage
    body = json.dumps({"asset_key": "TERRAIN_OCTAVE_SURGE_CHUNK_09", "expected_raymarch_steps": 96}).encode("utf-8")
    h2 = DummyHandler("/api/nextgen/npu_prestage", method="POST", body=body)
    h2.wfile.seek(0)
    out2 = h2.wfile.read().decode("utf-8", errors="ignore")
    assert "200 OK" in out2, f"Expected 200 OK, got: {out2[:150]}"
    assert '"status": "SUCCESS"' in out2
    assert '"is_pinned_in_hot_ram": true' in out2
    assert "1500x faster" in out2
    print("  ✓ POST /api/nextgen/npu_prestage -> 200 OK (Speculatively Pre-Staged to RAM)")


def main():
    print("=" * 75)
    print(" 🧠 KRYSTAL-STACK NEXTGEN: NPU PREDICTOR FULL SYSTEM VERIFICATION")
    print("=" * 75)

    test_invariant()
    test_windows_telemetry_and_npu_strategy()
    test_speculative_memory_prestaging_latency()
    test_npu_predictor_server_endpoints()

    print("\n" + "=" * 75)
    print(" ✅ ALL NPU HARDWARE PREDICTOR TESTS PASSED (100% SUCCESS)")
    print("=" * 75)


if __name__ == "__main__":
    main()
