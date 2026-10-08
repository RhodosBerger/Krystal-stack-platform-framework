#!/usr/bin/env python3
"""
VERIFICATION TEST: TPU BENCHMARK, ONTOLOGICAL REVERSE PROMPTS & UTF-8 INTEGRITY
=============================================================================
Verifies:
  1. Inviolable Invariant: VITAL_MAX_HP = 6.
  2. TPU Systolic Tensor Acceleration Benchmark (FP32, FP16, INT8).
  3. Ontological Reverse-Engineering Engine & Prompt Catalog.
  4. UTF-8 HTTP Header & Encoding Integrity (preventing mojibake in control panel).
  5. Web Hub REST Endpoints:
     - GET /api/nextgen/tpu_benchmark
     - GET /api/nextgen/ontological_prompts
     - GET / (Index.html with text/html; charset=utf-8)

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
    TensorPrecision,
    TPUBenchmarkResult,
    TPUSweepSummary,
    TPUTensorBenchmark,
    GLOBAL_TPU_BENCHMARK,
    OntologicalDomain,
    OntologicalPrompt,
    OntologicalReverseEngineeringEngine,
    GLOBAL_ONTOLOGICAL_ENGINE
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
    print("\n[TEST 1] Verifying System Invariant VITAL_MAX_HP = 6...")
    assert VITAL_MAX_HP == 6, f"Expected VITAL_MAX_HP == 6, got {VITAL_MAX_HP}"
    print(f"  ✓ Invariant strictly verified: VITAL_MAX_HP = {VITAL_MAX_HP}")


def test_tpu_tensor_benchmark():
    print("\n[TEST 2] Verifying TPU Tensor Acceleration Benchmark...")
    bench = GLOBAL_TPU_BENCHMARK

    # 1. Targeted 128x128 FP16 benchmark
    r_fp16 = bench.run_benchmark(dimension=128, precision=TensorPrecision.FP16)
    print(f"  ✓ Dimension: 128x128 FP16 | CPU: {r_fp16.cpu_scalar_duration_ms:.2f}ms | TPU: {r_fp16.tpu_tensor_duration_ms:.3f}ms | Speedup: {r_fp16.tpu_acceleration_factor:.1f}x")
    assert r_fp16.tpu_acceleration_factor > 100.0
    assert r_fp16.vital_max_hp == 6

    # 2. Targeted 128x128 INT8 benchmark
    r_int8 = bench.run_benchmark(dimension=128, precision=TensorPrecision.INT8)
    print(f"  ✓ Dimension: 128x128 INT8 | CPU: {r_int8.cpu_scalar_duration_ms:.2f}ms | TPU: {r_int8.tpu_tensor_duration_ms:.3f}ms | Speedup: {r_int8.tpu_acceleration_factor:.1f}x")
    assert r_int8.tpu_acceleration_factor > r_fp16.tpu_acceleration_factor

    # 3. Sweep
    sweep = bench.run_full_sweep()
    print(f"  ✓ Full Sweep: {sweep.total_benchmarks_run} configurations tested | Peak TOPS: {sweep.peak_tpu_tops:.4f}")
    assert sweep.total_benchmarks_run == 12
    assert sweep.max_tpu_acceleration > 1000.0


def test_ontological_reverse_prompts():
    print("\n[TEST 3] Verifying Ontological Reverse-Engineering Prompts...")
    engine = GLOBAL_ONTOLOGICAL_ENGINE

    assert len(engine.catalog) >= 5
    for p in engine.catalog:
        print(f"  ✓ [{p.prompt_id}] {p.title} (Domain: {p.domain.value})")
        assert len(p.ontological_axioms) >= 3
        assert len(p.reverse_engineering_directive_sk) > 30
        assert len(p.reverse_engineering_directive_en) > 30
        assert p.vital_max_hp == 6

    md = engine.export_all_prompts_markdown()
    assert "# KRYSTAL-STACK: KNIHA PROMPTOV PRE ONTOLOGICKÉ REVERZNÉ INŽINIERSTVO" in md
    assert "VITAL_MAX_HP = 6" in md
    print(f"  ✓ Exported markdown document verified ({len(md)} chars).")


def test_utf8_encoding_and_control_panel_integrity():
    print("\n[TEST 4] Verifying UTF-8 Encodings & Control Panel Integrity...")

    # Test GET / (index.html)
    h_root = DummyHandler("/")
    h_root.wfile.seek(0)
    out_root = h_root.wfile.read().decode("utf-8", errors="strict")
    assert "200 OK" in out_root
    assert "Content-Type: text/html; charset=utf-8" in out_root
    assert '<meta charset="UTF-8">' in out_root
    assert '<meta http-equiv="Content-Type" content="text/html; charset=utf-8">' in out_root
    assert "panel panel-controls" in out_root
    print("  ✓ Root index.html served with strict 'Content-Type: text/html; charset=utf-8'")


def test_server_nextgen_endpoints():
    print("\n[TEST 5] Verifying Web Hub TPU & Ontological REST Endpoints...")

    # 1. GET /api/nextgen/tpu_benchmark
    h_tpu = DummyHandler("/api/nextgen/tpu_benchmark?sweep=true")
    h_tpu.wfile.seek(0)
    out_tpu = h_tpu.wfile.read().decode("utf-8", errors="strict")
    assert "200 OK" in out_tpu
    assert '"status": "SUCCESS"' in out_tpu
    assert '"benchmark_summary"' in out_tpu
    assert '"vital_max_hp": 6' in out_tpu
    print("  ✓ GET /api/nextgen/tpu_benchmark -> 200 OK (Measured TPU Leap)")

    # 2. GET /api/nextgen/ontological_prompts
    h_onto = DummyHandler("/api/nextgen/ontological_prompts?format=markdown")
    h_onto.wfile.seek(0)
    out_onto = h_onto.wfile.read().decode("utf-8", errors="strict")
    assert "200 OK" in out_onto
    assert "Content-Type: text/markdown; charset=utf-8" in out_onto
    assert "KNIHA PROMPTOV PRE ONTOLOGICKÉ REVERZNÉ INŽINIERSTVO" in out_onto
    print("  ✓ GET /api/nextgen/ontological_prompts -> 200 OK (Markdown Stream)")


def main():
    print("=" * 80)
    print(" 🔬 KRYSTAL-STACK NEXTGEN: TPU BENCHMARK & ONTOLOGICAL PROMPT VERIFICATION")
    print("=" * 80)

    test_invariant()
    test_tpu_tensor_benchmark()
    test_ontological_reverse_prompts()
    test_utf8_encoding_and_control_panel_integrity()
    test_server_nextgen_endpoints()

    print("\n" + "=" * 80)
    print(" ✅ ALL TPU BENCHMARK & ONTOLOGICAL REVERSE TESTS PASSED (100% SUCCESS)")
    print("=" * 80)


if __name__ == "__main__":
    main()
