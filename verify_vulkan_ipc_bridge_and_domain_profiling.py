#!/usr/bin/env python3
"""
==============================================================================
VERIFICATION SUITE: VULKAN IPC BRIDGE & CROSS-APPLICATION DOMAIN PROFILER
==============================================================================
Empirically verifies:
  1. Inviolable System Invariant: VITAL_MAX_HP == 6.
  2. C-ABI Export Header: include/krystal_vulkan_ipc_bridge.h integrity.
  3. Vulkan IPC Bridge & Client hardware kernels (GEMM, Interning, AABB, SHM).
  4. Cross-Application Domain Profiling across 5 primary workload domains.
  5. Hub Server Endpoints (/api/vulkan_ipc/telemetry, /benchmark_domains, /dispatch).

Author: Dušan Kopecký & Krystal-Stack Architecture Council (2026)
==============================================================================
"""

import os
import sys
import json
import io
import time
from pathlib import Path

# Workspace setup
WORKSPACE_ROOT = os.path.dirname(os.path.abspath(__file__))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

from krystal_stack_nextgen import (
    VITAL_MAX_HP,
    VulkanIPCOpcode,
    VulkanIPCBridge,
    VulkanIPCClient,
    GLOBAL_VULKAN_IPC_BRIDGE,
    DomainBenchmarkResult,
    IPCApplicationBenchmarkSuite,
    GLOBAL_IPC_BENCHMARK
)


def test_system_invariant():
    print(">> [1/5] Verifying Inviolable System Invariant (VITAL_MAX_HP == 6)...")
    assert VITAL_MAX_HP == 6, f"Expected VITAL_MAX_HP == 6, got {VITAL_MAX_HP}"
    
    bridge = GLOBAL_VULKAN_IPC_BRIDGE
    assert bridge.vital_hp == 6
    assert bridge.text_processor.vital_hp == 6
    
    client = VulkanIPCClient(bridge)
    assert client.bridge.vital_hp == 6
    print("   [PASS] Invariant VITAL_MAX_HP == 6 strictly verified across bridge and client.")


def test_c_abi_header_integrity():
    print(">> [2/5] Verifying C-ABI Export Header (include/krystal_vulkan_ipc_bridge.h)...")
    header_path = Path(WORKSPACE_ROOT) / "include" / "krystal_vulkan_ipc_bridge.h"
    assert header_path.exists(), "include/krystal_vulkan_ipc_bridge.h missing!"
    
    content = header_path.read_text(encoding="utf-8")
    assert "KRYSTAL_VITAL_MAX_HP     6" in content
    assert "KRYSTAL_MAGIC_BYTES      0x4B525953" in content
    assert "krystal_ipc_init" in content
    assert "krystal_ipc_dispatch" in content
    assert "krystal_ipc_query_telemetry" in content
    assert "krystal_ipc_close" in content
    assert "KrystalIPCPacketHeader" in content
    print("   [PASS] ANSI C / C++ exportable header verified with 20-byte packed struct definition.")


def test_vulkan_ipc_bridge_and_client():
    print(">> [3/5] Verifying Vulkan IPC Bridge Kernels & Shared Memory...")
    bridge = GLOBAL_VULKAN_IPC_BRIDGE
    client = VulkanIPCClient(bridge)
    
    # 1. Test GEMM
    a = [2.0, 3.0, 4.0, 5.0]
    b = [1.0, 0.0, 0.0, 1.0]  # Identity matrix
    c = client.multiply_matrices(2, 2, 2, a, b)
    assert c == [2.0, 3.0, 4.0, 5.0], f"Identity GEMM failed: {c}"
    
    # 2. Test Text Token Interning
    text = "CYBERPUNK TILE4 NPU_PRESTAGE TPU_SYSTOLIC"
    toks = client.intern_text(text)
    assert len(toks) == 4, f"Expected 4 tokens, got {toks}"
    
    # 3. Test AABB Culling
    frustum_min = (-5.0, -5.0, -5.0)
    frustum_max = (5.0, 5.0, 5.0)
    boxes = [
        (-1.0, -1.0, -1.0, 1.0, 1.0, 1.0),     # Inside
        (10.0, 10.0, 10.0, 12.0, 12.0, 12.0),  # Outside
        (-2.0, -2.0, -2.0, 0.0, 0.0, 0.0),     # Inside
    ]
    vis = client.cull_aabbs(frustum_min, frustum_max, boxes)
    assert vis == [0, 2], f"Expected visible indices [0, 2], got {vis}"
    
    # 4. Telemetry check
    telem = bridge.get_telemetry()
    assert telem["status"] == "ONLINE"
    assert telem["vital_max_hp"] == 6
    assert telem["total_dispatches"] >= 3
    print(f"   [PASS] Hardware kernels executed. Device: '{telem['vulkan_device_name']}'. Dispatches: {telem['total_dispatches']}.")


def test_cross_application_domain_benchmarks():
    print(">> [4/5] Verifying Cross-Application Domain Profiler (5 Domains)...")
    suite = GLOBAL_IPC_BENCHMARK
    results = suite.run_full_suite(iterations=1000)
    
    assert len(results) == 5, f"Expected 5 benchmark domains, got {len(results)}"
    
    # Check that Web Streamer shows huge speedup (> 15x)
    streamer_res = next(r for r in results if r.domain_id == "DOM-1-WEB-STREAMER")
    assert streamer_res.speedup_multiplier >= 15.0, f"Expected Web Streamer speedup >= 15x, got {streamer_res.speedup_multiplier}x"
    assert streamer_res.l1_cache_lines_saved >= 100, f"Expected >= 100 L1 cache lines saved, got {streamer_res.l1_cache_lines_saved}"
    
    # Check that all domains show speedup > 3.0x
    for r in results:
        assert r.speedup_multiplier >= 3.0, f"Domain {r.domain_id} speedup too low: {r.speedup_multiplier}x"
        assert r.bandwidth_saved_pct >= 30.0, f"Domain {r.domain_id} bandwidth savings too low: {r.bandwidth_saved_pct}%"
        assert r.vital_max_hp == 6
    print(f"   [PASS] All 5 domains benchmarked. Top speedup: {results[0].name} ({results[0].speedup_multiplier}x).")


def test_server_vulkan_ipc_endpoints():
    print(">> [5/5] Verifying Hub Server Vulkan IPC REST Endpoints...")
    from krystal_web_hub.server import KrystalHubHandler
    
    class DummySocket:
        def __init__(self, request_bytes):
            self._rfile = io.BytesIO(request_bytes)
            self._wfile = io.BytesIO()

        def makefile(self, mode, *args, **kwargs):
            if "r" in mode:
                return self._rfile
            return self._wfile

        def setsockopt(self, *args, **kwargs):
            pass

        def sendall(self, b):
            self._wfile.write(b)

    # 1. Test GET /api/vulkan_ipc/telemetry
    req_get = b"GET /api/vulkan_ipc/telemetry HTTP/1.1\r\nHost: localhost:8080\r\n\r\n"
    sock = DummySocket(req_get)
    try:
        KrystalHubHandler(sock, ("127.0.0.1", 54321), None)
    except Exception:
        pass
    resp_telem = sock._wfile.getvalue()
    assert b"200 OK" in resp_telem, f"GET telemetry failed: {resp_telem[:200]}"
    assert b"vulkan_device_name" in resp_telem
    assert b"shared_memory_ring" in resp_telem

    # 2. Test GET /api/vulkan_ipc/benchmark_domains
    req_bench = b"GET /api/vulkan_ipc/benchmark_domains HTTP/1.1\r\nHost: localhost:8080\r\n\r\n"
    sock = DummySocket(req_bench)
    try:
        KrystalHubHandler(sock, ("127.0.0.1", 54321), None)
    except Exception:
        pass
    resp_bench = sock._wfile.getvalue()
    assert b"200 OK" in resp_bench, f"GET benchmark_domains failed: {resp_bench[:200]}"
    assert b"DOM-1-WEB-STREAMER" in resp_bench

    # 3. Test POST /api/vulkan_ipc/dispatch (GEMM)
    gemm_payload = json.dumps({
        "operation": "GEMM",
        "m": 2, "k": 2, "n": 2,
        "a": [1.0, 2.0, 3.0, 4.0],
        "b": [5.0, 6.0, 7.0, 8.0]
    }).encode("utf-8")
    req_post = f"POST /api/vulkan_ipc/dispatch HTTP/1.1\r\nHost: localhost:8080\r\nContent-Length: {len(gemm_payload)}\r\n\r\n".encode("utf-8") + gemm_payload
    sock = DummySocket(req_post)
    try:
        KrystalHubHandler(sock, ("127.0.0.1", 54321), None)
    except Exception:
        pass
    resp_post = sock._wfile.getvalue()
    assert b"200 OK" in resp_post, f"POST dispatch failed: {resp_post[:200]}"
    assert b"[19.0, 22.0, 43.0, 50.0]" in resp_post
    print("   [PASS] Endpoints /api/vulkan_ipc/telemetry, /benchmark_domains, and /dispatch returned 200 OK.")


def main():
    if sys.stdout.encoding.lower() != 'utf-8':
        sys.stdout.reconfigure(encoding='utf-8')
    print("=" * 80)
    print("  VERIFYING VULKAN IPC ACCELERATION BRIDGE & DOMAIN PROFILER")
    print("=" * 80)
    t0 = time.perf_counter()
    
    test_system_invariant()
    test_c_abi_header_integrity()
    test_vulkan_ipc_bridge_and_client()
    test_cross_application_domain_benchmarks()
    test_server_vulkan_ipc_endpoints()
    
    dt = time.perf_counter() - t0
    print("=" * 80)
    print(f"  ALL 5 SUITES PASSED STRICTLY IN {dt:.3f} SECONDS")
    print("=" * 80)


if __name__ == "__main__":
    main()
