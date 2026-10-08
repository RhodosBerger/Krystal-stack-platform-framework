#!/usr/bin/env python3
"""
verify_wsl_gui_compositor_bridge.py
===================================
Automated verification suite for the WSL2 Low-Latency Linux GUI Compositor,
Cross-OS UMA Shared Surfaces, and Zero-Copy Latency Benchmark.

Enforces system invariant: VITAL_MAX_HP == 6.
"""

import sys
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

def test_1_create_cross_os_surface():
    print("[TEST 1] Creating Zero-Copy Cross-OS D3D12/Vulkan Surface...")
    from krystal_kernel.wsl_gui_compositor_bridge import (
        GLOBAL_WSL_GUI_COMPOSITOR_BRIDGE,
        WslGuiLatencyMode,
        VITAL_MAX_HP
    )
    assert VITAL_MAX_HP == 6

    bridge = GLOBAL_WSL_GUI_COMPOSITOR_BRIDGE
    surface = bridge.create_shared_surface(
        window_title="Hyprland Linux Shell (Wayland)",
        linux_pid=5120,
        width=2560,
        height=1440,
        mode=WslGuiLatencyMode.KRYSTAL_DIRECT_UMA
    )

    assert surface.zero_copy_active is True
    assert surface.render_latency_ms < 0.5
    assert surface.framerate_fps == 120.0
    assert surface.vital_max_hp == 6
    assert surface.dxgi_shared_handle.startswith("0x")
    print(f"  -> PASS: Surface {surface.surface_id} created: {surface.render_latency_ms} ms latency, 120 FPS lock.")

def test_2_benchmark_latency_vs_wslg():
    print("[TEST 2] Benchmarking Krystal Direct UMA vs Standard WSLg (RDP rail)...")
    from krystal_kernel.wsl_gui_compositor_bridge import GLOBAL_WSL_GUI_COMPOSITOR_BRIDGE
    
    bridge = GLOBAL_WSL_GUI_COMPOSITOR_BRIDGE
    bench = bridge.benchmark_gui_latency()

    assert bench.standard_wslg_latency_ms >= 15.0
    assert bench.krystal_uma_latency_ms <= 0.5
    assert bench.latency_reduction_factor >= 30.0
    assert bench.bandwidth_saved_pct > 70.0
    assert bench.kisa_speculation_active is True
    assert bench.vital_max_hp == 6
    print(f"  -> PASS: {bench.krystal_uma_latency_ms} ms vs {bench.standard_wslg_latency_ms} ms ({bench.latency_reduction_factor}x reduction, {bench.bandwidth_saved_pct}% bandwidth saved).")

def test_3_header_bridge_validation():
    print("[TEST 3] Validating C/C++ Header Bridge...")
    header_path = REPO_ROOT / "include" / "krystal_wsl_gui_bridge.h"
    assert header_path.exists(), f"Missing header: {header_path}"
    
    content = header_path.read_text(encoding="utf-8")
    assert "KRYSTAL_VITAL_MAX_HP 6" in content
    assert "KRYSTAL_WSL_GUI_DIRECT_UMA" in content
    assert "krystal_estimate_cross_os_latency" in content
    print("  -> PASS: include/krystal_wsl_gui_bridge.h structurally valid.")

def test_4_openapi_spec_validation():
    print("[TEST 4] Validating OpenAPI 3.1.0 Endpoints and Schemas...")
    from krystal_kernel.openapi_spec import get_openapi_specification

    spec = get_openapi_specification()
    paths = spec.get("paths", {})
    schemas = spec.get("components", {}).get("schemas", {})

    assert "/api/wsl/gui_status" in paths
    assert "/api/wsl/benchmark_gui_latency" in paths
    assert "/api/wsl/create_shared_surface" in paths

    assert "CrossOsSurfaceDescriptor" in schemas
    assert "WslGuiBenchmarkComparison" in schemas
    assert "WslGuiCreateSurfaceRequest" in schemas

    assert len(paths) >= 28
    assert len(schemas) >= 38
    print(f"  -> PASS: OpenAPI specification contains {len(paths)} endpoints and {len(schemas)} schemas.")

def main():
    print("================================================================================")
    print("  KRYSTAL STACK: WSL2 LOW-LATENCY GUI & CROSS-OS UMA BRIDGE VERIFICATION")
    print("================================================================================")
    test_1_create_cross_os_surface()
    test_2_benchmark_latency_vs_wslg()
    test_3_header_bridge_validation()
    test_4_openapi_spec_validation()
    print("================================================================================")
    print("  ALL 4 WSL2 GUI COMPOSITOR TESTS PASSED! (100% SUCCESS, VITAL_MAX_HP == 6)")
    print("================================================================================")

if __name__ == "__main__":
    main()
