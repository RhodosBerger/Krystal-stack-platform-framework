#!/usr/bin/env python3
"""
==============================================================================
VERIFICATION SUITE: WSL2 HYPERVISOR GNOME PORT & CROSS-OS BRIDGE
==============================================================================
Tests:
  1. Hypervisor port parameters (Port 19283, AF_VSOCK, 0.38ms latency).
  2. Windows Registry query & GNOME .desktop entry generator.
  3. Overlay mode toggling & RAM reclamation calculation.
  4. Janet DSL layout definitions and WSLg speedup evaluation.
  5. Live REST API endpoints:
     - GET  /api/wsl/hypervisor_port
     - GET  /api/wsl/registry_inspect
     - POST /api/wsl/toggle_gnome_overlay
==============================================================================
"""

import sys
import json
import urllib.request
import urllib.error

if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from krystal_kernel.wsl_hypervisor_gnome_port import (
    GLOBAL_GNOME_OVERLAY_MANAGER,
    WindowsRegistryAccessPort,
    WslOverlayMode,
    VITAL_MAX_HP,
    KPHP_PORT
)


def test_1_hypervisor_port_telemetry():
    print("[TEST 1] Testing KPHP Hypervisor Port telemetry and latency...")
    status = GLOBAL_GNOME_OVERLAY_MANAGER.get_port_status()
    assert status.port_number == 19283
    assert "AF_VSOCK" in status.transport_protocol
    assert status.roundtrip_latency_ms <= 0.5, f"Expected <0.5ms latency, got {status.roundtrip_latency_ms}"
    assert status.vital_max_hp == 6
    assert status.active_win32_apps_in_gnome >= 5
    print(f"  -> PASSED: Port {status.port_number} active with {status.roundtrip_latency_ms} ms UMA latency.")


def test_2_registry_and_gnome_apps():
    print("[TEST 2] Testing Windows Registry inspection and GNOME .desktop mapping...")
    apps = WindowsRegistryAccessPort.get_installed_applications()
    assert len(apps) >= 5, "Expected at least 5 mapped applications"

    # Verify VS Code entry
    vscode = next((a for a in apps if a.app_id == "win32_code"), None)
    assert vscode is not None
    assert "Code.exe" in vscode.executable_path
    assert "[Desktop Entry]" in vscode.desktop_entry_content
    assert "X-Krystal-VITAL-MAX-HP=6" in vscode.desktop_entry_content

    # Verify Registry query
    reg_data = WindowsRegistryAccessPort.inspect_registry_key("HKLM\\SOFTWARE\\Microsoft\\Windows\\CurrentVersion")
    assert reg_data["exists"] is True
    assert reg_data["vital_max_hp"] == 6
    print(f"  -> PASSED: {len(apps)} Win32 apps converted to GNOME entries, registry access verified.")


def test_3_overlay_toggling_and_ram_reclamation():
    print("[TEST 3] Testing On-Demand Overlay Mode toggling...")
    mgr = GLOBAL_GNOME_OVERLAY_MANAGER

    # Switch to GNOME_PURE_SOVEREIGN
    pure_status = mgr.toggle_mode(WslOverlayMode.GNOME_PURE_SOVEREIGN)
    assert pure_status.active_mode == WslOverlayMode.GNOME_PURE_SOVEREIGN
    assert pure_status.explorer_process_state == "SUSPENDED"
    assert pure_status.reclaimed_ram_mb >= 3000

    # Switch to HYBRID_SEAMLESS_OVERLAY
    hybrid_status = mgr.toggle_mode(WslOverlayMode.HYBRID_SEAMLESS_OVERLAY)
    assert hybrid_status.active_mode == WslOverlayMode.HYBRID_SEAMLESS_OVERLAY
    assert hybrid_status.explorer_process_state == "RUNNING"

    # Switch to WINDOWS_CLASSIC_BYPASS
    classic_status = mgr.toggle_mode(WslOverlayMode.WINDOWS_CLASSIC_BYPASS)
    assert classic_status.active_mode == WslOverlayMode.WINDOWS_CLASSIC_BYPASS

    # Restore to Hybrid
    mgr.toggle_mode(WslOverlayMode.HYBRID_SEAMLESS_OVERLAY)
    print("  -> PASSED: All 3 overlay states transitioned seamlessly with verified RAM reclamation.")


def test_4_janet_dsl_validation():
    print("[TEST 4] Testing Janet WSL GNOME Hypervisor Bridge DSL...")
    from krystal_janet.janet_bridge import JanetValidator
    res = JanetValidator.validate_file("krystal_janet/wsl_gnome_hypervisor_bridge.janet")
    assert res["valid"] is True, f"Janet file invalid: {res}"
    assert "SUPPORTED-LAYOUTS" in res["definitions"]
    assert "calculate-hypervisor-ipc-speedup" in res["definitions"]
    print(f"  -> PASSED: Janet DSL validated with {len(res['definitions'])} exported primitives.")


def test_5_live_rest_api():
    print("[TEST 5] Testing Live REST API Endpoints on http://127.0.0.1:8080...")
    base_url = "http://127.0.0.1:8080"

    # 1. GET /api/wsl/hypervisor_port
    try:
        req = urllib.request.Request(f"{base_url}/api/wsl/hypervisor_port")
        with urllib.request.urlopen(req, timeout=5) as resp:
            assert resp.status == 200
            data = json.loads(resp.read().decode("utf-8"))
            assert data["port_number"] == 19283
            assert data["vital_max_hp"] == 6
            print("  -> GET /api/wsl/hypervisor_port: 200 OK")
    except Exception as e:
        print(f"  -> FAIL on GET /api/wsl/hypervisor_port: {e}")
        raise

    # 2. GET /api/wsl/registry_inspect
    try:
        req = urllib.request.Request(f"{base_url}/api/wsl/registry_inspect")
        with urllib.request.urlopen(req, timeout=5) as resp:
            assert resp.status == 200
            data = json.loads(resp.read().decode("utf-8"))
            assert data["status"] == "OK"
            assert len(data["installed_win32_apps"]) >= 5
            print("  -> GET /api/wsl/registry_inspect: 200 OK")
    except Exception as e:
        print(f"  -> FAIL on GET /api/wsl/registry_inspect: {e}")
        raise

    # 3. POST /api/wsl/toggle_gnome_overlay
    try:
        payload = json.dumps({"target_mode": "GNOME_PURE_SOVEREIGN"}).encode("utf-8")
        req = urllib.request.Request(
            f"{base_url}/api/wsl/toggle_gnome_overlay",
            data=payload,
            headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            assert resp.status == 200
            data = json.loads(resp.read().decode("utf-8"))
            assert data["active_mode"] == "GNOME_PURE_SOVEREIGN"
            assert data["reclaimed_ram_mb"] >= 3000
            print("  -> POST /api/wsl/toggle_gnome_overlay: 200 OK")
    except Exception as e:
        print(f"  -> FAIL on POST /api/wsl/toggle_gnome_overlay: {e}")
        raise

    print("  -> PASSED: All REST API endpoints operational.")


if __name__ == "__main__":
    print("================================================================================")
    print("  RUNNING VERIFICATION SUITE: WSL2 HYPERVISOR GNOME PORT & OVERLAY")
    print("================================================================================")
    test_1_hypervisor_port_telemetry()
    test_2_registry_and_gnome_apps()
    test_3_overlay_toggling_and_ram_reclamation()
    test_4_janet_dsl_validation()
    test_5_live_rest_api()
    print("================================================================================")
    print("  ALL 5 VERIFICATION TESTS PASSED SUCCESSFULLY! (VITAL_MAX_HP = 6)")
    print("================================================================================")
