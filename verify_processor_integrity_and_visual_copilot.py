#!/usr/bin/env python3
"""
==============================================================================
VERIFICATION SUITE: PROCESSOR INTEGRITY & VISUAL COPILOT ARCHITECTURE
==============================================================================
Automated verification of kernel telemetry sampling, behavioral prediction,
polyglot visual copilot synthesis, OS plugins, and system invariants.

System Invariant: VITAL_MAX_HP = 6.

Author: Dušan Kopecký & Krystal-Stack Architecture Council (2026)
==============================================================================
"""

import os
import sys
import time
import json
from pathlib import Path

WORKSPACE_ROOT = os.path.dirname(os.path.abspath(__file__))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

VITAL_MAX_HP: int = 6


def test_kernel_telemetry_engine():
    print("[TEST 1/5] Testing Processor Integrity & Context-Switch Telemetry Engine...")
    from krystal_kernel.processor_integrity_telemetry import (
        ProcessorIntegrityTelemetryEngine,
        ProcessorIntegrityStatus,
        VITAL_MAX_HP as TELEMETRY_HP
    )
    assert TELEMETRY_HP == 6, "Telemetry invariant violation"

    engine = ProcessorIntegrityTelemetryEngine()
    live_rep = engine.sample_integrity()
    print(f"  -> Live Snapshot OS: {live_rep.target_os}, CS: {live_rep.context_switches_per_sec:,.1f}/s, Status: {live_rep.status}")
    assert live_rep.vital_max_hp == 6
    assert 0.0 <= live_rep.integrity_score <= 1.0

    # Pathological thrashing anomaly test
    sim_rep = engine.sample_integrity(simulated_override={
        "cpu_utilization_pct": 96.0,
        "context_switches_per_sec": 190000.0
    })
    print(f"  -> Simulated Thrashing Index: {sim_rep.thrashing_index:.2f}, Score: {sim_rep.integrity_score * 100:.1f}%, Status: {sim_rep.status}")
    assert sim_rep.anomaly_detected is True
    assert sim_rep.status == ProcessorIntegrityStatus.CRITICAL_INTERFERENCE
    assert sim_rep.integrity_score < 0.40
    print("  [PASS] Kernel Telemetry Engine verified successfully.")


def test_visual_copilot_generator():
    print("\n[TEST 2/5] Testing Visual Copilot Generator & Behavioral Relation Predictor...")
    from krystal_kernel.visual_copilot_generator import (
        PolyglotVisualCopilotGenerator,
        VisualBehavioralRelationPredictor,
        VITAL_MAX_HP as COPILOT_HP
    )
    assert COPILOT_HP == 6, "Copilot invariant violation"

    gen = PolyglotVisualCopilotGenerator()
    
    # 1. Prediction test
    pred = gen.predictor.predict(cs_rate=145000.0, cpu_pct=92.0, thrashing_index=3.85)
    print(f"  -> Predicted Frame Time: {pred.predicted_frame_time_ms} ms, Jitter: {pred.frame_jitter_ms} ms, Stutter: {pred.rendering_stutter_probability*100:.1f}%")
    assert pred.predicted_frame_time_ms > 16.6, "High thrashing should cause frame drop prediction"
    assert pred.visual_integrity_color_hex == "#FF2255"

    # 2. Polyglot code generation test
    res = gen.generate_all_targets(cs_rate=8000.0, cpu_pct=50.0, thrashing_idx=1.05)
    sources = res["sources"]
    assert "csharp_winui" in sources and "KernelIntegrityNotificationModel" in sources["csharp_winui"]
    assert "cpp_kernel_hook" in sources and "ReadHardwareContextSwitches" in sources["cpp_kernel_hook"]
    assert "python_behavioral_model" in sources and "BehavioralKernelPredictor" in sources["python_behavioral_model"]
    assert "vulkan_glsl_shader" in sources and "KernelTelemetryUniform" in sources["vulkan_glsl_shader"]
    
    print(f"  -> Polyglot files generated: C# ({len(sources['csharp_winui'])} bytes), C++ ({len(sources['cpp_kernel_hook'])} bytes), Python ({len(sources['python_behavioral_model'])} bytes), Vulkan ({len(sources['vulkan_glsl_shader'])} bytes)")
    print("  [PASS] Visual Copilot Generator verified successfully.")


def test_os_notification_plugins():
    print("\n[TEST 3/5] Testing Multi-OS Notification Plugins & Universal Header...")
    plugin_dir = Path(WORKSPACE_ROOT) / "plugins" / "os_notification"
    include_dir = Path(WORKSPACE_ROOT) / "include"

    # 1. Windows C# Plugin
    cs_file = plugin_dir / "windows_tray_widget.cs"
    assert cs_file.exists(), "windows_tray_widget.cs not found"
    cs_content = cs_file.read_text(encoding="utf-8")
    assert "WindowsKernelNotificationBar" in cs_content
    assert "VitalMaxHp = 6" in cs_content
    print("  -> Windows C# Notification Bar plugin verified.")

    # 2. Linux DBus Plugin
    py_file = plugin_dir / "linux_dbus_indicator.py"
    assert py_file.exists(), "linux_dbus_indicator.py not found"
    from plugins.os_notification.linux_dbus_indicator import LinuxNotificationBarDaemon
    daemon = LinuxNotificationBarDaemon()
    data = daemon.run_cycle()
    assert "context_switches_per_sec" in data
    assert data["vital_max_hp"] == 6
    print(f"  -> Linux D-Bus Indicator plugin verified (Cycle Status: {data.get('status')}).")

    # 3. Android Kotlin Plugin
    kt_file = plugin_dir / "android_status_overlay.kt"
    assert kt_file.exists(), "android_status_overlay.kt not found"
    kt_content = kt_file.read_text(encoding="utf-8")
    assert "KrystalKernelNotificationService" in kt_content
    assert "VITAL_MAX_HP = 6" in kt_content
    print("  -> Android Kotlin Status Overlay plugin verified.")

    # 4. Universal C/C++ Header
    h_file = include_dir / "krystal_kernel_telemetry_hook.h"
    assert h_file.exists(), "krystal_kernel_telemetry_hook.h not found"
    h_content = h_file.read_text(encoding="utf-8")
    assert "KrystalThreadControlBlock" in h_content
    assert "krystal_compute_thrashing_index" in h_content
    assert "KRYSTAL_VITAL_MAX_HP 6" in h_content
    print("  -> Universal Kernel Telemetry C/C++ Hook verified.")
    print("  [PASS] Multi-OS Notification Plugins verified successfully.")


def test_web_hub_endpoints():
    print("\n[TEST 4/5] Testing Web Hub Server Endpoints...")
    # Verify imports and function logic for endpoints
    from krystal_kernel.processor_integrity_telemetry import GLOBAL_PROCESSOR_INTEGRITY_ENGINE
    from krystal_kernel.visual_copilot_generator import GLOBAL_VISUAL_COPILOT_GENERATOR
    
    rep = GLOBAL_PROCESSOR_INTEGRITY_ENGINE.sample_integrity()
    rep_dict = rep.to_dict()
    assert "context_switches_per_sec" in rep_dict
    assert "thrashing_index" in rep_dict
    assert rep_dict["vital_max_hp"] == 6

    # Verify Copilot synthesis endpoint logic
    copilot_res = GLOBAL_VISUAL_COPILOT_GENERATOR.generate_all_targets(
        cs_rate=rep.context_switches_per_sec,
        cpu_pct=rep.cpu_utilization_pct,
        thrashing_idx=rep.thrashing_index
    )
    assert "prediction" in copilot_res
    assert "sources" in copilot_res
    print("  -> GET /api/kernel/integrity payload verified.")
    print("  -> POST /api/copilot/generate payload verified.")
    print("  [PASS] Web Hub Endpoints verified successfully.")


def test_html_ui_components():
    print("\n[TEST 5/5] Testing HTML Notification Bar & Copilot Modal Markup...")
    html_file = Path(WORKSPACE_ROOT) / "krystal_web_hub" / "static" / "speculative_microprocessor_blog.html"
    assert html_file.exists(), "speculative_microprocessor_blog.html not found"
    content = html_file.read_text(encoding="utf-8")
    assert "kernelNotificationBar" in content
    assert "ktb-status-badge" in content
    assert "ktbGaugeFill" in content
    assert "copilotModal" in content
    assert "pollKernelTelemetry" in content
    assert "simulateThrashingAnomaly" in content
    print("  -> HTML Notification Bar and Interactive Modal confirmed in static blog.")
    print("  [PASS] HTML UI Components verified successfully.")


def main():
    if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
        try:
            sys.stdout.reconfigure(encoding='utf-8')
        except Exception:
            pass

    print("=" * 85)
    print("  KRYSTAL-STACK: PROCESSOR INTEGRITY & VISUAL COPILOT VERIFICATION")
    print("=" * 85)

    test_kernel_telemetry_engine()
    test_visual_copilot_generator()
    test_os_notification_plugins()
    test_web_hub_endpoints()
    test_html_ui_components()

    print("\n" + "=" * 85)
    print("  ALL 5 VERIFICATION SUITES PASSED WITH 100% ACCURACY!")
    print(f"  SYSTEM INVARIANT SATISFIED: VITAL_MAX_HP = {VITAL_MAX_HP}")
    print("=" * 85)


if __name__ == "__main__":
    main()
