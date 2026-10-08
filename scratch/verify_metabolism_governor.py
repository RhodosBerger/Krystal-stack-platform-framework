"""
Krystal-Stack Standalone Verification & Profiling: Tech Stack Metabolic Governor
================================================================================
Profiles evaluation latency, validates phase transitions (ALPHA, BETA, GAMMA, OMEGA),
verifies fidelity stepping down under thermodynamic pressure, and prints an ASCII HUD.
"""

import sys
import os
import time
import json

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from krystal_kernel.metabolic_governor import (
    VITAL_MAX_HP,
    BrainwavePhase,
    MetabolicGovernor,
    PHASE_PROFILES
)

def run_metabolic_governor_verification():
    print("=" * 80)
    print(" [KRYSTAL-STACK] METABOLIC GOVERNOR & BRAINWAVE PHASE PROFILING")
    print("=" * 80)

    governor = MetabolicGovernor(initial_phase=BrainwavePhase.BETA)

    scenarios = [
        ("IDLE_CONSOLIDATION", {"cpu": 0.10, "vram": 0.15, "entropy": 0.20, "temp": 39.0, "hp": 6}, BrainwavePhase.ALPHA),
        ("ACTIVE_FLOW_60FPS", {"cpu": 0.45, "vram": 0.50, "entropy": 0.45, "temp": 52.0, "hp": 6}, BrainwavePhase.BETA),
        ("HYPER_FOCUS_120FPS", {"cpu": 0.80, "vram": 0.85, "entropy": 0.65, "temp": 71.0, "hp": 5}, BrainwavePhase.GAMMA),
        ("THERMAL_SPIKE_THROTTLE", {"cpu": 0.95, "vram": 0.95, "entropy": 0.60, "temp": 88.5, "hp": 4}, BrainwavePhase.OMEGA),
        ("CRITICAL_ENTROPY_DAMP", {"cpu": 0.60, "vram": 0.60, "entropy": 0.82, "temp": 62.0, "hp": 3}, BrainwavePhase.OMEGA),
        ("RECOVERY_TO_FLOW", {"cpu": 0.40, "vram": 0.40, "entropy": 0.40, "temp": 50.0, "hp": 6}, BrainwavePhase.BETA),
    ]

    audit_log = {
        "timestamp": time.time(),
        "status": "PASS",
        "scenarios_tested": len(scenarios),
        "latencies_us": [],
        "transitions": []
    }

    for name, params, expected_phase in scenarios:
        t0 = time.perf_counter()
        state = governor.evaluate_telemetry(
            cpu_utilization=params["cpu"],
            vram_pressure=params["vram"],
            visual_entropy=params["entropy"],
            thermal_temp_c=params["temp"],
            vital_hp=params["hp"]
        )
        t_us = (time.perf_counter() - t0) * 1e6
        audit_log["latencies_us"].append(round(t_us, 2))

        assert state.active_phase == expected_phase, f"Scenario '{name}' expected {expected_phase}, got {state.active_phase}"
        assert state.vital_hp <= VITAL_MAX_HP, f"HP {state.vital_hp} exceeded VITAL_MAX_HP {VITAL_MAX_HP}"

        print(f"  [{state.fidelity.symbol}] Scenario: {name:<24} -> Phase: {state.active_phase.value:<6} | Target: {state.fidelity.target_fps:>3} FPS | Steps: {state.fidelity.raymarch_steps:>2} | Glyph: {state.fidelity.glyph_mode:<16} ({t_us:.2f} us)")
        audit_log["transitions"].append({
            "scenario": name,
            "phase": state.active_phase.value,
            "symbol": state.fidelity.symbol,
            "target_fps": state.fidelity.target_fps,
            "raymarch_steps": state.fidelity.raymarch_steps,
            "terrain_octaves": state.fidelity.terrain_octaves,
            "glyph_mode": state.fidelity.glyph_mode,
            "latency_us": round(t_us, 2)
        })

    avg_latency = sum(audit_log["latencies_us"]) / len(audit_log["latencies_us"])
    audit_log["avg_latency_us"] = round(avg_latency, 2)

    # Visual ASCII HUD Display
    print("\n" + "=" * 80)
    print(" [VISUAL TELEMETRY HUD] REAL-TIME METABOLIC STATUS")
    print("=" * 80)
    last_state = governor.history[-1]
    hud_lines = [
        f"┌────────────────────────────────────────────────────────────────────────┐",
        f"│ KRYSTAL COMPUTE KERNEL // BIO-CYBERNETIC METABOLIC HUD                 │",
        f"├────────────────────────────────────────────────────────────────────────┤",
        f"│ BRAINWAVE PHASE: [{last_state.fidelity.symbol}] {last_state.active_phase.value:<8}  │ TRANSITIONS: {last_state.transition_count:<4} │ VITAL HP: {last_state.vital_hp}/6 HP    │",
        f"│ CPU LOAD: [{'█' * int(last_state.cpu_utilization * 15):<15}] {last_state.cpu_utilization*100:>5.1f}% │ THERMAL TEMP: {last_state.thermal_temp_c:>5.1f}°C (Limit: 85°C)│",
        f"│ VRAM RES: [{'█' * int(last_state.vram_pressure * 15):<15}] {last_state.vram_pressure*100:>5.1f}% │ VISUAL ENTROPY: {last_state.visual_entropy:>4.2f} (Limit: 0.70)│",
        f"├────────────────────────────────────────────────────────────────────────┤",
        f"│ FIDELITY PROFILE: {last_state.fidelity.target_fps:>3} FPS │ RAYMARCH STEPS: {last_state.fidelity.raymarch_steps:>2} │ TERRAIN OCTAVES: {last_state.fidelity.terrain_octaves:>2} │",
        f"│ GLYPH RASTERIZER: {last_state.fidelity.glyph_mode:<16} │ STATUS: {'THROTTLED (BACKPRESSURE)' if last_state.is_throttled else 'OPTIMAL NOMINAL':<22}  │",
        f"└────────────────────────────────────────────────────────────────────────┘"
    ]
    hud_str = "\n".join(hud_lines)
    print(hud_str)
    audit_log["ascii_hud"] = hud_str

    # Write audit JSON to scratch
    scratch_dir = os.path.join(ROOT_DIR, "scratch")
    os.makedirs(scratch_dir, exist_ok=True)
    out_file = os.path.join(scratch_dir, "metabolism_governor_audit.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(audit_log, f, indent=2)

    print(f"\nAudit results successfully saved to: {out_file}")
    print(f"Average evaluation overhead: {avg_latency:.2f} microseconds (Target < 100 us)")
    print("=" * 80)
    return audit_log

if __name__ == "__main__":
    run_metabolic_governor_verification()
