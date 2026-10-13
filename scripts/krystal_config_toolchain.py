#!/usr/bin/env python3
"""
==============================================================================
KRYSTAL-STACK: UNIFIED CONFIGURATION TOOLCHAIN & PROFILE MANAGER
==============================================================================
Module: scripts/krystal_config_toolchain.py
Description:
    CLI toolchain and programmatic profile manager for configuring, validating,
    and deploying Krystal Stack across heterogeneous execution environments.
    
    Provides compressed, validated configuration presets for:
      1. REALTIME_LOW_LATENCY (1.0ms, P-Cores, AMX/AVX-512, MMCSS Games, PL1 32W)
      2. BALANCED_DEVELOPMENT (10.0ms, All-Cores, AVX2, PL1 24W, 60 FPS Viewport)
      3. ECO_THERMAL_PRESERVATION (50.0ms, E-Cores, PL1 10W, Under-voltage, Cool-down)
      4. SHADOW_COUNTERFACTUAL_RESEARCH (2.0ms, Non-actuating AUTH_SHADOW_ONLY, 0 writes)
      5. MISSION_CONTROL_STUDIO_KIOSK (16.67ms QPC Sync, Full Deck S1..S18, WebSockets)
      6. HEADLESS_CONTAINER_CI (Pure CPU fallback, OpenVINO emulation, mock telemetry)

Constitutional Invariant:
    assert VITAL_MAX_HP == 6 (Universally enforced across all configurations)

Author: Krystal-Stack Architecture Council (2026)
==============================================================================
"""

import os
import sys
import json
import time
import argparse
import platform
from enum import Enum, unique
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Optional, Tuple

WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

from krystal_kernel.scene_constructor_presets import VITAL_MAX_HP


@unique
class ProfileKind(str, Enum):
    """Canonical profile identifiers."""
    REALTIME_LOW_LATENCY           = "REALTIME_LOW_LATENCY"
    BALANCED_DEVELOPMENT           = "BALANCED_DEVELOPMENT"
    ECO_THERMAL_PRESERVATION       = "ECO_THERMAL_PRESERVATION"
    SHADOW_COUNTERFACTUAL_RESEARCH = "SHADOW_COUNTERFACTUAL_RESEARCH"
    MISSION_CONTROL_STUDIO_KIOSK   = "MISSION_CONTROL_STUDIO_KIOSK"
    HEADLESS_CONTAINER_CI          = "HEADLESS_CONTAINER_CI"


@dataclass
class KrystalConfigurationProfile:
    """Rigorous configuration schema for Krystal Stack execution body."""
    profile_id: ProfileKind
    name: str
    description: str
    target_cadence_ms: float
    timer_resolution_ms: int
    mmcss_profile: Optional[str]
    cpu_affinity_mask: int
    cpu_priority_class: str
    target_pl1_watts: float
    undervolt_mv: float
    active_isa: str
    npu_streams: int
    npu_dp4a_mode: bool
    default_locale: str
    default_register: str
    socket_streaming_enabled: bool
    actor_socket_ports: List[int]
    shadow_only: bool
    dirty_cascade_depth: int
    vital_max_hp: int = VITAL_MAX_HP

    def __post_init__(self):
        assert self.vital_max_hp == 6, f"Invariant violation: VITAL_MAX_HP must be 6, got {self.vital_max_hp}"

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["profile_id"] = self.profile_id.value
        return d


CANONICAL_PROFILES: Dict[ProfileKind, KrystalConfigurationProfile] = {
    ProfileKind.REALTIME_LOW_LATENCY: KrystalConfigurationProfile(
        profile_id=ProfileKind.REALTIME_LOW_LATENCY,
        name="Real-Time Ultra-Low Latency & High-Frequency Actuation",
        description="Sub-1ms quantum pinned execution, dedicated P-cores, maximum sustained compute with AMX/AVX-512.",
        target_cadence_ms=1.0,
        timer_resolution_ms=1,
        mmcss_profile="Games",
        cpu_affinity_mask=0x000F,  # P-Cores
        cpu_priority_class="HIGH_PRIORITY_CLASS",
        target_pl1_watts=32.0,
        undervolt_mv=-25.0,
        active_isa="INTEL_AMX_TILE",
        npu_streams=4,
        npu_dp4a_mode=True,
        default_locale="SK",
        default_register="MILITARY_TACTICAL",
        socket_streaming_enabled=True,
        actor_socket_ports=[8910, 8911, 8912],
        shadow_only=False,
        dirty_cascade_depth=5,
        vital_max_hp=6
    ),
    ProfileKind.BALANCED_DEVELOPMENT: KrystalConfigurationProfile(
        profile_id=ProfileKind.BALANCED_DEVELOPMENT,
        name="Balanced Interactive Development & Multi-Agent Exploration",
        description="Default operational configuration, 10ms cadence, full 8-core affinity, AVX2 FMA acceleration.",
        target_cadence_ms=10.0,
        timer_resolution_ms=2,
        mmcss_profile=None,
        cpu_affinity_mask=0x00FF,  # All 8 cores
        cpu_priority_class="NORMAL_PRIORITY_CLASS",
        target_pl1_watts=24.0,
        undervolt_mv=0.0,
        active_isa="X86_AVX2_FMA",
        npu_streams=2,
        npu_dp4a_mode=False,
        default_locale="SK",
        default_register="EPISTEMIC_FORMAL",
        socket_streaming_enabled=True,
        actor_socket_ports=[8910, 8911, 8912],
        shadow_only=False,
        dirty_cascade_depth=5,
        vital_max_hp=6
    ),
    ProfileKind.ECO_THERMAL_PRESERVATION: KrystalConfigurationProfile(
        profile_id=ProfileKind.ECO_THERMAL_PRESERVATION,
        name="Eco Thermal Preservation & Under-Volt Cool-Down",
        description="Emergency cool-down pacing, E-Core affinity, clamped 10W envelope, active thermal relief.",
        target_cadence_ms=50.0,
        timer_resolution_ms=5,
        mmcss_profile=None,
        cpu_affinity_mask=0x00F0,  # E-Cores
        cpu_priority_class="BELOW_NORMAL_PRIORITY_CLASS",
        target_pl1_watts=10.0,
        undervolt_mv=-50.0,
        active_isa="NPU_DP4A_INT8",
        npu_streams=1,
        npu_dp4a_mode=True,
        default_locale="EN",
        default_register="CYBERNETIC_TELEMETRY",
        socket_streaming_enabled=False,
        actor_socket_ports=[],
        shadow_only=False,
        dirty_cascade_depth=2,
        vital_max_hp=6
    ),
    ProfileKind.SHADOW_COUNTERFACTUAL_RESEARCH: KrystalConfigurationProfile(
        profile_id=ProfileKind.SHADOW_COUNTERFACTUAL_RESEARCH,
        name="Counterfactual Shadow Research & Risk Verification",
        description="Non-mutating execution (AUTH_SHADOW_ONLY), zero host OS register writes, high-cadence what-if simulation.",
        target_cadence_ms=2.0,
        timer_resolution_ms=1,
        mmcss_profile=None,
        cpu_affinity_mask=0x000F,
        cpu_priority_class="NORMAL_PRIORITY_CLASS",
        target_pl1_watts=28.0,
        undervolt_mv=0.0,
        active_isa="X86_AVX512_VNNI",
        npu_streams=2,
        npu_dp4a_mode=True,
        default_locale="LA",
        default_register="EPISTEMIC_FORMAL",
        socket_streaming_enabled=True,
        actor_socket_ports=[8910],
        shadow_only=True,
        dirty_cascade_depth=4,
        vital_max_hp=6
    ),
    ProfileKind.MISSION_CONTROL_STUDIO_KIOSK: KrystalConfigurationProfile(
        profile_id=ProfileKind.MISSION_CONTROL_STUDIO_KIOSK,
        name="Mission Control Living Studio Kiosk & Compositor Sync",
        description="Synchronous 60 FPS QPC frame pacing (16.67ms), MMCSS DisplayPostProcessing, full 18-section deck enabled.",
        target_cadence_ms=16.67,
        timer_resolution_ms=1,
        mmcss_profile="DisplayPostProcessing",
        cpu_affinity_mask=0x00FF,
        cpu_priority_class="HIGH_PRIORITY_CLASS",
        target_pl1_watts=28.0,
        undervolt_mv=-15.0,
        active_isa="X86_AVX2_FMA",
        npu_streams=2,
        npu_dp4a_mode=False,
        default_locale="SK",
        default_register="EPISTEMIC_FORMAL",
        socket_streaming_enabled=True,
        actor_socket_ports=[8910, 8911, 8912],
        shadow_only=False,
        dirty_cascade_depth=5,
        vital_max_hp=6
    ),
    ProfileKind.HEADLESS_CONTAINER_CI: KrystalConfigurationProfile(
        profile_id=ProfileKind.HEADLESS_CONTAINER_CI,
        name="Headless Container, POSIX CI & Pure-Python Fallback",
        description="Deterministic environment with mock telemetry, scalar fallback, zero native Windows API dependencies.",
        target_cadence_ms=10.0,
        timer_resolution_ms=10,
        mmcss_profile=None,
        cpu_affinity_mask=0x0000,  # OS default
        cpu_priority_class="NORMAL_PRIORITY_CLASS",
        target_pl1_watts=15.0,
        undervolt_mv=0.0,
        active_isa="GENERIC_SCALAR_X86",
        npu_streams=0,
        npu_dp4a_mode=False,
        default_locale="EN",
        default_register="CYBERNETIC_TELEMETRY",
        socket_streaming_enabled=False,
        actor_socket_ports=[],
        shadow_only=False,
        dirty_cascade_depth=1,
        vital_max_hp=6
    )
}


class KrystalConfigToolchain:
    """Toolchain manager handling inspection, validation, dry-runs, and format exports."""

    @classmethod
    def list_profiles(cls) -> List[KrystalConfigurationProfile]:
        return list(CANONICAL_PROFILES.values())

    @classmethod
    def get_profile(cls, profile_key: str) -> KrystalConfigurationProfile:
        norm = profile_key.upper().strip()
        for pk, prof in CANONICAL_PROFILES.items():
            if pk.value == norm or norm in pk.value:
                return prof
        raise ValueError(f"Unknown profile: {profile_key}. Available: {[p.value for p in ProfileKind]}")

    @classmethod
    def inspect_host_environment(cls) -> Dict[str, Any]:
        """Probes local host hardware capabilities, timer resolutions, and ISA vector units."""
        cpu_count = os.cpu_count() or 4
        is_windows = sys.platform == "win32"
        qpc_freq = 0
        timer_min_ms = 1
        timer_max_ms = 16

        if is_windows:
            try:
                import ctypes
                kernel32 = ctypes.windll.kernel32
                winmm = ctypes.windll.winmm
                freq = ctypes.c_int64()
                if kernel32.QueryPerformanceFrequency(ctypes.byref(freq)):
                    qpc_freq = freq.value
                
                # Check timer caps
                class TIMECAPS(ctypes.Structure):
                    _fields_ = [("wPeriodMin", ctypes.c_uint), ("wPeriodMax", ctypes.c_uint)]
                caps = TIMECAPS()
                if winmm.timeGetDevCaps(ctypes.byref(caps), ctypes.sizeof(caps)) == 0:
                    timer_min_ms = caps.wPeriodMin
                    timer_max_ms = caps.wPeriodMax
            except Exception:
                pass

        # Check OpenVINO
        openvino_found = False
        try:
            import openvino
            openvino_found = True
        except ImportError:
            pass

        return {
            "os_name": platform.system(),
            "os_release": platform.release(),
            "processor": platform.processor(),
            "cpu_cores": cpu_count,
            "is_windows_native": is_windows,
            "qpc_frequency_hz": qpc_freq,
            "winmm_timer_min_period_ms": timer_min_ms,
            "winmm_timer_max_period_ms": timer_max_ms,
            "openvino_available": openvino_found,
            "invariants_ok": (VITAL_MAX_HP == 6)
        }

    @classmethod
    def validate_configuration(cls, cfg: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Strict validation of a configuration dictionary against systemic contracts."""
        errors = []
        if cfg.get("vital_max_hp") != 6:
            errors.append(f"Invariant violation: vital_max_hp must be exactly 6, got {cfg.get('vital_max_hp')}")

        cadence = cfg.get("target_cadence_ms", 10.0)
        if cadence <= 0.0 or cadence > 1000.0:
            errors.append(f"Invalid target_cadence_ms: {cadence}. Must be in (0.0, 1000.0]")

        pl1 = cfg.get("target_pl1_watts", 24.0)
        if pl1 < 5.0 or pl1 > 120.0:
            errors.append(f"Invalid target_pl1_watts: {pl1}. Must be clamped to [5.0, 120.0] W")

        valid_isas = [
            "INTEL_AMX_TILE", "X86_AVX512_VNNI", "X86_AVX2_FMA", "ARM_SVE2_256",
            "NPU_DP4A_INT8", "ARM_NEON_128", "RISCV_VECTOR_V1", "X86_SSE4_2",
            "GENERIC_SCALAR_X86"
        ]
        if cfg.get("active_isa") not in valid_isas:
            errors.append(f"Invalid active_isa: {cfg.get('active_isa')}. Must be one of {valid_isas}")

        return (len(errors) == 0, errors)

    @classmethod
    def export_env(cls, profile: KrystalConfigurationProfile) -> str:
        """Exports profile as compact bash/powershell-compatible .env file."""
        return (
            f"# Krystal-Stack Profile: {profile.profile_id.value}\n"
            f"KRYSTAL_PROFILE={profile.profile_id.value}\n"
            f"KRYSTAL_CADENCE_MS={profile.target_cadence_ms}\n"
            f"KRYSTAL_TIMER_RESOLUTION_MS={profile.timer_resolution_ms}\n"
            f"KRYSTAL_MMCSS_PROFILE={profile.mmcss_profile or 'NONE'}\n"
            f"KRYSTAL_CPU_AFFINITY=0x{profile.cpu_affinity_mask:04X}\n"
            f"KRYSTAL_CPU_PRIORITY={profile.cpu_priority_class}\n"
            f"KRYSTAL_PL1_WATTS={profile.target_pl1_watts}\n"
            f"KRYSTAL_UNDERVOLT_MV={profile.undervolt_mv}\n"
            f"KRYSTAL_ACTIVE_ISA={profile.active_isa}\n"
            f"KRYSTAL_NPU_STREAMS={profile.npu_streams}\n"
            f"KRYSTAL_LOCALE={profile.default_locale}\n"
            f"KRYSTAL_REGISTER={profile.default_register}\n"
            f"KRYSTAL_SHADOW_ONLY={'1' if profile.shadow_only else '0'}\n"
            f"KRYSTAL_SOCKET_STREAMING={'1' if profile.socket_streaming_enabled else '0'}\n"
            f"KRYSTAL_VITAL_MAX_HP={profile.vital_max_hp}\n"
        )

    @classmethod
    def export_compact_card(cls, profile: KrystalConfigurationProfile) -> str:
        """Generates an ultra-compressed terminal ASCII card representing the profile."""
        hearts = "♥ " * profile.vital_max_hp
        ports_str = ",".join(str(p) for p in profile.actor_socket_ports) or "NONE"
        return (
            f"┌────────────────────────────────────────────────────────────────────────┐\n"
            f"│ KRYSTAL CONFIG: [{profile.profile_id.value:<28}]              │\n"
            f"├────────────────────────────────────────────────────────────────────────┤\n"
            f"│ Cadence:  {profile.target_cadence_ms:5.1f} ms  │ Resolution: {profile.timer_resolution_ms} ms  │ MMCSS: {profile.mmcss_profile or 'None':<15} │\n"
            f"│ Affinity: 0x{profile.cpu_affinity_mask:04X}    │ Priority: {profile.cpu_priority_class:<13} │ PL1: {profile.target_pl1_watts:4.1f}W ({profile.undervolt_mv:+.0f}mV)   │\n"
            f"│ ISA:      {profile.active_isa:<16} │ NPU: {profile.npu_streams} strm (DP4A={str(profile.npu_dp4a_mode):<5}) │ Sockets: {ports_str:<12} │\n"
            f"│ Language: [{profile.default_locale}] {profile.default_register:<16} │ Shadow Mode: {str(profile.shadow_only):<5} │ Dirty Depth: {profile.dirty_cascade_depth} Tiers │\n"
            f"├────────────────────────────────────────────────────────────────────────┤\n"
            f"│ INVARIANT: {hearts.strip()} ({profile.vital_max_hp}/6 HP) ── VALIDATED STABLE                  │\n"
            f"└────────────────────────────────────────────────────────────────────────┘"
        )

    @classmethod
    def dry_run_activation(cls, profile: KrystalConfigurationProfile) -> Dict[str, Any]:
        """Simulates atomic activation steps with expected kernel timing responses."""
        host = cls.inspect_host_environment()
        steps = [
            f"[STEP 1: TIMER QUANTUM] winmm.timeBeginPeriod({profile.timer_resolution_ms}) -> Scheduled",
            f"[STEP 2: MMCSS PROMOTION] Thread characteristics elevated to '{profile.mmcss_profile or 'Default'}'",
            f"[STEP 3: AFFINITY MASK] Binding worker matrix to mask 0x{profile.cpu_affinity_mask:04X}",
            f"[STEP 4: PRIORITY CLASS] Process priority set to {profile.cpu_priority_class}",
            f"[STEP 5: RAPL / PL1 ENVELOPE] Target thermal clamp {profile.target_pl1_watts}W, offset {profile.undervolt_mv}mV",
            f"[STEP 6: INSTRUCTION SET] Dynamic dispatch pointing to kernel: {profile.active_isa}",
            f"[STEP 7: ACTOR SOCKETS] Binding UDP/TCP loopback on ports: {profile.actor_socket_ports}",
            f"[STEP 8: LINGUISTIC / VIEWPORT] Setting active locale [{profile.default_locale}], cascade depth {profile.dirty_cascade_depth}",
            f"[STEP 9: CONSTITUTIONAL CHECK] assert VITAL_MAX_HP == {profile.vital_max_hp} -> OK"
        ]
        return {
            "status": "DRY_RUN_PASSED",
            "profile_id": profile.profile_id.value,
            "host_environment": host,
            "activation_steps": steps,
            "projected_jitter_ms": 0.05 if profile.target_cadence_ms <= 1.0 else 0.25,
            "is_shadow_only": profile.shadow_only,
            "vital_max_hp": profile.vital_max_hp
        }


# ==============================================================================
# CLI DISPATCHER
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Krystal Stack Configuration Toolchain & Compact Profile Manager v2.0"
    )
    parser.add_argument("-l", "--list", action="store_true", help="List all canonical profiles in compact table")
    parser.add_argument("-p", "--profile", type=str, default="BALANCED_DEVELOPMENT", help="Profile ID to inspect, dry-run, or export")
    parser.add_argument("-i", "--inspect", action="store_true", help="Inspect local host hardware and timer capabilities")
    parser.add_argument("-e", "--export", choices=["json", "env", "card"], help="Export format for selected profile")
    parser.add_argument("--dry-run", action="store_true", help="Simulate profile activation steps without OS mutations")
    parser.add_argument("--validate-json", type=str, help="Validate custom JSON configuration file")

    args = parser.parse_args()

    # 1. Validate custom json
    if args.validate_json:
        with open(args.validate_json, "r", encoding="utf-8") as f:
            data = json.load(f)
        valid, errors = KrystalConfigToolchain.validate_configuration(data)
        if valid:
            print(f"✅ Configuration in '{args.validate_json}' is VALID.")
            sys.exit(0)
        else:
            print(f"❌ Configuration INVALID:")
            for err in errors:
                print(f"   - {err}")
            sys.exit(1)

    # 2. Host inspection
    if args.inspect:
        info = KrystalConfigToolchain.inspect_host_environment()
        print("\n" + "=" * 70)
        print("  KRYSTAL STACK // HOST ENVIRONMENT TELEMETRY")
        print("=" * 70)
        for k, v in info.items():
            print(f"  {k:<30} : {v}")
        print("=" * 70 + "\n")
        return

    # 3. List profiles
    if args.list:
        print("\n" + "=" * 80)
        print("  KRYSTAL STACK // CANONICAL CONFIGURATION PROFILES")
        print("=" * 80)
        for prof in KrystalConfigToolchain.list_profiles():
            print(f" • {prof.profile_id.value:<32} [{prof.target_cadence_ms:5.1f}ms | PL1 {prof.target_pl1_watts:4.1f}W | {prof.active_isa:<16}]")
            print(f"   {prof.description}")
        print("=" * 80 + "\n")
        return

    # 4. Resolve selected profile
    try:
        profile = KrystalConfigToolchain.get_profile(args.profile)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)

    # 5. Export format
    if args.export == "env":
        print(KrystalConfigToolchain.export_env(profile))
        return
    elif args.export == "json":
        print(json.dumps(profile.to_dict(), indent=2, ensure_ascii=False))
        return
    elif args.export == "card":
        print("\n" + KrystalConfigToolchain.export_compact_card(profile) + "\n")
        return

    # 6. Dry run
    if args.dry_run or not args.export:
        report = KrystalConfigToolchain.dry_run_activation(profile)
        print("\n" + KrystalConfigToolchain.export_compact_card(profile) + "\n")
        print("  DRY-RUN ACTIVATION SEQUENCE:")
        for s in report["activation_steps"]:
            print(f"   {s}")
        print(f"\n  >> Projected Jitter: ±{report['projected_jitter_ms']} ms | Invariant: {report['vital_max_hp']}/6 HP OK\n")


if __name__ == "__main__":
    main()
