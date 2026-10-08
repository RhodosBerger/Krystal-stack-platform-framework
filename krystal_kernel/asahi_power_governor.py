#!/usr/bin/env python3
"""
==============================================================================
KRYSTAL-STACK: ASAHI-INSPIRED POWER GOVERNOR & THERMAL FUSE CONTROLLER
==============================================================================
Module: krystal_kernel/asahi_power_governor.py
Description: Advanced power management system inspired by Asahi Linux's
             fine-grained Apple Silicon DVFS, energy-aware scheduling (EAS),
             and dynamic power domain balancing for Intel Willow Cove & Iris Xe.

Principles:
  1. No Aggressive Panic-Throttling: Does not forcefully over-cool the system
     at the expense of performance, but rather sustains maximum hardware
     acceleration as long as the thermal fuse is preserved.
  2. Multi-Domain Voltage Balancing: Balances V_core (CPU), V_gt (Iris Xe GPU),
     and V_uncore (Memory Bus & Ring).
  3. Thermal Fuse Watchdog: Hard safety trip at 100°C, regulation limit at 95°C.
  4. System Invariant: VITAL_MAX_HP = 6.

Author: Dušan Kopecký & Krystal-Stack Architecture Council (2026)
==============================================================================
"""

import os
import sys
import time
import math
from dataclasses import dataclass, asdict
from enum import Enum
from typing import Dict, Any, List, Optional, Tuple

WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

VITAL_MAX_HP: int = 6

# Thermal Constants (°C)
THERMAL_SAFE_TARGET_C: float = 78.0
THERMAL_FUSE_REGULATION_C: float = 95.0
THERMAL_HARD_TRIP_C: float = 100.0


class AsahiPState(str, Enum):
    P0_QUIESCENT = "P0_QUIESCENT"        # Idle / minimal power (0.70V, 800 MHz)
    P1_EFFICIENCY = "P1_EFFICIENCY"      # Energy-aware mode (0.80V, 1600 MHz)
    P2_BALANCED = "P2_BALANCED"          # Nominal load (0.92V, 2800 MHz)
    P3_BURST_ACCEL = "P3_BURST_ACCEL"    # Maximum acceleration (1.12V, 4200 MHz CPU, 1350 MHz Iris Xe)
    P4_THERMAL_GUARD = "P4_THERMAL_GUARD"# Damped state when near fuse limit (0.88V, stable floor)


@dataclass
class AsahiPowerProfile:
    p_state: AsahiPState
    voltage_core_v: float
    voltage_gpu_gt_v: float
    clock_cpu_mhz: int
    clock_gpu_mhz: int
    power_budget_watts: float
    burst_duration_max_sec: float
    description: str


ASAHI_PROFILES: Dict[AsahiPState, AsahiPowerProfile] = {
    AsahiPState.P0_QUIESCENT: AsahiPowerProfile(
        p_state=AsahiPState.P0_QUIESCENT,
        voltage_core_v=0.70,
        voltage_gpu_gt_v=0.68,
        clock_cpu_mhz=800,
        clock_gpu_mhz=300,
        power_budget_watts=5.0,
        burst_duration_max_sec=9999.0,
        description="Quiescent: Minimálna spotreba, vypnuté nepotrebné domény."
    ),
    AsahiPState.P1_EFFICIENCY: AsahiPowerProfile(
        p_state=AsahiPState.P1_EFFICIENCY,
        voltage_core_v=0.80,
        voltage_gpu_gt_v=0.78,
        clock_cpu_mhz=1600,
        clock_gpu_mhz=600,
        power_budget_watts=12.0,
        burst_duration_max_sec=9999.0,
        description="Efficiency: Optimálny pomer výkonu na watt."
    ),
    AsahiPState.P2_BALANCED: AsahiPowerProfile(
        p_state=AsahiPState.P2_BALANCED,
        voltage_core_v=0.92,
        voltage_gpu_gt_v=0.90,
        clock_cpu_mhz=2800,
        clock_gpu_mhz=1000,
        power_budget_watts=25.0,
        burst_duration_max_sec=300.0,
        description="Balanced: Štandardný výpočtový profil pre plynulý chod."
    ),
    AsahiPState.P3_BURST_ACCEL: AsahiPowerProfile(
        p_state=AsahiPState.P3_BURST_ACCEL,
        voltage_core_v=1.12,
        voltage_gpu_gt_v=1.10,
        clock_cpu_mhz=4200,
        clock_gpu_mhz=1350,
        power_budget_watts=45.0,
        burst_duration_max_sec=30.0,
        description="Burst Acceleration: Maximálny HW výkon pre garanciu VSync snímkovania."
    ),
    AsahiPState.P4_THERMAL_GUARD: AsahiPowerProfile(
        p_state=AsahiPState.P4_THERMAL_GUARD,
        voltage_core_v=0.88,
        voltage_gpu_gt_v=0.86,
        clock_cpu_mhz=2200,
        clock_gpu_mhz=850,
        power_budget_watts=20.0,
        burst_duration_max_sec=60.0,
        description="Thermal Guard: Adaptívna stabilizácia pred dosiahnutím tepelnej poistky."
    )
}


@dataclass
class PowerGovernorTelemetry:
    timestamp: float
    current_p_state: str
    voltage_core_v: float
    voltage_gpu_gt_v: float
    clock_cpu_mhz: int
    clock_gpu_mhz: int
    junction_temp_c: float
    thermal_headroom_c: float
    thermal_fuse_breached: bool
    hardware_acceleration_ratio: float  # [0.0, 1.0]
    estimated_power_watts: float
    governor_diagnosis: str
    vital_max_hp: int = VITAL_MAX_HP

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class AsahiInspiredPowerGovernor:
    """
    Coordinates CPU and Iris Xe GPU DVFS curves without aggressive throttling cliffs,
    safeguarding the thermal fuse while prioritizing peak hardware acceleration.
    """

    def __init__(self):
        assert VITAL_MAX_HP == 6, "Invariant VITAL_MAX_HP must remain 6"
        self.vital_hp = VITAL_MAX_HP
        self.current_state = AsahiPState.P2_BALANCED
        self.simulated_temp_c = 68.0
        self.burst_start_time = 0.0
        self.last_update_time = time.time()

    def regulate(
        self,
        demand_burst: bool = False,
        render_frame_time_ms: float = 8.33,
        vsync_budget_ms: float = 8.333,
        ambient_temp_c: float = 24.0,
        external_temp_override: Optional[float] = None
    ) -> PowerGovernorTelemetry:
        """
        Calculates next P-state, balances voltages, and regulates thermal headroom.
        """
        now = time.time()
        dt = max(0.05, now - self.last_update_time)
        self.last_update_time = now

        profile = ASAHI_PROFILES[self.current_state]

        # Update physical thermal model
        if external_temp_override is not None:
            self.simulated_temp_c = external_temp_override
        else:
            # Heat generated ~ Power, heat dissipated ~ (T - Ambient)
            pwr = profile.power_budget_watts
            heat_in = pwr * 0.14 * dt
            heat_out = (self.simulated_temp_c - ambient_temp_c) * 0.08 * dt
            self.simulated_temp_c = max(ambient_temp_c, min(105.0, self.simulated_temp_c + (heat_in - heat_out)))

        t_j = self.simulated_temp_c
        headroom = round(THERMAL_FUSE_REGULATION_C - t_j, 1)
        fuse_breached = (t_j >= THERMAL_HARD_TRIP_C)

        # Asahi Linux Regulation Logic:
        # 1. Thermal Safety Guard: If T >= 95°C, step down to P4_THERMAL_GUARD
        if t_j >= THERMAL_FUSE_REGULATION_C:
            self.current_state = AsahiPState.P4_THERMAL_GUARD
            diag = "Thermal Guard aktívny: Udržiava frekvenciu bez kolapsu pod VSync."
        # 2. Demand Burst for VSync: If frame render latency is near VSync limit and headroom > 10°C, grant BURST
        elif demand_burst or (render_frame_time_ms > (vsync_budget_ms * 0.88)):
            if headroom > 10.0:
                self.current_state = AsahiPState.P3_BURST_ACCEL
                diag = "Burst Acceleration: Udelené maximálne napätie a takt pre garanciu VSync."
            else:
                self.current_state = AsahiPState.P2_BALANCED
                diag = "Balanced: Zvýšený výkon limitovaný termálnou rezervou."
        # 3. Quiescent: If frame time is negligible (< 3.0ms) and no burst demanded
        elif render_frame_time_ms < 3.0 and headroom > 20.0:
            self.current_state = AsahiPState.P1_EFFICIENCY
            diag = "Efficiency: Úsporné napätie, dostatočné pre render."
        else:
            self.current_state = AsahiPState.P2_BALANCED
            diag = "Optimálne vybalansované napätie na čipe."

        active_profile = ASAHI_PROFILES[self.current_state]
        accel_ratio = round(active_profile.clock_gpu_mhz / 1350.0, 2)

        return PowerGovernorTelemetry(
            timestamp=now,
            current_p_state=self.current_state.value,
            voltage_core_v=active_profile.voltage_core_v,
            voltage_gpu_gt_v=active_profile.voltage_gpu_gt_v,
            clock_cpu_mhz=active_profile.clock_cpu_mhz,
            clock_gpu_mhz=active_profile.clock_gpu_mhz,
            junction_temp_c=round(t_j, 1),
            thermal_headroom_c=headroom,
            thermal_fuse_breached=fuse_breached,
            hardware_acceleration_ratio=accel_ratio,
            estimated_power_watts=active_profile.power_budget_watts,
            governor_diagnosis=diag,
            vital_max_hp=self.vital_hp
        )


GLOBAL_ASAHI_POWER_GOVERNOR = AsahiInspiredPowerGovernor()


def main():
    if sys.stdout.encoding.lower() != 'utf-8':
        sys.stdout.reconfigure(encoding='utf-8')
    print("=" * 85)
    print("  KRYSTAL-STACK: ASAHI-INSPIRED POWER GOVERNOR (INTEL WILLOW COVE & IRIS XE)")
    print("=" * 85)
    gov = GLOBAL_ASAHI_POWER_GOVERNOR
    
    # 1. Normal frame pacing
    t1 = gov.regulate(render_frame_time_ms=6.2, vsync_budget_ms=8.333)
    print(f"State: {t1.current_p_state} | V_core: {t1.voltage_core_v}V | V_gt: {t1.voltage_gpu_gt_v}V | Temp: {t1.junction_temp_c}°C (Headroom: {t1.thermal_headroom_c}°C)")
    print(f"Diagnosis: {t1.governor_diagnosis}")
    print("-" * 85)

    # 2. VSync under pressure -> Request Burst
    print(">> Simulating heavy raymarching frame latency (7.9ms / 8.33ms budget)...")
    t2 = gov.regulate(demand_burst=True, render_frame_time_ms=7.9, vsync_budget_ms=8.333)
    print(f"State: {t2.current_p_state} | V_core: {t2.voltage_core_v}V | GPU Clock: {t2.clock_gpu_mhz} MHz | Power: {t2.estimated_power_watts}W")
    print(f"HW Accel: {t2.hardware_acceleration_ratio*100:.0f}% | Diagnosis: {t2.governor_diagnosis}")
    print("=" * 85)


if __name__ == "__main__":
    main()
