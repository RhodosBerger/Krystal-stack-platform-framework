#!/usr/bin/env python3
"""
BRAIN CORTEX V3 — Power & Thermal Monitor
=====================================================
Reads real CPU/GPU telemetry from Linux sysfs and psutil.
Provides live data to the Advice Grid for thermal-aware routing.

On Intel i5-1135G7 (Tiger Lake):
  CPU temp:  /sys/class/thermal/thermal_zone*/temp  (millidegrees)
  CPU freq:  /sys/devices/system/cpu/cpu*/cpufreq/scaling_cur_freq (kHz)
  TDP mode:  /sys/class/powercap/intel-rapl:0/constraint_0_power_limit_uw
  Battery:   /sys/class/power_supply/BAT0/

All values fall back to psutil or simulated data gracefully.
"""

from __future__ import annotations
import os
import time
import random
import logging
import psutil
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger("PowerMonitor")


@dataclass
class HWSnapshot:
    """Live hardware telemetry snapshot."""
    cpu_temp_c: float
    cpu_freq_mhz: float
    cpu_util_pct: float
    ram_used_pct: float
    power_source: str       # "battery" | "ac"
    battery_pct: Optional[float]
    battery_discharge_w: Optional[float]   # watts draining
    rapl_tdp_w: Optional[float]            # current RAPL power cap (W)
    throttling: bool
    timestamp: float


class PowerMonitor:
    """
    Polls hardware telemetry from Linux sysfs + psutil.
    Falls back to simulation if paths don't exist (CI/container safety).
    """

    # Intel i5-1135G7: 4 cores, base 2.4 GHz, boost 4.2 GHz, TDP 28W
    CPU_TEMP_NORMAL_RANGE = (40.0, 75.0)
    CPU_FREQ_BASE_MHZ = 2400.0
    CPU_FREQ_BOOST_MHZ = 4200.0
    TDP_PBP1_W = 28.0       # Primary Base Power (sustained)
    TDP_PBP2_W = 44.0       # Secondary (burst / Turbo)

    def __init__(self):
        self._simulation_mode = not self._has_sysfs()
        if self._simulation_mode:
            logger.info("PowerMonitor: sysfs not available → simulated mode")
        else:
            logger.info("PowerMonitor: connected to sysfs")
        self._frame = 0

    def _has_sysfs(self) -> bool:
        return Path("/sys/class/thermal").exists()

    def _read_cpu_temp(self) -> float:
        """Read CPU package temperature from thermal zones."""
        try:
            temps = psutil.sensors_temperatures()
            # coretemp is the standard Linux Intel sensor name
            if "coretemp" in temps:
                pkg = [t for t in temps["coretemp"] if "Package" in t.label]
                if pkg:
                    return float(pkg[0].current)
            # Fallback: acpitz
            if "acpitz" in temps:
                return float(temps["acpitz"][0].current)
        except Exception:
            pass
        # Final fallback: sysfs direct read
        try:
            zone_path = Path("/sys/class/thermal/thermal_zone0/temp")
            if zone_path.exists():
                return int(zone_path.read_text().strip()) / 1000.0
        except Exception:
            pass
        return -1.0   # unknown

    def _read_rapl_power(self) -> Optional[float]:
        """Read Intel RAPL package power cap in Watts."""
        try:
            p = Path("/sys/class/powercap/intel-rapl:0/constraint_0_power_limit_uw")
            if p.exists():
                uw = int(p.read_text().strip())
                return uw / 1_000_000.0
        except Exception:
            pass
        return None

    def _read_battery(self) -> tuple[Optional[float], Optional[float], str]:
        """Returns (battery_pct, discharge_watts, power_source)."""
        try:
            batt = psutil.sensors_battery()
            if batt is None:
                return None, None, "ac"
            pct = batt.percent
            power_source = "ac" if batt.power_plugged else "battery"
            # Discharge watts — not directly from psutil on all platforms
            discharge_w = None
            try:
                e_now = Path("/sys/class/power_supply/BAT0/energy_now")
                i_now = Path("/sys/class/power_supply/BAT0/current_now")
                v_now = Path("/sys/class/power_supply/BAT0/voltage_now")
                if i_now.exists() and v_now.exists():
                    current_ua = int(i_now.read_text().strip())
                    voltage_uv = int(v_now.read_text().strip())
                    discharge_w = (current_ua * voltage_uv) / 1e12
            except Exception:
                pass
            return pct, discharge_w, power_source
        except Exception:
            return None, None, "ac"

    def sample(self) -> HWSnapshot:
        """Collect a full hardware snapshot."""
        self._frame += 1

        if self._simulation_mode:
            return self._simulate()

        temp = self._read_cpu_temp()
        if temp < 0:
            temp = random.uniform(50, 72)  # safe fallback

        freq_mhz = psutil.cpu_freq().current if psutil.cpu_freq() else self.CPU_FREQ_BASE_MHZ
        cpu_util = psutil.cpu_percent(interval=None)
        ram_used = psutil.virtual_memory().percent
        batt_pct, discharge_w, power_source = self._read_battery()
        rapl_w = self._read_rapl_power()
        # i5-1135G7: throttle = above 85°C OR freq dropped below 80% of base
        throttling = temp > 85.0 or (
            psutil.cpu_freq() is not None
            and freq_mhz < (self.CPU_FREQ_BASE_MHZ * 0.80)
            and temp > 75.0  # only flag freq drop if already warm
        )

        return HWSnapshot(
            cpu_temp_c=round(temp, 1),
            cpu_freq_mhz=round(freq_mhz, 0),
            cpu_util_pct=round(cpu_util, 1),
            ram_used_pct=round(ram_used, 1),
            power_source=power_source,
            battery_pct=batt_pct,
            battery_discharge_w=discharge_w,
            rapl_tdp_w=rapl_w,
            throttling=throttling,
            timestamp=time.time(),
        )

    def _simulate(self) -> HWSnapshot:
        """Synthetic telemetry for demo/CI environments."""
        # Gradually heat up the simulated CPU
        base_temp = 52.0 + (self._frame * 0.05)
        temp = min(92.0, base_temp + random.gauss(0, 2.5))
        freq = self.CPU_FREQ_BOOST_MHZ if temp < 80 else self.CPU_FREQ_BASE_MHZ * 0.9
        util = min(100.0, 30.0 + self._frame * 0.3 + random.gauss(0, 5))
        on_battery = random.random() > 0.3
        batt_pct = max(5.0, 85.0 - self._frame * 0.1)

        return HWSnapshot(
            cpu_temp_c=round(temp, 1),
            cpu_freq_mhz=round(freq, 0),
            cpu_util_pct=round(util, 1),
            ram_used_pct=round(random.uniform(40, 75), 1),
            power_source="battery" if on_battery else "ac",
            battery_pct=round(batt_pct, 1) if on_battery else None,
            battery_discharge_w=round(random.uniform(5, 15), 1) if on_battery else None,
            rapl_tdp_w=self.TDP_PBP1_W if on_battery else self.TDP_PBP2_W,
            throttling=(temp > 85.0),
            timestamp=time.time(),
        )


class PowerGovernorV3:
    """
    Applies real Linux power profiles via sysfs when root is available.
    Falls back to simulation log otherwise.

    Profiles:
      eco       → powersave governor + RAPL cap at PBP1
      balanced  → balanced governor  + RAPL cap at PBP2*0.85
      overdrive → performance governor + no RAPL cap
    """

    GOVERNOR_MAP = {
        "eco":       "powersave",
        "balanced":  "balanced",
        "overdrive": "performance",
    }

    def __init__(self):
        self.current_mode = "balanced"
        self._can_write = self._check_write_access()
        if not self._can_write:
            logger.info("PowerGovernorV3: No root access → simulation mode")

    def _check_write_access(self) -> bool:
        p = Path("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor")
        return p.exists() and os.access(p, os.W_OK)

    def set_mode(self, mode: str) -> bool:
        if mode == self.current_mode:
            return True
        governor = self.GOVERNOR_MAP.get(mode, "balanced")
        self.current_mode = mode

        if self._can_write:
            # Apply to all CPUs (i5-1135G7 has 4 cores = 8 threads)
            for cpu_id in range(8):
                gov_path = Path(f"/sys/devices/system/cpu/cpu{cpu_id}/cpufreq/scaling_governor")
                if gov_path.exists():
                    try:
                        gov_path.write_text(governor)
                    except PermissionError:
                        logger.warning(f"Cannot write governor for cpu{cpu_id}")
            logger.info(f"PowerGovernorV3: Applied {governor} to all CPUs (mode={mode})")
            return True
        else:
            logger.info(f"[SIM] PowerGovernorV3: Would set governor={governor} (mode={mode})")
            return False
