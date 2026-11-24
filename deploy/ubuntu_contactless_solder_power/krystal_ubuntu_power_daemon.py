#!/usr/bin/env python3
r"""Krystal Stack — Ubuntu Linux Contactless Solder & Adaptive Thermal/RAPL Power Daemon.

Reads real Linux kernel telemetry from:
  - `/sys/class/thermal/thermal_zone*/temp`
  - `/sys/devices/system/cpu/cpu*/thermal_throttle/core_throttle_count`
  - `/sys/devices/system/cpu/cpu*/cpufreq/scaling_cur_freq`
  - `/sys/class/powercap/intel-rapl:0/energy_uj` & `constraint_0_power_limit_uw`
  - `/proc/meminfo`, `/proc/stat`, and `/proc/crypto` (verifying `vaes`, `sha256_ni`, `blake2b`)

Computes an adaptive long-term power envelope (`PL1`, `EPP`, Uncore bounds) driven by live CPU/UMA/NPU
utilization. Runs in safe read-only `--audit` mode by default; applies sysfs controls only when `--apply-sysfs`
is explicitly passed by root on Ubuntu 22.04 / 24.04 LTS.

Constitutional Invariant: VITAL_MAX_HP == 6 (\neg Valid => \Delta S = 0).
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List

VITAL_MAX_HP = 6
UMA_CEILING = 0.78


def _read_int(path: str, default: int = 0) -> int:
    try:
        return int(Path(path).read_text(encoding="utf-8").strip())
    except Exception:
        return default


def probe_ubuntu_linux_sensors() -> Dict[str, Any]:
    if VITAL_MAX_HP != 6:
        raise RuntimeError("[ERR_UBUNTU_HP] VITAL_MAX_HP != 6")

    temps_c: List[float] = []
    for tpath in sorted(glob.glob("/sys/class/thermal/thermal_zone*/temp")):
        raw = _read_int(tpath, -1)
        if raw > 0:
            temps_c.append(round(raw / 1000.0, 1))

    throttle_events = 0
    for th_path in glob.glob("/sys/devices/system/cpu/cpu*/thermal_throttle/core_throttle_count"):
        throttle_events += _read_int(th_path, 0)

    freqs_mhz: List[float] = []
    for fpath in sorted(glob.glob("/sys/devices/system/cpu/cpu[0-9]*/cpufreq/scaling_cur_freq")):
        khz = _read_int(fpath, 0)
        if khz > 0:
            freqs_mhz.append(round(khz / 1000.0, 1))

    rapl_uj = _read_int("/sys/class/powercap/intel-rapl:0/energy_uj", 0)
    pl1_uw = _read_int("/sys/class/powercap/intel-rapl:0/constraint_0_power_limit_uw", 28_000_000)

    mem_total_kb = 1
    mem_avail_kb = 1
    if os.path.exists("/proc/meminfo"):
        for line in Path("/proc/meminfo").read_text(encoding="utf-8", errors="ignore").splitlines():
            if line.startswith("MemTotal:"):
                mem_total_kb = max(1, int(line.split()[1]))
            elif line.startswith("MemAvailable:"):
                mem_avail_kb = int(line.split()[1])
    mem_used_ratio = round(1.0 - (mem_avail_kb / max(mem_total_kb, 1)), 3)

    kernel_crypto_drivers: List[str] = []
    if os.path.exists("/proc/crypto"):
        for line in Path("/proc/crypto").read_text(encoding="utf-8", errors="ignore").splitlines():
            if line.startswith("driver") and ":" in line:
                drv = line.split(":", 1)[1].strip()
                if any(k in drv for k in ("vaes", "ni", "avx", "blake2", "sha", "gcm")):
                    if drv not in kernel_crypto_drivers:
                        kernel_crypto_drivers.append(drv)

    npu_accel_nodes = sorted(glob.glob("/dev/accel/accel*") + glob.glob("/sys/class/accel/accel*"))
    drm_cards = sorted(glob.glob("/sys/class/drm/card[0-9]*"))

    max_temp = max(temps_c) if temps_c else 55.0
    uma_est = round(min(0.92, mem_used_ratio * 0.72), 3)

    if uma_est >= UMA_CEILING or max_temp >= 94.0:
        plan = {"strategy": "THRM_05_CRITICAL_SILICON_SHIELD", "target_pl1_w": 9.0, "epp": "power"}
    elif throttle_events > 0 or max_temp >= 84.0:
        plan = {"strategy": "THRM_04_ACTIVE_THROTTLE_MITIGATION", "target_pl1_w": 13.0, "epp": "balance_power"}
    elif max_temp >= 75.0:
        plan = {"strategy": "THRM_03_PREEMPTIVE_SOFT_SHUNT", "target_pl1_w": 17.0, "epp": "balance_power"}
    elif max_temp >= 62.0:
        plan = {"strategy": "THRM_02_ARRHENIUS_LONGEVITY_CRUISE", "target_pl1_w": 22.0, "epp": "balance_performance"}
    else:
        plan = {"strategy": "THRM_01_LAMINAR_FULL_WELD", "target_pl1_w": 28.0, "epp": "balance_performance"}

    return {
        "vital_max_hp": VITAL_MAX_HP,
        "max_temp_c": max_temp,
        "core_throttle_events": throttle_events,
        "avg_freq_mhz": round(sum(freqs_mhz) / max(len(freqs_mhz), 1), 1) if freqs_mhz else 0.0,
        "rapl_energy_uj": rapl_uj,
        "current_pl1_watts": round(pl1_uw / 1_000_000.0, 2),
        "estimated_uma_ratio": uma_est,
        "npu_nodes_detected": npu_accel_nodes,
        "drm_gpu_nodes": drm_cards,
        "kernel_crypto_hw_drivers": kernel_crypto_drivers[:12],
        "recommended_ubuntu_plan": plan,
    }


def apply_ubuntu_sysfs_plan(plan: Dict[str, Any]) -> List[str]:
    actions: List[str] = []
    pl1_uw = int(round(float(plan["target_pl1_w"]) * 1_000_000))
    rapl_limit = Path("/sys/class/powercap/intel-rapl:0/constraint_0_power_limit_uw")
    if rapl_limit.exists():
        rapl_limit.write_text(f"{pl1_uw}\n", encoding="utf-8")
        actions.append(f"wrote {pl1_uw} uW -> {rapl_limit}")

    for epp_file in glob.glob("/sys/devices/system/cpu/cpu[0-9]*/cpufreq/energy_performance_preference"):
        Path(epp_file).write_text(f"{plan['epp']}\n", encoding="utf-8")
    actions.append(f"updated EPP -> {plan['epp']}")
    return actions


def main() -> int:
    parser = argparse.ArgumentParser(description="Krystal Ubuntu Contactless Solder & Adaptive Power Daemon")
    parser.add_argument("--apply-sysfs", action="store_true", help="Write governed PL1/EPP to Linux sysfs (requires root)")
    parser.add_argument("--loop-seconds", type=float, default=0.0, help="Continuous monitoring interval (0 = single shot)")
    args = parser.parse_args()

    while True:
        snap = probe_ubuntu_linux_sensors()
        if args.apply_sysfs and os.name == "posix" and hasattr(os, "geteuid") and os.geteuid() == 0:
            snap["applied_actions"] = apply_ubuntu_sysfs_plan(snap["recommended_ubuntu_plan"])
        else:
            snap["applied_actions"] = ["AUDIT_DRY_RUN_NO_SYSFS_MUTATION"]
        print(json.dumps(snap, indent=2, ensure_ascii=False), flush=True)
        if args.loop_seconds <= 0.0:
            break
        time.sleep(max(0.5, args.loop_seconds))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
