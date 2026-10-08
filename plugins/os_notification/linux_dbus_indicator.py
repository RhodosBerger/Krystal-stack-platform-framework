#!/usr/bin/env python3
"""
==============================================================================
KRYSTAL-STACK: LINUX D-BUS NOTIFICATION BAR & DESKTOP INDICATOR
==============================================================================
Component: plugins/os_notification/linux_dbus_indicator.py
Description: Linux desktop status indicator & D-Bus notification daemon
             monitoring kernel context switches via /proc/stat.
Targets: Ubuntu, Debian, Fedora, Arch (GNOME, KDE Plasma, XFCE, Wayland)
System Invariant: VITAL_MAX_HP = 6

Author: Dušan Kopecký & Krystal-Stack Architecture Council (2026)
==============================================================================
"""

import os
import sys
import time
import json
import urllib.request
import subprocess
from typing import Dict, Any, Optional, Tuple

VITAL_MAX_HP: int = 6


class LinuxKernelTelemetryMonitor:
    """Monitors /proc/stat directly or queries Krystal Stack HTTP API."""

    def __init__(self, api_url: str = "http://localhost:8080/api/kernel/integrity"):
        assert VITAL_MAX_HP == 6, "Invariant VITAL_MAX_HP must remain 6"
        self.api_url = api_url
        self.last_ctxt = 0
        self.last_cpu_time = 0
        self.last_idle_time = 0
        self.last_timestamp = time.time()
        self.prime_proc_stat()

    def prime_proc_stat(self):
        """Reads initial snapshot from /proc/stat."""
        if os.path.exists("/proc/stat"):
            try:
                with open("/proc/stat", "r", encoding="utf-8") as f:
                    for line in f:
                        if line.startswith("ctxt "):
                            self.last_ctxt = int(line.split()[1])
                        elif line.startswith("cpu "):
                            parts = [int(p) for p in line.split()[1:]]
                            self.last_idle_time = parts[3]
                            self.last_cpu_time = sum(parts)
            except Exception:
                pass

    def sample_direct_proc_stat(self) -> Dict[str, Any]:
        """Calculates delta rates directly from /proc/stat."""
        now = time.time()
        dt = max(0.05, now - self.last_timestamp)
        curr_ctxt = self.last_ctxt
        curr_idle = self.last_idle_time
        curr_total = self.last_cpu_time

        if os.path.exists("/proc/stat"):
            try:
                with open("/proc/stat", "r", encoding="utf-8") as f:
                    for line in f:
                        if line.startswith("ctxt "):
                            curr_ctxt = int(line.split()[1])
                        elif line.startswith("cpu "):
                            parts = [int(p) for p in line.split()[1:]]
                            curr_idle = parts[3]
                            curr_total = sum(parts)
            except Exception:
                pass

        delta_ctxt = max(0, curr_ctxt - self.last_ctxt)
        delta_total = max(1, curr_total - self.last_cpu_time)
        delta_idle = max(0, curr_idle - self.last_idle_time)

        cs_rate = delta_ctxt / dt
        cpu_pct = 100.0 * (1.0 - (delta_idle / delta_total)) if delta_total > 0 else 0.0

        self.last_ctxt = curr_ctxt
        self.last_cpu_time = curr_total
        self.last_idle_time = curr_idle
        self.last_timestamp = now

        # Compute thrashing index
        expected_cs = max(1000.0, 6500.0 * (1.0 + (cpu_pct / 100.0) * 0.5))
        thrashing_idx = round((cs_rate / expected_cs) * (max(cpu_pct, 10.0) / 50.0), 2)

        if thrashing_idx <= 1.2:
            status = "OPTIMAL"
            score = 1.0
            diag = "Plánovač Linux kernelu (CFS/EEVDF) beží optimálne."
        elif thrashing_idx <= 2.2:
            status = "NOMINAL"
            score = 0.85
            diag = "Bežné prepínanie vlákien."
        elif thrashing_idx <= 3.5:
            status = "THRASHING_WARNING"
            score = 0.55
            diag = "Varovanie: Detekované nadmerné prepínanie kontextu!"
        else:
            status = "CRITICAL_INTERFERENCE"
            score = 0.20
            diag = "Kritická anomália: Patologický thread thrashing storm!"

        return {
            "context_switches_per_sec": round(cs_rate, 1),
            "cpu_utilization_pct": round(cpu_pct, 1),
            "thrashing_index": thrashing_idx,
            "integrity_score": score,
            "status": status,
            "behavioral_diagnosis": diag,
            "vital_max_hp": VITAL_MAX_HP
        }

    def fetch_telemetry(self) -> Dict[str, Any]:
        """Attempts to poll HTTP hub, falls back to direct /proc/stat."""
        try:
            req = urllib.request.Request(self.api_url, headers={"User-Agent": "KrystalLinuxIndicator/1.0"})
            with urllib.request.urlopen(req, timeout=0.8) as resp:
                if resp.status == 200:
                    return json.loads(resp.read().decode("utf-8"))
        except Exception:
            pass
        return self.sample_direct_proc_stat()


class LinuxNotificationBarDaemon:
    """Manages desktop notifications and notification bar status."""

    def __init__(self):
        self.monitor = LinuxKernelTelemetryMonitor()
        self.last_notification_time = 0
        self.alert_cooldown_sec = 10.0

    def send_desktop_notification(self, title: str, message: str, urgency: str = "normal"):
        """Dispatches notification via notify-send or D-Bus."""
        try:
            subprocess.run(
                ["notify-send", "-u", urgency, "-a", "Krystal Kernel Guard", title, message],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
        except Exception:
            pass

    def run_cycle(self) -> Dict[str, Any]:
        data = self.monitor.fetch_telemetry()
        status = data.get("status", "OPTIMAL")
        thrash_idx = data.get("thrashing_index", 1.0)
        cs_rate = data.get("context_switches_per_sec", 0.0)

        # Trigger desktop alerts on warning or critical
        now = time.time()
        if thrash_idx > 2.2 and (now - self.last_notification_time) > self.alert_cooldown_sec:
            urgency = "critical" if thrash_idx > 3.5 else "normal"
            title = f"⚠️ Krystal Kernel Alert [{status}]"
            body = f"Context Switches: {cs_rate:,.0f}/s (Thrashing Index: {thrash_idx:.2f})\n{data.get('behavioral_diagnosis')}"
            self.send_desktop_notification(title, body, urgency=urgency)
            self.last_notification_time = now

        return data


def main():
    if sys.stdout.encoding.lower() != 'utf-8':
        sys.stdout.reconfigure(encoding='utf-8')
    print("=" * 80)
    print("  KRYSTAL-STACK: LINUX NOTIFICATION BAR & DESKTOP INDICATOR")
    print("=" * 80)
    daemon = LinuxNotificationBarDaemon()
    data = daemon.run_cycle()
    print(f"Status:                      {data.get('status')}")
    print(f"CS Rate / sec:               {data.get('context_switches_per_sec'):,.1f}")
    print(f"CPU Utilization:             {data.get('cpu_utilization_pct')}%")
    print(f"Thrashing Index:             {data.get('thrashing_index')}")
    print(f"Integrity Score:             {data.get('integrity_score') * 100:.1f}%")
    print(f"Diagnosis:                   {data.get('behavioral_diagnosis')}")
    print("=" * 80)


if __name__ == "__main__":
    main()
