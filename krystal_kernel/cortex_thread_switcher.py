#!/usr/bin/env python3
"""
==============================================================================
KRYSTAL-STACK: CORTEX THREAD SWITCHER & PROCESSOR AFFINITY ORCHESTRATOR
==============================================================================
Module: krystal_kernel/cortex_thread_switcher.py
Description: Dynamic CPU thread and core affinity arbitrator connecting
             Windows NT kernel APIs with the Cortex Janet Deductive Scheduler.

Capabilities:
  1. Windows NT / Linux CPU Affinity Switching:
     - Sets process and thread affinity masks (kernel32.SetProcessAffinityMask).
  2. Hybrid Core Strategy Evaluation:
     - P-Core (Performance) vs E-Core (Efficiency) vs UMA Coprocessor.
     - Strategies:
       * STRATEGY_ISOLATED_P_CORE_BURST
       * STRATEGY_COOPERATIVE_E_CORE_THROTTLE
       * STRATEGY_HYBRID_SYMMETRIC
       * STRATEGY_CRITICAL_HEALTH_DEFENSE
  3. Real-Time Telemetry Correlation:
     - Samples live context-switch and thrashing telemetry from
       ProcessorIntegrityTelemetryEngine to dictate optimal core routing.
  4. System Invariant: VITAL_MAX_HP = 6.

Author: Dušan Kopecký & Krystal-Stack Architecture Council (2026)
==============================================================================
"""

import os
import sys
import time
import ctypes
from dataclasses import dataclass, asdict
from typing import Dict, Any, List, Optional

WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

from krystal_kernel.processor_integrity_telemetry import (
    GLOBAL_PROCESSOR_INTEGRITY_ENGINE,
    VITAL_MAX_HP
)

# Core Affinity Strategy Enums
STRATEGIES = {
    "STRATEGY_ISOLATED_P_CORE_BURST": {
        "id": "STRATEGY_ISOLATED_P_CORE_BURST",
        "name": "P-Core Isolated Burst",
        "affinity_mask": 0x000F,  # Cores 0-3
        "target_lane": "P-CORES",
        "target_freq_mode": "MAX_TURBO",
        "description": "Pins rendering and tensor synthesis to high-frequency P-Cores; blocks migration interrupts."
    },
    "STRATEGY_COOPERATIVE_E_CORE_THROTTLE": {
        "id": "STRATEGY_COOPERATIVE_E_CORE_THROTTLE",
        "name": "E-Core Cooperative Offload",
        "affinity_mask": 0x00F0,  # Cores 4-7
        "target_lane": "E-CORES",
        "target_freq_mode": "STABILIZED_ECO",
        "description": "Diverts background telemetry, economic governor, and arena checks to E-Cores."
    },
    "STRATEGY_HYBRID_SYMMETRIC": {
        "id": "STRATEGY_HYBRID_SYMMETRIC",
        "name": "Hybrid Symmetric Homeostasis",
        "affinity_mask": 0x01FF,  # All 9 execution units
        "target_lane": "ALL-CORES",
        "target_freq_mode": "DYNAMIC",
        "description": "Balanced thread distribution across all physical and logical execution units."
    },
    "STRATEGY_CRITICAL_HEALTH_DEFENSE": {
        "id": "STRATEGY_CRITICAL_HEALTH_DEFENSE",
        "name": "Critical Health Isolation",
        "affinity_mask": 0x0001,  # Core 0 only
        "target_lane": "ISOLATED-CORE-0",
        "target_freq_mode": "THROTTLED",
        "description": "Emergency lockdown when thrashing storm or memory corruption is detected."
    }
}


@dataclass
class CoreSwitchingDecision:
    timestamp: float
    current_strategy: str
    affinity_mask_hex: str
    affinity_mask_int: int
    target_lane: str
    target_freq_mode: str
    context_switches_per_sec: float
    thrashing_index: float
    cpu_utilization_pct: float
    applied_to_os: bool
    vital_max_hp: int = VITAL_MAX_HP

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class CortexThreadSwitcher:
    """
    Arbitrates and applies OS-level CPU affinity masks based on real-time
    kernel telemetry and deductive Cortex rules.
    """

    def __init__(self):
        self.is_windows = (sys.platform == "win32")
        self.active_strategy_id = "STRATEGY_HYBRID_SYMMETRIC"
        self.last_switch_time = 0.0
        self.total_switches = 0
        self.min_switch_interval_sec = 0.5  # Prevents rapid flapping

    def _formulate_decision(self, chosen: Dict[str, Any], applied: bool) -> CoreSwitchingDecision:
        report = GLOBAL_PROCESSOR_INTEGRITY_ENGINE.sample_integrity()
        now = time.time()
        self.active_strategy_id = chosen["id"]
        return CoreSwitchingDecision(
            timestamp=now,
            current_strategy=chosen["id"],
            affinity_mask_hex=f"0x{chosen['affinity_mask']:04X}",
            affinity_mask_int=chosen["affinity_mask"],
            target_lane=chosen["target_lane"],
            target_freq_mode=chosen["target_freq_mode"],
            context_switches_per_sec=round(report.context_switches_per_sec, 1),
            thrashing_index=round(report.thrashing_index, 3),
            cpu_utilization_pct=round(report.cpu_utilization_pct, 1),
            applied_to_os=applied,
            vital_max_hp=VITAL_MAX_HP
        )

    def evaluate_optimal_strategy(
        self,
        manual_override: Optional[str] = None
    ) -> CoreSwitchingDecision:
        """
        Samples live telemetry and deduces the optimal CPU core allocation strategy.
        """
        if manual_override and manual_override in STRATEGIES:
            chosen = STRATEGIES[manual_override]
            return self._formulate_decision(chosen, applied=self.apply_affinity(chosen["affinity_mask"]))

        # Read live processor telemetry
        report = GLOBAL_PROCESSOR_INTEGRITY_ENGINE.sample_integrity()
        cs_rate = report.context_switches_per_sec
        thrash = report.thrashing_index
        cpu_load = report.cpu_utilization_pct

        # Cortex Deductive Rules
        if thrash > 0.75 or cs_rate > (report.expected_baseline_cs_per_sec * 3.5):
            # Thrashing storm -> Offload to E-cores to stop invalidation cascades
            strat_key = "STRATEGY_COOPERATIVE_E_CORE_THROTTLE"
        elif thrash < 0.35 and cpu_load > 12.0:
            # High compute / render load -> P-Core burst
            strat_key = "STRATEGY_ISOLATED_P_CORE_BURST"
        else:
            # Nominal load -> Symmetric hybrid
            strat_key = "STRATEGY_HYBRID_SYMMETRIC"

        chosen = STRATEGIES[strat_key]
        now = time.time()
        applied = False

        # Apply if strategy changed and minimum interval elapsed
        if strat_key != self.active_strategy_id and (now - self.last_switch_time) >= self.min_switch_interval_sec:
            applied = self.apply_affinity(chosen["affinity_mask"])
            self.active_strategy_id = strat_key
            self.last_switch_time = now
            self.total_switches += 1

        decision = CoreSwitchingDecision(
            timestamp=now,
            current_strategy=chosen["id"],
            affinity_mask_hex=f"0x{chosen['affinity_mask']:04X}",
            affinity_mask_int=chosen["affinity_mask"],
            target_lane=chosen["target_lane"],
            target_freq_mode=chosen["target_freq_mode"],
            context_switches_per_sec=round(cs_rate, 1),
            thrashing_index=round(thrash, 3),
            cpu_utilization_pct=round(cpu_load, 1),
            applied_to_os=applied,
            vital_max_hp=VITAL_MAX_HP
        )
        return decision

    def apply_affinity(self, affinity_mask: int) -> bool:
        """Applies process affinity mask on Windows via kernel32."""
        if not self.is_windows:
            return False
        try:
            h_proc = ctypes.windll.kernel32.GetCurrentProcess()
            # SetProcessAffinityMask(HANDLE hProcess, DWORD_PTR dwProcessAffinityMask)
            res = ctypes.windll.kernel32.SetProcessAffinityMask(
                h_proc, ctypes.c_size_t(affinity_mask)
            )
            return bool(res)
        except Exception:
            return False

    def get_current_state(self) -> Dict[str, Any]:
        """Returns the full status descriptor of the thread switcher."""
        strat = STRATEGIES[self.active_strategy_id]
        return {
            "active_strategy": strat,
            "total_switches": self.total_switches,
            "available_strategies": list(STRATEGIES.values()),
            "vital_max_hp": VITAL_MAX_HP
        }


GLOBAL_CORTEX_THREAD_SWITCHER = CortexThreadSwitcher()
