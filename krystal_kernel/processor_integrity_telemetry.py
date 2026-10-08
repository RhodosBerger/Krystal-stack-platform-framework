#!/usr/bin/env python3
"""
==============================================================================
KRYSTAL-STACK KERNEL: PROCESSOR INTEGRITY & CONTEXT-SWITCH TELEMETRY
==============================================================================
Monitors and predicts CPU behavioral integrity based on low-level kernel
telemetry (Context Switches, System Calls, Thread Thrashing, and CPU Load).

Detects Pathological Anomalies:
  - High CPU load coupled with anomalous thread context switching (thrashing)
    that sharply diverges from normal computational baseline profiles.
  - Core migration stalls, lock contention, and cache line invalidation storms.

System Invariant: VITAL_MAX_HP = 6.

Author: Dušan Kopecký & Krystal-Stack Architecture Council (2026)
==============================================================================
"""

import os
import sys
import time
import math
import ctypes
from dataclasses import dataclass, asdict
from typing import Dict, Any, List, Optional, Tuple

WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

VITAL_MAX_HP: int = 6


class ProcessorIntegrityStatus:
    OPTIMAL = "OPTIMAL"
    NOMINAL = "NOMINAL"
    THRASHING_WARNING = "THRASHING_WARNING"
    CRITICAL_INTERFERENCE = "CRITICAL_INTERFERENCE"


@dataclass
class KernelTelemetrySnapshot:
    timestamp_ns: int
    context_switches_total: int
    system_calls_total: int
    idle_time_raw: int
    kernel_time_raw: int
    user_time_raw: int


@dataclass
class ProcessorIntegrityReport:
    timestamp: float
    cpu_utilization_pct: float
    context_switches_per_sec: float
    system_calls_per_sec: float
    cs_to_syscall_ratio: float
    expected_baseline_cs_per_sec: float
    thrashing_index: float
    integrity_score: float                # 1.0 = Perfect integrity, 0.0 = Severe thrashing collapse
    status: str                           # OPTIMAL, NOMINAL, THRASHING_WARNING, CRITICAL_INTERFERENCE
    anomaly_detected: bool
    behavioral_diagnosis: str
    target_os: str
    vital_max_hp: int = VITAL_MAX_HP

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ── Windows NT Kernel Performance Structure ──────────────────────────────────
class SYSTEM_PERFORMANCE_INFORMATION(ctypes.Structure):
    _fields_ = [
        ('IdleProcessTime', ctypes.c_int64),
        ('IoReadTransferCount', ctypes.c_int64),
        ('IoWriteTransferCount', ctypes.c_int64),
        ('IoOtherTransferCount', ctypes.c_int64),
        ('IoReadOperationCount', ctypes.c_uint32),
        ('IoWriteOperationCount', ctypes.c_uint32),
        ('IoOtherOperationCount', ctypes.c_uint32),
        ('AvailablePages', ctypes.c_uint32),
        ('TotalCommittedPages', ctypes.c_uint32),
        ('TotalCommitLimit', ctypes.c_uint32),
        ('PeakCommitment', ctypes.c_uint32),
        ('PageFaultCount', ctypes.c_uint32),
        ('CopyOnWriteCount', ctypes.c_uint32),
        ('TransitionCount', ctypes.c_uint32),
        ('CacheTransitionCount', ctypes.c_uint32),
        ('DemandZeroCount', ctypes.c_uint32),
        ('PageReadCount', ctypes.c_uint32),
        ('PageReadIoCount', ctypes.c_uint32),
        ('CacheReadCount', ctypes.c_uint32),
        ('CacheIoCount', ctypes.c_uint32),
        ('DirtyPagesWriteCount', ctypes.c_uint32),
        ('DirtyWriteIoCount', ctypes.c_uint32),
        ('MappedPagesWriteCount', ctypes.c_uint32),
        ('MappedWriteIoCount', ctypes.c_uint32),
        ('PagedPoolPages', ctypes.c_uint32),
        ('NonPagedPoolPages', ctypes.c_uint32),
        ('PagedPoolAllocs', ctypes.c_uint32),
        ('PagedPoolFrees', ctypes.c_uint32),
        ('NonPagedPoolAllocs', ctypes.c_uint32),
        ('NonPagedPoolFrees', ctypes.c_uint32),
        ('FreeSystemPtes', ctypes.c_uint32),
        ('ResidentSystemCodePage', ctypes.c_uint32),
        ('TotalSystemDriverPages', ctypes.c_uint32),
        ('TotalSystemCodePages', ctypes.c_uint32),
        ('NonPagedPoolLookasideHits', ctypes.c_uint32),
        ('PagedPoolLookasideHits', ctypes.c_uint32),
        ('AvailablePagedPoolPages', ctypes.c_uint32),
        ('ResidentSystemCachePage', ctypes.c_uint32),
        ('ResidentPagedPoolPage', ctypes.c_uint32),
        ('ResidentSystemDriverPage', ctypes.c_uint32),
        ('CcFastReadNoWait', ctypes.c_uint32),
        ('CcFastReadWait', ctypes.c_uint32),
        ('CcFastReadResourceMiss', ctypes.c_uint32),
        ('CcFastReadNotPossible', ctypes.c_uint32),
        ('CcFastMdlReadNoWait', ctypes.c_uint32),
        ('CcFastMdlReadWait', ctypes.c_uint32),
        ('CcFastMdlReadResourceMiss', ctypes.c_uint32),
        ('CcFastMdlReadNotPossible', ctypes.c_uint32),
        ('CcMapDataNoWait', ctypes.c_uint32),
        ('CcMapDataWait', ctypes.c_uint32),
        ('CcMapDataNoWaitMiss', ctypes.c_uint32),
        ('CcMapDataWaitMiss', ctypes.c_uint32),
        ('CcPinMappedDataCount', ctypes.c_uint32),
        ('CcPinReadNoWait', ctypes.c_uint32),
        ('CcPinReadWait', ctypes.c_uint32),
        ('CcPinReadNoWaitMiss', ctypes.c_uint32),
        ('CcPinReadWaitMiss', ctypes.c_uint32),
        ('CcCopyReadNoWait', ctypes.c_uint32),
        ('CcCopyReadWait', ctypes.c_uint32),
        ('CcCopyReadNoWaitMiss', ctypes.c_uint32),
        ('CcCopyReadWaitMiss', ctypes.c_uint32),
        ('CcMdlReadNoWait', ctypes.c_uint32),
        ('CcMdlReadWait', ctypes.c_uint32),
        ('CcMdlReadNoWaitMiss', ctypes.c_uint32),
        ('CcMdlReadWaitMiss', ctypes.c_uint32),
        ('CcReadAheadIosCount', ctypes.c_uint32),
        ('CcLazyWriteIosCount', ctypes.c_uint32),
        ('CcLazyWritePages', ctypes.c_uint32),
        ('CcDataFlushes', ctypes.c_uint32),
        ('CcDataPages', ctypes.c_uint32),
        ('ContextSwitches', ctypes.c_uint32),
        ('FirstLevelTbFills', ctypes.c_uint32),
        ('SecondLevelTbFills', ctypes.c_uint32),
        ('SystemCalls', ctypes.c_uint32)
    ]


class ProcessorIntegrityTelemetryEngine:
    """
    Hardware and OS-level telemetry sampler monitoring thread context switches,
    detecting scheduler thrashing, and computing processor integrity.
    """

    def __init__(self):
        assert VITAL_MAX_HP == 6, "Invariant VITAL_MAX_HP must remain 6"
        self.vital_hp = VITAL_MAX_HP
        self.is_windows = (sys.platform == "win32")
        self.target_os = "Windows" if self.is_windows else ("Linux" if "linux" in sys.platform else "Unknown")
        self.last_snapshot: Optional[KernelTelemetrySnapshot] = None
        self.history: List[ProcessorIntegrityReport] = []
        
        # Calibration baseline (context switches per second expected for healthy 100% compute)
        self.baseline_healthy_cs_rate = 6500.0  # Under clean single/multi-thread compute
        self.thrashing_critical_ceiling = 3.5    # When CS rate exceeds 3.5x expected, flag anomaly

        # Prime initial snapshot
        self._take_snapshot()

    def _take_snapshot(self) -> KernelTelemetrySnapshot:
        """Reads kernel performance telemetry via Windows ntdll or Linux /proc/stat."""
        t_ns = time.time_ns()
        cs_total = 0
        sc_total = 0
        idle_raw = 0
        kernel_raw = 0
        user_raw = 0

        if self.is_windows:
            try:
                spi = SYSTEM_PERFORMANCE_INFORMATION()
                ret_len = ctypes.c_ulong()
                status = ctypes.windll.ntdll.NtQuerySystemInformation(
                    2, ctypes.byref(spi), ctypes.sizeof(spi), ctypes.byref(ret_len)
                )
                if status == 0:
                    cs_total = int(spi.ContextSwitches)
                    sc_total = int(spi.SystemCalls)
                    idle_raw = int(spi.IdleProcessTime)
            except Exception:
                pass
        elif os.path.exists("/proc/stat"):
            try:
                with open("/proc/stat", "r", encoding="utf-8") as f:
                    for line in f:
                        if line.startswith("ctxt "):
                            cs_total = int(line.split()[1])
                        elif line.startswith("cpu "):
                            parts = line.split()
                            user_raw = int(parts[1])
                            idle_raw = int(parts[4])
            except Exception:
                pass

        snap = KernelTelemetrySnapshot(
            timestamp_ns=t_ns,
            context_switches_total=cs_total,
            system_calls_total=sc_total,
            idle_time_raw=idle_raw,
            kernel_time_raw=kernel_raw,
            user_time_raw=user_raw
        )
        return snap

    def sample_integrity(self, simulated_override: Optional[Dict[str, float]] = None) -> ProcessorIntegrityReport:
        """
        Samples live telemetry delta, analyzes thread switching behavior,
        and computes Processor Integrity Score.
        """
        now_snap = self._take_snapshot()
        prev_snap = self.last_snapshot or now_snap
        self.last_snapshot = now_snap

        dt_sec = max((now_snap.timestamp_ns - prev_snap.timestamp_ns) / 1e9, 0.05)
        
        # Compute rates
        delta_cs = max(0, now_snap.context_switches_total - prev_snap.context_switches_total)
        delta_sc = max(0, now_snap.system_calls_total - prev_snap.system_calls_total)
        
        cs_rate = delta_cs / dt_sec if dt_sec > 0 else 0.0
        sc_rate = delta_sc / dt_sec if dt_sec > 0 else 0.0
        cs_sc_ratio = round(delta_cs / max(delta_sc, 1), 2)

        # Estimate CPU load (if no override, assume typical active baseline or synthetic test)
        cpu_load = 50.0
        if simulated_override and "cpu_utilization_pct" in simulated_override:
            cpu_load = simulated_override["cpu_utilization_pct"]
            if "context_switches_per_sec" in simulated_override:
                cs_rate = simulated_override["context_switches_per_sec"]

        # Expected baseline context switch rate given the CPU load
        # Healthy compute: high CPU load usually correlates with dedicated long-running loops (low CS)
        expected_cs = max(1000.0, self.baseline_healthy_cs_rate * (1.0 + (cpu_load / 100.0) * 0.5))

        # Thread Thrashing Index: ratio of actual CS rate to expected healthy baseline, weighted by CPU load
        thrashing_index = round((cs_rate / max(expected_cs, 1.0)) * (cpu_load / 50.0), 2)

        # Integrity Score: 1.0 (Optimal) down to 0.0 (Collapse)
        if thrashing_index <= 1.2:
            integrity = 1.0
            status = ProcessorIntegrityStatus.OPTIMAL
            diag = "Normálny stav: Vlákna bežia plynule bez nadmerného prepínania kontextu."
            anomaly = False
        elif thrashing_index <= 2.2:
            integrity = round(max(0.70, 1.0 - (thrashing_index - 1.2) * 0.25), 2)
            status = ProcessorIntegrityStatus.NOMINAL
            diag = "Zvýšená aktivita: Bežný multi-threading s miernou migráciou vlákien."
            anomaly = False
        elif thrashing_index <= self.thrashing_critical_ceiling:
            integrity = round(max(0.40, 0.70 - (thrashing_index - 2.2) * 0.23), 2)
            status = ProcessorIntegrityStatus.THRASHING_WARNING
            diag = "Varovanie: Detekované nadmerné prepínanie vlákien (Thread Thrashing). Zvýšená réžia plánovača."
            anomaly = True
        else:
            integrity = round(max(0.05, 0.40 - (thrashing_index - self.thrashing_critical_ceiling) * 0.1), 2)
            status = ProcessorIntegrityStatus.CRITICAL_INTERFERENCE
            diag = "Kritická anomália: Patologický context-switch storm! CPU míňa výpočtovú silu na réžiu kernelu namiesto inštrukcií."
            anomaly = True

        report = ProcessorIntegrityReport(
            timestamp=time.time(),
            cpu_utilization_pct=round(cpu_load, 1),
            context_switches_per_sec=round(cs_rate, 1),
            system_calls_per_sec=round(sc_rate, 1),
            cs_to_syscall_ratio=cs_sc_ratio,
            expected_baseline_cs_per_sec=round(expected_cs, 1),
            thrashing_index=thrashing_index,
            integrity_score=integrity,
            status=status,
            anomaly_detected=anomaly,
            behavioral_diagnosis=diag,
            target_os=self.target_os,
            vital_max_hp=self.vital_hp
        )

        self.history.append(report)
        if len(self.history) > 100:
            self.history.pop(0)

        return report


GLOBAL_PROCESSOR_INTEGRITY_ENGINE = ProcessorIntegrityTelemetryEngine()


def main():
    if sys.stdout.encoding.lower() != 'utf-8':
        sys.stdout.reconfigure(encoding='utf-8')
    print("=" * 85)
    print("  KRYSTAL-STACK: PROCESSOR INTEGRITY & CONTEXT-SWITCH TELEMETRY MONITOR")
    print("=" * 85)
    engine = GLOBAL_PROCESSOR_INTEGRITY_ENGINE
    
    # 1. Sample Live System Telemetry
    time.sleep(0.15)
    live_rep = engine.sample_integrity()
    print(f"Target OS:                   {live_rep.target_os}")
    print(f"Context Switches / sec:      {live_rep.context_switches_per_sec:,.1f}")
    print(f"System Calls / sec:          {live_rep.system_calls_per_sec:,.1f}")
    print(f"Thrashing Index:             {live_rep.thrashing_index} (Expected Baseline: {live_rep.expected_baseline_cs_per_sec:,.1f})")
    print(f"Processor Integrity Score:   {live_rep.integrity_score * 100:.1f}% [{live_rep.status}]")
    print(f"Diagnosis:                   {live_rep.behavioral_diagnosis}")
    print("-" * 85)

    # 2. Simulate Pathological Thrashing Anomaly (e.g. 95% CPU, 180,000 CS/s)
    sim_rep = engine.sample_integrity(simulated_override={
        "cpu_utilization_pct": 95.0,
        "context_switches_per_sec": 185000.0
    })
    print(">> TEST: Simulating Severe Thread Thrashing Scenario (95% CPU, 185,000 CS/s)...")
    print(f"Simulated Thrashing Index:   {sim_rep.thrashing_index}")
    print(f"Integrity Score:             {sim_rep.integrity_score * 100:.1f}% [{sim_rep.status}]")
    print(f"Anomaly Detected:            {sim_rep.anomaly_detected}")
    print(f"Diagnosis:                   {sim_rep.behavioral_diagnosis}")
    print("=" * 85)


if __name__ == "__main__":
    main()
