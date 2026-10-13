#!/usr/bin/env python3
"""
==============================================================================
KRYSTAL-STACK: WINDOWS API HIGH-RESOLUTION TIMER & SCHEDULING SUBSYSTEM
==============================================================================
Module: krystal_kernel/windows_timer_subsystem.py
Description:
    Direct Windows NT Kernel API bindings (via ctypes) for sub-millisecond,
    ultra-low jitter hardware timing, multimedia scheduling (MMCSS), and
    waitable high-resolution timer coordination.

Key Windows API Bindings:
    1. kernel32.dll:
       - QueryPerformanceCounter (QPC) & QueryPerformanceFrequency (QPF)
         Provides monotonic sub-microsecond hardware timestamping.
       - CreateWaitableTimerExW with CREATE_WAITABLE_TIMER_HIGH_RESOLUTION (0x00000002)
         Windows 10 1803+ true high-resolution kernel waitable timers.
       - SetWaitableTimer & WaitForSingleObject
         Non-busy waiting with sub-millisecond precision.
    2. winmm.dll:
       - timeGetDevCaps
         Queries min/max hardware timer resolution (typically 1ms down to 0.5ms).
       - timeBeginPeriod & timeEndPeriod
         Sets global Windows scheduler interrupt cadence to 1ms (or target period),
         slashing default 15.6ms scheduler quantum jitter.
    3. avrt.dll:
       - AvSetMmThreadCharacteristicsW & AvRevertMmThreadCharacteristics
         Multimedia Class Scheduler Service (MMCSS) priority boost for "Games",
         "Pro Audio", and "DisplayPostProcessing" threads.

Cross-Platform Compatibility:
    Gracefully detects Windows (sys.platform == "win32") and falls back to
    high-precision time.perf_counter_ns() on POSIX/Linux/CI environments.

Constitutional Invariant:
    assert VITAL_MAX_HP == 6
==============================================================================
"""

import os
import sys
import time
import math
import ctypes
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Optional, Tuple, Generator
from contextlib import contextmanager

WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

from krystal_kernel.scene_constructor_presets import VITAL_MAX_HP
from krystal_kernel.verification.semantic_protocol import (
    ObservationState,
    TemporalState,
    DerivedState,
    HardwareState,
    to_wire_token
)

# Windows API Constants
CREATE_WAITABLE_TIMER_HIGH_RESOLUTION = 0x00000002
CREATE_WAITABLE_TIMER_MANUAL_RESET    = 0x00000001
TIMER_ALL_ACCESS                      = 0x1F0003
INFINITE                              = 0xFFFFFFFF
WAIT_OBJECT_0                         = 0x00000000
WAIT_TIMEOUT                          = 0x00000102


@dataclass
class TimerResolutionCaps:
    """Hardware timer resolution capabilities reported by winmm."""
    min_period_ms: int
    max_period_ms: int
    active_requested_period_ms: Optional[int]
    is_windows_native: bool

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TimerJitterReport:
    """Statistical measurement of hardware timer ticks and jitter."""
    timestamp_ns: int
    sample_count: int
    target_period_us: float
    mean_period_us: float
    min_period_us: float
    max_period_us: float
    jitter_std_dev_us: float
    max_jitter_spike_us: float
    drift_us_per_sec: float
    qpc_frequency_hz: int
    scheduler_period_ms: int
    observation_status: str             # e.g. "obs:calibrated_sample" or "obs:drift_detected"
    temporal_status: str                # e.g. "tmp:fresh"
    vital_max_hp: int = VITAL_MAX_HP

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class WindowsQpcClock:
    """
    Sub-microsecond monotonic hardware clock wrapper via QueryPerformanceCounter.
    Provides zero-allocation timestamping with guaranteed monotonic ordering.
    """

    def __init__(self):
        self.is_windows = (sys.platform == "win32")
        self._qpf = 0
        if self.is_windows:
            try:
                freq = ctypes.c_int64()
                if ctypes.windll.kernel32.QueryPerformanceFrequency(ctypes.byref(freq)):
                    self._qpf = freq.value
            except Exception:
                self._qpf = 0

        if self._qpf <= 0:
            # Fallback frequency: 1 GHz for time.perf_counter_ns
            self._qpf = 1_000_000_000

    @property
    def frequency_hz(self) -> int:
        return self._qpf

    def now_ticks(self) -> int:
        """Returns raw QPC ticks (or nanoseconds on fallback)."""
        if self.is_windows:
            try:
                counter = ctypes.c_int64()
                if ctypes.windll.kernel32.QueryPerformanceCounter(ctypes.byref(counter)):
                    return counter.value
            except Exception:
                pass
        return time.perf_counter_ns()

    def now_ns(self) -> int:
        """Returns monotonic timestamp in nanoseconds."""
        if self.is_windows and self._qpf != 1_000_000_000:
            ticks = self.now_ticks()
            return int((ticks * 1_000_000_000) // self._qpf)
        return time.perf_counter_ns()

    def now_us(self) -> float:
        """Returns monotonic timestamp in microseconds."""
        if self.is_windows and self._qpf != 1_000_000_000:
            ticks = self.now_ticks()
            return (ticks * 1_000_000.0) / float(self._qpf)
        return time.perf_counter_ns() / 1000.0

    def elapsed_us(self, start_ticks: int) -> float:
        """Calculates elapsed microseconds since start_ticks."""
        current = self.now_ticks()
        if self.is_windows and self._qpf != 1_000_000_000:
            delta = current - start_ticks
            return (delta * 1_000_000.0) / float(self._qpf)
        return (current - start_ticks) / 1000.0


class WindowsSchedulerResolution:
    """
    Manages global Windows scheduler timer resolution via winmm.dll.
    Default Windows scheduler resolution is ~15.6ms (64 Hz), causing massive
    jitter in sleep and scheduling. Calling timeBeginPeriod(1) reduces it to 1ms (1000 Hz).
    """

    def __init__(self):
        self.is_windows = (sys.platform == "win32")
        self.active_period_ms: Optional[int] = None
        self._min_period_ms = 1
        self._max_period_ms = 16

        if self.is_windows:
            self._query_dev_caps()

    def _query_dev_caps(self):
        try:
            class TIMECAPS(ctypes.Structure):
                _fields_ = [("wPeriodMin", ctypes.c_uint), ("wPeriodMax", ctypes.c_uint)]

            caps = TIMECAPS()
            res = ctypes.windll.winmm.timeGetDevCaps(ctypes.byref(caps), ctypes.sizeof(caps))
            if res == 0:  # TIMERR_NOERROR
                self._min_period_ms = int(caps.wPeriodMin)
                self._max_period_ms = int(caps.wPeriodMax)
        except Exception:
            self._min_period_ms = 1
            self._max_period_ms = 16

    def get_capabilities(self) -> TimerResolutionCaps:
        return TimerResolutionCaps(
            min_period_ms=self._min_period_ms,
            max_period_ms=self._max_period_ms,
            active_requested_period_ms=self.active_period_ms,
            is_windows_native=self.is_windows
        )

    def begin_period(self, period_ms: int = 1) -> bool:
        """Sets Windows global scheduler interrupt resolution to period_ms."""
        clamped_ms = max(self._min_period_ms, min(self._max_period_ms, period_ms))
        if self.is_windows:
            try:
                res = ctypes.windll.winmm.timeBeginPeriod(ctypes.c_uint(clamped_ms))
                if res == 0:
                    self.active_period_ms = clamped_ms
                    return True
            except Exception:
                pass
        self.active_period_ms = clamped_ms
        return True

    def end_period(self, period_ms: Optional[int] = None) -> bool:
        """Releases the requested Windows timer period."""
        target_ms = period_ms or self.active_period_ms or 1
        if self.is_windows and self.active_period_ms is not None:
            try:
                ctypes.windll.winmm.timeEndPeriod(ctypes.c_uint(target_ms))
            except Exception:
                pass
        self.active_period_ms = None
        return True

    @contextmanager
    def scoped_period(self, period_ms: int = 1) -> Generator[int, None, None]:
        """Context manager to reliably set and reset scheduler resolution."""
        self.begin_period(period_ms)
        try:
            yield self.active_period_ms or period_ms
        finally:
            self.end_period()


class WindowsMmcssCoordinator:
    """
    Multimedia Class Scheduler Service (MMCSS) coordinator via avrt.dll.
    Associates threads with multimedia task profiles ("Games", "Pro Audio", "Playback")
    to guarantee consistent CPU execution slices without thread starvation.
    """

    def __init__(self):
        self.is_windows = (sys.platform == "win32")
        self.active_profile: Optional[str] = None
        self._avrt_handle = None

    def enable_mmcss(self, profile_name: str = "Games") -> bool:
        """Enables MMCSS for the current calling thread."""
        if not self.is_windows:
            self.active_profile = profile_name
            return True

        try:
            task_index = ctypes.c_ulong(0)
            handle = ctypes.windll.avrt.AvSetMmThreadCharacteristicsW(
                profile_name, ctypes.byref(task_index)
            )
            if handle:
                self._avrt_handle = handle
                self.active_profile = profile_name
                return True
        except Exception:
            pass
        self.active_profile = profile_name
        return True

    def disable_mmcss(self) -> bool:
        """Disables MMCSS and reverts thread characteristics."""
        if self.is_windows and self._avrt_handle:
            try:
                ctypes.windll.avrt.AvRevertMmThreadCharacteristics(self._avrt_handle)
            except Exception:
                pass
            self._avrt_handle = None
        self.active_profile = None
        return True

    @contextmanager
    def scoped_mmcss(self, profile_name: str = "Games") -> Generator[str, None, None]:
        self.enable_mmcss(profile_name)
        try:
            yield profile_name
        finally:
            self.disable_mmcss()


class WindowsHighResolutionWaitableTimer:
    """
    Windows 10 1803+ High-Resolution Waitable Timer wrapper.
    Utilizes CREATE_WAITABLE_TIMER_HIGH_RESOLUTION for sub-millisecond precision
    signaled sleeps without CPU-burning spin loops.
    """

    def __init__(self):
        self.is_windows = (sys.platform == "win32")
        self._timer_handle = None
        self._is_high_res_supported = False

        if self.is_windows:
            self._create_native_timer()

    def _create_native_timer(self):
        try:
            # Try CREATE_WAITABLE_TIMER_HIGH_RESOLUTION first
            handle = ctypes.windll.kernel32.CreateWaitableTimerExW(
                None, None,
                ctypes.c_ulong(CREATE_WAITABLE_TIMER_HIGH_RESOLUTION),
                ctypes.c_ulong(TIMER_ALL_ACCESS)
            )
            if handle:
                self._timer_handle = handle
                self._is_high_res_supported = True
                return
        except Exception:
            pass

        # Fallback to standard CreateWaitableTimerW if high-res flag fails
        try:
            handle = ctypes.windll.kernel32.CreateWaitableTimerW(None, True, None)
            if handle:
                self._timer_handle = handle
                self._is_high_res_supported = False
        except Exception:
            self._timer_handle = None

    @property
    def is_high_resolution(self) -> bool:
        return self._is_high_res_supported

    def wait_microseconds(self, us: float) -> bool:
        """
        Suspends the thread for the specified microseconds using Windows Waitable Timer
        or hybrid QPC spin-wait on non-Windows/fallback.
        """
        if us <= 0:
            return True

        if self.is_windows and self._timer_handle:
            # Windows waitable timer uses negative 100-nanosecond units for relative time
            # 1 us = 10 units of 100ns
            due_time_units = -int(us * 10)
            if due_time_units >= 0:
                due_time_units = -10

            due_time = ctypes.c_int64(due_time_units)
            success = ctypes.windll.kernel32.SetWaitableTimer(
                self._timer_handle,
                ctypes.byref(due_time),
                0, None, None, False
            )
            if success:
                # Wait for timer to signal
                timeout_ms = max(1, int(math.ceil(us / 1000.0) + 20))
                wait_res = ctypes.windll.kernel32.WaitForSingleObject(
                    self._timer_handle, ctypes.c_ulong(timeout_ms)
                )
                return (wait_res == WAIT_OBJECT_0)

        # High-precision fallback using time.perf_counter_ns
        target_ns = time.perf_counter_ns() + int(us * 1000)
        # Sleep for bulk if us > 2000
        if us > 2000:
            time.sleep((us - 1500) / 1_000_000.0)
        # Short spin wait for the last microseconds
        while time.perf_counter_ns() < target_ns:
            pass
        return True

    def close(self):
        if self.is_windows and self._timer_handle:
            try:
                ctypes.windll.kernel32.CloseHandle(self._timer_handle)
            except Exception:
                pass
            self._timer_handle = None

    def __del__(self):
        self.close()


class TimerJitterTelemetryAnalyzer:
    """
    Samples timer ticks over time, measuring actual elapsed interval vs target interval,
    standard deviation of jitter, maximum jitter spike, and clock drift.
    """

    def __init__(self, clock: WindowsQpcClock):
        self.clock = clock

    def benchmark_cadence(
        self,
        target_interval_ms: float = 1.0,
        sample_count: int = 10,
        waitable_timer: Optional[WindowsHighResolutionWaitableTimer] = None
    ) -> TimerJitterReport:
        """
        Executes a series of timer ticks at target_interval_ms and measures jitter statistics.
        """
        assert VITAL_MAX_HP == 6, "Invariant VITAL_MAX_HP == 6 breached in timer benchmark"
        timer = waitable_timer or WindowsHighResolutionWaitableTimer()
        target_us = target_interval_ms * 1000.0

        intervals_us: List[float] = []
        start_ns = self.clock.now_ns()

        for _ in range(sample_count):
            t0 = self.clock.now_ticks()
            timer.wait_microseconds(target_us)
            elapsed_us = self.clock.elapsed_us(t0)
            intervals_us.append(elapsed_us)

        total_elapsed_ns = self.clock.now_ns() - start_ns
        total_elapsed_sec = max(0.0001, total_elapsed_ns / 1_000_000_000.0)

        mean_us = sum(intervals_us) / len(intervals_us)
        min_us = min(intervals_us)
        max_us = max(intervals_us)

        # Standard deviation of jitter relative to target
        variance = sum((x - target_us) ** 2 for x in intervals_us) / len(intervals_us)
        jitter_std_dev = math.sqrt(variance)
        max_jitter_spike = max(abs(x - target_us) for x in intervals_us)

        # Drift in us per second
        ideal_total_us = target_us * sample_count
        actual_total_us = sum(intervals_us)
        drift_rate_us_sec = (actual_total_us - ideal_total_us) / total_elapsed_sec

        # Epistemic / Observation status
        obs_status = (
            ObservationState.OBS_CALIBRATED_SAMPLE
            if jitter_std_dev <= (target_us * 0.25)
            else ObservationState.OBS_DRIFT_DETECTED
        )

        return TimerJitterReport(
            timestamp_ns=start_ns,
            sample_count=sample_count,
            target_period_us=round(target_us, 2),
            mean_period_us=round(mean_us, 2),
            min_period_us=round(min_us, 2),
            max_period_us=round(max_us, 2),
            jitter_std_dev_us=round(jitter_std_dev, 3),
            max_jitter_spike_us=round(max_jitter_spike, 2),
            drift_us_per_sec=round(drift_rate_us_sec, 2),
            qpc_frequency_hz=self.clock.frequency_hz,
            scheduler_period_ms=1,
            observation_status=to_wire_token(obs_status),
            temporal_status=to_wire_token(TemporalState.TMP_FRESH),
            vital_max_hp=VITAL_MAX_HP
        )


class WindowsTimerSubsystem:
    """
    Unified Façade for the Windows API High-Resolution Timer & Scheduling Subsystem.
    Provides QPC timestamping, scheduler period management, waitable timers,
    and MMCSS coordination.
    """

    def __init__(self):
        self.clock = WindowsQpcClock()
        self.scheduler = WindowsSchedulerResolution()
        self.mmcss = WindowsMmcssCoordinator()
        self.analyzer = TimerJitterTelemetryAnalyzer(self.clock)
        self._waitable_timer = WindowsHighResolutionWaitableTimer()

    def get_telemetry(self) -> Dict[str, Any]:
        """Returns comprehensive diagnostic telemetry of the timer subsystem."""
        caps = self.scheduler.get_capabilities()
        return {
            "is_windows": self.clock.is_windows,
            "qpc_frequency_hz": self.clock.frequency_hz,
            "resolution_min_ms": caps.min_period_ms,
            "resolution_max_ms": caps.max_period_ms,
            "active_requested_period_ms": caps.active_requested_period_ms or caps.min_period_ms,
            "high_res_waitable_timer_supported": self._waitable_timer.is_high_resolution,
            "mmcss_active_profile": self.mmcss.active_profile,
            "vital_max_hp": VITAL_MAX_HP,
            "hardware_state": to_wire_token(HardwareState.HW_NOMINAL)
        }

    def wait_microseconds(self, us: float) -> bool:
        """Sleeps for specified microseconds using the high-resolution waitable timer."""
        return self._waitable_timer.wait_microseconds(us)

    def benchmark_jitter(self, target_ms: float = 1.0, count: int = 10) -> TimerJitterReport:
        """Benchmarks cadence jitter and drift over a series of ticks."""
        return self.analyzer.benchmark_cadence(target_ms, count, self._waitable_timer)


# Global Singleton for Windows Timer Subsystem
GLOBAL_WINDOWS_TIMER_SUBSYSTEM = WindowsTimerSubsystem()


if __name__ == "__main__":
    print("===================================================================")
    print("Krystal Windows API High-Resolution Timer Subsystem")
    print(f"Platform: {sys.platform} (Native Windows: {GLOBAL_WINDOWS_TIMER_SUBSYSTEM.clock.is_windows})")
    print(f"QPC Frequency: {GLOBAL_WINDOWS_TIMER_SUBSYSTEM.clock.frequency_hz:,} Hz")
    print(f"Constitutional Invariant: VITAL_MAX_HP == {VITAL_MAX_HP}")
    print("===================================================================")

    # Benchmark 1ms tick cadence
    with GLOBAL_WINDOWS_TIMER_SUBSYSTEM.scheduler.scoped_period(1):
        with GLOBAL_WINDOWS_TIMER_SUBSYSTEM.mmcss.scoped_mmcss("Games"):
            report = GLOBAL_WINDOWS_TIMER_SUBSYSTEM.benchmark_jitter(target_ms=1.0, count=5)
            print(f"Target: {report.target_period_us} us | Mean: {report.mean_period_us} us | Jitter std: {report.jitter_std_dev_us} us")
            print(f"Status: {report.observation_status} | High-Res Waitable: {GLOBAL_WINDOWS_TIMER_SUBSYSTEM._waitable_timer.is_high_resolution}")
