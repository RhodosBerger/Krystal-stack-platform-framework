"""Hardware profile: what this host actually has, plus an explicit, cached calibration.

`detect_profile()` is cheap (no benchmarks). `calibrate()` runs short measured micro-benchmarks
(thread vs process scaling, IPC round-trip) and caches them in logs/hw_calibration.json so the
kernel's parameters are derived from *measurements*, not assumptions.
"""
from __future__ import annotations

import importlib.util
import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CALIBRATION_PATH = os.path.join(REPO_ROOT, "logs", "hw_calibration.json")


@dataclass
class HardwareProfile:
    cpu_name: str = "unknown"
    logical_cores: int = 1
    physical_cores: int = 1
    ram_gb: float = 0.0
    os_name: str = ""
    python: str = ""
    gil_enabled: bool = True
    openvino_installed: bool = False
    openvino_devices: List[str] = field(default_factory=list)
    npu_present: Optional[bool] = None  # None = unknown (needs OpenVINO or a PnP query)
    calibration: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _ram_gb() -> float:
    try:
        import ctypes

        class MEMSTAT(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
        m = MEMSTAT()
        m.dwLength = ctypes.sizeof(MEMSTAT)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m)):  # type: ignore[attr-defined]
            return round(m.ullTotalPhys / 2 ** 30, 1)
    except Exception:
        pass
    try:  # POSIX fallback
        return round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2 ** 30, 1)
    except Exception:
        return 0.0


def _cpu_name() -> str:
    try:
        import winreg
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as k:
            return str(winreg.QueryValueEx(k, "ProcessorNameString")[0]).strip()
    except Exception:
        return platform.processor() or "unknown"


def _physical_cores(logical: int) -> int:
    """Physical core count via GetLogicalProcessorInformation (Windows); else logical // 2 heuristic is NOT used."""
    try:
        import ctypes

        class SLPI(ctypes.Structure):
            _fields_ = [("ProcessorMask", ctypes.c_size_t), ("Relationship", ctypes.c_int), ("Reserved", ctypes.c_ubyte * 16)]
        k32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
        n = ctypes.c_ulong(0)
        k32.GetLogicalProcessorInformation(None, ctypes.byref(n))
        count = n.value // ctypes.sizeof(SLPI)
        arr = (SLPI * count)()
        if k32.GetLogicalProcessorInformation(arr, ctypes.byref(n)):
            cores = sum(1 for e in arr if e.Relationship == 0)  # RelationProcessorCore
            if cores:
                return cores
    except Exception:
        pass
    return logical  # unknown: do not assume hyper-threading


def _openvino() -> tuple:
    if importlib.util.find_spec("openvino") is None:
        return False, [], None
    try:
        import openvino as ov  # type: ignore
        devices = list(ov.Core().available_devices)
        return True, devices, any(d.startswith("NPU") for d in devices)
    except Exception:
        return True, [], None


def load_calibration() -> Dict[str, Any]:
    try:
        with open(CALIBRATION_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def detect_profile(use_cached_calibration: bool = True) -> HardwareProfile:
    logical = os.cpu_count() or 1
    ov_installed, ov_devices, npu = _openvino()
    return HardwareProfile(
        cpu_name=_cpu_name(), logical_cores=logical, physical_cores=_physical_cores(logical), ram_gb=_ram_gb(),
        os_name=f"{platform.system()} {platform.release()}", python=platform.python_version(),
        gil_enabled=getattr(sys, "_is_gil_enabled", lambda: True)(),
        openvino_installed=ov_installed, openvino_devices=ov_devices, npu_present=npu,
        calibration=load_calibration() if use_cached_calibration else {},
    )


def _burn(n: int) -> int:
    x = 0
    for i in range(n):
        x += i * i % 7
    return x


def calibrate(work_units: int = 1_500_000, repeats: int = 2, save: bool = True) -> Dict[str, Any]:
    """Measure thread/process scaling and IPC floor. Takes ~3-6 s. Results are *this run's* numbers."""
    from concurrent.futures import ThreadPoolExecutor

    prof = detect_profile(use_cached_calibration=False)
    t0 = time.perf_counter()
    _burn(work_units)
    single = time.perf_counter() - t0

    def best(fn):
        return min(fn() for _ in range(repeats))

    def run_threads(w):
        with ThreadPoolExecutor(w) as ex:
            t = time.perf_counter(); list(ex.map(_burn, [work_units] * w)); return time.perf_counter() - t

    cal: Dict[str, Any] = {"single_ms": round(single * 1000, 1), "work_units": work_units, "thread_speedup": {}, "process_speedup": {}}
    for w in sorted({2, max(2, prof.physical_cores)}):
        cal["thread_speedup"][str(w)] = round(w * single / best(lambda: run_threads(w)), 2)

    # process scaling via our own worker pool (the same machinery the kernel uses)
    from .lanes import ProcessLane
    for w in sorted({2, max(2, prof.physical_cores), prof.logical_cores}):
        lane = ProcessLane(workers=w)
        try:
            lane.warm()

            def run_procs(w=w, lane=lane):
                t = time.perf_counter()
                futs = [lane.submit_raw("burn", {"n": work_units}) for _ in range(w)]
                for f in futs:
                    f.result(timeout=60)
                return time.perf_counter() - t
            cal["process_speedup"][str(w)] = round(w * single / best(run_procs), 2)
        finally:
            lane.close()

    lane = ProcessLane(workers=1)
    try:
        lane.warm()
        lat = []
        for _ in range(100):
            t = time.perf_counter(); lane.submit_raw("echo", {"v": 1}).result(timeout=10); lat.append(time.perf_counter() - t)
        lat.sort()
        cal["ipc_rtt_us_p50"] = round(lat[50] * 1e6)
        cal["ipc_rtt_us_p95"] = round(lat[95] * 1e6)
    finally:
        lane.close()

    # knee: the smallest worker count within 10% of the best speedup. Fewer workers = less heat,
    # less contention with the rest of the program, nearly the same throughput.
    top = max(cal["process_speedup"].values())
    knee = min(int(k) for k, v in cal["process_speedup"].items() if v >= 0.9 * top)
    cal["best_process_workers"] = knee
    cal["peak_process_speedup"] = top
    cal["measured_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    if save:
        try:
            os.makedirs(os.path.dirname(CALIBRATION_PATH), exist_ok=True)
            with open(CALIBRATION_PATH, "w", encoding="utf-8") as f:
                json.dump(cal, f, indent=2)
        except OSError:
            pass
    return cal
