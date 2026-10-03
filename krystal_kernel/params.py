"""Parameter derivation: every operating parameter, its value, and *why*.

Values come from (in priority order): measured calibration -> hardware profile -> documented
defaults. `explain()` returns the full table so the API/docs can show provenance.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .hwprofile import HardwareProfile


@dataclass
class KernelParams:
    # -- lanes
    thread_workers: int = 4
    process_workers: int = 1
    reserve_cores: int = 1
    # -- latency / queueing
    latency_target_ms: float = 250.0
    inflation_limit: float = 2.0
    expected_service_ms: float = 20.0
    max_queue_wait_ms: float = 2000.0
    max_queue: int = 256
    max_payload_bytes: int = 4 * 1024 * 1024
    # -- admission (per API key)
    rate_per_s: float = 50.0
    burst: float = 100.0
    # -- learning
    epsilon: float = 0.10
    epsilon_floor: float = 0.02
    epsilon_decay: float = 0.995
    min_samples_for_pattern: int = 8
    # -- healing
    health_interval_s: float = 1.0
    stall_after_s: float = 5.0
    breaker_error_rate: float = 0.5
    breaker_window: int = 20
    breaker_cooldown_s: float = 10.0
    max_restarts_per_min: int = 3
    heal_cooldown_s: float = 5.0
    provenance: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = {k: v for k, v in self.__dict__.items() if k != "provenance"}
        return d

    def explain(self) -> List[Dict[str, Any]]:
        return [{"param": k, "value": getattr(self, k), **self.provenance.get(k, {"source": "default", "why": "documented default"})}
                for k in self.to_dict()]


def derive_params(profile: HardwareProfile, *, expected_service_ms: float = 20.0, latency_target_ms: float = 250.0,
                  max_queue_wait_ms: float = 2000.0) -> KernelParams:
    p = KernelParams(expected_service_ms=expected_service_ms, latency_target_ms=latency_target_ms,
                     max_queue_wait_ms=max_queue_wait_ms)
    prov = p.provenance
    cal = profile.calibration or {}

    # Reserve one core for the hub's own CPU-bound Python engine thread, so the kernel's workers do
    # not starve the thing that serves the API (a counter-productive effect we want to avoid).
    p.reserve_cores = 1 if profile.physical_cores > 1 else 0
    prov["reserve_cores"] = {"source": "derived", "why": "1 core kept free for the hub's pure-Python engine/HTTP threads"}

    if not profile.gil_enabled:
        proc_budget = 1
        prov["process_workers"] = {"source": "derived", "why": "free-threaded Python: threads scale, process lane kept minimal"}
        p.thread_workers = max(2, profile.logical_cores - p.reserve_cores)
    else:
        usable = max(1, profile.physical_cores - p.reserve_cores)
        best = cal.get("best_process_workers")
        if best:
            # The knee was *measured* across physical..logical worker counts, so if it exceeds the
            # physical core count the measurement itself shows SMT helps; cap at logical - reserve.
            cap = max(usable, profile.logical_cores - p.reserve_cores) if int(best) > profile.physical_cores else usable
            proc_budget = max(1, min(int(best), cap))
            prov["process_workers"] = {"source": "measured", "why": f"min(measured scaling knee {best}, cap {cap} = cores - {p.reserve_cores} reserved)",
                                       "evidence": {"process_speedup": cal.get("process_speedup"), "peak": cal.get("peak_process_speedup")}}
        else:
            proc_budget = usable
            prov["process_workers"] = {"source": "derived", "why": f"physical cores ({profile.physical_cores}) - reserve ({p.reserve_cores}); run calibrate() to replace with a measurement"}
        # GIL: threads add ~no CPU parallelism, so the thread lane is for short / IO-bound kernels only.
        p.thread_workers = max(2, min(8, profile.logical_cores))
    p.process_workers = proc_budget
    prov["thread_workers"] = {"source": "derived", "why": "short/IO-bound kernels only (GIL gives ~1.1x CPU scaling); capped at 8",
                              "evidence": {"thread_speedup": cal.get("thread_speedup")}}

    total_workers = p.thread_workers + p.process_workers
    # Little's law: L = lambda * W. A queue that can hold more work than can be served within
    # max_queue_wait_ms only converts overload into timeouts, so cap it there.
    served_per_wait = p.process_workers * (max_queue_wait_ms / max(1.0, expected_service_ms))
    p.max_queue = int(max(16, min(4096, math.ceil(served_per_wait + p.thread_workers))))
    prov["max_queue"] = {"source": "derived", "why": "Little's law: workers * max_queue_wait / expected_service (clamped 16..4096)"}
    prov["max_queue_wait_ms"] = {"source": "default", "why": "beyond ~2 s a queued request is usually already useless to a caller"}
    prov["expected_service_ms"] = {"source": "default", "why": "typical small-batch kernel; the live EWMA replaces it at runtime"}
    prov["latency_target_ms"] = {"source": "default", "why": "a single task slower than this counts as a congestion signal for AIMD"}
    prov["inflation_limit"] = {"source": "derived", "why": "service time > 2x its uncontended baseline => lane oversubscribed (GIL convoy / thermal / SMT); measured: 8 threads on a CPU-bound kernel gave 1.0-1.15x total speedup"}

    # sustained admission rate with 20% headroom; burst = 2 s of that
    cap_per_s = p.process_workers * 1000.0 / max(1.0, expected_service_ms) + p.thread_workers * 1000.0 / max(1.0, expected_service_ms) * 0.25
    p.rate_per_s = round(0.8 * cap_per_s, 1)
    p.burst = round(2 * p.rate_per_s, 1)
    prov["rate_per_s"] = {"source": "derived", "why": "80% of estimated sustainable throughput (GIL-discounted thread lane)"}
    prov["burst"] = {"source": "derived", "why": "2 s of sustained rate"}

    if cal.get("ipc_rtt_us_p50"):
        prov["expected_service_ms"] = {"source": "default", "why": f"default; IPC floor measured at {cal['ipc_rtt_us_p50']} us p50, so any kernel >= ~1 ms amortises dispatch",
                                       "evidence": {"ipc_rtt_us_p50": cal["ipc_rtt_us_p50"]}}
    return p
