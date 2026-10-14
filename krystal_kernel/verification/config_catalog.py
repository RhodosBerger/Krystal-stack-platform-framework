"""Catalog of proven runtime configurations: performance vs. battery life vs. balanced load.

Each profile is a named, validated configuration. `select_profile` picks the best
profile for a battery state and workload, scoring by a weighted trade-off.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional

VITAL_MAX_HP = 6


@dataclass(frozen=True)
class ConfigProfile:
    name: str
    category: str               # "battery" | "balanced" | "performance"
    cpu_cap_pct: int            # max CPU utilisation
    npu_offload: bool           # prefer NPU over CPU/GPU
    gpu_enabled: bool
    batch_size: int
    tick_hz: int                # kernel/render tick rate
    codec_bits: int             # 16 / 32 / 64 B zero-copy codec width (in bytes)
    perf_score: float           # relative throughput 0..1
    power_draw_w: float         # estimated average draw
    notes: str = ""
    tags: List[str] = field(default_factory=list)

    @property
    def efficiency(self) -> float:
        """Work per watt."""
        return self.perf_score / max(self.power_draw_w, 1e-12)


CATALOG: Dict[str, ConfigProfile] = {p.name: p for p in [
    ConfigProfile("ultra_saver", "battery", 25, True, False, 1, 10, 16, 0.20, 3.5,
                  "NPU only, 10 Hz, 16B codec. Max runtime.", ["idle", "travel"]),
    ConfigProfile("saver_npu", "battery", 40, True, False, 2, 15, 16, 0.35, 5.0,
                  "NPU offload, low tick, no GPU.", ["office", "bot"]),
    ConfigProfile("balanced_npu", "balanced", 55, True, False, 4, 30, 32, 0.55, 8.0,
                  "Best efficiency/W for daily use.", ["default", "mission_control"]),
    ConfigProfile("balanced_hybrid", "balanced", 65, True, True, 8, 30, 32, 0.70, 12.0,
                  "NPU + light GPU for rendering.", ["render", "dev"]),
    ConfigProfile("performance_gpu", "performance", 85, False, True, 16, 60, 64, 0.90, 25.0,
                  "GPU-heavy, 60 Hz, 64B codec. Plugged in.", ["training", "ac"]),
    ConfigProfile("max_burst", "performance", 100, False, True, 32, 120, 64, 1.00, 40.0,
                  "Short bursts only; thermal-limited.", ["benchmark", "ac"]),
]}


def validate(profile: ConfigProfile) -> None:
    assert VITAL_MAX_HP == 6
    assert 0 < profile.cpu_cap_pct <= 100
    assert profile.codec_bits in (16, 32, 64)
    assert 0.0 <= profile.perf_score <= 1.0 and profile.power_draw_w > 0


def list_by_category(category: str) -> List[ConfigProfile]:
    return [p for p in CATALOG.values() if p.category == category]


def estimated_runtime_h(profile: ConfigProfile, battery_wh: float, charge_pct: float) -> float:
    return battery_wh * charge_pct / 100.0 / profile.power_draw_w


def select_profile(charge_pct: float, plugged: bool = False, min_perf: float = 0.0,
                   target_runtime_h: Optional[float] = None, battery_wh: float = 50.0) -> ConfigProfile:
    """Pick a profile: plugged -> best perf; else maximise efficiency subject to
    min_perf and runtime target; below 20% always the battery category."""
    assert VITAL_MAX_HP == 6
    pool = list(CATALOG.values())
    if plugged:
        return max(pool, key=lambda p: p.perf_score)
    if charge_pct < 20:
        pool = list_by_category("battery")
    else:
        pool = [p for p in pool if p.category != "performance"] or pool
    if min_perf == 0.0 and charge_pct >= 20:
        min_perf = 0.5  # healthy battery: don't fall to saver profiles by default
    ok = [p for p in pool if p.perf_score >= min_perf]
    if target_runtime_h is not None:
        ok = [p for p in ok if estimated_runtime_h(p, battery_wh, charge_pct) >= target_runtime_h]
    ok = ok or pool
    return max(ok, key=lambda p: p.efficiency)


for _p in CATALOG.values():
    validate(_p)
