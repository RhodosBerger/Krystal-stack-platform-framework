import platform
import time
from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class WindowsApiState:
    enabled: bool
    tick_ms: int
    monotonic_ms: int
    scheduler_hint: str

    def as_dict(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "tick_ms": self.tick_ms,
            "monotonic_ms": self.monotonic_ms,
            "scheduler_hint": self.scheduler_hint,
        }


class WindowsApiAdapter:
    def sample(self) -> WindowsApiState:
        if platform.system().lower() != "windows":
            return WindowsApiState(
                enabled=False,
                tick_ms=0,
                monotonic_ms=int(time.perf_counter() * 1000.0),
                scheduler_hint="non_windows",
            )

        tick = self._get_tick_count_ms()
        return WindowsApiState(
            enabled=True,
            tick_ms=tick,
            monotonic_ms=int(time.perf_counter() * 1000.0),
            scheduler_hint="windows_priority_ready",
        )

    @staticmethod
    def _get_tick_count_ms() -> int:
        try:
            import ctypes

            return int(ctypes.windll.kernel32.GetTickCount64())
        except Exception:
            return int(time.time() * 1000.0)

