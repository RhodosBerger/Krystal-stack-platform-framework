"""Rolling-window quotas shared by the bot's LLM gateway and the pending-log evaluator (stdlib only).

A quota is `limit` units per `window_s` seconds. Unlike a token bucket it never allows a burst
above the stated budget inside any window, which is what you want for a paid/remote LLM budget.
State is persisted atomically so a daily token budget survives restarts.
"""
from __future__ import annotations

import json
import os
import threading
import time
from collections import deque
from typing import Any, Callable, Deque, Dict, Optional, Tuple


class Quota:
    def __init__(self, name: str, limit: float, window_s: float, clock: Callable[[], float] = time.time):
        if limit < 0 or window_s <= 0:
            raise ValueError("limit must be >= 0 and window_s > 0")
        self.name, self.limit, self.window_s, self._clock = name, float(limit), float(window_s), clock
        self._events: Deque[Tuple[float, float]] = deque()
        self._used = 0.0
        self.denied = 0

    def _expire(self, now: float) -> None:
        while self._events and self._events[0][0] <= now - self.window_s:
            self._used -= self._events.popleft()[1]
        if not self._events:
            self._used = 0.0  # kill float drift

    def used(self) -> float:
        self._expire(self._clock())
        return self._used

    def remaining(self) -> float:
        return max(0.0, self.limit - self.used())

    def try_consume(self, n: float = 1.0) -> Tuple[bool, float]:
        """Returns (granted, retry_after_s). A request larger than the whole limit can never be granted."""
        now = self._clock()
        self._expire(now)
        if n > self.limit:
            self.denied += 1
            return False, float("inf")
        if self._used + n <= self.limit + 1e-9:
            self._events.append((now, n))
            self._used += n
            return True, 0.0
        self.denied += 1
        need, freed = self._used + n - self.limit, 0.0
        for t, amt in self._events:
            freed += amt
            if freed >= need - 1e-9:
                return False, max(0.0, t + self.window_s - now)
        return False, self.window_s

    def refund(self, n: float) -> None:
        """Return unused units (e.g. a reservation larger than the actual usage)."""
        if n <= 0:
            return
        left = n
        while left > 1e-9 and self._events:
            t, amt = self._events[-1]
            take = min(amt, left)
            if take >= amt - 1e-9:
                self._events.pop()
            else:
                self._events[-1] = (t, amt - take)
            self._used -= take
            left -= take
        self._used = max(0.0, self._used)

    def snapshot(self) -> Dict[str, Any]:
        return {"limit": self.limit, "window_s": self.window_s, "used": round(self.used(), 3),
                "remaining": round(self.remaining(), 3), "denied": self.denied}


class QuotaManager:
    def __init__(self, path: Optional[str] = None, clock: Callable[[], float] = time.time):
        self._q: Dict[str, Quota] = {}
        self._lock = threading.Lock()
        self.path, self._clock = path, clock
        self._loaded: Dict[str, Any] = {}
        if path and os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    self._loaded = json.load(f)
            except (OSError, ValueError):
                self._loaded = {}

    def define(self, name: str, limit: float, window_s: float) -> Quota:
        with self._lock:
            q = Quota(name, limit, window_s, self._clock)
            saved = self._loaded.get(name)
            if saved and abs(saved.get("window_s", 0) - window_s) < 1e-6:
                now = self._clock()
                for t, amt in saved.get("events", []):
                    if t > now - window_s:
                        q._events.append((float(t), float(amt)))
                        q._used += float(amt)
            self._q[name] = q
            return q

    def get(self, name: str) -> Quota:
        return self._q[name]

    def try_consume(self, name: str, n: float = 1.0) -> Tuple[bool, float]:
        with self._lock:
            return self._q[name].try_consume(n)

    def refund(self, name: str, n: float) -> None:
        with self._lock:
            self._q[name].refund(n)

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            return {n: q.snapshot() for n, q in self._q.items()}

    def save(self) -> None:
        if not self.path:
            return
        with self._lock:
            data = {n: {"window_s": q.window_s, "events": list(q._events)} for n, q in self._q.items()}
        os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f)
        os.replace(tmp, self.path)
