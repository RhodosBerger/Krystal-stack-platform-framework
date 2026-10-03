"""Self-healing supervisor.

Detects: dead/hung process workers, stalled dispatcher, unresponsive lanes (canary tasks that never
return), open backend breakers. Remedies are rate-limited (per-action cooldown + global budget per
minute) so a persistent fault degrades the service visibly instead of causing a restart storm,
which would itself be a counter-productive effect.
"""
from __future__ import annotations

import threading
import time
from collections import deque
from concurrent.futures import Future
from typing import Any, Callable, Deque, Dict, List, Optional, Tuple

from .lanes import ProcessLane


class Supervisor:
    def __init__(self, kernel, clock: Callable[[], float] = time.monotonic, interval_s: Optional[float] = None):
        self.k = kernel
        self.p = kernel.params
        self.clock = clock
        self.interval_s = interval_s if interval_s is not None else self.p.health_interval_s
        self._last_action: Dict[Tuple[str, str], float] = {}
        self._actions: Deque[float] = deque()
        self._canary: Dict[str, Tuple[Future, float]] = {}
        self._heal_history: Dict[str, Deque[float]] = {}
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.suppressed = 0
        self.heals = 0
        self._last_state = "ok"

    # ------------------------------------------------------------------ control
    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True, name="kk-supervisor")
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def _loop(self) -> None:
        while not self._stop.wait(self.interval_s):
            try:
                self.check()
            except Exception as e:  # noqa: BLE001 - the supervisor itself must never die
                self.k.log.emit("supervisor_error", error=f"{type(e).__name__}: {e}")

    # ------------------------------------------------------------------ remedy gate
    def _may_heal(self, lane: str, action: str) -> bool:
        now = self.clock()
        last = self._last_action.get((lane, action))
        while self._actions and now - self._actions[0] > 60.0:
            self._actions.popleft()
        if (last is not None and now - last < self.p.heal_cooldown_s) or len(self._actions) >= self.p.max_restarts_per_min:
            self.suppressed += 1
            self.k.log.emit("heal_suppressed", lane=lane, action=action,
                            reason="cooldown" if last is not None and now - last < self.p.heal_cooldown_s else "budget_exhausted")
            return False
        self._last_action[(lane, action)] = now
        self._actions.append(now)
        return True

    def _heal(self, lane: str, action: str, reason: str, fn: Callable[[], Any]) -> bool:
        if not self._may_heal(lane, action):
            return False
        fn()
        self.heals += 1
        hist = self._heal_history.setdefault(lane, deque(maxlen=16))
        hist.append(self.clock())
        self.k.log.emit("heal", lane=lane, action=action, reason=reason)
        recent = [t for t in hist if self.clock() - t < 300]
        if len(recent) >= 3:
            self.k.log.emit("pattern", pattern_id=f"unstable_lane:{lane}", pattern_kind="unstable_lane",
                            rule=f"{lane} lane needed {len(recent)} repairs in 5 min - investigate the kernels routed to it")
        return True

    # ------------------------------------------------------------------ checks
    def check(self) -> Dict[str, Any]:
        k, now = self.k, self.clock()
        reasons: List[str] = []
        critical = False

        if not k.dispatcher_alive():
            if self._heal("dispatcher", "restart", "dispatcher thread dead", k.restart_dispatcher):
                pass
            else:
                reasons.append("dispatcher dead and heal suppressed")
                critical = True

        for name, lane in k.lanes.items():
            if isinstance(lane, ProcessLane):
                dead, hung = lane.dead_workers(), lane.hung_workers()
                if dead or hung:
                    reason = f"{dead} dead / {hung} hung worker(s)"
                    if not self._heal(name, "respawn_workers", reason, lane.ensure_alive):
                        reasons.append(f"{name}: {reason} (heal suppressed)")
            # canary: a trivial kernel pushed directly into the lane proves it can still make progress
            pending = self._canary.get(name)
            if pending:
                fut, t0 = pending
                if fut.done():
                    if fut.exception() is not None:
                        reasons.append(f"{name}: canary failed ({fut.exception()})")
                    del self._canary[name]
                elif now - t0 > self.p.stall_after_s:
                    if self._heal(name, "restart_lane", "canary unresponsive", lane.restart):
                        lane.reset_inflight()
                        del self._canary[name]
                    else:
                        reasons.append(f"{name}: unresponsive (heal suppressed)")
                        critical = True
            if name not in self._canary:
                try:
                    self._canary[name] = (lane.submit_raw("echo", {"canary": True}), now)
                except Exception as e:  # noqa: BLE001
                    reasons.append(f"{name}: canary submit failed ({e})")

        # dispatcher stall: work is queued, lanes are idle, nothing has completed for a while
        if k.queue_len > 0 and now - getattr(k, "last_completion", now) > self.p.stall_after_s and \
                all(l.inflight == 0 for l in k.lanes.values()):
            if self._heal("dispatcher", "restart", "queue not draining while lanes idle", k.restart_dispatcher):
                pass
            else:
                reasons.append("queue stalled (heal suppressed)")

        router = getattr(k, "backends", None)
        if router is not None:
            for b in router.status():
                if b["breaker"] != "closed":
                    reasons.append(f"backend {b['name']} breaker {b['breaker']}")

        if k.queue_len > 0.8 * self.p.max_queue:
            reasons.append("queue above 80% of capacity")
        recent_heals = [t for h in self._heal_history.values() for t in h if now - t < 60]
        if recent_heals:
            reasons.append(f"{len(recent_heals)} heal(s) in the last minute")

        state = "critical" if critical else ("degraded" if reasons else "ok")
        k.health = {"state": state, "reasons": reasons, "heals": self.heals, "suppressed": self.suppressed}
        if state != self._last_state:
            k.log.emit("health_change", old=self._last_state, new=state, reasons=reasons)
            self._last_state = state
        return k.health
