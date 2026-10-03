"""Pending-log evaluator: log records wait in a bounded queue and are evaluated in quotas.

Each cycle may only spend (a) the remaining `records/minute` quota, (b) a per-cycle batch cap and (c)
a wall-clock budget, so a log flood can never starve the hub. Pending records are scored riskiest-first
(cheap heuristic) so scarce quota goes where it matters; the accepted model then refines the score.
Records that cannot be evaluated in time are *counted as dropped*, never silently lost.
"""
from __future__ import annotations

import threading
import time
from collections import deque
from typing import Any, Callable, Deque, Dict, List, Optional

from .quota import QuotaManager
from .risk import RiskModel, featurize, heuristic_risk, label


class PendingEvaluator:
    def __init__(self, model: RiskModel, quotas: QuotaManager, on_flag: Optional[Callable[[Dict[str, Any]], None]] = None,
                 max_pending: int = 2000, batch: int = 64, flag_threshold: float = 0.8, records_per_min: int = 600,
                 cycle_budget_ms: float = 50.0, history: int = 20000):
        self.model, self.quotas, self.on_flag = model, quotas, on_flag
        self.max_pending, self.batch, self.flag_threshold, self.cycle_budget_ms = max_pending, batch, flag_threshold, cycle_budget_ms
        self.quotas.define("eval_records", records_per_min, 60.0)
        self._pending: Deque[Dict[str, Any]] = deque()
        self._history: Deque[Dict[str, Any]] = deque(maxlen=history)
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self.counters = {"submitted": 0, "evaluated": 0, "flagged": 0, "dropped_overflow": 0, "quota_deferred_cycles": 0,
                         "model_errors": 0}

    def submit(self, ev: Dict[str, Any]) -> bool:
        """Subscribe-safe: O(1), never blocks, ignores non-`task_done` records."""
        if ev.get("type") != "task_done":
            return False
        with self._lock:
            self._history.append(ev)
            if len(self._pending) >= self.max_pending:
                self._pending.popleft()  # drop oldest: freshest evidence is the most actionable
                self.counters["dropped_overflow"] += 1
            self._pending.append(ev)
            self.counters["submitted"] += 1
        return True

    def history(self) -> List[Dict[str, Any]]:
        with self._lock:
            return list(self._history)

    def cycle(self) -> Dict[str, Any]:
        t0 = time.perf_counter()
        with self._lock:
            batch = list(self._pending)
        if not batch:
            return {"processed": 0, "flagged": 0, "pending": 0}
        feats = [(ev, featurize(ev)) for ev in batch]
        feats = [(ev, f) for ev, f in feats if f is not None]
        feats.sort(key=lambda p: -heuristic_risk(p[1]))          # riskiest first
        want = min(self.batch, len(feats))
        granted = 0
        q = self.quotas.get("eval_records")
        while want > 0:                                            # take as much of the quota as is available
            ok, _ = self.quotas.try_consume("eval_records", want)
            if ok:
                granted = want
                break
            want = min(want - 1, int(q.remaining()))
        if granted == 0:
            self.counters["quota_deferred_cycles"] += 1
            return {"processed": 0, "flagged": 0, "pending": len(feats), "deferred_by_quota": True}
        take = feats[:granted]
        try:
            scores = self.model.predict([f for _, f in take])
        except Exception:  # noqa: BLE001
            self.counters["model_errors"] += 1
            scores = [heuristic_risk(f) for _, f in take]
        flagged = 0
        done_ids = set()
        for (ev, f), s in zip(take, scores):
            done_ids.add(id(ev))
            self.counters["evaluated"] += 1
            if s >= self.flag_threshold:
                flagged += 1
                self.counters["flagged"] += 1
                if self.on_flag:
                    try:
                        self.on_flag({"event": ev, "risk": round(float(s), 4), "backend": self.model.backend, "actual_bad": bool(label(ev))})
                    except Exception:  # noqa: BLE001
                        pass
            if (time.perf_counter() - t0) * 1000 > self.cycle_budget_ms and len(done_ids) < len(take):
                # Out of wall-clock budget: refund what we did not evaluate.
                self.quotas.refund("eval_records", granted - len(done_ids))
                break
        with self._lock:
            self._pending = deque(e for e in self._pending if id(e) not in done_ids)
            left = len(self._pending)
        return {"processed": len(done_ids), "flagged": flagged, "pending": left}

    def start(self, interval_s: float = 2.0) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()

        def loop() -> None:
            while not self._stop.wait(interval_s):
                try:
                    self.cycle()
                except Exception:  # noqa: BLE001
                    self.counters["model_errors"] += 1

        self._thread = threading.Thread(target=loop, name="krystal-bot-eval", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=3)

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            pend = len(self._pending)
        return {"pending": pend, "max_pending": self.max_pending, "counters": dict(self.counters),
                "quota": self.quotas.get("eval_records").snapshot(), "model": self.model.status()}
