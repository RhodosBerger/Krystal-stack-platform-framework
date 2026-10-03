"""Structured event log with subscribers - the "circulation" that feeds the learning loop.

Every scheduling decision, completion, deadline miss, rejection and healing action is an event.
The learner and the healer consume these events and emit new events (policy updates, remedies),
which are in turn visible to the next iteration.
"""
from __future__ import annotations

import json
import os
import threading
import time
from collections import deque
from typing import Any, Callable, Deque, Dict, Iterable, List, Optional


class EventLog:
    def __init__(self, capacity: int = 5000, path: Optional[str] = None, max_file_bytes: int = 2_000_000):
        self._buf: Deque[Dict[str, Any]] = deque(maxlen=capacity)
        self._lock = threading.Lock()
        self._seq = 0
        self._subs: List[Callable[[Dict[str, Any]], None]] = []
        self.path, self.max_file_bytes = path, max_file_bytes
        self.dropped_sub_errors = 0
        if path:
            os.makedirs(os.path.dirname(path), exist_ok=True)

    def subscribe(self, fn: Callable[[Dict[str, Any]], None]) -> None:
        self._subs.append(fn)

    def emit(self, etype: str, **fields: Any) -> Dict[str, Any]:
        with self._lock:
            self._seq += 1
            ev = {"seq": self._seq, "t": round(time.time(), 4), "type": etype, **fields}
            self._buf.append(ev)
            subs = list(self._subs)
        if self.path:
            self._append_file(ev)
        for fn in subs:
            try:
                fn(ev)
            except Exception:
                self.dropped_sub_errors += 1  # a faulty subscriber must never break the data path
        return ev

    def _append_file(self, ev: Dict[str, Any]) -> None:
        try:
            if os.path.exists(self.path) and os.path.getsize(self.path) > self.max_file_bytes:
                bak = self.path + ".1"
                if os.path.exists(bak):
                    os.remove(bak)
                os.replace(self.path, bak)
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(ev, ensure_ascii=True) + "\n")
        except OSError:
            pass

    def tail(self, n: int = 50, etype: Optional[str] = None) -> List[Dict[str, Any]]:
        with self._lock:
            items = list(self._buf)
        if etype:
            items = [e for e in items if e["type"] == etype]
        return items[-n:]

    def counts(self) -> Dict[str, int]:
        with self._lock:
            c: Dict[str, int] = {}
            for e in self._buf:
                c[e["type"]] = c.get(e["type"], 0) + 1
            return c

    @property
    def seq(self) -> int:
        return self._seq

    @staticmethod
    def read_file(path: str, limit: int = 20000) -> Iterable[Dict[str, Any]]:
        out: Deque[Dict[str, Any]] = deque(maxlen=limit)
        for p in (path + ".1", path):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    for line in f:
                        try:
                            out.append(json.loads(line))
                        except ValueError:
                            continue
            except OSError:
                continue
        return list(out)
