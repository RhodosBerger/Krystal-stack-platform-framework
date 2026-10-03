"""The bot palette: an append-only conversation ledger that every client (editor extension, web UI,
mesh node) writes into, plus a bounded in-memory view and optional RAG ingestion (stdlib only).

Entries are plain data with a UUID, a channel tag and an optional parent so a conversation can be
reconstructed as a tree. Text is length-capped; nothing is executed from palette content.
"""
from __future__ import annotations

import json
import os
import threading
import time
import uuid
from collections import deque
from typing import Any, Callable, Deque, Dict, List, Optional

MAX_TEXT = 8000
ROLES = ("user", "bot", "system")
CHANNELS = ("extension", "web", "mesh", "cli", "internal")


class Palette:
    def __init__(self, path: Optional[str] = None, capacity: int = 2000, max_file_bytes: int = 2_000_000,
                 on_entry: Optional[Callable[[Dict[str, Any]], None]] = None):
        self.path, self.max_file_bytes, self.on_entry = path, max_file_bytes, on_entry
        self._buf: Deque[Dict[str, Any]] = deque(maxlen=capacity)
        self._lock = threading.Lock()
        if path:
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
            for e in self._read(path):
                self._buf.append(e)

    @staticmethod
    def _read(path: str) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
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
        return out[-2000:]

    def add(self, role: str, text: str, channel: str = "internal", conversation: Optional[str] = None,
            parent: Optional[str] = None, meta: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        if role not in ROLES:
            raise ValueError(f"role must be one of {ROLES}")
        if channel not in CHANNELS:
            raise ValueError(f"channel must be one of {CHANNELS}")
        e = {"id": uuid.uuid4().hex, "t": round(time.time(), 3), "role": role, "channel": channel,
             "conversation": conversation or uuid.uuid4().hex, "parent": parent, "text": str(text)[:MAX_TEXT], "meta": meta or {}}
        with self._lock:
            self._buf.append(e)
            if self.path:
                try:
                    if os.path.exists(self.path) and os.path.getsize(self.path) > self.max_file_bytes:
                        bak = self.path + ".1"
                        if os.path.exists(bak):
                            os.remove(bak)
                        os.replace(self.path, bak)
                    with open(self.path, "a", encoding="utf-8") as f:
                        f.write(json.dumps(e, ensure_ascii=False) + "\n")
                except OSError:
                    pass
        if self.on_entry:
            try:
                self.on_entry(e)
            except Exception:  # noqa: BLE001 - ingestion must never break the conversation path
                pass
        return e

    def tail(self, n: int = 50, conversation: Optional[str] = None) -> List[Dict[str, Any]]:
        with self._lock:
            items = list(self._buf)
        if conversation:
            items = [e for e in items if e["conversation"] == conversation]
        return items[-max(1, min(n, 500)):]

    def conversations(self) -> List[Dict[str, Any]]:
        with self._lock:
            items = list(self._buf)
        seen: Dict[str, Dict[str, Any]] = {}
        for e in items:
            c = seen.setdefault(e["conversation"], {"conversation": e["conversation"], "messages": 0, "first": e["t"], "last": e["t"], "channels": set()})
            c["messages"] += 1
            c["last"] = e["t"]
            c["channels"].add(e["channel"])
        return [{**c, "channels": sorted(c["channels"])} for c in sorted(seen.values(), key=lambda c: -c["last"])]

    def __len__(self) -> int:
        return len(self._buf)
