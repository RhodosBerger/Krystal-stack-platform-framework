"""Skill manifests and a deterministic message gateway, adapted from ideas in OpenClaw (MIT).

What is borrowed is the *architecture*, re-implemented from scratch: a trusted local gateway,
untrusted execution, deterministic policy, SKILL.md-style manifests and sender pairing. No OpenClaw
code is included. See docs/research/OPENCLAW_PATTERNS.md for the mapping.

Safety model:
* A manifest file is metadata only (name, description, triggers, requirements). It can never
  contain code that is executed. A skill's behaviour is a Python handler registered by the host.
* Policy is evaluated before any handler runs and is a pure function of (skill, channel, sender, flags).
* Channels `extension`, `web` and `cli` are loopback-trusted. Any other channel (for example `mesh`)
  must be *paired*: the sender asks for a one-time code and the local operator approves it.
"""
from __future__ import annotations

import json
import os
import re
import secrets
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

TRUSTED_CHANNELS = {"extension", "web", "cli", "internal"}
_FRONT = re.compile(r"\A---\s*\n(.*?)\n---\s*\n?", re.S)


@dataclass
class Skill:
    name: str
    description: str = ""
    triggers: List[str] = field(default_factory=list)       # slash commands, e.g. ["/art", "/draw"]
    channels: List[str] = field(default_factory=lambda: sorted(TRUSTED_CHANNELS))
    requires_donor: bool = False
    requires_debug: bool = False
    source: str = ""


def parse_manifest(text: str, source: str = "") -> Skill:
    """Parse the tiny YAML subset used by SKILL.md front matter (scalars and `[a, b]` lists)."""
    m = _FRONT.match(text)
    if not m:
        raise ValueError("manifest has no front matter")
    meta: Dict[str, Any] = {}
    for line in m.group(1).splitlines():
        if not line.strip() or line.lstrip().startswith("#") or ":" not in line:
            continue
        k, v = line.split(":", 1)
        k, v = k.strip(), v.strip()
        if v.startswith("[") and v.endswith("]"):
            meta[k] = [x.strip().strip("'\"") for x in v[1:-1].split(",") if x.strip()]
        elif v.lower() in ("true", "false"):
            meta[k] = v.lower() == "true"
        else:
            meta[k] = v.strip("'\"")
    name = str(meta.get("name", "")).strip()
    if not re.fullmatch(r"[a-z][a-z0-9_\-]{1,40}", name):
        raise ValueError("manifest name must match [a-z][a-z0-9_-]{1,40}")
    trig = [t for t in meta.get("triggers", []) if re.fullmatch(r"/[a-z][a-z0-9_\-]{0,23}", t)]
    return Skill(name=name, description=str(meta.get("description", ""))[:300], triggers=trig,
                 channels=[c for c in meta.get("channels", sorted(TRUSTED_CHANNELS)) if c in TRUSTED_CHANNELS | {"mesh"}] or sorted(TRUSTED_CHANNELS),
                 requires_donor=bool(meta.get("requires_donor", False)), requires_debug=bool(meta.get("requires_debug", False)), source=source)


def load_manifests(root: str) -> List[Skill]:
    out: List[Skill] = []
    if not os.path.isdir(root):
        return out
    for d in sorted(os.listdir(root)):
        p = os.path.join(root, d, "SKILL.md")
        if os.path.isfile(p):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    out.append(parse_manifest(f.read(), p))
            except (OSError, ValueError):
                continue
    return out


class Pairing:
    """One-time codes for non-trusted senders. Codes expire and the pending list is bounded."""

    def __init__(self, ttl_s: float = 600.0, max_pending: int = 20, clock: Callable[[], float] = time.time):
        self.ttl_s, self.max_pending, self._clock = ttl_s, max_pending, clock
        self._pending: Dict[str, Tuple[str, float]] = {}
        self._paired: Dict[str, float] = {}
        self._lock = threading.Lock()

    def request(self, sender: str) -> str:
        with self._lock:
            now = self._clock()
            self._pending = {c: v for c, v in self._pending.items() if v[1] > now}
            if len(self._pending) >= self.max_pending:
                raise ValueError("too many pending pairing requests")
            code = f"{secrets.randbelow(10 ** 6):06d}"
            self._pending[code] = (sender[:64], now + self.ttl_s)
            return code

    def approve(self, code: str) -> Optional[str]:
        with self._lock:
            item = self._pending.pop(code, None)
            if not item or item[1] <= self._clock():
                return None
            self._paired[item[0]] = self._clock()
            return item[0]

    def is_paired(self, sender: str) -> bool:
        with self._lock:
            return sender in self._paired

    def revoke(self, sender: str) -> bool:
        with self._lock:
            return self._paired.pop(sender, None) is not None


@dataclass
class Decision:
    allowed: bool
    reason: str


class Gateway:
    """Routes a text message to a registered handler after a deterministic policy check."""

    def __init__(self, pairing: Optional[Pairing] = None, is_donor: Callable[[], bool] = lambda: False, is_debug: Callable[[], bool] = lambda: False):
        self.pairing = pairing or Pairing()
        self.is_donor, self.is_debug = is_donor, is_debug
        self._skills: Dict[str, Skill] = {}
        self._handlers: Dict[str, Callable[[str, Dict[str, Any]], str]] = {}
        self._by_trigger: Dict[str, str] = {}

    def register(self, skill: Skill, handler: Callable[[str, Dict[str, Any]], str]) -> None:
        self._skills[skill.name] = skill
        self._handlers[skill.name] = handler
        for t in skill.triggers:
            self._by_trigger[t] = skill.name

    def skills(self) -> List[Dict[str, Any]]:
        return [{"name": s.name, "description": s.description, "triggers": s.triggers, "channels": s.channels,
                 "requires_donor": s.requires_donor, "requires_debug": s.requires_debug} for s in self._skills.values()]

    def decide(self, skill: Skill, channel: str, sender: str) -> Decision:
        if channel not in skill.channels:
            return Decision(False, f"skill '{skill.name}' is not enabled on channel '{channel}'")
        if channel not in TRUSTED_CHANNELS and not self.pairing.is_paired(sender):
            return Decision(False, "sender is not paired; request a pairing code and have the operator approve it")
        if skill.requires_donor and not self.is_donor():
            return Decision(False, "this skill is part of the donor easter egg")
        if skill.requires_debug and not self.is_debug():
            return Decision(False, "this skill is only available in debug mode")
        return Decision(True, "ok")

    def dispatch(self, text: str, channel: str = "internal", sender: str = "local", fallback: Optional[str] = None) -> Dict[str, Any]:
        text = (text or "").strip()
        name = fallback
        arg = text
        if text.startswith("/"):
            head, _, arg = text.partition(" ")
            name = self._by_trigger.get(head.lower())
            if name is None:
                return {"ok": False, "skill": None, "reply": f"Unknown command {head}. Try /help."}
        if name is None or name not in self._skills:
            return {"ok": False, "skill": None, "reply": "No skill handles this message."}
        sk = self._skills[name]
        d = self.decide(sk, channel, sender)
        if not d.allowed:
            return {"ok": False, "skill": name, "denied": True, "reply": f"Denied: {d.reason}"}
        try:
            reply = self._handlers[name](arg.strip(), {"channel": channel, "sender": sender})
            return {"ok": True, "skill": name, "reply": reply}
        except Exception as e:  # noqa: BLE001 - untrusted execution: a handler failure never crashes the gateway
            return {"ok": False, "skill": name, "reply": f"Skill '{name}' failed: {type(e).__name__}: {e}"}
