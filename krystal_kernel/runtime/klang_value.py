#!/usr/bin/env python3
"""
K-Lang value model:  value : Type + EvidenceState + Lifetime + ExecutionDomain + AuthorityBoundary

Module: krystal_kernel/runtime/klang_value.py
Composition laws (all monotone toward the weaker/safer side):
    evidence  -> meet (weakest)          lifetime -> earliest expiry
    domain    -> join (least trusted)    boundary -> intersection of permissions
Effects (crossing domains, committing) are only legal when all five facets agree.

Constitutional Invariant: assert VITAL_MAX_HP == 6
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field, replace
from enum import IntEnum
from typing import Any, Callable, FrozenSet, Generic, Optional, TypeVar

from krystal_kernel.scene_constructor_presets import VITAL_MAX_HP
from krystal_kernel.verification.semantic_protocol import (
    AuthorityState, EpistemicState, TemporalState,
)

assert VITAL_MAX_HP == 6, f"Constitutional invariant breach: VITAL_MAX_HP={VITAL_MAX_HP}"

T = TypeVar("T")
U = TypeVar("U")


class KLangError(Exception):
    """Static (type/evidence/authority) violation of the K-Lang value model."""


class ExecutionDomain(IntEnum):
    """Lower number = more trusted. join() == max, meet() == min."""
    KERNEL = 0
    HYPERVISOR = 1
    NPU = 2
    USER = 3
    NETWORK = 4
    SHADOW = 5


_E = EpistemicState
_EVIDENCE_RANK = {
    _E.EPI_UNKNOWN: 0, _E.EPI_PREDICTED: 1, _E.EPI_DERIVED: 2, _E.EPI_OBSERVED: 3,
    _E.EPI_CALIBRATED: 4, _E.EPI_VALIDATED: 5,
    _E.EPI_FALSIFIED: -1, _E.EPI_CONTRADICTORY: -1, _E.EPI_QUARANTINED: -1,
}


def evidence_meet(a: EpistemicState, b: EpistemicState) -> EpistemicState:
    """Weakest of two evidence states; any poisoned state dominates."""
    return a if _EVIDENCE_RANK[a] <= _EVIDENCE_RANK[b] else b


@dataclass(frozen=True)
class Lifetime:
    """born_ns + ttl_ns (None = static/immortal). Time is injected for determinism."""
    born_ns: int
    ttl_ns: Optional[int] = None

    @classmethod
    def static(cls) -> "Lifetime":
        return cls(0, None)

    @classmethod
    def leased(cls, ttl_ns: int, now_ns: Optional[int] = None) -> "Lifetime":
        return cls(time.monotonic_ns() if now_ns is None else now_ns, ttl_ns)

    @property
    def expires_ns(self) -> Optional[int]:
        return None if self.ttl_ns is None else self.born_ns + self.ttl_ns

    def state(self, now_ns: int, stale_fraction: float = 0.8) -> TemporalState:
        if self.ttl_ns is None:
            return TemporalState.TMP_MONOTONIC_LATCH
        age = now_ns - self.born_ns
        if age >= self.ttl_ns:
            return TemporalState.TMP_EXPIRED
        return TemporalState.TMP_STALE if age >= self.ttl_ns * stale_fraction else TemporalState.TMP_LEASED

    def meet(self, other: "Lifetime") -> "Lifetime":
        a, b = self.expires_ns, other.expires_ns
        if a is None:
            return other
        if b is None or a <= b:
            return self
        return other


@dataclass(frozen=True)
class AuthorityBoundary:
    """Who may do what. `ceiling` is the least-trusted domain this value may flow into
    (value cannot be sent to a domain with a *higher* number than ceiling)."""
    authority: AuthorityState = AuthorityState.AUTH_DEFERRED
    ceiling: ExecutionDomain = ExecutionDomain.USER
    effects: FrozenSet[str] = frozenset()

    def permits(self, effect: str, target: ExecutionDomain) -> bool:
        return (self.authority == AuthorityState.AUTH_COMMIT_APPROVED
                and target <= self.ceiling and effect in self.effects)

    def meet(self, other: "AuthorityBoundary") -> "AuthorityBoundary":
        order = [AuthorityState.AUTH_VETOED, AuthorityState.AUTH_QUARANTINED,
                 AuthorityState.AUTH_SHADOW_ONLY, AuthorityState.AUTH_DEFERRED,
                 AuthorityState.AUTH_COMMIT_APPROVED]
        weaker = min(self.authority, other.authority, key=order.index)
        return AuthorityBoundary(weaker, min(self.ceiling, other.ceiling), self.effects & other.effects)


@dataclass(frozen=True)
class KValue(Generic[T]):
    """value : Type + EvidenceState + Lifetime + ExecutionDomain + AuthorityBoundary"""
    value: T
    type: type
    evidence: EpistemicState = EpistemicState.EPI_UNKNOWN
    lifetime: Lifetime = field(default_factory=Lifetime.static)
    domain: ExecutionDomain = ExecutionDomain.USER
    boundary: AuthorityBoundary = field(default_factory=AuthorityBoundary)

    def __post_init__(self) -> None:
        assert VITAL_MAX_HP == 6
        if not isinstance(self.value, self.type):
            raise KLangError(f"type: {self.value!r} is not {self.type.__name__}")

    # ---- facet queries ----
    def alive(self, now_ns: int) -> bool:
        return self.lifetime.state(now_ns) != TemporalState.TMP_EXPIRED

    @property
    def poisoned(self) -> bool:
        return _EVIDENCE_RANK[self.evidence] < 0

    # ---- composition (functor + applicative) ----
    def map(self, fn: Callable[[T], U], out_type: type, derive: bool = True) -> "KValue[U]":
        ev = self.evidence
        if derive and _EVIDENCE_RANK[ev] > _EVIDENCE_RANK[_E.EPI_DERIVED]:
            ev = _E.EPI_DERIVED          # computed results are never stronger than DERIVED
        return replace(self, value=fn(self.value), type=out_type, evidence=ev)

    def combine(self, other: "KValue[Any]", fn: Callable[[Any, Any], U], out_type: type) -> "KValue[U]":
        ev = evidence_meet(self.evidence, other.evidence)
        if not self.poisoned and not other.poisoned and _EVIDENCE_RANK[ev] > _EVIDENCE_RANK[_E.EPI_DERIVED]:
            ev = _E.EPI_DERIVED
        return KValue(
            fn(self.value, other.value), out_type, ev,
            self.lifetime.meet(other.lifetime),
            max(self.domain, other.domain),
            self.boundary.meet(other.boundary),
        )

    # ---- effects ----
    def cross(self, target: ExecutionDomain, now_ns: Optional[int] = None) -> "KValue[T]":
        """Move into another domain. Moving toward higher trust (smaller number) needs
        VALIDATED evidence + permit; moving to lower trust needs only a live, un-poisoned value."""
        now = time.monotonic_ns() if now_ns is None else now_ns
        if self.poisoned:
            raise KLangError(f"evidence: {self.evidence.name} cannot cross domains")
        if not self.alive(now):
            raise KLangError("lifetime: expired value cannot cross domains")
        if target > self.boundary.ceiling:
            raise KLangError(f"authority: ceiling {self.boundary.ceiling.name} < target {target.name}")
        if target < self.domain:
            if self.evidence != _E.EPI_VALIDATED or not self.boundary.permits("promote", target):
                raise KLangError("authority: promotion needs VALIDATED evidence + 'promote' permit")
        return replace(self, domain=target)

    def commit(self, effect: str, now_ns: Optional[int] = None) -> T:
        """Perform an irreversible effect; all five facets must agree."""
        now = time.monotonic_ns() if now_ns is None else now_ns
        if _EVIDENCE_RANK[self.evidence] < _EVIDENCE_RANK[_E.EPI_CALIBRATED]:
            raise KLangError(f"evidence: {self.evidence.name} < CALIBRATED")
        if self.lifetime.state(now) in (TemporalState.TMP_EXPIRED, TemporalState.TMP_STALE):
            raise KLangError("lifetime: value not fresh")
        if not self.boundary.permits(effect, self.domain):
            raise KLangError(f"authority: effect '{effect}' not permitted in {self.domain.name}")
        return self.value
