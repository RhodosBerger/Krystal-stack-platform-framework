"""
Signal-First Scheduling System (Python)

Decisions follow telemetry strength and domain rankings,
so resources flow to highest-value workloads while safety stays enforced.
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum, auto
import heapq
import time


class SignalSource(Enum):
    TELEMETRY = auto()
    USER = auto()
    POLICY = auto()
    SAFETY = auto()
    LEARNING = auto()
    EXTERNAL = auto()
    IPC = auto()


class SignalKind(Enum):
    # Performance signals
    FRAMETIME_SPIKE = auto()
    CPU_BOTTLENECK = auto()
    GPU_BOTTLENECK = auto()
    MEMORY_PRESSURE = auto()

    # Thermal signals
    THERMAL_WARNING = auto()
    THERMAL_CRITICAL = auto()
    COOLING_OPPORTUNITY = auto()

    # Workload signals
    WORKLOAD_CHANGE = auto()
    FOREGROUND_SWITCH = auto()
    IDLE_DETECTED = auto()
    BURST_DETECTED = auto()

    # User signals
    USER_BOOST_REQUEST = auto()
    USER_PROFILE_CHANGE = auto()

    # Safety signals
    GUARDRAIL_TRIGGERED = auto()
    EMERGENCY_STOP = auto()

    # IPC signals
    DRIVER_EVENT = auto()
    OPENVINO_READY = auto()


class Domain(Enum):
    SAFETY = 0
    THERMAL = 1
    USER = 2
    PERFORMANCE = 3
    POWER = 4


@dataclass
class Signal:
    """Signal with strength and metadata."""
    id: str
    source: SignalSource
    kind: SignalKind
    strength: float  # 0.0 - 1.0 normalized
    confidence: float = 0.9  # 0.0 - 1.0
    timestamp: float = field(default_factory=time.time)
    ttl_ms: int = 5000
    payload: Dict[str, Any] = field(default_factory=dict)

    def is_expired(self) -> bool:
        return (time.time() - self.timestamp) * 1000 > self.ttl_ms


@dataclass(order=True)
class RankedSignal:
    """Signal with computed priority for heap ordering."""
    priority: float  # Negated for max-heap behavior
    domain_rank: int
    signal: Signal = field(compare=False)


class SignalScheduler:
    """Signal scheduler with domain-based priority."""

    def __init__(self):
        self._queue: List[RankedSignal] = []
        self._processed: List[Signal] = []

        # Domain weights (higher = more important)
        self.domain_weights: Dict[Domain, float] = {
            Domain.SAFETY: 1.0,
            Domain.THERMAL: 0.85,
            Domain.USER: 0.8,
            Domain.PERFORMANCE: 0.7,
            Domain.POWER: 0.5,
        }

        # Kind base priorities
        self.kind_priorities: Dict[SignalKind, float] = {
            SignalKind.EMERGENCY_STOP: 1.0,
            SignalKind.THERMAL_CRITICAL: 0.95,
            SignalKind.GUARDRAIL_TRIGGERED: 0.9,
            SignalKind.THERMAL_WARNING: 0.8,
            SignalKind.USER_BOOST_REQUEST: 0.75,
            SignalKind.FRAMETIME_SPIKE: 0.7,
            SignalKind.CPU_BOTTLENECK: 0.65,
            SignalKind.GPU_BOTTLENECK: 0.65,
            SignalKind.MEMORY_PRESSURE: 0.6,
            SignalKind.WORKLOAD_CHANGE: 0.5,
            SignalKind.FOREGROUND_SWITCH: 0.5,
            SignalKind.DRIVER_EVENT: 0.45,
            SignalKind.OPENVINO_READY: 0.4,
            SignalKind.BURST_DETECTED: 0.4,
            SignalKind.USER_PROFILE_CHANGE: 0.35,
            SignalKind.COOLING_OPPORTUNITY: 0.3,
            SignalKind.IDLE_DETECTED: 0.2,
        }

    def enqueue(self, signal: Signal) -> None:
        """Enqueue a signal with computed priority."""
        domain = self._classify_domain(signal)
        priority = self._compute_priority(signal, domain)

        ranked = RankedSignal(
            priority=-priority,  # Negate for min-heap to act as max-heap
            domain_rank=domain.value,
            signal=signal,
        )
        heapq.heappush(self._queue, ranked)

    def dequeue(self) -> Optional[Signal]:
        """Dequeue highest priority non-expired signal."""
        while self._queue:
            ranked = heapq.heappop(self._queue)
            if not ranked.signal.is_expired():
                self._processed.append(ranked.signal)
                return ranked.signal
        return None

    def peek(self) -> Optional[Signal]:
        """Peek at highest priority signal without removing."""
        for ranked in sorted(self._queue):
            if not ranked.signal.is_expired():
                return ranked.signal
        return None

    def drain_by_priority(self) -> List[Signal]:
        """Get all pending signals sorted by priority."""
        signals = []
        while True:
            sig = self.dequeue()
            if sig is None:
                break
            signals.append(sig)
        return signals

    def filter_by_domain(self, domain: Domain) -> List[Signal]:
        """Get signals by domain."""
        return [
            r.signal for r in self._queue
            if self._classify_domain(r.signal) == domain and not r.signal.is_expired()
        ]

    def filter_by_kind(self, kind: SignalKind) -> List[Signal]:
        """Get signals by kind."""
        return [
            r.signal for r in self._queue
            if r.signal.kind == kind and not r.signal.is_expired()
        ]

    def _classify_domain(self, signal: Signal) -> Domain:
        """Classify signal into domain."""
        domain_map = {
            SignalKind.EMERGENCY_STOP: Domain.SAFETY,
            SignalKind.GUARDRAIL_TRIGGERED: Domain.SAFETY,
            SignalKind.THERMAL_WARNING: Domain.THERMAL,
            SignalKind.THERMAL_CRITICAL: Domain.THERMAL,
            SignalKind.COOLING_OPPORTUNITY: Domain.THERMAL,
            SignalKind.USER_BOOST_REQUEST: Domain.USER,
            SignalKind.USER_PROFILE_CHANGE: Domain.USER,
            SignalKind.FRAMETIME_SPIKE: Domain.PERFORMANCE,
            SignalKind.CPU_BOTTLENECK: Domain.PERFORMANCE,
            SignalKind.GPU_BOTTLENECK: Domain.PERFORMANCE,
            SignalKind.WORKLOAD_CHANGE: Domain.PERFORMANCE,
            SignalKind.FOREGROUND_SWITCH: Domain.PERFORMANCE,
            SignalKind.BURST_DETECTED: Domain.PERFORMANCE,
            SignalKind.MEMORY_PRESSURE: Domain.POWER,
            SignalKind.IDLE_DETECTED: Domain.POWER,
            SignalKind.DRIVER_EVENT: Domain.PERFORMANCE,
            SignalKind.OPENVINO_READY: Domain.PERFORMANCE,
        }
        return domain_map.get(signal.kind, Domain.PERFORMANCE)

    def _compute_priority(self, signal: Signal, domain: Domain) -> float:
        """Compute signal priority."""
        base = self.kind_priorities.get(signal.kind, 0.5)
        domain_weight = self.domain_weights.get(domain, 0.5)
        return base * domain_weight * signal.strength * signal.confidence

    def set_domain_weight(self, domain: Domain, weight: float) -> None:
        """Update domain weight dynamically."""
        self.domain_weights[domain] = max(0.0, min(1.0, weight))

    def stats(self) -> Dict[str, Any]:
        """Get scheduler statistics."""
        by_domain: Dict[Domain, int] = {}
        by_kind: Dict[SignalKind, int] = {}

        for ranked in self._queue:
            if not ranked.signal.is_expired():
                domain = self._classify_domain(ranked.signal)
                by_domain[domain] = by_domain.get(domain, 0) + 1
                by_kind[ranked.signal.kind] = by_kind.get(ranked.signal.kind, 0) + 1

        return {
            "queue_size": len([r for r in self._queue if not r.signal.is_expired()]),
            "processed_count": len(self._processed),
            "by_domain": {d.name: c for d, c in by_domain.items()},
            "by_kind": {k.name: c for k, c in by_kind.items()},
        }

    def clear_history(self) -> None:
        """Clear processed signal history."""
        self._processed.clear()


# Signal factory helpers
def telemetry_signal(kind: SignalKind, strength: float, **payload) -> Signal:
    """Create a telemetry-sourced signal."""
    return Signal(
        id=f"tel-{kind.name.lower()}-{int(time.time()*1000)}",
        source=SignalSource.TELEMETRY,
        kind=kind,
        strength=strength,
        payload=payload,
    )


def safety_signal(kind: SignalKind, strength: float = 1.0, **payload) -> Signal:
    """Create a safety-sourced signal (high priority)."""
    return Signal(
        id=f"safety-{kind.name.lower()}-{int(time.time()*1000)}",
        source=SignalSource.SAFETY,
        kind=kind,
        strength=strength,
        confidence=1.0,  # Safety signals are always confident
        ttl_ms=10000,    # Longer TTL for safety
        payload=payload,
    )


def user_signal(kind: SignalKind, strength: float = 0.9, **payload) -> Signal:
    """Create a user-sourced signal."""
    return Signal(
        id=f"user-{kind.name.lower()}-{int(time.time()*1000)}",
        source=SignalSource.USER,
        kind=kind,
        strength=strength,
        payload=payload,
    )


def ipc_signal(kind: SignalKind, strength: float, source_component: str, **payload) -> Signal:
    """Create an IPC-sourced signal from C/driver layer."""
    return Signal(
        id=f"ipc-{source_component}-{int(time.time()*1000)}",
        source=SignalSource.IPC,
        kind=kind,
        strength=strength,
        payload={"source_component": source_component, **payload},
    )
