"""
GAMESA Event Bus - Inter-component Communication System

Provides:
- Pub/sub event system for decoupled component communication
- Priority-based event queuing
- Event filtering and transformation
- Async event handlers with timeout support
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Any, Set
from enum import Enum, auto
from collections import deque
import time
import threading
import uuid


class EventPriority(Enum):
    """Event priority levels."""
    CRITICAL = 0
    HIGH = 1
    NORMAL = 2
    LOW = 3
    BACKGROUND = 4


class EventType(Enum):
    """Standard event types."""
    # System events
    SYSTEM_START = auto()
    SYSTEM_STOP = auto()
    SYSTEM_ERROR = auto()

    # Thermal events
    THERMAL_WARNING = auto()
    THERMAL_CRITICAL = auto()
    THERMAL_NORMAL = auto()

    # Resource events
    RESOURCE_LOW = auto()
    RESOURCE_EXHAUSTED = auto()
    RESOURCE_AVAILABLE = auto()

    # Cognitive events
    INSIGHT_GENERATED = auto()
    DOMAIN_ACTIVATED = auto()
    PATTERN_DETECTED = auto()

    # Trading events
    ORDER_SUBMITTED = auto()
    ORDER_FILLED = auto()
    ORDER_CANCELLED = auto()
    TIER_CHANGED = auto()

    # Learning events
    PHASE_TRANSITION = auto()
    REWARD_RECEIVED = auto()
    EXPLORATION_SPIKE = auto()

    # Custom
    CUSTOM = auto()


@dataclass
class Event:
    """Event container."""
    event_type: EventType
    source: str
    data: Dict[str, Any] = field(default_factory=dict)
    priority: EventPriority = EventPriority.NORMAL
    timestamp: float = field(default_factory=time.time)
    event_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])

    def __lt__(self, other):
        """Compare by priority for queue ordering."""
        return self.priority.value < other.priority.value


@dataclass
class Subscription:
    """Event subscription."""
    subscriber_id: str
    event_types: Set[EventType]
    handler: Callable[[Event], None]
    filter_fn: Optional[Callable[[Event], bool]] = None
    priority_threshold: EventPriority = EventPriority.BACKGROUND


class EventBus:
    """
    Central event bus for GAMESA components.

    Features:
    - Priority-based event dispatch
    - Subscription filtering
    - Event history tracking
    - Dead letter queue for failed events
    """

    def __init__(self, history_size: int = 1000):
        self._subscriptions: Dict[str, Subscription] = {}
        self._event_queue: deque = deque()
        self._history: deque = deque(maxlen=history_size)
        self._dead_letters: deque = deque(maxlen=100)
        self._lock = threading.RLock()
        self._running = False
        self._stats = {
            "events_published": 0,
            "events_delivered": 0,
            "events_dropped": 0,
            "dead_letters": 0
        }

    def subscribe(self, subscriber_id: str, event_types: Set[EventType],
                  handler: Callable[[Event], None],
                  filter_fn: Optional[Callable[[Event], bool]] = None,
                  priority_threshold: EventPriority = EventPriority.BACKGROUND) -> str:
        """
        Subscribe to events.

        Args:
            subscriber_id: Unique subscriber identifier
            event_types: Set of event types to subscribe to
            handler: Callback function for events
            filter_fn: Optional filter function
            priority_threshold: Minimum priority to receive

        Returns:
            Subscription ID
        """
        with self._lock:
            sub = Subscription(
                subscriber_id=subscriber_id,
                event_types=event_types,
                handler=handler,
                filter_fn=filter_fn,
                priority_threshold=priority_threshold
            )
            self._subscriptions[subscriber_id] = sub
            return subscriber_id

    def unsubscribe(self, subscriber_id: str) -> bool:
        """Remove subscription."""
        with self._lock:
            if subscriber_id in self._subscriptions:
                del self._subscriptions[subscriber_id]
                return True
            return False

    def publish(self, event: Event) -> int:
        """
        Publish event to all matching subscribers.

        Returns:
            Number of subscribers notified
        """
        with self._lock:
            self._stats["events_published"] += 1
            self._history.append(event)

            delivered = 0
            for sub in self._subscriptions.values():
                if self._matches(event, sub):
                    try:
                        sub.handler(event)
                        delivered += 1
                        self._stats["events_delivered"] += 1
                    except Exception as e:
                        self._dead_letters.append({
                            "event": event,
                            "subscriber": sub.subscriber_id,
                            "error": str(e),
                            "timestamp": time.time()
                        })
                        self._stats["dead_letters"] += 1

            if delivered == 0:
                self._stats["events_dropped"] += 1

            return delivered

    def publish_async(self, event: Event):
        """Queue event for async delivery."""
        with self._lock:
            self._event_queue.append(event)

    def _matches(self, event: Event, sub: Subscription) -> bool:
        """Check if event matches subscription criteria."""
        # Type match
        if event.event_type not in sub.event_types and EventType.CUSTOM not in sub.event_types:
            return False

        # Priority threshold
        if event.priority.value > sub.priority_threshold.value:
            return False

        # Custom filter
        if sub.filter_fn and not sub.filter_fn(event):
            return False

        return True

    def emit(self, event_type: EventType, source: str, data: Dict = None,
             priority: EventPriority = EventPriority.NORMAL) -> int:
        """Convenience method to create and publish event."""
        event = Event(
            event_type=event_type,
            source=source,
            data=data or {},
            priority=priority
        )
        return self.publish(event)

    def get_history(self, event_type: Optional[EventType] = None,
                    source: Optional[str] = None, limit: int = 100) -> List[Event]:
        """Get event history with optional filtering."""
        with self._lock:
            events = list(self._history)

            if event_type:
                events = [e for e in events if e.event_type == event_type]
            if source:
                events = [e for e in events if e.source == source]

            return events[-limit:]

    def get_dead_letters(self) -> List[Dict]:
        """Get failed event deliveries."""
        with self._lock:
            return list(self._dead_letters)

    def get_stats(self) -> Dict[str, int]:
        """Get bus statistics."""
        with self._lock:
            return self._stats.copy()

    def clear_history(self):
        """Clear event history."""
        with self._lock:
            self._history.clear()


class EventAggregator:
    """
    Aggregates related events over time windows.

    Useful for:
    - Batching high-frequency events
    - Detecting event patterns
    - Rate limiting notifications
    """

    def __init__(self, window_ms: int = 1000):
        self.window_ms = window_ms
        self._buffers: Dict[EventType, List[Event]] = {}
        self._window_start: Dict[EventType, float] = {}
        self._lock = threading.Lock()

    def add(self, event: Event) -> Optional[List[Event]]:
        """
        Add event to aggregation buffer.

        Returns:
            List of aggregated events if window expired, None otherwise
        """
        with self._lock:
            now = time.time() * 1000
            etype = event.event_type

            if etype not in self._buffers:
                self._buffers[etype] = []
                self._window_start[etype] = now

            self._buffers[etype].append(event)

            # Check window expiration
            if now - self._window_start[etype] >= self.window_ms:
                events = self._buffers[etype]
                self._buffers[etype] = []
                self._window_start[etype] = now
                return events

            return None

    def flush(self, event_type: EventType) -> List[Event]:
        """Force flush buffer for event type."""
        with self._lock:
            events = self._buffers.get(event_type, [])
            self._buffers[event_type] = []
            self._window_start[event_type] = time.time() * 1000
            return events


class EventRouter:
    """
    Routes events between multiple event buses.

    Enables hierarchical or federated event systems.
    """

    def __init__(self):
        self._buses: Dict[str, EventBus] = {}
        self._routes: List[tuple] = []  # (source_bus, target_bus, event_types, transform_fn)

    def register_bus(self, name: str, bus: EventBus):
        """Register an event bus."""
        self._buses[name] = bus

    def add_route(self, source: str, target: str,
                  event_types: Set[EventType],
                  transform_fn: Optional[Callable[[Event], Event]] = None):
        """Add routing rule between buses."""
        self._routes.append((source, target, event_types, transform_fn))

        # Setup forwarding subscription
        source_bus = self._buses.get(source)
        target_bus = self._buses.get(target)

        if source_bus and target_bus:
            def forward(event: Event):
                if transform_fn:
                    event = transform_fn(event)
                target_bus.publish(event)

            source_bus.subscribe(
                f"router_{source}_{target}",
                event_types,
                forward
            )

    def broadcast(self, event: Event):
        """Broadcast event to all registered buses."""
        for bus in self._buses.values():
            bus.publish(event)


# ============================================================
# GAMESA EVENT PATTERNS
# ============================================================

def create_thermal_event(source: str, temp: float, threshold: float) -> Event:
    """Create appropriate thermal event based on temperature."""
    if temp >= threshold + 15:
        etype = EventType.THERMAL_CRITICAL
        priority = EventPriority.CRITICAL
    elif temp >= threshold:
        etype = EventType.THERMAL_WARNING
        priority = EventPriority.HIGH
    else:
        etype = EventType.THERMAL_NORMAL
        priority = EventPriority.LOW

    return Event(
        event_type=etype,
        source=source,
        data={"temperature": temp, "threshold": threshold},
        priority=priority
    )


def create_resource_event(source: str, resource: str,
                          available: float, required: float) -> Event:
    """Create resource availability event."""
    ratio = available / max(required, 0.001)

    if ratio <= 0:
        etype = EventType.RESOURCE_EXHAUSTED
        priority = EventPriority.CRITICAL
    elif ratio < 0.2:
        etype = EventType.RESOURCE_LOW
        priority = EventPriority.HIGH
    else:
        etype = EventType.RESOURCE_AVAILABLE
        priority = EventPriority.NORMAL

    return Event(
        event_type=etype,
        source=source,
        data={"resource": resource, "available": available, "required": required},
        priority=priority
    )


# ============================================================
# DEMO
# ============================================================

def demo():
    """Demonstrate event bus functionality."""
    print("=== GAMESA Event Bus Demo ===\n")

    bus = EventBus()
    received = []

    # Subscribe to thermal events
    def thermal_handler(event: Event):
        received.append(event)
        print(f"  [{event.priority.name}] {event.event_type.name}: {event.data}")

    bus.subscribe(
        "thermal_monitor",
        {EventType.THERMAL_WARNING, EventType.THERMAL_CRITICAL, EventType.THERMAL_NORMAL},
        thermal_handler
    )

    # Subscribe to all high priority events
    def critical_handler(event: Event):
        print(f"  !! CRITICAL: {event.event_type.name} from {event.source}")

    bus.subscribe(
        "critical_monitor",
        set(EventType),
        critical_handler,
        priority_threshold=EventPriority.HIGH
    )

    # Publish events
    print("Publishing events:")
    bus.emit(EventType.THERMAL_NORMAL, "GPU", {"temp": 55})
    bus.emit(EventType.THERMAL_WARNING, "GPU", {"temp": 78}, EventPriority.HIGH)
    bus.emit(EventType.THERMAL_CRITICAL, "GPU", {"temp": 92}, EventPriority.CRITICAL)
    bus.emit(EventType.INSIGHT_GENERATED, "cognitive", {"insight": "pattern_found"})

    print(f"\nStats: {bus.get_stats()}")
    print(f"History size: {len(bus.get_history())}")

    # Test aggregator
    print("\n--- Aggregator Test ---")
    agg = EventAggregator(window_ms=100)

    for i in range(5):
        event = Event(EventType.REWARD_RECEIVED, "optimizer", {"reward": 0.8 + i * 0.01})
        result = agg.add(event)
        if result:
            print(f"  Window flushed: {len(result)} events")

    flushed = agg.flush(EventType.REWARD_RECEIVED)
    print(f"  Manual flush: {len(flushed)} events")


if __name__ == "__main__":
    demo()
