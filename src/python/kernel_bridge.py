"""
Kernel Bridge - Python interface to GAMESA Kernel

Provides:
- Component lifecycle management
- River (data stream) creation and routing
- Cross-language IPC multiplexing
- Driver registration
"""

import ctypes
import json
import time
import logging
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import IntEnum
from threading import Thread, Event, Lock
from queue import Queue

logger = logging.getLogger(__name__)


class ComponentType(IntEnum):
    GUARDIAN = 0
    RUST_BOT = 1
    THREAD_BOOST = 2
    RPG_CRAFT = 3
    OPENVINO = 4
    MESA_DRIVER = 5
    DMA_MODULE = 6
    TWEAKER_UI = 7
    CUSTOM = 8


class ComponentState(IntEnum):
    INIT = 0
    RUNNING = 1
    PAUSED = 2
    ERROR = 3
    STOPPED = 4


class RiverTransport(IntEnum):
    SHMEM = 0       # Shared memory (fastest)
    SOCKET = 1      # Unix socket
    PIPE = 2        # Named pipe
    QUEUE = 3       # Message queue
    DMA = 4         # DMA buffer (GPU)


class RiverDirection(IntEnum):
    UNIDIRECTIONAL = 0
    BIDIRECTIONAL = 1


@dataclass
class RiverQoS:
    """Quality of Service settings for rivers."""
    max_latency_us: int = 1000
    min_throughput_bps: int = 0
    priority: int = 128
    lossy: bool = False
    ordered: bool = True


@dataclass
class Component:
    """Component descriptor."""
    id: int
    name: str
    type: ComponentType
    state: ComponentState = ComponentState.INIT
    pid: int = 0
    start_time: float = 0.0
    cpu_time: float = 0.0
    restart_count: int = 0
    context: Any = None


@dataclass
class River:
    """Data stream between components."""
    id: int
    name: str
    source_id: int
    sink_id: int
    direction: RiverDirection
    transport: RiverTransport
    qos: RiverQoS
    active: bool = False
    bytes_transferred: int = 0
    messages_sent: int = 0
    messages_dropped: int = 0


@dataclass
class RiverMessage:
    """Message flowing through a river."""
    type: int
    data: bytes
    timestamp: float
    source_id: int
    sequence: int


# Message types
MSG_TELEMETRY = 0x0001
MSG_SIGNAL = 0x0002
MSG_DECISION = 0x0003
MSG_PRESET = 0x0004
MSG_EVENT = 0x0005
MSG_HEARTBEAT = 0x00FF


class RiverEndpoint:
    """Endpoint for sending/receiving on a river."""

    def __init__(self, river: River, is_source: bool):
        self.river = river
        self.is_source = is_source
        self._queue: Queue = Queue(maxsize=1000)
        self._sequence = 0
        self._lock = Lock()

    def send(self, msg_type: int, data: bytes) -> bool:
        """Send message on river."""
        if not self.river.active:
            return False

        with self._lock:
            self._sequence += 1
            msg = RiverMessage(
                type=msg_type,
                data=data,
                timestamp=time.time(),
                source_id=self.river.source_id,
                sequence=self._sequence,
            )

            try:
                self._queue.put_nowait(msg)
                self.river.messages_sent += 1
                self.river.bytes_transferred += len(data)
                return True
            except:
                self.river.messages_dropped += 1
                return False

    def recv(self, timeout: float = 1.0) -> Optional[RiverMessage]:
        """Receive message from river."""
        try:
            return self._queue.get(timeout=timeout)
        except:
            return None

    def recv_nowait(self) -> Optional[RiverMessage]:
        """Non-blocking receive."""
        try:
            return self._queue.get_nowait()
        except:
            return None


class GamesaKernel:
    """
    Python-side kernel for component and river management.

    Manages:
    - Component lifecycle (start, stop, monitor)
    - River creation and message routing
    - Cross-language bridges (Python ↔ Rust ↔ C)
    """

    def __init__(self):
        self.components: Dict[int, Component] = {}
        self.rivers: Dict[int, River] = {}
        self.endpoints: Dict[int, RiverEndpoint] = {}

        self._next_comp_id = 1
        self._next_river_id = 1
        self._running = False
        self._lock = Lock()

        self._scheduler_thread: Optional[Thread] = None
        self._monitor_thread: Optional[Thread] = None
        self._stop_event = Event()

        self._callbacks: Dict[str, List[Callable]] = {}
        self.boot_time = time.time()

    # Component Management

    def register_component(
        self,
        name: str,
        comp_type: ComponentType,
        context: Any = None,
    ) -> int:
        """Register a new component."""
        with self._lock:
            comp_id = self._next_comp_id
            self._next_comp_id += 1

            comp = Component(
                id=comp_id,
                name=name,
                type=comp_type,
                context=context,
            )
            self.components[comp_id] = comp

            logger.info(f"Registered component: {name} (id={comp_id})")
            return comp_id

    def start_component(self, comp_id: int) -> bool:
        """Start a component."""
        comp = self.components.get(comp_id)
        if not comp:
            return False

        comp.state = ComponentState.RUNNING
        comp.start_time = time.time()
        logger.info(f"Started component: {comp.name}")
        self._emit("component_start", comp)
        return True

    def stop_component(self, comp_id: int) -> bool:
        """Stop a component."""
        comp = self.components.get(comp_id)
        if not comp:
            return False

        comp.state = ComponentState.STOPPED
        logger.info(f"Stopped component: {comp.name}")
        self._emit("component_stop", comp)
        return True

    def get_component(self, comp_id: int) -> Optional[Component]:
        """Get component by ID."""
        return self.components.get(comp_id)

    def find_component(self, name: str) -> Optional[Component]:
        """Find component by name."""
        for comp in self.components.values():
            if comp.name == name:
                return comp
        return None

    # River Management

    def create_river(
        self,
        name: str,
        source_id: int,
        sink_id: int,
        transport: RiverTransport = RiverTransport.QUEUE,
        qos: Optional[RiverQoS] = None,
        direction: RiverDirection = RiverDirection.UNIDIRECTIONAL,
    ) -> int:
        """Create a data river between components."""
        with self._lock:
            river_id = self._next_river_id
            self._next_river_id += 1

            river = River(
                id=river_id,
                name=name,
                source_id=source_id,
                sink_id=sink_id,
                direction=direction,
                transport=transport,
                qos=qos or RiverQoS(),
                active=True,
            )
            self.rivers[river_id] = river

            # Create endpoints
            self.endpoints[river_id] = RiverEndpoint(river, is_source=True)

            logger.info(f"Created river: {name} ({source_id} → {sink_id})")
            return river_id

    def destroy_river(self, river_id: int) -> bool:
        """Destroy a river."""
        with self._lock:
            if river_id in self.rivers:
                river = self.rivers[river_id]
                river.active = False
                del self.rivers[river_id]
                del self.endpoints[river_id]
                logger.info(f"Destroyed river: {river.name}")
                return True
        return False

    def get_river_endpoint(self, river_id: int) -> Optional[RiverEndpoint]:
        """Get endpoint for a river."""
        return self.endpoints.get(river_id)

    def send_on_river(self, river_id: int, msg_type: int, data: bytes) -> bool:
        """Send data on a river."""
        endpoint = self.endpoints.get(river_id)
        if endpoint:
            return endpoint.send(msg_type, data)
        return False

    def recv_from_river(self, river_id: int, timeout: float = 1.0) -> Optional[RiverMessage]:
        """Receive data from a river."""
        endpoint = self.endpoints.get(river_id)
        if endpoint:
            return endpoint.recv(timeout)
        return None

    # Standard Rivers

    def create_telemetry_river(self, source_id: int) -> int:
        """Create standard telemetry river (high frequency, lossy ok)."""
        return self.create_river(
            name=f"telemetry_{source_id}",
            source_id=source_id,
            sink_id=0,  # Broadcast
            transport=RiverTransport.SHMEM,
            qos=RiverQoS(
                max_latency_us=500,
                priority=200,
                lossy=True,
                ordered=False,
            ),
        )

    def create_signal_river(self, sink_id: int) -> int:
        """Create standard signal river (reliable, ordered)."""
        return self.create_river(
            name=f"signals_{sink_id}",
            source_id=0,  # Any source
            sink_id=sink_id,
            transport=RiverTransport.QUEUE,
            qos=RiverQoS(
                max_latency_us=1000,
                priority=255,
                lossy=False,
                ordered=True,
            ),
        )

    def create_decision_river(self, sink_id: int) -> int:
        """Create standard decision river (high priority, reliable)."""
        return self.create_river(
            name=f"decisions_{sink_id}",
            source_id=0,
            sink_id=sink_id,
            transport=RiverTransport.QUEUE,
            qos=RiverQoS(
                max_latency_us=100,
                priority=255,
                lossy=False,
                ordered=True,
            ),
        )

    # Kernel Lifecycle

    def start(self):
        """Start the kernel."""
        if self._running:
            return

        self._running = True
        self._stop_event.clear()

        # Start scheduler thread
        self._scheduler_thread = Thread(target=self._scheduler_loop, daemon=True)
        self._scheduler_thread.start()

        # Start monitor thread
        self._monitor_thread = Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()

        logger.info("GAMESA Kernel started")

    def stop(self):
        """Stop the kernel."""
        self._stop_event.set()

        if self._scheduler_thread:
            self._scheduler_thread.join(timeout=2.0)
        if self._monitor_thread:
            self._monitor_thread.join(timeout=2.0)

        self._running = False
        logger.info("GAMESA Kernel stopped")

    def _scheduler_loop(self):
        """Main scheduler loop."""
        while not self._stop_event.is_set():
            # Process rivers with QoS priorities
            for river in sorted(self.rivers.values(), key=lambda r: -r.qos.priority):
                if not river.active:
                    continue
                # Route messages based on QoS
                # (In real impl, would handle transport-specific routing)

            time.sleep(0.001)  # 1ms tick

    def _monitor_loop(self):
        """Component health monitoring."""
        while not self._stop_event.is_set():
            for comp in self.components.values():
                if comp.state == ComponentState.RUNNING:
                    # Health check (placeholder)
                    pass

            time.sleep(1.0)  # 1s health check interval

    # Events

    def on(self, event: str, callback: Callable):
        """Register event callback."""
        if event not in self._callbacks:
            self._callbacks[event] = []
        self._callbacks[event].append(callback)

    def _emit(self, event: str, data: Any):
        """Emit event to callbacks."""
        for cb in self._callbacks.get(event, []):
            try:
                cb(data)
            except Exception as e:
                logger.error(f"Callback error: {e}")

    # Statistics

    def get_stats(self) -> Dict[str, Any]:
        """Get kernel statistics."""
        return {
            "uptime": time.time() - self.boot_time,
            "components": len(self.components),
            "rivers": len(self.rivers),
            "running": self._running,
            "component_states": {
                c.name: c.state.name for c in self.components.values()
            },
            "river_stats": {
                r.name: {
                    "bytes": r.bytes_transferred,
                    "messages": r.messages_sent,
                    "dropped": r.messages_dropped,
                }
                for r in self.rivers.values()
            },
        }


# Factory
def create_kernel() -> GamesaKernel:
    """Create and initialize GAMESA kernel."""
    kernel = GamesaKernel()
    return kernel


def create_full_stack_kernel() -> GamesaKernel:
    """Create kernel with all standard components registered."""
    kernel = GamesaKernel()

    # Register standard components
    guardian_id = kernel.register_component("guardian", ComponentType.GUARDIAN)
    rust_bot_id = kernel.register_component("rust_bot", ComponentType.RUST_BOT)
    thread_boost_id = kernel.register_component("thread_boost", ComponentType.THREAD_BOOST)
    rpg_craft_id = kernel.register_component("rpg_craft", ComponentType.RPG_CRAFT)
    openvino_id = kernel.register_component("openvino", ComponentType.OPENVINO)

    # Create standard rivers
    kernel.create_telemetry_river(thread_boost_id)  # C → Python/Rust
    kernel.create_signal_river(thread_boost_id)      # Python → C
    kernel.create_decision_river(thread_boost_id)    # Rust → C

    # Create component interconnects
    kernel.create_river("guardian_to_rust", guardian_id, rust_bot_id)
    kernel.create_river("rust_to_guardian", rust_bot_id, guardian_id)
    kernel.create_river("openvino_presets", guardian_id, openvino_id)

    return kernel
