"""
Crystal-Vino Socket Control Unit (crystal_socketd)

The central nervous system of GAMESA Cross-Forex runtime:
- Ingests telemetry from kernel, Vulkan, OS sources
- Broadcasts market ticker to all agents
- Arbitrates trade orders through Guardian logic
- Issues directives to agents
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Set, Any
from enum import Enum, auto
from collections import deque
import threading
import socket
import select
import time
import json
import os

from .crystal_protocol import (
    MessageType, MarketStatus, Scenario, AgentType, DirectiveStatus,
    MarketTicker, TradeOrder, Directive, CommodityPrices, MarketState,
    AgentRegistration, ProtocolCodec, DirectiveParams
)


class SocketMode(Enum):
    """Socket communication mode."""
    UNIX = auto()      # Unix domain socket
    TCP = auto()       # TCP socket
    MEMORY = auto()    # In-memory (for testing)


@dataclass
class ConnectedAgent:
    """Represents a connected agent."""
    agent_id: str
    agent_type: str
    socket: Optional[Any] = None
    capabilities: List[str] = field(default_factory=list)
    subscriptions: List[str] = field(default_factory=list)
    connected_at: float = field(default_factory=time.time)
    last_heartbeat: float = field(default_factory=time.time)
    orders_submitted: int = 0
    directives_received: int = 0


class CrystalSocketd:
    """
    Crystal-Vino Socket Control Unit.

    Central hub for Cross-Forex resource market:
    - Broadcasts market ticker every 16-50ms
    - Receives trade orders from agents
    - Routes directives from Guardian
    """

    SOCKET_PATH = "/tmp/gamesa_crystal.sock"
    TICK_INTERVAL_MS = 16
    HEARTBEAT_TIMEOUT_S = 5.0

    def __init__(self, mode: SocketMode = SocketMode.MEMORY,
                 tick_interval_ms: int = 16):
        self.mode = mode
        self.tick_interval_ms = tick_interval_ms

        # State
        self._running = False
        self._cycle = 0
        self._market_status = MarketStatus.CLOSED
        self._scenario = Scenario.IDLE

        # Connected agents
        self._agents: Dict[str, ConnectedAgent] = {}
        self._agent_lock = threading.RLock()

        # Current commodity prices
        self._commodities = CommodityPrices()

        # Order book
        self._pending_orders: deque = deque(maxlen=1000)
        self._processed_orders: deque = deque(maxlen=10000)

        # Directive queue
        self._directive_queue: deque = deque(maxlen=1000)

        # Guardian callback
        self._guardian_callback: Optional[Callable[[TradeOrder], Directive]] = None

        # Event callbacks
        self._on_ticker: List[Callable[[MarketTicker], None]] = []
        self._on_order: List[Callable[[TradeOrder], None]] = []
        self._on_directive: List[Callable[[Directive], None]] = []

        # Socket (for non-memory modes)
        self._server_socket: Optional[socket.socket] = None
        self._client_sockets: List[socket.socket] = []

        # Telemetry buffer
        self._telemetry_buffer: deque = deque(maxlen=10000)

        # Stats
        self._stats = {
            "tickers_broadcast": 0,
            "orders_received": 0,
            "directives_issued": 0,
            "agents_connected": 0
        }

    # --------------------------------------------------------
    # LIFECYCLE
    # --------------------------------------------------------

    def start(self):
        """Start the socket control unit."""
        self._running = True
        self._market_status = MarketStatus.OPEN

        if self.mode == SocketMode.UNIX:
            self._start_unix_socket()
        elif self.mode == SocketMode.TCP:
            self._start_tcp_socket()

        print(f"[crystal_socketd] Started (mode={self.mode.name})")

    def stop(self):
        """Stop the socket control unit."""
        self._running = False
        self._market_status = MarketStatus.CLOSED

        if self._server_socket:
            self._server_socket.close()
            if self.mode == SocketMode.UNIX and os.path.exists(self.SOCKET_PATH):
                os.unlink(self.SOCKET_PATH)

        for sock in self._client_sockets:
            sock.close()

        print("[crystal_socketd] Stopped")

    def _start_unix_socket(self):
        """Initialize Unix domain socket."""
        if os.path.exists(self.SOCKET_PATH):
            os.unlink(self.SOCKET_PATH)

        self._server_socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._server_socket.bind(self.SOCKET_PATH)
        self._server_socket.listen(10)
        self._server_socket.setblocking(False)

    def _start_tcp_socket(self):
        """Initialize TCP socket."""
        self._server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._server_socket.bind(('127.0.0.1', 9876))
        self._server_socket.listen(10)
        self._server_socket.setblocking(False)

    # --------------------------------------------------------
    # AGENT MANAGEMENT
    # --------------------------------------------------------

    def register_agent(self, agent_id: str, agent_type: str,
                       capabilities: List[str] = None,
                       socket_conn: Any = None) -> bool:
        """Register a new agent."""
        with self._agent_lock:
            if agent_id in self._agents:
                return False

            agent = ConnectedAgent(
                agent_id=agent_id,
                agent_type=agent_type,
                socket=socket_conn,
                capabilities=capabilities or []
            )
            self._agents[agent_id] = agent
            self._stats["agents_connected"] = len(self._agents)
            return True

    def unregister_agent(self, agent_id: str):
        """Unregister an agent."""
        with self._agent_lock:
            if agent_id in self._agents:
                del self._agents[agent_id]
                self._stats["agents_connected"] = len(self._agents)

    def get_agents(self) -> List[Dict]:
        """Get list of connected agents."""
        with self._agent_lock:
            return [
                {
                    "id": a.agent_id,
                    "type": a.agent_type,
                    "connected_at": a.connected_at,
                    "orders": a.orders_submitted
                }
                for a in self._agents.values()
            ]

    # --------------------------------------------------------
    # MARKET OPERATIONS
    # --------------------------------------------------------

    def update_commodities(self, **kwargs):
        """Update commodity prices."""
        for key, value in kwargs.items():
            if hasattr(self._commodities, key):
                setattr(self._commodities, key, value)

    def set_scenario(self, scenario: Scenario):
        """Set current scenario."""
        self._scenario = scenario

    def set_market_status(self, status: MarketStatus):
        """Set market status."""
        self._market_status = status

    def set_guardian(self, callback: Callable[[TradeOrder], Directive]):
        """Set Guardian callback for order arbitration."""
        self._guardian_callback = callback

    # --------------------------------------------------------
    # MAIN TICK
    # --------------------------------------------------------

    def tick(self, external_telemetry: Dict = None) -> Dict:
        """
        Execute one market tick.

        1. Update commodities from telemetry
        2. Broadcast market ticker
        3. Process pending orders
        4. Issue directives
        """
        self._cycle += 1
        results = {"cycle": self._cycle, "timestamp": time.time()}

        # Update from external telemetry
        if external_telemetry:
            self._ingest_telemetry(external_telemetry)

        # Create and broadcast ticker
        ticker = self._create_ticker()
        self._broadcast_ticker(ticker)
        results["ticker"] = ticker.to_json()

        # Process orders
        orders_processed = self._process_orders()
        results["orders_processed"] = orders_processed

        # Issue pending directives
        directives_issued = self._issue_directives()
        results["directives_issued"] = directives_issued

        # Clean up stale agents
        self._cleanup_stale_agents()

        return results

    def _ingest_telemetry(self, telemetry: Dict):
        """Ingest telemetry and update commodities."""
        self._telemetry_buffer.append({
            "cycle": self._cycle,
            "ts": time.time(),
            "data": telemetry
        })

        # Map telemetry to commodity prices
        mappings = {
            "cpu_util": ("hex_compute", lambda v: int(v * 255)),
            "gpu_util": ("hex_compute", lambda v: int(v * 255)),
            "memory_util": ("hex_memory", lambda v: int(v * 255)),
            "io_util": ("hex_io", lambda v: int(v * 255)),
            "gpu_temp": ("thermal_headroom_gpu", lambda v: max(0, 85 - v)),
            "cpu_temp": ("thermal_headroom_cpu", lambda v: max(0, 80 - v)),
        }

        for tel_key, (comm_key, transform) in mappings.items():
            if tel_key in telemetry:
                value = transform(telemetry[tel_key])
                if hasattr(self._commodities, comm_key):
                    setattr(self._commodities, comm_key, value)

    def _create_ticker(self) -> MarketTicker:
        """Create market ticker message."""
        return MarketTicker(
            ts=int(time.time() * 1000000),
            cycle=self._cycle,
            commodities=self._commodities,
            state=MarketState(
                scenario=self._scenario.value,
                market_status=self._market_status.value,
                tick_interval_ms=self.tick_interval_ms
            )
        )

    def _broadcast_ticker(self, ticker: MarketTicker):
        """Broadcast ticker to all agents."""
        self._stats["tickers_broadcast"] += 1

        # Memory mode - call callbacks
        for callback in self._on_ticker:
            try:
                callback(ticker)
            except Exception:
                pass

        # Socket mode - send to all connected sockets
        if self.mode != SocketMode.MEMORY:
            data = ticker.to_json().encode('utf-8') + b'\n'
            for sock in self._client_sockets[:]:
                try:
                    sock.send(data)
                except Exception:
                    self._client_sockets.remove(sock)

    def _process_orders(self) -> int:
        """Process pending trade orders."""
        processed = 0

        while self._pending_orders:
            order = self._pending_orders.popleft()

            # Arbitrate through Guardian
            directive = self._arbitrate_order(order)

            if directive:
                self._directive_queue.append(directive)

            self._processed_orders.append({
                "order": order,
                "directive": directive,
                "processed_at": time.time()
            })
            processed += 1

            # Call order callbacks
            for callback in self._on_order:
                try:
                    callback(order)
                except Exception:
                    pass

        return processed

    def _arbitrate_order(self, order: TradeOrder) -> Optional[Directive]:
        """Arbitrate order through Guardian."""
        if self._guardian_callback:
            return self._guardian_callback(order)

        # Default approval logic
        return Directive(
            target=order.source,
            permit_id=order.order_id,
            status=DirectiveStatus.APPROVED.value,
            params=DirectiveParams(duration_ms=5000)
        )

    def _issue_directives(self) -> int:
        """Issue pending directives to agents."""
        issued = 0

        while self._directive_queue:
            directive = self._directive_queue.popleft()

            # Find target agent
            with self._agent_lock:
                agent = self._agents.get(directive.target)
                if agent:
                    agent.directives_received += 1

            self._stats["directives_issued"] += 1
            issued += 1

            # Call directive callbacks
            for callback in self._on_directive:
                try:
                    callback(directive)
                except Exception:
                    pass

        return issued

    def _cleanup_stale_agents(self):
        """Remove agents that haven't sent heartbeat."""
        now = time.time()
        with self._agent_lock:
            stale = [
                aid for aid, agent in self._agents.items()
                if now - agent.last_heartbeat > self.HEARTBEAT_TIMEOUT_S
            ]
            for aid in stale:
                del self._agents[aid]

    # --------------------------------------------------------
    # API
    # --------------------------------------------------------

    def submit_order(self, order: TradeOrder):
        """Submit a trade order for processing."""
        self._pending_orders.append(order)
        self._stats["orders_received"] += 1

        with self._agent_lock:
            agent = self._agents.get(order.source)
            if agent:
                agent.orders_submitted += 1

    def on_ticker(self, callback: Callable[[MarketTicker], None]):
        """Register ticker callback."""
        self._on_ticker.append(callback)

    def on_order(self, callback: Callable[[TradeOrder], None]):
        """Register order callback."""
        self._on_order.append(callback)

    def on_directive(self, callback: Callable[[Directive], None]):
        """Register directive callback."""
        self._on_directive.append(callback)

    def get_stats(self) -> Dict:
        """Get socket control unit statistics."""
        return {
            **self._stats,
            "cycle": self._cycle,
            "market_status": self._market_status.value,
            "scenario": self._scenario.value,
            "pending_orders": len(self._pending_orders),
            "commodities": self._commodities.to_dict()
        }


# ============================================================
# DEMO
# ============================================================

def demo():
    """Demonstrate crystal_socketd."""
    print("=== Crystal-Vino Socket Control Unit Demo ===\n")

    socketd = CrystalSocketd(mode=SocketMode.MEMORY)
    socketd.start()

    # Register agents
    socketd.register_agent("AGENT_IRIS_XE", "GPU")
    socketd.register_agent("AGENT_SILICON", "CPU")
    socketd.register_agent("AGENT_NEURAL", "NPU")

    print(f"Agents: {socketd.get_agents()}")

    # Set scenario
    socketd.set_scenario(Scenario.GAME_COMBAT)

    # Ticker callback
    def on_ticker(ticker):
        print(f"  Ticker cycle={ticker.cycle}, scenario={ticker.state.scenario}")

    socketd.on_ticker(on_ticker)

    # Run ticks with telemetry
    print("\nRunning 5 market ticks:")
    for i in range(5):
        telemetry = {
            "cpu_util": 0.6 + i * 0.05,
            "gpu_util": 0.7 + i * 0.03,
            "gpu_temp": 65 + i * 2
        }

        # Submit order on tick 2
        if i == 2:
            order = TradeOrder(
                source="AGENT_IRIS_XE",
                action="REQUEST_FP16_OVERRIDE",
                bid={"reason": "HEX_COMPUTE_HIGH", "est_thermal_saving": 2.0, "priority": 7}
            )
            socketd.submit_order(order)
            print(f"  -> Submitted order: {order.order_id}")

        result = socketd.tick(telemetry)

    print(f"\nStats: {socketd.get_stats()}")
    socketd.stop()


if __name__ == "__main__":
    demo()
