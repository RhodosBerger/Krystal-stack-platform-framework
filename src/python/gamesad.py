"""
GAMESA Daemon (gamesad) - Crystal-Vino Runtime Orchestrator

Container process for the Cross-Forex runtime:
1. Load libgamesa_hex (Guardian brain)
2. Spawn crystal_socketd thread
3. Initialize Guardian safety gates
4. Coordinate all agents in the trading network
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Any
from enum import Enum, auto
from collections import deque
import threading
import time

from .crystal_protocol import (
    MessageType, MarketStatus, Scenario, AgentType,
    MarketTicker, TradeOrder, Directive, DirectiveParams,
    DirectiveStatus, ProtocolCodec
)
from .crystal_socketd import CrystalSocketd, SocketMode
from .crystal_agents import (
    BaseAgent, IrisXeTrader, SiliconTrader, NeuralTrader, CacheAgent,
    create_all_agents
)
from .guardian_hex import GuardianHexEngine, GuardianMode, HexDepth


class RuntimeState(Enum):
    """Daemon runtime state."""
    INIT = auto()
    STARTING = auto()
    RUNNING = auto()
    PAUSED = auto()
    STOPPING = auto()
    STOPPED = auto()
    ERROR = auto()


@dataclass
class RuntimeConfig:
    """Runtime configuration."""
    tick_interval_ms: int = 16
    socket_mode: SocketMode = SocketMode.MEMORY
    auto_start_agents: bool = True
    guardian_mode: GuardianMode = GuardianMode.NORMAL
    enable_telemetry_logging: bool = True
    max_telemetry_history: int = 10000


@dataclass
class RuntimeMetrics:
    """Aggregated runtime metrics."""
    cycles: int = 0
    uptime_seconds: float = 0.0
    orders_processed: int = 0
    directives_issued: int = 0
    thermal_events: int = 0
    avg_tick_ms: float = 0.0


class GamesaDaemon:
    """
    GAMESA Daemon - Crystal-Vino Runtime.

    The central orchestrator that:
    - Manages crystal_socketd (market exchange)
    - Coordinates trading agents
    - Runs Guardian arbitration
    - Handles telemetry flow
    """

    VERSION = "1.0.0"
    CODENAME = "Cross-Forex"

    def __init__(self, config: Optional[RuntimeConfig] = None):
        self.config = config or RuntimeConfig()
        self.state = RuntimeState.INIT

        # Core components
        self.socketd: Optional[CrystalSocketd] = None
        self.guardian: Optional[GuardianHexEngine] = None
        self.agents: Dict[str, BaseAgent] = {}

        # Runtime state
        self._running = False
        self._cycle = 0
        self._start_time: Optional[float] = None
        self._metrics = RuntimeMetrics()

        # Threading
        self._lock = threading.RLock()
        self._tick_thread: Optional[threading.Thread] = None

        # Telemetry
        self._telemetry_history: deque = deque(maxlen=self.config.max_telemetry_history)
        self._current_telemetry: Dict = {}

        # Callbacks
        self._on_tick: List[Callable[[Dict], None]] = []
        self._on_state_change: List[Callable[[RuntimeState, RuntimeState], None]] = []

    # --------------------------------------------------------
    # LIFECYCLE
    # --------------------------------------------------------

    def start(self) -> bool:
        """
        Start the GAMESA daemon.

        Startup sequence:
        1. Initialize Guardian (Hex Engine)
        2. Spawn crystal_socketd
        3. Register and connect agents
        4. Begin market operations
        """
        with self._lock:
            if self.state != RuntimeState.INIT:
                return False

            self._set_state(RuntimeState.STARTING)
            print(f"[gamesad] Starting GAMESA v{self.VERSION} ({self.CODENAME})")

            try:
                # 1. Initialize Guardian
                self.guardian = GuardianHexEngine()
                self.guardian.mode = self.config.guardian_mode
                self.guardian.on_intervention(self._on_guardian_intervention)
                print("[gamesad] Guardian/Hex Engine initialized")

                # 2. Spawn crystal_socketd
                self.socketd = CrystalSocketd(
                    mode=self.config.socket_mode,
                    tick_interval_ms=self.config.tick_interval_ms
                )

                # Connect Guardian as order arbitrator
                self.socketd.set_guardian(self._guardian_arbitrate)

                # Register socketd callbacks
                self.socketd.on_ticker(self._on_market_ticker)
                self.socketd.on_directive(self._on_directive_issued)

                self.socketd.start()
                print("[gamesad] crystal_socketd started")

                # 3. Initialize agents
                if self.config.auto_start_agents:
                    self._init_agents()

                # 4. Begin operations
                self._running = True
                self._start_time = time.time()
                self._set_state(RuntimeState.RUNNING)

                print("[gamesad] Market OPEN - Cross-Forex trading active")
                return True

            except Exception as e:
                print(f"[gamesad] Startup failed: {e}")
                self._set_state(RuntimeState.ERROR)
                return False

    def stop(self):
        """Stop the GAMESA daemon."""
        with self._lock:
            if self.state not in [RuntimeState.RUNNING, RuntimeState.PAUSED]:
                return

            self._set_state(RuntimeState.STOPPING)
            print("[gamesad] Stopping...")

            self._running = False

            if self._tick_thread and self._tick_thread.is_alive():
                self._tick_thread.join(timeout=2.0)

            if self.socketd:
                self.socketd.stop()

            self._set_state(RuntimeState.STOPPED)
            print("[gamesad] Stopped")

    def pause(self):
        """Pause trading operations."""
        with self._lock:
            if self.state == RuntimeState.RUNNING:
                self._set_state(RuntimeState.PAUSED)
                self.socketd.set_market_status(MarketStatus.HALTED)
                print("[gamesad] Market HALTED")

    def resume(self):
        """Resume trading operations."""
        with self._lock:
            if self.state == RuntimeState.PAUSED:
                self._set_state(RuntimeState.RUNNING)
                self.socketd.set_market_status(MarketStatus.OPEN)
                print("[gamesad] Market RESUMED")

    def _set_state(self, new_state: RuntimeState):
        """Set runtime state and notify callbacks."""
        old_state = self.state
        self.state = new_state

        for callback in self._on_state_change:
            try:
                callback(old_state, new_state)
            except Exception:
                pass

    # --------------------------------------------------------
    # AGENTS
    # --------------------------------------------------------

    def _init_agents(self):
        """Initialize default trading agents."""
        self.agents = create_all_agents()

        for agent_id, agent in self.agents.items():
            # Register with socketd
            self.socketd.register_agent(
                agent_id,
                agent.agent_type.value
            )

            # Set order callback
            agent.set_order_callback(self._on_agent_order)

        print(f"[gamesad] Registered {len(self.agents)} agents")

    def register_agent(self, agent: BaseAgent) -> bool:
        """Register a custom agent."""
        with self._lock:
            if agent.agent_id in self.agents:
                return False

            self.agents[agent.agent_id] = agent

            if self.socketd:
                self.socketd.register_agent(
                    agent.agent_id,
                    agent.agent_type.value
                )

            agent.set_order_callback(self._on_agent_order)
            return True

    def unregister_agent(self, agent_id: str) -> bool:
        """Unregister an agent."""
        with self._lock:
            if agent_id not in self.agents:
                return False

            del self.agents[agent_id]

            if self.socketd:
                self.socketd.unregister_agent(agent_id)

            return True

    # --------------------------------------------------------
    # MAIN TICK LOOP
    # --------------------------------------------------------

    def tick(self, external_telemetry: Dict = None) -> Dict:
        """
        Execute one runtime tick.

        1. Ingest telemetry
        2. Run socketd tick (broadcasts ticker)
        3. Agents analyze and submit orders
        4. Guardian clears orders
        5. Directives issued to agents
        """
        if self.state != RuntimeState.RUNNING:
            return {"error": "not_running"}

        tick_start = time.time()
        self._cycle += 1
        results = {"cycle": self._cycle}

        # Update telemetry
        telemetry = self._prepare_telemetry(external_telemetry)
        self._current_telemetry = telemetry
        results["telemetry"] = telemetry

        # Run socketd tick
        socketd_result = self.socketd.tick(telemetry)
        results["socketd"] = socketd_result

        # Update Guardian metrics
        commodities = {
            "hex_compute": telemetry.get("hex_compute", 0x50),
            "hex_memory": telemetry.get("hex_memory", 0x30),
            "thermal_headroom_gpu": telemetry.get("thermal_headroom_gpu", 15),
            "thermal_headroom_cpu": telemetry.get("thermal_headroom_cpu", 12),
        }
        self.guardian.update_metrics(commodities)
        results["guardian"] = self.guardian.get_metrics()

        # Calculate tick duration
        tick_duration = (time.time() - tick_start) * 1000
        self._update_metrics(tick_duration, socketd_result)
        results["tick_ms"] = tick_duration

        # Store telemetry
        if self.config.enable_telemetry_logging:
            self._telemetry_history.append({
                "cycle": self._cycle,
                "ts": time.time(),
                "telemetry": telemetry,
                "guardian": results["guardian"]
            })

        # Callbacks
        for callback in self._on_tick:
            try:
                callback(results)
            except Exception:
                pass

        return results

    def run(self, cycles: int = 0, telemetry_fn: Callable[[], Dict] = None) -> List[Dict]:
        """
        Run multiple ticks.

        Args:
            cycles: Number of cycles (0 = infinite)
            telemetry_fn: Function to generate telemetry each tick
        """
        results = []
        count = 0

        while self._running:
            if cycles > 0 and count >= cycles:
                break

            telemetry = telemetry_fn() if telemetry_fn else None
            result = self.tick(telemetry)
            results.append(result)
            count += 1

            # Sleep to maintain tick rate
            sleep_time = self.config.tick_interval_ms / 1000.0 - result.get("tick_ms", 0) / 1000.0
            if sleep_time > 0:
                time.sleep(sleep_time)

        return results

    def _prepare_telemetry(self, external: Dict = None) -> Dict:
        """Prepare telemetry for tick."""
        telemetry = {
            "cycle": self._cycle,
            "cpu_util": 0.5,
            "gpu_util": 0.6,
            "cpu_temp": 60,
            "gpu_temp": 70,
            "power_draw": 150,
            "memory_util": 0.4,
            "io_util": 0.3,
        }

        if external:
            telemetry.update(external)

        # Derive hex values
        telemetry["hex_compute"] = int(max(telemetry.get("cpu_util", 0.5),
                                            telemetry.get("gpu_util", 0.6)) * 255)
        telemetry["hex_memory"] = int(telemetry.get("memory_util", 0.4) * 255)
        telemetry["hex_io"] = int(telemetry.get("io_util", 0.3) * 255)
        telemetry["thermal_headroom_gpu"] = max(0, 85 - telemetry.get("gpu_temp", 70))
        telemetry["thermal_headroom_cpu"] = max(0, 80 - telemetry.get("cpu_temp", 60))

        return telemetry

    def _update_metrics(self, tick_ms: float, socketd_result: Dict):
        """Update runtime metrics."""
        self._metrics.cycles = self._cycle
        self._metrics.uptime_seconds = time.time() - self._start_time if self._start_time else 0
        self._metrics.orders_processed += socketd_result.get("orders_processed", 0)
        self._metrics.directives_issued += socketd_result.get("directives_issued", 0)

        # Running average tick time
        alpha = 0.1
        self._metrics.avg_tick_ms = (
            (1 - alpha) * self._metrics.avg_tick_ms + alpha * tick_ms
        )

    # --------------------------------------------------------
    # CALLBACKS & HANDLERS
    # --------------------------------------------------------

    def _on_market_ticker(self, ticker: MarketTicker):
        """Handle market ticker broadcast - notify agents."""
        for agent in self.agents.values():
            try:
                agent.on_ticker(ticker)
            except Exception:
                pass

    def _on_agent_order(self, order: TradeOrder):
        """Handle order from agent."""
        if self.socketd:
            self.socketd.submit_order(order)

    def _guardian_arbitrate(self, order: TradeOrder) -> Directive:
        """Guardian arbitration callback."""
        result = self.guardian.clear_order(order, self._current_telemetry)

        params = DirectiveParams(duration_ms=5000)

        # Apply adjustments from clearing
        if result.adjustments:
            for key, val in result.adjustments.items():
                if hasattr(params, key):
                    setattr(params, key, val)

        return Directive(
            target=order.source,
            permit_id=order.order_id,
            status=DirectiveStatus.APPROVED.value if result.approved else DirectiveStatus.DENIED.value,
            params=params,
            reason=result.reason,
            expires_at=int(time.time() * 1000000) + 5000000
        )

    def _on_directive_issued(self, directive: Directive):
        """Handle directive issued to agent."""
        agent = self.agents.get(directive.target)
        if agent:
            agent.on_directive(directive)

    def _on_guardian_intervention(self, intervention: Dict):
        """Handle Guardian intervention."""
        self._metrics.thermal_events += 1
        print(f"[gamesad] Guardian intervention: {intervention['type']}")

    # --------------------------------------------------------
    # PUBLIC API
    # --------------------------------------------------------

    def set_scenario(self, scenario: Scenario):
        """Set current scenario."""
        if self.socketd:
            self.socketd.set_scenario(scenario)

    def emergency_stop(self):
        """Trigger emergency stop."""
        if self.guardian:
            self.guardian.emergency_halt()
        if self.socketd:
            self.socketd.set_market_status(MarketStatus.EMERGENCY)
        self.pause()

    def on_tick(self, callback: Callable[[Dict], None]):
        """Register tick callback."""
        self._on_tick.append(callback)

    def on_state_change(self, callback: Callable[[RuntimeState, RuntimeState], None]):
        """Register state change callback."""
        self._on_state_change.append(callback)

    def get_state(self) -> Dict:
        """Get comprehensive runtime state."""
        return {
            "version": self.VERSION,
            "codename": self.CODENAME,
            "state": self.state.name,
            "metrics": {
                "cycles": self._metrics.cycles,
                "uptime": self._metrics.uptime_seconds,
                "orders": self._metrics.orders_processed,
                "directives": self._metrics.directives_issued,
                "thermal_events": self._metrics.thermal_events,
                "avg_tick_ms": self._metrics.avg_tick_ms
            },
            "guardian": self.guardian.get_metrics() if self.guardian else {},
            "socketd": self.socketd.get_stats() if self.socketd else {},
            "agents": [a.get_stats() for a in self.agents.values()]
        }

    def report(self) -> str:
        """Generate runtime report."""
        state = self.get_state()
        return f"""
{'='*60}
GAMESA DAEMON REPORT - {self.CODENAME}
{'='*60}

Version: {state['version']}
State: {state['state']}
Uptime: {state['metrics']['uptime']:.1f}s

METRICS:
  Cycles: {state['metrics']['cycles']}
  Orders Processed: {state['metrics']['orders']}
  Directives Issued: {state['metrics']['directives']}
  Thermal Events: {state['metrics']['thermal_events']}
  Avg Tick: {state['metrics']['avg_tick_ms']:.2f}ms

GUARDIAN:
  Mode: {state['guardian'].get('mode', 'N/A')}
  Hex Depth: {state['guardian'].get('hex_depth', 'N/A')}
  Approval Rate: {state['guardian'].get('approval_rate', 0)*100:.1f}%
  Stability: {state['guardian'].get('stability_score', 0):.2f}

AGENTS:
"""  + '\n'.join([f"  {a['agent_id']}: orders={a['metrics']['orders']}, approved={a['metrics']['approved']}"
                 for a in state['agents']])


# ============================================================
# FACTORY
# ============================================================

def create_gamesad(config: RuntimeConfig = None) -> GamesaDaemon:
    """Create GAMESA daemon instance."""
    return GamesaDaemon(config)


# ============================================================
# DEMO
# ============================================================

def demo():
    """Demonstrate GAMESA daemon."""
    import random

    print("=" * 60)
    print("GAMESA DAEMON - Cross-Forex Runtime")
    print("=" * 60)

    daemon = create_gamesad()
    daemon.start()

    # Set gaming scenario
    daemon.set_scenario(Scenario.GAME_COMBAT)

    print("\nRunning 20 market ticks...\n")

    def generate_telemetry():
        return {
            "cpu_util": 0.5 + random.gauss(0, 0.1),
            "gpu_util": 0.7 + random.gauss(0, 0.1),
            "cpu_temp": 60 + random.gauss(0, 5),
            "gpu_temp": 70 + random.gauss(0, 8),
            "power_draw": 150 + random.gauss(0, 20),
            "memory_util": 0.4 + random.gauss(0, 0.05),
            "io_util": 0.3 + random.gauss(0, 0.05)
        }

    for i in range(20):
        result = daemon.tick(generate_telemetry())

        if i % 5 == 0:
            print(f"Tick {i}: cycle={result['cycle']}, "
                  f"hex_depth={result['guardian']['hex_depth']}, "
                  f"tick_ms={result['tick_ms']:.2f}")

    print("\n" + daemon.report())
    daemon.stop()


if __name__ == "__main__":
    demo()
