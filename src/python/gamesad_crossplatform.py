"""
GAMESA Cross-Platform Daemon (gamesad_crossplatform)

Unified Crystal-Vino runtime for Intel, AMD, and ARM:
- Platform-aware telemetry collection
- Cross-platform trading agents
- Unified accelerator management
- Hardware-abstracted Guardian
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Any
from enum import Enum, auto
import threading
import time

from .crystal_protocol import (
    MessageType, MarketStatus, Scenario, DirectiveStatus,
    MarketTicker, TradeOrder, Directive, DirectiveParams,
    ProtocolCodec
)
from .crystal_socketd import CrystalSocketd, SocketMode
from .guardian_hex import GuardianHexEngine, GuardianMode, HexDepth
from .platform_hal import BaseHAL, HALFactory, Vendor, AcceleratorType
from .platform_telemetry import TelemetryAggregator, TelemetrySample
from .crossplatform_agents import (
    CrossPlatformGPUTrader, CrossPlatformCPUTrader, CrossPlatformNPUTrader,
    CrossPlatformAgentFactory, BaseAgent
)
from .accelerator_manager import AcceleratorManager, WorkloadRequest, WorkloadType, PrecisionMode


class RuntimeState(Enum):
    """Runtime state."""
    INIT = auto()
    STARTING = auto()
    RUNNING = auto()
    PAUSED = auto()
    STOPPING = auto()
    STOPPED = auto()
    ERROR = auto()


@dataclass
class CrossPlatformConfig:
    """Cross-platform runtime configuration."""
    tick_interval_ms: int = 16
    socket_mode: SocketMode = SocketMode.MEMORY
    guardian_mode: GuardianMode = GuardianMode.NORMAL
    enable_telemetry: bool = True
    enable_accelerator_routing: bool = True
    auto_precision_scaling: bool = True
    thermal_throttle_threshold: float = 10.0


@dataclass
class RuntimeMetrics:
    """Runtime metrics."""
    cycles: int = 0
    uptime_seconds: float = 0.0
    orders_processed: int = 0
    directives_issued: int = 0
    precision_switches: int = 0
    thermal_events: int = 0
    platform: str = ""


class GamesadCrossPlatform:
    """
    Cross-Platform GAMESA Daemon.

    Integrates:
    - Platform HAL for hardware abstraction
    - Cross-platform telemetry collection
    - Platform-aware trading agents
    - Unified accelerator management
    - Crystal-Vino market exchange
    """

    VERSION = "1.0.0"
    CODENAME = "Cross-Forex Universal"

    def __init__(self, config: Optional[CrossPlatformConfig] = None):
        self.config = config or CrossPlatformConfig()
        self.state = RuntimeState.INIT

        # Platform abstraction
        self.hal: Optional[BaseHAL] = None

        # Core components
        self.socketd: Optional[CrystalSocketd] = None
        self.guardian: Optional[GuardianHexEngine] = None
        self.telemetry: Optional[TelemetryAggregator] = None
        self.accelerator_mgr: Optional[AcceleratorManager] = None

        # Agents
        self.agents: Dict[str, BaseAgent] = {}

        # State
        self._running = False
        self._cycle = 0
        self._start_time: Optional[float] = None
        self._metrics = RuntimeMetrics()

        # Threading
        self._lock = threading.RLock()

        # Callbacks
        self._on_tick: List[Callable[[Dict], None]] = []

    # --------------------------------------------------------
    # LIFECYCLE
    # --------------------------------------------------------

    def start(self) -> bool:
        """Start cross-platform daemon."""
        with self._lock:
            if self.state != RuntimeState.INIT:
                return False

            self.state = RuntimeState.STARTING
            print(f"[gamesad] Starting v{self.VERSION} ({self.CODENAME})")

            try:
                # 1. Initialize HAL
                self.hal = HALFactory.create()
                self._metrics.platform = self.hal.info.vendor.name
                print(f"[gamesad] Platform: {self.hal.info.vendor.name} / {self.hal.info.arch.name}")
                print(f"[gamesad] Cores: {self.hal.info.core_count}, "
                      f"Accelerators: {[a.name for a in self.hal.info.accelerators]}")

                # 2. Initialize telemetry
                if self.config.enable_telemetry:
                    self.telemetry = TelemetryAggregator(self.hal)
                    print("[gamesad] Telemetry collector initialized")

                # 3. Initialize accelerator manager
                if self.config.enable_accelerator_routing:
                    self.accelerator_mgr = AcceleratorManager(self.hal)
                    print("[gamesad] Accelerator manager initialized")

                # 4. Initialize Guardian with platform awareness
                self.guardian = GuardianHexEngine()
                self.guardian.mode = self.config.guardian_mode
                self._configure_guardian_for_platform()
                self.guardian.on_intervention(self._on_guardian_intervention)
                print("[gamesad] Guardian/Hex initialized")

                # 5. Initialize socket exchange
                self.socketd = CrystalSocketd(
                    mode=self.config.socket_mode,
                    tick_interval_ms=self.config.tick_interval_ms
                )
                self.socketd.set_guardian(self._guardian_arbitrate)
                self.socketd.on_ticker(self._on_market_ticker)
                self.socketd.on_directive(self._on_directive_issued)
                self.socketd.start()
                print("[gamesad] Crystal-Vino exchange started")

                # 6. Initialize cross-platform agents
                self._init_agents()

                # 7. Start runtime
                self._running = True
                self._start_time = time.time()
                self.state = RuntimeState.RUNNING

                print(f"[gamesad] Market OPEN - Cross-Forex trading active")
                return True

            except Exception as e:
                print(f"[gamesad] Startup failed: {e}")
                self.state = RuntimeState.ERROR
                return False

    def stop(self):
        """Stop daemon."""
        with self._lock:
            if self.state not in [RuntimeState.RUNNING, RuntimeState.PAUSED]:
                return

            self.state = RuntimeState.STOPPING
            self._running = False

            if self.socketd:
                self.socketd.stop()

            self.state = RuntimeState.STOPPED
            print("[gamesad] Stopped")

    def _configure_guardian_for_platform(self):
        """Configure Guardian based on platform."""
        vendor = self.hal.info.vendor

        # Adjust thermal limits per platform
        if vendor == Vendor.AMD:
            # AMD has higher Tj max
            self.guardian.GPU_TEMP_CRITICAL = 100
            self.guardian.GPU_TEMP_WARNING = 90
        elif vendor in (Vendor.ARM_GENERIC, Vendor.QUALCOMM):
            # ARM has lower thermal limits
            self.guardian.GPU_TEMP_CRITICAL = 85
            self.guardian.GPU_TEMP_WARNING = 75

    def _init_agents(self):
        """Initialize platform-aware agents."""
        factory = CrossPlatformAgentFactory(self.hal)
        self.agents = factory.create_all()

        for agent_id, agent in self.agents.items():
            self.socketd.register_agent(agent_id, agent.agent_type.value)
            agent.set_order_callback(self._on_agent_order)

        print(f"[gamesad] Registered {len(self.agents)} cross-platform agents")

    # --------------------------------------------------------
    # MAIN TICK
    # --------------------------------------------------------

    def tick(self, external_telemetry: Dict = None) -> Dict:
        """Execute one tick cycle."""
        if self.state != RuntimeState.RUNNING:
            return {"error": "not_running"}

        tick_start = time.time()
        self._cycle += 1
        results = {"cycle": self._cycle, "platform": self._metrics.platform}

        # 1. Collect telemetry
        if self.telemetry:
            sample = self.telemetry.sample()
            telemetry = self.telemetry.to_commodities(sample)
        else:
            telemetry = external_telemetry or self._default_telemetry()

        results["telemetry"] = telemetry

        # 2. Auto precision scaling
        if self.config.auto_precision_scaling:
            self._check_precision_scaling(telemetry)

        # 3. Run socket tick
        socketd_result = self.socketd.tick(telemetry)
        results["socketd"] = socketd_result

        # 4. Update Guardian metrics
        self.guardian.update_metrics({
            "hex_compute": telemetry.get("hex_compute", 0x50),
            "thermal_headroom_gpu": telemetry.get("thermal_headroom_gpu", 15),
            "thermal_headroom_cpu": telemetry.get("thermal_headroom_cpu", 12)
        })
        results["guardian"] = self.guardian.get_metrics()

        # 5. Route accelerator workloads if enabled
        if self.accelerator_mgr:
            results["accelerators"] = self.accelerator_mgr.get_status()

        # 6. Update metrics
        tick_duration = (time.time() - tick_start) * 1000
        self._update_metrics(tick_duration, socketd_result)
        results["tick_ms"] = tick_duration

        # Callbacks
        for callback in self._on_tick:
            try:
                callback(results)
            except Exception:
                pass

        return results

    def _default_telemetry(self) -> Dict:
        """Generate default telemetry."""
        return {
            "hex_compute": 0x50,
            "hex_memory": 0x30,
            "hex_io": 0x20,
            "thermal_headroom_gpu": 15.0,
            "thermal_headroom_cpu": 12.0,
            "cpu_util": 0.5,
            "gpu_util": 0.5,
            "cpu_temp": 60.0,
            "gpu_temp": 70.0,
            "power_draw": 100.0,
            "memory_util": 0.4,
            "io_util": 0.3
        }

    def _check_precision_scaling(self, telemetry: Dict):
        """Auto-scale precision based on thermal/compute."""
        thermal = telemetry.get("thermal_headroom_gpu", 20)
        hex_compute = telemetry.get("hex_compute", 0x50)

        if thermal < self.config.thermal_throttle_threshold:
            # Force low precision on all GPU workloads
            for agent in self.agents.values():
                if hasattr(agent, 'current_precision'):
                    if agent.current_precision == "FP32":
                        self._metrics.precision_switches += 1

    def _update_metrics(self, tick_ms: float, socketd_result: Dict):
        """Update runtime metrics."""
        self._metrics.cycles = self._cycle
        self._metrics.uptime_seconds = time.time() - self._start_time if self._start_time else 0
        self._metrics.orders_processed += socketd_result.get("orders_processed", 0)
        self._metrics.directives_issued += socketd_result.get("directives_issued", 0)

    # --------------------------------------------------------
    # CALLBACKS
    # --------------------------------------------------------

    def _on_market_ticker(self, ticker: MarketTicker):
        """Handle ticker broadcast."""
        for agent in self.agents.values():
            try:
                agent.on_ticker(ticker)
            except Exception:
                pass

    def _on_agent_order(self, order: TradeOrder):
        """Handle agent order."""
        if self.socketd:
            self.socketd.submit_order(order)

    def _guardian_arbitrate(self, order: TradeOrder) -> Directive:
        """Guardian arbitration with platform awareness."""
        # Get current telemetry for arbitration
        telemetry = {}
        if self.telemetry:
            telemetry = self.telemetry.to_commodities()

        result = self.guardian.clear_order(order, telemetry)

        params = DirectiveParams(duration_ms=5000)

        # Apply adjustments
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
        """Handle directive to agent."""
        agent = self.agents.get(directive.target)
        if agent:
            agent.on_directive(directive)

    def _on_guardian_intervention(self, intervention: Dict):
        """Handle Guardian intervention."""
        self._metrics.thermal_events += 1
        print(f"[gamesad] Intervention: {intervention['type']}")

    # --------------------------------------------------------
    # API
    # --------------------------------------------------------

    def route_workload(self, workload_type: WorkloadType,
                       precision: PrecisionMode = PrecisionMode.FP32) -> Dict:
        """Route workload to optimal accelerator."""
        if not self.accelerator_mgr:
            return {"error": "Accelerator manager not enabled"}

        thermal = 20.0
        if self.telemetry:
            commodities = self.telemetry.to_commodities()
            thermal = commodities.get("thermal_headroom_gpu", 20.0)

        request = WorkloadRequest(workload_type, precision, priority=5)
        assignment = self.accelerator_mgr.select_accelerator(request, thermal)

        return {
            "accelerator": assignment.accelerator.name,
            "precision": assignment.precision.value,
            "latency_ms": assignment.estimated_latency_ms,
            "power_w": assignment.estimated_power_w,
            "reason": assignment.reason
        }

    def set_scenario(self, scenario: Scenario):
        """Set runtime scenario."""
        if self.socketd:
            self.socketd.set_scenario(scenario)

    def on_tick(self, callback: Callable[[Dict], None]):
        """Register tick callback."""
        self._on_tick.append(callback)

    def get_state(self) -> Dict:
        """Get comprehensive state."""
        return {
            "version": self.VERSION,
            "codename": self.CODENAME,
            "state": self.state.name,
            "platform": {
                "vendor": self.hal.info.vendor.name if self.hal else "unknown",
                "arch": self.hal.info.arch.name if self.hal else "unknown",
                "cores": self.hal.info.core_count if self.hal else 0,
                "accelerators": [a.name for a in self.hal.info.accelerators] if self.hal else []
            },
            "metrics": {
                "cycles": self._metrics.cycles,
                "uptime": self._metrics.uptime_seconds,
                "orders": self._metrics.orders_processed,
                "directives": self._metrics.directives_issued,
                "precision_switches": self._metrics.precision_switches,
                "thermal_events": self._metrics.thermal_events
            },
            "guardian": self.guardian.get_metrics() if self.guardian else {},
            "agents": [a.get_stats() for a in self.agents.values()]
        }

    def report(self) -> str:
        """Generate runtime report."""
        state = self.get_state()
        return f"""
{'='*60}
GAMESA CROSS-PLATFORM DAEMON REPORT
{'='*60}

Version: {state['version']} ({state['codename']})
State: {state['state']}
Uptime: {state['metrics']['uptime']:.1f}s

PLATFORM:
  Vendor: {state['platform']['vendor']}
  Arch: {state['platform']['arch']}
  Cores: {state['platform']['cores']}
  Accelerators: {state['platform']['accelerators']}

METRICS:
  Cycles: {state['metrics']['cycles']}
  Orders Processed: {state['metrics']['orders']}
  Directives Issued: {state['metrics']['directives']}
  Precision Switches: {state['metrics']['precision_switches']}
  Thermal Events: {state['metrics']['thermal_events']}

GUARDIAN:
  Mode: {state['guardian'].get('mode', 'N/A')}
  Hex Depth: {state['guardian'].get('hex_depth', 'N/A')}
  Approval Rate: {state['guardian'].get('approval_rate', 0)*100:.1f}%

AGENTS:
""" + '\n'.join([f"  {a['agent_id']}: orders={a['metrics']['orders']}, "
                  f"approved={a['metrics']['approved']}"
                 for a in state['agents']])


# ============================================================
# FACTORY
# ============================================================

def create_crossplatform_gamesad(config: CrossPlatformConfig = None) -> GamesadCrossPlatform:
    """Create cross-platform GAMESA daemon."""
    return GamesadCrossPlatform(config)


# ============================================================
# DEMO
# ============================================================

def demo():
    """Demonstrate cross-platform gamesad."""
    import random

    print("=" * 60)
    print("GAMESA CROSS-PLATFORM DAEMON")
    print("=" * 60)

    daemon = create_crossplatform_gamesad()
    daemon.start()

    daemon.set_scenario(Scenario.GAME_COMBAT)

    print("\nRunning 20 market ticks...\n")

    for i in range(20):
        result = daemon.tick()

        if i % 5 == 0:
            print(f"Tick {i}: cycle={result['cycle']}, "
                  f"platform={result['platform']}, "
                  f"hex_depth={result['guardian']['hex_depth']}")

    # Test workload routing
    print("\n--- Workload Routing ---")
    for wt in [WorkloadType.INFERENCE, WorkloadType.SPEECH, WorkloadType.SHADER]:
        routing = daemon.route_workload(wt, PrecisionMode.FP16)
        print(f"{wt.name}: {routing['accelerator']} @ {routing['precision']}")

    print("\n" + daemon.report())
    daemon.stop()


if __name__ == "__main__":
    demo()
