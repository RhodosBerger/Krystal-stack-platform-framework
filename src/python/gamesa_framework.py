"""
GAMESA Unified Framework

Integrates all subsystems into a cohesive runtime:
- Cognitive Synthesis Engine
- Cross-Forex Exchange
- MAVB (Memory-Augmented Virtual Bus)
- TPU Boost Bridge
- Thread Boost Layer
- Knowledge Optimizer

The ultimate integration of all GAMESA components.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable
from enum import Enum, auto
import time
import threading
from collections import deque

# Import all subsystems
from .cognitive_synthesis import CognitiveSynthesisEngine
from .cross_forex import CrossForexExchange, Commodity, OrderType, TierDirective
from .mavb import MAVBus, ResourceType
from .tpu_bridge import TPUBoostBridge, WorkloadType
from .thread_boost import ThreadBoostLayer, FlowPriority
from .knowledge_optimizer import KnowledgeOptimizer
from .krystal_sdk import Krystal, KrystalConfig


# ============================================================
# FRAMEWORK STATE
# ============================================================

class FrameworkState(Enum):
    """Framework operational state."""
    INIT = auto()
    STARTING = auto()
    RUNNING = auto()
    OPTIMIZING = auto()
    THROTTLING = auto()
    EMERGENCY = auto()
    STOPPING = auto()
    STOPPED = auto()


@dataclass
class FrameworkMetrics:
    """Aggregated framework metrics."""
    cycle: int = 0
    uptime_seconds: float = 0.0

    # Performance
    throughput_ops: float = 0.0
    avg_latency_ms: float = 0.0

    # Resource usage
    cpu_util: float = 0.0
    gpu_util: float = 0.0
    memory_util: float = 0.0

    # Health
    thermal_headroom: float = 20.0
    power_headroom: float = 50.0
    stability_score: float = 0.8

    # Learning
    total_reward: float = 0.0
    insights_generated: int = 0
    emergent_behaviors: int = 0


# ============================================================
# GAMESA FRAMEWORK
# ============================================================

class GamesaFramework:
    """
    GAMESA Unified Framework - The Ultimate Integration.

    Orchestrates all subsystems:
    - Cognitive Synthesis: Self-aware decision making
    - Cross-Forex: Resource trading market
    - MAVB: 3D memory fabric
    - TPU Bridge: AI acceleration
    - Thread Boost: Core/memory optimization
    - Knowledge Optimizer: Learning from experience
    """

    def __init__(self):
        # Core systems
        self.cognitive = CognitiveSynthesisEngine()
        self.exchange = CrossForexExchange()
        self.mavb = MAVBus()
        self.tpu = TPUBoostBridge()
        self.thread_boost = ThreadBoostLayer()
        self.optimizer = KnowledgeOptimizer(state_dim=16, action_dim=8)
        self.krystal = Krystal(KrystalConfig(state_dim=16, action_dim=8))

        # State
        self.state = FrameworkState.INIT
        self.metrics = FrameworkMetrics()
        self.start_time: Optional[float] = None

        # Event loop
        self._running = False
        self._lock = threading.RLock()
        self._cycle = 0

        # Telemetry buffer
        self.telemetry: deque = deque(maxlen=10000)

        # Registered agents
        self.agents: Dict[str, Dict] = {}

        # Callbacks
        self.on_cycle: Optional[Callable[[Dict], None]] = None
        self.on_insight: Optional[Callable[[Dict], None]] = None
        self.on_emergency: Optional[Callable[[Dict], None]] = None

    # --------------------------------------------------------
    # LIFECYCLE
    # --------------------------------------------------------

    def start(self):
        """Start the framework."""
        with self._lock:
            if self.state != FrameworkState.INIT:
                return

            self.state = FrameworkState.STARTING
            self.start_time = time.time()
            self._running = True

            # Initialize subsystems
            self.cognitive.start()
            self._setup_default_zones()
            self._register_default_agents()

            self.state = FrameworkState.RUNNING
            print("[GAMESA] Framework started")

    def stop(self):
        """Stop the framework."""
        with self._lock:
            self.state = FrameworkState.STOPPING
            self._running = False
            self.cognitive.stop()
            self.state = FrameworkState.STOPPED
            print("[GAMESA] Framework stopped")

    def _setup_default_zones(self):
        """Setup default thread boost zones."""
        self.thread_boost.create_zone(
            "COGNITIVE_CORE",
            grid_coverage=((0, 4), (0, 16), (0, 16)),
            memory_mb=256,
            priority=FlowPriority.HIGH
        )
        self.thread_boost.create_zone(
            "EXCHANGE_ENGINE",
            grid_coverage=((0, 4), (0, 8), (16, 32)),
            memory_mb=128,
            priority=FlowPriority.NORMAL
        )
        self.thread_boost.create_zone(
            "TPU_INFERENCE",
            grid_coverage=((4, 7), (0, 16), (0, 32)),
            memory_mb=512,
            priority=FlowPriority.REALTIME
        )

    def _register_default_agents(self):
        """Register default trading agents."""
        default_agents = [
            ("COGNITIVE_AGENT", "STRATEGIC", 10000),
            ("TPU_AGENT", "CREATIVE", 5000),
            ("MEMORY_AGENT", "ANALYTICAL", 5000),
            ("THERMAL_AGENT", "PROTECTIVE", 3000),
        ]
        for agent_id, domain, credits in default_agents:
            self.exchange.register_agent(agent_id, credits)
            self.exchange.update_domain(agent_id, domain)
            self.agents[agent_id] = {"domain": domain, "active": True}

    # --------------------------------------------------------
    # MAIN LOOP
    # --------------------------------------------------------

    def tick(self, external_telemetry: Dict = None) -> Dict:
        """
        Execute one framework cycle.

        Integrates all subsystems in coordinated flow.
        """
        with self._lock:
            self._cycle += 1
            results = {"cycle": self._cycle, "timestamp": time.time()}

            # 1. Gather telemetry
            telemetry = self._gather_telemetry(external_telemetry)
            results["telemetry"] = telemetry

            # 2. Update thermal state
            self._update_thermal(telemetry)

            # 3. Cognitive cycle
            cognitive_result = self.cognitive.tick(telemetry)
            results["cognitive"] = {
                "fitness": cognitive_result.get("fitness", 0),
                "plan": cognitive_result.get("plan"),
                "safe": cognitive_result.get("safe", True)
            }

            # 4. Knowledge optimizer
            self.optimizer.observe(telemetry)
            action = self.optimizer.decide()
            results["optimizer_action"] = action

            # 5. Exchange tick
            exchange_tick = self.exchange.tick()
            results["exchange"] = {
                "thermal_response": exchange_tick["thermal_response"],
                "active_orders": exchange_tick["active_orders"]
            }

            # 6. MAVB tick
            mavb_tick = self.mavb.tick()
            results["mavb"] = mavb_tick

            # 7. Thread boost tick
            boost_tick = self.thread_boost.tick()
            results["thread_boost"] = boost_tick

            # 8. TPU inference (if needed)
            if self._should_run_inference(telemetry):
                inference = self.tpu.run_inference(
                    WorkloadType.INFERENCE,
                    {"telemetry": telemetry},
                    domain="COGNITIVE_AGENT"
                )
                results["inference"] = {
                    "latency_ms": inference.latency_ms,
                    "accelerator": inference.accelerator.name
                }

            # 9. Compute reward and learn
            reward = self._compute_reward(telemetry, cognitive_result)
            optimizer_learn = self.optimizer.reward(reward)
            self.krystal.observe(telemetry)
            self.krystal.decide()
            self.krystal.reward(reward)
            results["reward"] = reward

            # 10. Update state
            self._update_state(telemetry, exchange_tick)

            # 11. Update metrics
            self._update_metrics(telemetry, results)

            # 12. Store telemetry
            self.telemetry.append(results)

            # Callbacks
            if self.on_cycle:
                self.on_cycle(results)

            if cognitive_result.get("insight") and self.on_insight:
                self.on_insight({"insight": cognitive_result["insight"]})

            return results

    def run(self, cycles: int, telemetry_fn: Callable[[], Dict] = None) -> List[Dict]:
        """Run multiple cycles."""
        results = []
        for _ in range(cycles):
            if not self._running:
                break
            telemetry = telemetry_fn() if telemetry_fn else None
            result = self.tick(telemetry)
            results.append(result)
        return results

    # --------------------------------------------------------
    # INTERNAL HELPERS
    # --------------------------------------------------------

    def _gather_telemetry(self, external: Dict = None) -> Dict:
        """Gather telemetry from all sources."""
        telemetry = {
            "cycle": self._cycle,
            "cpu_util": 0.5,
            "gpu_util": 0.6,
            "memory_util": 0.4,
            "cpu_temp": 55.0,
            "gpu_temp": 65.0,
            "power_draw": 150.0,
            "performance": 0.7,
            "latency_ms": 16.0,
        }

        # Merge external telemetry
        if external:
            telemetry.update(external)

        # Add MAVB heatmap summary
        heatmap = self.mavb.get_heatmap()
        telemetry["mavb_active_voxels"] = len(heatmap.get("mavb_heatmap", []))

        # Add thread boost summary
        boost = self.thread_boost.tick()
        telemetry["boost_elevated"] = boost.get("elevated", 0)

        return telemetry

    def _update_thermal(self, telemetry: Dict):
        """Update thermal state across subsystems."""
        cpu_temp = telemetry.get("cpu_temp", 55)
        gpu_temp = telemetry.get("gpu_temp", 65)
        power = telemetry.get("power_draw", 150)

        self.exchange.update_thermal(cpu_temp, gpu_temp, power)
        self.tpu.update_thermal(gpu_temp, power / 15)

    def _should_run_inference(self, telemetry: Dict) -> bool:
        """Determine if TPU inference should run."""
        return self._cycle % 10 == 0  # Every 10 cycles

    def _compute_reward(self, telemetry: Dict, cognitive: Dict) -> float:
        """Compute unified reward signal."""
        reward = 0.0

        # Performance reward
        perf = telemetry.get("performance", 0.5)
        reward += 0.3 * perf

        # Thermal reward
        thermal_ok = telemetry.get("gpu_temp", 70) < 80
        reward += 0.2 if thermal_ok else -0.1

        # Cognitive fitness
        fitness = cognitive.get("fitness", 0.5)
        reward += 0.3 * fitness

        # Safety reward
        if cognitive.get("safe", True):
            reward += 0.2

        return max(0, min(1, reward))

    def _update_state(self, telemetry: Dict, exchange: Dict):
        """Update framework state."""
        thermal = exchange.get("thermal_response", "SAFE")

        if thermal == "EMERGENCY":
            self.state = FrameworkState.EMERGENCY
            if self.on_emergency:
                self.on_emergency({"thermal": thermal})
        elif thermal == "THROTTLE":
            self.state = FrameworkState.THROTTLING
        elif self.metrics.stability_score < 0.5:
            self.state = FrameworkState.OPTIMIZING
        else:
            self.state = FrameworkState.RUNNING

    def _update_metrics(self, telemetry: Dict, results: Dict):
        """Update aggregated metrics."""
        self.metrics.cycle = self._cycle
        self.metrics.uptime_seconds = time.time() - self.start_time if self.start_time else 0

        self.metrics.cpu_util = telemetry.get("cpu_util", 0.5)
        self.metrics.gpu_util = telemetry.get("gpu_util", 0.5)
        self.metrics.memory_util = telemetry.get("memory_util", 0.4)

        self.metrics.thermal_headroom = max(0, 80 - telemetry.get("gpu_temp", 65))
        self.metrics.power_headroom = max(0, 250 - telemetry.get("power_draw", 150))

        self.metrics.total_reward += results.get("reward", 0)

        # From cognitive
        cog_state = self.cognitive.get_state()
        self.metrics.insights_generated = cog_state.get("insights_count", 0)
        self.metrics.emergent_behaviors = cog_state.get("emergent_behaviors", 0)

    # --------------------------------------------------------
    # PUBLIC API
    # --------------------------------------------------------

    def submit_trade(self, agent_id: str, commodity: Commodity,
                     order_type: OrderType, quantity: float, price: float) -> Dict:
        """Submit trade to exchange."""
        order, trades = self.exchange.submit_order(
            agent_id, commodity, order_type, quantity, price
        )
        return {
            "order_id": order.order_id,
            "status": order.status.name,
            "trades": len(trades)
        }

    def request_voxel(self, agent_id: str, voxel: tuple,
                      resource: ResourceType, bytes_req: int) -> Dict:
        """Request MAVB voxel allocation."""
        return self.mavb.submit_trade(
            agent_id, voxel, resource, bytes_req, duration_ms=16
        )

    def run_inference(self, workload: WorkloadType, input_data: Dict,
                      domain: str = None) -> Dict:
        """Run TPU inference."""
        result = self.tpu.run_inference(workload, input_data, domain)
        return {
            "request_id": result.request_id,
            "accelerator": result.accelerator.name,
            "latency_ms": result.latency_ms,
            "success": result.success
        }

    def get_state(self) -> Dict:
        """Get comprehensive framework state."""
        return {
            "state": self.state.name,
            "metrics": {
                "cycle": self.metrics.cycle,
                "uptime": self.metrics.uptime_seconds,
                "cpu_util": self.metrics.cpu_util,
                "gpu_util": self.metrics.gpu_util,
                "thermal_headroom": self.metrics.thermal_headroom,
                "total_reward": self.metrics.total_reward,
                "insights": self.metrics.insights_generated,
                "emergent": self.metrics.emergent_behaviors
            },
            "cognitive": self.cognitive.get_state(),
            "exchange": {
                "agents": len(self.agents),
                "rankings": self.exchange.ranking.get_rankings()[:3]
            },
            "mavb": self.mavb.stats(),
            "tpu": self.tpu.get_stats(),
            "thread_boost": self.thread_boost.tick()
        }

    def report(self) -> str:
        """Generate comprehensive report."""
        state = self.get_state()
        return f"""
{'='*60}
GAMESA UNIFIED FRAMEWORK REPORT
{'='*60}

State: {state['state']}
Cycle: {state['metrics']['cycle']}
Uptime: {state['metrics']['uptime']:.1f}s

METRICS:
  CPU Util: {state['metrics']['cpu_util']*100:.1f}%
  GPU Util: {state['metrics']['gpu_util']*100:.1f}%
  Thermal Headroom: {state['metrics']['thermal_headroom']:.1f}C
  Total Reward: {state['metrics']['total_reward']:.2f}
  Insights: {state['metrics']['insights']}
  Emergent Behaviors: {state['metrics']['emergent']}

COGNITIVE:
  Awareness: {state['cognitive'].get('awareness', {})}

EXCHANGE:
  Agents: {state['exchange']['agents']}
  Top Rankings: {state['exchange']['rankings']}

MAVB:
  {state['mavb']}

TPU:
  {state['tpu']}

THREAD BOOST:
  {state['thread_boost']}
"""


# ============================================================
# FACTORY
# ============================================================

def create_gamesa_framework() -> GamesaFramework:
    """Create and return GAMESA framework instance."""
    return GamesaFramework()


# ============================================================
# DEMO
# ============================================================

def demo():
    """Demonstrate GAMESA Framework."""
    import random

    print("=" * 60)
    print("GAMESA UNIFIED FRAMEWORK")
    print("=" * 60)

    framework = create_gamesa_framework()
    framework.start()

    print("\nRunning 50 cycles...\n")

    def generate_telemetry():
        return {
            "cpu_util": 0.4 + random.gauss(0, 0.1),
            "gpu_util": 0.6 + random.gauss(0, 0.1),
            "cpu_temp": 55 + random.gauss(0, 5),
            "gpu_temp": 65 + random.gauss(0, 8),
            "power_draw": 150 + random.gauss(0, 20),
            "performance": 0.7 + random.gauss(0, 0.1)
        }

    for i in range(50):
        result = framework.tick(generate_telemetry())

        if i % 10 == 0:
            print(f"Cycle {i}: State={framework.state.name}, "
                  f"Reward={result['reward']:.3f}, "
                  f"Thermal={result['exchange']['thermal_response']}")

    print("\n" + framework.report())

    framework.stop()
    print("\nDemo complete!")


if __name__ == "__main__":
    demo()
