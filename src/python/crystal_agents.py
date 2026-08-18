"""
GAMESA Crystal-Vino Trading Agents

Implements the Cross-Forex traders:
- Iris Xe Trader (GPU/Vulkan layer)
- Silicon Trader (CPU agent)
- Neural Trader (OpenVINO/NPU)
- Cache Agent (Memory prefetch)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Any
from enum import Enum, auto
from abc import ABC, abstractmethod
from collections import deque
import time
import random

from .crystal_protocol import (
    MarketTicker, TradeOrder, Directive, TradeBid,
    AgentType, OrderAction, DirectiveStatus, Precision,
    CommodityPrices, ProtocolCodec
)


class AgentState(Enum):
    """Agent operational state."""
    IDLE = auto()
    TRADING = auto()
    EXECUTING = auto()
    THROTTLED = auto()
    ERROR = auto()


@dataclass
class AgentMetrics:
    """Agent performance metrics."""
    ticks_processed: int = 0
    orders_submitted: int = 0
    orders_approved: int = 0
    orders_denied: int = 0
    thermal_savings: float = 0.0
    latency_impact: float = 0.0


class BaseAgent(ABC):
    """
    Base class for Cross-Forex trading agents.

    Agents observe market tickers and submit trade orders
    to acquire or release resources.
    """

    def __init__(self, agent_id: str, agent_type: AgentType):
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.state = AgentState.IDLE
        self.metrics = AgentMetrics()

        # Current directives
        self._active_directives: Dict[str, Directive] = {}
        self._directive_history: deque = deque(maxlen=1000)

        # Decision thresholds
        self.thresholds: Dict[str, float] = {}

        # Callbacks
        self._on_order: Optional[Callable[[TradeOrder], None]] = None

    def set_order_callback(self, callback: Callable[[TradeOrder], None]):
        """Set callback for order submission."""
        self._on_order = callback

    @abstractmethod
    def analyze_market(self, ticker: MarketTicker) -> Optional[TradeOrder]:
        """
        Analyze market ticker and decide whether to trade.

        Returns:
            TradeOrder if agent wants to trade, None otherwise
        """
        pass

    def on_ticker(self, ticker: MarketTicker) -> Optional[TradeOrder]:
        """Process market ticker."""
        self.metrics.ticks_processed += 1

        # Clean expired directives
        self._clean_expired_directives()

        # Analyze and potentially trade
        order = self.analyze_market(ticker)

        if order and self._on_order:
            self._on_order(order)
            self.metrics.orders_submitted += 1

        return order

    def on_directive(self, directive: Directive):
        """Process directive from Guardian."""
        self._directive_history.append(directive)

        if directive.status == DirectiveStatus.APPROVED.value:
            self._active_directives[directive.permit_id] = directive
            self.metrics.orders_approved += 1
            self._execute_directive(directive)
        else:
            self.metrics.orders_denied += 1

    @abstractmethod
    def _execute_directive(self, directive: Directive):
        """Execute approved directive."""
        pass

    def _clean_expired_directives(self):
        """Remove expired directives."""
        now = int(time.time() * 1000000)
        expired = [
            pid for pid, d in self._active_directives.items()
            if d.expires_at > 0 and now > d.expires_at
        ]
        for pid in expired:
            del self._active_directives[pid]

    def has_active_directive(self, action: str) -> bool:
        """Check if agent has active directive for action."""
        for d in self._active_directives.values():
            if action in d.permit_id or action in str(d.params.to_dict()):
                return True
        return False

    def get_stats(self) -> Dict:
        """Get agent statistics."""
        return {
            "agent_id": self.agent_id,
            "type": self.agent_type.value,
            "state": self.state.name,
            "metrics": {
                "ticks": self.metrics.ticks_processed,
                "orders": self.metrics.orders_submitted,
                "approved": self.metrics.orders_approved,
                "denied": self.metrics.orders_denied,
                "thermal_savings": self.metrics.thermal_savings
            },
            "active_directives": len(self._active_directives)
        }


class IrisXeTrader(BaseAgent):
    """
    Iris Xe GPU Trader (Vulkan layer).

    Bids for shader time and offers lower precision (FP16)
    to save thermal headroom when compute prices are high.
    """

    def __init__(self):
        super().__init__("AGENT_IRIS_XE", AgentType.IRIS_XE)

        self.thresholds = {
            "hex_compute_high": 0x80,      # Switch to FP16 above this
            "hex_compute_low": 0x40,       # Switch back to FP32 below this
            "thermal_critical": 5.0,        # Emergency threshold
        }

        # Current precision mode
        self.current_precision = Precision.FP32
        self.xmx_enabled = False

    def analyze_market(self, ticker: MarketTicker) -> Optional[TradeOrder]:
        """Analyze market and decide on precision trading."""
        commodities = ticker.commodities
        hex_compute = commodities.hex_compute
        thermal_gpu = commodities.thermal_headroom_gpu

        # Emergency thermal response
        if thermal_gpu < self.thresholds["thermal_critical"]:
            if self.current_precision != Precision.FP16:
                return self._create_fp16_request("THERMAL_CRITICAL", priority=10)

        # High compute pressure - switch to FP16
        if hex_compute > self.thresholds["hex_compute_high"]:
            if self.current_precision == Precision.FP32:
                if not self.has_active_directive("FP16"):
                    return self._create_fp16_request("HEX_COMPUTE_HIGH", priority=7)

        # Low compute pressure - restore FP32
        if hex_compute < self.thresholds["hex_compute_low"]:
            if self.current_precision == Precision.FP16:
                if not self.has_active_directive("FP32"):
                    return self._create_fp32_request("HEX_COMPUTE_LOW", priority=3)

        return None

    def _create_fp16_request(self, reason: str, priority: int) -> TradeOrder:
        """Create FP16 override request."""
        return TradeOrder(
            source=self.agent_id,
            action=OrderAction.REQUEST_FP16_OVERRIDE.value,
            bid=TradeBid(
                reason=reason,
                est_thermal_saving=3.0,
                priority=priority
            )
        )

    def _create_fp32_request(self, reason: str, priority: int) -> TradeOrder:
        """Create FP32 restore request."""
        return TradeOrder(
            source=self.agent_id,
            action=OrderAction.REQUEST_FP32_RESTORE.value,
            bid=TradeBid(
                reason=reason,
                est_thermal_saving=-1.0,
                priority=priority
            )
        )

    def _execute_directive(self, directive: Directive):
        """Execute precision change directive."""
        params = directive.params

        if params.force_precision == "FP16":
            self.current_precision = Precision.FP16
            self.xmx_enabled = True
            self.metrics.thermal_savings += 3.0
            print(f"[IrisXe] Switched to FP16 (XMX enabled)")
        elif params.force_precision == "FP32":
            self.current_precision = Precision.FP32
            self.xmx_enabled = False
            print(f"[IrisXe] Restored FP32")


class SiliconTrader(BaseAgent):
    """
    Silicon CPU Trader.

    Bids for frequency boosts and offers to park background
    threads to buy latency budget.
    """

    def __init__(self):
        super().__init__("AGENT_SILICON", AgentType.SILICON)

        self.thresholds = {
            "hex_io_high": 0x60,           # GPU starving for data
            "thermal_cpu_low": 8.0,         # Can boost
            "latency_budget_low": 10.0,     # Need more latency
        }

        # CPU state
        self.current_freq_boost = False
        self.parked_threads = 0
        self.prefetch_active = False

    def analyze_market(self, ticker: MarketTicker) -> Optional[TradeOrder]:
        """Analyze market for CPU trading opportunities."""
        commodities = ticker.commodities
        hex_io = commodities.hex_io
        thermal_cpu = commodities.thermal_headroom_cpu
        latency = commodities.latency_budget_ms

        # High IO traffic - GPU starving for data
        if hex_io > self.thresholds["hex_io_high"]:
            if not self.prefetch_active:
                return self._create_prefetch_request("HEX_IO_HIGH")

        # Low latency budget - park threads
        if latency < self.thresholds["latency_budget_low"]:
            if self.parked_threads < 4:
                return self._create_thread_park_request("LATENCY_LOW")

        # Thermal headroom available - request boost
        if thermal_cpu > self.thresholds["thermal_cpu_low"]:
            if not self.current_freq_boost:
                return self._create_boost_request("THERMAL_HEADROOM")

        return None

    def _create_prefetch_request(self, reason: str) -> TradeOrder:
        """Create cache prefetch request."""
        return TradeOrder(
            source=self.agent_id,
            action=OrderAction.REQUEST_CACHE_PREFETCH.value,
            bid=TradeBid(
                reason=reason,
                est_latency_impact=-2.0,
                priority=6
            )
        )

    def _create_thread_park_request(self, reason: str) -> TradeOrder:
        """Create thread park request."""
        return TradeOrder(
            source=self.agent_id,
            action=OrderAction.REQUEST_THREAD_PARK.value,
            bid=TradeBid(
                reason=reason,
                est_latency_impact=-1.0,
                est_thermal_saving=1.0,
                priority=5
            )
        )

    def _create_boost_request(self, reason: str) -> TradeOrder:
        """Create frequency boost request."""
        return TradeOrder(
            source=self.agent_id,
            action=OrderAction.REQUEST_FREQ_BOOST.value,
            bid=TradeBid(
                reason=reason,
                est_thermal_saving=-2.0,
                est_compute_cost=50.0,
                priority=4
            )
        )

    def _execute_directive(self, directive: Directive):
        """Execute CPU directive."""
        params = directive.params
        action = directive.permit_id

        if "PREFETCH" in action or params.cache_addresses:
            self.prefetch_active = True
            print(f"[Silicon] Cache prefetch activated")
        elif "THREAD_PARK" in action or params.thread_mask:
            self.parked_threads += 2
            self.metrics.thermal_savings += 1.0
            print(f"[Silicon] Parked threads (total: {self.parked_threads})")
        elif "FREQ_BOOST" in action or params.target_frequency_mhz:
            self.current_freq_boost = True
            print(f"[Silicon] Frequency boost enabled")


class NeuralTrader(BaseAgent):
    """
    Neural Trader (OpenVINO/NPU).

    Bids for NPU/GPU slots and offers batching flexibility
    to reduce resource contention.
    """

    def __init__(self):
        super().__init__("AGENT_NEURAL", AgentType.NEURAL)

        self.thresholds = {
            "hex_compute_high": 0x90,      # Reduce batch when high
            "game_mode_priority": False,    # Lower priority in games
        }

        # Inference state
        self.current_batch_size = 4
        self.npu_slot_active = False
        self.inference_paused = False

    def analyze_market(self, ticker: MarketTicker) -> Optional[TradeOrder]:
        """Analyze market for neural trading."""
        commodities = ticker.commodities
        state = ticker.state
        hex_compute = commodities.hex_compute

        # In game mode, offer to reduce batch
        if "GAME" in state.scenario:
            if self.current_batch_size > 1:
                return self._create_batch_reduce_request("GAME_MODE_DEFER")

        # High compute - reduce batch
        if hex_compute > self.thresholds["hex_compute_high"]:
            if self.current_batch_size > 1:
                return self._create_batch_reduce_request("HEX_COMPUTE_HIGH")

        # Request NPU slot when available
        if hex_compute < 0x50 and not self.npu_slot_active:
            return self._create_npu_request("HEX_COMPUTE_LOW")

        return None

    def _create_batch_reduce_request(self, reason: str) -> TradeOrder:
        """Create batch reduction offer."""
        return TradeOrder(
            source=self.agent_id,
            action=OrderAction.REQUEST_BATCH_REDUCE.value,
            bid=TradeBid(
                reason=reason,
                est_thermal_saving=1.5,
                est_compute_cost=-30.0,
                priority=3
            )
        )

    def _create_npu_request(self, reason: str) -> TradeOrder:
        """Create NPU slot request."""
        return TradeOrder(
            source=self.agent_id,
            action=OrderAction.REQUEST_NPU_SLOT.value,
            bid=TradeBid(
                reason=reason,
                est_thermal_saving=2.0,
                priority=4
            )
        )

    def _execute_directive(self, directive: Directive):
        """Execute neural directive."""
        params = directive.params

        if params.batch_size:
            self.current_batch_size = params.batch_size
            self.metrics.thermal_savings += 1.5
            print(f"[Neural] Batch size -> {self.current_batch_size}")
        elif "NPU" in directive.permit_id:
            self.npu_slot_active = True
            print(f"[Neural] NPU slot acquired")


class CacheAgent(BaseAgent):
    """
    Cache Agent (Memory Bridge).

    Handles Just-in-Time memory supply chain via
    prefetch directives.
    """

    def __init__(self):
        super().__init__("AGENT_CACHE", AgentType.CACHE)

        self.thresholds = {
            "hex_memory_high": 0x70,
            "hex_io_high": 0x60,
        }

        # Cache state
        self.prefetch_queue: List[str] = []
        self.llc_pressure = 0.0

    def analyze_market(self, ticker: MarketTicker) -> Optional[TradeOrder]:
        """Analyze memory market."""
        commodities = ticker.commodities
        hex_memory = commodities.hex_memory
        hex_io = commodities.hex_io

        # High memory/IO - proactive prefetch
        if hex_memory > self.thresholds["hex_memory_high"]:
            return self._create_prefetch_request("HEX_MEMORY_HIGH")

        if hex_io > self.thresholds["hex_io_high"]:
            return self._create_prefetch_request("HEX_IO_HIGH")

        return None

    def _create_prefetch_request(self, reason: str) -> TradeOrder:
        """Create prefetch request."""
        return TradeOrder(
            source=self.agent_id,
            action=OrderAction.REQUEST_CACHE_PREFETCH.value,
            bid=TradeBid(
                reason=reason,
                est_latency_impact=-3.0,
                priority=5
            )
        )

    def _execute_directive(self, directive: Directive):
        """Execute cache directive."""
        params = directive.params

        if params.cache_addresses:
            self.prefetch_queue.extend(params.cache_addresses)
            print(f"[Cache] Prefetch queued: {len(params.cache_addresses)} addresses")
        else:
            # Simulated prefetch
            self.llc_pressure += 0.1
            self.metrics.latency_impact -= 2.0
            print(f"[Cache] Prefetch executed")


# ============================================================
# AGENT FACTORY
# ============================================================

def create_all_agents() -> Dict[str, BaseAgent]:
    """Create all standard Cross-Forex agents."""
    return {
        "AGENT_IRIS_XE": IrisXeTrader(),
        "AGENT_SILICON": SiliconTrader(),
        "AGENT_NEURAL": NeuralTrader(),
        "AGENT_CACHE": CacheAgent()
    }


# ============================================================
# DEMO
# ============================================================

def demo():
    """Demonstrate Crystal agents."""
    print("=== Crystal-Vino Trading Agents Demo ===\n")

    agents = create_all_agents()
    codec = ProtocolCodec()

    # Simulate market conditions
    scenarios = [
        {"hex_compute": 0x40, "thermal_headroom_gpu": 15.0, "scenario": "IDLE"},
        {"hex_compute": 0x90, "thermal_headroom_gpu": 10.0, "scenario": "GAME_COMBAT"},
        {"hex_compute": 0xB0, "thermal_headroom_gpu": 5.0, "scenario": "GAME_COMBAT"},
        {"hex_compute": 0x30, "thermal_headroom_gpu": 20.0, "scenario": "DESKTOP"},
    ]

    for i, scenario in enumerate(scenarios):
        print(f"\n--- Tick {i+1}: {scenario['scenario']} ---")
        print(f"hex_compute=0x{scenario['hex_compute']:02X}, thermal={scenario['thermal_headroom_gpu']}")

        ticker = codec.create_ticker(
            cycle=i,
            commodities=scenario,
            scenario=scenario["scenario"]
        )

        for agent in agents.values():
            order = agent.on_ticker(ticker)
            if order:
                print(f"  [{agent.agent_id}] -> {order.action}")

                # Simulate approval
                directive = codec.create_directive(
                    target=agent.agent_id,
                    order_id=order.order_id,
                    approved=True,
                    duration_ms=5000,
                    force_precision="FP16" if "FP16" in order.action else None,
                    batch_size=1 if "BATCH" in order.action else None
                )
                agent.on_directive(directive)

    print("\n--- Agent Stats ---")
    for agent in agents.values():
        stats = agent.get_stats()
        print(f"{stats['agent_id']}: orders={stats['metrics']['orders']}, "
              f"approved={stats['metrics']['approved']}")


if __name__ == "__main__":
    demo()
