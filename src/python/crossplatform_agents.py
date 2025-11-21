"""
GAMESA Cross-Platform Trading Agents

Platform-aware agents that adapt to Intel, AMD, and ARM:
- Unified GPU Trader (Iris Xe / RDNA / Mali)
- Unified CPU Trader (P/E cores / Zen / big.LITTLE)
- Unified NPU Trader (GNA / XDNA / Ethos)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum, auto
from abc import ABC

from .crystal_protocol import (
    MarketTicker, TradeOrder, Directive, TradeBid,
    AgentType, OrderAction, DirectiveStatus
)
from .crystal_agents import BaseAgent, AgentState
from .platform_hal import (
    BaseHAL, HALFactory, PlatformDetector, Vendor,
    AcceleratorType, CoreType, PlatformInfo
)


# ============================================================
# CROSS-PLATFORM GPU TRADER
# ============================================================

class CrossPlatformGPUTrader(BaseAgent):
    """
    Unified GPU trader for all platforms.

    Adapts to:
    - Intel: Iris Xe with XMX units
    - AMD: RDNA with WMMA
    - ARM: Mali/Adreno compute shaders
    """

    def __init__(self, hal: BaseHAL):
        super().__init__("AGENT_GPU", AgentType.IRIS_XE)
        self.hal = hal
        self.platform = hal.info.vendor

        # Platform-specific thresholds
        self._init_thresholds()

        # Current state
        self.current_precision = "FP32"
        self.matrix_units_active = False

    def _init_thresholds(self):
        """Initialize platform-specific thresholds."""
        if self.platform == Vendor.INTEL:
            self.thresholds = {
                "hex_compute_high": 0x80,
                "hex_compute_low": 0x40,
                "thermal_critical": 5.0,
                "thermal_warning": 12.0,
                "matrix_unit": "XMX",
                "low_precision": "FP16"
            }
        elif self.platform == Vendor.AMD:
            self.thresholds = {
                "hex_compute_high": 0x85,  # AMD runs hotter
                "hex_compute_low": 0x45,
                "thermal_critical": 8.0,   # Higher Tj max
                "thermal_warning": 15.0,
                "matrix_unit": "WMMA",
                "low_precision": "FP16"
            }
        else:  # ARM
            self.thresholds = {
                "hex_compute_high": 0x70,  # More conservative
                "hex_compute_low": 0x35,
                "thermal_critical": 5.0,
                "thermal_warning": 10.0,
                "matrix_unit": "COMPUTE_SHADER",
                "low_precision": "FP16"
            }

    def analyze_market(self, ticker: MarketTicker) -> Optional[TradeOrder]:
        """Platform-aware market analysis."""
        commodities = ticker.commodities
        hex_compute = commodities.hex_compute
        thermal_gpu = commodities.thermal_headroom_gpu

        # Get platform-specific thermal from HAL
        hal_thermal = self.hal.get_thermal_headroom("gpu")

        # Use more accurate thermal if available
        effective_thermal = min(thermal_gpu, hal_thermal)

        # Emergency thermal response
        if effective_thermal < self.thresholds["thermal_critical"]:
            if self.current_precision != self.thresholds["low_precision"]:
                return self._create_low_precision_request(
                    "THERMAL_CRITICAL",
                    priority=10
                )

        # High compute pressure
        if hex_compute > self.thresholds["hex_compute_high"]:
            if self.current_precision == "FP32":
                return self._create_low_precision_request(
                    f"HEX_COMPUTE_HIGH_{self.platform.name}",
                    priority=7
                )

        # Recovery - restore precision
        if hex_compute < self.thresholds["hex_compute_low"]:
            if effective_thermal > self.thresholds["thermal_warning"]:
                if self.current_precision != "FP32":
                    return self._create_high_precision_request(
                        "HEX_COMPUTE_LOW",
                        priority=3
                    )

        return None

    def _create_low_precision_request(self, reason: str, priority: int) -> TradeOrder:
        """Create platform-appropriate low precision request."""
        precision = self.thresholds["low_precision"]
        matrix_unit = self.thresholds["matrix_unit"]

        return TradeOrder(
            source=self.agent_id,
            action=f"REQUEST_{precision}_OVERRIDE",
            bid=TradeBid(
                reason=f"{reason}_{matrix_unit}",
                est_thermal_saving=3.0 if self.platform == Vendor.INTEL else 2.5,
                priority=priority
            )
        )

    def _create_high_precision_request(self, reason: str, priority: int) -> TradeOrder:
        """Request return to high precision."""
        return TradeOrder(
            source=self.agent_id,
            action="REQUEST_FP32_RESTORE",
            bid=TradeBid(
                reason=reason,
                est_thermal_saving=-1.5,
                priority=priority
            )
        )

    def _execute_directive(self, directive: Directive):
        """Execute precision change."""
        params = directive.params

        if params.force_precision:
            self.current_precision = params.force_precision
            self.matrix_units_active = params.force_precision in ("FP16", "BF16")
            self.metrics.thermal_savings += 3.0 if self.matrix_units_active else -1.0

            print(f"[GPU/{self.platform.name}] Precision -> {self.current_precision} "
                  f"({self.thresholds['matrix_unit']} {'ON' if self.matrix_units_active else 'OFF'})")


# ============================================================
# CROSS-PLATFORM CPU TRADER
# ============================================================

class CrossPlatformCPUTrader(BaseAgent):
    """
    Unified CPU trader for all platforms.

    Adapts to:
    - Intel: P/E cores with Thread Director
    - AMD: Zen cores with CCD topology
    - ARM: big.LITTLE with EAS
    """

    def __init__(self, hal: BaseHAL):
        super().__init__("AGENT_CPU", AgentType.SILICON)
        self.hal = hal
        self.platform = hal.info.vendor

        self._init_thresholds()

        # Current state
        self.boost_active = False
        self.parked_cores: List[int] = []
        self.prefetch_active = False
        self.affinity_mode = "balanced"

    def _init_thresholds(self):
        """Initialize platform-specific thresholds."""
        if self.platform == Vendor.INTEL:
            self.thresholds = {
                "hex_io_high": 0x60,
                "thermal_boost_min": 10.0,
                "latency_low": 10.0,
                "boost_type": "TURBO_BOOST_3",
                "core_parking": "E_CORE_PARK",
                "prefetch_type": "PREFETCHW"
            }
        elif self.platform == Vendor.AMD:
            self.thresholds = {
                "hex_io_high": 0x55,
                "thermal_boost_min": 12.0,
                "latency_low": 12.0,
                "boost_type": "PRECISION_BOOST",
                "core_parking": "CCD_PARK",
                "prefetch_type": "PREFETCH"
            }
        else:  # ARM
            self.thresholds = {
                "hex_io_high": 0x50,
                "thermal_boost_min": 8.0,
                "latency_low": 15.0,
                "boost_type": "DVFS_BOOST",
                "core_parking": "LITTLE_PARK",
                "prefetch_type": "PRFM"
            }

    def analyze_market(self, ticker: MarketTicker) -> Optional[TradeOrder]:
        """Platform-aware CPU trading."""
        commodities = ticker.commodities
        hex_io = commodities.hex_io
        thermal_cpu = commodities.thermal_headroom_cpu
        latency = commodities.latency_budget_ms

        # Get HAL thermal
        hal_thermal = self.hal.get_thermal_headroom("cpu")
        effective_thermal = min(thermal_cpu, hal_thermal)

        # High IO - GPU starving for data
        if hex_io > self.thresholds["hex_io_high"]:
            if not self.prefetch_active:
                return self._create_prefetch_request(
                    f"HEX_IO_HIGH_{self.platform.name}"
                )

        # Low latency budget - park background cores
        if latency < self.thresholds["latency_low"]:
            if len(self.parked_cores) < 4:
                return self._create_park_request("LATENCY_LOW")

        # Thermal headroom available - request boost
        if effective_thermal > self.thresholds["thermal_boost_min"]:
            if not self.boost_active:
                return self._create_boost_request("THERMAL_HEADROOM")

        return None

    def _create_prefetch_request(self, reason: str) -> TradeOrder:
        """Create platform-specific prefetch request."""
        return TradeOrder(
            source=self.agent_id,
            action=f"REQUEST_CACHE_PREFETCH_{self.thresholds['prefetch_type']}",
            bid=TradeBid(
                reason=reason,
                est_latency_impact=-3.0,
                priority=6
            )
        )

    def _create_park_request(self, reason: str) -> TradeOrder:
        """Create core parking request."""
        return TradeOrder(
            source=self.agent_id,
            action=f"REQUEST_{self.thresholds['core_parking']}",
            bid=TradeBid(
                reason=reason,
                est_thermal_saving=1.5,
                est_latency_impact=-1.0,
                priority=5
            )
        )

    def _create_boost_request(self, reason: str) -> TradeOrder:
        """Create frequency boost request."""
        return TradeOrder(
            source=self.agent_id,
            action=f"REQUEST_{self.thresholds['boost_type']}",
            bid=TradeBid(
                reason=reason,
                est_thermal_saving=-2.0,
                est_compute_cost=50.0,
                priority=4
            )
        )

    def _execute_directive(self, directive: Directive):
        """Execute CPU directive with HAL."""
        params = directive.params
        action = directive.permit_id

        if "PREFETCH" in action:
            self.prefetch_active = True
            print(f"[CPU/{self.platform.name}] Prefetch active ({self.thresholds['prefetch_type']})")

        elif "PARK" in action:
            # Get background affinity from HAL
            affinity = self.hal.get_thread_affinity("background")
            park_cores = list(affinity.values())[0] if affinity else []
            self.parked_cores.extend(park_cores[:2])
            self.metrics.thermal_savings += 1.5
            print(f"[CPU/{self.platform.name}] Parked cores: {self.parked_cores}")

        elif "BOOST" in action:
            self.boost_active = True
            print(f"[CPU/{self.platform.name}] {self.thresholds['boost_type']} enabled")


# ============================================================
# CROSS-PLATFORM NPU TRADER
# ============================================================

class CrossPlatformNPUTrader(BaseAgent):
    """
    Unified NPU trader for all platforms.

    Adapts to:
    - Intel: GNA / Meteor Lake NPU
    - AMD: XDNA / Ryzen AI
    - ARM: Ethos / Hexagon
    """

    def __init__(self, hal: BaseHAL):
        super().__init__("AGENT_NPU", AgentType.NEURAL)
        self.hal = hal
        self.platform = hal.info.vendor

        self._init_thresholds()

        # Current state
        self.current_batch_size = 4
        self.npu_active = False
        self.offload_mode = "GPU"  # GPU, NPU, or CPU

    def _init_thresholds(self):
        """Initialize platform-specific NPU thresholds."""
        if self.platform == Vendor.INTEL:
            self.thresholds = {
                "hex_compute_high": 0x85,
                "npu_type": "GNA_2",
                "npu_workloads": ["speech", "audio", "keyword"],
                "fallback": "IRIS_XE",
                "min_batch": 1,
                "max_batch": 8
            }
        elif self.platform == Vendor.AMD:
            self.thresholds = {
                "hex_compute_high": 0x80,
                "npu_type": "XDNA",
                "npu_workloads": ["inference", "vision"],
                "fallback": "RDNA",
                "min_batch": 1,
                "max_batch": 4
            }
        elif self.platform == Vendor.QUALCOMM:
            self.thresholds = {
                "hex_compute_high": 0x70,
                "npu_type": "HEXAGON",
                "npu_workloads": ["speech", "vision", "inference"],
                "fallback": "ADRENO",
                "min_batch": 1,
                "max_batch": 4
            }
        else:  # Generic ARM
            self.thresholds = {
                "hex_compute_high": 0x70,
                "npu_type": "ETHOS",
                "npu_workloads": ["inference", "vision"],
                "fallback": "MALI",
                "min_batch": 1,
                "max_batch": 2
            }

    def analyze_market(self, ticker: MarketTicker) -> Optional[TradeOrder]:
        """Platform-aware NPU trading."""
        commodities = ticker.commodities
        state = ticker.state
        hex_compute = commodities.hex_compute

        # In game mode, reduce batch / defer
        if "GAME" in state.scenario:
            if self.current_batch_size > self.thresholds["min_batch"]:
                return self._create_batch_reduce_request("GAME_MODE_DEFER")

        # High compute - reduce load
        if hex_compute > self.thresholds["hex_compute_high"]:
            if self.current_batch_size > self.thresholds["min_batch"]:
                return self._create_batch_reduce_request("HEX_COMPUTE_HIGH")

        # Low compute - request NPU slot
        if hex_compute < 0x50 and not self.npu_active:
            return self._create_npu_request("HEX_COMPUTE_LOW")

        return None

    def _create_batch_reduce_request(self, reason: str) -> TradeOrder:
        """Create batch reduction request."""
        return TradeOrder(
            source=self.agent_id,
            action=f"REQUEST_BATCH_REDUCE_{self.thresholds['npu_type']}",
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
            action=f"REQUEST_{self.thresholds['npu_type']}_SLOT",
            bid=TradeBid(
                reason=reason,
                est_thermal_saving=2.0,
                priority=4
            )
        )

    def _execute_directive(self, directive: Directive):
        """Execute NPU directive."""
        params = directive.params

        if params.batch_size:
            self.current_batch_size = max(
                self.thresholds["min_batch"],
                min(self.thresholds["max_batch"], params.batch_size)
            )
            self.metrics.thermal_savings += 1.5
            print(f"[NPU/{self.thresholds['npu_type']}] Batch -> {self.current_batch_size}")

        elif "SLOT" in directive.permit_id:
            self.npu_active = True
            self.offload_mode = "NPU"
            print(f"[NPU/{self.thresholds['npu_type']}] Slot acquired")


# ============================================================
# UNIFIED AGENT FACTORY
# ============================================================

class CrossPlatformAgentFactory:
    """
    Create platform-optimized agents.
    """

    def __init__(self, hal: Optional[BaseHAL] = None):
        self.hal = hal or HALFactory.create()

    def create_all(self) -> Dict[str, BaseAgent]:
        """Create all cross-platform agents."""
        return {
            "AGENT_GPU": CrossPlatformGPUTrader(self.hal),
            "AGENT_CPU": CrossPlatformCPUTrader(self.hal),
            "AGENT_NPU": CrossPlatformNPUTrader(self.hal)
        }

    def create_gpu_trader(self) -> CrossPlatformGPUTrader:
        return CrossPlatformGPUTrader(self.hal)

    def create_cpu_trader(self) -> CrossPlatformCPUTrader:
        return CrossPlatformCPUTrader(self.hal)

    def create_npu_trader(self) -> CrossPlatformNPUTrader:
        return CrossPlatformNPUTrader(self.hal)

    def get_platform_info(self) -> Dict:
        """Get platform information."""
        return {
            "vendor": self.hal.info.vendor.name,
            "arch": self.hal.info.arch.name,
            "cores": self.hal.info.core_count,
            "accelerators": [a.name for a in self.hal.info.accelerators],
            "precision_modes": self.hal.get_precision_modes()
        }


# ============================================================
# DEMO
# ============================================================

def demo():
    """Demonstrate cross-platform agents."""
    print("=== Cross-Platform Agents Demo ===\n")

    # Create factory
    factory = CrossPlatformAgentFactory()
    print(f"Platform: {factory.get_platform_info()}\n")

    # Create agents
    agents = factory.create_all()

    for name, agent in agents.items():
        print(f"{name}: platform={agent.platform.name if hasattr(agent, 'platform') else 'N/A'}")

    # Simulate market conditions
    from .crystal_protocol import ProtocolCodec, CommodityPrices, MarketState

    scenarios = [
        {"hex_compute": 0x90, "thermal_headroom_gpu": 8.0, "scenario": "GAME_COMBAT"},
        {"hex_compute": 0x40, "thermal_headroom_gpu": 20.0, "scenario": "IDLE"},
    ]

    codec = ProtocolCodec()

    print("\n--- Market Simulation ---")
    for i, scenario in enumerate(scenarios):
        print(f"\nTick {i}: {scenario['scenario']}")

        ticker = codec.create_ticker(
            cycle=i,
            commodities=scenario,
            scenario=scenario["scenario"]
        )

        for agent in agents.values():
            order = agent.on_ticker(ticker)
            if order:
                print(f"  [{agent.agent_id}] -> {order.action}")


if __name__ == "__main__":
    demo()
