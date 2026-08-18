"""
GAMESA TPU Cross-Forex Resource Trading

Economic trading system for TPU resources that integrates with the
existing cross-forex framework but focuses on TPU-specific resources
like compute units, precision modes, on-chip memory, and thermal headroom.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum, auto
import time
import threading
import uuid
from decimal import Decimal
from datetime import datetime
import random

from . import (
    # Core GAMESA components
    ResourceType, Priority, AllocationRequest, Allocation,
    Effect, Capability, create_guardian_checker,
    Contract, create_guardian_validator,
    TelemetrySnapshot, Signal, SignalKind, Domain,
    Runtime, RuntimeFunc
)
from .tpu_bridge import TPUBoostBridge, TPUPreset, PresetLibrary
from .accelerator_manager import AcceleratorManager, WorkloadRequest
from .platform_hal import BaseHAL, HALFactory
from .cross_forex_memory_trading import (
    CrossForexManager, Portfolio, MarketOrder, MarketOrderType,
    MemoryResourceType as BaseMemoryResourceType
)


# ============================================================
# ENUMS
# ============================================================

class TPUResourceType(Enum):
    """TPU-specific resource types that can be traded."""
    COMPUTE_UNITS = "tpu_compute_units"
    ON_CHIP_MEMORY = "tpu_on_chip_memory"  # Fast TPU memory
    PRECISION_MODE = "tpu_precision_mode"  # FP32, FP16, INT8, etc
    THROUGHPUT_CAPACITY = "tpu_throughput_capacity"  # Inferences/sec
    LATENCY_BUDGET = "tpu_latency_budget"  # Max acceptable latency
    THERMAL_HEADROOM = "tpu_thermal_headroom"  # Temperature headroom
    POWER_BUDGET = "tpu_power_budget"  # Power consumption allowance
    INFERENCE_QUOTA = "tpu_inference_quota"  # Number of inferences
    BANDWIDTH_ALLOCATION = "tpu_bandwidth_allocation"  # Memory bandwidth


class TPUPrecisionMode(Enum):
    """TPU-specific precision modes."""
    FP32 = "FP32"  # Full precision
    FP16 = "FP16"  # Half precision
    BF16 = "BF16"  # Brain floating point
    INT8 = "INT8"  # 8-bit integer
    INT4 = "INT4"  # 4-bit integer
    DYNAMIC_QUANTIZE = "dynamic_quantize"  # Adaptive precision


class TPUTradingStrategy(Enum):
    """Trading strategies for TPU resources."""
    AGGRESSIVE = "aggressive"  # High demand for performance
    CONSERVATIVE = "conservative"  # Low power, thermal safety
    BALANCED = "balanced"  # Balanced performance/power
    OPPORTUNISTIC = "opportunistic"  # Take advantage of price drops


# ============================================================
# DATA CLASSES
# ============================================================

@dataclass
class TPUResourceRequest:
    """Request for TPU resources in the trading system."""
    request_id: str
    agent_id: str
    resource_type: TPUResourceType
    quantity: Decimal
    priority: int = Priority.NORMAL.value
    max_price: Decimal = Decimal('100.00')  # Max price willing to pay
    order_type: MarketOrderType = MarketOrderType.MARKET
    duration_minutes: int = 60  # How long to hold resource
    thermal_constraint: float = 20.0  # Thermal headroom requirement
    power_constraint: float = 15.0  # Power budget constraint
    performance_target: Dict[str, float] = field(default_factory=dict)
    deadline: Optional[float] = None  # Absolute deadline


@dataclass
class TPUResourceAllocation:
    """Result of TPU resource allocation."""
    allocation_id: str
    request_id: str
    agent_id: str
    resource_type: TPUResourceType
    quantity_allocated: Decimal
    price_paid: Decimal
    allocated_at: float
    expires_at: float
    status: str = "active"
    tpu_preset: Optional[TPUPreset] = None
    performance_metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class TPUMarketState:
    """Current state of the TPU resource market."""
    timestamp: float = field(default_factory=time.time)
    resource_prices: Dict[TPUResourceType, Decimal] = field(default_factory=dict)
    supply_levels: Dict[TPUResourceType, Decimal] = field(default_factory=dict)
    demand_levels: Dict[TPUResourceType, Decimal] = field(default_factory=dict)
    volatility: Dict[TPUResourceType, float] = field(default_factory=dict)


@dataclass
class TPUTradingMetrics:
    """Metrics for TPU trading performance."""
    total_trades: int = 0
    total_volume: Decimal = Decimal('0')
    average_price: Decimal = Decimal('0')
    success_rate: float = 1.0
    avg_response_time: float = 0.0
    thermal_efficiency: float = 1.0
    power_efficiency: float = 1.0
    timestamp: float = field(default_factory=time.time)


# ============================================================
# TPU RESOURCE PRICE GENERATOR
# ============================================================

class TPUResourcePriceGenerator:
    """
    Generates realistic TPU resource prices based on:
    - Supply and demand
    - Thermal conditions
    - Power constraints
    - Time of day (for predictive pricing)
    """

    def __init__(self):
        # Base prices for different resources
        self.base_prices = {
            TPUResourceType.COMPUTE_UNITS: Decimal('0.50'),  # $0.50 per compute unit
            TPUResourceType.ON_CHIP_MEMORY: Decimal('0.25'),  # $0.25 per MB
            TPUResourceType.PRECISION_MODE: Decimal('1.00'),  # $1.00 per precision mode
            TPUResourceType.THROUGHPUT_CAPACITY: Decimal('0.10'),  # $0.10 per inference/sec
            TPUResourceType.LATENCY_BUDGET: Decimal('2.00'),  # $2.00 per ms improvement
            TPUResourceType.THERMAL_HEADROOM: Decimal('0.75'),  # $0.75 per degree C
            TPUResourceType.POWER_BUDGET: Decimal('0.20'),  # $0.20 per watt
            TPUResourceType.INFERENCE_QUOTA: Decimal('0.05'),  # $0.05 per inference
            TPUResourceType.BANDWIDTH_ALLOCATION: Decimal('0.15')  # $0.15 per GB/s
        }

        # Price volatility factors
        self.volatility_factors = {
            TPUResourceType.COMPUTE_UNITS: 0.2,  # 20% volatility
            TPUResourceType.ON_CHIP_MEMORY: 0.15,  # 15% volatility
            TPUResourceType.PRECISION_MODE: 0.1,  # 10% volatility
            TPUResourceType.THROUGHPUT_CAPACITY: 0.25,  # 25% volatility
            TPUResourceType.LATENCY_BUDGET: 0.3,  # 30% volatility (highly demanded)
            TPUResourceType.THERMAL_HEADROOM: 0.4,  # 40% volatility (critical resource)
            TPUResourceType.POWER_BUDGET: 0.18,  # 18% volatility
            TPUResourceType.INFERENCE_QUOTA: 0.12,  # 12% volatility
            TPUResourceType.BANDWIDTH_ALLOCATION: 0.22  # 22% volatility
        }

    def get_current_price(self, resource_type: TPUResourceType,
                         supply: Decimal, demand: Decimal,
                         thermal_headroom: float = 20.0,
                         power_budget: float = 15.0) -> Decimal:
        """Get current price for a TPU resource."""
        base_price = self.base_prices[resource_type]

        # Calculate supply/demand ratio
        if supply > 0:
            supply_demand_ratio = demand / supply
        else:
            supply_demand_ratio = Decimal('10')  # Extremely high if no supply

        # Apply supply/demand multiplier (price increases with demand)
        sdr = max(0.1, float(supply_demand_ratio))  # Avoid negative prices
        supply_demand_multiplier = max(0.1, min(5.0, 1.0 + (sdr - 1.0) * 0.5))

        # Apply thermal constraint multiplier
        thermal_multiplier = 1.0
        if thermal_headroom < 10:
            # Thermal scarcity increases prices significantly
            thermal_multiplier = max(1.0, (20.0 - thermal_headroom) / 10.0)

        # Apply power constraint multiplier
        power_multiplier = 1.0
        if power_budget < 5:
            # Power scarcity increases prices
            power_multiplier = max(1.0, (15.0 - power_budget) / 10.0)

        # Apply volatility
        vol_factor = self.volatility_factors[resource_type]
        volatility = random.uniform(1.0 - vol_factor, 1.0 + vol_factor)

        # Calculate final price
        final_price = base_price * supply_demand_multiplier * thermal_multiplier * power_multiplier * volatility

        return Decimal(str(round(final_price, 2)))


# ============================================================
# TPU CROSS-FOREX MANAGER
# ============================================================

class TPUCrossForexManager:
    """
    TPU-specific cross-forex trading system that integrates with
    the broader GAMESA resource trading ecosystem.
    """

    def __init__(self, hal: Optional[BaseHAL] = None):
        self.hal = hal or HALFactory.create()
        self.price_generator = TPUResourcePriceGenerator()
        self.trading_lock = threading.RLock()
        self.trade_history = []
        self.metrics = TPUTradingMetrics()
        self.active_allocations = {}
        self.market_state = TPUMarketState()

        # Initialize market state
        self._initialize_market()

    def _initialize_market(self):
        """Initialize the TPU resource market state."""
        with self.trading_lock:
            for resource_type in TPUResourceType:
                # Start with balanced supply and demand
                self.market_state.supply_levels[resource_type] = Decimal('100.0')
                self.market_state.demand_levels[resource_type] = Decimal('80.0')
                # Set initial price
                self.market_state.resource_prices[resource_type] = self.price_generator.get_current_price(
                    resource_type,
                    self.market_state.supply_levels[resource_type],
                    self.market_state.demand_levels[resource_type]
                )

    def request_resources(self, request: TPUResourceRequest) -> Optional[TPUResourceAllocation]:
        """Request TPU resources through the trading system."""
        with self.trading_lock:
            start_time = time.time()

            # Validate request against thermal and power constraints
            if not self._validate_constraints(request):
                print(f"Request {request.request_id} failed constraints validation")
                return None

            # Calculate market price
            market_price = self.price_generator.get_current_price(
                request.resource_type,
                self.market_state.supply_levels[request.resource_type],
                self.market_state.demand_levels[request.resource_type],
                request.thermal_constraint,
                request.power_constraint
            )

            # Check if price is acceptable
            if market_price > request.max_price:
                print(f"Request {request.request_id}: Market price ${market_price} exceeds max ${request.max_price}")
                return None

            # Calculate quantity to allocate
            if request.order_type == MarketOrderType.MARKET:
                quantity = min(request.quantity, self.market_state.supply_levels[request.resource_type])
            else:  # LIMIT order would be handled differently
                quantity = min(request.quantity, self.market_state.supply_levels[request.resource_type])

            if quantity <= 0:
                print(f"Request {request.request_id}: Insufficient supply")
                return None

            # Calculate cost
            total_cost = market_price * quantity

            # Process the allocation
            allocation = self._process_allocation(
                request, quantity, market_price, total_cost
            )

            if allocation:
                # Update market state
                self._update_market_state(request.resource_type, quantity, market_price)

                # Update metrics
                self.metrics.total_trades += 1
                self.metrics.total_volume += total_cost
                avg_price = self.metrics.total_volume / self.metrics.total_trades
                self.metrics.average_price = avg_price

                response_time = time.time() - start_time
                self.metrics.avg_response_time = (self.metrics.avg_response_time + response_time) / 2

                # Record trade
                self.trade_history.append({
                    'timestamp': start_time,
                    'request_id': request.request_id,
                    'allocation_id': allocation.allocation_id,
                    'resource_type': request.resource_type.value,
                    'quantity': float(quantity),
                    'price': float(market_price),
                    'total_cost': float(total_cost)
                })

                if len(self.trade_history) > 10000:
                    self.trade_history.pop(0)

                return allocation

            return None

    def _validate_constraints(self, request: TPUResourceRequest) -> bool:
        """Validate thermal and power constraints."""
        # Check thermal constraint
        if request.thermal_constraint < 0:
            return False

        # Check power constraint
        if request.power_constraint <= 0:
            return False

        return True

    def _process_allocation(self, request: TPUResourceRequest, quantity: Decimal,
                           unit_price: Decimal, total_cost: Decimal) -> Optional[TPUResourceAllocation]:
        """Process the actual allocation."""
        # Select appropriate TPU preset based on resource type and requirements
        preset = self._select_preset_for_resource(request)

        allocation = TPUResourceAllocation(
            allocation_id=f"TPU_ALLOC_{uuid.uuid4().hex[:8]}",
            request_id=request.request_id,
            agent_id=request.agent_id,
            resource_type=request.resource_type,
            quantity_allocated=quantity,
            price_paid=total_cost,
            allocated_at=time.time(),
            expires_at=time.time() + (request.duration_minutes * 60),
            tpu_preset=preset
        )

        self.active_allocations[allocation.allocation_id] = allocation
        return allocation

    def _select_preset_for_resource(self, request: TPUResourceRequest) -> Optional[TPUPreset]:
        """Select appropriate TPU preset based on resource request."""
        # Map resource type to preset selection logic
        if request.resource_type == TPUResourceType.PRECISION_MODE:
            # Based on performance_target, select precision
            if request.performance_target.get('latency_improvement', 0) > 0.5:
                return PresetLibrary.get("LOW_LATENCY_FP16")
            else:
                return PresetLibrary.get("HIGH_THROUGHPUT_INT8")
        elif request.resource_type == TPUResourceType.THROUGHPUT_CAPACITY:
            if request.quantity > Decimal('500'):
                return PresetLibrary.get("HIGH_THROUGHPUT_FP16")
            else:
                return PresetLibrary.get("LOW_LATENCY_FP16")
        elif request.resource_type == TPUResourceType.THERMAL_HEADROOM:
            return PresetLibrary.get("EFFICIENT_GNA")  # GNA is very power efficient
        elif request.resource_type == TPUResourceType.ON_CHIP_MEMORY:
            # High memory request needs performance preset
            return PresetLibrary.get("HIGH_THROUGHPUT_FP16")
        else:
            # Default to balanced preset
            return PresetLibrary.get("LOW_LATENCY_FP16")

    def _update_market_state(self, resource_type: TPUResourceType, quantity: Decimal, price: Decimal):
        """Update market supply/demand based on trade."""
        # Reduce supply by allocated quantity
        self.market_state.supply_levels[resource_type] -= quantity

        # Update demand based on successful allocation
        # If allocation was successful, demand decreases slightly
        self.market_state.demand_levels[resource_type] = max(
            Decimal('0'),
            self.market_state.demand_levels[resource_type] - (quantity * Decimal('0.1'))
        )

        # Update price based on new supply/demand
        self.market_state.resource_prices[resource_type] = self.price_generator.get_current_price(
            resource_type,
            self.market_state.supply_levels[resource_type],
            self.market_state.demand_levels[resource_type]
        )

        # Update timestamp
        self.market_state.timestamp = time.time()

    def get_market_state(self) -> TPUMarketState:
        """Get current market state."""
        with self.trading_lock:
            return self.market_state

    def get_trading_metrics(self) -> TPUTradingMetrics:
        """Get trading performance metrics."""
        with self.trading_lock:
            return self.metrics

    def adjust_supply_for_resource(self, resource_type: TPUResourceType, adjustment: Decimal):
        """Manually adjust supply for a resource (used for dynamic rebalancing)."""
        with self.trading_lock:
            self.market_state.supply_levels[resource_type] += adjustment
            # Ensure supply doesn't go negative
            self.market_state.supply_levels[resource_type] = max(
                Decimal('0'),
                self.market_state.supply_levels[resource_type]
            )

    def optimize_for_trading_strategy(self, strategy: TPUTradingStrategy,
                                    thermal_headroom: float = 20.0,
                                    power_budget: float = 15.0) -> Dict[TPUResourceType, Decimal]:
        """
        Optimize resource allocation based on trading strategy.
        Returns suggested quantities for each resource type.
        """
        suggestions = {}

        if strategy == TPUTradingStrategy.AGGRESSIVE:
            # Prioritize performance resources
            suggestions[TPUResourceType.COMPUTE_UNITS] = Decimal('100.0')
            suggestions[TPUResourceType.ON_CHIP_MEMORY] = Decimal('50.0')
            suggestions[TPUResourceType.PRECISION_MODE] = Decimal('1.0')  # High precision
            suggestions[TPUResourceType.THROUGHPUT_CAPACITY] = Decimal('800.0')
            suggestions[TPUResourceType.LATENCY_BUDGET] = Decimal('10.0')  # Low latency
        elif strategy == TPUTradingStrategy.CONSERVATIVE:
            # Prioritize efficiency and safety
            suggestions[TPUResourceType.THERMAL_HEADROOM] = Decimal('15.0')
            suggestions[TPUResourceType.POWER_BUDGET] = Decimal('5.0')
            suggestions[TPUResourceType.PRECISION_MODE] = Decimal('1.0')
            suggestions[TPUResourceType.COMPUTE_UNITS] = Decimal('30.0')
        elif strategy == TPUTradingStrategy.BALANCED:
            # Balanced approach
            suggestions[TPUResourceType.COMPUTE_UNITS] = Decimal('60.0')
            suggestions[TPUResourceType.ON_CHIP_MEMORY] = Decimal('25.0')
            suggestions[TPUResourceType.POWER_BUDGET] = Decimal('10.0')
            suggestions[TPUResourceType.THROUGHPUT_CAPACITY] = Decimal('200.0')
        else:  # OPPORTUNISTIC
            # Take advantage of low prices
            for resource_type in TPUResourceType:
                current_price = self.market_state.resource_prices[resource_type]
                base_price = self.price_generator.base_prices[resource_type]
                # Buy more if price is below base
                if current_price < base_price * Decimal('0.8'):
                    suggestions[resource_type] = Decimal('50.0')
                else:
                    suggestions[resource_type] = Decimal('10.0')

        return suggestions

    def get_resource_recommendations(self, telemetry: TelemetrySnapshot) -> List[TPUResourceRequest]:
        """
        Generate resource trading recommendations based on system telemetry.
        """
        recommendations = []

        # High CPU utilization suggests TPU offloading opportunity
        if telemetry.cpu_util > 0.85:
            request = TPUResourceRequest(
                request_id=f"RECOMMEND_CPU_OFFLOAD_{uuid.uuid4().hex[:8]}",
                agent_id="TELEMETRY_ANALYZER",
                resource_type=TPUResourceType.COMPUTE_UNITS,
                quantity=Decimal('200.0'),
                priority=9,
                max_price=Decimal('200.00'),
                thermal_constraint=telemetry.temp_cpu if hasattr(telemetry, 'temp_cpu') else 20.0,
                power_constraint=15.0,
                performance_target={"cpu_offload": 0.5}
            )
            recommendations.append(request)

        # High thermal pressure suggests efficiency focus
        if hasattr(telemetry, 'temp_tpu'):
            if telemetry.temp_tpu > 75:
                request = TPUResourceRequest(
                    request_id=f"RECOMMEND_EFFICIENCY_{uuid.uuid4().hex[:8]}",
                    agent_id="THERMAL_MANAGER",
                    resource_type=TPUResourceType.POWER_BUDGET,
                    quantity=Decimal('5.0'),
                    priority=10,
                    max_price=Decimal('50.00'),
                    thermal_constraint=15.0,  # Very conservative thermal budget
                    power_constraint=5.0,
                    performance_target={"thermal_stability": 0.9}
                )
                recommendations.append(request)

        # Memory pressure might indicate TPU offloading
        if telemetry.memory_util > 0.85:
            request = TPUResourceRequest(
                request_id=f"RECOMMEND_MEMORY_OFFLOAD_{uuid.uuid4().hex[:8]}",
                agent_id="MEMORY_MANAGER",
                resource_type=TPUResourceType.COMPUTE_UNITS,
                quantity=Decimal('100.0'),
                priority=7,
                max_price=Decimal('100.00'),
                thermal_constraint=20.0,
                power_constraint=10.0,
                performance_target={"memory_utilization_reduction": 0.3}
            )
            recommendations.append(request)

        return recommendations

    def cleanup_expired_allocations(self) -> int:
        """Clean up expired allocations."""
        cleaned = 0
        now = time.time()

        expired_ids = []
        for alloc_id, allocation in self.active_allocations.items():
            if allocation.expires_at < now:
                expired_ids.append(alloc_id)
                cleaned += 1

        for alloc_id in expired_ids:
            del self.active_allocations[alloc_id]

        return cleaned


# ============================================================
# TPU TRADING CONTROLLER
# ============================================================

class TPUTradingController:
    """Controller for TPU resource trading operations."""

    def __init__(self, hal: Optional[BaseHAL] = None):
        self.tpu_manager = TPUCrossForexManager(hal)
        self.performance_monitor = TPUTradingPerformanceMonitor(self.tpu_manager)
        self.strategy_engine = TPUTradingStrategyEngine(self.tpu_manager)

    def process_trading_cycle(self, telemetry: TelemetrySnapshot,
                            signals: List[Signal]) -> Dict[str, Any]:
        """Process one trading cycle."""
        results = {
            'resource_requests': [],
            'allocations_made': [],
            'signals_processed': 0,
            'recommendations_made': 0,
            'actions_taken': []
        }

        # Generate recommendations based on telemetry
        recommendations = self.tpu_manager.get_resource_recommendations(telemetry)
        results['resource_requests'].extend(recommendations)
        results['recommendations_made'] = len(recommendations)

        # Process signals to generate additional requests
        for signal in signals:
            signal_requests = self._process_signal_for_trading(signal)
            results['resource_requests'].extend(signal_requests)
            results['signals_processed'] += 1

        # Execute resource requests
        for request in results['resource_requests']:
            allocation = self.tpu_manager.request_resources(request)
            if allocation:
                results['allocations_made'].append(allocation)
                results['actions_taken'].append(
                    f"Traded {request.quantity} {request.resource_type.value} for ${allocation.price_paid}"
                )

        # Clean up expired allocations
        expired_count = self.tpu_manager.cleanup_expired_allocations()
        if expired_count > 0:
            results['actions_taken'].append(f"Cleaned up {expired_count} expired allocations")

        return results

    def _process_signal_for_trading(self, signal: Signal) -> List[TPUResourceRequest]:
        """Process GAMESA signals for TPU resource trading."""
        requests = []

        if signal.kind in [SignalKind.CPU_BOTTLENECK, SignalKind.GPU_BOTTLENECK]:
            request = TPUResourceRequest(
                request_id=f"SIG_TPU_OFFLOAD_{signal.id}",
                agent_id="SIGNAL_PROCESSOR",
                resource_type=TPUResourceType.COMPUTE_UNITS,
                quantity=Decimal('300.0'),
                priority=9,
                max_price=Decimal('300.00'),
                thermal_constraint=20.0,
                power_constraint=15.0,
                performance_target={"bottleneck_relief": signal.strength}
            )
            requests.append(request)

        elif signal.kind == SignalKind.THERMAL_WARNING:
            request = TPUResourceRequest(
                request_id=f"SIG_THERMAL_OPT_{signal.id}",
                agent_id="THERMAL_GUARDIAN",
                resource_type=TPUResourceType.POWER_BUDGET,
                quantity=Decimal('3.0'),
                priority=10,
                max_price=Decimal('50.00'),
                thermal_constraint=15.0,
                power_constraint=3.0,
                performance_target={"thermal_stability": 0.95}
            )
            requests.append(request)

        elif signal.kind == SignalKind.USER_BOOST_REQUEST:
            request = TPUResourceRequest(
                request_id=f"SIG_USER_BOOST_{signal.id}",
                agent_id="USER_AGENT",
                resource_type=TPUResourceType.THROUGHPUT_CAPACITY,
                quantity=Decimal('500.0'),
                priority=10,
                max_price=Decimal('200.00'),
                thermal_constraint=15.0,
                power_constraint=10.0,
                performance_target={"performance_boost": 0.8}
            )
            requests.append(request)

        return requests

    def optimize_based_on_strategy(self, strategy: TPUTradingStrategy) -> List[TPUResourceRequest]:
        """Generate resource requests based on trading strategy."""
        suggestions = self.tpu_manager.optimize_for_trading_strategy(strategy)
        requests = []

        for resource_type, quantity in suggestions.items():
            request = TPUResourceRequest(
                request_id=f"STRAT_{strategy.value}_{resource_type.value}_{uuid.uuid4().hex[:6]}",
                agent_id="STRATEGY_ENGINE",
                resource_type=resource_type,
                quantity=quantity,
                priority=7,
                max_price=quantity * Decimal('5'),  # Estimate max price based on quantity
                thermal_constraint=20.0,
                power_constraint=15.0
            )
            requests.append(request)

        return requests

    def get_status_report(self) -> Dict[str, Any]:
        """Get comprehensive trading status report."""
        market_state = self.tpu_manager.get_market_state()
        trading_metrics = self.tpu_manager.get_trading_metrics()

        return {
            'market_state': {
                'timestamp': market_state.timestamp,
                'resource_prices': {k.value: float(v) for k, v in market_state.resource_prices.items()},
                'supply_levels': {k.value: float(v) for k, v in market_state.supply_levels.items()},
                'demand_levels': {k.value: float(v) for k, v in market_state.demand_levels.items()},
            },
            'trading_metrics': {
                'total_trades': trading_metrics.total_trades,
                'total_volume': float(trading_metrics.total_volume),
                'average_price': float(trading_metrics.average_price),
                'success_rate': trading_metrics.success_rate,
                'avg_response_time': trading_metrics.avg_response_time,
                'active_allocations': len(self.tpu_manager.active_allocations)
            },
            'timestamp': time.time()
        }


class TPUTradingPerformanceMonitor:
    """Monitors TPU trading performance metrics."""

    def __init__(self, manager: TPUCrossForexManager):
        self.manager = manager
        self.performance_history = []

    def collect_metrics(self) -> Dict[str, Any]:
        """Collect current trading performance metrics."""
        metrics = self.manager.get_trading_metrics()
        market_state = self.manager.get_market_state()

        performance = {
            'total_trades': metrics.total_trades,
            'total_volume': float(metrics.total_volume),
            'average_price': float(metrics.average_price),
            'success_rate': metrics.success_rate,
            'avg_response_time': metrics.avg_response_time,
            'most_expensive_resource': max(
                market_state.resource_prices.items(),
                key=lambda x: x[1]
            ) if market_state.resource_prices else ("none", 0),
            'least_available_resource': min(
                market_state.supply_levels.items(),
                key=lambda x: x[1]
            ) if market_state.supply_levels else ("none", 0)
        }

        self.performance_history.append((time.time(), performance))
        if len(self.performance_history) > 100:
            self.performance_history.pop(0)

        return performance


class TPUTradingStrategyEngine:
    """Applies trading strategies to optimize resource allocation."""

    def __init__(self, manager: TPUCrossForexManager):
        self.manager = manager
        self.active_strategies = []

    def apply_strategy(self, strategy: TPUTradingStrategy,
                      thermal_headroom: float = 20.0,
                      power_budget: float = 15.0) -> bool:
        """Apply a trading strategy."""
        suggestions = self.manager.optimize_for_trading_strategy(
            strategy, thermal_headroom, power_budget
        )

        success_count = 0
        for resource_type, quantity in suggestions.items():
            request = TPUResourceRequest(
                request_id=f"STRAT_{strategy.value}_{uuid.uuid4().hex[:8]}",
                agent_id="STRATEGY_ENGINE",
                resource_type=resource_type,
                quantity=quantity,
                priority=7,
                max_price=quantity * Decimal('3'),  # Conservative max price
                thermal_constraint=thermal_headroom,
                power_constraint=power_budget
            )

            allocation = self.manager.request_resources(request)
            if allocation:
                success_count += 1

        return success_count > 0


# ============================================================
# DEMO
# ============================================================

def demo_tpu_cross_forex_trading():
    """Demonstrate TPU cross-forex resource trading."""
    print("=== GAMESA TPU Cross-Forex Trading Demo ===\n")

    # Initialize controller
    controller = TPUTradingController()
    print("TPU Trading Controller initialized")

    # Simulate telemetry data
    telemetry = TelemetrySnapshot(
        timestamp=datetime.now().isoformat(),
        cpu_util=0.90,  # High CPU utilization
        gpu_util=0.60,  # Moderate GPU utilization
        temp_cpu=78,    # 78°C CPU
        temp_gpu=65,    # 65°C GPU
        memory_util=0.85,
        frametime_ms=22.0,
        active_process_category="ai_inference"
    )

    print(f"Input Telemetry: CPU={telemetry.cpu_util*100:.1f}%, GPU={telemetry.gpu_util*100:.1f}%, "
          f"Mem={telemetry.memory_util*100:.1f}%")

    # Simulate signals
    signals = [
        Signal(
            id="SIGNAL_001",
            source="TELEMETRY",
            kind=SignalKind.CPU_BOTTLENECK,
            strength=0.85,
            confidence=0.9,
            payload={"bottleneck_type": "compute", "recommended_action": "tpu_offload"}
        ),
        Signal(
            id="SIGNAL_002",
            source="TELEMETRY",
            kind=SignalKind.MEMORY_PRESSURE,
            strength=0.75,
            confidence=0.85,
            payload={"memory_util": 0.85, "recommended_action": "memory_offload"}
        )
    ]

    print(f"\nProcessing {len(signals)} signals...")

    # Execute trading cycle
    results = controller.process_trading_cycle(telemetry, signals)

    print(f"\nTrading Cycle Results:")
    print(f"  Resource Requests Generated: {len(results['resource_requests'])}")
    print(f"  Allocations Made: {len(results['allocations_made'])}")
    print(f"  Recommendations Made: {results['recommendations_made']}")
    print(f"  Signals Processed: {results['signals_processed']}")
    print(f"  Actions Taken: {len(results['actions_taken'])}")

    if results['actions_taken']:
        print(f"\nActions Taken:")
        for action in results['actions_taken']:
            print(f"  - {action}")

    # Show market state
    print(f"\nMarket State:")
    status = controller.get_status_report()
    for resource, price in status['market_state']['resource_prices'].items():
        supply = status['market_state']['supply_levels'][resource]
        demand = status['market_state']['demand_levels'][resource]
        print(f"  {resource}: ${price:.2f} (Supply: {supply:.0f}, Demand: {demand:.0f})")

    print(f"\nTrading Metrics:")
    metrics = status['trading_metrics']
    print(f"  Total Trades: {metrics['total_trades']}")
    print(f"  Total Volume: ${metrics['total_volume']:.2f}")
    print(f"  Average Price: ${metrics['average_price']:.2f}")
    print(f"  Success Rate: {metrics['success_rate']:.2%}")
    print(f"  Avg Response Time: {metrics['avg_response_time']:.4f}s")
    print(f"  Active Allocations: {metrics['active_allocations']}")

    # Test different strategies
    print(f"\nTesting Trading Strategies:")
    strategies = [TPUTradingStrategy.AGGRESSIVE, TPUTradingStrategy.CONSERVATIVE, TPUTradingStrategy.BALANCED]
    for strategy in strategies:
        reqs = controller.optimize_based_on_strategy(strategy)
        print(f"  {strategy.value}: Generated {len(reqs)} requests")

    print(f"\nGAMESA TPU Cross-Forex Trading demo completed successfully!")


if __name__ == "__main__":
    demo_tpu_cross_forex_trading()