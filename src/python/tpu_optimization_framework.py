"""
GAMESA TPU Optimization Framework

Comprehensive TPU-specific performance optimization system that integrates
with the existing GAMESA architecture for economic resource trading,
3D grid memory management, and safety-validated operations.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum, auto
import time
import threading
import uuid
from datetime import datetime
from decimal import Decimal

from . import (
    ResourceType, Priority, AllocationRequest, Allocation,
    Effect, Capability, create_guardian_checker,
    Contract, create_guardian_validator,
    TelemetrySnapshot, Signal, SignalKind, Domain,
    Runtime, RuntimeVar, RuntimeFunc
)
from .tpu_bridge import (
    TPUBoostBridge, TPUPreset, PresetLibrary,
    AcceleratorType, PrecisionMode, WorkloadType
)
from .accelerator_manager import (
    AcceleratorManager, AcceleratorAssignment, WorkloadRequest
)
from .platform_hal import BaseHAL, HALFactory
from .cross_forex_memory_trading import CrossForexManager
from .memory_coherence_protocol import MemoryCoherenceProtocol


# ============================================================
# ENUMS
# ============================================================

class TPUOptimizationStrategy(Enum):
    """TPU optimization strategies."""
    PERFORMANCE_FIRST = "performance_first"
    POWER_EFFICIENCY = "power_efficiency"
    THERMAL_AWARE = "thermal_aware"
    COST_OPTIMIZED = "cost_optimized"
    BALANCED = "balanced"


class TPUResourceType(Enum):
    """Types of TPU resources that can be traded."""
    COMPUTE_UNITS = "compute_units"
    ON_CHIP_MEMORY = "on_chip_memory"
    HOST_MEMORY = "host_memory"
    PRECISION_MODE = "precision_mode"
    THROUGHPUT_BUDGET = "throughput_budget"
    LATENCY_BUDGET = "latency_budget"
    THERMAL_HEADROOM = "thermal_headroom"
    POWER_BUDGET = "power_budget"


class TPUAllocationStrategy(Enum):
    """Strategies for TPU resource allocation."""
    ROUND_ROBIN = "round_robin"
    LOAD_BALANCED = "load_balanced"
    PRIORITY_BASED = "priority_based"
    ECONOMIC_BIDDING = "economic_bidding"
    THERMAL_AWARE = "thermal_aware"


# ============================================================
# DATA CLASSES
# ============================================================

@dataclass
class TPUAllocationRequest:
    """Request for TPU resources."""
    request_id: str
    agent_id: str
    resource_type: TPUResourceType
    amount: float
    priority: int = Priority.NORMAL.value
    duration_ms: int = 1000
    bid_credits: Decimal = Decimal('10.0')
    constraints: Dict[str, Any] = field(default_factory=dict)
    tpu_preference: Optional[AcceleratorType] = None
    performance_goals: Dict[str, float] = field(default_factory=dict)
    thermal_budget: float = 20.0
    power_budget: float = 15.0


@dataclass
class TPUAllocation:
    """Granted TPU resource allocation."""
    allocation_id: str
    request_id: str
    agent_id: str
    resource_type: TPUResourceType
    amount: float
    tpu_assigned: Optional[AcceleratorType]
    granted_at: float
    expires_at: float
    status: str = "active"
    trading_cost: Decimal = Decimal('0')
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    preset_config: Optional[TPUPreset] = None


@dataclass
class TPUIntegrationMetrics:
    """Metrics for TPU integration with GAMESA."""
    pipeline_utilization: float = 0.0
    memory_efficiency: float = 0.0
    coherence_success_rate: float = 1.0
    cross_forex_volume: Decimal = Decimal('0')
    average_latency_us: float = 0.0
    power_consumption_w: float = 0.0
    thermal_efficiency: float = 1.0
    tpu_switching_events: int = 0
    resource_trading_events: int = 0
    timestamp: float = field(default_factory=time.time)


@dataclass
class TPUConfig:
    """Configuration for TPU optimization."""
    enable_cross_forex: bool = True
    enable_coherence: bool = True
    optimization_strategy: TPUOptimizationStrategy = TPUOptimizationStrategy.BALANCED
    allocation_strategy: TPUAllocationStrategy = TPUAllocationStrategy.ECONOMIC_BIDDING
    coherence_timeout_secs: float = 0.001
    max_trading_credits: Decimal = Decimal('1000.00')
    thermal_safety_margin: float = 10.0
    power_limit_w: float = 15.0


@dataclass
class TPUPerformanceProfile:
    """Performance profile for different TPU configurations."""
    name: str
    accelerator_type: AcceleratorType
    precision_mode: PrecisionMode
    max_compute_units: float
    memory_bandwidth_gb_s: float
    peak_tflops: float
    power_efficiency: float  # TFLOPS/W
    thermal_profile: Dict[str, float]  # temp vs utilization
    latency_profile: Dict[str, float]  # latency vs batch size


# ============================================================
# TPU OPTIMIZATION MANAGER
# ============================================================

class TPUOptimizationManager:
    """
    Main TPU optimization manager that integrates with GAMESA systems.
    Manages TPU resources, applies optimization strategies, and ensures
    safety through GAMESA contracts and validation.
    """

    def __init__(self, config: Optional[TPUConfig] = None):
        self.config = config or TPUConfig()
        self.metrics = TPUIntegrationMetrics()
        self.integration_lock = threading.RLock()

        # Core components
        self.tpu_bridge = TPUBoostBridge()
        self.accelerator_manager = AcceleratorManager()
        self.hal = HALFactory.create()
        self.cross_forex_manager = CrossForexManager()
        self.coherence_protocol = MemoryCoherenceProtocol()

        # GAMESA integration
        self.effect_checker = create_guardian_checker()
        self.contract_validator = create_guardian_validator()
        self.runtime = Runtime()

        # Performance tracking
        self.performance_history = []
        self.allocation_history = []

        # Initialize TPU optimization
        self._initialize_optimization()

    def _initialize_optimization(self):
        """Initialize TPU optimization components."""
        with self.integration_lock:
            print("Initializing TPU Optimization Framework...")

            # Validate TPU capabilities
            if not self._validate_tpu_capabilities():
                print("Warning: TPU capabilities not fully validated")

            # Initialize economic trading
            if self.config.enable_cross_forex:
                portfolio = self.cross_forex_manager.memory_engine.create_portfolio("TPU_SYSTEM")
                if not portfolio:
                    print("Warning: Could not create TPU cross-forex portfolio")

            # Initialize coherence protocol
            if self.config.enable_coherence:
                self.coherence_protocol.initialize_coherence()

            print("TPU Optimization Framework initialized successfully")

    def _validate_tpu_capabilities(self) -> bool:
        """Validate that TPU capabilities are available."""
        # Check for available accelerators
        available = self.accelerator_manager.get_available_accelerators()
        tpu_types = [
            AcceleratorType.GNA, AcceleratorType.XDNA,
            AcceleratorType.ETHOS, AcceleratorType.HEXAGON
        ]

        has_tpu = any(accel in available for accel in tpu_types)
        return has_tpu

    def request_tpu_resources(self, request: TPUAllocationRequest) -> Optional[TPUAllocation]:
        """Request TPU resources with safety validation."""
        with self.integration_lock:
            print(f"Processing TPU allocation request: {request.request_id}")

            # Validate request against safety contracts
            contract_result = self.contract_validator.check_invariants("tpu_allocation", {
                "resource_type": request.resource_type.value,
                "amount": request.amount,
                "priority": request.priority,
                "thermal_budget": request.thermal_budget,
                "power_budget": request.power_budget
            })

            if not contract_result.valid:
                print(f"TPU allocation contract validation failed: {contract_result.errors}")
                return None

            # Determine optimal TPU based on allocation strategy
            tpu_assignment = self._determine_tpu_assignment(request)

            if tpu_assignment is None:
                print(f"No suitable TPU available for request {request.request_id}")
                return None

            # Select optimal preset based on request
            preset = self._select_optimal_preset(request, tpu_assignment)

            # Create the TPU allocation
            allocation_id = f"TPU_ALLOC_{uuid.uuid4().hex[:8]}"
            now = time.time()

            tpu_allocation = TPUAllocation(
                allocation_id=allocation_id,
                request_id=request.request_id,
                agent_id=request.agent_id,
                resource_type=request.resource_type,
                amount=request.amount,
                tpu_assigned=tpu_assignment,
                granted_at=now,
                expires_at=now + (request.duration_ms / 1000.0),
                trading_cost=request.bid_credits,
                status="active",
                preset_config=preset
            )

            # Update metrics
            self.metrics.resource_trading_events += 1

            # Track allocation
            self.allocation_history.append(tpu_allocation)
            if len(self.allocation_history) > 1000:
                self.allocation_history.pop(0)

            print(f"TPU allocation successful: {allocation_id} on {tpu_assignment.name}")
            return tpu_allocation

    def _determine_tpu_assignment(self, request: TPUAllocationRequest) -> Optional[AcceleratorType]:
        """Determine which TPU to assign based on allocation strategy."""
        # Get available TPU accelerators
        available_tpus = self._get_available_tpus()

        if not available_tpus:
            return None

        # Apply allocation strategy
        if self.config.allocation_strategy == TPUAllocationStrategy.PRIORITY_BASED:
            # Prioritize based on request priority and thermal budget
            if request.priority >= 8:
                # High priority requests get best performance TPUs
                high_perf_tpus = [t for t in available_tpus if self._is_high_performance(t)]
                if high_perf_tpus and request.thermal_budget > 15:
                    return high_perf_tpus[0]
                elif high_perf_tpus:
                    # Use power-efficient TPU instead
                    return self._get_power_efficient_tpu(available_tpus)

            # Check thermal budget constraint
            if request.thermal_budget < self.config.thermal_safety_margin:
                # Use power-efficient TPU to avoid overheating
                return self._get_power_efficient_tpu(available_tpus)

            return available_tpus[0]  # Default assignment

        elif self.config.allocation_strategy == TPUAllocationStrategy.ECONOMIC_BIDDING:
            # Use economic bidding system
            winning_bid = 0
            winner_tpu = available_tpus[0]

            for tpu_id in available_tpus:
                bid_value = self._calculate_tpu_bid_value(tpu_id, request)
                if bid_value > winning_bid:
                    winning_bid = bid_value
                    winner_tpu = tpu_id

            return winner_tpu

        elif self.config.allocation_strategy == TPUAllocationStrategy.THERMAL_AWARE:
            # Select TPU based on thermal conditions
            thermal_headroom = self.hal.get_thermal_headroom("tpu")
            if thermal_headroom < 15:
                # Use power-efficient TPU under thermal stress
                return self._get_power_efficient_tpu(available_tpus)
            else:
                # Use performance TPU under normal conditions
                return self._get_performance_tpu(available_tpus)

        elif self.config.allocation_strategy == TPUAllocationStrategy.LOAD_BALANCED:
            # Choose TPU based on current load
            loads = {}
            for tpu_id in available_tpus:
                loads[tpu_id] = self._get_tpu_load(tpu_id)

            # Return TPU with minimum load
            return min(loads, key=loads.get)

        else:  # ROUND_ROBIN
            # Simple round robin assignment
            if hasattr(self, '_round_robin_index'):
                index = getattr(self, '_round_robin_index', 0)
                assigned_tpu = available_tpus[index % len(available_tpus)]
                self._round_robin_index = (index + 1) % len(available_tpus)
            else:
                self._round_robin_index = 1
                assigned_tpu = available_tpus[0]

            return assigned_tpu

    def _select_optimal_preset(self, request: TPUAllocationRequest,
                               tpu_type: AcceleratorType) -> Optional[TPUPreset]:
        """Select optimal TPU preset based on request requirements."""
        # Get available presets for the TPU type
        if tpu_type == AcceleratorType.GNA:
            return PresetLibrary.get("EFFICIENT_GNA")
        elif tpu_type in [AcceleratorType.XDNA, AcceleratorType.ETHOS]:
            if request.performance_goals.get('latency', 10.0) < 10:  # Low latency requirement
                return PresetLibrary.get("LOW_LATENCY_FP16")
            else:  # Balanced or throughput
                return PresetLibrary.get("HIGH_THROUGHPUT_INT8")
        elif tpu_type == AcceleratorType.HEXAGON:
            return PresetLibrary.get("VPU_VISION")
        else:  # Default or fallback
            return PresetLibrary.get("CPU_FALLBACK")

    def _get_available_tpus(self) -> List[AcceleratorType]:
        """Get list of available TPU accelerators."""
        # Get all accelerators from the manager
        all_accelerators = self.accelerator_manager.get_available_accelerators()

        # TPU-specific accelerators
        tpu_types = [
            AcceleratorType.GNA,      # Gaussian Neural Accelerator
            AcceleratorType.XDNA,     # XDNA (AMD)
            AcceleratorType.ETHOS,    # Ethos (ARM)
            AcceleratorType.HEXAGON,  # Hexagon (Qualcomm)
            AcceleratorType.ANE       # Apple Neural Engine
        ]

        # Filter for TPU-like accelerators
        available_tpus = [accel for accel in all_accelerators if accel in tpu_types]
        return available_tpus

    def _is_high_performance(self, tpu_type: AcceleratorType) -> bool:
        """Check if TPU is high performance."""
        high_perf = [
            AcceleratorType.XDNA,   # AMD XDNA
            AcceleratorType.HEXAGON, # Qualcomm Hexagon
            AcceleratorType.ANE     # Apple Neural Engine
        ]
        return tpu_type in high_perf

    def _get_power_efficient_tpu(self, available_tpus: List[AcceleratorType]) -> AcceleratorType:
        """Get most power-efficient TPU from available list."""
        # Prioritize GNA, Ethos, then others
        priority_order = [
            AcceleratorType.GNA,    # Most power efficient
            AcceleratorType.ETHOS,  # ARM Ethos
            AcceleratorType.ANE,    # Apple ANE
            AcceleratorType.HEXAGON, # Qualcomm Hexagon
            AcceleratorType.XDNA    # AMD XDNA
        ]

        for tpu_type in priority_order:
            if tpu_type in available_tpus:
                return tpu_type

        return available_tpus[0]  # Fallback to first available

    def _get_performance_tpu(self, available_tpus: List[AcceleratorType]) -> AcceleratorType:
        """Get most performance-oriented TPU from available list."""
        # Prioritize performance accelerators
        perf_priority = [
            AcceleratorType.XDNA,   # AMD XDNA
            AcceleratorType.HEXAGON, # Qualcomm Hexagon
            AcceleratorType.ANE,    # Apple ANE
            AcceleratorType.GNA,    # GNA (if needed for performance)
            AcceleratorType.ETHOS   # ARM Ethos
        ]

        for tpu_type in perf_priority:
            if tpu_type in available_tpus:
                return tpu_type

        return available_tpus[0]  # Fallback to first available

    def _calculate_tpu_bid_value(self, tpu_id: AcceleratorType, request: TPUAllocationRequest) -> int:
        """Calculate bid value for economic TPU allocation."""
        # Base value based on available resources
        base_value = 100

        # Adjust based on performance capabilities
        capabilities = self.accelerator_manager.get_capabilities(tpu_id)
        if capabilities:
            base_value += int(capabilities.peak_tflops * 10)

        # Adjust based on current load
        current_load = self._get_tpu_load(tpu_id)
        load_penalty = int(current_load * 100)

        # Adjust based on thermal constraints
        thermal_penalty = 0
        if request.thermal_budget < self.config.thermal_safety_margin:
            thermal_penalty = 200

        # Adjust based on bid credits
        bid_bonus = min(int(request.bid_credits), 500)

        return max(0, base_value + bid_bonus - load_penalty - thermal_penalty)

    def _get_tpu_load(self, tpu_id: AcceleratorType) -> float:
        """Get current load of a TPU."""
        # In this simple implementation, return a simulated load
        # In reality, you'd get actual load metrics from the TPU
        import random
        return random.uniform(0.1, 0.9)

    def process_telemetry(self, telemetry: TelemetrySnapshot) -> List[TPUAllocationRequest]:
        """Process telemetry to generate TPU resource requests."""
        requests = []

        # High CPU utilization might indicate need for TPU offloading
        if telemetry.cpu_util > 0.85:
            request = TPUAllocationRequest(
                request_id=f"TPU_OFFLOAD_{uuid.uuid4().hex[:8]}",
                agent_id="TELEMETRY_ANALYZER",
                resource_type=TPUResourceType.COMPUTE_UNITS,
                amount=2000,  # 2000 compute units
                priority=8,
                bid_credits=Decimal('75.0'),
                thermal_budget=telemetry.temp_cpu,  # Use CPU temp as proxy
                power_budget=self.config.power_limit_w,
                performance_goals={"cpu_offload": 0.4, "latency_reduction": 0.25}
            )
            requests.append(request)

        # High thermal pressure might indicate switching to power-efficient TPU
        if telemetry.temp_cpu > 75 or getattr(telemetry, 'temp_tpu', 0) > 70:
            request = TPUAllocationRequest(
                request_id=f"THERMAL_REDUCTION_{uuid.uuid4().hex[:8]}",
                agent_id="THERMAL_MANAGER",
                resource_type=TPUResourceType.COMPUTE_UNITS,
                amount=500,
                priority=9,
                thermal_budget=30,  # Thermal budget of 30C
                bid_credits=Decimal('50.0'),
                constraints={"compute_intensity": "medium", "power_efficiency": True},
                performance_goals={"thermal_reduction": 0.15}
            )
            requests.append(request)

        # Memory pressure might indicate need for TPU offloading
        if telemetry.memory_util > 0.85:
            request = TPUAllocationRequest(
                request_id=f"MEMORY_OFFLOAD_{uuid.uuid4().hex[:8]}",
                agent_id="MEMORY_MANAGER",
                resource_type=TPUResourceType.COMPUTE_UNITS,
                amount=1000,
                priority=7,
                bid_credits=Decimal('60.0'),
                constraints={"memory_region": "tpu", "offload_target": "tpu"},
                thermal_budget=25,
                performance_goals={"memory_utilization": 0.75}
            )
            requests.append(request)

        return requests

    def process_signal(self, signal: Signal) -> List[TPUAllocationRequest]:
        """Process GAMESA signals for TPU resource allocation."""
        requests = []

        # Process different signal types
        if signal.kind in [SignalKind.CPU_BOTTLENECK, SignalKind.GPU_BOTTLENECK]:
            # Offload to TPU
            request = TPUAllocationRequest(
                request_id=f"SIGNAL_OFFLOAD_{signal.id}",
                agent_id="SIGNAL_PROCESSOR",
                resource_type=TPUResourceType.COMPUTE_UNITS,
                amount=3000,  # High compute demand
                priority=9,
                bid_credits=Decimal('125.0'),
                constraints={"offload_target": "tpu"},
                thermal_budget=20,
                performance_goals={"bottleneck_relief": signal.strength}
            )
            requests.append(request)

        elif signal.kind == SignalKind.MEMORY_PRESSURE:
            # Request TPU offload to reduce memory pressure
            request = TPUAllocationRequest(
                request_id=f"MEMORY_PRESSURE_{signal.id}",
                agent_id="MEMORY_MANAGER",
                resource_type=TPUResourceType.COMPUTE_UNITS,
                amount=int(signal.strength * 2000),  # Scale by signal strength
                priority=8,
                bid_credits=Decimal(str(signal.strength * 80.0)),
                thermal_budget=25,
                performance_goals={"memory_pressure_reduction": signal.strength * 0.8}
            )
            requests.append(request)

        elif signal.kind == SignalKind.THERMAL_WARNING:
            # Switch to power-efficient TPU operations
            request = TPUAllocationRequest(
                request_id=f"THERMAL_WARNING_{signal.id}",
                agent_id="THERMAL_GUARDIAN",
                resource_type=TPUResourceType.POWER_BUDGET,
                amount=5.0,  # Low power budget
                priority=10,
                bid_credits=Decimal('100.0'),
                constraints={"power_efficiency": True, "compute_intensity": "low"},
                thermal_budget=15,  # Very conservative thermal budget
                performance_goals={"thermal_stability": 0.95}
            )
            requests.append(request)

        elif signal.kind == SignalKind.USER_BOOST_REQUEST:
            # High-priority performance boost via TPU
            request = TPUAllocationRequest(
                request_id=f"USER_BOOST_{signal.id}",
                agent_id="USER_AGENT",
                resource_type=TPUResourceType.COMPUTE_UNITS,
                amount=4000,  # Very high demand
                priority=10,
                bid_credits=Decimal('250.0'),
                thermal_budget=15,  # Be conservative with thermal limits
                performance_goals={"performance_boost": 0.9}
            )
            requests.append(request)

        return requests

    def optimize_for_workload(self, workload_type: WorkloadType,
                            precision_mode: PrecisionMode = PrecisionMode.INT8) -> bool:
        """Optimize TPU configuration for specific workload type."""
        with self.integration_lock:
            try:
                # Select appropriate preset based on workload
                if workload_type == WorkloadType.INFERENCE:
                    if precision_mode in [PrecisionMode.INT8, PrecisionMode.INT4]:
                        preset = PresetLibrary.get("HIGH_THROUGHPUT_INT8")
                    else:
                        preset = PresetLibrary.get("LOW_LATENCY_FP16")
                elif workload_type == WorkloadType.SPEECH:
                    preset = PresetLibrary.get("EFFICIENT_GNA")  # GNA is good for speech
                elif workload_type == WorkloadType.VISION:
                    preset = PresetLibrary.get("VPU_VISION")  # VPU/Hexagon for vision
                else:  # General inference
                    preset = PresetLibrary.get("BALANCED") or PresetLibrary.get("LOW_LATENCY_FP16")

                if preset:
                    # Apply preset to TPU bridge
                    self.tpu_bridge.active_preset = preset
                    print(f"Applied {preset.preset_id} preset for {workload_type.name} workload")
                    return True

                return False

            except Exception as e:
                print(f"TPU workload optimization failed: {e}")
                return False

    def get_optimization_status(self) -> Dict[str, Any]:
        """Get current TPU optimization status."""
        with self.integration_lock:
            return {
                'config': {
                    'optimization_strategy': self.config.optimization_strategy.value,
                    'allocation_strategy': self.config.allocation_strategy.value,
                    'enable_cross_forex': self.config.enable_cross_forex,
                    'enable_coherence': self.config.enable_coherence,
                    'thermal_safety_margin': self.config.thermal_safety_margin,
                },
                'metrics': {
                    'pipeline_utilization': self.metrics.pipeline_utilization,
                    'memory_efficiency': self.metrics.memory_efficiency,
                    'coherence_success_rate': self.metrics.coherence_success_rate,
                    'cross_forex_volume': float(self.metrics.cross_forex_volume),
                    'average_latency_us': self.metrics.average_latency_us,
                    'power_consumption_w': self.metrics.power_consumption_w,
                    'tpu_switching_events': self.metrics.tpu_switching_events,
                    'resource_trading_events': self.metrics.resource_trading_events,
                },
                'tpu_status': self._get_tpu_status(),
                'accelerator_status': self.accelerator_manager.get_status(),
                'active_preset': self.tpu_bridge.active_preset.preset_id if self.tpu_bridge.active_preset else None,
                'timestamp': time.time()
            }

    def _get_tpu_status(self) -> Dict[str, Any]:
        """Get detailed TPU status."""
        available_tpus = self._get_available_tpus()
        thermal_headroom = self.hal.get_thermal_headroom("tpu")

        return {
            'available_tpus': [t.name for t in available_tpus],
            'thermal_headroom': thermal_headroom,
            'active_allocations': len([a for a in self.allocation_history if a.status == "active"]),
            'total_allocations': len(self.allocation_history),
            'bridge_stats': self.tpu_bridge.get_stats()
        }

    def update_metrics(self, new_metrics: Optional[Dict] = None):
        """Update optimization metrics."""
        with self.integration_lock:
            if new_metrics:
                for key, value in new_metrics.items():
                    if hasattr(self.metrics, key):
                        setattr(self.metrics, key, value)

    def cleanup_expired_allocations(self) -> int:
        """Clean up expired TPU allocations."""
        cleaned = 0
        now = time.time()

        active_allocations = []
        for alloc in self.allocation_history:
            if alloc.expires_at >= now:
                active_allocations.append(alloc)
            else:
                cleaned += 1

        self.allocation_history = active_allocations
        return cleaned


# ============================================================
# TPU OPTIMIZATION CONTROLLER
# ============================================================

class TPUOptimizationController:
    """Controller for TPU optimization cycles."""

    def __init__(self, config: Optional[TPUConfig] = None):
        self.manager = TPUOptimizationManager(config)
        self.performance_monitor = TPUPerformanceMonitor(self.manager)
        self.policy_engine = TPUPolicyEngine(self.manager)

    def process_cycle(self, telemetry: TelemetrySnapshot, signals: List[Signal]) -> Dict[str, Any]:
        """Process one optimization cycle."""
        results = {
            'allocation_requests': [],
            'signals_processed': 0,
            'telemetry_processed': 0,
            'actions_taken': [],
            'workload_optimizations': []
        }

        # Process telemetry
        telemetry_requests = self.manager.process_telemetry(telemetry)
        results['allocation_requests'].extend(telemetry_requests)
        results['telemetry_processed'] = len(telemetry_requests)

        # Process signals
        for signal in signals:
            signal_requests = self.manager.process_signal(signal)
            results['allocation_requests'].extend(signal_requests)
            results['signals_processed'] += 1

        # Execute allocation requests
        allocations = []
        for request in results['allocation_requests']:
            allocation = self.manager.request_tpu_resources(request)
            if allocation:
                allocations.append(allocation)
                results['actions_taken'].append(
                    f"Allocated {request.amount} {request.resource_type.value} on {allocation.tpu_assigned.name if allocation.tpu_assigned else 'Unknown'}"
                )

        # Apply workload-specific optimizations
        if telemetry.active_process_category == "ai_inference":
            success = self.manager.optimize_for_workload(WorkloadType.INFERENCE)
            if success:
                results['workload_optimizations'].append("Applied AI inference optimization")
        elif telemetry.active_process_category == "speech_processing":
            success = self.manager.optimize_for_workload(WorkloadType.SPEECH)
            if success:
                results['workload_optimizations'].append("Applied speech processing optimization")

        # Update metrics
        self.manager.update_metrics()

        return results

    def get_status_report(self) -> Dict[str, Any]:
        """Get comprehensive status report."""
        return self.manager.get_optimization_status()


class TPUPerformanceMonitor:
    """Monitors TPU performance metrics."""

    def __init__(self, manager: TPUOptimizationManager):
        self.manager = manager
        self.perf_history = []

    def collect_metrics(self) -> Dict[str, float]:
        """Collect current TPU performance metrics."""
        status = self.manager.get_optimization_status()

        metrics = {
            'pipeline_utilization': status['metrics']['pipeline_utilization'],
            'memory_efficiency': status['metrics']['memory_efficiency'],
            'coherence_rate': status['metrics']['coherence_success_rate'],
            'tpu_throughput': status['bridge_stats'].get('avg_latency_ms', 0.0),
            'latency_avg': status['metrics']['average_latency_us'],
            'power_efficiency': self._calculate_power_efficiency()
        }

        self.perf_history.append(metrics)
        if len(self.perf_history) > 100:
            self.perf_history.pop(0)

        return metrics

    def _calculate_power_efficiency(self) -> float:
        """Calculate power efficiency."""
        # Placeholder implementation
        return 0.85  # 85% efficiency


class TPUPolicyEngine:
    """Manages TPU optimization policies."""

    def __init__(self, manager: TPUOptimizationManager):
        self.manager = manager
        self.policies = {
            'thermal_aware': True,
            'performance_aware': True,
            'economically_aware': True,
            'safety_aware': True
        }

    def load_policies(self, policy_config: Dict):
        """Load TPU allocation policies."""
        self.policies.update(policy_config)

    def apply_policy(self, request: TPUAllocationRequest) -> TPUAllocationRequest:
        """Apply policy constraints to a request."""
        # Apply thermal policy
        if self.policies.get('thermal_aware', False):
            if request.agent_id == "THERMAL_MANAGER":
                request.priority = min(request.priority + 2, 10)
                request.thermal_budget = min(request.thermal_budget, 15)  # Conservative thermal

        # Apply performance policy
        if self.policies.get('performance_aware', False):
            if request.performance_goals.get('latency_reduction', 0) > 0.5:
                request.priority = min(request.priority + 1, 10)

        # Apply economic policy
        if self.policies.get('economically_aware', False):
            max_bid = self.manager.config.max_trading_credits
            request.bid_credits = min(request.bid_credits, max_bid)

        # Apply safety policy
        if self.policies.get('safety_aware', False):
            # Ensure thermal safety margin
            request.thermal_budget = max(
                request.thermal_budget,
                self.manager.config.thermal_safety_margin
            )

        return request


# ============================================================
# DEMO
# ============================================================

def demo_tpu_optimization_framework():
    """Demonstrate TPU optimization framework."""
    print("=== GAMESA TPU Optimization Framework Demo ===\n")

    # Initialize controller
    controller = TPUOptimizationController()

    # Simulate telemetry data
    telemetry = TelemetrySnapshot(
        timestamp=datetime.now().isoformat(),
        cpu_util=0.90,  # High CPU utilization
        gpu_util=0.60,  # Moderate GPU utilization
        temp_cpu=78,    # 78°C CPU
        temp_gpu=65,    # 65°C GPU
        frametime_ms=22.0,  # 45 FPS
        memory_util=0.85,
        active_process_category="ai_inference"
    )

    print(f"Input Telemetry: CPU={telemetry.cpu_util*100:.1f}%, GPU={telemetry.gpu_util*100:.1f}%, "
          f"Temp CPU={telemetry.temp_cpu}°C, Temp GPU={telemetry.temp_gpu}°C\n")

    # Simulate various signals
    signals = [
        Signal(
            id="SIGNAL_001",
            source="TELEMETRY",
            kind=SignalKind.CPU_BOTTLENECK,
            strength=0.9,
            confidence=0.9,
            payload={"bottleneck_type": "compute", "recommended_action": "tpu_offload"}
        ),
        Signal(
            id="SIGNAL_002",
            source="TELEMETRY",
            kind=SignalKind.MEMORY_PRESSURE,
            strength=0.7,
            confidence=0.85,
            payload={"memory_util": 0.85, "recommended_action": "memory_offload"}
        ),
        Signal(
            id="SIGNAL_003",
            source="USER",
            kind=SignalKind.USER_BOOST_REQUEST,
            strength=0.8,
            confidence=0.95,
            payload={"request_type": "performance", "priority": "high"}
        )
    ]

    print("Processing signals:")
    for signal in signals:
        print(f"  - {signal.kind.value} (strength: {signal.strength})")
    print()

    # Execute one optimization cycle
    results = controller.process_cycle(telemetry, signals)

    print(f"Optimization Cycle Results:")
    print(f"  Allocation Requests Generated: {len(results['allocation_requests'])}")
    print(f"  Signals Processed: {results['signals_processed']}")
    print(f"  Telemetry Processed: {results['telemetry_processed']}")
    print(f"  Actions Taken: {len(results['actions_taken'])}")
    print(f"  Workload Optimizations: {len(results['workload_optimizations'])}")

    if results['actions_taken']:
        print("\nActions Taken:")
        for action in results['actions_taken']:
            print(f"  - {action}")

    if results['workload_optimizations']:
        print("\nWorkload Optimizations:")
        for opt in results['workload_optimizations']:
            print(f"  - {opt}")

    if results['allocation_requests']:
        print(f"\nGenerated Allocation Requests:")
        for req in results['allocation_requests']:
            print(f"  - {req.resource_type.value}: {req.amount:.0f} units, "
                  f"Priority: {req.priority}, Credits: {req.bid_credits}")

    print(f"\nTPU Integration Status:")
    status = controller.get_status_report()
    print(f"  Available TPUs: {status['tpu_status']['available_tpus']}")
    print(f"  Thermal Headroom: {status['tpu_status']['thermal_headroom']:.1f}°C")
    print(f"  Active Allocations: {status['tpu_status']['active_allocations']}")
    print(f"  TPU Throughput Events: {status['tpu_switching_events']}")

    print(f"\nGAMESA TPU Optimization Framework demo completed successfully!")


if __name__ == "__main__":
    demo_tpu_optimization_framework()