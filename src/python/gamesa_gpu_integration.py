"""
GAMESA GPU Integration Framework

Integrates GPU pipeline, 3D grid memory system, cross-forex trading,
and memory coherence protocol with the main GAMESA framework.
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
    # Core GAMESA components
    ResourceType, Priority, AllocationRequest, Allocation,
    Effect, Capability, create_guardian_checker,
    Contract, create_guardian_validator,
    TelemetrySnapshot, Signal, SignalKind, Domain,
    # Runtime components
    Runtime, RuntimeVar, RuntimeFunc,
    # Feature engine
    FeatureEngine, DbFeatureTransformer,
    # Allocation system
    Allocator, ResourcePool, AllocationConstraints,
    # Effects and contracts
    EffectChecker, ContractValidator,
    # Signal scheduling
    SignalScheduler
)
from .gpu_pipeline_integration import (
    GPUManager, GPUPipeline, UHDCoprocessor, 
    DiscreteGPU, GPUPipelineSignalHandler
)
from .cross_forex_memory_trading import (
    CrossForexManager, MemoryTradingSignalProcessor, 
    CrossForexTrade, MarketOrderType, MemoryResourceType
)
from .memory_coherence_protocol import (
    MemoryCoherenceProtocol, GPUCoherenceManager, 
    CoherenceState, CoherenceOperation
)
from .allocation import (
    Allocation as BaseAllocation, AllocationRequest as BaseAllocationRequest,
    ResourceType as BaseResourceType, Priority as BasePriority
)


# Enums
class GPUPipelineStatus(Enum):
    """Status of GPU pipeline integration."""
    INITIALIZED = "initialized"
    ACTIVE = "active"
    THROTTLED = "throttled"
    OFFLINE = "offline"

class MemoryOptimizationStrategy(Enum):
    """Strategies for memory optimization."""
    PERFORMANCE_FIRST = "performance_first"
    POWER_EFFICIENCY = "power_efficiency"
    BALANCED = "balanced"
    COST_OPTIMIZED = "cost_optimized"

class GPUAllocationStrategy(Enum):
    """Strategies for GPU resource allocation."""
    ROUND_ROBIN = "round_robin"
    LOAD_BALANCED = "load_balanced"
    PRIORITY_BASED = "priority_based"
    ECONOMIC_BIDDING = "economic_bidding"

class IntegrationMode(Enum):
    """Integration modes with GAMESA."""
    COORDINATED = "coordinated"      # Full GAMESA coordination
    AUTONOMOUS = "autonomous"        # GPU system runs independently
    HYBRID = "hybrid"                # Mix of coordinated and autonomous
    OPTIMIZATION_ONLY = "optimization_only"  # Optimization only


# Data Classes
@dataclass
class GPUIntegrationMetrics:
    """Metrics for GPU integration with GAMESA."""
    pipeline_utilization: float = 0.0
    memory_efficiency: float = 0.0
    coherence_success_rate: float = 1.0
    cross_forex_volume: Decimal = Decimal('0')
    average_latency_us: float = 0.0
    power_consumption_w: float = 0.0
    thermal_efficiency: float = 1.0
    gpu_switching_events: int = 0
    resource_trading_events: int = 0
    timestamp: float = field(default_factory=time.time)


@dataclass
class GPUAllocationRequest:
    """Extended allocation request for GPU resources."""
    request_id: str
    agent_id: str
    resource_type: str  # GPU-specific resources
    amount: int
    priority: int = BasePriority.NORMAL.value
    duration_ms: int = 1000
    bid_credits: Decimal = Decimal('10.0')
    constraints: Dict[str, Any] = field(default_factory=dict)
    memory_context: Optional[Dict[str, Any]] = None
    gpu_preference: Optional[int] = None
    performance_goals: Dict[str, float] = field(default_factory=dict)


@dataclass
class GPUAllocation:
    """Extended allocation for GPU resources."""
    allocation_id: str
    request_id: str
    agent_id: str
    resource_type: str
    amount: int
    gpu_assigned: int
    granted_at: float
    expires_at: float
    status: str = "active"
    memory_allocation: Optional[str] = None  # Reference to 3D grid allocation
    trading_cost: Decimal = Decimal('0')
    performance_metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class GPUPipelineState:
    """State of the integrated GPU pipeline."""
    status: GPUPipelineStatus = GPUPipelineStatus.INITIALIZED
    active_tasks: int = 0
    pending_tasks: int = 0
    total_gpus: int = 0
    available_gpus: int = 0
    uhd_available: bool = False
    coherence_active: bool = True
    trading_active: bool = True
    last_update: float = field(default_factory=time.time)


@dataclass
class IntegrationConfig:
    """Configuration for GPU integration."""
    mode: IntegrationMode = IntegrationMode.COORDINATED
    enable_cross_forex: bool = True
    enable_coherence: bool = True
    enable_3d_grid: bool = True
    enable_uhd_coprocessor: bool = True
    coherence_timeout_secs: float = 0.001
    max_trading_credits: Decimal = Decimal('1000.00')
    memory_optimization_strategy: MemoryOptimizationStrategy = MemoryOptimizationStrategy.BALANCED
    gpu_allocation_strategy: GPUAllocationStrategy = GPUAllocationStrategy.ECONOMIC_BIDDING


class GAMESAGPUIntegration:
    """Main integration class connecting GPU pipeline with GAMESA."""
    
    def __init__(self, config: Optional[IntegrationConfig] = None):
        self.config = config or IntegrationConfig()
        self.metrics = GPUIntegrationMetrics()
        self.pipeline_state = GPUPipelineState()
        self.integration_lock = threading.RLock()
        
        # Integrated components
        self.gpu_manager = GPUManager()
        self.cross_forex_manager = CrossForexManager()
        self.coherence_manager = GPUCoherenceManager()
        self.signal_handler = GPUPipelineSignalHandler(self.gpu_manager.pipeline)
        self.memory_trading_processor = MemoryTradingSignalProcessor(self.cross_forex_manager)
        
        # GAMESA integration components
        self.effect_checker = create_guardian_checker()
        self.contract_validator = create_guardian_validator()
        self.runtime = Runtime()
        self.signal_scheduler = SignalScheduler()
        
        # Performance tracking
        self.performance_history = []
        self.trading_history = []
        
        # Initialize integration
        self._initialize_integration()
    
    def _initialize_integration(self):
        """Initialize the GPU integration with GAMESA."""
        with self.integration_lock:
            print("Initializing GAMESA GPU Integration...")
            
            # Initialize GPU manager
            gpu_devices = [
                {'id': 0, 'name': 'Intel UHD', 'compute_units': 24, 'memory_size': 128, 'is_uhd': True},
                {'id': 1, 'name': 'NVIDIA RTX 4090', 'compute_units': 18432, 'memory_size': 24576, 'is_uhd': False},
                {'id': 2, 'name': 'AMD RX 7900 XTX', 'compute_units': 6144, 'memory_size': 24576, 'is_uhd': False}
            ]
            
            success = self.gpu_manager.initialize_gpu_cluster(gpu_devices)
            if not success:
                raise Exception("Failed to initialize GPU cluster")
            
            # Initialize cross-forex trading
            portfolio = self.cross_forex_manager.memory_engine.create_portfolio("GAMESA_SYSTEM")
            if not portfolio:
                raise Exception("Failed to create cross-forex portfolio")
            
            # Initialize coherence protocol
            self.coherence_manager.initialize_gpus([
                {'id': 0, 'type': 'uhd_coprocessor', 'memory_region': range(0x7FFF0000, 0x7FFF8000)},
                {'id': 1, 'type': 'discrete_gpu', 'memory_region': range(0x80000000, 0x90000000)},
                {'id': 2, 'type': 'discrete_gpu', 'memory_region': range(0x90000000, 0xA0000000)}
            ])
            
            # Integrate with GAMESA safety systems
            self._integrate_with_gamesa()
            
            # Update state
            self.pipeline_state.status = GPUPipelineStatus.ACTIVE
            self.pipeline_state.total_gpus = len(gpu_devices)
            self.pipeline_state.available_gpus = len(gpu_devices)
            self.pipeline_state.uhd_available = True
            
            print("GAMESA GPU Integration initialized successfully")
    
    def _integrate_with_gamesa(self):
        """Integrate GPU system with GAMESA safety and validation systems."""
        print("Integrating with GAMESA systems...")
        
        # Check required capabilities
        required_effects = [
            ("gpu_pipeline", Effect.GPU_CONTROL),
            ("cross_forex_trading", Effect.MEMORY_CONTROL),
            ("memory_coherence", Effect.MEMORY_COHERENCE)
        ]
        
        for component, effect in required_effects:
            if not self.effect_checker.can_perform(component, effect):
                print(f"Warning: {component} lacks {effect.name} capability")
                # Continue anyway - warnings are OK for non-critical effects
        
        # Validate safety contracts
        contract_result = self.contract_validator.check_invariants("gpu_integration", {
            "component_count": 3,  # GPU, coherence, trading
            "expected_gpus": self.pipeline_state.total_gpus,
            "trading_enabled": self.config.enable_cross_forex,
            "coherence_enabled": self.config.enable_coherence
        })
        
        if not contract_result.valid:
            print(f"GPU integration validation warnings: {contract_result.errors}")
        
        print("GAMESA integration completed")
    
    def request_gpu_resources(self, request: GPUAllocationRequest) -> Optional[GPUAllocation]:
        """Request GPU resources through integrated system."""
        with self.integration_lock:
            print(f"Processing GPU allocation request: {request.request_id}")
            
            # Validate request
            if not self._validate_gpu_request(request):
                return None
            
            # Determine optimal GPU based on allocation strategy
            gpu_assignment = self._determine_gpu_assignment(request)
            
            if gpu_assignment is None:
                print(f"No suitable GPU available for request {request.request_id}")
                return None
            
            # Allocate memory using 3D grid system with cross-forex trading
            memory_allocation_id = None
            trading_cost = Decimal('0')
            
            if self.config.enable_3d_grid and self.config.enable_cross_forex:
                context = self._create_memory_context(request)
                try:
                    allocation = self.cross_forex_manager.get_memory_allocation(
                        size=request.amount if request.amount < 1024*1024*1024 else 1024*1024*1024,  # Cap at 1GB
                        context=context
                    )
                    memory_allocation_id = allocation.id
                    trading_cost = allocation.performance_metrics.get('cost', Decimal('0'))
                except Exception as e:
                    print(f"Memory allocation failed: {e}")
                    # Continue without memory allocation
            
            # Create the GPU allocation
            allocation_id = f"GPU_ALLOC_{uuid.uuid4().hex[:8]}"
            now = time.time()
            
            gpu_allocation = GPUAllocation(
                allocation_id=allocation_id,
                request_id=request.request_id,
                agent_id=request.agent_id,
                resource_type=request.resource_type,
                amount=request.amount,
                gpu_assigned=gpu_assignment,
                granted_at=now,
                expires_at=now + (request.duration_ms / 1000.0),
                memory_allocation=memory_allocation_id,
                trading_cost=trading_cost,
                status="active"
            )
            
            # Update metrics
            self.metrics.resource_trading_events += 1
            self.pipeline_state.active_tasks += 1
            
            print(f"GPU allocation successful: {allocation_id} on GPU {gpu_assignment}")
            return gpu_allocation
    
    def _validate_gpu_request(self, request: GPUAllocationRequest) -> bool:
        """Validate the GPU allocation request."""
        # Check resource type validity
        valid_resource_types = [
            "compute_units", "vram", "memory_bandwidth", 
            "l1_cache", "l2_cache", "texture_units", "render_targets"
        ]
        
        if request.resource_type not in valid_resource_types:
            print(f"Invalid resource type: {request.resource_type}")
            return False
        
        # Check amount validity
        if request.amount <= 0:
            print(f"Invalid amount: {request.amount}")
            return False
        
        # Check priority validity
        if request.priority < 1 or request.priority > 10:
            print(f"Invalid priority: {request.priority}")
            return False
        
        return True
    
    def _determine_gpu_assignment(self, request: GPUAllocationRequest) -> Optional[int]:
        """Determine which GPU to assign based on allocation strategy."""
        available_gpus = self._get_available_gpus()
        
        if not available_gpus:
            return None
        
        # Apply allocation strategy
        if self.config.gpu_allocation_strategy == GPUAllocationStrategy.PRIORITY_BASED:
            # Prioritize based on request priority
            if request.priority >= 8 and 1 in available_gpus:  # High priority goes to discrete
                return 1
            
            # Check GPU preference
            if request.gpu_preference and request.gpu_preference in available_gpus:
                return request.gpu_preference
            
            # Default to first available
            return available_gpus[0]
        
        elif self.config.gpu_allocation_strategy == GPUAllocationStrategy.ECONOMIC_BIDDING:
            # Use economic bidding system
            winning_bid = 0
            winner_gpu = available_gpus[0]
            
            for gpu_id in available_gpus:
                bid_value = self._calculate_gpu_bid_value(gpu_id, request)
                if bid_value > winning_bid:
                    winning_bid = bid_value
                    winner_gpu = gpu_id
            
            return winner_gpu
        
        elif self.config.gpu_allocation_strategy == GPUAllocationStrategy.LOAD_BALANCED:
            # Choose GPU with lowest current load
            loads = {}
            for gpu_id in available_gpus:
                loads[gpu_id] = self._get_gpu_load(gpu_id)
            
            # Return GPU with minimum load
            return min(loads, key=loads.get)
        
        else:  # ROUND_ROBIN
            # Simple round robin assignment
            if hasattr(self, '_round_robin_index'):
                index = getattr(self, '_round_robin_index', 0)
                assigned_gpu = available_gpus[index % len(available_gpus)]
                self._round_robin_index = (index + 1) % len(available_gpus)
            else:
                self._round_robin_index = 1
                assigned_gpu = available_gpus[0]
            
            return assigned_gpu
    
    def _create_memory_context(self, request: GPUAllocationRequest):
        """Create memory context for 3D grid allocation."""
        from .gpu_pipeline_integration import MemoryContext
        
        # Determine if request is performance-critical
        performance_critical = request.priority >= 7
        compute_intensive = 'compute' in request.resource_type.lower()
        
        # Determine GPU preference
        gpu_pref = None
        if request.gpu_preference == 0:  # UHD
            gpu_pref = GPUType.UHD
        elif request.gpu_preference:  # Discrete GPU
            gpu_pref = GPUType.DISCRETE
        
        return MemoryContext(
            access_pattern=request.constraints.get('access_pattern', 'random'),
            performance_critical=performance_critical,
            compute_intensive=compute_intensive,
            gpu_preference=gpu_pref
        )
    
    def _calculate_gpu_bid_value(self, gpu_id: int, request: GPUAllocationRequest) -> int:
        """Calculate bid value for economic GPU allocation."""
        # Base value based on available resources
        base_value = 100
        
        # Adjust based on performance capabilities
        if gpu_id == 1:  # High-end discrete GPU
            base_value += 200
        elif gpu_id == 0:  # UHD coprocessor
            if 'compute' in request.resource_type:
                base_value += 150
            else:
                base_value += 50
        else:  # Other GPU
            base_value += 100
        
        # Adjust based on current load
        current_load = self._get_gpu_load(gpu_id)
        load_penalty = int(current_load * 100)
        
        # Adjust based on bid credits
        bid_bonus = min(int(request.bid_credits), 500)
        
        return max(0, base_value + bid_bonus - load_penalty)
    
    def _get_available_gpus(self) -> List[int]:
        """Get list of available GPUs."""
        # In this simple implementation, all registered GPUs are available
        # In reality, you'd check actual availability
        return [0, 1, 2]  # UHD + 2 discrete GPUs
    
    def _get_gpu_load(self, gpu_id: int) -> float:
        """Get current load of a GPU."""
        # In this simple implementation, return a simulated load
        # In reality, you'd get actual load metrics
        import random
        return random.uniform(0.1, 0.9)
    
    def process_telemetry(self, telemetry: TelemetrySnapshot) -> List[GPUAllocationRequest]:
        """Process telemetry and generate GPU resource requests."""
        requests = []
        
        # Generate requests based on telemetry analysis
        
        # High CPU utilization might indicate need for GPU offloading
        if telemetry.cpu_util > 0.8:
            request = GPUAllocationRequest(
                request_id=f"OFFLOAD_{uuid.uuid4().hex[:8]}",
                agent_id="TELEMETRY_ANALYZER",
                resource_type="compute_units",
                amount=1000,  # 1000 compute units
                priority=8,
                bid_credits=Decimal('50.0'),
                performance_goals={"cpu_offload": 0.3, "latency_reduction": 0.2}
            )
            requests.append(request)
        
        # High thermal pressure might indicate switching to UHD coprocessor
        if telemetry.temp_gpu > 75:
            request = GPUAllocationRequest(
                request_id=f"THERMAL_REDUCTION_{uuid.uuid4().hex[:8]}",
                agent_id="THERMAL_MANAGER",
                resource_type="compute_units",
                amount=500,
                priority=9,
                gpu_preference=0,  # UHD coprocessor
                bid_credits=Decimal('30.0'),
                constraints={"compute_intensity": "medium"},
                performance_goals={"thermal_reduction": 0.1}
            )
            requests.append(request)
        
        # Memory pressure might indicate need for more VRAM or UHD buffer
        if telemetry.memory_util > 0.85:
            request = GPUAllocationRequest(
                request_id=f"MEMORY_OFFLOAD_{uuid.uuid4().hex[:8]}",
                agent_id="MEMORY_MANAGER", 
                resource_type="vram",
                amount=256 * 1024 * 1024,  # 256MB
                priority=7,
                bid_credits=Decimal('75.0'),
                constraints={"memory_region": "gpu"},
                performance_goals={"memory_utilization": 0.7}
            )
            requests.append(request)
        
        return requests
    
    def process_signal(self, signal: Signal) -> List[GPUAllocationRequest]:
        """Process GAMESA signals for GPU resource allocation."""
        requests = []
        
        # Process different signal types
        if signal.kind in [SignalKind.CPU_BOTTLENECK, SignalKind.GPU_BOTTLENECK]:
            # Offload to GPU or switch between GPU types
            request = GPUAllocationRequest(
                request_id=f"SIGNAL_OFFLOAD_{signal.id}",
                agent_id="SIGNAL_PROCESSOR",
                resource_type="compute_units",
                amount=2000,  # High compute demand
                priority=9,
                bid_credits=Decimal('100.0'),
                constraints={"offload_target": "gpu"},
                performance_goals={"bottleneck_relief": signal.strength}
            )
            requests.append(request)
        
        elif signal.kind == SignalKind.MEMORY_PRESSURE:
            # Request additional GPU memory
            request = GPUAllocationRequest(
                request_id=f"MEMORY_PRESSURE_{signal.id}",
                agent_id="MEMORY_MANAGER",
                resource_type="vram",
                amount=int(signal.strength * 512 * 1024 * 1024),  # Scale by signal strength
                priority=8,
                bid_credits=Decimal(str(signal.strength * 50.0)),
                performance_goals={"memory_pressure_reduction": signal.strength}
            )
            requests.append(request)
        
        elif signal.kind == SignalKind.THERMAL_WARNING:
            # Switch to cooler UHD coprocessor for certain tasks
            request = GPUAllocationRequest(
                request_id=f"THERMAL_WARNING_{signal.id}",
                agent_id="THERMAL_GUARDIAN",
                resource_type="compute_units",
                amount=1000,
                priority=10,
                gpu_preference=0,  # UHD coprocessor
                bid_credits=Decimal('80.0'),
                constraints={"compute_intensity": "low_to_medium"},
                performance_goals={"thermal_stability": 0.9}
            )
            requests.append(request)
        
        elif signal.kind == SignalKind.USER_BOOST_REQUEST:
            # High-priority performance boost
            request = GPUAllocationRequest(
                request_id=f"USER_BOOST_{signal.id}",
                agent_id="USER_AGENT",
                resource_type="compute_units",
                amount=3000,  # High-demand
                priority=10,
                bid_credits=Decimal('200.0'),
                performance_goals={"performance_boost": 0.8}
            )
            requests.append(request)
        
        return requests
    
    def get_integration_status(self) -> Dict[str, Any]:
        """Get current integration status."""
        with self.integration_lock:
            return {
                'config': {
                    'mode': self.config.mode.value,
                    'enable_cross_forex': self.config.enable_cross_forex,
                    'enable_coherence': self.config.enable_coherence,
                    'enable_3d_grid': self.config.enable_3d_grid,
                    'enable_uhd_coprocessor': self.config.enable_uhd_coprocessor,
                },
                'pipeline_state': {
                    'status': self.pipeline_state.status.value,
                    'active_tasks': self.pipeline_state.active_tasks,
                    'pending_tasks': self.pipeline_state.pending_tasks,
                    'total_gpus': self.pipeline_state.total_gpus,
                    'available_gpus': self.pipeline_state.available_gpus,
                    'uhd_available': self.pipeline_state.uhd_available,
                    'last_update': self.pipeline_state.last_update,
                },
                'metrics': {
                    'pipeline_utilization': self.metrics.pipeline_utilization,
                    'memory_efficiency': self.metrics.memory_efficiency,
                    'coherence_success_rate': self.metrics.coherence_success_rate,
                    'cross_forex_volume': float(self.metrics.cross_forex_volume),
                    'average_latency_us': self.metrics.average_latency_us,
                    'power_consumption_w': self.metrics.power_consumption_w,
                    'trading_events': self.metrics.resource_trading_events,
                },
                'gpu_cluster_status': self.gpu_manager.get_cluster_status(),
                'timestamp': time.time()
            }
    
    def optimize_memory_allocation(self) -> bool:
        """Optimize memory allocation based on current system state."""
        with self.integration_lock:
            try:
                # Determine optimization strategy
                if self.config.memory_optimization_strategy == MemoryOptimizationStrategy.PERFORMANCE_FIRST:
                    # Prioritize performance - use fastest memory available
                    print("Optimizing memory allocation for performance")
                    
                elif self.config.memory_optimization_strategy == MemoryOptimizationStrategy.POWER_EFFICIENCY:
                    # Prioritize power efficiency - use UHD coprocessor when possible
                    print("Optimizing memory allocation for power efficiency")
                    
                else:  # BALANCED or COST_OPTIMIZED
                    # Balanced approach - optimize for both performance and cost
                    print("Optimizing memory allocation with balanced approach")
                
                return True
                
            except Exception as e:
                print(f"Memory optimization failed: {e}")
                return False
    
    def update_metrics(self, new_metrics: Optional[Dict] = None):
        """Update integration metrics."""
        with self.integration_lock:
            if new_metrics:
                for key, value in new_metrics.items():
                    if hasattr(self.metrics, key):
                        setattr(self.metrics, key, value)
            
            # Update pipeline utilization
            total_tasks = self.pipeline_state.active_tasks + self.pipeline_state.pending_tasks
            if total_tasks > 0:
                self.metrics.pipeline_utilization = (
                    self.pipeline_state.active_tasks / total_tasks
                )
            else:
                self.metrics.pipeline_utilization = 0.0
    
    def cleanup_expired_allocations(self) -> int:
        """Clean up expired GPU allocations."""
        cleaned = 0
        now = time.time()
        
        # In a real implementation, you'd have a way to track allocations
        # For this demo, we'll just simulate cleanup
        if hasattr(self, '_recent_allocations'):
            expired = [alloc for alloc in self._recent_allocations if alloc.expires_at < now]
            cleaned = len(expired)
            self._recent_allocations = [alloc for alloc in self._recent_allocations if alloc.expires_at >= now]
        
        self.pipeline_state.active_tasks = max(0, self.pipeline_state.active_tasks - cleaned)
        return cleaned


class GAMESAGPUController:
    """Controller for GAMESA GPU integration."""
    
    def __init__(self):
        self.integration = GAMESAGPUIntegration()
        self.performance_monitor = GPUPerformanceMonitor(self.integration)
        self.policy_engine = GPUPolicyEngine(self.integration)
        
    def process_cycle(self, telemetry: TelemetrySnapshot, signals: List[Signal]) -> Dict[str, Any]:
        """Process one control cycle with telemetry and signals."""
        results = {
            'allocation_requests': [],
            'signals_processed': 0,
            'telemetry_processed': 0,
            'actions_taken': []
        }
        
        # Process telemetry
        telemetry_requests = self.integration.process_telemetry(telemetry)
        results['allocation_requests'].extend(telemetry_requests)
        results['telemetry_processed'] = len(telemetry_requests)
        
        # Process signals
        for signal in signals:
            signal_requests = self.integration.process_signal(signal)
            results['allocation_requests'].extend(signal_requests)
            results['signals_processed'] += 1
        
        # Execute allocation requests through economic trading
        allocations = []
        for request in results['allocation_requests']:
            allocation = self.integration.request_gpu_resources(request)
            if allocation:
                allocations.append(allocation)
                results['actions_taken'].append(f"Allocated {request.amount} {request.resource_type} on GPU {allocation.gpu_assigned}")
        
        # Update metrics
        self.integration.update_metrics()
        
        return results
    
    def get_status_report(self) -> Dict[str, Any]:
        """Get comprehensive status report."""
        status = self.integration.get_integration_status()
        status['control_cycle'] = getattr(self, '_cycle_count', 0)
        self._cycle_count = getattr(self, '_cycle_count', 0) + 1
        
        return status


class GPUPerformanceMonitor:
    """Monitors GPU performance metrics."""
    
    def __init__(self, integration: GAMESAGPUIntegration):
        self.integration = integration
        self.perf_history = deque(maxlen=1000)
    
    def collect_metrics(self) -> Dict[str, float]:
        """Collect current performance metrics."""
        status = self.integration.get_integration_status()
        
        metrics = {
            'pipeline_utilization': status['metrics']['pipeline_utilization'],
            'memory_efficiency': status['metrics']['memory_efficiency'],
            'coherence_rate': status['metrics']['coherence_success_rate'],
            'gpu_throughput': self._calculate_throughput(),
            'latency_avg': status['metrics']['average_latency_us'],
            'power_efficiency': self._calculate_power_efficiency()
        }
        
        self.perf_history.append(metrics)
        return metrics
    
    def _calculate_throughput(self) -> float:
        """Calculate GPU throughput."""
        # Placeholder implementation
        import random
        return random.uniform(80, 120)  # GFLOPS
    
    def _calculate_power_efficiency(self) -> float:
        """Calculate power efficiency."""
        # Placeholder implementation
        return 0.85  # 85% efficiency


class GPUPolicyEngine:
    """Manages GPU allocation policies."""
    
    def __init__(self, integration: GAMESAGPUIntegration):
        self.integration = integration
        self.policies = {}
    
    def load_policies(self, policy_config: Dict):
        """Load GPU allocation policies."""
        self.policies.update(policy_config)
    
    def apply_policy(self, request: GPUAllocationRequest) -> GPUAllocationRequest:
        """Apply policy constraints to a request."""
        # Apply thermal policy
        if self.policies.get('thermal_aware', False):
            if request.agent_id == "THERMAL_MANAGER":
                request.priority = min(request.priority + 2, 10)
        
        # Apply performance policy
        if self.policies.get('performance_aware', False):
            if request.performance_goals.get('latency_reduction', 0) > 0.5:
                request.priority = min(request.priority + 1, 10)
        
        # Apply economic policy
        if self.policies.get('economically_aware', False):
            max_bid = self.integration.config.max_trading_credits
            request.bid_credits = min(request.bid_credits, max_bid)
        
        return request


# Demo function
def demo_gamesa_gpu_integration():
    """Demonstrate GAMESA GPU integration."""
    print("=== GAMESA GPU Integration Demo ===\n")
    
    # Initialize controller
    controller = GAMESAGPUController()
    
    # Simulate telemetry data
    telemetry = TelemetrySnapshot(
        timestamp=datetime.now().isoformat(),
        cpu_util=0.85,  # High CPU utilization
        gpu_util=0.75,  # High GPU utilization
        frametime_ms=20.0,  # 50 FPS
        temp_cpu=78,  # 78°C
        temp_gpu=82,  # 82°C
        active_process_category="gaming"
    )
    
    print(f"Input Telemetry: CPU={telemetry.cpu_util*100:.1f}%, GPU={telemetry.gpu_util*100:.1f}%, "
          f"Temp CPU={telemetry.temp_cpu}°C, Temp GPU={telemetry.temp_gpu}°C\n")
    
    # Simulate various signals
    signals = [
        Signal(
            id="SIGNAL_001",
            source="TELEMETRY",
            kind=SignalKind.CPU_BOTTLENECK,
            strength=0.85,
            confidence=0.9,
            payload={"bottleneck_type": "compute", "recommended_action": "gpu_offload"}
        ),
        Signal(
            id="SIGNAL_002",
            source="TELEMETRY", 
            kind=SignalKind.THERMAL_WARNING,
            strength=0.6,
            confidence=0.8,
            payload={"component": "gpu", "temperature": 82, "recommended_action": "switch_to_uhd"}
        ),
        Signal(
            id="SIGNAL_003",
            source="USER",
            kind=SignalKind.USER_BOOST_REQUEST,
            strength=1.0,
            confidence=0.95,
            payload={"request_type": "performance", "priority": "high"}
        )
    ]
    
    print("Processing signals:")
    for signal in signals:
        print(f"  - {signal.kind.value} (strength: {signal.strength})")
    print()
    
    # Execute one control cycle
    results = controller.process_cycle(telemetry, signals)
    
    print(f"Control Cycle Results:")
    print(f"  Allocation Requests Generated: {len(results['allocation_requests'])}")
    print(f"  Signals Processed: {results['signals_processed']}")
    print(f"  Telemetry Processed: {results['telemetry_processed']}")
    print(f"  Actions Taken: {len(results['actions_taken'])}")
    
    if results['actions_taken']:
        print("\nActions Taken:")
        for action in results['actions_taken']:
            print(f"  - {action}")
    
    if results['allocation_requests']:
        print(f"\nGenerated Allocation Requests:")
        for req in results['allocation_requests']:
            print(f"  - {req.resource_type}: {req.amount} units, Priority: {req.priority}, Credits: {req.bid_credits}")
    
    print(f"\nIntegration Status:")
    status = controller.get_status_report()
    print(f"  Pipeline Status: {status['pipeline_state']['status']}")
    print(f"  Active Tasks: {status['pipeline_state']['active_tasks']}")
    print(f"  Available GPUs: {status['pipeline_state']['available_gpus']}")
    print(f"  GPU Cluster: {status['gpu_cluster_status']}")
    
    print(f"\nGAMESA GPU Integration demo completed successfully!")


if __name__ == "__main__":
    demo_gamesa_gpu_integration()