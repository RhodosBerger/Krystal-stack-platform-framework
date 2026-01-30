"""
Functional Layer for GAMESA GPU Integration

Comprehensive functional layer that integrates all previously implemented
tools and systems into a cohesive, working framework.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum, auto
import time
import threading
import asyncio
from concurrent.futures import ThreadPoolExecutor
import logging
from decimal import Decimal
from datetime import datetime
import uuid
import platform
import psutil
from functools import wraps


# Import all previously implemented modules
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
    DiscreteGPU, GPUPipelineSignalHandler,
    MemoryGridCoordinate, MemoryContext,
    GPUGridMemoryManager, GPUCacheCoherenceManager,
    TaskType, GPUPipelineStage, GPUType
)

from .cross_forex_memory_trading import (
    CrossForexManager, MemoryTradingSignalProcessor, 
    CrossForexTrade, MarketOrderType, MemoryResourceType,
    MemoryMarketEngine
)

from .memory_coherence_protocol import (
    MemoryCoherenceProtocol, GPUCoherenceManager, 
    CoherenceState, CoherenceOperation, CoherenceEntry
)

from .gamesa_gpu_integration import (
    GAMESAGPUIntegration, GPUAllocationRequest, IntegrationConfig,
    MemoryOptimizationStrategy, GPUAllocationStrategy, IntegrationMode,
    GAMESAGPUController, GPUPerformanceMonitor, GPUPolicyEngine
)


# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Enums
class LayerStatus(Enum):
    """Status of functional layers."""
    STARTING = "starting"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"

class TaskPriority(Enum):
    """Task priority levels."""
    CRITICAL = 10
    HIGH = 8
    NORMAL = 5
    LOW = 3
    BACKGROUND = 1

class ExecutionMode(Enum):
    """Execution mode for tasks."""
    SYNCHRONOUS = "synchronous"
    ASYNCHRONOUS = "asynchronous"
    BATCH = "batch"
    STREAMING = "streaming"


# Data Classes for Functional Layer
@dataclass
class FunctionalLayerMetrics:
    """Metrics for functional layer performance."""
    tasks_completed: int = 0
    tasks_failed: int = 0
    average_execution_time: float = 0.0  # in seconds
    memory_usage_mb: float = 0.0
    gpu_utilization: Dict[int, float] = field(default_factory=dict)
    coherence_success_rate: float = 1.0
    cross_forex_volume: Decimal = Decimal('0')
    latency_microseconds: float = 0.0
    timestamp: float = field(default_factory=time.time)

@dataclass
class LayerTask:
    """Task to be executed by functional layer."""
    task_id: str
    task_type: str
    data: Dict[str, Any]
    priority: TaskPriority = TaskPriority.NORMAL
    execution_mode: ExecutionMode = ExecutionMode.ASYNCHRONOUS
    dependencies: List[str] = field(default_factory=list)
    timeout_seconds: float = 30.0
    callback: Optional[callable] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class LayerResult:
    """Result from functional layer execution."""
    task_id: str
    success: bool
    result: Optional[Any] = None
    error: Optional[str] = None
    execution_time: float = 0.0
    metrics: FunctionalLayerMetrics = field(default_factory=FunctionalLayerMetrics)
    timestamp: float = field(default_factory=time.time)

@dataclass
class LayerConfiguration:
    """Configuration for functional layers."""
    enable_gpu_integration: bool = True
    enable_memory_coherence: bool = True
    enable_cross_forex_trading: bool = True
    enable_3d_grid_memory: bool = True
    enable_uhd_coprocessor: bool = True
    max_parallel_tasks: int = 16
    execution_mode: ExecutionMode = ExecutionMode.ASYNCHRONOUS
    priority_scheduling: bool = True
    enable_telemetry: bool = True
    enable_safety_checks: bool = True


class FunctionalLayer:
    """Base class for functional layers."""
    
    def __init__(self, name: str, config: LayerConfiguration):
        self.name = name
        self.config = config
        self.status = LayerStatus.STARTING
        self.lock = threading.RLock()
        self.tasks: Dict[str, LayerTask] = {}
        self.results: Dict[str, LayerResult] = {}
        self.metrics = FunctionalLayerMetrics()
        self.executor = ThreadPoolExecutor(max_workers=config.max_parallel_tasks)
        
        # Initialize with default configuration
        self._initialize_components()
        
    def _initialize_components(self):
        """Initialize layer components."""
        pass
    
    def execute_task(self, task: LayerTask) -> LayerResult:
        """Execute a single task."""
        start_time = time.time()
        
        try:
            result = self._execute(task)
            success = True
            error = None
        except Exception as e:
            result = None
            success = False
            error = str(e)
            logger.error(f"Task {task.task_id} failed: {error}")
        
        execution_time = time.time() - start_time
        
        # Update metrics
        with self.lock:
            if success:
                self.metrics.tasks_completed += 1
            else:
                self.metrics.tasks_failed += 1
            
            # Update average execution time
            total_tasks = self.metrics.tasks_completed + self.metrics.tasks_failed
            if total_tasks > 1:
                self.metrics.average_execution_time = (
                    (self.metrics.average_execution_time * (total_tasks - 1) + execution_time) / total_tasks
                )
            else:
                self.metrics.average_execution_time = execution_time
        
        layer_result = LayerResult(
            task_id=task.task_id,
            success=success,
            result=result,
            error=error,
            execution_time=execution_time,
            metrics=self.metrics,
            timestamp=time.time()
        )
        
        # Execute callback if provided
        if task.callback:
            try:
                task.callback(layer_result)
            except Exception as e:
                logger.error(f"Callback for task {task.task_id} failed: {e}")
        
        # Store result
        self.results[task.task_id] = layer_result
        
        return layer_result
    
    def _execute(self, task: LayerTask) -> Any:
        """Execute the actual task (to be overridden by subclasses)."""
        raise NotImplementedError("_execute method must be implemented by subclass")
    
    def submit_task(self, task: LayerTask) -> str:
        """Submit a task to the layer."""
        task_id = task.task_id or f"{self.name}_TASK_{uuid.uuid4().hex[:8]}"
        self.tasks[task_id] = task
        
        if self.config.execution_mode == ExecutionMode.ASYNCHRONOUS:
            # Execute asynchronously
            future = self.executor.submit(self.execute_task, task)
            return task_id
        else:
            # Execute synchronously
            result = self.execute_task(task)
            return task_id
    
    def get_result(self, task_id: str) -> Optional[LayerResult]:
        """Get result of a completed task."""
        return self.results.get(task_id)
    
    def wait_for_completion(self, task_id: str, timeout: Optional[float] = None) -> Optional[LayerResult]:
        """Wait for a task to complete."""
        start_time = time.time()
        while task_id not in self.results:
            if timeout and (time.time() - start_time) > timeout:
                return None
            time.sleep(0.001)
        return self.results[task_id]
    
    def get_metrics(self) -> FunctionalLayerMetrics:
        """Get current layer metrics."""
        return self.metrics
    
    def start(self):
        """Start the functional layer."""
        with self.lock:
            if self.status in [LayerStatus.STOPPED, LayerStatus.ERROR]:
                self.status = LayerStatus.RUNNING
                logger.info(f"Started functional layer: {self.name}")
    
    def stop(self):
        """Stop the functional layer."""
        with self.lock:
            if self.status in [LayerStatus.RUNNING, LayerStatus.PAUSED]:
                self.status = LayerStatus.STOPPING
                # Wait for tasks to complete
                time.sleep(0.1)
                self.executor.shutdown(wait=True)
                self.status = LayerStatus.STOPPED
                logger.info(f"Stopped functional layer: {self.name}")
    
    def pause(self):
        """Pause the functional layer."""
        with self.lock:
            if self.status == LayerStatus.RUNNING:
                self.status = LayerStatus.PAUSED
                logger.info(f"Paused functional layer: {self.name}")
    
    def resume(self):
        """Resume the functional layer."""
        with self.lock:
            if self.status == LayerStatus.PAUSED:
                self.status = LayerStatus.RUNNING
                logger.info(f"Resumed functional layer: {self.name}")


class GPULayer(FunctionalLayer):
    """GPU integration layer."""
    
    def __init__(self, config: LayerConfiguration):
        self.gpu_integration = None
        super().__init__("GPU_INTEGRATION", config)
    
    def _initialize_components(self):
        """Initialize GPU layer components."""
        if self.config.enable_gpu_integration:
            # Initialize GAMESA GPU Integration
            integration_config = IntegrationConfig()
            integration_config.mode = IntegrationMode.HYBRID
            integration_config.enable_cross_forex = self.config.enable_cross_forex_trading
            integration_config.enable_coherence = self.config.enable_memory_coherence
            integration_config.enable_3d_grid = self.config.enable_3d_grid_memory
            integration_config.enable_uhd_coprocessor = self.config.enable_uhd_coprocessor
            
            self.gpu_integration = GAMESAGPUIntegration(integration_config)
            self.status = LayerStatus.RUNNING
            logger.info("GPU Integration layer initialized")
        
        # Update metrics
        self._update_metrics()
    
    def _execute(self, task: LayerTask) -> Any:
        """Execute GPU-related tasks."""
        if task.task_type == "allocate_gpu_resources":
            return self._allocate_gpu_resources(task)
        elif task.task_type == "process_telemetry":
            return self._process_telemetry(task)
        elif task.task_type == "process_signal":
            return self._process_signal(task)
        elif task.task_type == "gpu_memory_allocation":
            return self._allocate_gpu_memory(task)
        else:
            raise ValueError(f"Unknown GPU task type: {task.task_type}")
    
    def _allocate_gpu_resources(self, task: LayerTask) -> Any:
        """Allocate GPU resources."""
        request_data = task.data
        request = GPUAllocationRequest(
            request_id=request_data.get('request_id', f"REQ_{uuid.uuid4().hex[:8]}"),
            agent_id=request_data.get('agent_id', 'GPU_LAYER'),
            resource_type=request_data.get('resource_type', 'compute_units'),
            amount=request_data.get('amount', 1000),
            priority=request_data.get('priority', 5),
            bid_credits=Decimal(str(request_data.get('bid_credits', 10.0))),
            constraints=request_data.get('constraints', {}),
            memory_context=request_data.get('memory_context'),
            gpu_preference=request_data.get('gpu_preference'),
            performance_goals=request_data.get('performance_goals', {})
        )
        
        if self.gpu_integration:
            return self.gpu_integration.request_gpu_resources(request)
        else:
            raise RuntimeError("GPU Integration not initialized")
    
    def _process_telemetry(self, task: LayerTask) -> List[GPUAllocationRequest]:
        """Process telemetry for GPU allocation requests."""
        telemetry = task.data.get('telemetry')
        if self.gpu_integration and telemetry:
            return self.gpu_integration.process_telemetry(telemetry)
        return []
    
    def _process_signal(self, task: LayerTask) -> List[GPUAllocationRequest]:
        """Process signal for GPU allocation requests."""
        signal = task.data.get('signal')
        if self.gpu_integration and signal:
            return self.gpu_integration.process_signal(signal)
        return []
    
    def _allocate_gpu_memory(self, task: LayerTask) -> Any:
        """Allocate GPU memory using 3D grid system."""
        if not self.gpu_integration:
            raise RuntimeError("GPU Integration not initialized")
        
        # Get memory manager from integration
        manager = self.gpu_integration.cross_forex_manager.grid_memory_manager
        
        size = task.data.get('size', 1024 * 1024)  # 1MB default
        context_data = task.data.get('context', {})
        
        context = MemoryContext(
            access_pattern=context_data.get('access_pattern', 'random'),
            performance_critical=context_data.get('performance_critical', False),
            compute_intensive=context_data.get('compute_intensive', False),
            gpu_preference=GPUType(context_data.get('gpu_preference')) if context_data.get('gpu_preference') else None
        )
        
        return manager.allocate_optimized(size, context)
    
    def _update_metrics(self):
        """Update GPU-specific metrics."""
        if self.gpu_integration:
            status = self.gpu_integration.get_integration_status()
            self.metrics.gpu_utilization = status.get('gpu_cluster_status', {}).get('gpu_utilization', {})
            self.metrics.coherence_success_rate = status.get('metrics', {}).get('coherence_success_rate', 1.0)
            self.metrics.cross_forex_volume = Decimal(str(status.get('metrics', {}).get('cross_forex_volume', 0)))


class MemoryLayer(FunctionalLayer):
    """Memory management layer."""
    
    def __init__(self, config: LayerConfiguration):
        self.memory_coherence = None
        self.cross_forex_manager = None
        super().__init__("MEMORY_MANAGEMENT", config)
    
    def _initialize_components(self):
        """Initialize memory layer components."""
        if self.config.enable_memory_coherence:
            self.memory_coherence = MemoryCoherenceProtocol()
            self.memory_coherence.register_gpu(0, 'uhd_coprocessor', range(0x7FFF0000, 0x80000000))
            self.memory_coherence.register_gpu(1, 'discrete_gpu', range(0x80000000, 0x90000000))
        
        if self.config.enable_cross_forex_trading:
            self.cross_forex_manager = CrossForexManager()
        
        self.status = LayerStatus.RUNNING
        logger.info("Memory Management layer initialized")
        
        # Update metrics
        self._update_metrics()
    
    def _execute(self, task: LayerTask) -> Any:
        """Execute memory-related tasks."""
        if task.task_type == "memory_read":
            return self._memory_read(task)
        elif task.task_type == "memory_write":
            return self._memory_write(task)
        elif task.task_type == "memory_allocate":
            return self._memory_allocate(task)
        elif task.task_type == "memory_deallocate":
            return self._memory_deallocate(task)
        elif task.task_type == "coherence_sync":
            return self._coherence_sync(task)
        elif task.task_type == "cross_forex_trade":
            return self._cross_forex_trade(task)
        else:
            raise ValueError(f"Unknown memory task type: {task.task_type}")
    
    def _memory_read(self, task: LayerTask) -> bytes:
        """Read from GPU memory with coherence."""
        gpu_id = task.data.get('gpu_id', 0)
        address = task.data.get('address', 0x7FFF0000)
        
        if self.memory_coherence:
            result = self.memory_coherence.read_access(gpu_id, address)
            if result.success:
                return result.data or b""
            else:
                raise RuntimeError(f"Memory read failed: {result.error_message}")
        else:
            # Simulate read
            return b"simulated_data"
    
    def _memory_write(self, task: LayerTask) -> bool:
        """Write to GPU memory with coherence."""
        gpu_id = task.data.get('gpu_id', 0)
        address = task.data.get('address', 0x7FFF0000)
        data = task.data.get('data', b"")
        
        if self.memory_coherence:
            result = self.memory_coherence.write_access(gpu_id, address, data)
            return result.success
        else:
            # Simulate write
            return True
    
    def _memory_allocate(self, task: LayerTask) -> str:
        """Allocate memory using 3D grid system."""
        from .gpu_pipeline_integration import GPUGridMemoryManager
        
        manager = GPUGridMemoryManager()
        size = task.data.get('size', 1024 * 1024)  # 1MB default
        
        context_data = task.data.get('context', {})
        context = MemoryContext(
            access_pattern=context_data.get('access_pattern', 'random'),
            performance_critical=context_data.get('performance_critical', False),
            compute_intensive=context_data.get('compute_intensive', False)
        )
        
        allocation = manager.allocate_optimized(size, context)
        return allocation.id
    
    def _memory_deallocate(self, task: LayerTask) -> bool:
        """Deallocate memory."""
        # In this implementation, deallocation is handled by garbage collection
        # Real implementation would track allocations and remove them
        return True
    
    def _coherence_sync(self, task: LayerTask) -> bool:
        """Synchronize coherence."""
        gpu_id = task.data.get('gpu_id', 0)
        address = task.data.get('address', 0x7FFF0000)
        
        if self.memory_coherence:
            result = self.memory_coherence.sync_request(gpu_id, address)
            return result.success
        else:
            return True
    
    def _cross_forex_trade(self, task: LayerTask) -> Any:
        """Execute cross-forex memory trade."""
        if not self.cross_forex_manager:
            raise RuntimeError("Cross-forex manager not initialized")
        
        trade_data = task.data
        trade = CrossForexTrade(
            trade_id=trade_data.get('trade_id', f"TRADE_{uuid.uuid4().hex[:8]}"),
            trader_id=trade_data.get('trader_id', 'MEMORY_LAYER'),
            order_type=MarketOrderType(trade_data.get('order_type', 'MARKET_BUY')),
            resource_type=MemoryResourceType(trade_data.get('resource_type', 'VRAM')),
            quantity=trade_data.get('quantity', 1024 * 1024 * 1024),  # 1GB default
            bid_credits=Decimal(str(trade_data.get('bid_credits', 10.0))),
            collateral=Decimal(str(trade_data.get('collateral', 20.0)))
        )
        
        success, message = self.cross_forex_manager.memory_engine.place_trade(trade)
        
        return {
            'success': success,
            'message': message,
            'trade_id': trade.trade_id
        }
    
    def _update_metrics(self):
        """Update memory-specific metrics."""
        if self.memory_coherence:
            stats = self.memory_coherence.get_coherence_stats()
            self.metrics.coherence_success_rate = stats.cache_hits / max(1, stats.cache_hits + stats.cache_misses) if (stats.cache_hits + stats.cache_misses) > 0 else 1.0
            self.metrics.latency_microseconds = stats.average_latency_us


class SignalProcessingLayer(FunctionalLayer):
    """Signal processing layer."""
    
    def __init__(self, config: LayerConfiguration):
        self.signal_scheduler = SignalScheduler()
        self.gpu_signal_handler = None
        self.memory_trading_processor = None
        super().__init__("SIGNAL_PROCESSING", config)
    
    def _initialize_components(self):
        """Initialize signal processing components."""
        # Initialize signal schedulers and handlers
        self.gpu_signal_handler = GPUPipelineSignalHandler(None)  # Will be bound later
        self.memory_trading_processor = MemoryTradingSignalProcessor(None)  # Will be bound later
        
        self.status = LayerStatus.RUNNING
        logger.info("Signal Processing layer initialized")
    
    def _execute(self, task: LayerTask) -> Any:
        """Execute signal processing tasks."""
        if task.task_type == "process_signal":
            return self._process_signal(task)
        elif task.task_type == "schedule_signal":
            return self._schedule_signal(task)
        elif task.task_type == "enqueue_signal":
            return self._enqueue_signal(task)
        elif task.task_type == "dispatch_signals":
            return self._dispatch_signals(task)
        else:
            raise ValueError(f"Unknown signal task type: {task.task_type}")
    
    def _process_signal(self, task: LayerTask) -> List[LayerTask]:
        """Process a signal."""
        signal = task.data.get('signal')
        if not signal:
            raise ValueError("Signal parameter required")
        
        gpu_tasks = []
        memory_tasks = []
        
        # Process signal based on type
        if signal.kind in [SignalKind.CPU_BOTTLENECK, SignalKind.GPU_BOTTLENECK]:
            # Generate GPU allocation task
            gpu_tasks.append(LayerTask(
                task_id=f"GPU_OFFLOAD_{signal.id}",
                task_type="allocate_gpu_resources",
                data={
                    'resource_type': 'compute_units',
                    'amount': int(signal.strength * 2000),
                    'priority': 7 + int(signal.strength * 3)
                },
                priority=TaskPriority.HIGH
            ))
        
        elif signal.kind == SignalKind.MEMORY_PRESSURE:
            # Generate memory allocation task
            memory_tasks.append(LayerTask(
                task_id=f"MEMORY_ALLOC_{signal.id}",
                task_type="memory_allocate",
                data={
                    'size': int(signal.strength * 1024 * 1024 * 1024),  # Scale by 1GB
                    'context': {
                        'access_pattern': 'random',
                        'performance_critical': signal.strength > 0.7
                    }
                },
                priority=TaskPriority.NORMAL
            ))
        
        elif signal.kind == SignalKind.THERMAL_WARNING:
            # Generate UHD coprocessor offload
            gpu_tasks.append(LayerTask(
                task_id=f"THERMAL_REDUCTION_{signal.id}",
                task_type="allocate_gpu_resources",
                data={
                    'resource_type': 'compute_units',
                    'amount': 1000,
                    'gpu_preference': 0,  # UHD GPU
                    'priority': 9
                },
                priority=TaskPriority.CRITICAL
            ))
        
        # Return tasks to be processed by other layers
        return gpu_tasks + memory_tasks
    
    def _schedule_signal(self, task: LayerTask) -> bool:
        """Schedule a signal."""
        signal = task.data.get('signal')
        if signal:
            self.signal_scheduler.enqueue(signal)
            return True
        return False
    
    def _enqueue_signal(self, task: LayerTask) -> str:
        """Enqueue a signal."""
        signal = task.data.get('signal')
        if signal:
            self.signal_scheduler.enqueue(signal)
            return signal.id
        raise ValueError("Signal parameter required")
    
    def _dispatch_signals(self, task: LayerTask) -> List[str]:
        """Dispatch all pending signals."""
        dispatched = []
        while True:
            signal = self.signal_scheduler.dequeue()
            if not signal:
                break
            dispatched.append(signal.id)
        return dispatched


class FunctionalLayerOrchestrator:
    """Orchestrates all functional layers."""
    
    def __init__(self, config: Optional[LayerConfiguration] = None):
        self.config = config or LayerConfiguration()
        self.layers: Dict[str, FunctionalLayer] = {}
        self.status = LayerStatus.STARTING
        self.lock = threading.RLock()
        self.executor = ThreadPoolExecutor(max_workers=32)  # For orchestrator tasks
        
        # Initialize layers
        self._initialize_layers()
        
    def _initialize_layers(self):
        """Initialize all functional layers."""
        logger.info("Initializing functional layers...")
        
        # GPU Layer
        gpu_layer = GPULayer(self.config)
        self.layers['gpu'] = gpu_layer
        
        # Memory Layer
        memory_layer = MemoryLayer(self.config)
        self.layers['memory'] = memory_layer
        
        # Signal Processing Layer
        signal_layer = SignalProcessingLayer(self.config)
        self.layers['signal'] = signal_layer
        
        self.status = LayerStatus.RUNNING
        logger.info("All functional layers initialized and running")
    
    def submit_task(self, layer_name: str, task: LayerTask) -> str:
        """Submit a task to a specific layer."""
        layer = self.layers.get(layer_name)
        if not layer:
            raise ValueError(f"Unknown layer: {layer_name}")
        return layer.submit_task(task)
    
    def execute_cross_layer_task(self, tasks: List[Tuple[str, LayerTask]]) -> List[LayerResult]:
        """Execute tasks across multiple layers."""
        results = []
        
        for layer_name, task in tasks:
            result = self.submit_and_wait(layer_name, task)
            results.append(result)
        
        return results
    
    def submit_and_wait(self, layer_name: str, task: LayerTask, 
                       timeout: Optional[float] = None) -> LayerResult:
        """Submit a task and wait for completion."""
        task_id = self.submit_task(layer_name, task)
        
        start_time = time.time()
        while task_id not in self.layers[layer_name].results:
            if timeout and (time.time() - start_time) > timeout:
                raise TimeoutError(f"Task {task_id} timed out")
            time.sleep(0.001)
        
        return self.layers[layer_name].results[task_id]
    
    def get_layer(self, layer_name: str) -> Optional[FunctionalLayer]:
        """Get a layer by name."""
        return self.layers.get(layer_name)
    
    def get_all_metrics(self) -> Dict[str, FunctionalLayerMetrics]:
        """Get metrics from all layers."""
        metrics = {}
        for name, layer in self.layers.items():
            metrics[name] = layer.get_metrics()
        return metrics
    
    def get_overall_status(self) -> Dict[str, str]:
        """Get overall system status."""
        status = {}
        for name, layer in self.layers.items():
            status[name] = layer.status.value
        return status
    
    def start(self):
        """Start the orchestrator and all layers."""
        with self.lock:
            for name, layer in self.layers.items():
                layer.start()
            self.status = LayerStatus.RUNNING
            logger.info("Functional Layer Orchestrator started")
    
    def stop(self):
        """Stop the orchestrator and all layers."""
        with self.lock:
            self.status = LayerStatus.STOPPING
            for name, layer in self.layers.items():
                layer.stop()
            self.executor.shutdown(wait=True)
            self.status = LayerStatus.STOPPED
            logger.info("Functional Layer Orchestrator stopped")
    
    def process_game_scenario(self, scenario_data: Dict) -> Dict[str, Any]:
        """Process a complete gaming scenario involving all layers."""
        results = {
            'gpu_tasks': [],
            'memory_tasks': [],
            'signal_processing': [],
            'errors': []
        }
        
        try:
            # Step 1: Process telemetry (could come from game engine)
            telemetry_task = LayerTask(
                task_id=f"TELEMETRY_{uuid.uuid4().hex[:8]}",
                task_type="process_telemetry",
                data={'telemetry': TelemetrySnapshot(
                    timestamp=datetime.now().isoformat(),
                    cpu_util=scenario_data.get('cpu_util', 0.8),
                    gpu_util=scenario_data.get('gpu_util', 0.85),
                    frametime_ms=scenario_data.get('frametime_ms', 16.67),
                    temp_cpu=scenario_data.get('temp_cpu', 70),
                    temp_gpu=scenario_data.get('temp_gpu', 75),
                    active_process_category=scenario_data.get('category', 'gaming')
                )}
            )
            
            # Submit telemetry processing
            gpu_requests = self.submit_and_wait('signal', telemetry_task)
            if gpu_requests.result:
                results['gpu_tasks'].extend(gpu_requests.result)
            
            # Step 2: Process additional signals
            signals = []
            if scenario_data.get('cpu_bottleneck'):
                signals.append(Signal(
                    id=f"CPU_BOTTLENECK_{uuid.uuid4().hex[:8]}",
                    source="SCENARIO_ENGINE",
                    kind=SignalKind.CPU_BOTTLENECK,
                    strength=0.9,
                    confidence=0.85,
                    payload={"bottleneck_type": "compute", "recommended_action": "gpu_offload"}
                ))
            
            if scenario_data.get('thermal_warning'):
                signals.append(Signal(
                    id=f"THERMAL_WARNING_{uuid.uuid4().hex[:8]}",
                    source="SCENARIO_ENGINE",
                    kind=SignalKind.THERMAL_WARNING,
                    strength=0.7,
                    confidence=0.9,
                    payload={"component": "gpu", "temperature": 85, "recommended_action": "switch_to_cooler_path"}
                ))
            
            # Process signals
            for signal in signals:
                signal_task = LayerTask(
                    task_id=f"SIG_TASK_{uuid.uuid4().hex[:8]}",
                    task_type="process_signal",
                    data={'signal': signal}
                )
                
                signal_result = self.submit_and_wait('signal', signal_task)
                if signal_result.success and signal_result.result:
                    results['signal_processing'].append(signal_result.result)
            
            # Step 3: Execute GPU allocation requests
            for gpu_request in results['gpu_tasks']:
                if hasattr(gpu_request, 'request_id'):
                    gpu_task = LayerTask(
                        task_id=f"GPU_REQ_{uuid.uuid4().hex[:8]}",
                        task_type="allocate_gpu_resources",
                        data={'request': gpu_request}
                    )
                    
                    gpu_result = self.submit_and_wait('gpu', gpu_task)
                    results['gpu_tasks'].append(gpu_result)
            
            # Step 4: Execute memory allocation requests
            for signal_result in results['signal_processing']:
                if isinstance(signal_result, list):
                    for task_item in signal_result:
                        if hasattr(task_item, 'task_type') and 'memory' in task_item.task_type:
                            memory_result = self.submit_and_wait('memory', task_item)
                            results['memory_tasks'].append(memory_result)
        
        except Exception as e:
            results['errors'].append(str(e))
            logger.error(f"Game scenario processing failed: {e}")
        
        return results


class SystemMonitor:
    """Monitor system resources and health."""
    
    def __init__(self, orchestrator: FunctionalLayerOrchestrator):
        self.orchestrator = orchestrator
        self.monitoring = False
        self.health_metrics = {}
        self.system_resources = {}
    
    def start_monitoring(self):
        """Start system monitoring."""
        self.monitoring = True
        logger.info("System monitoring started")
    
    def stop_monitoring(self):
        """Stop system monitoring."""
        self.monitoring = False
        logger.info("System monitoring stopped")
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get current system health."""
        # Get CPU and memory usage
        cpu_percent = psutil.cpu_percent(interval=1)
        memory_percent = psutil.virtual_memory().percent
        
        # Get orchestrator status
        layer_status = self.orchestrator.get_overall_status()
        metrics = self.orchestrator.get_all_metrics()
        
        health = {
            'timestamp': time.time(),
            'system_resources': {
                'cpu_percent': cpu_percent,
                'memory_percent': memory_percent,
                'platform': platform.system(),
                'processors': psutil.cpu_count(),
                'total_memory_gb': round(psutil.virtual_memory().total / (1024**3), 2)
            },
            'layer_status': layer_status,
            'performance_metrics': {name: {
                'tasks_completed': m.tasks_completed,
                'tasks_failed': m.tasks_failed,
                'avg_execution_time': m.average_execution_time,
                'coherence_rate': m.coherence_success_rate
            } for name, m in metrics.items()},
            'health_score': self._calculate_health_score(metrics, cpu_percent, memory_percent)
        }
        
        return health
    
    def _calculate_health_score(self, metrics: Dict, cpu_percent: float, memory_percent: float) -> float:
        """Calculate system health score (0.0 to 1.0)."""
        # Calculate based on various factors
        cpu_score = max(0, min(1, 1 - (cpu_percent / 100)))
        memory_score = max(0, min(1, 1 - (memory_percent / 100)))
        
        # Average of metrics
        avg_completion_rate = 0
        total_layers = 0
        for m in metrics.values():
            total = m.tasks_completed + m.tasks_failed
            if total > 0:
                avg_completion_rate += m.tasks_completed / total
                total_layers += 1
        
        completion_score = avg_completion_rate / total_layers if total_layers > 0 else 1.0
        coherence_score = statistics.mean(
            [m.coherence_success_rate for m in metrics.values()] 
            if metrics.values() else [1.0]
        )
        
        # Weighted average
        health_score = (
            0.3 * cpu_score +
            0.3 * memory_score + 
            0.2 * completion_score + 
            0.2 * coherence_score
        )
        
        return health_score


def demo_functional_layers():
    """Demonstrate the functional layers system."""
    print("=== GAMESA Functional Layers Demo ===\n")
    
    # Create configuration
    config = LayerConfiguration()
    config.enable_gpu_integration = True
    config.enable_memory_coherence = True
    config.enable_cross_forex_trading = True
    config.enable_3d_grid_memory = True
    config.enable_uhd_coprocessor = True
    config.max_parallel_tasks = 8
    
    # Initialize orchestrator
    orchestrator = FunctionalLayerOrchestrator(config)
    monitor = SystemMonitor(orchestrator)
    
    print("Starting functional layers...")
    orchestrator.start()
    monitor.start_monitoring()
    
    print("Functional layer statuses:", orchestrator.get_overall_status())
    
    # Submit some test tasks
    print("\nSubmitting test tasks...")
    
    # 1. GPU resource allocation task
    gpu_task = LayerTask(
        task_id="GPU_TEST_001",
        task_type="allocate_gpu_resources",
        data={
            'resource_type': 'compute_units',
            'amount': 2048,
            'priority': 8,
            'bid_credits': 50.0
        },
        priority=TaskPriority.HIGH
    )
    gpu_task_id = orchestrator.submit_task('gpu', gpu_task)
    print(f"Submitted GPU task: {gpu_task_id}")
    
    # 2. Memory allocation task
    memory_task = LayerTask(
        task_id="MEMORY_TEST_001",
        task_type="memory_allocate",
        data={
            'size': 1024 * 1024 * 256,  # 256MB
            'context': {
                'access_pattern': 'sequential',
                'performance_critical': True,
                'compute_intensive': True
            }
        },
        priority=TaskPriority.NORMAL
    )
    memory_task_id = orchestrator.submit_task('memory', memory_task)
    print(f"Submitted memory task: {memory_task_id}")
    
    # 3. Signal processing task
    signal = Signal(
        id="SIGNAL_TEST_001",
        source="DEMO",
        kind=SignalKind.CPU_BOTTLENECK,
        strength=0.8,
        confidence=0.9,
        payload={"bottleneck_type": "compute", "recommended_action": "gpu_offload"}
    )
    
    signal_task = LayerTask(
        task_id="SIGNAL_TEST_001",
        task_type="process_signal",
        data={'signal': signal},
        priority=TaskPriority.HIGH
    )
    signal_task_id = orchestrator.submit_task('signal', signal_task)
    print(f"Submitted signal task: {signal_task_id}")
    
    # Wait for tasks to complete and get results
    print("\nWaiting for task completion...")
    gpu_result = orchestrator.submit_and_wait('gpu', gpu_task, timeout=10.0)
    memory_result = orchestrator.submit_and_wait('memory', memory_task, timeout=10.0)
    signal_result = orchestrator.submit_and_wait('signal', signal_task, timeout=10.0)
    
    print(f"\nTask Results:")
    print(f"  GPU Task: Success={gpu_result.success}, Execution Time={gpu_result.execution_time:.3f}s")
    print(f"  Memory Task: Success={memory_result.success}, Execution Time={memory_result.execution_time:.3f}s")
    print(f"  Signal Task: Success={signal_result.success}, Execution Time={signal_result.execution_time:.3f}s")
    
    # Get system health
    print("\nSystem Health:")
    health = monitor.get_system_health()
    print(f"  Health Score: {health['health_score']:.3f}")
    print(f"  CPU Usage: {health['system_resources']['cpu_percent']:.1f}%")
    print(f"  Memory Usage: {health['system_resources']['memory_percent']:.1f}%")
    print(f"  Platform: {health['system_resources']['platform']}")
    
    # Process a complete scenario
    print("\nProcessing gaming scenario...")
    scenario_result = orchestrator.process_game_scenario({
        'cpu_util': 0.9,
        'gpu_util': 0.85,
        'frametime_ms': 14.0,
        'temp_cpu': 80,
        'temp_gpu': 78,
        'category': 'intensive_gaming',
        'cpu_bottleneck': True,
        'thermal_warning': True
    })
    
    print(f"Scenario processed:")
    print(f"  GPU Tasks: {len(scenario_result['gpu_tasks'])}")
    print(f"  Memory Tasks: {len(scenario_result['memory_tasks'])}")
    print(f"  Signal Processing: {len(scenario_result['signal_processing'])}")
    print(f"  Errors: {len(scenario_result['errors'])}")
    
    # Get all metrics
    print(f"\nAll Layer Metrics:")
    all_metrics = orchestrator.get_all_metrics()
    for layer_name, metrics in all_metrics.items():
        print(f"  {layer_name}:")
        print(f"    Tasks Completed: {metrics.tasks_completed}")
        print(f"    Tasks Failed: {metrics.tasks_failed}")
        print(f"    Avg Execution Time: {metrics.average_execution_time:.4f}s")
        print(f"    Coherence Rate: {metrics.coherence_success_rate:.3f}")
    
    # Stop everything
    print(f"\nStopping functional layers...")
    monitor.stop_monitoring()
    orchestrator.stop()
    
    print(f"\nFunctional layers demo completed successfully!")


if __name__ == "__main__":
    demo_functional_layers()