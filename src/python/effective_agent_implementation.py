"""
Effective Agent Implementation for GAMESA/KrystalStack Framework

Implements agent-based architecture for the GPU pipeline with 3D grid memory,
cross-forex trading, and memory coherence integration.
"""

import asyncio
import threading
import time
from typing import Dict, List, Optional, Any, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum, auto
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
from decimal import Decimal
from datetime import datetime
import uuid
import queue
import weakref
from collections import deque
import statistics


# Import existing components
from . import (
    # Core GAMESA components
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

from .functional_layer import (
    FunctionalLayerOrchestrator, SystemMonitor, LayerTask, TaskPriority,
    ExecutionMode, LayerStatus, LayerConfiguration
)


# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Agent-related enums
class AgentType(Enum):
    """Types of agents in the system."""
    CPU_BOTTLENECK_OPTIMIZER = "cpu_bottleneck_optimizer"
    GPU_BOTTLENECK_OPTIMIZER = "gpu_bottleneck_optimizer"
    THERMAL_MANAGEMENT = "thermal_management"
    MEMORY_OPTIMIZER = "memory_optimizer"
    POWER_EFFICIENCY = "power_efficiency"
    PERFORMANCE_BOOST = "performance_boost"
    SAFETY_MONITOR = "safety_monitor"
    TRADING_AGENT = "trading_agent"
    COHERENCE_AGENT = "coherence_agent"
    UHD_SCHEDULER = "uhd_scheduler"


class AgentStatus(Enum):
    """Status of agents."""
    IDLE = "idle"
    ACTIVE = "active"
    PROCESSING = "processing"
    WAITING = "waiting"
    ERROR = "error"
    TERMINATED = "terminated"


class TaskPriority(Enum):
    """Task priority levels."""
    CRITICAL = 10
    HIGH = 8
    NORMAL = 5
    LOW = 3
    BACKGROUND = 1


class AgentRole(Enum):
    """Role of the agent in the system."""
    OPTIMIZER = "optimizer"
    MONITOR = "monitor"
    CONTROLLER = "controller"
    TRADER = "trader"
    COORDINATOR = "coordinator"
    SAFETY_GUARD = "safety_guard"


# Data Classes
@dataclass
class AgentMetrics:
    """Metrics for agent performance."""
    tasks_executed: int = 0
    tasks_failed: int = 0
    execution_time_avg: float = 0.0
    success_rate: float = 1.0
    resource_efficiency: float = 0.0
    trading_profit: Decimal = Decimal('0.00')
    coherence_success_rate: float = 1.0
    memory_efficiency: float = 0.0
    latency_us: float = 0.0
    timestamp: float = field(default_factory=time.time)


@dataclass
class AgentTask:
    """Task for an agent to execute."""
    task_id: str
    agent_type: AgentType
    task_type: str  # "allocation", "optimization", "monitoring", "trading", etc.
    data: Dict[str, Any]
    priority: TaskPriority = TaskPriority.NORMAL
    deadline: Optional[float] = None  # Unix timestamp
    max_attempts: int = 3
    current_attempt: int = 0
    callback: Optional[Callable[[Any], None]] = None
    timeout: float = 10.0  # seconds
    dependencies: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentResult:
    """Result from agent execution."""
    task_id: str
    success: bool
    result: Optional[Any] = None
    error: Optional[str] = None
    execution_time: float = 0.0
    metrics: AgentMetrics = field(default_factory=AgentMetrics)
    timestamp: float = field(default_factory=time.time)


@dataclass
class AgentProfile:
    """Profile defining agent capabilities and responsibilities."""
    agent_type: AgentType
    role: AgentRole
    capabilities: List[str]
    priority_weight: float = 1.0
    resource_shares: Dict[str, float] = field(default_factory=dict)  # CPU, GPU, Memory shares
    trading_capital: Decimal = Decimal('1000.00')
    max_concurrent_tasks: int = 5
    performance_threshold: float = 0.8  # Minimum performance required


class BaseAgent(ABC):
    """Base class for all agents."""
    
    def __init__(self, agent_id: str, profile: AgentProfile, controller_ref: 'weakref.ReferenceType'):
        self.agent_id = agent_id
        self.profile = profile
        self.controller_ref = controller_ref
        self.status = AgentStatus.IDLE
        self.metrics = AgentMetrics()
        self.task_queue = queue.Queue()
        self.active_tasks: Dict[str, AgentTask] = {}
        self.executor = ThreadPoolExecutor(max_workers=min(4, profile.max_concurrent_tasks))
        self.lock = threading.RLock()
        self.last_update = time.time()
        
        logger.info(f"Agent {self.agent_id} initialized with profile {self.profile.agent_type.value}")
    
    def submit_task(self, task: AgentTask) -> str:
        """Submit a task to this agent."""
        task_id = task.task_id or f"TASK_{uuid.uuid4().hex[:8]}"
        task.task_id = task_id
        
        with self.lock:
            self.task_queue.put(task)
            self.status = AgentStatus.ACTIVE
        
        logger.debug(f"Task {task_id} submitted to agent {self.agent_id}")
        return task_id
    
    def process_next_task(self) -> Optional[AgentResult]:
        """Process the next task in the queue."""
        try:
            task = self.task_queue.get_nowait()
        except queue.Empty:
            return None
        
        with self.lock:
            self.active_tasks[task.task_id] = task
            self.status = AgentStatus.PROCESSING
        
        start_time = time.time()
        
        try:
            result_data = self._execute_task(task)
            success = True
            error = None
        except Exception as e:
            result_data = None
            success = False
            error = str(e)
            logger.error(f"Agent {self.agent_id} task {task.task_id} failed: {e}")
        
        execution_time = time.time() - start_time
        
        # Update metrics
        with self.lock:
            if success:
                self.metrics.tasks_executed += 1
            else:
                self.metrics.tasks_failed += 1
            
            # Update average execution time
            total_completed = self.metrics.tasks_executed + self.metrics.tasks_failed
            self.metrics.execution_time_avg = (
                (self.metrics.execution_time_avg * (total_completed - 1) + execution_time) / total_completed
                if total_completed > 0 else execution_time
            )
            
            self.metrics.success_rate = (
                self.metrics.tasks_executed / total_completed 
                if total_completed > 0 else 1.0
            )
        
        result = AgentResult(
            task_id=task.task_id,
            success=success,
            result=result_data,
            error=error,
            execution_time=execution_time,
            metrics=self.metrics,
            timestamp=time.time()
        )
        
        # Execute callback if provided
        if task.callback:
            try:
                task.callback(result)
            except Exception as e:
                logger.error(f"Callback for task {task.task_id} failed: {e}")
        
        # Remove from active tasks
        with self.lock:
            self.active_tasks.pop(task.task_id, None)
            if not self.active_tasks and self.task_queue.empty():
                self.status = AgentStatus.IDLE
        
        return result
    
    @abstractmethod
    def _execute_task(self, task: AgentTask) -> Any:
        """Execute the specific task (to be implemented by subclasses)."""
        pass
    
    def get_status(self) -> Dict[str, Any]:
        """Get agent status."""
        with self.lock:
            return {
                'agent_id': self.agent_id,
                'agent_type': self.profile.agent_type.value,
                'role': self.profile.role.value,
                'status': self.status.value,
                'active_tasks': len(self.active_tasks),
                'queue_size': self.task_queue.qsize(),
                'metrics': {
                    'tasks_executed': self.metrics.tasks_executed,
                    'tasks_failed': self.metrics.tasks_failed,
                    'success_rate': self.metrics.success_rate,
                    'avg_execution_time': self.metrics.execution_time_avg,
                    'resource_efficiency': self.metrics.resource_efficiency
                },
                'timestamp': time.time()
            }
    
    def stop(self):
        """Stop the agent."""
        with self.lock:
            self.status = AgentStatus.TERMINATED
            self.executor.shutdown(wait=False)
            logger.info(f"Agent {self.agent_id} stopped")


class GPUOptimizerAgent(BaseAgent):
    """Optimizes GPU resource allocation and performance."""
    
    def _execute_task(self, task: AgentTask) -> Any:
        """Execute GPU optimization task."""
        controller = self.controller_ref()
        if not controller:
            raise RuntimeError("Controller reference lost")
        
        if task.task_type == "gpu_resource_allocation":
            return self._optimize_gpu_allocation(task)
        elif task.task_type == "performance_optimization":
            return self._optimize_performance(task)
        elif task.task_type == "thermal_management":
            return self._manage_thermal(task)
        else:
            raise ValueError(f"Unknown task type: {task.task_type}")
    
    def _optimize_gpu_allocation(self, task: AgentTask) -> Any:
        """Optimize GPU resource allocation."""
        from .gamesa_gpu_integration import GPUAllocationRequest
        
        request_data = task.data
        request = GPUAllocationRequest(
            request_id=f"OPT_{task.task_id}",
            agent_id=self.agent_id,
            resource_type=request_data.get('resource_type', 'compute_units'),
            amount=request_data.get('amount', 1024),
            priority=request_data.get('priority', 5),
            bid_credits=Decimal(str(request_data.get('bid_credits', 10.0))),
            constraints=request_data.get('constraints', {}),
            memory_context=request_data.get('memory_context'),
            performance_goals=request_data.get('performance_goals', {})
        )
        
        # Use the GPU integration to process request
        integration = self.controller_ref().integration
        return integration.request_gpu_resources(request)
    
    def _optimize_performance(self, task: AgentTask) -> Any:
        """Optimize GPU performance."""
        # Get current system state
        integration = self.controller_ref().integration
        status = integration.get_integration_status()
        
        performance_improvements = []
        
        # Check if we can improve VRAM utilization
        if status.get('metrics', {}).get('memory_efficiency', 0.0) < 0.7:
            performance_improvements.append({
                'improvement': 'memory_optimization',
                'type': 'allocation_strategy',
                'parameters': {'strategy': '3d_grid_optimized'},
                'expected_improvement': 0.15
            })
        
        # Check thermal efficiency
        if status.get('metrics', {}).get('thermal_efficiency', 1.0) < 0.8:
            performance_improvements.append({
                'improvement': 'thermal_optimization',
                'type': 'task_distribution',
                'parameters': {'prefer_cooler_gpus': True},
                'expected_improvement': 0.10
            })
        
        # Update agent metrics based on performance
        with self.lock:
            self.metrics.resource_efficiency = status.get('metrics', {}).get('gpu_utilization', 0.0)
        
        return performance_improvements
    
    def _manage_thermal(self, task: AgentTask) -> Any:
        """Manage thermal conditions."""
        telemetry = task.data.get('telemetry')
        if not telemetry:
            return {"error": "Telemetry data required for thermal management"}
        
        thermal_actions = []
        thermal_threshold = 75  # Degrees Celsius
        
        if telemetry.temp_gpu > thermal_threshold:
            # Switch to cooler UHD coprocessor
            thermal_actions.append({
                'action': 'switch_to_cooler_gpu',
                'target_gpu': 0,  # UHD coprocessor
                'reason': 'thermal_reduction',
                'priority': TaskPriority.HIGH
            })
        
        if telemetry.temp_cpu > thermal_threshold:
            # Reduce CPU-intensive tasks
            thermal_actions.append({
                'action': 'cpu_load_reduction',
                'target_reduction': 0.2,
                'reason': 'thermal_management',
                'priority': TaskPriority.NORMAL
            })
        
        return thermal_actions


class MemoryOptimizerAgent(BaseAgent):
    """Optimizes memory allocation using 3D grid system and cross-forex trading."""
    
    def _execute_task(self, task: AgentTask) -> Any:
        """Execute memory optimization task."""
        if task.task_type == "memory_allocation":
            return self._optimize_memory_allocation(task)
        elif task.task_type == "memory_trading":
            return self._execute_memory_trade(task)
        elif task.task_type == "coherence_optimization":
            return self._optimize_coherence(task)
        else:
            raise ValueError(f"Unknown memory task type: {task.task_type}")
    
    def _optimize_memory_allocation(self, task: AgentTask) -> Any:
        """Optimize memory allocation using 3D grid system."""
        from .gpu_pipeline_integration import MemoryContext
        
        size = task.data.get('size', 1024 * 1024)  # 1MB default
        context_data = task.data.get('context', {})
        
        # Create memory context
        context = MemoryContext(
            access_pattern=context_data.get('access_pattern', 'random'),
            performance_critical=context_data.get('performance_critical', False),
            compute_intensive=context_data.get('compute_intensive', False)
        )
        
        # Use grid memory manager
        integration = self.controller_ref().integration
        grid_manager = integration.cross_forex_manager.grid_memory_manager
        
        # Allocate optimized memory
        allocation = grid_manager.allocate_optimized(size, context)
        
        # Update metrics
        with self.lock:
            self.metrics.memory_efficiency = allocation.performance_metrics.get('latency', 0.0)
        
        return allocation
    
    def _execute_memory_trade(self, task: AgentTask) -> Any:
        """Execute cross-forex memory trading."""
        integration = self.controller_ref().integration
        cross_forex = integration.cross_forex_manager
        
        trade_data = task.data
        trade = CrossForexTrade(
            trade_id=f"MEMORY_TRADE_{task.task_id}",
            trader_id=self.agent_id,
            order_type=MarketOrderType(trade_data.get('order_type', 'MARKET_BUY')),
            resource_type=MemoryResourceType(trade_data.get('resource_type', 'VRAM')),
            quantity=trade_data.get('quantity', 1024 * 1024 * 1024),  # 1GB default
            bid_credits=Decimal(str(trade_data.get('bid_credits', 50.0))),
            collateral=Decimal(str(trade_data.get('collateral', 100.0)))
        )
        
        success, message = cross_forex.memory_engine.place_trade(trade)
        
        if success:
            with self.lock:
                self.metrics.trading_profit += Decimal(str(trade_data.get('profit', 0.0)))
        
        return {
            'success': success,
            'message': message,
            'trade_id': trade.trade_id if success else None
        }
    
    def _optimize_coherence(self, task: AgentTask) -> Any:
        """Optimize memory coherence performance."""
        integration = self.controller_ref().integration
        coherence_manager = integration.coherence_manager
        
        # Get coherence statistics
        stats = coherence_manager.protocol.get_coherence_stats()
        
        optimization_suggestions = []
        
        # Check coherence success rate
        if stats.success_rate < 0.95:
            optimization_suggestions.append({
                'suggestion': 'increase_coherence_frequency',
                'type': 'protocol_optimization',
                'parameters': {'frequency_factor': 1.2},
                'expected_improvement': 0.05
            })
        
        # Check latency
        if stats.average_latency > 1.0:  # 1ms threshold
            optimization_suggestions.append({
                'suggestion': 'optimize_cache_prefetching',
                'type': 'cache_optimization',
                'parameters': {'prefetch_distance': 8},
                'expected_improvement': 0.15
            })
        
        # Update metrics
        with self.lock:
            self.metrics.coherence_success_rate = stats.success_rate
            self.metrics.latency_us = stats.average_latency * 1000  # Convert to microseconds
        
        return optimization_suggestions


class TradingAgent(BaseAgent):
    """Manages cross-forex memory trading operations."""
    
    def _execute_task(self, task: AgentTask) -> Any:
        """Execute trading task."""
        if task.task_type == "execute_trade":
            return self._execute_trade(task)
        elif task.task_type == "analyze_market":
            return self._analyze_market(task)
        elif task.task_type == "manage_portfolio":
            return self._manage_portfolio(task)
        else:
            raise ValueError(f"Unknown trading task type: {task.task_type}")
    
    def _execute_trade(self, task: AgentTask) -> Any:
        """Execute a cross-forex trade."""
        integration = self.controller_ref().integration
        cross_forex = integration.cross_forex_manager
        
        trade_data = task.data
        trade = CrossForexTrade(
            trade_id=f"AGENT_TRADE_{task.task_id}",
            trader_id=self.agent_id,
            order_type=MarketOrderType(trade_data.get('order_type', 'MARKET_BUY')),
            resource_type=MemoryResourceType(trade_data.get('resource_type', 'VRAM')),
            quantity=trade_data.get('quantity', 1024 * 1024 * 1024),  # 1GB
            bid_credits=Decimal(str(trade_data.get('bid_credits', 100.00))),
            collateral=Decimal(str(trade_data.get('collateral', 200.00))),
            stop_loss=Decimal(str(trade_data.get('stop_loss', 0.0))),
            take_profit=Decimal(str(trade_data.get('take_profit', 0.0)))
        )
        
        success, message = cross_forex.memory_engine.place_trade(trade)
        
        if success:
            with self.lock:
                self.metrics.trading_profit += Decimal(str(trade_data.get('expected_profit', 0.0)))
        
        return {
            'success': success,
            'message': message,
            'trade_id': trade.trade_id,
            'profit': float(trade_data.get('expected_profit', 0.0))
        }
    
    def _analyze_market(self, task: AgentTask) -> Any:
        """Analyze memory market conditions."""
        integration = self.controller_ref().integration
        cross_forex = integration.cross_forex_manager
        
        market_data = cross_forex.memory_engine.get_market_data()
        analysis = {
            'current_prices': market_data.get('current_prices', {}),
            'volume_trends': market_data.get('volume_trends', {}),
            'volatility': market_data.get('volatility', {}),
            'trading_opportunities': []
        }
        
        # Look for arbitrage opportunities
        resource_types = list(analysis['current_prices'].keys())
        for i, resource_a in enumerate(resource_types):
            for j, resource_b in enumerate(resource_types[i+1:], i+1):
                price_a = analysis['current_prices'][resource_a]
                price_b = analysis['current_prices'][resource_b]
                
                # Simple arbitrage check (in practice, this would be more sophisticated)
                if abs(price_a - price_b) / min(price_a, price_b + 0.001) > 0.05:  # 5% difference
                    analysis['trading_opportunities'].append({
                        'opportunity': 'arbitrage',
                        'resources': [resource_a, resource_b],
                        'price_difference': abs(price_a - price_b),
                        'profit_potential': 0.03  # 3% potential profit
                    })
        
        return analysis
    
    def _manage_portfolio(self, task: AgentTask) -> Any:
        """Manage trading portfolio."""
        integration = self.controller_ref().integration
        cross_forex = integration.cross_forex_manager
        
        portfolio = cross_forex.memory_engine.portfolios.get(self.agent_id)
        if not portfolio:
            portfolio = cross_forex.memory_engine.create_portfolio(self.agent_id)
        
        portfolio_analysis = {
            'portfolio_id': portfolio.portfolio_id,
            'cash_balance': float(portfolio.cash_balance),
            'total_value': float(portfolio.total_value),
            'resources': {str(k): v for k, v in portfolio.resources.items()},
            'allocation_diversity': len(portfolio.resources),
            'risk_metrics': self._calculate_risk_metrics(portfolio),
            'recommendations': []
        }
        
        # Risk management recommendations
        if portfolio_analysis['risk_metrics']['max_allocation_per_resource'] > 0.5:
            portfolio_analysis['recommendations'].append({
                'recommendation': 'diversify_portfolio',
                'reason': 'Too much concentration in single resource type',
                'action': 'distribute_to_other_resources'
            })
        
        if portfolio_analysis['risk_metrics']['volatility'] > 0.3:
            portfolio_analysis['recommendations'].append({
                'recommendation': 'reduce_risk',
                'reason': 'Portfolio volatility too high',
                'action': 'move_to_lower_risk_resources'
            })
        
        return portfolio_analysis
    
    def _calculate_risk_metrics(self, portfolio) -> Dict[str, float]:
        """Calculate risk metrics for portfolio."""
        if not portfolio.resources:
            return {'volatility': 0.0, 'max_allocation_per_resource': 0.0, 'diversification_score': 0.0}
        
        # Calculate allocation distribution
        allocations = [r.quantity for r in portfolio.resources.values()]
        total_allocation = sum(allocations)
        
        if total_allocation == 0:
            return {'volatility': 0.0, 'max_allocation_per_resource': 0.0, 'diversification_score': 0.0}
        
        allocation_ratios = [qty / total_allocation for qty in allocations]
        max_ratio = max(allocation_ratios) if allocation_ratios else 0.0
        
        # Calculate diversification (0-1 scale, 1 = perfectly diversified)
        if len(allocation_ratios) > 1:
            diversification_score = 1.0 - statistics.variance(allocation_ratios)
        else:
            diversification_score = 0.0
        
        return {
            'volatility': statistics.stdev(allocations) if len(allocations) > 1 else 0.0,
            'max_allocation_per_resource': max_ratio,
            'diversification_score': diversification_score
        }


class SafetyMonitorAgent(BaseAgent):
    """Monitors system safety and enforces constraints."""
    
    def _execute_task(self, task: AgentTask) -> Any:
        """Execute safety monitoring task."""
        if task.task_type == "check_safety":
            return self._check_safety(task)
        elif task.task_type == "enforce_constraints":
            return self._enforce_constraints(task)
        elif task.task_type == "emergency_procedures":
            return self._execute_emergency_procedures(task)
        else:
            raise ValueError(f"Unknown safety task type: {task.task_type}")
    
    def _check_safety(self, task: AgentTask) -> Dict[str, Any]:
        """Check system safety conditions."""
        integration = self.controller_ref().integration
        status = integration.get_integration_status()
        
        safety_status = {
            'system_health': 'good',
            'violations': [],
            'risk_assessment': 'low',
            'recommended_actions': []
        }
        
        # Check thermal limits
        thermal_violations = []
        if status.get('metrics', {}).get('gpu_temperature', 0) > 85:
            thermal_violations.append({'component': 'gpu', 'level': 'critical'})
        elif status.get('metrics', {}).get('gpu_temperature', 0) > 75:
            thermal_violations.append({'component': 'gpu', 'level': 'warning'})
        
        if status.get('metrics', {}).get('cpu_temperature', 0) > 90:
            thermal_violations.append({'component': 'cpu', 'level': 'critical'})
        elif status.get('metrics', {}).get('cpu_temperature', 0) > 80:
            thermal_violations.append({'component': 'cpu', 'level': 'warning'})
        
        if thermal_violations:
            safety_status['violations'].extend(thermal_violations)
            safety_status['risk_assessment'] = 'high' if any(v['level'] == 'critical' for v in thermal_violations) else 'medium'
        
        # Check memory pressure
        if status.get('metrics', {}).get('memory_utilization', 0) > 0.95:
            safety_status['violations'].append({'component': 'memory', 'level': 'critical'})
            safety_status['risk_assessment'] = 'high'
        elif status.get('metrics', {}).get('memory_utilization', 0) > 0.85:
            safety_status['violations'].append({'component': 'memory', 'level': 'warning'})
            if safety_status['risk_assessment'] == 'low':
                safety_status['risk_assessment'] = 'medium'
        
        # Generate recommended actions
        if thermal_violations:
            safety_status['recommended_actions'].append({
                'action': 'thermal_mitigation',
                'priority': 'high',
                'steps': ['reduce_gpu_intensity', 'switch_to_uhd_coprocessor', 'activate_cooling']
            })
        
        if 'memory' in [v.get('component', '') for v in safety_status['violations']]:
            safety_status['recommended_actions'].append({
                'action': 'memory_optimization',
                'priority': 'high',
                'steps': ['garbage_collect', 'reduce_memory_footprint', 'allocate_more_memory']
            })
        
        # Update health status based on violations
        if safety_status['violations']:
            safety_status['system_health'] = 'warning' if safety_status['risk_assessment'] == 'medium' else 'critical'
        else:
            safety_status['system_health'] = 'good'
        
        return safety_status
    
    def _enforce_constraints(self, task: AgentTask) -> Dict[str, Any]:
        """Enforce system constraints."""
        integration = self.controller_ref().integration
        
        constraints = task.data.get('constraints', {})
        violations_found = []
        
        # Check each constraint
        for constraint_type, constraint_data in constraints.items():
            if constraint_type == "thermal":
                current_temp = integration.get_current_temperature()
                if current_temp > constraint_data['max_temperature']:
                    violations_found.append({
                        'type': 'thermal',
                        'current': current_temp,
                        'max_allowed': constraint_data['max_temperature'],
                        'action': constraint_data.get('action', 'throttle')
                    })
            
            elif constraint_type == "power":
                current_power = integration.get_current_power_draw()
                if current_power > constraint_data['max_power']:
                    violations_found.append({
                        'type': 'power',
                        'current': current_power,
                        'max_allowed': constraint_data['max_power'],
                        'action': constraint_data.get('action', 'reduce_performance')
                    })
            
            elif constraint_type == "memory":
                current_memory = integration.get_current_memory_usage()
                if current_memory > constraint_data['max_memory']:
                    violations_found.append({
                        'type': 'memory',
                        'current': current_memory,
                        'max_allowed': constraint_data['max_memory'],
                        'action': constraint_data.get('action', 'garbage_collect')
                    })
        
        # Execute enforcement actions
        enforcement_results = []
        for violation in violations_found:
            action = violation['action']
            if action == "throttle":
                integration.perform_throttle_action()
            elif action == "reduce_performance":
                integration.reduce_performance_for_safety()
            elif action == "garbage_collect":
                integration.perform_garbage_collection()
            elif action == "switch_to_cooler_path":
                integration.switch_to_uhd_coprocessor()
            
            enforcement_results.append({
                'violation': violation,
                'action_taken': action,
                'success': True
            })
        
        return {
            'violations_found': violations_found,
            'enforcement_results': enforcement_results,
            'total_violations': len(violations_found)
        }
    
    def _execute_emergency_procedures(self, task: AgentTask) -> Dict[str, Any]:
        """Execute emergency safety procedures."""
        integration = self.controller_ref().integration
        
        procedure_data = task.data
        procedure_type = procedure_data.get('procedure_type', 'cooldown')
        severity = procedure_data.get('severity', 'medium')
        
        emergency_actions = []
        
        if procedure_type == 'cooldown':
            # Perform immediate thermal cooldown
            emergency_actions.append({
                'action': 'emergency cooldown initiated',
                'severity': severity,
                'steps': ['reduce_gpu_frequency', 'maximize_fan_speed', 'terminate_non_essential_processes']
            })
            integration.emergency_cooldown()
        
        elif procedure_type == 'power_crisis':
            # Handle power emergency
            emergency_actions.append({
                'action': 'power_crisis_procedures',
                'severity': severity,
                'steps': ['reduce_power_draw', 'switch_to_efficiency_mode', 'pause_non_critical_tasks']
            })
            integration.activate_power_saving_mode()
        
        elif procedure_type == 'memory_crisis':
            # Handle memory crisis
            emergency_actions.append({
                'action': 'memory_crisis_procedures',
                'severity': severity,
                'steps': ['memory_compaction', 'garbage_collect', 'reduce_memory_allocation']
            })
            integration.perform_emergency_memory_cleanup()
        
        return {
            'procedure_type': procedure_type,
            'severity': severity,
            'actions_taken': emergency_actions,
            'procedure_status': 'completed'
        }


class AgentCoordinator:
    """Coordinates all agents in the system."""
    
    def __init__(self, controller_ref: 'weakref.ReferenceType'):
        self.controller_ref = controller_ref
        self.agents: Dict[str, BaseAgent] = {}
        self.agent_profiles: Dict[AgentType, AgentProfile] = {}
        self.coordination_rules: List[Dict] = []
        self.message_queue = queue.Queue()
        self.running = False
        self.lock = threading.RLock()
        
        self._initialize_agent_profiles()
        self._initialize_coordination_rules()
    
    def _initialize_agent_profiles(self):
        """Initialize default agent profiles."""
        self.agent_profiles = {
            AgentType.CPU_BOTTLENECK_OPTIMIZER: AgentProfile(
                agent_type=AgentType.CPU_BOTTLENECK_OPTIMIZER,
                role=AgentRole.OPTIMIZER,
                capabilities=["cpu_offload", "task_distribution", "scheduling"],
                priority_weight=0.9,
                resource_shares={"cpu": 0.1, "gpu": 0.05, "memory": 0.05},
                trading_capital=Decimal('500.00'),
                max_concurrent_tasks=3,
                performance_threshold=0.85
            ),
            AgentType.GPU_BOTTLENECK_OPTIMIZER: AgentProfile(
                agent_type=AgentType.GPU_BOTTLENECK_OPTIMIZER,
                role=AgentRole.OPTIMIZER,
                capabilities=["gpu_allocation", "render_optimization", "memory_management"],
                priority_weight=0.95,
                resource_shares={"cpu": 0.05, "gpu": 0.3, "memory": 0.1},
                trading_capital=Decimal('1000.00'),
                max_concurrent_tasks=4,
                performance_threshold=0.90
            ),
            AgentType.THERMAL_MANAGEMENT: AgentProfile(
                agent_type=AgentType.THERMAL_MANAGEMENT,
                role=AgentRole.CONTROLLER,
                capabilities=["thermal_monitoring", "cooling_control", "temperature_optimization"],
                priority_weight=1.0,
                resource_shares={"cpu": 0.05, "gpu": 0.05, "memory": 0.05},
                trading_capital=Decimal('250.00'),
                max_concurrent_tasks=2,
                performance_threshold=0.95
            ),
            AgentType.MEMORY_OPTIMIZER: AgentProfile(
                agent_type=AgentType.MEMORY_OPTIMIZER,
                role=AgentRole.OPTIMIZER,
                capabilities=["3d_grid_allocation", "cross_forex_trading", "memory_coherence"],
                priority_weight=0.85,
                resource_shares={"cpu": 0.1, "memory": 0.2},
                trading_capital=Decimal('2000.00'),
                max_concurrent_tasks=5,
                performance_threshold=0.85
            ),
            AgentType.POWER_EFFICIENCY: AgentProfile(
                agent_type=AgentType.POWER_EFFICIENCY,
                role=AgentRole.OPTIMIZER,
                capabilities=["power_management", "efficiency_optimization"],
                priority_weight=0.7,
                resource_shares={"cpu": 0.05, "gpu": 0.05},
                trading_capital=Decimal('500.00'),
                max_concurrent_tasks=3,
                performance_threshold=0.75
            ),
            AgentType.PERFORMANCE_BOOST: AgentProfile(
                agent_type=AgentType.PERFORMANCE_BOOST,
                role=AgentRole.OPTIMIZER,
                capabilities=["gpu_boost", "cpu_boost", "memory_boost"],
                priority_weight=1.0,
                resource_shares={"cpu": 0.2, "gpu": 0.4, "memory": 0.15},
                trading_capital=Decimal('1500.00'),
                max_concurrent_tasks=6,
                performance_threshold=0.95
            ),
            AgentType.SAFETY_MONITOR: AgentProfile(
                agent_type=AgentType.SAFETY_MONITOR,
                role=AgentRole.SAFETY_GUARD,
                capabilities=["safety_monitoring", "constraint_enforcement", "emergency_procedures"],
                priority_weight=1.0,
                resource_shares={"cpu": 0.05, "memory": 0.05},
                trading_capital=Decimal('0.00'),  # Non-trading agent
                max_concurrent_tasks=2,
                performance_threshold=1.0  # Must always perform perfectly
            ),
            AgentType.TRADING_AGENT: AgentProfile(
                agent_type=AgentType.TRADING_AGENT,
                role=AgentRole.TRADER,
                capabilities=["cross_forex_trading", "portfolio_management", "market_analysis"],
                priority_weight=0.6,
                resource_shares={"cpu": 0.15, "memory": 0.1},
                trading_capital=Decimal('5000.00'),
                max_concurrent_tasks=8,
                performance_threshold=0.80
            ),
            AgentType.COHERENCE_AGENT: AgentProfile(
                agent_type=AgentType.COHERENCE_AGENT,
                role=AgentRole.CONTROLLER,
                capabilities=["memory_coherence", "cache_management", "synchronization"],
                priority_weight=0.8,
                resource_shares={"cpu": 0.1, "memory": 0.05},
                trading_capital=Decimal('100.00'),
                max_concurrent_tasks=4,
                performance_threshold=0.90
            ),
            AgentType.UHD_SCHEDULER: AgentProfile(
                agent_type=AgentType.UHD_SCHEDULER,
                role=AgentRole.CONTROLLER,
                capabilities=["uhd_scheduling", "coprocessor_management", "background_tasks"],
                priority_weight=0.75,
                resource_shares={"cpu": 0.05, "gpu": 0.1, "memory": 0.05},
                trading_capital=Decimal('200.00'),
                max_concurrent_tasks=3,
                performance_threshold=0.85
            )
        }
    
    def _initialize_coordination_rules(self):
        """Initialize coordination rules between agents."""
        self.coordination_rules = [
            {
                "condition": {"agent_type": AgentType.THERMAL_MANAGEMENT, "signal": "thermal_warning"},
                "action": "notify_all",
                "target_agents": [AgentType.GPU_BOTTLENECK_OPTIMIZER, AgentType.PERFORMANCE_BOOST, AgentType.POWER_EFFICIENCY],
                "message_template": "THERMAL_EXCEEDED - reduce_intensity"
            },
            {
                "condition": {"agent_type": AgentType.SAFETY_MONITOR, "signal": "safety_violation"},
                "action": "terminate",
                "target_agents": ["all_high_risk"],
                "message_template": "EMERGENCY_TERMINATION_REQUIRED"
            },
            {
                "condition": {"agent_type": AgentType.MEMORY_OPTIMIZER, "signal": "memory_pressure"},
                "action": "assist",
                "target_agents": [AgentType.CPU_BOTTLENECK_OPTIMIZER, AgentType.GPU_BOTTLENECK_OPTIMIZER],
                "message_template": "MEMORY_AVAILABLE_AFTER_OPTIMIZATION"
            },
            {
                "condition": {"agent_type": AgentType.TRADING_AGENT, "signal": "profit_opportunity"},
                "action": "inform",
                "target_agents": [AgentType.PERFORMANCE_BOOST],
                "message_template": "AVAILABLE_RESOURCES_FROM_PROFITABLE_TRADE"
            }
        ]
    
    def create_agent(self, agent_type: AgentType) -> BaseAgent:
        """Create an agent of the specified type."""
        with self.lock:
            if agent_type not in self.agent_profiles:
                raise ValueError(f"Unknown agent type: {agent_type}")
            
            profile = self.agent_profiles[agent_type]
            agent_id = f"{agent_type.value.upper()}_{uuid.uuid4().hex[:8]}"
            
            # Create appropriate agent type
            if agent_type in [AgentType.CPU_BOTTLENECK_OPTIMIZER, AgentType.GPU_BOTTLENECK_OPTIMIZER, 
                             AgentType.THERMAL_MANAGEMENT, AgentType.POWER_EFFICIENCY, AgentType.PERFORMANCE_BOOST]:
                agent = GPUOptimizerAgent(agent_id, profile, self.controller_ref)
            elif agent_type in [AgentType.MEMORY_OPTIMIZER, AgentType.COHERENCE_AGENT]:
                agent = MemoryOptimizerAgent(agent_id, profile, self.controller_ref)
            elif agent_type == AgentType.TRADING_AGENT:
                agent = TradingAgent(agent_id, profile, self.controller_ref)
            elif agent_type == AgentType.SAFETY_MONITOR:
                agent = SafetyMonitorAgent(agent_id, profile, self.controller_ref)
            elif agent_type == AgentType.UHD_SCHEDULER:
                agent = UHDSchedulerAgent(agent_id, profile, self.controller_ref)
            else:
                raise ValueError(f"No agent class for type: {agent_type}")
            
            self.agents[agent_id] = agent
            logger.info(f"Created agent {agent_id} of type {agent_type.value}")
            
            return agent
    
    def get_agent(self, agent_id: str) -> Optional[BaseAgent]:
        """Get an agent by ID."""
        return self.agents.get(agent_id)
    
    def distribute_task(self, task: AgentTask, target_agents: Optional[List[str]] = None) -> List[Tuple[str, AgentResult]]:
        """Distribute a task to appropriate agents."""
        results = []
        
        if target_agents is None:
            # Auto-distribute based on task type
            target_agents = self._auto_distribute_task(task)
        
        for agent_id in target_agents:
            agent = self.get_agent(agent_id)
            if agent:
                try:
                    result = agent.submit_task(task)
                    results.append((agent_id, result))
                except Exception as e:
                    logger.error(f"Failed to submit task to agent {agent_id}: {e}")
        
        return results
    
    def _auto_distribute_task(self, task: AgentTask) -> List[str]:
        """Auto-distribute task to appropriate agents based on task type."""
        eligible_agents = []
        
        # Map task types to agent types
        task_to_agent_map = {
            "gpu_resource_allocation": [AgentType.GPU_BOTTLENECK_OPTIMIZER, AgentType.PERFORMANCE_BOOST],
            "memory_allocation": [AgentType.MEMORY_OPTIMIZER, AgentType.COHERENCE_AGENT],
            "memory_trading": [AgentType.TRADING_AGENT],
            "safety_check": [AgentType.SAFETY_MONITOR],
            "thermal_management": [AgentType.THERMAL_MANAGEMENT, AgentType.POWER_EFFICIENCY],
            "performance_optimization": [AgentType.PERFORMANCE_BOOST, AgentType.GPU_BOTTLENECK_OPTIMIZER],
            "coherence_optimization": [AgentType.COHERENCE_AGENT, AgentType.MEMORY_OPTIMIZER]
        }
        
        target_types = task_to_agent_map.get(task.task_type, [])
        
        # Find agents of matching types
        for agent_id, agent in self.agents.items():
            if agent.profile.agent_type in target_types:
                eligible_agents.append(agent_id)
        
        return eligible_agents
    
    def process_messages(self):
        """Process coordination messages between agents."""
        messages_to_process = []
        
        # Get all pending messages
        while not self.message_queue.empty():
            try:
                message = self.message_queue.get_nowait()
                messages_to_process.append(message)
            except queue.Empty:
                break
        
        # Process each message
        for message in messages_to_process:
            self._handle_coordination_message(message)
    
    def _handle_coordination_message(self, message: Dict[str, Any]):
        """Handle a coordination message."""
        rule = self._find_matching_rule(message)
        if not rule:
            return
        
        action = rule["action"]
        target_agents = rule["target_agents"]
        template = rule["message_template"]
        
        # Resolve target agents
        if target_agents == ["all_high_risk"]:
            target_agents = [aid for aid, agent in self.agents.items() 
                           if agent.profile.priority_weight > 0.8]
        elif target_agents == ["all"]:
            target_agents = list(self.agents.keys())
        
        # Execute action
        if action == "notify_all":
            for agent_id in target_agents:
                if agent_id in self.agents:
                    notification_task = AgentTask(
                        task_id=f"NOTIFY_{uuid.uuid4().hex[:8]}",
                        agent_type=self.agents[agent_id].profile.agent_type,
                        task_type="notification",
                        data={"message": template, "origin": message.get("origin")},
                        priority=TaskPriority.HIGH
                    )
                    self.agents[agent_id].submit_task(notification_task)
        
        elif action == "terminate":
            for agent_id in target_agents:
                if agent_id in self.agents:
                    # In real system, this would perform safe termination
                    logger.warning(f"Terminating agent {agent_id} due to safety rule")
        
        elif action == "assist":
            # Forward the message to assist target agents
            for agent_id in target_agents:
                if agent_id in self.agents:
                    assist_task = AgentTask(
                        task_id=f"ASSIST_{uuid.uuid4().hex[:8]}",
                        agent_type=self.agents[agent_id].profile.agent_type,
                        task_type="assistance_request",
                        data={"request": template, "from_agent": message.get("origin")},
                        priority=TaskPriority.NORMAL
                    )
                    self.agents[agent_id].submit_task(assist_task)
    
    def _find_matching_rule(self, message: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Find a matching coordination rule for a message."""
        for rule in self.coordination_rules:
            condition = rule["condition"]
            
            if (condition.get("agent_type") == message.get("sender_type") and
                condition.get("signal") == message.get("signal_type")):
                return rule
        
        return None
    
    def get_overall_status(self) -> Dict[str, Any]:
        """Get overall status of all agents."""
        status = {
            'total_agents': len(self.agents),
            'agent_statuses': {},
            'coordination_rules': len(self.coordination_rules),
            'pending_messages': self.message_queue.qsize(),
            'timestamp': time.time()
        }
        
        for agent_id, agent in self.agents.items():
            status['agent_statuses'][agent_id] = agent.get_status()
        
        return status
    
    def start_coordination_cycle(self):
        """Start coordination processing cycle."""
        self.running = True
        
        def coordination_worker():
            while self.running:
                self.process_messages()
                time.sleep(0.01)  # 10ms intervals
        
        self.coordination_thread = threading.Thread(target=coordination_worker, daemon=True)
        self.coordination_thread.start()
    
    def stop_coordination(self):
        """Stop coordination processing."""
        self.running = False
        if hasattr(self, 'coordination_thread'):
            self.coordination_thread.join(timeout=1.0)


class UHDSchedulerAgent(BaseAgent):
    """Specialized agent for UHD coprocessor scheduling."""
    
    def _execute_task(self, task: AgentTask) -> Any:
        """Execute UHD scheduling task."""
        if task.task_type == "schedule_uhd_task":
            return self._schedule_uhd_task(task)
        elif task.task_type == "manage_uhd_load":
            return self._manage_uhd_load(task)
        elif task.task_type == "optimize_uhd_coherence":
            return self._optimize_uhd_coherence(task)
        else:
            raise ValueError(f"Unknown UHD task type: {task.task_type}")
    
    def _schedule_uhd_task(self, task: AgentTask) -> Any:
        """Schedule a task for UHD coprocessor."""
        integration = self.controller_ref().integration
        gpu_manager = integration.gpu_manager
        
        # Check UHD coprocessor availability
        if not gpu_manager.uhd_coprocessor or not gpu_manager.uhd_coprocessor.is_available():
            return {"status": "uhd_not_available", "alternative_gpu": 1}
        
        task_data = task.data
        task_type = task_data.get('task_type', 'general_compute')
        
        # Determine if task is suitable for UHD coprocessor
        is_suitable = self._is_suitable_for_uhd(task_data)
        
        if is_suitable:
            # Submit to UHD coprocessor
            uhd_task = PipelineTask(
                id=f"UHD_TASK_{task.task_id}",
                task_type=TaskType.COPROCESSOR_OPTIMIZED,
                data=task_data.get('kernel_data', {}),
                priority=task.priority.value
            )
            
            success = gpu_manager.uhd_coprocessor.submit_compute_task(uhd_task)
            result = {
                'scheduled_to': 'uhd_coprocessor' if success else 'discrete_gpu',
                'success': success,
                'task_id': uhd_task.id if success else task.task_id,
                'estimated_completion': time.time() + (task_data.get('estimated_duration', 0.01))
            }
        else:
            result = {
                'scheduled_to': 'discrete_gpu',
                'success': True,
                'reason': 'task_not_suitable_for_uhd',
                'estimated_completion': time.time() + (task_data.get('estimated_duration', 0.02))
            }
        
        return result
    
    def _is_suitable_for_uhd(self, task_data: Dict) -> bool:
        """Determine if a task is suitable for UHD coprocessor."""
        compute_intensity = task_data.get('compute_intensity', 'medium')
        memory_requirements = task_data.get('memory_mb', 128)
        thermal_sensitivity = task_data.get('thermal_sensitivity', 'low')
        
        # UHD is good for:
        # - Medium to low compute intensity tasks
        # - Low to moderate memory requirements (< 256MB)
        # - Tasks that help with thermal management
        return (
            compute_intensity in ['low', 'medium'] and
            memory_requirements <= 256 and
            thermal_sensitivity in ['low', 'medium']
        )
    
    def _manage_uhd_load(self, task: AgentTask) -> Any:
        """Manage UHD coprocessor load."""
        integration = self.controller_ref().integration
        gpu_manager = integration.gpu_manager
        
        if not gpu_manager.uhd_coprocessor:
            return {"status": "uhd_not_available"}
        
        uhd_stats = gpu_manager.uhd_coprocessor.get_performance_stats()
        
        load_management = {
            'current_load': uhd_stats.get('utilization', 0.0),
            'temperature': uhd_stats.get('temperature', 60),
            'active_tasks': uhd_stats.get('active_tasks', 0),
            'performance_score': uhd_stats.get('performance_score', 0.8),
            'recommendations': []
        }
        
        # Generate recommendations based on current state
        if load_management['current_load'] > 0.85:
            load_management['recommendations'].append({
                'action': 'reduce_uhd_load',
                'priority': 'high',
                'reason': 'overloaded_coprocessor'
            })
        elif load_management['current_load'] < 0.2:
            load_management['recommendations'].append({
                'action': 'increase_uhd_utilization',
                'priority': 'medium',
                'reason': 'underutilized_coprocessor'
            })
        
        if load_management['temperature'] > 75:
            load_management['recommendations'].append({
                'action': 'throttle_uhd',
                'priority': 'high',
                'reason': 'thermal_constraint'
            })
        
        return load_management
    
    def _optimize_uhd_coherence(self, task: AgentTask) -> Any:
        """Optimize coherence for UHD coprocessor."""
        integration = self.controller_ref().integration
        coherence_manager = integration.coherence_manager
        
        # Get coherence statistics for UHD coprocessor (GPU 0 typically)
        stats = coherence_manager.protocol.get_coherence_stats_for_gpu(0)
        
        optimization_recommendations = []
        
        if stats.get('cache_miss_rate', 1.0) > 0.1:
            optimization_recommendations.append({
                'recommendation': 'improve_uhd_cache_behavior',
                'type': 'prefetching',
                'parameters': {'prefetch_distance': 8},
                'expected_improvement': 0.15
            })
        
        if stats.get('latency_average_us', 1000.0) > 500:  # High latency
            optimization_recommendations.append({
                'recommendation': 'optimize_uhd_memory_access',
                'type': 'access_pattern',
                'parameters': {'access_pattern': 'sequential'},
                'expected_improvement': 0.10
            })
        
        return {
            'current_stats': stats,
            'optimization_recommendations': optimization_recommendations,
            'gpu_id': 0  # UHD coprocessor
        }


class EffectiveAgentSystem:
    """Main system for effective agent implementation."""
    
    def __init__(self, framework_config):
        self.config = framework_config
        self.framework = None
        self.agent_coordinator = None
        self.start_time = None
        
    def initialize(self):
        """Initialize the effective agent system."""
        logger.info("Initializing Effective Agent System...")
        
        # Initialize main framework
        self.framework = GAMESAGPUFramework(self.config)
        self.framework.initialize()
        
        # Create weak reference to controller for agents
        controller_ref = weakref.ref(self.framework.controller)
        
        # Initialize agent coordinator
        self.agent_coordinator = AgentCoordinator(controller_ref)
        
        # Create default agents based on configuration
        self._create_default_agents()
        
        # Start coordination
        self.agent_coordinator.start_coordination_cycle()
        
        self.start_time = time.time()
        logger.info("Effective Agent System initialized successfully")
    
    def _create_default_agents(self):
        """Create default agents based on system configuration."""
        required_agents = []
        
        # Always create essential monitoring agents
        required_agents.extend([
            AgentType.SAFETY_MONITOR,
            AgentType.THERMAL_MANAGEMENT
        ])
        
        # Add optimization agents based on configuration
        if self.config.enable_gpu_integration:
            required_agents.extend([
                AgentType.GPU_BOTTLENECK_OPTIMIZER,
                AgentType.PERFORMANCE_BOOST
            ])
        
        if self.config.enable_memory_coherence:
            required_agents.extend([
                AgentType.MEMORY_OPTIMIZER,
                AgentType.COHERENCE_AGENT
            ])
        
        if self.config.enable_cross_forex_trading:
            required_agents.append(AgentType.TRADING_AGENT)
        
        if self.config.enable_uhd_coprocessor:
            required_agents.append(AgentType.UHD_SCHEDULER)
        
        # Create all required agents
        for agent_type in required_agents:
            try:
                agent = self.agent_coordinator.create_agent(agent_type)
                logger.info(f"Created agent: {agent.agent_id} ({agent_type.value})")
            except Exception as e:
                logger.error(f"Failed to create agent {agent_type}: {e}")
    
    def run_demonstration(self, duration: int = 30):
        """Run demonstration of effective agent system."""
        logger.info(f"Starting effective agent demonstration for {duration} seconds...")
        
        start_time = time.time()
        
        while (time.time() - start_time) < duration:
            # Generate simulation telemetry
            telemetry = self.framework._generate_simulation_telemetry()
            
            # Process through GAMESA controller
            signals = self.framework._generate_simulation_signals()
            controller_results = self.framework.controller.process_cycle(telemetry, signals)
            
            # Create tasks for agents based on controller results
            for request in controller_results.get('allocation_requests', []):
                agent_task = AgentTask(
                    task_id=f"AGENT_TASK_{uuid.uuid4().hex[:8]}",
                    agent_type=AgentType.GPU_BOTTLENECK_OPTIMIZER,
                    task_type="gpu_resource_allocation",
                    data={
                        'request': request,
                        'telemetry': telemetry
                    },
                    priority=TaskPriority.NORMAL
                )
                
                # Distribute to appropriate agents
                self.agent_coordinator.distribute_task(agent_task)
            
            # Periodic coordination
            self.agent_coordinator.process_messages()
            
            # Log status periodically
            if int(time.time() - start_time) % 5 == 0:
                status = self.agent_coordinator.get_overall_status()
                logger.info(f"Agent System Status: {status['total_agents']} agents active")
            
            # Brief pause to allow processing
            time.sleep(0.05)
        
        logger.info("Agent demonstration completed")
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get complete system status."""
        framework_status = self.framework.get_status() if self.framework else {}
        agent_status = self.agent_coordinator.get_overall_status() if self.agent_coordinator else {}
        
        return {
            'framework': framework_status,
            'agents': agent_status,
            'system_uptime': time.time() - self.start_time if self.start_time else 0,
            'timestamp': time.time()
        }
    
    def shutdown(self):
        """Shutdown the agent system."""
        logger.info("Shutting down Effective Agent System...")
        
        if self.agent_coordinator:
            self.agent_coordinator.stop_coordination()
        
        if self.framework:
            self.framework.shutdown()
        
        logger.info("Effective Agent System shutdown complete")


def main():
    """Main entry point for the effective agent system."""
    parser = argparse.ArgumentParser(description='GAMESA/KrystalStack Effective Agent System')
    parser.add_argument('--mode', type=str, default='demonstration',
                       choices=['demonstration', 'benchmark', 'integration', 'production', 'debug'],
                       help='Operating mode')
    parser.add_argument('--duration', type=int, default=60,
                       help='Duration for demonstration mode (seconds)')
    parser.add_argument('--enable-gpu', action='store_true',
                       help='Enable GPU integration')
    parser.add_argument('--enable-memory', action='store_true',
                       help='Enable memory coherence')
    parser.add_argument('--enable-trading', action='store_true',
                       help='Enable cross-forex trading')
    parser.add_argument('--enable-uhd', action='store_true',
                       help='Enable UHD coprocessor')
    
    args = parser.parse_args()
    
    # Create configuration
    config = FrameworkConfig()
    config.mode = FrameworkMode(args.mode)
    config.enable_gpu_integration = args.enable_gpu or True  # Default to true
    config.enable_memory_coherence = args.enable_memory or True
    config.enable_cross_forex_trading = args.enable_trading or True
    config.enable_uhd_coprocessor = args.enable_uhd or True
    config.log_level = "INFO"
    
    print("GAMESA/KrystalStack Effective Agent System")
    print("=" * 50)
    print(f"Mode: {config.mode.value}")
    print(f"GPU Integration: {'ENABLED' if config.enable_gpu_integration else 'DISABLED'}")
    print(f"Memory Coherence: {'ENABLED' if config.enable_memory_coherence else 'DISABLED'}")
    print(f"Cross-Forex Trading: {'ENABLED' if config.enable_cross_forex_trading else 'DISABLED'}")
    print(f"UHD Coprocessor: {'ENABLED' if config.enable_uhd_coprocessor else 'DISABLED'}")
    print(f"Duration: {args.duration}s" if config.mode == FrameworkMode.DEMONSTRATION else "")
    print("=" * 50)
    
    # Initialize and run system
    agent_system = EffectiveAgentSystem(config)
    
    try:
        agent_system.initialize()
        
        if config.mode == FrameworkMode.DEMONSTRATION:
            agent_system.run_demonstration(args.duration)
        else:
            # For other modes, run until interrupted
            print(f"Running in {config.mode.value} mode...")
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                print("\nInterrupted by user")
        
        # Show final status
        final_status = agent_system.get_system_status()
        print(f"\nFinal System Status:")
        print(f"  Agents Active: {final_status['agents']['total_agents']}")
        print(f"  Framework Status: {final_status['framework'].get('status', 'unknown')}")
        print(f"  Uptime: {final_status['system_uptime']:.1f}s")
        
    except Exception as e:
        logger.error(f"System error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        agent_system.shutdown()


if __name__ == "__main__":
    main()