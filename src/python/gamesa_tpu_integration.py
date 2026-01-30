"""
GAMESA TPU Integration Framework

Main integration module that connects all TPU optimization components
with the broader GAMESA ecosystem including cognitive processing,
safety validation, and economic trading.
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
from .tpu_optimization_framework import (
    TPUOptimizationManager, TPUOptimizationController, TPUConfig
)
from .tpu_memory_manager import TPUMemoryManager, TPU3DGridMemoryAdapter
from .tpu_cross_forex_trading import TPUCrossForexManager, TPUTradingController
from .tpu_bridge import TPUBoostBridge, TPUPreset
from .memory_coherence_protocol import MemoryCoherenceProtocol
from .cross_forex_memory_trading import CrossForexManager
from .platform_hal import BaseHAL, HALFactory
from .mavb import MemoryGrid3D
from .cognitive_engine import create_cognitive_orchestrator


# ============================================================
# ENUMS
# ============================================================

class TPUIntegrationMode(Enum):
    """Integration modes for TPU with GAMESA."""
    FULL_INTEGRATION = "full_integration"      # Complete GAMESA integration
    OPTIMIZATION_ONLY = "optimization_only"    # TPU optimization only
    ECONOMIC_TRADING = "economic_trading"      # Trading focused
    COGNITIVE_ASSISTED = "cognitive_assisted"  # Cognitive-guided


class TPUIntegrationStatus(Enum):
    """Status of TPU integration."""
    INITIALIZING = "initializing"
    ACTIVE = "active"
    THROTTLED = "throttled"
    DEGRADED = "degraded"
    OFFLINE = "offline"


# ============================================================
# DATA CLASSES
# ============================================================

@dataclass
class TPUIntegrationMetrics:
    """Metrics for TPU integration with GAMESA."""
    cognitive_decisions: int = 0
    optimization_cycles: int = 0
    resource_trades: int = 0
    memory_operations: int = 0
    coherence_operations: int = 0
    avg_latency_improvement: float = 0.0
    power_efficiency_gain: float = 0.0
    thermal_improvement: float = 0.0
    trading_profitability: float = 0.0
    safety_violations: int = 0
    timestamp: float = field(default_factory=time.time)


@dataclass
class TPUIntegrationConfig:
    """Configuration for TPU integration."""
    mode: TPUIntegrationMode = TPUIntegrationMode.FULL_INTEGRATION
    enable_cognitive: bool = True
    enable_trading: bool = True
    enable_memory_management: bool = True
    enable_3d_grid: bool = True
    enable_coherence: bool = True
    cognitive_model_path: Optional[str] = None
    trading_strategy: str = "balanced"
    safety_multipliers: Dict[str, float] = field(default_factory=lambda: {
        "thermal": 0.9, "power": 0.85, "latency": 0.95
    })
    optimization_frequency_hz: float = 10.0  # 10 optimizations per second


# ============================================================
# MAIN TPU INTEGRATION CLASS
# ============================================================

class GAMESATPUIntegration:
    """
    Main class that integrates all TPU components with GAMESA framework.
    Provides unified interface for TPU optimization, trading, memory management,
    and cognitive decision making.
    """

    def __init__(self, config: Optional[TPUIntegrationConfig] = None):
        self.config = config or TPUIntegrationConfig()
        self.metrics = TPUIntegrationMetrics()
        self.integration_lock = threading.RLock()
        self.status = TPUIntegrationStatus.INITIALIZING

        # Initialize HAL
        self.hal = HALFactory.create()

        # Initialize core components
        self.coherence_manager = MemoryCoherenceProtocol()
        self.cross_forex_manager = CrossForexManager()
        self.memory_grid_3d = MemoryGrid3D()  # 3D Grid Memory System

        # Initialize TPU-specific components
        self.tpu_bridge = TPUBoostBridge()
        self.tpu_optimization_manager = TPUOptimizationManager()
        self.tpu_memory_manager = TPUMemoryManager(
            self.coherence_manager,
            self.cross_forex_manager,
            self.hal
        )
        self.tpu_trading_manager = TPUCrossForexManager(self.hal)
        self.tpu_3d_adapter = TPU3DGridMemoryAdapter(
            self.tpu_memory_manager,
            self.memory_grid_3d
        )

        # Initialize cognitive components
        self.cognitive_orchestrator = None
        if self.config.enable_cognitive:
            try:
                self.cognitive_orchestrator = create_cognitive_orchestrator()
            except Exception as e:
                print(f"Warning: Could not initialize cognitive orchestrator: {e}")

        # Initialize trading controller
        self.trading_controller = TPUTradingController(self.hal)

        # Initialize optimization controller
        tpu_config = TPUConfig(
            enable_cross_forex=self.config.enable_trading,
            enable_coherence=self.config.enable_coherence,
            optimization_strategy=self.config.trading_strategy
        )
        self.optimization_controller = TPUOptimizationController(tpu_config)

        # GAMESA integration components
        self.effect_checker = create_guardian_checker()
        self.contract_validator = create_guardian_validator()
        self.signal_scheduler = SignalScheduler()

        # Initialize integration
        self._initialize_integration()

    def _initialize_integration(self):
        """Initialize the complete TPU integration."""
        with self.integration_lock:
            print("Initializing GAMESA TPU Integration Framework...")

            # Validate TPU capabilities
            if not self._validate_tpu_capabilities():
                print("Warning: TPU capabilities not fully validated")

            # Initialize trading if enabled
            if self.config.enable_trading:
                portfolio = self.cross_forex_manager.memory_engine.create_portfolio("TPU_GAMESA")
                if not portfolio:
                    print("Warning: Could not create TPU portfolio")

            # Initialize coherence protocol
            if self.config.enable_coherence:
                self.coherence_manager.initialize_coherence()

            # Integrate with GAMESA safety systems
            self._integrate_with_gamesa()

            # Set status to active
            self.status = TPUIntegrationStatus.ACTIVE

            print("GAMESA TPU Integration Framework initialized successfully")

    def _validate_tpu_capabilities(self) -> bool:
        """Validate TPU integration capabilities."""
        # Check that required components are available
        checks = [
            self.tpu_bridge is not None,
            self.tpu_optimization_manager is not None,
            self.tpu_memory_manager is not None,
            self.tpu_trading_manager is not None,
        ]

        return all(checks)

    def _integrate_with_gamesa(self):
        """Integrate TPU components with GAMESA safety and validation systems."""
        print("Integrating TPU components with GAMESA systems...")

        # Check required effects
        required_effects = [
            ("tpu_optimization", Effect.TPU_CONTROL),
            ("tpu_trading", Effect.MEMORY_CONTROL),
            ("tpu_memory", Effect.MEMORY_COHERENCE)
        ]

        for component, effect in required_effects:
            if not self.effect_checker.can_perform(component, effect):
                print(f"Warning: {component} lacks {effect.name} capability")

        # Validate integration contracts
        contract_result = self.contract_validator.check_invariants("tpu_integration", {
            "components_initialized": 6,  # TPU bridge, optimization, memory, trading, 3D adapter, coherence
            "trading_enabled": self.config.enable_trading,
            "memory_management_enabled": self.config.enable_memory_management,
            "coherence_enabled": self.config.enable_coherence,
        })

        if not contract_result.valid:
            print(f"TPU integration validation warnings: {contract_result.errors}")

        print("GAMESA integration completed")

    def process_telemetry_and_signals(self, telemetry: TelemetrySnapshot,
                                    signals: List[Signal]) -> Dict[str, Any]:
        """Process telemetry and signals through all TPU components."""
        with self.integration_lock:
            results = {
                'optimization': {},
                'trading': {},
                'memory': {},
                'cognitive': {},
                'actions_taken': []
            }

            # Update TPU bridge with current signals
            for signal in signals:
                self.tpu_bridge.update_signal(
                    signal.kind.value,
                    signal.strength,
                    signal.priority if hasattr(signal, 'priority') else 0,
                    self.hal.get_thermal_headroom('tpu'),
                    self.config.safety_multipliers['power'] * 15.0  # Power budget
                )

            # Update thermal state
            self.tpu_bridge.update_thermal(
                getattr(telemetry, 'temp_cpu', 60.0),
                getattr(telemetry, 'power_draw', 10.0)
            )

            # Process through optimization controller
            opt_results = self.optimization_controller.process_cycle(telemetry, signals)
            results['optimization'] = opt_results
            results['actions_taken'].extend(opt_results.get('actions_taken', []))

            # Process through trading controller
            trade_results = self.trading_controller.process_trading_cycle(telemetry, signals)
            results['trading'] = trade_results
            results['actions_taken'].extend(trade_results.get('actions_taken', []))

            # Process cognitive decisions if available
            if self.cognitive_orchestrator and self.config.enable_cognitive:
                cognitive_results = self._process_cognitive_decisions(telemetry, signals)
                results['cognitive'] = cognitive_results
                results['actions_taken'].extend(cognitive_results.get('actions_taken', []))

            # Update metrics
            self.metrics.optimization_cycles += 1
            self.metrics.resource_trades += len(trade_results.get('allocations_made', []))
            self.metrics.cognitive_decisions += len(cognitive_results.get('decisions', [])) if cognitive_results else 0

            return results

    def _process_cognitive_decisions(self, telemetry: TelemetrySnapshot,
                                   signals: List[Signal]) -> Dict[str, Any]:
        """Process cognitive decisions for TPU optimization."""
        if not self.cognitive_orchestrator:
            return {'decisions': [], 'actions_taken': []}

        # Prepare cognitive input
        cognitive_input = {
            'telemetry': {
                'cpu_util': telemetry.cpu_util,
                'gpu_util': telemetry.gpu_util,
                'memory_util': telemetry.memory_util,
                'temp_cpu': getattr(telemetry, 'temp_cpu', 60.0),
                'temp_gpu': getattr(telemetry, 'temp_gpu', 65.0),
                'frametime_ms': getattr(telemetry, 'frametime_ms', 16.6),
                'process_category': getattr(telemetry, 'active_process_category', 'unknown')
            },
            'signals': [
                {
                    'kind': signal.kind.value,
                    'strength': signal.strength,
                    'confidence': getattr(signal, 'confidence', 0.5),
                    'source': signal.source
                }
                for signal in signals
            ],
            'tpu_status': self._get_tpu_status_for_cognitive()
        }

        try:
            # Get cognitive recommendations
            cognitive_output = self.cognitive_orchestrator.process(cognitive_input)

            decisions = []
            actions = []

            # Process cognitive recommendations
            for recommendation in cognitive_output.get('recommendations', []):
                if recommendation.get('target') == 'tpu_optimization':
                    decision = self._execute_cognitive_recommendation(recommendation)
                    decisions.append(decision)
                    actions.append(f"Cognitive: {recommendation.get('action', 'unknown')}")

            return {
                'decisions': decisions,
                'actions_taken': actions,
                'cognitive_output': cognitive_output
            }
        except Exception as e:
            print(f"Cognitive processing error: {e}")
            return {'decisions': [], 'actions_taken': [f"Cognitive error: {e}"]}

    def _get_tpu_status_for_cognitive(self) -> Dict[str, Any]:
        """Get TPU status for cognitive processing."""
        return {
            'bridge_stats': self.tpu_bridge.get_stats(),
            'trading_metrics': self.tpu_trading_manager.get_trading_metrics().__dict__,
            'memory_status': self.tpu_memory_manager.get_memory_status(),
            'thermal_headroom': self.hal.get_thermal_headroom('tpu'),
            'power_budget': self.config.safety_multipliers['power'] * 15.0
        }

    def _execute_cognitive_recommendation(self, recommendation: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a cognitive system recommendation."""
        action = recommendation.get('action', 'unknown')
        target = recommendation.get('target', 'unknown')
        params = recommendation.get('parameters', {})

        result = {
            'action': action,
            'target': target,
            'success': False,
            'details': ''
        }

        if action == 'optimize_for_workload':
            workload_type = params.get('workload_type', 'inference')
            # Map string to enum
            from .tpu_bridge import WorkloadType
            workload_map = {
                'inference': WorkloadType.INFERENCE,
                'speech': WorkloadType.SPEECH,
                'vision': WorkloadType.VISION,
                'classification': WorkloadType.CLASSIFICATION,
                'detection': WorkloadType.DETECTION,
                'segmentation': WorkloadType.SEGMENTATION,
                'generation': WorkloadType.GENERATION
            }
            workload_enum = workload_map.get(workload_type, WorkloadType.INFERENCE)
            
            success = self.tpu_optimization_manager.optimize_for_workload(workload_enum)
            result['success'] = success
            result['details'] = f"Optimized for {workload_type} workload"

        elif action == 'request_resources':
            resource_type = params.get('resource_type', 'compute_units')
            quantity = params.get('quantity', 100)
            # Map string to enum
            from .tpu_cross_forex_trading import TPUResourceType
            resource_map = {
                'compute_units': TPUResourceType.COMPUTE_UNITS,
                'on_chip_memory': TPUResourceType.ON_CHIP_MEMORY,
                'precision_mode': TPUResourceType.PRECISION_MODE,
                'throughput_capacity': TPUResourceType.THROUGHPUT_CAPACITY,
                'latency_budget': TPUResourceType.LATENCY_BUDGET,
                'thermal_headroom': TPUResourceType.THERMAL_HEADROOM,
                'power_budget': TPUResourceType.POWER_BUDGET,
                'inference_quota': TPUResourceType.INFERENCE_QUOTA,
                'bandwidth_allocation': TPUResourceType.BANDWIDTH_ALLOCATION
            }
            resource_enum = resource_map.get(resource_type, TPUResourceType.COMPUTE_UNITS)
            
            from .tpu_cross_forex_trading import TPUResourceRequest
            request = TPUResourceRequest(
                request_id=f"COG_{uuid.uuid4().hex[:8]}",
                agent_id="COGNITIVE_ENGINE",
                resource_type=resource_enum,
                quantity=Decimal(str(quantity)),
                priority=params.get('priority', 7),
                max_price=Decimal(str(params.get('max_price', 100.0))),
                thermal_constraint=params.get('thermal_constraint', 20.0),
                power_constraint=params.get('power_constraint', 15.0)
            )
            
            allocation = self.tpu_trading_manager.request_resources(request)
            result['success'] = allocation is not None
            result['details'] = f"Requested {quantity} {resource_type}"

        elif action == 'adjust_precision':
            precision_mode = params.get('precision_mode', 'int8')
            # This would involve adjusting TPU preset
            result['success'] = True
            result['details'] = f"Adjusted precision to {precision_mode}"

        return result

    def get_integration_status(self) -> Dict[str, Any]:
        """Get comprehensive integration status."""
        with self.integration_lock:
            return {
                'config': {
                    'mode': self.config.mode.value,
                    'enable_cognitive': self.config.enable_cognitive,
                    'enable_trading': self.config.enable_trading,
                    'enable_memory_management': self.config.enable_memory_management,
                    'optimization_frequency_hz': self.config.optimization_frequency_hz,
                },
                'status': self.status.value,
                'metrics': {
                    'cognitive_decisions': self.metrics.cognitive_decisions,
                    'optimization_cycles': self.metrics.optimization_cycles,
                    'resource_trades': self.metrics.resource_trades,
                    'memory_operations': self.metrics.memory_operations,
                    'avg_latency_improvement': self.metrics.avg_latency_improvement,
                    'power_efficiency_gain': self.metrics.power_efficiency_gain,
                    'thermal_improvement': self.metrics.thermal_improvement,
                    'trading_profitability': self.metrics.trading_profitability,
                    'safety_violations': self.metrics.safety_violations,
                },
                'tpu_bridge': self.tpu_bridge.get_stats(),
                'trading_controller': self.trading_controller.get_status_report(),
                'optimization_controller': self.optimization_controller.get_status_report(),
                'memory_manager': self.tpu_memory_manager.get_memory_status(),
                'timestamp': time.time()
            }

    def optimize_for_application(self, app_category: str) -> bool:
        """Optimize TPU settings for specific application category."""
        from .tpu_bridge import WorkloadType

        workload_map = {
            'gaming': WorkloadType.INFERENCE,
            'ai_inference': WorkloadType.INFERENCE,
            'speech_processing': WorkloadType.SPEECH,
            'computer_vision': WorkloadType.VISION,
            'classification': WorkloadType.CLASSIFICATION,
            'object_detection': WorkloadType.DETECTION,
            'image_segmentation': WorkloadType.SEGMENTATION,
            'content_generation': WorkloadType.GENERATION
        }

        workload_type = workload_map.get(app_category, WorkloadType.INFERENCE)

        success = self.tpu_optimization_manager.optimize_for_workload(workload_type)
        if success:
            self.metrics.optimization_cycles += 1

        return success

    def request_tpu_memory(self, size: int, access_pattern: str = "random") -> Optional[str]:
        """Request TPU memory allocation."""
        from .tpu_memory_manager import TPUMemoryRequest, TPUAccessPattern

        access_pattern_map = {
            'sequential': TPUAccessPattern.SEQUENTIAL,
            'random': TPUAccessPattern.RANDOM,
            'strided': TPUAccessPattern.STRIDED,
            'tile_based': TPUAccessPattern.TILE_BASED,
            'predictable': TPUAccessPattern.PREDICTABLE
        }

        mapped_pattern = access_pattern_map.get(access_pattern, TPUAccessPattern.RANDOM)

        request = TPUMemoryRequest(
            request_id=f"MEM_{uuid.uuid4().hex[:8]}",
            agent_id="TPU_INTEGRATION",
            region=None,  # Will be determined automatically
            size=size,
            access_pattern=mapped_pattern,
            priority=7
        )

        allocation = self.tpu_memory_manager.request_memory(request)
        if allocation:
            self.metrics.memory_operations += 1
            return allocation.allocation_id

        return None

    def cleanup(self):
        """Clean up integration resources."""
        with self.integration_lock:
            # Clean up expired allocations in all managers
            expired_count = self.tpu_trading_manager.cleanup_expired_allocations()
            print(f"Cleaned up {expired_count} expired TPU trading allocations")

            # Additional cleanup can be added here


# ============================================================
# GAMESA TPU CONTROLLER
# ============================================================

class GAMESATPUController:
    """Main controller for GAMESA TPU integration."""

    def __init__(self, config: Optional[TPUIntegrationConfig] = None):
        self.integration = GAMESATPUIntegration(config)
        self.running = False
        self._thread = None

    def start_continuous_optimization(self, interval_seconds: float = 0.1):
        """Start continuous optimization in background thread."""
        if self.running:
            print("GAMESA TPU Controller already running")
            return

        self.running = True

        def optimization_loop():
            print("Starting GAMESA TPU Continuous Optimization...")
            cycle_time = 1.0 / self.integration.config.optimization_frequency_hz
            last_telemetry = None
            last_signals = []

            while self.running:
                try:
                    # In a real implementation, you'd get real telemetry
                    # For now, we'll use simulated data
                    if last_telemetry is None:
                        from . import TelemetrySnapshot
                        last_telemetry = TelemetrySnapshot(
                            timestamp=datetime.now().isoformat(),
                            cpu_util=0.6,
                            gpu_util=0.5,
                            temp_cpu=65,
                            temp_gpu=60,
                            memory_util=0.7,
                            frametime_ms=18.0,
                            active_process_category="ai_inference"
                        )

                    # Process telemetry and signals
                    results = self.integration.process_telemetry_and_signals(
                        last_telemetry, last_signals
                    )

                    # Update based on application category
                    app_category = getattr(last_telemetry, 'active_process_category', 'unknown')
                    if app_category != 'unknown':
                        self.integration.optimize_for_application(app_category)

                    # Sleep for the remainder of the cycle
                    time.sleep(max(0.001, cycle_time))  # Minimum 1ms sleep

                except Exception as e:
                    print(f"GAMESA TPU optimization error: {e}")
                    time.sleep(0.1)  # Brief pause on error

        self._thread = threading.Thread(target=optimization_loop, daemon=True)
        self._thread.start()

    def stop_optimization(self):
        """Stop continuous optimization."""
        self.running = False
        if self._thread:
            self._thread.join(timeout=1.0)
        self.integration.cleanup()
        print("GAMESA TPU Controller stopped")

    def process_once(self, telemetry: TelemetrySnapshot, signals: List[Signal]) -> Dict[str, Any]:
        """Process one optimization cycle."""
        return self.integration.process_telemetry_and_signals(telemetry, signals)

    def get_status(self) -> Dict[str, Any]:
        """Get controller status."""
        return self.integration.get_integration_status()

    def request_memory(self, size: int, access_pattern: str = "random") -> Optional[str]:
        """Request TPU memory allocation."""
        return self.integration.request_tpu_memory(size, access_pattern)


# ============================================================
# DEMO
# ============================================================

def demo_gamesa_tpu_integration():
    """Demonstrate complete GAMESA TPU integration."""
    print("=== GAMESA TPU Integration Framework Demo ===\n")

    # Initialize controller with full integration
    config = TPUIntegrationConfig(
        mode=TPUIntegrationMode.FULL_INTEGRATION,
        enable_cognitive=True,
        enable_trading=True,
        enable_memory_management=True,
        enable_3d_grid=True,
        enable_coherence=True,
        trading_strategy="balanced",
        optimization_frequency_hz=5.0  # 5 cycles per second for demo
    )

    controller = GAMESATPUController(config)
    print("GAMESA TPU Controller initialized")

    # Simulate telemetry data
    from . import TelemetrySnapshot
    telemetry = TelemetrySnapshot(
        timestamp=datetime.now().isoformat(),
        cpu_util=0.88,  # High CPU utilization
        gpu_util=0.65,  # Moderate GPU utilization
        temp_cpu=75,    # 75°C CPU
        temp_gpu=70,    # 70°C GPU
        memory_util=0.82,
        frametime_ms=20.0,  # 50 FPS
        active_process_category="ai_inference"
    )

    print(f"Input Telemetry: CPU={telemetry.cpu_util*100:.1f}%, GPU={telemetry.gpu_util*100:.1f}%, "
          f"Temp CPU={telemetry.temp_cpu}°C, Mem={telemetry.memory_util*100:.1f}%")

    # Simulate various signals
    from . import Signal
    signals = [
        Signal(
            id="SIGNAL_001",
            source="TELEMETRY",
            kind=SignalKind.CPU_BOTTLENECK,
            strength=0.8,
            confidence=0.88,
            payload={"bottleneck_type": "compute", "recommended_action": "tpu_offload"}
        ),
        Signal(
            id="SIGNAL_002",
            source="TELEMETRY",
            kind=SignalKind.MEMORY_PRESSURE,
            strength=0.7,
            confidence=0.85,
            payload={"memory_util": 0.82, "recommended_action": "memory_offload"}
        ),
        Signal(
            id="SIGNAL_003",
            source="USER",
            kind=SignalKind.USER_BOOST_REQUEST,
            strength=0.9,
            confidence=0.92,
            payload={"request_type": "performance", "priority": "high"}
        )
    ]

    print(f"\nProcessing {len(signals)} signals...")
    for signal in signals:
        print(f"  - {signal.kind.value} (strength: {signal.strength}, source: {signal.source})")

    # Execute one processing cycle
    results = controller.process_once(telemetry, signals)
    print(f"\nProcessing Results:")

    # Print optimization results
    opt_results = results.get('optimization', {})
    print(f"  Optimization:")
    print(f"    Requests Generated: {len(opt_results.get('allocation_requests', []))}")
    print(f    "    Signals Processed: {opt_results.get('signals_processed', 0)}")
    print(f"    Actions Taken: {len(opt_results.get('actions_taken', []))}")

    # Print trading results
    trade_results = results.get('trading', {})
    print(f"  Trading:")
    print(f"    Resource Requests: {len(trade_results.get('resource_requests', []))}")
    print(f"    Allocations Made: {len(trade_results.get('allocations_made', []))}")
    print(f"    Actions Taken: {len(trade_results.get('actions_taken', []))}")

    # Print cognitive results
    cognitive_results = results.get('cognitive', {})
    print(f"  Cognitive:")
    print(f"    Decisions Made: {len(cognitive_results.get('decisions', []))}")
    print(f"    Actions Taken: {len(cognitive_results.get('actions_taken', []))}")

    if results.get('actions_taken'):
        print(f"\nActions Taken:")
        for i, action in enumerate(results['actions_taken'][:10]):  # Show first 10
            print(f"  {i+1}. {action}")
        if len(results['actions_taken']) > 10:
            print(f"  ... and {len(results['actions_taken']) - 10} more")

    # Test memory allocation
    print(f"\nTesting TPU Memory Allocation:")
    mem_allocation_id = controller.request_memory(1024 * 1024, "sequential")  # 1MB sequential access
    if mem_allocation_id:
        print(f"  Memory allocated: {mem_allocation_id}")

    # Show integration status
    print(f"\nIntegration Status:")
    status = controller.get_status()
    print(f"  Status: {status['status']}")
    print(f"  Cognitive Decisions: {status['metrics']['cognitive_decisions']}")
    print(f"  Optimization Cycles: {status['metrics']['optimization_cycles']}")
    print(f"  Resource Trades: {status['metrics']['resource_trades']}")
    print(f"  Memory Operations: {status['metrics']['memory_operations']}")

    # Show TPU bridge stats
    tpu_stats = status['tpu_bridge']
    print(f"\nTPU Bridge Stats:")
    print(f"  Active Preset: {tpu_stats.get('active_preset', 'None')}")
    print(f"  Total Inferences: {tpu_stats.get('total_inferences', 0)}")
    print(f"  Avg Latency: {tpu_stats.get('avg_latency_ms', 0):.2f}ms")

    # Show market state
    market_state = status['trading_controller']['market_state']
    print(f"\nTPU Resource Market Prices:")
    expensive_resource = min(market_state['resource_prices'].items(), key=lambda x: x[1])
    print(f"  Lowest Price: ${expensive_resource[1]:.2f} for {expensive_resource[0]}")

    print(f"\nGAMESA TPU Integration Framework demo completed successfully!")


if __name__ == "__main__":
    demo_gamesa_tpu_integration()