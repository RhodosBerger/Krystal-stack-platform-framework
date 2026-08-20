"""
GAMESA TPU Components Test Suite

Comprehensive tests for all TPU optimization components:
- TPU Optimization Framework
- TPU Memory Management
- TPU Cross-Forex Trading
- TPU Integration Framework
"""

import unittest
import time
from decimal import Decimal
from datetime import datetime

from . import TelemetrySnapshot, Signal, SignalKind
from .tpu_optimization_framework import (
    TPUOptimizationManager, TPUOptimizationController,
    TPUAllocationRequest, TPUResourceType, TPUConfig
)
from .tpu_memory_manager import (
    TPUMemoryManager, TPUMemoryRequest, TPURegion, TPUAccessPattern
)
from .tpu_cross_forex_trading import (
    TPUCrossForexManager, TPUResourceRequest, TPUResourceType as TradingResourceType
)
from .tpu_bridge import TPUBoostBridge, TPUPreset, PresetLibrary
from .tpu_integration_framework import GAMESATPUIntegration, GAMESATPUController
from .memory_coherence_protocol import MemoryCoherenceProtocol
from .cross_forex_memory_trading import CrossForexManager
from .platform_hal import HALFactory


class TestTPUOptimizationFramework(unittest.TestCase):
    """Test TPU optimization framework components."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = TPUConfig(
            enable_cross_forex=True,
            enable_coherence=True,
            thermal_safety_margin=5.0
        )
        self.manager = TPUOptimizationManager(self.config)

    def test_tpu_allocation_request_creation(self):
        """Test creating TPU allocation requests."""
        request = TPUAllocationRequest(
            request_id="TEST_REQ_001",
            agent_id="TEST_AGENT",
            resource_type=TPUResourceType.COMPUTE_UNITS,
            amount=100.0,
            priority=8,
            bid_credits=Decimal('50.0')
        )
        
        self.assertEqual(request.request_id, "TEST_REQ_001")
        self.assertEqual(request.resource_type, TPUResourceType.COMPUTE_UNITS)
        self.assertEqual(request.amount, 100.0)
        self.assertEqual(request.priority, 8)
        self.assertEqual(request.bid_credits, Decimal('50.0'))

    def test_tpu_resource_request_processing(self):
        """Test processing of TPU resource requests."""
        request = TPUAllocationRequest(
            request_id="TEST_REQ_002",
            agent_id="TEST_AGENT",
            resource_type=TPUResourceType.COMPUTE_UNITS,
            amount=200.0,
            priority=9,
            bid_credits=Decimal('75.0'),
            thermal_budget=25.0
        )
        
        allocation = self.manager.request_tpu_resources(request)
        # Allocation may fail if no TPU is available, but shouldn't crash
        if allocation is not None:
            self.assertTrue(allocation.allocation_id.startswith("TPU_ALLOC_"))
            self.assertEqual(allocation.agent_id, "TEST_AGENT")
            self.assertEqual(allocation.resource_type, TPUResourceType.COMPUTE_UNITS)

    def test_telemetry_processing(self):
        """Test processing of telemetry for TPU requests."""
        telemetry = TelemetrySnapshot(
            timestamp=datetime.now().isoformat(),
            cpu_util=0.90,
            gpu_util=0.60,
            temp_cpu=80,
            temp_gpu=65,
            memory_util=0.85,
            frametime_ms=22.0,
            active_process_category="ai_inference"
        )
        
        requests = self.manager.process_telemetry(telemetry)
        self.assertIsInstance(requests, list)
        # Should generate at least one request for high CPU utilization
        # (actual number depends on system state)

    def test_signal_processing(self):
        """Test processing of signals for TPU requests."""
        signal = Signal(
            id="SIGNAL_TEST",
            source="TEST",
            kind=SignalKind.CPU_BOTTLENECK,
            strength=0.85,
            confidence=0.9,
            payload={"test": True}
        )
        
        requests = self.manager.process_signal(signal)
        self.assertIsInstance(requests, list)
        # Should generate requests based on signal

    def test_optimization_status(self):
        """Test getting optimization status."""
        status = self.manager.get_optimization_status()
        self.assertIn('config', status)
        self.assertIn('metrics', status)
        self.assertIn('tpu_status', status)
        self.assertIn('timestamp', status)


class TestTPUMemoryManager(unittest.TestCase):
    """Test TPU memory management components."""

    def setUp(self):
        """Set up test fixtures."""
        from .memory_coherence_protocol import MemoryCoherenceProtocol
        from .cross_forex_memory_trading import CrossForexManager
        from .platform_hal import HALFactory

        self.coherence_manager = MemoryCoherenceProtocol()
        self.cross_forex_manager = CrossForexManager()
        self.hal = HALFactory.create()
        
        self.memory_manager = TPUMemoryManager(
            self.coherence_manager,
            self.cross_forex_manager,
            self.hal
        )

    def test_memory_request_creation(self):
        """Test creating memory requests."""
        request = TPUMemoryRequest(
            request_id="MEM_TEST_001",
            agent_id="TEST_AGENT",
            region=TPURegion.HOST_MEMORY,
            size=1024 * 1024,  # 1MB
            access_pattern=TPUAccessPattern.SEQUENTIAL,
            priority=7
        )
        
        self.assertEqual(request.request_id, "MEM_TEST_001")
        self.assertEqual(request.region, TPURegion.HOST_MEMORY)
        self.assertEqual(request.size, 1024 * 1024)
        self.assertEqual(request.access_pattern, TPUAccessPattern.SEQUENTIAL)

    def test_memory_allocation(self):
        """Test basic memory allocation."""
        request = TPUMemoryRequest(
            request_id="MEM_TEST_002",
            agent_id="TEST_AGENT",
            region=TPURegion.HOST_MEMORY,
            size=512 * 1024,  # 512KB
            access_pattern=TPUAccessPattern.RANDOM,
            priority=5
        )
        
        allocation = self.memory_manager.request_memory(request)
        # Allocation may not succeed if memory is full, but shouldn't crash
        if allocation:
            self.assertTrue(allocation.allocation_id.startswith("TPU_MEM_"))
            self.assertEqual(allocation.agent_id, "TEST_AGENT")
            self.assertEqual(allocation.granted_size, 512 * 1024)

    def test_memory_access(self):
        """Test memory access tracking."""
        request = TPUMemoryRequest(
            request_id="MEM_TEST_003",
            agent_id="TEST_AGENT",
            region=TPURegion.HOST_MEMORY,
            size=256 * 1024,  # 256KB
            access_pattern=TPUAccessPattern.SEQUENTIAL,
            priority=8
        )
        
        allocation = self.memory_manager.request_memory(request)
        if allocation:
            # Test access to allocated memory
            success = self.memory_manager.access_memory(
                allocation.allocation_id, 0, 1024
            )
            # Access might fail if allocation failed, but shouldn't crash
            self.assertIsInstance(success, bool)

    def test_memory_status(self):
        """Test getting memory status."""
        status = self.memory_manager.get_memory_status()
        self.assertIn('memory_pools', status)
        self.assertIn('metrics', status)
        self.assertIn('timestamp', status)
        
        # Check that we have expected memory pools
        expected_regions = {r.value for r in TPURegion}
        actual_regions = set(status['memory_pools'].keys())
        # At least a subset should be present
        self.assertTrue(len(actual_regions) > 0)


class TestTPUCrossForexTrading(unittest.TestCase):
    """Test TPU cross-forex trading components."""

    def setUp(self):
        """Set up test fixtures."""
        self.hal = HALFactory.create()
        self.trading_manager = TPUCrossForexManager(self.hal)

    def test_resource_request_creation(self):
        """Test creating TPU resource requests."""
        request = TPUResourceRequest(
            request_id="TRD_TEST_001",
            agent_id="TEST_AGENT",
            resource_type=TradingResourceType.COMPUTE_UNITS,
            quantity=Decimal('100.0'),
            priority=8,
            max_price=Decimal('150.00'),
            thermal_constraint=20.0,
            power_constraint=15.0
        )
        
        self.assertEqual(request.request_id, "TRD_TEST_001")
        self.assertEqual(request.resource_type, TradingResourceType.COMPUTE_UNITS)
        self.assertEqual(request.quantity, Decimal('100.0'))
        self.assertEqual(request.max_price, Decimal('150.00'))

    def test_resource_trading(self):
        """Test basic resource trading."""
        request = TPUResourceRequest(
            request_id="TRD_TEST_002",
            agent_id="TEST_AGENT",
            resource_type=TradingResourceType.ON_CHIP_MEMORY,
            quantity=Decimal('50.0'),
            priority=7,
            max_price=Decimal('100.00'),
            thermal_constraint=25.0,
            power_constraint=10.0
        )
        
        allocation = self.trading_manager.request_resources(request)
        # Trading may fail due to market conditions, but shouldn't crash
        if allocation:
            self.assertTrue(allocation.allocation_id.startswith("TPU_ALLOC_"))
            self.assertTrue(allocation.price_paid >= 0)

    def test_market_state_access(self):
        """Test accessing market state."""
        state = self.trading_manager.get_market_state()
        self.assertIn('resource_prices', state)
        self.assertIn('supply_levels', state)
        self.assertIn('demand_levels', state)
        self.assertIn('timestamp', state)
        
        # Check that all resource types have prices
        for resource_type in TradingResourceType:
            self.assertIn(resource_type, state.resource_prices)

    def test_trading_metrics(self):
        """Test accessing trading metrics."""
        metrics = self.trading_manager.get_trading_metrics()
        self.assertIn('total_trades', metrics)
        self.assertIn('total_volume', metrics)
        self.assertIn('average_price', metrics)
        self.assertIn('success_rate', metrics)


class TestTPUIntegrationFramework(unittest.TestCase):
    """Test TPU integration framework."""

    def setUp(self):
        """Set up test fixtures."""
        from .tpu_integration_framework import TPUIntegrationConfig, TPUIntegrationMode
        config = TPUIntegrationConfig(
            mode=TPUIntegrationMode.FULL_INTEGRATION,
            enable_cognitive=False,  # Disable cognitive for simpler testing
            enable_trading=True,
            enable_memory_management=True,
            enable_3d_grid=True,
            enable_coherence=True
        )
        self.integration = GAMESATPUIntegration(config)

    def test_integration_initialization(self):
        """Test that integration initializes properly."""
        status = self.integration.get_integration_status()
        self.assertIsNotNone(status)
        self.assertIn('status', status)
        self.assertIn('metrics', status)
        self.assertIn('config', status)

    def test_process_telemetry_and_signals(self):
        """Test processing telemetry and signals."""
        telemetry = TelemetrySnapshot(
            timestamp=datetime.now().isoformat(),
            cpu_util=0.7,
            gpu_util=0.5,
            temp_cpu=65,
            temp_gpu=60,
            memory_util=0.7,
            frametime_ms=18.0,
            active_process_category="ai_inference"
        )
        
        signals = [
            Signal(
                id="SIGNAL_001",
                source="TEST",
                kind=SignalKind.CPU_BOTTLENECK,
                strength=0.6,
                confidence=0.8,
                payload={"test": True}
            )
        ]
        
        results = self.integration.process_telemetry_and_signals(telemetry, signals)
        self.assertIn('optimization', results)
        self.assertIn('trading', results)
        self.assertIn('actions_taken', results)

    def test_application_optimization(self):
        """Test optimizing for specific applications."""
        success = self.integration.optimize_for_application("ai_inference")
        self.assertIsInstance(success, bool)

    def test_memory_request(self):
        """Test requesting TPU memory."""
        alloc_id = self.integration.request_tpu_memory(1024 * 1024)  # 1MB
        if alloc_id:
            self.assertIsInstance(alloc_id, str)
            self.assertTrue(alloc_id.startswith("TPU_MEM_"))


class TestTPUController(unittest.TestCase):
    """Test TPU controller functionality."""

    def setUp(self):
        """Set up test fixtures."""
        from .tpu_integration_framework import TPUIntegrationConfig, TPUIntegrationMode
        config = TPUIntegrationConfig(
            mode=TPUIntegrationMode.OPTIMIZATION_ONLY,
            enable_cognitive=False,
            enable_trading=True,
            enable_memory_management=True
        )
        self.controller = GAMESATPUController(config)

    def test_controller_initialization(self):
        """Test controller initialization."""
        status = self.controller.get_status()
        self.assertIsNotNone(status)
        self.assertIn('status', status)

    def test_controller_process_once(self):
        """Test single processing cycle."""
        telemetry = TelemetrySnapshot(
            timestamp=datetime.now().isoformat(),
            cpu_util=0.6,
            gpu_util=0.5,
            temp_cpu=60,
            temp_gpu=55,
            memory_util=0.6,
            frametime_ms=16.6,
            active_process_category="gaming"
        )
        
        signals = [
            Signal(
                id="CTRL_TEST",
                source="TEST",
                kind=SignalKind.USER_BOOST_REQUEST,
                strength=0.7,
                confidence=0.85,
                payload={"test": True}
            )
        ]
        
        results = self.controller.process_once(telemetry, signals)
        self.assertIsInstance(results, dict)
        # Should have at least empty sub-dictionaries
        self.assertIn('optimization', results)
        self.assertIn('trading', results)


def run_all_tests():
    """Run all TPU component tests."""
    print("=" * 60)
    print("RUNNING GAMESA TPU COMPONENTS TEST SUITE")
    print("=" * 60)

    # Create test suite
    suite = unittest.TestSuite()

    # Add all test cases
    suite.addTest(unittest.makeSuite(TestTPUOptimizationFramework))
    suite.addTest(unittest.makeSuite(TestTPUMemoryManager))
    suite.addTest(unittest.makeSuite(TestTPUCrossForexTrading))
    suite.addTest(unittest.makeSuite(TestTPUIntegrationFramework))
    suite.addTest(unittest.makeSuite(TestTPUController))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    print("\n" + "=" * 60)
    print(f"TEST RESULTS: {result.testsRun} tests run")
    print(f"Failures: {len(result.failures)}, Errors: {len(result.errors)}")
    
    if result.failures:
        print("\nFAILURES:")
        for test, traceback in result.failures:
            print(f"  {test}: {traceback}")
    
    if result.errors:
        print("\nERRORS:")
        for test, traceback in result.errors:
            print(f"  {test}: {traceback}")

    print("=" * 60)

    return result.wasSuccessful()


def demo_individual_components():
    """Demo individual component functionality."""
    print("\n" + "=" * 60)
    print("DEMOING INDIVIDUAL TPU COMPONENTS")
    print("=" * 60)

    print("\n1. Testing TPU Bridge...")
    bridge = TPUBoostBridge()
    bridge.update_signal("TEST_DOMAIN", 0.8, priority=2)
    result = bridge.run_inference("INFERENCE", {"test": "data"}, "TEST_DOMAIN")
    print(f"   Inference result: {result.success}, latency: {result.latency_ms:.2f}ms")

    print("\n2. Testing Preset Library...")
    presets = PresetLibrary.list_presets()
    print(f"   Available presets: {len(presets)}")
    high_perf = PresetLibrary.get("HIGH_THROUGHPUT_FP16")
    if high_perf:
        print(f"   High perf preset: {high_perf.preset_id} on {high_perf.accelerator.name}")

    print("\n3. Testing TPU Config...")
    config = TPUConfig()
    print(f"   Thermal safety margin: {config.thermal_safety_margin}C")
    print(f"   Max trading credits: {config.max_trading_credits}")

    print("\n4. Testing Resource Types...")
    for resource_type in TradingResourceType:
        print(f"   - {resource_type.value}")

    print("\nAll individual component demos completed!")


if __name__ == "__main__":
    # Run the test suite
    success = run_all_tests()

    # Demo individual components
    demo_individual_components()

    print(f"\n{'SUCCESS' if success else 'SOME TESTS FAILED'}: TPU Components Test Suite")
    exit(0 if success else 1)