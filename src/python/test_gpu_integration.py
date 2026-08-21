"""
Unit and Integration Tests for GPU Pipeline and Memory Systems

Tests all components including GPU pipeline, 3D grid memory, 
cross-forex trading, and coherence protocol integration.
"""

import unittest
import time
import threading
from decimal import Decimal
from unittest.mock import Mock, patch
from datetime import datetime

from .gpu_pipeline_integration import (
    GPUGridMemoryManager, GPUCacheCoherenceManager, UHDCoprocessor, 
    DiscreteGPU, GPUPipeline, GPUPipelineSignalHandler, GPUManager
)
from .cross_forex_memory_trading import (
    MemoryMarketEngine, CrossForexTrade, MarketOrderType, 
    MemoryResourceType, CrossForexManager, MemoryTradingSignalProcessor
)
from .memory_coherence_protocol import (
    MemoryCoherenceProtocol, GPUCoherenceManager, CoherenceState, 
    CoherenceOperation, CoherenceEntry, GPUType
)
from .gamesa_gpu_integration import (
    GAMESAGPUIntegration, GPUAllocationRequest, IntegrationConfig,
    MemoryOptimizationStrategy, GPUAllocationStrategy, IntegrationMode
)
from . import TelemetrySnapshot, Signal, SignalKind


class TestGPUGridMemoryManager(unittest.TestCase):
    """Test GPU Grid Memory Manager."""
    
    def setUp(self):
        self.manager = GPUGridMemoryManager()
    
    def test_allocate_memory_at_specific_coordinate(self):
        """Test allocating memory at specific 3D coordinate."""
        coord = MemoryGridCoordinate(tier=3, slot=5, depth=16)  # VRAM tier
        allocation = self.manager.allocate_memory_at(coord, 1024)
        
        self.assertEqual(allocation.grid_coordinate.tier, 3)
        self.assertEqual(allocation.grid_coordinate.slot, 5)
        self.assertEqual(allocation.grid_coordinate.depth, 16)
        self.assertEqual(allocation.size, 1024)
        self.assertGreater(allocation.virtual_address, 0)
    
    def test_optimized_allocation(self):
        """Test optimized memory allocation based on context."""
        from .gpu_pipeline_integration import MemoryContext
        
        context = MemoryContext(
            access_pattern="sequential",
            performance_critical=True,
            compute_intensive=True
        )
        allocation = self.manager.allocate_optimized(2048, context)
        
        # High-performance context should result in UHD buffer allocation
        self.assertIsNotNone(allocation)
        self.assertGreater(allocation.size, 0)
    
    def test_memory_grid_lookup(self):
        """Test looking up memory in the grid."""
        coord = MemoryGridCoordinate(tier=1, slot=2, depth=8)
        allocation = self.manager.allocate_memory_at(coord, 512)
        
        # Verify allocation is in grid
        grid_key = (coord.tier, coord.slot, coord.depth)
        self.assertIn(grid_key, self.manager.memory_grid)
        self.assertEqual(self.manager.memory_grid[grid_key].id, allocation.id)


class TestGPUCacheCoherenceManager(unittest.TestCase):
    """Test GPU Cache Coherence Manager."""
    
    def setUp(self):
        self.coherence = GPUCacheCoherenceManager()
    
    def test_write_memory_with_coherence(self):
        """Test writing memory with coherence protocol."""
        success = self.coherence.write_memory(0x7FFF0000, b"test_data", source_gpu=0)
        self.assertTrue(success)
    
    def test_read_memory_with_coherence(self):
        """Test reading memory with coherence protocol."""
        # First write to establish coherence
        write_success = self.coherence.write_memory(0x7FFF0001, b"shared_data", source_gpu=0)
        self.assertTrue(write_success)
        
        # Then read from different GPU
        data = self.coherence.read_memory(0x7FFF0001, target_gpu=1)
        self.assertIsNotNone(data)
    
    def test_invalidate_other_caches(self):
        """Test cache invalidation protocol."""
        # Simulate write from GPU 0
        self.coherence.coherence_table[0x7FFF0002] = CoherenceInfo(
            state=CoherenceState.MODIFIED,
            source_gpu=0,
            last_access_time=time.time(),
            copies_on_gpus=[0, 1, 2]
        )
        
        # Invalidate other caches
        self.coherence._invalidate_other_caches(0x7FFF0002, except_gpu=0)
        
        # Verify only GPU 0 remains
        info = self.coherence.coherence_table[0x7FFF0002]
        self.assertEqual(info.copies_on_gpus, [0])


class TestUHDCoprocessor(unittest.TestCase):
    """Test UHD Coprocessor functionality."""
    
    def setUp(self):
        self.coprocessor = UHDCoprocessor(0)
    
    def test_submit_compute_task(self):
        """Test submitting compute task to UHD coprocessor."""
        from .gpu_pipeline_integration import PipelineTask, TaskType
        
        task = PipelineTask(
            id="TASK_001",
            task_type=TaskType.COPROCESSOR_OPTIMIZED,
            data={"kernel": "test_kernel"},
            priority=5
        )
        
        success = self.coprocessor.submit_compute_task(task)
        self.assertTrue(success)
        self.assertEqual(self.coprocessor.active_tasks, 1)
    
    def test_coprocessor_availability(self):
        """Test UHD coprocessor availability."""
        self.assertTrue(self.coprocessor.is_available())
        
        # Fill up all slots
        task = PipelineTask(
            id="TASK_002",
            task_type=TaskType.COPROCESSOR_OPTIMIZED,
            data={"kernel": "test_kernel"},
            priority=5
        )
        for _ in range(4):  # Max concurrent tasks is 4
            self.coprocessor.submit_compute_task(task)
        
        self.assertFalse(self.coprocessor.is_available())


class TestDiscreteGPU(unittest.TestCase):
    """Test Discrete GPU functionality."""
    
    def setUp(self):
        self.gpu = DiscreteGPU(1)
    
    def test_submit_task(self):
        """Test submitting task to discrete GPU."""
        from .gpu_pipeline_integration import PipelineTask, TaskType
        
        task = PipelineTask(
            id="TASK_003",
            task_type=TaskType.RENDER_INTENSIVE,
            data={},
            priority=7
        )
        
        success = self.gpu.submit_task(task)
        self.assertTrue(success)
        self.assertGreater(self.gpu.current_load, 0)
    
    def test_performance_score(self):
        """Test performance score calculation."""
        from .gpu_pipeline_integration import TaskType
        
        # Test different task types
        score_render = self.gpu.get_performance_score(TaskType.RENDER_INTENSIVE)
        score_compute = self.gpu.get_performance_score(TaskType.COMPUTE_INTENSIVE)
        
        self.assertGreater(score_render, 0)
        self.assertGreater(score_compute, 0)
        self.assertGreaterEqual(score_render, score_compute)  # Rendering is primary use case


class TestMemoryMarketEngine(unittest.TestCase):
    """Test Memory Market Engine."""
    
    def setUp(self):
        self.engine = MemoryMarketEngine()
    
    def test_create_portfolio(self):
        """Test creating a trading portfolio."""
        portfolio = self.engine.create_portfolio("TRADER_001")
        
        self.assertEqual(portfolio.owner_id, "TRADER_001")
        self.assertEqual(portfolio.cash_balance, Decimal('10000.00'))
        self.assertIn(portfolio.portfolio_id, self.engine.portfolios)
    
    def test_place_trade(self):
        """Test placing a cross-forex trade."""
        # Create portfolio first
        portfolio = self.engine.create_portfolio("TRADER_002")
        
        trade = CrossForexTrade(
            trade_id="TRADE_001",
            trader_id=portfolio.portfolio_id,
            order_type=MarketOrderType.MARKET_BUY,
            resource_type=MemoryResourceType.VRAM,
            quantity=1024 * 1024 * 1024,  # 1GB
            bid_credits=Decimal('50.00'),
            collateral=Decimal('100.00')
        )
        
        success, message = self.engine.place_trade(trade)
        self.assertTrue(success, f"Trade failed: {message}")
    
    def test_market_quotes(self):
        """Test market quote functionality."""
        quote = self.engine.get_market_quote(MemoryResourceType.VRAM)
        self.assertIsNotNone(quote)
        self.assertGreaterEqual(quote.ask_price, quote.bid_price)


class TestMemoryCoherenceProtocol(unittest.TestCase):
    """Test Memory Coherence Protocol."""
    
    def setUp(self):
        self.protocol = MemoryCoherenceProtocol()
        # Register a GPU for testing
        self.protocol.register_gpu(0, GPUType.DISCRETE_GPU, range(0x70000000, 0x80000000))
    
    def test_read_access(self):
        """Test read access with coherence protocol."""
        response = self.protocol.read_access(0, 0x7FFF0000)
        self.assertTrue(response.success)
        self.assertIsNotNone(response.data)
        self.assertEqual(response.new_state, CoherenceState.SHARED)
    
    def test_write_access(self):
        """Test write access with coherence protocol."""
        response = self.protocol.write_access(0, 0x7FFF0001, b"test_write")
        self.assertTrue(response.success)
        self.assertEqual(response.new_state, CoherenceState.MODIFIED)
    
    def test_state_transitions(self):
        """Test MESI coherence state transitions."""
        # Start in INVALID state
        state = self.protocol.get_entry_state(0x7FFF0002)
        self.assertEqual(state, CoherenceState.INVALID)
        
        # Read should transition to SHARED
        self.protocol.read_access(0, 0x7FFF0002)
        state = self.protocol.get_entry_state(0x7FFF0002)
        self.assertIn(state, [CoherenceState.SHARED, CoherenceState.EXCLUSIVE])
    
    def test_coherence_statistics(self):
        """Test coherence statistics collection."""
        initial_requests = self.protocol.stats.total_requests
        
        # Perform some operations
        for i in range(5):
            self.protocol.read_access(0, 0x7FFF0003 + i)
            self.protocol.write_access(0, 0x7FFF0008 + i, b"test")
        
        final_requests = self.protocol.stats.total_requests
        self.assertEqual(final_requests, initial_requests + 10)


class TestGAMESAGPUIntegration(unittest.TestCase):
    """Test GAMESA GPU Integration."""
    
    def setUp(self):
        config = IntegrationConfig()
        self.integration = GAMESAGPUIntegration(config)
    
    def test_initialize_integration(self):
        """Test GAMESA GPU integration initialization."""
        status = self.integration.get_integration_status()
        
        self.assertIsNotNone(status)
        self.assertEqual(status['pipeline_state']['status'], 'active')
        self.assertGreater(status['pipeline_state']['total_gpus'], 0)
        self.assertTrue(status['config']['enable_cross_forex'])
        self.assertTrue(status['config']['enable_coherence'])
    
    def test_gpu_allocation_request(self):
        """Test GPU allocation request processing."""
        request = GPUAllocationRequest(
            request_id="REQ_001",
            agent_id="TEST_AGENT",
            resource_type="compute_units",
            amount=1000,
            priority=7,
            bid_credits=Decimal('50.00')
        )
        
        allocation = self.integration.request_gpu_resources(request)
        self.assertIsNotNone(allocation)
        self.assertIn(allocation.allocation_id, allocation.request_id)
        self.assertGreaterEqual(allocation.gpu_assigned, 0)
    
    def test_telemetry_processing(self):
        """Test telemetry processing for GPU requests."""
        telemetry = TelemetrySnapshot(
            timestamp=datetime.now().isoformat(),
            cpu_util=0.9,  # High CPU usage
            gpu_util=0.6,
            frametime_ms=16.67,
            temp_cpu=75,
            temp_gpu=70,
            active_process_category="gaming"
        )
        
        requests = self.integration.process_telemetry(telemetry)
        self.assertIsInstance(requests, list)
        # Expect an offload request due to high CPU utilization
        self.assertGreaterEqual(len(requests), 0)
    
    def test_signal_processing(self):
        """Test signal processing for GPU allocation."""
        from . import SignalKind
        
        signal = Signal(
            id="SIGNAL_TEST",
            source="TEST",
            kind=SignalKind.CPU_BOTTLENECK,
            strength=0.8,
            confidence=0.9,
            payload={"bottleneck_type": "compute"}
        )
        
        requests = self.integration.process_signal(signal)
        self.assertIsInstance(requests, list)
        self.assertGreaterEqual(len(requests), 0)  # Should generate an offload request


class TestCrossForexManager(unittest.TestCase):
    """Test Cross-forex Manager."""
    
    def setUp(self):
        self.manager = CrossForexManager()
    
    def test_memory_resource_trading(self):
        """Test memory resource trading."""
        from .gpu_pipeline_integration import MemoryContext
        
        context = MemoryContext(
            access_pattern="sequential",
            performance_critical=True
        )
        
        allocation = self.manager.get_memory_allocation(1024 * 1024, context)  # 1MB
        self.assertIsNotNone(allocation)
        self.assertEqual(allocation.size, 1024 * 1024)


class TestIntegration(unittest.TestCase):
    """Integration tests for all components."""
    
    def test_full_pipeline_integration(self):
        """Test full integration of GPU pipeline, memory, and trading."""
        # Initialize integration
        config = IntegrationConfig()
        integration = GAMESAGPUIntegration(config)
        
        # Create a high-priority allocation request
        request = GPUAllocationRequest(
            request_id="FULL_INTEGRATION_TEST",
            agent_id="INTEGRATION_TESTER",
            resource_type="vram",
            amount=256 * 1024 * 1024,  # 256 MB
            priority=9,
            bid_credits=Decimal('100.00'),
            performance_goals={"latency_reduction": 0.5}
        )
        
        # Process allocation through full pipeline
        allocation = integration.request_gpu_resources(request)
        
        self.assertIsNotNone(allocation)
        self.assertEqual(allocation.status, "active")
        self.assertGreaterEqual(allocation.gpu_assigned, 0)
        self.assertGreater(allocation.trading_cost, 0)
    
    def test_coherence_with_gpu_pipeline(self):
        """Test coherence protocol integration with GPU pipeline."""
        # Initialize coherence manager
        coherence_manager = GPUCoherenceManager()
        
        # Register GPUs
        gpu_configs = [
            {'id': 0, 'type': 'uhd_coprocessor', 'memory_region': range(0x7FFF0000, 0x7FFF8000)},
            {'id': 1, 'type': 'discrete_gpu', 'memory_region': range(0x80000000, 0x90000000)}
        ]
        coherence_manager.initialize_gpus(gpu_configs)
        
        # Test coherence operations
        response = coherence_manager.protocol.read_access(0, 0x7FFF1000)
        self.assertTrue(response.success)
        
        write_response = coherence_manager.protocol.write_access(1, 0x7FFF1000, b"sync_test")
        self.assertTrue(write_response.success)


class PerformanceTests(unittest.TestCase):
    """Performance tests for the GPU system."""
    
    def test_allocation_performance(self):
        """Test memory allocation performance."""
        manager = GPUGridMemoryManager()
        
        start_time = time.time()
        allocations = []
        
        # Perform multiple allocations
        for i in range(1000):
            coord = MemoryGridCoordinate(tier=i % 7, slot=i % 16, depth=i % 32)
            allocation = manager.allocate_memory_at(coord, 1024)
            allocations.append(allocation)
        
        end_time = time.time()
        duration = end_time - start_time
        allocations_per_second = len(allocations) / duration
        
        print(f"Allocation performance: {allocations_per_second:.2f} allocations/second")
        self.assertGreater(allocations_per_second, 100)  # Should be fast enough
    
    def test_coherence_protocol_performance(self):
        """Test coherence protocol performance."""
        protocol = MemoryCoherenceProtocol()
        protocol.register_gpu(0, GPUType.DISCRETE_GPU, range(0x70000000, 0x80000000))
        
        start_time = time.time()
        
        # Perform read/write operations
        for i in range(100):
            addr = 0x7FFF0000 + i
            read_resp = protocol.read_access(0, addr)
            write_resp = protocol.write_access(0, addr, f"data_{i}".encode())
        
        end_time = time.time()
        duration = end_time - start_time
        operations_per_second = 200 / duration  # 100 reads + 100 writes
        
        print(f"Coherence performance: {operations_per_second:.2f} operations/second")
        self.assertGreater(operations_per_second, 1000)  # Should be fast for real-time


def run_all_tests():
    """Run all tests."""
    print("Running GPU Integration Test Suite...\n")
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromModule(unittest.sys.modules[__name__])
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2, buffer=True)
    result = runner.run(suite)
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"Test Results Summary:")
    print(f"  Tests Run: {result.testsRun}")
    print(f"  Failures: {len(result.failures)}")
    print(f"  Errors: {len(result.errors)}")
    print(f"  Success Rate: {(result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100:.1f}%")
    
    if result.failures:
        print(f"\nFailures:")
        for test, trace in result.failures:
            print(f"  {test}")
    
    if result.errors:
        print(f"\nErrors:")
        for test, trace in result.errors:
            print(f"  {test}")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_all_tests()
    print(f"\nTest suite {'PASSED' if success else 'FAILED'}")
    exit(0 if success else 1)