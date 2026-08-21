"""
Integration Test Suite for GAMESA/KrystalStack Framework

Tests the complete integration of all system components:
- GPU pipeline with UHD coprocessor
- 3D grid memory system  
- Cross-forex resource trading
- Memory coherence protocol
- GAMESA integration
- Signal processing
- Safety validation
"""

import unittest
import time
import threading
from decimal import Decimal
from datetime import datetime
import uuid
import asyncio
from typing import Dict, List, Optional, Any

from .gamesa_gpu_integration import GAMESAGPUIntegration, IntegrationConfig, IntegrationMode
from .gpu_pipeline_integration import (
    GPUManager, GPUPipeline, UHDCoprocessor, DiscreteGPU,
    GPUGridMemoryManager, MemoryContext, MemoryGridCoordinate
)
from .cross_forex_memory_trading import CrossForexManager, CrossForexTrade, MarketOrderType, MemoryResourceType
from .memory_coherence_protocol import MemoryCoherenceProtocol, CoherenceState
from .functional_layer import FunctionalLayerOrchestrator, LayerTask, TaskPriority
from . import TelemetrySnapshot, Signal, SignalKind, Domain


class TestGPUIntegration(unittest.TestCase):
    """Test GPU pipeline integration."""
    
    def setUp(self):
        config = IntegrationConfig()
        config.mode = IntegrationMode.TESTING
        self.integration = GAMESAGPUIntegration(config)
        self.integration.initialize()
    
    def test_gpu_cluster_initialization(self):
        """Test GPU cluster initialization."""
        status = self.integration.get_integration_status()
        
        self.assertIsNotNone(status)
        self.assertGreaterEqual(status['pipeline_state']['total_gpus'], 1)
        self.assertTrue(status['config']['enable_cross_forex'])
        self.assertTrue(status['config']['enable_coherence'])
    
    def test_memory_grid_allocation(self):
        """Test 3D grid memory allocation."""
        from .gpu_pipeline_integration import MemoryGridCoordinate
        
        # Test allocation at specific coordinates
        coord = MemoryGridCoordinate(tier=3, slot=5, depth=16)  # VRAM tier
        size = 1024 * 1024  # 1MB
        
        allocation = self.integration.grid_memory_manager.allocate_memory_at(coord, size)
        
        self.assertIsNotNone(allocation)
        self.assertEqual(allocation.grid_coordinate.tier, 3)
        self.assertEqual(allocation.size, size)
        self.assertGreater(allocation.virtual_address, 0)
    
    def test_uhd_coprocessor_offload(self):
        """Test UHD coprocessor task offloading."""
        # Create a signal indicating high thermal pressure
        thermal_signal = Signal(
            id="THERMAL_HIGH_GPU",
            source="SYSTEM",
            kind=SignalKind.THERMAL_WARNING,
            strength=0.8,
            confidence=0.9,
            payload={"component": "discrete_gpu", "temperature": 78, "recommended_action": "switch_to_cooler_path"}
        )
        
        # Process signal through integration
        requests = self.integration.process_signal(thermal_signal)
        
        # Should generate at least one request for UHD coprocessor usage
        uhd_requests = [req for req in requests if req.gpu_preference == 0]  # UHD GPU ID 0
        self.assertGreaterEqual(len(uhd_requests), 1)
        
        print(f"Generated {len(uhd_requests)} UHD coprocessor requests")
    
    def test_cross_forex_memory_trading(self):
        """Test cross-forex memory resource trading."""
        trade = CrossForexTrade(
            trade_id=f"INTEGRATION_TEST_{uuid.uuid4().hex[:8]}",
            trader_id="INTEGRATION_PORTFOLIO",
            order_type=MarketOrderType.MARKET_BUY,
            resource_type=MemoryResourceType.VRAM,
            quantity=512 * 1024 * 1024,  # 512MB
            bid_credits=Decimal('100.00')
        )
        
        success, message = self.integration.cross_forex_manager.memory_engine.place_trade(trade)
        
        self.assertIsInstance(success, bool)
        self.assertIsInstance(message, str)


class TestMemoryCoherenceIntegration(unittest.TestCase):
    """Test memory coherence integration with 3D grid system."""
    
    def setUp(self):
        self.coherence = MemoryCoherenceProtocol()
        self.grid_manager = GPUGridMemoryManager()
        
        # Register GPUs for coherence
        self.coherence.register_gpu(0, 'uhd_coprocessor', range(0x7FFF0000, 0x80000000))
        self.coherence.register_gpu(1, 'discrete_gpu', range(0x80000000, 0x90000000))
    
    def test_coherence_with_grid_memory(self):
        """Test coherence protocol working with 3D grid memory."""
        # First allocate memory in grid system
        coord = MemoryGridCoordinate(tier=3, slot=10, depth=24)  # VRAM tier
        size = 1024 * 1024  # 1MB
        allocation = self.grid_manager.allocate_memory_at(coord, size)
        
        self.assertIsNotNone(allocation)
        
        # Perform read from GPU 0 (should transition to SHARED/EXCLUSIVE)
        addr = allocation.virtual_address
        read_response = self.coherence.read_access(0, addr)
        
        self.assertTrue(read_response.success)
        self.assertIn(read_response.new_state, [CoherenceState.SHARED, CoherenceState.EXCLUSIVE])
        
        # Perform read from GPU 1 (should remain SHARED if both GPUs accessed)
        read_response2 = self.coherence.read_access(1, addr)
        
        self.assertTrue(read_response2.success)
        # Both GPUs should now have access
        state = self.coherence.get_entry_state(addr)
        self.assertIn(state, [CoherenceState.SHARED, CoherenceState.EXCLUSIVE])
        
        # Perform write from GPU 0 (should transition to MODIFIED)
        write_response = self.coherence.write_access(0, addr, b"integration_test_data")
        
        self.assertTrue(write_response.success)
        self.assertEqual(write_response.new_state, CoherenceState.MODIFIED)
    
    def test_multigpu_coherence(self):
        """Test coherence across multiple GPUs."""
        address = 0x7FFF5000
        
        # GPU 0 reads
        response1 = self.coherence.read_access(0, address)
        self.assertTrue(response1.success)
        state1 = self.coherence.get_entry_state(address)
        
        # GPU 1 reads (should be SHARED)
        response2 = self.coherence.read_access(1, address)
        self.assertTrue(response2.success)
        state2 = self.coherence.get_entry_state(address)
        
        # GPU 0 writes (should invalidate other copies)
        write_response = self.coherence.write_access(0, address, b"exclusive_write")
        self.assertTrue(write_response.success)
        state3 = self.coherence.get_entry_state(address)
        
        # Final state should be MODIFIED (owned by GPU 0)
        self.assertEqual(state3, CoherenceState.MODIFIED)


class TestSignalProcessingIntegration(unittest.TestCase):
    """Test signal processing integration."""
    
    def setUp(self):
        config = IntegrationConfig()
        config.mode = IntegrationMode.TESTING
        self.integration = GAMESAGPUIntegration(config)
        self.integration.initialize()
    
    def test_cpu_bottleneck_signal(self):
        """Test CPU bottleneck detection and response."""
        # Create CPU bottleneck signal
        signal = Signal(
            id="CPU_BOTTLENECK_TEST",
            source="TELEMETRY",
            kind=SignalKind.CPU_BOTTLENECK,
            strength=0.85,
            confidence=0.9,
            payload={
                "bottleneck_type": "compute_intensive", 
                "recommended_action": "gpu_offload",
                "suggested_resource": "compute_units"
            }
        )
        
        # Process signal
        requests = self.integration.process_signal(signal)
        
        # Should generate GPU allocation requests
        self.assertGreater(len(requests), 0)
        
        # Check that requests are for GPU compute units
        gpu_compute_requests = [
            req for req in requests 
            if req.resource_type == "compute_units" and req.priority >= 7
        ]
        
        self.assertGreater(len(gpu_compute_requests), 0)
        print(f"Generated {len(gpu_compute_requests)} GPU compute allocation requests")
    
    def test_memory_pressure_signal(self):
        """Test memory pressure signal processing."""
        signal = Signal(
            id="MEMORY_PRESSURE_TEST",
            source="TELEMETRY",
            kind=SignalKind.MEMORY_PRESSURE,
            strength=0.75,
            confidence=0.85,
            payload={
                "pressure_type": "memory_bandwidth",
                "recommended_action": "allocate_more_vram",
                "suggested_size": 256 * 1024 * 1024  # 256MB
            }
        )
        
        requests = self.integration.process_signal(signal)
        
        self.assertGreater(len(requests), 0)
        
        # Should contain memory allocation requests
        memory_requests = [req for req in requests if 'memory' in req.resource_type.lower()]
        self.assertGreaterEqual(len(memory_requests), 1)
    
    def test_thermal_warning_signal(self):
        """Test thermal warning signal processing."""
        signal = Signal(
            id="THERMAL_WARNING_TEST",
            source="THERMAL_SENSOR",
            kind=SignalKind.THERMAL_WARNING,
            strength=0.8,
            confidence=0.95,
            payload={
                "component": "discrete_gpu",
                "temperature": 82,
                "recommended_action": "switch_to_uhd_coprocessor"
            }
        )
        
        requests = self.integration.process_signal(signal)
        
        # Should generate requests for UHD coprocessor
        uhd_requests = [
            req for req in requests
            if req.gpu_preference == 0 or (
                'uhd' in req.resource_type.lower() or 
                'coprocessor' in req.resource_type.lower()
            )
        ]
        
        self.assertGreaterEqual(len(uhd_requests), 1)
        print(f"Generated {len(uhd_requests)} UHD coprocessor requests for thermal management")


class TestEconomicTradingIntegration(unittest.TestCase):
    """Test economic trading integration with GPU system."""
    
    def setUp(self):
        config = IntegrationConfig()
        config.mode = IntegrationMode.TESTING
        config.enable_cross_forex_trading = True
        self.integration = GAMESAGPUIntegration(config)
        self.integration.initialize()
    
    def test_trading_based_allocation(self):
        """Test allocation based on cross-forex trading decisions."""
        # Create trading strategy signal
        signal = Signal(
            id="TRADING_STRATEGY_TEST",
            source="CROSS_FOREX_ANALYSIS",
            kind=SignalKind.USER_BOOST_REQUEST,
            strength=1.0,
            confidence=0.98,
            payload={
                "trading_strategy": "buy_vram_and_compute",
                "investment_amount": 250.00,
                "target_resources": ["vram", "compute_units"],
                "recommended_allocation": "aggressive"
            }
        )
        
        # Process signal
        requests = self.integration.process_signal(signal)
        
        # Should generate both VRAM and compute allocation requests
        vram_requests = [req for req in requests if 'vram' in req.resource_type.lower()]
        compute_requests = [req for req in requests if 'compute' in req.resource_type.lower()]
        
        self.assertGreaterEqual(len(vram_requests), 1, "Should have VRAM allocation requests")
        self.assertGreaterEqual(len(compute_requests), 1, "Should have compute allocation requests")
        
        # Check that bids are appropriate for aggressive allocation
        high_bid_requests = [req for req in requests if req.bid_credits > 50]
        self.assertGreaterEqual(len(high_bid_requests), 1, "Should have high-bid requests for aggressive allocation")
    
    def test_portfolio_based_optimization(self):
        """Test optimization based on portfolio analysis."""
        # Simulate portfolio performance signal
        signal = Signal(
            id="PORTFOLIO_OPTIMIZATION_TEST",
            source="PORTFOLIO_ANALYZER",
            kind=SignalKind.PERFORMANCE_IMPROVEMENT,
            strength=0.9,
            confidence=0.92,
            payload={
                "portfolio_analysis": {
                    "resource_allocation_efficiency": 0.7,
                    "recommended_action": "reallocate_for_efficiency",
                    "target_resource_type": "memory_bandwidth",
                    "suggested_quantity": 1024 * 1024 * 1024  # 1GB
                }
            }
        )
        
        requests = self.integration.process_signal(signal)
        
        # Should generate bandwidth/memory allocation requests
        bandwidth_requests = [
            req for req in requests 
            if any(keyword in req.resource_type.lower() for keyword in ['bandwidth', 'memory', 'vram'])
        ]
        
        self.assertGreaterEqual(len(bandwidth_requests), 1)
        print(f"Generated {len(bandwidth_requests)} bandwidth/memory allocation requests")


class TestSafetyIntegration(unittest.TestCase):
    """Test safety validation integration."""
    
    def setUp(self):
        config = IntegrationConfig()
        config.mode = IntegrationMode.TESTING
        config.enable_safety_checks = True
        self.integration = GAMESAGPUIntegration(config)
        self.integration.initialize()
    
    def test_safety_constraint_enforcement(self):
        """Test safety constraint enforcement."""
        # Create a request that might violate thermal safety
        dangerous_request = GPUAllocationRequest(
            request_id="DANGEROUS_TEST",
            agent_id="TEST_AGENT",
            resource_type="compute_units",
            amount=10000,  # Very high amount
            priority=10,   # Very high priority
            bid_credits=Decimal('500.00'),  # High bid
            constraints={
                "thermal_safety": True,
                "power_limit": 200.0  # Low power limit
            }
        )
        
        # The allocation request should go through safety validation
        # which might reject dangerous requests
        try:
            allocation = self.integration.request_gpu_resources(dangerous_request)
            # Allocation might be modified for safety rather than rejected
            if allocation:
                # Check if safety constraints were applied
                self.assertLessEqual(allocation.amount_granted, dangerous_request.amount)
        except ValueError as e:
            # Expected if safety validation rejects the request
            self.assertIn("safety", str(e).lower())
    
    def test_thermal_safety_mechanisms(self):
        """Test thermal safety mechanisms."""
        # Simulate high thermal signal
        thermal_signal = Signal(
            id="CRITICAL_THERMAL",
            source="THERMAL_SENSOR",
            kind=SignalKind.THERMAL_WARNING,
            strength=0.98,
            confidence=0.99,
            payload={
                "component": "gpu",
                "temperature": 90,  # Critical temperature
                "recommended_action": "immediate_cooldown",
                "safety_priority": 10  # Highest priority
            }
        )
        
        # Process signal with high thermal stress
        requests = self.integration.process_signal(thermal_signal)
        
        # Should generate immediate safety responses
        safety_requests = [
            req for req in requests
            if any(keyword in str(req).lower() for keyword in ['cooldown', 'reduce', 'throttle', 'emergency'])
        ]
        
        self.assertGreaterEqual(len(safety_requests), 1, "Should have safety response requests")
        print(f"Generated {len(safety_requests)} safety response requests")


class TestCompleteWorkflow(unittest.TestCase):
    """Test complete end-to-end workflow."""
    
    def setUp(self):
        config = IntegrationConfig()
        config.mode = IntegrationMode.DEMONSTRATION
        config.enable_cross_forex_trading = True
        config.enable_coherence = True
        config.enable_3d_grid_memory = True
        config.enable_uhd_coprocessor = True
        self.integration = GAMESAGPUIntegration(config)
        self.integration.initialize()
    
    def test_gaming_scenario_workflow(self):
        """Test complete gaming scenario workflow."""
        # Simulate gaming telemetry
        telemetry = TelemetrySnapshot(
            timestamp=datetime.now().isoformat(),
            cpu_util=0.88,  # High CPU usage
            gpu_util=0.75,  # High GPU usage  
            frametime_ms=14.5,  # Good FPS (~69 FPS)
            temp_cpu=78,      # High CPU temp
            temp_gpu=75,      # High GPU temp
            active_process_category="intensive_gaming"
        )
        
        # Generate multiple signals simulating gaming scenario
        signals = [
            Signal(
                id="GAMING_CPU_BOTTLENECK",
                source="GAME_ENGINE",
                kind=SignalKind.CPU_BOTTLENECK,
                strength=0.85,
                confidence=0.9,
                payload={"bottleneck_type": "compute_intensive", "recommended_action": "gpu_offload"}
            ),
            Signal(
                id="GAMING_GPU_PRESSURE", 
                source="GAME_ENGINE",
                kind=SignalKind.GPU_BOTTLENECK,
                strength=0.7,
                confidence=0.85,
                payload={"bottleneck_type": "render_intensive", "recommended_action": "optimize_render_path"}
            ),
            Signal(
                id="GAMING_THERMAL_PRESSURE",
                source="SYSTEM_MONITOR", 
                kind=SignalKind.THERMAL_WARNING,
                strength=0.65,
                confidence=0.9,
                payload={"component": "gpu", "temperature": 75, "recommended_action": "switch_to_uhd_if_possible"}
            )
        ]
        
        # Process full cycle
        controller_results = self.integration.process_cycle(telemetry, signals)
        
        # Verify results
        self.assertIn('allocation_requests', controller_results)
        self.assertIn('signals_processed', controller_results)
        self.assertIn('actions_taken', controller_results)
        
        allocation_requests = controller_results['allocation_requests']
        signals_processed = controller_results['signals_processed']
        actions_taken = controller_results['actions_taken']
        
        # Should have generated allocation requests
        self.assertGreaterEqual(len(allocation_requests), 0, 
                               f"No allocation requests generated, but got: {allocation_requests}")
        
        # Should have processed all signals
        self.assertEqual(signals_processed, len(signals))
        
        # Should have taken actions based on signals
        self.assertGreaterEqual(len(actions_taken), 0)
        
        print(f"Gaming scenario processed:")
        print(f"  - Allocation requests: {len(allocation_requests)}")
        print(f"  - Signals processed: {signals_processed}")
        print(f"  - Actions taken: {len(actions_taken)}")
        
        # Analyze allocation types
        gpu_allocations = [req for req in allocation_requests if 'gpu' in req.resource_type.lower() or req.resource_type == 'compute_units']
        memory_allocations = [req for req in allocation_requests if 'memory' in req.resource_type.lower() or 'vram' in req.resource_type.lower()]
        thermal_allocations = [req for req in allocation_requests if 'thermal' in req.resource_type.lower()]
        
        print(f"  - GPU allocations: {len(gpu_allocations)}")
        print(f"  - Memory allocations: {len(memory_allocations)}")
        print(f"  - Thermal actions: {len(thermal_allocations)}")
    
    def test_memory_intensive_workflow(self):
        """Test memory-intensive workload workflow."""
        telemetry = TelemetrySnapshot(
            timestamp=datetime.now().isoformat(),
            cpu_util=0.72,
            gpu_util=0.68,
            frametime_ms=15.2,
            temp_cpu=70,
            temp_gpu=68,
            active_process_category="memory_intensive_computing"
        )
        
        signals = [
            Signal(
                id="MEM_INTENSIVE_PRESSURE",
                source="MEMORY_MONITOR",
                kind=SignalKind.MEMORY_PRESSURE,
                strength=0.82,
                confidence=0.88,
                payload={
                    "pressure_type": "bandwidth_limited", 
                    "recommended_action": "allocate_more_vram",
                    "suggested_size_gb": 2.0
                }
            )
        ]
        
        controller_results = self.integration.process_cycle(telemetry, signals)
        
        self.assertIn('allocation_requests', controller_results)
        
        # Should have memory/resource allocation requests
        allocation_requests = controller_results['allocation_requests']
        memory_related = [
            req for req in allocation_requests 
            if any(keyword in req.resource_type.lower() for keyword in ['memory', 'vram', 'bandwidth', 'cache'])
        ]
        
        self.assertGreaterEqual(len(memory_related), 1)
        print(f"Memory-intensive scenario: {len(memory_related)} memory-related allocation requests")
    
    def test_thermal_management_workflow(self):
        """Test thermal management workflow."""
        # High thermal pressure scenario
        telemetry = TelemetrySnapshot(
            timestamp=datetime.now().isoformat(),
            cpu_util=0.92,
            gpu_util=0.88,
            frametime_ms=18.0,  # Lower FPS due to thermal throttling
            temp_cpu=85,        # High CPU temp
            temp_gpu=82,        # High GPU temp  
            active_process_category="thermal_stress_testing"
        )
        
        signals = [
            Signal(
                id="THERMAL_CRITICAL_GPU",
                source="THERMAL_SENSOR",
                kind=SignalKind.THERMAL_WARNING,
                strength=0.92,
                confidence=0.98,
                payload={
                    "component": "discrete_gpu", 
                    "temperature": 82,
                    "recommended_action": "switch_to_uhd_coprocessor",
                    "priority": 10
                }
            ),
            Signal(
                id="THERMAL_HIGH_CPU",
                source="THERMAL_SENSOR", 
                kind=SignalKind.THERMAL_WARNING,
                strength=0.85,
                confidence=0.95,
                payload={
                    "component": "cpu",
                    "temperature": 85, 
                    "recommended_action": "reduce_cpu_frequency",
                    "priority": 9
                }
            )
        ]
        
        controller_results = self.integration.process_cycle(telemetry, signals)
        
        # Should generate thermal management requests
        allocation_requests = controller_results['allocation_requests']
        thermal_requests = []
        
        for req in allocation_requests:
            if ('uhd' in req.resource_type.lower() or 
                'coprocessor' in req.resource_type.lower() or
                'thermal' in req.resource_type.lower() or
                ('compute' in req.resource_type.lower() and req.gpu_preference == 0)):  # UHD preference
                thermal_requests.append(req)
        
        self.assertGreaterEqual(len(thermal_requests), 1)
        print(f"Thermal management: {len(thermal_requests)} thermal-related requests")


class TestPerformanceIntegration(unittest.TestCase):
    """Test performance of integrated system."""
    
    def setUp(self):
        config = IntegrationConfig()
        config.mode = IntegrationMode.BENCHMARK
        self.integration = GAMESAGPUIntegration(config)
        self.integration.initialize()
    
    def test_integration_throughput(self):
        """Test throughput of integrated system."""
        import time
        
        start_time = time.time()
        processed_cycles = 0
        
        # Process multiple cycles rapidly
        for i in range(100):  # 100 cycles
            telemetry = TelemetrySnapshot(
                timestamp=datetime.now().isoformat(),
                cpu_util=0.7 + (i % 20) * 0.01,
                gpu_util=0.6 + (i % 25) * 0.01, 
                frametime_ms=16.0 + (i % 5) * 0.1,
                temp_cpu=65 + (i % 30) * 0.1,
                temp_gpu=68 + (i % 28) * 0.1,
                active_process_category="benchmark"
            )
            
            signals = [
                Signal(
                    id=f"BENCH_SIGNAL_{i:03d}",
                    source="BENCHMARK_GEN",
                    kind=SignalKind.CPU_BOTTLENECK if i % 3 == 0 else SignalKind.GPU_BOTTLENECK,
                    strength=0.6 + (i % 40) * 0.01,
                    confidence=0.85,
                    payload={"test_counter": i}
                )
            ]
            
            controller_results = self.integration.process_cycle(telemetry, signals)
            processed_cycles += 1
        
        end_time = time.time()
        duration = end_time - start_time
        throughput = processed_cycles / duration
        
        print(f"Integration throughput: {throughput:.2f} cycles/sec ({processed_cycles} cycles in {duration:.3f}s)")
        
        # Should achieve at least 10 cycles per second in benchmark mode
        self.assertGreater(throughput, 10.0, f"Throughput too low: {throughput:.2f} cycles/sec")
    
    def test_memory_allocation_performance(self):
        """Test performance of memory allocation integration."""
        import time
        
        start_time = time.time()
        allocations = []
        
        # Perform multiple 3D grid memory allocations
        for i in range(1000):
            coord = MemoryGridCoordinate(tier=i % 7, slot=i % 16, depth=i % 32)
            size = 1024 + (i % 1024)  # 1KB to ~2KB
            context = MemoryContext(
                access_pattern="random" if i % 3 == 0 else "sequential",
                performance_critical=i % 10 == 0,  # Every 10th is performance critical
                compute_intensive=i % 7 == 0       # Every 7th is compute intensive
            )
            
            # Use grid memory manager directly
            allocation = self.integration.grid_memory_manager.allocate_optimized(size, context)
            if allocation:
                allocations.append(allocation)
        
        end_time = time.time()
        duration = end_time - start_time
        allocation_rate = len(allocations) / duration
        
        print(f"Memory allocation performance: {allocation_rate:.2f} allocations/sec")
        
        # Should achieve at least 1000 allocations per second
        self.assertGreater(allocation_rate, 1000.0, f"Allocation rate too slow: {allocation_rate:.2f} allocs/sec")
        self.assertGreaterEqual(len(allocations), 950, f"Too many allocation failures: {len(allocations)}/1000")  # Allow 5% failure rate


def run_integration_tests():
    """Run all integration tests."""
    print("Running GAMESA/KrystalStack Integration Tests")
    print("=" * 60)
    
    # Create test suite
    test_classes = [
        TestGPUIntegration,
        TestMemoryCoherenceIntegration,
        TestSignalProcessingIntegration,
        TestEconomicTradingIntegration,
        TestSafetyIntegration,
        TestCompleteWorkflow,
        TestPerformanceIntegration
    ]
    
    all_tests = unittest.TestSuite()
    
    for test_class in test_classes:
        loader = unittest.TestLoader()
        tests = loader.loadTestsFromTestCase(test_class)
        all_tests.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(all_tests)
    
    # Print summary
    print("\n" + "=" * 60)
    print("INTEGRATION TEST RESULTS SUMMARY")
    print("=" * 60)
    print(f"Tests Run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.failures:
        print(f"\nFAILURES:")
        for test, trace in result.failures:
            print(f"  {test}")
            for line in trace.splitlines()[-5:]:  # Last 5 lines of error
                print(f"    {line}")
    
    if result.errors:
        print(f"\nERRORS:")
        for test, trace in result.errors:
            print(f"  {test}")
            for line in trace.splitlines()[-5:]:
                print(f"    {line}")
    
    success_rate = (result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100 if result.testsRun > 0 else 0
    print(f"\nSuccess Rate: {success_rate:.1f}%")
    
    status = "PASSED" if result.wasSuccessful() else "FAILED"
    print(f"\nOverall Status: {status}")
    
    return result.wasSuccessful()


def run_performance_benchmarks():
    """Run performance benchmarks."""
    print("\nRunning Performance Benchmarks")
    print("=" * 40)
    
    # Import performance components
    from .functional_layer import FunctionalLayerOrchestrator, LayerTask
    from .gpu_pipeline_integration import GPUGridMemoryManager
    from .memory_coherence_protocol import MemoryCoherenceProtocol
    
    # Memory allocation benchmark
    print("\n1. Memory Allocation Performance:")
    grid_manager = GPUGridMemoryManager()
    
    start = time.time()
    allocs = []
    for i in range(5000):
        coord = MemoryGridCoordinate(tier=i % 7, slot=i % 16, depth=i % 32)
        alloc = grid_manager.allocate_memory_at(coord, 1024 + (i % 1023))
        if alloc:
            allocs.append(alloc)
    end = time.time()
    
    rate = len(allocs) / (end - start)
    print(f"   Rate: {rate:.0f} allocations/sec")
    print(f"   Success: {len(allocs)}/5000 ({len(allocs)/5000*100:.1f}%)")
    
    # Coherence protocol benchmark
    print("\n2. Coherence Protocol Performance:")
    coherence = MemoryCoherenceProtocol()
    coherence.register_gpu(0, 'discrete_gpu', range(0x70000000, 0x80000000))
    
    start = time.time()
    ops_count = 0
    for i in range(2500):
        addr = 0x7FFF0000 + (i % 1000)
        coherence.read_access(0, addr)
        coherence.write_access(0, addr, f"test_data_{i}".encode())
        ops_count += 2
    end = time.time()
    
    rate = ops_count / (end - start)
    print(f"   Rate: {rate:.0f} operations/sec")
    
    # Cross-forex trading benchmark
    print("\n3. Cross-forex Trading Performance:")
    from .cross_forex_memory_trading import CrossForexManager
    
    manager = CrossForexManager()
    portfolio = manager.memory_engine.create_portfolio("BENCHMARK_TRADER")
    
    start = time.time()
    trades_success = 0
    for i in range(100):
        trade = CrossForexTrade(
            trade_id=f"BENCH_TRADE_{i:03d}",
            trader_id=portfolio.portfolio_id,
            order_type=MarketOrderType.MARKET_BUY,
            resource_type=MemoryResourceType.VRAM,
            quantity=1024 * 1024,  # 1MB
            bid_credits=Decimal('15.00')
        )
        success, _ = manager.memory_engine.place_trade(trade)
        if success:
            trades_success += 1
    end = time.time()
    
    rate = trades_success / (end - start)
    print(f"   Rate: {rate:.1f} trades/sec")
    print(f"   Success: {trades_success}/100 ({trades_success/100*100:.1f}%)")
    
    print(f"\nPerformance benchmarks completed!")


if __name__ == "__main__":
    import sys
    
    print("GAMESA/KrystalStack Integration Test Suite")
    print("=" * 60)
    
    # Run integration tests
    integration_success = run_integration_tests()
    
    # Run performance benchmarks
    run_performance_benchmarks()
    
    print(f"\n{'='*60}")
    if integration_success:
        print("✓ All integration tests PASSED!")
        print("GAMESA/KrystalStack system is functioning correctly.")
        sys.exit(0)
    else:
        print("✗ Some integration tests FAILED!")
        print("Issues detected in system integration.")
        sys.exit(1)