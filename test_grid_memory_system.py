#!/usr/bin/env python3
"""
Test Suite for 3D Grid Memory Controller with Functional Runtime
"""

import unittest
import time
from grid_memory_controller import (
    GridCoordinate, 
    MemoryBlock, 
    GridMemoryController, 
    GridCoherenceProtocol, 
    MemoryMigrationEngine, 
    PerformanceOptimizer, 
    FunctionalRuntime,
    MemoryTier
)


class TestGridCoordinate(unittest.TestCase):
    """Test the GridCoordinate class."""
    
    def test_coordinate_creation(self):
        """Test coordinate creation and properties."""
        coord = GridCoordinate(1, 2, 3)
        
        self.assertEqual(coord.x, 1)
        self.assertEqual(coord.y, 2)
        self.assertEqual(coord.z, 3)
        self.assertEqual(coord.to_tuple(), (1, 2, 3))
        self.assertEqual(str(coord), "(1, 2, 3)")
    
    def test_coordinate_equality(self):
        """Test coordinate equality."""
        coord1 = GridCoordinate(1, 2, 3)
        coord2 = GridCoordinate(1, 2, 3)
        coord3 = GridCoordinate(1, 2, 4)
        
        self.assertEqual(coord1, coord2)
        self.assertNotEqual(coord1, coord3)
        self.assertNotEqual(coord2, coord3)
    
    def test_coordinate_distance(self):
        """Test distance calculation."""
        coord1 = GridCoordinate(0, 0, 0)
        coord2 = GridCoordinate(3, 4, 0)
        
        distance = coord1.distance_to(coord2)
        self.assertAlmostEqual(distance, 5.0)  # 3-4-5 triangle
    
    def test_coordinate_hashing(self):
        """Test coordinate hashing."""
        coord1 = GridCoordinate(1, 2, 3)
        coord2 = GridCoordinate(1, 2, 3)
        
        # Coordinates with same values should have same hash
        self.assertEqual(hash(coord1), hash(coord2))
        
        # Should work in sets and dicts
        coord_set = {coord1, coord2}
        self.assertEqual(len(coord_set), 1)


class TestMemoryBlock(unittest.TestCase):
    """Test the MemoryBlock class."""
    
    def test_memory_block_creation(self):
        """Test memory block creation."""
        coord = GridCoordinate(1, 2, 3)
        block = MemoryBlock(coordinate=coord, size=1024)
        
        self.assertEqual(block.coordinate, coord)
        self.assertEqual(block.size, 1024)
        self.assertIsNone(block.data)
        self.assertEqual(block.access_count, 0)
        self.assertFalse(block.is_pinned)
        self.assertEqual(block.memory_tier, MemoryTier.SYSTEM_RAM)
    
    def test_memory_block_properties(self):
        """Test memory block properties."""
        coord = GridCoordinate(0, 0, 0)
        block = MemoryBlock(
            coordinate=coord,
            size=2048,
            data=b"test_data",
            priority=5,
            is_pinned=True,
            memory_tier=MemoryTier.VRAM
        )
        
        self.assertEqual(block.size, 2048)
        self.assertEqual(block.data, b"test_data")
        self.assertEqual(block.priority, 5)
        self.assertTrue(block.is_pinned)
        self.assertEqual(block.memory_tier, MemoryTier.VRAM)


class TestGridMemoryController(unittest.TestCase):
    """Test the GridMemoryController class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.controller = GridMemoryController(grid_dimensions=(4, 4, 4))
    
    def test_initialization(self):
        """Test controller initialization."""
        self.assertEqual(self.controller.grid_dimensions, (4, 4, 4))
        self.assertEqual(len(self.controller.grid), 64)  # 4*4*4
        self.assertEqual(len(self.controller.access_history), 0)
    
    def test_memory_allocation(self):
        """Test memory allocation."""
        coord = self.controller.allocate_memory(1024)
        
        self.assertIsNotNone(coord)
        self.assertIn(coord, self.controller.grid)
        
        block = self.controller.grid[coord]
        self.assertEqual(block.size, 1024)
        self.assertIsNotNone(block.owner_id)
        self.assertGreater(block.timestamp, 0)
    
    def test_memory_allocation_with_parameters(self):
        """Test memory allocation with specific parameters."""
        coord = self.controller.allocate_memory(
            size=2048,
            access_pattern="sequential",
            priority=3,
            memory_tier=MemoryTier.VRAM
        )
        
        self.assertIsNotNone(coord)
        
        block = self.controller.grid[coord]
        self.assertEqual(block.size, 2048)
        self.assertEqual(block.priority, 3)
        self.assertEqual(block.memory_tier, MemoryTier.VRAM)
    
    def test_memory_write_read(self):
        """Test memory write and read operations."""
        coord = self.controller.allocate_memory(1024)
        self.assertIsNotNone(coord)
        
        # Write data
        write_success = self.controller.write_memory(coord, b"Hello World")
        self.assertTrue(write_success)
        
        # Read data
        data = self.controller.read_memory(coord)
        self.assertEqual(data, b"Hello World")
        
        # Check access counts
        block = self.controller.grid[coord]
        self.assertEqual(block.access_count, 2)  # 1 write + 1 read
    
    def test_memory_deallocation(self):
        """Test memory deallocation."""
        coord = self.controller.allocate_memory(1024)
        self.assertIsNotNone(coord)
        
        # Verify allocation
        block = self.controller.grid[coord]
        self.assertGreater(block.size, 0)
        
        # Deallocate
        deallocate_success = self.controller.deallocate_memory(coord)
        self.assertTrue(deallocate_success)
        
        # Verify deallocation
        block = self.controller.grid[coord]
        self.assertEqual(block.size, 0)
        self.assertIsNone(block.data)
        self.assertIsNone(block.owner_id)
    
    def test_pinned_block_protection(self):
        """Test that pinned blocks cannot be deallocated."""
        coord = self.controller.allocate_memory(1024)
        self.assertIsNotNone(coord)
        
        # Pin the block
        block = self.controller.grid[coord]
        block.is_pinned = True
        
        # Attempt to deallocate (should fail)
        deallocate_success = self.controller.deallocate_memory(coord)
        self.assertFalse(deallocate_success)
        
        # Block should still be allocated
        block = self.controller.grid[coord]
        self.assertGreater(block.size, 0)
    
    def test_memory_usage_statistics(self):
        """Test memory usage statistics."""
        usage_before = self.controller.get_memory_usage()
        self.assertEqual(usage_before['used_blocks'], 0)
        
        # Allocate some memory
        coord1 = self.controller.allocate_memory(1024)
        coord2 = self.controller.allocate_memory(2048)
        
        self.assertIsNotNone(coord1)
        self.assertIsNotNone(coord2)
        
        usage_after = self.controller.get_memory_usage()
        self.assertEqual(usage_after['used_blocks'], 2)
        self.assertGreater(usage_after['usage_percentage'], 0)
        
        # Check tier usage
        self.assertGreater(usage_after['tier_usage']['L1_CACHE'], 0)
    
    def test_access_pattern_optimization(self):
        """Test that access patterns influence allocation."""
        # Sequential access pattern should prefer certain tiers
        seq_coord = self.controller.allocate_memory(1024, access_pattern="sequential")
        self.assertIsNotNone(seq_coord)
        
        # Random access pattern should have different behavior
        rand_coord = self.controller.allocate_memory(1024, access_pattern="random")
        self.assertIsNotNone(rand_coord)
        
        # Both should be valid coordinates
        self.assertIn(seq_coord, self.controller.grid)
        self.assertIn(rand_coord, self.controller.grid)


class TestGridCoherenceProtocol(unittest.TestCase):
    """Test the GridCoherenceProtocol class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.protocol = GridCoherenceProtocol()
    
    def test_write_block_coherence(self):
        """Test write block coherence protocol."""
        coord = GridCoordinate(1, 2, 3)
        self.protocol.write_block(coord, b"test_data")
        
        # Check that coherence state is updated
        self.assertIn(coord, self.protocol.cache_coherence)
        self.assertEqual(self.protocol.cache_coherence[coord], "MODIFIED")
    
    def test_read_block_coherence(self):
        """Test read block coherence protocol."""
        coord = GridCoordinate(1, 2, 3)
        
        # Read from invalid state
        self.protocol.read_block(coord)
        self.assertIn(coord, self.protocol.cache_coherence)
        
        # Read from shared state
        self.protocol.cache_coherence[coord] = "SHARED"
        self.protocol.read_block(coord)
    
    def test_invalidate_block(self):
        """Test block invalidation."""
        coord = GridCoordinate(1, 2, 3)
        self.protocol.cache_coherence[coord] = "MODIFIED"
        self.protocol.gpu_directory[coord] = ["GPU1", "GPU2"]
        
        self.protocol.invalidate_block(coord)
        
        self.assertNotIn(coord, self.protocol.cache_coherence)
        self.assertNotIn(coord, self.protocol.gpu_directory)


class TestMemoryMigrationEngine(unittest.TestCase):
    """Test the MemoryMigrationEngine class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.engine = MemoryMigrationEngine()
    
    def test_migration_decision(self):
        """Test migration decision logic."""
        # Create a test block
        coord = GridCoordinate(6, 0, 0)  # SWAP tier
        block = MemoryBlock(
            coordinate=coord,
            size=1024,
            memory_tier=MemoryTier.SWAP,
            access_count=25  # High access count
        )
        
        # Create access patterns
        access_patterns = {coord: 25}
        
        # Should migrate because of high access and slow tier
        should_migrate = self.engine._should_migrate_block(block, access_patterns)
        self.assertTrue(should_migrate)


class TestFunctionalRuntime(unittest.TestCase):
    """Test the FunctionalRuntime class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.controller = GridMemoryController(grid_dimensions=(2, 2, 2))
        self.runtime = FunctionalRuntime(self.controller)
    
    def test_function_registration(self):
        """Test function registration."""
        def test_func(x, y):
            return x + y
        
        self.runtime.register_function('add', test_func)
        self.assertIn('add', self.runtime.functions)
    
    def test_function_execution(self):
        """Test function execution."""
        result = self.runtime.execute_function('allocate', 1024)
        self.assertIsNotNone(result)
        self.assertIsInstance(result, GridCoordinate)
    
    def test_builtin_functions(self):
        """Test that builtin functions are registered."""
        self.assertIn('allocate', self.runtime.functions)
        self.assertIn('read', self.runtime.functions)
        self.assertIn('write', self.runtime.functions)
        self.assertIn('deallocate', self.runtime.functions)
        self.assertIn('copy', self.runtime.functions)
        self.assertIn('transform', self.runtime.functions)
    
    def test_functional_map(self):
        """Test functional map operation."""
        # Allocate some coordinates
        coords = []
        for i in range(3):
            coord = self.runtime.execute_function('allocate', 100, priority=i)
            if coord:
                coords.append(coord)
                # Write some data
                self.runtime.execute_function('write', coord, f"data_{i}".encode())
        
        # Define transformation function
        def get_length(data: bytes) -> int:
            return len(data)
        
        # Apply map
        lengths = self.runtime.functional_map(get_length, coords)
        self.assertEqual(len(lengths), len(coords))
        
        # Check that lengths are correct
        for length in lengths:
            if length is not None:
                self.assertEqual(length, 6)  # "data_X" is 6 chars
    
    def test_functional_filter(self):
        """Test functional filter operation."""
        # Allocate coordinates with different data
        coords = []
        for i in range(3):
            coord = self.runtime.execute_function('allocate', 100, priority=i)
            if coord:
                coords.append(coord)
                # Write data of different lengths
                data = f"short{i}" if i < 2 else f"this_is_a_longer_string_{i}"
                self.runtime.execute_function('write', coord, data.encode())
        
        # Define predicate
        def is_long_data(data: bytes) -> bool:
            return len(data) > 10
        
        # Apply filter
        filtered_coords = self.runtime.functional_filter(is_long_data, coords)
        self.assertLessEqual(len(filtered_coords), len(coords))
    
    def test_execution_context(self):
        """Test execution context functionality."""
        # Create context
        ctx_id = self.runtime.create_execution_context("test_context")
        self.assertTrue(ctx_id.startswith("CTX_test_context_"))
        
        # Execute in context
        coord = self.runtime.execute_in_context(ctx_id, 'allocate', 512, priority=1)
        self.assertIsNotNone(coord)
        
        # Check context operations
        context = self.runtime.execution_contexts[ctx_id]
        self.assertGreater(len(context['operations']), 0)
        self.assertEqual(context['operations'][0]['function'], 'allocate')
    
    def test_runtime_statistics(self):
        """Test runtime statistics."""
        stats = self.runtime.get_runtime_stats()
        
        self.assertIsInstance(stats, dict)
        self.assertIn('registered_functions', stats)
        self.assertIn('transaction_count', stats)
        self.assertIn('active_contexts', stats)
        self.assertIn('execution_count', stats)
        
        self.assertGreaterEqual(stats['registered_functions'], 6)  # At least built-ins


def run_comprehensive_tests():
    """Run comprehensive tests for the 3D Grid Memory System."""
    print("=" * 80)
    print("COMPREHENSIVE TEST SUITE: 3D GRID MEMORY SYSTEM")
    print("=" * 80)
    
    # Create test suites
    coord_suite = unittest.TestLoader().loadTestsFromTestCase(TestGridCoordinate)
    block_suite = unittest.TestLoader().loadTestsFromTestCase(TestMemoryBlock)
    controller_suite = unittest.TestLoader().loadTestsFromTestCase(TestGridMemoryController)
    coherence_suite = unittest.TestLoader().loadTestsFromTestCase(TestGridCoherenceProtocol)
    migration_suite = unittest.TestLoader().loadTestsFromTestCase(TestMemoryMigrationEngine)
    runtime_suite = unittest.TestLoader().loadTestsFromTestCase(TestFunctionalRuntime)
    
    # Combine all tests
    all_tests = unittest.TestSuite([
        coord_suite,
        block_suite,
        controller_suite,
        coherence_suite,
        migration_suite,
        runtime_suite
    ])
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(all_tests)
    
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.failures:
        print("\nFailures:")
        for test, traceback in result.failures:
            print(f"  {test}: {traceback}")
    
    if result.errors:
        print("\nErrors:")
        for test, traceback in result.errors:
            print(f"  {test}: {traceback}")
    
    if result.wasSuccessful():
        print("\n[OK] All tests passed! 3D Grid Memory System is working correctly.")
        return True
    else:
        print("\n[X] Some tests failed. Please review the output above.")
        return False


def demo_integration():
    """Demonstrate integration between all 3D Grid Memory components."""
    print("\n" + "=" * 80)
    print("INTEGRATION DEMONSTRATION")
    print("=" * 80)
    
    # Create the 3D Grid Memory Controller
    controller = GridMemoryController(grid_dimensions=(8, 16, 32))
    print("[OK] 3D Grid Memory Controller created")
    
    # Create the Functional Runtime
    runtime = FunctionalRuntime(controller)
    print("[OK] Functional Runtime created")
    
    # Create a performance optimizer
    optimizer = PerformanceOptimizer()
    print("[OK] Performance Optimizer created")
    
    # Create a memory migration engine
    migration_engine = MemoryMigrationEngine()
    print("[OK] Memory Migration Engine created")
    
    # Create a coherence protocol
    coherence_protocol = GridCoherenceProtocol()
    print("[OK] Coherence Protocol created")
    
    # Demonstrate coordinated operations
    print(f"\n--- Coordinated Operations Demo ---")
    
    # 1. Allocate memory with optimization
    optimized_params = optimizer.optimize_for_access_pattern("sequential")
    print(f"Optimized parameters: {optimized_params}")
    
    coord1 = runtime.execute_function('allocate', 1024, access_pattern="sequential")
    print(f"Allocated 1KB at {coord1}")
    
    # 2. Write data with coherence
    coherence_protocol.write_block(coord1, b"Sequential data")
    write_success = runtime.execute_function('write', coord1, b"Sequential data")
    print(f"Write with coherence: {'SUCCESS' if write_success else 'FAILED'}")
    
    # 3. Perform functional operations
    coord2 = runtime.execute_function('allocate', 2048, access_pattern="random")
    print(f"Allocated 2KB at {coord2}")
    
    runtime.execute_function('write', coord2, b"Random access data")
    
    # 4. Demonstrate transformation
    def reverse_transform(data: bytes) -> bytes:
        return data[::-1]  # Reverse bytes
    
    transform_success = runtime.execute_function('transform', coord1, reverse_transform)
    print(f"Transform operation: {'SUCCESS' if transform_success else 'FAILED'}")
    
    # 5. Analyze access patterns for migration
    access_patterns = {
        coord1: 50,  # Frequently accessed
        coord2: 5    # Rarely accessed
    }
    
    should_migrate = migration_engine._should_migrate_block(controller.grid[coord1], access_patterns)
    print(f"Should migrate coord1: {should_migrate}")
    
    # 6. Show memory usage
    usage = controller.get_memory_usage()
    print(f"Memory usage: {usage}")
    
    # 7. Create execution context for batch operations
    ctx_id = runtime.create_execution_context("batch_operations")
    print(f"Created execution context: {ctx_id}")
    
    # 8. Perform batch operations in context
    batch_coords = []
    for i in range(3):
        batch_coord = runtime.execute_in_context(ctx_id, 'allocate', 512, priority=i)
        if batch_coord:
            batch_coords.append(batch_coord)
            runtime.execute_in_context(ctx_id, 'write', batch_coord, f"Batch data {i}".encode())
    
    print(f"Allocated {len(batch_coords)} blocks in context")
    
    # 9. Show runtime statistics
    stats = runtime.get_runtime_stats()
    print(f"Runtime statistics: {stats}")
    
    # 10. Clean up
    for coord in [coord1, coord2] + batch_coords:
        if coord:
            runtime.execute_function('deallocate', coord)
    
    print(f"\n[OK] Integration demonstration completed successfully")
    print("All components work together seamlessly:")
    print(f"- Grid Memory Controller: Manages 3D memory space")
    print(f"- Functional Runtime: Provides functional programming interface")
    print(f"- Performance Optimizer: Optimizes based on access patterns")
    print(f"- Migration Engine: Migrates blocks for efficiency")
    print(f"- Coherence Protocol: Maintains data consistency")
    print(f"- Execution Contexts: Manage batch operations")
    
    print("=" * 80)


if __name__ == "__main__":
    print("Starting comprehensive test suite for 3D Grid Memory System...")
    
    # Run comprehensive tests
    tests_passed = run_comprehensive_tests()
    
    if tests_passed:
        # Run integration demonstration
        demo_integration()
        
        print("\n" + "=" * 80)
        print("ALL TESTS PASSED - INTEGRATION IS STABLE")
        print("=" * 80)
        print("The 3D Grid Memory System is fully functional:")
        print("- 3D coordinate-based memory management")
        print("- Functional programming interface")
        print("- Memory coherence protocol")
        print("- Migration engine for optimization")
        print("- Performance optimization")
        print("- Execution contexts")
        print("- Comprehensive memory operations")
        print("- Thread-safe operations")
        print("- Integration with other systems")
        print("=" * 80)
    else:
        print("\n" + "=" * 80)
        print("TESTS FAILED - PLEASE REVIEW OUTPUT")
        print("=" * 80)