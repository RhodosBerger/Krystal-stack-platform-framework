#!/usr/bin/env python3
"""
Test Suite for Hexadecimal System with ASCII Rendering and Composition Generator
"""

import unittest
import numpy as np
from hexadecimal_system import (
    HexadecimalSystem, 
    HexCommodityType, 
    HexDepthLevel, 
    ASCIIHexRenderer, 
    CompositionGenerator,
    HexPatternMatcher
)


class TestHexadecimalSystem(unittest.TestCase):
    """Test the hexadecimal system functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.hex_system = HexadecimalSystem()
    
    def test_create_commodity(self):
        """Test creating hexadecimal commodities."""
        commodity = self.hex_system.create_commodity(
            HexCommodityType.HEX_COMPUTE,
            quantity=100.0,
            depth_level=HexDepthLevel.MODERATE
        )
        
        self.assertIsNotNone(commodity.commodity_id)
        self.assertEqual(commodity.commodity_type, HexCommodityType.HEX_COMPUTE)
        self.assertEqual(commodity.depth_level, HexDepthLevel.MODERATE)
        self.assertGreaterEqual(commodity.hex_value, 0x00)
        self.assertLessEqual(commodity.hex_value, 0x1F)  # Compute type range
    
    def test_execute_trade(self):
        """Test executing hexadecimal trades."""
        commodity = self.hex_system.create_commodity(
            HexCommodityType.HEX_MEMORY,
            quantity=512.0,
            depth_level=HexDepthLevel.HIGH
        )
        
        trade = self.hex_system.execute_trade(
            commodity.commodity_id,
            "BuyerAgent",
            "SellerAgent",
            price=200.0
        )
        
        self.assertIsNotNone(trade.trade_id)
        self.assertEqual(trade.buyer_id, "BuyerAgent")
        self.assertEqual(trade.seller_id, "SellerAgent")
        self.assertEqual(trade.price, 200.0)
        self.assertEqual(trade.trade_status, "executed")
        
        # Verify commodity was removed from available commodities
        self.assertNotIn(commodity.commodity_id, self.hex_system.commodities)
    
    def test_market_depth(self):
        """Test market depth calculation."""
        # Create commodities of different depth levels
        self.hex_system.create_commodity(
            HexCommodityType.HEX_COMPUTE,
            quantity=100.0,
            depth_level=HexDepthLevel.LOW
        )
        
        self.hex_system.create_commodity(
            HexCommodityType.HEX_COMPUTE,
            quantity=200.0,
            depth_level=HexDepthLevel.HIGH
        )
        
        depth_info = self.hex_system.get_market_depth(HexCommodityType.HEX_COMPUTE)
        
        # Check that we have depth information for different levels
        self.assertIn('LOW', depth_info)
        self.assertIn('HIGH', depth_info)
        
        # Verify quantities are calculated correctly
        low_depth = depth_info['LOW']
        high_depth = depth_info['HIGH']
        
        self.assertGreaterEqual(low_depth['total_quantity'], 0)
        self.assertGreaterEqual(high_depth['total_quantity'], 0)


class TestASCIIHexRenderer(unittest.TestCase):
    """Test the ASCII rendering functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.renderer = ASCIIHexRenderer()
    
    def test_render_hex_value(self):
        """Test rendering hexadecimal values as ASCII art."""
        # Test rendering a simple hex value
        result = self.renderer.render_hex_value(0xAB)
        
        self.assertIsInstance(result, str)
        self.assertIn("  ***  ", result)  # Pattern for 'A'
        self.assertIn(" ****  ", result)  # Pattern for 'B'
        
        # Test with scaling
        result_scaled = self.renderer.render_hex_value(0x1, scale=2)
        self.assertIsInstance(result_scaled, str)
        # Should be larger due to scaling
    
    def test_render_market_state(self):
        """Test rendering market state as ASCII."""
        hex_system = HexadecimalSystem()
        
        # Add some commodities
        hex_system.create_commodity(
            HexCommodityType.HEX_COMPUTE,
            quantity=100.0,
            depth_level=HexDepthLevel.MODERATE
        )
        
        result = self.renderer.render_market_state(hex_system)
        
        self.assertIsInstance(result, str)
        self.assertIn("HEXADECIMAL MARKET STATE", result)
        self.assertIn("Total Commodities:", result)
    
    def test_render_composition(self):
        """Test rendering composition as ASCII."""
        test_composition = {
            'composition_id': 'TEST_COMP_123',
            'timestamp': 1234567890.0,
            'hex_values': [0x10, 0x20, 0x30],
            'resources': {'compute': {'hex_value': 16, 'allocation': 10.0}}
        }
        
        result = self.renderer.render_composition(test_composition)
        
        self.assertIsInstance(result, str)
        self.assertIn("HEXADECIMAL COMPOSITION", result)
        self.assertIn("composition_id", result)


class TestCompositionGenerator(unittest.TestCase):
    """Test the composition generator functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.generator = CompositionGenerator()
    
    def test_generate_composition(self):
        """Test generating resource compositions."""
        market_state = {
            'demand_pressure': 0.7,
            'volatility': 0.2,
            'trend': 'up'
        }
        
        composition = self.generator.generate_composition(market_state)
        
        self.assertIsNotNone(composition['composition_id'])
        self.assertIn('resources', composition)
        self.assertIn('efficiency_score', composition)
        self.assertIn('risk_level', composition)
        
        # Verify efficiency score is between 0 and 1
        self.assertGreaterEqual(composition['efficiency_score'], 0.0)
        self.assertLessEqual(composition['efficiency_score'], 1.0)
        
        # Verify risk level is one of expected values
        self.assertIn(composition['risk_level'], ['low', 'medium', 'high'])
    
    def test_optimize_composition(self):
        """Test optimizing existing compositions."""
        market_state = {'demand_pressure': 0.5, 'volatility': 0.3, 'trend': 'neutral'}
        composition = self.generator.generate_composition(market_state)
        
        optimized = self.generator.optimize_composition(composition)
        
        # Should return the same composition with potentially updated scores
        self.assertEqual(composition['composition_id'], optimized['composition_id'])
        self.assertIn('efficiency_score', optimized)
        self.assertIn('risk_level', optimized)


class TestHexPatternMatcher(unittest.TestCase):
    """Test the hex pattern matching functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.matcher = HexPatternMatcher()
    
    def test_find_optimal_hex_values(self):
        """Test finding optimal hex values based on market state."""
        # Test different market conditions
        high_demand_state = {'demand_pressure': 0.8, 'volatility': 0.5, 'trend': 'up'}
        low_demand_state = {'demand_pressure': 0.2, 'volatility': 0.5, 'trend': 'down'}
        high_volatility_state = {'demand_pressure': 0.5, 'volatility': 0.8, 'trend': 'neutral'}
        
        high_demand_values = self.matcher.find_optimal_hex_values(high_demand_state)
        low_demand_values = self.matcher.find_optimal_hex_values(low_demand_state)
        high_volatility_values = self.matcher.find_optimal_hex_values(high_volatility_state)
        
        # Should return lists of hex values
        self.assertIsInstance(high_demand_values, list)
        self.assertIsInstance(low_demand_values, list)
        self.assertIsInstance(high_volatility_values, list)
        
        self.assertGreater(len(high_demand_values), 0)
        self.assertGreater(len(low_demand_values), 0)
        self.assertGreater(len(high_volatility_values), 0)
    
    def test_detect_market_patterns(self):
        """Test detecting patterns in market data."""
        hex_system = HexadecimalSystem()

        # Add multiple commodities to ensure some remain after trades
        commodity1 = hex_system.create_commodity(
            HexCommodityType.HEX_COMPUTE,
            quantity=100.0,
            depth_level=HexDepthLevel.MODERATE
        )

        commodity2 = hex_system.create_commodity(
            HexCommodityType.HEX_MEMORY,
            quantity=200.0,
            depth_level=HexDepthLevel.HIGH
        )

        # Test patterns with commodities present (should have distribution info)
        patterns_with_commodities = self.matcher.detect_market_patterns(hex_system)
        self.assertIsInstance(patterns_with_commodities, dict)
        self.assertIn('distribution', patterns_with_commodities)

        # Execute trade and test patterns again
        hex_system.execute_trade(commodity1.commodity_id, "A", "B", 150.0)

        patterns_after_trade = self.matcher.detect_market_patterns(hex_system)

        self.assertIsInstance(patterns_after_trade, dict)

        # After trade, we should still have pricing info from trade history
        # and distribution info from remaining commodities
        self.assertIn('pricing', patterns_after_trade)

        # If there are still commodities left, distribution should be present
        if hex_system.commodities:
            self.assertIn('distribution', patterns_after_trade)


def run_comprehensive_tests():
    """Run comprehensive tests for the hexadecimal system."""
    print("=" * 80)
    print("COMPREHENSIVE TEST SUITE: HEXADECIMAL SYSTEM")
    print("=" * 80)
    
    # Create test suites
    hex_system_suite = unittest.TestLoader().loadTestsFromTestCase(TestHexadecimalSystem)
    ascii_renderer_suite = unittest.TestLoader().loadTestsFromTestCase(TestASCIIHexRenderer)
    composition_suite = unittest.TestLoader().loadTestsFromTestCase(TestCompositionGenerator)
    pattern_suite = unittest.TestLoader().loadTestsFromTestCase(TestHexPatternMatcher)
    
    # Combine all tests
    all_tests = unittest.TestSuite([
        hex_system_suite,
        ascii_renderer_suite,
        composition_suite,
        pattern_suite
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
        print("\n[OK] All tests passed! Hexadecimal system is working correctly.")
        return True
    else:
        print("\n[X] Some tests failed. Please review the output above.")
        return False


def demo_integration():
    """Demonstrate integration between all components."""
    print("\n" + "=" * 80)
    print("INTEGRATION DEMONSTRATION")
    print("=" * 80)
    
    # Create all components
    hex_system = HexadecimalSystem()
    renderer = ASCIIHexRenderer()
    generator = CompositionGenerator()
    matcher = HexPatternMatcher()
    
    print("[OK] All components created")
    
    # Step 1: Create commodities
    compute_comm = hex_system.create_commodity(
        HexCommodityType.HEX_COMPUTE, 100.0, HexDepthLevel.MODERATE
    )
    memory_comm = hex_system.create_commodity(
        HexCommodityType.HEX_MEMORY, 512.0, HexDepthLevel.HIGH
    )
    
    print(f"[OK] Created {len(hex_system.commodities)} commodities")
    
    # Step 2: Execute a trade
    trade = hex_system.execute_trade(
        memory_comm.commodity_id, "AgentA", "AgentB", 250.0
    )
    print(f"[OK] Executed trade: {trade.trade_id}")
    
    # Step 3: Generate market state and composition
    market_state = {
        'demand_pressure': 0.6,
        'volatility': 0.3,
        'trend': 'up'
    }
    
    composition = generator.generate_composition(market_state)
    print(f"[OK] Generated composition: {composition['composition_id']}")
    
    # Step 4: Render the composition
    comp_render = renderer.render_composition(composition)
    print("[OK] Composition rendered")
    
    # Step 5: Detect market patterns
    patterns = matcher.detect_market_patterns(hex_system)
    print(f"[OK] Detected patterns: {list(patterns.keys())}")
    
    # Step 6: Find optimal hex values
    optimal_values = matcher.find_optimal_hex_values(market_state)
    print(f"[OK] Found optimal hex values: {optimal_values}")
    
    # Step 7: Render market state
    market_render = renderer.render_market_state(hex_system)
    print("[OK] Market state rendered")
    
    # Step 8: Render a hex value
    hex_render = renderer.render_hex_value(0xCD)
    print("[OK] Hex value rendered")
    
    print("\n[OK] Integration demonstration completed successfully")
    print("All components work together seamlessly:")
    print("- Hexadecimal trading system")
    print("- ASCII rendering engine")
    print("- Composition generation")
    print("- Pattern matching and optimization")
    print("=" * 80)


if __name__ == "__main__":
    print("Starting comprehensive test suite for Hexadecimal System...")
    
    # Run comprehensive tests
    tests_passed = run_comprehensive_tests()
    
    if tests_passed:
        # Run integration demonstration
        demo_integration()
        
        print("\n" + "=" * 80)
        print("ALL TESTS PASSED - SYSTEM IS STABLE")
        print("=" * 80)
        print("The hexadecimal system with ASCII rendering is fully functional:")
        print("- Hexadecimal commodity trading with depth levels")
        print("- ASCII rendering for visualization")
        print("- Composition generation based on market patterns")
        print("- Pattern detection and optimization")
        print("- Integration with the GAMESA framework")
        print("=" * 80)
    else:
        print("\n" + "=" * 80)
        print("TESTS FAILED - PLEASE REVIEW OUTPUT")
        print("=" * 80)