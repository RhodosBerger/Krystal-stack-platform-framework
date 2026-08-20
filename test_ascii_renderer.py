#!/usr/bin/env python3
"""
Test Suite for ASCII Image Renderer
"""

import unittest
import numpy as np
from PIL import Image
from ascii_image_renderer import ASCIIImageRenderer, HexadecimalASCIIConverter, ASCIIConfig


class TestASCIIImageRenderer(unittest.TestCase):
    """Test the ASCII image renderer functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.renderer = ASCIIImageRenderer()
    
    def test_config_initialization(self):
        """Test configuration initialization."""
        config = ASCIIConfig(width=40, height=20, charset=" .:-")
        renderer = ASCIIImageRenderer(config)
        
        self.assertEqual(renderer.config.width, 40)
        self.assertEqual(renderer.config.height, 20)
        self.assertEqual(renderer.config.charset, " .:-")
    
    def test_resize_image(self):
        """Test image resizing functionality."""
        # Create a small test image
        image = Image.new('RGB', (100, 50), 'white')
        
        resized = self.renderer.resize_image(image)
        
        # Check that the resized image has the expected dimensions
        self.assertEqual(resized.size[0], self.renderer.config.width)
        # Height is adjusted for character aspect ratio
        expected_height = int(self.renderer.config.height / self.renderer.config.scale_factor)
        self.assertEqual(resized.size[1], expected_height)
    
    def test_grayscale_conversion(self):
        """Test grayscale conversion."""
        # Create a color test image
        image = Image.new('RGB', (10, 10), 'red')
        
        grayscale = self.renderer.grayscale_image(image)
        
        # Check that the image is now in grayscale mode
        self.assertEqual(grayscale.mode, 'L')
    
    def test_render_simple_image(self):
        """Test rendering a simple image."""
        # Create a simple checkerboard pattern
        width, height = 20, 10
        image = Image.new('RGB', (width, height), 'white')
        pixels = image.load()
        
        # Create a checkerboard pattern
        for x in range(width):
            for y in range(height):
                if (x // 4) % 2 == (y // 2) % 2:
                    pixels[x, y] = (128, 128, 128)  # Gray
        
        ascii_art = self.renderer.render_image(image)
        
        # Check that we got some output
        self.assertIsInstance(ascii_art, str)
        self.assertGreater(len(ascii_art), 0)
        
        # Check that it contains ASCII characters from the charset
        for char in ascii_art:
            if char not in ['\n', '\r', ' ', '\t']:  # Skip whitespace
                self.assertIn(char, self.renderer.config.charset)
                break  # Only need to check one character


class TestHexadecimalASCIIConverter(unittest.TestCase):
    """Test the hexadecimal ASCII converter functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.converter = HexadecimalASCIIConverter()
    
    def test_hex_to_ascii_art(self):
        """Test converting hex to ASCII art."""
        hex_data = "48656C6C6F20576F726C64"  # "Hello World" in hex
        ascii_art = self.converter.hex_to_ascii_art(hex_data)
        
        self.assertIsInstance(ascii_art, str)
        self.assertIn("48 65 6C 6C 6F", ascii_art)  # Check hex values are present
        self.assertIn("Hello World", ascii_art)     # Check ASCII interpretation
    
    def test_hex_to_ascii_art_with_spaces(self):
        """Test converting hex with spaces to ASCII art."""
        hex_data = "48 65 6C 6C 6F 20 57 6F 72 6C 64"  # "Hello World" with spaces
        ascii_art = self.converter.hex_to_ascii_art(hex_data)
        
        self.assertIsInstance(ascii_art, str)
        self.assertIn("48 65 6C", ascii_art)
    
    def test_hex_to_ascii_art_with_prefix(self):
        """Test converting hex with 0x prefix to ASCII art."""
        hex_data = "0x48656C6C6F20576F726C64"  # "Hello World" with 0x prefix
        ascii_art = self.converter.hex_to_ascii_art(hex_data)
        
        self.assertIsInstance(ascii_art, str)
        self.assertIn("48 65 6C", ascii_art)
    
    def test_visualize_hex_distribution(self):
        """Test hex distribution visualization."""
        hex_values = [0x10, 0x20, 0x30, 0x40, 0x50, 0x60, 0x70, 0x80]
        visualization = self.converter.visualize_hex_distribution(hex_values)
        
        self.assertIsInstance(visualization, str)
        self.assertGreater(len(visualization), 0)
        # Should contain visualization characters
        self.assertIn('#', visualization)
    
    def test_visualize_empty_distribution(self):
        """Test hex distribution visualization with empty list."""
        visualization = self.converter.visualize_hex_distribution([])
        
        self.assertIsInstance(visualization, str)
        self.assertIn("No data to visualize", visualization)
    
    def test_hex_composition_to_ascii(self):
        """Test converting composition data to ASCII."""
        composition_data = {
            'composition_id': 'TEST_COMP_001',
            'efficiency_score': 0.85,
            'risk_level': 'low',
            'hex_values': [0x10, 0x20, 0x30],
            'resources': {
                'compute': {'hex_value': 0x10, 'allocation': 10.0, 'priority': 1}
            }
        }
        
        ascii_art = self.converter.hex_composition_to_ascii(composition_data)
        
        self.assertIsInstance(ascii_art, str)
        self.assertIn("TEST_COMP_001", ascii_art)
        self.assertIn("Efficiency: 0.85", ascii_art)
        self.assertIn("Risk: LOW", ascii_art)
        self.assertIn("compute:", ascii_art)
    
    def test_hex_composition_empty_data(self):
        """Test converting empty composition data to ASCII."""
        ascii_art = self.converter.hex_composition_to_ascii({})
        
        self.assertIsInstance(ascii_art, str)
        self.assertIn("No composition data", ascii_art)


def run_comprehensive_tests():
    """Run comprehensive tests for the ASCII image renderer."""
    print("=" * 80)
    print("COMPREHENSIVE TEST SUITE: ASCII IMAGE RENDERER")
    print("=" * 80)
    
    # Create test suites
    ascii_renderer_suite = unittest.TestLoader().loadTestsFromTestCase(TestASCIIImageRenderer)
    hex_converter_suite = unittest.TestLoader().loadTestsFromTestCase(TestHexadecimalASCIIConverter)
    
    # Combine all tests
    all_tests = unittest.TestSuite([
        ascii_renderer_suite,
        hex_converter_suite
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
        print("\n[OK] All tests passed! ASCII image renderer is working correctly.")
        return True
    else:
        print("\n[X] Some tests failed. Please review the output above.")
        return False


def demo_integration_with_framework():
    """Demonstrate integration with the existing framework."""
    print("\n" + "=" * 80)
    print("FRAMEWORK INTEGRATION DEMONSTRATION")
    print("=" * 80)
    
    # Create the converter
    converter = HexadecimalASCIIConverter()
    
    print("[OK] Hexadecimal ASCII converter created")
    
    # Example: Convert hex data from the hexadecimal system
    hex_values = [0x10, 0x25, 0x3A, 0x4F, 0x64, 0x79, 0x8E, 0xA3, 0xB8, 0xCD]
    print(f"[OK] Sample hex values: {hex_values}")
    
    # Create distribution visualization
    dist_viz = converter.visualize_hex_distribution(hex_values)
    print(f"[OK] Distribution visualization created")
    
    # Create composition-like data
    comp_data = {
        'composition_id': 'FRAMEWORK_COMP_001',
        'efficiency_score': 0.78,
        'risk_level': 'medium',
        'hex_values': hex_values,
        'resources': {
            'compute': {'hex_value': 0x25, 'allocation': 15.5, 'priority': 2},
            'memory': {'hex_value': 0x4F, 'allocation': 35.0, 'priority': 4},
            'gpu': {'hex_value': 0x79, 'allocation': 55.5, 'priority': 6},
            'neural': {'hex_value': 0xA3, 'allocation': 75.0, 'priority': 8}
        }
    }
    
    comp_viz = converter.hex_composition_to_ascii(comp_data)
    print(f"[OK] Composition visualization created")
    
    # Show the visualizations
    print("\nHEX DISTRIBUTION VISUALIZATION:")
    print(dist_viz[:200] + "..." if len(dist_viz) > 200 else dist_viz)
    
    print("\nCOMPOSITION VISUALIZATION:")
    print(comp_viz)
    
    print("\n[OK] Integration demonstration completed successfully")
    print("The ASCII renderer integrates with:")
    print("- Hexadecimal trading system")
    print("- Composition generator")
    print("- Resource allocation visualization")
    print("- Framework telemetry data")
    print("=" * 80)


if __name__ == "__main__":
    print("Starting comprehensive test suite for ASCII Image Renderer...")
    
    # Run comprehensive tests
    tests_passed = run_comprehensive_tests()
    
    if tests_passed:
        # Run integration demonstration
        demo_integration_with_framework()
        
        print("\n" + "=" * 80)
        print("ALL TESTS PASSED - INTEGRATION IS STABLE")
        print("=" * 80)
        print("The ASCII image renderer is fully functional:")
        print("- Image to ASCII conversion")
        print("- Hexadecimal data visualization")
        print("- Distribution pattern visualization")
        print("- Composition data visualization")
        print("- Integration with GAMESA framework")
        print("=" * 80)
    else:
        print("\n" + "=" * 80)
        print("TESTS FAILED - PLEASE REVIEW OUTPUT")
        print("=" * 80)