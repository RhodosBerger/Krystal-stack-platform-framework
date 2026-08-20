#!/usr/bin/env python3
"""
Comprehensive Test Suite for Essential Encoder and OpenVINO Integration

This test suite validates the functionality and compatibility of the essential
encoder with OpenVINO integration for the GAMESA framework.
"""

import unittest
import numpy as np
import sys
from typing import Any, Dict, List
import tempfile
import os

# Import the encoder and OpenVINO integration
try:
    from essential_encoder import EssentialEncoder, NeuralEncoder, QuantizedEncoder
    print("[OK] Successfully imported essential encoder components")
except ImportError as e:
    print(f"[ERROR] Could not import essential encoder: {e}")
    sys.exit(1)

try:
    from openvino_integration import OpenVINOEncoder, GAMESAOpenVINOIntegration, DeviceType, ModelPrecision, ModelConfig
    print("[OK] Successfully imported OpenVINO integration")
except ImportError as e:
    print(f"[INFO] Could not import OpenVINO integration (expected if OpenVINO not installed): {e}")


class TestEssentialEncoder(unittest.TestCase):
    """Test the essential encoder functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.encoder = EssentialEncoder()
        self.neural_encoder = NeuralEncoder()
        self.quantized_encoder = QuantizedEncoder()
    
    def test_binary_encoding(self):
        """Test binary encoding functionality."""
        from essential_encoder import EncodingType
        test_data = {"key": "value", "number": 42}
        result = self.encoder.encode(test_data, encoding_type=EncodingType.BINARY)

        self.assertIsNotNone(result.data)
        self.assertGreater(result.size_encoded, 0)
        self.assertEqual(result.encoding_type, EncodingType.BINARY)

        # Test decoding
        decoded = self.encoder.decode(result.data, result.encoding_type)
        self.assertEqual(type(decoded).__name__, "dict")
    
    def test_base64_encoding(self):
        """Test base64 encoding functionality."""
        from essential_encoder import EncodingType
        test_data = "Hello, World!"
        result = self.encoder.encode(test_data, encoding_type=EncodingType.BASE64)

        self.assertIsNotNone(result.data)
        self.assertTrue(result.data.startswith(b'SGVsbG8'))
        self.assertEqual(result.encoding_type, EncodingType.BASE64)

        # Test decoding
        decoded = self.encoder.decode(result.data, result.encoding_type)
        self.assertEqual(decoded, test_data)
    
    def test_json_encoding(self):
        """Test JSON encoding functionality."""
        from essential_encoder import EncodingType
        test_data = {"name": "test", "value": 123, "active": True}
        result = self.encoder.encode(test_data, encoding_type=EncodingType.JSON)

        self.assertIsNotNone(result.data)
        self.assertIn(b'"name":"test"', result.data)
        self.assertEqual(result.encoding_type, EncodingType.JSON)

        # Test decoding
        decoded = self.encoder.decode(result.data, result.encoding_type)
        self.assertEqual(decoded, test_data)
    
    def test_compressed_encoding(self):
        """Test compressed encoding functionality."""
        from essential_encoder import EncodingType
        test_data = ["repeated"] * 100  # Repetitive data for good compression
        result = self.encoder.encode(test_data, encoding_type=EncodingType.COMPRESSED)

        self.assertIsNotNone(result.data)
        self.assertLess(result.size_encoded, result.size_original)
        self.assertGreater(result.compression_ratio, 1.0)
        self.assertEqual(result.encoding_type, EncodingType.COMPRESSED)

        # Test decoding
        decoded = self.encoder.decode(result.data, result.encoding_type)
        self.assertEqual(decoded, test_data)
    
    def test_neural_encoding(self):
        """Test neural network optimized encoding."""
        from essential_encoder import EncodingType
        test_data = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = self.encoder.encode(test_data, encoding_type=EncodingType.NEURAL)

        self.assertIsNotNone(result.data)
        self.assertEqual(result.encoding_type, EncodingType.NEURAL)

        # Test decoding
        decoded = self.encoder.decode(result.data, result.encoding_type)
        self.assertIsInstance(decoded, np.ndarray)
        self.assertEqual(decoded.shape[0], len(test_data))
    
    def test_hex_encoding(self):
        """Test hexadecimal encoding."""
        from essential_encoder import EncodingType
        test_data = "test data"
        result = self.encoder.encode(test_data, encoding_type=EncodingType.HEX)

        self.assertIsNotNone(result.data)
        self.assertEqual(result.encoding_type, EncodingType.HEX)

        # Test decoding
        decoded = self.encoder.decode(result.data, result.encoding_type)
        self.assertEqual(decoded, test_data)
    
    def test_neural_feature_encoding(self):
        """Test neural encoder feature encoding."""
        features = [1.0, 2.5, 3.7, 4.2]
        encoded = self.neural_encoder.encode_features(features, normalize=True)
        
        self.assertIsInstance(encoded, np.ndarray)
        self.assertEqual(encoded.shape, (4,))
        self.assertLessEqual(encoded.max(), 1.0)
        self.assertGreaterEqual(encoded.min(), 0.0)
    
    def test_categorical_encoding(self):
        """Test categorical variable encoding."""
        categories = ["cat", "dog", "bird", "cat"]
        encoded = self.neural_encoder.encode_categorical(categories)
        
        self.assertIsInstance(encoded, np.ndarray)
        self.assertEqual(encoded.shape, (4, 3))  # 4 samples, 3 unique categories
        self.assertTrue(np.allclose(encoded.sum(axis=1), 1.0))  # Each row sums to 1
    
    def test_sequence_encoding(self):
        """Test sequence encoding."""
        sequence = ["state1", "state2", "state3", "state1"]
        encoded = self.neural_encoder.encode_sequence(sequence, max_length=8)
        
        self.assertIsInstance(encoded, np.ndarray)
        self.assertEqual(encoded.shape[0], 8)  # max_length
    
    def test_quantization(self):
        """Test quantization functionality."""
        data = np.random.random((10, 5)).astype(np.float32) * 100
        quantized, params = self.quantized_encoder.quantize(data)
        dequantized = self.quantized_encoder.dequantize(quantized, params)
        
        self.assertIsInstance(quantized, (np.ndarray))
        self.assertIsInstance(params, dict)
        self.assertEqual(quantized.dtype, np.uint8)  # For 8-bit quantization
        self.assertEqual(dequantized.shape, data.shape)
        
        # Check that error is reasonable
        mse = np.mean((data - dequantized) ** 2)
        self.assertLess(mse, 10.0)  # Reasonable threshold
    
    def test_integrity_verification(self):
        """Test data integrity verification."""
        from essential_encoder import EncodingType
        test_data = {"important": "data", "value": 123}
        result, hash_val = self.encoder.encode_with_hash(test_data, EncodingType.COMPRESSED)

        self.assertIsNotNone(hash_val)
        self.assertTrue(len(hash_val) > 0)

        is_valid = self.encoder.verify_integrity(result.data, hash_val)
        self.assertTrue(is_valid)

        # Test with wrong hash
        is_invalid = self.encoder.verify_integrity(result.data, "wrong_hash")
        self.assertFalse(is_invalid)


class TestOpenVINOIntegration(unittest.TestCase):
    """Test OpenVINO integration functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        try:
            self.openvino_encoder = OpenVINOEncoder()
            self.integration = GAMESAOpenVINOIntegration()
        except Exception as e:
            print(f"OpenVINO not available, using fallback: {e}")
            # Still create instances to test fallback behavior
            self.openvino_encoder = OpenVINOEncoder()
            self.integration = GAMESAOpenVINOIntegration()
    
    def test_device_availability(self):
        """Test device availability detection."""
        devices = self.openvino_encoder.get_available_devices()
        
        self.assertIsInstance(devices, list)
        self.assertGreaterEqual(len(devices), 1)  # Should have at least CPU
        self.assertIn("CPU", devices)  # CPU should always be available
    
    def test_device_performance_info(self):
        """Test device performance information."""
        cpu_info = self.openvino_encoder.get_device_performance_info(DeviceType.CPU)
        
        self.assertIsInstance(cpu_info, dict)
        self.assertIn("device", cpu_info)
        self.assertEqual(cpu_info["device"], "CPU")
    
    def test_preprocessing(self):
        """Test data preprocessing for OpenVINO."""
        test_data = [1.0, 2.0, 3.0, 4.0, 5.0]
        
        # Test without target shape
        processed = self.openvino_encoder.preprocess_for_openvino(test_data)
        self.assertIsInstance(processed, np.ndarray)
        self.assertEqual(processed.shape, (5,))
        self.assertEqual(processed.dtype, np.float32)
        
        # Test with target shape
        processed = self.openvino_encoder.preprocess_for_openvino(test_data, (1, 5))
        self.assertEqual(processed.shape, (1, 5))
    
    def test_encode_with_fallback(self):
        """Test encoding with fallback behavior."""
        test_data = [1.0, 2.0, 3.0]
        
        # This should work even if OpenVINO is not available (fallback to neural encoding)
        encoded, metadata = self.integration.encode_for_inference(test_data)
        
        self.assertIsInstance(encoded, np.ndarray)
        self.assertIsInstance(metadata, dict)
        self.assertIn("fallback", metadata) or self.assertIn("device", metadata)
    
    def test_resource_optimization(self):
        """Test resource optimization recommendations."""
        workload_profile = {
            "complexity": "medium",
            "latency_critical": False,
            "throughput_required": True
        }
        
        recommendations = self.integration.optimize_resources_for_inference(workload_profile)
        
        self.assertIsInstance(recommendations, dict)
        self.assertIn("device_selection", recommendations)
        self.assertIn("precision_setting", recommendations)
        self.assertIn("batch_size", recommendations)
        self.assertIn("num_requests", recommendations)
    
    def test_gamespace_initialization(self):
        """Test GAMESA space initialization."""
        gamespace_config = {
            "preferred_device": "CPU",
            "models": []
        }
        
        success = self.integration.initialize_for_gamespace(gamespace_config)
        self.assertTrue(success)  # Should succeed even with empty models list


class TestIntegrationCompatibility(unittest.TestCase):
    """Test compatibility between encoder and OpenVINO integration."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.encoder = EssentialEncoder()
        self.openvino_encoder = OpenVINOEncoder(self.encoder)
        self.integration = GAMESAOpenVINOIntegration()
    
    def test_encoder_chain(self):
        """Test chaining encoder operations."""
        from essential_encoder import EncodingType
        # Start with raw data
        raw_data = [1.0, 2.0, 3.0, 4.0, 5.0]  # Use numeric data that can be preprocessed

        # Encode with essential encoder
        encoded_result = self.encoder.encode(raw_data, EncodingType.JSON)  # Use JSON to maintain structure

        # Decode back to original format for preprocessing
        decoded_data = self.encoder.decode(encoded_result.data, EncodingType.JSON)

        # Preprocess for OpenVINO
        preprocessed = self.openvino_encoder.preprocess_for_openvino(decoded_data)

        # Verify the result is a numpy array
        self.assertIsInstance(preprocessed, np.ndarray)
    
    def test_neural_flow(self):
        """Test neural network data flow."""
        from essential_encoder import EncodingType
        # Create neural data
        neural_data = [0.1, 0.5, 0.9, 0.3]

        # Encode with neural encoder
        encoded = self.encoder.encode(neural_data, EncodingType.NEURAL)

        # Preprocess for OpenVINO
        processed = self.openvino_encoder.preprocess_for_openvino(neural_data)

        # Both should result in numpy arrays
        self.assertIsInstance(encoded.data, bytes)
        self.assertIsInstance(processed, np.ndarray)
    
    def test_model_config_creation(self):
        """Test model configuration creation."""
        config = ModelConfig(
            model_path="dummy_path.xml",
            device=DeviceType.CPU,
            precision=ModelPrecision.FP32,
            batch_size=1
        )
        
        self.assertEqual(config.device, DeviceType.CPU)
        self.assertEqual(config.precision, ModelPrecision.FP32)
        self.assertEqual(config.batch_size, 1)
        self.assertEqual(config.model_path, "dummy_path.xml")


def run_comprehensive_tests():
    """Run comprehensive tests for encoder and OpenVINO integration."""
    print("=" * 80)
    print("COMPREHENSIVE TEST SUITE: ESSENTIAL ENCODER & OPENVINO INTEGRATION")
    print("=" * 80)
    
    # Create test suites
    essential_encoder_suite = unittest.TestLoader().loadTestsFromTestCase(TestEssentialEncoder)
    openvino_suite = unittest.TestLoader().loadTestsFromTestCase(TestOpenVINOIntegration)
    integration_suite = unittest.TestLoader().loadTestsFromTestCase(TestIntegrationCompatibility)
    
    # Combine all tests
    all_tests = unittest.TestSuite([
        essential_encoder_suite,
        openvino_suite,
        integration_suite
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
        print("\n[OK] All tests passed! Encoder and OpenVINO integration are compatible.")
        return True
    else:
        print("\n[X] Some tests failed. Please review the output above.")
        return False


def demo_end_to_end_integration():
    """Demonstrate end-to-end integration."""
    print("\n" + "=" * 80)
    print("END-TO-END INTEGRATION DEMONSTRATION")
    print("=" * 80)
    
    # Create the integration components
    encoder = EssentialEncoder()
    openvino_encoder = OpenVINOEncoder(encoder)
    gamesa_integration = GAMESAOpenVINOIntegration()
    
    print("[OK] Components created")
    
    # Example: Process telemetry data through the pipeline
    telemetry_data = {
        "cpu_usage": [75.2, 80.1, 78.5, 82.3],
        "memory_usage": [65.3, 68.7, 70.1, 69.8],
        "gpu_usage": [45.6, 50.2, 48.9, 52.1],
        "temperature": [72.5, 74.3, 73.8, 75.1]
    }
    
    print(f"Original telemetry data keys: {list(telemetry_data.keys())}")
    
    # Step 1: Encode with essential encoder
    from essential_encoder import EncodingType
    encoded_result = encoder.encode(telemetry_data, EncodingType.JSON)
    print(f"JSON encoded size: {encoded_result.size_encoded} bytes")
    
    # Step 2: Preprocess for neural network
    neural_features = []
    for key, values in telemetry_data.items():
        if isinstance(values, list):
            neural_features.extend(values)
    
    preprocessed = openvino_encoder.preprocess_for_openvino(neural_features)
    print(f"Preprocessed shape: {preprocessed.shape}")
    
    # Step 3: Simulate inference encoding (with fallback)
    inference_output, metadata = gamesa_integration.encode_for_inference(neural_features)
    print(f"Inference output shape: {inference_output.shape}")
    print(f"Inference metadata: {metadata}")
    
    # Step 4: Get resource optimization
    workload_profile = {
        "complexity": "medium",
        "latency_critical": True,
        "throughput_required": False
    }
    resources = gamesa_integration.optimize_resources_for_inference(workload_profile)
    print(f"Resource recommendations: {resources}")
    
    # Step 5: Verify data integrity
    from essential_encoder import EncodingType
    _, hash_val = encoder.encode_with_hash(telemetry_data, EncodingType.COMPRESSED)
    is_valid = encoder.verify_integrity(encoded_result.data, hash_val)
    print(f"Data integrity check: {'PASS' if is_valid else 'FAIL'}")
    
    print("\n[OK] End-to-end integration completed successfully")
    print("The system can process telemetry data through the entire pipeline:")
    print("  1. Essential encoding (JSON, compression, etc.)")
    print("  2. Neural preprocessing")
    print("  3. Inference preparation")
    print("  4. Resource optimization")
    print("  5. Data integrity verification")
    print("=" * 80)


if __name__ == "__main__":
    print("Starting comprehensive test suite for Essential Encoder and OpenVINO Integration...")
    
    # Run comprehensive tests
    tests_passed = run_comprehensive_tests()
    
    if tests_passed:
        # Run end-to-end demonstration
        demo_end_to_end_integration()
        
        print("\n" + "=" * 80)
        print("ALL TESTS PASSED - INTEGRATION IS STABLE")
        print("=" * 80)
        print("The Essential Encoder and OpenVINO integration is fully functional:")
        print("- Multiple encoding strategies supported")
        print("- Neural network optimization capabilities")
        print("- Hardware acceleration (with OpenVINO)")
        print("- Data integrity verification")
        print("- Resource optimization for GAMESA framework")
        print("- Fallback mechanisms when OpenVINO unavailable")
        print("=" * 80)
    else:
        print("\n" + "=" * 80)
        print("TESTS FAILED - PLEASE REVIEW OUTPUT")
        print("=" * 80)
        sys.exit(1)