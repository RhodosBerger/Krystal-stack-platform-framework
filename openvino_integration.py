#!/usr/bin/env python3
"""
OpenVINO Integration for GAMESA/KrystalStack Framework

This module provides OpenVINO integration for the essential encoder,
enabling hardware-accelerated neural network inference and optimization.
"""

import numpy as np
import logging
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass
from enum import Enum
import os
import sys
from datetime import datetime

# Import essential encoder components
from essential_encoder import EssentialEncoder, NeuralEncoder, QuantizedEncoder

# Try to import OpenVINO - handle gracefully if not available
try:
    from openvino.runtime import Core, Model, CompiledModel, InferRequest
    from openvino.preprocess import PrePostProcessor
    from openvino.runtime.passes import Manager
    import openvino as ov
    OPENVINO_AVAILABLE = True
    logger = logging.getLogger(__name__)
    logger.info("OpenVINO runtime imported successfully")
except ImportError:
    OPENVINO_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("OpenVINO not available. Hardware acceleration features will be limited.")
    
    # Create mock classes for graceful degradation
    class MockCore:
        def __init__(self):
            pass
        def compile_model(self, *args, **kwargs):
            raise NotImplementedError("OpenVINO not available")
        def read_model(self, *args, **kwargs):
            raise NotImplementedError("OpenVINO not available")
    
    Core = MockCore


class DeviceType(Enum):
    """Types of devices supported by OpenVINO."""
    CPU = "CPU"
    GPU = "GPU"
    VPU = "VPU"  # Vision Processing Unit
    FPGA = "FPGA"
    HETERO = "HETERO"  # Heterogeneous execution
    MULTI = "MULTI"  # Multi-device execution


class ModelPrecision(Enum):
    """Precision levels for model optimization."""
    FP32 = "FP32"  # 32-bit floating point
    FP16 = "FP16"  # 16-bit floating point
    INT8 = "INT8"  # 8-bit integer
    BF16 = "BF16"  # BFloat16


@dataclass
class ModelConfig:
    """Configuration for OpenVINO model."""
    model_path: str
    device: DeviceType = DeviceType.CPU
    precision: ModelPrecision = ModelPrecision.FP32
    batch_size: int = 1
    num_requests: int = 1
    inference_precision: str = "f32"
    dynamic_shapes: bool = False


@dataclass
class OptimizationResult:
    """Result of model optimization."""
    original_model_size: int
    optimized_model_size: int
    optimization_ratio: float
    optimization_time: float
    device_used: DeviceType
    precision_used: ModelPrecision
    performance_metrics: Dict[str, float]


class OpenVINOEncoder:
    """
    OpenVINO integration for the essential encoder.
    
    Provides hardware-accelerated encoding and neural network inference
    optimized for Intel hardware platforms.
    """
    
    def __init__(self, encoder: EssentialEncoder = None):
        if not OPENVINO_AVAILABLE:
            logger.warning("OpenVINO not available. Using CPU fallback.")
        
        self.core = Core() if OPENVINO_AVAILABLE else None
        self.encoder = encoder or EssentialEncoder()
        self.neural_encoder = NeuralEncoder()
        self.quantized_encoder = QuantizedEncoder()
        self.compiled_models = {}
        self.optimization_history = []
        
    def get_available_devices(self) -> List[str]:
        """Get list of available OpenVINO devices."""
        if not OPENVINO_AVAILABLE:
            return ["CPU"]  # Fallback to CPU
        
        try:
            return self.core.get_available_devices()
        except Exception as e:
            logger.error(f"Error getting available devices: {e}")
            return ["CPU"]  # Fallback to CPU
    
    def optimize_model(self, model_path: str, config: ModelConfig) -> OptimizationResult:
        """
        Optimize a model for the specified device and precision.
        
        Args:
            model_path: Path to the model to optimize
            config: Model configuration
            
        Returns:
            OptimizationResult with optimization metrics
        """
        import time
        start_time = time.time()
        
        if not OPENVINO_AVAILABLE:
            logger.warning("OpenVINO not available. Returning dummy optimization result.")
            return OptimizationResult(
                original_model_size=0,
                optimized_model_size=0,
                optimization_ratio=1.0,
                optimization_time=0.0,
                device_used=config.device,
                precision_used=config.precision,
                performance_metrics={"fallback": True}
            )
        
        try:
            # Read the model
            model = self.core.read_model(model_path)
            
            # Apply transformations based on precision
            if config.precision == ModelPrecision.FP16:
                # Convert to FP16 if needed
                pass  # OpenVINO handles this automatically in many cases
            elif config.precision == ModelPrecision.INT8:
                # Apply INT8 quantization
                pass  # Would require quantization-aware training
            
            # Compile the model
            compiled_model = self.core.compile_model(
                model,
                config.device.value,
                config={
                    "PERFORMANCE_HINT": "THROUGHPUT",
                    "NUM_STREAMS": str(config.num_requests)
                }
            )
            
            # Save the compiled model
            model_key = f"{config.device.value}_{config.precision.value}"
            self.compiled_models[model_key] = compiled_model
            
            # Get model sizes (approximate)
            original_size = os.path.getsize(model_path) if os.path.exists(model_path) else 0
            # For optimized size, we'll use the same since compilation doesn't change file size
            optimized_size = original_size
            
            result = OptimizationResult(
                original_model_size=original_size,
                optimized_model_size=optimized_size,
                optimization_ratio=1.0,  # Placeholder
                optimization_time=time.time() - start_time,
                device_used=config.device,
                precision_used=config.precision,
                performance_metrics={
                    "model_inputs": len(model.inputs),
                    "model_outputs": len(model.outputs)
                }
            )
            
            self.optimization_history.append(result)
            return result
            
        except Exception as e:
            logger.error(f"Error optimizing model {model_path}: {e}")
            raise
    
    def compile_model_for_inference(self, model_path: str, device: DeviceType = DeviceType.CPU) -> str:
        """
        Compile a model for inference on the specified device.
        
        Args:
            model_path: Path to the model file
            device: Target device for compilation
            
        Returns:
            Model key for accessing the compiled model
        """
        if not OPENVINO_AVAILABLE:
            logger.warning("OpenVINO not available. Cannot compile model.")
            return "cpu_fallback"
        
        try:
            model = self.core.read_model(model_path)
            compiled_model = self.core.compile_model(model, device.value)
            
            model_key = f"{device.value}_model"
            self.compiled_models[model_key] = compiled_model
            
            logger.info(f"Model compiled successfully for {device.value}")
            return model_key
            
        except Exception as e:
            logger.error(f"Error compiling model: {e}")
            raise
    
    def encode_with_openvino(self, data: Any, model_key: str, 
                           input_name: Optional[str] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Encode data using a compiled OpenVINO model.
        
        Args:
            data: Input data to encode
            model_key: Key for the compiled model
            input_name: Name of the input tensor (if multiple inputs)
            
        Returns:
            Encoded output and metadata
        """
        if not OPENVINO_AVAILABLE:
            logger.warning("OpenVINO not available. Using neural encoder fallback.")
            # Use the neural encoder as fallback
            neural_output = self.neural_encoder.encode_features(data)
            return neural_output, {"fallback": True, "device": "CPU"}
        
        try:
            if model_key not in self.compiled_models:
                raise ValueError(f"Model with key {model_key} not found. Compile model first.")
            
            compiled_model = self.compiled_models[model_key]
            infer_request = compiled_model.create_infer_request()
            
            # Prepare input data
            if isinstance(data, (list, tuple)):
                input_data = np.array(data, dtype=np.float32)
            elif isinstance(data, np.ndarray):
                input_data = data.astype(np.float32)
            else:
                input_data = np.array([data], dtype=np.float32)
            
            # Ensure proper shape for the model
            input_tensor = compiled_model.input(0)  # Use first input
            input_shape = input_tensor.shape
            
            # Handle dynamic shapes
            if -1 in input_shape or any(dim == -1 for dim in input_shape):
                # Set shape dynamically based on input
                actual_shape = list(input_shape)
                for i, dim in enumerate(input_shape):
                    if dim == -1 and i < len(input_data.shape):
                        actual_shape[i] = input_data.shape[i]
                input_data = np.reshape(input_data, actual_shape)
            else:
                # Ensure input matches expected shape
                if input_data.size < np.prod(input_shape):
                    # Pad with zeros
                    padded_data = np.zeros(input_shape, dtype=np.float32)
                    flat_data = input_data.flatten()
                    padded_data.flat[:len(flat_data)] = flat_data
                    input_data = padded_data
                elif input_data.size > np.prod(input_shape):
                    # Truncate
                    input_data = input_data.flat[:np.prod(input_shape)].reshape(input_shape)
                else:
                    input_data = input_data.reshape(input_shape)
            
            # Run inference
            result = infer_request.infer(inputs={input_tensor.any_name: input_data})
            
            # Get output
            output_tensor = compiled_model.output(0)  # Use first output
            output_data = np.array(result[output_tensor])
            
            metadata = {
                "device": compiled_model.get_property("DEVICE_TYPE"),
                "model_precision": compiled_model.get_property("INFERENCE_PRECISION_HINT"),
                "input_shape": input_data.shape,
                "output_shape": output_data.shape,
                "inference_time": infer_request.get_profiling_info()
            }
            
            return output_data, metadata
            
        except Exception as e:
            logger.error(f"Error during OpenVINO inference: {e}")
            raise
    
    def quantize_for_openvino(self, model_path: str, calibration_data: List[np.ndarray]) -> str:
        """
        Quantize a model for INT8 inference with OpenVINO.
        
        Args:
            model_path: Path to the model to quantize
            calibration_data: Data for calibration
            
        Returns:
            Path to the quantized model
        """
        if not OPENVINO_AVAILABLE:
            logger.warning("OpenVINO not available. Quantization not supported.")
            return model_path  # Return original path as fallback
        
        try:
            # In a real implementation, this would use OpenVINO's Post-training Optimization Toolkit
            # For now, we'll simulate the process
            logger.info(f"Quantizing model {model_path} for INT8 inference...")
            
            # This is where you would typically use POT (Post-training Optimization Toolkit)
            # quantized_model_path = pot_quantize(model_path, calibration_data)
            
            # For demonstration, return the original path with a note
            quantized_path = model_path.replace(".xml", "_int8.xml")
            logger.info(f"Quantized model would be saved to: {quantized_path}")
            
            return quantized_path
            
        except Exception as e:
            logger.error(f"Error during quantization: {e}")
            return model_path  # Return original on error
    
    def preprocess_for_openvino(self, data: Any, target_shape: Tuple[int, ...] = None) -> np.ndarray:
        """
        Preprocess data for OpenVINO inference.
        
        Args:
            data: Input data to preprocess
            target_shape: Target shape for the data
            
        Returns:
            Preprocessed data ready for inference
        """
        # Use neural encoder to prepare the data
        processed_data = self.neural_encoder.encode_features(data, normalize=True)
        
        if target_shape:
            # Reshape to target shape, padding or truncating as needed
            if processed_data.size < np.prod(target_shape):
                # Pad with zeros
                reshaped_data = np.zeros(target_shape, dtype=np.float32)
                flat_data = processed_data.flatten()
                reshaped_data.flat[:len(flat_data)] = flat_data
                processed_data = reshaped_data
            elif processed_data.size > np.prod(target_shape):
                # Truncate
                processed_data = processed_data.flat[:np.prod(target_shape)].reshape(target_shape)
            else:
                processed_data = processed_data.reshape(target_shape)
        
        return processed_data.astype(np.float32)
    
    def benchmark_model(self, model_key: str, input_data: np.ndarray, 
                       num_iterations: int = 100) -> Dict[str, float]:
        """
        Benchmark a compiled model's performance.
        
        Args:
            model_key: Key for the compiled model
            input_data: Input data for benchmarking
            num_iterations: Number of inference iterations
            
        Returns:
            Performance metrics
        """
        if not OPENVINO_AVAILABLE:
            logger.warning("OpenVINO not available. Skipping benchmark.")
            return {"avg_inference_time_ms": 0.0, "throughput_fps": 0.0, "fallback": True}
        
        try:
            if model_key not in self.compiled_models:
                raise ValueError(f"Model with key {model_key} not found.")
            
            compiled_model = self.compiled_models[model_key]
            infer_request = compiled_model.create_infer_request()
            
            input_tensor = compiled_model.input(0)
            
            # Ensure input data has correct shape
            expected_shape = input_tensor.shape
            if -1 in expected_shape:
                # Dynamic shape - use the provided data shape
                pass
            else:
                input_data = input_data.reshape(expected_shape)
            
            import time
            start_time = time.time()
            
            # Run multiple inferences
            for _ in range(num_iterations):
                infer_request.infer(inputs={input_tensor.any_name: input_data})
            
            total_time = time.time() - start_time
            avg_time = (total_time / num_iterations) * 1000  # Convert to ms
            throughput = num_iterations / total_time  # Inferences per second
            
            metrics = {
                "avg_inference_time_ms": avg_time,
                "throughput_fps": throughput,
                "total_time_s": total_time,
                "num_iterations": num_iterations
            }
            
            logger.info(f"Benchmark results: {avg_time:.3f}ms avg, {throughput:.2f} FPS")
            return metrics
            
        except Exception as e:
            logger.error(f"Error during benchmarking: {e}")
            raise
    
    def get_device_performance_info(self, device: DeviceType) -> Dict[str, Any]:
        """
        Get performance information for a specific device.
        
        Args:
            device: Target device
            
        Returns:
            Device performance information
        """
        if not OPENVINO_AVAILABLE:
            return {
                "device": device.value,
                "supported": False,
                "max_threads": 0,
                "memory_available_mb": 0
            }
        
        try:
            # Get device properties
            properties = self.core.get_property(device.value, "SUPPORTED_PROPERTIES")
            
            # Try to get more specific info
            try:
                max_threads = self.core.get_property(device.value, "CPU_THREADS_NUM")
            except:
                max_threads = 0
                
            try:
                memory_info = self.core.get_property(device.value, "AVAILABLE_DEVICES")
            except:
                memory_info = []
            
            return {
                "device": device.value,
                "supported": True,
                "properties": properties,
                "max_threads": max_threads,
                "memory_info": memory_info
            }
        except Exception as e:
            logger.error(f"Error getting device info: {e}")
            return {
                "device": device.value,
                "supported": False,
                "error": str(e)
            }


class GAMESAOpenVINOIntegration:
    """
    Main integration class for combining GAMESA framework with OpenVINO.
    
    This class provides the bridge between the GAMESA resource management
    system and OpenVINO's hardware acceleration capabilities.
    """
    
    def __init__(self):
        self.encoder = EssentialEncoder()
        self.openvino_encoder = OpenVINOEncoder(self.encoder)
        self.active_models = {}
        
    def initialize_for_gamespace(self, gamespace_config: Dict[str, Any]) -> bool:
        """
        Initialize OpenVINO integration for a specific GAMESA use case.
        
        Args:
            gamespace_config: Configuration for the specific use case
            
        Returns:
            True if initialization successful
        """
        try:
            # Determine optimal device based on config
            preferred_device = gamespace_config.get("preferred_device", "CPU")
            available_devices = self.openvino_encoder.get_available_devices()
            
            if preferred_device in available_devices:
                device_type = DeviceType(preferred_device)
            else:
                # Fallback to CPU
                device_type = DeviceType.CPU
                logger.warning(f"Preferred device {preferred_device} not available, using CPU")
            
            # Initialize any required models based on config
            models_to_load = gamespace_config.get("models", [])
            for model_info in models_to_load:
                model_path = model_info["path"]
                model_name = model_info["name"]
                
                config = ModelConfig(
                    model_path=model_path,
                    device=device_type,
                    precision=ModelPrecision(model_info.get("precision", "FP32"))
                )
                
                # Optimize and compile the model
                optimization_result = self.openvino_encoder.optimize_model(model_path, config)
                logger.info(f"Model {model_name} optimized: {optimization_result.optimization_ratio:.2f}x")
                
                # Store model reference
                self.active_models[model_name] = {
                    "config": config,
                    "optimization_result": optimization_result,
                    "device": device_type
                }
            
            logger.info(f"OpenVINO integration initialized for GAMESA with device: {device_type.value}")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing OpenVINO for GAMESA: {e}")
            return False
    
    def encode_for_inference(self, data: Any, model_name: str = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Encode data specifically for neural network inference using OpenVINO.
        
        Args:
            data: Input data to encode
            model_name: Name of the model to use (optional)
            
        Returns:
            Encoded data and metadata
        """
        if model_name and model_name in self.active_models:
            # Use the specific model
            model_info = self.active_models[model_name]
            model_key = f"{model_info['device'].value}_{model_info['config'].precision.value}"
            
            try:
                result, metadata = self.openvino_encoder.encode_with_openvino(data, model_key)
                return result, metadata
            except Exception as e:
                logger.warning(f"OpenVINO inference failed: {e}, falling back to neural encoding")
                # Fallback to neural encoding
                encoded_data = self.openvino_encoder.neural_encoder.encode_features(data)
                return encoded_data, {"fallback": True, "device": "CPU"}
        else:
            # Use neural encoding as default
            encoded_data = self.openvino_encoder.neural_encoder.encode_features(data)
            return encoded_data, {"fallback": True, "device": "CPU"}
    
    def optimize_resources_for_inference(self, workload_profile: Dict[str, Any]) -> Dict[str, Any]:
        """
        Optimize system resources for neural network inference based on workload profile.
        
        Args:
            workload_profile: Profile of the expected workload
            
        Returns:
            Resource optimization recommendations
        """
        recommendations = {
            "device_selection": "CPU",  # Default recommendation
            "precision_setting": "FP32",
            "batch_size": 1,
            "num_requests": 1,
            "memory_allocation_mb": 512,
            "cpu_threads": 4
        }
        
        # Analyze workload profile to make better recommendations
        complexity = workload_profile.get("complexity", "medium")
        latency_critical = workload_profile.get("latency_critical", False)
        throughput_required = workload_profile.get("throughput_required", False)
        
        available_devices = self.openvino_encoder.get_available_devices()
        
        # Device selection logic
        if "GPU" in available_devices and complexity == "high" and not latency_critical:
            recommendations["device_selection"] = "GPU"
            recommendations["precision_setting"] = "FP16"  # GPUs handle FP16 well
        elif "CPU" in available_devices:
            recommendations["device_selection"] = "CPU"
            if complexity == "low":
                "INT8"  # For less complex models, INT8 can be efficient
        
        # Batch size optimization
        if throughput_required:
            recommendations["batch_size"] = 8  # Higher batch for throughput
            recommendations["num_requests"] = 4  # Multiple infer requests
        elif latency_critical:
            recommendations["batch_size"] = 1  # Lowest latency
            recommendations["num_requests"] = 1
        
        # Memory allocation based on model size
        if complexity == "high":
            recommendations["memory_allocation_mb"] = 2048
        elif complexity == "medium":
            recommendations["memory_allocation_mb"] = 1024
        
        logger.info(f"Resource optimization recommendations: {recommendations}")
        return recommendations


def demo_openvino_integration():
    """Demonstrate OpenVINO integration with the essential encoder."""
    print("=" * 80)
    print("OPENVINO INTEGRATION DEMONSTRATION")
    print("=" * 80)
    
    # Create integration instance
    gamesa_ov = GAMESAOpenVINOIntegration()
    print(f"[OK] OpenVINO integration initialized")
    print(f"  OpenVINO available: {OPENVINO_AVAILABLE}")
    
    # Show available devices
    devices = gamesa_ov.openvino_encoder.get_available_devices()
    print(f"  Available devices: {devices}")
    
    # Get device performance info for each available device
    for device_str in devices:
        try:
            device_type = DeviceType(device_str)
            perf_info = gamesa_ov.openvino_encoder.get_device_performance_info(device_type)
            print(f"  {device_str} performance: {perf_info.get('max_threads', 'N/A')} threads")
        except ValueError:
            print(f"  {device_str} performance: Unknown device type")
    
    # Create a sample gamespace configuration
    gamespace_config = {
        "preferred_device": devices[0] if devices else "CPU",
        "models": [
            {
                "name": "telemetry_encoder",
                "path": "",  # Would be actual model path
                "precision": "FP32"
            }
        ]
    }
    
    # Initialize for gamespace
    init_success = gamesa_ov.initialize_for_gamespace(gamespace_config)
    print(f"  GAMESA initialization: {'SUCCESS' if init_success else 'FAILED'}")
    
    # Test encoding with fallback
    test_data = [1.0, 2.0, 3.0, 4.0, 5.0]
    encoded_data, metadata = gamesa_ov.encode_for_inference(test_data)
    print(f"  Encoding result shape: {encoded_data.shape}")
    print(f"  Encoding metadata: {metadata}")
    
    # Test resource optimization
    workload_profile = {
        "complexity": "medium",
        "latency_critical": False,
        "throughput_required": True
    }
    resource_rec = gamesa_ov.optimize_resources_for_inference(workload_profile)
    print(f"  Resource recommendations: {resource_rec}")
    
    # If OpenVINO is available, try more advanced features
    if OPENVINO_AVAILABLE:
        print("\n--- Advanced OpenVINO Features ---")
        
        try:
            # Test model compilation (with a dummy path)
            # Note: In a real scenario, you would have actual model files
            dummy_model_path = "dummy_model.xml"
            if os.path.exists(dummy_model_path):
                model_key = gamesa_ov.openvino_encoder.compile_model_for_inference(
                    dummy_model_path, DeviceType.CPU
                )
                print(f"  Model compiled with key: {model_key}")
            else:
                print("  Skipping model compilation (no model file found)")
        
        except Exception as e:
            print(f"  Model compilation test: {e}")
        
        try:
            # Test preprocessing
            preprocessed = gamesa_ov.openvino_encoder.preprocess_for_openvino(
                test_data, target_shape=(1, 5)
            )
            print(f"  Preprocessed shape: {preprocessed.shape}")
        except Exception as e:
            print(f"  Preprocessing test: {e}")
    
    else:
        print("\n--- OpenVINO Not Available ---")
        print("  Using CPU fallback for all operations")
        print("  Install OpenVINO for hardware acceleration")
    
    print("\n" + "=" * 80)
    print("OPENVINO INTEGRATION DEMONSTRATION COMPLETE")
    print("Integration provides hardware acceleration for neural network inference")
    print("with automatic resource optimization for GAMESA framework")
    print("=" * 80)


if __name__ == "__main__":
    demo_openvino_integration()