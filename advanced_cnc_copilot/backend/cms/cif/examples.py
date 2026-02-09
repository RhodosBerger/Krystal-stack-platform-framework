"""
CIF Framework Examples and Demos
Demonstrates usage patterns for CNC Intelligence Framework
"""

import numpy as np
from backend.cms.cif import CIFCore, CIFModelConverter, Pipeline


def example_1_basic_inference():
    """
    Example 1: Basic inference workflow
    Load model, compile for device, run inference
    """
    print("=" * 70)
    print("Example 1: Basic Inference")
    print("=" * 70)
    
    # Create inference engine
    core = CIFCore()
    
    # Show available devices
    devices = core.get_available_devices()
    print(f"Available devices: {devices}\n")
    
    # For demo, create a simple synthetic model
    # In production, you would load an actual .cif file
    from backend.cms.cif.cif_core import CIFModel
    
    # Create dummy model
    dummy_model = CIFModel(
        model_data={'weights': np.random.randn(10, 10)},
        metadata={'name': 'demo_model', 'framework': 'synthetic'},
        input_spec={'features': (1, 10)},
        output_spec={'prediction': (1, 1)},
        optimization_config={}
    )
    
    # Compile for AUTO device selection
    compiled = core.compile_model(dummy_model, device='AUTO')
    print(f"Model compiled for: {compiled.device}")
    
    # Run inference
    inputs = {'features': np.random.randn(1, 10)}
    result = compiled(inputs)
    print(f"Inference result shape: {result.get('output', np.array([])).shape}")
    
    # Performance stats
    stats = compiled.get_performance_stats()
    print(f"\nPerformance: {stats['average_time']*1000:.2f}ms average")
    print()


def example_2_async_inference():
    """
    Example 2: Asynchronous inference
    Don't block - continue processing while AI runs
    """
    print("=" * 70)
    print("Example 2: Asynchronous Inference")
    print("=" * 70)
    
    core = CIFCore()
    
    # Create dummy model
    from backend.cms.cif.cif_core import CIFModel
    dummy_model = CIFModel(
        model_data={'weights': np.random.randn(10, 10)},
        metadata={'name': 'async_demo'},
        input_spec={'sensor_data': (1, 10)},
        output_spec={'prediction': (1, 1)},
        optimization_config={}
    )
    
    compiled = core.compile_model(dummy_model)
    
    # Create async request
    request = compiled.create_infer_request()
    
    # Start async inference
    print("Starting async inference...")
    request.start_async({'sensor_data': np.random.randn(1, 10)})
    
    # Do other work while inference runs
    print("Doing other work while AI thinks...")
    import time
    time.sleep(0.01)  # Simulate other processing
    
    # Check if complete
    if request.is_complete():
        print("Inference complete!")
        result = request.wait()
        print(f"Result: {result}")
    else:
        print("Still processing, waiting...")
        result = request.wait(timeout=5.0)
        print(f"Result: {result}")
    
    print()


def example_3_model_conversion():
    """
    Example 3: Convert PyTorch model to CIF format
    """
    print("=" * 70)
    print("Example 3: Model Conversion")
    print("=" * 70)
    
    # For demo purposes, create a simple PyTorch model
    try:
        import torch
        import torch.nn as nn
        
        # Simple neural network
        class SimpleNN(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(10, 20)
                self.fc2 = nn.Linear(20, 1)
                self.relu = nn.ReLU()
            
            def forward(self, x):
                x = self.relu(self.fc1(x))
                x = self.fc2(x)
                return x
        
        # Create and save model
        model = SimpleNN()
        torch.save(model, 'temp_demo_model.pt')
        
        # Convert to CIF
        print("Converting PyTorch model to CIF format...")
        converter = CIFModelConverter('temp_demo_model.pt', 'pytorch')
        
        cif_model = converter.optimize(
            quantization='INT8',  # 4x compression
            pruning_threshold=0.01,  # Remove 1% smallest weights
            target_device='CPU'
        )
        
        # Save CIF model
        cif_model.save('demo_model.cif')
        print(f"Model saved: demo_model.cif")
        print(f"Model size: {cif_model.size / 1024:.1f} KB")
        print(f"Estimated FLOPs: {cif_model.ops:,}")
        
        # Clean up
        import os
        os.remove('temp_demo_model.pt')
        
    except ImportError:
        print("PyTorch not installed - skipping conversion demo")
    
    print()


def example_4_heterogeneous_pipeline():
    """
    Example 4: Heterogeneous computing pipeline
    Different tasks on different devices
    """
    print("=" * 70)
    print("Example 4: Heterogeneous Pipeline")
    print("=" * 70)
    
    # Define processing functions
    def preprocess(inputs):
        """Preprocessing on CPU"""
        data = inputs['raw_data']
        processed = data * 2.0  # Simple transformation
        return {'processed': processed}
    
    def transform(inputs):
        """Heavy compute on GPU (or CPU if no GPU)"""
        data = inputs['processed']
        # FFT or other heavy operation
        transformed = np.fft.fft(data)
        magnitude = np.abs(transformed)
        return {'features': magnitude}
    
    def classify(inputs):
        """Decision logic on CPU"""
        features = inputs['features']
        decision = 'OK' if np.mean(features) < 100 else 'ANOMALY'
        return {'decision': decision, 'confidence': 0.95}
    
    # Create pipeline
    pipeline = Pipeline([
        ('preprocess', 'CPU', preprocess),
        ('transform', 'GPU', transform),  # Falls back to CPU if no GPU
        ('classify', 'CPU', classify)
    ])
    
    # Execute pipeline
    input_data = {'raw_data': np.random.randn(1024)}
    
    print("Executing heterogeneous pipeline...")
    result = pipeline(input_data)
    
    print(f"Decision: {result['decision']}")
    print(f"Confidence: {result['confidence']}")
    
    # Performance report
    print()
    pipeline.print_performance_report()


def example_5_manufacturing_use_case():
    """
    Example 5: Real manufacturing scenario
    Tool wear prediction during machining
    """
    print("=" * 70)
    print("Example 5: Tool Wear Prediction")
    print("=" * 70)
    
    # Simulate sensor data
    def simulate_sensors():
        """Simulate real-time sensor readings"""
        return {
            'vibration_x': np.random.randn(1000) * 0.5,
            'vibration_y': np.random.randn(1000) * 0.5,
            'vibration_z': np.random.randn(1000) * 0.5,
            'spindle_load': np.random.uniform(50, 80),
            'feed_rate': 500,  # mm/min
            'temperature': np.random.uniform(25, 35)
        }
    
    # Feature extraction
    def extract_features(inputs):
        """Extract features from raw sensor data"""
        vib_x = inputs['vibration_x']
        vib_y = inputs['vibration_y']
        vib_z = inputs['vibration_z']
        
        features = {
            'rms_x': np.sqrt(np.mean(vib_x**2)),
            'rms_y': np.sqrt(np.mean(vib_y**2)),
            'rms_z': np.sqrt(np.mean(vib_z**2)),
            'peak_x': np.max(np.abs(vib_x)),
            'peak_y': np.max(np.abs(vib_y)),
            'peak_z': np.max(np.abs(vib_z)),
            'spindle_load': inputs['spindle_load'],
            'temperature': inputs['temperature']
        }
        
        return {'ml_features': np.array(list(features.values()))}
    
    # Prediction (simplified - would use actual model)
    def predict_wear(inputs):
        """Predict tool wear"""
        features = inputs['ml_features']
        
        # Simplified prediction
        wear_estimate = np.mean(features) * 0.01  # mm
        
        return {
            'wear_mm': wear_estimate,
            'remaining_life_pct': max(0, 100 - wear_estimate * 100)
        }
    
    # Create pipeline
    pipeline = Pipeline([
        ('feature_extraction', 'CPU', extract_features),
        ('prediction', 'CPU', predict_wear)  # Would be Edge TPU in production
    ])
    
    # Simulate machining cycle
    print("Simulating 5 measurement cycles...")
    for cycle in range(5):
        sensor_data = simulate_sensors()
        result = pipeline(sensor_data)
        
        print(f"Cycle {cycle+1}: Wear={result['wear_mm']:.3f}mm, "
              f"Life={result['remaining_life_pct']:.1f}%")
    
    print()
    pipeline.print_performance_report()


def run_all_examples():
    """Run all examples"""
    examples = [
        example_1_basic_inference,
        example_2_async_inference,
        example_3_model_conversion,
        example_4_heterogeneous_pipeline,
        example_5_manufacturing_use_case
    ]
    
    for example in examples:
        try:
            example()
        except Exception as e:
            print(f"Error in {example.__name__}: {e}")
            print()
    
    print("=" * 70)
    print("All examples complete!")
    print("=" * 70)


if __name__ == "__main__":
    # Run all examples
    run_all_examples()
