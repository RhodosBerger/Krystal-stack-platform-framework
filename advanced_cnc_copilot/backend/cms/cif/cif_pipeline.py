"""
CIF Heterogeneous Pipeline
Inspired by oneAPI task graphs

Distribute computation across multiple devices:
- FPGA for preprocessing (ultra-low latency)
- GPU for heavy compute (FFT, matrix ops)
- Edge TPU for ML inference (power efficient)
- CPU for control logic (sequential)
"""

from typing import List, Tuple, Dict, Any, Callable, Optional
from dataclasses import dataclass
import time

from .cif_core import CIFCore


@dataclass
class PipelineStage:
    """
    Single stage in processing pipeline
    
    Args:
        name: Stage identifier
        device: Target device ('CPU', 'GPU', 'EDGE_TPU', 'FPGA')
        function: Processing function or model
        inputs: List of input names
        outputs: List of output names
    """
    name: str
    device: str
    function: Any
    inputs: List[str]
    outputs: List[str]


class Pipeline:
    """
    Heterogeneous computing pipeline
    
    Distributes tasks across optimal hardware similar to oneAPI task graphs
    """
    
    def __init__(self, stages: Optional[List[Tuple[str, str, Any]]] = None):
        """
        Create pipeline
        
        Args:
            stages: List of (name, device, function) tuples
        """
        self.stages: List[PipelineStage] = []
        self.core = CIFCore()
        
        # Statistics
        self.execution_count = 0
        self.total_time = 0.0
        self.stage_times: Dict[str, float] = {}
        
        if stages:
            for name, device, func in stages:
                self.add_stage(name, device, func)
    
    def add_stage(self, 
                  name: str,
                  device: str,
                  function: Any,
                  inputs: Optional[List[str]] = None,
                  outputs: Optional[List[str]] = None):
        """
        Add stage to pipeline
        
        Args:
            name: Stage name
            device: Target device
            function: Processing function or CIF model
            inputs: Input tensor names (auto-inferred if None)
            outputs: Output tensor names (auto-inferred if None)
        """
        stage = PipelineStage(
            name=name,
            device=device,
            function=function,
            inputs=inputs or ['input'],
            outputs=outputs or ['output']
        )
        
        self.stages.append(stage)
        self.stage_times[name] = 0.0
    
    def execute(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute pipeline on input data
        
        Args:
            data: Input dictionary
        
        Returns:
            Output dictionary from final stage
        """
        start_time = time.time()
        
        current_data = data.copy()
        
        # Execute each stage sequentially
        for stage in self.stages:
            stage_start = time.time()
            
            # Extract inputs for this stage
            stage_inputs = {
                k: current_data[k] 
                for k in stage.inputs 
                if k in current_data
            }
            
            # Execute on target device
            if isinstance(stage.function, str):
                # It's a CIF model path
                model = self.core.load_model(stage.function)
                compiled = self.core.compile_model(model, device=stage.device)
                stage_outputs = compiled(stage_inputs)
            
            elif callable(stage.function):
                # It's a Python function
                stage_outputs = stage.function(stage_inputs)
            
            else:
                raise ValueError(f"Invalid function type for stage {stage.name}")
            
            # Merge outputs into current data
            current_data.update(stage_outputs)
            
            # Track stage time
            stage_time = time.time() - stage_start
            self.stage_times[stage.name] += stage_time
        
        # Track total time
        total_time = time.time() - start_time
        self.total_time += total_time
        self.execution_count += 1
        
        return current_data
    
    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Allow pipeline() syntax"""
        return self.execute(data)
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        stats = {
            'total_executions': self.execution_count,
            'total_time': self.total_time,
            'average_time': self.total_time / max(self.execution_count, 1),
            'throughput': self.execution_count / max(self.total_time, 0.001),
            'stage_times': {}
        }
        
        for name, total_time in self.stage_times.items():
            stats['stage_times'][name] = {
                'total': total_time,
                'average': total_time / max(self.execution_count, 1),
                'percentage': (total_time / max(self.total_time, 0.001)) * 100
            }
        
        return stats
    
    def print_performance_report(self):
        """Print detailed performance report"""
        stats = self.get_performance_stats()
        
        print("=" * 70)
        print("Pipeline Performance Report")
        print("=" * 70)
        print(f"Total Executions: {stats['total_executions']}")
        print(f"Total Time: {stats['total_time']:.3f}s")
        print(f"Average Time: {stats['average_time']*1000:.2f}ms")
        print(f"Throughput: {stats['throughput']:.1f} inferences/sec")
        print()
        print("Stage Breakdown:")
        print("-" * 70)
        
        for stage_name, stage_stats in stats['stage_times'].items():
            # Find device for this stage
            device = next(
                (s.device for s in self.stages if s.name == stage_name),
                'Unknown'
            )
            
            print(f"{stage_name:20s} [{device:10s}] "
                  f"{stage_stats['average']*1000:7.2f}ms  "
                  f"({stage_stats['percentage']:5.1f}%)")
        
        print("=" * 70)


class AsyncPipeline(Pipeline):
    """
    Asynchronous pipeline with parallel stage execution
    
    Can execute independent stages in parallel
    """
    
    def __init__(self, stages: Optional[List[Tuple[str, str, Any]]] = None):
        super().__init__(stages)
        self.parallel_execution = True
    
    def execute(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute with parallelism where possible
        
        TODO: Implement dependency graph analysis and parallel execution
        For now, falls back to sequential
        """
        # Would analyze stage dependencies and execute in parallel
        # using threading or asyncio
        
        return super().execute(data)


# Example pipeline builders
def create_vibration_analysis_pipeline() -> Pipeline:
    """
    Create pipeline for vibration analysis
    
    Stages:
    1. FPGA: Sensor preprocessing (0.5ms)
    2. GPU: FFT transformation (2ms)
    3. Edge TPU: Anomaly detection (3ms)
    4. CPU: Decision logic (1ms)
    """
    def preprocess(inputs):
        # Simplified preprocessing
        import numpy as np
        data = inputs['raw_signal']
        # Remove DC offset, apply window
        processed = data - np.mean(data)
        return {'preprocessed': processed}
    
    def fft_transform(inputs):
        import numpy as np
        signal = inputs['preprocessed']
        spectrum = np.fft.fft(signal)
        magnitude = np.abs(spectrum)
        return {'spectrum': magnitude}
    
    def decision_logic(inputs):
        anomaly_score = inputs.get('anomaly_score', 0)
        
        if anomaly_score > 0.8:
            action = 'STOP_MACHINE'
        elif anomaly_score > 0.5:
            action = 'WARNING'
        else:
            action = 'OK'
        
        return {'action': action, 'score': anomaly_score}
    
    pipeline = Pipeline([
        ('preprocess', 'CPU', preprocess),  # Would be FPGA in production
        ('fft', 'GPU', fft_transform),
        # ('anomaly_detection', 'EDGE_TPU', 'anomaly_model.cif'),  # Would load model
        ('decision', 'CPU', decision_logic)
    ])
    
    return pipeline


def create_quality_inspection_pipeline() -> Pipeline:
    """
    Create pipeline for visual quality inspection
    
    Stages:
    1. CPU: Image preprocessing
    2. GPU: Defect detection CNN
    3. CPU: Classification and reporting
    """
    def preprocess_image(inputs):
        import numpy as np
        image = inputs['image']
        # Resize, normalize
        processed = image / 255.0
        return {'processed_image': processed}
    
    def generate_report(inputs):
        defects = inputs.get('defects', [])
        
        report = {
            'defect_count': len(defects),
            'pass': len(defects) == 0,
            'defect_types': [d['type'] for d in defects]
        }
        
        return {'report': report}
    
    pipeline = Pipeline([
        ('preprocess', 'CPU', preprocess_image),
        # ('detect', 'GPU', 'defect_detector.cif'),  # Would load model
        ('report', 'CPU', generate_report)
    ])
    
    return pipeline
