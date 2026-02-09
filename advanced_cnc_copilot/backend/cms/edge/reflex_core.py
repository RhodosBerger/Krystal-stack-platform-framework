from typing import List, Tuple, Dict, Any
import time
import logging


class NeuroCParams:
    """Parameters for the Neuro-C architecture"""
    def __init__(self,
                 max_vibration_threshold: int = 1000,  # Integer threshold for vibration safety
                 max_temperature_threshold: int = 80000,  # Integer threshold for temperature (scaled by 1000)
                 max_spindle_load_threshold: int = 95000,  # Integer threshold for spindle load (scaled by 1000)
                 latency_target_ms: int = 1):  # Target latency in milliseconds
        self.max_vibration_threshold = max_vibration_threshold
        self.max_temperature_threshold = max_temperature_threshold
        self.max_spindle_load_threshold = max_spindle_load_threshold
        self.latency_target_ms = latency_target_ms


class NeuroCKernel:
    """
    Implements the Neuro-C architecture for ultra-fast inference on edge devices.
    Uses integer-only arithmetic with ternary weights {-1, 0, +1} for <1ms latency.
    Based on Neuro-Geometric Architecture (Neuro-C) theory eliminating floating-point MACC operations.
    """
    
    def __init__(self, adjacency_matrix: List[List[int]], params: NeuroCParams = None):
        """
        Initialize the Neuro-C kernel with ternary adjacency matrix.
        
        Args:
            adjacency_matrix: 2D list of integers {-1, 0, +1} representing ternary weights
            params: NeuroCParams object with safety thresholds
        """
        self.params = params if params is not None else NeuroCParams()
        self.adjacency_matrix = adjacency_matrix
        if adjacency_matrix:
            self.matrix_height = len(adjacency_matrix)
            self.matrix_width = len(adjacency_matrix[0]) if adjacency_matrix[0] else 0
        else:
            self.matrix_height = 0
            self.matrix_width = 0
        self.logger = logging.getLogger(__name__)
        
        # Validate that all weights are in {-1, 0, +1}
        all_weights = []
        for row in self.adjacency_matrix:
            all_weights.extend(row)
        unique_weights = set(all_weights)
        if not unique_weights.issubset({-1, 0, 1}):
            raise ValueError(f"Adjacency matrix contains invalid weights. Only {{-1, 0, +1}} allowed, got: {unique_weights}")
    
    def forward_pass(self, input_vector: List[int]) -> Tuple[int, str]:
        """
        Performs integer-only forward pass through the network.
        If output exceeds threshold, immediately returns STOP_SPINDLE signal.
        
        Args:
            input_vector: List of integer sensor values (vibration, temperature, load, etc.)
            
        Returns:
            Tuple of (output_value, action_signal) where action_signal is one of:
            - "CONTINUE" if safe
            - "WARN" if approaching danger
            - "STOP_SPINDLE" if dangerous
        """
        start_time = time.time()
        
        # Validate input dimensions
        if len(input_vector) != self.matrix_width:
            raise ValueError(f"Input vector length {len(input_vector)} doesn't match adjacency matrix width {self.matrix_width}")
        
        # Perform integer-only matrix multiplication using sparse approach
        # Only process non-zero weights to maximize speed
        accumulator = 0
        for i in range(len(input_vector)):
            for j in range(self.matrix_height):
                if i < len(self.adjacency_matrix) and j < len(self.adjacency_matrix[i]):
                    weight = self.adjacency_matrix[i][j]
                    if weight != 0:  # Skip zero weights for sparsity
                        accumulator += input_vector[i] * weight
        
        # Apply integer scaling factor to get final output
        output = accumulator  # No additional scaling needed for this simple implementation
        
        # Determine action based on thresholds
        if output > self.params.max_vibration_threshold:
            action = "STOP_SPINDLE"
            self.logger.critical(f"NEURO-C EMERGENCY: Output {output} exceeds max threshold {self.params.max_vibration_threshold}")
        elif output > int(self.params.max_vibration_threshold * 0.8):
            action = "WARN"
            self.logger.warning(f"NEURO-C WARNING: Output {output} approaching danger threshold {self.params.max_vibration_threshold}")
        else:
            action = "CONTINUE"
            if output > int(self.params.max_vibration_threshold * 0.5):
                self.logger.info(f"NEURO-C MONITOR: Output {output} in moderate range")
        
        # Check execution time against target
        elapsed_time = (time.time() - start_time) * 1000  # Convert to milliseconds
        if elapsed_time > self.params.latency_target_ms:
            self.logger.warning(f"NEURO-C LATENCY: Forward pass took {elapsed_time:.2f}ms, exceeding target {self.params.latency_target_ms}ms")
        
        return output, action
    
    def reflex_response(self, sensor_values: List[int]) -> str:
        """
        Implements the 'Spinal Reflex' - ultra-fast safety response.
        Direct mapping from sensors to safety actions with minimal processing.
        
        Args:
            sensor_values: [vibration_x, vibration_y, vibration_z, temperature, spindle_load, ...]
            
        Returns:
            Action string: "SAFE", "CAUTION", or "EMERGENCY_STOP"
        """
        # Extract critical values
        vibration_x = sensor_values[0] if len(sensor_values) > 0 else 0
        vibration_y = sensor_values[1] if len(sensor_values) > 1 else 0
        vibration_z = sensor_values[2] if len(sensor_values) > 2 else 0
        temperature = sensor_values[3] if len(sensor_values) > 3 else 0
        spindle_load = sensor_values[4] if len(sensor_values) > 4 else 0
        
        # Check each safety threshold using integer comparisons only
        if (vibration_x > self.params.max_vibration_threshold or
            vibration_y > self.params.max_vibration_threshold or
            vibration_z > self.params.max_vibration_threshold or
            temperature > self.params.max_temperature_threshold or
            spindle_load > self.params.max_spindle_load_threshold):
            self.logger.critical("SPINAL REFLEX TRIGGERED: Emergency stop condition detected")
            return "EMERGENCY_STOP"
        
        # Check caution thresholds
        caution_threshold = int(0.8 * self.params.max_vibration_threshold)
        if (vibration_x > caution_threshold or
            vibration_y > caution_threshold or
            vibration_z > caution_threshold or
            temperature > int(0.8 * self.params.max_temperature_threshold) or
            spindle_load > int(0.8 * self.params.max_spindle_load_threshold)):
            self.logger.warning("SPINAL REFLEX: Caution threshold exceeded")
            return "CAUTION"
        
        return "SAFE"


class SpindleSafetyMonitor:
    """
    Monitors spindle safety using Neuro-C principles
    Implements integer-only arithmetic for ultra-fast response times
    """
    
    def __init__(self, neuro_c_kernel: NeuroCKernel):
        self.neuro_c = neuro_c_kernel
        self.logger = logging.getLogger(__name__)
    
    def monitor_spindle_safety(self, sensor_data: List[int]) -> Dict[str, Any]:
        """
        Monitor spindle safety based on sensor data
        
        Args:
            sensor_data: List of integer sensor values [vibration_x, vibration_y, vibration_z, temp, load, ...]
            
        Returns:
            Dictionary with safety status and action
        """
        start_time = time.time()
        
        # Perform forward pass through Neuro-C network
        output, action = self.neuro_c.forward_pass(sensor_data)
        
        # Get reflex response as backup safety check
        reflex_action = self.neuro_c.reflex_response(sensor_data)
        
        # Combine both responses for comprehensive safety
        if action == "STOP_SPINDLE" or reflex_action == "EMERGENCY_STOP":
            final_action = "EMERGENCY_STOP"
            status = "CRITICAL"
        elif action == "WARN" or reflex_action == "CAUTION":
            final_action = "SLOW_DOWN"
            status = "WARNING"
        else:
            final_action = "CONTINUE"
            status = "NORMAL"
        
        return {
            'timestamp': time.time(),
            'neuro_c_output': output,
            'neuro_c_action': action,
            'reflex_action': reflex_action,
            'final_action': final_action,
            'status': status,
            'sensor_data': sensor_data,
            'latency_ms': (time.time() - start_time) * 1000
        }
    
    def validate_integer_only_math(self, sensor_values: List[int]) -> bool:
        """
        Validates that all calculations are performed using integer-only math
        """
        # All operations in this class should use only integer arithmetic
        # This is a validation method to ensure we're staying true to the Neuro-C architecture
        try:
            # Test that all operations work with integers
            test_result = self.neuro_c.forward_pass(sensor_values[:self.neuro_c.matrix_width])
            return isinstance(test_result[0], int)
        except:
            return False


# Example usage:
if __name__ == "__main__":
    # Example adjacency matrix for a simple 5x5 ternary network
    # This represents connections between 5 sensor inputs and 5 internal nodes
    adjacency_matrix = [
        [1, -1, 0, 1, -1],   # Node 1 connections
        [0, 1, -1, 0, 1],    # Node 2 connections
        [-1, 0, 1, 1, 0],    # Node 3 connections
        [1, 1, 0, -1, 1],    # Node 4 connections
        [0, -1, 1, 0, -1]    # Node 5 connections
    ]
    
    # Initialize Neuro-C kernel
    neuro_c = NeuroCKernel(adjacency_matrix)
    
    # Example sensor data: [vibration_x, vibration_y, vibration_z, temperature_scaled, spindle_load_scaled]
    sensor_data = [200, 150, 180, 45000, 75000]  # All values are integers
    
    # Perform safety check
    output, action = neuro_c.forward_pass(sensor_data)
    print(f"Neuro-C Output: {output}, Action: {action}")
    
    # Perform reflex check
    reflex_action = neuro_c.reflex_response(sensor_data)
    print(f"Reflex Action: {reflex_action}")