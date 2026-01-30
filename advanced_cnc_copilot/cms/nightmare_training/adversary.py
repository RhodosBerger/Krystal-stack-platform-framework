"""
Nightmare Training - Adversary Component
Injects synthetic failures into clean telemetry logs for offline learning
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import random
from datetime import datetime, timedelta


class FailureType(Enum):
    """Types of synthetic failures that can be injected"""
    SPINDLE_LOAD_SPIKE = "spindle_load_spike"
    THERMAL_RUNAWAY = "thermal_runaway"
    PHANTOM_TRAUMA = "phantom_trauma"
    VIBRATION_ANOMALY = "vibration_anomaly"
    COOLANT_FAILURE = "coolant_failure"
    TOOL_BREAKAGE = "tool_breakage"


@dataclass
class FailureInjectionConfig:
    """Configuration for failure injection"""
    failure_type: FailureType
    severity: float  # 0.0 to 1.0 scale
    duration: int  # Duration in number of data points
    start_time_index: int  # Index in the telemetry data where failure starts
    probability: float = 0.5  # Probability of this failure occurring


class Adversary:
    """
    The Adversary component of Nightmare Training.
    Injects synthetic failures into clean telemetry logs to create training scenarios.
    """
    
    def __init__(self):
        self.failure_configs = {
            FailureType.SPINDLE_LOAD_SPIKE: self._inject_spindle_load_spike,
            FailureType.THERMAL_RUNAWAY: self._inject_thermal_runaway,
            FailureType.PHANTOM_TRAUMA: self._inject_phantom_trauma,
            FailureType.VIBRATION_ANOMALY: self._inject_vibration_anomaly,
            FailureType.COOLANT_FAILURE: self._inject_coolant_failure,
            FailureType.TOOL_BREAKAGE: self._inject_tool_breakage,
        }
        
    def inject_failures(self, telemetry_data: pd.DataFrame, 
                      configs: List[FailureInjectionConfig]) -> pd.DataFrame:
        """
        Inject multiple failure scenarios into the telemetry data
        
        Args:
            telemetry_data: Original clean telemetry data
            configs: List of failure injection configurations
            
        Returns:
            DataFrame with injected failures
        """
        modified_data = telemetry_data.copy()
        
        for config in configs:
            if random.random() < config.probability:
                failure_func = self.failure_configs[config.failure_type]
                modified_data = failure_func(modified_data, config)
                
        return modified_data
    
    def _inject_spindle_load_spike(self, data: pd.DataFrame, 
                                  config: FailureInjectionConfig) -> pd.DataFrame:
        """
        Inject spindle load spike (tool breakage scenario)
        Spindle load jumps to 200% instantly
        """
        modified_data = data.copy()
        start_idx = config.start_time_index
        duration = config.duration
        severity = config.severity
        
        # Calculate the spike magnitude (up to 200% of normal)
        max_spike = 200.0  # 200% of normal load
        spike_magnitude = max_spike * severity
        
        # Apply the spike over the specified duration
        end_idx = min(start_idx + duration, len(modified_data))
        
        for i in range(start_idx, end_idx):
            if 'spindle_load' in modified_data.columns:
                original_load = modified_data.iloc[i]['spindle_load']
                new_load = min(100.0, original_load + (spike_magnitude * (i - start_idx + 1) / duration))
                modified_data.at[modified_data.index[i], 'spindle_load'] = new_load
                
        return modified_data
    
    def _inject_thermal_runaway(self, data: pd.DataFrame, 
                               config: FailureInjectionConfig) -> pd.DataFrame:
        """
        Inject thermal runaway (coolant failure scenario)
        Temperature rises 10°C/sec
        """
        modified_data = data.copy()
        start_idx = config.start_time_index
        duration = config.duration
        severity = config.severity
        
        # Calculate temperature rise rate (up to 10°C/sec scaled by severity)
        temp_rise_rate = 10.0 * severity  # degrees per time interval
        
        # Apply the temperature increase over the specified duration
        end_idx = min(start_idx + duration, len(modified_data))
        
        for i in range(start_idx, end_idx):
            if 'temperature' in modified_data.columns:
                original_temp = modified_data.iloc[i]['temperature']
                time_factor = (i - start_idx + 1)  # Incremental factor
                new_temp = original_temp + (temp_rise_rate * time_factor)
                modified_data.at[modified_data.index[i], 'temperature'] = new_temp
                
        return modified_data
    
    def _inject_phantom_trauma(self, data: pd.DataFrame, 
                              config: FailureInjectionConfig) -> pd.DataFrame:
        """
        Inject phantom trauma (sensor drift scenario)
        High vibration with low load
        """
        modified_data = data.copy()
        start_idx = config.start_time_index
        duration = config.duration
        severity = config.severity
        
        # Apply high vibration with low load over the specified duration
        end_idx = min(start_idx + duration, len(modified_data))
        
        for i in range(start_idx, end_idx):
            if 'vibration_x' in modified_data.columns:
                original_vibration = modified_data.iloc[i]['vibration_x']
                new_vibration = min(10.0, original_vibration + (2.0 * severity))
                modified_data.at[modified_data.index[i], 'vibration_x'] = new_vibration
            
            if 'spindle_load' in modified_data.columns:
                original_load = modified_data.iloc[i]['spindle_load']
                # Reduce load to simulate phantom trauma
                new_load = max(0.0, original_load * (1.0 - 0.3 * severity))
                modified_data.at[modified_data.index[i], 'spindle_load'] = new_load
                
        return modified_data
    
    def _inject_vibration_anomaly(self, data: pd.DataFrame, 
                                 config: FailureInjectionConfig) -> pd.DataFrame:
        """
        Inject vibration anomaly (bearing wear, imbalance, etc.)
        """
        modified_data = data.copy()
        start_idx = config.start_time_index
        duration = config.duration
        severity = config.severity
        
        # Apply vibration anomaly over the specified duration
        end_idx = min(start_idx + duration, len(modified_data))
        
        for i in range(start_idx, end_idx):
            if 'vibration_x' in modified_data.columns:
                original_vibration = modified_data.iloc[i]['vibration_x']
                # Add anomalous vibration based on severity
                anomaly = 1.5 * severity * (i - start_idx + 1) / duration
                new_vibration = min(5.0, original_vibration + anomaly)
                modified_data.at[modified_data.index[i], 'vibration_x'] = new_vibration
            
            if 'vibration_y' in modified_data.columns:
                original_vibration = modified_data.iloc[i]['vibration_y']
                anomaly = 1.5 * severity * (i - start_idx + 1) / duration
                new_vibration = min(5.0, original_vibration + anomaly)
                modified_data.at[modified_data.index[i], 'vibration_y'] = new_vibration
                
        return modified_data
    
    def _inject_coolant_failure(self, data: pd.DataFrame, 
                               config: FailureInjectionConfig) -> pd.DataFrame:
        """
        Inject coolant failure scenario
        Temperature increases while coolant flow decreases
        """
        modified_data = data.copy()
        start_idx = config.start_time_index
        duration = config.duration
        severity = config.severity
        
        # Apply coolant failure over the specified duration
        end_idx = min(start_idx + duration, len(modified_data))
        
        for i in range(start_idx, end_idx):
            if 'temperature' in modified_data.columns:
                original_temp = modified_data.iloc[i]['temperature']
                temp_increase = 5.0 * severity * (i - start_idx + 1) / duration
                new_temp = original_temp + temp_increase
                modified_data.at[modified_data.index[i], 'temperature'] = new_temp
            
            if 'coolant_flow' in modified_data.columns:
                original_flow = modified_data.iloc[i]['coolant_flow']
                flow_decrease = original_flow * 0.8 * severity
                new_flow = max(0.0, original_flow - flow_decrease)
                modified_data.at[modified_data.index[i], 'coolant_flow'] = new_flow
                
        return modified_data
    
    def _inject_tool_breakage(self, data: pd.DataFrame, 
                            config: FailureInjectionConfig) -> pd.DataFrame:
        """
        Inject tool breakage scenario
        Combines multiple symptoms: high load, high vibration, temperature increase
        """
        modified_data = data.copy()
        start_idx = config.start_time_index
        duration = config.duration
        severity = config.severity
        
        # Apply tool breakage symptoms over the specified duration
        end_idx = min(start_idx + duration, len(modified_data))
        
        for i in range(start_idx, end_idx):
            # Increase spindle load dramatically
            if 'spindle_load' in modified_data.columns:
                original_load = modified_data.iloc[i]['spindle_load']
                load_increase = 50.0 * severity * (i - start_idx + 1) / duration
                new_load = min(100.0, original_load + load_increase)
                modified_data.at[modified_data.index[i], 'spindle_load'] = new_load
            
            # Increase vibration significantly
            if 'vibration_x' in modified_data.columns:
                original_vibration = modified_data.iloc[i]['vibration_x']
                vibration_increase = 2.0 * severity * (i - start_idx + 1) / duration
                new_vibration = min(5.0, original_vibration + vibration_increase)
                modified_data.at[modified_data.index[i], 'vibration_x'] = new_vibration
            
            if 'vibration_y' in modified_data.columns:
                original_vibration = modified_data.iloc[i]['vibration_y']
                vibration_increase = 2.0 * severity * (i - start_idx + 1) / duration
                new_vibration = min(5.0, original_vibration + vibration_increase)
                modified_data.at[modified_data.index[i], 'vibration_y'] = new_vibration
            
            # Increase temperature
            if 'temperature' in modified_data.columns:
                original_temp = modified_data.iloc[i]['temperature']
                temp_increase = 8.0 * severity * (i - start_idx + 1) / duration
                new_temp = original_temp + temp_increase
                modified_data.at[modified_data.index[i], 'temperature'] = new_temp
                
        return modified_data
    
    def generate_random_failures(self, telemetry_data: pd.DataFrame, 
                                num_scenarios: int = 5) -> List[Tuple[pd.DataFrame, List[str]]]:
        """
        Generate multiple random failure scenarios from clean telemetry data
        
        Args:
            telemetry_data: Original clean telemetry data
            num_scenarios: Number of failure scenarios to generate
            
        Returns:
            List of tuples containing (modified_data, list_of_applied_failures)
        """
        scenarios = []
        
        for _ in range(num_scenarios):
            # Randomly select a few failure types for this scenario
            selected_failures = random.sample(list(FailureType), 
                                           k=random.randint(1, 3))
            
            configs = []
            applied_failures = []
            
            for failure_type in selected_failures:
                # Random parameters for this failure
                severity = random.uniform(0.3, 1.0)
                duration = random.randint(5, 50)  # 5-50 data points
                start_time_index = random.randint(0, len(telemetry_data) - duration - 1)
                probability = 1.0  # Always apply for random generation
                
                config = FailureInjectionConfig(
                    failure_type=failure_type,
                    severity=severity,
                    duration=duration,
                    start_time_index=start_time_index,
                    probability=probability
                )
                configs.append(config)
                applied_failures.append(failure_type.value)
            
            # Apply all failures to create the scenario
            modified_data = self.inject_failures(telemetry_data, configs)
            scenarios.append((modified_data, applied_failures))
        
        return scenarios


def create_sample_telemetry_data(num_points: int = 1000) -> pd.DataFrame:
    """
    Create sample telemetry data for testing the Adversary
    """
    timestamps = [datetime.now() - timedelta(seconds=i*10) for i in range(num_points)]
    timestamps.reverse()
    
    # Generate realistic baseline telemetry data
    base_spindle_load = np.random.normal(60, 10, num_points)
    base_temperature = np.random.normal(35, 5, num_points)
    base_vibration_x = np.random.normal(0.5, 0.2, num_points)
    base_vibration_y = np.random.normal(0.5, 0.2, num_points)
    base_coolant_flow = np.random.normal(2.0, 0.5, num_points)
    
    # Ensure values are within reasonable bounds
    base_spindle_load = np.clip(base_spindle_load, 0, 100)
    base_temperature = np.clip(base_temperature, 20, 80)
    base_vibration_x = np.clip(base_vibration_x, 0, 3)
    base_vibration_y = np.clip(base_vibration_y, 0, 3)
    base_coolant_flow = np.clip(base_coolant_flow, 0.5, 5)
    
    df = pd.DataFrame({
        'timestamp': timestamps,
        'spindle_load': base_spindle_load,
        'temperature': base_temperature,
        'vibration_x': base_vibration_x,
        'vibration_y': base_vibration_y,
        'coolant_flow': base_coolant_flow,
        'feed_rate': np.random.normal(2000, 500, num_points),
        'rpm': np.random.normal(4000, 1000, num_points)
    })
    
    return df


# Example usage and testing
if __name__ == "__main__":
    # Create sample telemetry data
    sample_data = create_sample_telemetry_data(500)
    print("Original data shape:", sample_data.shape)
    print("Original data stats:")
    print(sample_data[['spindle_load', 'temperature', 'vibration_x']].describe())
    
    # Initialize the Adversary
    adversary = Adversary()
    
    # Create specific failure configurations
    configs = [
        FailureInjectionConfig(
            failure_type=FailureType.SPINDLE_LOAD_SPIKE,
            severity=0.8,
            duration=20,
            start_time_index=100,
            probability=1.0
        ),
        FailureInjectionConfig(
            failure_type=FailureType.THERMAL_RUNAWAY,
            severity=0.6,
            duration=30,
            start_time_index=200,
            probability=1.0
        ),
        FailureInjectionConfig(
            failure_type=FailureType.PHANTOM_TRAUMA,
            severity=0.7,
            duration=25,
            start_time_index=300,
            probability=1.0
        )
    ]
    
    # Inject failures
    modified_data = adversary.inject_failures(sample_data, configs)
    
    print("\nModified data stats:")
    print(modified_data[['spindle_load', 'temperature', 'vibration_x']].describe())
    
    # Show the differences at injection points
    print(f"\nSpindle load at injection point (idx 100): {modified_data.iloc[100]['spindle_load']:.2f}")
    print(f"Temperature at injection point (idx 200): {modified_data.iloc[200]['temperature']:.2f}")
    print(f"Vibration X at injection point (idx 300): {modified_data.iloc[300]['vibration_x']:.2f}")
    
    # Generate random scenarios
    print("\nGenerating random failure scenarios...")
    random_scenarios = adversary.generate_random_failures(sample_data, 3)
    
    for i, (scenario_data, applied_failures) in enumerate(random_scenarios):
        print(f"Scenario {i+1}: Applied failures: {applied_failures}")
        print(f"  Data shape: {scenario_data.shape}")
        print(f"  Modified points: {len(scenario_data[scenario_data != sample_data].dropna(how='all'))}")