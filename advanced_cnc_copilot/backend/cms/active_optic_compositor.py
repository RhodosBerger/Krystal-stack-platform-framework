"""
Active Optic Compositor
KrystalStack Core Module

PARADIGM: Treat data streams as Visual Fields
Calculate Entropy (Chaos) and Coherence (Order/Health)

METAPHOR: Like analyzing video for quality issues,
we analyze data streams for operational issues
"""

import numpy as np
from typing import Dict, Tuple, Optional
from dataclasses import dataclass
from scipy import fft
from scipy.stats import entropy as scipy_entropy


@dataclass
class EntropyMetrics:
    """
    Entropy metrics for a visual/data field
    
    Attributes:
        spatial_entropy: Chaos in space (variance across positions)
        temporal_entropy: Chaos in time (variance across frames)
        frequency_entropy: Chaos in frequency domain
        total_entropy: Combined entropy metric (0.0-1.0)
        coherence: Inverse of entropy (1.0 = perfect order)
    """
    spatial_entropy: float
    temporal_entropy: float
    frequency_entropy: float
    total_entropy: float
    coherence: float


class ActiveOpticCompositor:
    """
    Calculates visual/data entropy for health monitoring
    
    Uses multi-scale analysis:
    - Spatial: How chaotic is the current frame?
    - Temporal: How chaotic is the change over time?
    - Frequency: Are there unexpected oscillations?
    
    HIGH ENTROPY → System degrading/failing
    LOW ENTROPY → System stable/healthy
    """
    
    def __init__(self):
        """Initialize compositor"""
        self.frame_history = []
        self.max_history = 100
    
    def calculate_entropy(self, visual_data: np.ndarray) -> EntropyMetrics:
        """
        Calculate entropy metrics for visual/data field
        
        Args:
            visual_data: Array of shape (height, width, channels)
                        Can be actual image or data mapped to visual domain
        
        Returns:
            EntropyMetrics with all entropy components
        """
        # Store in history
        self.frame_history.append(visual_data)
        if len(self.frame_history) > self.max_history:
            self.frame_history.pop(0)
        
        # Calculate each entropy component
        spatial = self._calculate_spatial_entropy(visual_data)
        temporal = self._calculate_temporal_entropy()
        frequency = self._calculate_frequency_entropy(visual_data)
        
        # Combine entropies (weighted average)
        total = (spatial * 0.4 + temporal * 0.3 + frequency * 0.3)
        
        # Coherence is inverse of entropy
        coherence = 1.0 - total
        
        return EntropyMetrics(
            spatial_entropy=spatial,
            temporal_entropy=temporal,
            frequency_entropy=frequency,
            total_entropy=total,
            coherence=coherence
        )
    
    def _calculate_spatial_entropy(self, frame: np.ndarray) -> float:
        """
        Calculate spatial entropy (variance within frame)
        
        High variance = High entropy = Chaotic
        Low variance = Low entropy = Ordered
        
        Args:
            frame: Current frame
        
        Returns:
            Normalized entropy (0.0-1.0)
        """
        # Calculate variance across all channels
        variance = np.var(frame)
        
        # Normalize to 0-1 (assume max variance ~1.0)
        entropy = min(1.0, variance / 1.0)
        
        return entropy
    
    def _calculate_temporal_entropy(self) -> float:
        """
        Calculate temporal entropy (variance across frames)
        
        High frame-to-frame changes = High entropy = Unstable
        Low frame-to-frame changes = Low entropy = Stable
        
        Returns:
            Normalized entropy (0.0-1.0)
        """
        if len(self.frame_history) < 2:
            return 0.0
        
        # Calculate frame-to-frame differences
        diffs = []
        for i in range(1, len(self.frame_history)):
            diff = np.mean(np.abs(self.frame_history[i] - self.frame_history[i-1]))
            diffs.append(diff)
        
        # Variance of differences
        temporal_variance = np.var(diffs)
        
        # Normalize
        entropy = min(1.0, temporal_variance / 0.1)
        
        return entropy
    
    def _calculate_frequency_entropy(self, frame: np.ndarray) -> float:
        """
        Calculate frequency domain entropy
        
        Unexpected high frequencies = High entropy = Noise/Oscillation
        Smooth frequency spectrum = Low entropy = Clean signal
        
        Args:
            frame: Current frame
        
        Returns:
            Normalized entropy (0.0-1.0)
        """
        # Flatten frame for FFT
        signal = frame.flatten()
        
        # FFT
        fft_result = fft.fft(signal)
        magnitudes = np.abs(fft_result)
        
        # Normalize magnitudes to probability distribution
        probs = magnitudes / (np.sum(magnitudes) + 1e-10)
        
        # Shannon entropy
        freq_entropy = scipy_entropy(probs)
        
        # Normalize (max entropy for uniform distribution ≈ log(N))
        max_entropy = np.log(len(probs))
        normalized = freq_entropy / max_entropy
        
        return normalized
    
    def should_throttle(self, entropy_metrics: EntropyMetrics, 
                       threshold: float = 0.7) -> bool:
        """
        Determine if system should throttle based on entropy
        
        Args:
            entropy_metrics: Calculated entropy
            threshold: Entropy threshold for throttling
        
        Returns:
            True if should throttle
        """
        return entropy_metrics.total_entropy > threshold
    
    def get_throttle_factor(self, entropy_metrics: EntropyMetrics,
                           threshold: float = 0.7,
                           min_factor: float = 0.5) -> float:
        """
        Calculate throttle factor based on entropy
        
        Args:
            entropy_metrics: Calculated entropy
            threshold: Entropy threshold
            min_factor: Minimum throttle factor (0.5 = 50%)
        
        Returns:
            Throttle factor (0.5-1.0)
        """
        if entropy_metrics.total_entropy <= threshold:
            return 1.0  # No throttling
        
        # Linear reduction above threshold
        excess_entropy = entropy_metrics.total_entropy - threshold
        max_excess = 1.0 - threshold
        
        reduction = excess_entropy / max_excess
        factor = 1.0 - (reduction * (1.0 - min_factor))
        
        return max(min_factor, factor)


# Example usage for manufacturing data
def example_servo_error_entropy():
    """
    Example: Analyze servo error as visual entropy
    """
    compositor = ActiveOpticCompositor()
    
    # Simulate servo error data
    # Stable operation
    stable_errors = np.random.normal(0, 0.01, (100, 1, 3))  # Low noise
    
    # Unstable operation (chatter)
    unstable_errors = np.random.normal(0, 0.5, (100, 1, 3))  # High noise
    
    # Analyze stable
    stable_metrics = compositor.calculate_entropy(stable_errors)
    print(f"Stable: Entropy={stable_metrics.total_entropy:.3f}, "
          f"Coherence={stable_metrics.coherence:.3f}")
    
    # Analyze unstable
    compositor.frame_history = []  # Reset
    unstable_metrics = compositor.calculate_entropy(unstable_errors)
    print(f"Unstable: Entropy={unstable_metrics.total_entropy:.3f}, "
          f"Coherence={unstable_metrics.coherence:.3f}")
    
    # Check throttling
    should_throttle_stable = compositor.should_throttle(stable_metrics)
    should_throttle_unstable = compositor.should_throttle(unstable_metrics)
    
    print(f"\nShould throttle stable: {should_throttle_stable}")
    print(f"Should throttle unstable: {should_throttle_unstable}")


if __name__ == "__main__":
    example_servo_error_entropy()
