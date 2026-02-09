"""
Predictive Maintenance - Feature Engineering
Transform raw sensor data into ML-ready features

FEATURES:
- Statistical aggregations over sliding windows
- Frequency domain analysis (FFT)
- Trend detection and rate of change
- Cross-sensor correlations
- Anomaly scoring relative to baseline

TIME WINDOWS:
- 1 minute
- 10 minutes
- 1 hour
- 24 hours
"""

from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import numpy as np
from collections import deque


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class TimeSeriesWindow:
    """Time series data for a single sensor over a window"""
    sensor_id: str
    start_time: datetime
    end_time: datetime
    values: np.ndarray
    timestamps: np.ndarray
    
    def duration_seconds(self) -> float:
        """Get window duration in seconds"""
        return (self.end_time - self.start_time).total_seconds()
    
    def sampling_rate_hz(self) -> float:
        """Calculate average sampling rate"""
        if len(self.values) < 2:
            return 0.0
        return len(self.values) / self.duration_seconds()


@dataclass
class FeatureVector:
    """Computed features for a time window"""
    timestamp: datetime
    operation_id: int
    features: Dict[str, float]
    
    def to_array(self, feature_names: List[str]) -> np.ndarray:
        """Convert to numpy array in specified order"""
        return np.array([self.features.get(name, 0.0) for name in feature_names])


# =============================================================================
# STATISTICAL FEATURE EXTRACTOR
# =============================================================================

class StatisticalFeatureExtractor:
    """
    Extract statistical features from time-series data
    
    Features:
    - Mean, median, std, min, max
    - Percentiles (25th, 75th, 95th)
    - Range, interquartile range
    - Skewness, kurtosis
    """
    
    @staticmethod
    def extract(window: TimeSeriesWindow) -> Dict[str, float]:
        """
        Extract statistical features
        
        Args:
            window: Time series window
        
        Returns:
            Dictionary of features
        """
        if len(window.values) == 0:
            return {}
        
        values = window.values
        
        features = {
            f'{window.sensor_id}_mean': np.mean(values),
            f'{window.sensor_id}_median': np.median(values),
            f'{window.sensor_id}_std': np.std(values),
            f'{window.sensor_id}_min': np.min(values),
            f'{window.sensor_id}_max': np.max(values),
            f'{window.sensor_id}_range': np.max(values) - np.min(values),
        }
        
        # Percentiles
        percentiles = np.percentile(values, [25, 75, 95])
        features[f'{window.sensor_id}_p25'] = percentiles[0]
        features[f'{window.sensor_id}_p75'] = percentiles[1]
        features[f'{window.sensor_id}_p95'] = percentiles[2]
        
        # Interquartile range
        features[f'{window.sensor_id}_iqr'] = percentiles[1] - percentiles[0]
        
        # Skewness and kurtosis (if scipy available)
        try:
            from scipy import stats
            features[f'{window.sensor_id}_skew'] = stats.skew(values)
            features[f'{window.sensor_id}_kurtosis'] = stats.kurtosis(values)
        except ImportError:
            pass
        
        return features


# =============================================================================
# FREQUENCY DOMAIN FEATURE EXTRACTOR
# =============================================================================

class FrequencyFeatureExtractor:
    """
    Extract frequency domain features using FFT
    
    Features:
    - Dominant frequency
    - Peak magnitude
    - Spectral centroid
    - Spectral energy in bands
    """
    
    @staticmethod
    def extract(window: TimeSeriesWindow) -> Dict[str, float]:
        """
        Extract frequency domain features
        
        Args:
            window: Time series window
        
        Returns:
            Dictionary of features
        """
        if len(window.values) < 16:  # Need minimum samples for FFT
            return {}
        
        values = window.values
        sampling_rate = window.sampling_rate_hz()
        
        # Apply FFT
        fft_values = np.fft.fft(values)
        fft_freq = np.fft.fftfreq(len(values), 1.0/sampling_rate)
        
        # Only positive frequencies
        positive_freq_idx = fft_freq > 0
        fft_magnitude = np.abs(fft_values[positive_freq_idx])
        fft_freq_positive = fft_freq[positive_freq_idx]
        
        features = {}
        
        # Dominant frequency (peak in spectrum)
        if len(fft_magnitude) > 0:
            peak_idx = np.argmax(fft_magnitude)
            features[f'{window.sensor_id}_dominant_freq'] = fft_freq_positive[peak_idx]
            features[f'{window.sensor_id}_peak_magnitude'] = fft_magnitude[peak_idx]
            
            # Spectral centroid
            spectral_centroid = np.sum(fft_freq_positive * fft_magnitude) / np.sum(fft_magnitude)
            features[f'{window.sensor_id}_spectral_centroid'] = spectral_centroid
            
            # Energy in frequency bands
            # Low: 0-10 Hz, Mid: 10-50 Hz, High: 50+ Hz
            low_band = (fft_freq_positive < 10)
            mid_band = (fft_freq_positive >= 10) & (fft_freq_positive < 50)
            high_band = (fft_freq_positive >= 50)
            
            total_energy = np.sum(fft_magnitude**2)
            if total_energy > 0:
                features[f'{window.sensor_id}_energy_low'] = np.sum(fft_magnitude[low_band]**2) / total_energy
                features[f'{window.sensor_id}_energy_mid'] = np.sum(fft_magnitude[mid_band]**2) / total_energy
                features[f'{window.sensor_id}_energy_high'] = np.sum(fft_magnitude[high_band]**2) / total_energy
        
        return features


# =============================================================================
# TREND FEATURE EXTRACTOR
# =============================================================================

class TrendFeatureExtractor:
    """
    Extract trend and temporal features
    
    Features:
    - Linear regression slope
    - Rate of change
    - Acceleration (2nd derivative)
    - Number of peaks/valleys
    - Crossings of mean
    """
    
    @staticmethod
    def extract(window: TimeSeriesWindow) -> Dict[str, float]:
        """
        Extract trend features
        
        Args:
            window: Time series window
        
        Returns:
            Dictionary of features
        """
        if len(window.values) < 3:
            return {}
        
        values = window.values
        features = {}
        
        # Linear regression slope
        x = np.arange(len(values))
        slope, intercept = np.polyfit(x, values, 1)
        features[f'{window.sensor_id}_trend_slope'] = slope
        features[f'{window.sensor_id}_trend_intercept'] = intercept
        
        # Rate of change (first derivative)
        rate_of_change = np.diff(values)
        features[f'{window.sensor_id}_roc_mean'] = np.mean(rate_of_change)
        features[f'{window.sensor_id}_roc_std'] = np.std(rate_of_change)
        features[f'{window.sensor_id}_roc_max'] = np.max(rate_of_change)
        features[f'{window.sensor_id}_roc_min'] = np.min(rate_of_change)
        
        # Acceleration (2nd derivative)
        if len(rate_of_change) > 1:
            acceleration = np.diff(rate_of_change)
            features[f'{window.sensor_id}_accel_mean'] = np.mean(acceleration)
            features[f'{window.sensor_id}_accel_std'] = np.std(acceleration)
        
        # Count peaks and valleys
        peaks = 0
        valleys = 0
        for i in range(1, len(values) - 1):
            if values[i] > values[i-1] and values[i] > values[i+1]:
                peaks += 1
            elif values[i] < values[i-1] and values[i] < values[i+1]:
                valleys += 1
        
        features[f'{window.sensor_id}_num_peaks'] = peaks
        features[f'{window.sensor_id}_num_valleys'] = valleys
        
        # Mean crossings
        mean_val = np.mean(values)
        crossings = 0
        for i in range(1, len(values)):
            if (values[i-1] < mean_val and values[i] >= mean_val) or \
               (values[i-1] >= mean_val and values[i] < mean_val):
                crossings += 1
        
        features[f'{window.sensor_id}_mean_crossings'] = crossings
        
        return features



# =============================================================================
# CROSS-SENSOR CORRELATION EXTRACTOR
# =============================================================================

class CrossSensorCorrelationExtractor:
    """
    Extract features based on relationships between sensors
    
    Features:
    - Pearson correlation coefficient
    - Rolling correlation (if windows allow)
    - Ratio of means (e.g., Load / Speed)
    """
    
    @staticmethod
    def extract(window: TimeSeriesWindow, other_windows: Dict[str, TimeSeriesWindow]) -> Dict[str, float]:
        """
        Extract correlation features between this window and others
        
        Args:
            window: Primary sensor window
            other_windows: Dictionary of other sensor windows covering the same time
            
        Returns:
            Dictionary of features
        """
        if len(window.values) < 2:
            return {}
            
        features = {}
        sensor_a = window.sensor_id
        
        for sensor_b, window_b in other_windows.items():
            if sensor_a == sensor_b:
                continue
                
            # Ensure windows are compatible (same length/timestamps ideally)
            # For this MVP, we truncate to the shorter length
            min_len = min(len(window.values), len(window_b.values))
            if min_len < 2:
                continue
                
            vals_a = window.values[:min_len]
            vals_b = window_b.values[:min_len]
            
            # Pearson Correlation
            try:
                corr_matrix = np.corrcoef(vals_a, vals_b)
                if corr_matrix.shape == (2, 2):
                    corr = corr_matrix[0, 1]
                    if not np.isnan(corr):
                        features[f'corr_{sensor_a}_vs_{sensor_b}'] = corr
            except Exception:
                pass
                
            # Ratio of Means
            mean_b = np.mean(vals_b)
            if abs(mean_b) > 1e-6:
                mean_a = np.mean(vals_a)
                features[f'ratio_mean_{sensor_a}_div_{sensor_b}'] = mean_a / mean_b

        return features

# =============================================================================
# SLIDING WINDOW GENERATOR
# =============================================================================

class SlidingWindowGenerator:
    """
    Generate sliding windows from time-series data
    
    Supports multiple window sizes and step sizes
    """
    
    def __init__(self, window_sizes: List[timedelta], step_size: timedelta = timedelta(seconds=10)):
        """
        Initialize generator
        
        Args:
            window_sizes: List of window durations
            step_size: How far to slide window each step
        """
        self.window_sizes = window_sizes
        self.step_size = step_size
    
    def generate_windows(self, sensor_data: Dict[str, List[Tuple[datetime, float]]]) -> List[Dict[str, TimeSeriesWindow]]:
        """
        Generate windows for all sensors
        
        Args:
            sensor_data: Dict mapping sensor_id to list of (timestamp, value) tuples
        
        Returns:
            List of window sets (one per time point)
        """
        # Find time range
        all_timestamps = []
        for sensor_id, data in sensor_data.items():
            all_timestamps.extend([ts for ts, _ in data])
        
        if not all_timestamps:
            return []
        
        min_time = min(all_timestamps)
        max_time = max(all_timestamps)
        
        # Generate windows
        window_sets = []
        current_time = min_time + max(self.window_sizes)  # Start after longest window available
        
        while current_time <= max_time:
            window_set = {}
            
            for sensor_id, data in sensor_data.items():
                for window_size in self.window_sizes:
                    start_time = current_time - window_size
                    
                    # Filter data in window
                    window_data = [(ts, val) for ts, val in data if start_time <= ts <= current_time]
                    
                    if window_data:
                        timestamps = np.array([ts.timestamp() for ts, _ in window_data])
                        values = np.array([val for _, val in window_data])
                        
                        window_name = f"{sensor_id}_{int(window_size.total_seconds())}s"
                        window_set[window_name] = TimeSeriesWindow(
                            sensor_id=sensor_id,
                            start_time=start_time,
                            end_time=current_time,
                            values=values,
                            timestamps=timestamps
                        )
            
            if window_set:
                window_sets.append(window_set)
            
            current_time += self.step_size
        
        return window_sets

# =============================================================================
# FEATURE CACHE
# =============================================================================

class FeatureCache:
    """
    Simple in-memory cache for computed features to prevent re-calculation.
    Keys are based on (sensor_id, window_start, window_end, extractor_type).
    """
    def __init__(self, max_size: int = 10000):
        self.cache = {}
        self.max_size = max_size
        
    def get(self, key: str) -> Optional[Dict[str, float]]:
        return self.cache.get(key)
        
    def set(self, key: str, value: Dict[str, float]):
        if len(self.cache) >= self.max_size:
            # Simple eviction: clear 20% random items or oldest. 
            # For simplicity: Clear half.
            keys = list(self.cache.keys())
            for k in keys[:self.max_size // 2]:
                del self.cache[k]
        self.cache[key] = value

    def generate_key(self, window: TimeSeriesWindow, extractor_name: str) -> str:
        # Key format: sensor|start_iso|end_iso|extractor
        return f"{window.sensor_id}|{window.start_time.isoformat()}|{window.end_time.isoformat()}|{extractor_name}"


# =============================================================================
# FEATURE ENGINEERING PIPELINE
# =============================================================================

class FeatureEngineeringPipeline:
    """
    Complete feature engineering pipeline
    
    Orchestrates all feature extractors
    """
    
    def __init__(self, window_sizes: List[timedelta] = None):
        """
        Initialize pipeline
        
        Args:
            window_sizes: Window sizes to use (default: 1min, 10min, 1hr)
        """
        if window_sizes is None:
            self.window_sizes = [
                timedelta(minutes=1),
                timedelta(minutes=10),
                timedelta(hours=1)
            ]
        else:
            self.window_sizes = window_sizes
        
        self.window_generator = SlidingWindowGenerator(self.window_sizes)
        self.stat_extractor = StatisticalFeatureExtractor()
        self.freq_extractor = FrequencyFeatureExtractor()
        self.trend_extractor = TrendFeatureExtractor()
        self.corr_extractor = CrossSensorCorrelationExtractor()
        self.cache = FeatureCache()

    
    def extract_features(self, window: TimeSeriesWindow, other_windows: Optional[Dict[str, TimeSeriesWindow]] = None) -> Dict[str, float]:
        """
        Extract all features for a single window
        """
        features = {}
        
        # Helper to check cache
        def get_or_compute(extractor, name):
            key = self.cache.generate_key(window, name)
            cached = self.cache.get(key)
            if cached is not None:
                return cached
            result = extractor.extract(window)
            self.cache.set(key, result)
            return result

        # Statistical features
        features.update(get_or_compute(self.stat_extractor, "stat"))
        
        # Frequency features
        features.update(get_or_compute(self.freq_extractor, "freq"))
        
        # Trend features
        features.update(get_or_compute(self.trend_extractor, "trend"))
        
        # Cross-Sensor Correlations (No caching for now due to complexity of key)
        if other_windows:
            features.update(self.corr_extractor.extract(window, other_windows))
        
        return features
    
    def process_sensor_data(self, sensor_data: Dict[str, List[Tuple[datetime, float]]], 
                           operation_id: int) -> List[FeatureVector]:
        """
        Process sensor data and extract features
        
        Args:
            sensor_data: Dict mapping sensor_id to list of (timestamp, value) tuples
            operation_id: Operation ID
        
        Returns:
            List of feature vectors
        """
        feature_vectors = []
        
        # Generate sliding windows
        window_sets = self.window_generator.generate_windows(sensor_data)
        
        # Extract features for each window set
        for window_set in window_sets:
            all_features = {}
            timestamp = None
            
            # Map of name -> window for correlation
            # We need to group by window size to do meaningful correlation?
            # Or just pass all concurrent windows
            
            for window_name, window in window_set.items():
                if timestamp is None:
                    timestamp = window.end_time
                
                # Identify potential partners for correlation (same window size, different sensor)
                # current window_name format: sensor_123s
                # We simply pass the whole set; the extractor will filter by index or logic
                
                # Extract features
                # Pass 'window_set' as 'other_windows' so we can correlate
                window_features = self.extract_features(window, window_set)
                all_features.update(window_features)
            
            if all_features and timestamp:
                feature_vector = FeatureVector(
                    timestamp=timestamp,
                    operation_id=operation_id,
                    features=all_features
                )
                feature_vectors.append(feature_vector)
        
        return feature_vectors


# Example usage
if __name__ == "__main__":
    print("=" * 70)
    print("Feature Engineering Pipeline - Demo")
    print("=" * 70)
    
    # Generate synthetic sensor data
    import math
    
    base_time = datetime.now()
    sensor_data = {}
    
    # Spindle load (with increasing trend - simulating bearing wear)
    timestamps_load = []
    values_load = []
    for i in range(600):  # 10 minutes at 1 Hz
        t = base_time + timedelta(seconds=i)
        # Normal operation with slight upward trend
        value = 60 + 10 * math.sin(i / 30) + 0.01 * i + np.random.normal(0, 2)
        timestamps_load.append(t)
        values_load.append(value)
    
    sensor_data['spindle_load'] = list(zip(timestamps_load, values_load))
    
    # Vibration (with increasing amplitude - sign of impending failure)
    timestamps_vib = []
    values_vib = []
    for i in range(600):
        t = base_time + timedelta(seconds=i)
        # Vibration increasing over time
        amplitude = 0.3 + 0.001 * i
        value = amplitude * math.sin(i * 20) + np.random.normal(0, 0.05)
        timestamps_vib.append(t)
        values_vib.append(value)
    
    sensor_data['vibration_x'] = list(zip(timestamps_vib, values_vib))
    
    print(f"\n📊 Sensor Data:")
    print(f"   Spindle load: {len(sensor_data['spindle_load'])} samples")
    print(f"   Vibration X: {len(sensor_data['vibration_x'])} samples")
    
    # Create pipeline
    pipeline = FeatureEngineeringPipeline()
    
    # Extract features
    print(f"\n⚙️  Extracting features...")
    feature_vectors = pipeline.process_sensor_data(sensor_data, operation_id=1)
    
    print(f"\n✅ Extracted {len(feature_vectors)} feature vectors")
    
    # Show first feature vector
    if feature_vectors:
        fv = feature_vectors[0]
        print(f"\n📋 Sample Feature Vector:")
        print(f"   Timestamp: {fv.timestamp}")
        print(f"   Operation ID: {fv.operation_id}")
        print(f"   Number of features: {len(fv.features)}")
        print(f"\n   Sample features:")
        for i, (name, value) in enumerate(list(fv.features.items())[:10]):
            print(f"     {name}: {value:.4f}")
        print(f"   ... ({len(fv.features) - 10} more features)")
        
        # Show trend indicator
        print(f"\n📈 Trend Indicators:")
        spindle_slope = fv.features.get('spindle_load_trend_slope', 0)
        vib_mean = fv.features.get('vibration_x_mean', 0)
        print(f"   Spindle load trend: {spindle_slope:+.6f} (positive = increasing!)")
        print(f"   Vibration amplitude: {vib_mean:.4f} mm")
        
        if spindle_slope > 0.01:
            print(f"\n   ⚠️ WARNING: Increasing spindle load detected!")
            print(f"   This could indicate bearing wear or degradation.")
