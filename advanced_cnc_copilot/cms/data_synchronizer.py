"""
Data Synchronizer
KrystalStack Core Module

PARADIGM: Multiple "Angles of View" consolidate into unified truth
Each data source provides one perspective on reality

METAPHOR: Like witnesses describing an accident from different angles,
each sees truth but from their perspective. Consolidation finds THE truth.
"""

import numpy as np
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime, timedelta


@dataclass
class AngleOfView:
    """
    Single perspective on system state
    
    Each data source (sensor, API, calculation) provides
    one "angle of view" on the system
    
    Attributes:
        source_id: Identifier for this data source
        timestamp: When this view was captured
        data: Dictionary of measurements
        confidence: How reliable is this view (0.0-1.0)
        latency_ms: How old is this data
    """
    source_id: str
    timestamp: datetime
    data: Dict[str, Any]
    confidence: float  # 0.0-1.0
    latency_ms: float  # Age of data


class DataSynchronizer:
    """
    Synchronizes data from multiple asynchronous sources
    
    PROBLEM: Manufacturing systems have many data sources at different rates:
    - FOCAS: 1000 Hz
    - PLC: 100 Hz
    - Vision system: 30 Hz
    - Temperature sensors: 1 Hz
    
    Traditional approach: Sample at slowest rate and lose information
    
    OUR APPROACH: Each source is an "Angle of View"
    - Collect all views within time window
    - Weight by confidence and latency
    - Interpolate missing data
    - Consolidate into unified state
    """
    
    def __init__(self, time_window_ms: float = 100.0):
        """
        Initialize data synchronizer
        
        Args:
            time_window_ms: Time window for collecting views
        """
        self.time_window_ms = time_window_ms
        self.views: List[AngleOfView] = []
        self.max_views = 1000  # Limit memory
    
    def add_view(self, view: AngleOfView):
        """
        Add angle of view to synchronizer
        
        Args:
            view: New perspective on system state
        """
        self.views.append(view)
        
        # Prune old views
        cutoff_time = datetime.now() - timedelta(milliseconds=self.time_window_ms * 10)
        self.views = [v for v in self.views if v.timestamp > cutoff_time]
        
        # Limit total views
        if len(self.views) > self.max_views:
            self.views = self.views[-self.max_views:]
    
    def consolidate(self, time_point: Optional[datetime] = None) -> Dict[str, Any]:
        """
        Consolidate all views into unified state
        
        Args:
            time_point: Time to consolidate at (None = now)
        
        Returns:
            Dictionary of consolidated data
        """
        if time_point is None:
            time_point = datetime.now()
        
        # Get views within time window
        window_start = time_point - timedelta(milliseconds=self.time_window_ms)
        relevant_views = [
            v for v in self.views
            if window_start <= v.timestamp <= time_point
        ]
        
        if not relevant_views:
            return {}
        
        # Consolidate data from all views
        consolidated = {}
        
        # Get all data keys across views
        all_keys = set()
        for view in relevant_views:
            all_keys.update(view.data.keys())
        
        # For each key, consolidate values from all views
        for key in all_keys:
            values_with_weights = []
            
            for view in relevant_views:
                if key in view.data:
                    value = view.data[key]
                    
                    # Calculate weight based on confidence and latency
                    time_diff = (time_point - view.timestamp).total_seconds() * 1000  # ms
                    latency_weight = 1.0 / (1.0 + time_diff / 100.0)  # Decay over 100ms
                    
                    total_weight = view.confidence * latency_weight
                    
                    values_with_weights.append((value, total_weight))
            
            # Consolidate values
            if values_with_weights:
                consolidated[key] = self._consolidate_values(values_with_weights)
        
        return consolidated
    
    def _consolidate_values(self, values_with_weights: List[Tuple[Any, float]]) -> Any:
        """
        Consolidate multiple values into single value
        
        Args:
            values_with_weights: List of (value, weight) tuples
        
        Returns:
            Consolidated value
        """
        if not values_with_weights:
            return None
        
        # Get value type from first value
        first_value = values_with_weights[0][0]
        
        # Numeric values: weighted average
        if isinstance(first_value, (int, float)):
            total_weight = sum(w for _, w in values_with_weights)
            if total_weight == 0:
                return first_value
            
            weighted_sum = sum(v * w for v, w in values_with_weights)
            return weighted_sum / total_weight
        
        # Dictionary values: recursive consolidation
        elif isinstance(first_value, dict):
            # Get all keys
            all_keys = set()
            for val, _ in values_with_weights:
                if isinstance(val, dict):
                    all_keys.update(val.keys())
            
            # Consolidate each key
            result = {}
            for key in all_keys:
                key_values = []
                for val, weight in values_with_weights:
                    if isinstance(val, dict) and key in val:
                        key_values.append((val[key], weight))
                
                if key_values:
                    result[key] = self._consolidate_values(key_values)
            
            return result
        
        # Other types: use most confident value
        else:
            max_weight_idx = max(range(len(values_with_weights)), 
                                key=lambda i: values_with_weights[i][1])
            return values_with_weights[max_weight_idx][0]
    
    def get_latest_by_source(self, source_id: str) -> Optional[AngleOfView]:
        """
        Get latest view from specific source
        
        Args:
            source_id: Source identifier
        
        Returns:
            Latest view from this source (or None)
        """
        matching = [v for v in self.views if v.source_id == source_id]
        if not matching:
            return None
        
        return max(matching, key=lambda v: v.timestamp)
    
    def get_source_health(self, source_id: str) -> Dict[str, Any]:
        """
        Get health metrics for a data source
        
        Args:
            source_id: Source identifier
        
        Returns:
            Dictionary with health metrics
        """
        matching = [v for v in self.views if v.source_id == source_id]
        
        if not matching:
            return {
                'status': 'no_data',
                'last_seen': None,
                'update_rate_hz': 0,
                'average_confidence': 0,
                'average_latency_ms': 0
            }
        
        # Calculate metrics
        timestamps = [v.timestamp for v in matching]
        latest = max(timestamps)
        
        # Update rate (views per second)
        if len(matching) > 1:
            time_span = (latest - min(timestamps)).total_seconds()
            update_rate = len(matching) / max(time_span, 0.001)
        else:
            update_rate = 0
        
        # Average confidence
        avg_confidence = np.mean([v.confidence for v in matching])
        
        # Average latency
        avg_latency = np.mean([v.latency_ms for v in matching])
        
        # Age
        age_seconds = (datetime.now() - latest).total_seconds()
        
        # Status
        if age_seconds < 1.0:
            status = 'healthy'
        elif age_seconds < 5.0:
            status = 'degraded'
        else:
            status = 'stale'
        
        return {
            'status': status,
            'last_seen': latest.isoformat(),
            'age_seconds': age_seconds,
            'update_rate_hz': update_rate,
            'average_confidence': avg_confidence,
            'average_latency_ms': avg_latency,
            'total_views': len(matching)
        }


# Example usage
def example_data_synchronization():
    """
    Example: Synchronize data from multiple sources at different rates
    """
    synchronizer = DataSynchronizer(time_window_ms=100.0)
    
    import time
    
    # Simulate high-frequency source (1000 Hz)
    for i in range(10):
        synchronizer.add_view(AngleOfView(
            source_id='high_freq_sensor',
            timestamp=datetime.now(),
            data={'value': 100 + i * 0.1},
            confidence=0.99,
            latency_ms=0.1
        ))
        time.sleep(0.001)  # 1ms = 1000 Hz
    
    # Simulate medium-frequency source (100 Hz)
    for i in range(2):
        synchronizer.add_view(AngleOfView(
            source_id='medium_freq_sensor',
            timestamp=datetime.now(),
            data={'value': 95 + i * 0.5},
            confidence=0.95,
            latency_ms=1.0
        ))
        time.sleep(0.010)  # 10ms = 100 Hz
    
    # Simulate low-frequency source (10 Hz)
    synchronizer.add_view(AngleOfView(
        source_id='low_freq_sensor',
        timestamp=datetime.now(),
        data={'value': 90},
        confidence=0.90,
        latency_ms=10.0
    ))
    
    # Consolidate
    consolidated = synchronizer.consolidate()
    print("Consolidated data:", consolidated)
    
    # Check health
    for source_id in ['high_freq_sensor', 'medium_freq_sensor', 'low_freq_sensor']:
        health = synchronizer.get_source_health(source_id)
        print(f"\n{source_id}:")
        print(f"  Status: {health['status']}")
        print(f"  Update rate: {health['update_rate_hz']:.1f} Hz")
        print(f"  Avg confidence: {health['average_confidence']:.2f}")


if __name__ == "__main__":
    example_data_synchronization()
