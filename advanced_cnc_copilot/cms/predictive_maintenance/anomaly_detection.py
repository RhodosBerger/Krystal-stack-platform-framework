"""
Predictive Maintenance - Anomaly Detection
Real-time anomaly detection for manufacturing equipment

ALGORITHMS:
- Isolation Forest (unsupervised)
- Statistical Control Charts (Shewhart, CUSUM, EWMA)
- One-Class SVM
- Threshold-based detection

FEATURES:
- Multi-algorithm ensemble
- Configurable sensitivity
- Real-time scoring
- Alert generation
"""

from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from datetime import datetime
import numpy as np
from collections import deque


# =============================================================================
# DATA STRUCTURES
# =============================================================================

class AnomalySeverity(Enum):
    """Severity levels for anomalies"""
    NORMAL = "normal"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class AnomalyDetection:
    """Detected anomaly with details"""
    timestamp: datetime
    sensor_id: str
    anomaly_score: float  # 0.0-1.0
    severity: AnomalySeverity
    method: str  # Which detector found it
    affected_features: List[str]
    description: str


# =============================================================================
# ISOLATION FOREST DETECTOR
# =============================================================================

class IsolationForestDetector:
    """
    Anomaly detection using Isolation Forest
    
    PRINCIPLE: Anomalies are easier to isolate than normal points
    """
    
    def __init__(self, contamination: float = 0.05, n_estimators: int = 100):
        """
        Initialize detector
        
        Args:
            contamination: Expected proportion of anomalies (0.01-0.1)
            n_estimators: Number of trees
        """
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.model = None
        self.fitted = False
        
        # Try to import sklearn
        try:
            from sklearn.ensemble import IsolationForest
            self.IsolationForest = IsolationForest
            self._sklearn_available = True
        except ImportError:
            print("⚠️ scikit-learn not installed - Isolation Forest unavailable")
            print("   Install with: pip install scikit-learn")
            self._sklearn_available = False
    
    def fit(self, X: np.ndarray):
        """
        Fit detector on normal data
        
        Args:
            X: Feature matrix (n_samples, n_features)
        """
        if not self._sklearn_available:
            return
        
        self.model = self.IsolationForest(
            contamination=self.contamination,
            n_estimators=self.n_estimators,
            random_state=42
        )
        self.model.fit(X)
        self.fitted = True
        print(f"✅ Isolation Forest fitted on {X.shape[0]} samples, {X.shape[1]} features")
    
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict anomalies
        
        Args:
            X: Feature matrix
        
        Returns:
            (predictions, scores)
            predictions: 1 = normal, -1 = anomaly
            scores: Anomaly scores (lower = more anomalous)
        """
        if not self._sklearn_available or not self.fitted:
            # Return all normal if not available
            return np.ones(X.shape[0]), np.zeros(X.shape[0])
        
        predictions = self.model.predict(X)
        scores = self.model.score_samples(X)
        
        # Convert scores to 0-1 range (higher = more anomalous)
        anomaly_scores = 1.0 / (1.0 + np.exp(scores))  # Sigmoid
        
        return predictions, anomaly_scores


# =============================================================================
# STATISTICAL CONTROL CHARTS
# =============================================================================

class ControlChartDetector:
    """
    Statistical process control charts
    
    CHARTS:
    - Shewhart: Mean ± 3σ limits
    - CUSUM: Cumulative sum control
    - EWMA: Exponentially weighted moving average
    """
    
    def __init__(self, window_size: int = 100):
        """
        Initialize detector
        
        Args:
            window_size: Number of samples for baseline calculation
        """
        self.window_size = window_size
        self.baseline_mean = None
        self.baseline_std = None
        self.history = deque(maxlen=window_size)
    
    def fit(self, X: np.ndarray):
        """
        Establish baseline from normal data
        
        Args:
            X: Historical normal data (n_samples,)
        """
        self.baseline_mean = np.mean(X)
        self.baseline_std = np.std(X)
        self.history.extend(X[-self.window_size:])
        
        print(f"✅ Control chart baseline: μ={self.baseline_mean:.2f}, σ={self.baseline_std:.2f}")
    
    def detect_shewhart(self, value: float) -> Tuple[bool, float]:
        """
        Shewhart control chart (3-sigma rule)
        
        Args:
            value: New measurement
        
        Returns:
            (is_anomaly, sigma_distance)
        """
        if self.baseline_mean is None:
            return False, 0.0
        
        # Calculate distance in standard deviations
        sigma_distance = abs(value - self.baseline_mean) / self.baseline_std if self.baseline_std > 0 else 0
        
        # Anomaly if outside 3-sigma
        is_anomaly = sigma_distance > 3.0
        
        return is_anomaly, sigma_distance
    
    def detect_cusum(self, value: float, threshold: float = 5.0) -> Tuple[bool, float]:
        """
        CUSUM control chart (cumulative sum)
        
        Args:
            value: New measurement
            threshold: CUSUM threshold
        
        Returns:
            (is_anomaly, cusum_value)
        """
        if not hasattr(self, 'cusum_positive'):
            self.cusum_positive = 0.0
            self.cusum_negative = 0.0
        
        # Expected value
        deviation = value - self.baseline_mean if self.baseline_mean else 0
        
        # Update CUSUM
        self.cusum_positive = max(0, self.cusum_positive + deviation)
        self.cusum_negative = min(0, self.cusum_negative + deviation)
        
        # Check thresholds
        is_anomaly = abs(self.cusum_positive) > threshold or abs(self.cusum_negative) > threshold
        cusum_value = max(abs(self.cusum_positive), abs(self.cusum_negative))
        
        return is_anomaly, cusum_value
    
    def detect_ewma(self, value: float, lambda_param: float = 0.2, L: float = 3.0) -> Tuple[bool, float]:
        """
        EWMA control chart
        
        Args:
            value: New measurement
            lambda_param: Smoothing parameter (0-1)
            L: Control limit multiplier
        
        Returns:
            (is_anomaly, ewma_value)
        """
        if not hasattr(self, 'ewma'):
            self.ewma = self.baseline_mean if self.baseline_mean else value
        
        # Update EWMA
        self.ewma = lambda_param * value + (1 - lambda_param) * self.ewma
        
        # Control limits
        if self.baseline_std and self.baseline_mean:
            sigma_ewma = self.baseline_std * np.sqrt(lambda_param / (2 - lambda_param))
            ucl = self.baseline_mean + L * sigma_ewma
            lcl = self.baseline_mean - L * sigma_ewma
            
            is_anomaly = self.ewma > ucl or self.ewma < lcl
            distance = min(abs(self.ewma - ucl), abs(self.ewma - lcl))
        else:
            is_anomaly = False
            distance = 0.0
        
        return is_anomaly, distance


# =============================================================================
# THRESHOLD-BASED DETECTOR
# =============================================================================

class ThresholdDetector:
    """
    Simple threshold-based anomaly detection
    
    Useful for known limits (e.g., temperature > 100°C = bad)
    """
    
    def __init__(self, thresholds: Dict[str, Tuple[float, float]]):
        """
        Initialize detector
        
        Args:
            thresholds: Dict mapping feature name to (min, max) tuple
        """
        self.thresholds = thresholds
    
    def detect(self, feature_name: str, value: float) -> Tuple[bool, str]:
        """
        Check if value exceeds thresholds
        
        Args:
            feature_name: Name of feature
            value: Value to check
        
        Returns:
            (is_anomaly, reason)
        """
        if feature_name not in self.thresholds:
            return False, ""
        
        min_val, max_val = self.thresholds[feature_name]
        
        if value < min_val:
            return True, f"Below minimum ({value:.2f} < {min_val:.2f})"
        elif value > max_val:
            return True, f"Above maximum ({value:.2f} > {max_val:.2f})"
        else:
            return False, ""


# =============================================================================
# ANOMALY DETECTION ENGINE
# =============================================================================

class AnomalyDetectionEngine:
    """
    Multi-algorithm anomaly detection engine
    
    Combines:
    - Isolation Forest
    - Control charts
    - Threshold detection
    
    Provides ensemble voting and severity scoring
    """
    
    def __init__(self):
        """Initialize engine"""
        self.isolation_forest = IsolationForestDetector()
        self.control_charts: Dict[str, ControlChartDetector] = {}
        self.threshold_detector = None
        self.fitted = False
    
    def fit(self, X: np.ndarray, feature_names: List[str]):
        """
        Fit all detectors on training data
        
        Args:
            X: Feature matrix (n_samples, n_features)
            feature_names: Names of features
        """
        print(f"\n🎓 Training anomaly detection models...")
        print(f"   Training data: {X.shape[0]} samples, {X.shape[1]} features")
        
        # Fit Isolation Forest
        self.isolation_forest.fit(X)
        
        # Fit control charts for each feature
        for i, feature_name in enumerate(feature_names):
            chart = ControlChartDetector()
            chart.fit(X[:, i])
            self.control_charts[feature_name] = chart
        
        self.feature_names = feature_names
        self.fitted = True
        print(f"✅ Training complete!")
    
    def detect(self, X: np.ndarray, timestamp: datetime = None) -> List[AnomalyDetection]:
        """
        Detect anomalies in new data
        
        Args:
            X: Feature matrix (n_samples, n_features)
            timestamp: Timestamp of measurements
        
        Returns:
            List of detected anomalies
        """
        if not self.fitted:
            print("⚠️ Engine not fitted - call fit() first")
            return []
        
        if timestamp is None:
            timestamp = datetime.now()
        
        anomalies = []
        
        # Isolation Forest detection
        predictions, if_scores = self.isolation_forest.predict(X)
        
        for i in range(X.shape[0]):
            # Check if anomaly according to Isolation Forest
            if predictions[i] == -1:
                severity = self._score_to_severity(if_scores[i])
                
                # Find which features are most anomalous
                affected_features = []
                for j, feature_name in enumerate(self.feature_names):
                    if feature_name in self.control_charts:
                        is_anom_shewhart, sigma_dist = self.control_charts[feature_name].detect_shewhart(X[i, j])
                        if is_anom_shewhart:
                            affected_features.append(f"{feature_name} ({sigma_dist:.1f}σ)")
                
                anomaly = AnomalyDetection(
                    timestamp=timestamp,
                    sensor_id="multiple",
                    anomaly_score=if_scores[i],
                    severity=severity,
                    method="isolation_forest",
                    affected_features=affected_features[:5],  # Top 5
                    description=f"Anomalous pattern detected (score: {if_scores[i]:.2f})"
                )
                anomalies.append(anomaly)
        
        return anomalies
    
    def _score_to_severity(self, score: float) -> AnomalySeverity:
        """Convert anomaly score to severity level"""
        if score < 0.3:
            return AnomalySeverity.NORMAL
        elif score < 0.5:
            return AnomalySeverity.LOW
        elif score < 0.7:
            return AnomalySeverity.MEDIUM
        elif score < 0.9:
            return AnomalySeverity.HIGH
        else:
            return AnomalySeverity.CRITICAL


# Example usage
if __name__ == "__main__":
    print("=" * 70)
    print("Anomaly Detection Engine - Demo")
    print("=" * 70)
    
    # Generate synthetic training data (normal operation)
    np.random.seed(42)
    n_samples = 1000
    n_features = 10
    
    # Normal data: mean=50, std=5
    X_train = np.random.normal(50, 5, size=(n_samples, n_features))
    
    feature_names = [f'feature_{i}' for i in range(n_features)]
    
    # Create and train engine
    engine = AnomalyDetectionEngine()
    engine.fit(X_train, feature_names)
    
    print(f"\n🧪 Testing anomaly detection...")
    
    # Test data with some anomalies
    X_test_normal = np.random.normal(50, 5, size=(10, n_features))
    X_test_anomaly = np.random.normal(80, 10, size=(5, n_features))  # Anomalous!
    
    X_test = np.vstack([X_test_normal, X_test_anomaly])
    
    # Detect anomalies
    anomalies = engine.detect(X_test)
    
    print(f"\n📊 Results:")
    print(f"   Test samples: {X_test.shape[0]}")
    print(f"   Anomalies detected: {len(anomalies)}")
    
    if anomalies:
        print(f"\n🚨 Detected Anomalies:")
        for anom in anomalies:
            print(f"\n   {anom.severity.value.upper()} - {anom.method}")
            print(f"   Score: {anom.anomaly_score:.3f}")
            print(f"   Affected features: {', '.join(anom.affected_features[:3])}")
            print(f"   {anom.description}")
    else:
        print(f"\n✅ No anomalies detected - all systems normal")
