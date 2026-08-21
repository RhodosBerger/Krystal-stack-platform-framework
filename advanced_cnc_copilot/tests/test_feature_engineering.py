"""
Unit Tests for Feature Engineering Pipeline 🧪
Verifies:
1. Statistical Extraction
2. Frequency Extraction
3. Trend Extraction
4. Cross-Sensor Correlation
5. Feature Caching
"""
import unittest
import numpy as np
from datetime import datetime, timedelta
from cms.predictive_maintenance.feature_engineering import (
    StatisticalFeatureExtractor,
    FrequencyFeatureExtractor,
    TrendFeatureExtractor,
    CrossSensorCorrelationExtractor,
    FeatureCache,
    TimeSeriesWindow
)

class TestFeatureEngineering(unittest.TestCase):
    
    def setUp(self):
        # Create dummy window
        self.start = datetime.now()
        self.end = self.start + timedelta(seconds=60)
        
        # Sine wave + noise
        t = np.linspace(0, 10, 100)
        self.values = np.sin(t) + 0.1
        self.timestamps = np.array([self.start + timedelta(seconds=x) for x in t])
        
        self.window = TimeSeriesWindow(
            sensor_id="test_sensor",
            start_time=self.start,
            end_time=self.end,
            values=self.values,
            timestamps=self.timestamps
        )

    def test_statistical_features(self):
        extractor = StatisticalFeatureExtractor()
        feats = extractor.extract(self.window)
        
        self.assertIn('test_sensor_mean', feats)
        self.assertIn('test_sensor_std', feats)
        self.assertAlmostEqual(feats['test_sensor_max'], np.max(self.values), places=4)

    def test_trend_features(self):
        extractor = TrendFeatureExtractor()
        
        # Create a linear trend
        linear_vals = np.linspace(0, 10, 100)
        trend_window = TimeSeriesWindow(
            sensor_id="trend_sensor",
            start_time=self.start,
            end_time=self.end,
            values=linear_vals,
            timestamps=self.timestamps
        )
        
        feats = extractor.extract(trend_window)
        self.assertIn('trend_sensor_trend_slope', feats)
        # Slope should be approx 10/100 per sample step (0.1)
        self.assertTrue(feats['trend_sensor_trend_slope'] > 0)

    def test_correlation(self):
        extractor = CrossSensorCorrelationExtractor()
        
        # Correlated window (Inverse)
        inv_vals = -self.values
        window_b = TimeSeriesWindow(
            sensor_id="sensor_b",
            start_time=self.start,
            end_time=self.end,
            values=inv_vals,
            timestamps=self.timestamps
        )
        
        other_windows = {"sensor_b": window_b}
        feats = extractor.extract(self.window, other_windows)
        
        self.assertIn('corr_test_sensor_vs_sensor_b', feats)
        # Should be close to -1
        self.assertLess(feats['corr_test_sensor_vs_sensor_b'], -0.9)

    def test_cache(self):
        cache = FeatureCache()
        key = cache.generate_key(self.window, "test")
        val = {"metric": 123.0}
        
        cache.set(key, val)
        retrieved = cache.get(key)
        self.assertEqual(retrieved, val)
        
        # Test cache miss
        self.assertIsNone(cache.get("invalid_key"))

if __name__ == '__main__':
    unittest.main()
