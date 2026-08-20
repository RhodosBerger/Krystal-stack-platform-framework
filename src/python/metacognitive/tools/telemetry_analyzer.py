"""
GAMESA Metacognitive - Telemetry Analyzer Tool

Analyzes GAMESA telemetry data and provides insights.
"""

from typing import List, Dict, Any
from .tool_registry import BaseTool, ToolParameter
import json


class TelemetryAnalyzer(BaseTool):
    """
    Analyzes GAMESA telemetry for patterns and anomalies.

    Provides statistical analysis, trend detection, and correlation.
    """

    def __init__(self, telemetry_buffer: List[Dict] = None):
        """
        Initialize with telemetry buffer.

        Args:
            telemetry_buffer: List of telemetry snapshots
        """
        self.telemetry_buffer = telemetry_buffer or []

    @property
    def name(self) -> str:
        return "telemetry_analyzer"

    @property
    def description(self) -> str:
        return (
            "Analyze GAMESA telemetry data for patterns, trends, and correlations. "
            "Supports queries like: 'temperature_stats', 'fps_correlation', "
            "'power_trend', 'anomaly_detection'"
        )

    def get_parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="query_type",
                type="string",
                description="Type of analysis to perform",
                required=True,
                enum=[
                    "temperature_stats",
                    "fps_correlation",
                    "power_trend",
                    "anomaly_detection",
                    "summary",
                    "time_series"
                ]
            ),
            ToolParameter(
                name="metric",
                type="string",
                description="Specific metric to analyze (optional)",
                required=False
            ),
            ToolParameter(
                name="window_size",
                type="number",
                description="Number of recent samples to analyze (default: 60)",
                required=False
            )
        ]

    def execute(self, query_type: str, metric: str = None, window_size: int = 60) -> Dict[str, Any]:
        """
        Analyze telemetry data.

        Args:
            query_type: Type of analysis
            metric: Specific metric to analyze
            window_size: Number of samples

        Returns:
            Analysis results
        """
        if not self.telemetry_buffer:
            return {"error": "No telemetry data available"}

        # Get recent window
        data = self.telemetry_buffer[-window_size:]

        if query_type == "temperature_stats":
            return self._temperature_stats(data)
        elif query_type == "fps_correlation":
            return self._fps_correlation(data)
        elif query_type == "power_trend":
            return self._power_trend(data)
        elif query_type == "anomaly_detection":
            return self._anomaly_detection(data, metric)
        elif query_type == "summary":
            return self._summary(data)
        elif query_type == "time_series":
            return self._time_series(data, metric)
        else:
            return {"error": f"Unknown query type: {query_type}"}

    def _temperature_stats(self, data: List[Dict]) -> Dict:
        """Calculate temperature statistics."""
        temps = [d.get("temperature", 0) for d in data]

        if not temps:
            return {"error": "No temperature data"}

        return {
            "min": min(temps),
            "max": max(temps),
            "mean": sum(temps) / len(temps),
            "current": temps[-1],
            "trend": "rising" if temps[-1] > temps[0] else "falling",
            "samples": len(temps)
        }

    def _fps_correlation(self, data: List[Dict]) -> Dict:
        """Analyze FPS correlation with other metrics."""
        fps_data = [d.get("fps", 60) for d in data]
        cpu_data = [d.get("cpu_util", 0.5) for d in data]
        gpu_data = [d.get("gpu_util", 0.5) for d in data]
        temp_data = [d.get("temperature", 70) for d in data]

        # Simple correlation (not true Pearson, but approximate)
        def simple_correlation(x, y):
            if len(x) != len(y) or len(x) < 2:
                return 0.0
            mean_x = sum(x) / len(x)
            mean_y = sum(y) / len(y)
            numerator = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(len(x)))
            denom_x = sum((x[i] - mean_x) ** 2 for i in range(len(x)))
            denom_y = sum((y[i] - mean_y) ** 2 for i in range(len(y)))
            if denom_x == 0 or denom_y == 0:
                return 0.0
            return numerator / (denom_x * denom_y) ** 0.5

        return {
            "fps_mean": sum(fps_data) / len(fps_data),
            "fps_current": fps_data[-1],
            "correlation_with_cpu": simple_correlation(fps_data, cpu_data),
            "correlation_with_gpu": simple_correlation(fps_data, gpu_data),
            "correlation_with_temp": simple_correlation(fps_data, temp_data),
            "bottleneck_hint": self._detect_bottleneck(cpu_data[-5:], gpu_data[-5:])
        }

    def _power_trend(self, data: List[Dict]) -> Dict:
        """Analyze power draw trend."""
        power_data = [d.get("power_draw", 20) for d in data]

        if not power_data:
            return {"error": "No power data"}

        # Simple linear trend
        n = len(power_data)
        x_mean = (n - 1) / 2
        y_mean = sum(power_data) / n
        numerator = sum((i - x_mean) * (power_data[i] - y_mean) for i in range(n))
        denominator = sum((i - x_mean) ** 2 for i in range(n))
        slope = numerator / denominator if denominator != 0 else 0

        return {
            "current": power_data[-1],
            "mean": y_mean,
            "trend": "increasing" if slope > 0.1 else "decreasing" if slope < -0.1 else "stable",
            "slope": slope,
            "samples": n
        }

    def _anomaly_detection(self, data: List[Dict], metric: str = "temperature") -> Dict:
        """Detect anomalies in a metric."""
        values = [d.get(metric, 0) for d in data]

        if len(values) < 10:
            return {"error": "Insufficient data for anomaly detection"}

        # Simple threshold-based anomaly detection
        mean = sum(values) / len(values)
        variance = sum((v - mean) ** 2 for v in values) / len(values)
        std_dev = variance ** 0.5

        anomalies = []
        threshold = 2.0  # 2 standard deviations

        for i, value in enumerate(values):
            z_score = abs(value - mean) / std_dev if std_dev > 0 else 0
            if z_score > threshold:
                anomalies.append({"index": i, "value": value, "z_score": z_score})

        return {
            "metric": metric,
            "mean": mean,
            "std_dev": std_dev,
            "anomaly_count": len(anomalies),
            "anomalies": anomalies[:5],  # Top 5
            "current_z_score": abs(values[-1] - mean) / std_dev if std_dev > 0 else 0
        }

    def _summary(self, data: List[Dict]) -> Dict:
        """Generate overall summary."""
        if not data:
            return {"error": "No data available"}

        latest = data[-1]

        return {
            "samples": len(data),
            "latest": {
                "temperature": latest.get("temperature", 0),
                "cpu_util": latest.get("cpu_util", 0),
                "gpu_util": latest.get("gpu_util", 0),
                "fps": latest.get("fps", 0),
                "power_draw": latest.get("power_draw", 0)
            },
            "temperature_stats": self._temperature_stats(data),
            "power_stats": self._power_trend(data)
        }

    def _time_series(self, data: List[Dict], metric: str = "temperature") -> Dict:
        """Return time series data for a metric."""
        values = [d.get(metric, 0) for d in data]

        return {
            "metric": metric,
            "values": values,
            "count": len(values),
            "min": min(values) if values else 0,
            "max": max(values) if values else 0,
            "mean": sum(values) / len(values) if values else 0
        }

    def _detect_bottleneck(self, cpu_util: List[float], gpu_util: List[float]) -> str:
        """Detect CPU vs GPU bottleneck."""
        avg_cpu = sum(cpu_util) / len(cpu_util)
        avg_gpu = sum(gpu_util) / len(gpu_util)

        if avg_cpu > 0.85 and avg_gpu < 0.7:
            return "CPU bottleneck detected"
        elif avg_gpu > 0.85 and avg_cpu < 0.7:
            return "GPU bottleneck detected"
        elif avg_cpu > 0.85 and avg_gpu > 0.85:
            return "Both CPU and GPU heavily utilized"
        else:
            return "No clear bottleneck"

    def update_buffer(self, telemetry_buffer: List[Dict]):
        """Update telemetry buffer."""
        self.telemetry_buffer = telemetry_buffer
