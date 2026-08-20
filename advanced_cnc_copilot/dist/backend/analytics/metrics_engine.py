"""
Metrics Engine 📊
Responsibility:
1. Collect and aggregate system telemetry.
2. Track generation statistics, API usage, and performance metrics.
"""
import time
from typing import Dict, Any, List
from datetime import datetime, timezone

class MetricsEngine:
    def __init__(self):
        self._metrics = {
            "total_generations": 0,
            "total_payloads": 0,
            "total_products": 0,
            "total_api_calls": 0,
            "avg_generation_time_ms": 0,
            "success_rate": 100.0
        }
        self._timeline: List[Dict[str, Any]] = []
        self._generation_times: List[float] = []

    def record_generation(self, payload_count: int, duration_ms: float, success: bool = True):
        """Records a generation event."""
        self._metrics["total_generations"] += 1
        self._metrics["total_payloads"] += payload_count
        self._generation_times.append(duration_ms)
        self._metrics["avg_generation_time_ms"] = sum(self._generation_times) / len(self._generation_times)
        
        if not success:
            total = self._metrics["total_generations"]
            fails = (1 - self._metrics["success_rate"] / 100) * (total - 1) + 1
            self._metrics["success_rate"] = ((total - fails) / total) * 100

        self._timeline.append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event": "GENERATION",
            "payload_count": payload_count,
            "duration_ms": duration_ms,
            "success": success
        })

    def record_api_call(self, endpoint: str):
        """Records an API call."""
        self._metrics["total_api_calls"] += 1

    def increment_products(self):
        """Increments product count."""
        self._metrics["total_products"] += 1

    def get_metrics(self) -> Dict[str, Any]:
        return {
            **self._metrics,
            "last_updated": datetime.now(timezone.utc).isoformat()
        }

    def get_timeline(self, limit: int = 20) -> List[Dict[str, Any]]:
        return self._timeline[-limit:]

# Global Instance
metrics_engine = MetricsEngine()
