"""
Performance Profiler 📊
Responsibility:
1. Track API response times.
2. Identify slow endpoints.
3. Generate performance reports.
"""
import time
import statistics
from typing import Dict, Any, List
from collections import defaultdict
from datetime import datetime, timezone

class PerformanceProfiler:
    """Tracks and reports API performance metrics."""
    
    def __init__(self, max_samples: int = 1000):
        self.max_samples = max_samples
        self.response_times: Dict[str, List[float]] = defaultdict(list)
        self.error_counts: Dict[str, int] = defaultdict(int)
        self.request_counts: Dict[str, int] = defaultdict(int)
        self._start_times: Dict[str, float] = {}

    def start_request(self, request_id: str, endpoint: str):
        """Marks the start of a request."""
        self._start_times[request_id] = time.time()
        self.request_counts[endpoint] += 1

    def end_request(self, request_id: str, endpoint: str, error: bool = False):
        """Marks the end of a request and records timing."""
        if request_id in self._start_times:
            duration = (time.time() - self._start_times[request_id]) * 1000  # ms
            del self._start_times[request_id]
            
            self.response_times[endpoint].append(duration)
            if len(self.response_times[endpoint]) > self.max_samples:
                self.response_times[endpoint] = self.response_times[endpoint][-self.max_samples:]
            
            if error:
                self.error_counts[endpoint] += 1

    def get_endpoint_stats(self, endpoint: str) -> Dict[str, Any]:
        """Returns statistics for a specific endpoint."""
        times = self.response_times.get(endpoint, [])
        if not times:
            return {"endpoint": endpoint, "no_data": True}
        
        return {
            "endpoint": endpoint,
            "request_count": self.request_counts[endpoint],
            "error_count": self.error_counts[endpoint],
            "error_rate": round(self.error_counts[endpoint] / max(self.request_counts[endpoint], 1) * 100, 2),
            "avg_ms": round(statistics.mean(times), 2),
            "min_ms": round(min(times), 2),
            "max_ms": round(max(times), 2),
            "median_ms": round(statistics.median(times), 2),
            "p95_ms": round(self._percentile(times, 95), 2),
            "p99_ms": round(self._percentile(times, 99), 2),
            "sample_count": len(times)
        }

    def get_all_stats(self) -> Dict[str, Any]:
        """Returns statistics for all endpoints."""
        endpoints = {}
        for endpoint in self.response_times.keys():
            endpoints[endpoint] = self.get_endpoint_stats(endpoint)
        return {
            "endpoints": endpoints,
            "total_requests": sum(self.request_counts.values()),
            "total_errors": sum(self.error_counts.values()),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

    def get_slowest_endpoints(self, top_n: int = 10) -> List[Dict[str, Any]]:
        """Returns the slowest endpoints by average response time."""
        stats = []
        for endpoint in self.response_times.keys():
            s = self.get_endpoint_stats(endpoint)
            if not s.get("no_data"):
                stats.append(s)
        return sorted(stats, key=lambda x: x.get("avg_ms", 0), reverse=True)[:top_n]

    def _percentile(self, data: List[float], percentile: int) -> float:
        """Calculates percentile of data."""
        if not data:
            return 0
        sorted_data = sorted(data)
        index = (len(sorted_data) - 1) * percentile / 100
        lower = int(index)
        upper = lower + 1
        if upper >= len(sorted_data):
            return sorted_data[-1]
        weight = index - lower
        return sorted_data[lower] * (1 - weight) + sorted_data[upper] * weight

    def reset(self):
        """Resets all profiling data."""
        self.response_times.clear()
        self.error_counts.clear()
        self.request_counts.clear()
        self._start_times.clear()


# Global Instance
performance_profiler = PerformanceProfiler()
