"""
GAMESA/KrystalStack Metrics Logger

Lightweight metrics collection and export:
- Latency tracking
- Thermal monitoring
- Budget usage
- Decision statistics
- CSV/log export
"""

import time
import csv
import os
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from collections import deque
from datetime import datetime
import threading


@dataclass
class MetricSample:
    """Single metric sample."""
    name: str
    value: float
    timestamp: float
    tags: Dict[str, str] = field(default_factory=dict)


class MetricsCollector:
    """Collects and aggregates metrics."""

    def __init__(self, buffer_size: int = 10000):
        self.samples: deque = deque(maxlen=buffer_size)
        self.counters: Dict[str, int] = {}
        self.gauges: Dict[str, float] = {}
        self.histograms: Dict[str, List[float]] = {}
        self._lock = threading.Lock()

    def record(self, name: str, value: float, tags: Optional[Dict[str, str]] = None):
        """Record metric sample."""
        with self._lock:
            sample = MetricSample(
                name=name,
                value=value,
                timestamp=time.time(),
                tags=tags or {}
            )
            self.samples.append(sample)
            self.gauges[name] = value

            if name not in self.histograms:
                self.histograms[name] = []
            self.histograms[name].append(value)
            if len(self.histograms[name]) > 1000:
                self.histograms[name] = self.histograms[name][-1000:]

    def increment(self, name: str, delta: int = 1):
        """Increment counter."""
        with self._lock:
            self.counters[name] = self.counters.get(name, 0) + delta

    def get_gauge(self, name: str) -> float:
        """Get current gauge value."""
        return self.gauges.get(name, 0.0)

    def get_counter(self, name: str) -> int:
        """Get counter value."""
        return self.counters.get(name, 0)

    def get_percentile(self, name: str, p: float) -> float:
        """Get percentile of histogram."""
        with self._lock:
            if name not in self.histograms or not self.histograms[name]:
                return 0.0
            sorted_vals = sorted(self.histograms[name])
            idx = int(len(sorted_vals) * p / 100)
            return sorted_vals[min(idx, len(sorted_vals) - 1)]

    def get_stats(self, name: str) -> Dict[str, float]:
        """Get statistics for metric."""
        with self._lock:
            if name not in self.histograms or not self.histograms[name]:
                return {"count": 0, "min": 0, "max": 0, "avg": 0, "p50": 0, "p95": 0, "p99": 0}

            vals = self.histograms[name]
            sorted_vals = sorted(vals)
            n = len(vals)

            return {
                "count": n,
                "min": min(vals),
                "max": max(vals),
                "avg": sum(vals) / n,
                "p50": sorted_vals[n // 2],
                "p95": sorted_vals[int(n * 0.95)],
                "p99": sorted_vals[int(n * 0.99)]
            }


class CSVExporter:
    """Export metrics to CSV."""

    def __init__(self, output_dir: str = "./metrics"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def export_samples(self, collector: MetricsCollector, filename: str = "samples.csv"):
        """Export raw samples to CSV."""
        path = os.path.join(self.output_dir, filename)
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "name", "value", "tags"])
            for sample in collector.samples:
                writer.writerow([
                    datetime.fromtimestamp(sample.timestamp).isoformat(),
                    sample.name,
                    sample.value,
                    str(sample.tags)
                ])
        return path

    def export_summary(self, collector: MetricsCollector, filename: str = "summary.csv"):
        """Export summary statistics to CSV."""
        path = os.path.join(self.output_dir, filename)
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["metric", "count", "min", "max", "avg", "p50", "p95", "p99"])
            for name in collector.histograms:
                stats = collector.get_stats(name)
                writer.writerow([
                    name,
                    stats["count"],
                    f"{stats['min']:.4f}",
                    f"{stats['max']:.4f}",
                    f"{stats['avg']:.4f}",
                    f"{stats['p50']:.4f}",
                    f"{stats['p95']:.4f}",
                    f"{stats['p99']:.4f}",
                ])
        return path


# =============================================================================
# Metric Hooks for Integration
# =============================================================================

class BrainMetrics:
    """Metrics hooks for UnifiedBrain."""

    def __init__(self, collector: Optional[MetricsCollector] = None):
        self.collector = collector or MetricsCollector()
        self._start_times: Dict[str, float] = {}

    def start_timer(self, name: str):
        """Start timing operation."""
        self._start_times[name] = time.time()

    def stop_timer(self, name: str):
        """Stop timing and record latency."""
        if name in self._start_times:
            elapsed = (time.time() - self._start_times[name]) * 1000  # ms
            self.collector.record(f"latency_{name}_ms", elapsed)
            del self._start_times[name]

    def record_decision(self, action: str, source: str):
        """Record decision metric."""
        self.collector.increment(f"decisions_{action}")
        self.collector.increment(f"decisions_source_{source}")

    def record_thermal(self, headroom: float, temperature: float):
        """Record thermal metrics."""
        self.collector.record("thermal_headroom_c", headroom)
        self.collector.record("temperature_c", temperature)

    def record_budgets(self, cpu_mw: float, gpu_mw: float, thermal_c: float, latency_ms: float):
        """Record budget usage."""
        self.collector.record("budget_cpu_mw", cpu_mw)
        self.collector.record("budget_gpu_mw", gpu_mw)
        self.collector.record("budget_thermal_c", thermal_c)
        self.collector.record("budget_latency_ms", latency_ms)

    def record_violation(self, constraint: str, severity: str):
        """Record safety violation."""
        self.collector.increment(f"violations_{constraint}")
        self.collector.increment(f"violations_severity_{severity}")

    def record_stress(self, stress: float):
        """Record homeostatic stress."""
        self.collector.record("homeostatic_stress", stress)

    def record_surprise(self, surprise: float):
        """Record prediction surprise."""
        self.collector.record("predictive_surprise", surprise)

    def get_kpi_summary(self) -> Dict[str, Any]:
        """Get KPI summary for reporting."""
        return {
            "latency": {
                "process": self.collector.get_stats("latency_process_ms"),
                "cognitive": self.collector.get_stats("latency_cognitive_ms"),
                "invention": self.collector.get_stats("latency_invention_ms"),
            },
            "thermal": {
                "headroom": self.collector.get_stats("thermal_headroom_c"),
                "temperature": self.collector.get_stats("temperature_c"),
            },
            "violations": {
                name.replace("violations_", ""): count
                for name, count in self.collector.counters.items()
                if name.startswith("violations_")
            },
            "decisions": {
                name.replace("decisions_", ""): count
                for name, count in self.collector.counters.items()
                if name.startswith("decisions_")
            },
            "stress": self.collector.get_stats("homeostatic_stress"),
            "surprise": self.collector.get_stats("predictive_surprise"),
        }


# =============================================================================
# Feature Toggle for Invention/Emergent
# =============================================================================

@dataclass
class FeatureFlags:
    """Feature flags for system configuration."""
    enable_invention_engine: bool = True
    enable_emergent_system: bool = True
    enable_distributed_brain: bool = False
    enable_metrics_logging: bool = True
    enable_shadow_rules: bool = True
    environment: str = "lab"  # "lab" or "prod"

    @classmethod
    def production(cls) -> "FeatureFlags":
        """Production-safe defaults."""
        return cls(
            enable_invention_engine=False,
            enable_emergent_system=False,
            enable_distributed_brain=False,
            enable_metrics_logging=True,
            enable_shadow_rules=False,
            environment="prod"
        )

    @classmethod
    def laboratory(cls) -> "FeatureFlags":
        """Lab environment with all features."""
        return cls(
            enable_invention_engine=True,
            enable_emergent_system=True,
            enable_distributed_brain=True,
            enable_metrics_logging=True,
            enable_shadow_rules=True,
            environment="lab"
        )


# Global metrics instance
_global_metrics: Optional[BrainMetrics] = None
_global_flags: Optional[FeatureFlags] = None


def get_metrics() -> BrainMetrics:
    """Get global metrics instance."""
    global _global_metrics
    if _global_metrics is None:
        _global_metrics = BrainMetrics()
    return _global_metrics


def get_flags() -> FeatureFlags:
    """Get global feature flags."""
    global _global_flags
    if _global_flags is None:
        _global_flags = FeatureFlags.laboratory()
    return _global_flags


def set_flags(flags: FeatureFlags):
    """Set global feature flags."""
    global _global_flags
    _global_flags = flags


def configure_for_production():
    """Configure system for production."""
    set_flags(FeatureFlags.production())


def configure_for_lab():
    """Configure system for lab/development."""
    set_flags(FeatureFlags.laboratory())
