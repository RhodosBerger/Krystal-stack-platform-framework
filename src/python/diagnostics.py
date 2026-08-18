"""
GAMESA Diagnostics - Health Monitoring and Self-Test System

Provides:
- Component health checks
- Performance benchmarking
- Anomaly detection
- Self-healing suggestions
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Any, Tuple
from enum import Enum, auto
from collections import deque
import time
import statistics


class HealthStatus(Enum):
    """Component health status."""
    HEALTHY = auto()
    DEGRADED = auto()
    WARNING = auto()
    CRITICAL = auto()
    UNKNOWN = auto()


class DiagnosticCategory(Enum):
    """Diagnostic test categories."""
    PERFORMANCE = auto()
    THERMAL = auto()
    MEMORY = auto()
    LEARNING = auto()
    INTEGRATION = auto()


@dataclass
class DiagnosticResult:
    """Result of a diagnostic test."""
    name: str
    category: DiagnosticCategory
    status: HealthStatus
    message: str
    value: Optional[float] = None
    threshold: Optional[float] = None
    timestamp: float = field(default_factory=time.time)
    suggestions: List[str] = field(default_factory=list)


@dataclass
class HealthReport:
    """Comprehensive health report."""
    overall_status: HealthStatus
    component_status: Dict[str, HealthStatus]
    diagnostics: List[DiagnosticResult]
    metrics: Dict[str, float]
    timestamp: float = field(default_factory=time.time)
    uptime: float = 0.0


class AnomalyDetector:
    """
    Detects anomalies in time series data using statistical methods.
    """

    def __init__(self, window_size: int = 100, threshold_sigma: float = 3.0):
        self.window_size = window_size
        self.threshold_sigma = threshold_sigma
        self._buffers: Dict[str, deque] = {}
        self._anomaly_counts: Dict[str, int] = {}

    def observe(self, metric: str, value: float) -> Optional[Dict]:
        """
        Observe a metric value and detect anomalies.

        Returns:
            Anomaly info dict if detected, None otherwise
        """
        if metric not in self._buffers:
            self._buffers[metric] = deque(maxlen=self.window_size)
            self._anomaly_counts[metric] = 0

        buffer = self._buffers[metric]

        if len(buffer) >= 10:
            mean = statistics.mean(buffer)
            std = statistics.stdev(buffer) if len(buffer) > 1 else 0.001

            z_score = abs(value - mean) / max(std, 0.001)

            if z_score > self.threshold_sigma:
                self._anomaly_counts[metric] += 1
                buffer.append(value)
                return {
                    "metric": metric,
                    "value": value,
                    "mean": mean,
                    "std": std,
                    "z_score": z_score,
                    "direction": "high" if value > mean else "low"
                }

        buffer.append(value)
        return None

    def get_stats(self, metric: str) -> Dict:
        """Get statistics for a metric."""
        buffer = self._buffers.get(metric, [])
        if not buffer:
            return {}

        return {
            "count": len(buffer),
            "mean": statistics.mean(buffer),
            "std": statistics.stdev(buffer) if len(buffer) > 1 else 0,
            "min": min(buffer),
            "max": max(buffer),
            "anomalies": self._anomaly_counts.get(metric, 0)
        }


class DiagnosticEngine:
    """
    Runs diagnostic tests and generates health reports.
    """

    def __init__(self):
        self._tests: Dict[str, Callable[[], DiagnosticResult]] = {}
        self._history: deque = deque(maxlen=1000)
        self._anomaly_detector = AnomalyDetector()
        self._start_time = time.time()
        self._component_status: Dict[str, HealthStatus] = {}

        # Register built-in tests
        self._register_builtin_tests()

    def _register_builtin_tests(self):
        """Register default diagnostic tests."""

        def test_memory_usage():
            # Simulated - in real use would check actual memory
            import sys
            objects = len([1 for _ in range(100)])  # Placeholder
            status = HealthStatus.HEALTHY if objects < 10000 else HealthStatus.WARNING
            return DiagnosticResult(
                name="memory_usage",
                category=DiagnosticCategory.MEMORY,
                status=status,
                message=f"Memory objects tracked: {objects}",
                value=objects,
                threshold=10000
            )

        def test_response_time():
            start = time.time()
            # Simulated work
            _ = [i * i for i in range(1000)]
            elapsed = (time.time() - start) * 1000

            if elapsed < 1:
                status = HealthStatus.HEALTHY
            elif elapsed < 10:
                status = HealthStatus.DEGRADED
            else:
                status = HealthStatus.WARNING

            return DiagnosticResult(
                name="response_time",
                category=DiagnosticCategory.PERFORMANCE,
                status=status,
                message=f"Basic operation: {elapsed:.2f}ms",
                value=elapsed,
                threshold=10.0
            )

        self._tests["memory_usage"] = test_memory_usage
        self._tests["response_time"] = test_response_time

    def register_test(self, name: str, test_fn: Callable[[], DiagnosticResult]):
        """Register a diagnostic test."""
        self._tests[name] = test_fn

    def run_test(self, name: str) -> Optional[DiagnosticResult]:
        """Run a specific diagnostic test."""
        test_fn = self._tests.get(name)
        if not test_fn:
            return None

        try:
            result = test_fn()
            self._history.append(result)
            return result
        except Exception as e:
            return DiagnosticResult(
                name=name,
                category=DiagnosticCategory.INTEGRATION,
                status=HealthStatus.CRITICAL,
                message=f"Test failed: {str(e)}"
            )

    def run_all_tests(self) -> List[DiagnosticResult]:
        """Run all registered diagnostic tests."""
        results = []
        for name in self._tests:
            result = self.run_test(name)
            if result:
                results.append(result)
        return results

    def run_category(self, category: DiagnosticCategory) -> List[DiagnosticResult]:
        """Run tests in a specific category."""
        results = []
        for name, test_fn in self._tests.items():
            result = self.run_test(name)
            if result and result.category == category:
                results.append(result)
        return results

    def update_component(self, component: str, status: HealthStatus):
        """Update component health status."""
        self._component_status[component] = status

    def observe_metric(self, metric: str, value: float) -> Optional[Dict]:
        """Observe metric and check for anomalies."""
        return self._anomaly_detector.observe(metric, value)

    def generate_report(self) -> HealthReport:
        """Generate comprehensive health report."""
        results = self.run_all_tests()

        # Determine overall status
        statuses = [r.status for r in results] + list(self._component_status.values())

        if HealthStatus.CRITICAL in statuses:
            overall = HealthStatus.CRITICAL
        elif HealthStatus.WARNING in statuses:
            overall = HealthStatus.WARNING
        elif HealthStatus.DEGRADED in statuses:
            overall = HealthStatus.DEGRADED
        elif all(s == HealthStatus.HEALTHY for s in statuses):
            overall = HealthStatus.HEALTHY
        else:
            overall = HealthStatus.UNKNOWN

        # Collect metrics
        metrics = {}
        for name in self._anomaly_detector._buffers:
            stats = self._anomaly_detector.get_stats(name)
            if stats:
                metrics[f"{name}_mean"] = stats["mean"]
                metrics[f"{name}_anomalies"] = stats["anomalies"]

        return HealthReport(
            overall_status=overall,
            component_status=self._component_status.copy(),
            diagnostics=results,
            metrics=metrics,
            uptime=time.time() - self._start_time
        )

    def get_suggestions(self, report: HealthReport) -> List[str]:
        """Generate suggestions based on health report."""
        suggestions = []

        for diag in report.diagnostics:
            if diag.status in [HealthStatus.WARNING, HealthStatus.CRITICAL]:
                suggestions.extend(diag.suggestions)

                if diag.category == DiagnosticCategory.THERMAL:
                    suggestions.append("Consider activating power_saver profile")
                elif diag.category == DiagnosticCategory.MEMORY:
                    suggestions.append("Consider clearing caches or reducing buffer sizes")
                elif diag.category == DiagnosticCategory.PERFORMANCE:
                    suggestions.append("Consider reducing workload or batch size")

        return list(set(suggestions))


class SelfHealingManager:
    """
    Automated recovery actions based on diagnostics.
    """

    def __init__(self, diagnostics: DiagnosticEngine):
        self.diagnostics = diagnostics
        self._recovery_actions: Dict[str, Callable[[], bool]] = {}
        self._recovery_history: List[Dict] = []

    def register_recovery(self, condition: str, action: Callable[[], bool]):
        """Register recovery action for a condition."""
        self._recovery_actions[condition] = action

    def check_and_heal(self) -> List[Dict]:
        """Check health and attempt recovery actions."""
        report = self.diagnostics.generate_report()
        actions_taken = []

        for diag in report.diagnostics:
            if diag.status in [HealthStatus.WARNING, HealthStatus.CRITICAL]:
                action = self._recovery_actions.get(diag.name)
                if action:
                    try:
                        success = action()
                        actions_taken.append({
                            "condition": diag.name,
                            "success": success,
                            "timestamp": time.time()
                        })
                        self._recovery_history.append(actions_taken[-1])
                    except Exception as e:
                        actions_taken.append({
                            "condition": diag.name,
                            "success": False,
                            "error": str(e),
                            "timestamp": time.time()
                        })

        return actions_taken


# ============================================================
# FRAMEWORK DIAGNOSTIC TESTS
# ============================================================

def create_framework_tests(framework) -> Dict[str, Callable[[], DiagnosticResult]]:
    """Create diagnostic tests for GAMESA framework."""

    tests = {}

    def test_cognitive_health():
        try:
            state = framework.cognitive.get_state()
            fitness = state.get("fitness", 0)

            if fitness > 0.7:
                status = HealthStatus.HEALTHY
            elif fitness > 0.4:
                status = HealthStatus.DEGRADED
            else:
                status = HealthStatus.WARNING

            return DiagnosticResult(
                name="cognitive_health",
                category=DiagnosticCategory.LEARNING,
                status=status,
                message=f"Cognitive fitness: {fitness:.2f}",
                value=fitness,
                threshold=0.5
            )
        except Exception as e:
            return DiagnosticResult(
                name="cognitive_health",
                category=DiagnosticCategory.LEARNING,
                status=HealthStatus.CRITICAL,
                message=f"Failed: {e}"
            )

    def test_thermal_status():
        metrics = framework.metrics
        headroom = metrics.thermal_headroom

        if headroom > 15:
            status = HealthStatus.HEALTHY
        elif headroom > 5:
            status = HealthStatus.DEGRADED
        else:
            status = HealthStatus.WARNING

        return DiagnosticResult(
            name="thermal_status",
            category=DiagnosticCategory.THERMAL,
            status=status,
            message=f"Thermal headroom: {headroom:.1f}C",
            value=headroom,
            threshold=10.0,
            suggestions=["Reduce GPU load", "Increase fan speed"] if headroom < 10 else []
        )

    def test_exchange_liquidity():
        try:
            state = framework.exchange.tick()
            orders = state.get("active_orders", 0)

            status = HealthStatus.HEALTHY if orders < 100 else HealthStatus.DEGRADED

            return DiagnosticResult(
                name="exchange_liquidity",
                category=DiagnosticCategory.INTEGRATION,
                status=status,
                message=f"Active orders: {orders}",
                value=orders,
                threshold=100
            )
        except Exception as e:
            return DiagnosticResult(
                name="exchange_liquidity",
                category=DiagnosticCategory.INTEGRATION,
                status=HealthStatus.CRITICAL,
                message=f"Failed: {e}"
            )

    tests["cognitive_health"] = test_cognitive_health
    tests["thermal_status"] = test_thermal_status
    tests["exchange_liquidity"] = test_exchange_liquidity

    return tests


# ============================================================
# DEMO
# ============================================================

def demo():
    """Demonstrate diagnostics system."""
    print("=== GAMESA Diagnostics Demo ===\n")

    engine = DiagnosticEngine()

    # Run all tests
    print("Running diagnostic tests:")
    results = engine.run_all_tests()
    for r in results:
        print(f"  [{r.status.name}] {r.name}: {r.message}")

    # Observe metrics and detect anomalies
    print("\n--- Anomaly Detection ---")
    import random

    for i in range(50):
        val = 50 + random.gauss(0, 5)
        engine.observe_metric("cpu_temp", val)

    # Inject anomaly
    anomaly = engine.observe_metric("cpu_temp", 90)
    if anomaly:
        print(f"  Anomaly detected: {anomaly}")

    print(f"  Stats: {engine._anomaly_detector.get_stats('cpu_temp')}")

    # Generate report
    print("\n--- Health Report ---")
    report = engine.generate_report()
    print(f"  Overall: {report.overall_status.name}")
    print(f"  Uptime: {report.uptime:.1f}s")
    print(f"  Metrics: {report.metrics}")

    suggestions = engine.get_suggestions(report)
    if suggestions:
        print(f"  Suggestions: {suggestions}")


if __name__ == "__main__":
    demo()
