"""
GAMESA/KrystalStack Benchmark Harness

Measures actual KPIs against PERFORMANCE_PROJECTIONS.md targets:
- Decision latency (target: <10ms p99)
- Thermal stability (target: <5°C variance)
- Resource utilization efficiency
- Rule throughput
- Anomaly detection rate
"""

import time
import statistics
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import random


@dataclass
class BenchmarkResult:
    """Results from a benchmark run."""
    name: str
    iterations: int
    latencies_ms: List[float] = field(default_factory=list)
    throughput_ops: float = 0.0
    memory_mb: float = 0.0
    errors: int = 0

    @property
    def p50(self) -> float:
        if not self.latencies_ms:
            return 0.0
        return statistics.median(self.latencies_ms)

    @property
    def p95(self) -> float:
        if len(self.latencies_ms) < 20:
            return max(self.latencies_ms) if self.latencies_ms else 0.0
        sorted_lat = sorted(self.latencies_ms)
        idx = int(len(sorted_lat) * 0.95)
        return sorted_lat[idx]

    @property
    def p99(self) -> float:
        if len(self.latencies_ms) < 100:
            return max(self.latencies_ms) if self.latencies_ms else 0.0
        sorted_lat = sorted(self.latencies_ms)
        idx = int(len(sorted_lat) * 0.99)
        return sorted_lat[idx]

    @property
    def mean(self) -> float:
        return statistics.mean(self.latencies_ms) if self.latencies_ms else 0.0


@dataclass
class KPITargets:
    """KPI targets from PERFORMANCE_PROJECTIONS.md."""
    decision_latency_p99_ms: float = 10.0
    thermal_variance_c: float = 5.0
    rule_throughput_ops: float = 10000.0
    memory_efficiency_percent: float = 80.0
    anomaly_detection_rate: float = 0.95


class BenchmarkHarness:
    """
    Run benchmarks against GAMESA systems.
    """

    def __init__(self):
        self.results: List[BenchmarkResult] = []
        self.targets = KPITargets()

    def benchmark_decision_latency(self, iterations: int = 100) -> BenchmarkResult:
        """Benchmark UnifiedBrain decision latency."""
        from unified_brain import UnifiedBrain
        from metrics_logger import FeatureFlags

        # Use production flags for fast benchmarking (no invention/emergent)
        flags = FeatureFlags.production()
        brain = UnifiedBrain(flags=flags)

        result = BenchmarkResult(name="decision_latency", iterations=iterations)

        # Warm-up (reduced)
        for _ in range(3):
            telemetry = self._generate_telemetry()
            brain.process(telemetry)

        # Benchmark
        for i in range(iterations):
            telemetry = self._generate_telemetry()
            start = time.perf_counter()
            try:
                brain.process(telemetry)
            except Exception:
                result.errors += 1
            elapsed_ms = (time.perf_counter() - start) * 1000
            result.latencies_ms.append(elapsed_ms)

        result.throughput_ops = iterations / (sum(result.latencies_ms) / 1000)
        self.results.append(result)
        return result

    def benchmark_rule_engine(self, iterations: int = 5000) -> BenchmarkResult:
        """Benchmark rule evaluation throughput."""
        from cognitive_engine import RuleEngine, MicroInferenceRule, SafetyTier

        engine = RuleEngine()

        # Add typical rules
        rules = [
            MicroInferenceRule(
                rule_id=f"bench_rule_{i}",
                condition=f"temp > {60 + i}",
                action="throttle",
                priority=i,
                safety_tier=SafetyTier.STRICT,
            )
            for i in range(20)
        ]
        for r in rules:
            engine.add_rule(r)

        result = BenchmarkResult(name="rule_engine", iterations=iterations)

        for _ in range(iterations):
            context = {"temp": random.uniform(50, 100), "cpu_util": random.uniform(0.3, 1.0)}
            start = time.perf_counter()
            try:
                engine.evaluate(context)
            except Exception:
                result.errors += 1
            elapsed_ms = (time.perf_counter() - start) * 1000
            result.latencies_ms.append(elapsed_ms)

        result.throughput_ops = iterations / (sum(result.latencies_ms) / 1000)
        self.results.append(result)
        return result

    def benchmark_safety_guardrails(self, iterations: int = 2000) -> BenchmarkResult:
        """Benchmark safety validation throughput."""
        from cognitive_engine import SafetyGuardrails

        guardrails = SafetyGuardrails()
        result = BenchmarkResult(name="safety_guardrails", iterations=iterations)

        for _ in range(iterations):
            telemetry = self._generate_telemetry()
            start = time.perf_counter()
            try:
                guardrails.validate(telemetry)
            except Exception:
                result.errors += 1
            elapsed_ms = (time.perf_counter() - start) * 1000
            result.latencies_ms.append(elapsed_ms)

        result.throughput_ops = iterations / (sum(result.latencies_ms) / 1000)
        self.results.append(result)
        return result

    def benchmark_invention_engine(self, iterations: int = 50) -> BenchmarkResult:
        """Benchmark invention engine processing."""
        from invention_engine import create_invention_engine

        engine = create_invention_engine()
        result = BenchmarkResult(name="invention_engine", iterations=iterations)

        for _ in range(iterations):
            telemetry = self._generate_telemetry()
            start = time.perf_counter()
            try:
                engine.process(telemetry)
            except Exception:
                result.errors += 1
            elapsed_ms = (time.perf_counter() - start) * 1000
            result.latencies_ms.append(elapsed_ms)

        result.throughput_ops = iterations / (sum(result.latencies_ms) / 1000)
        self.results.append(result)
        return result

    def benchmark_emergent_system(self, iterations: int = 50) -> BenchmarkResult:
        """Benchmark emergent system processing."""
        from emergent_system import create_emergent_system

        system = create_emergent_system()
        result = BenchmarkResult(name="emergent_system", iterations=iterations)

        for _ in range(iterations):
            telemetry = self._generate_telemetry()
            start = time.perf_counter()
            try:
                system.process(telemetry)
            except Exception:
                result.errors += 1
            elapsed_ms = (time.perf_counter() - start) * 1000
            result.latencies_ms.append(elapsed_ms)

        result.throughput_ops = iterations / (sum(result.latencies_ms) / 1000)
        self.results.append(result)
        return result

    def _generate_telemetry(self) -> Dict[str, float]:
        """Generate realistic telemetry."""
        return {
            "temperature": random.uniform(55, 85),
            "thermal_headroom": random.uniform(5, 25),
            "power_draw": random.uniform(15, 28),
            "cpu_util": random.uniform(0.3, 0.9),
            "gpu_util": random.uniform(0.2, 0.8),
            "memory_util": random.uniform(0.4, 0.8),
            "fps": random.uniform(30, 120),
            "frametime": random.uniform(8, 33),
            "latency": random.uniform(5, 20),
        }

    def run_all(self) -> Dict[str, BenchmarkResult]:
        """Run all benchmarks."""
        print("Running GAMESA Benchmark Suite...\n")

        benchmarks = [
            ("Safety Guardrails", self.benchmark_safety_guardrails),
            ("Rule Engine", self.benchmark_rule_engine),
            ("Invention Engine", self.benchmark_invention_engine),
            ("Emergent System", self.benchmark_emergent_system),
            ("Full Decision Loop", self.benchmark_decision_latency),
        ]

        results = {}
        for name, bench_fn in benchmarks:
            print(f"  [{name}]...", end=" ", flush=True)
            result = bench_fn()
            status = "PASS" if result.p99 < self.targets.decision_latency_p99_ms else "WARN"
            print(f"{status} p99={result.p99:.2f}ms, throughput={result.throughput_ops:.0f} ops/s")
            results[result.name] = result

        return results

    def generate_report(self) -> str:
        """Generate benchmark report."""
        lines = [
            "=" * 60,
            "GAMESA/KrystalStack Benchmark Report",
            "=" * 60,
            "",
        ]

        for result in self.results:
            target_met = result.p99 < self.targets.decision_latency_p99_ms
            status = "✓ PASS" if target_met else "✗ FAIL"

            lines.extend([
                f"## {result.name}",
                f"   Status: {status}",
                f"   Iterations: {result.iterations}",
                f"   Latency p50: {result.p50:.3f} ms",
                f"   Latency p95: {result.p95:.3f} ms",
                f"   Latency p99: {result.p99:.3f} ms (target: <{self.targets.decision_latency_p99_ms} ms)",
                f"   Throughput: {result.throughput_ops:.0f} ops/s",
                f"   Errors: {result.errors}",
                "",
            ])

        # KPI summary
        lines.extend([
            "=" * 60,
            "KPI Summary vs Targets",
            "=" * 60,
        ])

        decision_result = next((r for r in self.results if r.name == "decision_latency"), None)
        if decision_result:
            met = decision_result.p99 < self.targets.decision_latency_p99_ms
            lines.append(f"Decision Latency p99: {decision_result.p99:.2f}ms / {self.targets.decision_latency_p99_ms}ms - {'MET' if met else 'NOT MET'}")

        rule_result = next((r for r in self.results if r.name == "rule_engine"), None)
        if rule_result:
            met = rule_result.throughput_ops > self.targets.rule_throughput_ops
            lines.append(f"Rule Throughput: {rule_result.throughput_ops:.0f} / {self.targets.rule_throughput_ops:.0f} ops/s - {'MET' if met else 'NOT MET'}")

        return "\n".join(lines)


def run_benchmarks():
    """Run benchmark suite and print report."""
    import sys
    sys.path.insert(0, "/home/user/Dev-contitional/src/python")

    harness = BenchmarkHarness()
    harness.run_all()
    print("\n" + harness.generate_report())


if __name__ == "__main__":
    run_benchmarks()
