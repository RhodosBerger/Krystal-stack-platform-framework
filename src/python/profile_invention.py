"""
Profiling script for invention_engine.py

Identifies performance hotspots to guide optimization efforts.
"""

import cProfile
import pstats
import io
import time
import json
import random
from typing import Dict, List
from invention_engine import create_invention_engine


def generate_realistic_telemetry() -> Dict[str, float]:
    """Generate realistic telemetry data."""
    return {
        "temperature": random.uniform(60, 85),
        "thermal_headroom": random.uniform(5, 25),
        "power_draw": random.uniform(15, 28),
        "cpu_util": random.uniform(0.3, 0.9),
        "gpu_util": random.uniform(0.2, 0.8),
        "npu_util": random.uniform(0.1, 0.5),
        "memory_util": random.uniform(0.4, 0.8),
        "fps": random.uniform(30, 120),
        "frametime": random.uniform(8, 33),
        "latency": random.uniform(5, 20),
    }


def benchmark_invention_engine(iterations: int = 50):
    """Benchmark invention engine processing."""
    engine = create_invention_engine()

    print(f"Running {iterations} iterations...")
    print("Warming up...", flush=True)

    # Warm-up
    for _ in range(5):
        telemetry = generate_realistic_telemetry()
        engine.process(telemetry)

    print("Benchmarking...", flush=True)
    latencies = []

    for i in range(iterations):
        telemetry = generate_realistic_telemetry()
        start = time.perf_counter()
        result = engine.process(telemetry)
        elapsed_ms = (time.perf_counter() - start) * 1000
        latencies.append(elapsed_ms)

        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{iterations} completed", flush=True)

    # Statistics
    latencies.sort()
    p50 = latencies[len(latencies) // 2]
    p95 = latencies[int(len(latencies) * 0.95)]
    p99 = latencies[int(len(latencies) * 0.99)]
    mean = sum(latencies) / len(latencies)

    print(f"\nBenchmark Results:")
    print(f"  Iterations: {iterations}")
    print(f"  Mean:  {mean:.2f} ms")
    print(f"  p50:   {p50:.2f} ms")
    print(f"  p95:   {p95:.2f} ms")
    print(f"  p99:   {p99:.2f} ms")
    print(f"  Target: <10.00 ms")
    print(f"  Status: {'✓ PASS' if p99 < 10.0 else '✗ FAIL (%.2fx over target)' % (p99/10.0)}")

    return latencies


def profile_invention_engine():
    """Run profiling analysis."""
    print("=" * 70)
    print("Invention Engine Profiling")
    print("=" * 70)
    print()

    # Create profiler
    profiler = cProfile.Profile()

    # Run under profiler
    print("Running profiled benchmark...")
    profiler.enable()
    benchmark_invention_engine(iterations=50)
    profiler.disable()

    # Capture stats
    stream = io.StringIO()
    stats = pstats.Stats(profiler, stream=stream)

    print("\n" + "=" * 70)
    print("Top 20 Functions by Cumulative Time")
    print("=" * 70)
    stats.sort_stats('cumulative')
    stats.print_stats(20)
    print(stream.getvalue())

    stream = io.StringIO()
    stats = pstats.Stats(profiler, stream=stream)

    print("\n" + "=" * 70)
    print("Top 20 Functions by Total Time (tottime)")
    print("=" * 70)
    stats.sort_stats('tottime')
    stats.print_stats(20)
    print(stream.getvalue())

    # Save profile data for visualization
    profiler.dump_stats('/home/user/Dev-contitional/invention_profile.stats')
    print("\nProfile data saved to: invention_profile.stats")
    print("Visualize with: snakeviz invention_profile.stats")

    return stats


def component_breakdown():
    """Benchmark individual components."""
    print("\n" + "=" * 70)
    print("Component-Level Breakdown")
    print("=" * 70)

    from invention_engine import (
        SuperpositionScheduler,
        SpikeTimingAllocator,
        SwarmOptimizer,
        CausalInferenceEngine,
        HyperdimensionalEncoder,
        ReservoirComputer
    )

    components = {
        "SuperpositionScheduler": lambda: test_quantum(),
        "SpikeTimingAllocator": lambda: test_spiking(),
        "HyperdimensionalEncoder": lambda: test_hd(),
        "CausalInferenceEngine": lambda: test_causal(),
        "ReservoirComputer": lambda: test_reservoir(),
    }

    results = {}

    for name, test_fn in components.items():
        print(f"\n[{name}]", flush=True)
        latencies = []

        for _ in range(100):
            start = time.perf_counter()
            test_fn()
            elapsed_ms = (time.perf_counter() - start) * 1000
            latencies.append(elapsed_ms)

        latencies.sort()
        mean = sum(latencies) / len(latencies)
        p99 = latencies[99]

        print(f"  Mean: {mean:.3f} ms")
        print(f"  p99:  {p99:.3f} ms")

        results[name] = {"mean": mean, "p99": p99}

    # Summary
    print("\n" + "=" * 70)
    print("Component Performance Summary")
    print("=" * 70)

    total_mean = sum(r["mean"] for r in results.values())
    total_p99 = sum(r["p99"] for r in results.values())

    for name, result in sorted(results.items(), key=lambda x: x[1]["p99"], reverse=True):
        pct = (result["p99"] / total_p99) * 100 if total_p99 > 0 else 0
        print(f"  {name:30s}  p99={result['p99']:6.3f} ms  ({pct:5.1f}%)")

    print(f"\n  {'TOTAL (sum of components)':30s}  p99={total_p99:6.3f} ms")

    return results


def test_quantum():
    """Test quantum scheduler."""
    from invention_engine import SuperpositionScheduler

    scheduler = SuperpositionScheduler()
    task = scheduler.create_superposition(
        "test_task",
        ["boost", "throttle", "migrate", "idle"]
    )
    scheduler.interference(task.task_id, {"temp": 70, "cpu": 0.5})
    action = scheduler.measure(task.task_id)
    return action


def test_spiking():
    """Test spiking allocator."""
    from invention_engine import SpikeTimingAllocator

    allocator = SpikeTimingAllocator()
    resources = ["cpu", "gpu", "npu"]
    for r in resources:
        allocator.add_neuron(r)

    allocation = allocator.allocate_resources({
        "cpu": 0.8,
        "gpu": 0.5,
        "npu": 0.3,
    })
    return allocation


def test_hd():
    """Test hyperdimensional encoder."""
    from invention_engine import HyperdimensionalEncoder

    encoder = HyperdimensionalEncoder(dimensions=5000)
    state = {"temp": 70, "cpu": 0.5, "gpu": 0.6}

    encoded = encoder.encode_state(state)
    encoder.store("test_state", state)
    similar = encoder.query(state, top_k=3)

    return similar


def test_causal():
    """Test causal inference."""
    from invention_engine import CausalInferenceEngine

    engine = CausalInferenceEngine()

    for _ in range(20):
        obs = {
            "temp": random.uniform(60, 80),
            "cpu": random.uniform(0.3, 0.9),
            "fps": random.uniform(30, 120),
        }
        engine.observe(obs)

    edges = engine.discover_causes("temp", ["cpu", "fps"])
    return edges


def test_reservoir():
    """Test reservoir computer."""
    from invention_engine import ReservoirComputer

    reservoir = ReservoirComputer(input_size=5, reservoir_size=200)

    # Simple forward pass
    input_vec = [0.5, 0.3, 0.7, 0.2, 0.9]
    state = reservoir.step(input_vec)

    return state


if __name__ == "__main__":
    import sys
    sys.path.insert(0, "/home/user/Dev-contitional/src/python")

    # Run full profile
    profile_invention_engine()

    # Component breakdown
    component_breakdown()

    print("\n" + "=" * 70)
    print("Profiling Complete!")
    print("=" * 70)
    print("\nNext Steps:")
    print("  1. Review 'Top Functions by Cumulative Time' for hotspots")
    print("  2. Check 'Component-Level Breakdown' for major contributors")
    print("  3. Visualize with: snakeviz invention_profile.stats")
    print("  4. Focus optimization on functions taking >50% of time")
    print()
