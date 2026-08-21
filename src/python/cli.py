"""
GAMESA CLI - Command Line Interface

Main entry point for running the unified system.
"""

import argparse
import sys
import time
import json
from typing import Optional

def cmd_run(args):
    """Run the unified system."""
    from .unified_system import create_unified_system

    print(f"Starting GAMESA Unified System...")
    print(f"  Cycles: {args.cycles}")
    print(f"  Verbose: {args.verbose}")
    print()

    system = create_unified_system()
    system.start()

    results = []
    start_time = time.time()

    for i in range(args.cycles):
        result = system.tick()
        results.append(result)

        if args.verbose or (i + 1) % 10 == 0:
            print(f"[{i+1:4d}] Mode={result['mode']:12s} "
                  f"Reward={result['reward']:.3f} "
                  f"Phase={result['emergence']['phase']:8s}")

    elapsed = time.time() - start_time
    system.stop()

    # Summary
    print()
    print("=" * 50)
    print("Summary")
    print("=" * 50)

    summary = system.get_state_summary()
    print(f"  Final Mode: {summary['mode']}")
    print(f"  Cycles: {summary['cycle_count']}")
    print(f"  Elapsed: {elapsed:.2f}s ({args.cycles/elapsed:.1f} cycles/sec)")
    print(f"  Avg Reward: {sum(r['reward'] for r in results)/len(results):.3f}")
    print(f"  Consciousness: {summary['consciousness']}")
    print(f"  Attractors: {summary['emergence']['attractor_count']}")

    if args.output:
        with open(args.output, 'w') as f:
            json.dump({
                "summary": summary,
                "results": results[-100:]  # Last 100 cycles
            }, f, indent=2, default=str)
        print(f"\nResults saved to: {args.output}")


def cmd_monitor(args):
    """Run real-time monitor."""
    from .unified_system import create_unified_system
    from .realtime_monitor import create_monitor

    system = create_unified_system()
    system.start()

    monitor = create_monitor(system)
    monitor.refresh_rate = args.refresh

    print("Starting real-time monitor (Ctrl+C to exit)...")
    time.sleep(1)

    try:
        monitor.run_interactive(duration=args.duration)
    except KeyboardInterrupt:
        pass

    system.stop()


def cmd_benchmark(args):
    """Run performance benchmark."""
    from .unified_system import create_unified_system
    import numpy as np

    print("GAMESA Performance Benchmark")
    print("=" * 50)

    system = create_unified_system()
    system.start()

    # Warmup
    print("Warming up...")
    for _ in range(100):
        system.tick()

    # Benchmark
    print(f"Benchmarking {args.iterations} iterations...")
    times = []

    for _ in range(args.iterations):
        start = time.perf_counter()
        system.tick()
        elapsed = time.perf_counter() - start
        times.append(elapsed * 1000)  # Convert to ms

    system.stop()

    # Statistics
    times = np.array(times)
    print()
    print("Results:")
    print(f"  Mean:   {np.mean(times):.3f} ms/tick")
    print(f"  Median: {np.median(times):.3f} ms/tick")
    print(f"  Std:    {np.std(times):.3f} ms")
    print(f"  Min:    {np.min(times):.3f} ms")
    print(f"  Max:    {np.max(times):.3f} ms")
    print(f"  P95:    {np.percentile(times, 95):.3f} ms")
    print(f"  P99:    {np.percentile(times, 99):.3f} ms")
    print()
    print(f"  Throughput: {1000/np.mean(times):.1f} ticks/sec")


def cmd_verify(args):
    """Verify system installation."""
    print("GAMESA System Verification")
    print("=" * 50)

    checks = []

    # Check Python imports
    print("\nChecking Python imports...")
    try:
        from . import (
            create_unified_system,
            create_cognitive_engine,
            create_breakthrough_engine,
            create_hypervisor,
            create_emergent_intelligence,
            create_generative_engine,
            create_gpu_optimizer,
        )
        checks.append(("Python imports", True))
        print("  ✓ All imports successful")
    except ImportError as e:
        checks.append(("Python imports", False))
        print(f"  ✗ Import failed: {e}")

    # Check unified system
    print("\nChecking unified system...")
    try:
        system = create_unified_system()
        system.start()
        result = system.tick()
        system.stop()
        checks.append(("Unified system", True))
        print(f"  ✓ System tick successful (reward={result['reward']:.3f})")
    except Exception as e:
        checks.append(("Unified system", False))
        print(f"  ✗ System failed: {e}")

    # Check cognitive engine
    print("\nChecking cognitive engine...")
    try:
        engine = create_cognitive_engine()
        checks.append(("Cognitive engine", True))
        print("  ✓ Cognitive engine created")
    except Exception as e:
        checks.append(("Cognitive engine", False))
        print(f"  ✗ Failed: {e}")

    # Check emergent intelligence
    print("\nChecking emergent intelligence...")
    try:
        emergence = create_emergent_intelligence()
        props = emergence.get_emergent_properties()
        checks.append(("Emergent intelligence", True))
        print(f"  ✓ Phase={props['phase']}, Attractors={props['attractor_count']}")
    except Exception as e:
        checks.append(("Emergent intelligence", False))
        print(f"  ✗ Failed: {e}")

    # Summary
    print("\n" + "=" * 50)
    passed = sum(1 for _, ok in checks if ok)
    total = len(checks)
    print(f"Verification: {passed}/{total} checks passed")

    return 0 if passed == total else 1


def cmd_info(args):
    """Show system information."""
    print("GAMESA System Information")
    print("=" * 50)

    print("\nLayers:")
    layers = [
        ("Level 0", "Hardware Abstraction", "Telemetry, presets"),
        ("Level 1", "Signal Processing", "PID control"),
        ("Level 2", "Learning", "TD, Bayesian, Evolution"),
        ("Level 3", "Prediction", "Temporal, Neural fabric"),
        ("Level 4", "Emergence", "Attractors, Phase, Consciousness"),
        ("Level 5", "Generation", "Presets, Content, Code"),
    ]
    for level, name, desc in layers:
        print(f"  {level}: {name:20s} - {desc}")

    print("\nComponents:")
    components = [
        "UnifiedSystem", "CognitiveEngine", "BreakthroughEngine",
        "Hypervisor", "EmergentIntelligence", "GenerativeEngine",
        "GpuOptimizer", "RealtimeMonitor"
    ]
    for comp in components:
        print(f"  • {comp}")

    print("\nModes:")
    modes = ["INIT", "LEARNING", "OPTIMIZING", "STABLE", "GENERATING", "EMERGENCY"]
    for mode in modes:
        print(f"  • {mode}")

    print("\nPhases:")
    phases = ["SOLID", "LIQUID", "GAS", "PLASMA", "CRITICAL"]
    for phase in phases:
        print(f"  • {phase}")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="gamesa",
        description="GAMESA Unified System CLI"
    )
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Run command
    run_parser = subparsers.add_parser("run", help="Run the unified system")
    run_parser.add_argument("-c", "--cycles", type=int, default=100, help="Number of cycles")
    run_parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    run_parser.add_argument("-o", "--output", type=str, help="Output JSON file")

    # Monitor command
    monitor_parser = subparsers.add_parser("monitor", help="Real-time monitoring")
    monitor_parser.add_argument("-r", "--refresh", type=int, default=10, help="Refresh rate (Hz)")
    monitor_parser.add_argument("-d", "--duration", type=float, help="Duration in seconds")

    # Benchmark command
    bench_parser = subparsers.add_parser("benchmark", help="Performance benchmark")
    bench_parser.add_argument("-i", "--iterations", type=int, default=1000, help="Iterations")

    # Verify command
    subparsers.add_parser("verify", help="Verify installation")

    # Info command
    subparsers.add_parser("info", help="Show system information")

    args = parser.parse_args()

    if args.command == "run":
        cmd_run(args)
    elif args.command == "monitor":
        cmd_monitor(args)
    elif args.command == "benchmark":
        cmd_benchmark(args)
    elif args.command == "verify":
        sys.exit(cmd_verify(args))
    elif args.command == "info":
        cmd_info(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
