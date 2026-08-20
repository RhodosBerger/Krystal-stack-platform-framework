#!/usr/bin/env python3
"""
Cross-Forex Resource Market Demo

Demonstrates the GAMESA/KrystalStack architecture:
- Resource trading via Allocator (the "stock exchange")
- Autonomous agents requesting resources
- Signal-based priority scheduling
- Guardian oversight with effects/contracts
- Telemetry → Decision → Learning loop
"""

import sys
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "python"))

from allocation import (
    Allocator, AllocationRequest, ResourceType, Priority,
    AllocationConstraints, AllocationStrategy, create_default_allocator
)
from signals import (
    SignalScheduler, Signal, SignalKind, SignalSource, Domain,
    telemetry_signal, safety_signal, user_signal
)
from effects import (
    Effect, Capability, EffectDeclaration, EffectChecker,
    create_guardian_checker
)
from contracts import (
    ContractValidator, create_guardian_validator,
    TELEMETRY_CONTRACT, SAFETY_CONTRACT
)
from runtime import Runtime
from feature_engine import ScaleParams


def print_header(title: str):
    print(f"\n{'='*60}")
    print(f" {title}")
    print('='*60)


def demo_resource_market():
    """Demo the Cross-Forex resource trading system."""
    print_header("CROSS-FOREX RESOURCE MARKET")

    # Create the "stock exchange"
    allocator = create_default_allocator()
    print("\n[Stock Exchange] Initialized with pools:")
    for rt, stats in allocator.stats().items():
        print(f"  {rt.value}: {stats.total_capacity} units")

    # CPU Agent trades
    print("\n[CPU Agent] Requesting 4 cores for gaming workload...")
    cpu_alloc = allocator.allocate(AllocationRequest(
        id="cpu-gaming-001",
        resource_type=ResourceType.CPU_CORES,
        amount=4,
        priority=Priority.HIGH,
        constraints=AllocationConstraints(
            affinity=[0, 2, 4, 6],  # P-cores only
            preemptible=False
        ),
        metadata={"workload": "gaming", "process": "game.exe"}
    ))
    print(f"  ✓ Allocated: {cpu_alloc.amount} cores (ID: {cpu_alloc.id})")

    # GPU Agent trades
    print("\n[GPU Agent] Requesting 2048MB VRAM...")
    gpu_alloc = allocator.allocate(AllocationRequest(
        id="gpu-vram-001",
        resource_type=ResourceType.GPU_MEMORY,
        amount=2048,
        priority=Priority.HIGH,
        constraints=AllocationConstraints(alignment=256),
        metadata={"purpose": "render_target"}
    ))
    print(f"  ✓ Allocated: {gpu_alloc.amount}MB (ID: {gpu_alloc.id})")

    # Power Agent trades
    print("\n[Power Agent] Requesting 60% power budget...")
    power_alloc = allocator.allocate(AllocationRequest(
        id="power-budget-001",
        resource_type=ResourceType.POWER_BUDGET,
        amount=60,
        priority=Priority.NORMAL,
    ))
    print(f"  ✓ Allocated: {power_alloc.amount}% (ID: {power_alloc.id})")

    # Market status
    print("\n[Market Status]")
    for rt, stats in allocator.stats().items():
        util = stats.utilization * 100
        print(f"  {rt.value}: {stats.allocated}/{stats.total_capacity} "
              f"({util:.1f}% utilized)")

    return allocator, [cpu_alloc, gpu_alloc, power_alloc]


def demo_signal_scheduling():
    """Demo the signal-first scheduling system."""
    print_header("SIGNAL-FIRST SCHEDULING")

    scheduler = SignalScheduler()

    # Simulate incoming signals
    print("\n[Incoming Signals]")

    signals = [
        telemetry_signal(SignalKind.FRAMETIME_SPIKE, 0.7, frametime_ms=22.5),
        telemetry_signal(SignalKind.CPU_BOTTLENECK, 0.6, cpu_util=0.95),
        safety_signal(SignalKind.THERMAL_WARNING, 0.8, temp_cpu=88),
        user_signal(SignalKind.USER_BOOST_REQUEST, 0.9, profile="gaming"),
        telemetry_signal(SignalKind.IDLE_DETECTED, 0.3),
    ]

    for sig in signals:
        scheduler.enqueue(sig)
        print(f"  ← {sig.kind.name} (strength={sig.strength:.1f})")

    # Process by priority
    print("\n[Processing by Priority]")
    print("  Domain ranking: SAFETY > THERMAL > USER > PERFORMANCE > POWER\n")

    while True:
        sig = scheduler.dequeue()
        if not sig:
            break
        domain = scheduler._classify_domain(sig)
        print(f"  → {sig.kind.name} [Domain: {domain.name}] "
              f"(strength={sig.strength:.1f})")

    return scheduler


def demo_guardian_oversight():
    """Demo the Guardian/Hex central bank oversight."""
    print_header("GUARDIAN OVERSIGHT (Central Bank)")

    # Effect-based capability control
    print("\n[Effect Checker] Validating agent capabilities...")
    checker = create_guardian_checker()

    agents = ["guardian", "openvino_bridge", "thread_boost"]
    for agent in agents:
        caps = []
        for effect in [Effect.CPU_CONTROL, Effect.GPU_CONTROL, Effect.READ_TELEMETRY]:
            if checker.can_perform(agent, effect):
                caps.append(effect.name)
        print(f"  {agent}: {', '.join(caps) or 'none'}")

    # Composition validation
    print("\n[Composition] Validating safe composition...")
    result = checker.validate_composition(agents)
    print(f"  Valid: {result.valid}")
    if result.warnings:
        for w in result.warnings:
            print(f"  ⚠ {w}")

    # Contract validation
    print("\n[Contract Validator] Checking telemetry invariants...")
    validator = create_guardian_validator()

    # Good telemetry
    good_context = {"cpu_util": 0.75, "gpu_util": 0.70, "temp_cpu": 72, "temp_gpu": 68, "frametime_ms": 14.2}
    result = validator.check_invariants("telemetry_snapshot", good_context)
    print(f"  Normal operation: {'✓ PASS' if result.passed else '✗ FAIL'}")

    # Thermal warning
    hot_context = {"cpu_util": 0.90, "gpu_util": 0.85, "temp_cpu": 96, "temp_gpu": 78, "frametime_ms": 18.5}
    result = validator.check_invariants("safety_check", hot_context)
    print(f"  High thermal: {'✓ PASS' if result.passed else '✗ FAIL (triggers cooldown)'}")
    for v in result.violations:
        print(f"    ⚠ {v.description}")

    return checker, validator


def demo_telemetry_loop():
    """Demo the Telemetry → Decision → Learning loop."""
    print_header("TELEMETRY → DECISION → LEARNING LOOP")

    # Runtime for variable fetching
    runtime = Runtime()

    # Simulate telemetry update
    print("\n[Telemetry] Incoming snapshot...")
    runtime.update_telemetry_dict({
        "cpu_util": 0.78,
        "gpu_util": 0.72,
        "frametime_ms": 15.2,
        "temp_cpu": 74,
        "temp_gpu": 70,
    })

    # Fetch and transform features
    print("\n[Feature Engine] Computing derived features...")
    runtime.register_computed_var("fps", "1000 / frametime_ms")
    runtime.register_computed_var("thermal_risk", "sigmoid((temp_cpu - 80) / 10)")
    runtime.register_computed_var("util_combined", "(cpu_util + gpu_util) / 2")

    features = ["cpu_util", "gpu_util", "fps", "thermal_risk", "util_combined"]
    for name in features:
        val = runtime.fetch_var(name)
        print(f"  {name}: {val:.3f}")

    # Decision based on signals
    print("\n[Decision Engine] Evaluating signals...")
    scheduler = SignalScheduler()

    # Generate signals from telemetry
    if runtime.fetch_var("thermal_risk") > 0.5:
        scheduler.enqueue(telemetry_signal(SignalKind.THERMAL_WARNING, 0.7))
    if runtime.fetch_var("fps") < 60:
        scheduler.enqueue(telemetry_signal(SignalKind.FRAMETIME_SPIKE, 0.6))
    if runtime.fetch_var("cpu_util") > 0.85:
        scheduler.enqueue(telemetry_signal(SignalKind.CPU_BOTTLENECK, 0.5))

    print(f"  Generated {len(scheduler._queue)} signals")

    # Process top signal
    top = scheduler.dequeue()
    if top:
        print(f"  Top priority: {top.kind.name}")
        print(f"  → Decision: Apply optimization for {top.kind.name}")

    # Learning feedback
    print("\n[Experience Store] Recording outcome...")
    reward = 0.15 if top else 0.0
    print(f"  Action: {top.kind.name if top else 'none'}")
    print(f"  Reward: {reward:.2f}")
    print(f"  → Stored for future learning")


def demo_alpha_beta_theta():
    """Demo α/β/θ scaling for feature engineering."""
    print_header("ALPHA-BETA-THETA SCALING")

    from feature_engine import FeatureEngine, ScaleParams

    engine = FeatureEngine()

    # Raw values
    raw_values = [0.3, 0.5, 0.7, 0.9]
    print("\n[Raw Values]", raw_values)

    # Different presets
    presets = {
        "Balanced": ScaleParams(alpha=1.0, beta=0.0, theta=0.0),
        "Gaming Boost": ScaleParams(alpha=1.5, beta=0.2, theta=0.1),
        "Power Save": ScaleParams(alpha=0.7, beta=-0.1, theta=0.0),
        "Thermal Aware": ScaleParams(alpha=1.0, beta=0.0, theta=0.5),
    }

    for name, params in presets.items():
        scaled = [engine.scale_abt(v, params) for v in raw_values]
        print(f"\n[{name}] α={params.alpha}, β={params.beta}, θ={params.theta}")
        print(f"  Scaled: [{', '.join(f'{v:.3f}' for v in scaled)}]")


def main():
    print("\n" + "="*60)
    print(" GAMESA/KrystalStack - Cross-Forex Demo")
    print(" Real-time Hardware Optimization as High-Frequency Trading")
    print("="*60)

    # Run all demos
    allocator, allocs = demo_resource_market()
    demo_signal_scheduling()
    demo_guardian_oversight()
    demo_telemetry_loop()
    demo_alpha_beta_theta()

    # Cleanup
    print_header("CLEANUP")
    print("\n[Releasing Allocations]")
    for alloc in allocs:
        allocator.free(alloc.resource_type, alloc.id)
        print(f"  ✓ Released {alloc.id}")

    print("\n" + "="*60)
    print(" Demo Complete!")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
