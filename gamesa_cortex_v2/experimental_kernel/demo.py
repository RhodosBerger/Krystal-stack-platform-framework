#!/usr/bin/env python3
"""
Experimental Kernel Demo for Krystal Stack

This script demonstrates the advanced features of the Experimental Kernel layer:
- UUID tracking with nanosecond precision
- Kernel scheduling with multiple algorithms
- Economic governor for resource allocation
- Feature management system

Author: Dušan Kopecký
Email: dusan.kopecky0101@gmail.com
"""

import sys
import time
import json

try:
    from experimental_kernel import (
        UUIDTracker,
        KernelScheduler,
        EconomicGovernor,
        FeatureContainer,
        ModuleRegistry,
        HotReloadManager,
    )
except ImportError:
    print("Error: experimental_kernel module not found.")
    print("Please build and install with: maturin develop --release")
    sys.exit(1)


def demo_uuid_tracker():
    """Demonstrate UUID tracking with latency measurement."""
    print("\n" + "=" * 60)
    print("🎯 UUID Tracker Demo")
    print("=" * 60)
    
    tracker = UUIDTracker()
    
    # Create multiple tasks with different priorities
    print("\n📝 Creating tasks...")
    tasks = []
    for i in range(5):
        uuid = tracker.create_task(
            task_type=f"inference_task_{i}",
            priority=100 + (i * 20),
            credits_cost=10 + i,
            tags=[f"batch_{i}", "demo"]
        )
        tasks.append(uuid)
        print(f"  Created task {i}: {uuid[:8]}... (priority={100 + (i * 20)})")
    
    # Simulate execution with timing
    print("\n⏱️  Executing tasks...")
    for i, uuid in enumerate(tasks):
        tracker.start_task(uuid)
        
        # Simulate work (1-10ms)
        work_time = 0.001 * (i + 1)
        time.sleep(work_time)
        
        latency = tracker.complete_task(uuid)
        print(f"  Task {i} completed: {latency:,} ns ({latency/1_000_000:.2f} ms)")
    
    # Get statistics
    print("\n📊 Latency Statistics:")
    stats = tracker.get_latency_stats()
    print(f"  Min:     {stats['min_ns']:,} ns ({stats['min_ns']/1_000_000:.2f} ms)")
    print(f"  Max:     {stats['max_ns']:,} ns ({stats['max_ns']/1_000_000:.2f} ms)")
    print(f"  Average: {stats['avg_ns']:,} ns ({stats['avg_ns']/1_000_000:.2f} ms)")
    print(f"  P50:     {stats['p50_ns']:,} ns ({stats['p50_ns']/1_000_000:.2f} ms)")
    print(f"  P95:     {stats['p95_ns']:,} ns ({stats['p95_ns']/1_000_000:.2f} ms)")
    print(f"  P99:     {stats['p99_ns']:,} ns ({stats['p99_ns']/1_000_000:.2f} ms)")
    print(f"  Count:   {stats['count']}")
    
    # Export to JSON
    print("\n💾 Exporting task history...")
    json_data = tracker.export_to_json()
    print(f"  Exported {len(json_data)} bytes of task history")
    
    return tracker


def demo_kernel_scheduler():
    """Demonstrate kernel scheduling algorithms."""
    print("\n" + "=" * 60)
    print("📅 Kernel Scheduler Demo")
    print("=" * 60)
    
    scheduler = KernelScheduler()
    
    # Initialize budgets
    print("\n💰 Initializing credit budgets...")
    scheduler.init_budget("gpu", 1000, 100, 100)
    scheduler.init_budget("cpu", 500, 50, 200)
    scheduler.init_budget("npu", 300, 30, 500)
    print("  GPU: 1000 credits (replenish 100/100ms)")
    print("  CPU: 500 credits (replenish 50/200ms)")
    print("  NPU: 300 credits (replenish 30/500ms)")
    
    # Submit tasks with different deadlines and priorities
    print("\n📤 Submitting tasks...")
    task_configs = [
        ("critical_safety", 10, 255, 20, "gpu"),
        ("inference_batch", 50, 150, 15, "npu"),
        ("vision_pipeline", 30, 180, 25, "gpu"),
        ("background_log", 500, 50, 5, "cpu"),
        ("realtime_control", 5, 200, 10, "cpu"),
    ]
    
    for task_type, deadline, priority, cost, domain in task_configs:
        uuid = scheduler.submit_task(
            task_type=task_type,
            deadline_ms=deadline,
            priority=priority,
            credits_cost=cost,
            domain=domain,
            tags=["demo", task_type]
        )
        if uuid:
            print(f"  ✓ {task_type}: {uuid[:8]}... (deadline={deadline}ms, priority={priority})")
        else:
            print(f"  ✗ {task_type}: REJECTED (insufficient credits)")
    
    # Test different scheduling modes
    print("\n🔄 Testing scheduling modes...")
    
    for mode in ["normal", "realtime", "eco", "performance"]:
        scheduler.set_mode(mode)
        actual_mode = scheduler.get_mode()
        print(f"\n  Mode: {actual_mode}")
        
        # Schedule a few tasks
        scheduled = 0
        for _ in range(3):
            task = scheduler.schedule_next()
            if task:
                scheduled += 1
                print(f"    Scheduled: {task['task_type']} (priority={task['priority']})")
                scheduler.complete_task(task['uuid'])
            else:
                break
        
        print(f"    Tasks processed: {scheduled}")
    
    # Get final statistics
    print("\n📊 Final Statistics:")
    stats = scheduler.get_stats()
    print(f"  Tasks Scheduled:  {stats['tasks_scheduled']}")
    print(f"  Tasks Completed:  {stats['tasks_completed']}")
    print(f"  Deadline Misses:  {stats['tasks_deadline_missed']}")
    print(f"  Active Tasks:     {stats['active_tasks']}")
    print(f"  Completed Tasks:  {stats['completed_tasks']}")
    
    # Budget status
    print("\n💰 Budget Status:")
    for budget in stats.get('budgets', []):
        print(f"  {budget['domain']}: {budget['available']}/{budget['total']} credits")
    
    # Latency stats
    print("\n⏱️  Latency Statistics:")
    latency = scheduler.get_latency_stats()
    print(f"  P50: {latency['p50_ns']:,} ns")
    print(f"  P95: {latency['p95_ns']:,} ns")
    print(f"  P99: {latency['p99_ns']:,} ns")
    
    return scheduler


def demo_economic_governor():
    """Demonstrate economic governor for resource allocation."""
    print("\n" + "=" * 60)
    print("💰 Economic Governor Demo")
    print("=" * 60)
    
    governor = EconomicGovernor()
    
    # Allocate budgets
    print("\n📊 Allocating budgets...")
    governor.allocate_budget("render", 2000, 200, 50)
    governor.allocate_budget("mining", 1500, 150, 100)
    governor.allocate_budget("ai_inference", 1000, 100, 75)
    
    # Request credits
    print("\n💸 Requesting credits...")
    requests = [
        ("render", 500, "4K render job"),
        ("render", 800, "8K render job"),
        ("mining", 300, "SHA-256 batch"),
        ("ai_inference", 200, "LLM inference"),
        ("ai_inference", 900, "Large batch inference"),  # Should fail
    ]
    
    for domain, cost, description in requests:
        approved = governor.request_credits(domain, cost)
        status = "✓ APPROVED" if approved else "✗ REJECTED"
        print(f"  {status}: {description} ({cost} credits from {domain})")
    
    # Check budget status
    print("\n📊 Budget Status:")
    all_budgets = governor.get_all_budgets()
    for budget in all_budgets:
        usage = (1 - budget['available'] / budget['total']) * 100
        print(f"  {budget['domain']}:")
        print(f"    Available: {budget['available']}/{budget['total']} ({usage:.1f}% used)")
        print(f"    Replenish: {budget['replenishment_rate']} credits/interval")
    
    return governor


def demo_feature_container():
    """Demonstrate feature management system."""
    print("\n" + "=" * 60)
    print("🧩 Feature Container Demo")
    print("=" * 60)
    
    container = FeatureContainer()
    
    # Register features
    print("\n🔧 Registering features...")
    features = [
        ("thermal_monitor", "Thermal Monitor", "1.0.0", 255, []),
        ("power_manager", "Power Manager", "1.0.0", 200, ["thermal_monitor"]),
        ("performance_logger", "Performance Logger", "1.0.0", 50, []),
        ("telemetry_uploader", "Telemetry Uploader", "1.0.0", 30, ["performance_logger"]),
    ]
    
    for fid, name, version, priority, deps in features:
        container.register_feature(fid, name, version, priority, deps)
        print(f"  Registered: {name} v{version} (priority={priority})")
    
    # List features
    print("\n📋 Registered Features:")
    for feature in container.list_features():
        print(f"  - {feature['id']} (priority={feature['priority']})")
    
    # Initialize
    print("\n🚀 Initializing features...")
    if container.initialize_all():
        print("  ✓ All features initialized")
    
    # Execute
    print("\n▶️  Executing features...")
    results = container.execute_all()
    for result in results:
        status = "✓" if result.get('success', False) else "✗"
        print(f"  {status} {result['feature_id']}")
    
    # Context data
    print("\n📝 Setting context data...")
    container.set_context_data("environment", "production")
    container.set_context_data("gpu_count", "8")
    container.set_context_data("debug_mode", "false")
    
    print(f"  environment: {container.get_context_data('environment')}")
    print(f"  gpu_count: {container.get_context_data('gpu_count')}")
    print(f"  debug_mode: {container.get_context_data('debug_mode')}")
    
    # Module registry
    print("\n📦 Module Registry:")
    registry = ModuleRegistry()
    registry.register_module(
        "krystal_bitboard",
        "2.0.0",
        "/opt/krystal/modules/bitboard.so",
        ["mining", "rendering", "ai"]
    )
    registry.register_module(
        "gamesa_cortex",
        "3.0.0",
        "/opt/krystal/modules/cortex.so",
        ["planning", "vision", "control"]
    )
    
    print("  Registered modules:")
    for module in registry.list_modules():
        status = "LOADED" if module['is_loaded'] else "UNLOADED"
        print(f"    {module['name']} v{module['version']} [{status}]")
    
    # Load a module
    if registry.load_module("krystal_bitboard"):
        print("\n  ✓ Loaded krystal_bitboard module")
    
    # Hot reload
    print("\n🔄 Hot Reload Manager:")
    hot_reload = HotReloadManager()
    hot_reload.watch_module("krystal_bitboard", "/opt/krystal/modules/bitboard.so")
    
    reload_count = hot_reload.get_reload_count("krystal_bitboard")
    print(f"  Initial reload count: {reload_count}")
    
    if hot_reload.trigger_reload("krystal_bitboard"):
        reload_count = hot_reload.get_reload_count("krystal_bitboard")
        print(f"  After trigger: {reload_count} reloads")
    
    return container


def main():
    """Run all demos."""
    print("\n" + "=" * 60)
    print("🚀 EXPERIMENTAL KERNEL DEMO - Krystal Stack")
    print("=" * 60)
    print("\nThis demo showcases the advanced features of the Experimental Kernel layer.")
    print("Author: Dušan Kopecký <dusan.kopecky0101@gmail.com>")
    
    try:
        # Run all demos
        tracker = demo_uuid_tracker()
        scheduler = demo_kernel_scheduler()
        governor = demo_economic_governor()
        container = demo_feature_container()
        
        # Final summary
        print("\n" + "=" * 60)
        print("✅ DEMO COMPLETE")
        print("=" * 60)
        print("\nThe Experimental Kernel is ready for integration with Krystal Stack!")
        print("\nNext steps:")
        print("  1. Review the documentation: README.md")
        print("  2. Integrate with Gamesa Cortex V2")
        print("  3. Configure credit budgets for your workload")
        print("  4. Enable telemetry export to your BI system")
        print("\n" + "=" * 60 + "\n")
        
    except Exception as e:
        print(f"\n❌ Error during demo: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
