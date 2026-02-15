import time
import sys
import os
import random
import json
import statistics
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add parent directory to path to import modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from gamesa_cortex_v2.src.core.npu_coordinator import NPUCoordinator

def heavy_compute(n):
    """Simulates a heavy AI inference or planning task."""
    start = time.perf_counter_ns()
    # Matrix multiplication simulation (CPU bound)
    result = 0
    for i in range(n):
        result += (i * i) % 100
    end = time.perf_counter_ns()
    return (end - start) / 1_000_000.0

def safe_io_task(n):
    """Simulates a safety check or telemetry I/O."""
    time.sleep(n / 1000.0) # sleep n ms
    return n

def baseline_execution(tasks):
    """Run tasks using standard Python ThreadPoolExecutor without governance."""
    print(f"Running Baseline ({len(tasks)} tasks)...")
    results = []
    start_total = time.perf_counter()
    
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = []
        for task_type, workload in tasks:
            if task_type == "AI_INFERENCE":
                futures.append(executor.submit(heavy_compute, workload))
            elif task_type == "SAFETY_CHECK":
                futures.append(executor.submit(safe_io_task, workload))
                
        for f in as_completed(futures):
            results.append(f.result())
            
    end_total = time.perf_counter()
    duration = end_total - start_total
    return duration, results

def managed_execution(tasks):
    """Run tasks using Gamesa Cortex V2 NPUCoordinator with Economic Governance."""
    print(f"Running Cortex V2 Managed ({len(tasks)} tasks)...", flush=True)
    coordinator = NPUCoordinator()
    
    # Pre-replenish budget for benchmark
    coordinator.economics.budget_credits = 50000 
    
    results = []
    start_total = time.perf_counter()
    
    futures = []
    for task_type, workload in tasks:
        # Mock deadline logic
        deadline = 100.0 # ms relative budget
        
        future = None
        if task_type == "AI_INFERENCE":
            future = coordinator.dispatch_task(heavy_compute, task_type, deadline, workload)
        elif task_type == "SAFETY_CHECK":
            # Safety checks are high priority/low cost usually
            future = coordinator.dispatch_task(safe_io_task, "INTERDICTION_PROTOCOL", deadline, workload)
            
        if future:
            futures.append(future)
        else:
            # Task denied by governor
            results.append(None) 
            
    for f in as_completed(futures):
        results.append(f.result())
        
    end_total = time.perf_counter()
    duration = end_total - start_total
    
    # Shutdown executor to avoid hanging
    coordinator.executor.shutdown(wait=False)
    
    return duration, results, coordinator.economics.budget_credits

def benchmark_suite():
    print("=== Gamesa Cortex V2 Benchmark Suite ===", flush=True)
    
    # Define Workload - REDUCED
    # Mix of Heavy Compute (Inference) and Light IO (Safety)
    workload = []
    for _ in range(5):
        workload.append(("AI_INFERENCE", 10000)) # Approx 1-2ms compute
    for _ in range(5):
        workload.append(("SAFETY_CHECK", 5)) # 5ms sleep
        
    random.shuffle(workload)
    
    # Run Baseline
    base_time, base_results = baseline_execution(workload)
    base_tps = len(workload) / base_time
    
    print(f"Baseline: {base_time:.4f}s | TPS: {base_tps:.2f}", flush=True)
    
    # Run Managed
    managed_time, managed_results, remaining_credits = managed_execution(workload)
    # Filter out denied tasks (None)
    completed_tasks = [r for r in managed_results if r is not None]
    denied_count = len(workload) - len(completed_tasks)
    managed_tps = len(completed_tasks) / managed_time if managed_time > 0 else 0
    
    print(f"Managed:  {managed_time:.4f}s | TPS: {managed_tps:.2f} | Denied: {denied_count} | Credits Left: {remaining_credits}", flush=True)
    
    # Report
    print("\n=== Benchmark Report ===", flush=True)
    print(f"Workload: {len(workload)} tasks (50% Inference, 50% Safety)")
    print(f"Baseline Throughput: {base_tps:.2f} tasks/sec")
    print(f"Managed Throughput:  {managed_tps:.2f} tasks/sec")
    
    improvement = ((managed_tps - base_tps) / base_tps) * 100
    print(f"Throughput Delta: {improvement:+.2f}%")
    
    if denied_count > 0:
        print(f"Note: {denied_count} tasks were denied by Economic Governor to preserve system stability.")
        
    # Save results
    report = {
        "baseline_time": base_time,
        "baseline_tps": base_tps,
        "managed_time": managed_time,
        "managed_tps": managed_tps,
        "improvement_percent": improvement,
        "tasks_total": len(workload),
        "tasks_completed": len(completed_tasks),
        "tasks_denied": denied_count,
        "remaining_credits": remaining_credits
    }
    
    with open("gamesa_cortex_v2/scripts/benchmark_results.json", "w") as f:
        json.dump(report, f, indent=2)
        
    print("Results saved to gamesa_cortex_v2/scripts/benchmark_results.json")

if __name__ == "__main__":
    benchmark_suite()
