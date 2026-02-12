
import sys
import os
import time
import random
import cProfile
import pstats
from dataclasses import dataclass
from typing import Dict, List
import asyncio

# Setup Path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

from advanced_cnc_copilot.backend.cms.services.dopamine_engine import DopamineEngine, NeuroState


# Mock Repository
class MockRepository:
    def __init__(self):
        # Pre-generate 100 mock records once
        self.pregenerated_data = []
        for i in range(100):
            self.pregenerated_data.append(MockRecord(
                spindle_load=50.0 + random.random() * 20,
                vibration_x=0.5 + random.random() * 0.5,
                temperature=35.0 + random.random() * 10,
                tool_wear=0.1 + random.random() * 0.1,
                feed_rate=1000.0 + random.random() * 100
            ))

    def get_recent_by_machine(self, machine_id, minutes):
        # Return the pre-generated list (simulation of DB fetch)
        return self.pregenerated_data


@dataclass
class MockRecord:
    spindle_load: float = 50.0
    vibration_x: float = 0.5
    temperature: float = 35.0
    tool_wear: float = 0.1
    feed_rate: float = 1000.0


def run_cpu_benchmark(iterations=5000):
    print(f"🚀 Starting CPU Benchmark (Dopamine Engine) - {iterations} iterations...", flush=True)
    repo = MockRepository()
    engine = DopamineEngine(repo)
    
    current_metrics = {
        'spindle_load': 75.0,
        'vibration_x': 0.8,
        'temperature': 40.0,
        'feed_rate': 1200.0,
        'tool_wear': 0.3
    }

    start_time = time.perf_counter()
    
    for _ in range(iterations):
        _ = engine.calculate_current_state(machine_id=1, current_metrics=current_metrics)
        
    end_time = time.perf_counter()
    elapsed = end_time - start_time
    ops = iterations / elapsed if elapsed > 0 else 0
    
    print(f"✅ Computation Finished!", flush=True)
    print(f"Time: {elapsed:.4f}s", flush=True)
    print(f"Score: {ops:.2f} OPS (Operations Per Second)", flush=True)
    
    return ops

def profile_cpu_benchmark(iterations=1000):
    print("\n🔬 Profiling...", flush=True)
    profiler = cProfile.Profile()
    profiler.enable()
    run_cpu_benchmark(iterations)
    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats('cumtime')
    stats.print_stats(10)

if __name__ == "__main__":
    iterations = 5000
    if len(sys.argv) > 1:
        iterations = int(sys.argv[1])
    
    run_cpu_benchmark(iterations)
    # profile_cpu_benchmark(iterations // 5)

