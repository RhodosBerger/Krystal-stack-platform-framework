#!/usr/bin/env python3
"""
FANUC RISE - INTEGRATION VERIFICATION
Demonstrates the full CNC-VINO Loop:
1. Optimizer (G-Code -> IR)
2. Bridge (Wave Metrics)
3. Dopamine Engine (Safety Scoring)
4. Scanner (Visualization)
"""

import asyncio
import logging
import sys
import os

# Ensure imports work
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.cms.cnc_vino_optimizer import CNCOptimizer
from backend.cms.dopamine_engine import DopamineEngine
from backend.cms.quadratic_scanner import QuadraticScanner
from backend.cms.fanuc_rise_bridge import FanucRiseBridge
from backend.cms.parameter_standard import IdealMetric, Characteristic, Mantinel, MantinelType

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(name)-10s | %(message)s')

async def run_fanuc_rise_demo():
    print("\n" + "="*60)
    print("   FANUC RISE: INTELLIGENT MACHINING DEMO")
    print("="*60 + "\n")

    # 1. DEFINE IDEAL STANDARDS
    print(">>> [STEP 1] Defining Gold Standard (Ideal Run)")
    ideal_speed = IdealMetric(target_value=4500, tolerance_scale=500)
    print(f"   Ideal RPM: 4500 (+/- 500 Sigma)")
    await asyncio.sleep(1)

    # 2. RUN THE OPTIMIZER (The "Compiler")
    print("\n>>> [STEP 2] Running CNC-VINO Optimizer on Raw G-Code")
    raw_gcode = [
        "S5000 M3",
        "G1 X100 F1000",   # Safe
        "G1 X200 F3000",   # Safe (15M < 8M? No. 5000*3000=15M. Unsafe.)
    ]
    
    optimizer = CNCOptimizer()
    ir = optimizer.optimize_model(raw_gcode)
    
    print("   [OPTIMIZED IR OUTPUT]:")
    for line in ir:
        print(f"   {line}")
    await asyncio.sleep(1)

    # 3. RUN THE RUNTIME (Simulating Execution with Dopamine)
    print("\n>>> [STEP 3] Runtime Execution with Fanuc Rise Wave Metrics")
    brain = DopamineEngine()
    bridge = FanucRiseBridge()
    await bridge.connect()
    
    # Start stream in background
    bridge_task = asyncio.create_task(bridge.stream_wave_metrics())
    
    # Simulate a cut with varying conditions
    for i in range(5):
        # Simulate receiving data from Bridge (Mocked here for sync logic)
        # In real app, this is event driven via MessageBus
        
        # Scenario: Metric spikes (Bad Vibration)
        vibe = 0.1
        if i == 3: 
            vibe = 0.95 # SPIKE at step 3
            print(f"   !!! VIBRATION SPIKE DETECTED !!!")

        # Calculate Scale/Deviation
        # Let's say we are running at 5000 RPM (Ideal is 4500)
        deviation = ideal_speed.calculate_deviation(5000) # (5000-4500)/500 = 1.0 (Acceptable)
        
        action = brain.evaluate_stimuli(speed_factor=1.0, vibration_level=vibe, deviation_score=deviation, result_quality=0.9)
        print(f"   T={i}: Dev={deviation:.1f} | NeuroState={brain.state} -> {action}")
        await asyncio.sleep(0.5)

    bridge.stop()
    await bridge_task
    print("\n" + "="*60)
    print("   DEMO COMPLETE")
    print("="*60)

if __name__ == "__main__":
    asyncio.run(run_fanuc_rise_demo())
