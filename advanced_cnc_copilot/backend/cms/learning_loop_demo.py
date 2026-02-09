#!/usr/bin/env python3
"""
LEARNING LOOP DEMO
Verifies that "The Machine Learns While It Sleeps".

1. Seed Hippocampus with "Trauma" (High Cortisol episodes).
2. Run Nightly Training.
3. Verify Policy Update (Machine "fears" the material).
"""

import sys
import os
import json
import asyncio

# Ensure imports work
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.cms.hippocampus import Hippocampus
from backend.cms.nightly_training import NightlyTrainer, POLICY_FILE
from backend.cms.cnc_vino_optimizer import CNCOptimizer

def run_learning_demo():
    print("\n" + "="*60)
    print("   THE LEARNING LOOP: MEMORY & ADAPTATION DEMO")
    print("="*60 + "\n")

    # 1. Setup Memory
    memory_file = "demo_memory.json"
    if os.path.exists(memory_file): os.remove(memory_file)
    if os.path.exists(POLICY_FILE): os.remove(POLICY_FILE)

    brain = Hippocampus(memory_file)
    
    # 2. Simulate a "Bad Day" with Inconel (Hard Material)
    print(">>> [DAY 1] Simulation: Cutting Inconel 718...")
    print("    ! EVENT: Vibration Spikes detected. Cortisol rising.")
    # Record 5 bad episodes
    for i in range(5):
        brain.remember(
            material="Inconel718",
            rpm=3000,
            feed=500,
            action="ACTION_STANDARD_MODE",
            cortisol=85.0, # High Stress
            dopamine=10.0,
            notes=f"Chatter event {i}"
        )
    print("    [MEMORY] Recorded 5 Traumatic Episodes for Inconel.")

    # 3. Simulate a "Good Day" with Aluminum
    print("\n>>> [DAY 1] Simulation: Cutting Aluminum 6061...")
    for i in range(5):
        brain.remember(
            material="Aluminum6061",
            rpm=12000,
            feed=4000,
            action="ACTION_RUSH_MODE",
            cortisol=5.0, # Low Stress
            dopamine=90.0, # High Reward
            notes="Smooth cutting"
        )
    print("    [MEMORY] Recorded 5 Successful Episodes for Aluminum.")
    
    # 4. Trigger Sleep Cycle (Nightly Training)
    print("\n>>> [NIGHT] System Entering Sleep Mode...")
    trainer = NightlyTrainer(memory_file)
    trainer.dream_and_learn()
    
    # 5. Verify the Lesson Learned
    print("\n>>> [DAY 2] Checking Updated Policy...")
    if os.path.exists(POLICY_FILE):
        with open(POLICY_FILE, 'r') as f:
            policy = json.load(f)
            print(json.dumps(policy, indent=2))
            
            # Validation
            if policy.get("Inconel718", {}).get("caution_flag"):
                print("\n[SUCCESS] System has learned to fear Inconel (Caution Flag SET).")
            else:
                print("\n[FAIL] System did not learn caution.")
    else:
        print("[FAIL] No Policy File generated.")

    # 6. Verify Realization (Optimizer Adaptation)
    print("\n>>> [DAY 2] Running Optimizer on Aluminum Part...")
    optimizer = CNCOptimizer()
    # Mock G-Code
    gcode = ["S10000 M3", "G1 X100 F500"]
    ir = optimizer.optimize_model(gcode, material="Aluminum6061")
    
    print("   [OPTIMIZER OUTPUT]:")
    for line in ir:
        print(f"   {line}")
        
    print("\n" + "="*60)
    print("   DEMO COMPLETE")
    print("="*60)

if __name__ == "__main__":
    run_learning_demo()
