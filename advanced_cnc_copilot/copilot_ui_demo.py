#!/usr/bin/env python3
"""
Advanced CNC Copilot - Interactive Demo
Shows the "Manifest of Adaptability" in action.
"""

import sys
import time
import logging

# Import our new system
from manufacturing_economics import CostFactors, ProjectParameters
from cnc_optimization_engine import OptimizationCopilot

def slow_print(text, delay=0.02):
    """Cinematic printing effect."""
    for char in text:
        sys.stdout.write(char)
        sys.stdout.flush()
        time.sleep(delay)
    print()

def main():
    print("\n" + "="*50)
    print("   ADVANCED CNC COPILOT // MANIFEST SYSTEM")
    print("="*50 + "\n")
    
    # 1. Initialize System
    print("[SYSTEM] Initializing Economic & Optimization Bridges...")
    costs = CostFactors(machine_hourly_rate=120.0, labor_hourly_rate=65.0, kilowatt_price=0.22)
    copilot = OptimizationCopilot(costs)
    time.sleep(1)
    print("[SYSTEM] Ready.\n")
    
    # 2. Input Bridge (Simulated UI)
    slow_print(">>> NEW PROJECT ENTRY DETECTED <<<")
    
    # In a real app, these would be form inputs
    p_name = input("Enter Part Name [Default: TurbineBlade]: ") or "TurbineBlade"
    try:
        p_batch = int(input("Enter Batch Size [Default: 50]: ") or "50")
        p_quota = int(input("Enter Daily Quota (0 for none) [Default: 0]: ") or "0")
    except ValueError:
        print("Invalid number, using defaults.")
        p_batch, p_quota = 50, 0
        
    project = ProjectParameters(
        project_id="DEMO-001",
        part_name=p_name,
        material_cost_per_unit=45.0,
        estimated_cycle_time_minutes=25.0,
        batch_size=p_batch,
        daily_quota=p_quota
    )
    
    print("\n" + "-"*40)
    slow_print(f"Injecting Data for '{p_name}' into Neural Bridge...", delay=0.01)
    time.sleep(1)
    
    # 3. Processing (The Mind)
    slow_print("Running Economic Simulation Models...", delay=0.05)
    plan = copilot.create_optimized_plan(project)
    
    # 4. Output (The Result)
    print("\n" + "="*50)
    print("   OPTIMIZATION REPORT")
    print("="*50)
    
    strategy = plan['strategy']
    print(f"RECOMMENDED STRATEGY:  [{strategy.mode.name.upper()}]")
    print(f"LOGIC REASONING:       {plan['reason']}")
    print("-" * 30)
    print(f"EST. COST PER UNIT:    ${strategy.cost_per_unit:.2f}")
    print(f"TOTAL BATCH COST:      ${strategy.total_cost:.2f}")
    print(f"TOTAL TIME ESTIMATE:   {strategy.total_time_hours:.1f} Hours")
    print("-" * 30)
    
    slow_print("\nGenerating Manifest G-Code...", delay=0.05)
    print("\n[PREVIEW OF GENERATED CODE]")
    print(f">>> {strategy.mode.name.upper()} PROFILE LOADED")
    print("-" * 20)
    print("\n".join(plan['gcode'].split("\n")[:12]))
    print("... (truncated)")
    
    print("\n[SYSTEM] Project Successfully Bridged.")

if __name__ == "__main__":
    main()
