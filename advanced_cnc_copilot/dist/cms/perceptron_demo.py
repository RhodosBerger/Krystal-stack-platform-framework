#!/usr/bin/env python3
"""
PERCEPTRON DEMO
Verification of the Unified Sensation & Impact System.
"""

import sys
import os
import random

# Ensure imports work
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cms.sensory_cortex import SensoryCortex, HALParser, SolidworksParser
from cms.hal_fanuc import FanucAdapter
from cms.impact_cortex import ImpactCortex

def run_perceptron_demo():
    print("\n" + "="*60)
    print("   UNIFIED PERCEPTRON: SYSTEM DEMO")
    print("="*60 + "\n")

    # 1. Initialize The Sensory Cortex (The Collector)
    senses = SensoryCortex()
    
    # NEW: Register HAL Adapter
    fanuc_hal = FanucAdapter() 
    senses.register_parser(HALParser(fanuc_hal))
    
    senses.register_parser(SolidworksParser({}))
    print(" [INIT] Sensory Cortex Online. Parsers Registered.")

    # 2. Initialize The Impact Cortex (The Brain)
    brain = ImpactCortex()
    print(" [INIT] Impact Cortex Online. Logic Loaded.")

    # 3. Simulate Scenarios
    scenarios = [
        ("IDLE_STATE", {"hal_override": {"load": 0.05, "rpm": 0, "vibration": 0.0}, "solidworks": {"curvature": 0.0}}),
        ("OPTIMAL_CUT", {"hal_override": {"load": 0.4, "rpm": 12000, "vibration": 0.1}, "solidworks": {"curvature": 0.1}}),
        ("CHATTER_EVENT", {"hal_override": {"load": 0.6, "rpm": 8000, "vibration": 0.8}, "solidworks": {"curvature": 0.2}}),
        ("TOOL_CRASH", {"hal_override": {"load": 1.2, "rpm": 0, "vibration": 1.0}, "solidworks": {"curvature": 0.0}})
    ]

    for name, inputs in scenarios:
        print(f"\n >>> SCENARIO: {name}")
        
        # Step A: Sensation
        raw_stream = senses.collect_all(inputs)
        print(f"   [SENSORY] Packet Received. {len(raw_stream)} Datums.")
        
        # Step B: Perception (Impact Analysis)
        impact = brain.process(raw_stream)
        print(f"   {impact}")
        
        # Step C: Decision
        if impact.safety < 30:
            print("   [!!!] CRITICAL: EMERGENCY STOP TRIGGERED")
        elif impact.quality < 60:
            print("   [WARN] QUALITY ALERT: Adjust Feed Rate")
        else:
            print("   [OK] System Stable.")

    print("\n" + "="*60)
    print("   DEMO COMPLETE")
    print("="*60)

if __name__ == "__main__":
    run_perceptron_demo()
