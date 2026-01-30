#!/usr/bin/env python3
"""
AGGREGATION DEMO
Verifies the Synthesis capabilities of the Framework.
"""

import sys
import os
import random

# Ensure imports work
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cms.hippocampus import Hippocampus, HISTORY_FILE
from cms.hippocampus_aggregator import AggregationFramework

def run_aggregation_demo():
    print("\n" + "="*60)
    print("   DATA SYNTHESIS: AGGREGATION DEMO")
    print("="*60 + "\n")

    # 1. Setup & Seed Memory with Random Data
    memory_file = "agg_demo_memory.json"
    if os.path.exists(memory_file): os.remove(memory_file)
    
    mem = Hippocampus(memory_file)
    
    materials = ["Steel4140", "Aluminum6061", "Titanium6Al4V"]
    strategies = ["RUSH", "STANDARD", "CAUTIOUS"]
    
    print(">>> [SEEDING] Generating 50 Random Episodes...")
    for _ in range(50):
        mat = random.choice(materials)
        strat = random.choice(strategies)
        
        # Simulate Physics
        cortisol = 0
        if mat == "Titanium6Al4V": cortisol = random.uniform(40, 90)
        elif mat == "Aluminum6061": cortisol = random.uniform(0, 20)
        else: cortisol = random.uniform(20, 60)
        
        mem.remember(mat, 5000, 2000, strat, cortisol, 100-cortisol)

    # 2. Run Aggregation
    agg = AggregationFramework(mem)
    
    print("\n>>> [REPORT 1] Analysis by Material")
    agg.generate_report("material")

    print("\n>>> [REPORT 2] Analysis by Strategy")
    agg.generate_report("strategy")

    print("\n>>> [REPORT 3] Combined (Granular)")
    agg.generate_report("combined")
    
    print("\n" + "="*60)
    print("   DEMO COMPLETE")
    print("="*60)

if __name__ == "__main__":
    run_aggregation_demo()
