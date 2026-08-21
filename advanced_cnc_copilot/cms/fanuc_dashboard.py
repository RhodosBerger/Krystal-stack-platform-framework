#!/usr/bin/env python3
"""
FANUC RISE COMMAND CENTER
The "Bridge" between Human Intent and Machine Experience.
Integrates Elements DB, Optimizer, and Dopamine Engine into a UI.
"""

import os
import sys
import time
import random
import asyncio

# Ensure imports work
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cms.elements_db import MATERIALS_DB, STRATEGY_DB, TOOLS_DB
from cms.dopamine_engine import DopamineEngine
from cms.cnc_vino_optimizer import CNCOptimizer
from cms.quadratic_scanner import QuadraticScanner, Mantinel

class FanucDashboard:
    def __init__(self):
        self.selected_material = "Steel4140"
        self.selected_strategy = "ACTION_STANDARD_MODE"
        self.selected_tool = "EM_12mm_Carbide_3Flute"
        self.engine = DopamineEngine()
        self.scanner = QuadraticScanner("rpm * feed < 8000000") # Default
        self.optimizer = CNCOptimizer()

    def clear_screen(self):
        # Disabled for test stability if running in non-interactive mode usually, 
        # but kept for real usage. 
        if os.name == 'nt': os.system('cls')
        else: os.system('clear')

    def print_header(self):
        # self.clear_screen() # Commented out to prevent flickering in some terminals or tests
        print("="*60)
        print("   FANUC RISE // ADVANCED COPILOT // COMMAND CENTER")
        print("="*60)
        print(f" [M] Material: {self.selected_material} | [S] Strategy: {self.selected_strategy}")
        print(f" [T] Tool:     {self.selected_tool}")
        print("-" * 60)

    def menu(self):
        while True:
            self.print_header()
            print("\n [1] CONFIG: Select Material")
            print(" [2] CONFIG: Select Strategy")
            print(" [3] RUN:    Simulate Optimzed Machining")
            print(" [X] EXIT")
            
            try:
                choice = input("\n > COMMAND: ").strip().upper()
            except EOFError:
                break
            
            if choice == '1': self.select_material()
            elif choice == '2': self.select_strategy()
            elif choice == '3': asyncio.run(self.run_simulation())
            elif choice == 'X': break

    def select_material(self):
        print("\n AVAILABLE MATERIALS:")
        keys = list(MATERIALS_DB.keys())
        for idx, mat in enumerate(keys):
            print(f" [{idx+1}] {mat}")
        try:
            sel = int(input(" > Select # : ")) - 1
            if 0 <= sel < len(keys):
                self.selected_material = keys[sel]
        except: pass

    def select_strategy(self):
        print("\n AVAILABLE STRATEGIES:")
        keys = list(STRATEGY_DB.keys())
        for idx, strat in enumerate(keys):
            print(f" [{idx+1}] {strat}")
        try:
            sel = int(input(" > Select # : ")) - 1
            if 0 <= sel < len(keys):
                self.selected_strategy = keys[sel]
        except: pass

    async def run_simulation(self):
        self.print_header()
        print("\n >>> INITIALIZING OPTIMIZER...")
        
        # 1. Load Profiles
        mat_profile = MATERIALS_DB[self.selected_material]
        strat_profile = STRATEGY_DB[self.selected_strategy]
        
        # 2. Configure Optimizer
        # Mocking a G-Code Sample
        raw_gcode = [
            "G01 X10 Y10 F100 S1000",
            "G01 X50 Y50 F2000 S5000", # Aggressive
            "G01 X100 Y100 F5000 S12000" # Dangerous
        ]
        
        # 3. Simulate "In-Process"
        print(" >>> STARTING CYCLE INTERPOLATION (ASCII VINO SCAN)")
        
        limit = mat_profile["mantinel_limit"]
        self.scanner.formula = f"rpm * feed < {limit}"
        
        # Interactive Simulation Loop
        for step in range(5): # Reduced steps for brevity
            # Simulate changing conditions
            rpm = random.randint(1000, 12000)
            feed = random.randint(100, 4000)
            
            # Visualize
            print(f"\n --- STEP {step+1}/5 : {self.selected_material} @ {self.selected_strategy} ---")
            
            self.scanner.visualize(rpm, feed)
            
            # Show Dopamine
            # Simulate "Wave"
            vib = random.uniform(0, 1.0)
            stress = vib / mat_profile["cortisol_threshold"] # Normalized stress
            
            # Neuro-Metrics
            dopamine = (rpm * feed) / limit * 100 * strat_profile["dopamine_weight"]
            cortisol = stress * 100 * strat_profile["cortisol_penalty"]
            
            bar_d = "#" * int(dopamine / 5)
            bar_c = "!" * int(cortisol / 5)
            
            print(f"\n [NEURO-METRICS]")
            print(f" DOPAMINE (Reward): {dopamine:.1f} |{bar_d}")
            print(f" CORTISOL (Stress): {cortisol:.1f} |{bar_c}")
            
            if cortisol > 90:
                print("\n *** EMERGENCY STOP TARGETED ***")
                break
                
            # time.sleep(0.5) # Removed for test speed, rely on print
            
        print("\n [CYCLE COMPLETE]")
        # input("\n Press Enter to return...") # Removed to avoid hanging tests


if __name__ == "__main__":
    app = FanucDashboard()
    app.menu()
