#!/usr/bin/env python3
"""
DASHBOARD TEST (Headless)
Verifies the logic of the Command Center without requiring interaction.
"""
import sys
import os
import asyncio
from unittest.mock import MagicMock, patch

# Ensure imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.cms.fanuc_dashboard import FanucDashboard

def test_dashboard_logic():
    print(">>> TESTING DASHBOARD LOGIC...")
    
    # 1. Initialize
    dash = FanucDashboard()
    print(f" [OK] Initialized. Default Material: {dash.selected_material}")
    
    # 2. Test Selection Logic
    print(" >>> Testing Material Selection Mock...")
    with patch('builtins.input', return_value='2'): # Select 2nd material
        dash.select_material()
        print(f" [OK] Material Update: {dash.selected_material}")
        
    # 3. Test Simulation Logic (Async)
    print(" >>> Testing Simulation (Short Run)...")
    # We mock 'time.sleep' to be instant, and 'input' to return immediately
    with patch('time.sleep', return_value=None), \
         patch('builtins.input', return_value=''):
        asyncio.run(dash.run_simulation())
    
    print(" [OK] Simulation completed without crash.")

if __name__ == "__main__":
    test_dashboard_logic()
