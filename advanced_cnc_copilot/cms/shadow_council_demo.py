#!/usr/bin/env python3
"""
THE SHADOW COUNCIL - INTEGRATION DEMO
Verifies that all 4 "Minds" work in parallel.
"""

import asyncio
import logging
import sys
import os

# Add parent dir to path to find all modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cms.message_bus import global_bus
from cms.interaction_supervisor import InteractionSupervisor
from cms.solidworks_tf_bridge import SolidworksTFBridge
from cms.cms_core import AuditorWrapper
from manufacturing_economics import AccountantWrapper

# Configure logging to be very clean for the demo
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(name)-15s | %(message)s', datefmt='%H:%M:%S')

async def run_scenario():
    print("\n" + "="*60)
    print("   THE SHADOW COUNCIL: PARALLEL INTERACTION DEMO")
    print("="*60 + "\n")

    # 1. Initialize The Council Members
    supervisor = InteractionSupervisor()
    visualizer = SolidworksTFBridge() # The Eye
    auditor = AuditorWrapper()        # The Auditor
    accountant = AccountantWrapper()  # The Accountant

    # Start the Supervisor (Listener)
    # We run it in background
    supervisor_task = asyncio.create_task(supervisor.start())
    
    # 2. Simulate "The Eye" seeing a part
    print(">>> [SCENARIO_START] Visualizer scans a Solidworks model...")
    await asyncio.sleep(1)
    await visualizer.analyze_part("C:/CAD/Titanium_Flange_X.SLDPRT")
    
    # 3. Simulate User Input (Triggers the Chain)
    print("\n>>> [USER_INPUT] Engineer asks: 'Generate a plan for this Titanium part at 5000 RPM.'")
    await asyncio.sleep(1)
    
    # Simulate the input message
    await global_bus.publish("USER_INTENT", {"text": "Plan for Titanium"}, sender_id="USER")
    
    # 4. Wait for the Council to debate (Auditor should BLOCK it because Titanium limit is 3000)
    await asyncio.sleep(3)
    
    print("\n" + "="*60)
    print("   SCENARIO COMPLETE")
    print("="*60)
    
    # Cleanup
    supervisor.active = False
    await supervisor_task

if __name__ == "__main__":
    try:
        asyncio.run(run_scenario())
    except KeyboardInterrupt:
        pass
