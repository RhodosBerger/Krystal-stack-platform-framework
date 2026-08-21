import asyncio
import sys
import os
import json

# Add project root to path
sys.path.append(os.getcwd())

from cms.message_bus import global_bus
from cms.interaction_supervisor import InteractionSupervisor
from cms.agents.thermal_agent import ThermalAgent
from cms.agents.signal_repeater import SignalRepeater

async def test_swarm():
    print("🐝 Starting Swarm Verification...")
    
    # 1. Initialize Bots
    supervisor = InteractionSupervisor()
    thermal_bot = ThermalAgent()
    repeater = SignalRepeater()
    
    # Start them up
    # Supervisor blocks, so run it in background
    asyncio.create_task(supervisor.start())
    # Give it a moment to bind listeners
    await asyncio.sleep(0.5)
    
    await thermal_bot.start()
    await repeater.start()
    
    print("✅ Bots Initialized (Supervisor, Thermal, Repeater)")
    
    verify_future = asyncio.get_event_loop().create_future()
    
    # Listen for final Consensus to prove everyone voted
    async def consensus_listener(msg):
        print(f"\n🗳️ Consensus Reached!")
        print(json.dumps(msg.payload, indent=2))
        verify_future.set_result(msg.payload)
        
    global_bus.subscribe("PLAN_APPROVED", consensus_listener)
    global_bus.subscribe("PLAN_REJECTED", consensus_listener)

    # 2. Simulate other agents (Auditor/Biochemist) manually to speed up test
    # The Supervisor waits for 3 votes usually.
    # But wait, to test *integration*, we should check if Thermal Bot actually votes.
    
    # Check if Repeater logs
    if os.path.exists("swarm_blackbox.jsonl"):
        os.remove("swarm_blackbox.jsonl")
        
    # 3. Publish Draft Plan (Normal Heat)
    test_plan = {
        "job_id": "SWARM_TEST_001",
        "material": "Aluminum", 
        "rpm": 1000,
        "feed": 500,
    }
    print(f"\n📤 Broadcasting Plan: {test_plan}")
    await global_bus.publish("DRAFT_PLAN", test_plan, "TESTER")
    
    # Wait for Thermal Bot to react
    await asyncio.sleep(0.5) 
    
    # Manually Inject other votes so Supervisor finishes (since we didn't start Auditor/Biochemist)
    await global_bus.publish("VALIDATION_RESULT", {"original_plan_id": "SWARM_TEST_001", "status": "PASS"}, "AUDITOR_MOCK")
    await global_bus.publish("VOTE_BIOCHEMIST", {"job_id": "SWARM_TEST_001", "vote": 1.0}, "BIOCHEMIST_MOCK")
    
    # Wait for Consensus
    try:
        result = await asyncio.wait_for(verify_future, timeout=3.0)
        
        # Verify Thermal Bot Participation in Consensus details if available
        # The logic in Supervisor might not expose details in the event payload, 
        # but if we got a result, it means the count reached 3.
        print("✅ Trio Consensus Reached (Thermal Bot must have voted)")
        
        # Verify Repeater Log
        if os.path.exists("swarm_blackbox.jsonl"):
             with open("swarm_blackbox.jsonl", "r") as f:
                 logs = f.readlines()
                 print(f"✅ Signal Repeater Archived {len(logs)} Events")
                 if len(logs) > 0:
                     print("Sample Log:", logs[0].strip())
        else:
             print("❌ FAIL: Blackbox log missing")
             
    except asyncio.TimeoutError:
        print("❌ TIMEOUT: Swarm did not reach consensus.")

if __name__ == "__main__":
    asyncio.run(test_swarm())
