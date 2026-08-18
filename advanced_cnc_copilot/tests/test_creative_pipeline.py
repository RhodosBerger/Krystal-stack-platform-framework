import asyncio
import sys
import os

# Add project root to path
sys.path.append(os.getcwd())

from backend.core.synaptic_bridge import synaptic_bridge
from cms.protocol_conductor import ProtocolConductor
from cms.message_bus import global_bus

async def test_creative_pipeline():
    print("🧪 Starting Creative Pipeline Verification...")
    
    # 1. Test Bridge
    print("\n--- Testing Synaptic Bridge ---")
    intents = {
        "I want a very aggressive cut": "AGGRESSIVE",
        "Make it smooth and serene": "SERENE",
        "Just adapt to the material flow": "FLUID"
    }
    
    for text, expected in intents.items():
        result = synaptic_bridge.interpret_intent(text)
        print(f"'{text}' -> {result}")
        if result != expected:
            print(f"❌ FAIL: Expected {expected}, got {result}")
            return
    print("✅ Synaptic Bridge: PASSED")
    
    # 2. Test Conductor with Bias
    print("\n--- Testing Protocol Conductor (Emotional Bias) ---")
    conductor = ProtocolConductor()
    # Mocking dependencies
    conductor._mock_llm_extraction = lambda x: {"rpm": 8000, "feed": 500}
    conductor.repo.validate_and_save = lambda n, s: {"corrected_criteria": s, "message": "Specs Accepted."}
    conductor.topology.find_knobs_for_gauge = lambda g: ["RPM_KNOB"]
    
    emotion = "AGGRESSIVE"
    result = conductor.conduct_scenario("TEST_JOB_001", "Make it fast", emotion=emotion)
    
    narrative = "\n".join(result["narrative"])
    print("Narrative Output:")
    print(narrative)
    
    if "EMOTIONAL BIAS APPLIED: AGGRESSIVE" in narrative:
        print("✅ Conductor: Bias Injected")
    else:
        print("❌ FAIL: Bias header missing from narrative")
        return

    # 3. Test Shadow Council Trigger
    print("\n--- Testing Shadow Council Bus Trigger ---")
    
    received_events = []
    
    async def plan_listener(msg):
        print(f"📡 Bus Received [{msg.channel}]: {msg.payload}")
        received_events.append(msg)
        
    global_bus.subscribe("DRAFT_PLAN", plan_listener)
    
    # Simulate what API does
    await global_bus.publish("DRAFT_PLAN", {"job_id": "TEST_JOB_001", "rpm": 8000}, "TESTER")
    
    # Wait for async processing
    await asyncio.sleep(0.1)
    
    if len(received_events) > 0:
        print("✅ Message Bus: Event Published & Received")
    else:
        print("❌ FAIL: Event not received")
        return
        
    print("\n🎉 ALL SYSTEMS GO: Creative Pipeline Verified.")

if __name__ == "__main__":
    asyncio.run(test_creative_pipeline())
