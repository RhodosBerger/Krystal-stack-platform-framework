import asyncio
import sys
import os
import json

# Add project root to path
sys.path.append(os.getcwd())

from cms.auditor_agent import AuditorAgent, AuditLevel
from cms.message_bus import global_bus

async def test_deep_thought():
    print("🧠 Starting Deep Thought Verification...")
    
    agent = AuditorAgent()
    await agent.start()
    
    verify_future = asyncio.get_event_loop().create_future()
    
    async def validation_listener(msg):
        print(f"\n🔍 Auditor Output Received:")
        print(json.dumps(msg.payload, indent=2))
        verify_future.set_result(msg.payload)
        
    global_bus.subscribe("VALIDATION_RESULT", validation_listener)
    
    # Simulate a "Deep Thought" worthy plan (Hard Material + High Complexity)
    test_plan = {
        "job_id": "DEEP_THOUGHT_TEST_001",
        "material": "Titanium", # Trigger for DEEP_THOUGHT
        "rpm": 3000,
        "feed": 500,
        "complexity": 9.0, # High complexity
        "source": "UNIT_TEST"
    }
    
    print(f"\n📤 Sending High-Complexity Plan: {test_plan}")
    await global_bus.publish("DRAFT_PLAN", test_plan, "TESTER")
    
    # Wait for result
    try:
        result = await asyncio.wait_for(verify_future, timeout=5.0)
        
        # Verify Math Proofs
        if "math_proof" in result:
            flux = result["math_proof"].get("thermal_flux_watts")
            print(f"\n✅ Math Proof Found: Thermal Flux = {flux} W")
            
            # Verify Calculation: (3000 * (500/100)) / 7.0 for Titanium
            # (3000 * 5) / 7 = 15000 / 7 = 2142.85
            expected = 2142.86 # Rounding diffs possible
            if abs(flux - expected) < 10.0:
                 print("✅ Calculation Accuracy: PASSED")
            else:
                 print(f"❌ Calculation Failure: Expected ~{expected}, Got {flux}")
        else:
            print("❌ FAIL: No 'math_proof' in validation result.")

        # Verify Geometry Proofs (Mock Voxel)
        if "geometry_proof" in result:
             print("✅ Geometry Proof Found (Voxel Analysis)")
        else:
             print("❌ FAIL: No 'geometry_proof' in validation result.")
             
    except asyncio.TimeoutError:
        print("❌ TIMEOUT: Auditor did not respond.")

if __name__ == "__main__":
    asyncio.run(test_deep_thought())
