"""
Verification Script for Phase 16: Evolutionary Runtime 🧬
Tests the Point-to-Point flow: Intent -> Geometry -> Simulation -> Feedback
"""
import asyncio
import sys
import os
import io

# Setup Path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__))))

# Mock Dependencies to avoid full backend load
class MockVoxelizer:
    def process_geometry_intent(self, geometry):
        return {"status": "VOXELIZED"}

class MockPhysicist:
    def process_simulation_query(self, query):
        print(f"   [Physicist] Simulating {query['material']} bracket...")
        # Fail safety check if material is ALUMINUM (for test case)
        safety_factor = 0.5 if query["material"] == "WEAK_ALUMINUM" else 2.5
        return {
            "status": "COMPLETED",
            "data": {"safety_factor": safety_factor}
        }

class MockBuilder:
    def execute_logic(self, context):
        return {"status": "BUILT"}

async def test_evolution_flow():
    from backend.core.evolution_runtime import EvolutionRuntime, LifecycleState
    
    print("🚀 Starting Evolutionary Runtime Verification...")
    
    # 1. Initialize Runtime with Mocks
    runtime = EvolutionRuntime(backend_refs={
        "voxelizer": MockVoxelizer(),
        "physicist": MockPhysicist(),
        "builder": MockBuilder()
    })
    
    # 2. Trigger Flow 1: Successful Pass
    print("\n[Test 1] Standard Steel Bracket (Should Pass)")
    ctx1 = {"intent": "Make a strong bracket", "material": "STEEL", "feature": "BRACKET"}
    
    # Step A: Concept -> Draft
    res1 = await runtime.trigger_evolution(ctx1)
    print(f" -> Step A Result: {res1['status']} ({res1['state']})")
    if res1['state'] != "DRAFT":
        print("❌ Failed to reach DRAFT state")
        return

    # Step B: Draft -> Simulated
    res2 = await runtime.trigger_evolution({}) # Check next step
    print(f" -> Step B Result: {res2['status']} ({res2['state']})")
    
    if res2['state'] == "SIMULATED":
        print("✅ Test 1 Passed: reached SIMULATED state.")
    else:
        print(f"❌ Test 1 Failed: Stalled at {res2['state']}")

    # 3. Trigger Flow 2: Failure & Regression
    print("\n[Test 2] Weak Aluminum Bracket (Should Fail Simulation)")
    # Reset State manualy for test
    runtime.state = LifecycleState.CONCEPT 
    ctx2 = {"intent": "Cheap bracket", "material": "WEAK_ALUMINUM", "feature": "BRACKET"}
    
    # Step A: Concept -> Draft
    await runtime.trigger_evolution(ctx2)
    
    # Step B: Draft -> Simulation Check
    res3 = await runtime.trigger_evolution({})
    print(f" -> Step B Result: {res3['status']} ({res3.get('reason', 'No Reason')})")
    
    if res3['status'] == "REGRESSION":
        print("✅ Test 2 Passed: Correctly REJECTED unsafe design.")
    else:
        print(f"❌ Test 2 Failed: Should have rejected, but status is {res3['status']}")

if __name__ == "__main__":
    asyncio.run(test_evolution_flow())
