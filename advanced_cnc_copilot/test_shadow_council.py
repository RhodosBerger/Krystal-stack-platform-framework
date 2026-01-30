import asyncio
from cms.message_bus import global_bus
from cms.auditor_agent import AuditorAgent
from cms.interaction_supervisor import InteractionSupervisor

async def test_audit_flow():
    print("--- STARTING SHADOW COUNCIL SIMULATION ---")
    
    # 1. Init System
    supervisor = InteractionSupervisor()
    # We won't call supervisor.start() fully because it has a while True loop.
    # Instead, we manually start the specific component we want to test.
    await supervisor.auditor.start()
    
    # 2. Mock a "Bad Plan" (Creator's Role)
    bad_plan = {
        "job_id": "TEST_JOB_001",
        "action": "MILLING",
        "material": "Titanium",
        "rpm": 12000, # WAY too fast for Titanium
        "feed": 500
    }
    
    # 3. Subscribe to the result so we can see it
    async def print_verdict(msg):
        print(f"\n>>> [FINAL VERDICT] from {msg.sender_id}")
        print(f"Status: {msg.payload['status']}")
        if msg.payload['errors']:
            print(f"Errors: {msg.payload['errors']}")
        print("----------------------------------------\n")

    global_bus.subscribe("VALIDATION_RESULT", print_verdict)
    
    # 4. Publish the intent
    print(f"Creator proposing plan: {bad_plan}")
    await global_bus.publish("DRAFT_PLAN", bad_plan, "CREATOR_MOCK")
    
    # 5. Wait for async processing
    await asyncio.sleep(1)
    
    # 6. Mock a "Good Plan"
    good_plan = {
        "job_id": "TEST_JOB_002",
        "action": "MILLING",
        "material": "Aluminum",
        "rpm": 8000,
        "feed": 1200
    }
    print(f"Creator proposing plan: {good_plan}")
    await global_bus.publish("DRAFT_PLAN", good_plan, "CREATOR_MOCK")
    
    await asyncio.sleep(1)

if __name__ == "__main__":
    asyncio.run(test_audit_flow())
