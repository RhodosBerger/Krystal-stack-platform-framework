
import asyncio
import logging
import sys
import os

# Setup paths
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from backend.core.orchestrator import orchestrator
from cms.message_bus import global_bus

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("VERIFY_PHASE_2")

async def test_shadow_council():
    logger.info("--- STARTING SHADOW COUNCIL VERIFICATION ---")
    
    # 1. Initialize System
    logger.info("Initializing Orchestrator & Council...")
    await orchestrator.initialize()
    
    # 2. Test Safe Job (Should Pass)
    logger.info("\n[TEST 1] Submitting SAFE Job (Aluminum, 8000 RPM)...")
    safe_payload = {
        "description": "Safe milling operation",
        "material": "Aluminum",
        "rpm": 8000,
        "feed": 1000
    }
    result_safe = await orchestrator.process_request("GENERATE_GCODE", safe_payload, "tester")
    logger.info(f"Safe Job Status: {result_safe.get('status')}")
    assert result_safe.get('status') != "BLOCKED", "Safe job was incorrectly blocked!"

    # 3. Test Dangerous Job (Should be Vetoed by Auditor)
    logger.info("\n[TEST 2] Submitting DANGEROUS Job (Titanium, 10000 RPM)...")
    logger.info("Note: Auditor limit for Titanium is 4000 RPM")
    
    danger_payload = {
        "description": "Dangerous operation",
        "material": "Titanium",
        "rpm": 10000, # Way above limit
        "feed": 2000
    }
    result_danger = await orchestrator.process_request("GENERATE_GCODE", danger_payload, "tester")
    logger.info(f"Danger Job Status: {result_danger.get('status')}")
    logger.info(f"Block Reason: {result_danger.get('errors')}")
    
    assert result_danger.get('status') == "BLOCKED", "Dangerous job was NOT blocked!"
    assert "exceeds safety limit" in str(result_danger.get('errors')), "Incorrect error message"

    # 4. Test Dopamine Response (Neuro-System)
    logger.info("\n[TEST 3] Injecting High-Stress Telemetry...")
    
    # Subscribe to verify we get the Neuro-Update
    future = asyncio.get_event_loop().create_future()
    async def neuro_listener(msg):
        if msg.sender_id == "DOPAMINE_CORE":
            future.set_result(msg.payload)
            
    global_bus.subscribe("NEURO_STATE_UPDATE", neuro_listener)
    
    # Inject "Stress" (High Vibration)
    await global_bus.publish("TELEMETRY_UPDATE", {
        "machine_id": "TEST-01",
        "rpm": 5000,
        "vibration": 0.8 # Very high (0.8g)
    }, sender_id="TESTER")
    
    try:
        neuro_state = await asyncio.wait_for(future, timeout=2.0)
        logger.info(f"Received Neuro-State: {neuro_state['state']}")
        
        cortisol = neuro_state['state']['cortisol']
        logger.info(f"Cortisol Level: {cortisol}")
        
        # Initial cortisol was 20. High vib should spike it.
        assert cortisol > 25, "Cortisol did not rise enough from stress!"
        
    except asyncio.TimeoutError:
        logger.error("Dopamine Engine did not respond!")
        raise

    logger.info("\n--- VERIFICATION SUCCESSFUL: THE COUNCIL IS WATCHING ---")

if __name__ == "__main__":
    try:
        if sys.platform == 'win32':
             asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        asyncio.run(test_shadow_council())
    except Exception as e:
        logger.error(f"Test Failed: {e}")
        sys.exit(1)
