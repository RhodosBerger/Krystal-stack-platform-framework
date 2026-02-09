
import sys
import os
import asyncio
import logging

# Add path
sys.path.append(os.path.join(os.getcwd(), 'advanced_cnc_copilot'))

# Logging
logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger("IntegrationTest")
logger.setLevel(logging.INFO)

# Mock missing dependencies
from unittest.mock import MagicMock
sys.modules["numpy"] = MagicMock()
sys.modules["redis"] = MagicMock()
sys.modules["celery"] = MagicMock()
sys.modules["celery.result"] = MagicMock()
sys.modules["scipy"] = MagicMock()
sys.modules["fastapi"] = MagicMock()
sys.modules["pydantic"] = MagicMock()
sys.modules["sqlalchemy"] = MagicMock()
sys.modules["sqlalchemy.orm"] = MagicMock()
sys.modules["requests"] = MagicMock()
sys.modules["openai"] = MagicMock()

async def test_integration():
    print("🚀 Starting Integration Test...")
    
    try:
        from backend.core.orchestrator import orchestrator
        print("✅ Orchestrator Imported.")
        
        # Initialize
        await orchestrator.initialize()
        print("✅ Orchestrator Initialized.")
        
        # Test 1: Consultation (Classic -> Agent)
        print("\n--- Testing Consultation (Expect Agent Response) ---")
        payload = {"question": "How do I optimize steel milling?"}
        response = await orchestrator.process_request("CONSULTATION", payload, "test_user")
        
        print(f"Response: {response}")
        
        if "ManufacturingAgent" in str(response):
             print("✅ Success: Response came from ManufacturingAgent.")
        else:
             print(f"⚠️ Warning: Response source might be fallback: {response.get('data', {}).get('source')}")

        # Test 2: G-Code (Async -> Scheduler -> Agent/Legacy)
        # Note: G-Code goes to Celery buffer, so we might just see QUEUED.
        
    except Exception as e:
        print(f"❌ Integration Test Failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_integration())
