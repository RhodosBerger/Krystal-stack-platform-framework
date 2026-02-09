
import sys
import os
import logging

# Add backend parent dir to path
sys.path.append(os.path.join(os.getcwd(), 'advanced_cnc_copilot'))

# Configure logging to see our "Initialing Augmented LLM Brain..." message
logging.basicConfig(level=logging.INFO)

print("Attempting to import LLMRouter...")
try:
    from backend.core.llm_brain import llm_router, LLMProvider
    print("✅ Import Successful.")
    
    print("Attempting Query...")
    response = llm_router.query(
        system_prompt="Test System",
        user_prompt="Hello Brain",
        provider=None # Should default to Augmented -> OpenVINO/Mock
    )
    print(f"Response: {response}")
    
    if "Simulated Response" in response or "OpenVINO" in response:
         print("✅ Verification Passed: Response came from Augmented Core (Mock/OpenVINO).")
    else:
         print(f"⚠️ Unexpected Response: {response}")

except Exception as e:
    print(f"❌ Verification Failed: {e}")
    import traceback
    traceback.print_exc()
