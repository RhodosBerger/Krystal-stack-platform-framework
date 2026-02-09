
import sys
import os
import logging

# Add path
sys.path.append(os.path.join(os.getcwd(), 'advanced_cnc_copilot'))

# Logging
logging.basicConfig(level=logging.INFO)

print("Attempting to import ManufacturingAgent...")
try:
    from backend.agent.manufacturing import ManufacturingAgent
    from backend.core.augmented.llm_processor import LLMProcessor
    from backend.core.augmented.openvino_engine import OpenVINOEngine

    print("✅ Import Successful.")
    
    # Init dependencies (Mock/CPU)
    engine = OpenVINOEngine(device="CPU")
    processor = LLMProcessor(openvino_engine=engine)
    
    # Init Agent
    agent = ManufacturingAgent(engine, processor)
    print("✅ Agent Initialized.")

    # Test Task 1: Alpha (Plan)
    print("\n--- Testing Alpha Flow (Planning) ---")
    response_alpha = agent.run_task("Create a manufacturing process plan for a generic bracket.")
    print(f"Alpha Response Length: {len(response_alpha)} chars")
    
    # Test Task 2: Beta (G-Code)
    print("\n--- Testing Beta Flow (G-Code) ---")
    response_beta = agent.run_task("Generate G-Code for a 50mm square pocket.")
    print(f"Beta Response Length: {len(response_beta)} chars")
    
    if "QA Status" in response_beta:
        print("✅ QA Tool executed successfully.")
    else:
        print("⚠️ QA Tool output missing.")

    print("\n✅ Verification COMPLETE.")

except Exception as e:
    print(f"❌ Verification Failed: {e}")
    import traceback
    traceback.print_exc()
