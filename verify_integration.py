
import sys
import os
import asyncio
from typing import Dict, Any

# Ensure path to find backend
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from advanced_cnc_copilot.backend.cms.vulkan_ingestor import grid_state
from advanced_cnc_copilot.backend.cms.services.shadow_council import AuditorAgent, DecisionPolicy

async def run_test():
    print("🧪 Starting Integration Verification...")
    
    # 1. Setup Environment
    policy = DecisionPolicy()
    auditor = AuditorAgent(policy)
    
    # 2. Simulate Normal State
    print("\n--- Test Case 1: Normal Grid State (0.2 Saturation) ---")
    grid_state["saturation"] = 0.2
    
    proposal_normal = {
        'proposed_parameters': {'feed_rate': 2000.0},
        'confidence': 0.9
    }
    current_state = {'feed_rate': 1000.0}
    
    result = auditor.validate_proposal(proposal_normal, current_state)
    print(f"Result: Approved={result['is_approved']}")
    if result['is_approved']:
        print("✅ PASS: Low saturation allowed action.")
    else:
        print("❌ FAIL: Low saturation blocked action.")

    # 3. Simulate Saturated State
    print("\n--- Test Case 2: Saturated Grid State (0.95 Saturation) ---")
    grid_state["saturation"] = 0.95
    
    result_saturated = auditor.validate_proposal(proposal_normal, current_state)
    print(f"Result: Approved={result_saturated['is_approved']}")
    
    has_saturation_violation = any(v['parameter'] == 'grid_saturation' for v in result_saturated['constraint_violations'])
    
    if not result_saturated['is_approved'] and has_saturation_violation:
        print("✅ PASS: High saturation blocked action.")
    else:
        print("❌ FAIL: High saturation did NOT block action.")
        print(result_saturated)

if __name__ == "__main__":
    asyncio.run(run_test())
