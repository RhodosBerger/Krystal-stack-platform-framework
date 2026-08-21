from cms.protocols.user_mirror import mirror_protocol
from backend.core.semantic_meaning_library import semantic_library
import json
import time

def test_persistence():
    print("💾 Verifying Persistence & Meaning (Long-Term Memory)...")

    # 1. Create a User Mirror Object (The Data)
    print("\n[Protocol] Creating User Mirror...")
    project_id = f"PROJ-{int(time.time())}"
    
    intent_data = {
        "intent": "Create a lightweight bracket for aerospace",
        "constraints": ["Max weight 200g", "AL6061-T6"],
        "priority": "Quality"
    }
    
    geometry_data = {
        "shape_type": "Prismatic",
        "dimensions": {"length": 120.0, "width": 45.5},
        "features": ["M6 Thread", "Pocket 10mm"],
        "material": "Aluminum6061",
        "tolerance": "+/- 0.05mm"
    }
    
    mirror = mirror_protocol.create_mirror(project_id, intent_data, geometry_data)
    print(f"✅ Created Mirror: {mirror.project_id}")
    print(f"   Reflected Will: {mirror_protocol.reflect_will(mirror)}")
    
    # 2. Save to Long-Term Memory (The Save)
    print("\n[Library] Saving to LTM...")
    success = semantic_library.save_mirror(mirror)
    
    if success:
        print("✅ Save Operation Successful")
    else:
        print("❌ Save Failed")
        
    # 3. Recall from LTM (The Read)
    print("\n[Library] Recalling Project...")
    memory = semantic_library.recall_project(project_id)
    if memory:
        print(f"✅ Recalled Intent: {memory['intent']}")
        print(f"✅ Recalled Geometry: {json.loads(memory['geometry'])['shape_type']}")
    else:
        print("⚠️ Recall returned empty (Using Ephemeral Mode?)")

if __name__ == "__main__":
    test_persistence()
