"""
Phase 8 Verification Script 🧪
Validates Sustainability Engine & Cortex Analytics
"""
import requests
import time
import sys

BASE_URL = "http://localhost:8000"

def test_sustainability():
    print(f"🌍 Testing Sustainability Engine...")
    payload = {
        "description": "Milling steel bracket",
        "material": "Steel"
    }
    try:
        resp = requests.post(f"{BASE_URL}/api/sustainability/estimate", json=payload)
        if resp.status_code == 200:
            data = resp.json()
            if "carbon_footprint_kg" in data:
                print(f"✅ Sustainability OK: Calculated {data['carbon_footprint_kg']}kg CO2e")
                return True
            else:
                print(f"❌ Sustainability Error: Missing fields in {data}")
        else:
            print(f"❌ Sustainability API Error: {resp.status_code} - {resp.text}")
    except Exception as e:
        print(f"❌ Connection Error: {e}")
    return False

def test_cortex_analytics():
    print(f"🧠 Testing Cortex Analytics...")
    try:
        resp = requests.get(f"{BASE_URL}/api/cortex/stats?limit=5")
        if resp.status_code == 200:
            data = resp.json()
            if "intents" in data:
                print(f"✅ Cortex Analytics OK: Retrieved {len(data['intents'])} records")
                # print sample
                if data["intents"]:
                    print(f"   Latest Intent: {data['intents'][0].get('action')}")
                return True
            else:
                print(f"❌ Cortex Error: Invalid structure {data}")
        else:
            print(f"❌ Cortex API Error: {resp.status_code} - {resp.text}")
    except Exception as e:
        print(f"❌ Connection Error: {e}")
    return False

def main():
    print("🚀 Starting Phase 8 Verification")
    passed = 0
    total = 2
    
    if test_sustainability(): passed += 1
    if test_cortex_analytics(): passed += 1
    
    print("-" * 30)
    print(f"Tests Passed: {passed}/{total}")
    
    if passed == total:
        print("✅ PHASE 8 VERIFIED SUCCESS")
        sys.exit(0)
    else:
        print("❌ PHASE 8 VERIFIED FAILED")
        sys.exit(1)

if __name__ == "__main__":
    main()
