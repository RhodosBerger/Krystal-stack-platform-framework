import requests

BASE_URL = "http://localhost:8000"

def verify_frontend():
    print("🖥️ Verifying Generative Frontend (Multisite Bridge)...")
    
    views = [
        ("operator", "RISE | operator"),
        ("knowledge", "RISE | knowledge"),
        ("cortex", "RISE | cortex"),
        ("engineer", "RISE | engineer")
    ]
    
    success = True
    
    for view, expected_title in views:
        print(f"\n[Testing] /view/{view}")
        try:
            resp = requests.get(f"{BASE_URL}/view/{view}")
            if resp.status_code == 200:
                if expected_title in resp.text:
                    print(f"✅ Success! Title matched: '{expected_title}'")
                else:
                    print(f"❌ Content Mismatch. Expected '{expected_title}'")
                    success = False
            else:
                print(f"❌ HTTP Error {resp.status_code}")
                success = False
        except Exception as e:
            print(f"❌ Connection Failed: {e}")
            success = False

    if success:
        print("\n✅ All Generative Layouts Functional")
    else:
        print("\n❌ Frontend Verification Failed")

if __name__ == "__main__":
    verify_frontend()
