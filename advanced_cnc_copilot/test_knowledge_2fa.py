import time
import requests
import json
from backend.worker import celery_app

API_URL = "http://localhost:8000"

def test_2fa_flow():
    print("🔐 Verifying Knowledge 2FA (Factor 1 & 2)...")
    
    # 1. Login
    print("\n[Auth] Logging in...")
    try:
        token_resp = requests.post(f"{API_URL}/token", json={"username": "admin", "password": "admin123"})
        token = token_resp.json().get("access_token")
        headers = {"Authorization": f"Bearer {token}"}
        print("✅ Logged in")
    except Exception as e:
        print(f"❌ Login Failed: {e}")
        return

    # 2. Trigger Crawler (Factor 1: Create Pending)
    print("\n[Factor 1] Triggering Crawler Task...")
    task = celery_app.send_task("tasks.check_industry_blogs")
    
    # Poll for Task Completion to get Verification ID
    verification_id = None
    for _ in range(10):
        if task.ready():
            res = task.get()
            verification_id = res.get("verification_id")
            print(f"✅ Crawler Finished. Held for Verification: {verification_id}")
            break
        time.sleep(1)
        
    if not verification_id:
        print("❌ Crawler failed to produce verification ID")
        return

    # 3. Check Pending Queue (User View)
    print("\n[User View] Checking Pending Queue...")
    pending_resp = requests.get(f"{API_URL}/api/knowledge/pending", headers=headers)
    items = pending_resp.json().get("items", [])
    print(f"   Pending Items Count: {len(items)}")
    found = False
    for item in items:
        if item["id"] == verification_id:
            print(f"   Found Target: {item['name']} (Source: {item['source']})")
            found = True
    
    if not found:
        print("❌ Verification ID not found in pending list")
        return

    # 4. Verify/Approve (Factor 2: Commit)
    print(f"\n[Factor 2] Approving Artifact {verification_id}...")
    verify_resp = requests.post(f"{API_URL}/api/knowledge/verify/{verification_id}", headers=headers)
    if verify_resp.status_code == 200:
        print("✅ Verification Successful!")
        print(json.dumps(verify_resp.json(), indent=2))
    else:
        print(f"❌ Verification Failed: {verify_resp.text}")

if __name__ == "__main__":
    test_2fa_flow()
