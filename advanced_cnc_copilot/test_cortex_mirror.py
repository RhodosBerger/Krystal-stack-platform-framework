import redis
import json
import time
import requests

REDIS_URL = "redis://localhost:6379/0"
API_URL = "http://localhost:8000"

def verify_cortex_mirror():
    print("👁️ Verifying Cortex Observability & Intent Mirroring...")
    
    try:
        r = redis.Redis.from_url(REDIS_URL, decode_responses=True)
        r.ping()
        print("✅ Connected to Shared Memory (Redis)")
    except Exception as e:
        print(f"❌ Redis Connection Failed: {e}")
        return

    # 1. Trigger an Action via API (to generate logs/intent)
    print("\n[1] Generating Traffic (Manufacturing Request)...")
    try:
        # Login
        token_resp = requests.post(f"{API_URL}/token", json={"username": "admin", "password": "admin123"})
        token = token_resp.json().get("access_token")
        
        # Submit Job
        headers = {"Authorization": f"Bearer {token}"}
        payload = {
            "type": "GENERATE_GCODE",
            "payload": {
                "description": "Cortex Mirror Test",
                "material": "Titanium6Al4V"
            }
        }
        resp = requests.post(f"{API_URL}/api/manufacturing/request", json=payload, headers=headers)
        print(f"    Request Status: {resp.status_code}")
        time.sleep(2) # Wait for processing
        
    except Exception as e:
        print(f"⚠️ API Trigger Failed: {e}")

    # 2. Inspect Cortex Logs (Mirror)
    print("\n[2] Inspecting 'cortex:logs' (Mirror)...")
    logs = r.lrange("cortex:logs", 0, 5)
    if logs:
        for log_str in logs:
            log = json.loads(log_str)
            print(f"    [{log['level']}] {log['component']}: {log['message']}")
        print("✅ Log Mirroring Active")
    else:
        print("❌ No Logs Found in Cortex Mirror")

    # 3. Inspect Database of Intent
    print("\n[3] Inspecting 'cortex:intent_database'...")
    intents = r.lrange("cortex:intent_database", 0, 5)
    if intents:
        for intent_str in intents:
            intent = json.loads(intent_str)
            print(f"    ACTOR: {intent['actor']}")
            print(f"    ACTION: {intent['action']}")
            print(f"    REASONING: {intent['reasoning']}")
            print("    ---")
        print("✅ Database of Intent Active")
    else:
        print("❌ No Intent Records Found")

if __name__ == "__main__":
    verify_cortex_mirror()
