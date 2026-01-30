"""
Async Flow Verification 🌊
Tests the full lifecycle of a Celery Job:
1. Submit Job -> API returns Job ID
2. Poll API -> Check for "QUEUED" then "COMPLETED"
3. Retrieve Result
"""
import requests
import time
import sys

BASE_URL = "http://localhost:8000"

def test_async_gcode_gen():
    print(f"🌊 Testing Async G-Code Generation...")
    
    # 0. Authenticate
    auth_payload = {"username": "admin", "password": "admin123"}
    try:
        auth_resp = requests.post(f"{BASE_URL}/token", json=auth_payload)
        if auth_resp.status_code != 200:
            print(f"❌ Auth Failed: {auth_resp.text}")
            return False
        token = auth_resp.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}
    except Exception as e:
        print(f"❌ Auth Exception: {e}")
        return False
    
    # 1. Submit
    payload = {
        "type": "GENERATE_GCODE",
        "payload": {
            "description": "Simple titanium bracket 50x50mm",
            "material": "Titanium Grade 5"
        }
    }
    
    try:
        resp = requests.post(
            f"{BASE_URL}/api/manufacturing/request", 
            json=payload,
            headers=headers
        )
        if resp.status_code != 200:
            print(f"❌ Submit Failed: {resp.text}")
            return False
            
        data = resp.json()
        print(f"DEBUG: Response Type: {type(data)}")
        print(f"DEBUG: Response Data: {data}")
        
        if isinstance(data, str):
            import json
            try:
                data = json.loads(data)
            except:
                pass
                
        job_id = data.get("job_id")
        
        if not job_id:
            # It might have fallen back to sync if redis failed?
            print(f"⚠️ No Job ID returned. Response: {data}")
            return False
            
        print(f"   ✅ Job Submitted: {job_id} (Status: {data.get('status')})")
        
        # 2. Poll
        max_retries = 20
        for i in range(max_retries):
            time.sleep(1)
            status_resp = requests.get(
                f"{BASE_URL}/api/jobs/{job_id}",
                headers=headers
            )
            if status_resp.status_code == 200:
                s_data = status_resp.json()
                print(f"DEBUG: Poll Data: {s_data}")
                
                # Handle potential error response which is flat
                if isinstance(s_data.get("status"), str):
                    status = s_data.get("status")
                else:
                    status = s_data.get("status", {}).get("status")
                print(f"   ... Polling {i+1}: {status}")
                
                if status == "COMPLETED":
                    print(f"   ✅ Job Completed!")
                    print(f"   Result Summary: {s_data.get('status', {}).get('result_summary')}")
                    return True
                if status == "FAILED":
                    print(f"   ❌ Job Failed: {s_data}")
                    return False
                    
        print("❌ Timeout waiting for job")
        return False
        
    except Exception as e:
        print(f"❌ Exception: {e}")
        return False

if __name__ == "__main__":
    if test_async_gcode_gen():
        sys.exit(0)
    else:
        sys.exit(1)
