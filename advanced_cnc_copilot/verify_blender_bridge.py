"""
Verify Blender Bridge & Voxelizer 🎨
Simulates a request coming from the Blender Add-on.
"""
import requests
import json
import sys
import time

BASE_URL = "http://localhost:8000"

def test_blender_bridge():
    print("🎨 Testing Blender -> Voxelizer Bridge...")
    
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

    # 1. Send Geometry Payload (Mocking Blender Client)
    payload = {
        "type": "ANALYZE_GEOMETRY",
        "payload": {
            "name": "Complex_Turbine_Blade_v4",
            "vertex_count": 15420,
            "dimensions": [10.5, 5.2, 3.1]
        }
    }
    
    try:
        resp = requests.post(
            f"{BASE_URL}/api/manufacturing/request", 
            json=payload,
            headers=headers
        )
        
        if resp.status_code == 200:
            data = resp.json()
            print("✅ Bridge Connection Successful!")
            print(f"   Status: {data.get('status')}")
            
            if data.get('status') == 'ERROR':
                 print(f"   ❌ Backend Error Message: {data.get('message')}")
                 return False

            result = data.get('data', {})
            graphs = result.get('graphs', {})
            
            # Check for Precise Graphs
            if 'curvature' in graphs and 'thickness' in graphs:
                print("   ✅ Precise Parameter Graphs Returned")
                print(f"      Curvature Mean: {graphs['curvature'].get('mean'):.4f}")
                print(f"      Thickness Range: {graphs['thickness'].get('min_mm')} - {graphs['thickness'].get('max_mm')} mm")
                return True
            else:
                print("   ❌ Missing Graphs in Response")
                return False
        else:
            print(f"❌ Bridge Failed: {resp.text}")
            return False
            
    except Exception as e:
        print(f"❌ Exception: {e}")
        return False

if __name__ == "__main__":
    if test_blender_bridge():
        sys.exit(0)
    else:
        sys.exit(1)
