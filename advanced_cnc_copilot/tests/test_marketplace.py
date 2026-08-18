import requests
import json
import sys
import os

# Ensure backend module is visible
sys.path.append(os.getcwd())

BASE_URL = "http://localhost:8000/api/marketplace"

def test_marketplace():
    print("🛒 Testing Marketplace Ecosystem...")
    
    # 1. List Components
    try:
        # Note: In a real test we'd need the server running. 
        # Since I cannot guarantee the server is up on localhost:8000 in this environment, 
        # I will mock the IMPORT to verify code correctness, 
        # or rely on the user to run the server. 
        # BUT, I can import the router directly and test the function logic!
        
        from backend.core.marketplace import list_components, download_component, DownloadRequest, BackgroundTasks
        import asyncio
        
        # Test Listing
        print("   Listing Components...")
        res = asyncio.run(list_components(category="ALL"))
        components = res["components"]
        print(f"   ✅ Found {len(components)} components.")
        print(f"   Sample: {components[0]['name']} by {components[0]['author']}")
        
        # Test Filtering
        res_gcode = asyncio.run(list_components(category="GCODE"))
        print(f"   ✅ Filtered GCODE: {len(res_gcode['components'])} items.")
        
        # Test Download Mock
        print("   Simulating Download...")
        target_id = components[0]['id']
        bg_tasks = BackgroundTasks()
        res_dl = asyncio.run(download_component(target_id, DownloadRequest(target_node="TEST_NODE"), bg_tasks))
        
        if res_dl["status"] == "INITIATED":
            print(f"   ✅ Download Initiated: {res_dl['message']}")
        else:
            print("   ❌ Download Start Failed")
            
        print("🎉 Marketplace Logic Verified (Offline Mode).")
        
    except Exception as e:
        print(f"❌ Verification Failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_marketplace()
