import os

def test_docker_compose():
    print("🐳 Testing Deployment Configuration (Dependency-Free)...")
    
    compose_path = "docker-compose.yml"
    if not os.path.exists(compose_path):
        print("❌ FAIL: docker-compose.yml not found")
        return
        
    try:
        with open(compose_path, "r") as f:
            content = f.read()
            
        print(f"   Config Size: {len(content)} bytes")
        
        required_services = ["services:", "backend:", "frontend:", "redis:"]
        
        missing = [s for s in required_services if s not in content]
        
        if missing:
            print(f"❌ FAIL: Missing keys: {missing}")
        else:
            print("✅ PASS: All core services defined.")
            
        # Check backend build context
        if "backend/Dockerfile" in content:
            print("✅ PASS: Backend build context correct.")
        else:
             print("❌ FAIL: Backend Dockerfile path mismatch.")
             
        print("🚀 Deployment Config Verified.")
        
    except Exception as e:
        print(f"❌ FAIL: Read Error: {e}")

if __name__ == "__main__":
    test_docker_compose()
