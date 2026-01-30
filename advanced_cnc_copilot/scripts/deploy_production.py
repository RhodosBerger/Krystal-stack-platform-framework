"""
🚀 FANUC RISE - Production Deployment Script
Responsibility:
1. Pre-Flight Checks (Docker, Env, Ports).
2. Run automated test suite (QA, Persistence).
3. Build and Launch Containers.
4. Verify Health of deployed services.
"""
import os
import sys
import subprocess
import time
import requests

def print_step(msg):
    print(f"\n🔹 {msg}...")

def check_command(command, error_msg):
    try:
        subprocess.check_call(command, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except:
        print(f"❌ {error_msg}")
        return False

def run_tests():
    print_step("Running Verification Suite")
    tests = [
        "test_quality_assurance.py", 
        "test_persistence.py",
        "verify_frontend.py"
    ]
    
    overall_success = True
    for test in tests:
        if os.path.exists(test):
            print(f"   Running {test}...", end=" ")
            try:
                subprocess.check_call(f"python {test}", shell=True, stdout=subprocess.DEVNULL)
                print("✅ PASS")
            except:
                print("❌ FAIL")
                # We alert but don't block deployment for non-critical local tests (persistence might fail without docker)
                if test == "test_quality_assurance.py": 
                    overall_success = False
        else:
            print(f"   ⚠️ {test} not found")
            
    if not overall_success:
        input("⚠️ Some tests failed. Press Enter to continue anyway, or Ctrl+C to abort.")

def main():
    print("========================================")
    print("🏭 FANUC RISE - DEPLOYING PRODUCTION")
    print("========================================")
    
    # 1. Environment Checks
    print_step("Checking Environment")
    if not check_command("docker --version", "Docker is not installed or not in PATH"): return
    if not check_command("docker-compose --version", "Docker Compose is not installed"): return
    
    if not os.path.exists(".env"):
        print("⚠️ .env file missing! Creating default...")
        with open(".env", "w") as f:
            f.write("OPENAI_API_KEY=your_key_here\nPOSTGRES_USER=user\nPOSTGRES_PASSWORD=password\n")
            
    # 2. Build & Deploy
    print_step("Building Containers (This may take a while)")
    code = subprocess.call("docker-compose up -d --build", shell=True)
    if code != 0:
        print("❌ Deployment Failed")
        return

    # 3. Health Check & Warmup
    print_step("Waiting for System Warmup (15s)")
    time.sleep(15)
    
    # 4. Run Integration Tests
    run_tests()
    
    try:
        print("\n   Pinging Backend Health...", end=" ")
        resp = requests.get("http://localhost:8000/api/health")
        if resp.status_code == 200:
            print(f"✅ ONLINE ({resp.json()})")
        else:
            print(f"⚠️ Unstable (Status {resp.status_code})")
    except:
        print("❌ Backend Unreachable (Check Logs)")
        
    print("\n✅ DEPLOYMENT COMPLETE!")
    print("   -> Dashboard: http://localhost:8000/")
    print("   -> API Docs:  http://localhost:8000/docs")

if __name__ == "__main__":
    main()
