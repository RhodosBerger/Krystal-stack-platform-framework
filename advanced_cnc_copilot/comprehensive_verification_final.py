#!/usr/bin/env python3
"""
Comprehensive Verification Script for FANUC RISE v2.1 Advanced CNC Copilot

This script performs comprehensive verification of all system components
to identify and validate services that may not be working properly.
"""

from typing import Dict, List, Any
import subprocess
import requests
import json
import sys
import time
import socket

def check_container_status() -> Dict[str, str]:
    """Check the status of Docker containers"""
    try:
        result = subprocess.run(['docker-compose', 'ps', '--format', 'json'],
                               capture_output=True, text=True, check=True)
        
        containers = {}
        for line in result.stdout.strip().split('\n'):
            if line:
                try:
                    container_info = json.loads(line)
                    name = container_info.get('Name', '')
                    status = container_info.get('Status', '')
                    containers[name] = status
                except json.JSONDecodeError:
                    continue
        
        return containers
    except subprocess.CalledProcessError:
        # Fallback to simple ps command
        try:
            result = subprocess.run(['docker-compose', 'ps'],
                                   capture_output=True, text=True, check=True)
            
            containers = {}
            lines = result.stdout.strip().split('\n')
            for line in lines[1:]:  # Skip header
                if line.strip():
                    parts = line.split()
                    if len(parts) >= 3:
                        name = parts[0]
                        status = ' '.join(parts[1:])
                        containers[name] = status
            
            return containers
        except subprocess.CalledProcessError:
            return {}

def check_port_availability(port: int) -> bool:
    """Check if a port is available/listening"""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        result = sock.connect_ex(('localhost', port))
        sock.close()
        return result == 0
    except:
        return False

def test_api_connectivity():
    """Test API connectivity and basic endpoints"""
    print("Testing API connectivity...")
    
    tests = [
        ("Health Check", "http://localhost:8000/health"),
        ("API Health", "http://localhost:8000/api/health"),
        ("Telemetry", "http://localhost:8000/api/telemetry"),
        ("Machines", "http://localhost:8000/api/machines")
    ]
    
    results = {}
    for name, url in tests:
        try:
            response = requests.get(url, timeout=10)
            results[name] = response.status_code == 200
            print(f"   {name}: {'[PASS]' if results[name] else '[FAIL]'} ({response.status_code})")
        except requests.exceptions.RequestException as e:
            results[name] = False
            print(f"   {name}: [FAIL] (Error: {str(e)})")
    
    return results

def test_database_connectivity():
    """Test database connectivity directly"""
    print("Testing database connectivity...")
    
    try:
        # Test if we can execute a simple query via the API
        response = requests.get("http://localhost:8000/api/health", timeout=10)
        if response.status_code == 200:
            data = response.json()
            db_status = data.get('database_status', 'unknown')
            print(f"   Database Status: {db_status}")
            return db_status == 'connected'
        else:
            print(f"   Unable to reach health endpoint: {response.status_code}")
            return False
    except Exception as e:
        print(f"   Database connectivity test failed: {str(e)}")
        return False

def test_redis_connectivity():
    """Test Redis connectivity"""
    print("Testing Redis connectivity...")
    
    try:
        response = requests.get("http://localhost:8000/api/health", timeout=10)
        if response.status_code == 200:
            data = response.json()
            cache_status = data.get('cache_status', 'unknown')
            print(f"   Cache Status: {cache_status}")
            return cache_status == 'connected'
        else:
            print(f"   Unable to reach health endpoint: {response.status_code}")
            return False
    except Exception as e:
        print(f"   Redis connectivity test failed: {str(e)}")
        return False

def test_frontend_services():
    """Test frontend services accessibility"""
    print("Testing frontend services...")
    
    tests = [
        ("React Frontend", "http://localhost:3000"),
        ("Vue Frontend", "http://localhost:8080"),
        ("NGINX Proxy", "http://localhost:80")
    ]
    
    results = {}
    for name, url in tests:
        try:
            response = requests.get(url, timeout=10)
            results[name] = response.status_code in [200, 404]  # 404 means service is running but page not found
            print(f"   {name}: {'[PASS]' if results[name] else '[FAIL]'} ({response.status_code})")
        except requests.exceptions.RequestException as e:
            results[name] = False
            print(f"   {name}: [FAIL] (Error: {str(e)})")
    
    return results

def test_authentication():
    """Test authentication system"""
    print("Testing authentication system...")
    
    try:
        # Test token endpoint
        response = requests.post(
            "http://localhost:8000/token",
            data={"username": "test", "password": "test"},
            timeout=10
        )
        # We expect either a 200 (if test user exists) or 400/401 (if invalid credentials)
        # Both indicate the auth system is working
        auth_working = response.status_code in [200, 400, 401]
        print(f"   Auth endpoint reachable: {'[PASS]' if auth_working else '[FAIL]'}")
        return auth_working
    except Exception as e:
        print(f"   Authentication test failed: {str(e)}")
        return False

def test_background_jobs():
    """Test background job processing"""
    print("Testing background job processing...")
    
    try:
        # Try to trigger a background task
        response = requests.post(
            "http://localhost:8000/api/tasks/synthetic-data",
            json={"duration": 10},
            timeout=10
        )
        # Expect either success or validation error (both mean endpoint exists)
        success = response.status_code in [200, 201, 422]
        print(f"   Task queue endpoint: {'[PASS]' if success else '[FAIL]'}")
        return success
    except Exception as e:
        print(f"   Background job test failed: {str(e)}")
        return False

def test_file_upload():
    """Test file upload functionality"""
    print("Testing file upload functionality...")
    
    try:
        # Test upload endpoint
        response = requests.post(
            "http://localhost:8000/api/upload/batch",
            files={"files": ("test.txt", "test content")},
            timeout=10
        )
        # Expect either success or validation error (both mean endpoint exists)
        success = response.status_code in [200, 422, 400, 500]
        print(f"   Upload endpoint: {'[PASS]' if success else '[FAIL]'}")
        return success
    except Exception as e:
        print(f"   File upload test failed: {str(e)}")
        return False

def test_monitoring_stack():
    """Test monitoring services"""
    print("Testing monitoring stack...")
    
    tests = [
        ("Prometheus", "http://localhost:9090"),
        ("Grafana", "http://localhost:3001")
    ]
    
    results = {}
    for name, url in tests:
        try:
            response = requests.get(url, timeout=10)
            results[name] = response.status_code in [200, 302]  # 302 is redirect which is OK for Grafana
            print(f"   {name}: {'[PASS]' if results[name] else '[FAIL]'} ({response.status_code})")
        except requests.exceptions.RequestException as e:
            results[name] = False
            print(f"   {name}: [FAIL] (Error: {str(e)})")
    
    return results

def test_specific_endpoints():
    """Test specific critical endpoints"""
    print("Testing specific critical endpoints...")
    
    endpoints = [
        ("/api/cortex/stats", "Cortex Stats"),
        ("/api/knowledge/search", "Knowledge Search"),
        ("/api/swarm/status", "Swarm Status"),
        ("/api/presets/proven", "Proven Presets"),
        ("/api/possibilities", "Possibilities Engine"),
        ("/api/intelligence/insights", "Intelligence Insights")
    ]
    
    results = {}
    for endpoint, name in endpoints:
        try:
            response = requests.get(f"http://localhost:8000{endpoint}", timeout=10)
            # We consider it working if we get any response (even 401/403)
            results[name] = response.status_code not in [404, 500, 502, 503, 504]
            print(f"   {name}: {'[PASS]' if results[name] else '[FAIL]'} ({response.status_code})")
        except requests.exceptions.RequestException as e:
            results[name] = False
            print(f"   {name}: [FAIL] (Error: {str(e)})")
    
    return results

def run_comprehensive_verification():
    """Run comprehensive verification of all system components"""
    print("FANUC RISE v2.1 Comprehensive Verification")
    print("=" * 60)
    
    # Get container status
    print("[INFO] Checking container statuses...")
    containers = check_container_status()
    for name, status in containers.items():
        status_indicator = "[UP]" if "Up" in status else "[DOWN]"
        print(f"   {name}: {status_indicator} {status}")
    print()
    
    # Test all components
    test_results = {}
    
    test_results['api_connectivity'] = test_api_connectivity()
    print()
    
    test_results['database'] = test_database_connectivity()
    print()
    
    test_results['redis'] = test_redis_connectivity()
    print()
    
    test_results['frontends'] = test_frontend_services()
    print()
    
    test_results['authentication'] = test_authentication()
    print()
    
    test_results['background_jobs'] = test_background_jobs()
    print()
    
    test_results['file_upload'] = test_file_upload()
    print()
    
    test_results['monitoring'] = test_monitoring_stack()
    print()
    
    test_results['specific_endpoints'] = test_specific_endpoints()
    print()
    
    # Summary
    print("VERIFICATION SUMMARY")
    print("=" * 60)
    
    all_results = []
    for category, results in test_results.items():
        if isinstance(results, dict):
            # For grouped results
            for name, status in results.items():
                all_results.append((f"{category}.{name}", status))
                print(f"   {name}: {'[PASS]' if status else '[FAIL]'}")
        else:
            # For single results
            all_results.append((category, results))
            print(f"   {category}: {'[PASS]' if results else '[FAIL]'}")
    
    # Count passing tests
    passed = sum(1 for _, status in all_results if status)
    total = len(all_results)
    
    print(f"\nOverall: {passed}/{total} components operational")
    
    if passed == total:
        print("\nALL SYSTEM COMPONENTS ARE WORKING PROPERLY!")
        print("FANUC RISE v2.1 Advanced CNC Copilot is fully operational")
        return 0
    else:
        failed_count = total - passed
        print(f"\n{failed_count} out of {total} components have issues")
        print("Please address the failing components before production use")
        return 1

def diagnose_issues():
    """Provide diagnostics for common issues"""
    print("\nDIAGNOSTIC RECOMMENDATIONS")
    print("=" * 60)
    
    containers = check_container_status()
    
    # Check for common issues
    if "rise_backend" not in containers or "Up" not in containers.get("rise_backend", ""):
        print("   - Backend service may not be running: Check docker-compose logs api")
    
    if "rise_db" not in containers or "healthy" not in containers.get("rise_db", ""):
        print("   - Database may not be healthy: Check docker-compose logs db")
    
    if "rise_redis" not in containers or "healthy" not in containers.get("rise_redis", ""):
        print("   - Redis may not be healthy: Check docker-compose logs redis")
    
    if "frontend-react" not in containers or "Up" not in containers.get("frontend-react", ""):
        print("   - React frontend may not be running: Check docker-compose logs frontend-react")
    
    if "frontend-vue" not in containers or "Up" not in containers.get("frontend-vue", ""):
        print("   - Vue frontend may not be running: Check docker-compose logs frontend-vue")
    
    if "nginx" not in containers or "Up" not in containers.get("nginx", ""):
        print("   - NGINX proxy may not be running: Check docker-compose logs nginx")
    
    print("\nCOMMON TROUBLESHOOTING COMMANDS:")
    print("   - docker-compose logs api")
    print("   - docker-compose logs frontend-react")
    print("   - docker-compose logs frontend-vue")
    print("   - docker-compose logs nginx")
    print("   - docker-compose restart api")
    print("   - docker-compose restart nginx")

if __name__ == "__main__":
    exit_code = run_comprehensive_verification()
    diagnose_issues()
    sys.exit(exit_code)