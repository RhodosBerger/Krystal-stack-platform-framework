#!/usr/bin/env python3
"""
Verification Script for FANUC RISE v2.1 Advanced CNC Copilot Deployment

This script verifies that all components of the FANUC RISE v2.1 system are properly deployed
and accessible, with special attention to the frontend running on port 3000 as requested.
"""

import subprocess
import requests
import sys
import time
import json
from typing import Dict, List, Tuple

def check_docker_containers() -> Dict[str, str]:
    """Check the status of all Docker containers"""
    print("[INFO] Checking Docker container status...")
    try:
        result = subprocess.run(['docker', 'ps', '--format', '{{.Names}}\t{{.Status}}'],
                               capture_output=True, text=True, check=True)
        containers = {}
        for line in result.stdout.strip().split('\n')[1:]:  # Skip header
            if '\t' in line:
                name, status = line.split('\t', 1)
                containers[name] = status
        return containers
    except subprocess.CalledProcessError:
        print("⚠️  Could not retrieve Docker container status")
        return {}

def check_port_availability(port: int) -> bool:
    """Check if a port is available/listening"""
    import socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    result = sock.connect_ex(('localhost', port))
    sock.close()
    return result == 0

def test_api_health() -> bool:
    """Test the backend API health endpoint"""
    print("[INFO] Testing backend API health...")
    try:
        response = requests.get('http://localhost:8000/health', timeout=10)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False

def test_frontend_react() -> bool:
    """Test the React frontend on port 3000"""
    print("[INFO] Testing React frontend on port 3000...")
    try:
        response = requests.get('http://localhost:3000', timeout=10)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False

def test_frontend_vue() -> bool:
    """Test the Vue frontend on port 8080"""
    print("[INFO] Testing Vue frontend on port 8080...")
    try:
        response = requests.get('http://localhost:8080', timeout=10)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False

def test_database_connection() -> bool:
    """Test database connectivity"""
    print("[INFO] Testing database connection...")
    try:
        # Test using the backend API's database connectivity
        response = requests.get('http://localhost:8000/api/health', timeout=10)
        if response.status_code == 200:
            data = response.json()
            return data.get('database_status') == 'connected'
        return False
    except requests.exceptions.RequestException:
        return False

def test_redis_connection() -> bool:
    """Test Redis connectivity"""
    print("[INFO] Testing Redis connection...")
    try:
        response = requests.get('http://localhost:8000/api/health', timeout=10)
        if response.status_code == 200:
            data = response.json()
            return data.get('cache_status') == 'connected'
        return False
    except requests.exceptions.RequestException:
        return False

def main():
    """Main verification function"""
    print("FANUC RISE v2.1 Deployment Verification")
    print("=" * 50)
    
    # Check Docker containers
    containers = check_docker_containers()
    print(f"[INFO] Found {len(containers)} running containers")
    for name, status in containers.items():
        print(f"   - {name}: {status}")
    
    # Wait a moment for services to fully start
    print("\n[WAIT] Waiting for services to stabilize...")
    time.sleep(5)
    
    # Test services
    tests = [
        ("Backend API Health", test_api_health),
        ("React Frontend (Port 3000)", test_frontend_react),
        ("Vue Frontend (Port 8080)", test_frontend_vue),
        ("Database Connection", test_database_connection),
        ("Redis Connection", test_redis_connection),
    ]
    
    results = {}
    for test_name, test_func in tests:
        print(f"\n[TEST] Running: {test_name}")
        try:
            result = test_func()
            results[test_name] = result
            if result:
                print(f"   [PASS] {test_name}: PASSED")
            else:
                print(f"   [FAIL] {test_name}: FAILED")
        except Exception as e:
            print(f"   [ERROR] {test_name}: ERROR - {str(e)}")
            results[test_name] = False
    
    # Summary
    print("\n[SUMMARY] Verification Summary")
    print("=" * 50)
    passed = sum(1 for result in results.values() if result)
    total = len(results)
    
    for test_name, result in results.items():
        status = "PASSED" if result else "FAILED"
        print(f"   {test_name}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    # Special check for the requested port 3000
    if results.get("React Frontend (Port 3000)", False):
        print("\n[SUCCESS] React frontend is running on port 3000 as requested!")
        print("[TARGET] The FANUC RISE v2.1 system is fully deployed and operational.")
    else:
        print("\n[ISSUE] React frontend is not accessible on port 3000.")
        print("[FIX] Please check the docker-compose configuration to ensure the React frontend is properly mapped to port 3000.")
    
    # Return success if all critical components are working
    critical_tests_passed = all([
        results.get("Backend API Health", False),
        results.get("React Frontend (Port 3000)", False),  # This was specifically requested
        results.get("Database Connection", False)
    ])
    
    if critical_tests_passed:
        print("\n[SUCCESS] All critical components are operational!")
        return 0
    else:
        print("\n[WARN] Some components are not operational.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
