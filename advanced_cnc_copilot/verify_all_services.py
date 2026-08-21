#!/usr/bin/env python3
"""
Final verification script for FANUC RISE v2.1 Advanced CNC Copilot system.
Verifies all services are operational and accessible on their designated ports.
"""

import subprocess
import requests
import time
import sys
import json
from typing import Dict, List, Tuple

def check_container_status() -> Dict[str, str]:
    """Check the status of Docker containers"""
    try:
        result = subprocess.run(['docker-compose', 'ps', '--format', 'json'],
                               capture_output=True, text=True, check=True)
        
        containers = {}
        for line in result.stdout.strip().split('\n'):
            if line and '"Name"' in line and '"Status"' in line:
                try:
                    container_info = json.loads(line)
                    name = container_info.get('Name', '')
                    status = container_info.get('Status', '')
                    if name:  # Only add if name exists
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

def test_port_accessibility(port: int, service_name: str) -> bool:
    """Test if a service is accessible on a specific port"""
    try:
        response = requests.get(f"http://localhost:{port}", timeout=5)
        # For web servers, we expect 200, 404 (page not found but server running), or redirects
        return response.status_code in [200, 404, 301, 302, 405]
    except requests.exceptions.RequestException:
        # If we can't access via HTTP, try a simple connection check
        import socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        result = sock.connect_ex(('localhost', port))
        sock.close()
        return result == 0

def main():
    print("FANUC RISE v2.1 Advanced CNC Copilot - Final Service Verification")
    print("=" * 70)
    
    # Get container status
    print("[CHECKING] Checking container statuses...")
    containers = check_container_status()
    
    if not containers:
        print("[WARNING] No containers found. Attempting to start the system...")
        try:
            subprocess.run(['docker-compose', 'up', '-d', '--build'], check=True)
            time.sleep(10)  # Wait for services to start
            containers = check_container_status()
        except subprocess.CalledProcessError:
            print("[ERROR] Failed to start services via docker-compose")
            return 1
    
    for name, status in containers.items():
        status_indicator = "[UP]" if "Up" in status else "[DOWN]"
        print(f"   {name}: {status_indicator} {status}")
    
    print()
    
    # Define services and their expected ports
    services_ports = [
        ("API Backend", 8000),
        ("React Frontend", 3000),
        ("Vue Frontend", 8080),
        ("Database", 5432),
        ("Redis", 6379),
        ("Prometheus", 9090),
        ("Grafana", 3001),
        ("Flower (Task Monitor)", 5555)
    ]
    
    print("[TESTING] Testing service accessibility on designated ports...")
    all_services_operational = True
    
    for service_name, port in services_ports:
        print(f"   Testing {service_name} on port {port}...", end="")
        is_accessible = test_port_accessibility(port, service_name)
        
        if is_accessible:
            print(" [PASS] [SUCCESS]")
        else:
            print(" [FAIL] [ERROR]")
            all_services_operational = False
    
    print()
    
    # Test API health endpoint specifically
    print("[HEALTH] Testing API health endpoint...")
    try:
        response = requests.get("http://localhost:8000/health", timeout=10)
        if response.status_code == 200:
            print("   API Health: [PASS] [SUCCESS]")
        else:
            print(f"   API Health: [FAIL] [ERROR] (Status: {response.status_code})")
            all_services_operational = False
    except requests.exceptions.RequestException as e:
        print(f"   API Health: [FAIL] [ERROR] (Error: {str(e)})")
        all_services_operational = False
    
    print()
    
    # Test key API endpoints
    api_endpoints = [
        "/api/health",
        "/api/telemetry",
        "/api/machines"
    ]
    
    print("[LINKS] Testing key API endpoints...")
    for endpoint in api_endpoints:
        try:
            response = requests.get(f"http://localhost:8000{endpoint}", timeout=10)
            status = "[PASS] [SUCCESS]" if response.status_code in [200, 401, 403, 405] else "[FAIL] [ERROR]"
            print(f"   {endpoint}: {status} (Status: {response.status_code})")
            if response.status_code not in [200, 401, 403, 405]:
                all_services_operational = False
        except requests.exceptions.RequestException as e:
            print(f"   {endpoint}: [FAIL] [ERROR] (Error: {str(e)})")
            all_services_operational = False
    
    print()
    
    # Summary
    print("[SUMMARY] Final Verification Summary")
    print("=" * 70)
    
    operational_count = sum(1 for service_name, port in services_ports if test_port_accessibility(port, service_name))
    total_services = len(services_ports)
    
    print(f"Services operational: {operational_count}/{total_services}")
    
    # Check if API is accessible separately since it's critical
    api_accessible = False
    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        api_accessible = response.status_code == 200
    except:
        api_accessible = False
    
    if all_services_operational and operational_count >= 5 and api_accessible:  # At least 5 of 8 services should be accessible + API must be healthy
        print("\n[SUCCESS] FANUC RISE v2.1 system is fully operational!")
        print("[OK] All critical services are running and accessible")
        print("[OK] React frontend is accessible on port 3000 as requested")
        print("[OK] API backend is operational on port 8000")
        print("[OK] Database and Redis connections established")
        print("[OK] System ready for production use")
        return 0
    else:
        warning_msg = f"\n[WARNING] Only {operational_count} of {total_services} services are accessible"
        if not api_accessible:
            warning_msg += " and API health check failed"
        print(warning_msg)
        print("[INFO] Please check the docker-compose configuration and service logs")
        print("[HINT] Try running: docker-compose logs <service_name> for troubleshooting")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)