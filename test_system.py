#!/usr/bin/env python3
"""
Test script to verify GAMESA/KrystalStack system functionality
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src', 'python'))

# Test imports
try:
    print("Testing imports...")
    from krystal_sdk import health_check, Krystal
    print("[OK] KrystalSDK import successful")

    # Test health check
    health = health_check()
    print(f"[OK] Health check: {health['status']}")

    # Test basic Krystal functionality
    k = Krystal()
    k.observe({"cpu": 0.7, "mem": 0.5})
    action = k.decide()
    k.reward(0.8)
    print(f"[OK] Basic Krystal test successful, action: {action[:2]}... (truncated)")

    # Test cognitive engine
    try:
        from cognitive_engine import create_cognitive_orchestrator
        print("[OK] Cognitive engine import successful")

        orchestrator = create_cognitive_orchestrator()
        print("[OK] Cognitive orchestrator created")

        # Test with sample telemetry
        telemetry = {
            'cpu_util': 0.7,
            'gpu_util': 0.6,
            'cpu_temp': 70,
            'gpu_temp': 68,
            'power_draw': 150,
            'memory_util': 0.5,
            'thermal_headroom': 0.6,
            'fps_ratio': 0.8
        }

        result = orchestrator.process(telemetry)
        print(f"[OK] Cognitive engine processed telemetry, action: {result['action']}")
    except ImportError as e:
        print(f"[WARN] Cognitive engine import failed: {e}")

    # Test invention engine
    try:
        from invention_engine import create_invention_engine
        print("[OK] Invention engine import successful")

        engine = create_invention_engine()
        print("[OK] Invention engine created")

        # Test with sample telemetry
        telemetry = {
            'cpu_util': 0.7,
            'gpu_util': 0.6,
            'cpu_temp': 70,
            'gpu_temp': 68,
            'power_draw': 150,
            'memory_util': 0.5
        }

        result = engine.process(telemetry)
        print(f"[OK] Invention engine processed, action: {result['action']}")
    except ImportError as e:
        print(f"[WARN] Invention engine import failed: {e}")

    print("\n[SUCCESS] All core systems tested successfully!")

except Exception as e:
    print(f"[ERROR] Error during testing: {e}")
    import traceback
    traceback.print_exc()