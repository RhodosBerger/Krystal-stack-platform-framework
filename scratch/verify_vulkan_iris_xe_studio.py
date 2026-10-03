"""
Verification Script for Vulkan Iris Xe Custom Engine Studio & Endpoints
========================================================================
Runs timeout-guarded checks against localhost:8089 for:
1. Static HTML page serving (vulkan_iris_xe_custom_engine_studio.html).
2. Telemetry endpoint (GET /api/vulkan-iris-xe/telemetry).
3. Dispatch simulation (POST /api/vulkan-iris-xe/simulate-dispatch).
4. Whisperer tuning (POST /api/vulkan-iris-xe/tune-whisperer).
5. Fast-swap trigger (POST /api/vulkan-iris-xe/trigger-swap).
6. Strict 6 Max HP Vital Invariant verification.
"""

import json
import urllib.request
import urllib.error

BASE_URL = "http://127.0.0.1:8089"


def test_endpoint(name, method="GET", path="", payload=None):
    url = f"{BASE_URL}{path}"
    headers = {"Content-Type": "application/json"} if payload else {}
    data = json.dumps(payload).encode("utf-8") if payload else None

    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=5.0) as resp:
            content = resp.read()
            status = resp.status
            if "application/json" in resp.headers.get("Content-Type", ""):
                parsed = json.loads(content.decode("utf-8"))
            elif "image/" in resp.headers.get("Content-Type", ""):
                parsed = content
            else:
                parsed = content.decode("utf-8")
            print(f"[PASS] {name} -> Status: {status}")
            return parsed
    except Exception as e:
        print(f"[FAIL] {name} -> Error: {e}")
        return None


def main():
    print("=== VERIFYING VULKAN IRIS XE STUDIO & API ENDPOINTS ===")
    
    # 1. Static HTML
    html = test_endpoint("Serve Studio HTML", "GET", "/static/vulkan_iris_xe_custom_engine_studio.html")
    assert html is not None and "Krystal-Vulkan Custom Engine" in html, "HTML content mismatch"

    # 2. Studio Image Asset
    img = test_endpoint("Serve Studio Image", "GET", "/static/img/vulkan_iris_xe_custom_engine.jpg")
    assert img is not None and len(img) > 1000, "Image asset missing"

    # 3. Telemetry GET
    telem = test_endpoint("Get Telemetry", "GET", "/api/vulkan-iris-xe/telemetry")
    assert telem is not None, "Telemetry failed"
    assert telem.get("vital_max_hp_rule") == 6, "Vital Max HP rule not 6"
    assert telem["gpu_hardware"]["total_execution_units"] == 96, "EUs mismatch"
    assert telem["cpu_whisperer"]["simd_mode"] == "AVX2 + FMA 256-bit", "SIMD mismatch"
    print(f"       Measured Stock FPS: {telem['gpu_hardware']['measured_stock_fps']} | Optimized: {telem['gpu_hardware']['krystal_optimized_fps']}")
    print(f"       WDDM Latency: {telem['gpu_hardware']['wddm_legacy_latency_us']} us -> Krystal Bypass: {telem['gpu_hardware']['krystal_bypass_latency_us']} us")

    # 4. Dispatch Simulation POST
    disp = test_endpoint("Simulate Dispatch", "POST", "/api/vulkan-iris-xe/simulate-dispatch", {
        "workload_chunks": 32,
        "eu_load_percent": 90.0
    })
    assert disp is not None and disp.get("status") == "DISPATCH_COMPLETED", "Dispatch failed"
    assert disp.get("vital_max_hp") == 6, "Vital Max HP in dispatch not 6"
    print(f"       Dispatch Duration: {disp['dispatch_duration_us']} us | Effective GFLOPS: {disp['effective_gflops']}")

    # 5. Whisperer Tuning POST
    tune = test_endpoint("Tune Whisperer", "POST", "/api/vulkan-iris-xe/tune-whisperer", {
        "enable_whisperer": True,
        "enable_zero_copy": True,
        "enable_npu": True,
        "enable_ssd_prefetch": True
    })
    assert tune is not None and tune.get("success") is True, "Tune whisperer failed"
    print(f"       Active Latency: {tune['active_latency_us']} us | Thread Switch: {tune['active_thread_switch_ns']} ns")

    # 6. Trigger Fast-Swap POST
    swap = test_endpoint("Trigger Fast-Swap", "POST", "/api/vulkan-iris-xe/trigger-swap", {
        "coordinates": [3, 2, 1]
    })
    assert swap is not None and swap.get("cache_hit") is True, "Fast-swap failed"
    assert swap.get("vital_max_hp") == 6, "Vital Max HP in swap not 6"
    print(f"       Sector: {swap['chunk_id']} | Prefetch: {swap['prefetch_time_ms']} ms | Frame Drop Risk: {swap['frame_drop_risk']}")

    print("=== ALL CHECKS PASSED SUCCESSFULLY ===")


if __name__ == "__main__":
    main()
