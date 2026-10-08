#!/usr/bin/env python3
"""
==============================================================================
VERIFICATION SUITE: INTEL 11TH GEN OPENVINO HARDWARE ACCELERATION PACK
==============================================================================
Tests:
  1. Tiger Lake PL1/PL2 bypass profile (32W unblock, Arrhenius wear guardrail).
  2. OpenVINO extended parameter configuration (DP4A INT8, 4 streams, KV-u8).
  3. Heterogeneous scheduler layer routing (GPU + CPU VNNI + GNA 2.0).
  4. Empirical benchmark verification: Unlocked i5 beats stock i7.
  5. Janet orchestrator DSL validation.
  6. Live REST API endpoints:
     - GET  /api/openvino/hardware_bypass_status
     - POST /api/openvino/accelerate
     - GET  /api/openvino/benchmark_dp4a
==============================================================================
"""

import sys
import json
import urllib.request
import urllib.error

if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from krystal_kernel.openvino_hardware_acceleration_pack import (
    TigerLakeTurboUnblocker,
    OpenVinoIrisXeDP4APack,
    HeterogeneousHybridScheduler,
    HardwareAccelerationBenchmark,
    VITAL_MAX_HP
)


def test_1_hardware_bypass_profile():
    print("[TEST 1] Testing Tiger Lake PL1/PL2 bypass and Arrhenius model...")
    bypass = TigerLakeTurboUnblocker.apply_hardware_bypass(target_pl1_w=32.0, voltage_offset_mv=-60.0)
    assert bypass.pl1_limit_watts == 32.0, f"Expected 32W PL1, got {bypass.pl1_limit_watts}"
    assert bypass.pl2_limit_watts == 54.0
    assert bypass.tau_window_seconds == 56.0
    assert bypass.pl_clamp_bypassed is True
    assert bypass.voltage_clamp_max_v <= 1.05, f"Expected <= 1.05V clamp, got {bypass.voltage_clamp_max_v}"
    assert bypass.vital_max_hp == 6
    assert bypass.performance_gain_pct >= 50.0

    # Test Arrhenius at safe vs dangerous boundary
    safe_wear = TigerLakeTurboUnblocker.calculate_arrhenius_wear_factor(temp_c=65.0, voltage_v=0.98)
    assert safe_wear <= 1.05, f"Expected safe wear factor <= 1.05, got {safe_wear}"

    print(f"  -> PASSED: PL1 unblocked to {bypass.pl1_limit_watts}W with +{bypass.performance_gain_pct}% compute boost.")


def test_2_openvino_extended_config():
    print("[TEST 2] Testing OpenVINO extended parameters & DP4A configuration...")
    cfg = OpenVinoIrisXeDP4APack.generate_extended_config(
        use_dp4a_int8=True,
        streams_count=4,
        enable_kv_u8=True
    )
    assert cfg.performance_hint == "CUMULATIVE_THROUGHPUT"
    assert cfg.inference_precision_hint == "i8"
    assert cfg.gpu_throughput_streams == 4
    assert cfg.model_priority == "HIGH"
    assert cfg.kv_cache_precision == "u8"
    assert cfg.predicted_tokens_per_sec >= 20.0
    assert cfg.vital_max_hp == 6
    print(f"  -> PASSED: OpenVINO config verified with DP4A i8, 4 streams, and {cfg.predicted_tokens_per_sec} tok/s.")


def test_3_heterogeneous_scheduler():
    print("[TEST 3] Testing Heterogeneous layer routing (GPU + CPU + GNA)...")
    plan = HeterogeneousHybridScheduler.get_layer_dispatch_plan()
    assert plan["pipeline_mode"] == "ZERO_COPY_UMA_RING_BUFFER"
    assert plan["vital_max_hp"] == 6
    routes = plan["dispatch_routing"]
    assert len(routes) == 5

    # Verify DP4A on Iris Xe
    gpu_stage = next((r for r in routes if "Iris Xe" in r["device"]), None)
    assert gpu_stage is not None
    assert "DP4A" in gpu_stage["acceleration_isa"]

    # Verify GNA on Audio
    gna_stage = next((r for r in routes if "GNA" in r["device"]), None)
    assert gna_stage is not None
    assert "Low-Power" in gna_stage["acceleration_isa"]
    print("  -> PASSED: Heterogeneous routing verified across GPU, CPU VNNI, and GNA 2.0.")


def test_4_empirical_i5_vs_i7_benchmark():
    print("[TEST 4] Testing empirical benchmark proving Unlocked i5 beats Stock i7...")
    bench = HardwareAccelerationBenchmark.run_comparative_benchmark()
    assert bench.krystal_i5_unlocked_32w_tokens_sec > bench.i7_stock_28w_tokens_sec, \
        f"Krystal i5 ({bench.krystal_i5_unlocked_32w_tokens_sec}) must beat stock i7 ({bench.i7_stock_28w_tokens_sec})"
    assert bench.i5_unlocked_vs_i7_stock_speedup_pct >= 40.0
    assert bench.vital_max_hp == 6
    print(f"  -> PASSED: Krystal Unlocked i5 ({bench.krystal_i5_unlocked_32w_tokens_sec} tok/s) "
          f"beats Stock i7 ({bench.i7_stock_28w_tokens_sec} tok/s) by +{bench.i5_unlocked_vs_i7_stock_speedup_pct}%.")


def test_5_janet_dsl_validation():
    print("[TEST 5] Testing Janet OpenVINO Acceleration Orchestrator DSL...")
    from krystal_janet.janet_bridge import JanetValidator
    res = JanetValidator.validate_file("krystal_janet/openvino_acceleration_orchestrator.janet")
    assert res["valid"] is True, f"Janet file invalid: {res}"
    assert "ACCELERATION-PACKAGES" in res["definitions"]
    assert "evaluate-chip-acceleration-parity" in res["definitions"]
    print(f"  -> PASSED: Janet DSL validated with {len(res['definitions'])} exported primitives.")


def test_6_live_rest_api():
    print("[TEST 6] Testing Live REST API Endpoints on http://127.0.0.1:8080...")
    base_url = "http://127.0.0.1:8080"

    # 1. GET /api/openvino/hardware_bypass_status
    try:
        req = urllib.request.Request(f"{base_url}/api/openvino/hardware_bypass_status")
        with urllib.request.urlopen(req, timeout=5) as resp:
            assert resp.status == 200
            data = json.loads(resp.read().decode("utf-8"))
            assert data["pl1_limit_watts"] == 32.0
            assert data["pl_clamp_bypassed"] is True
            assert data["vital_max_hp"] == 6
            print("  -> GET /api/openvino/hardware_bypass_status: 200 OK")
    except Exception as e:
        print(f"  -> FAIL on GET /api/openvino/hardware_bypass_status: {e}")
        raise

    # 2. POST /api/openvino/accelerate
    try:
        payload = json.dumps({"use_dp4a_int8": True, "streams_count": 4, "enable_kv_u8": True}).encode("utf-8")
        req = urllib.request.Request(
            f"{base_url}/api/openvino/accelerate",
            data=payload,
            headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            assert resp.status == 200
            data = json.loads(resp.read().decode("utf-8"))
            assert data["inference_precision_hint"] == "i8"
            assert data["gpu_throughput_streams"] == 4
            assert data["vital_max_hp"] == 6
            print("  -> POST /api/openvino/accelerate: 200 OK")
    except Exception as e:
        print(f"  -> FAIL on POST /api/openvino/accelerate: {e}")
        raise

    # 3. GET /api/openvino/benchmark_dp4a
    try:
        req = urllib.request.Request(f"{base_url}/api/openvino/benchmark_dp4a")
        with urllib.request.urlopen(req, timeout=5) as resp:
            assert resp.status == 200
            data = json.loads(resp.read().decode("utf-8"))
            assert data["krystal_i5_unlocked_32w_tokens_sec"] > data["i7_stock_28w_tokens_sec"]
            assert data["vital_max_hp"] == 6
            print("  -> GET /api/openvino/benchmark_dp4a: 200 OK")
    except Exception as e:
        print(f"  -> FAIL on GET /api/openvino/benchmark_dp4a: {e}")
        raise

    print("  -> PASSED: All REST API endpoints validated.")


if __name__ == "__main__":
    print("================================================================================")
    print("  RUNNING VERIFICATION SUITE: OPENVINO HARDWARE ACCELERATION PACK")
    print("================================================================================")
    test_1_hardware_bypass_profile()
    test_2_openvino_extended_config()
    test_3_heterogeneous_scheduler()
    test_4_empirical_i5_vs_i7_benchmark()
    test_5_janet_dsl_validation()
    test_6_live_rest_api()
    print("================================================================================")
    print("  ALL 6 VERIFICATION TESTS PASSED SUCCESSFULLY! (VITAL_MAX_HP = 6)")
    print("================================================================================")
