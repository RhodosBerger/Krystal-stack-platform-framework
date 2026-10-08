#!/usr/bin/env python3
"""
verify_godot_render_and_llm_vram_governor.py
=============================================
Comprehensive automated verification suite for the Godot 4.x Render Engine Integration,
Intel Iris Xe VRAM Aperture Unlocker, Local Quantized LLM Token Benchmark,
K-ISA Speculative Instruction Set, and Arrhenius Silicon Lifespan Model.

Enforces system invariant: VITAL_MAX_HP == 6.
"""

import sys
import os
from pathlib import Path

# Ensure repo root is on path
REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

def test_1_iris_xe_vram_unlocker():
    print("[TEST 1] Intel Iris Xe UMA VRAM Aperture Unlocker...")
    from krystal_kernel.iris_xe_llm_vram_governor import (
        GLOBAL_IRIS_XE_LLM_VRAM_GOVERNOR,
        VramApertureTier,
        VITAL_MAX_HP
    )
    
    gov = GLOBAL_IRIS_XE_LLM_VRAM_GOVERNOR
    assert VITAL_MAX_HP == 6, f"Invariant violated: VITAL_MAX_HP is {VITAL_MAX_HP}"
    
    # 1. Check default clamped state
    res_clamped = gov.unlock_vram_aperture(VramApertureTier.TIER_LEGACY_CLAMPED_128MB)
    assert res_clamped["wddm_clamp_bypassed"] is False
    assert res_clamped["aperture_mb"] == 128
    
    # 2. Unlock 4GB tier
    res_4gb = gov.unlock_vram_aperture(VramApertureTier.TIER_UNLOCKED_4GB)
    assert res_4gb["unlocked"] is True
    assert res_4gb["wddm_clamp_bypassed"] is True
    assert res_4gb["direct_uma_coherence"] is True
    assert res_4gb["aperture_mb"] == 4096
    assert res_4gb["vital_max_hp"] == 6
    assert "INT4" in res_4gb["max_supported_llm_parameters"]
    
    # 3. Unlock 8GB tier
    res_8gb = gov.unlock_vram_aperture(VramApertureTier.TIER_UNLOCKED_8GB)
    assert res_8gb["aperture_mb"] == 8192
    assert res_8gb["wddm_clamp_bypassed"] is True
    
    print("  -> PASS: WDDM 128MB limit successfully bypassed with host-coherent UMA tiers (4GB & 8GB).")

def test_2_local_quantized_llm_benchmark():
    print("[TEST 2] Local Quantized LLM Token Throughput Benchmark...")
    from krystal_kernel.iris_xe_llm_vram_governor import (
        GLOBAL_IRIS_XE_LLM_VRAM_GOVERNOR,
        VramApertureTier,
        QuantizationFormat
    )
    
    gov = GLOBAL_IRIS_XE_LLM_VRAM_GOVERNOR
    
    # Baseline: 128MB clamped tier
    bench_128 = gov.benchmark_local_llm_throughput(
        VramApertureTier.TIER_LEGACY_CLAMPED_128MB,
        QuantizationFormat.INT4_GGUF_AWQ
    )
    assert bench_128.tokens_per_second <= 10.0, f"Expected low throughput under clamp, got {bench_128.tokens_per_second}"
    assert bench_128.ring_bus_stalls_per_sec > 1000.0, "Expected ring bus thrashing under 128MB clamp"
    assert bench_128.speedup_vs_clamped == 1.0
    
    # Unlocked: 4GB tier (Zero-Copy UMA)
    bench_4gb = gov.benchmark_local_llm_throughput(
        VramApertureTier.TIER_UNLOCKED_4GB,
        QuantizationFormat.INT4_GGUF_AWQ
    )
    assert bench_4gb.tokens_per_second >= 40.0, f"Expected >=40 tok/s, got {bench_4gb.tokens_per_second}"
    assert bench_4gb.speedup_vs_clamped >= 5.0, f"Expected >=5.0x speedup, got {bench_4gb.speedup_vs_clamped}x"
    assert bench_4gb.ring_bus_stalls_per_sec == 0.0, "Expected 0 ring bus stalls with direct UMA pool"
    assert bench_4gb.thermal_throttling_prevented is True
    assert bench_4gb.token_latency_ms < 25.0
    
    print(f"  -> PASS: 4GB UMA unlocked delivers {bench_4gb.tokens_per_second} tok/s ({bench_4gb.speedup_vs_clamped}x speedup vs {bench_128.tokens_per_second} tok/s clamped).")

def test_3_k_isa_speculative_instructions():
    print("[TEST 3] K-ISA Speculative Instruction Pipeline & Latency Hiding...")
    from krystal_kernel.speculative_instruction_set import (
        GLOBAL_SPECULATIVE_PREDICTOR_ENGINE,
        KrystalInstructionOpcode,
        VITAL_MAX_HP
    )
    
    eng = GLOBAL_SPECULATIVE_PREDICTOR_ENGINE
    assert VITAL_MAX_HP == 6
    
    rep = eng.evaluate_and_dispatch_speculation(
        predicted_stall_cycles=1200,
        speculation_confidence=0.95
    )
    
    assert rep.frame_drop_prevented is True
    assert rep.stall_cycles_masked >= 1000
    assert rep.latency_hidden_ms >= 10.0
    assert rep.safety_voltage_clamp_v <= 1.05
    assert rep.pipeline_integrity_preserved is True
    assert rep.vital_max_hp == 6
    
    opcodes = [inst.opcode for inst in rep.instructions_dispatched]
    assert KrystalInstructionOpcode.K_SPEC_PREFETCH_UMA in opcodes
    assert KrystalInstructionOpcode.K_SPEC_INTERPOLATE_FRAME in opcodes
    assert KrystalInstructionOpcode.K_SAFE_VOLT_CLAMP in opcodes
    assert KrystalInstructionOpcode.K_VERIFY_INVARIANT_HP in opcodes
    
    print(f"  -> PASS: K-ISA speculation masked {rep.stall_cycles_masked} cycles, hiding {rep.latency_hidden_ms} ms stall latency at {rep.safety_voltage_clamp_v}V.")

def test_4_optimal_process_calculator_and_arrhenius():
    print("[TEST 4] Optimal Process Calculator & Arrhenius Lifespan Model...")
    from krystal_kernel.iris_xe_llm_vram_governor import OptimalProcessCalculator
    
    calc = OptimalProcessCalculator()
    
    # 1. Standard optimal plan
    plan = calc.synthesize_optimal_plan(demand_high_throughput=True, max_safe_temp_c=85.0)
    assert plan.safe_to_operate is True
    assert plan.effective_chip_lifespan_years >= 9.8, f"Expected ~10 years lifespan, got {plan.effective_chip_lifespan_years}"
    assert plan.optimal_voltage_v <= 1.05
    assert plan.projected_junction_temp_c < 80.0
    assert plan.thermal_headroom_c > 15.0
    assert len(plan.governor_prescriptions) >= 3
    
    # 2. Extreme over-voltage / thermal stress test
    stress_life = calc.calculate_arrhenius_lifespan_factor(voltage_v=1.28, temp_celsius=98.0)
    assert stress_life < 0.35, f"Expected severe lifespan degradation under 1.28V / 98C, got {stress_life}"
    
    print(f"  -> PASS: Voltage clamped to {plan.optimal_voltage_v}V, preserving {plan.effective_chip_lifespan_years} years nominal MTTF.")

def test_5_godot_integration_and_shader():
    print("[TEST 5] Godot 4.x GDScript Integration & GDShader Package...")
    gdscript_path = REPO_ROOT / "godot_project" / "scripts" / "KrystalRenderEngineIntegration.gd"
    gdshader_path = REPO_ROOT / "godot_project" / "shaders" / "krystal_knss_godot_viewport.gdshader"
    bridge_h_path = REPO_ROOT / "include" / "krystal_godot_llm_vram_bridge.h"
    
    assert gdscript_path.exists(), f"Missing {gdscript_path}"
    assert gdshader_path.exists(), f"Missing {gdshader_path}"
    assert bridge_h_path.exists(), f"Missing {bridge_h_path}"
    
    gdscript_content = gdscript_path.read_text(encoding="utf-8")
    assert "class_name KrystalRenderEngineIntegration" in gdscript_content
    assert "VITAL_MAX_HP: int = 6" in gdscript_content
    assert "http://127.0.0.1:8080" in gdscript_content
    assert "request_llm_vram_unlock" in gdscript_content
    assert "trigger_kisa_speculation" in gdscript_content
    
    gdshader_content = gdshader_path.read_text(encoding="utf-8")
    assert "shader_type canvas_item;" in gdshader_content
    assert "rgb_to_ycocg" in gdshader_content
    assert "history_texture" in gdshader_content
    assert "clamp(history_ycocg" in gdshader_content
    
    bridge_h_content = bridge_h_path.read_text(encoding="utf-8")
    assert "KRYSTAL_VITAL_MAX_HP 6" in bridge_h_content
    assert "krystal_calculate_arrhenius_lifespan" in bridge_h_content
    assert "krystal_resolve_token_speedup" in bridge_h_content
    
    print("  -> PASS: Godot 4.x GDScript, GDShader, and C/C++ bridge verified and cross-referenced.")

def test_6_openapi_endpoints_and_server():
    print("[TEST 6] OpenAPI 3.1.0 Specification & REST Endpoints...")
    from krystal_kernel.openapi_spec import generate_openapi_spec
    
    spec = generate_openapi_spec()
    paths = spec.get("paths", {})
    schemas = spec.get("components", {}).get("schemas", {})
    
    required_paths = [
        "/api/llm/vram_unlock",
        "/api/llm/benchmark_tokens",
        "/api/llm/optimal_plan",
        "/api/isa/speculate",
        "/api/godot/render_package"
    ]
    for p in required_paths:
        assert p in paths, f"Path {p} not found in OpenAPI spec!"
        
    required_schemas = [
        "VramUnlockRequest",
        "VramUnlockResponse",
        "LlmTokenBenchmarkRequest",
        "LlmTokenBenchmarkResult",
        "OptimalProcessPlan",
        "KIsaSpeculationReport",
        "GodotRenderPackageResponse"
    ]
    for s in required_schemas:
        assert s in schemas, f"Schema {s} not found in OpenAPI components!"
        
    assert len(paths) >= 25, f"Expected >=25 endpoints, got {len(paths)}"
    assert len(schemas) >= 35, f"Expected >=35 schemas, got {len(schemas)}"
    
    print(f"  -> PASS: OpenAPI 3.1.0 validated with {len(paths)} endpoints and {len(schemas)} schemas.")

def main():
    print("================================================================================")
    print("  KRYSTAL STACK: GODOT 4.X RENDER ENGINE & IRIS XE LLM VRAM VERIFICATION SUITE")
    print("================================================================================")
    
    test_1_iris_xe_vram_unlocker()
    test_2_local_quantized_llm_benchmark()
    test_3_k_isa_speculative_instructions()
    test_4_optimal_process_calculator_and_arrhenius()
    test_5_godot_integration_and_shader()
    test_6_openapi_endpoints_and_server()
    
    print("================================================================================")
    print("  ALL 6 TESTS PASSED SUCCESSFULLY! (100% SUCCESS, VITAL_MAX_HP == 6)")
    print("================================================================================")

if __name__ == "__main__":
    main()
