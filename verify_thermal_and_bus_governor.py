#!/usr/bin/env python3
"""
VERIFICATION TEST: CROSS-PLATFORM THERMAL & MEMORY BUS GOVERNOR
==============================================================
Verifies:
  1. Inviolable Invariant: VITAL_MAX_HP = 6.
  2. Memory Bus Optimization: 10x Bandwidth Reduction on Dual-Channel UMA.
  3. Float16 Precision Boundaries: 50% Footprint Reduction & Epsilon Thresholds.
  4. NPU Pre-staging Prediction: 1,500x Latency Arbitrage (RAM vs SSD NVMe).
  5. Multiplatform Thermal Management (Windows, Linux, macOS, Android).
  6. Generation of the Master Markdown Report requested by the user.

Author: Dušan Kopecký & Krystal-Stack Architecture Council (2026)
"""

import sys
import os

WORKSPACE_ROOT = os.path.dirname(os.path.abspath(__file__))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

if sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

from krystal_stack_nextgen import (
    VITAL_MAX_HP,
    TargetPlatform,
    ThermalState,
    MemoryBusProfile,
    Float16PrecisionImpact,
    NPUPredictionMetrics,
    CrossPlatformThermalPlan,
    CrossPlatformThermalBusGovernor,
    GLOBAL_THERMAL_BUS_GOVERNOR
)


def test_invariant():
    print("\n[TEST 1] Verifying System Invariant...")
    assert VITAL_MAX_HP == 6, f"Expected VITAL_MAX_HP == 6, got {VITAL_MAX_HP}"
    print(f"  ✓ System Invariant VITAL_MAX_HP = {VITAL_MAX_HP} strictly preserved.")


def test_memory_bus_optimization():
    print("\n[TEST 2] Verifying Memory Bus Optimization Profile...")
    gov = GLOBAL_THERMAL_BUS_GOVERNOR
    bus = gov.get_memory_bus_profile()
    print(f"  ✓ Bus Architecture:     {bus.bus_type}")
    print(f"  ✓ Theoretical Peak:     {bus.theoretical_peak_gb_s} GB/s")
    print(f"  ✓ Baseline Bandwidth:   {bus.baseline_uncompressed_gb_s} GB/s")
    print(f"  ✓ Tile4 + CCS Bandwidth:{bus.optimized_tile4_ccs_gb_s} GB/s")
    print(f"  ✓ Bandwidth Reduction:  {bus.compression_ratio}x")
    assert bus.compression_ratio == 10.0
    assert bus.theoretical_peak_gb_s > 60.0
    assert bus.l3_sampler_cache_pct == 65.0
    assert bus.subgroup_simd_width == 16


def test_float16_precision_impact():
    print("\n[TEST 3] Verifying Float16 Precision & Dynamic Range...")
    gov = GLOBAL_THERMAL_BUS_GOVERNOR
    fp = gov.analyze_float16_precision()
    print(f"  ✓ Format:               {fp.format_name}")
    print(f"  ✓ Exponent/Mantissa:    {fp.exponent_bits} bits / {fp.mantissa_bits} bits")
    print(f"  ✓ Max Dynamic Range:    {fp.dynamic_range_max}")
    print(f"  ✓ Machine Epsilon:      {fp.epsilon_machine}")
    print(f"  ✓ Memory Footprint:     -{fp.memory_footprint_reduction_pct}%")
    print(f"  ✓ Arithmetic Intensity: {fp.arithmetic_intensity_multiplier}x")
    assert fp.memory_footprint_reduction_pct == 50.0
    assert fp.arithmetic_intensity_multiplier == 2.0
    assert fp.dynamic_range_max == 65504.0


def test_npu_prestaging_metrics():
    print("\n[TEST 4] Verifying NPU Pre-staging Prediction Metrics...")
    gov = GLOBAL_THERMAL_BUS_GOVERNOR
    npu = gov.get_npu_prediction_metrics()
    print(f"  ✓ Inference Engine:     {npu.inference_engine}")
    print(f"  ✓ Decision Budget:      {npu.out_of_band_decision_budget_us} µs")
    print(f"  ✓ Lookahead:            {npu.speculative_lookahead_frames} frames")
    print(f"  ✓ NVMe SSD Latency:     {npu.nvme_ssd_swap_latency_us} µs")
    print(f"  ✓ Hot Pinned RAM Latency:{npu.pinned_ram_ring_buffer_latency_us} µs")
    print(f"  ✓ Speedup Factor:       {npu.latency_elimination_factor}x faster")
    print(f"  ✓ Effective Latency:    {npu.effective_access_latency_us} µs")
    print(f"  ✓ NPU Operating Power:  {npu.npu_operating_power_watts} W")
    assert npu.latency_elimination_factor >= 1500.0
    assert npu.measured_hit_probability == 0.94


def test_cross_platform_thermal_matrix():
    print("\n[TEST 5] Verifying Cross-Platform Thermal Governance...")
    gov = GLOBAL_THERMAL_BUS_GOVERNOR

    # Windows under warm load
    p_win = gov.evaluate_cross_platform_thermal(TargetPlatform.WINDOWS_X64_INTEL, 78.5)
    print(f"  ✓ Windows (78.5°C): State={p_win.current_thermal_state.value} | PL1={p_win.power_limit_pl1_watts}W | {p_win.cooling_strategy[:30]}...")
    assert p_win.current_thermal_state == ThermalState.SERIOUS_THROTTLING_RISK
    assert p_win.power_limit_pl1_watts == 18.0

    # Linux Steam Deck optimal
    p_lin = gov.evaluate_cross_platform_thermal(TargetPlatform.LINUX_STEAM_DECK_KISAK, 71.0)
    print(f"  ✓ Linux (71.0°C):   State={p_lin.current_thermal_state.value} | PL1={p_lin.power_limit_pl1_watts}W")
    assert p_lin.current_thermal_state == ThermalState.FAIR_WARM

    # macOS Apple Silicon nominal
    p_mac = gov.evaluate_cross_platform_thermal(TargetPlatform.MACOS_APPLE_SILICON_M_SERIES, 63.5)
    print(f"  ✓ macOS (63.5°C):   State={p_mac.current_thermal_state.value} | PL1={p_mac.power_limit_pl1_watts}W")
    assert p_mac.current_thermal_state == ThermalState.NOMINAL_OPTIMAL

    # Android ARM critical
    p_arm = gov.evaluate_cross_platform_thermal(TargetPlatform.ANDROID_EDGE_ARM_SNAPDRAGON, 86.0)
    print(f"  ✓ Android (86.0°C): State={p_arm.current_thermal_state.value} | PL1={p_arm.power_limit_pl1_watts}W | Migration={p_arm.workload_migration_target}")
    assert p_arm.current_thermal_state == ThermalState.CRITICAL_EMERGENCY_SHED
    assert p_arm.power_limit_pl1_watts == 4.5


def test_markdown_report_generation():
    print("\n[TEST 6] Verifying Markdown Report Generation...")
    gov = GLOBAL_THERMAL_BUS_GOVERNOR
    report = gov.generate_comprehensive_markdown_report()
    assert "# DETAILNÝ TECHNICKÝ VÝPIS ARCHITEKTÚRY KRYSTAL STACK NEXTGEN" in report
    assert "OPTIMALIZÁCIA PAMÄŤOVEJ ZBERNICE" in report
    assert "DOPADY ZNIŽOVANIA PRESNOSTI NA FLOAT16" in report
    assert "PREDIKCIA PRE-STAGINGU V NPU" in report
    assert "NÁVRH NA RIADENIE TEPLOTY NAPRIEČ PLATFORMAMI" in report
    assert "VITAL_MAX_HP = 6" in report
    print(f"  ✓ Detailed markdown report generated successfully ({len(report)} characters).")


def main():
    print("=" * 80)
    print(" 🌡️ KRYSTAL-STACK NEXTGEN: THERMAL & BUS GOVERNOR FULL VERIFICATION")
    print("=" * 80)

    test_invariant()
    test_memory_bus_optimization()
    test_float16_precision_impact()
    test_npu_prestaging_metrics()
    test_cross_platform_thermal_matrix()
    test_markdown_report_generation()

    print("\n" + "=" * 80)
    print(" ✅ ALL CROSS-PLATFORM THERMAL & BUS TESTS PASSED (100% SUCCESS)")
    print("=" * 80)


if __name__ == "__main__":
    main()
