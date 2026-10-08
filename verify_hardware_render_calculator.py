#!/usr/bin/env python3
# ==============================================================================
# VERIFICATION SUITE: HARDWARE RENDER BUDGET CALCULATOR & GODOT SYNTHESIZER
# ==============================================================================
# Validates:
#   1. Non-negotiable Invariant: VITAL_MAX_HP == 6
#   2. Hardware Cache & Memory Probing (GPU L2 3.84MB, VRAM, CPU L1/L2/L3, RAM)
#   3. Engine Function Aggregation into Real Render Functions
#   4. Scene Diversity Maximization & GPU L2 Working Set Bounds
#   5. Adaptive RAM Ring Buffer vs SSD Swapping (Low Quota vs Burst Eviction)
#   6. Structured SSD Log Writing & Readback Loop
#   7. Closed-Loop Janet-to-Godot 4.x Photorealistic Shader & Stage Generation
# ==============================================================================

import os
import sys
import json
import time

if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

def test_hardware_render_calculator_and_synthesis():
    print("=" * 70)
    print(" 💎 KRYSTAL-STACK: HARDWARE RENDER BUDGET & GODOT SYNTHESIS VERIFICATION")
    print("=" * 70)

    # 1. Invariant Assertion
    from krystal_kernel import VITAL_MAX_HP
    assert VITAL_MAX_HP == 6, f"VIOLATION: VITAL_MAX_HP is {VITAL_MAX_HP}, must be 6!"
    print(f"\n[1/7] Invariant Verified: VITAL_MAX_HP = {VITAL_MAX_HP}")

    # 2. Hardware Topology Probing
    from krystal_kernel.hardware_render_calculator import (
        probe_hardware_topology,
        HardwareRenderBudgetCalculator,
        RenderFunctionRegistry,
        AdaptiveSwapManager,
        SwapQuotaMode,
        LogDrivenJanetGodotSynthesizer
    )

    hw = probe_hardware_topology()
    print(f"\n[2/7] Hardware Profile Probed:")
    print(f"      GPU:             {hw.gpu_name} (VRAM: {hw.gpu_vram_mb} MB)")
    print(f"      GPU L2 Cache:    {hw.gpu_l2_cache_kb:.1f} KB (3.84 MB Iris Xe / dGPU class)")
    print(f"      CPU:             {hw.cpu_name}")
    print(f"      CPU L1/L2/L3:    L1D={hw.cpu_l1_data_cache_kb:.0f}KB, L2={hw.cpu_l2_cache_kb:.0f}KB, L3={hw.cpu_l3_llc_kb:.0f}KB")
    print(f"      Host System RAM: Total={hw.total_ram_gb:.1f}GB, Avail={hw.avail_ram_gb:.1f}GB, Load={hw.ram_load_percent:.1f}%")
    assert hw.gpu_l2_cache_kb > 1000, "GPU L2 Cache must be modeled (>1MB)"
    assert hw.total_ram_gb > 0, "System RAM must be probed"

    # 3. Engine Function Aggregation
    registry = RenderFunctionRegistry.get_aggregated_functions()
    print(f"\n[3/7] Aggregated Engine Render Functions ({len(registry)} functions):")
    expected_fns = [
        "terrain_multioctave", "sdf_chalice", "sdf_athame", "urban_extrude_spire",
        "ballistics_mortar", "coxeter_dihedral_reflections", "bayer_dither_sample",
        "bullet_time_dilation"
    ]
    for fn_id in expected_fns:
        assert fn_id in registry, f"Missing aggregated function: {fn_id}"
        desc = registry[fn_id]
        print(f"      • [{desc.category}] {desc.display_name} -> L2 Working Set: {desc.gpu_l2_working_set_kb:.0f} KB, Diversity: +{desc.diversity_contribution}")

    # 4. Hardware Render Budget & Scene Diversity Maximization
    calc = HardwareRenderBudgetCalculator(profile=hw)

    # Test Balanced Plan
    plan_balanced = calc.compute_render_budget(target_fps=60, quality_preference="BALANCED")
    print(f"\n[4/7] Render Budget Plan (Balanced):")
    print(f"      Target FPS:           {plan_balanced.target_fps} (Frame Budget: {plan_balanced.frame_budget_ms} ms)")
    print(f"      Scene Diversity Score:{plan_balanced.scene_diversity_score} / 100.0")
    print(f"      GPU L2 Footprint:     {plan_balanced.total_gpu_l2_footprint_kb:.1f} KB ({plan_balanced.gpu_l2_utilization_percent}% of L2)")
    print(f"      Raymarch Max Steps:   {plan_balanced.max_raymarching_steps} (Epsilon: {plan_balanced.raymarching_epsilon})")
    print(f"      Coxeter D_N Folds:    {plan_balanced.coxeter_folds}")
    print(f"      Bayer Dither Matrix:  {plan_balanced.bayer_matrix_size}x{plan_balanced.bayer_matrix_size}")
    print(f"      Advice:               {plan_balanced.hardware_advice}")
    assert plan_balanced.vital_max_hp == 6
    assert plan_balanced.scene_diversity_score > 50.0
    assert not plan_balanced.l2_cache_budget_exceeded, "Balanced plan should fit within GPU L2 cache"

    # Test Cache-Constrained Plan with heavy load
    all_fns = list(registry.keys())
    plan_heavy = calc.compute_render_budget(requested_functions=all_fns, target_fps=120, quality_preference="MINIMAL")
    print(f"      Minimal Cache Plan:   Steps={plan_heavy.max_raymarching_steps}, Folds={plan_heavy.coxeter_folds}, L2 Util={plan_heavy.gpu_l2_utilization_percent}%")

    # 5. Adaptive RAM Ring Buffer vs SSD Swapping
    from krystal_kernel import GLOBAL_PROCESSOR_WHISPERER
    test_ssd_path = os.path.join("logs", "test_verification_structured_audit.jsonl")
    if os.path.exists(test_ssd_path):
        os.remove(test_ssd_path)

    swap_mgr = AdaptiveSwapManager(whisperer=GLOBAL_PROCESSOR_WHISPERER, structured_log_path=test_ssd_path)

    # Populate ring buffer with sample pulses
    for i in range(12):
        GLOBAL_PROCESSOR_WHISPERER.record_hardware_pulse(
            pc=0x1000 + (i * 0x10),
            is_current_switch=(i % 4 == 0),
            is_l1_write=(i % 2 == 0),
            is_l3_write=(i % 3 == 0),
            is_dram_spill=(i % 7 == 0)
        )
        GLOBAL_PROCESSOR_WHISPERER.whisper_instruction_hint(pc=0x1000 + (i * 0x10))

    # Evaluate Adaptive Quota
    swap_telem = swap_mgr.evaluate_adaptive_quota()
    print(f"\n[5/7] Adaptive Swap Telemetry:")
    print(f"      Mode:                 {swap_telem.mode}")
    print(f"      RAM Ring Utilization: {swap_telem.ram_ring_utilization_pct}% ({swap_telem.buffered_count}/{swap_telem.ring_capacity})")
    print(f"      System Free RAM:      {swap_telem.system_free_ram_gb:.2f} GB")
    print(f"      Current Flush Quota:  {swap_telem.current_flush_quota} entries/batch")
    print(f"      Summary:              {swap_telem.status_summary}")
    assert swap_telem.mode in [SwapQuotaMode.LOW_QUOTA_STRUCTURED, SwapQuotaMode.HIGH_BURST_EVICTION]

    # Execute Structured Swap Cycle
    swap_res = swap_mgr.execute_structured_swap_cycle()
    print(f"      Swap Cycle Result:    Status={swap_res['status']}, Written Entries={swap_res['written_entries']}")
    assert swap_res["status"] in ["SWAP_SUCCESS", "IDLE"]
    assert os.path.exists(test_ssd_path), "Structured log file must be created on SSD"

    # 6. Structured SSD Log Readback Loop
    readback = swap_mgr.read_structured_ssd_logs(max_blocks=5)
    print(f"\n[6/7] Closed-Loop SSD Log Readback:")
    print(f"      Blocks Read:          {readback['blocks_count']}")
    print(f"      Total Items:          {readback['total_entries']}")
    print(f"      Synthesizer Metrics:  L1 Locality={readback['synthesizer_telemetry']['l1_locality_ratio']}, "
          f"L3 Writeback={readback['synthesizer_telemetry']['l3_write_ratio']}, "
          f"PGO Hint={readback['synthesizer_telemetry']['pgo_hint']}")
    assert readback["blocks_count"] >= 1, "Must read back at least 1 structured block"

    # 7. Closed-Loop Janet-to-Godot 4.x Synthesizer
    synthesizer = LogDrivenJanetGodotSynthesizer(swap_manager=swap_mgr, budget_calculator=calc)
    synth_res = synthesizer.synthesize_hardware_tuned_scene(
        scene_name="NeoPraha_Alchemical_Hologram",
        seed=101
    )
    print(f"\n[7/7] Synthesized Closed-Loop Godot 4.x Stage:")
    print(f"      Scene Name:           {synth_res['scene_name']}")
    print(f"      Diversity Score:      {synth_res['scene_diversity_score']} / 100.0")
    print(f"      Generated Shader:     {synth_res['generated_godot_shader_path']}")
    print(f"      Generated Scene:      {synth_res['generated_godot_scene_path']}")
    print(f"      Generated Bridge:     {synth_res['generated_godot_bridge_path']}")

    # Verify shader and scene files were written to disk
    assert os.path.exists(synth_res["generated_godot_shader_path"]), "Shader file must exist!"
    assert os.path.exists(synth_res["generated_godot_scene_path"]), "Scene file must exist!"
    assert os.path.exists(synth_res["generated_godot_bridge_path"]), "Bridge file must exist!"

    with open(synth_res["generated_godot_shader_path"], "r", encoding="utf-8") as f:
        shader_txt = f.read()
    assert "shader_type canvas_item;" in shader_txt
    assert "VITAL_MAX_HP = 6" in shader_txt
    assert "fold_dihedral" in shader_txt
    assert "BAYER_4x4" in shader_txt
    assert "sd_chalice" in shader_txt
    assert "sd_athame" in shader_txt

    print("\n--- SAMPLE ENRICHED JANET SCRIPT ---")
    print("\n".join(synth_res["janet_script"].splitlines()[:16]))

    print("\n" + "=" * 70)
    print(" ✅ ALL 7 HARDWARE RENDER CALCULATOR & GODOT TESTS PASSED (100% INTEGRITY)")
    print("=" * 70)

if __name__ == "__main__":
    test_hardware_render_calculator_and_synthesis()
