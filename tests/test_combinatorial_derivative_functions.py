"""Tests for Combinatorial Derivative Functions & Incremental Knowledge Engine."""

from __future__ import annotations

from pathlib import Path

from projects.trigonometric_hybrid_platform_framework.combinatorial_derivative_functions_engine import (
    build_combinatorial_incremental_knowledge_ledger,
)


def test_combinatorial_derivative_functions_and_incremental_ledger() -> None:
    ledger = build_combinatorial_incremental_knowledge_ledger()
    assert ledger["vital_max_hp"] == 6
    assert ledger["combinatorics"]["pairwise_c_6_2"] == 15
    assert ledger["combinatorics"]["triadic_c_6_3"] == 20
    assert ledger["combinatorics"]["total_multi_pillar_subsets"] == 57
    assert len(ledger["verified_derivative_functions"]) == 5

    for item in ledger["verified_derivative_functions"]:
        assert item["initial_status"] == "PATTERN_CANDIDATE"
        assert item["promoted_status"] == "VERIFIED_WITH_SCOPE"

    df01 = ledger["verified_derivative_functions"][0]["output"]
    assert df01["total_bytes"] == 192 and df01["cache_lines"] == 3
    assert df01["wntc_crc32_valid"] is True and df01["kpkt_sha256_valid"] is True

    df03 = ledger["verified_derivative_functions"][2]["output"]
    assert Path(df03["tres_path"]).exists()

    df05 = ledger["verified_derivative_functions"][4]["output"]
    assert df05["bits_used"] == 37 and df05["live_units_popcount"] == 8


def test_neurosymbolic_theories_case_study_and_kb_generation() -> None:
    from projects.trigonometric_hybrid_platform_framework.neurosymbolic_theories_case_study_and_kb import (
        run_neurosymbolic_pipeline_case_study_and_build_kb,
    )

    out = run_neurosymbolic_pipeline_case_study_and_build_kb()
    assert out["vital_max_hp"] == 6
    assert out["kb_records_count"] == 11  # 4 theories + 5 derivatives + 1 case study + 1 ceilings record
    assert Path(out["kb_path"]).exists()
    assert Path(out["md_path"]).exists()
    cs = out["case_study"]
    assert cs["dsl_calls_verified_in_ast"] is True
    assert cs["adversarial_dsl_blocked_delta_s_zero"] is True
    assert cs["henc_grounded_output"] is True
    assert cs["datagram_192b_verified"] is True


def test_unified_backend_ui_gateway_and_sprint_prompt_corpus() -> None:
    import pytest
    from krystal_kernel.constants import ContractViolation
    from projects.trigonometric_hybrid_platform_framework.unified_backend_ui_gateway_and_sprint_corpus import (
        UnifiedBackendUiAccessGateway,
        build_unified_gateway_and_sprint_prompt_corpus,
    )

    corpus = build_unified_gateway_and_sprint_prompt_corpus()
    assert corpus["vital_max_hp"] == 6
    assert corpus["actual_codebase_state"]["unresolved_nomenclature_terms"] == 0
    assert corpus["derivative_decision_ranking"]["top_recommended_choice"] == "DF_01_192B_WNTC_KPKT_DATAGRAM"
    assert len(corpus["multi_sprint_long_sequence"]) == 6

    gw = UnifiedBackendUiAccessGateway()
    env_shadow = gw.project_for_ui("wntc_kpkt_192b", uma_utilization=0.84)
    assert env_shadow["view_tokens"]["visual_tier"] == "WIREFRAME_SHADOW_0x1f"
    assert env_shadow["doc_exists_on_disk"] is True

    with pytest.raises(ContractViolation):
        gw.project_for_ui("unknown_capability")


def test_uma_cuda_vulkan_scrum_publication_and_universal_kb_kernel() -> None:
    import numpy as np
    from projects.trigonometric_hybrid_platform_framework.krystal_uma_cuda_vulkan_library import (
        kuAdviseAndDefend,
        kuMallocUMA,
    )
    from projects.trigonometric_hybrid_platform_framework.seasonal_scrum_prompt_kb_engine import (
        build_seasonal_scrum_kb_and_publication,
    )
    from projects.trigonometric_hybrid_platform_framework.universal_kb_hierarchy_kernel import (
        UniversalKnowledgeHierarchyKernel,
    )

    # Verify subnormal flushing + 64B alignment in kuMallocUMA
    sub_arr = np.array([1e-42, 2.5, -3e-43, 4.0], dtype=np.float32)
    ubuf = kuMallocUMA("sub_test", sub_arr)
    assert ubuf.aligned_64b is True
    assert float(ubuf.host_array[0]) == 0.0 and float(ubuf.host_array[1]) == 2.5
    adv_sat = kuAdviseAndDefend(ubuf, batch_n=128, reduction_k=512, uma_utilization=0.85)
    assert adv_sat["vulkan_gpu_allowed"] is False

    pub = build_seasonal_scrum_kb_and_publication()
    assert pub["vital_max_hp"] == 6
    assert Path(pub["kb_path"]).exists()
    assert Path(pub["publication_md_path"]).exists()
    assert pub["ku_matmul_defended"]["tiles_executed"] == 2
    assert pub["injected_op_demo"]["hypertransport_192b_ok"] is True

    ukh = UniversalKnowledgeHierarchyKernel()
    sol = ukh.push_custom_engineering_solution("xerj_vulkan_mesh_copilot", reduction_k=512, uma_utilization=0.35)
    assert sol["schema_validated"] is True
    assert sol["hypertransport_192b_verified"] is True
    assert sol["protocol_solution_envelope"]["scope_environment"]["superglobal_scope"]["vital_max_hp"] == 6


def test_monetization_corporate_rules_and_openvino_addon_kernel() -> None:
    from projects.trigonometric_hybrid_platform_framework.monetization_corporate_and_openvino_addon_kernel import (
        ComputeHierarchyOpenVinoTelemetryAddonKernel,
        decide_corporate_declarative_rule_option,
    )

    corp = decide_corporate_declarative_rule_option(airgapped_required=True, multi_tenant_hypervisor=False)
    assert corp["vital_max_hp"] == 6
    assert corp["selected_option_id"] == "OPT_A_AIRGAPPED_DEFENSE_FINANCE"
    assert len(corp["incremental_onboarding_ladder"]) == 4

    addon = ComputeHierarchyOpenVinoTelemetryAddonKernel().execute_five_phases_over_existing_data(
        uma_utilization=0.36,
        airgapped_company=True,
    )
    assert addon["vital_max_hp"] == 6
    assert len(addon["inter_sprint_axioms"]) == 6
    assert len(addon["monetization_capabilities"]) == 5
    assert len(addon["phases"]) == 5
    assert addon["phases"][4]["datagram_bytes"] == 192
    assert addon["phases"][4]["wntc_crc32_ok"] is True
    assert addon["phases"][4]["kpkt_sha256_ok"] is True


def test_unix_rust_vulkan_drivers_kernel_ds_and_ai_chip_automation() -> None:
    import ast
    from projects.trigonometric_hybrid_platform_framework.unix_rust_vulkan_driver_and_kernel_ds_framework import (
        build_full_unix_rust_vulkan_oneapi_framework_report,
    )

    rep = build_full_unix_rust_vulkan_oneapi_framework_report()
    assert rep["vital_max_hp"] == 6
    assert len(rep["unix_linux_big_query_theories"]) == 6
    assert len(rep["strict_allocation_hardware_limits"]["exposed_hardware_limits"]) == 7
    assert rep["kernel_data_structure_benchmark"]["bitmap_zero_copy_speedup_vs_legacy_x"] > 1.5
    assert rep["dynamic_threading_log_benchmark"]["speedup_sharded_vs_mutex_x"] > 1.2
    assert rep["dynamic_threading_log_benchmark"]["krystal_sharded_spsc_128b_priority_log"]["p0_constitutional_frames_dropped"] == 0
    assert len(rep["lightweight_rust_vulkan_drivers"]) == 4
    assert rep["hardware_ai_chip_automation"]["logged_connections_count"] == 3
    assert Path(rep["report_json_path"]).exists()
    assert Path(rep["architecture_md_path"]).exists()
    assert Path("src/rust-bot/src/unix_windows_vulkan_driver_core.rs").exists()

    # Zero bare `assert` audit across all production modules in trigonometric_hybrid_platform_framework
    pkg_dir = Path("projects/trigonometric_hybrid_platform_framework")
    bare_asserts = []
    for py_file in sorted(pkg_dir.glob("*.py")):
        tree = ast.parse(py_file.read_text(encoding="utf-8"), filename=str(py_file))
        for node in ast.walk(tree):
            if isinstance(node, ast.Assert):
                bare_asserts.append(f"{py_file.name}:{node.lineno}")
    assert bare_asserts == []


def test_stock_vs_krystal_platform_benchmark_suite() -> None:
    from projects.trigonometric_hybrid_platform_framework.stock_vs_krystal_platform_benchmark_suite import (
        run_stock_vs_krystal_benchmark_suite,
    )

    ledger = run_stock_vs_krystal_benchmark_suite()
    assert ledger["vital_max_hp"] == 6
    assert ledger["summary"]["total_benchmarks"] == 8
    assert ledger["summary"]["krystal_wins"] >= 7
    assert Path(ledger["json_path"]).exists()
    assert Path(ledger["csv_path"]).exists()
    assert Path(ledger["md_path"]).exists()


def test_universal_mmio_hierarchy_coprocessor_and_isa_matrix() -> None:
    from projects.trigonometric_hybrid_platform_framework.universal_hardware_mmio_aperture_kernel import (
        build_universal_mmio_hierarchy_and_absorption_report,
    )

    rep = build_universal_mmio_hierarchy_and_absorption_report()
    assert rep["vital_max_hp"] == 6
    assert len(rep["phase_1_in_memory_hardware_hierarchy"]) == 9
    assert len(rep["phase_2_absorption_options"]) == 4
    assert rep["phase_2_executed_absorption"]["total_unit_families_absorbed"] == 9
    assert rep["phase_2_executed_absorption"]["total_repository_files_indexed"] > 100
    assert rep["constitutional_safety_verification"]["read_only_vital_hp_write_blocked"] is True
    assert rep["constitutional_safety_verification"]["vital_hp_before_and_after"] == [6, 6]
    assert rep["mmio_vs_stock_lookup_benchmark"]["speedup_x"] > 2.0
    assert rep["coprocessor_end_to_end_benchmark"]["optimal_hetero_speedup_vs_stock_x"] > 1.5
    assert len(rep["cross_isa_and_hardware_configurations"]) == 6
    assert len(rep["linux_driver_capabilities"]) == 7
    assert Path(rep["json_path"]).exists()
    assert Path(rep["md_path"]).exists()


def test_hybrid_rust_linux_drivers_and_intel_xe_cuda_streams() -> None:
    from projects.trigonometric_hybrid_platform_framework.hybrid_rust_and_intel_xe_cuda_streams import (
        build_hybrid_rust_and_intel_xe_cuda_report,
    )

    out = build_hybrid_rust_and_intel_xe_cuda_report()
    assert out["vital_max_hp"] == 6
    assert len(out["hybrid_rust_planes"]) == 5
    assert len(out["linux_driver_capabilities"]) == 7
    assert out["intel_xe_device_properties"]["fp32_stream_processors_total"] == 768
    assert out["rust_128b_descriptor_attestation"]["byte_length"] == 128
    assert len(out["unified_accelerator_targets"]) == 3
    assert out["occupancy_calculation"]["grf_spill_free"] is True
    assert len(out["rust_language_and_ecosystem_gains"]) == 4
    assert len(out["causal_mechanisms_behind_parameters"]) == 6
    assert len(out["usable_algorithms_catalog"]) == 6
    assert out["verified_flash_attention_slm_demo"]["fits_in_64kb_subslice_slm"] is True
    assert out["verified_flash_attention_slm_demo"]["relative_l2_error_vs_exact"] < 1e-5
    assert out["win11_d3d12_64kb_heap_probe"]["ready_for_d3d12_open_existing_heap"] is True
    assert "CPU" in out["intel_gen11_plus_stability_and_process_governor"]["openvino_live_device_telemetry"]
    assert out["intel_gen11_plus_stability_and_process_governor"]["os_process_management_bridge"]["pid"] > 0
    assert out["high_frequency_telemetry_overhead_audit"]["hotpath_overhead_reduction_x"] > 5.0
    assert out["visual_quality_and_performance_gain"]["psnr_improvement_db"] > 8.0
    assert out["visual_quality_and_performance_gain"]["fits_within_120hz_budget"] is True
    assert out["openvino_multithreading_presets"]["live_compile_verified"] is True
    assert out["cpu_lookahead_whisperer_benchmark"]["cpu_whisperer_speedup_x"] > 1.5
    assert Path(out["json_path"]).exists()
    assert Path(out["md_path"]).exists()


def test_bot_integrated_efficiency_overseer_and_core_activation() -> None:
    import tempfile
    from krystal_bot.integrated_efficiency_overseer import IntegratedEfficiencyOverseer
    from krystal_bot.service import BotService

    ov = IntegratedEfficiencyOverseer()
    normal = ov.evaluate_and_select("coprocessor_visual_ai", uma_utilization=0.36, junction_temp_c=60.0)
    assert normal["vital_max_hp"] == 6
    assert normal["selected_core_activation_strategy"] == "CORE_WAKE_04_HETERO_3P_1VSYNC_IRIS_XE"
    assert normal["selected_openvino_preset"] == "OV_MT_03_HETERO_IRIS_XE_SINGLE_STREAM_PLUS_3T_CPU"

    emerg = ov.evaluate_and_select("coprocessor_visual_ai", uma_utilization=0.84, junction_temp_c=88.0)
    assert emerg["selected_core_activation_strategy"] == "CORE_WAKE_02_UMA_THERMAL_SHIELD_2C"
    assert emerg["core_activation_details"]["igpu_awake"] is False

    # Overenie 12-krokovej hysterézie proti okamžitému kmitaniu späť
    hold = ov.evaluate_and_select("coprocessor_visual_ai", uma_utilization=0.40, junction_temp_c=62.0)
    assert hold["resolved_mode"] == "uma_thermal_emergency"

    with tempfile.TemporaryDirectory() as tmp:
        svc = BotService(root=".", data_dir=tmp, use_keras=False)
        msg = svc.message("/auto-select batch_benchmark", channel="cli", sender="local")
        assert msg["ok"] is True
        assert "CORE_WAKE_06_BATCH_BURST_8T_ALL_THREADS" in msg["reply"]

        arb_msg = svc.message("/arbitrate", channel="cli", sender="local")
        assert arb_msg["ok"] is True
        assert "UI_120HZ_SCANOUT_GUARD" in arb_msg["reply"]

        opp_msg = svc.message("/prilezitosti", channel="cli", sender="local")
        assert opp_msg["ok"] is True
        assert "OPP_GPU_01_ZERO_BOUNCE_64KB_USM_BRIDGE" in opp_msg["reply"]


def test_observable_ml_study_glmark_fhd_4k_and_adrenalin_cms() -> None:
    from projects.trigonometric_hybrid_platform_framework.observable_study_and_intel_adrenalin_cms import (
        IntelAdrenalinKrystalCmsEngine,
    )

    bundle = IntelAdrenalinKrystalCmsEngine().build_complete_cms_and_study_bundle()
    assert bundle["vital_max_hp"] == 6
    assert len(bundle["new_process_definitions"]) == 5
    ml = bundle["perceptual_ml_study"]
    assert ml["models_test_evaluation"]["M2_PERCEPTUAL_COUPLED_MLP"]["accuracy"] >= ml["models_test_evaluation"]["M0_NAIVE_MAJORITY_BASELINE"]["accuracy"]
    gl = bundle["glmark_fhd_and_4k_benchmark"]
    assert gl["monitored_frames_logged"] == 16
    assert gl["resolutions"]["FHD_1920x1080"]["speedup_x"] > 1.1
    assert gl["resolutions"]["UHD_4K_3840x2160"]["speedup_x"] > 1.1
    assert len(bundle["fair_licensing_system"]["fair_price_to_performance_tiers"]) == 4
    assert bundle["fair_licensing_system"]["authorized_partner_addendum"]["grantee"] == "Ivan Novikov"
    assert Path(bundle["summary_json_path"]).exists()
    assert Path(bundle["study_md_path"]).exists()


def test_mcp_parameter_grep_comm_pipeline_and_bot_configurator() -> None:
    from krystal_web_hub.hybrid_mcp_antigravity_connector import GLOBAL_HYBRID_MCP_CONNECTOR
    from projects.trigonometric_hybrid_platform_framework.mcp_parameter_grep_pipeline_and_bot_configurator import (
        render_bot_configurator_studio_html,
    )

    r_grep = GLOBAL_HYBRID_MCP_CONNECTOR.execute_tool("krystal-grep-parameters", {"pattern": "VITAL_MAX_HP|NUM_STREAMS"})
    assert r_grep["vital_max_hp"] == 6
    assert r_grep["matches_count"] >= 3
    assert r_grep["validation_status"] == "CLEAN_NO_DRIFT"

    r_pipe = GLOBAL_HYBRID_MCP_CONNECTOR.execute_tool("krystal-automated-comm-pipeline", {})
    assert r_pipe["vital_max_hp"] == 6
    assert Path(r_pipe["announcements_jsonl_path"]).exists()

    r_diff = GLOBAL_HYBRID_MCP_CONNECTOR.execute_tool("krystal-bot-configurator-diff", {})
    assert r_diff["vital_max_hp"] == 6
    assert r_diff["speedups_vs_baseline"]["telemetry_x"] > 200.0
    assert "KRYSTAL" in render_bot_configurator_studio_html()


def test_compiled_nostd_concurrent_rust_modules_and_low_footprint() -> None:
    import tempfile
    from krystal_bot.service import BotService
    from projects.trigonometric_hybrid_platform_framework.concurrent_rust_ffi_runtime_bridge import (
        run_concurrent_rust_efficiency_and_resource_audit,
    )

    rep = run_concurrent_rust_efficiency_and_resource_audit()
    assert rep["vital_max_hp"] == 6
    mods = rep["compiled_nostd_rust_modules"]
    assert mods["krystal_concurrent_rust_core"]["binary_built"] is True
    assert mods["pulse_logger_core"]["binary_built"] is True
    assert mods["combined_binary_kb"] <= 32.0
    assert mods["fits_inside_32kb_l1i_cache"] is True

    frug = rep["low_resource_frugality"]
    assert frug["stack_kb_per_worker"] == 128
    assert frug["stack_memory_reduction_x"] >= 32.0
    assert frug["idle_cpu_spin_percent"] == 0.0
    assert frug["rust_heap_allocations_bytes"] == 0

    coex = rep["concurrent_coexecution_metrics"]
    assert coex["total_frames_submitted"] == 512
    assert coex["completed_valid_frames"] == 480
    assert coex["shed_p2_frames_under_uma_pressure"] == 28
    assert coex["rejected_invalid_hp_frames_delta_s_zero"] == 4
    assert coex["bar0_live_completed_head"] == 480

    with tempfile.TemporaryDirectory() as tmp:
        svc = BotService(root=".", data_dir=tmp, use_keras=False)
        msg = svc.message("/rust-moduly", channel="cli", sender="local")
        assert msg["ok"] is True
        assert "krystal_concurrent_rust_core" in msg["reply"]


def test_expanded_10_stage_neurosymbolic_pipeline_v2() -> None:
    import tempfile
    from krystal_bot.service import BotService
    from projects.trigonometric_hybrid_platform_framework.expanded_neurosymbolic_pipeline_v2 import (
        run_expanded_10_stage_neurosymbolic_pipeline,
    )

    out = run_expanded_10_stage_neurosymbolic_pipeline()
    assert out["vital_max_hp"] == 6
    assert len(out["proven_capabilities"]) == 6
    assert out["theories_count"] == 8
    assert len(out["stages"]) == 10
    assert out["kb_records_count"] == 24
    assert out["ns05_query1"]["lookup_nostd_kb"] == 24.5
    assert out["ns05_query1"]["conflict_verdict"] == "CONFLICT"
    assert out["ns05_query1"]["conflict_branches_count"] == 2
    assert out["ns06_hetero_parity"]["greedy_token_parity"] is True
    assert out["ns06_hetero_parity"]["max_abs_logit_diff"] < 1e-2
    assert out["ns07_grammar_masking"]["masked_fabrication_rate"] == 0.0
    assert out["ns07_grammar_masking"]["masked_grounded_count"] == 40
    assert out["ns09_parameter_grep_status"] == "CLEAN_NO_DRIFT"
    assert Path(out["kb_v2_path"]).exists()
    assert Path(out["doc_v2_path"]).exists()

    with tempfile.TemporaryDirectory() as tmp:
        svc = BotService(root=".", data_dir=tmp, use_keras=False)
        msg = svc.message("/neurosymbolika", channel="cli", sender="local")
        assert msg["ok"] is True
        assert "10-stupňová" in msg["reply"]


def test_contactless_solder_metal_vulkan_ubuntu_and_speculative_loop() -> None:
    from deploy.ubuntu_contactless_solder_power.krystal_ubuntu_power_daemon import probe_ubuntu_linux_sensors
    from projects.trigonometric_hybrid_platform_framework.metal_vulkan_endpoints_and_speculative_bench_loop import (
        execute_all_milestones_and_export_roadmap,
        render_contactless_solder_live_studio_html,
    )

    out = execute_all_milestones_and_export_roadmap()
    assert out["vital_max_hp"] == 6
    assert len(out["milestones"]) == 7

    cs = out["contactless_solder_bundle"]
    assert cs["nomenclature_and_theory_th09"]["marketing_name_sk"] == "Bezkontaktná letovačka"
    assert cs["contactless_solder_bytecode"]["bytecode_size_bytes"] == 8 + 8 * 8
    assert len(cs["contactless_solder_bytecode"]["progression_timeline"]) == 16
    assert cs["strategic_uma_process_pairing_pipeline"]["stays_below_078_ceiling"] is True
    assert cs["strategic_uma_process_pairing_pipeline"]["pairing_speedup_x"] > 1.0

    vk_gate = out["vulkan_endpoint_gate_results"]
    assert vk_gate["committed_ep1"]["status"] == "COMMITTED_AFTER_GLMARK_PASS"
    assert vk_gate["rolled_back_regression_ep"]["status"] == "ROLLED_BACK_GLMARK_REGRESSION_DELTA_S_ZERO"
    assert Path(vk_gate["layer_manifest_path"]).exists()

    spec = out["speculative_debug_and_benchmark_loop"]
    assert spec["best_verified_speedup_x"] > 2.0
    assert any("REJECTED_REGRESSION" in c["verdict"] for c in spec["spawned_child_branches"])

    w32 = out["win32_gen11_and_rust_vulkan_quota"]
    assert w32["rust_vulkan_acceleration_quota_64b"]["rust_symbol_exported_in_dll"] is True
    assert 0 < w32["rust_vulkan_acceleration_quota_64b"]["allowed_iris_xe_eus_of_96"] <= 96

    ub = probe_ubuntu_linux_sensors()
    assert ub["vital_max_hp"] == 6
    assert "Bezkontaktná letovačka" in render_contactless_solder_live_studio_html()


def test_sovereign_rust_firmware_validator_stream_cuda_and_kisak_gallium_presets() -> None:
    import tempfile
    from krystal_bot.service import BotService
    from projects.trigonometric_hybrid_platform_framework.sovereign_rust_firmware_cuda_gallium_suite import (
        execute_sovereign_rust_firmware_cuda_gallium_suite,
        match_vulkan_engine_preset,
        render_sovereign_rust_gallium_cuda_studio_html,
    )

    res = execute_sovereign_rust_firmware_cuda_gallium_suite(uma_util=0.34, base_vulkan_fps=60.0)
    assert res["vital_max_hp"] == 6

    fw = res["sovereign_rust_firmware"]
    assert fw["fits_in_32kb_l1i_cache"] is True
    assert fw["rust_binary_size_kb"] <= 24.0
    assert fw["firmware_control_speedup_x"] > 1.5
    assert fw["constitutional_rollback_audit"]["fault_return_code"] == -6
    assert fw["constitutional_rollback_audit"]["svn_epoch_preserved_delta_s_zero"] is True

    rv = res["retroactive_validator_generator_aggregator"]
    assert rv["emission_authorized"] is True
    assert rv["proposer_not_equal_verifier"] is True
    assert rv["sanitized_subnormals_ftz_count"] == 128
    assert rv["retroactive_compute_speedup_x"] > 1.2
    assert rv["self_certification_quarantine_verified"] is True

    sc = res["stream_cuda_and_fps_booster"]
    assert sc["device_props"]["fp32_stream_processors_cuda_core_equivalent"] == 768
    assert sc["boosted_effective_fps"] > sc["base_native_fps"]
    assert sc["chronometric_shadow_multiplexing_ok"] is True

    gal = res["kisak_gallium_and_vulkan_presets"]
    assert gal["zero_touch_interposition_p1"] is True
    assert gal["gallium_pillars_count"] == 6
    assert gal["vulkan_engine_presets_count"] == 8
    assert Path(gal["kb_jsonl_path"]).exists()
    assert Path(gal["ubuntu_port_installer_path"]).exists()
    assert Path(gal["vulkan_layer_manifest_path"]).exists()

    ue5 = match_vulkan_engine_preset("UnrealEngine", uma_util=0.34, base_fps=60.0)
    assert ue5["code"] == "ENG_UNREAL_5_VULKAN"
    assert ue5["stream_boosted_vulkan_fps"] > 120.0

    # UMA >= 0.78 circuit breaker protects scanout and disables speculative framegen
    saturated = match_vulkan_engine_preset("Source 2", uma_util=0.82, base_fps=60.0)
    assert saturated["active_stream_processors_of_768"] == 128
    assert saturated["stream_boosted_vulkan_fps"] == 60.0

    assert "Kisak-Gallium" in render_sovereign_rust_gallium_cuda_studio_html()

    with tempfile.TemporaryDirectory() as tmp:
        svc = BotService(root=".", data_dir=tmp, use_keras=False)
        msg = svc.message("/kisak-gallium Source 2", channel="cli", sender="local")
        assert msg["ok"] is True
        assert "krystal_sovereign_firmware_rs" in msg["reply"]
        assert "Source 2" in msg["reply"]


def test_openvino_gallium_driver_bridge_and_unified_benchmark() -> None:
    import tempfile
    from krystal_bot.service import BotService
    from krystal_kernel.openvino_aep_runtime import OpenVinoAepRuntime
    from scripts.benchmark_unified_openvino_gallium_rust_stack import run_unified_openvino_gallium_rust_benchmark

    rt = OpenVinoAepRuntime()
    ch = rt.get_gallium_driver_channel(uma_utilization=0.34, payload_bytes=65_536)
    assert ch["vital_max_hp"] == 6
    assert ch["zero_copy_shared_tensor"] is True

    ch_spill = rt.get_gallium_driver_channel(uma_utilization=0.84, payload_bytes=65_536)
    assert ch_spill["active_openvino_device"] == "CPU"
    assert "LLVMPIPE" in ch_spill["gallium_driver_plane"]

    bench = run_unified_openvino_gallium_rust_benchmark(samples_n=10)
    assert bench["vital_max_hp"] == 6
    assert bench["plane_3_openvino_gallium_1mb_tensor_binding"]["zero_copy_binding_speedup_x"] > 1.0
    sy = bench["plane_4_openvino_gallium_neural_modules"]["neural_gallium_frame_synthesizer"]
    assert sy["zero_copy_address_parity"] is True
    assert sy["parity_verified"] is True
    assert len(bench["plane_4_openvino_gallium_neural_modules"]["connected_modules_catalog"]) == 4
    assert Path(bench["report_json_path"]).exists()
    assert Path(bench["report_md_path"]).exists()

    with tempfile.TemporaryDirectory() as tmp:
        svc = BotService(root=".", data_dir=tmp, use_keras=False)
        msg = svc.message("/openvino-gallium", channel="cli", sender="local")
        assert msg["ok"] is True
        assert "OpenVINO <-> Kisak-Mesa Gallium" in msg["reply"]















