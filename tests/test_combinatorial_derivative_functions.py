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









