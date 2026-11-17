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



