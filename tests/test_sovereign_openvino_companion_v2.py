"""Verification tests for `projects/sovereign_openvino_companion_v2`."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from krystal_kernel.constants import ContractViolation, VITAL_MAX_HP
from projects.sovereign_openvino_companion.operating_coordinates_community import (
    OperatingCoordinatesVector,
)
from projects.sovereign_openvino_companion_v2 import (
    AdaptiveSpendingOptimizer,
    BinaryParameterAspectCompiler,
    BufferedComputationLogSystem,
    SensoryCoordinateMorphismEngine,
    SensoryManifoldCoordinates,
    SovereignOpenVinoCompanionV2,
    StratifiedProcessMicroserviceFlowArchitect,
)
from projects.sovereign_openvino_companion_v2.cli import main as v2_cli_main


def test_buffered_logger_coalescing_and_tamper_detection(tmp_path: Path) -> None:
    logger = BufferedComputationLogSystem(log_dir=tmp_path, reset_ledger=True, flush_every=4)
    for i in range(6):
        logger.log_event("telemetry", f"OP_{i}", "S2", 0.1, {"frame_index": i})

    # 4 triggered 1 automatic flush; remaining 2 stay buffered until flush/verify
    assert logger.disk_flush_count == 1
    ok, report = logger.verify_chain_integrity()
    assert ok is True
    assert report["entries_checked"] == 6
    assert logger.disk_flush_count == 2


def test_smart_v2_session_prebrake_parasite_and_candidate_nodes(tmp_path: Path) -> None:
    v2 = SovereignOpenVinoCompanionV2(log_dir=tmp_path, reset_ledger=True, flush_every=16)
    session = v2.run_smart_session(num_frames=12)
    assert session["chain_valid"] is True
    assert session["vital_max_hp"] == 6

    rationales = [f["routing_rationale"] for f in session["sample_frames"]]
    assert "PREDICTIVE_UMA_VELOCITY_PREBRAKE_CPU" in rationales
    assert session["sample_frames"][4]["artifact_released"] is False  # entropy=0.0 parasite cutoff
    assert session["sample_frames"][4]["spending_plan"]["compressed_spend_mj"] == 0.0180

    # Candidate runtime nodes stay strictly separate from 56 frozen theory nodes
    cands = v2.prototype_candidate_runtime_nodes()
    assert cands["frozen_theory_nodes"] == 56
    assert cands["node_57_added_to_frozen_theory"] is False
    assert all(c["p7_directional_gate_pass"] is True for c in cands["candidate_runtime_nodes"])


def test_v1_vs_v2_paired_benchmark_and_cli(tmp_path: Path) -> None:
    v2 = SovereignOpenVinoCompanionV2(log_dir=tmp_path / "v2", reset_ledger=True)
    ab = v2.run_v1_vs_v2_ab_benchmark(iterations=12, tmp_v1_dir=tmp_path / "v1")
    assert ab["v2_chain_valid"] is True
    assert ab["v2_fine_tuned"]["median_ms"] <= ab["v1_baseline"]["median_ms"]
    assert ab["productivity_per_watt"]["v2_paced_12w_ops_per_w"] > ab["productivity_per_watt"]["v1_baseline_15w_ops_per_w"]

    rc = v2_cli_main(["--run-session", "--frames", "6", "--candidate-nodes", "--reset-ledger", "--log-dir", str(tmp_path / "cli")])
    assert rc == 0
    assert VITAL_MAX_HP == 6


def test_sensory_coordinate_morphism_continuous_and_p6_homotopic_fork(tmp_path: Path) -> None:
    logger = BufferedComputationLogSystem(log_dir=tmp_path, reset_ledger=True, flush_every=8)
    engine = SensoryCoordinateMorphismEngine(logger=logger)

    src_op = OperatingCoordinatesVector(4, 5, 0.25, 5.0, 15.0, 4.8201, VITAL_MAX_HP)
    dst_op = OperatingCoordinatesVector(7, 7, 0.82, 7.8, 7.5, 1.40, VITAL_MAX_HP)

    s_src = engine.project_operating_to_sensory(src_op)
    s_dst = engine.project_operating_to_sensory(dst_op)
    inv_op = engine.inverse_sensory_to_operating(s_src)
    assert inv_op.vital_hp_coordinate == VITAL_MAX_HP

    # Continuous Golden morph
    res_cont = engine.morph_sensory_coordinates(s_src, s_dst, morph_factor=0.618034, contradictory_evidence=False)
    assert res_cont["topology"] == "HOMEOMORPHIC_CONTINUOUS_MORPH"
    assert res_cont["convex_averaging_blocked"] is False
    assert res_cont["smrp_crc32_valid"] is True
    assert len(bytes.fromhex(res_cont["smrp_20b_hex"])) == 20
    assert res_cont["morphed_coordinates"]["vital_hp_anchor"] == VITAL_MAX_HP

    # P6 Orthogonal Fork when evidence contradicts
    res_fork = engine.morph_sensory_coordinates(s_src, s_dst, morph_factor=0.5, contradictory_evidence=True)
    assert res_fork["topology"] == "SUPERPOSED_HOMOTOPIC_FORK_ALPHA_PERP_BETA"
    assert res_fork["convex_averaging_blocked"] is True
    assert res_fork["branch_alpha"] != res_fork["branch_beta"]

    # Negative guards raise ContractViolation
    with pytest.raises(ContractViolation):
        engine.morph_sensory_coordinates(s_src, s_dst, morph_factor=1.5)
    with pytest.raises(ContractViolation):
        SensoryManifoldCoordinates(3, "L3_DEDUCTIVE_VERIFICATION", "SOLAR_CLARITY", 250.0, 1.2, 310.0, 0.9, vital_hp_anchor=5)


def test_binary_parameter_aspect_compiler_roundtrip_and_tamper_quarantine(tmp_path: Path) -> None:
    logger = BufferedComputationLogSystem(log_dir=tmp_path / "logs", reset_ledger=True)
    compiler = BinaryParameterAspectCompiler(logger=logger)
    bin_dir = tmp_path / "bins"
    out = compiler.compile_all_aspect_ratios(morph_factor=0.618034, output_bin_dir=bin_dir)

    assert out["vital_max_hp"] == 6
    assert out["frozen_theory_nodes"] == 56
    assert out["proposer_equals_verifier"] is False
    assert out["artifact_gate_released"] is True
    assert out["canonical_21_6_5"]["exact_integer_aspect_ratio"] == "21 : 6 : 5"
    assert out["phi_golden_79_49"]["lossless_roundtrip_verified"] is True
    assert out["silver_octagonal_90_38"]["lossless_roundtrip_verified"] is True

    canon_bytes = bytes.fromhex(out["canonical_21_6_5"]["frame_128b_hex"])
    kmb1_bytes = bytes.fromhex(out["canonical_21_6_5"]["kmb1_24b_hex"])
    assert len(canon_bytes) == 128
    assert len(kmb1_bytes) == 24
    assert Path(out["written_binary_files"]["canonical_21_6_5_128b.bin"]).read_bytes() == canon_bytes

    # Tamper 1 byte in [108:128] SMRP slice -> must raise ContractViolation
    tampered = bytearray(canon_bytes)
    tampered[115] ^= 0xFF
    with pytest.raises(ContractViolation):
        compiler.decompile_canonical_128b_frame(bytes(tampered))


def test_processes_microservices_flows_matrix_governor_cli_and_http(tmp_path: Path) -> None:
    import http.server
    import threading
    import urllib.error
    import urllib.request

    from krystal_kernel.sovereign_openvino_oneapi_app import SovereignOpenVinoRequestHandler

    arch = StratifiedProcessMicroserviceFlowArchitect()
    pm_audit = arch.define_processes_microservices_and_stack_conclusions()
    tiers = pm_audit["execution_tiers_comparison"]
    assert tiers["native_processes_s0_s3"]["can_self_certify"] is False
    assert tiers["microservices_cw01_cw06_s4_s8"]["can_self_certify"] is False
    assert tiers["stack_defined_conclusion_s9"]["can_self_certify"] is False
    assert pm_audit["constitutional_invariants"]["frozen_theory_nodes"] == 56

    flows = arch.synthesize_new_stack_flows()
    assert flows["total_flows"] == 6

    matrix_res = arch.synthesize_hypervisor_matrix_behavioral_and_knowledge_properties()
    assert matrix_res["choice_palette_matrix"]["total_choice_cells"] == 120
    assert matrix_res["choice_palette_matrix"]["exceeds_100_choices"] is True
    assert len(matrix_res["new_knowledge_properties"]) == 6

    surv = arch.synthesize_survival_governor_motivation_and_aspects(
        uma_utilization=0.72, uma_velocity=0.16, epistemic_gain_nats=1.3689, predicted_energy_mj=0.05
    )
    assert surv["cpu_governor_live_metrics"]["governor_profile"] == "CPU_SURVIVAL_THROTTLE"
    assert surv["compute_motivation_core"]["motivation_score_nats_per_risk_mj"] > 0.0

    # CLI flags
    rc = v2_cli_main(
        [
            "--morph-sense",
            "--compile-binary-ratios",
            "--stack-flows",
            "--emit-bin-dir",
            str(tmp_path / "cli_bins"),
            "--reset-ledger",
            "--log-dir",
            str(tmp_path / "cli_logs"),
        ]
    )
    assert rc == 0

    # Ephemeral HTTP GET & POST test on /api/v1/runtime/sensory_binary_aspect_compiler
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), SovereignOpenVinoRequestHandler)
    port = server.server_address[1]
    th = threading.Thread(target=server.serve_forever, daemon=True)
    th.start()
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/v1/runtime/sensory_binary_aspect_compiler", timeout=5) as resp:
            assert resp.status == 200
            body = json.loads(resp.read().decode("utf-8"))
            assert body["vital_max_hp"] == 6
            assert body["binary_aspect_ratios"]["artifact_gate_released"] is True

        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/api/v1/runtime/sensory_binary_aspect_compiler",
            data=json.dumps({"morph_factor": 2.0}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with pytest.raises(urllib.error.HTTPError) as http_err:
            urllib.request.urlopen(req, timeout=5)
        assert http_err.value.code == 400
    finally:
        server.shutdown()
        server.server_close()


def test_markdown_performance_module_doc_query_knowledge_graph_and_http(tmp_path: Path) -> None:
    import http.server
    import threading
    import urllib.error
    import urllib.request

    from krystal_kernel.epistemic_enlg_compiler import EnlgRole
    from krystal_kernel.sovereign_openvino_oneapi_app import SovereignOpenVinoRequestHandler
    from projects.sovereign_openvino_companion_v2 import (
        MarkdownBehavioralKnowledgeGraphModule,
        OntologicSolutionSynthesizer,
        TraceMarkEngine,
    )

    v2 = SovereignOpenVinoCompanionV2(log_dir=tmp_path, reset_ledger=True)
    kg_mod = MarkdownBehavioralKnowledgeGraphModule(logger=v2.logger)
    assert len(kg_mod.nodes) >= 5

    res = kg_mod.derive_knowledge_graph_and_outcomes(
        query="binary aspect ratio survival governor process microservice",
        companion_v2=v2,
    )
    assert res["vital_max_hp"] == 6
    assert res["frozen_theory_nodes_untouched"] == 56
    assert res["node_57_added_to_frozen_theory"] is False
    assert res["knowledge_graph"]["dag_cycles"] == 0
    assert res["knowledge_graph"]["dag_strictly_acyclic"] is True
    assert res["documentation_query_result"]["matches_found"] >= 1
    assert len(res["derived_outcomes"]) == 4
    assert len(res["outsourcing_strategies"]) == 4
    assert len(res["derived_subject_constraints_and_computation"]["constraint_to_compute_laws"]) == 4
    assert len(res["modular_self_goal_fetching_points"]) == 4
    assert res["synthesizer_codename"] == "OntologicSolutionSynthesizer"
    assert res["tracemark_engine_commit"]["commit_status"] == "TRACED_AND_COMMITTED_WITH_SCOPE"

    # OntologicSolutionSynthesizer & TraceMarkEngine direct verification + self-cert block
    synth = OntologicSolutionSynthesizer(logger=v2.logger)
    assert len(synth.nodes) == len(kg_mod.nodes)
    tmk = TraceMarkEngine(logger=v2.logger)
    mark1 = tmk.issue_commit_mark("FEAT_A", EnlgRole.PROPOSER, EnlgRole.VERIFIER, {"ok": True})
    assert mark1["tracemark_id"].startswith("TMK_0001_")
    with pytest.raises(ContractViolation):
        tmk.issue_commit_mark("FEAT_SELF", EnlgRole.PROPOSER, EnlgRole.PROPOSER)

    # Verify WSL2<->NT 64B Coexistence Binary, Candidate Theory & Inverted Reverse Engineering
    syn_full = v2.run_sensory_binary_and_flow_synthesis()
    assert len(syn_full["wsl_nt_coexistence"]["coexistence_matrix_4x4"]) == 4
    assert len(bytes.fromhex(syn_full["wsl_nt_coexistence"]["self_binary_64b_hex"])) == 64
    assert syn_full["candidate_theoretical_patterns"]["node_57_added_to_frozen_theory"] is False
    inv = res["inverted_reverse_engineering_and_mixed_derivatives"]
    assert inv["reconstructed_from_binaries"]["canonical_128b_roundtrip"] is True
    assert inv["mixed_derivatives"]["mixed_orthogonal_pov_index"] > 0.0
    assert res["retrace_over_contours"]["epoch_reset"] is True
    assert len(res["cross_parameter_gains_scripting_and_puzzle_gaps"]["scripting_paradigm_hierarchy"]) == 4
    assert len(res["cross_parameter_gains_scripting_and_puzzle_gaps"]["missing_parts_puzzle_slots"]) == 2

    with pytest.raises(ContractViolation):
        kg_mod.query_documentation("   ")

    rc = v2_cli_main(["--doc-query", "binary aspect ratio 128B", "--reset-ledger", "--log-dir", str(tmp_path / "doc_cli")])
    assert rc == 0

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), SovereignOpenVinoRequestHandler)
    port = server.server_address[1]
    th = threading.Thread(target=server.serve_forever, daemon=True)
    th.start()
    try:
        with urllib.request.urlopen(
            f"http://127.0.0.1:{port}/api/v1/runtime/markdown_knowledge_graph?q=binary+aspect+ratio",
            timeout=5,
        ) as resp:
            assert resp.status == 200
            data = json.loads(resp.read().decode("utf-8"))
            assert data["knowledge_graph"]["dag_cycles"] == 0

        bad_req = urllib.request.Request(
            f"http://127.0.0.1:{port}/api/v1/runtime/markdown_knowledge_graph",
            data=json.dumps({"query": ""}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with pytest.raises(urllib.error.HTTPError) as http_err:
            urllib.request.urlopen(bad_req, timeout=5)
        assert http_err.value.code == 400
    finally:
        server.shutdown()
        server.server_close()
