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


def test_constraint_composer_failure_guesser_non_numeric_authority_and_calibration(tmp_path: Path) -> None:
    import http.server
    import threading
    import urllib.error
    import urllib.request

    from krystal_kernel.claims_lattice import EvidenceLatticePoint
    from krystal_kernel.constants import CapabilityScope
    from krystal_kernel.sovereign_openvino_oneapi_app import SovereignOpenVinoRequestHandler
    from projects.sovereign_openvino_companion_v2 import (
        AlternativeHorizonExplorer,
        CohesiveNonScalarAuthorityAndProbabilisticEngine,
        ConstraintNarrativeAndFactComposer,
        FailureRegistryAndRedundantCognitiveGuesser,
    )

    v2 = SovereignOpenVinoCompanionV2(log_dir=tmp_path, reset_ledger=True)
    composer = ConstraintNarrativeAndFactComposer()
    comp_res = composer.compose_facts_and_narrative(logger=v2.logger)
    assert len(comp_res["composed_facts"]) == 2
    assert len(comp_res["quarantined_events"]) == 1
    assert comp_res["quarantined_events"][0]["delta_s"] == 0
    assert comp_res["render_mutates_state"] is False
    assert len(comp_res["filtered_taxonomy_improvements"]) == 3

    guesser = FailureRegistryAndRedundantCognitiveGuesser()
    fail_res = guesser.register_failure_and_infer_redundant_guesses(logger=v2.logger)
    assert len(fail_res["failure_registry_entries"]) == 4
    assert all(g["can_self_certify"] is False for g in fail_res["redundant_cognitive_guesses"])
    assert all(g["epistemic_status"] == "HYPOTHESIS_PATTERN_CANDIDATE" for g in fail_res["redundant_cognitive_guesses"])

    alt_res = AlternativeHorizonExplorer().evaluate_horizon_alternatives()
    assert alt_res["frozen_theory_nodes_untouched"] == 56
    assert alt_res["node_57_added_to_frozen_theory"] is False
    assert all(a["p7_directional_gate_pass"] is True for a in alt_res["alternatives_ranked_by_evoi"])

    auth = CohesiveNonScalarAuthorityAndProbabilisticEngine()
    ok_gate = auth.check_non_numeric_authority(
        proposer_scope=CapabilityScope.CAN_PROPOSE,
        verifier_scope=CapabilityScope.CAN_VERIFY,
        lattice_point=EvidenceLatticePoint("IMPLEMENTED", "UNIT_TESTED", "CONTRACT_BOUND", "HOST_BENCHMARKED"),
        vital_hp=6,
    )
    assert ok_gate["authority_admissible"] is True
    assert ok_gate["is_scalar_weighted"] is False

    veto_self = auth.check_non_numeric_authority(
        proposer_scope=CapabilityScope.CAN_PROPOSE,
        verifier_scope=CapabilityScope.CAN_PROPOSE,
        lattice_point=EvidenceLatticePoint("IMPLEMENTED", "UNIT_TESTED", "CONTRACT_BOUND", "HOST_BENCHMARKED"),
        vital_hp=6,
    )
    assert veto_self["authority_admissible"] is False

    veto_gap = auth.check_non_numeric_authority(
        proposer_scope=CapabilityScope.CAN_PROPOSE,
        verifier_scope=CapabilityScope.CAN_VERIFY,
        lattice_point=EvidenceLatticePoint("IMPLEMENTED", "UNIT_TESTED", "CONTRACT_BOUND", "HOST_BENCHMARKED"),
        has_unestimated_critical_axis=True,
    )
    assert veto_gap["authority_admissible"] is False

    with pytest.raises(ContractViolation):
        auth.check_non_numeric_authority(
            CapabilityScope.CAN_PROPOSE,
            CapabilityScope.CAN_VERIFY,
            EvidenceLatticePoint("IMPLEMENTED", "UNIT_TESTED", "CONTRACT_BOUND", "HOST_BENCHMARKED"),
            vital_hp=5,
        )

    prob_ok = auth.evaluate_probabilistic_reasoning_patterns(authority_admissible=True, logger=v2.logger)
    assert prob_ok["sample_size_n"] == 25
    assert prob_ok["mean_brier_score"] < 0.20
    assert prob_ok["expected_calibration_error_ece"] < 0.15
    assert prob_ok["stdev_brier_score"] > 0.0
    assert prob_ok["reasoning_vs_cost_and_variance_inhibitor"]["healthy_variance_status"] == "HEALTHY_BOUNDED_VARIANCE"
    assert len(prob_ok["modular_challenge_graph_coordinates"]) == 4

    prob_veto = auth.evaluate_probabilistic_reasoning_patterns(authority_admissible=False)
    assert all(p["effective_commit_probability"] == 0.0 for p in prob_veto["pattern_posteriors"])

    comp = auth.calculate_speedup_hierarchy_and_provider_competition()
    assert len(comp["alternate_speedup_hierarchy"]) == 4
    assert len(comp["provider_and_manufacturer_profiles"]) == 5
    assert len(comp["objectives_and_model_friction_priorities"]) == 4
    assert comp["boundary_friction_dissipation"]["concept_name"] == "Boundary Friction Dissipation"
    assert len(comp["cross_model_quest_table"]["development_quests"]) == 4
    assert comp["extended_qualified_statistics"]["calibration_summary"]["sample_size_n"] == 25
    assert comp["extended_qualified_statistics"]["hardware_baseline_summary"]["evidence_classification"] == "LOCAL_MEASURED"

    choice_log = auth.evaluate_choice_perspective_and_speculative_theory_log(authority_admissible=True)
    assert len(choice_log["beneficial_agreement_log"]) == 2
    assert len(choice_log["speculative_driven_theory_projections"]) == 2
    assert choice_log["authoritative_state_mutated"] is False
    refer = auth.build_behavioral_structure_refer()
    assert refer["frozen_theory_nodes_untouched"] == 56
    assert len(refer["new_behavioral_pillars"]) == 4

    gov_conspect = auth.synthesize_custom_patterns_governor_and_parameter_conspect(authority_admissible=True)
    assert len(gov_conspect["beneficial_cost_and_numa_consent"]) == 3
    assert len(gov_conspect["socket_power_management_schemas"]) == 3
    assert len(gov_conspect["conspect_parameter_workaround_model"]["workarounds"]) == 3
    assert gov_conspect["characteristic_metrics_set"]["measured_local"]["ops_per_watt"] == 30.391

    rc = v2_cli_main(
        [
            "--compose-constraints",
            "--probabilistic-authority",
            "--artifact-tree-query",
            "--reset-ledger",
            "--log-dir",
            str(tmp_path / "cli_auth"),
        ]
    )
    assert rc == 0

    from projects.sovereign_openvino_companion_v2 import MarkdownBehavioralKnowledgeGraphModule
    tree_q = MarkdownBehavioralKnowledgeGraphModule().query_beneficial_artifacts_in_tree()
    assert tree_q["verified_artifacts_count"] == 8

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), SovereignOpenVinoRequestHandler)
    port = server.server_address[1]
    th = threading.Thread(target=server.serve_forever, daemon=True)
    th.start()
    try:
        with urllib.request.urlopen(
            f"http://127.0.0.1:{port}/api/v1/runtime/probabilistic_authority_composer",
            timeout=5,
        ) as resp:
            assert resp.status == 200
            body = json.loads(resp.read().decode("utf-8"))
            assert body["non_numeric_authority_gate"]["authority_admissible"] is True
            assert body["probabilistic_reasoning_calibration"]["sample_size_n"] == 25

        bad_req = urllib.request.Request(
            f"http://127.0.0.1:{port}/api/v1/runtime/probabilistic_authority_composer",
            data=json.dumps({"vital_hp": 5}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with pytest.raises(urllib.error.HTTPError) as http_err:
            urllib.request.urlopen(bad_req, timeout=5)
        assert http_err.value.code == 400
    finally:
        server.shutdown()
        server.server_close()


def test_morph_vision_os_microservices_sql_scales_and_sprint_kg(tmp_path: Path):
    import http.server
    import threading
    import urllib.error
    import urllib.request
    from krystal_kernel.constants import ContractViolation, FRAME_SIZE_BYTES, VITAL_MAX_HP
    from krystal_kernel.sovereign_openvino_oneapi_app import SovereignOpenVinoRequestHandler
    from projects.sovereign_openvino_companion_v2 import (
        AutonomousOsMicroserviceSynthesizer,
        BeneficialHardwareBehaviorConstitution,
        EmergentLayerQueryInjector,
        EpistemicSqlDataTypeScaleEngine,
        EpistemicVsGeneralTelemetryMapper,
        FourBucketPreemptionAndCpuGovernor,
        KrystalThermodynamicVisionFramework,
        LinuxNativeFusionAndTelemetryProjector,
        LogDrivenComponentMorpher,
        SovereignOpenVinoCompanionV2,
        VariableConditionedWeightNeuronModulator,
    )
    from projects.sovereign_openvino_companion_v2.cli import main as v2_cli_main

    v2 = SovereignOpenVinoCompanionV2(log_dir=tmp_path / "morph_sprint", reset_ledger=True)

    # 1. Log-Driven Component Morpher + Reversible Morph Lineage
    morpher = LogDrivenComponentMorpher()
    m1 = morpher.morph_from_logs(logger=v2.logger, vital_hp=VITAL_MAX_HP)
    assert len(m1["morphed_operators"]) == 4
    m2 = morpher.re_morph_to_state("STATE_V2_7_5W_VBLANK_SURVIVAL", vital_hp=VITAL_MAX_HP, logger=v2.logger)
    assert m2["chain_verified"] is True
    assert len(morpher.checkpoints) == 2
    cp_before = len(morpher.checkpoints)
    with pytest.raises(ContractViolation):
        morpher.re_morph_to_state("STATE_ILLEGAL_HP5", vital_hp=5)
    assert len(morpher.checkpoints) == cp_before  # ΔS == 0 on invalid HP

    # 2. Epistemic vs General Telemetry + 5 Subject-Observation Mapping Quantities
    tele = EpistemicVsGeneralTelemetryMapper().evaluate_telemetry_and_subject_quantities(logger=v2.logger)
    qty = tele["subject_observation_quantities"]
    assert qty["provenance_independence_rank"] == 3
    assert qty["observer_perspective_entropy_nats"] > 0.0
    assert qty["calibration_reliability_index"] == pytest.approx(0.8375, rel=1e-4)

    # 3. Four-Bucket Preemption & CPU Governance
    gov = FourBucketPreemptionAndCpuGovernor().quantify_four_buckets_and_govern(
        uma_load=0.75, temp_c=72.5, scanout_active=True, logger=v2.logger
    )
    assert gov["operating_point"]["selected_target_watts"] == 7.5
    assert gov["operating_point"]["selected_k_spec"] == 1
    assert len(gov["four_buckets"]) == 4
    assert all(item["delta_s"] == 0 for item in gov["four_buckets"]["REJECTED_NON_EXECUTED_PARASITE"])

    # 4. Linux Native Projections (anti-promotion) + Code GENE & Unified Simulator Bridges + Constitution
    proj = LinuxNativeFusionAndTelemetryProjector().project_linux_deployment_and_bridges()
    assert proj["linux_native_projected_gains"]["classification"] == "ARCHITECTURAL_DERIVATION_PENDING_LINUX_CDT"
    assert proj["code_gene_raster_bridge"]["bounded_cage_volume_m3"] == 215.6
    assert proj["unified_simulator_bridge"]["render_mutates_state"] is False
    const = BeneficialHardwareBehaviorConstitution().compile_constitution()
    assert const["keyword_count"] == 10

    # 5. Emergent Layer Query Injector with Per-Branch Isolated SHA-256 Log
    inj = EmergentLayerQueryInjector()
    inj_out = inj.execute_cloned_branches(
        query="Compare same query across multiple lenses",
        environment_id="ENV_TEST_V2",
        logger=v2.logger,
    )
    assert inj_out["distinct_decisions_count"] == 4
    assert inj_out["distinct_leaf_hashes_count"] == 4
    for br in inj_out["branches"]:
        assert br["base_layer_digest_sha256"] == inj_out["base_layer_digest_sha256"]
        assert br["branch_log_verified"] is True
        assert inj.verify_branch_log(br["branch_log"]) is True

    # 6. Variable-Conditioned Weight & Neuron Modulator (W_0 immutable)
    mod = VariableConditionedWeightNeuronModulator()
    vec = [0.8, -0.4, 0.5, 0.3, -0.2, 0.7, 0.1, 0.6]
    r15 = mod.forward_with_variable_z(vec, z=0.15)
    r50 = mod.forward_with_variable_z(vec, z=0.50)
    r85 = mod.forward_with_variable_z(vec, z=0.85)
    assert r15["base_weights_mutated"] is False
    assert r15["base_weights_sha256"] == r50["base_weights_sha256"] == r85["base_weights_sha256"]
    assert r15["effective_weight_l1_norm"] != r85["effective_weight_l1_norm"]
    assert r15["output_vector"] != r85["output_vector"]
    with pytest.raises(ContractViolation):
        mod.forward_with_variable_z(vec, z=1.5)
    with pytest.raises(ContractViolation):
        mod.forward_with_variable_z(vec, z=0.5, vital_hp=5)

    # 7. Thermodynamic Computer Vision Framework (128B tiles + P5 0.0180 mJ skip + Code GENE)
    vis = KrystalThermodynamicVisionFramework().process_pixel_stream(
        width=16, height=8, conditioning_z=0.65, logger=v2.logger
    )
    assert vis["frame_bytes"] == FRAME_SIZE_BYTES == 128
    assert vis["tiles_total"] == 8
    assert vis["tiles_skipped_p5_cutoff"] >= 2
    assert vis["tiles_executed"] >= 2
    assert vis["render_mutates_state"] is False
    assert vis["proposer_role"] != vis["verifier_role"]

    # 8. Autonomous OS Microservice Synthesizer (Zero-Touch P1)
    auto_os = AutonomousOsMicroserviceSynthesizer().synthesize_and_observe_microservices(logger=v2.logger)
    assert auto_os["existing_source_delta_bytes"] == 0
    assert auto_os["host_os_mutated"] is False
    assert auto_os["proposer_role"] != auto_os["verifier_role"]
    assert len(auto_os["synthesized_microservices"]) == 4
    assert all(m["artifact_gate_receipt"]["proposer_neq_verifier"] for m in auto_os["synthesized_microservices"])

    # 9. Epistemic SQL Data-Type Scale Engine (real SQLite STRICT + triggers)
    sql_engine = EpistemicSqlDataTypeScaleEngine()
    sql_low = sql_engine.compare_conventional_vs_epistemic_sql_patterns(scale_influence_z=0.10, logger=v2.logger)
    sql_high = sql_engine.compare_conventional_vs_epistemic_sql_patterns(scale_influence_z=0.95, logger=v2.logger)
    assert sql_low["vital_hp_trigger_verified"] is True
    assert sql_low["anti_promotion_trigger_verified"] is True
    assert sql_low["sql_delta_s_on_violation"] == 0
    assert len(sql_low["pattern_comparison"]["krystal_epistemic_data_type_scale_sql_app"]["five_data_type_scales"]) == 5
    bands_low = {r["claim_id"]: r["governed_scale_band"] for r in sql_low["classified_rows"]}
    bands_high = {r["claim_id"]: r["governed_scale_band"] for r in sql_high["classified_rows"]}
    assert bands_low["CLM_UMA_PREBRAKE"] == "PREEMPT_SWEETSPOT_12W"
    assert bands_high["CLM_UMA_PREBRAKE"] == "STANDARD_ADMISSIBLE_12W"
    assert bands_low["CLM_PARASITE_SPIN"] == "QUARANTINE_PARASITE_0_018MJ"

    # 10. Full Orchestrator + CLI + HTTP Endpoint
    orch_out = v2.run_morph_vision_and_sprint_kg_orchestration()
    assert orch_out["dag_cycles"] == 0
    assert orch_out["frozen_theory_nodes_untouched"] == 56
    assert orch_out["chain_valid"] is True
    assert len(orch_out["sprint_milestone_knowledge_graph"]["nodes"]) == 5
    rev_calc = orch_out["reverse_engineering_residual_calculator"]
    assert rev_calc["chip_capacity_overflow_policy"]["excess_events_hz"] == 2400.0
    assert rev_calc["chip_capacity_overflow_policy"]["chip_overheat_prevented"] is True
    assert len(rev_calc["missing_unmeasured_parts"]) == 2
    assert all(p["status"] == "UNKNOWN_NOT_MEASURED" for p in rev_calc["missing_unmeasured_parts"])
    rubik = orch_out["rubik_combinatorial_theory_branching"]
    assert rubik["unpruned_cartesian_states"] == 4096
    assert rubik["computable_admissible_frontier_states"] == 5
    assert all(f["convex_averaged"] is False for f in rubik["combinatorial_forks"])
    arb = orch_out["adaptive_scenario_arbitration_layer"]
    assert arb["selected_scenario"] == "SCENARIO_PEAK_12W_SWEETSPOT_VISION"
    assert arb["blind_scalar_prediction_blocked"] is True
    assert len(arb["differentiation_microservices"]) == 3
    sem_lib = orch_out["adaptive_datatype_differential_and_semantic_library"]
    assert sem_lib["cross_domain_sql_delta_s_on_violation"] == 0
    assert sem_lib["cross_domain_anti_self_cert_verified"] is True
    assert len(sem_lib["morphed_semantic_library"]) == 6
    assert sem_lib["differential_classifier_probes"]["interlock_hp"]["classified_scale"] == "CONSTITUTIONAL_INVARIANT_FIXED"
    assert sem_lib["differential_classifier_probes"]["joule_draw_mj"]["classified_scale"] == "RATIO_THERMODYNAMIC_PHYSICAL"

    rc = v2_cli_main(
        [
            "--sprint-kg-orchestrate",
            "--thermodynamic-vision",
            "--autonomous-os-microservices",
            "--epistemic-sql-scales",
            "--conditioning-z",
            "0.75",
            "--scale-influence-z",
            "0.25",
            "--reset-ledger",
            "--log-dir",
            str(tmp_path / "cli_sprint_kg"),
        ]
    )
    assert rc == 0

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), SovereignOpenVinoRequestHandler)
    port = server.server_address[1]
    th = threading.Thread(target=server.serve_forever, daemon=True)
    th.start()
    try:
        with urllib.request.urlopen(
            f"http://127.0.0.1:{port}/api/v1/runtime/morph_vision_sprint_kg?z=0.65&sql_z=0.50",
            timeout=5,
        ) as resp:
            assert resp.status == 200
            body = json.loads(resp.read().decode("utf-8"))
            assert body["dag_cycles"] == 0
            assert body["frozen_theory_nodes_untouched"] == 56
            assert body["autonomous_os_microservices"]["host_os_mutated"] is False

        ok_post = urllib.request.Request(
            f"http://127.0.0.1:{port}/api/v1/runtime/morph_vision_sprint_kg",
            data=json.dumps({"vital_hp": 6, "conditioning_z": 0.75, "scale_influence_z": 0.40}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(ok_post, timeout=5) as resp_ok:
            assert resp_ok.status == 200
            post_body = json.loads(resp_ok.read().decode("utf-8"))
            assert post_body["chain_valid"] is True

        bad_post = urllib.request.Request(
            f"http://127.0.0.1:{port}/api/v1/runtime/morph_vision_sprint_kg",
            data=json.dumps({"vital_hp": 5}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with pytest.raises(urllib.error.HTTPError) as http_err:
            urllib.request.urlopen(bad_post, timeout=5)
        assert http_err.value.code == 400

        with urllib.request.urlopen(f"http://127.0.0.1:{port}/blender_vray_ui", timeout=5) as ui_resp:
            assert ui_resp.status == 200
            assert b"Krystal Stack // Blender" in ui_resp.read()

        with urllib.request.urlopen(f"http://127.0.0.1:{port}/vision_propedeutics_video", timeout=5) as vid_resp:
            assert vid_resp.status == 200
            assert b"Propedeutika" in vid_resp.read()

        with urllib.request.urlopen(f"http://127.0.0.1:{port}/kernel_eshop", timeout=5) as eshop_ui:
            assert eshop_ui.status == 200
            assert b"__KRYSTAL_FULLSTACK_VARS__" in eshop_ui.read()

        with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/v1/runtime/rss_feed.xml", timeout=5) as rss_resp:
            import xml.etree.ElementTree as ET
            assert rss_resp.status == 200
            rss_root = ET.fromstring(rss_resp.read().decode("utf-8"))
            assert rss_root.tag == "rss"
            assert len(rss_root.findall("./channel/item")) == 4

        for route in (
            "/api/v1/runtime/blender_vray_sdk_pipeline",
            "/api/v1/runtime/inter_pause_cognitive_parser",
            "/api/v1/runtime/back_office_prompt_spectrum",
            "/api/v1/runtime/airport_b2b_cpp_sdk",
            "/api/v1/runtime/core_processes_and_constellations",
            "/api/v1/runtime/procedural_svg_cache_governor",
            "/api/v1/runtime/provenance_marketing_zones",
            "/api/v1/runtime/kernel_eshop",
            "/api/v1/runtime/adaptive_vision_usb_swap",
            "/api/v1/runtime/oxygen_hybrid_mesh_crud",
            "/api/v1/runtime/process_supervisor_inspector",
        ):
            with urllib.request.urlopen(f"http://127.0.0.1:{port}{route}", timeout=5) as ep_resp:
                assert ep_resp.status == 200
                assert json.loads(ep_resp.read().decode("utf-8"))["vital_max_hp"] == 6

        eshop_post_ok = urllib.request.Request(
            f"http://127.0.0.1:{port}/api/v1/runtime/kernel_eshop",
            data=json.dumps(
                {
                    "sku": "KRYSTAL-SERVER-CLUSTER",
                    "scale_key": "RATIO_SPECTRAL_TENSOR",
                    "contributor_name": "Dušan Kopecký",
                    "github_handle": "RhodosBerger",
                }
            ).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(eshop_post_ok, timeout=5) as eshop_res:
            assert eshop_res.status == 200
            assert json.loads(eshop_res.read().decode("utf-8"))["identity_and_access_audit"]["github_handle"] == "RhodosBerger"

        eshop_post_bad = urllib.request.Request(
            f"http://127.0.0.1:{port}/api/v1/runtime/kernel_eshop",
            data=json.dumps({"contributor_name": "Unauthorized Rogue", "github_handle": "rogue"}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with pytest.raises(urllib.error.HTTPError) as eshop_err:
            urllib.request.urlopen(eshop_post_bad, timeout=5)
        assert eshop_err.value.code == 400

        bv_post = urllib.request.Request(
            f"http://127.0.0.1:{port}/api/v1/runtime/blender_vray_sdk_pipeline",
            data=json.dumps({"vital_hp": 6, "resolution_x": 1280, "resolution_y": 720, "vray_samples": 32}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(bv_post, timeout=5) as bv_ok:
            assert bv_ok.status == 200
            assert json.loads(bv_ok.read().decode("utf-8"))["pipeline_inputs"]["resolution"] == "1280x720"

        bv_bad = urllib.request.Request(
            f"http://127.0.0.1:{port}/api/v1/runtime/blender_vray_sdk_pipeline",
            data=json.dumps({"vital_hp": 5}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with pytest.raises(urllib.error.HTTPError) as bv_err:
            urllib.request.urlopen(bv_bad, timeout=5)
        assert bv_err.value.code == 400
    finally:
        server.shutdown()
        server.server_close()


def test_reversible_matrix_binary_correlator_multipole_inspectors_and_cruise_scheduler(tmp_path: Path) -> None:
    import http.server
    import json
    import threading
    import urllib.error
    import urllib.request

    from krystal_kernel.constants import ContractViolation, VITAL_MAX_HP
    from krystal_kernel.sovereign_openvino_oneapi_app import SovereignOpenVinoRequestHandler
    from projects.sovereign_openvino_companion_v2 import (
        AdaptiveVisionUsbSwapAndFileMatrixEngine,
        AirportB2BCppCompiledSdkSuite,
        BackOfficeHierarchyPromptSpectrumGovernor,
        BehavioralDeterminismAndCrossSystemMatrix,
        BinaryPatternCorrelationTool,
        BinarySetMatrixEngine,
        BlenderVRaySdkPipelineAndIsaCacheBridge,
        CadenceDictionLogarithmicCoreEngine,
        CombinatorialWiringAndOscillationMatrixSuite,
        CoreProcessesAndProductionConstellations,
        CrossParadigmCppTranspilerAndCognitionStandardsEngine,
        DeterministicSocketAndThreadFrequencyArchitect,
        DeterministicRelationalCrudDatabaseFlow,
        DirectionValidationAndEmergingKnowledgeMapper,
        EpistemicMemoryPatternTransducerAndFourScaleEvaluator,
        ExplorationParameterBenchmarkRunner,
        FourChipsetIsaHybridUxAndMeshSuite,
        Gamesa3DGridVulkanStrategicParserEngine,
        GeneticLineageTestMethodologyEnhancer,
        GpuVisualMemoryCoordinateAndFieldDefragmenter,
        HexadimensionalLogInterpreterAndPowerMorpher,
        IntenseThreadBurstMatrixViewBinder,
        InterPauseLatentInvariantParser,
        KernelEshopAndContributorLicensingEngine,
        MeasuredPropertyProfileAndLexiconMethodology,
        MorseHardwareStateParser,
        MultipoleLatticeInferenceAndControllabilityEngine,
        OperationalProcessSupervisorInspector,
        ProceduralSvgCacheGovernor,
        ProvenanceOriginAndCommercialStrategyEngine,
        RealtimeInspectorScoringGovernor,
        ReversibleCorrelationAndCruiseOrchestrator,
        SelfAnalyzingGeometricSyncScaleEngine,
        SovereignOpenVinoCompanionV2,
        SpatialInvariantAdaptiveCruiseScheduler,
        TectonicMeshMemoryGearboxAndUnitComparator,
        UxPrivilegeMappingFramework,
        VisionLanguagePropaedeuticsVideoExporter,
        VulkanEndpointExtensionRuntime,
        VulkanMemoryAdaptabilityLibrary,
        ZeroCopySocketObservationProcess,
        render_kernel_eshop_html,
    )
    from projects.sovereign_openvino_companion_v2.cli import main as v2_cli_main

    # 1. Involutive 6x6 matrix & cross-system reversibility + negative ContractViolation on wrong dimension
    mat_engine = BehavioralDeterminismAndCrossSystemMatrix()
    mat_out = mat_engine.evaluate_cross_system_reversibility()
    assert mat_out["is_reversible"] is True
    assert mat_out["matrix_involution_error"] <= 1e-12
    assert mat_out["roundtrip_max_abs_error"] <= 1e-5
    with pytest.raises(ContractViolation):
        mat_engine.transform_a_to_b([1.0, 2.0])

    # 2. 128B 'KCOR' frame encode/decode/correlate + CRC32/SHA-256 tamper rejection
    corr = BinaryPatternCorrelationTool()
    f_a = corr.encode_pattern_frame_128b(mat_out["state_a"], capability_mask=4, epoch=7, system_tag=1)
    assert len(f_a) == 128
    assert corr.decode_and_verify_frame_128b(f_a)["verified"] is True
    tampered = bytearray(f_a)
    tampered[20] ^= 0xFF
    with pytest.raises(ContractViolation):
        corr.decode_and_verify_frame_128b(bytes(tampered))

    # 3. Measured profiles & 6-step lexicon + anti-promotion
    prof = MeasuredPropertyProfileAndLexiconMethodology().evaluate_profiles_and_methodology()
    assert prof["lexicon_step_count"] == 6
    assert prof["anti_promotion_enforced"] is True
    assert prof["profiles"][-1]["evidence_level"] == "ARCHITECTURAL_DERIVATION_PENDING_LINUX_CDT"

    # 4. Multipole lattice (no scalar collapse) + anti-promotion join when verifier_clearance=False
    mp_unverified = MultipoleLatticeInferenceAndControllabilityEngine().evaluate_lattice_and_multipole(
        verifier_clearance=False
    )
    assert mp_unverified["scalar_collapsed"] is False
    assert mp_unverified["guarded_lattice_join"]["measurement"] == "PENDING"
    assert len(mp_unverified["multipole_poles"]) == 6

    # 5. Real-time inspector scoring governor (bounds + rejection gates)
    gov = RealtimeInspectorScoringGovernor()
    gov_ok = gov.evaluate_inspectors(proposer_id="AGENT_A", verifier_id="AGENT_B", evoi_per_mj=0.145, observed_latency_ms=2.7420)
    assert 0.0 < gov_ok["governor_score"] <= 100.0
    assert gov_ok["delta_s"] == 1
    assert gov_ok["scored_throughput_gain_pct"] > 0.0
    gov_bad_sep = gov.evaluate_inspectors(proposer_id="SAME", verifier_id="SAME")
    assert gov_bad_sep["governor_score"] == 0.0 and gov_bad_sep["delta_s"] == 0
    gov_bad_evoi = gov.evaluate_inspectors(evoi_per_mj=0.0100)
    assert gov_bad_evoi["governor_score"] == 0.0 and gov_bad_evoi["delta_s"] == 0

    # 6. Spatial-invariant ACC cruise scheduler + hard boundary ContractViolation
    acc = SpatialInvariantAdaptiveCruiseScheduler()
    acc_out = acc.run_cruise_profile()
    assert acc_out["cage_volume_m3"] == 215.6
    assert len(acc_out["segments"]) == 5
    assert acc_out["invariant_collisions"] == 0
    with pytest.raises(ContractViolation):
        acc.check_spatial_point(3.6, 0.0, 0.0)

    # 7. Cadence/Diction logarithmic core & UX privilege framework
    cad_out = CadenceDictionLogarithmicCoreEngine().evaluate_logarithmic_core()
    assert cad_out["source_mode"] == "AUTOMATIC_CADENCE_DICTION"
    assert cad_out["target_mode"] == "SEMI_AUTOMATIC_CADENCE"
    assert cad_out["diction_polarity"] == +1
    assert cad_out["cadence_polarity"] == -1
    assert cad_out["higher_telemetry_knowledge_alone_grants_authority"] is False

    ux_out = UxPrivilegeMappingFramework().evaluate_role_matrix()
    assert ux_out["role_count"] == 5
    assert ux_out["telemetry_escalation_blocked"] is True
    assert ux_out["escalation_delta_s"] == 0

    # 8. Hexadimensional log interpreter & power morpher (binary 1 -> 0.5 at max drop)
    hex_out = HexadimensionalLogInterpreterAndPowerMorpher().concentrate_and_morph_logs()
    assert hex_out["dimension_count"] == 6
    assert hex_out["target_half_recorded_state"] == 0.5
    assert hex_out["reached_half_state_at_min_power"] is True
    assert hex_out["max_power_drop_morphed_state"] == 0.5

    # 8b. Direction validation, Socket/Thread governance, Epistemic Memory Transducer, Exploration Benchmarks, Tectonic Gearbox
    dir_out = DirectionValidationAndEmergingKnowledgeMapper().evaluate_direction_and_horizons()
    assert dir_out["directional_test_metrics"]["direction_valid"] is True
    assert len(dir_out["emerging_knowledge_horizons"]) == 4
    assert all(not h["is_local_measured"] for h in dir_out["emerging_knowledge_horizons"])

    sock_out = DeterministicSocketAndThreadFrequencyArchitect().design_socket_and_thread_governance()
    assert sock_out["zero_copy_lane_bytes"] == 128
    assert sock_out["thread_switching_governance"]["switch_reduction_12w_pct"] == 95.0

    mem_out = EpistemicMemoryPatternTransducerAndFourScaleEvaluator().evaluate_four_scales_and_transducer()
    assert set(mem_out["four_measurement_scales"].keys()) == {"NOMINAL", "ORDINAL", "INTERVAL", "RATIO"}
    assert mem_out["transducer_batch"]["unauthorized_mutations"] == 0
    assert mem_out["transducer_batch"]["deduplicated_hits"] >= 1

    bench_out = ExplorationParameterBenchmarkRunner().run_exploration_benchmarks(trials=24)
    assert bench_out["trials"] == 24
    for stat_key in ("median_us", "p95_us", "p99_us", "stdev_us"):
        assert stat_key in bench_out["governed_timing_us"]
    assert bench_out["memory_bytes_saved_pct"] > 50.0
    assert bench_out["energy_saved_pct"] > 80.0
    assert bench_out["switch_reduction_pct"] == 95.0

    tect_engine = TectonicMeshMemoryGearboxAndUnitComparator()
    tect_out = tect_engine.evaluate_tectonic_mesh_gearbox()
    assert tect_out["mesh_plate_count"] == 4
    assert tect_out["adaptive_memory_gearbox_active"] is True
    assert len(tect_out["phase_fingerprints"]) == 4
    with pytest.raises(ContractViolation):
        tect_engine.convert_unit(12.0, "W", "m")

    # 8c. Vulkan endpoint extension runtime & higher-version kernel socket zero-copy observer
    vk_out = VulkanEndpointExtensionRuntime().run_vulkan_extension_mapping()
    assert vk_out["zero_touch_source_delta_bytes"] == 0
    assert vk_out["endpoint_count"] == 5
    assert vk_out["all_bound_kernels_present"] is True

    obs_out = ZeroCopySocketObservationProcess().observe_gearbox_vs_divergent_differential()
    assert obs_out["version_strictly_higher"] is True
    assert obs_out["observer_write_authority"] is False
    assert obs_out["low_record_signal_count"] == 4
    assert all(tr["low_record_signal"] and tr["observer_delta_s"] == 0 for tr in obs_out["traces"])

    gen_out = GeneticLineageTestMethodologyEnhancer().evaluate_genetic_test_methodology()
    assert gen_out["fate_certainty_fabricated"] is False
    assert gen_out["unknown_loci_quarantined"] == 1
    assert gen_out["p6_conflict_branches_preserved"] == 1
    assert gen_out["uma_bus_contention_eliminated"] is True

    # 8d. GPU visual-memory defragmenter, intense thread-burst binder, Binary Set Matrices, Geometric Sync Scales
    gpu_defrag = GpuVisualMemoryCoordinateAndFieldDefragmenter()
    defrag_out = gpu_defrag.project_defragment_and_name_processes()
    assert defrag_out["external_fragmentation_bytes"] == 0
    assert defrag_out["compaction_saved_bytes"] == 256
    assert len(defrag_out["named_active_processes"]) == 4
    with pytest.raises(ContractViolation):
        gpu_defrag.project_defragment_and_name_processes([{"tile_id": "BAD_OOB", "u": 1.25, "v": 0.5, "d": 0.5, "evoi_per_mj": 0.14}])

    burst_out = IntenseThreadBurstMatrixViewBinder().bind_views_under_intense_burst()
    assert burst_out["bound_view_count"] == 4
    assert burst_out["coalesced_switch_reduction_pct"] == 98.75
    assert burst_out["view_synchronization_drift"] == 0.0

    bsm = BinarySetMatrixEngine()
    bsm_out = bsm.evaluate_binary_set_matrices()
    assert bsm_out["idempotent_union_ok"] is True
    assert bsm_out["idempotent_intersection_ok"] is True
    assert bsm_out["double_transpose_exact"] is True
    assert bsm_out["illegal_self_cert_cells"] == 0
    bad_mat = [[2 if (r == 0 and c == 0) else 0 for c in range(6)] for r in range(6)]
    with pytest.raises(ContractViolation):
        bsm.validate_binary_matrix(bad_mat)

    geom_out = SelfAnalyzingGeometricSyncScaleEngine().evaluate_geometric_scales_and_loop()
    assert geom_out["strata_count"] == 5
    assert geom_out["loop_step_count"] == 6
    assert 0.0 <= geom_out["main_spectrum_coherence"] <= 1.0

    cpp_std_out = CrossParadigmCppTranspilerAndCognitionStandardsEngine().evaluate_cpp_gc_elimination_and_standards()
    assert cpp_std_out["gc_mode"] == "DETERMINISTIC_ZERO_GC_EPOCH_ARENA"
    assert cpp_std_out["standard_count"] == 5
    assert len(cpp_std_out["cognition_governance_schemas"]) == 4

    morse_parser = MorseHardwareStateParser()
    morse_out = morse_parser.evaluate_morse_roundtrip()
    assert morse_out["roundtrip_exact"] is True
    assert morse_out["swap_zero_preserved"] is True
    with pytest.raises(ContractViolation):
        morse_parser.parse_morse_to_hardware_image("........")

    osc_suite = CombinatorialWiringAndOscillationMatrixSuite()
    osc_out = osc_suite.evaluate_combinatorial_wiring_and_oscillation_matrices()
    assert osc_out["admissible_scheme_count"] == 6
    assert osc_out["combinatorial_pruning_pct"] > 99.0
    assert osc_out["max_spectral_row_norm"] < 1.0
    assert osc_out["monotonic_damping_ok"] is True
    assert osc_out["strictly_ascending_lanes"] is True
    assert len(osc_out["trigonometric_chipset_models"]) == 4
    assert osc_out["trig_unit_circle_error_max"] < 1e-4
    with pytest.raises(ContractViolation):
        osc_suite.evaluate_combinatorial_wiring_and_oscillation_matrices(vital_hp=5)

    gamesa_vk = Gamesa3DGridVulkanStrategicParserEngine()
    vk_out = gamesa_vk.evaluate_gamesa_3d_grid_and_vulkan_parser()
    assert vk_out["frozen_theory_nodes_match"] is True
    assert vk_out["combinatorial_basis_sum"] == 56
    assert vk_out["all_roundtrips_reversible_exact"] is True
    assert len(vk_out["four_sdk_adoption_domains"]) == 4
    assert len(vk_out["four_api_sdk_hierarchical_properties"]) == 4
    assert len(vk_out["unified_process_flow_matrix"]) == 4
    assert len(vk_out["airport_ops_sdk_blueprint"]["submodules"]) == 4
    glob_agg = vk_out["global_hardware_2d_25d_3d_vulkan_aggregates"]
    assert glob_agg["taxonomy_evidence_status"] == "SIMULATED_GLOBAL_HW_TAXONOMY"
    assert set(glob_agg["dimensional_parametric_scales"].keys()) == {"2D", "2.5D", "3D"}
    assert len(glob_agg["vulkan_abstract_kernel_aggregates"]) == 4
    assert glob_agg["manhattan_l1_span_cells"] > 0
    assert len(glob_agg["compiled_virtual_pixel_charsets"]) == 6
    with pytest.raises(ContractViolation):
        gamesa_vk.parse_vulkan_plan_reversible("BAD|TOKEN")

    bv_pipe = BlenderVRaySdkPipelineAndIsaCacheBridge()
    bv_out = bv_pipe.evaluate_pipeline()
    assert len(bv_out["isa_portability_cache_tiers"]) == 5
    assert len(bv_out["deduced_alternative_approaches"]) == 3
    with pytest.raises(ContractViolation):
        bv_pipe.evaluate_pipeline(uma_budget=0.90)

    ip_parser = InterPauseLatentInvariantParser()
    ip_out = ip_parser.parse_inter_pause_state()
    assert ip_out["latent_invariants"]["all_invariants_valid"] is True
    assert ip_out["eight_directional_vector_flow"]["direction_count"] == 8
    assert ip_out["eight_directional_vector_flow"]["closed_loop_net_displacement"] == [0, 0]
    with pytest.raises(ContractViolation):
        ip_parser.parse_inter_pause_state(scanout_active=True)

    bo_gov = BackOfficeHierarchyPromptSpectrumGovernor()
    assert bo_gov.evaluate_back_office_hierarchy()["proposer_verifier_separation_enforced"] is True
    b2b_out = AirportB2BCppCompiledSdkSuite().evaluate_b2b_sdk()
    assert b2b_out["module_count"] == 4
    const_out = CoreProcessesAndProductionConstellations().evaluate()
    assert len(const_out["captured_core_processes"]) == 5
    assert len(const_out["production_constellations"]) == 4

    svg_gov = ProceduralSvgCacheGovernor()
    svg_out = svg_gov.evaluate()
    assert svg_out["replay_from_cache"] is True
    assert svg_out["compression_saved_pct"] > 50.0
    assert svg_out["svg_scene_markup"].startswith("<svg ")
    with pytest.raises(ContractViolation):
        svg_gov.evaluate(uma_precache_budget_ratio=0.90)

    vid_exp = VisionLanguagePropaedeuticsVideoExporter()
    avi_meta = vid_exp.generate_avi_video(tmp_path / "propedeutics.avi")
    assert avi_meta["num_frames"] == 12
    assert (tmp_path / "propedeutics.avi").exists()
    assert (tmp_path / "propedeutics.avi").read_bytes()[:4] == b"RIFF"

    prov_eng = ProvenanceOriginAndCommercialStrategyEngine()
    prov_out = prov_eng.evaluate()
    assert prov_out["authorized_origin_audit"]["authorized"] is True
    assert prov_out["rust_bytecode_crate"]["toml_present"] is True
    assert prov_out["rust_bytecode_crate"]["lib_present"] is True
    assert len(prov_out["four_empirical_telemetry_zones"]) == 4
    assert len(prov_out["five_recordable_telemetry_layers"]) == 5
    assert len(prov_out["four_new_horizons"]) == 4
    assert len(prov_out["five_gpl_acquisition_interest_domains"]) == 5
    assert len(prov_out["commercial_phase1_marketing_domains"]) == 4
    with pytest.raises(ContractViolation):
        prov_eng.evaluate(origin_url="https://github.com/untrusted/clone")

    eshop_eng = KernelEshopAndContributorLicensingEngine()
    eshop_out = eshop_eng.evaluate()
    assert eshop_out["vital_max_hp"] == 6
    assert eshop_out["identity_and_access_audit"]["primary_author"] == "Dušan Kopecký"
    assert eshop_out["identity_and_access_audit"]["github_handle"] == "RhodosBerger"
    assert len(eshop_out["speculative_multithread_precache"]["precomputed_branches"]) == 4
    assert eshop_out["github_license_quotas"]["sovereign_author_licenses"] == 1
    assert eshop_out["fullstack_compiled_variables"]["krystal_github_handle"] == "RhodosBerger"
    assert len(eshop_out["pattern_equity_acquisition_catalog"]) == 5
    coauthor_out = eshop_eng.evaluate(contributor_name="Ivan Novikov", github_handle="", author_agreement_signed=True)
    assert coauthor_out["identity_and_access_audit"]["role"] == "SIGNED_CO_AUTHOR_DELEGATE"
    with pytest.raises(ContractViolation):
        eshop_eng.evaluate(contributor_name="Ivan Novikov", github_handle="", author_agreement_signed=False)
    with pytest.raises(ContractViolation):
        eshop_eng.evaluate(contributor_name="Unauthorized Outsider", github_handle="rogue")
    eshop_html = render_kernel_eshop_html(eshop_out)
    assert "window.__KRYSTAL_FULLSTACK_VARS__" in eshop_html

    usb_eng = AdaptiveVisionUsbSwapAndFileMatrixEngine()
    usb_out = usb_eng.evaluate()
    assert usb_out["vital_max_hp"] == 6
    assert len(usb_out["five_adaptive_schemes"]["schemes"]) == 5
    assert usb_out["linux_usb_swap_media"]["magic_header"] == "KRYSTAL_SWAPIMG_V1"
    assert usb_out["linux_usb_swap_media"]["image_size_bytes"] == 4096
    assert usb_out["utf8_ascii_redundant_compiler"]["all_rows_lossless_decoded"] is True
    assert usb_out["eight_css_state_vector_scripts"]["script_count"] == 8
    assert usb_out["eight_css_state_vector_scripts"]["all_fallback_self_verified"] is True
    assert usb_out["krystal_file_format_and_frontend_runtime"]["file_magic"] == "KRYSTAL_FILE_V1"
    assert len(usb_out["four_knowledge_materialization_domains"]) == 4
    assert (
        "4.19.183"
        in usb_out["unified_isa_and_asus_rt_ax58u_v2"]["asus_rt_ax58u_v2_kernel_options"]["recommended_custom_firmware_and_kernel"]["linux_kernel_version"]
    )
    assert len(usb_out["cybersecurity_cross_channel_and_compliance"]["krystal_cybersecurity_protocols"]) == 4
    with pytest.raises(ContractViolation):
        usb_eng.evaluate(uma_utilization=0.92)
    with pytest.raises(ContractViolation):
        usb_eng.evaluate(allow_unpermitted_rf_mutation=True)
    with pytest.raises(ContractViolation):
        usb_eng.evaluate(github_registered=False)

    manifest_md = Path("docs/architecture/KRYSTAL_UNIVERSAL_AGENT_KERNEL_MANIFEST.md")
    assert manifest_md.exists()
    manifest_text = manifest_md.read_text(encoding="utf-8")
    assert "Spec-K6" in manifest_text and "AGENT_PROMPT_06_CYBERSECURITY_ZERO_MUTATION" in manifest_text

    ox_suite = FourChipsetIsaHybridUxAndMeshSuite()
    ox_out = ox_suite.evaluate()
    assert len(ox_out["four_chipset_isa_hybrids"]) == 4
    assert ox_out["adaptive_oxygen_composer"]["composer_name"] == "KrystalAdaptiveOxygenComposerV1"
    assert ox_out["symbol_word_task_analysis"]["analogy_in_constants"]["constants"]["FRAME_SIZE_BYTES"] == 128
    assert len(ox_out["ordinal_ram_frequency_synthesizer"]["four_key_elements"]) == 4
    assert ox_out["ordinal_ram_frequency_synthesizer"]["interpolated_ordinal_slots"][-1]["speculative_freq_hz"] == 240.0
    assert len(ox_out["discovered_instruction_sets"]["new_instruction_sets_catalog"]) == 4
    assert len(ox_out["four_constitutional_symbolic_domains"]) == 4

    crud_db = DeterministicRelationalCrudDatabaseFlow(tmp_path / "krystal_crud.db")
    crud_res = crud_db.run_crud_roundtrip_demo()
    assert crud_res["post_cascade_counts"]["nodes"] == 1
    assert crud_res["post_cascade_counts"]["edges_cascaded_to_zero"] == 0
    assert crud_res["read_after_update_seq"] == 2
    assert crud_res["auditable_language_metrics"]["metric_constructs"]["metric_hash_chain_continuity_ratio"] == 1.0
    assert len(crud_res["auditable_language_metrics"]["auditable_language_log_entries"]) >= 4
    with pytest.raises(ContractViolation):
        crud_db.create_node("BAD_HP", "TEST", {}, vital_hp=5)
    assert crud_db.conn.execute("SELECT COUNT(*) FROM krystal_nodes").fetchone()[0] == 1
    with pytest.raises(ContractViolation):
        crud_db.link_nodes("OX_HYBRID_ROOT", "NON_EXISTENT", "BAD_FK")

    proc_insp = OperationalProcessSupervisorInspector().inspect()
    assert proc_insp["vital_max_hp"] == 6
    assert proc_insp["live_process_snapshot"]["pid"] > 0
    assert len(proc_insp["five_connection_schemas"]) == 5
    assert proc_insp["b2b_sdk_pricing_taxonomy"]["total_annual_integration_eur"] == 16200
    assert proc_insp["crypto_ml_crowdfunding_and_edu"]["btc_strategy_ml_model"]["normalized_strategy_score"] > 0.0
    assert OperationalProcessSupervisorInspector.compute_b2b_sdk_pricing_taxonomy(commercial_license=False)["total_eur"] == 0

    vk_lib_out = VulkanMemoryAdaptabilityLibrary.evaluate_memory_operations()
    assert vk_lib_out["vital_max_hp"] == 6
    assert len(vk_lib_out["four_vulkan_memory_paradigms"]) == 4
    geom_synth = vk_lib_out["deterministic_geometry_and_render_synthesis"]
    assert len(geom_synth["five_public_visibility_endpoints"]) == 5
    assert geom_synth["inverted_scalar_card"]["field_difficulty_coefficient"] > 0.0
    with pytest.raises(ContractViolation):
        VulkanMemoryAdaptabilityLibrary.evaluate_memory_operations(uma_utilization=0.90)

    # 9. End-to-end orchestrator + SovereignOpenVinoCompanionV2 + CLI + HTTP GET/POST
    comp = SovereignOpenVinoCompanionV2(log_dir=tmp_path / "rev_logs", reset_ledger=True)
    orch_out = comp.run_reversible_correlation_and_spatial_cruise()
    assert orch_out["vital_max_hp"] == VITAL_MAX_HP == 6
    assert orch_out["frozen_theory_nodes_untouched"] == 56
    assert orch_out["chain_valid"] is True
    assert orch_out["binary_pattern_correlation_tool"]["roundtrip_identity"]["xnor_bit_similarity"] >= 0.999999

    rc = v2_cli_main(
        [
            "--reversible-binary-correlate",
            "--spatial-cruise-scheduler",
            "--realtime-inspector-governor",
            "--ux-privilege-framework",
            "--hexadim-log-morph",
            "--epistemic-memory-transducer",
            "--socket-thread-governance",
            "--exploration-benchmarks",
            "--tectonic-mesh-gearbox",
            "--vulkan-endpoint-extension",
            "--zero-copy-socket-observer",
            "--gpu-visual-defrag",
            "--binary-set-matrices",
            "--geometric-sync-scales",
            "--morse-hardware-parser",
            "--oscillation-matrix-suite",
            "--gamesa-3d-grid-vulkan-parser",
            "--blender-vray-b2b-endpoints",
            "--reset-ledger",
            "--log-dir",
            str(tmp_path / "cli_rev_cruise"),
        ]
    )
    assert rc == 0

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), SovereignOpenVinoRequestHandler)
    port = server.server_address[1]
    th = threading.Thread(target=server.serve_forever, daemon=True)
    th.start()
    try:
        with urllib.request.urlopen(
            f"http://127.0.0.1:{port}/api/v1/runtime/reversible_correlation_and_cruise?evoi_per_mj=0.145&observed_latency_ms=2.7420",
            timeout=5,
        ) as resp:
            assert resp.status == 200
            body = json.loads(resp.read().decode("utf-8"))
            assert body["vital_max_hp"] == 6
            assert body["frozen_theory_nodes_untouched"] == 56
            assert body["hexadimensional_log_interpreter_and_power_morpher"]["reached_half_state_at_min_power"] is True

        ok_post = urllib.request.Request(
            f"http://127.0.0.1:{port}/api/v1/runtime/reversible_correlation_and_cruise",
            data=json.dumps({"vital_hp": 6, "evoi_per_mj": 0.145, "observed_latency_ms": 2.7420}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(ok_post, timeout=5) as resp_ok:
            assert resp_ok.status == 200
            assert json.loads(resp_ok.read().decode("utf-8"))["chain_valid"] is True

        bad_post = urllib.request.Request(
            f"http://127.0.0.1:{port}/api/v1/runtime/reversible_correlation_and_cruise",
            data=json.dumps({"vital_hp": 5}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with pytest.raises(urllib.error.HTTPError) as http_err:
            urllib.request.urlopen(bad_post, timeout=5)
        assert http_err.value.code == 400

        with urllib.request.urlopen(f"http://127.0.0.1:{port}/blender_vray_ui", timeout=5) as ui_resp:
            assert ui_resp.status == 200
            assert b"Krystal Stack // Blender" in ui_resp.read()

        with urllib.request.urlopen(f"http://127.0.0.1:{port}/vision_propedeutics_video", timeout=5) as vid_resp:
            assert vid_resp.status == 200
            assert b"Propedeutika" in vid_resp.read()

        with urllib.request.urlopen(f"http://127.0.0.1:{port}/kernel_eshop", timeout=5) as eshop_ui:
            assert eshop_ui.status == 200
            assert b"__KRYSTAL_FULLSTACK_VARS__" in eshop_ui.read()

        with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/v1/runtime/rss_feed.xml", timeout=5) as rss_resp:
            import xml.etree.ElementTree as ET
            assert rss_resp.status == 200
            rss_root = ET.fromstring(rss_resp.read().decode("utf-8"))
            assert rss_root.tag == "rss"
            assert len(rss_root.findall("./channel/item")) == 4

        for route in (
            "/api/v1/runtime/blender_vray_sdk_pipeline",
            "/api/v1/runtime/inter_pause_cognitive_parser",
            "/api/v1/runtime/back_office_prompt_spectrum",
            "/api/v1/runtime/airport_b2b_cpp_sdk",
            "/api/v1/runtime/core_processes_and_constellations",
            "/api/v1/runtime/procedural_svg_cache_governor",
            "/api/v1/runtime/provenance_marketing_zones",
            "/api/v1/runtime/kernel_eshop",
            "/api/v1/runtime/adaptive_vision_usb_swap",
            "/api/v1/runtime/oxygen_hybrid_mesh_crud",
            "/api/v1/runtime/process_supervisor_inspector",
        ):
            with urllib.request.urlopen(f"http://127.0.0.1:{port}{route}", timeout=5) as ep_resp:
                assert ep_resp.status == 200
                assert json.loads(ep_resp.read().decode("utf-8"))["vital_max_hp"] == 6

        eshop_post_ok = urllib.request.Request(
            f"http://127.0.0.1:{port}/api/v1/runtime/kernel_eshop",
            data=json.dumps(
                {
                    "sku": "KRYSTAL-SERVER-CLUSTER",
                    "scale_key": "RATIO_SPECTRAL_TENSOR",
                    "contributor_name": "Dušan Kopecký",
                    "github_handle": "RhodosBerger",
                }
            ).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(eshop_post_ok, timeout=5) as eshop_res:
            assert eshop_res.status == 200
            assert json.loads(eshop_res.read().decode("utf-8"))["identity_and_access_audit"]["github_handle"] == "RhodosBerger"

        eshop_post_bad = urllib.request.Request(
            f"http://127.0.0.1:{port}/api/v1/runtime/kernel_eshop",
            data=json.dumps({"contributor_name": "Unauthorized Rogue", "github_handle": "rogue"}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with pytest.raises(urllib.error.HTTPError) as eshop_err:
            urllib.request.urlopen(eshop_post_bad, timeout=5)
        assert eshop_err.value.code == 400

        bv_post = urllib.request.Request(
            f"http://127.0.0.1:{port}/api/v1/runtime/blender_vray_sdk_pipeline",
            data=json.dumps({"vital_hp": 6, "resolution_x": 1280, "resolution_y": 720, "vray_samples": 32}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(bv_post, timeout=5) as bv_ok:
            assert bv_ok.status == 200
            assert json.loads(bv_ok.read().decode("utf-8"))["pipeline_inputs"]["resolution"] == "1280x720"

        bv_bad = urllib.request.Request(
            f"http://127.0.0.1:{port}/api/v1/runtime/blender_vray_sdk_pipeline",
            data=json.dumps({"vital_hp": 5}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with pytest.raises(urllib.error.HTTPError) as bv_err:
            urllib.request.urlopen(bv_bad, timeout=5)
        assert bv_err.value.code == 400
    finally:
        server.shutdown()
        server.server_close()
