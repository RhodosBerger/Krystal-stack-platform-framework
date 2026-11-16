"""End-to-end verification tests for x86 platform findings, hex combat bandwidth & world AI,
spatial coordinators & VR hologram compositor, NeuroSymbolicMicroLLM, arena cache/optical
codec/sockets, code-sample 3D object library & texture nodes, adaptive 2B trainer & study
conclusion, playable field-lines & game theory, WSL/mortar/geotiles/Vulkan shaders, and
satellite 3D mesh ML + cluster PIDs + intranet 128 B packets + Gemma fine-tuning sets.
"""

from __future__ import annotations

import http.server
import json
import threading
import urllib.request
from pathlib import Path

import pytest

from krystal_kernel.constants import VITAL_MAX_HP, ContractViolation
from krystal_kernel.sovereign_openvino_oneapi_app import SovereignOpenVinoRequestHandler
from projects.trigonometric_hybrid_platform_framework.adaptive_2b_spatial_linguistic_trainer import (
    AdaptiveConversationTrainer,
    build_spatial_and_linguistic_2b_kb,
    calibrate_new_shapes_and_model_catalog,
    compile_convolutional_study_conclusion,
    compute_four_core_project_summary_and_roadmap,
)
from projects.trigonometric_hybrid_platform_framework.arena_cache_optical_sockets_and_consumption import (
    BattleArenaCacheAndTensorNavigator,
    build_option_geometry_holograms,
    read_optical_binary,
    render_wireframe_sockets_and_holograms_html,
    run_multiplayer_entity_script_round,
    synthesize_and_write_optical_binary,
)
from projects.trigonometric_hybrid_platform_framework.clash_battle_royale import ClashGame
from projects.trigonometric_hybrid_platform_framework.cluster_pid_gemma_finetune_and_mesh_ml import (
    build_cluster_entity_pid_registry,
    build_gemma_abstraction_variants_and_finetune_sets,
    pack_intranet_128b_packet,
    run_neurosymbolic_datalink_and_hypervisor_exchange,
    train_and_duplicate_satellite_3d_mesh,
    unpack_intranet_128b_packet,
)
from projects.trigonometric_hybrid_platform_framework.hex_combat_bandwidth_and_world_ai import (
    compute_full_combat_bandwidth_solution,
    evaluate_all_faction_duel_combinations,
    instantiate_combat_preset,
    resolve_action_with_world_and_ai_chain,
    verify_hex_combinatorial_identities,
)
from projects.trigonometric_hybrid_platform_framework.neurosymbolic_genesis_discipline_and_llm import (
    NeuroSymbolicMicroLLM,
)
from projects.trigonometric_hybrid_platform_framework.object_library_texture_nodes_and_synthetic_network import (
    PreallocatedPhaseRenderPool,
    build_renderable_object_library,
    build_synthetic_knowledge_training_network,
    compile_carbon_environment_script,
    execute_arena_texture_node_graph,
)
from projects.trigonometric_hybrid_platform_framework.playable_fieldlines_and_game_theory import (
    build_llm_engine_tool_integrations,
    evaluate_standardized_digital_game_theory,
    execute_syntactic_function_chain,
    fetch_staged_processes_and_rig_skeleton_animations,
    generate_operational_field_lines_and_learn,
    render_playable_syntactic_studio_html,
    solve_algebraic_tensor_word_problems,
)
from projects.trigonometric_hybrid_platform_framework.spatial_coordinators_and_vr_hologram_compositor import (
    build_and_trace_hologram_scene,
    evaluate_spatial_coordinators_and_procedural_render,
    render_vr_hologram_profiler_html,
    synthesize_vision_sensory_mindmap,
)
from projects.trigonometric_hybrid_platform_framework.wsl_hypervisor_mortar_and_geotile_adapter import (
    adapt_integrations_to_four_geospatial_tiles,
    compile_vulkan_spirv_shader_sets_and_isomorphisms,
    compute_mortar_css_trajectories_and_reticles,
    configure_wsl_ubuntu_hypervisor_bridge,
    convert_geotile_location_to_3d_arena_replica,
    define_os_call_visual_semantics,
    evaluate_decision_tensor_ui_and_2b_bridge,
)
from projects.trigonometric_hybrid_platform_framework.x86_platform_findings import (
    run_x86_platform_findings_computation,
)


def test_x86_findings_and_hex_bandwidth_equations(tmp_path: Path) -> None:
    x86 = run_x86_platform_findings_computation(
        runs=5,
        write_reports=True,
        json_path=tmp_path / "x86.json",
        md_path=tmp_path / "x86.md",
    )
    assert x86["vital_max_hp"] == VITAL_MAX_HP
    assert x86["findings"]["X01_cache_hierarchy"]["l1d_over_dram_bandwidth_ratio"] > 1.0
    assert x86["findings"]["X05_binary_128b_vs_json"]["binary_speedup_over_json"] > 1.0

    comb = verify_hex_combinatorial_identities(max_radius=4)
    assert comb["verdict"] == "VERIFIED_WITH_SCOPE"

    bw = compute_full_combat_bandwidth_solution()
    assert bw["combinatorics"]["hex_cells"] == 37
    assert bw["combinatorics"]["directed_attack_arcs"] == 180
    assert bw["bandwidth_solution"]["host_max_concurrent_60hz_arenas"] >= 1

    duels = evaluate_all_faction_duel_combinations()
    assert len(duels["pairwise_faction_duels"]) == 6
    assert len(duels["presets"]) == 6

    g, _ = instantiate_combat_preset("PRESET_02_CENTER_HEX_CRUCIBLE")
    chain = resolve_action_with_world_and_ai_chain(g, {"type": "attack", "unit": "C01", "target": "C03"})
    assert chain["pre_state_sha256"] != chain["post_state_sha256"]
    assert len(chain["ai_reactions"]) >= 1

    before = g.state_hash()
    with pytest.raises(ContractViolation):
        resolve_action_with_world_and_ai_chain(g, {"type": "move", "unit": "C01", "to": [99, 99]})
    assert g.state_hash() == before


def test_spatial_vr_hologram_mindmap_and_micro_llm() -> None:
    sp = evaluate_spatial_coordinators_and_procedural_render()
    assert sp["coordinators_total"] == 7
    assert sp["procedural_render_pipeline"]["draw_call_reduction_ratio"] > 5.0

    holo = build_and_trace_hologram_scene(ray_count=16)
    assert len(holo["hologram_nodes"]) == 12
    assert holo["ray_trace_summary"]["rays_cast"] == 16

    vr_html = render_vr_hologram_profiler_html()
    assert "box-shadow" not in vr_html and "backdrop-filter" not in vr_html

    mm = synthesize_vision_sensory_mindmap()
    assert len(mm["mindmap_nodes"]) == 6

    llm = NeuroSymbolicMicroLLM(d_model=24, max_seq_len=20, seed=6)
    m = llm.train_and_compile(epochs=35, lr=0.08, device="CPU")
    assert m["final_cross_entropy_loss"] < m["initial_cross_entropy_loss"]
    gen = llm.generate_and_verify("NumPy BLAS")
    assert gen["grounded"] is True


def test_arena_cache_optical_kopt_object_lib_and_carbon_bridge(tmp_path: Path) -> None:
    g = ClashGame("SYNC", seed=6)
    nav = BattleArenaCacheAndTensorNavigator()
    assert nav.cache_arena_state(g)["cache_hit"] is False
    assert nav.cache_arena_state(g)["cache_hit"] is True
    mv = nav.compute_algebraic_tensor_move(g, "C01", apply_move=True)
    assert mv["applied"] is True

    mp = run_multiplayer_entity_script_round(seed=6)
    assert mp["consumption_audit"]["render_cost_reduction_pct"] > 0.0

    kopt_file = tmp_path / "arena.kopt"
    written = synthesize_and_write_optical_binary(path=kopt_file)
    decoded = read_optical_binary(path=kopt_file)
    assert decoded["sha256_verified"] is True
    assert decoded["record_count"] == written["record_count"] == 4

    # Corrupt 1 byte -> must raise ContractViolation
    bad = bytearray(kopt_file.read_bytes())
    bad[20] ^= 0xFF
    bad_file = tmp_path / "bad.kopt"
    bad_file.write_bytes(bytes(bad))
    with pytest.raises(ContractViolation):
        read_optical_binary(path=bad_file)

    assert len(build_option_geometry_holograms()["option_holograms"]) >= 1
    wf_html = render_wireframe_sockets_and_holograms_html()
    assert "box-shadow" not in wf_html

    lib = build_renderable_object_library()
    assert lib["object_count"] == 6

    pool = PreallocatedPhaseRenderPool()
    tex = execute_arena_texture_node_graph(pool=pool)
    assert tex["phase_pool_audit"]["zero_reallocation_verified"] is True

    env_ok = compile_carbon_environment_script(
        [{"op": "SPAWN_CODE_OBJECT", "object_id": "OBJ_01_SYNC_GATE_OBELISK"}, {"op": "SET_ARENA_TEXTURE"}]
    )
    assert env_ok["openai_used"] is False
    with pytest.raises(ContractViolation):
        compile_carbon_environment_script([{"provider": "openai", "op": "SPAWN_CODE_OBJECT"}])

    net = build_synthetic_knowledge_training_network(jsonl_path=tmp_path / "syn_net.jsonl")
    assert net["synthetic_triples_exported"] >= 6


def test_adaptive_2b_study_fieldlines_game_theory_wsl_and_cluster_gemma(tmp_path: Path) -> None:
    cal = calibrate_new_shapes_and_model_catalog(cache_dir=tmp_path)
    assert len(cal["calibrated_shapes"]) == 3

    kb2b = build_spatial_and_linguistic_2b_kb(out_path=tmp_path / "kb2b.jsonl")
    assert kb2b["kb_entries_count"] >= 10

    trainer = AdaptiveConversationTrainer(d_model=32, max_seq_len=24, epochs=35, seed=6)
    ad = trainer.adapt_from_conversation_history([("Bad turn", "Uncited claim 12345.")])
    assert ad["rejected_unverified_turns"] == 1

    study = compile_convolutional_study_conclusion(
        x86_runs=3,
        svg_path=tmp_path / "proj.svg",
        md_path=tmp_path / "study.md",
    )
    assert 0.0 < study["convolutional_genesis_index_0_to_6"] <= 6.0
    assert (tmp_path / "proj.svg").exists() and (tmp_path / "study.md").exists()

    fl = generate_operational_field_lines_and_learn(streamline_count=6, steps_per_line=8)
    assert fl["learner_metrics"]["final_mse"] < fl["learner_metrics"]["initial_mse"]

    ch = execute_syntactic_function_chain(telemetry_out=tmp_path / "ag_telem.jsonl")
    assert len(ch["trace"]) == 5
    assert "box-shadow" not in render_playable_syntactic_studio_html()

    gt = evaluate_standardized_digital_game_theory()
    assert gt["verdict"] == "VERIFIED_WITH_SCOPE"

    assert configure_wsl_ubuntu_hypervisor_bridge()["vital_max_hp"] == 6
    assert len(define_os_call_visual_semantics()["os_semantic_mappings"]) == 4
    assert "@keyframes krystalMortarArc" in compute_mortar_css_trajectories_and_reticles()["css_keyframes_preset"]
    assert evaluate_decision_tensor_ui_and_2b_bridge()["inference_triggered"] is True
    assert adapt_integrations_to_four_geospatial_tiles()["adaptation_count"] == 4
    assert convert_geotile_location_to_3d_arena_replica()["replica_hex_count"] == 37
    assert compile_vulkan_spirv_shader_sets_and_isomorphisms()["bijection_hex37_to_uma_slot_verified"] is True

    mesh_ml = train_and_duplicate_satellite_3d_mesh(grid_n=6, epochs=40, obj_path=tmp_path / "clone.obj")
    assert mesh_ml["final_mse"] < mesh_ml["initial_mse"]
    assert (tmp_path / "clone.obj").exists()

    pids = build_cluster_entity_pid_registry()
    assert pids["unique_pids_verified"] is True

    raw_pkt = pack_intranet_128b_packet(1, 100, 200, 0.35, "HELLO_CLUSTER")
    assert len(raw_pkt) == 128 and unpack_intranet_128b_packet(raw_pkt)["sha256_verified"] is True
    assert len(run_neurosymbolic_datalink_and_hypervisor_exchange()["packets_transmitted"]) == 2

    gemma = build_gemma_abstraction_variants_and_finetune_sets(out_path=tmp_path / "gemma.jsonl")
    assert gemma["train_eval_disjoint_verified"] is True

    from projects.trigonometric_hybrid_platform_framework.cluster_pid_gemma_finetune_and_mesh_ml import (
        build_godot_scene_and_goniometric_behavioral_kernel,
    )
    from projects.trigonometric_hybrid_platform_framework.playable_fieldlines_and_game_theory import (
        render_lightweight_bot_perception_hud,
        run_engine_node_benchmark_and_recommend_preset,
    )
    from projects.trigonometric_hybrid_platform_framework.wsl_hypervisor_mortar_and_geotile_adapter import (
        wsl_hypervisor_render_dispatch,
    )

    summary4 = compute_four_core_project_summary_and_roadmap()
    assert len(summary4["four_core_characteristics"]) == 4
    assert len(solve_algebraic_tensor_word_problems()["word_problems"]) == 3
    assert build_llm_engine_tool_integrations()["symbolic_hex_polynomial"] == "3*R**2 + 3*R + 1"
    assert fetch_staged_processes_and_rig_skeleton_animations()["skeleton_is_tree"] is True
    assert wsl_hypervisor_render_dispatch()["vital_max_hp"] == 6
    assert render_lightweight_bot_perception_hud(svg_out=tmp_path / "hud.svg")["vital_max_hp"] == 6
    assert run_engine_node_benchmark_and_recommend_preset(runs=2)["combined_pipeline_median_ms"] > 0.0
    godot_gonio = build_godot_scene_and_goniometric_behavioral_kernel(tscn_path=tmp_path / "arena.tscn")
    assert godot_gonio["godot_layer_analysis"]["sustainable_simultaneous_layers_on_host"] == 10
    assert len(godot_gonio["goniometric_behavioral_kernel"]["descriptors"]) == 3


def test_new_http_endpoints_roundtrip() -> None:
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), SovereignOpenVinoRequestHandler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        for route in (
            "/vr_hologram_profiler",
            "/arena_wireframe_sockets",
            "/syntactic_playable_studio",
            "/api/v1/runtime/hex_combat_bandwidth",
            "/api/v1/runtime/game_theory",
            "/api/v1/runtime/geotile_vulkan_arena",
            "/api/v1/runtime/cluster_pid_and_gemma",
        ):
            with urllib.request.urlopen(base + route, timeout=30) as r:
                assert r.status == 200
    finally:
        server.shutdown()
        server.server_close()


def test_cross_layer_pattern_demonstration() -> None:
    from projects.trigonometric_hybrid_platform_framework.cross_layer_pattern_demo import (
        run_cross_layer_pattern_demonstration,
    )

    demo = run_cross_layer_pattern_demonstration(uma_utilization=0.82, layout_angle_deg=64.0, write_reports=True)
    assert demo["vital_max_hp"] == 6
    assert demo["telemetry_and_hypervisor_snapshot"]["wsl2_modulated_ram_gb"] == 8
    assert demo["layer_demonstrations"]["L1_zero_stutter_dcc"]["parasitic_task_decoupled"] is True
    assert demo["layer_demonstrations"]["L2_verified_robotics_twin"]["conflict_emergency_stop"] is True
    assert demo["layer_demonstrations"]["L3_self_tuning_workstation"]["committed_core_affinity"] == "CPU_4T_PHYSICAL_AVX512"
    assert demo["layer_demonstrations"]["L3_self_tuning_workstation"]["anti_oscillation_jitter_blocked"] is True
    assert demo["layer_demonstrations"]["L4_low_latency_ide_mesh"]["released_after_verifier_receipt"] is True
    assert all(p["evidence_status"] in ("VERIFIED_WITH_SCOPE", "LOCAL_MEASURED") for p in demo["pattern_6_tuples"])

