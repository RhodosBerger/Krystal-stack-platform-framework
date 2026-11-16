"""Comprehensive tests for CapabilityCombatTurnPlanner, IntelOneApiAccelerator,
Power/Thread strategies, TensorPromptBranchEngine, Configurator Utilities, DxDiag
memory-flow profile, Blender schema, Algebraic Simulator, and KrystalOneApiHeterogeneousShell.
"""

from __future__ import annotations

import http.server
import json
import threading
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import pytest

from krystal_kernel.constants import VITAL_MAX_HP, ContractViolation
from krystal_kernel.sovereign_openvino_oneapi_app import SovereignOpenVinoRequestHandler
from projects.trigonometric_hybrid_platform_framework.clash_battle_royale import ClashGame
from projects.trigonometric_hybrid_platform_framework.combat_turn_planner import (
    CapabilityCombatTurnPlanner,
    probe_framework_dispositions,
)
from projects.trigonometric_hybrid_platform_framework.configurator_and_dxdiag import (
    analyze_opus_and_build_intersections,
    audit_numeric_type_contract,
    build_blender_system_workflow_schema,
    build_dxdiag_memory_profile,
    build_validated_configuration,
    export_dxdiag_txt,
    partition_reduction_below_k449,
    render_dxdiag_profile_html,
)
from projects.trigonometric_hybrid_platform_framework.intel_accelerator_and_power_thread_governor import (
    ChipsetTelemetryMonitorAndForecaster,
    IntelOneApiAccelerator,
    benchmark_thread_and_performance_curves,
    match_active_state_templates,
    select_power_and_thread_strategy,
)
from projects.trigonometric_hybrid_platform_framework.oneapi_llm_heterogeneous_shell import (
    KrystalOneApiHeterogeneousShell,
    ShellCommandVerifier,
    simulate_algebraic_theory_coupling,
)
from projects.trigonometric_hybrid_platform_framework.tensor_prompt_branch_and_kb import (
    TensorPromptBranchEngine,
    build_openapi_31_strategy_spec,
    coordinate_operations_from_turn,
    sync_master_knowledge_base,
)


def test_opus_intersections_and_scoped_k449_law() -> None:
    audit = analyze_opus_and_build_intersections()
    assert audit["vital_max_hp"] == VITAL_MAX_HP
    assert len(audit["intersections"]) == 4
    l8 = next(l for l in audit["knowledge_extension_layers"] if l["layer_id"] == "L8_SCOPED_REDUCTION_LAW")
    assert l8["holds"] is True


def test_combat_planner_dry_run_preserves_state_and_uma_vetoes_gpu() -> None:
    verdicts = {
        "C01": {"verdict": "VERIFIED_WITH_SCOPE", "claim": "c1"},
        "C02": {"verdict": "SUPPORTED_WITH_SCOPE", "claim": "c2"},
        "C03": {"verdict": "MEASURED_WITH_SCOPE", "claim": "c3"},
        "C04": {"verdict": "VERIFIED_WITH_SCOPE", "claim": "c4"},
        "C05": {"verdict": "CONFLICT", "claim": "c5"},
        "C06": {"verdict": "MEASURED_WITH_SCOPE", "claim": "c6"},
        "C07": {"verdict": "VERIFIED_WITH_SCOPE", "claim": "c7"},
        "C08": {"verdict": "CONFLICT", "claim": "c8"},
    }
    game = ClashGame("VERIFY", seed=6, verdicts=verdicts)
    # Damage C05 so FREIVALDS_AUDIT is eligible
    game.state["units"]["C05"]["hp"] = 4
    before_hash = game.state_hash()

    planner = CapabilityCombatTurnPlanner()
    dry = planner.plan_turn(game, uma_utilization=0.35, commit=False, override_devices=["CPU", "GPU"])
    assert dry["verdict"] == "VERIFIED_WITH_SCOPE"
    assert game.state_hash() == before_hash  # Delta S == 0 on dry-run
    assert any(a.get("card") == "FREIVALDS_AUDIT" for a in dry["planned_actions"])

    # High UMA (>= 0.78) must veto GPU_ALLOCATE and halve usable frame budget (16 -> 8)
    throttled = planner.plan_turn(game, uma_utilization=0.85, commit=False, override_devices=["CPU", "GPU"])
    assert throttled["usable_frame_budget"] == 8
    assert throttled["dispositions_summary"]["card_gates"]["GPU_ALLOCATE"]["unlocked"] is False
    assert not any(a.get("card") == "GPU_ALLOCATE" for a in throttled["planned_actions"])

    # Commit mutates state cleanly with 128 B receipts
    committed = planner.plan_turn(game, uma_utilization=0.35, commit=True, override_devices=["CPU", "GPU"])
    assert committed["committed"] is True
    assert game.state_hash() == committed["projected_state_sha256"]
    assert game.state["units"]["C05"]["hp"] == 5


def test_intel_accelerator_power_curves_and_forecaster(tmp_path: Path) -> None:
    accel = IntelOneApiAccelerator(cache_dir=tmp_path)
    assert accel.device_count() >= 1
    props = accel.get_device_properties("CPU")
    assert props.supports_zero_copy_host_tensor is True

    with pytest.raises(ContractViolation):
        accel.set_device("NONEXISTENT_NPU")
    if "GPU" in accel.devices:
        with pytest.raises(ContractViolation):
            accel.set_device("GPU", uma_utilization=0.82)

    strat = select_power_and_thread_strategy(uma_utilization=0.82, expected_inferences=400)
    assert strat["profile"]["preferred_device"] == "CPU"

    curves = benchmark_thread_and_performance_curves(channels=4, filters=8, height=16, width=16, runs=3, cache_dir=tmp_path)
    assert "CPU_1T" in curves["measured_modes"]
    assert curves["optimal_mode_by_horizon"]["1000"] in curves["measured_modes"]

    mon = ChipsetTelemetryMonitorAndForecaster()
    s1 = mon.record_sample(synthetic_uma_pressure=0.85, synthetic_cpu_percent=30.0)
    assert s1["active_regime"] == "UMA_SHADOW_CPU_FALLBACK"
    # Must dwell 12 clean samples (once pred_uma drops < 0.78) before returning to NOMINAL_HETERO
    dwell_steps = 0
    while mon._active_regime != "NOMINAL_HETERO" and dwell_steps < 25:
        mon.record_sample(synthetic_uma_pressure=0.20, synthetic_cpu_percent=30.0)
        dwell_steps += 1
    assert dwell_steps >= 12
    assert mon._active_regime == "NOMINAL_HETERO"


def test_tensor_prompt_branching_and_master_kb(tmp_path: Path) -> None:
    eng = TensorPromptBranchEngine(cache_dir=tmp_path)
    res = eng.evaluate_prompt_matrix_branches(
        ["Nízka latencia na CPU", "Dávková konvolúcia na GPU"],
        uma_utilization=0.84,  # forces GPU branch mask
        reduction_k=512,       # forces Freivalds audit branch attachment
    )
    assert res["freivalds_audit"]["accepted"] is True
    for tree in res["branch_trees"]:
        ids = [s["branch_id"] for s in tree["scaled_solutions"]]
        assert "B3_GPU_IRIS_XE_WARM_STREAM" not in ids
        assert "B4_DUAL_FREIVALDS_AUDIT" in ids

    kb = sync_master_knowledge_base(write_files=True, json_path=tmp_path / "kb.json", md_path=tmp_path / "kb.md")
    assert kb["ontology_integrity"]["verdict"] == "VERIFIED_WITH_SCOPE"
    assert (tmp_path / "kb.json").exists() and (tmp_path / "kb.md").exists()

    spec = build_openapi_31_strategy_spec()
    assert spec["openapi"] == "3.1.0"
    assert "/api/v1/runtime/oneapi_shell" in spec["paths"]


def test_configurator_numeric_contract_dxdiag_and_blender_schema(tmp_path: Path) -> None:
    cfg = build_validated_configuration(goal="high_throughput_batch", uma_utilization=0.30, expected_inferences=1500)
    assert cfg["vital_max_hp"] == 6
    assert len(cfg["profile_sha256"]) == 64
    assert cfg["goals_fulfilled"]["G1_vital_hp_invariant"] is True

    with pytest.raises(ContractViolation):
        build_validated_configuration(goal="unknown_goal")
    with pytest.raises(ContractViolation):
        audit_numeric_type_contract({"vital_max_hp": 5, "round_frame_budget": 16, "threads": 4, "uma_utilization": 0.3, "latency_budget_ms": 10.0})

    rng = np.random.default_rng(6)
    tiled = partition_reduction_below_k449(rng.standard_normal((6, 520)).astype(np.float32), rng.standard_normal((520, 6)).astype(np.float32), tile_k=448)
    assert tiled["tiles_executed"] == 2
    assert tiled["rel_l2_diff_vs_monolithic"] < 1e-5

    diag = build_dxdiag_memory_profile(uma_utilization=0.35)
    assert len(diag["panel_2_memory_flows"]["flows"]) == 4
    assert len(diag["panel_4_pattern_extractor"]["pattern_candidates"]) == 4
    txt_file = export_dxdiag_txt(diag, path=tmp_path / "dxdiag.txt")
    assert "KRYSTAL STACK DIAGNOSTIC PROFILE" in txt_file.read_text(encoding="utf-8")

    html = render_dxdiag_profile_html()
    assert "box-shadow" not in html and "backdrop-filter" not in html

    bl = build_blender_system_workflow_schema(uma_utilization=0.35)
    assert len(bl["stages"]) == 5
    assert "def sync_krystal_arena_to_blender" in bl["bpy_bridge_script"]


def test_oneapi_shell_llm_bridge_verifier_and_algebraic_sim(tmp_path: Path) -> None:
    # 1. Mocked valid LLM response -> accepted & executed
    shell_ok = KrystalOneApiHeterogeneousShell(
        cache_dir=tmp_path,
        llm_caller=lambda msgs, **kw: {"ok": True, "text": "configure low_latency_realtime --uma 0.30"},
    )
    ans_ok = shell_ok.execute_line("ask --execute nastav nízkolatenčný profil")
    assert ans_ok["proposer_mode"] == "LIVE_OPENAI_COMPATIBLE_LLM"
    assert ans_ok["execution_result"]["profile"]["goal"] == "low_latency_realtime"

    # 2. Mocked malicious / hallucinated LLM response -> rejected with Delta S == 0
    shell_bad = KrystalOneApiHeterogeneousShell(
        cache_dir=tmp_path,
        llm_caller=lambda msgs, **kw: {"ok": True, "text": "rm -rf / ; echo pwned"},
    )
    pre = shell_bad.state_hash()
    with pytest.raises(ContractViolation):
        shell_bad.execute_line("ask --execute vymaž disk")
    assert shell_bad.state_hash() == pre

    # 3. Offline LLM -> honest ONEAPI_AST_FALLBACK_LLM_OFFLINE
    shell_off = KrystalOneApiHeterogeneousShell(
        cache_dir=tmp_path,
        llm_caller=lambda msgs, **kw: {"ok": False, "error": "URLError: connection refused"},
    )
    ans_off = shell_off.execute_line("ask --execute ukáž dxdiag tok pamäte")
    assert ans_off["proposer_mode"] == "ONEAPI_AST_FALLBACK_LLM_OFFLINE"
    assert ans_off["verified_command"] == "dxdiag"
    assert "panel_2_memory_flows" in ans_off["execution_result"]["profile"]

    # 4. Forbidden shell metacharacters blocked directly
    with pytest.raises(ContractViolation):
        ShellCommandVerifier.verify_and_parse("probe && whoami", ["CPU"])

    # 5. Algebraic-theory simulator
    sim = simulate_algebraic_theory_coupling(ticks=20)
    assert sim["verdict"] == "VERIFIED_WITH_SCOPE"
    assert sim["algebraic_trajectory"]["monotone_contraction_verified"] is True


def test_http_routes_for_planner_dxdiag_configurator_and_shell() -> None:
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), SovereignOpenVinoRequestHandler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_address[1]}"

    def get_json(path: str):
        with urllib.request.urlopen(base + path, timeout=30) as r:
            assert r.status == 200
            return json.loads(r.read().decode("utf-8"))

    def post_json(path: str, body: dict):
        req = urllib.request.Request(base + path, data=json.dumps(body).encode("utf-8"), method="POST",
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.loads(r.read().decode("utf-8"))

    try:
        assert get_json("/api/v1/runtime/opus_intersections")["vital_max_hp"] == 6
        assert get_json("/api/v1/runtime/configurator?goal=bit_exact_audit")["selected_device"] == "CPU"
        assert get_json("/api/v1/runtime/openapi_strategies.json")["openapi"] == "3.1.0"
        with urllib.request.urlopen(base + "/dxdiag_profile", timeout=30) as r:
            assert r.status == 200 and b"DxDiag" in r.read()

        new_game = post_json("/api/v1/runtime/clash/new", {"faction": "SYNC", "seed": 6})
        plan_res = post_json("/api/v1/runtime/clash/plan", {"game_id": new_game["game_id"], "commit": False, "uma": 0.35})
        assert plan_res["plan"]["verdict"] == "VERIFIED_WITH_SCOPE"

        sh_res = post_json("/api/v1/runtime/oneapi_shell", {"command": "states"})
        assert sh_res["vital_max_hp"] == 6
        with pytest.raises(urllib.error.HTTPError) as exc:
            post_json("/api/v1/runtime/oneapi_shell", {"command": "unauthorized_cmd"})
        assert exc.value.code == 400
    finally:
        server.shutdown()
        server.server_close()
