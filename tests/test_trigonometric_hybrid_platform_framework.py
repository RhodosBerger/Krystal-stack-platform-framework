"""End-to-end verification suite for Trigonometric Hybrid Platform Framework."""

from __future__ import annotations

import http.server
import json
import threading
import urllib.error
import urllib.request

import pytest

from krystal_kernel.constants import VITAL_MAX_HP, ContractViolation
from krystal_kernel.sovereign_openvino_oneapi_app import SovereignOpenVinoRequestHandler
from projects.trigonometric_hybrid_platform_framework import (
    CatalogProliferationAndCarouselReconciler,
    CodeDependencyGraphAnalyzer,
    ConstitutionalModeratorAndChoiceSolver,
    FamiliarHomeClassAndSevenProductSets,
    FrameworkNationsValidator,
    HierarchyOfRelatedTermsEngine,
    MasterPromptRouterConsolidator,
    ProcessDrivenGearboxAndCommKernel,
    ReloadedTrigonometricPatternStructures,
    SchemaConsistencySolver,
    StepExecutionGateProxy,
    TrigonometricHybridPlatformFramework,
    render_trigonometric_platform_html,
)


def test_trigonometric_reload_and_zero_copy_structures() -> None:
    reloader = ReloadedTrigonometricPatternStructures()
    out = reloader.reload_pattern(angle_deg=58.282525)
    trig = out["trigonometric_state_struct"]
    frame = out["zero_copy_frame_struct"]

    assert trig["vital_max_hp"] == VITAL_MAX_HP == 6
    assert trig["pythagorean_invariant_ok"] is True
    assert abs(trig["pythagorean_sum"] - 1.0) <= 1e-4
    assert len(trig["chebyshev_t1_to_t4"]) == 4
    assert len(trig["frame_128b_hex"]) == 256  # 128 bytes hex
    assert frame["total_frame_bytes"] == 128
    assert frame["authenticated_prefix_bytes"] + frame["hmac_sha256_tag_bytes"] == 128

    with pytest.raises(ContractViolation):
        ReloadedTrigonometricPatternStructures(vital_hp=5)
    with pytest.raises(ContractViolation):
        reloader.reload_pattern(angle_deg=float("nan"))


def test_z3_schema_solver_and_ast_code_dependencies() -> None:
    solved = SchemaConsistencySolver.solve_all_schemas(uma_utilization=0.58)
    assert solved["vital_max_hp"] == 6
    assert solved["solver_verdict"] == "Z3_SAT_VERIFIED"
    assert solved["z3_witness"]["frame_bytes"] == 128
    assert solved["z3_witness"]["b2b_fee_eur"] == 16200
    assert solved["z3_witness"]["gpl_fee_eur"] == 0

    with pytest.raises(ContractViolation):
        SchemaConsistencySolver.solve_all_schemas(uma_utilization=0.85)

    deps = CodeDependencyGraphAnalyzer.analyze_code_dependencies()
    assert deps["analyzed_module_count"] >= 8
    assert deps["internal_dependency_edge_count"] > 0
    assert deps["pyproject_present"] is True


def test_master_prompt_router_and_step_proxy_invariants() -> None:
    router = MasterPromptRouterConsolidator()
    routed = router.route_master_prompt("licensing_b2b_and_oss_governance")
    assert routed["routed_microprompt"]["prompt_id"] == "MP_06_HYBRID_MARKET_AND_ORIGIN_LOCK"
    assert len(routed["discovered_codebase_sources"]["agent_rules"]) >= 10

    receipt = router.edit_microprompt(
        "MP_01_EPISTEMIC_DUAL_MODE",
        "Enforce EXPLORE != VERIFY and PROPOSER != VERIFIER with zero self-certification.",
    )
    assert receipt["new_version"] == 2

    with pytest.raises(ContractViolation):
        router.edit_microprompt("MP_01_EPISTEMIC_DUAL_MODE", "x" * 1500)

    proxy = StepExecutionGateProxy()
    pre_snapshot = dict(proxy.committed_state)
    with pytest.raises(ContractViolation):
        proxy.execute_step(
            step_id="BAD_SELF_CERT",
            proposer_id="SAME_AGENT",
            verifier_id="SAME_AGENT",
            step_fn=lambda: {"ok": True},
        )
    assert proxy.committed_state == pre_snapshot  # Delta S == 0


def test_nations_catalogs_carousels_gearbox_subprocess_and_moderator() -> None:
    nations = FrameworkNationsValidator.validate_all_nations()
    assert nations["validated_nation_count"] == 5
    assert nations["all_nations_invariant_valid"] is True

    audit = CatalogProliferationAndCarouselReconciler.audit_catalogs_and_carousels()
    assert audit["active_domain_catalog_count"] >= 50
    assert audit["carousel_count"] == 4

    psets = FamiliarHomeClassAndSevenProductSets.synthesize_home_and_product_sets()
    assert psets["product_set_count"] == 7
    assert len(psets["familiar_home_class_solutions"]) == 3

    gear = ProcessDrivenGearboxAndCommKernel.sample_live_processes_and_gearbox()
    assert gear["active_os_process_count"] > 0
    assert "<svg" in gear["gearbox_svg_visualization"]

    comm = ProcessDrivenGearboxAndCommKernel.spawn_communication_kernel_process()
    assert comm["proceed_gate_verdict"] == "PROCEED_AUTHORIZED"
    assert comm["frame_bytes_verified"] == 128
    assert comm["spawned_child_pid"] != comm["parent_pid"]

    mod = ConstitutionalModeratorAndChoiceSolver.moderate_and_solve_consistent_choice()
    assert mod["moderation_verdict"] == "ALLOW_CONSISTENT_CHOICE"
    assert mod["z3_consistent_choice_status"] == "Z3_SAT_CONSISTENT"
    assert mod["consistent_choice_witness"]["resolved_cart_eur"] == 16200

    with pytest.raises(ContractViolation):
        ConstitutionalModeratorAndChoiceSolver.moderate_and_solve_consistent_choice(
            commercial_cart=True,
            signed_agreement=False,
        )


def test_full_platform_and_live_http_endpoints() -> None:
    fw = TrigonometricHybridPlatformFramework()
    res = fw.evaluate_full_platform(angle_deg=31.7175)
    assert res["vital_max_hp"] == 6
    assert len(res["step_proxy_receipts"]) == 6
    assert res["charter_and_hierarchy"]["hierarchy_of_related_terms"]["tier_count"] == 5

    html_doc = render_trigonometric_platform_html()
    assert "Krystal Trigonometric Hybrid Platform Framework" in html_doc
    assert "<svg" in html_doc

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), SovereignOpenVinoRequestHandler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/trigonometric_hybrid_platform", timeout=10) as r:
            assert r.status == 200
            body_html = r.read().decode("utf-8")
            assert "WAVE_2026_11_TRIGONOMETRIC_HYBRID_PLATFORM" in body_html

        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/api/v1/runtime/trigonometric_hybrid_platform",
            data=json.dumps({"angle_deg": 45.0, "uma_utilization": 0.50}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=10) as r2:
            assert r2.status == 200
            payload = json.loads(r2.read().decode("utf-8"))
            assert payload["solved_schemas"]["solver_verdict"] == "Z3_SAT_VERIFIED"

        bad_req = urllib.request.Request(
            f"http://127.0.0.1:{port}/api/v1/runtime/trigonometric_hybrid_platform",
            data=json.dumps({"vital_hp": 4}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with pytest.raises(urllib.error.HTTPError) as exc_info:
            urllib.request.urlopen(bad_req, timeout=10)
        assert exc_info.value.code == 400
    finally:
        server.shutdown()
        server.server_close()
