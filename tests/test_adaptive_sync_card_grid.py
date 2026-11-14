"""Tests for the adaptive sync card grid (design-manual compliance + data integrity)."""

from __future__ import annotations

import json
import re

from projects.trigonometric_hybrid_platform_framework.adaptive_sync_card_grid import (
    build_card_dataset,
    render_adaptive_sync_card_grid_html,
)


def test_card_dataset_covers_all_layers_and_zero_sim_desync() -> None:
    data = build_card_dataset(sim_ticks=400)
    layers = {c["layer"] for c in data["cards"]}
    assert layers == {"WORLD", "FACTION", "SYSTEM", "ACTION", "TELEMETRY"}
    assert data["card_count"] == 12 + 5 + 16 + 3
    assert data["graph"]["graph_valid"] is True
    assert data["total_sim_desync"] == 0
    for c in data["cards"]:
        if c["kind"] == "TELEMETRY":
            assert all(m["tag"] == "MODELED" for m in c["metrics"])
    # Drill-down links must resolve to existing cards and every card must have at least one link
    ids = {c["id"] for c in data["cards"]}
    for c in data["cards"]:
        assert c["links"], c["id"]
        assert set(c["links"]) <= ids, c["id"]


def test_html_follows_design_manual_tokens() -> None:
    html_doc = render_adaptive_sync_card_grid_html(sim_ticks=300)
    # Canonical tokens and the three display modes
    for token in ("--ks-gold:#D8AE4B", "--ks-cyan:#58D8FF", "--radius-ui:2px", "mode-world", "mode-arena", "mode-hybrid"):
        assert token in html_doc
    # Banned: shadows, glass blur, pill radii
    assert "box-shadow" not in html_doc
    assert "backdrop-filter" not in html_doc
    assert not re.search(r"border-radius:\s*(?:[5-9]|\d{2,})px", html_doc)
    # Hex grid view present
    assert ".ks-hexgrid" in html_doc and "shape-outside" in html_doc and 'data-v="hex"' in html_doc
    assert 'data-v="hex3d"' in html_doc and "preserve-3d" in html_doc and "perspective" in html_doc
    # Embedded dataset parses
    m = re.search(r"const DATA = (\{.*?\});\n", html_doc, re.S)
    assert m is not None
    assert json.loads(m.group(1).replace("<\\/", "</"))["vital_max_hp"] == 6


def test_dependency_scan_cache_hits_on_repeat() -> None:
    from projects.trigonometric_hybrid_platform_framework.platform_engine import CodeDependencyGraphAnalyzer

    first = CodeDependencyGraphAnalyzer.analyze_code_dependencies()
    second = CodeDependencyGraphAnalyzer.analyze_code_dependencies()
    assert second["ast_scan_cache_hits"] == second["analyzed_module_count"]
    assert first["internal_dependency_edge_count"] == second["internal_dependency_edge_count"]
    assert first["modules"] == second["modules"]
