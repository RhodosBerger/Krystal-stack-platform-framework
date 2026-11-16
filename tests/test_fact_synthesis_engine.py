"""Tests for the fact synthesis engine (induction, bridge, allocation, distance)."""

from __future__ import annotations

from projects.trigonometric_hybrid_platform_framework.fact_synthesis_engine import (
    fetch_distance_between_modules,
    induce_rules,
)


def _f(fid, value):
    return {"fact_id": fid, "statement": "", "source": "t", "key_path": "t", "evidence": "TEST", "value": value}


def test_bridge_detects_consistency_and_conflict() -> None:
    facts = [_f("F_C06_threshold_k", 449),
             _f("F_SW_A_blas_fma", {"K": 144, "holds": True}),
             _f("F_SW_B_blas_fma", {"K": 576, "holds": False})]
    rule = next(r for r in induce_rules(facts) if r["rule_id"] == "R_BRIDGE_BLAS_BOUNDARY")
    assert rule["verdict"] == "SUPPORTED_WITH_SCOPE"

    facts[2] = _f("F_SW_B_blas_fma", {"K": 576, "holds": True})  # contradicts the threshold
    rule = next(r for r in induce_rules(facts) if r["rule_id"] == "R_BRIDGE_BLAS_BOUNDARY")
    assert rule["verdict"] == "CONFLICT" and rule["counterexamples"] == ["F_SW_B_blas_fma"]


def test_allocation_break_even() -> None:
    facts = [_f("F_C03_GPU_compile_ms", 2240.0), _f("F_C03_CPU_compile_ms", 40.0),
             _f("F_C03_GPU_median_ms", 3.0), _f("F_C03_CPU_median_ms", 4.1)]
    rule = next(r for r in induce_rules(facts) if r["rule_id"] == "R_ALLOC_BREAK_EVEN")
    assert rule["break_even_inferences"] == 2000.0


def test_module_distance() -> None:
    d = fetch_distance_between_modules("virtual_convolution_contra", "virtual_convolution_pipeline")
    assert d["hops"] == 1
    assert fetch_distance_between_modules("nope", "platform_engine")["hops"] is None
