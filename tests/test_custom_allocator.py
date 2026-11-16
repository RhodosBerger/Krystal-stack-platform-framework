"""Tests for the fact-driven allocator and ontology integrity check."""

from __future__ import annotations

from projects.trigonometric_hybrid_platform_framework.custom_allocator import (
    FactDrivenAllocator,
    check_ontology_integrity,
)


def _f(fid, value, source="pyproject.toml"):
    return {"fact_id": fid, "statement": "", "source": source, "key_path": "t", "evidence": "TEST", "value": value}


FACTS = [_f("F_C03_GPU_compile_ms", 2240.0), _f("F_C03_CPU_compile_ms", 40.0),
         _f("F_C03_GPU_median_ms", 3.0), _f("F_C03_CPU_median_ms", 4.1),
         _f("F_C03_GPU_p95_ms", 5.6), _f("F_C03_CPU_p95_ms", 9.9)]


def test_allocator_break_even_and_hysteresis() -> None:
    a = FactDrivenAllocator(FACTS)
    assert a.allocate(100)["device"] == "CPU"      # 40+410 < 2240+300
    assert a.allocate(10_000)["device"] == "GPU"   # 2240+30000 < 40+41000
    # Tight latency budget excludes CPU (p95 9.9 ms) even for few inferences.
    assert a.allocate(100, latency_budget_ms=6.0)["device"] == "GPU"
    # Near break-even, the current device is kept (gain below 10 % margin).
    r = a.allocate(2100, current_device="CPU")
    assert r["device"] == "CPU" and "kept" in r["reason"]


def test_allocator_unknown_without_facts() -> None:
    assert FactDrivenAllocator([]).allocate(10)["verdict"] == "UNKNOWN"


def test_integrity_detects_dangling_and_missing() -> None:
    good = {"facts": FACTS, "rules": [{"rule_id": "R1", "derived_from": ["F_C03_GPU_median_ms"]}]}
    assert check_ontology_integrity(good)["verdict"] == "VERIFIED_WITH_SCOPE"
    bad = {"facts": FACTS + [_f("F_X", 1, source="does/not/exist.json")],
           "rules": [{"rule_id": "R2", "derived_from": ["F_MISSING"]}]}
    res = check_ontology_integrity(bad)
    assert res["verdict"] == "FALSIFIED"
    assert res["dangling_rule_citations"] == {"R2": ["F_MISSING"]}
    assert res["missing_sources"] == ["does/not/exist.json"]
