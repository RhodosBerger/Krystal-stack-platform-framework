"""Tests for the nomenclature propedeutics node."""

from __future__ import annotations

from projects.trigonometric_hybrid_platform_framework.nomenclature_registry import TERMS, validate_nomenclature


def test_all_terms_bind_to_real_code() -> None:
    res = validate_nomenclature()
    assert res["unresolved_terms"] == [], res["unresolved_terms"]
    assert res["verdict"] == "VERIFIED_WITH_SCOPE"


def test_fake_term_is_rejected() -> None:
    fake = TERMS + (("Memory isolation allocator", "claimed but not implemented",
                     "custom_allocator:IsolateGpuMemory", "tests/test_custom_allocator.py"),)
    res = validate_nomenclature(fake)
    assert res["verdict"] == "FALSIFIED"
    assert res["unresolved_terms"] == ["Memory isolation allocator"]
