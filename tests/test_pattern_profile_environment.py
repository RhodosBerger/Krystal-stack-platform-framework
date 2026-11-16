"""Tests for the pattern-over-profiles environment."""

from __future__ import annotations

import pytest

from krystal_kernel.constants import ContractViolation
from projects.trigonometric_hybrid_platform_framework.pattern_profile_environment import (
    ProfileSpec,
    run_pattern_sweep,
)

ALLOWED = {"SUPPORTED_WITH_SCOPE", "CONFLICT", "UNKNOWN"}


def test_sweep_structure_and_verdicts() -> None:
    pytest.importorskip("openvino")
    profiles = (ProfileSpec("T1", 2, 4, 8, 8, 1), ProfileSpec("T2", 4, 8, 12, 12, 2))
    res = run_pattern_sweep(profiles)
    assert res["profiles_run"] == 2
    for name, h in res["hypotheses"].items():
        assert h["verdict"] in ALLOWED, name
        assert h["holds_in"] + len(h["failing_profiles"]) == h["profiles_run"]


def test_duplicate_profiles_rejected() -> None:
    p = ProfileSpec("DUP", 2, 4, 8, 8, 1)
    with pytest.raises(ContractViolation):
        run_pattern_sweep((p, p))
