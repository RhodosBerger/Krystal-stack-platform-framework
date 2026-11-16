"""Tests for convolution contra modules (adjoint, inverse, ReLU partial inverse)."""

from __future__ import annotations

import pytest

from krystal_kernel.constants import ContractViolation
from projects.trigonometric_hybrid_platform_framework.virtual_convolution_contra import run_contra_modules


def test_adjoint_and_linear_inverse() -> None:
    res = run_contra_modules(channels=2, filters=4, height=12, width=12, cg_iterations=200, use_openvino=False)
    assert res["contra1_adjoint_dot_test_rel_gap"] < 1e-5
    assert res["contra2_linear_inverse"]["reconstruction_rel_error"] < 1e-3
    # ReLU removes information: partial inverse must be worse than the linear inverse.
    assert (
        res["contra3_relu_partial_inverse"]["reconstruction_rel_error"]
        > res["contra2_linear_inverse"]["reconstruction_rel_error"]
    )


def test_openvino_transpose_conv_matches_numpy_adjoint() -> None:
    pytest.importorskip("openvino")
    res = run_contra_modules(channels=2, filters=4, height=12, width=12, cg_iterations=20, use_openvino=True)
    ov_info = res["contra1_openvino_transpose_conv"]
    assert ov_info["used"] is True, ov_info
    assert ov_info["max_rel_error_vs_numpy_adjoint"] < 1e-4


def test_underdetermined_rejected() -> None:
    with pytest.raises(ContractViolation):
        run_contra_modules(channels=8, filters=4)
