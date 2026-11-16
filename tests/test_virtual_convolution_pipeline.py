"""Tests for the virtual convolution pipeline (NumPy reference + OpenVINO CPU)."""

from __future__ import annotations

import numpy as np
import pytest

from krystal_kernel.constants import ContractViolation
from projects.trigonometric_hybrid_platform_framework.virtual_convolution_pipeline import (
    conv2d_reference,
    run_virtual_convolution_pipeline,
)


def test_reference_conv_identity_kernel() -> None:
    x = np.arange(2 * 5 * 5, dtype=np.float32).reshape(1, 2, 5, 5)
    w = np.zeros((2, 2, 3, 3), dtype=np.float32)
    w[0, 0, 1, 1] = 1.0
    w[1, 1, 1, 1] = 1.0
    assert np.array_equal(conv2d_reference(x, w), x)


def test_reference_conv_padding_sum_kernel() -> None:
    x = np.ones((1, 1, 3, 3), dtype=np.float32)
    w = np.ones((1, 1, 3, 3), dtype=np.float32)
    out = conv2d_reference(x, w)[0, 0]
    assert out[1, 1] == 9 and out[0, 0] == 4 and out[0, 1] == 6


def test_openvino_cpu_matches_reference() -> None:
    pytest.importorskip("openvino")
    res = run_virtual_convolution_pipeline(channels=4, filters=8, height=16, width=16, iterations=3, devices=["CPU"])
    cpu = res["devices"]["CPU"]
    assert cpu["matches_reference"] is True
    assert cpu["verdict"] == "MEASURED_WITH_SCOPE"


def test_bounds() -> None:
    with pytest.raises(ContractViolation):
        run_virtual_convolution_pipeline(vital_hp=5)
    with pytest.raises(ContractViolation):
        run_virtual_convolution_pipeline(channels=512, height=256, width=256)


def test_summation_order_fingerprint_discriminates_orders() -> None:
    pytest.importorskip("openvino")
    from projects.trigonometric_hybrid_platform_framework.virtual_convolution_pipeline import diagnose_bit_identity

    r = diagnose_bit_identity(channels=4, filters=8, height=16, width=16)
    assert r["status"] == "PATTERN_CANDIDATE"
    # Different summation orders must be distinguishable, otherwise the fingerprint has no power.
    assert r["variant_cross_check"]["seq_natural_equals_seq_reversed"] is False
    for dev in r["devices"].values():
        assert dev["kernels"], "runtime model must report the convolution kernel"
