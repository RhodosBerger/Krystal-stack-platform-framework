"""Tests for matrix control processes (Freivalds + operator mashup)."""

from __future__ import annotations

import numpy as np

from projects.trigonometric_hybrid_platform_framework.matrix_control_processes import (
    freivalds_verify,
    run_operator_mashup,
)


def test_freivalds_accepts_correct_and_rejects_single_corruption() -> None:
    rng = np.random.default_rng(1)
    a = rng.standard_normal((128, 96)).astype(np.float32)
    b = rng.standard_normal((96, 80)).astype(np.float32)
    c = a @ b
    assert freivalds_verify(a, b, c)["accepted"] is True
    bad = c.copy()
    bad[17, 33] += 1.0
    assert freivalds_verify(a, b, bad)["accepted"] is False


def test_operator_mashup_laws_and_negative_control() -> None:
    res = run_operator_mashup(max_len=4)
    assert res["words_checked"] == 8          # K, KT, K·KT, KT·K, and the 3- and 4-letter alternations
    assert res["all_laws_hold"] is True
    assert res["negative_control_detected"] is True
    assert res["verdict"] == "VERIFIED_WITH_SCOPE"
