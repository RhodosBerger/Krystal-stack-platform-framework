"""Tests for the architectural consistency bot."""

from __future__ import annotations

import json

import pytest

from projects.trigonometric_hybrid_platform_framework.consistency_bot import CHECKS, BotCheck, run_bot

ALLOWED = {"VERIFIED_WITH_SCOPE", "MEASURED_WITH_SCOPE", "SUPPORTED_WITH_SCOPE", "CONFLICT", "FALSIFIED", "UNKNOWN"}


def test_bot_fast_run_writes_multimodal_report(tmp_path) -> None:
    pytest.importorskip("openvino")
    pytest.importorskip("PIL")
    from PIL import Image

    rep = run_bot(fast=True, out_dir=tmp_path)
    assert len(rep["checks"]) == len(CHECKS)
    assert all(r["verdict"] in ALLOWED for r in rep["checks"])
    data = json.loads((tmp_path / "consistency_bot_latest.json").read_text(encoding="utf-8"))
    assert data["summary"] == rep["summary"]
    assert "| C01 |" in (tmp_path / "consistency_bot_latest.md").read_text(encoding="utf-8")
    with Image.open(tmp_path / "consistency_bot_latest.png") as im:
        assert im.size[0] == 1200 and im.size[1] > 300


def test_exception_becomes_unknown_not_pass(tmp_path) -> None:
    pytest.importorskip("PIL")

    def boom(fast: bool):
        raise RuntimeError("simulated failure")

    rep = run_bot(fast=True, out_dir=tmp_path, checks=(BotCheck("X1", "always fails", "n/a", "TEST", boom),))
    assert rep["checks"][0]["verdict"] == "UNKNOWN"
    assert "simulated failure" in rep["checks"][0]["measured"]["error"]
