"""Verification tests for `projects/sovereign_openvino_companion_v2`."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from krystal_kernel.constants import ContractViolation, VITAL_MAX_HP
from projects.sovereign_openvino_companion_v2 import (
    AdaptiveSpendingOptimizer,
    BufferedComputationLogSystem,
    SovereignOpenVinoCompanionV2,
)
from projects.sovereign_openvino_companion_v2.cli import main as v2_cli_main


def test_buffered_logger_coalescing_and_tamper_detection(tmp_path: Path) -> None:
    logger = BufferedComputationLogSystem(log_dir=tmp_path, reset_ledger=True, flush_every=4)
    for i in range(6):
        logger.log_event("telemetry", f"OP_{i}", "S2", 0.1, {"frame_index": i})

    # 4 triggered 1 automatic flush; remaining 2 stay buffered until flush/verify
    assert logger.disk_flush_count == 1
    ok, report = logger.verify_chain_integrity()
    assert ok is True
    assert report["entries_checked"] == 6
    assert logger.disk_flush_count == 2


def test_smart_v2_session_prebrake_parasite_and_candidate_nodes(tmp_path: Path) -> None:
    v2 = SovereignOpenVinoCompanionV2(log_dir=tmp_path, reset_ledger=True, flush_every=16)
    session = v2.run_smart_session(num_frames=12)
    assert session["chain_valid"] is True
    assert session["vital_max_hp"] == 6

    rationales = [f["routing_rationale"] for f in session["sample_frames"]]
    assert "PREDICTIVE_UMA_VELOCITY_PREBRAKE_CPU" in rationales
    assert session["sample_frames"][4]["artifact_released"] is False  # entropy=0.0 parasite cutoff
    assert session["sample_frames"][4]["spending_plan"]["compressed_spend_mj"] == 0.0180

    # Candidate runtime nodes stay strictly separate from 56 frozen theory nodes
    cands = v2.prototype_candidate_runtime_nodes()
    assert cands["frozen_theory_nodes"] == 56
    assert cands["node_57_added_to_frozen_theory"] is False
    assert all(c["p7_directional_gate_pass"] is True for c in cands["candidate_runtime_nodes"])


def test_v1_vs_v2_paired_benchmark_and_cli(tmp_path: Path) -> None:
    v2 = SovereignOpenVinoCompanionV2(log_dir=tmp_path / "v2", reset_ledger=True)
    ab = v2.run_v1_vs_v2_ab_benchmark(iterations=12, tmp_v1_dir=tmp_path / "v1")
    assert ab["v2_chain_valid"] is True
    assert ab["v2_fine_tuned"]["median_ms"] <= ab["v1_baseline"]["median_ms"]
    assert ab["productivity_per_watt"]["v2_paced_12w_ops_per_w"] > ab["productivity_per_watt"]["v1_baseline_15w_ops_per_w"]

    rc = v2_cli_main(["--run-session", "--frames", "6", "--candidate-nodes", "--reset-ledger", "--log-dir", str(tmp_path / "cli")])
    assert rc == 0
    assert VITAL_MAX_HP == 6
