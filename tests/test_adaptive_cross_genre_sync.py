"""Tests for the adaptive cross-genre synchronization module."""

from __future__ import annotations

import pytest

from krystal_kernel.constants import ContractViolation
from projects.trigonometric_hybrid_platform_framework.adaptive_cross_genre_sync import (
    SYNC_CONNECTIONS,
    SYNC_NODES,
    AdaptiveHorizonController,
    SyncConnection,
    SyncMode,
    describe_adaptive_sync_architecture,
    join_sync_policy,
    simulate_adaptive_rollback,
    validate_connection_graph,
)


def test_connection_graph_valid_and_rejects_authority_bypass() -> None:
    res = validate_connection_graph()
    assert res["graph_valid"] is True
    assert res["edge_count"] == 16
    assert res["wire_edges"] == 9

    bypass = SyncConnection("EXX", "CLIENT_ACTION_6D", "KERNEL_STATE", "direct write", "WIRE_128B", 128, "none")
    with pytest.raises(ContractViolation):
        validate_connection_graph(SYNC_NODES, SYNC_CONNECTIONS + (bypass,))

    bad_frame = SyncConnection("EYY", "CLIENT_RTS_GRID", "PAIR_JOIN_ARBITER", "proposal", "WIRE_128B", 64, "none")
    with pytest.raises(ContractViolation):
        validate_connection_graph(SYNC_NODES, SYNC_CONNECTIONS + (bad_frame,))

    observer_emit = SyncConnection("EZZ", "CLIENT_TWIN_OBSERVER", "PAIR_JOIN_ARBITER", "x", "WIRE_128B", 128, "none")
    with pytest.raises(ContractViolation):
        validate_connection_graph(SYNC_NODES, SYNC_CONNECTIONS + (observer_emit,))

    assert describe_adaptive_sync_architecture()["graph_validation"]["graph_valid"] is True


def test_strictest_policy_join() -> None:
    assert join_sync_policy("GENRE_03_ACTION_BALLISTIC_6D", "GENRE_02_RTS_STRATEGIC_3D_GRID") == SyncMode.INPUT_DELAY_LOCKSTEP
    assert join_sync_policy("GENRE_03_ACTION_BALLISTIC_6D", "GENRE_04_RPG_QUEST_NARRATIVE") == SyncMode.AUTHORITATIVE_LEDGER
    with pytest.raises(ContractViolation):
        join_sync_policy("GENRE_03_ACTION_BALLISTIC_6D", "GENRE_05_SIMULATION_DIGITAL_TWIN")


def test_horizon_controller_upshift_and_hysteresis() -> None:
    ctrl = AdaptiveHorizonController()
    for _ in range(40):
        ctrl.observe(100.0)
        ctrl.tick()
    peak = ctrl.horizon
    assert peak == 6  # ceil(100 / 16.667) = 6 at zero variance

    for _ in range(5):
        ctrl.observe(10.0)
        ctrl.tick()
    assert ctrl.horizon == peak  # no immediate downshift

    for _ in range(400):
        ctrl.observe(10.0)
        ctrl.tick()
    assert ctrl.horizon == 1


@pytest.mark.parametrize(
    "latency_ms,jitter_ms",
    [(10.0, 2.0), (40.0, 6.0), (100.0, 20.0)],
)
def test_rollback_simulation_zero_desync(latency_ms: float, jitter_ms: float) -> None:
    res = simulate_adaptive_rollback(ticks=1500, one_way_latency_ms=latency_ms, jitter_sigma_ms=jitter_ms, seed=6)
    assert res["desync_ticks"] == 0
    assert res["kernel_final_checksum"] == res["client_final_checksum"]
    assert res["max_hp_observed"] <= 6


def test_horizon_grows_with_latency() -> None:
    lan = simulate_adaptive_rollback(ticks=1000, one_way_latency_ms=10.0, jitter_sigma_ms=2.0)
    wan = simulate_adaptive_rollback(ticks=1000, one_way_latency_ms=100.0, jitter_sigma_ms=20.0)
    assert lan["max_horizon"] < wan["max_horizon"]
