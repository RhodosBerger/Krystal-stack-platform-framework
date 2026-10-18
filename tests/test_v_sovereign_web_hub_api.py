"""
TEST SUITE: V-SOVEREIGN-WEB-HUB-API-01
=============================================================================
Milestone: V-SOVEREIGN-WEB-HUB-API-01
Normative Reference: docs/architecture/AEP_CANONICAL_SPEC.md
Constitutional Invariant: VITAL_MAX_HP == 6

Comprehensive integration test verifying live HTTP REST API endpoints on Krystal Web Hub:
  - GET  /api/plea/status
  - GET  /api/vulkan_usm/status
  - GET  /api/economy/ledger
  - GET  /api/research/status
  - POST /api/plea/evaluate
  - POST /api/vulkan_usm/reconstruct
  - POST /api/economy/transfer
  - POST /api/economy/craft
  - POST /api/research/experiment (commit & rollback)
"""

import json
import urllib.request
import urllib.error
import pytest

from krystal_kernel.constants import VITAL_MAX_HP

BASE_URL = "http://127.0.0.1:8000"


def make_request(path: str, data: dict = None) -> dict:
    url = f"{BASE_URL}{path}"
    headers = {"Content-Type": "application/json"} if data else {}
    body = json.dumps(data).encode("utf-8") if data else None
    req = urllib.request.Request(url, data=body, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            assert resp.status in (200, 202)
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.URLError as e:
        pytest.fail(f"Web Hub connection error on {path}: {e}")


def test_get_plea_status():
    res = make_request("/api/plea/status")
    assert res["status"] == "ONLINE"
    assert res["engine"] == "AepPleaIntegratedEngine"
    assert res["vital_max_hp"] == VITAL_MAX_HP


def test_get_vulkan_usm_status():
    res = make_request("/api/vulkan_usm/status")
    assert res["status"] == "ONLINE"
    assert res["bridge"] == "VulkanOpenVinoUsmBridge"
    assert len(res["registered_buffers"]) >= 2
    assert res["vital_max_hp"] == VITAL_MAX_HP


def test_get_economy_ledger():
    res = make_request("/api/economy/ledger")
    assert res["status"] == "ONLINE"
    assert res["engine"] == "AepEconomyProcessor"
    assert len(res["merkle_root"]) == 64
    assert "alice" in res["wallets"]
    assert "miner_bob" in res["wallets"]
    assert res["vital_max_hp"] == VITAL_MAX_HP


def test_get_research_status():
    res = make_request("/api/research/status")
    assert res["status"] == "ONLINE"
    assert res["actuator"] == "AepResearchActuator"
    assert "pl1_power_watts" in res["tuning_state"]
    assert res["vital_max_hp"] == VITAL_MAX_HP


def test_post_plea_evaluate():
    payload = {
        "action": "EVALUATE_PHYSICAL_HYPOTHESIS",
        "expected_entropy_reduction": 1.2,
        "uma_bytes_estimated": 2048,
        "delta_state": {"score": 99, "vital_max_hp": 6},
    }
    res = make_request("/api/plea/evaluate", payload)
    assert res["status"] == "SUCCESS"
    assert res["committed"] is True
    assert res["vital_max_hp"] == VITAL_MAX_HP
    assert "operation_id" in res["receipt"]


def test_post_vulkan_usm_reconstruct():
    payload = {
        "input_buffer_id": "viewport_540p",
        "output_buffer_id": "target_1080p",
        "bus_pressure": 0.12,
        "frame_time_ms": 7.50,
    }
    res = make_request("/api/vulkan_usm/reconstruct", payload)
    assert res["status"] == "COMMITTED"
    assert res["committed"] is True
    assert res["vital_max_hp"] == VITAL_MAX_HP
    assert res["receipt"]["execution"]["backend"] == "vulkan-implicit-usm"


def test_post_economy_transfer():
    payload = {
        "from_account": "alice",
        "to_account": "miner_bob",
        "asset": "CREDIT",
        "amount": 25.0,
        "system_fee": 0.5,
    }
    res = make_request("/api/economy/transfer", payload)
    assert res["status"] == "COMMITTED"
    assert res["committed"] is True
    assert len(res["merkle_root"]) == 64
    assert res["vital_max_hp"] == VITAL_MAX_HP


def test_post_economy_craft():
    payload = {
        "account_id": "miner_bob",
        "recipe_id": "recipe-alloy-synthesis",
    }
    res = make_request("/api/economy/craft", payload)
    assert res["status"] == "COMMITTED"
    assert res["committed"] is True
    assert len(res["merkle_root"]) == 64
    assert res["vital_max_hp"] == VITAL_MAX_HP


def test_post_research_experiment_commit():
    payload = {
        "target_parameter": "pl1_power_watts",
        "candidate_value": 30.0,
        "objective": "MINIMIZE",
        "simulated_gain": 4.0,
        "force_commit": True,
    }
    res = make_request("/api/research/experiment", payload)
    assert res["status"] == "COMMITTED"
    assert res["committed"] is True
    assert res["tuning_state"]["pl1_power_watts"] == 30.0
    assert res["vital_max_hp"] == VITAL_MAX_HP


def test_post_research_experiment_rollback():
    # Attempt candidate with force_commit=False (simulating neutral/regression)
    payload = {
        "target_parameter": "pl1_power_watts",
        "candidate_value": 45.0,
        "objective": "MINIMIZE",
        "simulated_gain": 0.0,
        "force_commit": False,
    }
    res = make_request("/api/research/experiment", payload)
    assert res["status"] == "ROLLED_BACK"
    assert res["committed"] is False
    # Value must remain unmutated (Delta S == 0)
    assert res["tuning_state"]["pl1_power_watts"] == 30.0
    assert res["vital_max_hp"] == VITAL_MAX_HP


def test_get_sovereign_benchmark_latest():
    res = make_request("/api/benchmark/sovereign")
    assert res["suite"] == "SOVEREIGN_FABRIC_EFFICIENCY_BENCHMARK"
    assert res["vital_max_hp"] == VITAL_MAX_HP
    assert "summary" in res
    assert res["summary"]["usm_latency_speedup_factor"] > 1.0


def test_post_run_sovereign_benchmark():
    res = make_request("/api/benchmark/sovereign", {"scale": 1})
    assert res["suite"] == "SOVEREIGN_FABRIC_EFFICIENCY_BENCHMARK"
    assert res["vital_max_hp"] == VITAL_MAX_HP
    assert res["summary"]["economy_sustained_tps"] > 500.0
    assert res["summary"]["delta_s_zero_verified"] is True
