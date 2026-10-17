"""
TEST SUITE: V-OPENVINO-AEP-RUNTIME-01
=============================================================================
Verification of OpenVINO Neural Inference Workload running under AEP L0–L8:
  1. OpenVINO runtime status and hardware topology probe
  2. OpenVINO process prioritization inference with AepReceipt generation
  3. L2 Invariant Gate (Probability normalization, confidence bounds, VITAL_MAX_HP)
  4. L7 Backpressure gating under high UMA bus pressure
  5. DP4A hardware acceleration configuration
"""

import json
import pytest

from krystal_kernel.aep_engine import (
    AepVerdict,
    WorkloadType,
    ExecutionLane,
)
from krystal_kernel.openvino_aep_runtime import (
    OpenVinoAepRuntime,
    OpenVinoInferenceInvariantRule,
    GLOBAL_OPENVINO_AEP_RUNTIME,
    VITAL_MAX_HP,
)


class TestOpenVinoAepRuntime:
    """Verifies OpenVINO execution over the canonical AEP cycle."""

    def test_runtime_status_online_and_backend_discovered(self):
        status = GLOBAL_OPENVINO_AEP_RUNTIME.get_runtime_status()
        assert status["status"] == "ONLINE"
        assert status["runtime"] == "Krystal-OpenVINO-AEP"
        assert status["vital_max_hp"] == 6
        assert status["active_workload"] == WorkloadType.INFERENCE.value
        assert "backend" in status
        assert "hardware_bypass" in status

    def test_models_catalog_structure(self):
        catalog = GLOBAL_OPENVINO_AEP_RUNTIME.get_models_catalog()
        assert len(catalog) >= 2
        model_ids = [m["model_id"] for m in catalog]
        assert "cortex-process-prioritizer-v2" in model_ids
        for m in catalog:
            assert m["vital_max_hp"] == 6
            assert "precision" in m

    def test_openvino_inference_execution_and_receipt(self):
        runtime_id = "session-openvino-test-01"
        res, receipt = GLOBAL_OPENVINO_AEP_RUNTIME.infer_process(
            runtime_id=runtime_id,
            pid=2048,
            name="blender.exe",
            features={
                "cpu_pct": 85.0,
                "cs_rate": 8000.0,
                "page_faults_per_sec": 300.0,
                "working_set_mb": 1024.0,
                "thread_count": 16,
                "io_ops_per_sec": 120.0,
                "kernel_time_ratio": 0.20,
                "thrashing_index": 1.4,
            },
            current_bus_pressure=0.15,
        )

        assert res is not None
        assert res.target_pid == 2048
        assert res.process_name == "blender.exe"
        assert res.predicted_priority_class in ["IDLE", "BELOW_NORMAL", "NORMAL", "ABOVE_NORMAL", "HIGH", "REALTIME"]
        assert 0.0 <= res.confidence_score <= 1.0

        # Receipt verification
        assert receipt.runtime_id == runtime_id
        assert receipt.workload == WorkloadType.INFERENCE
        assert receipt.action == "OPENVINO_PROCESS_INFERENCE"
        assert receipt.execution_lane == ExecutionLane.COMPUTE
        assert receipt.verdict == AepVerdict.COMMITTED
        assert receipt.state_before != receipt.state_after

    def test_openvino_backpressure_rejection(self):
        runtime = OpenVinoAepRuntime(max_bus_pressure=0.75)
        res, receipt = runtime.infer_process(
            runtime_id="session-congested",
            pid=9999,
            name="hog.exe",
            features={"cpu_pct": 99.0},
            current_bus_pressure=0.88, # Above 0.75 threshold
        )

        assert res is None
        assert receipt.verdict == AepVerdict.DISCARDED_BACKPRESSURE_REJECTED
        assert receipt.state_before == receipt.state_after

    def test_invariant_rule_probability_distribution(self):
        rule = OpenVinoInferenceInvariantRule()

        # Valid candidate
        valid_cand = {
            "confidence_score": 0.85,
            "class_probabilities": {"NORMAL": 0.70, "HIGH": 0.30},
            "predicted_priority_class": "NORMAL",
            "vital_max_hp": 6,
        }
        is_ok, reason = rule.evaluate(valid_cand)
        assert is_ok is True

        # Invalid probability sum (1.40 != 1.0)
        invalid_probs = {
            "confidence_score": 0.85,
            "class_probabilities": {"NORMAL": 0.90, "HIGH": 0.50},
            "predicted_priority_class": "NORMAL",
            "vital_max_hp": 6,
        }
        is_ok, reason = rule.evaluate(invalid_probs)
        assert is_ok is False
        assert "Probability distribution invariant violated" in reason

        # Invalid HP
        invalid_hp = {
            "confidence_score": 0.85,
            "predicted_priority_class": "NORMAL",
            "vital_max_hp": 99,
        }
        is_ok, reason = rule.evaluate(invalid_hp)
        assert is_ok is False
        assert "VITAL_MAX_HP invariant breach" in reason

    def test_hardware_acceleration_configuration(self):
        cfg = GLOBAL_OPENVINO_AEP_RUNTIME.configure_acceleration(
            use_dp4a=True,
            streams_count=4,
            enable_kv_u8=True,
        )
        assert cfg["status"] == "CONFIGURED"
        assert cfg["vital_max_hp"] == 6
        assert cfg["configuration"]["inference_precision_hint"] == "i8"
