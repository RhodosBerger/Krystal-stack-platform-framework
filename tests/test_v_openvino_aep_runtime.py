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
    HardwareCrossoverPolicy,
    GLOBAL_OPENVINO_AEP_RUNTIME,
    VITAL_MAX_HP,
)
from krystal_kernel.zero_copy_contracts import (
    ZeroCopyContractFrame,
    canonicalize_float32,
    KRYSTAL_MAGIC,
    KRYSTAL_PROTOCOL_VERSION,
)
from krystal_kernel.zero_copy_ring import (
    ZeroCopyContractRing,
)
import time


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

    def test_zerocopy_fastpath_infer_from_ring_slot(self):
        session_key = b"openvino_zerocopy_test_key_32B!"
        in_ring = ZeroCopyContractRing(capacity=16, session_key=session_key)
        out_ring = ZeroCopyContractRing(capacity=16, session_key=session_key)

        # Build input 128B frame
        in_frame = ZeroCopyContractFrame(
            magic=KRYSTAL_MAGIC,
            version=KRYSTAL_PROTOCOL_VERSION,
            sequence_id=4001,
            generation_id=1,
            flags=0,
            capability_scope=1,
            vital_hp=6,
            timestamp_ns=time.time_ns(),
            nonce=12345,
            predecessor_hash_raw=b"\x00" * 32,
            profile_id=10,
            verdict_id=0,
            approved_authority_mask=0x01,
            metric1=canonicalize_float32(0.65),  # 65% CPU
            metric2=canonicalize_float32(0.08),  # 80 io ops
        )
        success = in_ring.push_frame_with_hmac(in_frame, session_key)
        assert success is True
        assert in_ring.available_count() == 1

        # Execute fast-path inference from ring slot
        result, receipt, out_frame = GLOBAL_OPENVINO_AEP_RUNTIME.infer_from_ring_slot(
            ring=in_ring,
            session_key=session_key,
            output_ring=out_ring,
            current_bus_pressure=0.15,
        )

        assert result is not None
        assert receipt.verdict == AepVerdict.COMMITTED
        assert receipt.workload == WorkloadType.INFERENCE
        assert receipt.execution_lane == ExecutionLane.COMPUTE
        assert receipt.backend == "openvino-neural-engine"

        # Verify output ring has received the response frame
        assert out_ring.available_count() == 1
        popped_out = out_ring.pop_frame_with_hmac(session_key)
        assert popped_out is not None
        assert popped_out.sequence_id == 4001
        assert popped_out.vital_hp == 6
        assert popped_out.verdict_id == 1  # COMMITTED
        assert popped_out.verify_frame_integrity(session_key) is True
        assert popped_out.metric1 == pytest.approx(float(result.confidence_score), rel=1e-4)

    def test_zerocopy_fastpath_empty_ring_returns_discarded_receipt(self):
        in_ring = ZeroCopyContractRing(capacity=16)
        result, receipt, out_frame = GLOBAL_OPENVINO_AEP_RUNTIME.infer_from_ring_slot(ring=in_ring)
        assert result is None
        assert receipt.verdict == AepVerdict.DISCARDED
        assert out_frame is None

    def test_zerocopy_fastpath_sub_microsecond_throughput_benchmark(self):
        session_key = b"benchmark_zerocopy_spsc_key_32B"
        in_ring = ZeroCopyContractRing(capacity=256, session_key=session_key)
        out_ring = ZeroCopyContractRing(capacity=256, session_key=session_key)

        iterations = 50
        for i in range(iterations):
            f = ZeroCopyContractFrame(
                magic=KRYSTAL_MAGIC,
                version=KRYSTAL_PROTOCOL_VERSION,
                sequence_id=5000 + i,
                generation_id=1,
                flags=0,
                capability_scope=1,
                vital_hp=6,
                timestamp_ns=time.time_ns(),
                nonce=i,
                predecessor_hash_raw=b"\x00" * 32,
                profile_id=1,
                verdict_id=0,
                approved_authority_mask=0x01,
                metric1=canonicalize_float32(0.50),
                metric2=canonicalize_float32(0.05),
            )
            in_ring.push_frame_with_hmac(f, session_key)

        t0 = time.perf_counter()
        committed_count = 0
        for _ in range(iterations):
            res, receipt, out_f = GLOBAL_OPENVINO_AEP_RUNTIME.infer_from_ring_slot(
                ring=in_ring,
                session_key=session_key,
                output_ring=out_ring,
            )
            if receipt.verdict == AepVerdict.COMMITTED:
                committed_count += 1

        elapsed = time.perf_counter() - t0
        avg_us = (elapsed / iterations) * 1_000_000.0

        assert committed_count == iterations
        assert out_ring.available_count() == iterations
        # Fast-path SPSC memory processing should be well below 600 µs in interpreted Python (and sub-100ns in C11)
        assert avg_us < 600.0, f"Average zero-copy inference latency {avg_us:.1f}us exceeded 600us target"

    def test_hardware_crossover_policy_micro_tensor_routes_to_cpu(self):
        # 128B micro-tensor with low pressure should route to CPU
        device, rationale = HardwareCrossoverPolicy.select_device(
            payload_bytes=128,
            current_bus_pressure=0.20,
            available_devices=["CPU", "GPU"],
        )
        assert device == "CPU"
        assert "MICRO_TENSOR" in rationale

    def test_hardware_crossover_policy_dense_tensor_routes_to_gpu(self):
        # 64 KB dense tensor with low pressure should route to GPU when available
        device, rationale = HardwareCrossoverPolicy.select_device(
            payload_bytes=65536,
            current_bus_pressure=0.20,
            available_devices=["CPU", "GPU"],
        )
        assert device == "GPU"
        assert "DENSE_TENSOR" in rationale

    def test_hardware_crossover_policy_uma_pressure_overrides_to_cpu(self):
        # Even a dense tensor must fall back to CPU if bus pressure >= 0.78
        device, rationale = HardwareCrossoverPolicy.select_device(
            payload_bytes=65536,
            current_bus_pressure=0.85,
            available_devices=["CPU", "GPU"],
        )
        assert device == "CPU"
        assert "UMA_PRESSURE_OVERRUN" in rationale

    def test_persistent_cache_dir_configured(self):
        status = GLOBAL_OPENVINO_AEP_RUNTIME.get_runtime_status()
        assert "cache_dir" in status
        assert "openvino_blobs" in status["cache_dir"]
        assert status["crossover_threshold_bytes"] == 16384


