"""
KRYSTAL-STACK // OPENVINO AEP RUNTIME & INFERENCE FABRIC
=============================================================================
Workload: INFERENCE
Normative Architecture: docs/workloads/INFERENCE_WORKLOAD.md
Classification: [IMPLEMENTED]

Key Capabilities:
  1. OpenVinoAepRuntime executing over AEP layers L0–L8
  2. OpenVinoInferenceInvariantRule (probability normalization & confidence bounds)
  3. Integrated hardware accelerator topology probe (CPU DP4A / Iris Xe / NPU)
  4. Standardized AepReceipt generation with WorkloadType.INFERENCE
"""

from __future__ import annotations

import dataclasses
import json
import math
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

from krystal_kernel.aep_engine import (
    AepReceipt,
    AepTransactionalEngine,
    AepVerdict,
    ExecutionLane,
    InvariantRule,
    WorkloadType,
    compute_state_digest,
)
from krystal_kernel.cortex_openvino_engine import (
    GLOBAL_OPENVINO_GOVERNOR,
    OpenVINOInferenceResult,
    WIN32_PRIORITY_CLASSES,
)
from krystal_kernel.openvino_hardware_acceleration_pack import (
    HardwareAccelerationBenchmark,
    OpenVinoIrisXeDP4APack,
    TigerLakeTurboUnblocker,
)

VITAL_MAX_HP: int = 6


class OpenVinoInferenceInvariantRule(InvariantRule):
    """
    L2 Invariant:
      1. Confidence score must strictly remain within [0.0, 1.0].
      2. Sum of class probabilities must approximate 1.0 (|sum - 1.0| <= 0.05).
      3. Predicted priority class must belong to the valid priority class catalog.
      4. VITAL_MAX_HP must remain exactly 6.
    """
    def evaluate(self, candidate_state: Any) -> Tuple[bool, str]:
        if not isinstance(candidate_state, dict):
            if hasattr(candidate_state, "to_dict"):
                candidate_state = candidate_state.to_dict()
            else:
                return False, "Candidate inference state must be mapping or serializable"

        confidence = candidate_state.get("confidence_score")
        if confidence is not None:
            if not isinstance(confidence, (int, float)) or confidence < 0.0 or confidence > 1.0:
                return False, f"Confidence score invariant violated: {confidence} not in [0.0, 1.0]"

        probs = candidate_state.get("class_probabilities")
        if probs and isinstance(probs, dict):
            prob_sum = sum(probs.values())
            if abs(prob_sum - 1.0) > 0.05:
                return False, f"Probability distribution invariant violated: sum {prob_sum:.4f} != 1.0"

        priority = candidate_state.get("predicted_priority_class")
        if priority and priority not in WIN32_PRIORITY_CLASSES:
            return False, f"Invalid priority class predicted: {priority}"

        if candidate_state.get("vital_max_hp", VITAL_MAX_HP) != VITAL_MAX_HP:
            return False, f"VITAL_MAX_HP invariant breach in OpenVINO state: must be {VITAL_MAX_HP}"

        return True, "OPENVINO_INVARIANTS_SATISFIED"


class OpenVinoAepRuntime:
    """
    Authoritative OpenVINO runtime implementing the AEP L0–L8 execution cycle.
    """

    def __init__(self, max_bus_pressure: float = 0.80) -> None:
        self.aep_engine = AepTransactionalEngine(
            invariants=[OpenVinoInferenceInvariantRule()],
            max_pressure_threshold=max_bus_pressure,
        )
        self.governor = GLOBAL_OPENVINO_GOVERNOR

    def infer_process(
        self,
        runtime_id: str,
        pid: int = 1000,
        name: str = "worker.exe",
        features: Optional[Dict[str, float]] = None,
        current_bus_pressure: float = 0.10,
    ) -> Tuple[Optional[OpenVINOInferenceResult], AepReceipt]:
        """
        Executes a process prioritization neural inference under the AEP L0–L8 cycle.
        """
        f = features or {}
        cpu_pct = float(f.get("cpu_pct", 50.0))
        cs_rate = float(f.get("cs_rate", 6500.0))
        faults = float(f.get("page_faults_per_sec", 120.0))
        mem = float(f.get("working_set_mb", 256.0))
        threads = int(f.get("thread_count", 8))
        io_ops = float(f.get("io_ops_per_sec", 50.0))
        k_ratio = float(f.get("kernel_time_ratio", 0.15))
        thrash = float(f.get("thrashing_index", 1.1))

        initial_state = {
            "pid": pid,
            "name": name,
            "status": "INITIALIZED",
            "vital_max_hp": VITAL_MAX_HP,
        }

        def transition_inference(state: Dict[str, Any], payload: Dict[str, Any]) -> Dict[str, Any]:
            res = self.governor.infer_process_priority(
                pid=pid,
                name=name,
                cpu_pct=cpu_pct,
                cs_rate=cs_rate,
                page_faults_per_sec=faults,
                working_set_mb=mem,
                thread_count=threads,
                io_ops_per_sec=io_ops,
                kernel_time_ratio=k_ratio,
                thrashing_index=thrash,
            )
            state_dict = res.to_dict()
            state_dict["status"] = "INFERRED"
            return state_dict

        tx_result = self.aep_engine.execute_transition(
            runtime_id=runtime_id,
            workload=WorkloadType.INFERENCE,
            action="OPENVINO_PROCESS_INFERENCE",
            current_state=initial_state,
            transition_fn=transition_inference,
            action_payload=f,
            policy_id="openvino-dp4a-heterogeneous",
            policy_revision=1,
            execution_lane=ExecutionLane.COMPUTE,
            backend="openvino-neural-engine",
            device=getattr(self.governor, "preferred_device", "CPU"),
            current_bus_pressure=current_bus_pressure,
            deadline_budget_us=20000,
        )

        if tx_result.committed and isinstance(tx_result.state, dict):
            # Reconstruct result object
            inference_obj = OpenVINOInferenceResult(
                target_pid=tx_result.state.get("target_pid", pid),
                process_name=tx_result.state.get("process_name", name),
                input_feature_vector=tx_result.state.get("input_feature_vector", []),
                predicted_priority_class=tx_result.state.get("predicted_priority_class", "NORMAL"),
                win32_priority_code=tx_result.state.get("win32_priority_code", 32),
                confidence_score=tx_result.state.get("confidence_score", 0.5),
                class_probabilities=tx_result.state.get("class_probabilities", {}),
                inference_device=tx_result.state.get("inference_device", "CPU-OpenVINO"),
                inference_latency_us=tx_result.state.get("inference_latency_us", 100.0),
                openvino_backend=tx_result.state.get("openvino_backend", "OPENVINO_DP4A"),
                vital_max_hp=VITAL_MAX_HP,
            )
            return inference_obj, tx_result.receipt
        else:
            return None, tx_result.receipt

    def get_runtime_status(self) -> Dict[str, Any]:
        """Returns comprehensive OpenVINO device and hardware topology status."""
        unblocker_status = TigerLakeTurboUnblocker.apply_hardware_bypass().to_dict()
        return {
            "status": "ONLINE",
            "runtime": "Krystal-OpenVINO-AEP",
            "device": getattr(self.governor, "preferred_device", "CPU"),
            "backend": getattr(self.governor, "backend_name", "OpenVINO-DirectSim"),
            "openvino_version": "2026.4.1-LTS",
            "dp4a_accelerated": True,
            "vital_max_hp": VITAL_MAX_HP,
            "hardware_bypass": unblocker_status,
            "active_workload": WorkloadType.INFERENCE.value,
        }

    def get_models_catalog(self) -> List[Dict[str, Any]]:
        """Returns catalog of deployed OpenVINO models."""
        return [
            {
                "model_id": "cortex-process-prioritizer-v2",
                "domain": "PROCESS_SCHEDULING_AND_QOS",
                "format": "ONNX / OpenVINO IR",
                "precision": "INT8 (DP4A) / FP16",
                "latency_target_us": 500,
                "input_dimensions": [1, 8],
                "output_classes": 6,
                "vital_max_hp": VITAL_MAX_HP,
            },
            {
                "model_id": "cortex-integrity-evaluator-v3",
                "domain": "NEUROMORPHIC_ENTROPY_ANALYSIS",
                "format": "OpenVINO Poetic Tropes IR",
                "precision": "FP16 / INT8",
                "latency_target_us": 1200,
                "input_dimensions": [1, 16],
                "output_classes": 4,
                "vital_max_hp": VITAL_MAX_HP,
            }
        ]

    def configure_acceleration(
        self,
        use_dp4a: bool = True,
        streams_count: int = 4,
        enable_kv_u8: bool = True,
    ) -> Dict[str, Any]:
        """Applies hardware acceleration configuration to the OpenVINO pipeline."""
        cfg = OpenVinoIrisXeDP4APack.generate_extended_config(
            use_dp4a_int8=use_dp4a,
            streams_count=streams_count,
            enable_kv_u8=enable_kv_u8,
        )
        return {
            "status": "CONFIGURED",
            "configuration": cfg.to_dict(),
            "vital_max_hp": VITAL_MAX_HP,
        }

    def run_benchmark(self, iterations: int = 50) -> Dict[str, Any]:
        """Runs comparative hardware benchmark."""
        bench = HardwareAccelerationBenchmark.run_comparative_benchmark()
        bench_dict = bench.to_dict()
        bench_dict["benchmark_id"] = "BENCH-OPENVINO-DP4A-01"
        bench_dict["aep_workload"] = WorkloadType.INFERENCE.value
        return bench_dict


# Global Singleton for Web Hub Integration
GLOBAL_OPENVINO_AEP_RUNTIME = OpenVinoAepRuntime()
