"""Convolutional Interposer Level between Engine Operations and Hypervisor Steps.

Bridges last week's modules:
  - `StepExecutionGateProxy` (`projects/trigonometric_hybrid_platform_framework/platform_engine.py`)
  - `WslNtCoexistenceBinaryBridge` (`projects/sovereign_openvino_companion_v2/sensory_morphism_binary_compiler.py`)
with this week's modules:
  - OpenVINO `opset13` 2-stage temporal-operational convolution verified against NumPy
  - `128 B KPKT` intranet hypervisor packets & WSL2 Ubuntu quota modulation
  - Fixed `Double-Time` (`tau_real + tau_spec`) lookahead clamping.
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence

import numpy as np

from krystal_kernel.constants import UMA_SATURATION_THRESHOLD, VITAL_MAX_HP, ContractViolation
from krystal_kernel.epistemic_cognition_compositor import MultiDataHybridCompositor
from projects.sovereign_openvino_companion_v2.sensory_morphism_binary_compiler import WslNtCoexistenceBinaryBridge

from .combat_turn_planner import DEFAULT_CACHE_DIR
from .dep_bridge_numeric_and_packet_abi import emit_governed_128b_event, sanitize_vector_f32
from .intel_accelerator_and_power_thread_governor import get_shared_ov_core
from .platform_engine import StepExecutionGateProxy
from .wsl_hypervisor_mortar_and_geotile_adapter import configure_wsl_ubuntu_hypervisor_bridge

OP_CHANNELS: int = 6   # [uma, log_payload, log_batch, conflict_flag, keystroke_flag, gonio_cos]
WINDOW_STEPS: int = 16


def _deterministic_conv_kernels() -> tuple[np.ndarray, np.ndarray]:
    """Build deterministic (out_ch, in_ch, 1, 3) temporal-operational convolution kernels."""
    rng = np.random.default_rng(6)
    w1 = rng.normal(0.0, 0.28, size=(8, OP_CHANNELS, 1, 3)).astype(np.float32)
    # Emphasize temporal smoothing [0.25, 0.50, 0.25] on channel 0 (UMA) so 1-tick spikes don't thrash
    w1[0, 0, 0, :] = np.array([0.25, 0.50, 0.25], dtype=np.float32)
    w2 = rng.normal(0.0, 0.25, size=(4, 8, 1, 3)).astype(np.float32)
    w2[2, 0, 0, :] = np.array([0.35, 0.45, 0.20], dtype=np.float32)
    return np.ascontiguousarray(w1), np.ascontiguousarray(w2)


def _numpy_conv1d_relu_2stage(x: np.ndarray, w1: np.ndarray, w2: np.ndarray) -> np.ndarray:
    """Exact NumPy reference for 2-stage valid (1x3) convolution + ReLU over (1, C, 1, W)."""
    _, _, _, length = x.shape
    l1 = length - 2
    h1 = np.zeros((1, w1.shape[0], 1, l1), dtype=np.float32)
    for oc in range(w1.shape[0]):
        for ic in range(w1.shape[1]):
            for k in range(3):
                h1[0, oc, 0, :] += x[0, ic, 0, k : k + l1] * w1[oc, ic, 0, k]
    h1 = np.maximum(h1, np.float32(0.0))

    l2 = l1 - 2
    h2 = np.zeros((1, w2.shape[0], 1, l2), dtype=np.float32)
    for oc in range(w2.shape[0]):
        for ic in range(w2.shape[1]):
            for k in range(3):
                h2[0, oc, 0, :] += h1[0, ic, 0, k : k + l2] * w2[oc, ic, 0, k]
    return np.maximum(h2, np.float32(0.0))


class ConvolutionalHypervisorStepBridge:
    """OpenVINO convolutional level between operations streams and governed hypervisor steps."""

    def __init__(self) -> None:
        if VITAL_MAX_HP != 6:
            raise ContractViolation("VITAL_MAX_HP != 6 in ConvolutionalHypervisorStepBridge", violation_code="ERR_CONV_HV_HP")
        self.w1, self.w2 = _deterministic_conv_kernels()
        ov, core, _ = get_shared_ov_core(DEFAULT_CACHE_DIR)
        ops = ov.opset13
        inp = ops.parameter([1, OP_CHANNELS, 1, WINDOW_STEPS], np.float32, name="op_window")
        c1 = ops.relu(ops.convolution(inp, ops.constant(self.w1), [1, 1], [0, 0], [0, 0], [1, 1]))
        c2 = ops.relu(ops.convolution(c1, ops.constant(self.w2), [1, 1], [0, 0], [0, 0], [1, 1]))
        self._ov_model = ov.Model([c2], [inp], "conv_ops_to_hypervisor_bridge")
        self._compiled = core.compile_model(self._ov_model, "CPU", {"INFERENCE_PRECISION_HINT": "f32"})
        self.step_proxy = StepExecutionGateProxy()
        self.wntc_bridge = WslNtCoexistenceBinaryBridge()
        self.compositor = MultiDataHybridCompositor()

    def evaluate_operations_to_hypervisor_step(
        self,
        operations_window: Sequence[Sequence[float]],
        frame_budget_ms: float = 8.333,
    ) -> Dict[str, Any]:
        """Run OpenVINO conv interposer, verify against NumPy, and dispatch governed hypervisor steps."""
        if len(operations_window) != WINDOW_STEPS:
            raise ContractViolation(f"Expected {WINDOW_STEPS} operation steps, got {len(operations_window)}", violation_code="ERR_CONV_HV_LEN")

        sanitized_cols: List[List[float]] = []
        for step_vec in operations_window:
            if len(step_vec) != OP_CHANNELS:
                raise ContractViolation(f"Each operation vector must have {OP_CHANNELS} channels", violation_code="ERR_CONV_HV_CH")
            sanitized_cols.append(sanitize_vector_f32(step_vec))

        # Shape (1, C=6, H=1, W=16)
        x = np.array(sanitized_cols, dtype=np.float32).T.reshape(1, OP_CHANNELS, 1, WINDOW_STEPS)
        x = np.ascontiguousarray(x)

        req = self._compiled.create_infer_request()
        ov_feat = np.array(req.infer({0: x})[self._compiled.output(0)], copy=True)
        np_feat = _numpy_conv1d_relu_2stage(x, self.w1, self.w2)
        denom = float(np.linalg.norm(np_feat)) or 1.0
        rel_err = float(np.linalg.norm(ov_feat - np_feat) / denom)
        if rel_err >= 1e-4:
            raise ContractViolation(f"Conv interposer drift {rel_err:.3e} >= 1e-4", violation_code="ERR_CONV_HV_DRIFT")

        # Smooth UMA across the latest 3 operation steps (3-tap [0.25, 0.50, 0.25] filter) vs raw instant UMA
        raw_last_uma = float(x[0, 0, 0, -1])
        conv_smoothed_uma = float(0.25 * x[0, 0, 0, -3] + 0.50 * x[0, 0, 0, -2] + 0.25 * x[0, 0, 0, -1])
        effective_uma = round(max(raw_last_uma, conv_smoothed_uma) if raw_last_uma >= UMA_SATURATION_THRESHOLD else conv_smoothed_uma, 4)

        pooled = np.mean(ov_feat[0, :, 0, :], axis=1)
        scores = {
            "cpu_4t_affinity": round(float(pooled[0]), 4),
            "gpu_iris_xe_burst": round(float(pooled[1]), 4),
            "wsl2_quota_pressure": round(float(pooled[2]), 4),
            "double_time_lookahead": round(float(pooled[3]), 4),
        }

        wsl_base = int(configure_wsl_ubuntu_hypervisor_bridge()["hypervisor_allocation"]["assigned_ram_gb"])
        wntc = self.wntc_bridge.build_coexistence_matrix_and_binary(
            capability_scope_mask=0x0F if effective_uma >= UMA_SATURATION_THRESHOLD else 0x1F
        )
        dt = self.compositor.run_double_time_compositor_cycle(
            frame_budget_ms=frame_budget_ms,
            speculative_speedup_factor=4,
            uma_utilization=effective_uma,
        )

        # Pass through last week's StepExecutionGateProxy:
        # if UMA >= 0.78, StepExecutionGateProxy raises ContractViolation keeping committed_state untouched (Delta_S == 0),
        # and we route cleanly to the CPU shadow hypervisor fallback step!
        pre_step_count = self.step_proxy.committed_state["step_count"]
        try:
            proxy_res = self.step_proxy.execute_step(
                step_id=f"HV_STEP_{pre_step_count + 1}",
                proposer_id="openvino_conv_interposer",
                verifier_id="numpy_dual_verifier",
                step_fn=lambda: {
                    "selected_lane": "GPU_IRIS_XE_WARM" if scores["gpu_iris_xe_burst"] > scores["cpu_4t_affinity"] else "CPU_4T_PHYSICAL_AVX512",
                    "wsl2_ram_gb": wsl_base,
                    "wntc_crc32_valid": wntc["self_binary_decoded"]["crc32_valid"],
                },
                vital_hp=VITAL_MAX_HP,
                uma_utilization=effective_uma,
            )
            hypervisor_mode = "COMMITTED_VIA_STEP_PROXY"
            wsl_ram_gb = wsl_base
            lane = proxy_res["output"]["selected_lane"]
            proxy_receipt = proxy_res["receipt"]
        except ContractViolation:
            # Constitutional Delta_S == 0 preserved on proxy; trigger OP_01 hypervisor fallback
            hypervisor_mode = "OP_01_SHADOW_FALLBACK_DELTA_S_ZERO"
            wsl_ram_gb = max(4, wsl_base - 2)
            lane = "CPU_4T_PHYSICAL_AVX512"
            proxy_receipt = {
                "committed": False,
                "delta_s": 0,
                "step_count_unchanged": self.step_proxy.committed_state["step_count"] == pre_step_count,
            }

        pkt128 = emit_governed_128b_event(
            2,
            16777217,
            17104902,
            effective_uma,
            f"CONV_HV:{lane}:WSL={wsl_ram_gb}G:DT={dt['temporal_domains']['tau_spec_lookahead_axis']['lookahead_horizon_steps']}",
        )

        return {
            "vital_max_hp": VITAL_MAX_HP,
            "conv_interposer": {
                "input_shape": [1, OP_CHANNELS, 1, WINDOW_STEPS],
                "output_feature_shape": list(ov_feat.shape),
                "rel_error_vs_numpy": round(rel_err, 8),
                "raw_last_uma": round(raw_last_uma, 4),
                "conv_smoothed_uma": round(conv_smoothed_uma, 4),
                "effective_uma": effective_uma,
                "channel_scores": scores,
            },
            "hypervisor_step": {
                "mode": hypervisor_mode,
                "selected_compute_lane": lane,
                "wsl2_ram_gb": wsl_ram_gb,
                "wntc_64b_crc_valid": bool(wntc["self_binary_decoded"]["crc32_valid"]),
                "wntc_64b_hex_prefix": wntc["self_binary_64b_hex"][:32],
                "double_time_status": dt["status"],
                "double_time_lookahead_steps": dt["temporal_domains"]["tau_spec_lookahead_axis"]["lookahead_horizon_steps"],
                "step_proxy_receipt": proxy_receipt,
                "kpkt_128b_sha256": pkt128["frame_sha256_hex"],
            },
        }
