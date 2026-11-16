"""Virtual convolution pipeline: OpenVINO in-graph Conv2D stack vs. NumPy reference.

"Virtual" = the network is built in-memory from OpenVINO opset ops (no model file, no
download) and executed on every available device (CPU, GPU). Each device result is
checked against an independent NumPy reference before any timing is reported
(PROPOSER = OpenVINO device, VERIFIER = NumPy reference).

Pipeline: input [1, C, H, W] -> Conv3x3 (C -> F) -> ReLU -> Conv3x3 (F -> F) -> ReLU
Padding "same", stride 1, deterministic weights (seed 6).
"""

from __future__ import annotations

import statistics
import time
from typing import Any, Dict, List

import numpy as np

from krystal_kernel.constants import VITAL_MAX_HP, ContractViolation

MAX_ELEMENTS = 4 * 1024 * 1024  # input size bound (elements) to keep runs short and memory bounded


def _weights(c_in: int, c_out: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (rng.standard_normal((c_out, c_in, 3, 3)) / np.sqrt(c_in * 9)).astype(np.float32)


def conv2d_reference(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    """NCHW conv, 3x3, stride 1, zero 'same' padding, via im2col + matmul (independent of OpenVINO)."""
    n, c, h, wd = x.shape
    xp = np.pad(x, ((0, 0), (0, 0), (1, 1), (1, 1)))
    cols = np.empty((n, c, 3, 3, h, wd), dtype=np.float32)
    for i in range(3):
        for j in range(3):
            cols[:, :, i, j] = xp[:, :, i:i + h, j:j + wd]
    cols = cols.reshape(n, c * 9, h * wd)
    out = np.einsum("fk,nkp->nfp", w.reshape(w.shape[0], -1), cols, optimize=True)
    return out.reshape(n, w.shape[0], h, wd)


def reference_pipeline(x: np.ndarray, w1: np.ndarray, w2: np.ndarray) -> np.ndarray:
    return np.maximum(conv2d_reference(np.maximum(conv2d_reference(x, w1), 0.0), w2), 0.0)


def build_openvino_model(shape: tuple, w1: np.ndarray, w2: np.ndarray):
    import openvino as ov
    import openvino.opset13 as ops

    param = ops.parameter(list(shape), np.float32, name="input")
    c1 = ops.convolution(param, ops.constant(w1), [1, 1], [1, 1], [1, 1], [1, 1])
    r1 = ops.relu(c1)
    c2 = ops.convolution(r1, ops.constant(w2), [1, 1], [1, 1], [1, 1], [1, 1])
    r2 = ops.relu(c2)
    return ov.Model([r2], [param], "krystal_virtual_conv_pipeline")


def run_virtual_convolution_pipeline(
    channels: int = 16,
    filters: int = 32,
    height: int = 128,
    width: int = 128,
    iterations: int = 30,
    seed: int = 6,
    devices: List[str] | None = None,
    vital_hp: int = VITAL_MAX_HP,
) -> Dict[str, Any]:
    if vital_hp != VITAL_MAX_HP:
        raise ContractViolation(f"vital_hp={vital_hp} != {VITAL_MAX_HP}", violation_code="ERR_CONV_VITAL_HP")
    if min(channels, filters, height, width, iterations) < 1:
        raise ContractViolation("All dimensions and iterations must be >= 1", violation_code="ERR_CONV_SHAPE")
    if channels * height * width > MAX_ELEMENTS:
        raise ContractViolation("Input exceeds MAX_ELEMENTS bound", violation_code="ERR_CONV_TOO_LARGE")

    import openvino as ov

    shape = (1, channels, height, width)
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(shape).astype(np.float32)
    w1, w2 = _weights(channels, filters, seed + 1), _weights(filters, filters, seed + 2)

    t0 = time.perf_counter()
    ref = reference_pipeline(x, w1, w2)
    ref_ms = (time.perf_counter() - t0) * 1000.0

    core = ov.Core()
    model = build_openvino_model(shape, w1, w2)
    targets = devices or core.available_devices
    macs = 9 * height * width * (channels * filters + filters * filters)

    results: Dict[str, Any] = {}
    for dev in targets:
        entry: Dict[str, Any] = {"device_name": core.get_property(dev, "FULL_DEVICE_NAME")}
        try:
            t_c = time.perf_counter()
            # Force FP32 so the correctness check is meaningful on GPU (default hint there is FP16).
            compiled = core.compile_model(model, dev, {"INFERENCE_PRECISION_HINT": "f32"})
            entry["compile_ms"] = round((time.perf_counter() - t_c) * 1000.0, 2)
            req = compiled.create_infer_request()
            out = req.infer({0: x})[compiled.output(0)]
            if out.shape != ref.shape:
                raise ContractViolation(f"Output shape {out.shape} != {ref.shape}", violation_code="ERR_CONV_SHAPE")
            max_err = float(np.max(np.abs(out - ref)))
            rel_err = max_err / max(float(np.max(np.abs(ref))), 1e-12)
            entry["max_abs_error"] = round(max_err, 8)
            entry["max_rel_error"] = round(rel_err, 8)
            entry["matches_reference"] = rel_err < 1e-3
            if not entry["matches_reference"]:
                entry["verdict"] = "REJECTED_MISMATCH_NO_TIMING"
                results[dev] = entry
                continue
            req.infer({0: x})  # warm-up
            samples = []
            for _ in range(iterations):
                t = time.perf_counter()
                req.infer({0: x})
                samples.append((time.perf_counter() - t) * 1000.0)
            samples.sort()
            med = statistics.median(samples)
            entry.update({
                "n": iterations,
                "median_ms": round(med, 3),
                "p95_ms": round(samples[min(len(samples) - 1, int(0.95 * len(samples)))], 3),
                "stdev_ms": round(statistics.stdev(samples), 3) if len(samples) > 1 else 0.0,
                "gmac_per_s_at_median": round(macs / (med / 1000.0) / 1e9, 2),
                "verdict": "MEASURED_WITH_SCOPE",
            })
        except Exception as exc:  # device unavailable / plugin error -> recorded, not hidden
            entry["verdict"] = f"DEVICE_ERROR: {type(exc).__name__}: {exc}"
        results[dev] = entry

    return {
        "vital_max_hp": VITAL_MAX_HP,
        "pipeline": "Conv3x3(C->F) -> ReLU -> Conv3x3(F->F) -> ReLU, same padding, stride 1, FP32",
        "input_shape": list(shape),
        "filters": filters,
        "macs_per_inference": macs,
        "numpy_reference_ms_single_run": round(ref_ms, 2),
        "devices": results,
        "evidence_level": "LOCAL_MEASURED (this host, Python + OpenVINO, wall-clock per infer call)",
    }


if __name__ == "__main__":
    import json

    print(json.dumps(run_virtual_convolution_pipeline(), indent=2))
