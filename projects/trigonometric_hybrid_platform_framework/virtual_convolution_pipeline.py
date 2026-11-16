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


def _conv_sequential(x: np.ndarray, w: np.ndarray, reverse: bool = False, fma: bool = False) -> np.ndarray:
    """FP32 conv with an explicit, fixed summation order over k = (c, ky, kx).

    fma=False: multiply rounds to f32, then add rounds to f32.
    fma=True : emulated fused multiply-add — the f32*f32 product is exact in f64 and the
               sum is rounded once to f32 (rare double-rounding differences possible).
    Used only as a summation-order fingerprint, not as the main verifier.
    """
    n, c, h, wd = x.shape
    xp = np.pad(x, ((0, 0), (0, 0), (1, 1), (1, 1))).astype(np.float32)
    f = w.shape[0]
    order = [(ci, i, j) for ci in range(c) for i in range(3) for j in range(3)]
    if reverse:
        order.reverse()
    acc = np.zeros((n, f, h, wd), dtype=np.float32)
    for ci, i, j in order:
        patch = xp[:, ci:ci + 1, i:i + h, j:j + wd]                      # [n,1,h,w]
        wk = w[None, :, ci, i, j, None, None]
        if fma:
            acc = (acc.astype(np.float64) + wk.astype(np.float64) * patch.astype(np.float64)).astype(np.float32)
        else:
            acc = (acc + (wk * patch).astype(np.float32)).astype(np.float32)
    return acc


def diagnose_bit_identity(channels: int = 16, filters: int = 32, height: int = 64, width: int = 64,
                          seed: int = 6) -> Dict[str, Any]:
    """PATTERN_CANDIDATE 'summation-order fingerprint'.

    Question: why is one device bit-identical to the NumPy reference while another is not?
    H1 same summation order (both sum k sequentially in natural order) -> device == seq_natural.
    H2 flawed comparison (aliasing / same buffer)                        -> device == every variant.
    H3 genuine independent agreement by chance                          -> no structural match.
    Observation: OpenVINO runtime model reports which kernel (primitiveType) each device used.
    """
    import openvino as ov

    rng = np.random.default_rng(seed)
    shape = (1, channels, height, width)
    x = rng.standard_normal(shape).astype(np.float32)
    w = _weights(channels, filters, seed + 1)

    variants = {
        "numpy_einsum_blas": conv2d_reference(x, w),
        "seq_natural_k_order": _conv_sequential(x, w, reverse=False),
        "seq_natural_k_order_fma": _conv_sequential(x, w, reverse=False, fma=True),
        "seq_reversed_k_order": _conv_sequential(x, w, reverse=True),
        "seq_reversed_k_order_fma": _conv_sequential(x, w, reverse=True, fma=True),
    }
    import openvino.opset13 as ops

    param = ops.parameter(list(shape), np.float32, name="input")
    model = ov.Model([ops.convolution(param, ops.constant(w), [1, 1], [1, 1], [1, 1], [1, 1])], [param], "single_conv")

    core = ov.Core()
    devices: Dict[str, Any] = {}
    for dev in core.available_devices:
        compiled = core.compile_model(model, dev, {"INFERENCE_PRECISION_HINT": "f32"})
        out = np.array(compiled.create_infer_request().infer({0: x})[compiled.output(0)], copy=True)
        kernels = []
        for op in compiled.get_runtime_model().get_ordered_ops():
            rt = op.get_rt_info()
            if "layerType" in rt and str(rt["layerType"].astype(str)).lower().startswith("conv"):
                kernels.append({k: str(rt[k].astype(str)) for k in ("layerType", "primitiveType", "runtimePrecision") if k in rt})
        devices[dev] = {
            "kernels": kernels,
            "bit_identical_to": [name for name, ref in variants.items() if np.array_equal(out, ref)],
            "max_abs_diff": {name: float(np.max(np.abs(out - ref))) for name, ref in variants.items()},
        }
    variant_cross = {
        "einsum_equals_seq_natural": bool(np.array_equal(variants["numpy_einsum_blas"], variants["seq_natural_k_order"])),
        "einsum_equals_seq_natural_fma": bool(np.array_equal(variants["numpy_einsum_blas"], variants["seq_natural_k_order_fma"])),
        "einsum_vs_seq_natural_fma_mismatch_fraction": float(np.mean(variants["numpy_einsum_blas"] != variants["seq_natural_k_order_fma"])),
        "seq_natural_equals_seq_reversed": bool(np.array_equal(variants["seq_natural_k_order"], variants["seq_reversed_k_order"])),
    }
    return {"input_shape": list(shape), "filters": filters, "variant_cross_check": variant_cross, "devices": devices,
            "status": "PATTERN_CANDIDATE"}


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
