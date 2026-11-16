"""Contra modules for the virtual convolution pipeline: run the meaning backwards.

Forward:  y = K x            (Conv3x3, same padding, stride 1, C -> F channels)
Contra 1: K^T y              transposed convolution = exact adjoint of K
                             (NumPy flipped-kernel reference + OpenVINO ConvolutionBackpropData)
Contra 2: x ~ argmin ||Kx - y||   inverse via conjugate gradient on K^T K x = K^T y
Contra 3: x from ReLU(Kx)    only positive outputs carry equations; ReLU discards the rest,
                             so recovery is partial — the measured error quantifies that loss.

Verifier for every contra step: the dot-product (adjoint) test <Kx, y> == <x, K^T y>
and reconstruction error against the known ground-truth x.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

from krystal_kernel.constants import VITAL_MAX_HP, ContractViolation

from .virtual_convolution_pipeline import _weights, conv2d_reference


def conv2d_transpose_reference(y: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Adjoint of conv2d_reference for 3x3 / pad 1 / stride 1: swap in/out channels, flip spatially."""
    w_t = np.ascontiguousarray(np.flip(w, axis=(2, 3)).transpose(1, 0, 2, 3))
    return conv2d_reference(y, w_t)


def _cg_normal(apply_k, apply_kt, y: np.ndarray, x_shape: tuple, iters: int, tol: float) -> Dict[str, Any]:
    """Conjugate gradient on K^T K x = K^T y (CGLS form)."""
    x = np.zeros(x_shape, dtype=np.float64)
    r = y.astype(np.float64) - apply_k(x)
    s = apply_kt(r)
    p = s.copy()
    gamma = float(np.vdot(s, s))
    gamma0 = gamma
    it = 0
    for it in range(1, iters + 1):
        q = apply_k(p)
        alpha = gamma / max(float(np.vdot(q, q)), 1e-300)
        x += alpha * p
        r -= alpha * q
        s = apply_kt(r)
        gamma_new = float(np.vdot(s, s))
        if gamma_new <= (tol ** 2) * gamma0:
            gamma = gamma_new
            break
        p = s + (gamma_new / gamma) * p
        gamma = gamma_new
    return {"x": x, "iterations": it, "normal_residual_ratio": (gamma / gamma0) ** 0.5 if gamma0 else 0.0}


def openvino_transpose_conv(y: np.ndarray, w: np.ndarray, device: str = "CPU") -> np.ndarray:
    """OpenVINO ConvolutionBackpropData with filters [F, C, 3, 3] = the forward weights unchanged."""
    import openvino as ov
    import openvino.opset13 as ops

    param = ops.parameter(list(y.shape), np.float32, name="y")
    # Keywords are required: the 4th positional argument of convolution_backprop_data is output_shape.
    out = ops.convolution_backprop_data(
        param, ops.constant(w), strides=[1, 1], pads_begin=[1, 1], pads_end=[1, 1], dilations=[1, 1]
    )
    model = ov.Model([out], [param], "krystal_contra_transpose_conv")
    compiled = ov.Core().compile_model(model, device, {"INFERENCE_PRECISION_HINT": "f32"})
    return compiled.create_infer_request().infer({0: y.astype(np.float32)})[compiled.output(0)]


def run_contra_modules(
    channels: int = 4,
    filters: int = 8,
    height: int = 32,
    width: int = 32,
    cg_iterations: int = 300,
    seed: int = 6,
    use_openvino: bool = True,
    vital_hp: int = VITAL_MAX_HP,
) -> Dict[str, Any]:
    if vital_hp != VITAL_MAX_HP:
        raise ContractViolation(f"vital_hp={vital_hp} != {VITAL_MAX_HP}", violation_code="ERR_CONTRA_VITAL_HP")
    if filters < channels:
        raise ContractViolation(
            "filters < channels makes K non-injective; exact inversion is impossible by construction",
            violation_code="ERR_CONTRA_UNDERDETERMINED",
        )

    rng = np.random.default_rng(seed)
    x_true = rng.standard_normal((1, channels, height, width)).astype(np.float32)
    w = _weights(channels, filters, seed + 1)
    k = lambda v: conv2d_reference(v.astype(np.float32), w).astype(np.float64)
    kt = lambda v: conv2d_transpose_reference(v.astype(np.float32), w).astype(np.float64)

    y = k(x_true)

    # Contra 1 verifier: dot-product adjoint test with independent random probes.
    xa = rng.standard_normal(x_true.shape)
    ya = rng.standard_normal(y.shape)
    lhs, rhs = float(np.vdot(k(xa), ya)), float(np.vdot(xa, kt(ya)))
    adjoint_rel_gap = abs(lhs - rhs) / max(abs(lhs), 1e-12)

    ov_info: Dict[str, Any] = {"used": False}
    if use_openvino:
        try:
            ov_out = openvino_transpose_conv(ya.astype(np.float32), w)
            ref_out = kt(ya)
            if ov_out.shape != ref_out.shape:
                raise ContractViolation(
                    f"OpenVINO transpose-conv shape {ov_out.shape} != {ref_out.shape}",
                    violation_code="ERR_CONTRA_SHAPE",
                )
            ov_info = {
                "used": True,
                "max_rel_error_vs_numpy_adjoint": float(np.max(np.abs(ov_out - ref_out)) / np.max(np.abs(ref_out))),
            }
        except Exception as exc:
            ov_info = {"used": False, "error": f"{type(exc).__name__}: {exc}"}

    # Contra 2: invert the linear layer.
    lin = _cg_normal(k, kt, y, x_true.shape, cg_iterations, 1e-10)
    lin_err = float(np.linalg.norm(lin["x"] - x_true) / np.linalg.norm(x_true))

    # Contra 3: recover from ReLU output using only the active (positive) equations.
    y_relu = np.maximum(y, 0.0)
    mask = (y > 0).astype(np.float64)
    km = lambda v: mask * k(v)
    ktm = lambda v: kt(mask * v)
    rel = _cg_normal(km, ktm, y_relu * mask, x_true.shape, cg_iterations, 1e-10)
    relu_err = float(np.linalg.norm(rel["x"] - x_true) / np.linalg.norm(x_true))
    active = float(mask.mean())

    return {
        "vital_max_hp": VITAL_MAX_HP,
        "shape_x": list(x_true.shape),
        "shape_y": list(y.shape),
        "unknowns": int(x_true.size),
        "equations_linear": int(y.size),
        "equations_after_relu": int(mask.sum()),
        "relu_active_fraction": round(active, 4),
        "contra1_adjoint_dot_test_rel_gap": adjoint_rel_gap,
        "contra1_openvino_transpose_conv": ov_info,
        "contra2_linear_inverse": {
            "iterations": lin["iterations"],
            "normal_residual_ratio": lin["normal_residual_ratio"],
            "reconstruction_rel_error": lin_err,
        },
        "contra3_relu_partial_inverse": {
            "iterations": rel["iterations"],
            "normal_residual_ratio": rel["normal_residual_ratio"],
            "reconstruction_rel_error": relu_err,
        },
        "evidence_level": "LOCAL_MEASURED_NUMERICAL (synthetic x, seed 6)",
    }


if __name__ == "__main__":
    import json

    print(json.dumps(run_contra_modules(), indent=2, default=float))
