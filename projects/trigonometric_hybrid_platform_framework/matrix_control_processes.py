"""Matrix control processes and a verified combinatorial operator mashup.

1. Freivalds verification (Freivalds, 1977): checks C == A @ B in O(n^2) per round with
   random vectors r: A(Br) == Cr. A wrong C passes one round with probability <= 1/2
   (exact arithmetic), so `rounds` rounds give <= 2^-rounds. Floating point uses a tolerance.
2. Combinatorial mashup: every type-correct word over {K, KT} up to a given length
   (K: conv X->Y, KT: transposed conv Y->X) is built and checked against algebraic laws:
     L1 linearity        W(a x1 + b x2) == a W x1 + b W x2
     L2 adjoint          <W x, y> == <x, W^T y>, where W^T = reversed word with K<->KT
     L3 Gram positivity  <x, W^T W x> >= 0
   A deliberately nonlinear operator (ReLU after K) is included as a negative control: it
   must fail L1, otherwise the checks have no discriminating power.
"""

from __future__ import annotations

from itertools import product
from typing import Any, Callable, Dict, List, Tuple

import numpy as np

from .virtual_convolution_contra import conv2d_transpose_reference
from .virtual_convolution_pipeline import _weights, conv2d_reference


def freivalds_verify(a: np.ndarray, b: np.ndarray, c: np.ndarray, rounds: int = 20, rtol: float = 1e-4,
                     seed: int = 6) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    scale = max(float(np.max(np.abs(c))), 1e-12)
    worst = 0.0
    for i in range(rounds):
        r = rng.integers(0, 2, size=(b.shape[1], 1)).astype(np.float64)
        lhs = a.astype(np.float64) @ (b.astype(np.float64) @ r)
        rhs = c.astype(np.float64) @ r
        gap = float(np.max(np.abs(lhs - rhs))) / scale
        worst = max(worst, gap)
        if gap > rtol:
            return {"accepted": False, "failed_round": i + 1, "rel_gap": gap, "rounds": rounds}
    return {"accepted": True, "rounds": rounds, "worst_rel_gap": worst,
            "false_accept_bound_exact_arithmetic": 2.0 ** -rounds}


def _words(max_len: int) -> List[Tuple[str, ...]]:
    """Type-correct words: K maps X->Y, KT maps Y->X; apply right-to-left like matrices."""
    out = []
    for n in range(1, max_len + 1):
        for w in product(("K", "KT"), repeat=n):
            ok = all(w[i] != w[i + 1] for i in range(n - 1))  # alternating = type-correct
            if ok:
                out.append(w)
    return out


def run_operator_mashup(channels: int = 2, filters: int = 4, size: int = 10, max_len: int = 4, seed: int = 6,
                        tol: float = 1e-5) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    w = _weights(channels, filters, seed + 1)
    x_shape, y_shape = (1, channels, size, size), (1, filters, size, size)
    ops: Dict[str, Callable[[np.ndarray], np.ndarray]] = {
        "K": lambda v: conv2d_reference(v.astype(np.float32), w).astype(np.float64),
        "KT": lambda v: conv2d_transpose_reference(v.astype(np.float32), w).astype(np.float64),
    }
    in_space = {"K": x_shape, "KT": y_shape}
    out_space = {"K": y_shape, "KT": x_shape}

    def apply(word: Tuple[str, ...], v: np.ndarray, nonlinear: bool = False) -> np.ndarray:
        for op in reversed(word):  # rightmost operator acts first
            v = ops[op](v)
            if nonlinear and op == "K":
                v = np.maximum(v, 0.0)
        return v

    def transpose(word: Tuple[str, ...]) -> Tuple[str, ...]:
        return tuple("KT" if o == "K" else "K" for o in reversed(word))

    rows = []
    for word in _words(max_len):
        xs, ys = in_space[word[-1]], out_space[word[0]]
        x1, x2, y = rng.standard_normal(xs), rng.standard_normal(xs), rng.standard_normal(ys)
        a, b = 1.7, -0.6
        lhs = apply(word, a * x1 + b * x2)
        rhs = a * apply(word, x1) + b * apply(word, x2)
        lin_gap = float(np.linalg.norm(lhs - rhs) / max(np.linalg.norm(rhs), 1e-12))
        p, q = float(np.vdot(apply(word, x1), y)), float(np.vdot(x1, apply(transpose(word), y)))
        adj_gap = abs(p - q) / max(abs(p), 1e-12)
        gram = float(np.vdot(x1, apply(transpose(word) + word, x1)))
        rows.append({"word": "·".join(word), "maps": f"{'X' if xs == x_shape else 'Y'}->{'Y' if ys == y_shape else 'X'}",
                     "L1_linearity_gap": lin_gap, "L2_adjoint_gap": adj_gap, "L3_gram_value": gram,
                     "passes": lin_gap < tol and adj_gap < tol and gram >= -tol * abs(gram)})

    # Negative control: ReLU after every K -> must break linearity.
    x1, x2 = rng.standard_normal(x_shape), rng.standard_normal(x_shape)
    nl = apply(("K",), 1.7 * x1 - 0.6 * x2, nonlinear=True)
    nr = 1.7 * apply(("K",), x1, nonlinear=True) - 0.6 * apply(("K",), x2, nonlinear=True)
    control_gap = float(np.linalg.norm(nl - nr) / max(np.linalg.norm(nr), 1e-12))

    all_pass = all(r["passes"] for r in rows)
    control_detected = control_gap > 1e-3
    return {
        "words_checked": len(rows),
        "all_laws_hold": all_pass,
        "negative_control_relu_linearity_gap": control_gap,
        "negative_control_detected": control_detected,
        "verdict": "VERIFIED_WITH_SCOPE" if all_pass and control_detected else ("FALSIFIED" if not all_pass else "UNKNOWN"),
        "rows": rows,
    }
