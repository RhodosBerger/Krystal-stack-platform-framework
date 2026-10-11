"""Built-in node functions for diagnostic telemetry rails.

All nodes are stdlib-only. Real telemetry sources read clocks that need no elevated
privileges (``time.process_time``, ``time.perf_counter``, ...). There are deliberately
**no** MSR/RAPL or psutil sources here (see ``krystal_theory/05_telemetria_ako_diagnostika.md``).

Theory references (``theory_ref``) point to ``krystal_theory`` statements that a node
implements or tests, e.g. Definition 3.2 (Schmitt trigger) or Theorem 3.1 (2p(1-p)).
"""

from __future__ import annotations

import math
import random
import statistics
import time
from typing import Any, Dict, List

from .nodes import DEFAULT_REGISTRY, Param, Port

MAX_SAMPLES = 100_000
MAX_SOURCE_SECONDS = 5.0

VERDICT_NOT_REJECTED = "NOT_REJECTED"
VERDICT_REJECTED = "REJECTED"
VERDICT_INCONCLUSIVE = "INCONCLUSIVE"

node = DEFAULT_REGISTRY.node
NAN = float("nan")


def _finite(x: float) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def _claim_param(default: str = "") -> Param:
    return Param("claim", "string", default, doc="Human-readable statement under test.")


# ── Sources ───────────────────────────────────────────────────────────────────

@node("source.synthetic_series", category="source", provenance="synthetic",
      outputs=[Port("series", "series", doc="Generated samples")],
      params=[
          Param("n", "integer", 2000, 2, MAX_SAMPLES, doc="Number of samples"),
          Param("mean", "number", 85.0, -1e6, 1e6, doc="Mean level (e.g. theta in % Tj,max)"),
          Param("noise", "string", "uniform", enum=("uniform", "gaussian", "ar1"),
                doc="uniform: bounded |n|<=amplitude (A3.2); gaussian: iid N(0,amplitude^2) (A3.1); "
                    "ar1: n_k = phi*n_(k-1) + e_k, e_k ~ N(0,amplitude^2) (violates A3.1)"),
          Param("amplitude", "number", 3.0, 0.0, 1e6, doc="Half-width (uniform) or innovation std"),
          Param("phi", "number", 0.0, -0.99, 0.99, doc="AR(1) coefficient (ar1 only)"),
          Param("drift", "number", 0.0, -1e3, 1e3, doc="Added per sample (mean ramps)"),
          Param("seed", "integer", 1, 0, 2**31 - 1, doc="PRNG seed (reproducible)"),
      ],
      theory_ref="A3.1, A3.2")
def synthetic_series(inputs: Dict[str, Any], p: Dict[str, Any]) -> Dict[str, Any]:
    """Seeded synthetic signal: mean + noise (+ drift). Ground truth for testing diagnostics."""
    rng = random.Random(int(p["seed"]))
    n, mean, amp, kind = int(p["n"]), float(p["mean"]), float(p["amplitude"]), p["noise"]
    phi, drift = float(p["phi"]), float(p["drift"])
    out: List[float] = []
    if kind == "ar1":
        # Start in the stationary distribution so there is no warm-up transient.
        state = rng.gauss(0.0, amp / math.sqrt(1.0 - phi * phi)) if amp > 0 else 0.0
    for k in range(n):
        if kind == "uniform":
            noise = rng.uniform(-amp, amp)
        elif kind == "gaussian":
            noise = rng.gauss(0.0, amp) if amp > 0 else 0.0
        else:
            if k > 0:
                state = phi * state + (rng.gauss(0.0, amp) if amp > 0 else 0.0)
            noise = state
        out.append(mean + drift * k + noise)
    return {"series": out}


@node("source.constant", category="source", provenance="derived",
      outputs=[Port("value", "scalar")],
      params=[Param("value", "number", 0.0, -1e12, 1e12)])
def constant(inputs: Dict[str, Any], p: Dict[str, Any]) -> Dict[str, Any]:
    """A declared reference value (prediction, threshold, known ground truth)."""
    return {"value": float(p["value"])}


_CLOCKS = ("process_time", "perf_counter", "monotonic", "time")


@node("source.clock_quantum", category="source", provenance="measured",
      outputs=[
          Port("quantum_ms", "scalar", doc="Smallest observed positive clock increment"),
          Port("reported_resolution_ms", "scalar", doc="time.get_clock_info(...).resolution"),
          Port("increments_ms", "series", doc="Observed positive increments"),
      ],
      params=[
          Param("clock", "string", "process_time", enum=_CLOCKS),
          Param("samples", "integer", 16, 1, 64, doc="Increments to observe"),
      ],
      wall_budget=lambda p: 0.5,
      theory_ref="§5.3 kvantovanie")
def clock_quantum(inputs: Dict[str, Any], p: Dict[str, Any]) -> Dict[str, Any]:
    """Measure the effective tick of a clock by busy-waiting until it changes (<=0.5 s).

    On Windows ``process_time`` often *reports* 100 ns resolution while it actually
    advances in scheduler ticks (~15.6 ms). The measured quantum is what matters for
    diagnostics; the reported one is returned for comparison.
    """
    clock = getattr(time, p["clock"])
    reported = time.get_clock_info(p["clock"]).resolution * 1e3
    deadline = time.perf_counter() + 0.5
    increments: List[float] = []
    last = clock()
    while len(increments) < int(p["samples"]) and time.perf_counter() < deadline:
        now = clock()
        if now != last:
            if now > last:
                increments.append((now - last) * 1e3)
            last = now
    quantum = min(increments) if increments else NAN
    return {"quantum_ms": quantum, "reported_resolution_ms": reported, "increments_ms": increments}


def _cpu_cross_check(p: Dict[str, Any]) -> List[str]:
    if p["duration_s"] * 1000.0 / p["interval_ms"] < 2:
        return ["duration_s must cover at least 2 intervals"]
    return []


@node("source.process_cpu_sampler", category="source", provenance="measured",
      outputs=[
          Port("cpu_pct", "series", doc="Process CPU time / wall time per interval, %"),
          Port("interval_ms", "series", doc="Actual wall-clock length of each interval"),
          Port("quantum_pct", "scalar", doc="Smallest positive CPU step expressed in % of the median interval"),
      ],
      params=[
          Param("duration_s", "number", 1.0, 0.05, MAX_SOURCE_SECONDS),
          Param("interval_ms", "number", 20.0, 1.0, 1000.0),
          Param("duty", "number", 0.5, 0.0, 1.0, doc="Fraction of each interval spent busy-spinning"),
      ],
      cross_check=_cpu_cross_check,
      wall_budget=lambda p: float(p["duration_s"]),
      theory_ref="§5.3 kvantovanie, §5.5 záťaž pozorovateľa")
def process_cpu_sampler(inputs: Dict[str, Any], p: Dict[str, Any]) -> Dict[str, Any]:
    """Sample this Python process's CPU utilisation with a known synthetic duty cycle.

    The ground truth is ``duty`` (we spin for that fraction of each interval), so the
    node measures the *instrument*: how far ``time.process_time`` deltas deviate from
    a known load at the chosen interval. No elevated privileges are needed.
    """
    interval = float(p["interval_ms"]) / 1e3
    duty = float(p["duty"])
    end = time.perf_counter() + float(p["duration_s"])
    cpu_pct: List[float] = []
    lengths: List[float] = []
    steps: List[float] = []
    w0, c0 = time.perf_counter(), time.process_time()
    while w0 < end and len(cpu_pct) < MAX_SAMPLES:
        spin_until = w0 + duty * interval
        while time.perf_counter() < spin_until:
            pass
        remaining = w0 + interval - time.perf_counter()
        if remaining > 0:
            time.sleep(remaining)
        w1, c1 = time.perf_counter(), time.process_time()
        dw, dc = w1 - w0, c1 - c0
        if dw > 0:
            cpu_pct.append(100.0 * dc / dw)
            lengths.append(dw * 1e3)
            if dc > 0:
                steps.append(dc)
        w0, c0 = w1, c1
    median_len = statistics.median(lengths) if lengths else NAN
    quantum_pct = (100.0 * min(steps) * 1e3 / median_len) if steps and median_len > 0 else NAN
    return {"cpu_pct": cpu_pct, "interval_ms": lengths, "quantum_pct": quantum_pct}


def _jitter_cross_check(p: Dict[str, Any]) -> List[str]:
    if p["n"] * p["interval_ms"] > MAX_SOURCE_SECONDS * 1000.0:
        return [f"n * interval_ms must be <= {int(MAX_SOURCE_SECONDS * 1000)} ms"]
    return []


@node("source.sleep_jitter", category="source", provenance="measured",
      outputs=[Port("jitter_ms", "series", doc="Actual minus requested sleep length")],
      params=[
          Param("interval_ms", "number", 10.0, 1.0, 1000.0),
          Param("n", "integer", 100, 2, 1000),
      ],
      cross_check=_jitter_cross_check,
      wall_budget=lambda p: float(p["n"]) * float(p["interval_ms"]) / 1e3,
      theory_ref="§5.2 vzorkovanie")
def sleep_jitter(inputs: Dict[str, Any], p: Dict[str, Any]) -> Dict[str, Any]:
    """Measure how late ``time.sleep`` wakes up: the real sampling grid is not uniform."""
    req = float(p["interval_ms"]) / 1e3
    out: List[float] = []
    for _ in range(int(p["n"])):
        t0 = time.perf_counter()
        time.sleep(req)
        out.append((time.perf_counter() - t0 - req) * 1e3)
    return {"jitter_ms": out}


# ── Transforms ───────────────────────────────────────────────────────────────

@node("transform.moving_average", category="transform",
      inputs=[Port("series", "series")], outputs=[Port("series", "series")],
      params=[Param("window", "integer", 5, 1, 10_000)])
def moving_average(inputs: Dict[str, Any], p: Dict[str, Any]) -> Dict[str, Any]:
    """Trailing moving average (same length; the first values average the available prefix)."""
    xs, w = inputs["series"], int(p["window"])
    out: List[float] = []
    acc = 0.0
    for i, x in enumerate(xs):
        acc += x
        if i >= w:
            acc -= xs[i - w]
        out.append(acc / min(i + 1, w))
    return {"series": out}


@node("transform.quantize", category="transform",
      inputs=[Port("series", "series")],
      outputs=[Port("series", "series"), Port("quantum", "scalar")],
      params=[Param("q", "number", 0.5, 0.0, 1e6, doc="Quantisation step; 0 = pass-through")],
      theory_ref="§5.3, Lema 5.3")
def quantize(inputs: Dict[str, Any], p: Dict[str, Any]) -> Dict[str, Any]:
    """Round to the nearest multiple of q (|error| <= q/2), as a sensor ADC would."""
    q = float(p["q"])
    xs = inputs["series"]
    out = list(xs) if q == 0 else [q * round(x / q) for x in xs]
    return {"series": out, "quantum": q}


@node("transform.threshold_mode", category="transform",
      inputs=[Port("series", "series")], outputs=[Port("modes", "modes")],
      params=[Param("threshold", "number", 85.0, -1e9, 1e9)],
      theory_ref="Definícia 3.1")
def threshold_mode(inputs: Dict[str, Any], p: Dict[str, Any]) -> Dict[str, Any]:
    """Memoryless regime q_k = 1[x_k >= threshold] (what classify() does today)."""
    t = float(p["threshold"])
    return {"modes": [1 if x >= t else 0 for x in inputs["series"]]}


@node("transform.hysteresis_mode", category="transform",
      inputs=[Port("series", "series"),
              Port("h", "scalar", required=False, doc="Overrides param h when linked")],
      outputs=[Port("modes", "modes"), Port("h_used", "scalar")],
      params=[Param("threshold", "number", 85.0, -1e9, 1e9),
              Param("h", "number", 0.0, 0.0, 1e9),
              Param("initial", "integer", 0, enum=(0, 1))],
      theory_ref="Definícia 3.2, Veta 3.2")
def hysteresis_mode(inputs: Dict[str, Any], p: Dict[str, Any]) -> Dict[str, Any]:
    """Schmitt trigger: enter at x >= t, leave at x < t - h, otherwise keep the mode."""
    t = float(p["threshold"])
    h = inputs.get("h")
    h = float(p["h"]) if h is None else float(h)
    if not math.isfinite(h) or h < 0:
        raise ValueError(f"hysteresis width must be finite and >= 0, got {h!r}")
    q = int(p["initial"])
    out: List[int] = []
    for x in inputs["series"]:
        if x >= t:
            q = 1
        elif x < t - h:
            q = 0
        out.append(q)
    return {"modes": out, "h_used": h}


# ── Measures ─────────────────────────────────────────────────────────────────

@node("measure.exceed_fraction", category="measure",
      inputs=[Port("modes", "modes")], outputs=[Port("p", "scalar")],
      theory_ref="H-R1 krok 3")
def exceed_fraction(inputs: Dict[str, Any], p: Dict[str, Any]) -> Dict[str, Any]:
    """p_hat = share of samples in mode 1."""
    m = inputs["modes"]
    return {"p": (sum(m) / len(m)) if m else NAN}


@node("measure.switch_rate", category="measure",
      inputs=[Port("modes", "modes")],
      outputs=[Port("rate", "scalar"), Port("count", "scalar"), Port("pairs", "scalar")],
      theory_ref="H-R1 krok 3, Lema 5.1")
def switch_rate(inputs: Dict[str, Any], p: Dict[str, Any]) -> Dict[str, Any]:
    """s_hat = share of adjacent pairs whose mode differs; also the raw count."""
    m = inputs["modes"]
    pairs = max(len(m) - 1, 0)
    count = sum(1 for a, b in zip(m, m[1:]) if a != b)
    return {"rate": (count / pairs) if pairs else NAN, "count": float(count), "pairs": float(pairs)}


@node("measure.expected_switch_rate", category="measure",
      inputs=[Port("p", "scalar")], outputs=[Port("rate", "scalar")],
      theory_ref="Veta 3.1")
def expected_switch_rate(inputs: Dict[str, Any], p: Dict[str, Any]) -> Dict[str, Any]:
    """Prediction of Theorem 3.1 under iid samples: 2p(1-p)."""
    pv = inputs["p"]
    return {"rate": 2.0 * pv * (1.0 - pv) if _finite(pv) else NAN}


@node("measure.iid_switch_sigma", category="measure",
      inputs=[Port("p", "scalar"), Port("pairs", "scalar")], outputs=[Port("sigma", "scalar")],
      theory_ref="Lema 5.2")
def iid_switch_sigma(inputs: Dict[str, Any], p: Dict[str, Any]) -> Dict[str, Any]:
    """Std. error of s_hat under iid modes: sqrt(4a(1-3a)/m), a = p(1-p), m = pairs.

    Switch indicators of adjacent pairs are 1-dependent, so the naive binomial
    variance 2a(1-2a)/m is wrong; Lemma 5.2 derives the covariance term.
    """
    pv, m = inputs["p"], inputs["pairs"]
    if not (_finite(pv) and _finite(m)) or m < 1:
        return {"sigma": NAN}
    a = pv * (1.0 - pv)
    return {"sigma": math.sqrt(max(4.0 * a * (1.0 - 3.0 * a), 0.0) / m)}


@node("measure.lag1_autocorr", category="measure",
      inputs=[Port("series", "series")], outputs=[Port("rho", "scalar")],
      theory_ref="H-R1 krok 4")
def lag1_autocorr(inputs: Dict[str, Any], p: Dict[str, Any]) -> Dict[str, Any]:
    """Sample lag-1 autocorrelation (nan when the series is constant or too short)."""
    xs = inputs["series"]
    n = len(xs)
    if n < 3:
        return {"rho": NAN}
    mu = sum(xs) / n
    den = sum((x - mu) ** 2 for x in xs)
    if den == 0:
        return {"rho": NAN}
    num = sum((xs[i] - mu) * (xs[i + 1] - mu) for i in range(n - 1))
    return {"rho": num / den}


@node("measure.noise_range", category="measure",
      inputs=[Port("series", "series")],
      outputs=[Port("range", "scalar"), Port("delta_hat", "scalar")],
      theory_ref="A3.2, Lema 5.3")
def noise_range(inputs: Dict[str, Any], p: Dict[str, Any]) -> Dict[str, Any]:
    """Empirical range R and delta_hat = R/2. A *lower* estimate of the true bound delta."""
    xs = inputs["series"]
    if not xs:
        return {"range": NAN, "delta_hat": NAN}
    r = max(xs) - min(xs)
    return {"range": r, "delta_hat": r / 2.0}


@node("measure.stats", category="measure",
      inputs=[Port("series", "series")],
      outputs=[Port("stats", "record"), Port("mean", "scalar")])
def stats(inputs: Dict[str, Any], p: Dict[str, Any]) -> Dict[str, Any]:
    """n, mean, sample std, min, max."""
    xs = inputs["series"]
    if not xs:
        return {"stats": {"n": 0}, "mean": NAN}
    mean = statistics.fmean(xs)
    std = statistics.stdev(xs) if len(xs) > 1 else NAN
    rec = {"n": len(xs), "mean": mean, "std": std, "min": min(xs), "max": max(xs)}
    return {"stats": rec, "mean": mean}


@node("measure.hysteresis_width", category="measure",
      inputs=[Port("range", "scalar")], outputs=[Port("h", "scalar")],
      params=[Param("margin", "number", 0.0, 0.0, 1e9, doc="Safety margin added to the range")],
      theory_ref="Veta 3.2, Lema 5.3")
def hysteresis_width(inputs: Dict[str, Any], p: Dict[str, Any]) -> Dict[str, Any]:
    """h = R_hat + margin. Guarantees Theorem 3.2 only if R_hat + margin >= 2*delta'."""
    r = inputs["range"]
    return {"h": (r + float(p["margin"])) if _finite(r) else NAN}


# ── Diagnose ─────────────────────────────────────────────────────────────────

def _verdict(status: str, claim: str, **detail: Any) -> Dict[str, Any]:
    return {"kind": "verdict", "status": status, "claim": claim, **detail}


@node("diagnose.compare_prediction", category="diagnose",
      inputs=[Port("observed", "scalar"), Port("predicted", "scalar"),
              Port("sigma", "scalar", required=False, doc="Std. error; tolerance = z * sigma")],
      outputs=[Port("verdict", "record")],
      params=[Param("z", "number", 1.96, 0.0, 10.0),
              Param("abs_tol", "number", 0.0, 0.0, 1e9, doc="Used when sigma is not linked"),
              _claim_param()],
      theory_ref="§5.4")
def compare_prediction(inputs: Dict[str, Any], p: Dict[str, Any]) -> Dict[str, Any]:
    """REJECTED when |observed - predicted| > tolerance; INCONCLUSIVE on nan inputs."""
    obs, pred, sigma = inputs["observed"], inputs["predicted"], inputs.get("sigma")
    claim = p["claim"]
    use_sigma = sigma is not None
    tol = float(p["z"]) * sigma if use_sigma and _finite(sigma) else float(p["abs_tol"])
    if not (_finite(obs) and _finite(pred)) or (use_sigma and not _finite(sigma)):
        return {"verdict": _verdict(VERDICT_INCONCLUSIVE, claim, observed=obs, predicted=pred,
                                    reason="non-finite input")}
    dev = abs(obs - pred)
    detail: Dict[str, Any] = {"observed": obs, "predicted": pred, "deviation": dev, "tolerance": tol}
    if use_sigma:
        detail["z_score"] = (dev / sigma) if sigma > 0 else (0.0 if dev == 0 else math.inf)
    status = VERDICT_REJECTED if dev > tol else VERDICT_NOT_REJECTED
    return {"verdict": _verdict(status, claim, **detail)}


@node("diagnose.bound_check", category="diagnose",
      inputs=[Port("value", "scalar")], outputs=[Port("verdict", "record")],
      params=[Param("upper", "number", 1.0, -1e300, 1e300),
              Param("absolute", "boolean", False, doc="Compare |value| instead of value"),
              _claim_param()],
      theory_ref="§5.4")
def bound_check(inputs: Dict[str, Any], p: Dict[str, Any]) -> Dict[str, Any]:
    """NOT_REJECTED when value (or |value|) <= upper."""
    v = inputs["value"]
    claim = p["claim"]
    if not _finite(v):
        return {"verdict": _verdict(VERDICT_INCONCLUSIVE, claim, observed=v, reason="non-finite input")}
    x = abs(v) if p["absolute"] else v
    status = VERDICT_NOT_REJECTED if x <= float(p["upper"]) else VERDICT_REJECTED
    return {"verdict": _verdict(status, claim, observed=v, upper=float(p["upper"]),
                                absolute=bool(p["absolute"]))}


@node("diagnose.combine_verdicts", category="diagnose",
      inputs=[Port("a", "record"), Port("b", "record", required=False),
              Port("c", "record", required=False), Port("d", "record", required=False)],
      outputs=[Port("verdict", "record")],
      params=[_claim_param()],
      theory_ref="§5.4")
def combine_verdicts(inputs: Dict[str, Any], p: Dict[str, Any]) -> Dict[str, Any]:
    """Rejection criteria joined by OR: any REJECTED -> REJECTED; else any INCONCLUSIVE."""
    parts = [inputs[k] for k in ("a", "b", "c", "d") if inputs.get(k) is not None]
    statuses = [part.get("status") for part in parts]
    if VERDICT_REJECTED in statuses:
        status = VERDICT_REJECTED
    elif all(s == VERDICT_NOT_REJECTED for s in statuses):
        status = VERDICT_NOT_REJECTED
    else:
        status = VERDICT_INCONCLUSIVE
    return {"verdict": _verdict(status, p["claim"], parts=parts)}
