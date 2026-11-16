"""Architectural consistency bot with multimodal (data + text + image) reporting.

The bot turns knowledge claims into executable checks. Each check produces:
  claim -> measurement -> falsifier -> verdict (ENLG lattice) -> evidence class
and the run is published as JSON (data), Markdown (text) and PNG (image).

Verdicts: VERIFIED_WITH_SCOPE | MEASURED_WITH_SCOPE | SUPPORTED_WITH_SCOPE | CONFLICT | FALSIFIED | UNKNOWN
The bot never upgrades a verdict without a passing measurement, and records exceptions as UNKNOWN.
"""

from __future__ import annotations

import json
import re
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List

import numpy as np

from krystal_kernel.constants import VITAL_MAX_HP

REPO_ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = REPO_ROOT / "reports"

VERDICT_COLORS = {  # Krystal UI manual semantic colors
    "VERIFIED_WITH_SCOPE": "#43D264",
    "MEASURED_WITH_SCOPE": "#43D264",
    "SUPPORTED_WITH_SCOPE": "#50EEE0",
    "CONFLICT": "#F59E0B",
    "FALSIFIED": "#FF4757",
    "UNKNOWN": "#8D8997",
}


@dataclass(frozen=True)
class BotCheck:
    check_id: str
    claim: str
    falsifier: str
    evidence_class: str
    run: Callable[[bool], Dict[str, Any]]  # returns {"verdict": ..., "measured": {...}}


# ---------------------------------------------------------------------------
# Measurement helpers
# ---------------------------------------------------------------------------

def blas_sequential_fma_threshold(m: int = 32, n: int = 1024, lo: int = 432, hi: int = 468, seed: int = 6) -> Dict[str, Any]:
    """Smallest reduction length K at which NumPy's BLAS matmul stops being bit-identical to a
    sequential FMA accumulation in natural k order. Bisection between a known-equal lo and
    known-different hi; the neighbours of the result are re-checked (monotonicity is assumed,
    not proven)."""
    rng = np.random.default_rng(seed)

    def equal_at(k: int) -> bool:
        a = rng.standard_normal((m, k)).astype(np.float32)
        b = rng.standard_normal((k, n)).astype(np.float32)
        blas = a @ b
        acc = np.zeros((m, n), dtype=np.float32)
        for i in range(k):
            acc = (acc.astype(np.float64) + a[:, i:i + 1].astype(np.float64) * b[i:i + 1, :].astype(np.float64)).astype(np.float32)
        return bool(np.array_equal(blas, acc))

    if not equal_at(lo) or equal_at(hi):
        return {"threshold_k": None, "note": "bracket invalid on this host"}
    a_lo, a_hi = lo, hi
    while a_hi - a_lo > 1:
        mid = (a_lo + a_hi) // 2
        if equal_at(mid):
            a_lo = mid
        else:
            a_hi = mid
    return {
        "last_equal_k": a_lo,
        "threshold_k": a_hi,
        "neighbour_check": {a_lo - 1: equal_at(a_lo - 1), a_hi + 1: equal_at(a_hi + 1)},
        "matrix_shape_m_n": [m, n],
    }


def _app_default_port() -> int:
    src = (REPO_ROOT / "krystal_kernel" / "sovereign_openvino_oneapi_app.py").read_text(encoding="utf-8")
    m = re.search(r'KRYSTAL_PORT",\s*"(\d+)"', src)
    return int(m.group(1)) if m else -1


def _nginx_upstream_ports(path: Path) -> List[int]:
    if not path.exists():
        return []
    return [int(p) for p in re.findall(r"server\s+127\.0\.0\.1:(\d+);", path.read_text(encoding="utf-8"))]


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------

def _c01_graph(fast: bool) -> Dict[str, Any]:
    from .adaptive_cross_genre_sync import validate_connection_graph

    g = validate_connection_graph()
    return {"verdict": "VERIFIED_WITH_SCOPE" if g["graph_valid"] else "FALSIFIED", "measured": g}


def _c02_rollback(fast: bool) -> Dict[str, Any]:
    from .adaptive_cross_genre_sync import simulate_adaptive_rollback

    ticks = 300 if fast else 1000
    runs = {n: simulate_adaptive_rollback(ticks=ticks, one_way_latency_ms=l, jitter_sigma_ms=j)
            for n, l, j in (("LAN", 10.0, 2.0), ("REGIONAL", 40.0, 6.0), ("WAN", 100.0, 20.0))}
    desync = {n: r["desync_ticks"] for n, r in runs.items()}
    return {
        "verdict": "SUPPORTED_WITH_SCOPE" if not any(desync.values()) else "FALSIFIED",
        "measured": {"desync_ticks": desync, "max_horizon": {n: r["max_horizon"] for n, r in runs.items()}, "ticks": ticks},
    }


def _c03_conv(fast: bool) -> Dict[str, Any]:
    from .virtual_convolution_pipeline import run_virtual_convolution_pipeline

    kw = dict(channels=8, filters=16, height=32, width=32, iterations=10) if fast else dict(iterations=30)
    r = run_virtual_convolution_pipeline(**kw)
    devs = r["devices"]
    all_match = all(d.get("matches_reference") for d in devs.values())
    measured = {
        dev: {k: d.get(k) for k in ("device_name", "compile_ms", "max_rel_error", "median_ms", "p95_ms", "stdev_ms", "n", "verdict")}
        for dev, d in devs.items()
    }
    return {"verdict": "MEASURED_WITH_SCOPE" if all_match else "FALSIFIED",
            "measured": {"input_shape": r["input_shape"], "devices": measured}}


def _c04_contra(fast: bool) -> Dict[str, Any]:
    from .virtual_convolution_contra import run_contra_modules

    r = run_contra_modules(channels=2, filters=4, height=12, width=12, cg_iterations=200) if fast else run_contra_modules()
    ok = r["contra1_adjoint_dot_test_rel_gap"] < 1e-5 and r["contra2_linear_inverse"]["reconstruction_rel_error"] < 1e-3
    return {
        "verdict": "VERIFIED_WITH_SCOPE" if ok else "FALSIFIED",
        "measured": {
            "adjoint_gap": r["contra1_adjoint_dot_test_rel_gap"],
            "openvino_transpose": r["contra1_openvino_transpose_conv"],
            "linear_inverse_error": r["contra2_linear_inverse"]["reconstruction_rel_error"],
            "relu_inverse_error": r["contra3_relu_partial_inverse"]["reconstruction_rel_error"],
        },
    }


def _c05_sweep(fast: bool) -> Dict[str, Any]:
    from .pattern_profile_environment import DEFAULT_PROFILES, ProfileSpec, run_pattern_sweep

    profiles = (ProfileSpec("F1", 2, 4, 8, 8, 1), ProfileSpec("F2", 4, 8, 12, 12, 2)) if fast else DEFAULT_PROFILES
    r = run_pattern_sweep(profiles)
    verdicts = [h["verdict"] for h in r["hypotheses"].values()]
    overall = "CONFLICT" if "CONFLICT" in verdicts else ("UNKNOWN" if "UNKNOWN" in verdicts else "SUPPORTED_WITH_SCOPE")
    return {"verdict": overall, "measured": {"hypotheses": r["hypotheses"], "kernels_seen": r["kernels_seen"]}}


def _c06_blas_threshold(fast: bool) -> Dict[str, Any]:
    r = blas_sequential_fma_threshold()
    return {"verdict": "MEASURED_WITH_SCOPE" if r.get("threshold_k") else "UNKNOWN", "measured": r}


def _c07_design_manual(fast: bool) -> Dict[str, Any]:
    from .adaptive_sync_card_grid import render_adaptive_sync_card_grid_html

    html_doc = render_adaptive_sync_card_grid_html(sim_ticks=200)
    violations = [t for t in ("box-shadow", "backdrop-filter") if t in html_doc]
    if re.search(r"border-radius:\s*(?:[5-9]|\d{2,})px", html_doc):
        violations.append("border-radius > 4px")
    return {"verdict": "VERIFIED_WITH_SCOPE" if not violations else "FALSIFIED", "measured": {"violations": violations}}


def _c08_deploy_ports(fast: bool) -> Dict[str, Any]:
    app_port = _app_default_port()
    platform = _nginx_upstream_ports(REPO_ROOT / "deploy" / "nginx_krystal_platform.conf")
    subdomain = _nginx_upstream_ports(REPO_ROOT / "deploy" / "nginx_krystal_subdomain_ssl.conf")
    mismatched = [p for p in subdomain if p not in (80, app_port)]
    return {
        "verdict": "CONFLICT" if mismatched or app_port not in platform else "VERIFIED_WITH_SCOPE",
        "measured": {"app_default_port": app_port, "platform_conf_upstreams": platform,
                     "subdomain_conf_upstreams": subdomain, "mismatched_upstreams": mismatched,
                     "docker_build": "NOT_RUN (no Docker on host)"},
    }


CHECKS: tuple = (
    BotCheck("C01", "Sync connection graph obeys authority rules", "any edge bypasses STEP_GATE_PROXY", "STATIC_VALIDATION", _c01_graph),
    BotCheck("C02", "Adaptive rollback has zero desync on 3 latency profiles", "any checksum desync", "SIMULATED", _c02_rollback),
    BotCheck("C03", "OpenVINO conv pipeline matches NumPy reference on every device", "rel error >= 1e-3 on any device", "LOCAL_MEASURED", _c03_conv),
    BotCheck("C04", "Transposed conv is the exact adjoint; linear layer is invertible", "adjoint gap >= 1e-5 or inverse error >= 1e-3", "LOCAL_MEASURED_NUMERICAL", _c04_contra),
    BotCheck("C05", "Summation-order fingerprint pattern holds over profiles", "any hypothesis fails on a profile", "LOCAL_MEASURED", _c05_sweep),
    BotCheck("C06", "Locate K where NumPy BLAS leaves sequential-FMA order", "no threshold in bracket", "LOCAL_MEASURED", _c06_blas_threshold),
    BotCheck("C07", "Card grid respects the Krystal UI design manual bans", "shadow / blur / radius > 4px present", "STATIC_VALIDATION", _c07_design_manual),
    BotCheck("C08", "Deployment configs agree on the server port", "nginx upstream port != app port", "STATIC_VALIDATION", _c08_deploy_ports),
)


# ---------------------------------------------------------------------------
# Runner + multimodal report
# ---------------------------------------------------------------------------

def run_bot(fast: bool = False, out_dir: Path = REPORT_DIR, checks: tuple = CHECKS) -> Dict[str, Any]:
    results: List[Dict[str, Any]] = []
    for c in checks:
        t0 = time.perf_counter()
        try:
            r = c.run(fast)
        except Exception as exc:  # never hidden: an exception is UNKNOWN, not PASS
            r = {"verdict": "UNKNOWN", "measured": {"error": f"{type(exc).__name__}: {exc}",
                                                    "trace_tail": traceback.format_exc().splitlines()[-3:]}}
        results.append({"check_id": c.check_id, "claim": c.claim, "falsifier": c.falsifier,
                        "evidence_class": c.evidence_class, "verdict": r["verdict"],
                        "elapsed_s": round(time.perf_counter() - t0, 2), "measured": r["measured"]})

    summary: Dict[str, int] = {}
    for r in results:
        summary[r["verdict"]] = summary.get(r["verdict"], 0) + 1
    report = {"vital_max_hp": VITAL_MAX_HP, "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
              "mode": "fast" if fast else "full", "summary": summary, "checks": results}

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "consistency_bot_latest.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    (out_dir / "consistency_bot_latest.md").write_text(_render_markdown(report), encoding="utf-8")
    _render_png(report, out_dir / "consistency_bot_latest.png")
    report["artifacts"] = {k: str(out_dir / f"consistency_bot_latest.{k}") for k in ("json", "md", "png")}
    return report


def _render_markdown(report: Dict[str, Any]) -> str:
    lines = [f"# Consistency bot report ({report['mode']})", "", f"Generated: {report['generated_at']}", "",
             "| Check | Claim | Verdict | Evidence | Time (s) |", "| --- | --- | --- | --- | --- |"]
    for r in report["checks"]:
        lines.append(f"| {r['check_id']} | {r['claim']} | **{r['verdict']}** | {r['evidence_class']} | {r['elapsed_s']} |")
    lines += ["", "Falsifiers:", ""] + [f"- {r['check_id']}: {r['falsifier']}" for r in report["checks"]]
    return "\n".join(lines) + "\n"


def _render_png(report: Dict[str, Any], path: Path) -> None:
    from PIL import Image, ImageDraw, ImageFont

    w, row_h, top = 1200, 40, 96
    h = top + row_h * len(report["checks"]) + 160
    img = Image.new("RGB", (w, h), "#030406")
    d = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("consola.ttf", 16)
        big = ImageFont.truetype("consola.ttf", 28)
    except OSError:
        font = big = ImageFont.load_default()
    d.text((32, 24), "KRYSTAL CONSISTENCY BOT", fill="#58D8FF", font=big)
    d.text((32, 64), f"{report['generated_at']}  mode={report['mode']}  summary={report['summary']}", fill="#8D8997", font=font)
    for i, r in enumerate(report["checks"]):
        y = top + i * row_h
        d.rectangle([32, y, w - 32, y + row_h - 8], outline="#292A38", width=1)
        color = VERDICT_COLORS.get(r["verdict"], "#8D8997")
        d.rectangle([32, y, 40, y + row_h - 8], fill=color)
        d.text((52, y + 8), f"{r['check_id']}  {r['claim'][:70]}", fill="#F1EFE8", font=font)
        d.text((w - 330, y + 8), r["verdict"], fill=color, font=font)

    # Latency bars for C03 (median and p95 per device), if measured.
    c03 = next((r for r in report["checks"] if r["check_id"] == "C03"), None)
    y0 = top + row_h * len(report["checks"]) + 24
    if c03 and isinstance(c03["measured"], dict) and "devices" in c03["measured"]:
        devs = c03["measured"]["devices"]
        vals = [v for dv in devs.values() for v in (dv.get("median_ms"), dv.get("p95_ms")) if v]
        scale = (w - 400) / max(vals) if vals else 1
        d.text((32, y0), "C03 latency [MEASURED]  median / p95 (ms)", fill="#8D8997", font=font)
        for j, (dev, dv) in enumerate(devs.items()):
            y = y0 + 28 + j * 44
            d.text((32, y), dev, fill="#F1EFE8", font=font)
            if dv.get("median_ms"):
                d.rectangle([120, y, 120 + dv["median_ms"] * scale, y + 12], fill="#58D8FF")
                d.rectangle([120, y + 16, 120 + dv["p95_ms"] * scale, y + 28], outline="#58D8FF", width=1)
                d.text((130 + dv["p95_ms"] * scale, y + 6), f"{dv['median_ms']} / {dv['p95_ms']}", fill="#50EEE0", font=font)
    img.save(path, format="PNG")


if __name__ == "__main__":
    import sys

    rep = run_bot(fast="--fast" in sys.argv)
    print(json.dumps(rep["summary"]))
    for r in rep["checks"]:
        print(r["check_id"], r["verdict"], r["elapsed_s"], json.dumps(r["measured"], default=str)[:300])
    print(rep["artifacts"])
