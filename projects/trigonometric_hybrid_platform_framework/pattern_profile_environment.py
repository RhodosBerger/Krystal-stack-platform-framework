"""Pattern-over-profiles environment for the summation-order fingerprint.

Runs diagnose_bit_identity() across a grid of layer geometries and seeds and models,
per device, which arithmetic recipe the device's convolution kernel follows.

Pattern under test (PATTERN_CANDIDATE from 2026-11-16):
  P1  GPU convolution kernel == sequential FMA summation in natural k order (c, ky, kx)
  P2  NumPy BLAS reference   == the same recipe (so GPU/NumPy bit-identity is shared arithmetic)
  P3  CPU oneDNN kernel      == none of the sequential recipes (blocked accumulation)

Verdict rules (no averaging of conflicts):
  every profile agrees      -> SUPPORTED_WITH_SCOPE (scope = profiles run, this host)
  some profiles disagree    -> CONFLICT (failing profiles listed explicitly)
  no profile could run      -> UNKNOWN
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence

from krystal_kernel.constants import VITAL_MAX_HP, ContractViolation

from .virtual_convolution_pipeline import diagnose_bit_identity

FMA_NATURAL = "seq_natural_k_order_fma"
BLAS = "numpy_einsum_blas"


@dataclass(frozen=True)
class ProfileSpec:
    profile_id: str
    channels: int
    filters: int
    height: int
    width: int
    seed: int

    @property
    def reduction_k(self) -> int:
        return self.channels * 9


DEFAULT_PROFILES: tuple = (
    ProfileSpec("P01_tiny", 4, 8, 16, 16, 1),
    ProfileSpec("P02_baseline", 16, 32, 64, 64, 6),
    ProfileSpec("P03_seed_change", 16, 32, 64, 64, 7),
    ProfileSpec("P04_rgb_K27", 3, 16, 96, 96, 6),
    ProfileSpec("P05_K288", 32, 32, 64, 64, 6),
    ProfileSpec("P06_wide_out", 32, 64, 32, 32, 6),
    ProfileSpec("P07_K576", 64, 32, 32, 32, 6),
    ProfileSpec("P08_large_spatial", 16, 16, 128, 128, 11),
)


def _verdict(flags: List[bool]) -> str:
    if not flags:
        return "UNKNOWN"
    return "SUPPORTED_WITH_SCOPE" if all(flags) else "CONFLICT"


def run_pattern_sweep(profiles: Sequence[ProfileSpec] = DEFAULT_PROFILES, vital_hp: int = VITAL_MAX_HP) -> Dict[str, Any]:
    if vital_hp != VITAL_MAX_HP:
        raise ContractViolation(f"vital_hp={vital_hp} != {VITAL_MAX_HP}", violation_code="ERR_SWEEP_VITAL_HP")
    ids = [p.profile_id for p in profiles]
    if len(ids) != len(set(ids)):
        raise ContractViolation("Duplicate profile IDs", violation_code="ERR_SWEEP_DUP_PROFILE")

    rows: List[Dict[str, Any]] = []
    for p in profiles:
        t0 = time.perf_counter()
        try:
            d = diagnose_bit_identity(channels=p.channels, filters=p.filters, height=p.height, width=p.width, seed=p.seed)
        except Exception as exc:  # recorded, never hidden
            rows.append({"profile": asdict(p), "K": p.reduction_k, "error": f"{type(exc).__name__}: {exc}"})
            continue
        row: Dict[str, Any] = {
            "profile": asdict(p),
            "K": p.reduction_k,
            "elapsed_s": round(time.perf_counter() - t0, 2),
            "blas_equals_fma_natural": d["variant_cross_check"]["einsum_equals_seq_natural_fma"],
            "blas_vs_fma_mismatch_fraction": d["variant_cross_check"]["einsum_vs_seq_natural_fma_mismatch_fraction"],
            "devices": {},
        }
        for dev, info in d["devices"].items():
            row["devices"][dev] = {
                "kernel": info["kernels"][0].get("primitiveType") if info["kernels"] else None,
                "bit_identical_to": info["bit_identical_to"],
                "max_abs_diff_vs_blas": info["max_abs_diff"][BLAS],
            }
        rows.append(row)

    ok_rows = [r for r in rows if "error" not in r]
    devices = sorted({dev for r in ok_rows for dev in r["devices"]})

    def flags_for(pred) -> Dict[str, Any]:
        flags = [pred(r) for r in ok_rows]
        return {
            "verdict": _verdict(flags),
            "holds_in": sum(flags),
            "profiles_run": len(flags),
            "failing_profiles": [r["profile"]["profile_id"] for r, f in zip(ok_rows, flags) if not f],
        }

    hypotheses: Dict[str, Any] = {
        "P2_blas_is_fma_natural": flags_for(lambda r: r["blas_equals_fma_natural"]),
    }
    if "GPU" in devices:
        hypotheses["P1_gpu_is_fma_natural"] = flags_for(
            lambda r: "GPU" in r["devices"] and FMA_NATURAL in r["devices"]["GPU"]["bit_identical_to"]
        )
    if "CPU" in devices:
        hypotheses["P3_cpu_matches_no_sequential_recipe"] = flags_for(
            lambda r: "CPU" in r["devices"]
            and not any(v.startswith("seq_") for v in r["devices"]["CPU"]["bit_identical_to"])
        )

    kernels_seen = {
        dev: sorted({r["devices"][dev]["kernel"] for r in ok_rows if dev in r["devices"] and r["devices"][dev]["kernel"]})
        for dev in devices
    }
    return {
        "vital_max_hp": VITAL_MAX_HP,
        "profiles_requested": len(profiles),
        "profiles_run": len(ok_rows),
        "errors": [r for r in rows if "error" in r],
        "kernels_seen": kernels_seen,
        "hypotheses": hypotheses,
        "rows": rows,
        "scope": "this host only (i5-1135G7 + Iris Xe, OpenVINO 2026.4.1, NumPy BLAS of the local .venv), FP32, 3x3, stride 1",
    }


if __name__ == "__main__":
    out = run_pattern_sweep()
    target = Path(__file__).resolve().parents[2] / "reports" / "conv_fingerprint_profile_sweep.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps({k: out[k] for k in ("profiles_run", "kernels_seen", "hypotheses", "errors")}, indent=2))
    for r in out["rows"]:
        if "error" in r:
            continue
        print(r["profile"]["profile_id"], "K=", r["K"], "blas==fma:", r["blas_equals_fma_natural"],
              {d: (v["kernel"], v["bit_identical_to"]) for d, v in r["devices"].items()})
    print("REPORT", target)
