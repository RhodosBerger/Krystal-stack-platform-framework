#!/usr/bin/env python3
"""
================================================================================
KSL v0.2 — HTE-001 STANDALONE INDEPENDENT VERIFICATION EVALUATOR
================================================================================
File: repro/HTE001/standalone_evaluator.py

Zero external dependencies: uses Python 3 standard library only.
Audits run manifests, hardware transition receipts, exact-zero memory invariants,
envelope compliance, and bootstrap 95% confidence intervals for third-party review.

Normative Standard:
  - Manifest Preregistration Hash: ae6a1add5fcf1d3cb36c36859d3f38ef48a27df340b8aaf39daa3b47ea17f9cf
  - Constitutional Sentinel P10:  VITAL_MAX_HP == 6
  - Zero Mutation Contract:       Delta S == 0 on invariant violation
  - Exact Zero Set:               I_0 = {memory: R_memory == 0.0}
  - Envelope Compliance:          max_{j in I_+} |r_j| <= 1.0
  - Statistical Criterion:        95% CI of mean(r_j) subset of [-1.0, 1.0], N >= 30
================================================================================
"""

import sys
import json
import math
import hmac
import hashlib
import argparse
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

EXPECTED_MANIFEST_DIGEST = "ae6a1add5fcf1d3cb36c36859d3f38ef48a27df340b8aaf39daa3b47ea17f9cf"
VITAL_MAX_HP_EXPECTED = 6


def rfc8785_format_float(x: float) -> str:
    """
    Formats an IEEE-754 float in strict compliance with RFC 8785 Section 3.2.2.3 and
    ECMAScript 6 (ECMA-262) Section 7.1.12.1 (ToString Applied to the Number Type).
    """
    if math.isnan(x) or math.isinf(x):
        raise ValueError("RFC 8785 JCS does not permit NaN or Infinity")
    if x == 0.0:
        return "0"

    sign = "-" if math.copysign(1.0, x) < 0 else ""
    x = abs(x)

    r = repr(x)
    if "e" in r:
        mantissa, exp_s = r.split("e")
        exp = int(exp_s)
        if "." in mantissa:
            int_part, frac_part = mantissa.split(".")
            s = int_part + frac_part
        else:
            s = mantissa
        s = s.rstrip("0") if "." in mantissa else s
        k = len(s)
        n = exp + 1
    else:
        if "." in r:
            int_part, frac_part = r.split(".")
        else:
            int_part, frac_part = r, ""

        if int_part == "0":
            stripped_frac = frac_part.lstrip("0")
            leading_zeros = len(frac_part) - len(stripped_frac)
            s = stripped_frac
            k = len(s)
            n = -leading_zeros
        else:
            s = (int_part + frac_part).rstrip("0")
            if not s:
                return "0"
            k = len(s)
            n = len(int_part)

    if k <= n <= 21:
        res = s + "0" * (n - k)
    elif 0 < n <= 21:
        res = s[:n] + "." + s[n:]
    elif -6 < n <= 0:
        res = "0." + "0" * (-n) + s
    else:
        if k == 1:
            a = s
        else:
            a = s[0] + "." + s[1:]
        c = "+" if (n - 1) >= 0 else "-"
        d = str(abs(n - 1))
        res = a + "e" + c + d

    return sign + res


def canonical_jcs_serialize(obj: Any) -> bytes:
    """
    Serializes a Python object to JSON adhering strictly to RFC 8785 (JSON Canonicalization Scheme - JCS).
    Key properties:
      1. Verbatim Unicode preservation.
      2. Deterministic key sorting by UTF-16 code units (RFC 8785 section 3.2.3).
      3. IEEE-754 float canonicalization compliant with ECMAScript 6 Section 7.1.12.1.
      4. Strict whitespace elimination (separators=(',', ':')).
    """
    if obj is None:
        return b"null"
    elif isinstance(obj, bool):
        return b"true" if obj else b"false"
    elif isinstance(obj, int):
        return str(obj).encode("utf-8")
    elif isinstance(obj, float):
        return rfc8785_format_float(obj).encode("utf-8")
    elif isinstance(obj, str):
        return json.dumps(obj, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    elif isinstance(obj, bytes):
        return json.dumps(obj.hex()).encode("utf-8")
    elif isinstance(obj, (list, tuple)):
        items = [canonical_jcs_serialize(item) for item in obj]
        return b"[" + b",".join(items) + b"]"
    elif isinstance(obj, dict):
        sorted_keys = sorted(
            obj.keys(),
            key=lambda k: str(k).encode("utf-16-be")
        )
        parts = []
        for k in sorted_keys:
            k_bytes = canonical_jcs_serialize(str(k))
            v_bytes = canonical_jcs_serialize(obj[k])
            parts.append(k_bytes + b":" + v_bytes)
        return b"{" + b",".join(parts) + b"}"
    elif hasattr(obj, "__dict__"):
        return canonical_jcs_serialize(obj.__dict__)
    else:
        return canonical_jcs_serialize(str(obj))


def evaluate_single_receipt(
    receipt: Dict[str, Any],
    manifest: Dict[str, Any],
    secret_key: Optional[bytes] = None,
) -> Tuple[bool, List[str]]:
    """
    Evaluates a single HardwareTransitionReceipt against the preregistered manifest.
    Returns (passed, list_of_audit_notes).
    """
    notes: List[str] = []
    passed = True

    # 1. Manifest Hash Verification
    rec_manifest_hash = receipt.get("manifest_hash", "")
    if rec_manifest_hash != EXPECTED_MANIFEST_DIGEST:
        notes.append(f"FAIL: Manifest hash mismatch: receipt has {rec_manifest_hash[:16]}..., expected {EXPECTED_MANIFEST_DIGEST[:16]}...")
        passed = False
    else:
        notes.append("PASS: Manifest hash matches preregistered anchor.")

    # 2. Constitutional Invariant Check (P10)
    hp = receipt.get("vital_max_hp")
    if hp != VITAL_MAX_HP_EXPECTED:
        notes.append(f"FAIL: Constitutional invariant breached: vital_max_hp={hp} != {VITAL_MAX_HP_EXPECTED}")
        passed = False
    else:
        notes.append(f"PASS: Constitutional invariant verified: vital_max_hp == {VITAL_MAX_HP_EXPECTED}.")

    # 3. Exact Zero Invariant Check (I_0)
    raw_res = receipt.get("raw_residuals", {})
    mem_res = raw_res.get("memory", None)
    if mem_res is None or mem_res != 0.0:
        notes.append(f"FAIL: I_0 Exact Zero Violation: memory delta R_memory={mem_res} != 0.0")
        passed = False
    else:
        notes.append("PASS: I_0 Exact Zero Invariant holds: R_memory == 0.0 (Delta S_protected == 0).")

    # 4. Normalized Residual Envelope Check (I_+)
    norm_res = receipt.get("normalized_residuals", {})
    tolerance_fields = manifest.get("normalized_tolerance_fields", ["compute", "energy", "latency", "queue", "thermal"])
    envelope_ok = True
    for field in tolerance_fields:
        r_val = norm_res.get(field)
        if r_val is None:
            notes.append(f"FAIL: Missing normalized residual for field '{field}'")
            envelope_ok = False
            passed = False
        elif abs(r_val) > 1.0:
            notes.append(f"FAIL: Envelope breach on '{field}': |{r_val:.4f}| > 1.0")
            envelope_ok = False
            passed = False
    if envelope_ok:
        notes.append(f"PASS: Normalized envelope compliance: all |r_j| <= 1.0 across {len(tolerance_fields)} fields.")

    # 5. Decoupled Epistemic Status
    integrity = receipt.get("receipt_integrity")
    ev_class = receipt.get("evidence_class")
    verdict = receipt.get("semantic_verdict")
    notes.append(f"INFO: Epistemic Status: <Integrity={integrity}, EvidenceClass={ev_class}, Verdict={verdict}>")

    # 6. HMAC Signature Verification (if key provided)
    if secret_key:
        sig = receipt.get("hmac_signature", "")
        data_to_verify = {k: v for k, v in receipt.items() if k != "hmac_signature"}
        jcs_bytes = canonical_jcs_serialize(data_to_verify)
        expected_sig = hmac.new(secret_key, jcs_bytes, hashlib.sha256).hexdigest()
        if hmac.compare_digest(sig, expected_sig):
            notes.append("PASS: HMAC-SHA256 signature verified against key.")
        else:
            notes.append("FAIL: HMAC-SHA256 signature mismatch.")
            passed = False

    return passed, notes


def bootstrap_ci_95(samples: List[float], iterations: int = 1000) -> Tuple[float, float, float]:
    """
    Computes empirical bootstrap 95% Confidence Interval for sample mean.
    Returns (mean, ci_lower, ci_upper).
    """
    n = len(samples)
    if n == 0:
        return 0.0, 0.0, 0.0
    mean_val = sum(samples) / n
    if n < 5:
        return mean_val, min(samples), max(samples)

    # Deterministic pseudo-random generation using linear congruential generator for reproducibility
    seed = 42
    boot_means: List[float] = []
    for _ in range(iterations):
        b_sum = 0.0
        for _ in range(n):
            seed = (seed * 1103515245 + 12345) & 0x7FFFFFFF
            idx = seed % n
            b_sum += samples[idx]
        boot_means.append(b_sum / n)

    boot_means.sort()
    lower_idx = int(iterations * 0.025)
    upper_idx = int(iterations * 0.975)
    return mean_val, boot_means[lower_idx], boot_means[upper_idx]


def evaluate_batch_run(
    receipts: List[Dict[str, Any]],
    manifest: Dict[str, Any],
) -> Tuple[bool, Dict[str, Any]]:
    """
    Evaluates multi-run statistical compliance per manifest criterion.
    Requires at least 30 acquisition-valid runs, exact zero pass, and CI 95% within [-1.0, 1.0].
    """
    crit = manifest.get("statistical_criterion", {})
    min_runs = crit.get("min_acquisition_valid_runs", 30)

    acq_valid_count = sum(1 for r in receipts if r.get("acquisition_valid", False))
    transition_pass_count = sum(1 for r in receipts if r.get("transition_pass", False))

    field_samples: Dict[str, List[float]] = {}
    exact_zero_violations = 0

    for r in receipts:
        if not r.get("exact_zero_pass", False):
            exact_zero_violations += 1
        norm_res = r.get("normalized_residuals", {})
        for k, v in norm_res.items():
            field_samples.setdefault(k, []).append(float(v))

    ci_results: Dict[str, Dict[str, float]] = {}
    ci_pass = True
    for k, vals in field_samples.items():
        m, lo, hi = bootstrap_ci_95(vals, 1000)
        in_bounds = (-1.0 <= lo <= 1.0) and (-1.0 <= hi <= 1.0)
        ci_results[k] = {"mean": m, "ci_lower": lo, "ci_upper": hi, "in_bounds": in_bounds}
        if not in_bounds:
            ci_pass = False

    overall_pass = (
        (acq_valid_count >= min_runs)
        and (exact_zero_violations == 0)
        and ci_pass
    )

    summary = {
        "total_receipts": len(receipts),
        "acquisition_valid_runs": acq_valid_count,
        "required_runs": min_runs,
        "transition_pass_count": transition_pass_count,
        "exact_zero_violations": exact_zero_violations,
        "ci_results": ci_results,
        "ci_pass": ci_pass,
        "overall_batch_pass": overall_pass,
    }
    return overall_pass, summary


def main():
    parser = argparse.ArgumentParser(description="KSL v0.2 HTE-001 Standalone Independent Verifier")
    parser.add_argument("--receipt", help="Path to single receipt JSON file or directory")
    parser.add_argument("--manifest", default="repro/HTE001/HTE001_RUN_MANIFEST.json", help="Path to manifest JSON")
    parser.add_argument("--secret-key-hex", help="HMAC secret key in hex format for signature audit")
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        print(f"ERROR: Manifest file not found: {manifest_path}", file=sys.stderr)
        sys.exit(1)

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    # Check manifest canonical hash
    canonical_bytes = canonical_jcs_serialize(manifest)
    manifest_digest = hashlib.sha256(canonical_bytes).hexdigest()
    if manifest_digest != EXPECTED_MANIFEST_DIGEST:
        print(f"FATAL: Manifest hash {manifest_digest} != expected {EXPECTED_MANIFEST_DIGEST}", file=sys.stderr)
        sys.exit(1)

    print("=" * 80)
    print("KSL v0.2 HTE-001 INDEPENDENT VERIFICATION AUDIT")
    print("=" * 80)
    print(f"Manifest:          {manifest_path}")
    print(f"Manifest Digest:   {manifest_digest} [MATCHES PREREGISTERED ANCHOR]")
    print(f"Experiment:        {manifest.get('experiment_id')} / {manifest.get('run_series_id')}")
    print(f"Board & FPGA:      {manifest.get('board_model')} ({manifest.get('fpga_part')})")
    print(f"Clock / Sampling:  {manifest.get('clock_frequency_hz')} Hz / {manifest.get('sampling_rate_hz')} Hz")
    print(f"Tolerances:        Latency={manifest.get('epsilon_latency')}ns, Energy={manifest.get('epsilon_energy')}uJ, Memory=0.0B (EXACT)")
    print("-" * 80)

    secret_key = bytes.fromhex(args.secret_key_hex) if args.secret_key_hex else None

    if not args.receipt:
        print("No receipt file provided. Manifest verification completed successfully.")
        print("Run with --receipt <receipt.json> to verify a hardware execution receipt.")
        print("=" * 80)
        sys.exit(0)

    receipt_path = Path(args.receipt)
    receipts_to_test: List[Dict[str, Any]] = []

    if receipt_path.is_file():
        with open(receipt_path, "r", encoding="utf-8") as f:
            receipts_to_test.append(json.load(f))
    elif receipt_path.is_dir():
        for p in sorted(receipt_path.glob("*.json")):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if "experiment_id" in data:
                        receipts_to_test.append(data)
            except Exception:
                pass

    if not receipts_to_test:
        print(f"ERROR: No valid receipt JSON found at: {receipt_path}", file=sys.stderr)
        sys.exit(1)

    all_single_passed = True
    for idx, r in enumerate(receipts_to_test):
        rec_id = r.get("run_id", f"run_{idx}")
        print(f"\nAUDITING RECEIPT: {rec_id}")
        passed, notes = evaluate_single_receipt(r, manifest, secret_key)
        for n in notes:
            print(f"  {n}")
        if not passed:
            all_single_passed = False

    if len(receipts_to_test) >= 30:
        print("\n" + "-" * 80)
        print("STATISTICAL BATCH EVALUATION (N >= 30)")
        print("-" * 80)
        batch_passed, batch_summary = evaluate_batch_run(receipts_to_test, manifest)
        print(f"Acquisition-Valid Runs: {batch_summary['acquisition_valid_runs']} / {batch_summary['required_runs']}")
        print(f"Exact Zero Violations:  {batch_summary['exact_zero_violations']}")
        for fld, ci in batch_summary["ci_results"].items():
            status = "PASS" if ci["in_bounds"] else "FAIL"
            print(f"  [{status}] {fld:<10s}: Mean={ci['mean']:+.4f}, 95% CI=[{ci['ci_lower']:+.4f}, {ci['ci_upper']:+.4f}]")
        print(f"Batch Statistical Verdict: {'PASS' if batch_passed else 'FAIL'}")
        exit_code = 0 if (all_single_passed and batch_passed) else 1
    else:
        exit_code = 0 if all_single_passed else 1

    print("=" * 80)
    print(f"OVERALL AUDIT VERDICT: {'COMPLIANT' if exit_code == 0 else 'NON-COMPLIANT'}")
    print("=" * 80)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
