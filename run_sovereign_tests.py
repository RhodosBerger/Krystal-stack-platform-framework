#!/usr/bin/env python3
"""
run_sovereign_tests.py

One-Click Multi-Layer Automated Run-Testing & Benchmarking CLI.
==============================================================
Executes stress runs, latency percentiles, and invariant audits across
all 6 sovereign layers of Krystal Stack and Godot Krystal Edition.

Constitutional Invariant: VITAL_MAX_HP == 6
"""

import sys
import json
from krystal_kernel.sovereign_run_benchmark_harness import GLOBAL_SOVEREIGN_RUN_BENCHMARK_HARNESS
from krystal_kernel.constants import VITAL_MAX_HP


def main() -> int:
    print("==================================================================")
    print("  KRYSTAL STACK // SOVEREIGN LAYER AUTOMATED RUN-TESTING HARNESS")
    print(f"  Constitutional Invariant: VITAL_MAX_HP == {VITAL_MAX_HP}")
    print("==================================================================")

    receipt = GLOBAL_SOVEREIGN_RUN_BENCHMARK_HARNESS.run_full_suite()

    print(f"\n[RUN RECEIPT] ID: {receipt.run_id} | Timestamp: {receipt.timestamp_iso}")
    print(f"Total Duration: {receipt.total_duration_ms:.2f} ms | Layers Tested: {receipt.total_layers_tested}")
    print("------------------------------------------------------------------")

    for layer_id, bench in receipt.layer_benchmarks.items():
        status = "PASSED" if bench.passed else "FAILED"
        print(f"[{status}] {bench.layer_name:<26} | {bench.iterations:>6} iters | {bench.throughput_ops_per_sec:>9.1f} ops/s | p50: {bench.latency_p50_us:>5.2f} us | p95: {bench.latency_p95_us:>5.2f} us | p99: {bench.latency_p99_us:>5.2f} us")

    print("------------------------------------------------------------------")
    print(f"HMAC-SHA256 Digest: {receipt.receipt_hmac_digest}")
    print("==================================================================")

    if receipt.all_layers_passed:
        print(">> ALL SOVEREIGN LAYERS PASSED OPERATIONAL RUN-TESTING (100% OK)")
        return 0
    else:
        print(">> WARNING: ONE OR MORE SOVEREIGN LAYERS FAILED CRITICAL AUDIT")
        return 1


if __name__ == "__main__":
    sys.exit(main())
