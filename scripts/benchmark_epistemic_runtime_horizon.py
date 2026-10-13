#!/usr/bin/env python3
"""
==============================================================================
KRYSTAL-STACK: EPISTEMIC RUNTIME BENCHMARK PROGRAM (SECTION 8)
==============================================================================
Script: scripts/benchmark_epistemic_runtime_horizon.py
Description:
    Benchmarks all required feature horizon groups:
      Group A: Epistemic Type Overhead (Creation, TTL verification)
      Group B: Envelope Normalization (Raw hardware to EpistemicValue)
      Group C: Causal Event Logging (Lineage emission, parent tracking)
      Group D: State-Space Pruner (Constraint solving throughput)
      Group E: Dual Authority Gate (SafetyGate + EvidenceGate)
      Group J: Proof Bundle Generation & Hashing
      Group K: Deterministic Replay Throughput
      Group L: Memory Footprint & Garbage Overhead

    Metrics Reported:
      p50, p95, p99, Mean, StdDev, CV (%), RSS (MB), Bytes/Event, Ops/sec.
==============================================================================
"""

import os
import sys
import time
import math
import json
import tracemalloc
from typing import List, Dict, Any, Callable

# Add workspace root to sys.path
WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

from krystal_kernel.scene_constructor_presets import VITAL_MAX_HP
from krystal_kernel.verification.state_version import StateVersion
from krystal_kernel.verification.epistemic_value import (
    ValidityState,
    EvidenceKind,
    EpistemicValue
)
from krystal_kernel.verification.resource_evidence_envelope import (
    ResourceVector,
    ExpectedDelta,
    ResourceEvidenceEnvelope
)
from krystal_kernel.verification.authority_gate import (
    CandidateAction,
    SafetyGate,
    EvidenceGate,
    DualAuthorityGate
)
from krystal_kernel.verification.state_space_pruner import StateSpacePruner
from krystal_kernel.verification.causal_event_log import (
    CausalEventLog,
    CausalEventClass
)
from krystal_kernel.verification.proof_bundle import ReversibilityClass
from krystal_kernel.runtime.shadow_runtime import ShadowRuntime


def compute_statistics(latencies_us: List[float]) -> Dict[str, float]:
    latencies_us.sort()
    n = len(latencies_us)
    if n == 0:
        return {}

    mean = sum(latencies_us) / n
    variance = sum((x - mean) ** 2 for x in latencies_us) / (n - 1) if n > 1 else 0.0
    stddev = math.sqrt(variance)
    cv = (stddev / mean * 100.0) if mean > 0 else 0.0

    p50 = latencies_us[int(n * 0.50)]
    p95 = latencies_us[min(n - 1, int(n * 0.95))]
    p99 = latencies_us[min(n - 1, int(n * 0.99))]

    return {
        "count": n,
        "mean_us": round(mean, 3),
        "stddev_us": round(stddev, 3),
        "cv_pct": round(cv, 2),
        "p50_us": round(p50, 3),
        "p95_us": round(p95, 3),
        "p99_us": round(p99, 3),
        "ops_per_sec": round(1e6 / mean, 1) if mean > 0 else 0.0
    }


def benchmark_group_a_epistemic_type(iterations: int = 5000) -> Dict[str, Any]:
    """Group A: Epistemic Type Overhead (Creation, validity calculation)"""
    sv = StateVersion(1, 0, 1, time.time())
    now_ns = int(time.time() * 1e9)
    ttl = int(2e9)

    tracemalloc.start()
    latencies: List[float] = []

    for _ in range(iterations):
        t0 = time.perf_counter()
        ev = EpistemicValue(
            value=45.2,
            provenance="CppHardwareProbe:hwmon",
            evidence_kind=EvidenceKind.HARDWARE_MEASURED,
            uncertainty=0.015,
            valid_from_ns=now_ns,
            valid_to_ns=now_ns + ttl,
            domain="hardware.thermal",
            state_version=sv,
            source_adapter="CppHardwareProbe",
            confidence=0.98
        )
        st = ev.get_validity_state(now_ns)
        usable = ev.is_usable_for_decision(now_ns)
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1e6)

    current_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    stats = compute_statistics(latencies)
    stats["bytes_per_op"] = round(peak_mem / iterations, 2)
    stats["peak_rss_kb"] = round(peak_mem / 1024, 2)
    return stats


def benchmark_group_b_envelope_normalization(iterations: int = 2000) -> Dict[str, Any]:
    """Group B: ResourceEvidenceEnvelope Normalization & Serialization"""
    sv = StateVersion(1, 0, 1, time.time())
    now_ns = int(time.time() * 1e9)
    ttl = int(2e9)

    tracemalloc.start()
    latencies: List[float] = []

    for i in range(iterations):
        t0 = time.perf_counter()
        ev_cpu = EpistemicValue(42.0, "cgroups_v2", EvidenceKind.RUNTIME_MEASURED, 0.02, now_ns, now_ns + ttl, "cpu.usage", sv, "KvmQemuAdapter")
        ev_mem = EpistemicValue(1024*1024*1024, "cgroups_v2", EvidenceKind.RUNTIME_MEASURED, 0.01, now_ns, now_ns + ttl, "memory.rss", sv, "KvmQemuAdapter")
        ev_tmp = EpistemicValue(52.0, "hwmon", EvidenceKind.HARDWARE_MEASURED, 0.01, now_ns, now_ns + ttl, "hardware.thermal", sv, "CppHardwareProbe")
        
        env = ResourceEvidenceEnvelope(
            envelope_id=f"ENV-{i:06d}",
            state_version=sv,
            provenance="KvmQemuAdapter",
            host_monotonic_ns=now_ns,
            canonical_normalized={"cpu": ev_cpu, "memory": ev_mem, "thermal": ev_tmp}
        )
        d = env.to_dict()
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1e6)

    current_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    stats = compute_statistics(latencies)
    stats["bytes_per_op"] = round(peak_mem / iterations, 2)
    stats["peak_rss_kb"] = round(peak_mem / 1024, 2)
    return stats


def benchmark_group_c_causal_logging(iterations: int = 3000) -> Dict[str, Any]:
    """Group C: Causal Event Logging (Emit and DAG Lineage)"""
    log = CausalEventLog(capacity=5000)
    sv = StateVersion(1, 0, 1, time.time())
    now_ns = int(time.time() * 1e9)

    tracemalloc.start()
    latencies: List[float] = []
    parent_id = None

    for i in range(iterations):
        t0 = time.perf_counter()
        ev = log.emit(
            event_class=CausalEventClass.PREDICTION,
            state_version=sv,
            causal_parent=parent_id,
            payload={"step": i, "delta": -2.5},
            now_ns=now_ns + i * 100
        )
        parent_id = ev.event_id
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1e6)

    current_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    stats = compute_statistics(latencies)
    stats["bytes_per_op"] = round(peak_mem / iterations, 2)
    stats["peak_rss_kb"] = round(peak_mem / 1024, 2)
    return stats


def benchmark_group_d_state_space_pruner(iterations: int = 1500) -> Dict[str, Any]:
    """Group D: State Space Pruner (Constraint Solving Throughput)"""
    pruner = StateSpacePruner()
    safety_gate = SafetyGate()
    evidence_gate = EvidenceGate()
    sv = StateVersion(1, 0, 1, time.time())
    now_ns = int(time.time() * 1e9)
    ttl = int(2e9)

    env = ResourceEvidenceEnvelope(
        envelope_id="ENV-BENCH-PRUNE",
        state_version=sv,
        provenance="KvmQemuAdapter",
        host_monotonic_ns=now_ns,
        canonical_normalized={
            "cpu_usage_pct": EpistemicValue(40.0, "cgroups", EvidenceKind.RUNTIME_MEASURED, 0.02, now_ns, now_ns + ttl, "cpu.usage", sv, "KvmQemuAdapter"),
            "temperature_celsius": EpistemicValue(50.0, "hwmon", EvidenceKind.HARDWARE_MEASURED, 0.01, now_ns, now_ns + ttl, "hardware.thermal", sv, "CppHardwareProbe")
        }
    )

    # 4 candidates: 2 valid, 1 expired, 1 overflow
    cands = [
        CandidateAction("ACT-1", sv, now_ns, now_ns + int(1e9), "RESIZE_BALLOON", "g1", ExpectedDelta({"cpu": -1.0}, 0.9, int(1e9)), ResourceVector(cpu_normalized_share=0.1), [env.envelope_id], ReversibilityClass.REVERSIBLE),
        CandidateAction("ACT-2", sv, now_ns, now_ns + int(1e9), "ADJUST_CPU_SHARES", "g1", ExpectedDelta({"cpu": -2.0}, 0.85, int(1e9)), ResourceVector(cpu_normalized_share=0.2), [env.envelope_id], ReversibilityClass.REVERSIBLE),
        CandidateAction("ACT-3", sv, now_ns - int(2e9), now_ns - int(1e9), "RESIZE_BALLOON", "g1", ExpectedDelta({"cpu": -1.0}, 0.9, int(1e9)), ResourceVector(cpu_normalized_share=0.1), [env.envelope_id], ReversibilityClass.REVERSIBLE),
        CandidateAction("ACT-4", sv, now_ns, now_ns + int(1e9), "RESIZE_BALLOON", "g1", ExpectedDelta({"cpu": -1.0}, 0.9, int(1e9)), ResourceVector(cpu_normalized_share=3.0), [env.envelope_id], ReversibilityClass.REVERSIBLE),
    ]

    tracemalloc.start()
    latencies: List[float] = []

    for _ in range(iterations):
        t0 = time.perf_counter()
        rep = pruner.prune(cands, env, now_ns, safety_gate, evidence_gate)
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1e6)

    current_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    stats = compute_statistics(latencies)
    stats["bytes_per_op"] = round(peak_mem / iterations, 2)
    stats["peak_rss_kb"] = round(peak_mem / 1024, 2)
    return stats


def benchmark_group_j_proof_bundle(iterations: int = 1500) -> Dict[str, Any]:
    """Group J: Proof Bundle Generation and Cryptographic Hash Arbitrament"""
    gate = DualAuthorityGate()
    sv = StateVersion(1, 0, 1, time.time())
    now_ns = int(time.time() * 1e9)
    ttl = int(2e9)

    env = ResourceEvidenceEnvelope(
        envelope_id="ENV-BENCH-PROOF",
        state_version=sv,
        provenance="KvmQemuAdapter",
        host_monotonic_ns=now_ns,
        canonical_normalized={
            "cpu_usage_pct": EpistemicValue(38.0, "cgroups", EvidenceKind.RUNTIME_MEASURED, 0.02, now_ns, now_ns + ttl, "cpu.usage", sv, "KvmQemuAdapter"),
            "temperature_celsius": EpistemicValue(48.0, "hwmon", EvidenceKind.HARDWARE_MEASURED, 0.01, now_ns, now_ns + ttl, "hardware.thermal", sv, "CppHardwareProbe")
        }
    )
    cand = CandidateAction(
        "ACT-PROOF-BENCH", sv, now_ns, now_ns + int(1e9), "RESIZE_BALLOON", "g1",
        ExpectedDelta({"cpu": -2.0}, 0.9, int(1e9)), ResourceVector(cpu_normalized_share=0.1),
        [env.envelope_id], ReversibilityClass.REVERSIBLE
    )

    tracemalloc.start()
    latencies: List[float] = []

    for _ in range(iterations):
        t0 = time.perf_counter()
        bundle = gate.arbitrate(cand, sv, [env, env, env], now_ns)
        _ = bundle.audit_hash
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1e6)

    current_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    stats = compute_statistics(latencies)
    stats["bytes_per_op"] = round(peak_mem / iterations, 2)
    stats["peak_rss_kb"] = round(peak_mem / 1024, 2)
    return stats


def benchmark_group_k_deterministic_replay(iterations: int = 2000) -> Dict[str, Any]:
    """Group K: Deterministic Replay Throughput"""
    log = CausalEventLog(capacity=5000)
    sv = StateVersion(1, 0, 1, time.time())
    ev = log.emit(
        event_class=CausalEventClass.AUTHORITY_VERDICT,
        state_version=sv,
        payload={"verdict_type": "COMMIT_APPROVED", "candidate_id": "ACT-BENCH"}
    )

    tracemalloc.start()
    latencies: List[float] = []

    for _ in range(iterations):
        t0 = time.perf_counter()
        res = log.replay_verdict(ev.event_id)
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1e6)

    current_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    stats = compute_statistics(latencies)
    stats["bytes_per_op"] = round(peak_mem / iterations, 2)
    stats["peak_rss_kb"] = round(peak_mem / 1024, 2)
    return stats


def run_full_benchmark_program() -> Dict[str, Any]:
    print("==============================================================================")
    print(" KRYSTAL-STACK EPISTEMIC RUNTIME BENCHMARK PROGRAM (SECTION 8)")
    print("==============================================================================")
    print(f" Invariant: VITAL_MAX_HP == {VITAL_MAX_HP}")
    print(f" Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}\n")

    results = {
        "timestamp_utc": time.time(),
        "vital_max_hp": VITAL_MAX_HP,
        "groups": {
            "A_epistemic_type_overhead": benchmark_group_a_epistemic_type(),
            "B_envelope_normalization": benchmark_group_b_envelope_normalization(),
            "C_causal_logging": benchmark_group_c_causal_logging(),
            "D_state_space_pruner": benchmark_group_d_state_space_pruner(),
            "J_proof_bundle_generation": benchmark_group_j_proof_bundle(),
            "K_deterministic_replay": benchmark_group_k_deterministic_replay()
        }
    }

    # Print Formatted Comparison Table
    print(f"{'Benchmark Group':<32} | {'Mean (us)':<10} | {'p50 (us)':<9} | {'p95 (us)':<9} | {'p99 (us)':<9} | {'CV (%)':<8} | {'Throughput':<12} | {'Bytes/Op'}")
    print("-" * 115)

    for name, m in results["groups"].items():
        print(f"{name:<32} | {m['mean_us']:<10.2f} | {m['p50_us']:<9.2f} | {m['p95_us']:<9.2f} | {m['p99_us']:<9.2f} | {m['cv_pct']:<8.1f} | {m['ops_per_sec']:<12.0f} | {m['bytes_per_op']}")

    print("\nBenchmark program complete.")
    return results


if __name__ == "__main__":
    report = run_full_benchmark_program()
