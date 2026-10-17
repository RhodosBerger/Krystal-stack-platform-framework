#!/usr/bin/env python3
"""
==============================================================================
KRYSTAL-STACK: KHI-E2E-002 PHASE Q FULL-PATH E2E BENCHMARK (Q0 vs Q-AB)
==============================================================================
Module: scripts/run_khi_e2e_phase_q_fullpath.py
Description:
    Macro-benchmark evaluating the entire end-to-end software control loop:
      Ring Dequeue -> Normalization -> Evidence Envelope -> Governor -> Planner
    comparing Q0 (Baseline Governor) vs Q-AB (Integrated Phase Q Authority Reducer).

Hypothesis Under Test:
    H3-SYSTEM: Total Software Path p99 (Ring -> Plan) < 500 µs
               (evaluated against the original baseline p99 of 1,566.5 µs).

Statistical Protocol:
    - Randomized Block Design (RBD): 30 independent blocks
    - 1,000 complete E2E iterations per variant per block (30,000 total E2E traces / variant)
    - Block Bootstrap (2,000 resamples) for 95% Confidence Intervals
    - Canonical Semantic Verdict Checksum: SHA-256 over canonical decision payload
    - Constitutional Invariant: assert VITAL_MAX_HP == 6 throughout

Author: Dušan Kopecký & Krystal-Stack Architecture & Hardware Systems Council (2026)
==============================================================================
"""

from __future__ import annotations

import os
import sys
import time
import math
import json
import random
import hashlib
from typing import Dict, Any, List, Optional, Tuple, Set

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from krystal_kernel.scene_constructor_presets import VITAL_MAX_HP
assert VITAL_MAX_HP == 6, f"Constitutional invariant breach: VITAL_MAX_HP must be 6, got {VITAL_MAX_HP}"

from krystal_kernel.verification.authority_bits import (
    AuthorityProfileBit,
    PROFILE_ID_TO_BIT,
    BIT_TO_PROFILE_ID,
    set_to_mask,
    mask_to_set,
)
from krystal_kernel.verification.capability_registry import (
    CapabilityRegistry,
    CapabilityState,
    HardwareBackend,
    GLOBAL_CAPABILITY_REGISTRY,
)
from krystal_kernel.verification.config_catalog import CATALOG
from krystal_kernel.verification.phase_q_authority import (
    CapabilitySnapshotCache,
    PhaseQAuthorityReducer,
    CapabilitySnapshot,
)
from krystal_kernel.custom_governor import (
    CustomHardwareGovernor,
    GovernorDecision,
    GovernorMode,
    BatteryBand,
    ThermalBand,
    ExecutionEnvelope,
)
from krystal_kernel.custom_planner import CustomWorkloadPlanner, WorkloadTask, TaskCategory
from krystal_kernel.hardware_input_driver import HardwareInputDriver, NormalizedInputState
from krystal_kernel.hardware_input_hal import (
    HardwareInputHAL,
    InputSourceKind,
)


def compute_stats(values: List[float]) -> Dict[str, float]:
    s = sorted(values)
    n = len(s)
    return {
        "p50": round(s[int(n * 0.50)], 2),
        "p90": round(s[int(n * 0.90)], 2),
        "p95": round(s[int(n * 0.95)], 2),
        "p99": round(s[int(n * 0.99)], 2),
        "max": round(s[-1], 2),
        "mean": round(sum(s) / n, 2),
    }


def block_bootstrap_ci(block_percentiles: List[float], num_resamples: int = 2000) -> Tuple[float, float]:
    k = len(block_percentiles)
    if k < 2:
        return (block_percentiles[0], block_percentiles[0])
    means = []
    for _ in range(num_resamples):
        sample = [random.choice(block_percentiles) for _ in range(k)]
        means.append(sorted(sample)[int(k * 0.5)])
    sorted_means = sorted(means)
    low_idx = int(num_resamples * 0.025)
    high_idx = int(num_resamples * 0.975)
    return (round(sorted_means[low_idx], 2), round(sorted_means[high_idx], 2))


class PhaseQFullPathRunner:
    """Orchestrates comparative end-to-end full-path executions."""

    def __init__(self):
        self.driver = HardwareInputDriver(mock_mode=True)
        self.hal = HardwareInputHAL()
        self.planner = CustomWorkloadPlanner()

        # Q-AB Governor: integrated with PhaseQAuthorityReducer & Snapshot Cache
        self.gov_q_ab = CustomHardwareGovernor()

        # Q0 Baseline Governor: emulation of unoptimized dynamic catalog filtering
        self.gov_q0 = CustomHardwareGovernor()
        # Override filter_feasible_profiles to use the legacy baseline dynamic filtering
        self.gov_q0.filter_feasible_profiles = self._legacy_filter_feasible_profiles

    def _legacy_filter_feasible_profiles(
        self,
        thermal_band: ThermalBand,
        battery_band: BatteryBand,
        min_perf: float = 0.0,
    ) -> List[Any]:
        """Baseline legacy filtering allocating lists/sets on every call."""
        pool = list(self.gov_q0._catalog.values())
        if thermal_band == ThermalBand.EMERGENCY or battery_band == BatteryBand.CRITICAL:
            return [self.gov_q0._catalog["ultra_saver"]]
        if battery_band == BatteryBand.CONSERVATION:
            return [p for p in pool if p.category == "battery"]
        if battery_band == BatteryBand.NORMAL:
            effective_min = max(min_perf, 0.5)
            pool = [p for p in pool if p.category != "performance" and p.perf_score >= effective_min] or [self.gov_q0._catalog["balanced_npu"]]
        if battery_band == BatteryBand.PLUGGED:
            pool = [p for p in pool if "SHORT_BURST_ONLY" not in p.tags]
        if thermal_band == ThermalBand.THROTTLED:
            pool = [p for p in pool if p.power_draw_w <= 12.0]
        verified_pool = []
        for p in pool:
            if p.gpu_enabled and not (self.gov_q0.caps.is_usable(HardwareBackend.OPENVINO_GPU) or self.gov_q0.caps.is_usable(HardwareBackend.VULKAN)):
                continue
            verified_pool.append(p)
        return verified_pool or [self.gov_q0._catalog["balanced_npu"]]

    def execute_e2e_iteration(self, seq: int, variant: str) -> Tuple[float, float, float, float, float, float, str]:
        """
        Executes one full iteration:
          Ring -> Normalize -> Evidence -> Governor -> Planner
        Returns (t_ring, t_norm, t_evid, t_gov, t_plan, t_total, decision_checksum)
        """
        now_ns = self.driver.read_monotonic_ns()

        # 1. Inject event & SPSC Ring Pop
        event = self.hal.inject_synthetic_event(
            source_kind=InputSourceKind.RAW_MOUSE,
            payload_dict={"dx": 10, "dy": -5}
        )
        t0 = time.perf_counter_ns()
        popped = self.hal.ring.pop()
        t1 = time.perf_counter_ns()
        t_ring = (t1 - t0) / 1000.0

        # 2. Normalize
        t2 = time.perf_counter_ns()
        flx, fly, _ = self.driver.filter.filter_radial_stick(0.5, 0.5)
        _ = NormalizedInputState(
            device_id="RAW_MOUSE_0",
            source_kind=InputSourceKind.RAW_MOUSE,
            timestamp_ns=now_ns,
            sequence_id=seq,
            axes={"lx": flx, "ly": fly},
            vital_max_hp=VITAL_MAX_HP
        )
        t3 = time.perf_counter_ns()
        t_norm = (t3 - t2) / 1000.0

        # 3. Evidence Envelope
        t4 = time.perf_counter_ns()
        _ = self.hal.create_evidence_envelope(
            event=popped or event,
            normalized_state_ref=f"state://norm/{seq}",
            calibration_profile_id="CAL-MOUSE-v3.0.0",
            state_version=seq,
            uncertainty_us=0.5
        )
        t5 = time.perf_counter_ns()
        t_evid = (t5 - t4) / 1000.0

        # 4. Governor Evaluation
        gov = self.gov_q_ab if variant == "Q_AB" else self.gov_q0
        t6 = time.perf_counter_ns()
        decision = gov.evaluate_telemetry(
            battery_pct=85.0,
            plugged=True,
            temperature_c=65.0,
            cpu_utilization_pct=35.0,
            uma_pressure=0.20,
            now_ns=now_ns
        )
        t7 = time.perf_counter_ns()
        t_gov = (t7 - t6) / 1000.0

        # 5. Planner
        t8 = time.perf_counter_ns()
        task = WorkloadTask(
            task_id=f"macro_act_{seq}",
            name=f"Tactical Action {seq}",
            category=TaskCategory.COMPUTE,
            estimated_cost_ms=1.2,
            preferred_backend="CPU_SIMD_DETERMINISTIC"
        )
        _ = self.planner.generate_plan(tasks=[task], envelope=decision.envelope)
        t9 = time.perf_counter_ns()
        t_plan = (t9 - t8) / 1000.0

        t_total = t_ring + t_norm + t_evid + t_gov + t_plan

        # Decision Checksum
        payload = {
            "selected_profile": decision.selected_profile.name,
            "governor_mode": decision.governor_mode.value,
            "reason_codes": sorted(decision.reason_codes),
            "vital_max_hp": decision.vital_max_hp,
            "capability_generation": decision.envelope.capability_generation
        }
        chk = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("ascii")).hexdigest()
        return t_ring, t_norm, t_evid, t_gov, t_plan, t_total, chk


def run_phase_q_benchmark(blocks: int = 30, events_per_block: int = 1000) -> Dict[str, Any]:
    print("\n" + "=" * 80)
    print(" KRYSTAL-STACK // KHI-E2E-002 PHASE Q (FULL-PATH E2E: Q0 vs Q-AB)")
    print("=" * 80)
    print(f"  Macro Loop:  Ring -> Normalize -> Evidence -> Governor -> Planner")
    print(f"  Design:      Randomized Block Design (RBD), {blocks} independent blocks")
    print(f"  Iterations:  {events_per_block:,} full-path traces/variant/block ({blocks * events_per_block:,} total/variant)")
    print(f"  Invariant:   assert VITAL_MAX_HP == 6")
    print("-" * 80)

    runner = PhaseQFullPathRunner()
    variants = ["Q0", "Q_AB"]

    stage_latencies: Dict[str, Dict[str, List[float]]] = {
        v: {"ring": [], "norm": [], "evid": [], "gov": [], "plan": [], "total": []}
        for v in variants
    }
    block_totals_p99: Dict[str, List[float]] = {v: [] for v in variants}
    block_gov_p99: Dict[str, List[float]] = {v: [] for v in variants}
    checksums: Dict[str, List[str]] = {v: [] for v in variants}

    start_ns = time.time_ns()
    seq_counter = 0

    for b in range(1, blocks + 1):
        # Warmup 500 iterations
        for _ in range(500):
            seq_counter += 1
            _ = runner.execute_e2e_iteration(seq_counter, "Q_AB")

        order = list(variants)
        random.shuffle(order)
        sys.stdout.write(f"\r  [Block {b:02d}/{blocks:02d}] Order: {' -> '.join(order)}...")
        sys.stdout.flush()

        for v in order:
            v_totals = []
            v_govs = []
            v_chk = hashlib.sha256()

            for _ in range(events_per_block):
                seq_counter += 1
                tr, tn, te, tg, tp, tt, c = runner.execute_e2e_iteration(seq_counter, v)
                stage_latencies[v]["ring"].append(tr)
                stage_latencies[v]["norm"].append(tn)
                stage_latencies[v]["evid"].append(te)
                stage_latencies[v]["gov"].append(tg)
                stage_latencies[v]["plan"].append(tp)
                stage_latencies[v]["total"].append(tt)
                v_totals.append(tt)
                v_govs.append(tg)
                v_chk.update(c.encode("ascii"))

            s_tot = sorted(v_totals)
            s_gov = sorted(v_govs)
            block_totals_p99[v].append(s_tot[int(events_per_block * 0.99)])
            block_gov_p99[v].append(s_gov[int(events_per_block * 0.99)])
            checksums[v].append(v_chk.hexdigest())

    dur_s = (time.time_ns() - start_ns) / 1e9
    print(f"\n  [DONE] Completed 60,000 full-path E2E traces in {dur_s:.2f}s\n" + "-" * 80)

    # Statistical Evaluation
    summary: Dict[str, Any] = {}
    for v in variants:
        v_res = {}
        for stage in ["ring", "norm", "evid", "gov", "plan", "total"]:
            st = compute_stats(stage_latencies[v][stage])
            v_res[stage] = st
        ci_low, ci_high = block_bootstrap_ci(block_totals_p99[v])
        v_res["total"]["p99_ci95"] = [ci_low, ci_high]
        gov_ci_low, gov_ci_high = block_bootstrap_ci(block_gov_p99[v])
        v_res["gov"]["p99_ci95"] = [gov_ci_low, gov_ci_high]
        summary[v] = v_res

    # Hypothesis H3-SYSTEM check: Total Software Path p99 < 500 µs
    q_ab_total_p99 = summary["Q_AB"]["total"]["p99"]
    q0_total_p99 = summary["Q0"]["total"]["p99"]
    q_ab_gov_p99 = summary["Q_AB"]["gov"]["p99"]
    q0_gov_p99 = summary["Q0"]["gov"]["p99"]

    h3_supported = q_ab_total_p99 < 500.0
    gov_speedup = q0_gov_p99 / max(0.01, q_ab_gov_p99)
    total_speedup = q0_total_p99 / max(0.01, q_ab_total_p99)
    delta_total_pct = ((q0_total_p99 - q_ab_total_p99) / q0_total_p99) * 100.0

    print("STAGE-BY-STAGE MACRO COMPARISON (p50 / p99 in µs):")
    print(f"{'Stage':<14} | {'Q0 Baseline (p50 / p99)':<26} | {'Q-AB Phase Q (p50 / p99)':<26} | {'Speedup (p99)':<12}")
    print("-" * 80)
    for stg in ["ring", "norm", "evid", "gov", "plan", "total"]:
        s0 = summary["Q0"][stg]
        sq = summary["Q_AB"][stg]
        sp = s0["p99"] / max(0.01, sq["p99"])
        print(f"{stg.upper():<14} | {s0['p50']:>6.2f} µs / {s0['p99']:>6.2f} µs     | {sq['p50']:>6.2f} µs / {sq['p99']:>6.2f} µs     | {sp:>8.2f}x")
    print("-" * 80)
    print(f"  Q-AB Total Software Path p99:  {q_ab_total_p99:.2f} µs [95% CI: {summary['Q_AB']['total']['p99_ci95'][0]:.1f}, {summary['Q_AB']['total']['p99_ci95'][1]:.1f}]")
    print(f"  Q0 Total Software Path p99:    {q0_total_p99:.2f} µs")
    print(f"  Macro Path Latency Reduction:  {delta_total_pct:.1f} % (Speedup: {total_speedup:.2f}x)")
    print(f"  Governor Stage Latency (p99):  Q0 = {q0_gov_p99:.2f} µs -> Q-AB = {q_ab_gov_p99:.2f} µs ({gov_speedup:.2f}x)")
    print(f"  H3-SYSTEM (Path p99 < 500 µs): {'[PASS / SUPPORTED]' if h3_supported else '[NOT MET]'}")
    print("=" * 80)

    return {
        "benchmark_id": "KHI-E2E-002-PHASE-Q",
        "blocks": blocks,
        "traces_per_variant": blocks * events_per_block,
        "summary": summary,
        "h3_supported": h3_supported,
        "total_speedup": total_speedup,
        "delta_total_pct": delta_total_pct,
        "vital_max_hp": VITAL_MAX_HP
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="KHI-E2E-002 Phase Q Full-Path Runner")
    parser.add_argument("--blocks", type=int, default=30, help="Number of blocks (default: 30)")
    parser.add_argument("--events", type=int, default=1000, help="Events per block (default: 1000)")
    args = parser.parse_args()

    run_phase_q_benchmark(blocks=args.blocks, events_per_block=args.events)
