#!/usr/bin/env python3
"""
==============================================================================
KRYSTAL-STACK: KHI-E2E-003 SUSTAINED 8 kHz QUEUE STABILITY BENCHMARK
==============================================================================
Module: scripts/run_khi_e2e_003_queue_stability.py
Description:
    Evaluates whether the Krystal Software Control Path can sustain an 8 kHz
    input arrival stream (λ = 8,000 events/s, nominal interval T = 125.0 µs)
    without queue backlog growth, buffer overflow, or dropped frames.

Macro Pipeline Under Test:
    8,000 events/s (λ)
           │
           ▼
    SPSC Ring Buffer (capacity = 4,096)
           │
           ▼
    Normalize & Deadzone Filter
           │
           ▼
    Evidence Envelope Generation
           │
           ▼
    Governor Q-AB (Bitmask + Capability Snapshot Cache)
           │
           ▼
    Workload Planner (Task Graph Construction)

Core Scientific Question:
    Is λ_arrival <= μ_service sustained over large horizons (e.g. 80,000 to
    480,000 events), and does the SPSC queue remain strictly bounded with
    zero drops, even when individual p99 latency exceeds 125.0 µs?

Evaluation Modes:
    1. CLOCK_DRIVEN: Discrete-event arrival timeline where physical time
       advances by the exact CPU nanoseconds consumed by the actual Python
       execution of the pipeline stages. Eliminates OS thread scheduling
       artifacts to test pure mathematical queue stability (Little's Law).
    2. CONCURRENT_THREADED: Multi-threaded producer-consumer using SPSC ring
       to capture OS scheduling, context-switch jitter, and thread contention.

Constitutional Invariant:
    assert VITAL_MAX_HP == 6 across all events, envelopes, decisions, and plans.

Author: Dušan Kopecký & Krystal-Stack Architecture & Hardware Systems Council (2026)
==============================================================================
"""

from __future__ import annotations

import os
import sys
import time
import math
import json
import argparse
import threading
from typing import Dict, Any, List, Optional, Tuple

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from krystal_kernel.scene_constructor_presets import VITAL_MAX_HP
assert VITAL_MAX_HP == 6, f"Constitutional invariant breach: VITAL_MAX_HP must be 6, got {VITAL_MAX_HP}"

from krystal_kernel.hardware_input_hal import (
    HardwareInputHAL,
    InputSourceKind,
    HardwareInputEvent,
    FrameIntegrity,
    FrameAuthenticity,
    SPSCInputRingBuffer,
)
from krystal_kernel.hardware_input_driver import (
    HardwareInputDriver,
    NormalizedInputState,
)
from krystal_kernel.custom_governor import (
    CustomHardwareGovernor,
    BatteryBand,
    ThermalBand,
)
from krystal_kernel.custom_planner import (
    CustomWorkloadPlanner,
    WorkloadTask,
    TaskCategory,
)


def compute_distribution(values: List[float]) -> Dict[str, float]:
    """Computes standard percentiles and summary statistics."""
    if not values:
        return {"p50": 0.0, "p90": 0.0, "p95": 0.0, "p99": 0.0, "p99_9": 0.0, "max": 0.0, "mean": 0.0, "min": 0.0}
    s = sorted(values)
    n = len(s)
    return {
        "min": round(s[0], 2),
        "p50": round(s[int(n * 0.50)], 2),
        "p90": round(s[int(n * 0.90)], 2),
        "p95": round(s[int(n * 0.95)], 2),
        "p99": round(s[int(n * 0.99)], 2),
        "p99_9": round(s[min(n - 1, int(n * 0.999))], 2),
        "max": round(s[-1], 2),
        "mean": round(sum(s) / n, 2),
    }


class KhiQueueStabilityBenchmark:
    """Orchestrates sustained 8 kHz queue stability evaluations."""

    def __init__(self, ring_capacity: int = 4096, control_ratio: int = 1):
        self.hal = HardwareInputHAL()
        self.driver = HardwareInputDriver(mock_mode=True)
        self.gov_q_ab = CustomHardwareGovernor()
        self.planner = CustomWorkloadPlanner()
        self.ring = SPSCInputRingBuffer(capacity=ring_capacity)
        self.nominal_interval_ns = 125_000  # 8,000 Hz = 125.0 µs
        self.control_ratio = max(1, control_ratio)  # 1 = every event, 8 = 1 kHz control on 8 kHz input
        self._cached_decision = None
        self._cached_plan = None

    def _create_synthetic_event(self, seq: int, arrival_ns: int) -> HardwareInputEvent:
        return HardwareInputEvent(
            device_fingerprint="SIM-MOUSE-8KHZ-STABILITY",
            source_kind=InputSourceKind.RAW_MOUSE,
            sequence_number=seq,
            host_monotonic_ns=arrival_ns,
            hardware_timestamp_ns=arrival_ns,
            clock_domain_id="monotonic",
            payload_type="JSON_AXIS",
            payload=b'{"dx": 12, "dy": -8}',
            integrity_status=FrameIntegrity.VALIDATED,
            authenticity_status=FrameAuthenticity.UNAUTHENTICATED,
            provenance="KhiQueueStabilityBenchmark",
            simulated=True,
            vital_max_hp=VITAL_MAX_HP,
        )

    def run_clock_driven(
        self,
        total_events: int = 80000,
        warmup_events: int = 2000,
    ) -> Dict[str, Any]:
        """
        Executes discrete-event clock-driven queue simulation where timeline advances
        by the exact real CPU time consumed by the actual Python execution of each pipeline stage.
        """
        print(f"\n  Running CLOCK_DRIVEN Queue Stability: {total_events:,} events (warmup: {warmup_events:,})...")
        
        # Warmup execution
        for w in range(warmup_events):
            evt_w = self._create_synthetic_event(w + 1, w * self.nominal_interval_ns)
            self._process_single_event(evt_w)

        # Clear ring buffer
        self.ring.drain_all()

        t_sim_start_ns = 1_000_000_000
        current_time_ns = t_sim_start_ns
        next_arrival_idx = 0
        consumed = 0

        queue_depth_samples: List[float] = []
        transit_ages_us: List[float] = []
        stage_proc_us: List[float] = []
        queue_depth_timeline: List[Tuple[float, int]] = []  # (time_sec, depth)

        bench_wall_start = time.perf_counter()

        while (consumed + self.ring.dropped_packets) < total_events:
            # Advance scheduled arrivals up to current_time_ns
            while next_arrival_idx < total_events:
                sched_ns = t_sim_start_ns + next_arrival_idx * self.nominal_interval_ns
                if sched_ns <= current_time_ns:
                    evt = self._create_synthetic_event(next_arrival_idx + 1, sched_ns)
                    self.ring.push(evt)
                    next_arrival_idx += 1
                else:
                    break

            # Check queue
            qd = self.ring.size()
            if self.ring.is_empty():
                # Pipeline is idle waiting for next input packet; advance clock to next arrival
                if next_arrival_idx < total_events:
                    current_time_ns = t_sim_start_ns + next_arrival_idx * self.nominal_interval_ns
                    continue
                else:
                    break

            # Record queue depth snapshot
            queue_depth_samples.append(float(qd))
            if consumed % 1000 == 0:
                elapsed_sim_s = (current_time_ns - t_sim_start_ns) / 1e9
                queue_depth_timeline.append((round(elapsed_sim_s, 3), qd))

            # Pop event from SPSC ring
            popped_evt = self.ring.pop()
            if popped_evt is None:
                continue

            consumed += 1

            # Execute real pipeline stages and measure real CPU nanoseconds spent
            t0 = time.perf_counter_ns()
            self._process_single_event(popped_evt)
            t1 = time.perf_counter_ns()

            real_cpu_duration_ns = t1 - t0
            proc_us = real_cpu_duration_ns / 1000.0
            stage_proc_us.append(proc_us)

            # Advance simulated timeline by real CPU processing duration
            current_time_ns += real_cpu_duration_ns

            # Event transit age = when processing completed - when event was scheduled to arrive
            transit_age_us = (current_time_ns - popped_evt.host_monotonic_ns) / 1000.0
            transit_ages_us.append(transit_age_us)

        bench_wall_dur = time.perf_counter() - bench_wall_start
        total_sim_dur_s = (current_time_ns - t_sim_start_ns) / 1e9

        return self._build_benchmark_report(
            mode="CLOCK_DRIVEN",
            total_events=total_events,
            wall_duration_s=bench_wall_dur,
            sim_duration_s=total_sim_dur_s,
            queue_depths=queue_depth_samples,
            transit_ages_us=transit_ages_us,
            proc_latencies_us=stage_proc_us,
            timeline=queue_depth_timeline,
            drops=self.ring.dropped_packets,
        )

    def run_concurrent_threaded(
        self,
        duration_s: float = 10.0,
        warmup_s: float = 1.0,
    ) -> Dict[str, Any]:
        """
        Executes multi-threaded producer-consumer on Windows.
        Producer thread feeds SPSC ring buffer at 8 kHz; Consumer thread drains
        and executes the entire software pipeline.
        """
        print(f"\n  Running CONCURRENT_THREADED Queue Stability: {duration_s:.1f}s (warmup: {warmup_s:.1f}s)...")
        
        self.ring.drain_all()
        stop_signal = threading.Event()
        start_bench_signal = threading.Event()

        produced_count = 0
        consumed_count = 0

        queue_depth_samples: List[float] = []
        transit_ages_us: List[float] = []
        stage_proc_us: List[float] = []
        queue_depth_timeline: List[Tuple[float, int]] = []

        def producer_worker():
            nonlocal produced_count
            seq = 0
            t_start = time.perf_counter_ns()
            while not stop_signal.is_set():
                target_ns = t_start + seq * self.nominal_interval_ns
                while time.perf_counter_ns() < target_ns:
                    if stop_signal.is_set():
                        return
                    # micro yield if ahead of schedule
                    time.sleep(0)
                seq += 1
                now_ns = time.perf_counter_ns()
                evt = self._create_synthetic_event(seq, now_ns)
                self.ring.push(evt)
                if start_bench_signal.is_set():
                    produced_count += 1

        def consumer_worker():
            nonlocal consumed_count
            while not stop_signal.is_set() or not self.ring.is_empty():
                qd = self.ring.size()
                evt = self.ring.pop()
                if evt is None:
                    time.sleep(0.00005)  # 50 µs sleep when empty
                    continue

                if start_bench_signal.is_set():
                    queue_depth_samples.append(float(qd))
                    if consumed_count % 1000 == 0:
                        queue_depth_timeline.append((round(time.perf_counter(), 3), qd))

                t0 = time.perf_counter_ns()
                self._process_single_event(evt)
                t1 = time.perf_counter_ns()

                if start_bench_signal.is_set():
                    proc_us = (t1 - t0) / 1000.0
                    stage_proc_us.append(proc_us)
                    transit_age_us = (t1 - evt.host_monotonic_ns) / 1000.0
                    transit_ages_us.append(transit_age_us)
                    consumed_count += 1

        t_prod = threading.Thread(target=producer_worker, name="Khi8kHzProducer")
        t_cons = threading.Thread(target=consumer_worker, name="Khi8kHzConsumer")

        t_cons.start()
        t_prod.start()

        # Warmup phase
        time.sleep(warmup_s)
        start_bench_signal.set()
        bench_start = time.perf_counter()

        # Measurement duration
        time.sleep(duration_s)
        bench_dur = time.perf_counter() - bench_start
        stop_signal.set()

        t_prod.join(timeout=3.0)
        t_cons.join(timeout=3.0)

        return self._build_benchmark_report(
            mode="CONCURRENT_THREADED",
            total_events=consumed_count,
            wall_duration_s=bench_dur,
            sim_duration_s=bench_dur,
            queue_depths=queue_depth_samples,
            transit_ages_us=transit_ages_us,
            proc_latencies_us=stage_proc_us,
            timeline=queue_depth_timeline,
            drops=self.ring.dropped_packets,
            produced_count=produced_count,
        )

    def _process_single_event(self, evt: HardwareInputEvent) -> None:
        """Executes the macro control path: Normalize -> Evidence -> Governor Q-AB -> Planner."""
        # 1. Normalize
        flx, fly, _ = self.driver.filter.filter_radial_stick(0.5, 0.5)
        _ = NormalizedInputState(
            device_id=evt.device_fingerprint,
            source_kind=evt.source_kind,
            timestamp_ns=evt.host_monotonic_ns,
            sequence_id=evt.sequence_number,
            axes={"lx": flx, "ly": fly},
            vital_max_hp=VITAL_MAX_HP,
        )

        # 2. Evidence Envelope
        _ = self.hal.create_evidence_envelope(
            event=evt,
            normalized_state_ref=f"state://norm/{evt.sequence_number}",
            calibration_profile_id="CAL-MOUSE-v3.0.0",
            state_version=evt.sequence_number,
            uncertainty_us=0.5,
        )

        # 3. Governor Q-AB & Planner (evaluated at control_ratio rate)
        needs_replan = (self._cached_decision is None) or (evt.sequence_number % self.control_ratio == 0)

        if needs_replan:
            decision = self.gov_q_ab.evaluate_telemetry(
                battery_pct=85.0,
                plugged=True,
                temperature_c=65.0,
                cpu_utilization_pct=35.0,
                uma_pressure=0.20,
                now_ns=evt.host_monotonic_ns,
            )
            assert decision.vital_max_hp == 6
            self._cached_decision = decision

            # 4. Planner
            task = WorkloadTask(
                task_id=f"macro_act_{evt.sequence_number}",
                name="Tactical Action",
                category=TaskCategory.COMPUTE,
                estimated_cost_ms=1.2,
                preferred_backend="CPU_SIMD_DETERMINISTIC",
            )
            plan = self.planner.generate_plan(tasks=[task], envelope=decision.envelope)
            assert plan.vital_max_hp == 6
            self._cached_plan = plan
        else:
            decision = self._cached_decision
            plan = self._cached_plan

    def _build_benchmark_report(
        self,
        mode: str,
        total_events: int,
        wall_duration_s: float,
        sim_duration_s: float,
        queue_depths: List[float],
        transit_ages_us: List[float],
        proc_latencies_us: List[float],
        timeline: List[Tuple[float, int]],
        drops: int,
        produced_count: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Constructs comprehensive statistical and queue stability report."""
        effective_events = len(transit_ages_us)
        arrival_rate_hz = (produced_count or total_events) / max(0.001, sim_duration_s)
        service_rate_hz = effective_events / max(0.001, sim_duration_s)

        mean_proc_us = (sum(proc_latencies_us) / len(proc_latencies_us)) if proc_latencies_us else 100.0
        service_capacity_hz = 1_000_000.0 / mean_proc_us if mean_proc_us > 0 else 10000.0
        traffic_intensity_rho = 8000.0 / service_capacity_hz

        qd_dist = compute_distribution(queue_depths)
        age_dist = compute_distribution(transit_ages_us)
        proc_dist = compute_distribution(proc_latencies_us)

        # Backlog growth slope (linear regression over queue depth samples)
        n = len(queue_depths)
        if n > 10:
            x_vals = range(n)
            x_mean = (n - 1) / 2.0
            y_mean = sum(queue_depths) / n
            num = sum((i - x_mean) * (y - y_mean) for i, y in zip(x_vals, queue_depths))
            den = sum((i - x_mean) ** 2 for i in x_vals)
            backlog_slope = num / den if den != 0 else 0.0
        else:
            backlog_slope = 0.0

        # Deadline miss counts
        misses_125 = sum(1 for a in transit_ages_us if a > 125.0)
        misses_250 = sum(1 for a in transit_ages_us if a > 250.0)
        misses_500 = sum(1 for a in transit_ages_us if a > 500.0)

        # Stability Verdict
        is_bounded = qd_dist["max"] < self.ring.capacity * 0.90 and abs(backlog_slope) < 0.001
        is_lossless = drops == 0
        rho_stable = traffic_intensity_rho < 1.0

        if is_lossless and is_bounded and rho_stable:
            verdict = "STABLE_SUSTAINED"
            verdict_desc = f"Queue is bounded (max={qd_dist['max']}, slope={backlog_slope:.6f}), drops=0, rho={traffic_intensity_rho:.3f} < 1.0. 8 kHz throughput is fully sustainable."
        elif is_lossless and rho_stable:
            verdict = "STABLE_BOUNDED_WITH_BURSTS"
            verdict_desc = f"Queue fluctuates (max={qd_dist['max']}) but drops=0 and mean service rate ({service_capacity_hz:.0f} Hz) exceeds arrival rate."
        else:
            verdict = "UNSTABLE_OVERFLOW"
            verdict_desc = f"Queue overflow or drops detected ({drops} drops, rho={traffic_intensity_rho:.3f})."

        return {
            "benchmark_id": "KHI-E2E-003",
            "mode": mode,
            "control_ratio": self.control_ratio,
            "control_rate_hz": round(8000.0 / self.control_ratio, 1),
            "target_rate_hz": 8000.0,
            "nominal_interval_us": 125.0,
            "sample_count": total_events,
            "wall_duration_s": round(wall_duration_s, 2),
            "sim_duration_s": round(sim_duration_s, 2),
            "arrival_rate_hz": round(arrival_rate_hz, 2),
            "completion_rate_hz": round(service_rate_hz, 2),
            "service_capacity_hz": round(service_capacity_hz, 2),
            "traffic_intensity_rho": round(traffic_intensity_rho, 4),
            "ring_capacity": self.ring.capacity,
            "drops": drops,
            "backlog_slope": round(backlog_slope, 8),
            "queue_depth": qd_dist,
            "transit_age_us": age_dist,
            "processing_latency_us": proc_dist,
            "deadline_misses": {
                "gt_125_us": misses_125,
                "gt_125_pct": round(misses_125 / max(1, effective_events) * 100.0, 2),
                "gt_250_us": misses_250,
                "gt_250_pct": round(misses_250 / max(1, effective_events) * 100.0, 2),
                "gt_500_us": misses_500,
                "gt_500_pct": round(misses_500 / max(1, effective_events) * 100.0, 2),
            },
            "stability_verdict": verdict,
            "verdict_description": verdict_desc,
            "vital_max_hp": VITAL_MAX_HP,
        }


def print_report(res: Dict[str, Any]) -> None:
    print("\n" + "=" * 80)
    print(f" KRYSTAL-STACK // KHI-E2E-003 QUEUE STABILITY REPORT ({res['mode']})")
    print("=" * 80)
    print(f"  Target Input Rate (λ):   {res['target_rate_hz']:,.1f} Hz (T = {res['nominal_interval_us']} µs)")
    print(f"  Control Ratio:           1:{res['control_ratio']} ({res['control_rate_hz']:,.1f} Hz Governor/Planner ticks)")
    print(f"  Service Capacity (μ):    {res['service_capacity_hz']:,.1f} Hz")
    print(f"  Traffic Intensity (ρ):   {res['traffic_intensity_rho']:.4f} (λ / μ {'< 1.0 [STABLE]' if res['traffic_intensity_rho'] < 1.0 else '>= 1.0 [UNSTABLE]'})")
    print(f"  Processed Events:        {res['sample_count']:,} events in {res['sim_duration_s']:.2f}s")
    print(f"  Ring Buffer Capacity:    {res['ring_capacity']} slots")
    print(f"  Total Drops:             {res['drops']} drops")
    print(f"  Backlog Growth Slope:    {res['backlog_slope']:.8f} slots/event")
    print("-" * 80)
    print("QUEUE DEPTH DISTRIBUTION:")
    qd = res["queue_depth"]
    print(f"  p50: {qd['p50']:>6.1f} | p90: {qd['p90']:>6.1f} | p95: {qd['p95']:>6.1f} | p99: {qd['p99']:>6.1f} | p99.9: {qd['p99_9']:>6.1f} | max: {qd['max']:>6.1f}")
    print("-" * 80)
    print("EVENT TRANSIT AGE (ENQUEUE TO PLAN COMPLETION) (µs):")
    ta = res["transit_age_us"]
    print(f"  p50: {ta['p50']:>6.1f} µs | p90: {ta['p90']:>6.1f} µs | p95: {ta['p95']:>6.1f} µs | p99: {ta['p99']:>6.1f} µs | p99.9: {ta['p99_9']:>6.1f} µs | max: {ta['max']:>6.1f} µs")
    print("-" * 80)
    print("RAW SOFTWARE PROCESSING TIME (µs):")
    pr = res["processing_latency_us"]
    print(f"  p50: {pr['p50']:>6.1f} µs | p90: {pr['p90']:>6.1f} µs | p95: {pr['p95']:>6.1f} µs | p99: {pr['p99']:>6.1f} µs | p99.9: {pr['p99_9']:>6.1f} µs | max: {pr['max']:>6.1f} µs")
    print("-" * 80)
    print("DEADLINE MISS ANALYSIS:")
    dm = res["deadline_misses"]
    print(f"  > 125 µs (1 HW period):   {dm['gt_125_us']:>7,} misses ({dm['gt_125_pct']:>5.2f} %)")
    print(f"  > 250 µs (2 HW periods):  {dm['gt_250_us']:>7,} misses ({dm['gt_250_pct']:>5.2f} %)")
    print(f"  > 500 µs (H3 Target):     {dm['gt_500_us']:>7,} misses ({dm['gt_500_pct']:>5.2f} %)")
    print("-" * 80)
    print(f"STABILITY VERDICT: [{res['stability_verdict']}]")
    print(f"  {res['verdict_description']}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="KHI-E2E-003 Sustained 8 kHz Queue Stability Benchmark")
    parser.add_argument("--mode", choices=["clock_driven", "concurrent_threaded", "both"], default="clock_driven", help="Evaluation mode (default: clock_driven)")
    parser.add_argument("--events", type=int, default=80000, help="Total events for clock-driven mode (default: 80,000 = 10s)")
    parser.add_argument("--duration", type=float, default=10.0, help="Duration in seconds for concurrent mode (default: 10.0s)")
    parser.add_argument("--control-ratio", type=int, default=1, help="Ratio of input events per Governor/Planner tick (default: 1 = every event, 8 = 1 kHz control)")
    parser.add_argument("--ring-capacity", type=int, default=4096, help="SPSC ring buffer capacity (default: 4096)")
    parser.add_argument("--json-out", type=str, default=None, help="Optional path to write JSON output")
    args = parser.parse_args()

    bench = KhiQueueStabilityBenchmark(ring_capacity=args.ring_capacity, control_ratio=args.control_ratio)
    results = {}

    if args.mode in ("clock_driven", "both"):
        res_clock = bench.run_clock_driven(total_events=args.events)
        print_report(res_clock)
        results["clock_driven"] = res_clock

    if args.mode in ("concurrent_threaded", "both"):
        res_thread = bench.run_concurrent_threaded(duration_s=args.duration)
        print_report(res_thread)
        results["concurrent_threaded"] = res_thread

    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"  Results saved to {args.json_out}")
