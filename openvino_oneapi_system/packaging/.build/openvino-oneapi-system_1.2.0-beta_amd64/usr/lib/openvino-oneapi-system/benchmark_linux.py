import argparse
import json
import os
import re
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from typing import Dict, List, Optional, Tuple

from ovo.algorithmic_models import EconomicUtilityModel, UcbBanditPlanner
from ovo.inference import OpenVinoInferenceEngine
from ovo.logging_system import JsonLogger
from ovo.runtime_policy import RuntimePolicyApplier
from ovo.telemetry import TelemetryCollector


Action = Tuple[int, int]


@dataclass
class ScenarioStats:
    name: str
    mean_latency_ms: float
    p95_latency_ms: float
    mean_throughput: float
    mean_utility: float
    actions: List[Action]


def percentile(values: List[float], p: float) -> float:
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    idx = int((len(sorted_vals) - 1) * p)
    return sorted_vals[idx]


def run_scenario(
    name: str,
    cycles: int,
    fixed_action: Optional[Action],
    planner: Optional[UcbBanditPlanner],
    logger: JsonLogger,
) -> ScenarioStats:
    telemetry = TelemetryCollector()
    engine = OpenVinoInferenceEngine()
    policy = RuntimePolicyApplier()
    model = EconomicUtilityModel()

    latencies: List[float] = []
    throughputs: List[float] = []
    utilities: List[float] = []
    chosen_actions: List[Action] = []

    for i in range(cycles):
        snapshot = telemetry.sample()
        action = fixed_action if fixed_action is not None else planner.select()  # type: ignore[union-attr]
        threads, streams = action
        policy.apply(threads=threads, streams=streams, mode=name)
        result = engine.run(threads=threads, streams=streams)
        utility = model.score(result, snapshot, action)

        if planner is not None:
            planner.update(action, utility)

        latencies.append(result.latency_ms)
        throughputs.append(result.throughput)
        utilities.append(utility)
        chosen_actions.append(action)

        logger.event(
            "benchmark_cycle",
            {
                "scenario": name,
                "cycle": i,
                "threads": threads,
                "streams": streams,
                "latency_ms": result.latency_ms,
                "throughput": result.throughput,
                "utility": utility,
                "backend": result.backend,
            },
        )

    return ScenarioStats(
        name=name,
        mean_latency_ms=mean(latencies),
        p95_latency_ms=percentile(latencies, 0.95),
        mean_throughput=mean(throughputs),
        mean_utility=mean(utilities),
        actions=chosen_actions,
    )


def parse_sysbench_events_per_sec(text: str) -> Optional[float]:
    match = re.search(r"events per second:\s+([0-9.]+)", text)
    if not match:
        return None
    return float(match.group(1))


def run_sysbench(threads: int, max_prime: int) -> Optional[float]:
    try:
        proc = subprocess.run(
            ["sysbench", "cpu", f"--threads={threads}", f"--cpu-max-prime={max_prime}", "run"],
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        return None

    if proc.returncode != 0:
        return None

    return parse_sysbench_events_per_sec(proc.stdout)


def format_ascii_report(
    baseline: ScenarioStats,
    adaptive: ScenarioStats,
    sysbench_baseline: Optional[float],
    sysbench_adaptive: Optional[float],
) -> str:
    latency_gain = ((baseline.mean_latency_ms - adaptive.mean_latency_ms) / baseline.mean_latency_ms) * 100.0
    throughput_gain = ((adaptive.mean_throughput - baseline.mean_throughput) / baseline.mean_throughput) * 100.0
    utility_gain = ((adaptive.mean_utility - baseline.mean_utility) / abs(baseline.mean_utility or 1e-9)) * 100.0
    lines = [
        "=== LINUX PERFORMANCE REPORT (ASCII) ===",
        "",
        f"Baseline  mean latency: {baseline.mean_latency_ms:.4f} ms",
        f"Adaptive  mean latency: {adaptive.mean_latency_ms:.4f} ms",
        f"Latency improvement:    {latency_gain:.2f}%",
        "",
        f"Baseline  throughput:   {baseline.mean_throughput:.2f}",
        f"Adaptive  throughput:   {adaptive.mean_throughput:.2f}",
        f"Throughput improvement: {throughput_gain:.2f}%",
        "",
        f"Baseline  utility:      {baseline.mean_utility:.4f}",
        f"Adaptive  utility:      {adaptive.mean_utility:.4f}",
        f"Utility improvement:    {utility_gain:.2f}%",
    ]

    if sysbench_baseline is not None and sysbench_adaptive is not None:
        sysbench_gain = ((sysbench_adaptive - sysbench_baseline) / sysbench_baseline) * 100.0
        lines.extend(
            [
                "",
                f"Sysbench baseline events/s: {sysbench_baseline:.2f}",
                f"Sysbench adaptive events/s: {sysbench_adaptive:.2f}",
                f"Sysbench improvement:       {sysbench_gain:.2f}%",
            ]
        )
    else:
        lines.extend(["", "Sysbench: not available in current environment"])

    lines.extend(
        [
            "",
            "Action samples:",
            f"Baseline first 10 actions: {baseline.actions[:10]}",
            f"Adaptive first 10 actions: {adaptive.actions[:10]}",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Linux benchmark for OpenVINO ONE API performance system")
    parser.add_argument("--cycles", type=int, default=50, help="Cycles per scenario")
    parser.add_argument("--sysbench-max-prime", type=int, default=20000, help="sysbench cpu-max-prime")
    args = parser.parse_args()

    log_dir = Path(os.getenv("OVO_LOG_DIR", "openvino_oneapi_system/logs"))
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = JsonLogger(log_path=str(log_dir / "benchmark_runtime.jsonl"))

    baseline_action: Action = (2, 1)
    adaptive_actions: List[Action] = [
        (2, 1),
        (4, 1),
        (4, 2),
        (8, 2),
        (8, 3),
        (12, 3),
        (16, 4),
        (24, 6),
    ]

    baseline = run_scenario(
        name="baseline",
        cycles=args.cycles,
        fixed_action=baseline_action,
        planner=None,
        logger=logger,
    )
    adaptive = run_scenario(
        name="adaptive",
        cycles=args.cycles,
        fixed_action=None,
        planner=UcbBanditPlanner(actions=adaptive_actions),
        logger=logger,
    )

    best_threads = max(adaptive.actions, key=lambda a: a[0])[0] if adaptive.actions else baseline_action[0]
    sysbench_baseline = run_sysbench(threads=baseline_action[0], max_prime=args.sysbench_max_prime)
    sysbench_adaptive = run_sysbench(threads=best_threads, max_prime=args.sysbench_max_prime)

    report = format_ascii_report(baseline, adaptive, sysbench_baseline, sysbench_adaptive)
    print(report)

    result_json = {
        "baseline": asdict(baseline),
        "adaptive": asdict(adaptive),
        "sysbench_baseline_events_per_sec": sysbench_baseline,
        "sysbench_adaptive_events_per_sec": sysbench_adaptive,
        "note": "If sysbench metrics are null, sysbench is unavailable.",
    }

    (log_dir / "benchmark_latest.json").write_text(json.dumps(result_json, indent=2), encoding="utf-8")
    (log_dir / "benchmark_latest.txt").write_text(report + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
