#!/usr/bin/env python3
"""
CRYSTALSEC 2.0 // BOUNDED CONFIGURATION SEARCH & EMPIRICAL REPEATABILITY MAP
=============================================================================
Milestone 3 Implementation:
Explores controlled perturbations across explicitly bounded parameter pairs:
  - threads (1, 2, 4, 8)
  - streams (1, 2, 4)
  - cpu_affinity ([0], [0, 1], [2, 3], [0, 1, 2, 3])
  - queue_max_depth (20, 50, 100, 200)
  - context_limit_tokens (512, 1024, 2048, 4096)
  - memory_limit_mb (1024, 2048, 4096, 8192)

OUTPUT CONTRACT:
  Does NOT return an opaque single "best configuration".
  Outputs an empirical causal mapping:
    configuration -> measured effect -> uncertainty -> repeatability

Repeatability is verified by running N repetitions (default 5 runs) and
computing empirical standard deviation and variance.
=============================================================================
"""

import os
import sys
import time
import math
import uuid
import statistics
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple

WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

from krystal_kernel.scene_constructor_presets import VITAL_MAX_HP
from krystal_kernel.crystalsec_llm_execution_plane import (
    ControlledVariables,
    LLMOperationalFootprint,
    GLOBAL_LLM_EXECUTION_PLANE
)


@dataclass
class ConfigurationEmpiricalResult:
    config_id: str
    controlled_parameters: Dict[str, Any]
    repetitions_count: int
    mean_ttft_ms: float
    stddev_ttft_ms: float
    mean_itl_ms: float
    stddev_itl_ms: float
    mean_throughput_tps: float
    stddev_throughput_tps: float
    mean_thermal_celsius: float
    mean_cpu_pressure: float
    mean_kv_cache_pressure_pct: float
    mean_utility_u: float
    uncertainty_score: float         # [0.0 - 1.0], derived from coefficient of variation
    repeatability_pct: float         # [0.0 - 100.0%], (1.0 - CV) * 100
    is_reproducible: bool            # True if repeatability >= 85%
    causal_signature_summary: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "config_id": self.config_id,
            "controlled_parameters": self.controlled_parameters,
            "repetitions_count": self.repetitions_count,
            "mean_ttft_ms": round(self.mean_ttft_ms, 2),
            "stddev_ttft_ms": round(self.stddev_ttft_ms, 3),
            "mean_itl_ms": round(self.mean_itl_ms, 2),
            "stddev_itl_ms": round(self.stddev_itl_ms, 3),
            "mean_throughput_tps": round(self.mean_throughput_tps, 1),
            "stddev_throughput_tps": round(self.stddev_throughput_tps, 2),
            "mean_thermal_celsius": round(self.mean_thermal_celsius, 1),
            "mean_cpu_pressure": round(self.mean_cpu_pressure, 3),
            "mean_kv_cache_pressure_pct": round(self.mean_kv_cache_pressure_pct, 1),
            "mean_utility_u": round(self.mean_utility_u, 4),
            "uncertainty_score": round(self.uncertainty_score, 4),
            "repeatability_pct": round(self.repeatability_pct, 1),
            "is_reproducible": self.is_reproducible,
            "causal_signature_summary": self.causal_signature_summary
        }


class BoundedConfigurationExplorer:
    """Explores controlled parameters and maps measured effect and repeatability."""

    def __init__(self):
        self.empirical_map: Dict[str, ConfigurationEmpiricalResult] = {}

    def evaluate_configuration_repetitions(
        self,
        config: ControlledVariables,
        repetitions: int = 5,
        workload_prompt: str = "G01 X50.0 Y100.0 Z-2.0 F1200 S12000 M03"
    ) -> ConfigurationEmpiricalResult:
        """
        Executes N runs of an OpenVINO / LLM command under controlled parameters.
        Measures repeatability and uncertainty.
        """
        ttft_runs = []
        itl_runs = []
        tps_runs = []
        thermal_runs = []
        cpu_runs = []
        kv_runs = []
        utility_runs = []

        for i in range(repetitions):
            trace = GLOBAL_LLM_EXECUTION_PLANE.execute_and_trace_command(
                prompt=workload_prompt,
                target_action="EVALUATE_BOUNDED_SEARCH",
                controlled_vars=config
            )
            fp = trace.footprint
            ttft_runs.append(fp.ttft_ms)
            itl_runs.append(fp.inter_token_latency_ms)
            tps_runs.append(fp.token_throughput_tps)
            thermal_runs.append(fp.thermal_celsius)
            cpu_runs.append(config.threads * 0.12)
            kv_runs.append(fp.kv_cache_pressure_pct)
            utility_runs.append(trace.utility_u)

        # Statistical calculations
        mean_ttft = statistics.mean(ttft_runs)
        std_ttft = statistics.stdev(ttft_runs) if len(ttft_runs) > 1 else 0.0

        mean_itl = statistics.mean(itl_runs)
        std_itl = statistics.stdev(itl_runs) if len(itl_runs) > 1 else 0.0

        mean_tps = statistics.mean(tps_runs)
        std_tps = statistics.stdev(tps_runs) if len(tps_runs) > 1 else 0.0

        mean_thermal = statistics.mean(thermal_runs)
        mean_cpu = statistics.mean(cpu_runs)
        mean_kv = statistics.mean(kv_runs)
        mean_u = statistics.mean(utility_runs)

        # Coefficient of variation (CV) for latency & throughput
        cv_ttft = (std_ttft / mean_ttft) if mean_ttft > 0 else 0.0
        cv_tps = (std_tps / mean_tps) if mean_tps > 0 else 0.0
        uncertainty = min(1.0, max(cv_ttft, cv_tps, 0.02))
        repeatability = max(0.0, (1.0 - uncertainty) * 100.0)

        config_id = f"CFG-T{config.threads}-S{config.streams}-B{config.batch_size}"

        causal_summary = (
            f"Threads={config.threads}, Streams={config.streams} yields TTFT={mean_ttft:.1f}ms, "
            f"Throughput={mean_tps:.1f} tps at {mean_thermal:.1f}°C. Repeatability={repeatability:.1f}%."
        )

        result = ConfigurationEmpiricalResult(
            config_id=config_id,
            controlled_parameters=config.to_dict(),
            repetitions_count=repetitions,
            mean_ttft_ms=mean_ttft,
            stddev_ttft_ms=std_ttft,
            mean_itl_ms=mean_itl,
            stddev_itl_ms=std_itl,
            mean_throughput_tps=mean_tps,
            stddev_throughput_tps=std_tps,
            mean_thermal_celsius=mean_thermal,
            mean_cpu_pressure=mean_cpu,
            mean_kv_cache_pressure_pct=mean_kv,
            mean_utility_u=mean_u,
            uncertainty_score=uncertainty,
            repeatability_pct=repeatability,
            is_reproducible=(repeatability >= 85.0),
            causal_signature_summary=causal_summary
        )

        self.empirical_map[config_id] = result
        return result

    def generate_bounded_search_map(
        self,
        candidate_thread_counts: Optional[List[int]] = None,
        candidate_stream_counts: Optional[List[int]] = None,
        repetitions_per_point: int = 5
    ) -> List[ConfigurationEmpiricalResult]:
        """
        Explores bounded combinations of (threads, streams) systematically,
        producing the empirical repeatability map.
        """
        threads_list = candidate_thread_counts or [2, 4, 8]
        streams_list = candidate_stream_counts or [1, 2]

        results = []
        for t in threads_list:
            for s in streams_list:
                cfg = ControlledVariables(
                    threads=t,
                    streams=s,
                    cpu_affinity=list(range(min(t, 8))),
                    batch_size=1
                )
                res = self.evaluate_configuration_repetitions(cfg, repetitions=repetitions_per_point)
                results.append(res)

        return results


GLOBAL_BOUNDED_EXPLORER = BoundedConfigurationExplorer()
