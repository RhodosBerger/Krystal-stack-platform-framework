r"""
CrystalSec 3.0 Phase 5.1: Hardware Evidence Calibration, Placement Regret
& Cross-Tier Model Validation Engine.

Builds upon the Phase 5 Heterogeneous Execution Triad by replacing static
microbenchmark assumptions with empirical distribution models:
    p(latency | workload_class, tier, state)

Key Capabilities:
1. Workload Taxonomy:
   - INT8_DOT_PRODUCT (Compute-bound SIMD / Tensor)
   - FP32_VECTOR_MATH (Transcendental / Polynomial vector math)
   - MEMORY_BOUND_STREAM (Bandwidth-bound streaming memory)
   - BRANCH_HEAVY_CONTROL (Irregular conditional control flow)
   - MIXED_COMPUTE_MEMORY (Composite pipeline workload)

2. Empirical Latency & Cost Distribution:
   - Tracks median, p95, p99, std, IQR, and statistical uncertainty u_tier.

3. Calibration Error Tracking:
   - E_latency = |L_pred - L_obs| / max(L_obs, eps)
   - E_uma, E_thermal, E_migration
   - Mean Absolute Calibration Error (MACE) and Prediction Bias.

4. Placement Regret Metric:
   - R_placement = U_best_observed - U_selected
   - Cumulative regret R_T and regret rate R_bar = R_T / T.
   - Misplacement rate tracking.

5. Confidence-Aware Routing & Micro-Probing:
   - High uncertainty triggers non-invasive micro-probe runs before full routing.
   - Safe-tier fallback if confidence is low.
   - Online Bayesian updating of empirical cost priors.

Platform Invariant: VITAL_MAX_HP == 6 (Unconditional).

Phase 5.2 Evidence Integrity Revision:
   - Measured latency = wall clock only; the former per-tier constants (+0.120 ms "GPU warp
     divergence", +0.035 ms "CPU branch", ...) were removed because they made rankings circular.
   - T1 / T2 taxonomy paths are labelled HOST_CPU_EMULATED (pure Python); UMA / power / thermal
     are MODELED_PRIORs. Only HardwareEvidenceProbe (Vulkan SDF raymarch) can emit
     HARDWARE_MEASURED GPU numbers.
   - Regret / calibration verdicts are evidence-gated; 'OPTIMAL' is never emitted.
"""

from __future__ import annotations

import collections
import copy
import hashlib
import json
import math
import statistics
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from krystal_kernel.verification.crystalsec_v3_control_plane import (
    CrystalSecV3State,
    TelemetryItem,
)
from krystal_kernel.verification.crystalsec_v3_heterogeneous_execution import (
    VITAL_MAX_HP,
    DirectSimAdapter,
    CpuVnniAdapter,
    IrisXeVulkanAdapter,
    DedicatedNpuAdapter,
    EvidenceLabel,
    ExecutionMode,
    ExecutionPlacementDecision,
    ExecutionTierAdapter,
    ExecutionTierCapability,
    ExecutionTierID,
    HeterogeneousTriadController,
    RuntimePlacementEngine,
    TierAvailabilityStatus,
    TierOscillationGuard,
    TransactionalMigrationArbiter,
    create_default_live_state,
)

EPSILON: float = 1e-6


# ============================================================================
# 1. WORKLOAD TAXONOMY & TEST KERNELS
# ============================================================================

class WorkloadClass(str, Enum):
    """Canonical workload taxonomy across heterogeneous compute substrates."""
    INT8_DOT_PRODUCT = "INT8_DOT_PRODUCT"          # Compute-heavy INT8 (heavily favors VNNI / DP4A)
    FP32_VECTOR_MATH = "FP32_VECTOR_MATH"          # Vectorized FP32 math (activations, polynomials)
    MEMORY_BOUND_STREAM = "MEMORY_BOUND_STREAM"    # Streaming data transfer, UMA contention sensitive
    BRANCH_HEAVY_CONTROL = "BRANCH_HEAVY_CONTROL"  # Irregular branching (favors CPU OoO; GPU branch penalty)
    MIXED_COMPUTE_MEMORY = "MIXED_COMPUTE_MEMORY"  # Realistic composite pipeline


@dataclass
class SyntheticWorkload:
    """Workload definition carrying inputs and execution semantics."""
    workload_id: str
    workload_class: WorkloadClass
    data: Dict[str, Any]
    elements_count: int
    working_set_bytes: int
    expected_intensity_flop_per_byte: float


class WorkloadSynthesizer:
    """Generates deterministic, calibrated workloads for each taxonomy class."""

    @staticmethod
    def generate(
        workload_class: WorkloadClass,
        size: int = 128,
        workload_id: Optional[str] = None,
    ) -> SyntheticWorkload:
        wl_id = workload_id or f"wl_{workload_class.value}_{size}_{int(time.time()*1000)}"

        if workload_class == WorkloadClass.INT8_DOT_PRODUCT:
            # Saturated INT8 vector pairs
            a = [(i % 127) - 64 for i in range(size)]
            b = [((i * 3) % 127) - 64 for i in range(size)]
            return SyntheticWorkload(
                workload_id=wl_id,
                workload_class=workload_class,
                data={"vector_a": a, "vector_b": b},
                elements_count=size,
                working_set_bytes=size * 2,
                expected_intensity_flop_per_byte=2.0,
            )

        elif workload_class == WorkloadClass.FP32_VECTOR_MATH:
            # Continuous floats for polynomial activation math
            x = [float(i) * 0.05 - 3.14 for i in range(size)]
            return SyntheticWorkload(
                workload_id=wl_id,
                workload_class=workload_class,
                data={"inputs": x, "coeffs": [0.5, -0.2, 0.1, 0.05]},
                elements_count=size,
                working_set_bytes=size * 4,
                expected_intensity_flop_per_byte=8.0,
            )

        elif workload_class == WorkloadClass.MEMORY_BOUND_STREAM:
            # Large buffer stream with low arithmetic operations
            size_stream = max(size, 1024)
            buf = [float(i % 256) for i in range(size_stream)]
            return SyntheticWorkload(
                workload_id=wl_id,
                workload_class=workload_class,
                data={"buffer": buf, "stride": 4},
                elements_count=size_stream,
                working_set_bytes=size_stream * 4,
                expected_intensity_flop_per_byte=0.125,
            )

        elif workload_class == WorkloadClass.BRANCH_HEAVY_CONTROL:
            # Irregular control data with high entropy bit-patterns
            keys = [int((i * 7919) ^ (i << 3)) % 1024 for i in range(size)]
            return SyntheticWorkload(
                workload_id=wl_id,
                workload_class=workload_class,
                data={"branch_keys": keys},
                elements_count=size,
                working_set_bytes=size * 4,
                expected_intensity_flop_per_byte=1.5,
            )

        elif workload_class == WorkloadClass.MIXED_COMPUTE_MEMORY:
            a = [(i % 127) - 64 for i in range(size)]
            b = [((i * 2) % 127) - 64 for i in range(size)]
            buf = [float(i) for i in range(size * 2)]
            return SyntheticWorkload(
                workload_id=wl_id,
                workload_class=workload_class,
                data={"vector_a": a, "vector_b": b, "stream_buf": buf},
                elements_count=size * 3,
                working_set_bytes=(size * 2) + (size * 8),
                expected_intensity_flop_per_byte=1.2,
            )
        else:
            raise ValueError(f"Unknown workload class: {workload_class}")


# ============================================================================
# 2. WORKLOAD TAXONOMY EXECUTION ENGINE
# ============================================================================

# Phase 5.2: secondary telemetry is NOT measured on this host by this module. The values below are
# MODELED_PRIORs kept only so downstream consumers have a number; every telemetry dict carries
# `telemetry_evidence = MODELED_PRIOR` and calibration excludes them from error metrics.
MODELED_TELEMETRY_PRIORS: Dict[ExecutionTierID, Dict[str, float]] = {
    ExecutionTierID.TIER_0_DIRECTSIM: {"uma_pressure": 0.05, "power_w": 2.5, "temp_delta_c": 0.02, "thread_util": 0.25},
    ExecutionTierID.TIER_1_CPU_VNNI: {"uma_pressure": 0.12, "power_w": 14.5, "temp_delta_c": 0.18, "thread_util": 0.85},
    ExecutionTierID.TIER_2_IRIS_XE: {"uma_pressure": 0.18, "power_w": 18.0, "temp_delta_c": 0.45, "thread_util": 0.15},
}
MODELED_UMA_PRIOR_OVERRIDES: Dict[Tuple[ExecutionTierID, WorkloadClass], float] = {
    (ExecutionTierID.TIER_1_CPU_VNNI, WorkloadClass.MEMORY_BOUND_STREAM): 0.45,
    (ExecutionTierID.TIER_2_IRIS_XE, WorkloadClass.MEMORY_BOUND_STREAM): 0.72,
    (ExecutionTierID.TIER_2_IRIS_XE, WorkloadClass.MIXED_COMPUTE_MEMORY): 0.35,
}

# Which (tier, class) pairs have a code path that differs from the T0 reference at all.
# Everything else on T1/T2 executes the *same* Python loop as T0 and therefore cannot
# provide evidence about the named hardware.
_DISTINCT_ADAPTER_PATHS: Set[Tuple[ExecutionTierID, WorkloadClass]] = {
    (ExecutionTierID.TIER_1_CPU_VNNI, WorkloadClass.INT8_DOT_PRODUCT),
    (ExecutionTierID.TIER_1_CPU_VNNI, WorkloadClass.MIXED_COMPUTE_MEMORY),
    (ExecutionTierID.TIER_2_IRIS_XE, WorkloadClass.INT8_DOT_PRODUCT),
    (ExecutionTierID.TIER_2_IRIS_XE, WorkloadClass.MIXED_COMPUTE_MEMORY),
}


class TaxonomyExecutionEngine:
    """
    Executes specific workload classes against the respective tier adapters.

    Phase 5.2 evidence rules:
      - Returned latency is pure wall-clock (time.perf_counter) of the executed path.
        No per-tier constants are added (previous versions added e.g. +0.120 ms "GPU warp
        divergence" and +0.035 ms "CPU branch" penalties, which made tier rankings circular).
      - T1 / T2 paths in this module run on the host CPU in Python -> HOST_CPU_EMULATED.
      - Telemetry fields are MODELED_PRIORs, never measurements.
    """

    @staticmethod
    def _telemetry(tier: ExecutionTierID, wc: WorkloadClass, latency_evidence: EvidenceLabel) -> Dict[str, Any]:
        telem: Dict[str, Any] = dict(MODELED_TELEMETRY_PRIORS[tier])
        if (tier, wc) in MODELED_UMA_PRIOR_OVERRIDES:
            telem["uma_pressure"] = MODELED_UMA_PRIOR_OVERRIDES[(tier, wc)]
        telem["latency_evidence"] = latency_evidence.value
        telem["telemetry_evidence"] = EvidenceLabel.MODELED_PRIOR.value
        telem["distinct_code_path"] = (
            tier == ExecutionTierID.TIER_0_DIRECTSIM or (tier, wc) in _DISTINCT_ADAPTER_PATHS
        )
        return telem

    @staticmethod
    def execute_workload(
        workload: SyntheticWorkload,
        tier: ExecutionTierID,
        adapters: Dict[ExecutionTierID, ExecutionTierAdapter],
    ) -> Tuple[List[float], float, Dict[str, Any]]:
        """
        Executes workload on tier and returns: (output, measured_latency_ms, telemetry_with_provenance)
        """
        adapter = adapters[tier]
        if not adapter.available():
            raise RuntimeError(f"Tier {tier.value} is not available for execution.")

        wc = workload.workload_class

        if tier == ExecutionTierID.TIER_3_DEDICATED_NPU:
            raise RuntimeError("Tier 3 Dedicated NPU is PHYSICALLY_UNAVAILABLE on this host.")

        t_start = time.perf_counter()
        if tier == ExecutionTierID.TIER_0_DIRECTSIM:
            out = TaxonomyExecutionEngine._run_directsim(workload)
            evidence = EvidenceLabel.HOST_CPU_REFERENCE
        elif tier == ExecutionTierID.TIER_1_CPU_VNNI:
            out = TaxonomyExecutionEngine._run_cpu_vnni(workload, adapter)
            evidence = EvidenceLabel.HOST_CPU_EMULATED
        elif tier == ExecutionTierID.TIER_2_IRIS_XE:
            out = TaxonomyExecutionEngine._run_iris_xe(workload, adapter)
            evidence = EvidenceLabel.HOST_CPU_EMULATED
        else:
            raise ValueError(f"Unknown tier: {tier}")
        measured_lat = (time.perf_counter() - t_start) * 1000.0

        return out, max(0.0001, measured_lat), TaxonomyExecutionEngine._telemetry(tier, wc, evidence)

    @staticmethod
    def _run_directsim(workload: SyntheticWorkload) -> List[float]:
        wc = workload.workload_class
        data = workload.data

        if wc in (WorkloadClass.INT8_DOT_PRODUCT, WorkloadClass.MIXED_COMPUTE_MEMORY):
            a, b = data["vector_a"], data["vector_b"]
            return [max(0.0, float(x * y)) for x, y in zip(a, b)]
        elif wc == WorkloadClass.FP32_VECTOR_MATH:
            inputs, c = data["inputs"], data["coeffs"]
            return [c[0] + c[1]*x + c[2]*(x**2) + c[3]*(x**3) for x in inputs]
        elif wc == WorkloadClass.MEMORY_BOUND_STREAM:
            buf, stride = data["buffer"], data["stride"]
            return [buf[i] * 1.001 for i in range(0, len(buf), stride)]
        elif wc == WorkloadClass.BRANCH_HEAVY_CONTROL:
            keys = data["branch_keys"]
            res = []
            for k in keys:
                if k % 11 == 0:
                    v = k * 3 + 1
                elif k % 7 == 0:
                    v = k // 2
                elif k % 3 == 0:
                    v = k * 5
                elif k % 2 == 0:
                    v = k + 17
                else:
                    v = k - 1
                res.append(float(v))
            return res
        return [0.0]

    @staticmethod
    def _run_cpu_vnni(workload: SyntheticWorkload, adapter: ExecutionTierAdapter) -> List[float]:
        """T1 path. INT8 / MIXED go through CpuVnniBackend (pure-Python INT8 emulation);
        other classes have no distinct T1 implementation and reuse the reference loop."""
        wc = workload.workload_class
        data = workload.data

        if wc in (WorkloadClass.INT8_DOT_PRODUCT, WorkloadClass.MIXED_COMPUTE_MEMORY):
            out, _ = adapter.execute({"input": data["vector_a"], "weights": data["vector_b"]})
            return out
        return TaxonomyExecutionEngine._run_directsim(workload)

    @staticmethod
    def _run_iris_xe(workload: SyntheticWorkload, adapter: ExecutionTierAdapter) -> List[float]:
        """T2 path. IrisXeVulkanAdapter.execute is a host-CPU emulation (see adapter docstring);
        other classes have no GPU kernel at all and reuse the reference loop."""
        wc = workload.workload_class
        data = workload.data

        if wc in (WorkloadClass.INT8_DOT_PRODUCT, WorkloadClass.MIXED_COMPUTE_MEMORY):
            out, _ = adapter.execute({"input": data["vector_a"], "weights": data["vector_b"]})
            return out
        return TaxonomyExecutionEngine._run_directsim(workload)


# ============================================================================
# 3. EMPIRICAL LATENCY & COST DISTRIBUTION: p(latency | workload, tier, state)
# ============================================================================

@dataclass
class EmpiricalDistributionStats:
    """Distribution summary metrics for a (workload, tier) combination."""
    sample_count: int
    mean_val: float
    median_val: float
    p95_val: float
    p99_val: float
    std_val: float
    cv_val: float
    iqr_val: float
    uncertainty_score: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sample_count": self.sample_count,
            "mean": round(self.mean_val, 4),
            "median": round(self.median_val, 4),
            "p95": round(self.p95_val, 4),
            "p99": round(self.p99_val, 4),
            "std": round(self.std_val, 4),
            "cv": round(self.cv_val, 4),
            "iqr": round(self.iqr_val, 4),
            "uncertainty": round(self.uncertainty_score, 4),
        }


class EmpiricalCostModel:
    """
    Tracks empirical distributions p(latency | workload_class, tier, state).
    Replaces static scalar constants with rolling statistical distributions.
    """

    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        # (workload_class, tier_id) -> deque of measured latencies
        self.history: Dict[Tuple[WorkloadClass, ExecutionTierID], collections.deque] = collections.defaultdict(
            lambda: collections.deque(maxlen=window_size)
        )
        # Prior estimates for initial bootstrap
        self.priors: Dict[Tuple[WorkloadClass, ExecutionTierID], float] = {
            (WorkloadClass.INT8_DOT_PRODUCT, ExecutionTierID.TIER_0_DIRECTSIM): 0.50,
            (WorkloadClass.INT8_DOT_PRODUCT, ExecutionTierID.TIER_1_CPU_VNNI): 0.12,
            (WorkloadClass.INT8_DOT_PRODUCT, ExecutionTierID.TIER_2_IRIS_XE): 0.035,
            (WorkloadClass.FP32_VECTOR_MATH, ExecutionTierID.TIER_0_DIRECTSIM): 0.65,
            (WorkloadClass.FP32_VECTOR_MATH, ExecutionTierID.TIER_1_CPU_VNNI): 0.16,
            (WorkloadClass.FP32_VECTOR_MATH, ExecutionTierID.TIER_2_IRIS_XE): 0.040,
            (WorkloadClass.MEMORY_BOUND_STREAM, ExecutionTierID.TIER_0_DIRECTSIM): 0.75,
            (WorkloadClass.MEMORY_BOUND_STREAM, ExecutionTierID.TIER_1_CPU_VNNI): 0.18,
            (WorkloadClass.MEMORY_BOUND_STREAM, ExecutionTierID.TIER_2_IRIS_XE): 0.19,
            (WorkloadClass.BRANCH_HEAVY_CONTROL, ExecutionTierID.TIER_0_DIRECTSIM): 0.40,
            (WorkloadClass.BRANCH_HEAVY_CONTROL, ExecutionTierID.TIER_1_CPU_VNNI): 0.09,
            (WorkloadClass.BRANCH_HEAVY_CONTROL, ExecutionTierID.TIER_2_IRIS_XE): 0.22,
            (WorkloadClass.MIXED_COMPUTE_MEMORY, ExecutionTierID.TIER_0_DIRECTSIM): 0.55,
            (WorkloadClass.MIXED_COMPUTE_MEMORY, ExecutionTierID.TIER_1_CPU_VNNI): 0.14,
            (WorkloadClass.MIXED_COMPUTE_MEMORY, ExecutionTierID.TIER_2_IRIS_XE): 0.065,
        }

    def record_observation(
        self,
        workload_class: WorkloadClass,
        tier: ExecutionTierID,
        latency_ms: float,
    ):
        """Records an empirical execution observation."""
        self.history[(workload_class, tier)].append(max(0.0001, latency_ms))

    def predict_latency(
        self,
        workload_class: WorkloadClass,
        tier: ExecutionTierID,
    ) -> float:
        """Returns median empirical latency or prior if unobserved."""
        key = (workload_class, tier)
        samples = self.history[key]
        if not samples:
            return self.priors.get(key, 0.50)
        return float(statistics.median(samples))

    def get_distribution_stats(
        self,
        workload_class: WorkloadClass,
        tier: ExecutionTierID,
    ) -> EmpiricalDistributionStats:
        """Calculates statistical moments and uncertainty."""
        key = (workload_class, tier)
        samples = list(self.history[key])

        if not samples:
            prior = self.priors.get(key, 0.50)
            return EmpiricalDistributionStats(
                sample_count=0,
                mean_val=prior,
                median_val=prior,
                p95_val=prior * 1.5,
                p99_val=prior * 2.0,
                std_val=prior * 0.5,
                cv_val=0.5,
                iqr_val=prior * 0.25,
                uncertainty_score=1.0,
            )

        n = len(samples)
        mean_val = statistics.mean(samples)
        median_val = statistics.median(samples)
        std_val = statistics.stdev(samples) if n > 1 else 0.0
        cv_val = std_val / (mean_val + EPSILON)

        s_sorted = sorted(samples)
        p95_val = s_sorted[int(math.ceil(0.95 * n)) - 1]
        p99_val = s_sorted[int(math.ceil(0.99 * n)) - 1]
        q75 = s_sorted[int(math.ceil(0.75 * n)) - 1]
        q25 = s_sorted[int(math.ceil(0.25 * n)) - 1]
        iqr_val = q75 - q25

        sparsity_penalty = 1.0 / math.sqrt(n + 1)
        uncertainty = min(1.0, (cv_val * 0.5) + (sparsity_penalty * 0.5))

        return EmpiricalDistributionStats(
            sample_count=n,
            mean_val=mean_val,
            median_val=median_val,
            p95_val=p95_val,
            p99_val=p99_val,
            std_val=std_val,
            cv_val=cv_val,
            iqr_val=iqr_val,
            uncertainty_score=uncertainty,
        )


# ============================================================================
# 4. CALIBRATION ERROR TRACKER
# ============================================================================

@dataclass
class CalibrationSample:
    """Record comparing predicted vs observed cost metrics."""
    workload_id: str
    workload_class: WorkloadClass
    tier: ExecutionTierID
    predicted_latency_ms: float
    observed_latency_ms: float
    latency_error: float
    predicted_uma: float
    observed_uma: float
    uma_error: float
    predicted_temp_delta: float
    observed_temp_delta: float
    temp_error: float
    composite_error: float
    bias_latency: float
    timestamp: float


class CalibrationErrorTracker:
    """
    Evaluates placement model accuracy:
        E_latency = |L_pred - L_obs| / max(L_obs, eps)
    Maintains rolling MACE (Mean Absolute Calibration Error) and prediction bias.

    Phase 5.2:
      - UMA / thermal errors only enter the composite when the observed value is
        HARDWARE_MEASURED. Comparing a modeled prediction to a modeled "observation"
        is not calibration.
      - No verdict below `min_samples_for_verdict` samples (INSUFFICIENT_EVIDENCE).
    """

    def __init__(
        self,
        max_samples: int = 200,
        min_samples_for_verdict: int = 10,
        mace_tolerance: float = 0.30,
    ):
        self.samples: collections.deque[CalibrationSample] = collections.deque(maxlen=max_samples)
        self.min_samples_for_verdict = min_samples_for_verdict
        self.mace_tolerance = mace_tolerance
        self._secondary_included_count: int = 0

    def record_prediction(
        self,
        workload_id: str,
        workload_class: WorkloadClass,
        tier: ExecutionTierID,
        pred_lat: float,
        obs_lat: float,
        pred_uma: float = 0.20,
        obs_uma: float = 0.20,
        pred_temp: float = 0.20,
        obs_temp: float = 0.20,
        secondary_evidence: str = EvidenceLabel.HARDWARE_MEASURED.value,
    ) -> CalibrationSample:
        lat_err = abs(pred_lat - obs_lat) / max(obs_lat, EPSILON)
        uma_err = abs(pred_uma - obs_uma) / max(obs_uma, EPSILON)
        temp_err = abs(pred_temp - obs_temp) / max(abs(obs_temp), EPSILON)

        if secondary_evidence == EvidenceLabel.HARDWARE_MEASURED.value:
            comp_err = (0.50 * lat_err) + (0.30 * uma_err) + (0.20 * temp_err)
            self._secondary_included_count += 1
        else:
            comp_err = lat_err
        bias = pred_lat - obs_lat

        sample = CalibrationSample(
            workload_id=workload_id,
            workload_class=workload_class,
            tier=tier,
            predicted_latency_ms=pred_lat,
            observed_latency_ms=obs_lat,
            latency_error=lat_err,
            predicted_uma=pred_uma,
            observed_uma=obs_uma,
            uma_error=uma_err,
            predicted_temp_delta=pred_temp,
            observed_temp_delta=obs_temp,
            temp_error=temp_err,
            composite_error=comp_err,
            bias_latency=bias,
            timestamp=time.time(),
        )
        self.samples.append(sample)
        return sample

    def get_calibration_metrics(self) -> Dict[str, Any]:
        """Calculates aggregate calibration metrics with an evidence-gated verdict."""
        if not self.samples:
            return {
                "sample_count": 0,
                "mean_latency_error": 0.0,
                "median_latency_error": 0.0,
                "mace_composite": 0.0,
                "prediction_bias": 0.0,
                "is_well_calibrated": False,
                "calibration_status": "INSUFFICIENT_EVIDENCE",
                "secondary_metrics_included": False,
            }

        lat_errs = [s.latency_error for s in self.samples]
        comp_errs = [s.composite_error for s in self.samples]
        biases = [s.bias_latency for s in self.samples]

        m_lat = statistics.mean(lat_errs)
        med_lat = statistics.median(lat_errs)
        mace = statistics.mean(comp_errs)
        mean_bias = statistics.mean(biases)
        n = len(self.samples)

        if n < self.min_samples_for_verdict:
            status = "INSUFFICIENT_EVIDENCE"
        elif mace <= self.mace_tolerance:
            status = "WITHIN_TOLERANCE"
        else:
            status = "RECALIBRATION_REQUIRED"

        per_class: Dict[str, Dict[str, float]] = {}
        for wc in {s.workload_class for s in self.samples}:
            errs = [s.latency_error for s in self.samples if s.workload_class == wc]
            per_class[wc.value] = {
                "sample_count": len(errs),
                "median_latency_error": round(statistics.median(errs), 4),
            }

        return {
            "sample_count": n,
            "mean_latency_error": round(m_lat, 4),
            "median_latency_error": round(med_lat, 4),
            "mace_composite": round(mace, 4),
            "prediction_bias": round(mean_bias, 4),
            "is_well_calibrated": status == "WITHIN_TOLERANCE",
            "calibration_status": status,
            "mace_tolerance": self.mace_tolerance,
            "min_samples_for_verdict": self.min_samples_for_verdict,
            "secondary_metrics_included": self._secondary_included_count > 0,
            "per_class": per_class,
        }


# ============================================================================
# 5. PLACEMENT REGRET TRACKER
# ============================================================================

class PlacementQuality(str, Enum):
    """Evidence-scoped placement verdicts. There is deliberately no 'OPTIMAL'."""
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    ZERO_REGRET_ON_EVALUATED_SET = "ZERO_REGRET_ON_EVALUATED_SET"
    LOW_REGRET = "LOW_REGRET"
    PLACEMENT_RECALIBRATION_REQUIRED = "PLACEMENT_RECALIBRATION_REQUIRED"


@dataclass
class RegretObservation:
    """Record of a single placement regret event."""
    workload_id: str
    workload_class: WorkloadClass
    selected_tier: ExecutionTierID
    selected_utility: float
    best_tier: ExecutionTierID
    best_observed_utility: float
    regret: float
    is_misplacement: bool
    timestamp: float
    latency_regret_ms: Optional[float] = None
    evaluated_tiers: Tuple[str, ...] = ()
    noise_floor_ms: Optional[float] = None
    within_measurement_noise: bool = False


class PlacementRegretTracker:
    """
    Measures the quality of placement decisions:
        R_placement = U_best_observed - U_selected

    Phase 5.2 epistemic rules:
      - Regret is relative to the tiers actually evaluated; it never certifies global optimality.
      - Below `min_decisions_for_verdict` decisions the verdict is INSUFFICIENT_EVIDENCE.
      - Selected/best tier distributions are always reported so a degenerate
        "always pick the same tier" policy cannot hide behind R = 0.
    """

    def __init__(
        self,
        max_history: int = 500,
        misplacement_threshold: float = 0.05,
        min_decisions_for_verdict: int = 10,
        low_regret_rate: float = 0.03,
        low_misplacement_rate: float = 0.05,
    ):
        self.history: collections.deque[RegretObservation] = collections.deque(maxlen=max_history)
        self.misplacement_threshold = misplacement_threshold
        self.min_decisions_for_verdict = min_decisions_for_verdict
        self.low_regret_rate = low_regret_rate
        self.low_misplacement_rate = low_misplacement_rate
        self.cumulative_regret: float = 0.0

    def record_decision(
        self,
        workload_id: str,
        workload_class: WorkloadClass,
        selected_tier: ExecutionTierID,
        selected_utility: float,
        best_tier: ExecutionTierID,
        best_utility: float,
        selected_latency_ms: Optional[float] = None,
        best_latency_ms: Optional[float] = None,
        evaluated_tiers: Optional[List[ExecutionTierID]] = None,
        noise_floor_ms: Optional[float] = None,
    ) -> RegretObservation:
        # Regret cannot be negative
        regret = max(0.0, best_utility - selected_utility)
        lat_regret = None
        if selected_latency_ms is not None and best_latency_ms is not None:
            lat_regret = max(0.0, selected_latency_ms - best_latency_ms)
        # A latency gap smaller than the measured repeat spread is not distinguishable from jitter.
        within_noise = (
            lat_regret is not None and noise_floor_ms is not None and lat_regret <= noise_floor_ms
        )
        is_misplacement = regret > self.misplacement_threshold and not within_noise

        obs = RegretObservation(
            workload_id=workload_id,
            workload_class=workload_class,
            selected_tier=selected_tier,
            selected_utility=selected_utility,
            best_tier=best_tier,
            best_observed_utility=best_utility,
            regret=regret,
            is_misplacement=is_misplacement,
            timestamp=time.time(),
            latency_regret_ms=lat_regret,
            evaluated_tiers=tuple(t.value for t in (evaluated_tiers or [])),
            noise_floor_ms=noise_floor_ms,
            within_measurement_noise=within_noise,
        )
        self.history.append(obs)
        self.cumulative_regret += regret
        return obs

    def get_regret_metrics(self) -> Dict[str, Any]:
        """Summarizes regret metrics over the rolling window with an evidence-scoped verdict."""
        total = len(self.history)
        if total == 0:
            return {
                "decision_count": 0,
                "cumulative_regret": 0.0,
                "average_regret_rate": 0.0,
                "misplacement_count": 0,
                "misplacement_rate": 0.0,
                "placement_quality": PlacementQuality.INSUFFICIENT_EVIDENCE.value,
                "model_optimality": PlacementQuality.INSUFFICIENT_EVIDENCE.value,  # deprecated alias
                "selected_tier_distribution": {},
                "best_tier_distribution": {},
                "single_tier_selection": False,
            }

        window_regret = sum(obs.regret for obs in self.history)
        adjusted_regret = sum(0.0 if obs.within_measurement_noise else obs.regret for obs in self.history)
        misplacements = sum(1 for obs in self.history if obs.is_misplacement)
        avg_regret = window_regret / total
        adj_avg_regret = adjusted_regret / total
        misplacement_rate = misplacements / total

        if total < self.min_decisions_for_verdict:
            quality = PlacementQuality.INSUFFICIENT_EVIDENCE
        elif adjusted_regret <= 1e-12 and misplacements == 0:
            quality = PlacementQuality.ZERO_REGRET_ON_EVALUATED_SET
        elif adj_avg_regret <= self.low_regret_rate and misplacement_rate <= self.low_misplacement_rate:
            quality = PlacementQuality.LOW_REGRET
        else:
            quality = PlacementQuality.PLACEMENT_RECALIBRATION_REQUIRED

        selected_dist = dict(collections.Counter(obs.selected_tier.value for obs in self.history))
        best_dist = dict(collections.Counter(obs.best_tier.value for obs in self.history))
        evaluated = sorted({t for obs in self.history for t in obs.evaluated_tiers})
        lat_regrets = [obs.latency_regret_ms for obs in self.history if obs.latency_regret_ms is not None]

        return {
            "decision_count": total,
            "cumulative_regret": round(self.cumulative_regret, 4),
            "average_regret_rate": round(avg_regret, 4),
            "noise_adjusted_regret_rate": round(adj_avg_regret, 4),
            "misplacement_count": misplacements,
            "misplacement_rate": round(misplacement_rate, 4),
            "mean_latency_regret_ms": round(statistics.mean(lat_regrets), 5) if lat_regrets else None,
            "within_measurement_noise_count": sum(1 for obs in self.history if obs.within_measurement_noise),
            "placement_quality": quality.value,
            "model_optimality": quality.value,  # deprecated alias; never 'OPTIMAL'
            "selected_tier_distribution": selected_dist,
            "best_tier_distribution": best_dist,
            "single_tier_selection": len(selected_dist) == 1,
            "evaluated_tiers": evaluated,
            "scope_note": "Regret is relative to the tiers evaluated in this run on this host; "
                          "it is not a claim of global optimality.",
        }


# ============================================================================
# 6. CONFIDENCE-AWARE ROUTER & MICRO-PROBE RUNNER
# ============================================================================

class ConfidenceAwareRouter:
    """
    Directs workload placement based on cost-model confidence.
    If uncertainty exceeds threshold or data is sparse, executes a micro-probe
    run before committing full routing.
    """

    def __init__(
        self,
        cost_model: EmpiricalCostModel,
        adapters: Dict[ExecutionTierID, ExecutionTierAdapter],
        uncertainty_threshold: float = 0.35,
        min_probe_samples: int = 3,
    ):
        self.cost_model = cost_model
        self.adapters = adapters
        self.uncertainty_threshold = uncertainty_threshold
        self.min_probe_samples = min_probe_samples
        self.probe_executions_count: int = 0

    def evaluate_routing(
        self,
        workload: SyntheticWorkload,
        candidate_tier: ExecutionTierID,
        record_probe_observation: bool = True,
    ) -> Tuple[ExecutionTierID, bool, str]:
        """
        Determines whether candidate tier can be routed directly or requires probe execution.
        Returns: (final_tier, probe_triggered, routing_reason)

        `record_probe_observation=False` keeps 16-element probe latencies out of the class
        distribution (they are a different workload size and would bias calibration).
        """
        wc = workload.workload_class
        stats = self.cost_model.get_distribution_stats(wc, candidate_tier)

        # 1. Check if model has sufficient empirical confidence
        if stats.sample_count >= self.min_probe_samples and stats.uncertainty_score <= self.uncertainty_threshold:
            return candidate_tier, False, f"Confident routing (u={stats.uncertainty_score:.3f}, N={stats.sample_count})"

        # 2. Uncertainty is high -> Trigger non-invasive Micro-Probe Run
        self.probe_executions_count += 1
        probe_ok, probe_lat = self._run_micro_probe(workload, candidate_tier)

        if probe_ok:
            if record_probe_observation:
                self.cost_model.record_observation(wc, candidate_tier, probe_lat)
            return candidate_tier, True, f"Probed successfully ({probe_lat:.3f}ms), routed to {candidate_tier.value}"
        else:
            fallback = ExecutionTierID.TIER_1_CPU_VNNI if self.adapters[ExecutionTierID.TIER_1_CPU_VNNI].available() else ExecutionTierID.TIER_0_DIRECTSIM
            return fallback, True, f"Probe failed/degraded; falling back to safe tier {fallback.value}"

    def _run_micro_probe(
        self,
        workload: SyntheticWorkload,
        tier: ExecutionTierID,
    ) -> Tuple[bool, float]:
        """Executes a tiny probe (16-element slice) to test hardware responsiveness."""
        # Phase 5.2 fix: previously the full data was deep-copied, so the "probe" ran the
        # entire workload. Slice every list payload to 16 elements.
        probe_data = {
            k: (list(v[:16]) if isinstance(v, list) else copy.deepcopy(v))
            for k, v in workload.data.items()
        }
        probe_wl = SyntheticWorkload(
            workload_id=f"probe_{workload.workload_id}",
            workload_class=workload.workload_class,
            data=probe_data,
            elements_count=min(16, workload.elements_count),
            working_set_bytes=64,
            expected_intensity_flop_per_byte=workload.expected_intensity_flop_per_byte,
        )
        try:
            _, lat, _ = TaxonomyExecutionEngine.execute_workload(probe_wl, tier, self.adapters)
            return lat < 20.0, lat
        except Exception:
            return False, 999.0


# ============================================================================
# 7. HARDWARE EVIDENCE PROBE (real Vulkan dispatch, Phase 5.2)
# ============================================================================

class HardwareEvidenceProbe:
    """
    Collects the only GPU numbers in this module that may be labelled HARDWARE_MEASURED:
    the SDF raymarch kernel dispatched through src/python/vulkan_compute_driver.py.

    A row is HARDWARE_MEASURED only if every repeat reported compute_backend == VULKAN_GPU
    with real_gpu_dispatch == True immediately after the dispatch. Otherwise the GPU column
    is labelled HOST_CPU_EMULATED (driver CPU fallback).
    """

    DEFAULT_RESOLUTIONS: Tuple[Tuple[int, int], ...] = ((16, 8), (64, 32), (160, 80))

    def __init__(self, driver_factory: Optional[Callable[[], Any]] = None):
        self.driver_factory = driver_factory or self._default_driver_factory

    @staticmethod
    def _default_driver_factory() -> Any:
        from src.python.vulkan_compute_driver import VulkanComputeDriver
        return VulkanComputeDriver(prefer_gpu=True)

    def run_sdf_raymarch_sweep(
        self,
        resolutions: Optional[Tuple[Tuple[int, int], ...]] = None,
        repeats: int = 5,
    ) -> Dict[str, Any]:
        resolutions = resolutions or self.DEFAULT_RESOLUTIONS
        try:
            driver = self.driver_factory()
        except Exception as exc:  # pragma: no cover - depends on host drivers
            return {
                "workload": "SDF_RAYMARCH",
                "status": EvidenceLabel.UNAVAILABLE.value,
                "reason": f"Vulkan driver init failed: {exc}",
                "rows": [],
            }

        rows: List[Dict[str, Any]] = []
        device_name = getattr(driver, "device_name", "unknown")
        try:
            for (w, h) in resolutions:
                gpu_lats: List[float] = []
                cpu_lats: List[float] = []
                backends: Set[str] = set()
                matches = 0
                compared = 0
                for r in range(repeats):
                    t = 0.1 * r
                    t0 = time.perf_counter()
                    g_glyphs, _ = driver.execute_raymarch(w, h, t, readback=True)
                    gpu_lats.append((time.perf_counter() - t0) * 1000.0)
                    # Read provenance immediately: execute_raymarch_cpu resets these flags.
                    real = bool(getattr(driver, "real_gpu_dispatch", False))
                    backends.add(getattr(driver, "compute_backend", "UNKNOWN") if real else "CPU_FALLBACK")

                    t0 = time.perf_counter()
                    c_glyphs, _ = driver.execute_raymarch_cpu(w, h, t)
                    cpu_lats.append((time.perf_counter() - t0) * 1000.0)

                    if g_glyphs and c_glyphs and len(g_glyphs) == len(c_glyphs):
                        compared += len(c_glyphs)
                        matches += sum(1 for a, b in zip(g_glyphs, c_glyphs) if a == b)

                gpu_label = (
                    EvidenceLabel.HARDWARE_MEASURED if backends == {"VULKAN_GPU"}
                    else EvidenceLabel.HOST_CPU_EMULATED
                )
                t2_med = statistics.median(gpu_lats)
                t0_med = statistics.median(cpu_lats)
                rows.append({
                    "resolution": f"{w}x{h}",
                    "pixels": w * h,
                    "repeats": repeats,
                    "t2_median_ms": round(t2_med, 4),
                    "t2_evidence": gpu_label.value,
                    "t2_backends_observed": sorted(backends),
                    "t0_median_ms": round(t0_med, 4),
                    "t0_evidence": EvidenceLabel.HOST_CPU_REFERENCE.value,
                    "t0_over_t2_ratio": round(t0_med / max(t2_med, EPSILON), 2),
                    "glyph_match_ratio": round(matches / compared, 4) if compared else None,
                })
        finally:
            try:
                driver.cleanup()
            except Exception:
                pass

        measured = [r for r in rows if r["t2_evidence"] == EvidenceLabel.HARDWARE_MEASURED.value]
        faster = [r for r in measured if r["t2_median_ms"] < r["t0_median_ms"]]
        return {
            "workload": "SDF_RAYMARCH",
            "status": EvidenceLabel.HARDWARE_MEASURED.value if measured else EvidenceLabel.HOST_CPU_EMULATED.value,
            "device_name": device_name,
            "rows": rows,
            "smallest_evaluated_size_where_t2_faster_px": min((r["pixels"] for r in faster), default=None),
            "note": "T2 timings include dispatch + readback. Crossover statement is limited to "
                    "the evaluated resolutions; T0 here is the driver's pure-Python CPU reference.",
        }


# ============================================================================
# 8. UNIFIED CALIBRATION & VALIDATION CONTROLLER
# ============================================================================

class PlacementCalibrationController:
    """
    Unified coordinator for CrystalSec 3.0 Phase 5.1 (evidence revision 5.2):
    - Benchmarks the 5 taxonomy workload classes across tiers.
    - Calibrates empirical distributions p(L | W, T, S).
    - Measures calibration error and tracks placement regret.
    - Governs confidence-aware routing.
    """

    def __init__(self, triad_controller: Optional[HeterogeneousTriadController] = None):
        self.triad_controller = triad_controller or HeterogeneousTriadController()
        self.adapters = self.triad_controller.adapters
        self.cost_model = EmpiricalCostModel()
        self.calibration_tracker = CalibrationErrorTracker()
        self.regret_tracker = PlacementRegretTracker()
        self.confidence_router = ConfidenceAwareRouter(
            cost_model=self.cost_model,
            adapters=self.adapters,
        )
        # Connect empirical cost model to placement engine
        self.triad_controller.placement_engine.empirical_cost_model = self.cost_model

    def run_calibration_suite(
        self,
        iterations_per_class: int = 15,
        workload_size: int = 128,
        repeats_per_measurement: int = 5,
        include_hardware_evidence: bool = False,
        hardware_probe: Optional[HardwareEvidenceProbe] = None,
    ) -> Dict[str, Any]:
        """
        Runs calibration across all workload classes and available tiers.

        Order per iteration (Phase 5.2):
          1. Placement engine predicts BEFORE this iteration's measurements are recorded
             (previously the prediction already contained the sample it was scored against).
          2. Confidence router may probe the proposed tier (probe latencies are not recorded,
             since they are a different workload size).
          3. Counterfactual: every available tier is executed `repeats_per_measurement` times;
             the median wall-clock latency is the observation.
          4. Calibration error uses latency only (secondary telemetry is MODELED_PRIOR).
          5. Regret uses measured latency only: U_t = L_min / L_t, so U_best = 1.
        """
        classes = [
            WorkloadClass.INT8_DOT_PRODUCT,
            WorkloadClass.FP32_VECTOR_MATH,
            WorkloadClass.MEMORY_BOUND_STREAM,
            WorkloadClass.BRANCH_HEAVY_CONTROL,
            WorkloadClass.MIXED_COMPUTE_MEMORY,
        ]
        available_tiers = [
            tid for tid, adapter in self.adapters.items()
            if adapter.detect().physically_present and adapter.available()
        ]

        results_by_class: Dict[str, Any] = {}
        evidence_by_class: Dict[str, Dict[str, Dict[str, Any]]] = {}
        rerouted = 0

        for wc in classes:
            evidence_by_class[wc.value] = {}
            for i in range(iterations_per_class):
                wl = WorkloadSynthesizer.generate(wc, size=workload_size, workload_id=f"calib_{wc.value}_{i}")

                # 1. Predict first (no leakage of the current observation)
                decision = self.triad_controller.placement_engine.evaluate_placement(
                    workload_id=wl.workload_id,
                    action={"action_id": f"ACT_{wc.value}", "type": wc.value},
                    live_state=create_default_live_state(),
                    mode=ExecutionMode.BALANCED_MODE,
                )
                proposed_tier = ExecutionTierID(decision.selected_tier)
                pred_lat_by_tier = {t: self.cost_model.predict_latency(wc, t) for t in available_tiers}

                # 2. Confidence-aware routing
                routed_tier, _, _ = self.confidence_router.evaluate_routing(
                    wl, proposed_tier, record_probe_observation=False
                )
                if routed_tier not in available_tiers:
                    routed_tier = proposed_tier
                if routed_tier != proposed_tier:
                    rerouted += 1
                selected_tier = routed_tier

                # 3. Counterfactual measurement on every available tier
                tier_lat: Dict[ExecutionTierID, float] = {}
                tier_spread: Dict[ExecutionTierID, float] = {}
                tier_telem: Dict[ExecutionTierID, Dict[str, Any]] = {}
                for t in available_tiers:
                    lats = []
                    telem: Dict[str, Any] = {}
                    for _ in range(max(1, repeats_per_measurement)):
                        _, lat, telem = TaxonomyExecutionEngine.execute_workload(wl, t, self.adapters)
                        lats.append(lat)
                    med = float(statistics.median(lats))
                    tier_lat[t] = med
                    tier_spread[t] = max(lats) - min(lats)
                    tier_telem[t] = telem
                    self.cost_model.record_observation(wc, t, med)
                    evidence_by_class[wc.value][t.value] = {
                        "latency_evidence": telem["latency_evidence"],
                        "telemetry_evidence": telem["telemetry_evidence"],
                        "distinct_code_path": telem["distinct_code_path"],
                    }

                # 4. Calibration error (latency only)
                obs_lat = tier_lat[selected_tier]
                self.calibration_tracker.record_prediction(
                    workload_id=wl.workload_id,
                    workload_class=wc,
                    tier=selected_tier,
                    pred_lat=pred_lat_by_tier[selected_tier],
                    obs_lat=obs_lat,
                    pred_uma=decision.predicted_UMA_pressure,
                    obs_uma=tier_telem[selected_tier]["uma_pressure"],
                    pred_temp=decision.predicted_temperature_delta_c,
                    obs_temp=tier_telem[selected_tier]["temp_delta_c"],
                    secondary_evidence=tier_telem[selected_tier]["telemetry_evidence"],
                )

                # 5. Regret on measured latency only, over tiers with a distinct code path.
                #    T1/T2 paths that run the identical Python loop as T0 differ only by timer
                #    noise, so they are excluded from the evaluated set (the selected tier is
                #    always included).
                comparable = [
                    t for t in available_tiers
                    if tier_telem[t]["distinct_code_path"] or t == selected_tier
                ]
                comp_lat = {t: tier_lat[t] for t in comparable}
                best_tier = min(comp_lat, key=lambda k: comp_lat[k])
                l_min = comp_lat[best_tier]
                utilities = {t: l_min / max(l, EPSILON) for t, l in comp_lat.items()}
                self.regret_tracker.record_decision(
                    workload_id=wl.workload_id,
                    workload_class=wc,
                    selected_tier=selected_tier,
                    selected_utility=utilities[selected_tier],
                    best_tier=best_tier,
                    best_utility=utilities[best_tier],
                    selected_latency_ms=obs_lat,
                    best_latency_ms=l_min,
                    evaluated_tiers=comparable,
                    noise_floor_ms=max(tier_spread[selected_tier], tier_spread[best_tier]),
                )

            # Summarize class empirical stats
            results_by_class[wc.value] = {
                t.value: self.cost_model.get_distribution_stats(wc, t).to_dict()
                for t in available_tiers
            }

        evidence_counts = collections.Counter(
            v["latency_evidence"] for per_cls in evidence_by_class.values() for v in per_cls.values()
        )

        report: Dict[str, Any] = {
            "phase": "CrystalSec3_Phase5_1_Calibration",
            "evidence_revision": "PHASE_5_2_EVIDENCE_INTEGRITY",
            "timestamp": time.time(),
            "iterations_per_class": iterations_per_class,
            "workload_size": workload_size,
            "repeats_per_measurement": repeats_per_measurement,
            "classes_evaluated": [wc.value for wc in classes],
            "empirical_distributions": results_by_class,
            "tier_evidence_by_class": evidence_by_class,
            "latency_evidence_counts": dict(evidence_counts),
            "telemetry_evidence": EvidenceLabel.MODELED_PRIOR.value,
            "calibration_metrics": self.calibration_tracker.get_calibration_metrics(),
            "regret_metrics": self.regret_tracker.get_regret_metrics(),
            "regret_basis": "MEASURED_LATENCY_ONLY (U_t = L_min / L_t) over tiers with a distinct code path",
            "confidence_router_probes": self.confidence_router.probe_executions_count,
            "confidence_router_reroutes": rerouted,
            "unsupported_claims": [
                "No T1/T2 number in empirical_distributions measures AVX-512/VNNI or Iris Xe hardware: "
                "those paths are HOST_CPU_EMULATED Python.",
                "UMA / power / thermal values are MODELED_PRIORs, not telemetry.",
            ],
            "vital_max_hp_invariant": VITAL_MAX_HP == 6,
        }

        if include_hardware_evidence:
            probe = hardware_probe or HardwareEvidenceProbe()
            report["hardware_evidence"] = probe.run_sdf_raymarch_sweep()

        return report


# Global singleton instance for calibration and API endpoints
GLOBAL_PLACEMENT_CALIBRATION_CONTROLLER = PlacementCalibrationController()
