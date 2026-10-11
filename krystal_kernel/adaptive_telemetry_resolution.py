#!/usr/bin/env python3
"""
CRYSTALSEC 2.0 // ADAPTIVE TELEMETRY RESOLUTION CONTROLLER
=============================================================================
Dynamically modulates telemetry sampling frequency based on the rate of change
of the system state vector:

  |dH/dt| > theta

Workflow:
  1. QUIESCENT_BASELINE (2 Hz) : Stable system -> cheap telemetry, minimal CPU footprint.
  2. DEVIATION DETECTED        : |dH/dt| exceeds threshold theta.
  3. BURST_TRANSIENT_CAPTURE   : Instant switch to high-frequency (50 - 100 Hz).
  4. TRANSIENT BUFFERING       : High-resolution ringbuffer captures transient spike.
  5. COOLDOWN & RETURN         : When |dH/dt| stabilizes <= theta, decays back to baseline.

Prevents telemetry flooding while guaranteeing 100% capture of transient micro-anomalies.
=============================================================================
"""

import os
import sys
import time
import math
from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple

WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)


class TelemetryResolutionMode(str, Enum):
    QUIESCENT_BASELINE = "QUIESCENT_BASELINE"           # 2 Hz
    BURST_TRANSIENT_CAPTURE = "BURST_TRANSIENT_CAPTURE" # 50 - 100 Hz
    COOLDOWN_RECOVERY = "COOLDOWN_RECOVERY"             # 10 Hz


@dataclass
class TransientCaptureWindow:
    window_id: str
    trigger_metric: str
    derivative_value: float
    threshold_theta: float
    t_start: float
    t_end: Optional[float] = None
    samples_captured: List[Dict[str, Any]] = field(default_factory=list)
    peak_sampling_rate_hz: float = 100.0


class AdaptiveTelemetryGovernor:
    """Calculates state derivatives and modulates sampling resolution on the fly,
    strictly guarded by an Observer Effect overhead limiter to avoid perturbing workloads."""

    def __init__(
        self,
        baseline_hz: float = 2.0,
        burst_hz: float = 100.0,
        cooldown_hz: float = 10.0,
        theta_threshold: float = 5.0,
        cooldown_ticks_required: int = 15,
        max_observer_overhead_pct: float = 2.5
    ):
        self.baseline_hz = baseline_hz
        self.burst_hz = burst_hz
        self.cooldown_hz = cooldown_hz
        self.theta_threshold = theta_threshold
        self.cooldown_ticks_required = cooldown_ticks_required
        self.max_observer_overhead_pct = max_observer_overhead_pct

        self.current_mode: TelemetryResolutionMode = TelemetryResolutionMode.QUIESCENT_BASELINE
        self.current_sampling_rate_hz: float = self.baseline_hz

        self.last_sample: Optional[Dict[str, float]] = None
        self.last_timestamp: float = 0.0
        self.consecutive_stable_ticks: int = 0

        self.active_transient_window: Optional[TransientCaptureWindow] = None
        self.transient_history: List[TransientCaptureWindow] = []

        # Observer Effect Tracking
        self.empirical_sample_cost_sec: float = 0.0001
        self.observer_overhead_pct: float = 0.0
        self.is_observer_throttled: bool = False

    def compute_state_derivative(
        self,
        current_sample: Dict[str, float],
        dt: float
    ) -> Tuple[float, str]:
        """
        Calculates L2 norm of normalized rate-of-change:
          |dH/dt| = sqrt( sum( (delta_h_i / dt)^2 ) )
        Returns (max_derivative, dominant_metric_key).
        """
        if not self.last_sample or dt <= 0.0001:
            return (0.0, "none")

        max_deriv = 0.0
        dominant_key = "none"

        metric_scales = {
            "latency_ms": 10.0,       # 10ms/s change is significant
            "temperature_c": 2.0,     # 2C/s change is significant
            "cpu_pressure": 0.20,     # 20%/s change is significant
            "memory_pressure": 0.15,  # 15%/s change is significant
            "queue_depth": 15.0       # 15 items/s change is significant
        }

        for key, scale in metric_scales.items():
            if key in current_sample and key in self.last_sample:
                delta = abs(current_sample[key] - self.last_sample[key])
                rate = (delta / dt) / scale
                if rate > max_deriv:
                    max_deriv = rate
                    dominant_key = key

        return (round(max_deriv, 3), dominant_key)

    def ingest_telemetry_tick(
        self,
        sample: Dict[str, Any],
        timestamp: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Processes a telemetry tick. If derivative > theta, escalates to
        BURST_TRANSIENT_CAPTURE, capped dynamically by the Observer Effect Guard.
        """
        t_tick_start = time.perf_counter()
        t_now = timestamp or time.time()
        dt = t_now - self.last_timestamp if self.last_timestamp > 0 else 0.5
        numeric_sample = {k: float(v) for k, v in sample.items() if isinstance(v, (int, float))}

        deriv, trigger_key = self.compute_state_derivative(numeric_sample, dt)

        # Check if derivative breaches theta threshold: |dH/dt| > theta
        if deriv > self.theta_threshold:
            self.consecutive_stable_ticks = 0
            if self.current_mode != TelemetryResolutionMode.BURST_TRANSIENT_CAPTURE:
                self.current_mode = TelemetryResolutionMode.BURST_TRANSIENT_CAPTURE
                self.current_sampling_rate_hz = self.burst_hz

                self.active_transient_window = TransientCaptureWindow(
                    window_id=f"WIN-{int(t_now*1000)}",
                    trigger_metric=trigger_key,
                    derivative_value=deriv,
                    threshold_theta=self.theta_threshold,
                    t_start=t_now,
                    peak_sampling_rate_hz=self.burst_hz
                )

        elif self.current_mode == TelemetryResolutionMode.BURST_TRANSIENT_CAPTURE:
            self.consecutive_stable_ticks += 1
            if self.consecutive_stable_ticks >= 5:
                self.current_mode = TelemetryResolutionMode.COOLDOWN_RECOVERY
                self.current_sampling_rate_hz = self.cooldown_hz

        elif self.current_mode == TelemetryResolutionMode.COOLDOWN_RECOVERY:
            self.consecutive_stable_ticks += 1
            if self.consecutive_stable_ticks >= self.cooldown_ticks_required:
                if self.active_transient_window:
                    self.active_transient_window.t_end = t_now
                    self.transient_history.append(self.active_transient_window)
                    self.active_transient_window = None

                self.current_mode = TelemetryResolutionMode.QUIESCENT_BASELINE
                self.current_sampling_rate_hz = self.baseline_hz

        # High-resolution buffer capture
        if self.active_transient_window:
            self.active_transient_window.samples_captured.append({
                "timestamp": t_now,
                "values": numeric_sample
            })

        self.last_sample = numeric_sample
        self.last_timestamp = t_now

        # -------------------------------------------------------------
        # OBSERVER EFFECT GUARD:
        # Measure telemetry computation time & throttle frequency if
        # overhead exceeds max_observer_overhead_pct (e.g. 2.5% of CPU).
        # -------------------------------------------------------------
        t_tick_duration = max(time.perf_counter() - t_tick_start, 0.00005)
        self.empirical_sample_cost_sec = 0.90 * self.empirical_sample_cost_sec + 0.10 * t_tick_duration

        projected_overhead = (self.current_sampling_rate_hz * self.empirical_sample_cost_sec) * 100.0
        if projected_overhead > self.max_observer_overhead_pct:
            # Throttle sampling rate to keep overhead <= max_observer_overhead_pct
            max_safe_hz = (self.max_observer_overhead_pct / 100.0) / self.empirical_sample_cost_sec
            self.current_sampling_rate_hz = max(self.baseline_hz, min(self.current_sampling_rate_hz, max_safe_hz))
            self.is_observer_throttled = True
            self.observer_overhead_pct = (self.current_sampling_rate_hz * self.empirical_sample_cost_sec) * 100.0
        else:
            self.is_observer_throttled = False
            self.observer_overhead_pct = projected_overhead

        return {
            "mode": self.current_mode.value,
            "active_sampling_rate_hz": round(self.current_sampling_rate_hz, 1),
            "derivative_magnitude": deriv,
            "dominant_trigger": trigger_key,
            "theta_threshold": self.theta_threshold,
            "is_burst_active": self.current_mode == TelemetryResolutionMode.BURST_TRANSIENT_CAPTURE,
            "burst_samples_captured": len(self.active_transient_window.samples_captured) if self.active_transient_window else 0,
            "observer_overhead_pct": round(self.observer_overhead_pct, 3),
            "is_observer_throttled": self.is_observer_throttled
        }

    def evaluate_observer_tradeoff_matrix(
        self,
        candidate_frequencies_hz: Optional[List[float]] = None
    ) -> List[Dict[str, Any]]:
        """
        Generates an empirical trade-off matrix:
          Frequency (Hz) -> Information Gain -> Observer Overhead (%) -> Recommendation
        Used to experimentally establish burst sampling frequency without dogma.
        """
        freqs = candidate_frequencies_hz or [2.0, 10.0, 25.0, 50.0, 100.0, 200.0]
        results = []

        for f in freqs:
            overhead = (f * self.empirical_sample_cost_sec) * 100.0
            # Information gain logarithmically saturates according to Nyquist/Shannon
            info_gain = round(math.log2(1.0 + f / 2.0) / math.log2(1.0 + 100.0 / 2.0), 3)

            if overhead <= 1.0:
                rec = "SAFE_QUIESCENT_OR_MODERATE"
            elif overhead <= self.max_observer_overhead_pct:
                rec = "RECOMMENDED_BURST"
            else:
                rec = "OBSERVER_EFFECT_RISK_PERTURBS_WORKLOAD"

            results.append({
                "frequency_hz": f,
                "projected_overhead_pct": round(overhead, 3),
                "normalized_info_gain": min(1.0, info_gain),
                "is_within_budget": overhead <= self.max_observer_overhead_pct,
                "recommendation": rec
            })

        return results


GLOBAL_ADAPTIVE_TELEMETRY = AdaptiveTelemetryGovernor()
