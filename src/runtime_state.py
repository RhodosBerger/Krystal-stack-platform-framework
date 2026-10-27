#!/usr/bin/env python3
"""
==============================================================================
KRYSTAL-STACK: RUNTIME STATE & CONTROLLED INFERENCE WORKLOAD (M1)
==============================================================================
Module: src/runtime_state.py
Canonical 6D State Vector:
    S = [C, M, L, T, E, Q]
        C: compute pressure   [0.0, 1.0]
        M: memory pressure    [0.0, 1.0]
        L: latency pressure   [0.0, 1.0]
        T: thermal pressure   [0.0, 1.0]
        E: energy pressure    [0.0, 1.0]
        Q: queue pressure     [0.0, 1.0]
Constitutional Invariant: VITAL_MAX_HP == 6
==============================================================================
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Tuple


@dataclass
class RuntimeStateVector:
    """Canonical 6-Dimensional Normalized Operational State Vector S = [C, M, L, T, E, Q]."""
    c_compute: float = 0.50
    m_memory: float = 0.40
    l_latency: float = 0.35
    t_thermal: float = 0.45
    e_energy: float = 0.40
    q_queue: float = 0.20

    def to_list(self) -> List[float]:
        """Serializes the 6D vector in canonical order [C, M, L, T, E, Q]."""
        return [
            round(self.c_compute, 4),
            round(self.m_memory, 4),
            round(self.l_latency, 4),
            round(self.t_thermal, 4),
            round(self.e_energy, 4),
            round(self.q_queue, 4),
        ]

    @classmethod
    def from_list(cls, values: List[float]) -> RuntimeStateVector:
        """Instantiates 6D vector from a list, padding or truncating to 6."""
        if len(values) < 6:
            padded = values + [0.0] * (6 - len(values))
        else:
            padded = values[:6]
        return cls(
            c_compute=cls._clamp(padded[0]),
            m_memory=cls._clamp(padded[1]),
            l_latency=cls._clamp(padded[2]),
            t_thermal=cls._clamp(padded[3]),
            e_energy=cls._clamp(padded[4]),
            q_queue=cls._clamp(padded[5]),
        )

    @staticmethod
    def _clamp(v: float) -> float:
        return max(0.0, min(1.0, float(v)))

    def distance_to(self, other: RuntimeStateVector) -> float:
        """Euclidean distance in 6D normalized state space."""
        v1 = self.to_list()
        v2 = other.to_list()
        return math.sqrt(sum((v1[i] - v2[i]) ** 2 for i in range(6)))

    def mae_to(self, other: RuntimeStateVector) -> float:
        """Mean Absolute Error across the 6 normalized dimensions."""
        v1 = self.to_list()
        v2 = other.to_list()
        return sum(abs(v1[i] - v2[i]) for i in range(6)) / 6.0

    def residual_to(self, other: RuntimeStateVector) -> List[float]:
        """Directional residual vector R = Actual(self) - Predicted(other)."""
        v1 = self.to_list()
        v2 = other.to_list()
        return [round(v1[i] - v2[i], 4) for i in range(6)]


@dataclass
class ControlledInferenceWorkload:
    """
    Controlled Synthetic / AI Inference Workload for Milestone M1.
    Allows systematic variation of:
      - batch_size (1 to 64)
      - worker_count (1 to 8)
      - device_placement ('CPU_AVX512', 'GPU_IRIS_XE')
      - precision ('FP32', 'FP16', 'INT8')
      - queue_depth (0 to 64)
    """
    batch_size: int = 16
    worker_count: int = 4
    device_placement: str = "CPU_AVX512"
    precision: str = "FP16"
    queue_depth: int = 8
    target_fps: float = 60.0

    def compute_workload_state(self, ambient_temp_c: float = 25.0) -> RuntimeStateVector:
        """
        Deterministically evaluates physical state vector S = [C, M, L, T, E, Q]
        given workload configuration and physical constants.
        """
        # 1. Compute pressure: scaled by batch size and worker saturation
        compute_load = (self.batch_size / 64.0) * 0.60 + (self.worker_count / 8.0) * 0.40
        if self.precision == "INT8":
            compute_load *= 0.65
        elif self.precision == "FP32":
            compute_load *= 1.30

        # 2. Memory pressure: primary driver is batch size and device placement
        memory_load = (self.batch_size / 64.0) * 0.70 + (self.queue_depth / 64.0) * 0.30
        if self.device_placement == "GPU_IRIS_XE":
            memory_load *= 1.15 # UMA shared memory pressure

        # 3. Latency pressure: increases exponentially with batch size and queue depth
        latency_load = (self.batch_size / 64.0) * 0.50 + (self.queue_depth / 32.0) * 0.50
        if self.worker_count < 2:
            latency_load += 0.20 # Contention / head of line blocking

        # 4. Thermal pressure: function of workers, compute load, and ambient temperature
        thermal_base = (ambient_temp_c / 50.0) * 0.30
        thermal_load = thermal_base + (compute_load * 0.50) + (self.worker_count / 8.0) * 0.20

        # 5. Energy pressure: active power draw proportional to compute * workers
        energy_load = (compute_load * 0.60) + (self.worker_count / 8.0) * 0.40

        # 6. Queue pressure: current depth relative to ceiling
        queue_load = min(1.0, self.queue_depth / 32.0)

        return RuntimeStateVector.from_list([
            compute_load,
            memory_load,
            latency_load,
            thermal_load,
            energy_load,
            queue_load,
        ])

    def apply_action(self, batch_factor: float, worker_factor: float) -> ControlledInferenceWorkload:
        """Derives a new workload state after applying fractional control factors."""
        new_batch = max(1, min(64, int(round(self.batch_size * batch_factor))))
        new_workers = max(1, min(8, int(round(self.worker_count * worker_factor))))
        # Queue adjustment: reducing batch temporarily holds or clears queue
        new_queue = max(0, min(64, int(self.queue_depth * (1.0 - (1.0 - batch_factor) * 0.5))))
        return ControlledInferenceWorkload(
            batch_size=new_batch,
            worker_count=new_workers,
            device_placement=self.device_placement,
            precision=self.precision,
            queue_depth=new_queue,
            target_fps=self.target_fps,
        )
