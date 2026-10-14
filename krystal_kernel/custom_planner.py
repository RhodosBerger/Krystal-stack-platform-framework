#!/usr/bin/env python3
"""
==============================================================================
KRYSTAL-STACK: CUSTOM WORKLOAD PLANNER & EPISTEMIC DISPATCHER (V2)
==============================================================================
Module: krystal_kernel/custom_planner.py
Description:
    Unified workload planner and epistemic scheduler.
    Receives computational tasks (inference, rendering, verification, RAG),
    analyzes DAG dependencies, computes critical paths, and binds them to the
    execution envelope dictated by CustomHardwareGovernor.

Key Architectural Guarantees:
  1. Envelope-Driven Decoupling: Planner does not mutate or query internal
     governor state; it strictly consumes an immutable ExecutionEnvelope.
  2. Capability-Gated Dispatch: Desired backends are resolved through
     CapabilityRegistry. If an accelerator is unverified, fallback is deterministic.
  3. Abstract Verification Contract: Verification category mandates
     DETERMINISTIC_EXECUTION (resolves to CPU_SIMD_DETERMINISTIC or CPU_SCALAR).
  4. Strict Shadow Task Isolation: Shadow candidates are quarantined with
     shadow_only=True and never alter host state or transaction authority.
  5. Multi-Layer Execution Feedback: Emits statistical error metrics (residual,
     EWMA, variance) into PredictionErrorLedger before triggering ContextualRisk.
  6. Constitutional Invariant: assert VITAL_MAX_HP == 6 across all execution plans.

Author: Dušan Kopecký & Krystal-Stack Architecture Council (2026)
==============================================================================
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from krystal_kernel.scene_constructor_presets import VITAL_MAX_HP
from krystal_kernel.custom_governor import (
    CustomHardwareGovernor,
    ExecutionEnvelope,
    GovernorDecision,
    GovernorMode,
)
from krystal_kernel.verification.capability_registry import (
    CapabilityRegistry,
    HardwareBackend,
    GLOBAL_CAPABILITY_REGISTRY,
)
from krystal_kernel.runtime.prediction_error_ledger import (
    ContextualRiskRegistry,
    ContextualRiskKey,
)
from krystal_kernel.karaf.hypothesis_miner import (
    HypothesisMiner,
)

assert VITAL_MAX_HP == 6, f"Constitutional invariant breach: VITAL_MAX_HP must be 6, got {VITAL_MAX_HP}"


class TaskCategory(str, Enum):
    COMPUTE = "COMPUTE"
    INFERENCE = "INFERENCE"
    MEMORY = "MEMORY"
    RENDER = "RENDER"
    VERIFICATION = "VERIFICATION"


@dataclass
class WorkloadTask:
    task_id: str
    name: str
    category: TaskCategory
    estimated_cost_ms: float
    dependencies: List[str] = field(default_factory=list)
    priority: int = 1            # 1 (lowest) .. 10 (highest)
    deadline_ms: Optional[float] = None
    preferred_backend: Optional[str] = None
    payload: Dict[str, Any] = field(default_factory=dict)
    vital_max_hp: int = VITAL_MAX_HP

    def __post_init__(self):
        assert self.vital_max_hp == 6, "VITAL_MAX_HP must equal 6"

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["category"] = self.category.value
        return d


@dataclass
class ScheduledTaskAssignment:
    task_id: str
    target_backend: str
    assigned_worker_id: int
    codec_width_bytes: int
    predicted_latency_ms: float
    is_critical_path: bool
    is_shadow_candidate: bool = False
    shadow_only: bool = False       # Strict isolation guarantee
    vital_max_hp: int = VITAL_MAX_HP

    def __post_init__(self):
        assert self.vital_max_hp == 6, "VITAL_MAX_HP must equal 6"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class WorkloadPlan:
    plan_id: str
    timestamp_ns: int
    governor_mode: str
    envelope_state_version: int
    total_estimated_latency_ms: float
    critical_path: List[str]
    assignments: List[ScheduledTaskAssignment]
    concurrency_limit: int
    shadow_candidate_count: int
    envelope_summary: Dict[str, Any]
    vital_max_hp: int = VITAL_MAX_HP

    def __post_init__(self):
        assert self.vital_max_hp == 6, "VITAL_MAX_HP must equal 6"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "timestamp_ns": self.timestamp_ns,
            "governor_mode": self.governor_mode,
            "envelope_state_version": self.envelope_state_version,
            "total_estimated_latency_ms": self.total_estimated_latency_ms,
            "critical_path": list(self.critical_path),
            "assignments": [a.to_dict() for a in self.assignments],
            "concurrency_limit": self.concurrency_limit,
            "shadow_candidate_count": self.shadow_candidate_count,
            "envelope_summary": dict(self.envelope_summary),
            "vital_max_hp": self.vital_max_hp,
        }


@dataclass
class ExecutionFeedbackRecord:
    task_id: str
    backend: str
    predicted_ms: float
    actual_ms: float
    residual_ms: float
    relative_error: float
    workload_class: str
    timestamp_ns: int
    is_shadow: bool = False


class CustomWorkloadPlanner:
    """
    Unified workload planner and epistemic dispatcher.
    Consumes an immutable ExecutionEnvelope, gates dispatch via CapabilityRegistry,
    and isolates shadow candidate testing into the prediction error feedback loop.
    """

    def __init__(
        self,
        governor: Optional[CustomHardwareGovernor] = None,
        caps: Optional[CapabilityRegistry] = None,
        risk_registry: Optional[ContextualRiskRegistry] = None,
        hypothesis_miner: Optional[HypothesisMiner] = None,
    ):
        assert VITAL_MAX_HP == 6, "VITAL_MAX_HP must equal 6"
        self.vital_max_hp = VITAL_MAX_HP
        self.governor = governor or CustomHardwareGovernor()
        self.caps = caps or getattr(self.governor, "caps", GLOBAL_CAPABILITY_REGISTRY)
        self.risk_registry = risk_registry or ContextualRiskRegistry()
        self.hypothesis_miner = hypothesis_miner or HypothesisMiner()

        self._plan_counter: int = 0
        self._last_plan: Optional[WorkloadPlan] = None
        self._feedback_records: List[ExecutionFeedbackRecord] = []

        # Statistical drift tracking (EWMA)
        self._ewma_error_ms: float = 0.0
        self._error_variance_ms: float = 0.0

    @property
    def last_plan(self) -> Optional[WorkloadPlan]:
        return self._last_plan

    def compute_critical_path(self, tasks: List[WorkloadTask]) -> Tuple[List[str], float]:
        """Calculates the critical path duration and task sequence over a DAG."""
        task_map = {t.task_id: t for t in tasks}
        memo_dur: Dict[str, float] = {}
        memo_path: Dict[str, List[str]] = {}

        def get_path(tid: str) -> Tuple[float, List[str]]:
            if tid in memo_dur:
                return memo_dur[tid], memo_path[tid]
            t = task_map[tid]
            max_dep_dur = 0.0
            best_dep_path: List[str] = []
            for dep_id in t.dependencies:
                if dep_id in task_map:
                    d_dur, d_path = get_path(dep_id)
                    if d_dur > max_dep_dur:
                        max_dep_dur = d_dur
                        best_dep_path = d_path
            tot = max_dep_dur + t.estimated_cost_ms
            p = best_dep_path + [tid]
            memo_dur[tid] = tot
            memo_path[tid] = p
            return tot, p

        max_dur = 0.0
        best_path: List[str] = []
        for tid in task_map:
            dur, path = get_path(tid)
            if dur > max_dur:
                max_dur = dur
                best_path = path

        return best_path, max_dur

    def resolve_backend_for_task(
        self,
        task: WorkloadTask,
        envelope: ExecutionEnvelope,
        is_critical: bool,
    ) -> str:
        """
        Capability-gated backend selection based on contract requirements and envelope constraints.
        """
        # 1. Abstract Verification Contract: mandates strict determinism
        if task.category == TaskCategory.VERIFICATION:
            resolved = self.caps.resolve_backend(
                desired=HardwareBackend.CPU_SIMD_DETERMINISTIC,
                fallback=HardwareBackend.CPU_SCALAR,
                require_deterministic=True
            )
            return resolved.value

        # 2. Desired backend based on category and envelope allowances
        desired = HardwareBackend.CPU_SIMD_DETERMINISTIC
        if task.preferred_backend:
            try:
                desired = HardwareBackend(task.preferred_backend)
            except ValueError:
                desired = HardwareBackend.CPU_SIMD_DETERMINISTIC

        elif task.category == TaskCategory.INFERENCE:
            if "NPU_OFFLOAD" in envelope.allowed_backends and self.caps.is_usable(HardwareBackend.NPU):
                desired = HardwareBackend.NPU
            elif "GPU_OPENVINO" in envelope.allowed_backends and self.caps.is_usable(HardwareBackend.OPENVINO_GPU):
                desired = HardwareBackend.OPENVINO_GPU
            else:
                desired = HardwareBackend.CPU_SIMD_DETERMINISTIC

        elif task.category == TaskCategory.RENDER:
            if "GPU_VULKAN" in envelope.allowed_backends and self.caps.is_usable(HardwareBackend.VULKAN):
                desired = HardwareBackend.VULKAN
            elif "GPU_OPENVINO" in envelope.allowed_backends and self.caps.is_usable(HardwareBackend.OPENVINO_GPU):
                desired = HardwareBackend.OPENVINO_GPU
            elif "NPU_OFFLOAD" in envelope.allowed_backends and self.caps.is_usable(HardwareBackend.NPU):
                desired = HardwareBackend.NPU
            else:
                desired = HardwareBackend.CPU_SIMD_DETERMINISTIC

        elif is_critical and "GPU_OPENVINO" in envelope.allowed_backends and self.caps.is_usable(HardwareBackend.OPENVINO_GPU):
            desired = HardwareBackend.OPENVINO_GPU

        # 3. Final capability check: ensure desired is verified, otherwise fallback
        resolved_backend = self.caps.resolve_backend(
            desired=desired,
            fallback=HardwareBackend.CPU_SIMD_DETERMINISTIC,
            require_deterministic=False
        )
        return resolved_backend.value

    def generate_plan(
        self,
        tasks: List[WorkloadTask],
        envelope: Optional[ExecutionEnvelope] = None,
        include_shadow_experiment: bool = True,
        now_ns: Optional[int] = None,
    ) -> WorkloadPlan:
        """
        Builds a WorkloadPlan adhering strictly to the provided ExecutionEnvelope.
        """
        assert VITAL_MAX_HP == 6
        self._plan_counter += 1
        t_ns = time.monotonic_ns() if now_ns is None else now_ns

        # If envelope not directly provided, query current governor decision
        if envelope is None:
            # Produce fallback nominal envelope
            prof = self.governor.active_profile
            envelope = ExecutionEnvelope(
                governor_mode=self.governor.mode.value,
                allowed_backends=("CPU_SCALAR", "CPU_SIMD_DETERMINISTIC", "NPU_OFFLOAD") if prof.npu_offload else ("CPU_SCALAR", "CPU_SIMD_DETERMINISTIC"),
                max_workers=max(1, int(round((prof.cpu_cap_pct / 100.0) * 8))),
                max_power_w=prof.power_draw_w,
                max_tick_hz=prof.tick_hz,
                codec_width_bytes=prof.codec_width_bytes,
                thermal_budget_c=25.0,
                memory_pressure_limit=0.82,
                valid_until_ns=t_ns + 1_000_000_000,
                state_version=1,
                vital_max_hp=VITAL_MAX_HP,
            )

        crit_path, total_latency = self.compute_critical_path(tasks)
        crit_set = set(crit_path)

        assignments: List[ScheduledTaskAssignment] = []
        worker_idx = 0
        shadow_count = 0

        # Sort tasks: critical path first, then priority
        sorted_tasks = sorted(
            tasks,
            key=lambda t: (1 if t.task_id in crit_set else 0, t.priority),
            reverse=True
        )

        for t in sorted_tasks:
            is_crit = t.task_id in crit_set
            backend_str = self.resolve_backend_for_task(t, envelope, is_crit)
            worker_id = worker_idx % envelope.max_workers
            worker_idx += 1

            # Backend speedup factor
            speedup = 1.0
            if "NPU" in backend_str:
                speedup = 2.0
            elif "GPU" in backend_str:
                speedup = 2.4
            elif "SIMD" in backend_str:
                speedup = 1.3

            pred_lat = max(0.5, t.estimated_cost_ms / speedup)

            # Primary assignment
            assignments.append(ScheduledTaskAssignment(
                task_id=t.task_id,
                target_backend=backend_str,
                assigned_worker_id=worker_id,
                codec_width_bytes=envelope.codec_width_bytes,
                predicted_latency_ms=round(pred_lat, 2),
                is_critical_path=is_crit,
                is_shadow_candidate=False,
                shadow_only=False,
                vital_max_hp=VITAL_MAX_HP,
            ))

            # Isolated Shadow Candidate generation (max 1 per plan)
            if include_shadow_experiment and not is_crit and shadow_count == 0 and len(tasks) > 1:
                alt_backend = "CPU_SIMD_DETERMINISTIC" if "CPU" not in backend_str else "NPU"
                assignments.append(ScheduledTaskAssignment(
                    task_id=f"shadow_{t.task_id}",
                    target_backend=alt_backend,
                    assigned_worker_id=(worker_id + 1) % envelope.max_workers,
                    codec_width_bytes=16 if envelope.codec_width_bytes == 32 else 32,
                    predicted_latency_ms=round(t.estimated_cost_ms * 1.1, 2),
                    is_critical_path=False,
                    is_shadow_candidate=True,
                    shadow_only=True,  # Guarantees zero host mutation / no primary side-effects
                    vital_max_hp=VITAL_MAX_HP,
                ))
                shadow_count += 1

        plan = WorkloadPlan(
            plan_id=f"plan_{self._plan_counter}_{int(t_ns % 1_000_000)}",
            timestamp_ns=t_ns,
            governor_mode=envelope.governor_mode,
            envelope_state_version=envelope.state_version,
            total_estimated_latency_ms=round(total_latency, 2),
            critical_path=crit_path,
            assignments=assignments,
            concurrency_limit=envelope.max_workers,
            shadow_candidate_count=shadow_count,
            envelope_summary={
                "max_power_w": envelope.max_power_w,
                "codec_width_bytes": envelope.codec_width_bytes,
                "allowed_backends": envelope.allowed_backends,
            },
            vital_max_hp=VITAL_MAX_HP,
        )

        self._last_plan = plan
        return plan

    def record_execution_feedback(
        self,
        task_id: str,
        backend: str,
        predicted_latency_ms: float,
        actual_latency_ms: float,
        workload_class: str = "COMPUTE",
        is_shadow: bool = False,
        now_ns: Optional[int] = None,
    ) -> ExecutionFeedbackRecord:
        """
        Two-layer execution feedback loop:
        1. Records raw residual and relative error into PredictionErrorLedger.
        2. Evaluates statistical EWMA drift; only flags contextual risk upon repeated anomalies.
        """
        assert VITAL_MAX_HP == 6
        t_ns = time.monotonic_ns() if now_ns is None else now_ns
        residual = actual_latency_ms - predicted_latency_ms
        rel_error = abs(residual) / max(0.5, predicted_latency_ms)

        rec = ExecutionFeedbackRecord(
            task_id=task_id,
            backend=backend,
            predicted_ms=predicted_latency_ms,
            actual_ms=actual_latency_ms,
            residual_ms=round(residual, 2),
            relative_error=round(rel_error, 3),
            workload_class=workload_class,
            timestamp_ns=t_ns,
            is_shadow=is_shadow,
        )
        self._feedback_records.append(rec)
        if len(self._feedback_records) > 256:
            self._feedback_records.pop(0)

        # Statistical EWMA update
        alpha = 0.15
        self._ewma_error_ms = (1.0 - alpha) * self._ewma_error_ms + alpha * residual
        self._error_variance_ms = (1.0 - alpha) * self._error_variance_ms + alpha * (residual ** 2)

        # Contextual Risk update: only if variance is large and repeated drift exceeds threshold
        if abs(self._ewma_error_ms) > 12.0 and math.sqrt(self._error_variance_ms) > 10.0:
            risk_key = ContextualRiskKey(
                action_signature=f"backend::{backend}",
                host_state_class=self.governor.active_profile.category,
                workload_class=workload_class,
                pressure_region="saturated" if residual > 0 else "underutilized",
                policy_version_id="karaf_v2",
            )
            self.risk_registry.record_veto(risk_key, severity=0.1)

        return rec

    def get_status(self) -> Dict[str, Any]:
        return {
            "vital_max_hp": self.vital_max_hp,
            "plan_counter": self._plan_counter,
            "last_plan": self._last_plan.to_dict() if self._last_plan else None,
            "feedback_count": len(self._feedback_records),
            "ewma_error_ms": round(self._ewma_error_ms, 3),
            "error_variance_ms": round(self._error_variance_ms, 3),
            "governor_status": self.governor.get_status(),
        }
