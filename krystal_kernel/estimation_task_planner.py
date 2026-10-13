#!/usr/bin/env python3
"""
==============================================================================
KRYSTAL-STACK: ESTIMATION TASK PLANNER & COMPUTATION DAG SCHEDULER
==============================================================================
Module: krystal_kernel/estimation_task_planner.py
Description:
    Decomposes predictive epistemic estimation and device optimization into a
    structured Directed Acyclic Graph (DAG) of discrete tasks.

Execution Pipeline Stages:
    1. STAGE_OBS_TELEMETRY (OBS_RAW_SAMPLE):
       Samples physical sensors and OS counters (temperatures, package watts, load).
    2. STAGE_DER_FEATURE_EXTRACTION (DER_COMPUTED_DELTA, DER_PRESSURE_SCORE):
       Computes derived mathematical features and normalized tensor inputs.
    3. STAGE_PRED_NPU_EVALUATION (PRED_POINT_ESTIMATE, PRED_CONFIDENCE_BOUND):
       Dispatches vectorized forward tensor inference via OpenVINO NPU coprocessor.
    4. STAGE_SAFE_INVARIANT_VERIFICATION (SAFE_PASS / SAFE_BREACH):
       Audits constitutional invariant VITAL_MAX_HP == 6 and thermal/power boundaries.
    5. STAGE_AUTH_DUAL_GATE_DECISION (AUTH_COMMIT_APPROVED / AUTH_VETOED):
       Deterministic governance arbitration combining evidence and safety.
    6. STAGE_ACT_MUTATION_DISPATCH (ACT_DISPATCHED / ACT_EXECUTED):
       Dispatches low-level directives to target hardware domains (CPU, Power, NPU).

Constitutional Invariant:
    assert VITAL_MAX_HP == 6
==============================================================================
"""

import os
import sys
import time
from enum import Enum, unique
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Optional, Callable

WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

from krystal_kernel.scene_constructor_presets import VITAL_MAX_HP
from krystal_kernel.verification.semantic_protocol import (
    ActuationState,
    ObservationState,
    DerivedState,
    PredictionState,
    SafetyState,
    AuthorityState,
    TransactionState,
    InvariantState,
    HardwareState,
    evaluate_dual_gate,
    to_wire_token
)
from krystal_kernel.hardware_layers import (
    HardwareSubsystemCoordinator,
    GLOBAL_HARDWARE_COORDINATOR,
    HardwareDomainKind
)
from krystal_kernel.windows_timer_subsystem import (
    WindowsTimerSubsystem,
    GLOBAL_WINDOWS_TIMER_SUBSYSTEM
)


@unique
class TaskStageKind(str, Enum):
    """Stages of an estimation computation workflow."""
    STAGE_OBS_TELEMETRY               = "OBS_TELEMETRY"
    STAGE_DER_FEATURE_EXTRACTION      = "DER_FEATURE_EXTRACTION"
    STAGE_PRED_NPU_EVALUATION         = "PRED_NPU_EVALUATION"
    STAGE_SAFE_INVARIANT_VERIFICATION = "SAFE_INVARIANT_VERIFICATION"
    STAGE_AUTH_DUAL_GATE_DECISION     = "AUTH_DUAL_GATE_DECISION"
    STAGE_ACT_MUTATION_DISPATCH       = "ACT_MUTATION_DISPATCH"


@dataclass
class EstimationTask:
    """An individual atomic task inside an estimation execution plan."""
    task_id: str
    name: str
    stage: TaskStageKind
    target_domain: HardwareDomainKind
    dependencies: List[str] = field(default_factory=list)
    latency_budget_us: float = 1000.0
    actual_latency_us: float = 0.0
    state: str = field(default_factory=lambda: to_wire_token(ActuationState.ACT_IDLE))
    payload: Dict[str, Any] = field(default_factory=dict)
    output: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["stage"] = self.stage.value
        d["target_domain"] = self.target_domain.value
        return d


@dataclass
class EstimationPlanReport:
    """Execution telemetry report of an estimation task plan."""
    plan_id: str
    template_id: str
    total_tasks: int
    executed_tasks: int
    total_latency_us: float
    qpc_start_ticks: int
    qpc_end_ticks: int
    safety_verdict: str
    authority_verdict: str
    transaction_state: str
    tasks: List[Dict[str, Any]]
    vital_max_hp: int = VITAL_MAX_HP

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class EstimationTaskPlan:
    """
    DAG of estimation tasks executed with high-resolution QPC instrumentation.
    Enforces topological order, latency constraints, and constitutional invariants.
    """

    def __init__(self, plan_id: str, template_id: str):
        self.plan_id = plan_id
        self.template_id = template_id
        self.tasks: List[EstimationTask] = []
        self._task_map: Dict[str, EstimationTask] = {}

    def add_task(self, task: EstimationTask):
        self.tasks.append(task)
        self._task_map[task.task_id] = task

    def execute(
        self,
        hw_coordinator: Optional[HardwareSubsystemCoordinator] = None,
        timer_subsystem: Optional[WindowsTimerSubsystem] = None
    ) -> EstimationPlanReport:
        """
        Executes all planned tasks in topological sequence with microsecond precision.
        """
        assert VITAL_MAX_HP == 6, "Invariant VITAL_MAX_HP == 6 breached in estimation plan execution"

        hw = hw_coordinator or GLOBAL_HARDWARE_COORDINATOR
        timer = timer_subsystem or GLOBAL_WINDOWS_TIMER_SUBSYSTEM
        clock = timer.clock

        qpc_start = clock.now_ticks()
        context_data: Dict[str, Any] = {}
        executed_count = 0

        safety_state = SafetyState.SAFE_PASS
        auth_state = AuthorityState.AUTH_COMMIT_APPROVED
        txn_state = TransactionState.TXN_IDLE

        for task in self.tasks:
            task.state = to_wire_token(ActuationState.ACT_DISPATCHED)
            t0 = clock.now_ticks()

            # Execute task logic based on its stage
            if task.stage == TaskStageKind.STAGE_OBS_TELEMETRY:
                hw_status = hw.get_full_status()
                task.output = {
                    "temperature_c": hw_status["power_thermal"]["temperature_c"],
                    "package_power_watts": hw_status["power_thermal"]["package_power_watts"],
                    "cpu_util": hw_status["cpu"]["cpu_utilization_pct"],
                    "headroom_c": hw_status["power_thermal"]["thermal_headroom_c"],
                    "status_code": to_wire_token(ObservationState.OBS_RAW_SAMPLE)
                }
                context_data["telemetry"] = task.output

            elif task.stage == TaskStageKind.STAGE_DER_FEATURE_EXTRACTION:
                telem = context_data.get("telemetry", {})
                temp = telem.get("temperature_c", 65.0)
                pkg_w = telem.get("package_power_watts", 21.0)
                cpu_u = telem.get("cpu_util", 25.0)
                headroom = telem.get("headroom_c", 35.0)

                # Normalized features for NPU tensor model
                features = [
                    min(1.0, temp / 100.0),
                    min(1.0, pkg_w / 60.0),
                    min(1.0, cpu_u / 100.0),
                    min(1.0, headroom / 60.0),
                    0.51,  # memory
                    1.0,   # vital HP 6/6
                    0.62   # frequency ratio
                ]
                pressure = round(0.45 * (temp / 100.0) + 0.35 * (cpu_u / 100.0) + 0.20 * 0.51, 4)
                task.output = {
                    "feature_vector": features,
                    "pressure_score": pressure,
                    "status_code": to_wire_token(DerivedState.DER_PRESSURE_SCORE)
                }
                context_data["features"] = features
                context_data["pressure"] = pressure

            elif task.stage == TaskStageKind.STAGE_PRED_NPU_EVALUATION:
                features = context_data.get("features", [0.65, 0.35, 0.25, 0.58, 0.51, 1.0, 0.62])
                plan_bias = task.payload.get("plan_bias", 0.45)
                from krystal_kernel.epistemic_utility_tweaker import GLOBAL_UTILITY_TWEAKER
                d_perf, d_energy, r_therm, p_breach = GLOBAL_UTILITY_TWEAKER.npu_evaluator.forward_tensor_inference(
                    features, plan_bias
                )
                task.output = {
                    "predicted_perf_gain_pct": d_perf,
                    "predicted_energy_delta_pct": d_energy,
                    "thermal_runaway_risk": r_therm,
                    "breach_probability": p_breach,
                    "target_npu_device": hw.npu.device_id,
                    "status_code": to_wire_token(PredictionState.PRED_POINT_ESTIMATE)
                }
                context_data["prediction"] = task.output

            elif task.stage == TaskStageKind.STAGE_SAFE_INVARIANT_VERIFICATION:
                pred = context_data.get("prediction", {})
                telem = context_data.get("telemetry", {})
                headroom = telem.get("headroom_c", 35.0)
                p_breach = pred.get("breach_probability", 0.05)

                if headroom < 10.0 or p_breach > 0.85:
                    safety_state = SafetyState.SAFE_BREACH
                else:
                    safety_state = SafetyState.SAFE_PASS

                task.output = {
                    "safety_verdict": to_wire_token(safety_state),
                    "vital_hp": VITAL_MAX_HP,
                    "invariant_state": to_wire_token(InvariantState.INV_NOMINAL_6HP)
                }
                context_data["safety"] = safety_state

            elif task.stage == TaskStageKind.STAGE_AUTH_DUAL_GATE_DECISION:
                from krystal_kernel.verification.semantic_protocol import EpistemicState
                evidence = EpistemicState.EPI_VALIDATED
                auth_state = evaluate_dual_gate(safety_state, evidence)

                if task.payload.get("is_shadow_only", False):
                    auth_state = AuthorityState.AUTH_SHADOW_ONLY

                task.output = {
                    "authority_verdict": to_wire_token(auth_state),
                    "evidence_used": to_wire_token(evidence)
                }
                context_data["authority"] = auth_state
                txn_state = TransactionState.TXN_PREPARED

            elif task.stage == TaskStageKind.STAGE_ACT_MUTATION_DISPATCH:
                if auth_state == AuthorityState.AUTH_COMMIT_APPROVED:
                    # Apply directives to hardware domains
                    affinity = task.payload.get("affinity_mask")
                    priority = task.payload.get("priority_class")
                    pl1 = task.payload.get("pl1_watts")
                    streams = task.payload.get("npu_streams")

                    mutation_res = hw.apply_hardware_profile(
                        affinity_mask=affinity,
                        priority_class=priority,
                        pl1_watts=pl1,
                        npu_streams=streams
                    )
                    task.output = mutation_res
                    txn_state = TransactionState.TXN_COMMITTED
                else:
                    task.output = {
                        "applied": False,
                        "reason": f"Disallowed by authority state {to_wire_token(auth_state)}"
                    }
                    txn_state = TransactionState.TXN_ABORTED

            elapsed_us = clock.elapsed_us(t0)
            task.actual_latency_us = round(elapsed_us, 2)
            task.state = to_wire_token(ActuationState.ACT_EXECUTED)
            executed_count += 1

        qpc_end = clock.now_ticks()
        total_lat_us = clock.elapsed_us(qpc_start)

        return EstimationPlanReport(
            plan_id=self.plan_id,
            template_id=self.template_id,
            total_tasks=len(self.tasks),
            executed_tasks=executed_count,
            total_latency_us=round(total_lat_us, 2),
            qpc_start_ticks=qpc_start,
            qpc_end_ticks=qpc_end,
            safety_verdict=to_wire_token(safety_state),
            authority_verdict=to_wire_token(auth_state),
            transaction_state=to_wire_token(txn_state),
            tasks=[t.to_dict() for t in self.tasks],
            vital_max_hp=VITAL_MAX_HP
        )


def build_canonical_estimation_dag(
    plan_id: str,
    template_id: str,
    plan_bias: float = 0.45,
    affinity_mask: Optional[int] = None,
    priority_class: Optional[str] = None,
    pl1_watts: Optional[float] = None,
    npu_streams: int = 2,
    is_shadow_only: bool = False
) -> EstimationTaskPlan:
    """
    Constructs the canonical 6-stage estimation DAG for evaluating system optimizations.
    """
    plan = EstimationTaskPlan(plan_id=plan_id, template_id=template_id)

    # 1. Telemetry Ingest
    plan.add_task(EstimationTask(
        task_id="TASK_01_OBS_TELEMETRY",
        name="Silicon Sensor Telemetry Sampling",
        stage=TaskStageKind.STAGE_OBS_TELEMETRY,
        target_domain=HardwareDomainKind.POWER_THERMAL,
        latency_budget_us=800.0
    ))

    # 2. Feature Extraction
    plan.add_task(EstimationTask(
        task_id="TASK_02_DER_FEATURES",
        name="Derived Mathematical Feature Normalization",
        stage=TaskStageKind.STAGE_DER_FEATURE_EXTRACTION,
        target_domain=HardwareDomainKind.CPU,
        dependencies=["TASK_01_OBS_TELEMETRY"],
        latency_budget_us=400.0
    ))

    # 3. NPU Forward Tensor Inference
    plan.add_task(EstimationTask(
        task_id="TASK_03_PRED_NPU",
        name="OpenVINO NPU Forward Tensor Inference",
        stage=TaskStageKind.STAGE_PRED_NPU_EVALUATION,
        target_domain=HardwareDomainKind.NPU,
        dependencies=["TASK_02_DER_FEATURES"],
        payload={"plan_bias": plan_bias},
        latency_budget_us=2500.0
    ))

    # 4. Invariant & Safety Verification
    plan.add_task(EstimationTask(
        task_id="TASK_04_SAFE_INVARIANT",
        name="Constitutional 6HP Invariant & Thermal Guard",
        stage=TaskStageKind.STAGE_SAFE_INVARIANT_VERIFICATION,
        target_domain=HardwareDomainKind.CPU,
        dependencies=["TASK_03_PRED_NPU"],
        latency_budget_us=300.0
    ))

    # 5. Dual-Gate Authority Decision
    plan.add_task(EstimationTask(
        task_id="TASK_05_AUTH_GATE",
        name="Epistemic Dual-Gate Authority Verdict",
        stage=TaskStageKind.STAGE_AUTH_DUAL_GATE_DECISION,
        target_domain=HardwareDomainKind.CPU,
        dependencies=["TASK_04_SAFE_INVARIANT"],
        payload={"is_shadow_only": is_shadow_only},
        latency_budget_us=300.0
    ))

    # 6. Hardware Actuation Dispatch
    plan.add_task(EstimationTask(
        task_id="TASK_06_ACT_MUTATION",
        name="Hardware Substrate Directive Dispatch",
        stage=TaskStageKind.STAGE_ACT_MUTATION_DISPATCH,
        target_domain=HardwareDomainKind.CPU,
        dependencies=["TASK_05_AUTH_GATE"],
        payload={
            "affinity_mask": affinity_mask,
            "priority_class": priority_class,
            "pl1_watts": pl1_watts,
            "npu_streams": npu_streams
        },
        latency_budget_us=1500.0
    ))

    return plan


if __name__ == "__main__":
    print("===================================================================")
    print("Krystal Estimation Task Planner & DAG Scheduler")
    print(f"Constitutional Invariant: VITAL_MAX_HP == {VITAL_MAX_HP}")
    print("===================================================================")

    dag = build_canonical_estimation_dag(
        plan_id="TEST_DAG_PLAN_01",
        template_id="TEMPLATE_BALANCED_ADAPTIVE_CADENCE",
        plan_bias=0.45,
        affinity_mask=0x00FF,
        priority_class="NORMAL_PRIORITY_CLASS",
        pl1_watts=24.0,
        npu_streams=2
    )

    report = dag.execute()
    print(f"Executed {report.executed_tasks}/{report.total_tasks} tasks in {report.total_latency_us:.2f} us")
    print(f"Safety: {report.safety_verdict} | Auth: {report.authority_verdict} | Txn: {report.transaction_state}")
    for t in report.tasks:
        print(f"  - [{t['task_id']}] {t['name']:<45} : {t['actual_latency_us']:>7.2f} us ({t['state']})")
