#!/usr/bin/env python3
"""
==============================================================================
KRYSTAL-STACK: KRYSTAL RUNTIME CORE (MASTER ENGINE)
==============================================================================
Module: krystal_kernel/runtime/core_engine.py
Description:
    Master orchestrator implementing the comprehensive architecture:
      Krystal Runtime Core
      │
      ├─ Epistemic Type System
      ├─ State / Policy / Evidence Versioning
      ├─ SafetyGate + EvidenceGate
      ├─ PlannerSelector
      ├─ Worker / Thread / Layer Runtime
      ├─ Hypothesis Ledger
      │
      └─ Runtime Extensions
         ├─ Predictor Plugin
         ├─ Planner Plugin
         ├─ Worker Strategy Plugin
         ├─ Telemetry Probe Plugin
         ├─ Policy Plugin
         ├─ Hardware Adapter Plugin
         ├─ Hypervisor Adapter Plugin
         └─ Experiment Module

Constitutional Invariant:
    assert VITAL_MAX_HP == 6
==============================================================================
"""

import time
import math
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field, asdict

from krystal_kernel.scene_constructor_presets import VITAL_MAX_HP
from krystal_kernel.verification.semantic_protocol import (
    EpistemicState,
    AuthorityState,
    SafetyState,
    to_wire_token
)
from krystal_kernel.convolutional_matrix_workloads import BooleanMatrix

# Runtime Core Modules
from krystal_kernel.runtime.epistemic_type_system import (
    EpistemicValue,
    EpistemicTruthLattice,
    LatticeCertaintyLevel,
    ConfidenceInterval
)
from krystal_kernel.runtime.versioning_engine import (
    StatePolicyEvidenceVersion,
    StatePolicyEvidenceVersioningEngine,
    GLOBAL_VERSIONING_ENGINE
)
from krystal_kernel.runtime.dual_gate_engine import (
    SafetyGate,
    EvidenceGate,
    DualGateEvaluator,
    DualGateVerdict,
    GLOBAL_DUAL_GATE_EVALUATOR
)
from krystal_kernel.closed_loop_workload_fabric import (
    HypothesisLedger,
    EmpiricalHypothesis,
    WorkerMatrix,
    ThreadRouter,
    LayerGraph,
    WorkerRuntime
)

# Runtime Extension Plugins
from krystal_kernel.runtime.extensions import (
    ExtensionCategory,
    RuntimeExtensionRegistry,
    GLOBAL_RUNTIME_EXTENSION_REGISTRY,
    BasePredictorPlugin,
    BasePlannerPlugin,
    BaseWorkerStrategyPlugin,
    BaseTelemetryProbePlugin,
    BasePolicyPlugin,
    BaseHardwareAdapterPlugin,
    BaseHypervisorAdapterPlugin,
    BaseExperimentPlugin,
    PredictionOutput,
    PlanOutput,
    WorkerExecutionResult,
    TelemetryProbeSample,
    PolicyEvaluationResult,
    HardwareCapabilities,
    HypervisorIsolationReport,
    ExperimentResult
)

assert VITAL_MAX_HP == 6, f"Constitutional invariant breach: VITAL_MAX_HP must be 6, got {VITAL_MAX_HP}"


@dataclass
class CoreExecutionReport:
    """Artifact produced by KrystalRuntimeCore after executing a workload."""
    execution_id: str
    workload_id: str
    version: Dict[str, Any]
    epistemic_value: Dict[str, Any]
    dual_gate_verdict: Dict[str, Any]
    predictor_used: str
    predicted: Dict[str, Any]
    planner_used: str
    plan: Dict[str, Any]
    worker_strategy_used: str
    runtime_telemetry: Dict[str, Any]
    hardware_adapter_used: str
    hypervisor_adapter_used: str
    probes_sampled: List[Dict[str, Any]]
    active_hypotheses_count: int
    cycle_duration_us: float
    vital_max_hp: int = VITAL_MAX_HP

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class KrystalRuntimeCore:
    """
    Master Runtime Core orchestrating:
      - Epistemic Type System
      - State / Policy / Evidence Versioning
      - SafetyGate + EvidenceGate
      - PlannerSelector (integrating Planner Plugins & Hypothesis Ledger)
      - Worker / Thread / Layer Runtime (integrating Worker Strategy Plugins)
      - Hypothesis Ledger
      - Runtime Extensions Registry
      - Experiment Module
    """

    def __init__(
        self,
        versioning_engine: Optional[StatePolicyEvidenceVersioningEngine] = None,
        dual_gate_evaluator: Optional[DualGateEvaluator] = None,
        extension_registry: Optional[RuntimeExtensionRegistry] = None,
        hypothesis_ledger: Optional[HypothesisLedger] = None
    ):
        self.versioning = versioning_engine or GLOBAL_VERSIONING_ENGINE
        self.dual_gate = dual_gate_evaluator or GLOBAL_DUAL_GATE_EVALUATOR
        self.extensions = extension_registry or GLOBAL_RUNTIME_EXTENSION_REGISTRY
        self.hypotheses = hypothesis_ledger or HypothesisLedger()

        # Worker Subsystems
        self.worker_matrix = WorkerMatrix()
        self.thread_router = ThreadRouter()
        self.layer_graph = LayerGraph()
        self.worker_runtime = WorkerRuntime()

        self._executions_count: int = 0
        self._experiments_count: int = 0

    def get_runtime_architecture_tree(self) -> Dict[str, Any]:
        """Provides the full nested tree view requested by the architectural specification."""
        ver = self.versioning.current_version
        return {
            "root": "Krystal Runtime Core",
            "vital_max_hp": VITAL_MAX_HP,
            "core_subsystems": {
                "Epistemic Type System": {
                    "certainty_levels": [lvl.value for lvl in LatticeCertaintyLevel],
                    "lattice_operators": ["meet(∧)", "join(∨)", "detect_contradiction()"],
                    "formal_types": ["EpistemicValue[T]", "ConfidenceInterval", "EpistemicTruthLattice"]
                },
                "State / Policy / Evidence Versioning": {
                    "current_version": ver.to_dict(),
                    "vector_string": ver.vector_string,
                    "merkle_fingerprint": ver.compute_merkle_fingerprint()
                },
                "SafetyGate + EvidenceGate": {
                    "safety_gate": {
                        "max_thermal_celsius": self.dual_gate.safety_gate.max_thermal,
                        "max_power_watts": self.dual_gate.safety_gate.max_power,
                        "max_observer_overhead_pct": self.dual_gate.safety_gate.max_observer_overhead,
                        "invariant": f"assert VITAL_MAX_HP == {VITAL_MAX_HP}"
                    },
                    "evidence_gate": {
                        "min_belief_degree": self.dual_gate.evidence_gate.min_belief_degree,
                        "falsification_guard": True
                    },
                    "dual_gate_synthesis": "f(SafetyGate, EvidenceGate) -> AuthorityState"
                },
                "PlannerSelector": {
                    "registered_planners": [
                        p["plugin_id"] for p in self.extensions.list_plugins(ExtensionCategory.PLANNER)
                    ],
                    "active_planner": self.extensions.get_active_plugin(ExtensionCategory.PLANNER).plugin_id if self.extensions.get_active_plugin(ExtensionCategory.PLANNER) else None,
                    "guided_by_hypotheses": True
                },
                "Worker / Thread / Layer Runtime": {
                    "worker_matrix_slots": getattr(self.worker_matrix, "num_slots", 8),
                    "thread_router_threads": getattr(self.thread_router, "total_threads", 8),
                    "active_strategy": self.extensions.get_active_plugin(ExtensionCategory.WORKER_STRATEGY).plugin_id if self.extensions.get_active_plugin(ExtensionCategory.WORKER_STRATEGY) else None
                },
                "Hypothesis Ledger": {
                    "active_hypotheses": len(self.hypotheses.active_hypotheses),
                    "corroborated_rules": self.hypotheses.get_active_corroborated_rules()
                }
            },
            "runtime_extensions": self.extensions.get_extension_tree_summary()
        }

    def execute_workload(
        self,
        workload_id: str,
        workload_category: str = "COMPUTE_BOUND_CONVOLUTION",
        input_bits: Optional[int] = None,
        target_deadline_us: float = 50000.0,
        thermal_celsius: float = 52.0,
        power_watts: float = 18.0,
        observer_overhead_pct: float = 1.8
    ) -> CoreExecutionReport:
        """
        Executes a workload through the entire unified Core & Extensions architecture.
        """
        t_start = time.perf_counter_ns()
        self._executions_count += 1
        exec_id = f"CORE_EXEC_{self._executions_count}_{int(time.time()*1000)}"

        # 1. Epistemic Observation Formation
        ep_val = EpistemicValue(
            value={"workload_category": workload_category, "target_deadline_us": target_deadline_us},
            epistemic_state=EpistemicState.EPI_OBSERVED,
            belief_degree=0.95,
            certainty_level=LatticeCertaintyLevel.EVIDENCED_STRONG,
            confidence_interval=ConfidenceInterval(lower_bound=0.90, upper_bound=0.98),
            evidence_tokens=["OBS_WORKLOAD_INGEST", f"DEADLINE_{int(target_deadline_us)}us"],
            source_probe="CoreWorkloadIngestProbe"
        )

        # 2. Dual-Gate Governance Verification (SafetyGate + EvidenceGate)
        verdict = self.dual_gate.evaluate(
            epistemic_val=ep_val,
            vital_hp=VITAL_MAX_HP,
            thermal_celsius=thermal_celsius,
            power_watts=power_watts,
            observer_overhead_pct=observer_overhead_pct
        )

        if not verdict.can_actuate:
            # Rejection or Deferral
            elapsed_us = (time.perf_counter_ns() - t_start) / 1000.0
            return CoreExecutionReport(
                execution_id=exec_id,
                workload_id=workload_id,
                version=self.versioning.current_version.to_dict(),
                epistemic_value=ep_val.to_dict(),
                dual_gate_verdict=verdict.to_dict(),
                predictor_used="N/A",
                predicted={},
                planner_used="N/A",
                plan={},
                worker_strategy_used="N/A",
                runtime_telemetry={"status": "ACTUATION_BLOCKED", "reason": verdict.verdict_message},
                hardware_adapter_used="N/A",
                hypervisor_adapter_used="N/A",
                probes_sampled=[],
                active_hypotheses_count=len(self.hypotheses.active_hypotheses),
                cycle_duration_us=elapsed_us,
                vital_max_hp=VITAL_MAX_HP
            )

        # 3. Policy Plugin Evaluation
        policy_plugin = self.extensions.get_active_plugin(ExtensionCategory.POLICY)
        if policy_plugin and isinstance(policy_plugin, BasePolicyPlugin):
            pol_eval = policy_plugin.evaluate_policy(
                "EXECUTE_WORKLOAD",
                {"vital_max_hp": VITAL_MAX_HP, "observer_overhead_pct": observer_overhead_pct}
            )
            if not pol_eval.allowed:
                raise PermissionError(f"Policy Rejection ({policy_plugin.name}): {pol_eval.verdict}")

        # 4. Predictor Plugin Invocation
        predictor_plugin = self.extensions.get_active_plugin(ExtensionCategory.PREDICTOR)
        pred_out = None
        if predictor_plugin and isinstance(predictor_plugin, BasePredictorPlugin):
            pred_out = predictor_plugin.predict(
                {"workload_category": workload_category, "initial_density": 0.25},
                {"allocated_threads": 4}
            )

        # 5. PlannerSelector & Planner Plugin Invocation
        # Query active hypotheses for constraints
        active_rules = self.hypotheses.get_active_corroborated_rules()
        planner_plugin = self.extensions.get_active_plugin(ExtensionCategory.PLANNER)
        if "RULE_THERMAL_THROTTLE" in active_rules:
            # Dynamically switch to conservative thermal planner
            thermal_planner = self.extensions.get_plugin("PLANNER_CONSERVATIVE_THERMAL")
            if thermal_planner and isinstance(thermal_planner, BasePlannerPlugin):
                planner_plugin = thermal_planner

        plan_out = None
        if planner_plugin and isinstance(planner_plugin, BasePlannerPlugin):
            plan_out = planner_plugin.create_plan(
                {"workload_id": workload_id, "category": workload_category},
                {"thermal_celsius": thermal_celsius}
            )

        # 6. Worker Strategy Execution across Worker Matrix & Layer Graph
        worker_strategy = self.extensions.get_active_plugin(ExtensionCategory.WORKER_STRATEGY)
        if isinstance(input_bits, str):
            work_matrix = BooleanMatrix.from_hex_string(input_bits, 8, 8)
        elif isinstance(input_bits, int):
            work_matrix = BooleanMatrix.from_hex_string(hex(input_bits), 8, 8)
        elif isinstance(input_bits, list):
            work_matrix = BooleanMatrix(8, 8, initial_bits=input_bits)
        else:
            work_matrix = BooleanMatrix.from_hex_string("0x003C3C3C3C3C3C00", 8, 8)
        worker_res = None
        if worker_strategy and isinstance(worker_strategy, BaseWorkerStrategyPlugin):
            worker_res = worker_strategy.execute_stage(
                {"stage_id": "STAGE_PRIMARY_DISPATCH"},
                work_matrix
            )

        # 7. Telemetry Probes Sampling
        probe_samples = []
        for p in self.extensions.list_plugins(ExtensionCategory.TELEMETRY_PROBE):
            inst = self.extensions.get_plugin(p["plugin_id"])
            if inst and isinstance(inst, BaseTelemetryProbePlugin):
                s = inst.sample_probe({"workload_id": workload_id})
                probe_samples.append(asdict(s))

        # 8. Hardware & Hypervisor Adapters Verification
        hw_adapter = self.extensions.get_active_plugin(ExtensionCategory.HARDWARE_ADAPTER)
        hw_name = hw_adapter.name if hw_adapter else "DEFAULT_CPU"

        hyp_adapter = self.extensions.get_active_plugin(ExtensionCategory.HYPERVISOR_ADAPTER)
        hyp_name = hyp_adapter.name if hyp_adapter else "EMULATED_SANDBOX"

        # 9. Advance State / Policy / Evidence Versioning
        new_version = self.versioning.advance_state(
            state_delta={
                "execution_id": exec_id,
                "workload_id": workload_id,
                "density": work_matrix.density,
                "strategy": worker_strategy.plugin_id if worker_strategy else "UNKNOWN"
            },
            mutation_cause=f"WORKLOAD_EXEC_{workload_category}"
        )

        # Commit evidence fingerprint to Versioning Ledger
        ev_hash = ep_val.compute_evidence_hash()
        self.versioning.commit_evidence_batch([ev_hash])

        elapsed_us = (time.perf_counter_ns() - t_start) / 1000.0

        return CoreExecutionReport(
            execution_id=exec_id,
            workload_id=workload_id,
            version=new_version.to_dict(),
            epistemic_value=ep_val.to_dict(),
            dual_gate_verdict=verdict.to_dict(),
            predictor_used=predictor_plugin.plugin_id if predictor_plugin else "NONE",
            predicted=asdict(pred_out) if pred_out else {},
            planner_used=planner_plugin.plugin_id if planner_plugin else "NONE",
            plan=asdict(plan_out) if plan_out else {},
            worker_strategy_used=worker_strategy.plugin_id if worker_strategy else "NONE",
            runtime_telemetry={
                "elapsed_us": worker_res.elapsed_us if worker_res else 0.0,
                "output_density": worker_res.output_density if worker_res else work_matrix.density,
                "bits_mutated": worker_res.bits_mutated if worker_res else 0,
                "status": worker_res.status if worker_res else "COMMITTED"
            },
            hardware_adapter_used=hw_name,
            hypervisor_adapter_used=hyp_name,
            probes_sampled=probe_samples,
            active_hypotheses_count=len(self.hypotheses.active_hypotheses),
            cycle_duration_us=round(elapsed_us, 2),
            vital_max_hp=VITAL_MAX_HP
        )

    def run_experiment(self, experiment_spec: Optional[Dict[str, Any]] = None) -> ExperimentResult:
        """
        Executes an experiment trial via the active Experiment Module.
        """
        self._experiments_count += 1
        spec = experiment_spec or {}
        if "experiment_id" not in spec:
            spec["experiment_id"] = f"EXP_TRIAL_{self._experiments_count}_{int(time.time()*1000)}"

        exp_plugin = self.extensions.get_active_plugin(ExtensionCategory.EXPERIMENT_MODULE)
        if not exp_plugin or not isinstance(exp_plugin, BaseExperimentPlugin):
            raise RuntimeError("No active Experiment Module plugin registered in extensions.")

        res = exp_plugin.run_experiment(spec)

        # If hypothesis is corroborated, record in Hypothesis Ledger
        if res.hypothesis_corroborated:
            rule_id = f"EXP_RULE_{res.experiment_id[:16]}"
            hyp = EmpiricalHypothesis(
                hypothesis_id=f"HYP_{int(time.time()*1000)}",
                trigger_rule=rule_id,
                description=f"Divergence {res.divergence_score:.4f} within bounds; maintain current strategy",
                confidence_score=0.85,
                corroboration_count=1
            )
            self.hypotheses.commit_hypotheses([hyp])

        return res


# Global Singleton Master Runtime Core
GLOBAL_KRYSTAL_RUNTIME_CORE = KrystalRuntimeCore()
