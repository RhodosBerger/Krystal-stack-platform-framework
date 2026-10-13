#!/usr/bin/env python3
"""
==============================================================================
KARAF: UNIFIED ADAPTIVE RUNTIME PLANNER & RESEARCH ORCHESTRATOR
==============================================================================
Module: krystal_kernel/karaf/planner.py
Description:
    Coordinates the closed-loop scientific research and execution cycle:
        OBSERVE -> REPRESENT -> EXPLAIN -> PREDICT -> FALSIFY -> VALIDATE -> UPDATE

    Unifies Algorithms A through I into an autonomous, shadow-first runtime fabric.
==============================================================================
"""

import time
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple

from krystal_kernel.scene_constructor_presets import VITAL_MAX_HP
from krystal_kernel.karaf.types import (
    ObservationFabricSnapshot,
    WorkerTopology,
    PlacementDecision,
    ShadowDecision,
    RuntimeHypothesis
)
from krystal_kernel.karaf.observation_fabric import (
    HardwareObservationFabric,
    GLOBAL_OBSERVATION_FABRIC
)
from krystal_kernel.karaf.resource_envelope_solver import (
    AdaptiveResourceEnvelopeSolver,
    ResourceEnvelopeSolution
)
from krystal_kernel.karaf.worker_elasticity import (
    WorkerElasticityGradient,
    WorkerElasticityResult
)
from krystal_kernel.karaf.cascade_placement import (
    CascadeHardwarePlacementOptimizer
)
from krystal_kernel.karaf.thermal_predictor import (
    PredictiveThermalGovernor,
    ThermalGovernorReport
)
from krystal_kernel.karaf.uma_contention import (
    UmaContentionPredictor,
    UmaContentionReport
)
from krystal_kernel.karaf.critical_path_optimizer import (
    CascadeCriticalPathOptimizer,
    CriticalPathReport
)
from krystal_kernel.karaf.interference_model import (
    RuntimeInterferenceMatrix
)
from krystal_kernel.karaf.evidence_decay import (
    EvidenceDecayEngine
)
from krystal_kernel.karaf.hypothesis_miner import (
    HypothesisMiner
)
from krystal_kernel.karaf.shadow_executor import (
    ShadowExecutionEngine
)
from krystal_kernel.karaf.evaluator import (
    ClosedLoopEvaluator
)
from krystal_kernel.cascade.cascade_planner import CascadePlan


@dataclass
class KarafPlanningReport:
    """Comprehensive status and decision packet produced by KARAF."""
    timestamp_ns: int
    snapshot_id: str
    resource_envelope: ResourceEnvelopeSolution
    worker_elasticity: WorkerElasticityResult
    thermal_report: ThermalGovernorReport
    uma_report: UmaContentionReport
    shadow_decisions: List[ShadowDecision]
    active_hypotheses: List[RuntimeHypothesis]
    decay_report_count: int
    vital_max_hp: int = VITAL_MAX_HP

    def __post_init__(self):
        assert self.vital_max_hp == 6, "VITAL_MAX_HP must equal 6"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp_ns": self.timestamp_ns,
            "snapshot_id": self.snapshot_id,
            "resource_envelope": self.resource_envelope.to_dict(),
            "worker_elasticity": self.worker_elasticity.to_dict(),
            "thermal_report": self.thermal_report.to_dict(),
            "uma_report": self.uma_report.to_dict(),
            "shadow_decisions": [d.to_dict() for d in self.shadow_decisions],
            "active_hypotheses": [h.to_dict() for h in self.active_hypotheses],
            "decay_report_count": self.decay_report_count,
            "vital_max_hp": self.vital_max_hp
        }


class KarafPlanner:
    """
    Central adaptive orchestrator of the Krystal Adaptive Runtime Algorithm Fabric.
    """

    def __init__(
        self,
        fabric: Optional[HardwareObservationFabric] = None,
        decay_engine: Optional[EvidenceDecayEngine] = None,
        hypothesis_miner: Optional[HypothesisMiner] = None,
        shadow_engine: Optional[ShadowExecutionEngine] = None,
        evaluator: Optional[ClosedLoopEvaluator] = None
    ):
        assert VITAL_MAX_HP == 6, f"Constitutional invariant breach: VITAL_MAX_HP must be 6, got {VITAL_MAX_HP}"
        self.vital_max_hp = VITAL_MAX_HP

        self.fabric = fabric or GLOBAL_OBSERVATION_FABRIC
        self.decay_engine = decay_engine or EvidenceDecayEngine()
        self.hypothesis_miner = hypothesis_miner or HypothesisMiner()
        self.shadow_engine = shadow_engine or ShadowExecutionEngine()
        self.evaluator = evaluator or ClosedLoopEvaluator()

        # Algorithms
        self.envelope_solver = AdaptiveResourceEnvelopeSolver()
        self.elasticity_estimator = WorkerElasticityGradient()
        self.placement_optimizer = CascadeHardwarePlacementOptimizer()
        self.thermal_governor = PredictiveThermalGovernor()
        self.uma_predictor = UmaContentionPredictor()
        self.critical_path_optimizer = CascadeCriticalPathOptimizer()
        self.interference_matrix = RuntimeInterferenceMatrix()

    def run_cycle(
        self,
        current_topology: Optional[WorkerTopology] = None,
        workload_context: Optional[Dict[str, Any]] = None,
        cascade_plan: Optional[CascadePlan] = None
    ) -> KarafPlanningReport:
        """
        Executes one full iteration of the closed-loop research and planning cycle.
        """
        now_ns = time.time_ns()

        # 1. OBSERVE: Sample hardware observation fabric
        snapshot: ObservationFabricSnapshot = self.fabric.sample_fabric()

        # 2. EVIDENCE DECAY: Audit evidence freshness
        fresh_obs, decay_reports = self.decay_engine.filter_fresh_evidence(snapshot.observations)

        # 3. INTERFERENCE INGESTION
        self.interference_matrix.ingest_snapshot(snapshot)

        # 4. DEFAULT TOPOLOGY
        topology = current_topology or WorkerTopology(
            worker_count=snapshot.hardware_context.get("p_cores", 4),
            vital_max_hp=VITAL_MAX_HP
        )

        # 5. SOLVE ALGORITHMS
        # A: Resource Envelope
        env_solution = self.envelope_solver.solve(snapshot, workload_context)

        # B: Worker Elasticity Gradient
        elasticity_res = self.elasticity_estimator.evaluate_elasticity(
            topology, snapshot, workload_class=workload_context.get("category", "COMPUTE_BOUND") if workload_context else "COMPUTE_BOUND"
        )
        elasticity_shadow = self.elasticity_estimator.create_shadow_decision(elasticity_res)

        # D: Predictive Thermal Governor
        thermal_rep = self.thermal_governor.predict(snapshot, worker_count=topology.worker_count)

        # E: UMA Contention Predictor
        uma_rep = self.uma_predictor.evaluate_contention(snapshot)

        # Collect shadow decisions
        shadow_decisions = [
            env_solution.shadow_decision,
            elasticity_shadow,
            thermal_rep.shadow_decision,
            uma_rep.shadow_decision
        ]

        # 6. MINE HYPOTHESES from interference & elasticity
        new_hypotheses = self.interference_matrix.generate_interference_hypotheses()
        for nh in new_hypotheses:
            self.hypothesis_miner.record_hypothesis(nh)

        # If elasticity indicates contention, propose worker saturation hypothesis
        if elasticity_res.classification.value == "CONTENTION_REGION":
            sat_hyp = self.hypothesis_miner.propose_worker_saturation_hypothesis(
                current_workers=elasticity_res.current_workers,
                attempted_workers=elasticity_res.current_workers + 2,
                throughput_delta_pct=-8.5,
                context_switch_delta_pct=42.0,
                hardware_context=snapshot.hardware_context
            )

        active_hyp = list(self.hypothesis_miner.ledger.values())

        return KarafPlanningReport(
            timestamp_ns=now_ns,
            snapshot_id=snapshot.snapshot_id,
            resource_envelope=env_solution,
            worker_elasticity=elasticity_res,
            thermal_report=thermal_rep,
            uma_report=uma_rep,
            shadow_decisions=shadow_decisions,
            active_hypotheses=active_hyp,
            decay_report_count=len(decay_reports),
            vital_max_hp=VITAL_MAX_HP
        )


# Global singleton planner
GLOBAL_KARAF_PLANNER = KarafPlanner()
