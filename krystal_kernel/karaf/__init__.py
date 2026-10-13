#!/usr/bin/env python3
"""
==============================================================================
KARAF: KRYSTAL ADAPTIVE RUNTIME ALGORITHM FABRIC — PACKAGE ROOT
==============================================================================
Module: krystal_kernel/karaf/__init__.py
Description:
    Canonical exports for the KARAF system.
==============================================================================
"""

from krystal_kernel.scene_constructor_presets import VITAL_MAX_HP

# 1. Canonical Types & Domains
from krystal_kernel.karaf.types import (
    HardwareDomain,
    ObservationMetric,
    DerivedMetric,
    ObservationFabricSnapshot,
    WorkerTopology,
    ElasticityClassification,
    WorkerElasticityResult,
    PlacementDecision,
    HypothesisStatus,
    RuntimeHypothesis,
    ShadowActionType,
    ShadowDecision,
    ExperimentPlan,
    FalsificationReport,
)

# 2. Hardware Observation Fabric
from krystal_kernel.karaf.observation_fabric import (
    HardwareObservationFabric,
    GLOBAL_OBSERVATION_FABRIC,
)

# 3. Algorithm Template Protocol & Registry
from krystal_kernel.karaf.algorithm_template import (
    AlgorithmTemplate,
    TemplateIdentity,
    TemplateEpistemic,
    TemplateObjective,
)
from krystal_kernel.karaf.template_registry import (
    TemplateRegistry,
    GLOBAL_TEMPLATE_REGISTRY,
    parse_yaml_subset,
)

# 4. Next-Gen Algorithms A through I
from krystal_kernel.karaf.resource_envelope_solver import (
    AdaptiveResourceEnvelopeSolver,
    ResourceEnvelopeSolution,
)
from krystal_kernel.karaf.worker_elasticity import (
    WorkerElasticityGradient,
)
from krystal_kernel.karaf.cascade_placement import (
    CascadeHardwarePlacementOptimizer,
)
from krystal_kernel.karaf.thermal_predictor import (
    PredictiveThermalGovernor,
    ThermalGovernorReport,
    ThermalHorizonPrediction,
)
from krystal_kernel.karaf.uma_contention import (
    UmaContentionPredictor,
    UmaContentionReport,
)
from krystal_kernel.karaf.critical_path_optimizer import (
    CascadeCriticalPathOptimizer,
    CriticalPathReport,
)
from krystal_kernel.karaf.interference_model import (
    RuntimeInterferenceMatrix,
    InterferenceEdge,
)
from krystal_kernel.karaf.evidence_decay import (
    EvidenceDecayEngine,
    EvidenceFreshnessStatus,
    DecayedEvidenceReport,
)
from krystal_kernel.karaf.hypothesis_miner import (
    HypothesisMiner,
)

# 5. Shadow-First Execution Engine & Authority Dual-Gate
from krystal_kernel.karaf.shadow_executor import (
    ShadowExecutionEngine,
)

# 6. Closed-Loop Evaluator & State Validity
from krystal_kernel.karaf.evaluator import (
    ClosedLoopEvaluator,
    ScopedStateVersion,
    EvaluationRecord,
)

# 7. Adaptive Planner
from krystal_kernel.karaf.planner import (
    KarafPlanner,
    KarafPlanningReport,
    GLOBAL_KARAF_PLANNER,
)

# 8. Skill-Driven Experiment Generator
from krystal_kernel.karaf.skill_experiment_generator import (
    SkillDrivenExperimentGenerator,
)

__all__ = [
    "VITAL_MAX_HP",
    "HardwareDomain",
    "ObservationMetric",
    "DerivedMetric",
    "ObservationFabricSnapshot",
    "WorkerTopology",
    "ElasticityClassification",
    "WorkerElasticityResult",
    "PlacementDecision",
    "HypothesisStatus",
    "RuntimeHypothesis",
    "ShadowActionType",
    "ShadowDecision",
    "ExperimentPlan",
    "FalsificationReport",
    "HardwareObservationFabric",
    "GLOBAL_OBSERVATION_FABRIC",
    "AlgorithmTemplate",
    "TemplateIdentity",
    "TemplateEpistemic",
    "TemplateObjective",
    "TemplateRegistry",
    "GLOBAL_TEMPLATE_REGISTRY",
    "parse_yaml_subset",
    "AdaptiveResourceEnvelopeSolver",
    "ResourceEnvelopeSolution",
    "WorkerElasticityGradient",
    "CascadeHardwarePlacementOptimizer",
    "PredictiveThermalGovernor",
    "ThermalGovernorReport",
    "ThermalHorizonPrediction",
    "UmaContentionPredictor",
    "UmaContentionReport",
    "CascadeCriticalPathOptimizer",
    "CriticalPathReport",
    "RuntimeInterferenceMatrix",
    "InterferenceEdge",
    "EvidenceDecayEngine",
    "EvidenceFreshnessStatus",
    "DecayedEvidenceReport",
    "HypothesisMiner",
    "ShadowExecutionEngine",
    "ClosedLoopEvaluator",
    "ScopedStateVersion",
    "EvaluationRecord",
    "KarafPlanner",
    "KarafPlanningReport",
    "GLOBAL_KARAF_PLANNER",
    "SkillDrivenExperimentGenerator",
]
