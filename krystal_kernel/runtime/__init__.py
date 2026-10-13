#!/usr/bin/env python3
"""
==============================================================================
KRYSTAL-STACK: RUNTIME SUBSYSTEM PACKAGE (CORE & EXTENSIONS)
==============================================================================
"""

from .prediction_error_ledger import (
    PredictionErrorLedger,
    PredictionErrorRecord,
    ContextualRiskRegistry,
    ContextualRiskKey
)
from .shadow_runtime import ShadowRuntime

from .epistemic_type_system import (
    LatticeCertaintyLevel,
    ConfidenceInterval,
    EpistemicValue,
    EpistemicTruthLattice
)

from .versioning_engine import (
    StatePolicyEvidenceVersion,
    VersionSnapshot,
    StatePolicyEvidenceVersioningEngine,
    GLOBAL_VERSIONING_ENGINE
)

from .dual_gate_engine import (
    SafetyGateResult,
    EvidenceGateResult,
    DualGateVerdict,
    SafetyGate,
    EvidenceGate,
    DualGateEvaluator,
    GLOBAL_DUAL_GATE_EVALUATOR
)

from .extensions import (
    ExtensionCategory,
    BaseRuntimePlugin,
    BasePredictorPlugin,
    EwmaLatencyPredictorPlugin,
    ParetoNpuPredictorPlugin,
    BasePlannerPlugin,
    ParetoOptimalNpuPlannerPlugin,
    ConservativeThermalPlannerPlugin,
    LatencyCriticalPlannerPlugin,
    BaseWorkerStrategyPlugin,
    SequentialCpuWorkerStrategyPlugin,
    ParallelThreadedWorkerStrategyPlugin,
    ZeroCopyVectorWorkerStrategyPlugin,
    BaseTelemetryProbePlugin,
    NanosecondQpcTelemetryProbePlugin,
    SiliconThermalTelemetryProbePlugin,
    BasePolicyPlugin,
    ConstitutionalMaxHpPolicyPlugin,
    ObserverOverheadBudgetPolicyPlugin,
    BaseHardwareAdapterPlugin,
    IntelIrisXeHardwareAdapterPlugin,
    CpuBaremetalHardwareAdapterPlugin,
    BaseHypervisorAdapterPlugin,
    SovereignArchLinuxHypervisorAdapterPlugin,
    EmulatedVirtualizationAdapterPlugin,
    BaseExperimentPlugin,
    CanonicalExperimentModule,
    RuntimeExtensionRegistry,
    GLOBAL_RUNTIME_EXTENSION_REGISTRY,
    PredictionOutput,
    PlanOutput,
    WorkerExecutionResult,
    TelemetryProbeSample,
    PolicyEvaluationResult,
    HardwareCapabilities,
    HypervisorIsolationReport,
    ExperimentResult
)

from .core_engine import (
    CoreExecutionReport,
    KrystalRuntimeCore,
    GLOBAL_KRYSTAL_RUNTIME_CORE
)

__all__ = [
    # Legacy & Shadow
    "PredictionErrorLedger",
    "PredictionErrorRecord",
    "ContextualRiskRegistry",
    "ContextualRiskKey",
    "ShadowRuntime",

    # Epistemic Type System
    "LatticeCertaintyLevel",
    "ConfidenceInterval",
    "EpistemicValue",
    "EpistemicTruthLattice",

    # State / Policy / Evidence Versioning
    "StatePolicyEvidenceVersion",
    "VersionSnapshot",
    "StatePolicyEvidenceVersioningEngine",
    "GLOBAL_VERSIONING_ENGINE",

    # SafetyGate + EvidenceGate
    "SafetyGateResult",
    "EvidenceGateResult",
    "DualGateVerdict",
    "SafetyGate",
    "EvidenceGate",
    "DualGateEvaluator",
    "GLOBAL_DUAL_GATE_EVALUATOR",

    # Runtime Extensions
    "ExtensionCategory",
    "BaseRuntimePlugin",
    "BasePredictorPlugin",
    "EwmaLatencyPredictorPlugin",
    "ParetoNpuPredictorPlugin",
    "BasePlannerPlugin",
    "ParetoOptimalNpuPlannerPlugin",
    "ConservativeThermalPlannerPlugin",
    "LatencyCriticalPlannerPlugin",
    "BaseWorkerStrategyPlugin",
    "SequentialCpuWorkerStrategyPlugin",
    "ParallelThreadedWorkerStrategyPlugin",
    "ZeroCopyVectorWorkerStrategyPlugin",
    "BaseTelemetryProbePlugin",
    "NanosecondQpcTelemetryProbePlugin",
    "SiliconThermalTelemetryProbePlugin",
    "BasePolicyPlugin",
    "ConstitutionalMaxHpPolicyPlugin",
    "ObserverOverheadBudgetPolicyPlugin",
    "BaseHardwareAdapterPlugin",
    "IntelIrisXeHardwareAdapterPlugin",
    "CpuBaremetalHardwareAdapterPlugin",
    "BaseHypervisorAdapterPlugin",
    "SovereignArchLinuxHypervisorAdapterPlugin",
    "EmulatedVirtualizationAdapterPlugin",
    "BaseExperimentPlugin",
    "CanonicalExperimentModule",
    "RuntimeExtensionRegistry",
    "GLOBAL_RUNTIME_EXTENSION_REGISTRY",
    "PredictionOutput",
    "PlanOutput",
    "WorkerExecutionResult",
    "TelemetryProbeSample",
    "PolicyEvaluationResult",
    "HardwareCapabilities",
    "HypervisorIsolationReport",
    "ExperimentResult",

    # Master Core
    "CoreExecutionReport",
    "KrystalRuntimeCore",
    "GLOBAL_KRYSTAL_RUNTIME_CORE"
]
