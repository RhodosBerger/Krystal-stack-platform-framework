"""Krystal Kernel - a hardware-aware, user-space compute kernel (stdlib only).

It is *not* an OS kernel. It is a scheduler + pipeline runtime that routes named compute
kernels to execution lanes (threads, worker processes, optional OpenVINO accelerator),
enforces admission/back-pressure/deadlines, learns routing from its own event log, and
heals itself when a lane or backend misbehaves.

    from krystal_kernel import get_kernel
    k = get_kernel()
    fut = k.submit("hash_embed", {"texts": ["hello"]}, kind="embed", cost=1, deadline_ms=200)
    fut.result()
"""
from .hwprofile import HardwareProfile, detect_profile, calibrate  # noqa: F401
from .params import KernelParams, derive_params  # noqa: F401
from .eventlog import EventLog  # noqa: F401
from .kernel import ComputeKernel, Backpressure, DeadlineMissed, get_kernel, shutdown_kernel  # noqa: F401
from .metabolic_governor import BrainwavePhase, MetabolicFidelityProfile, SystemMetabolismState, MetabolicGovernor  # noqa: F401
from .processor_whisperer import (  # noqa: F401
    VITAL_MAX_HP,
    EngineOpcode,
    JanetSynthesizedScript,
    JanetScriptEngineSynthesizer,
    HardwareEventOrigin,
    SelfHealingLogEntry,
    BinaryMatrixSelfHealingLog,
    RecommendedISA,
    ISAGovernorRecommendation,
    InstructionSetMetaGovernor,
    ProcessorInstructionWhisperer,
    GLOBAL_PROCESSOR_WHISPERER
)

from .hardware_render_calculator import (  # noqa: F401
    HardwareCacheProfile,
    probe_hardware_topology,
    RenderFunctionDescriptor,
    RenderFunctionRegistry,
    RenderBudgetPlan,
    HardwareRenderBudgetCalculator,
    SwapQuotaMode,
    AdaptiveSwapTelemetry,
    AdaptiveSwapManager,
    LogDrivenJanetGodotSynthesizer,
    GLOBAL_RENDER_CALCULATOR,
    GLOBAL_ADAPTIVE_SWAP_MANAGER,
    GLOBAL_LOG_DRIVEN_SYNTHESIZER
)

from .cache_matrix_compressor import (  # noqa: F401
    CompressionStats,
    CacheMatrixRepetitionCompressor,
    GLOBAL_MATRIX_COMPRESSOR
)

from .llm_config_agent import (  # noqa: F401
    ConfigCell,
    ConfigPanelSchema,
    MutationResult,
    LLMConfigAgent,
    GLOBAL_LLM_CONFIG_AGENT
)

from .domain_derivatives import (  # noqa: F401
    HFTOrderBookTick,
    HFTOrderBookDeduplicator,
    DroneSpatialObservation,
    EdgeRoboticsSpatialGovernor,
    GenomicSignalPulse,
    GenomicSelfHealingSignalEncoder,
    GLOBAL_HFT_DEDUPLICATOR,
    GLOBAL_DRONE_GOVERNOR,
    GLOBAL_GENOMIC_ENCODER
)

from .intel_enterprise_performance_suite import (  # noqa: F401
    IntelMicroarchitecture,
    EnterprisePerformancePolicy,
    AccelerationExecutionState,
    AccelerationTargetUnit,
    HardwareTopologyProbe,
    ArrheniusGuardrailStatus,
    PowerEnvelopePlan,
    UmaMemoryAllocationMatrix,
    EnterpriseAccelerationReport,
    ArrheniusLongevityGuardrail,
    IntelHardwareProber,
    IntelArchitectureStrategy,
    TigerLakeStrategy,
    AlderRaptorLakeStrategy,
    MeteorLakeStrategy,
    LunarLakeStrategy,
    GenericIntelFallbackStrategy,
    IntelEnterprisePerformanceSuite,
    GLOBAL_INTEL_ENTERPRISE_SUITE
)

from .iris_xe_windows_kisak_driver import (  # noqa: F401
    WindowsKisakDriverState,
    MemoryTilingMode,
    WindowsKisakConfiguration,
    IrisXeWindowsTelemetry,
    IrisXeWindowsKisakDriver,
    GLOBAL_IRIS_XE_WINDOWS_KISAK_DRIVER
)

from .iris_xe_gaming_uma_driver_bridge import (  # noqa: F401
    TargetGameEngine,
    UmaAllocationStrategy,
    GraphicsApiBackend,
    EnginePerformanceTarget,
    AsahiMemorySlabReport,
    LlmGamingCoexistenceStatus,
    AsahiUmaMemoryManager,
    IrisXeGamingOptimizationEngine,
    LocalLlmGamingCoexistenceGovernor,
    CommunityMvpArtifactGenerator,
    GLOBAL_ASAHI_UMA_MANAGER,
    GLOBAL_GAMING_OPTIMIZATION_ENGINE,
    GLOBAL_COEXISTENCE_GOVERNOR
)

from .janet_openvino_core import (  # noqa: F401
    TensorOpcode,
    TargetSimdTier,
    CompiledTensorInstruction,
    JanetOpenVinoBenchmarkReport,
    JanetTensorBytecodeComposer,
    LegacyIntelSimdFallbackEngine,
    SovereignJanetOpenVinoEngine,
    GLOBAL_JANET_OPENVINO_ENGINE
)

from .game_socket_compositor_bridge import (  # noqa: F401
    SocketCompositorMode,
    GameFramePacket,
    SocketCompositorTelemetry,
    ProceduralAsciiCompositor,
    GameSocketCompositorBridge,
    GLOBAL_GAME_SOCKET_COMPOSITOR
)

from .psychological_coping_cortex_mcp_bridge import (  # noqa: F401
    PsychologicalCopingMechanism,
    CopingPriorityDescriptor,
    SiliconDieUnlockReport,
    SiliconDieUnlocker,
    SemanticSynonymCompilationOptimizer,
    CortexProcessorWhispererMcpBridge,
    GLOBAL_COPING_CORTEX_MCP_BRIDGE
)

from .epistemic_utility_tweaker import (  # noqa: F401
    TweakerPlanId,
    ScannedSystemLevels,
    CandidateExecutionPlan,
    NpuOpenVinoPlanEvaluator,
    KrystalEpistemicUtilityTweaker,
    GLOBAL_UTILITY_TWEAKER
)

from .windows_timer_subsystem import (  # noqa: F401
    WindowsQpcClock,
    WindowsSchedulerResolution,
    WindowsMmcssCoordinator,
    WindowsHighResolutionWaitableTimer,
    TimerJitterTelemetryAnalyzer,
    WindowsTimerSubsystem,
    TimerJitterReport,
    GLOBAL_WINDOWS_TIMER_SUBSYSTEM
)

from .hardware_layers import (  # noqa: F401
    HardwareDomainKind,
    CpuDomainStatus,
    GpuDomainStatus,
    NpuDomainStatus,
    ThermalPowerDomainStatus,
    CpuHardwareLayer,
    GpuHardwareLayer,
    NpuHardwareLayer,
    ThermalPowerHardwareLayer,
    HardwareSubsystemCoordinator,
    GLOBAL_HARDWARE_COORDINATOR
)

from .estimation_task_planner import (  # noqa: F401
    TaskStageKind,
    EstimationTask,
    EstimationPlanReport,
    EstimationTaskPlan,
    build_canonical_estimation_dag
)

from .coordination_templates import (  # noqa: F401
    CoordinationTemplateId,
    CoordinationTemplate,
    CoordinatedDispatchReport,
    CANONICAL_TEMPLATES,
    CoordinationTemplateEngine,
    GLOBAL_COORDINATION_TEMPLATE_ENGINE
)

from .automated_command_actors import (  # noqa: F401
    SocketCommandOpcode,
    BulkParameterSocketPacket,
    BulkParameterSocketServer,
    send_bulk_parameters_via_client,
    ActorLoopState,
    BaseAutonomousCommandLoopActor,
    HardwareRegulationLoopActor,
    NpuBatchTuningLoopActor,
    EpistemicGovernorLoopActor,
    AutonomousActorSystemCoordinator,
    GLOBAL_ACTOR_SYSTEM_COORDINATOR
)

from .convolutional_matrix_workloads import (  # noqa: F401
    BooleanMatrix,
    ConvolutionKernel,
    CANONICAL_CONVOLUTION_KERNELS,
    convolve_boolean_matrix,
    MatrixStreamingLogEntry,
    MatrixStreamingLogService,
    GLOBAL_MATRIX_STREAMING_LOG_SERVICE,
    BooleanEntity,
    BooleanMatrixCrudDatabase,
    GLOBAL_BOOLEAN_CRUD_DATABASE,
    WorkloadStageResult,
    ChainedPipelineExecutionReport,
    ChainedMatrixWorkloadPipeline,
    GLOBAL_CHAINED_MATRIX_WORKLOAD_PIPELINE
)

from .matrix_image_generator_mvp import (  # noqa: F401
    LatentSeedMode,
    ImageColorPalette,
    ImageGenerationConfig,
    GeneratedImageArtifact,
    LatentTopologyLayer,
    ConvolutionalDiffusionLayer,
    HarmonicColorTensorLayer,
    EpistemicHudInlayLayer,
    MatrixImageRasterExporter,
    MatrixImageGeneratorEngine,
    GLOBAL_MATRIX_IMAGE_GENERATOR_ENGINE
)

from .closed_loop_workload_fabric import (  # noqa: F401
    WorkloadCategory,
    Workload,
    WorkloadClassification,
    WorkloadClassifier,
    PlannerStrategyKind,
    ExecutionPlan,
    PlannerSelector,
    Planner,
    CompiledExperimentScript,
    ExperimentScriptCompiler,
    StateSpacePruner,
    SafetyEvidenceGates,
    MatrixPerturbation,
    WorkerMatrix,
    ThreadRouter,
    LayerGraph,
    WorkerRuntime,
    DiscrepancyReport,
    ExpectedVsActualComparator,
    EmpiricalHypothesis,
    HypothesisGenerator,
    HypothesisLedger,
    ClosedLoopExecutionReport,
    ClosedLoopWorkloadFabric,
    GLOBAL_CLOSED_LOOP_WORKLOAD_FABRIC
)

__all__ = [
    "HardwareProfile", "detect_profile", "calibrate", "KernelParams", "derive_params",
    "EventLog", "ComputeKernel", "Backpressure", "DeadlineMissed", "get_kernel", "shutdown_kernel",
    "BrainwavePhase", "MetabolicFidelityProfile", "SystemMetabolismState", "MetabolicGovernor",
    "EngineOpcode", "JanetSynthesizedScript", "JanetScriptEngineSynthesizer",
    "HardwareEventOrigin", "SelfHealingLogEntry", "BinaryMatrixSelfHealingLog",
    "RecommendedISA", "ISAGovernorRecommendation", "InstructionSetMetaGovernor",
    "ProcessorInstructionWhisperer", "GLOBAL_PROCESSOR_WHISPERER",
    "HardwareCacheProfile", "probe_hardware_topology", "RenderFunctionDescriptor",
    "RenderFunctionRegistry", "RenderBudgetPlan", "HardwareRenderBudgetCalculator",
    "SwapQuotaMode", "AdaptiveSwapTelemetry", "AdaptiveSwapManager",
    "LogDrivenJanetGodotSynthesizer", "GLOBAL_RENDER_CALCULATOR",
    "GLOBAL_ADAPTIVE_SWAP_MANAGER", "GLOBAL_LOG_DRIVEN_SYNTHESIZER",
    "CompressionStats", "CacheMatrixRepetitionCompressor", "GLOBAL_MATRIX_COMPRESSOR",
    "ConfigCell", "ConfigPanelSchema", "MutationResult", "LLMConfigAgent", "GLOBAL_LLM_CONFIG_AGENT",
    "HFTOrderBookTick", "HFTOrderBookDeduplicator", "DroneSpatialObservation",
    "EdgeRoboticsSpatialGovernor", "GenomicSignalPulse", "GenomicSelfHealingSignalEncoder",
    "GLOBAL_HFT_DEDUPLICATOR", "GLOBAL_DRONE_GOVERNOR", "GLOBAL_GENOMIC_ENCODER",
    "IntelMicroarchitecture", "EnterprisePerformancePolicy", "AccelerationExecutionState",
    "AccelerationTargetUnit", "HardwareTopologyProbe", "ArrheniusGuardrailStatus",
    "PowerEnvelopePlan", "UmaMemoryAllocationMatrix", "EnterpriseAccelerationReport",
    "ArrheniusLongevityGuardrail", "IntelHardwareProber", "IntelArchitectureStrategy",
    "TigerLakeStrategy", "AlderRaptorLakeStrategy", "MeteorLakeStrategy",
    "LunarLakeStrategy", "GenericIntelFallbackStrategy", "IntelEnterprisePerformanceSuite",
    "GLOBAL_INTEL_ENTERPRISE_SUITE",
    "WindowsKisakDriverState", "MemoryTilingMode", "WindowsKisakConfiguration",
    "IrisXeWindowsTelemetry", "IrisXeWindowsKisakDriver", "GLOBAL_IRIS_XE_WINDOWS_KISAK_DRIVER",
    "TargetGameEngine", "UmaAllocationStrategy", "GraphicsApiBackend",
    "EnginePerformanceTarget", "AsahiMemorySlabReport", "LlmGamingCoexistenceStatus",
    "AsahiUmaMemoryManager", "IrisXeGamingOptimizationEngine", "LocalLlmGamingCoexistenceGovernor",
    "CommunityMvpArtifactGenerator", "GLOBAL_ASAHI_UMA_MANAGER",
    "GLOBAL_GAMING_OPTIMIZATION_ENGINE", "GLOBAL_COEXISTENCE_GOVERNOR",
    "TensorOpcode", "TargetSimdTier", "CompiledTensorInstruction",
    "JanetOpenVinoBenchmarkReport", "JanetTensorBytecodeComposer",
    "LegacyIntelSimdFallbackEngine", "SovereignJanetOpenVinoEngine",
    "GLOBAL_JANET_OPENVINO_ENGINE",
    "SocketCompositorMode", "GameFramePacket", "SocketCompositorTelemetry",
    "ProceduralAsciiCompositor", "GameSocketCompositorBridge",
    "GLOBAL_GAME_SOCKET_COMPOSITOR",
    "PsychologicalCopingMechanism", "CopingPriorityDescriptor",
    "SiliconDieUnlockReport", "SiliconDieUnlocker",
    "SemanticSynonymCompilationOptimizer", "CortexProcessorWhispererMcpBridge",
    "GLOBAL_COPING_CORTEX_MCP_BRIDGE",
    "TweakerPlanId", "ScannedSystemLevels", "CandidateExecutionPlan",
    "NpuOpenVinoPlanEvaluator", "KrystalEpistemicUtilityTweaker", "GLOBAL_UTILITY_TWEAKER",
    "WindowsQpcClock", "WindowsSchedulerResolution", "WindowsMmcssCoordinator",
    "WindowsHighResolutionWaitableTimer", "TimerJitterTelemetryAnalyzer",
    "WindowsTimerSubsystem", "TimerJitterReport", "GLOBAL_WINDOWS_TIMER_SUBSYSTEM",
    "HardwareDomainKind", "CpuDomainStatus", "GpuDomainStatus", "NpuDomainStatus",
    "ThermalPowerDomainStatus", "CpuHardwareLayer", "GpuHardwareLayer",
    "NpuHardwareLayer", "ThermalPowerHardwareLayer", "HardwareSubsystemCoordinator",
    "GLOBAL_HARDWARE_COORDINATOR", "TaskStageKind", "EstimationTask",
    "EstimationPlanReport", "EstimationTaskPlan", "build_canonical_estimation_dag",
    "CoordinationTemplateId", "CoordinationTemplate", "CoordinatedDispatchReport",
    "CANONICAL_TEMPLATES", "CoordinationTemplateEngine", "GLOBAL_COORDINATION_TEMPLATE_ENGINE",
    "SocketCommandOpcode", "BulkParameterSocketPacket", "BulkParameterSocketServer",
    "send_bulk_parameters_via_client", "ActorLoopState", "BaseAutonomousCommandLoopActor",
    "HardwareRegulationLoopActor", "NpuBatchTuningLoopActor", "EpistemicGovernorLoopActor",
    "AutonomousActorSystemCoordinator", "GLOBAL_ACTOR_SYSTEM_COORDINATOR",
    "BooleanMatrix", "ConvolutionKernel", "CANONICAL_CONVOLUTION_KERNELS",
    "convolve_boolean_matrix", "MatrixStreamingLogEntry", "MatrixStreamingLogService",
    "GLOBAL_MATRIX_STREAMING_LOG_SERVICE", "BooleanEntity", "BooleanMatrixCrudDatabase",
    "GLOBAL_BOOLEAN_CRUD_DATABASE", "WorkloadStageResult", "ChainedPipelineExecutionReport",
    "ChainedMatrixWorkloadPipeline", "GLOBAL_CHAINED_MATRIX_WORKLOAD_PIPELINE",
    "LatentSeedMode", "ImageColorPalette", "ImageGenerationConfig", "GeneratedImageArtifact",
    "LatentTopologyLayer", "ConvolutionalDiffusionLayer", "HarmonicColorTensorLayer",
    "EpistemicHudInlayLayer", "MatrixImageRasterExporter", "MatrixImageGeneratorEngine",
    "GLOBAL_MATRIX_IMAGE_GENERATOR_ENGINE",
    "WorkloadCategory", "Workload", "WorkloadClassification", "WorkloadClassifier",
    "PlannerStrategyKind", "ExecutionPlan", "PlannerSelector", "Planner",
    "CompiledExperimentScript", "ExperimentScriptCompiler", "StateSpacePruner",
    "SafetyEvidenceGates", "MatrixPerturbation", "WorkerMatrix", "ThreadRouter",
    "LayerGraph", "WorkerRuntime", "DiscrepancyReport", "ExpectedVsActualComparator",
    "EmpiricalHypothesis", "HypothesisGenerator", "HypothesisLedger",
    "ClosedLoopExecutionReport", "ClosedLoopWorkloadFabric", "GLOBAL_CLOSED_LOOP_WORKLOAD_FABRIC"
]
