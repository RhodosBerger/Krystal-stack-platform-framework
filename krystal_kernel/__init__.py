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
from .custom_governor import (  # noqa: F401
    GovernorMode,
    GovernorLimits,
    GovernorDecision,
    ExecutionEnvelope,
    CustomHardwareGovernor,
)
from .custom_planner import (  # noqa: F401
    TaskCategory,
    WorkloadTask,
    ScheduledTaskAssignment,
    WorkloadPlan,
    ExecutionFeedbackRecord,
    CustomWorkloadPlanner,
)
from .verification.capability_registry import (  # noqa: F401
    CapabilityState,
    HardwareBackend,
    BackendCapability,
    CapabilityRegistry,
    GLOBAL_CAPABILITY_REGISTRY,
)
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

from .runtime import (  # noqa: F401
    EpistemicValue,
    EpistemicTruthLattice,
    LatticeCertaintyLevel,
    ConfidenceInterval,
    StatePolicyEvidenceVersion,
    StatePolicyEvidenceVersioningEngine,
    GLOBAL_VERSIONING_ENGINE,
    SafetyGateResult,
    EvidenceGateResult,
    DualGateVerdict,
    SafetyGate,
    EvidenceGate,
    DualGateEvaluator,
    GLOBAL_DUAL_GATE_EVALUATOR,
    ExtensionCategory,
    RuntimeExtensionRegistry,
    GLOBAL_RUNTIME_EXTENSION_REGISTRY,
    CoreExecutionReport,
    KrystalRuntimeCore,
    GLOBAL_KRYSTAL_RUNTIME_CORE
)

from .geometric_procedural_supersampling_engine import (  # noqa: F401
    ProceduralSceneKind,
    NpuReconstructionFilter,
    ColorBitDepth,
    NpuSupersamplingSoftwareProfile,
    CANONICAL_NPU_PROFILES,
    ProceduralFrame,
    GeometricManifoldSynthesizer,
    IntermediateFrameInterpolator,
    NpuMultiFrameSupersampler,
    ImageQualityEvaluationReport,
    ImageQualityEvaluator,
    SupersampledSceneBundle,
    GeometricProceduralSupersamplingEngine,
    GLOBAL_GEOMETRIC_PROCEDURAL_SUPERSAMPLING_ENGINE
)

from .isa_performance_taxonomies_and_profiler import (  # noqa: F401
    IsaArchitectureFamily,
    VectorRegisterWidth,
    WorkloadArithmeticTaxon,
    CacheLocalityTaxon,
    ThermalPowerTaxon,
    IsaInstructionDescriptor,
    CANONICAL_ISA_CATALOG,
    IsaPerformanceProfile,
    IsaRecommendationReport,
    IsaPerformanceProfiler,
    ProfileGuidedIsaGovernor,
    GLOBAL_ISA_PERFORMANCE_PROFILER,
    GLOBAL_ISA_GOVERNOR
)

from .multilevel_state_rerendering_and_multilingual_taxonomies import (  # noqa: F401
    SystemStateLevelTaxon,
    LanguageLocaleTaxon,
    LinguisticRegisterTaxon,
    SemanticIntentTaxon,
    MULTILINGUAL_LEXICON,
    MultiLevelSystemState,
    MultiLevelReRenderReport,
    MultilingualIntentClassifier,
    MultiLevelStateReRenderingEngine,
    GLOBAL_MULTILEVEL_STATE_RERENDERING_ENGINE
)

from .cascade import (  # noqa: F401
    ModelNode,
    ModelNodeSpec,
    IntentClassifierNode,
    CoreRagNode,
    ReasoningNode,
    CodingSpecialistNode,
    VerificationNode,
    FinalSynthesisNode,
    ModelRouter,
    NodeRouteCandidate,
    RoutingObjectiveWeights,
    ROUTING_PROFILES,
    CascadeEdge,
    CascadePlan,
    CascadePlanner,
    CascadeMerger,
    CascadeExecutor,
    CascadeNodeResult,
    CascadeExecutionTrace,
    ModelCascadeRuntime,
    GLOBAL_MODEL_CASCADE_RUNTIME
)

from .linux_hw_monitor_utility import (  # noqa: F401
    HardwareResourceCategory,
    HardwareAllocationSnapshot,
    CpuSchedulerAllocation,
    MemoryUmaAllocation,
    GpuAcceleratorAllocation,
    NpuCascadeAllocation,
    PowerThermalAllocation,
    StorageIoAllocation,
    NetworkSocketsAllocation,
    EpistemicInvariantsAllocation,
    LinuxHwMonitorUtility,
    GLOBAL_LINUX_HW_MONITOR
)

from .nextgen_algorithms_and_templates import (  # noqa: F401
    InterferenceContentionReport,
    MultiTenantInterferenceTensorAlgorithm,
    SymplecticState,
    SymplecticPhaseSpaceGovernorAlgorithm,
    PlacementDecision,
    HeterogeneousBreakEvenPlacementAlgorithm,
    CalibrationStatus,
    OnlineCalibrationLedgerAlgorithm,
    HarmonicTensorPoint,
    GoldenRatioHarmonicTensorAlgorithm,
    NextGenAlgorithmsSuite,
    GLOBAL_NEXTGEN_ALGORITHMS_SUITE
)

from .custom_governor import (  # noqa: F401
    GovernorMode,
    GovernorLimits,
    GovernorDecision,
    ExecutionEnvelope,
    CustomHardwareGovernor
)

from .custom_planner import (  # noqa: F401
    TaskCategory,
    WorkloadTask,
    ScheduledTaskAssignment,
    WorkloadPlan,
    ExecutionFeedbackRecord,
    CustomWorkloadPlanner
)

from .verification.capability_registry import (  # noqa: F401
    CapabilityState,
    HardwareBackend,
    BackendCapability,
    CapabilityRegistry,
    GLOBAL_CAPABILITY_REGISTRY
)

from .hardware_input_driver import (  # noqa: F401
    InputSourceKind,
    UserActivityState,
    RawHardwarePacket,
    NormalizedInputState,
    InputReplayFrame,
    LockFreeInputRingBuffer,
    DeadzoneAndJitterFilter,
    SerialTelemetryDecoder,
    DeterministicInputJournal,
    HardwareInputDriver,
    EpistemicInputBridge,
    GLOBAL_HARDWARE_INPUT_DRIVER
)

from .hardware_input_toolkit import (  # noqa: F401
    HardwareDeviceInfo,
    CalibrationProfile,
    BenchmarkResult,
    HardwareInputToolkit,
    GLOBAL_HARDWARE_INPUT_TOOLKIT
)

from .hardware_input_hal import (  # noqa: F401
    DeviceConnectionStatus,
    FrameIntegrity,
    FrameAuthenticity,
    ReplayVerificationScope,
    ClockCalibration,
    HardwareInputEvent,
    InputEvidenceEnvelope,
    HalDeviceInfo,
    AntiReplayFilter,
    SPSCInputRingBuffer,
    HardwareInputHAL,
    compute_device_fingerprint,
    compute_device_fingerprints,
    GLOBAL_HARDWARE_INPUT_HAL
)

from .llm_hw_acceleration_bridge import (  # noqa: F401
    HardwareInferenceReceipt,
    LlmHardwareAccelerationBridge,
    GLOBAL_LLM_HW_BRIDGE
)

from .procedural_sdf_hw_bridge import (  # noqa: F401
    SdfEvaluationReceipt,
    ProceduralSdfHwAccelerationBridge,
    GLOBAL_PROCEDURAL_SDF_BRIDGE
)

from .uma_inductive_voltage_governor import (  # noqa: F401
    TerminalChannel,
    TerminalChokeState,
    UmaVoltagePlan,
    SafetyThresholdEnvelopes,
    LatentHysteresisStateTracker,
    InductiveChokeTerminalBlock,
    UmaInductiveVoltageGovernor,
    GLOBAL_UMA_INDUCTIVE_VOLTAGE_GOVERNOR
)

from .two_phase_reasoning_computing_alternator import (  # noqa: F401
    AlternatorPhase,
    EpistemicClaimStatus,
    PatternEvidence6Tuple,
    ReasoningHypothesisProposal,
    VerificationAuditReceipt,
    AlternatorCycleReceipt,
    TwoPhaseVerifierGate,
    TwoPhaseReasoningComputingAlternator,
    GLOBAL_REASONING_COMPUTING_ALTERNATOR
)

from .verification.theory_dependency_graph import (  # noqa: F401
    TheoryCategory,
    TheoryEpistemicStatus,
    TheoryNode,
    ExperimentalTheoryDependencyGraph,
    GLOBAL_THEORY_DEPENDENCY_GRAPH
)

from .verification.theory_harvesting_engine import (  # noqa: F401
    HarvestedTheoryRecord,
    TheoryHarvestSummary,
    TheoryHarvestingEngine,
    GLOBAL_THEORY_HARVESTING_ENGINE
)

from .matrix_algorithms_two_phase_bridge import (  # noqa: F401
    MatrixExecutionReceipt,
    MatrixAlgorithmsTwoPhaseBridge,
    GLOBAL_MATRIX_ALGORITHMS_BRIDGE
)

from .cartesian_hex_hsl_grid import (  # noqa: F401
    GridCoefficients3D,
    CartesianGridPoint,
    CartesianHexHslGrid,
    GLOBAL_CARTESIAN_HEX_HSL_GRID
)

from .pattern_rhythm_cadence_calculator import (  # noqa: F401
    PatternCadenceType,
    CadenceAlignmentVerdict,
    PatternRhythmSpec,
    RhythmCalculationReport,
    PatternRhythmCadenceCalculator,
    GLOBAL_PATTERN_RHYTHM_CALCULATOR
)

from .chrono_symplectic_processor_architecture import (  # noqa: F401
    CoreClusterRole,
    ApplicationSpectrumKind,
    ProcessorClusterSpec,
    ApplicationSpectrumDescriptor,
    WorkloadExecutionReceipt,
    ChronoSymplecticProcessorSoC,
    GLOBAL_CHRONO_SYMPLECTIC_SOC
)

from .future_theory_integrations_orchestrator import (  # noqa: F401
    IntegrationTargetSubsystem,
    SubsystemUpgradeReport,
    FutureTheoryIntegrationsOrchestrator,
    GLOBAL_FUTURE_THEORY_INTEGRATIONS_ORCHESTRATOR
)

from .redundant_epistemic_inference_engine import (  # noqa: F401
    AspectCategory,
    InferenceLens,
    InferenceEpistemicStatus,
    AspectDefinition,
    EpistemicInferenceClaim,
    RedundantInferenceSynthesis,
    RedundantEpistemicInferenceEngine,
    GLOBAL_REDUNDANT_INFERENCE_ENGINE
)

from .sovereign_runtime import (  # noqa: F401
    SubsystemNodeId,
    InterpolationMode,
    SpatialHexHslInterpolationResult,
    CadenceHarmonicInterpolationResult,
    SymplecticVerletState,
    VoltageImpedanceState,
    CoordinationChannel,
    RuntimeExecutionCycleReceipt,
    SpatialHexHslInterpolator,
    CadenceHarmonicInterpolator,
    SymplecticPhaseSpaceInterpolator,
    VoltageImpedanceInterpolator,
    KrystalSovereignRuntime,
    GLOBAL_SOVEREIGN_RUNTIME
)

from .sovereign_operations_compiler import (  # noqa: F401
    CompilationStage,
    StageExecutionResult,
    SovereignOperationsReceipt,
    SovereignOperationsCompiler,
    GLOBAL_SOVEREIGN_OPERATIONS_COMPILER
)

from .sovereign_subsystem_turn_coordinator import (  # noqa: F401
    TurnState,
    OperationSafetyVerdict,
    OperationReversibilityClass,
    ArchitectureProfile,
    ArrheniusReliabilityModel,
    CandidateOperation,
    AgileDraftedEnvelope,
    SelectionEvaluationResult,
    TurnToken,
    InstructionReceipt,
    ProceduralAgileDraftingInterpolator,
    SafeOperationSelectionPack,
    run_static_vs_agile_drafting_benchmark,
    AblationConfigurationId,
    AblationConfigurationMetrics,
    ComponentAblationReport,
    run_four_configuration_component_ablation,
)

from .sovereign_hypervisor_toolset import (  # noqa: F401
    HypervisorGuestDomain,
    HypercallKind,
    HypercallSafetyVerdict,
    HypercallCandidate,
    HypercallTurnToken,
    HypercallExecutionReceipt,
    HypervisorGuestMetrics,
    HypervisorAblationMetrics,
    HypervisorAblationReport,
    HypervisorDomainLockArbiter,
    HypervisorReversibilityCoordinator,
    HypervisorUmaHexGovernor,
    HypervisorTurnDiagnostics,
    HypervisorComponentAblationTool,
    SovereignHypervisorConsoleTool,
    GLOBAL_SOVEREIGN_HYPERVISOR_TOOL,
)

from .sovereign_uma_inspector_governor import (  # noqa: F401
    HardwarePriorityLane,
    HexMemoryCellState,
    GovernorAdviceKind,
    MicroOpKind,
    HexGridCellDescriptor,
    TokenTrackingEntry,
    GovernorAdvice,
    MicroOperation,
    PlannedMicroOpBatch,
    UmaInspectorHexGrid,
    HardwarePrioritizationTokenMonitor,
    SovereignGovernorAdvisor,
    MicroOperationAllocationBot,
    GLOBAL_UMA_HEX_GRID,
    GLOBAL_HW_TOKEN_MONITOR,
    GLOBAL_GOVERNOR_ADVISOR,
    GLOBAL_MICRO_OP_BOT,
)

from .sovereign_behavioral_compiler import (  # noqa: F401
    CeilingDomain,
    SublevelDomainId,
    ResourceSwapKind,
    CanonicalPhaseId,
    PotentialRangeMetrics,
    BehavioralSchema,
    PhaseConfirmationResult,
    SublevelDomainPacket,
    EconomicCostQuote,
    UmaRelativeCorrelation,
    BehavioralValidatorEngine,
    VectorPhaseRagEngine,
    ComputationalDifferenceServo,
    SublevelDomainRouter,
    HypervisorEconomicResourceGovernor,
    VulkanUmaRuntimePredictor,
    SovereignBehavioralSchemaCompiler,
    GLOBAL_BEHAVIORAL_SCHEMA_COMPILER,
)

from .sovereign_promptbook import (  # noqa: F401
    PromptbookCategory,
    InvariantSeverity,
    KeyProvenance,
    hkdf_extract,
    hkdf_expand,
    hkdf_derive_domain_key,
    canonical_jcs_serialize,
    InvariantDefinition,
    compute_invariant_set_hash,
    ExecutableInvariant,
    InvariantEvaluationReport,
    FalsificationOracleContract,
    PromptbookRecipe,
    RenderedPrompt,
    OperationalAstProposal,
    PromptExecutionReceipt,
    resolve_promptbook_secret_key,
    canonical_recipe_json,
    SovereignPromptbookRegistry,
    GLOBAL_SOVEREIGN_PROMPTBOOK,
)

from .ksl import (  # noqa: F401
    KslCompiler,
    GLOBAL_KSL_COMPILER,
    KslOptimizationObjective,
    MicroOpDAG,
    KslOperationalAstBridge,
    GLOBAL_KSL_OPERATIONAL_AST_BRIDGE,
)

from .sovereign_reasoning_brain import (  # noqa: F401
    ReasoningMode,
    EvidentialStatus,
    EpistemicModelKind,
    PurposeCategory,
    PatternEvidenceTuple,
    PurposeDescriptor,
    EpistemicModelCandidate,
    ReasoningPatternDescriptor,
    ModelArbitrationReport,
    SovereignReasoningPatternBrain,
    GLOBAL_REASONING_PATTERN_BRAIN,
)

from .sovereign_recurrent_stream_emulator import (  # noqa: F401
    VariationKind,
    ComputationLayerId,
    ActionGenre,
    StreamVariationSample,
    DecodedHarmonicStream,
    VirtualExecutionTrace,
    HistoryDrawerEntry,
    SyntheticInferenceProfile,
    CrossoverActionReceipt,
    RecurrentVariationsStreamDecoder,
    VirtualLayerHostEmulator,
    RecurrentPathwaysGovernorExtension,
    DependencyGraphHistoryDrawer,
    SyntheticInferencePatternScaler,
    ApiCrossoverActionEndpointManager,
    GLOBAL_RECURRENT_STREAM_DECODER,
    GLOBAL_VIRTUAL_HOST_EMULATOR,
    GLOBAL_RECURRENT_GOVERNOR_EXT,
    GLOBAL_HISTORY_DRAWER,
    GLOBAL_SYNTHETIC_PATTERN_SCALER,
    GLOBAL_CROSSOVER_ENDPOINT_MANAGER,
)

from .textbook_inferable_patterns import (  # noqa: F401
    InferablePatternId,
    PatternEvidenceTuple,
    ReasoningTrace11Tuple,
    PhaseCrossoverInferenceResult,
    ChronometricShadowInferenceResult,
    SymplecticLeapfrogInferenceResult,
    DwellHysteresisInferenceResult,
    HexHslBijectionInferenceResult,
    InductiveVoltageInferenceResult,
    PulseTelemetryInferenceResult,
    ArtifactGateInferenceResult,
    TextbookPatternDescriptor,
    TextbookInferablePatternsEngine,
    infer_asymmetric_phase_crossover,
    infer_chronometric_vblank_shadow,
    infer_symplectic_leapfrog_step,
    infer_12frame_dwell_hysteresis,
    infer_hex_hsl_discrete_bijection,
    infer_uma_inductive_voltage_clamping,
    infer_sub50ns_pulse_telemetry,
    infer_artifact_gate_commit,
    GLOBAL_TEXTBOOK_PATTERNS_ENGINE
)

from .sovereign_bot_scale_dimensioner import (  # noqa: F401
    ScaleDomain,
    LayerSystemId,
    ScaleProfile,
    LayerGoverningVariable,
    VariableTuningReceipt,
    OptimalOperatingReceipt,
    SovereignBotScaleDimensioner,
    GLOBAL_SOVEREIGN_BOT_SCALE_DIMENSIONER
)

from .unified_dependency_coordinator import (  # noqa: F401
    RuntimeStreamKind,
    KeyDependencyCoordinatorId,
    StreamElementMapping,
    InterventionProposal,
    RealtimeInterventionReceipt,
    UnifiedStreamSyncReceipt,
    SafeRealtimeInterventionActuator,
    UnifiedCrossStreamDependencyCoordinator,
    GLOBAL_UNIFIED_DEPENDENCY_COORDINATOR
)

from .godot_krystal_edition_packager import (  # noqa: F401
    EngineLayerId,
    EngineLayerSpec,
    EditionPackageManifest,
    EditionPackagingReceipt,
    GodotKrystalEditionPackager,
    GLOBAL_GODOT_KRYSTAL_EDITION_PACKAGER
)

from .sovereign_layer_optimizer import (  # noqa: F401
    OptimizedPdnEvaluation,
    OptimizedSiliconPdnGovernor,
    OptimizedShadowWindowReport,
    OptimizedChronometricShadowScheduler,
    SymplecticParticleState,
    SymplecticStepResult,
    OptimizedSymplecticPhaseSpaceEngine,
    CadenceGovernorEvaluation,
    OptimizedCognitiveCadenceGovernor,
    SerializedContractFrame,
    OptimizedZeroCopyContractSerializer,
    FastInterventionReceipt,
    OptimizedConstitutionalSafetyActuator
)

from .sovereign_run_benchmark_harness import (  # noqa: F401
    LayerRunBenchmarkMetrics,
    FullSovereignRunReceipt,
    SovereignRunBenchmarkHarness,
    GLOBAL_SOVEREIGN_RUN_BENCHMARK_HARNESS
)

from .sovereign_guidebook_registry import (  # noqa: F401
    GuidebookCategory,
    ReadingDifficulty,
    InteractiveSandboxPreset,
    GuidebookSection,
    SovereignGuidebookRegistry,
    GLOBAL_SOVEREIGN_GUIDEBOOK_REGISTRY
)

from .symplectic_animation_transition_engine import (  # noqa: F401
    TransitionCurvePreset,
    PhaseSpacePoint,
    SymplecticTransitionTrajectory,
    HexHslColorKeyframe,
    SymplecticAnimationTransitionEngine,
    GLOBAL_SYMPLECTIC_TRANSITION_ENGINE
)

from krystal_janet.sovereign_janet_api_bridge import (  # noqa: F401
    JanetCommandKind,
    JanetCommandReceipt,
    SovereignJanetSystemBridge,
    GLOBAL_SOVEREIGN_JANET_SYSTEM_BRIDGE
)

from .goniometric_proposer_kernel import (  # noqa: F401
    GoniometricProposalCategory,
    GoniometricRadiusProfile,
    GoniometricActionProposal,
    GoniometricComputingBackend,
    GoniometricProposerKernel,
    get_goniometric_proposer_kernel,
    reset_goniometric_proposer_kernel
)

from .bot_diagnostic_and_tuning_toolkit import (  # noqa: F401
    TuningProfile,
    BottleneckSeverity,
    ScaleDomainProbeResult,
    CrossLayerCouplingAudit,
    BotDiagnosticReport,
    BotTuningAction,
    BotPerformanceTuningPlan,
    BotTuningExecutionReceipt,
    SovereignBotDiagnosticToolkit,
    get_bot_diagnostic_toolkit,
    reset_bot_diagnostic_toolkit
)

from .sovereign_extended_layers import (  # noqa: F401
    BackpressureState,
    ExecutionDispatchReceipt,
    BackpressureAuditReceipt,
    TransactionRecord,
    MerkleAuditReceipt,
    NineLayerSystemAuditReceipt,
    ExecutionFabricL6,
    BackpressureControllerL7,
    TransactionalCommitLedgerL8,
    SovereignExtendedNineLayerHypervisor,
    GLOBAL_SOVEREIGN_EXTENDED_NINE_LAYER_HYPERVISOR
)

from .datacenter_power_cascade_simulator import (  # noqa: F401
    PowerRegime,
    PhaseType,
    DatacenterTopology,
    RegimeParameters,
    DatacenterPowerMetrics,
    DatacenterEconomicReceipt,
    DatacenterPowerCascadeSimulator,
    GLOBAL_DATACENTER_POWER_CASCADE_SIMULATOR
)

from .kphp_protocol import (  # noqa: F401
    KPHP_MAGIC,
    KPHP_VERSION,
    KPHP_FRAME_SIZE_BYTES,
    OperationalStateVector,
    KphpHintFrame,
    KphpCodec
)

from .scenario_market_farm_engine import (  # noqa: F401
    ScenarioKind,
    MarketState,
    FarmState,
    NetworkState,
    ScenarioOutcome,
    FarmExpansionReport,
    CryptoMarketFarmScenarioEngine,
    GLOBAL_CRYPTO_MARKET_FARM_ENGINE
)

from .sovereign_subsystem_turn_coordinator import (  # noqa: F401
    TurnState,
    OperationSafetyVerdict,
    OperationReversibilityClass,
    ArchitectureProfile,
    AgileDraftedEnvelope,
    CandidateOperation,
    SelectionEvaluationResult,
    TurnToken,
    ArrheniusReliabilityModel,
    InstructionReceipt,
    ProceduralAgileDraftingInterpolator,
    SafeOperationSelectionPack,
    InstructionalDependencyCoordinator,
    GLOBAL_SUBSYSTEM_TURN_COORDINATOR,
    run_static_vs_agile_drafting_benchmark,
    AblationConfigurationId,
    AblationConfigurationMetrics,
    ComponentAblationReport,
    run_four_configuration_component_ablation,
)

from .sovereign_uma_inspector_governor import (  # noqa: F401
    HexMemoryCellState,
    GovernorAdviceKind,
    MicroOpKind,
    HexGridCellDescriptor,
    ResourceCapabilityToken,
    GovernorAdvice,
    TokenTrackingEntry,
    MicroOperation,
    PlannedMicroOpBatch,
    UmaInspectorHexGrid,
    HardwarePrioritizationTokenMonitor,
    SovereignGovernorAdvisor,
    MicroOperationAllocationBot,
    ModeBenchmarkMetrics,
    FourModeBenchmarkResult,
    calculate_physical_clock_period_ns,
    run_four_mode_uma_governor_benchmark,
    GLOBAL_UMA_HEX_GRID,
    GLOBAL_HW_TOKEN_MONITOR,
    GLOBAL_GOVERNOR_ADVISOR,
    GLOBAL_MICRO_OP_BOT,
)

from .sovereign_promptbook import (  # noqa: F401
    PromptbookCategory,
    InvariantSeverity,
    ExecutableInvariant,
    InvariantEvaluationReport,
    PromptbookRecipe,
    RenderedPrompt,
    OperationalAstProposal,
    PromptExecutionReceipt,
    SovereignPromptbookRegistry,
    GLOBAL_SOVEREIGN_PROMPTBOOK,
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
    "ClosedLoopExecutionReport", "ClosedLoopWorkloadFabric", "GLOBAL_CLOSED_LOOP_WORKLOAD_FABRIC",
    "EpistemicValue", "EpistemicTruthLattice", "LatticeCertaintyLevel", "ConfidenceInterval",
    "StatePolicyEvidenceVersion", "StatePolicyEvidenceVersioningEngine", "GLOBAL_VERSIONING_ENGINE",
    "SafetyGateResult", "EvidenceGateResult", "DualGateVerdict", "SafetyGate", "EvidenceGate",
    "DualGateEvaluator", "GLOBAL_DUAL_GATE_EVALUATOR", "ExtensionCategory", "RuntimeExtensionRegistry",
    "GLOBAL_RUNTIME_EXTENSION_REGISTRY", "CoreExecutionReport", "KrystalRuntimeCore",
    "GLOBAL_KRYSTAL_RUNTIME_CORE",
    "ProceduralSceneKind", "NpuReconstructionFilter", "ColorBitDepth",
    "NpuSupersamplingSoftwareProfile", "CANONICAL_NPU_PROFILES", "ProceduralFrame",
    "GeometricManifoldSynthesizer", "IntermediateFrameInterpolator", "NpuMultiFrameSupersampler",
    "ImageQualityEvaluationReport", "ImageQualityEvaluator", "SupersampledSceneBundle",
    "GeometricProceduralSupersamplingEngine", "GLOBAL_GEOMETRIC_PROCEDURAL_SUPERSAMPLING_ENGINE",
    "IsaArchitectureFamily", "VectorRegisterWidth", "WorkloadArithmeticTaxon",
    "CacheLocalityTaxon", "ThermalPowerTaxon", "IsaInstructionDescriptor",
    "CANONICAL_ISA_CATALOG", "IsaPerformanceProfile", "IsaRecommendationReport",
    "IsaPerformanceProfiler", "ProfileGuidedIsaGovernor", "GLOBAL_ISA_PERFORMANCE_PROFILER",
    "GLOBAL_ISA_GOVERNOR",
    "SystemStateLevelTaxon", "LanguageLocaleTaxon", "LinguisticRegisterTaxon",
    "SemanticIntentTaxon", "MULTILINGUAL_LEXICON", "MultiLevelSystemState",
    "MultiLevelReRenderReport", "MultilingualIntentClassifier",
    "MultiLevelStateReRenderingEngine", "GLOBAL_MULTILEVEL_STATE_RERENDERING_ENGINE",
    "ModelNode", "ModelNodeSpec", "IntentClassifierNode", "CoreRagNode",
    "ReasoningNode", "CodingSpecialistNode", "VerificationNode", "FinalSynthesisNode",
    "ModelRouter", "NodeRouteCandidate", "RoutingObjectiveWeights", "ROUTING_PROFILES",
    "CascadeEdge", "CascadePlan", "CascadePlanner", "CascadeMerger", "CascadeExecutor",
    "CascadeNodeResult", "CascadeExecutionTrace", "ModelCascadeRuntime", "GLOBAL_MODEL_CASCADE_RUNTIME",
    "HardwareResourceCategory", "HardwareAllocationSnapshot", "CpuSchedulerAllocation",
    "MemoryUmaAllocation", "GpuAcceleratorAllocation", "NpuCascadeAllocation",
    "PowerThermalAllocation", "StorageIoAllocation", "NetworkSocketsAllocation",
    "EpistemicInvariantsAllocation", "LinuxHwMonitorUtility", "GLOBAL_LINUX_HW_MONITOR",
    "InterferenceContentionReport", "MultiTenantInterferenceTensorAlgorithm",
    "SymplecticState", "SymplecticPhaseSpaceGovernorAlgorithm",
    "PlacementDecision", "HeterogeneousBreakEvenPlacementAlgorithm",
    "CalibrationStatus", "OnlineCalibrationLedgerAlgorithm",
    "HarmonicTensorPoint", "GoldenRatioHarmonicTensorAlgorithm",
    "NextGenAlgorithmsSuite", "GLOBAL_NEXTGEN_ALGORITHMS_SUITE",
    "GovernorMode", "GovernorLimits", "GovernorDecision", "ExecutionEnvelope", "CustomHardwareGovernor",
    "TaskCategory", "WorkloadTask", "ScheduledTaskAssignment", "WorkloadPlan", "ExecutionFeedbackRecord", "CustomWorkloadPlanner",
    "CapabilityState", "HardwareBackend", "BackendCapability", "CapabilityRegistry", "GLOBAL_CAPABILITY_REGISTRY",
    "InputSourceKind", "UserActivityState", "RawHardwarePacket", "NormalizedInputState",
    "InputReplayFrame", "LockFreeInputRingBuffer", "DeadzoneAndJitterFilter",
    "SerialTelemetryDecoder", "DeterministicInputJournal", "HardwareInputDriver",
    "EpistemicInputBridge", "GLOBAL_HARDWARE_INPUT_DRIVER",
    "HardwareDeviceInfo", "CalibrationProfile", "BenchmarkResult",
    "HardwareInputToolkit", "GLOBAL_HARDWARE_INPUT_TOOLKIT",
    "DeviceConnectionStatus", "FrameIntegrity", "FrameAuthenticity",
    "ReplayVerificationScope", "ClockCalibration", "HardwareInputEvent",
    "InputEvidenceEnvelope", "HalDeviceInfo", "AntiReplayFilter",
    "SPSCInputRingBuffer", "HardwareInputHAL", "compute_device_fingerprint",
    "compute_device_fingerprints", "GLOBAL_HARDWARE_INPUT_HAL",
    "HardwareInferenceReceipt", "LlmHardwareAccelerationBridge", "GLOBAL_LLM_HW_BRIDGE",
    "SdfEvaluationReceipt", "ProceduralSdfHwAccelerationBridge", "GLOBAL_PROCEDURAL_SDF_BRIDGE",
    "TerminalChannel", "TerminalChokeState", "UmaVoltagePlan",
    "SafetyThresholdEnvelopes", "LatentHysteresisStateTracker",
    "InductiveChokeTerminalBlock", "UmaInductiveVoltageGovernor",
    "GLOBAL_UMA_INDUCTIVE_VOLTAGE_GOVERNOR",
    "AlternatorPhase", "EpistemicClaimStatus", "PatternEvidence6Tuple",
    "ReasoningHypothesisProposal", "VerificationAuditReceipt", "AlternatorCycleReceipt",
    "TwoPhaseVerifierGate", "TwoPhaseReasoningComputingAlternator",
    "GLOBAL_REASONING_COMPUTING_ALTERNATOR",
    "TheoryCategory", "TheoryEpistemicStatus", "TheoryNode",
    "ExperimentalTheoryDependencyGraph", "GLOBAL_THEORY_DEPENDENCY_GRAPH",
    "HarvestedTheoryRecord", "TheoryHarvestSummary",
    "TheoryHarvestingEngine", "GLOBAL_THEORY_HARVESTING_ENGINE",
    "MatrixExecutionReceipt", "MatrixAlgorithmsTwoPhaseBridge",
    "GLOBAL_MATRIX_ALGORITHMS_BRIDGE",
    "GridCoefficients3D", "CartesianGridPoint",
    "CartesianHexHslGrid", "GLOBAL_CARTESIAN_HEX_HSL_GRID",
    "PatternCadenceType", "CadenceAlignmentVerdict",
    "PatternRhythmSpec", "RhythmCalculationReport",
    "PatternRhythmCadenceCalculator", "GLOBAL_PATTERN_RHYTHM_CALCULATOR",
    "CoreClusterRole", "ApplicationSpectrumKind",
    "ProcessorClusterSpec", "ApplicationSpectrumDescriptor",
    "WorkloadExecutionReceipt", "ChronoSymplecticProcessorSoC",
    "GLOBAL_CHRONO_SYMPLECTIC_SOC",
    "IntegrationTargetSubsystem", "SubsystemUpgradeReport",
    "FutureTheoryIntegrationsOrchestrator",
    "GLOBAL_FUTURE_THEORY_INTEGRATIONS_ORCHESTRATOR",
    "AspectCategory", "InferenceLens", "InferenceEpistemicStatus",
    "AspectDefinition", "EpistemicInferenceClaim",
    "RedundantInferenceSynthesis", "RedundantEpistemicInferenceEngine",
    "GLOBAL_REDUNDANT_INFERENCE_ENGINE",
    "SubsystemNodeId", "InterpolationMode",
    "SpatialHexHslInterpolationResult", "CadenceHarmonicInterpolationResult",
    "SymplecticVerletState", "VoltageImpedanceState",
    "CoordinationChannel", "RuntimeExecutionCycleReceipt",
    "SpatialHexHslInterpolator", "CadenceHarmonicInterpolator",
    "SymplecticPhaseSpaceInterpolator", "VoltageImpedanceInterpolator",
    "KrystalSovereignRuntime", "GLOBAL_SOVEREIGN_RUNTIME",
    "CompilationStage", "StageExecutionResult",
    "SovereignOperationsReceipt", "SovereignOperationsCompiler",
    "GLOBAL_SOVEREIGN_OPERATIONS_COMPILER",
    "TurnState", "OperationSafetyVerdict", "CandidateOperation",
    "AgileDraftedEnvelope", "SelectionEvaluationResult", "TurnToken",
    "InstructionReceipt", "ProceduralAgileDraftingInterpolator",
    "SafeOperationSelectionPack", "InstructionalDependencyCoordinator",
    "GLOBAL_SUBSYSTEM_TURN_COORDINATOR",
    "HardwarePriorityLane", "HexMemoryCellState", "GovernorAdviceKind",
    "MicroOpKind", "HexGridCellDescriptor", "TokenTrackingEntry",
    "GovernorAdvice", "MicroOperation", "PlannedMicroOpBatch",
    "UmaInspectorHexGrid", "HardwarePrioritizationTokenMonitor",
    "SovereignGovernorAdvisor", "MicroOperationAllocationBot",
    "GLOBAL_UMA_HEX_GRID", "GLOBAL_HW_TOKEN_MONITOR",
    "GLOBAL_GOVERNOR_ADVISOR", "GLOBAL_MICRO_OP_BOT",
    "CeilingDomain", "SublevelDomainId", "ResourceSwapKind",
    "CanonicalPhaseId", "PotentialRangeMetrics", "BehavioralSchema",
    "PhaseConfirmationResult", "SublevelDomainPacket", "EconomicCostQuote",
    "UmaRelativeCorrelation", "BehavioralValidatorEngine",
    "VectorPhaseRagEngine", "ComputationalDifferenceServo",
    "SublevelDomainRouter", "HypervisorEconomicResourceGovernor",
    "VulkanUmaRuntimePredictor", "SovereignBehavioralSchemaCompiler",
    "GLOBAL_BEHAVIORAL_SCHEMA_COMPILER",
    "PromptbookCategory", "PromptbookRecipe", "RenderedPrompt",
    "SovereignPromptbookRegistry", "GLOBAL_SOVEREIGN_PROMPTBOOK",
    "VariationKind", "ComputationLayerId", "ActionGenre",
    "StreamVariationSample", "DecodedHarmonicStream", "VirtualExecutionTrace",
    "HistoryDrawerEntry", "SyntheticInferenceProfile", "CrossoverActionReceipt",
    "RecurrentVariationsStreamDecoder", "VirtualLayerHostEmulator",
    "RecurrentPathwaysGovernorExtension", "DependencyGraphHistoryDrawer",
    "SyntheticInferencePatternScaler", "ApiCrossoverActionEndpointManager",
    "GLOBAL_RECURRENT_STREAM_DECODER", "GLOBAL_VIRTUAL_HOST_EMULATOR",
    "GLOBAL_RECURRENT_GOVERNOR_EXT", "GLOBAL_HISTORY_DRAWER",
    "GLOBAL_SYNTHETIC_PATTERN_SCALER", "GLOBAL_CROSSOVER_ENDPOINT_MANAGER",
    "InferablePatternId", "PatternEvidenceTuple", "ReasoningTrace11Tuple",
    "PhaseCrossoverInferenceResult", "ChronometricShadowInferenceResult",
    "SymplecticLeapfrogInferenceResult", "DwellHysteresisInferenceResult",
    "HexHslBijectionInferenceResult", "InductiveVoltageInferenceResult",
    "PulseTelemetryInferenceResult", "ArtifactGateInferenceResult",
    "TextbookPatternDescriptor", "TextbookInferablePatternsEngine",
    "infer_asymmetric_phase_crossover", "infer_chronometric_vblank_shadow",
    "infer_symplectic_leapfrog_step", "infer_12frame_dwell_hysteresis",
    "infer_hex_hsl_discrete_bijection", "infer_uma_inductive_voltage_clamping",
    "infer_sub50ns_pulse_telemetry", "infer_artifact_gate_commit",
    "GLOBAL_TEXTBOOK_PATTERNS_ENGINE",
    "ScaleDomain", "LayerSystemId", "ScaleProfile",
    "LayerGoverningVariable", "VariableTuningReceipt",
    "OptimalOperatingReceipt", "SovereignBotScaleDimensioner",
    "GLOBAL_SOVEREIGN_BOT_SCALE_DIMENSIONER",
    "RuntimeStreamKind", "KeyDependencyCoordinatorId",
    "StreamElementMapping", "InterventionProposal",
    "RealtimeInterventionReceipt", "UnifiedStreamSyncReceipt",
    "SafeRealtimeInterventionActuator",
    "UnifiedCrossStreamDependencyCoordinator",
    "GLOBAL_UNIFIED_DEPENDENCY_COORDINATOR",
    "EngineLayerId", "EngineLayerSpec",
    "EditionPackageManifest", "EditionPackagingReceipt",
    "GodotKrystalEditionPackager",
    "GLOBAL_GODOT_KRYSTAL_EDITION_PACKAGER",
    "OptimizedPdnEvaluation", "OptimizedSiliconPdnGovernor",
    "OptimizedShadowWindowReport", "OptimizedChronometricShadowScheduler",
    "SymplecticParticleState", "SymplecticStepResult",
    "OptimizedSymplecticPhaseSpaceEngine",
    "CadenceGovernorEvaluation", "OptimizedCognitiveCadenceGovernor",
    "SerializedContractFrame", "OptimizedZeroCopyContractSerializer",
    "FastInterventionReceipt", "OptimizedConstitutionalSafetyActuator",
    "LayerRunBenchmarkMetrics", "FullSovereignRunReceipt",
    "SovereignRunBenchmarkHarness",
    "GLOBAL_SOVEREIGN_RUN_BENCHMARK_HARNESS",
    "GuidebookCategory", "ReadingDifficulty",
    "InteractiveSandboxPreset", "GuidebookSection",
    "SovereignGuidebookRegistry",
    "GLOBAL_SOVEREIGN_GUIDEBOOK_REGISTRY",
    "TransitionCurvePreset", "PhaseSpacePoint",
    "SymplecticTransitionTrajectory", "HexHslColorKeyframe",
    "SymplecticAnimationTransitionEngine",
    "GLOBAL_SYMPLECTIC_TRANSITION_ENGINE",
    "JanetCommandKind", "JanetCommandReceipt",
    "SovereignJanetSystemBridge",
    "GoniometricProposalCategory", "GoniometricRadiusProfile",
    "GoniometricActionProposal", "GoniometricComputingBackend",
    "GoniometricProposerKernel", "get_goniometric_proposer_kernel",
    "reset_goniometric_proposer_kernel",
    "TuningProfile", "BottleneckSeverity",
    "ScaleDomainProbeResult", "CrossLayerCouplingAudit",
    "BotDiagnosticReport", "BotTuningAction",
    "BotPerformanceTuningPlan", "BotTuningExecutionReceipt",
    "SovereignBotDiagnosticToolkit", "get_bot_diagnostic_toolkit",
    "reset_bot_diagnostic_toolkit",
    "BackpressureState", "ExecutionDispatchReceipt",
    "BackpressureAuditReceipt", "TransactionRecord",
    "MerkleAuditReceipt", "NineLayerSystemAuditReceipt",
    "ExecutionFabricL6", "BackpressureControllerL7",
    "TransactionalCommitLedgerL8", "SovereignExtendedNineLayerHypervisor",
    "GLOBAL_SOVEREIGN_EXTENDED_NINE_LAYER_HYPERVISOR",
    "PowerRegime", "PhaseType", "DatacenterTopology",
    "RegimeParameters", "DatacenterPowerMetrics", "DatacenterEconomicReceipt",
    "DatacenterPowerCascadeSimulator", "GLOBAL_DATACENTER_POWER_CASCADE_SIMULATOR",
    "KPHP_MAGIC", "KPHP_VERSION", "KPHP_FRAME_SIZE_BYTES",
    "OperationalStateVector", "KphpHintFrame", "KphpCodec",
    "ScenarioKind", "MarketState", "FarmState", "NetworkState",
    "ScenarioOutcome", "FarmExpansionReport",
    "CryptoMarketFarmScenarioEngine", "GLOBAL_CRYPTO_MARKET_FARM_ENGINE",
    "TurnState", "OperationSafetyVerdict", "OperationReversibilityClass",
    "ArchitectureProfile", "AgileDraftedEnvelope", "CandidateOperation",
    "SelectionEvaluationResult", "TurnToken", "ArrheniusReliabilityModel",
    "InstructionReceipt", "ProceduralAgileDraftingInterpolator",
    "SafeOperationSelectionPack", "InstructionalDependencyCoordinator",
    "GLOBAL_SUBSYSTEM_TURN_COORDINATOR", "run_static_vs_agile_drafting_benchmark",
    "AblationConfigurationId", "AblationConfigurationMetrics",
    "ComponentAblationReport", "run_four_configuration_component_ablation",
    "HypervisorGuestDomain", "HypercallKind", "HypercallSafetyVerdict",
    "HypercallCandidate", "HypercallTurnToken", "HypercallExecutionReceipt",
    "HypervisorGuestMetrics", "HypervisorAblationMetrics", "HypervisorAblationReport",
    "HypervisorDomainLockArbiter", "HypervisorReversibilityCoordinator",
    "HypervisorUmaHexGovernor", "HypervisorTurnDiagnostics",
    "HypervisorComponentAblationTool", "SovereignHypervisorConsoleTool",
    "GLOBAL_SOVEREIGN_HYPERVISOR_TOOL",
    "HexMemoryCellState", "GovernorAdviceKind", "MicroOpKind",
    "HexGridCellDescriptor", "ResourceCapabilityToken",
    "GovernorAdvice", "TokenTrackingEntry", "MicroOperation",
    "PlannedMicroOpBatch", "UmaInspectorHexGrid",
    "HardwarePrioritizationTokenMonitor", "SovereignGovernorAdvisor",
    "MicroOperationAllocationBot", "ModeBenchmarkMetrics",
    "FourModeBenchmarkResult", "calculate_physical_clock_period_ns",
    "run_four_mode_uma_governor_benchmark",
    "GLOBAL_UMA_HEX_GRID", "GLOBAL_HW_TOKEN_MONITOR",
    "GLOBAL_GOVERNOR_ADVISOR", "GLOBAL_MICRO_OP_BOT",
    "PromptbookCategory", "InvariantSeverity", "KeyProvenance",
    "hkdf_extract", "hkdf_expand", "hkdf_derive_domain_key",
    "canonical_jcs_serialize", "InvariantDefinition", "compute_invariant_set_hash",
    "ExecutableInvariant", "InvariantEvaluationReport", "FalsificationOracleContract",
    "PromptbookRecipe", "RenderedPrompt",
    "OperationalAstProposal", "PromptExecutionReceipt",
    "resolve_promptbook_secret_key", "canonical_recipe_json",
    "SovereignPromptbookRegistry", "GLOBAL_SOVEREIGN_PROMPTBOOK",
    "ReasoningMode", "EvidentialStatus", "EpistemicModelKind",
    "PurposeCategory", "PatternEvidenceTuple", "PurposeDescriptor",
    "EpistemicModelCandidate", "ReasoningPatternDescriptor",
    "ModelArbitrationReport", "SovereignReasoningPatternBrain",
    "GLOBAL_REASONING_PATTERN_BRAIN",
    "KslCompiler", "GLOBAL_KSL_COMPILER",
    "KslOptimizationObjective", "MicroOpDAG",
    "KslOperationalAstBridge", "GLOBAL_KSL_OPERATIONAL_AST_BRIDGE"
]

