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
    "GLOBAL_COPING_CORTEX_MCP_BRIDGE"
]
