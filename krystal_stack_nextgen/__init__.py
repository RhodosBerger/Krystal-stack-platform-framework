"""
==============================================================================
KRYSTAL-STACK NEXTGEN (v2 ARCHITECTURE)
==============================================================================
Next-Generation Energy-Efficient Kernel, Iris Xe Bandwidth Optimization,
and Anti-Mining Proof-of-Visual-Work Governance.

Key Pillars:
  1. Iris Xe Kisak Driver Optimization:
     - Tile4/TileY 2D cache tiling, auxiliary buffer compression (CCS),
       and SIMD16 subgroup dispatch eliminating UMA DRAM bus saturation.
  2. Proof-of-Visual-Work & Anti-Mining Power Governor:
     - Discovers and halts pathological compute loops that burn watts without
       producing fluid display presentation.
     - Enforces auditable energy-tariff compliance (Joules/Frame attribution).
  3. Divergence-Free Bounded Subgroup Raymarcher:
     - AABB pre-culling and subgroup ballot execution preventing warp stalls.
  4. Inviolable Invariant:
     - VITAL_MAX_HP = 6.

Author: Dušan Kopecký & Krystal-Stack Architecture Council (2026)
==============================================================================
"""

from .iris_xe_kisak_optimizer import (
    VITAL_MAX_HP,
    IrisXeMemoryArchitecture,
    IrisXeKisakOptimizer,
    BandwidthOptimizationPlan,
    GLOBAL_IRIS_XE_OPTIMIZER
)

from .anti_mining_power_governor import (
    WorkloadCategory,
    WorkloadEnergyAudit,
    AntiMiningPowerGovernor,
    GLOBAL_ENERGY_GOVERNOR
)

from .subgroup_raymarch_kernel import (
    RaymarchKernelProfile,
    SubgroupRaymarchKernel,
    GLOBAL_SUBGROUP_KERNEL
)

from .npu_speculative_predictor import (
    NPUPredictionAction,
    NPUSpeculativeStrategy,
    NPUTelemetrySnapshot,
    NPUSpeculativePreStager,
    NPUHardwarePredictor,
    GLOBAL_NPU_PREDICTOR
)

from .cross_platform_thermal_bus_governor import (
    TargetPlatform,
    ThermalState,
    MemoryBusProfile,
    Float16PrecisionImpact,
    NPUPredictionMetrics,
    CrossPlatformThermalPlan,
    CrossPlatformThermalBusGovernor,
    GLOBAL_THERMAL_BUS_GOVERNOR
)

from .tpu_tensor_benchmark import (
    TensorPrecision,
    TPUBenchmarkResult,
    TPUSweepSummary,
    TPUTensorBenchmark,
    GLOBAL_TPU_BENCHMARK
)

from .ontological_reverse_prompts import (
    OntologicalDomain,
    OntologicalPrompt,
    OntologicalReverseEngineeringEngine,
    GLOBAL_ONTOLOGICAL_ENGINE
)

from .pattern_metrics_and_ipc_governor import (
    PatternDomain,
    OptimizationPattern,
    PatternMetricsReport,
    BinaryIPCPacket,
    FastTextProcessor,
    PatternMetricsAndIPCGovernor,
    GLOBAL_PATTERN_IPC_GOVERNOR
)

from .ipc_application_benchmark import (
    DomainBenchmarkResult,
    IPCApplicationBenchmarkSuite,
    GLOBAL_IPC_BENCHMARK
)

from .vulkan_ipc_bridge import (
    VulkanIPCOpcode,
    VulkanIPCBridge,
    VulkanIPCClient,
    GLOBAL_VULKAN_IPC_BRIDGE
)

from .multi_gpu_stream_scaler import (
    MultiGPUScalingTopology,
    TopologyEvaluation,
    MultiGPUScalingReport,
    P2PRingBufferGovernor,
    IndirectDispatchPlanner,
    MultiGPUDeviceManager,
    GLOBAL_MULTI_GPU_SCALER
)

__all__ = [
    "VITAL_MAX_HP",
    "IrisXeMemoryArchitecture",
    "IrisXeKisakOptimizer",
    "BandwidthOptimizationPlan",
    "GLOBAL_IRIS_XE_OPTIMIZER",
    "WorkloadCategory",
    "WorkloadEnergyAudit",
    "AntiMiningPowerGovernor",
    "GLOBAL_ENERGY_GOVERNOR",
    "RaymarchKernelProfile",
    "SubgroupRaymarchKernel",
    "GLOBAL_SUBGROUP_KERNEL",
    "NPUPredictionAction",
    "NPUSpeculativeStrategy",
    "NPUTelemetrySnapshot",
    "NPUSpeculativePreStager",
    "NPUHardwarePredictor",
    "GLOBAL_NPU_PREDICTOR",
    "TargetPlatform",
    "ThermalState",
    "MemoryBusProfile",
    "Float16PrecisionImpact",
    "NPUPredictionMetrics",
    "CrossPlatformThermalPlan",
    "CrossPlatformThermalBusGovernor",
    "GLOBAL_THERMAL_BUS_GOVERNOR",
    "TensorPrecision",
    "TPUBenchmarkResult",
    "TPUSweepSummary",
    "TPUTensorBenchmark",
    "GLOBAL_TPU_BENCHMARK",
    "OntologicalDomain",
    "OntologicalPrompt",
    "OntologicalReverseEngineeringEngine",
    "GLOBAL_ONTOLOGICAL_ENGINE",
    "PatternDomain",
    "OptimizationPattern",
    "PatternMetricsReport",
    "BinaryIPCPacket",
    "FastTextProcessor",
    "PatternMetricsAndIPCGovernor",
    "GLOBAL_PATTERN_IPC_GOVERNOR",
    "DomainBenchmarkResult",
    "IPCApplicationBenchmarkSuite",
    "GLOBAL_IPC_BENCHMARK",
    "VulkanIPCOpcode",
    "VulkanIPCBridge",
    "VulkanIPCClient",
    "GLOBAL_VULKAN_IPC_BRIDGE",
    "MultiGPUScalingTopology",
    "TopologyEvaluation",
    "MultiGPUScalingReport",
    "P2PRingBufferGovernor",
    "IndirectDispatchPlanner",
    "MultiGPUDeviceManager",
    "GLOBAL_MULTI_GPU_SCALER"
]


