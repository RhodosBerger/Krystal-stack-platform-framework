"""
GAMESA/KrystalStack - Cognitive Stream (Python)

This module implements the Cognitive Stream components:
- MetacognitiveInterface: Self-reflecting analysis of system performance
- PolicyProposalGenerator: LLM-driven policy proposals
- ExperienceStore: State-Action-Reward storage and retrieval
"""

from .metacognitive import MetacognitiveInterface
from .experience_store import ExperienceStore
from .policy_generator import PolicyProposalGenerator
from .feature_engine import (
    FeatureEngine, DbFeatureTransformer, ScaleParams,
    LogBase, TrigFunc, scale_alpha_beta_theta, log_scale, trig_encode
)
from .runtime import Runtime, RuntimeVar, RuntimeFunc, VarSource, quick_eval
from .allocation import (
    Allocator, ResourcePool, AllocationRequest, Allocation,
    AllocationStrategy, ResourceType, Priority, AllocationConstraints,
    PoolStats, create_default_allocator
)
from .effects import (
    Effect, Capability, CapabilityConstraints, EffectDeclaration,
    EffectChecker, CompositionResult, create_guardian_checker
)
from .contracts import (
    Contract, Condition, CompareOp, ViolationAction, ContractValidator,
    ValidationResult, Violation, range_check, not_null_check, compare_check,
    contract, ContractViolationError, create_guardian_validator,
    TELEMETRY_CONTRACT, DIRECTIVE_CONTRACT, SAFETY_CONTRACT
)
from .signals import (
    Signal, SignalSource, SignalKind, Domain, SignalScheduler,
    telemetry_signal, safety_signal, user_signal, ipc_signal
)
from .llm_bridge import (
    LLMBridge, LLMConfig, LLMResponse, LLMCache, LLMProvider, LLMError,
    create_llm_bridge
)
from .document_processor import (
    DocumentProcessor, DocumentRecord, ExtractedData, DocumentType,
    ProcessingStatus, OCREngine, PDFProcessor
)
from .mongo_store import (
    MongoStore, MongoConfig, CollectionName, InMemoryCollection,
    create_mongo_store
)
from .flask_app import create_app, run_app
from .content_generator import (
    ContentGenerator, ContentType, GameContext, GeneratedContent,
    create_content_generator
)
from .hybrid_event_pipeline import (
    HybridEventPipeline, TelemetrySnapshot, DirectiveDecision,
    PresetType, RewardSignal, OpenVINOBridge, ProcessPredictor,
    create_hybrid_pipeline
)
from .synthesis_dashboard import (
    SynthesisDashboard, SystemState, ZoneInfo, ViewMode,
    create_synthesis_dashboard
)
from .prioritized_replay import (
    PrioritizedReplayBuffer, Experience, SumTree,
    ThermalPredictor, GeneticPresetEvolver,
    create_prioritized_buffer, create_thermal_predictor, create_preset_evolver
)
from .shared_memory_ipc import (
    SharedMemoryRingBuffer, TelemetryPacket, SignalPacket,
    TelemetryChannel, SignalChannel, IPCBridge,
    create_ipc_server, create_ipc_client
)
from .guardian_hooks import (
    GuardianBridge, ValidatorBridge, ZoneState, BoostConfig, CraftPreset,
    BoostAction, CorePolicy, create_guardian_bridge, create_validator_bridge
)
from .kernel_bridge import (
    GamesaKernel, Component, River, RiverEndpoint, RiverMessage,
    ComponentType, ComponentState, RiverTransport, RiverDirection, RiverQoS,
    create_kernel, create_full_stack_kernel
)
from .cognitive_engine import (
    CognitiveEngine, FeedbackController, TDLearner, BayesianTracker,
    StatMechAllocator, EntropyAnalyzer, EvolutionaryOptimizer,
    SystemState, ControlSignal, Preset, create_cognitive_engine
)
from .nvidia_optimizer import (
    NvidiaOptimizer, NvidiaPreset, NvidiaGpuState, NvmlBridge,
    TensorRTBridge, create_nvidia_optimizer
)
from .amd_optimizer import (
    AmdOptimizer, AmdPreset, AmdGpuState, AmdgpuSysfs,
    RocmBridge, create_amd_optimizer
)
from .gpu_optimizer import (
    UnifiedGpuOptimizer, GpuVendor, UnifiedPreset, GpuInfo,
    OptimizationResult, GpuDetector, create_gpu_optimizer
)
from .generative_engine import (
    GenerativeEngine, GeneratorType, OutputType, QualityLevel,
    LatentVector, GeneratedOutput, GenerationRequest, GeneratorConfig,
    OpenVINOGenerator, ProceduralGenerator, RecipeGenerator,
    create_generative_engine
)
from .breakthrough_engine import (
    BreakthroughEngine, TemporalPredictor, NeuralHardwareFabric,
    QuantumInspiredOptimizer, SelfModifyingEngine, SwarmIntelligence,
    FutureState, HardwareNeuron, QuantumPreset, CodePatch, SwarmNode,
    create_breakthrough_engine
)
from .hypervisor_layer import (
    Hypervisor, ConsciousnessEngine, RealitySynthesizer, ZeroCopyFabric,
    ConsciousnessState, AwarenessLevel, RealityBudget, UnifiedPointer,
    MemoryRegion, SystemPhase, MetaLearner, create_hypervisor
)
from .emergent_intelligence import (
    EmergentIntelligence, AttractorLandscape, PhaseTransitionEngine,
    CollectiveIntelligence, SynapseNetwork, Attractor, Agent, Synapse,
    PhaseState, create_emergent_intelligence
)
from .unified_system import (
    UnifiedSystem, HardwareLevel, SignalLevel, LearningLevel,
    PredictionLevel, EmergenceLevel, GenerationLevel,
    SystemMode, SystemMetrics, create_unified_system
)
from .realtime_monitor import RealtimeMonitor, create_monitor
from .cli import CLI, main as cli_main

__all__ = [
    # Core
    "MetacognitiveInterface", "ExperienceStore", "PolicyProposalGenerator",
    # Features
    "FeatureEngine", "DbFeatureTransformer", "ScaleParams",
    "LogBase", "TrigFunc", "scale_alpha_beta_theta", "log_scale", "trig_encode",
    # Runtime
    "Runtime", "RuntimeVar", "RuntimeFunc", "VarSource", "quick_eval",
    # Allocation
    "Allocator", "ResourcePool", "AllocationRequest", "Allocation",
    "AllocationStrategy", "ResourceType", "Priority", "AllocationConstraints",
    "PoolStats", "create_default_allocator",
    # Effects
    "Effect", "Capability", "CapabilityConstraints", "EffectDeclaration",
    "EffectChecker", "CompositionResult", "create_guardian_checker",
    # Contracts
    "Contract", "Condition", "CompareOp", "ViolationAction", "ContractValidator",
    "ValidationResult", "Violation", "range_check", "not_null_check", "compare_check",
    "contract", "ContractViolationError", "create_guardian_validator",
    "TELEMETRY_CONTRACT", "DIRECTIVE_CONTRACT", "SAFETY_CONTRACT",
    # Signals
    "Signal", "SignalSource", "SignalKind", "Domain", "SignalScheduler",
    "telemetry_signal", "safety_signal", "user_signal", "ipc_signal",
    # LLM Bridge
    "LLMBridge", "LLMConfig", "LLMResponse", "LLMCache", "LLMProvider",
    "LLMError", "create_llm_bridge",
    # Document Processor
    "DocumentProcessor", "DocumentRecord", "ExtractedData", "DocumentType",
    "ProcessingStatus", "OCREngine", "PDFProcessor",
    # MongoDB Store
    "MongoStore", "MongoConfig", "CollectionName", "InMemoryCollection",
    "create_mongo_store",
    # Flask App
    "create_app", "run_app",
    # Content Generator
    "ContentGenerator", "ContentType", "GameContext", "GeneratedContent",
    "create_content_generator",
    # Hybrid Pipeline
    "HybridEventPipeline", "TelemetrySnapshot", "DirectiveDecision",
    "PresetType", "RewardSignal", "OpenVINOBridge", "ProcessPredictor",
    "create_hybrid_pipeline",
    # Synthesis Dashboard
    "SynthesisDashboard", "SystemState", "ZoneInfo", "ViewMode",
    "create_synthesis_dashboard",
    # Prioritized Replay & Learning
    "PrioritizedReplayBuffer", "Experience", "SumTree",
    "ThermalPredictor", "GeneticPresetEvolver",
    "create_prioritized_buffer", "create_thermal_predictor", "create_preset_evolver",
    # Shared Memory IPC
    "SharedMemoryRingBuffer", "TelemetryPacket", "SignalPacket",
    "TelemetryChannel", "SignalChannel", "IPCBridge",
    "create_ipc_server", "create_ipc_client",
    # Guardian Hooks
    "GuardianBridge", "ValidatorBridge", "ZoneState", "BoostConfig", "CraftPreset",
    "BoostAction", "CorePolicy", "create_guardian_bridge", "create_validator_bridge",
    # Kernel
    "GamesaKernel", "Component", "River", "RiverEndpoint", "RiverMessage",
    "ComponentType", "ComponentState", "RiverTransport", "RiverDirection", "RiverQoS",
    "create_kernel", "create_full_stack_kernel",
    # Cognitive Engine
    "CognitiveEngine", "FeedbackController", "TDLearner", "BayesianTracker",
    "StatMechAllocator", "EntropyAnalyzer", "EvolutionaryOptimizer",
    "SystemState", "ControlSignal", "Preset", "create_cognitive_engine",
    # NVIDIA Optimizer
    "NvidiaOptimizer", "NvidiaPreset", "NvidiaGpuState", "NvmlBridge",
    "TensorRTBridge", "create_nvidia_optimizer",
    # AMD Optimizer
    "AmdOptimizer", "AmdPreset", "AmdGpuState", "AmdgpuSysfs",
    "RocmBridge", "create_amd_optimizer",
    # Unified GPU Optimizer
    "UnifiedGpuOptimizer", "GpuVendor", "UnifiedPreset", "GpuInfo",
    "OptimizationResult", "GpuDetector", "create_gpu_optimizer",
    # Generative Engine
    "GenerativeEngine", "GeneratorType", "OutputType", "QualityLevel",
    "LatentVector", "GeneratedOutput", "GenerationRequest", "GeneratorConfig",
    "OpenVINOGenerator", "ProceduralGenerator", "RecipeGenerator",
    "create_generative_engine",
    # Breakthrough Engine
    "BreakthroughEngine", "TemporalPredictor", "NeuralHardwareFabric",
    "QuantumInspiredOptimizer", "SelfModifyingEngine", "SwarmIntelligence",
    "FutureState", "HardwareNeuron", "QuantumPreset", "CodePatch", "SwarmNode",
    "create_breakthrough_engine",
    # Hypervisor Layer
    "Hypervisor", "ConsciousnessEngine", "RealitySynthesizer", "ZeroCopyFabric",
    "ConsciousnessState", "AwarenessLevel", "RealityBudget", "UnifiedPointer",
    "MemoryRegion", "SystemPhase", "MetaLearner", "create_hypervisor",
    # Emergent Intelligence
    "EmergentIntelligence", "AttractorLandscape", "PhaseTransitionEngine",
    "CollectiveIntelligence", "SynapseNetwork", "Attractor", "Agent", "Synapse",
    "PhaseState", "create_emergent_intelligence",
    # Unified System
    "UnifiedSystem", "HardwareLevel", "SignalLevel", "LearningLevel",
    "PredictionLevel", "EmergenceLevel", "GenerationLevel",
    "SystemMode", "SystemMetrics", "create_unified_system",
    # Realtime Monitor
    "RealtimeMonitor", "create_monitor",
    # CLI
    "CLI", "cli_main",
]
