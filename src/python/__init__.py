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
]
