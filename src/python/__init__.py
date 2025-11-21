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
]
