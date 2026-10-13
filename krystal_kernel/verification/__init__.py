"""
KRYSTAL-STACK // UNIFIED SIMULATION & VERIFICATION LAYER
=============================================================================
Unified, multi-disciplinary verification and epistemic simulation framework:
  1. Game Simulation (Poslední Kmen)
  2. Causal LLM Runtime (CrystalSec 8-Stage Execution Plane)
  3. Cyber-Physical CNC Dynamics
  4. Linux Kernel & Resource Isolation
  5. Fault-Injection & Resilience Lab

Preserves VITAL_MAX_HP == 6 across all operations.
=============================================================================
"""

from krystal_kernel.verification.state_vector import (
    StateVector,
    StateGeometry,
    StateVectorDelta,
    TradeoffClassification,
    ParetoStatus,
)
from krystal_kernel.verification.invariants import (
    UniversalInvariantLayer,
    InvariantRule,
    InvariantViolationEvent,
    InvariantSeverity,
)
from krystal_kernel.verification.evidence import (
    EvidenceLevel,
    EpistemicState,
    EpistemicClaim,
    EvidenceRegistry,
)
from krystal_kernel.verification.policy_engine import (
    ProposedAction,
    VerificationPolicy,
    PolicyTransformationEngine,
)
from krystal_kernel.verification.outcome_evaluator import (
    EvaluationReport,
    OutcomeEvaluator,
)
from krystal_kernel.verification.replay_engine import (
    ReplayPackage,
    ReplayExecutionResult,
    ReplayEngine,
)
from krystal_kernel.verification.cross_domain_comparator import (
    DomainArchetype,
    DOMAIN_ARCHETYPES,
    DomainComparisonSummary,
    CrossDomainComparator,
)
from krystal_kernel.verification.unified_facade import (
    VerificationExperimentResult,
    UnifiedVerificationFacade,
    GLOBAL_VERIFICATION_FACADE,
)
from krystal_kernel.verification.live_policy_consumer import (
    LivePolicyConsumer,
    GLOBAL_LIVE_POLICY_CONSUMER,
    AntiOscillationGuard,
    ExperimentLedgerEntry,
    ConsumerMode,
)

__all__ = [
    "StateVector",
    "StateGeometry",
    "StateVectorDelta",
    "TradeoffClassification",
    "ParetoStatus",
    "UniversalInvariantLayer",
    "InvariantRule",
    "InvariantViolationEvent",
    "InvariantSeverity",
    "EvidenceLevel",
    "EpistemicState",
    "EpistemicClaim",
    "EvidenceRegistry",
    "ProposedAction",
    "VerificationPolicy",
    "PolicyTransformationEngine",
    "EvaluationReport",
    "OutcomeEvaluator",
    "ReplayPackage",
    "ReplayExecutionResult",
    "ReplayEngine",
    "DomainArchetype",
    "DOMAIN_ARCHETYPES",
    "DomainComparisonSummary",
    "CrossDomainComparator",
    "VerificationExperimentResult",
    "UnifiedVerificationFacade",
    "GLOBAL_VERIFICATION_FACADE",
    "LivePolicyConsumer",
    "GLOBAL_LIVE_POLICY_CONSUMER",
    "AntiOscillationGuard",
    "ExperimentLedgerEntry",
    "ConsumerMode",
    # Epistemic Runtime Skeleton (PROT Sprint A)
    "StateVersion",
    "ValidityState",
    "EvidenceKind",
    "EpistemicValue",
    "PolicyVersion",
    "DynamicSafetyPolicy",
    "DynamicEvidencePolicy",
    "PolicyRegistry",
    "GLOBAL_POLICY_REGISTRY",
    "ResourceVector",
    "ExpectedDelta",
    "ResourceEvidenceEnvelope",
    "SafetyTier",
    "SafetyVerdictType",
    "EvidenceVerdictType",
    "AuthorityVerdictType",
    "ReversibilityClass",
    "FailureLifecycleStage",
    "SafetyViolationRecord",
    "SafetyProof",
    "EvidenceProof",
    "AuthorityProofBundle",
    "CandidateAction",
    "SafetyGate",
    "EvidenceGate",
    "DualAuthorityGate",
    # Epistemic Taxonomies (Boolean, List, Predictive Graph)
    "EpistemicBoolean",
    "BooleanModalKind",
    "BooleanTemporalKind",
    "BooleanInvariantKind",
    "BooleanActionKind",
    "EpistemicList",
    "ListAlgebraicKind",
    "ListCapacityKind",
    "ListIntegrityKind",
    "ListMemoryKind",
    "create_vital_hearts_list",
    "PredictiveDependencyGraph",
    "PredictiveDependencyNode",
    "DependencyEdgeKind",
]

from krystal_kernel.verification.state_version import StateVersion
from krystal_kernel.verification.epistemic_value import (
    ValidityState,
    EvidenceKind,
    EpistemicValue
)
from krystal_kernel.verification.policy_registry import (
    PolicyVersion,
    DynamicSafetyPolicy,
    DynamicEvidencePolicy,
    PolicyRegistry,
    GLOBAL_POLICY_REGISTRY
)
from krystal_kernel.verification.resource_evidence_envelope import (
    ResourceVector,
    ExpectedDelta,
    ResourceEvidenceEnvelope
)
from krystal_kernel.verification.proof_bundle import (
    SafetyTier,
    SafetyVerdictType,
    EvidenceVerdictType,
    AuthorityVerdictType,
    ReversibilityClass,
    FailureLifecycleStage,
    SafetyViolationRecord,
    SafetyProof,
    EvidenceProof,
    AuthorityProofBundle
)
from krystal_kernel.verification.authority_gate import (
    CandidateAction,
    SafetyGate,
    EvidenceGate,
    DualAuthorityGate
)
from krystal_kernel.verification.epistemic_taxonomies import (
    EpistemicBoolean,
    BooleanModalKind,
    BooleanTemporalKind,
    BooleanInvariantKind,
    BooleanActionKind,
    EpistemicList,
    ListAlgebraicKind,
    ListCapacityKind,
    ListIntegrityKind,
    ListMemoryKind,
    create_vital_hearts_list,
    PredictiveDependencyGraph,
    PredictiveDependencyNode,
    DependencyEdgeKind,
)
