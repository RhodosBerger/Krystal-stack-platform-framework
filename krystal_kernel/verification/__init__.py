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
]
