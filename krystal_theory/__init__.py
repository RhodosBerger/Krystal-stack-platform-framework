"""
KRYSTAL-STACK FORMAL THEORY RESOLUTION PACKAGE
==============================================
Provides mathematical operators, reachability triage, and WSPT scheduling
derivations resolving P1 and P2 from OTVORENE_PROBLEMY.md.
"""

from .strain_operator import (
    StrainState,
    StrainDomainModel,
    compute_apex_strain,
    CandidateStrainTransform
)

from .triage_operator import (
    MechanismRule,
    ReachabilityStatus,
    ReachabilityResult,
    TriageOperator,
    MechanismDescriptor
)

from .wspt_ranker import (
    SchedulableAction,
    SmithRanker,
    derive_w_comp_upper_bound,
    StressWeightAmplifier
)

from .formal_theory_bridge import (
    FormalTheoryBridge,
    TheoryResolutionReport,
    InvariantSafetyGate
)

__all__ = [
    "StrainState",
    "StrainDomainModel",
    "compute_apex_strain",
    "CandidateStrainTransform",
    "MechanismRule",
    "ReachabilityStatus",
    "ReachabilityResult",
    "TriageOperator",
    "MechanismDescriptor",
    "SchedulableAction",
    "SmithRanker",
    "derive_w_comp_upper_bound",
    "StressWeightAmplifier",
    "FormalTheoryBridge",
    "TheoryResolutionReport",
    "InvariantSafetyGate",
]
