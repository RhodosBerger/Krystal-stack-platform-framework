"""Unified Transition Package for Krystal Stack Platform Framework (Milestone M2)."""

from .models import (
    DomainType,
    TransitionVerdict,
    DomainStateVector,
    ExecutionCandidate,
    ConstraintSet,
    KrystalTransitionEnvelope,
)
from .domain_adapter import DomainAdapter
from .runtime_core import KrystalRuntimeCore, GLOBAL_KRYSTAL_RUNTIME_CORE

__all__ = [
    "DomainType",
    "TransitionVerdict",
    "DomainStateVector",
    "ExecutionCandidate",
    "ConstraintSet",
    "KrystalTransitionEnvelope",
    "DomainAdapter",
    "KrystalRuntimeCore",
    "GLOBAL_KRYSTAL_RUNTIME_CORE",
]
