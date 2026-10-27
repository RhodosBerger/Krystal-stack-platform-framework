#!/usr/bin/env python3
"""
==============================================================================
KRYSTAL-STACK: ABSTRACT DOMAIN ADAPTER INTERFACE
==============================================================================
Module: krystal_kernel/transition/domain_adapter.py
Normative Reference:
    - docs/architecture/MASTER_BLUEPRINT_KRYSTAL_STACK_SOVEREIGN_OMNI_SYSTEM.md
==============================================================================
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Any

from .models import (
    DomainStateVector,
    ExecutionCandidate,
    ConstraintSet,
    KrystalTransitionEnvelope,
)


class DomainAdapter(ABC):
    """
    Abstract adapter interface required for any domain to plug into
    the unified Krystal Transition Pipeline without mutating the core.
    """

    @property
    @abstractmethod
    def domain_name(self) -> str:
        """Identifier of the domain (e.g. CNC, SILICON, SOVEREIGN, FARM)."""
        pass

    @abstractmethod
    def observe(self) -> Dict[str, Any]:
        """Samples raw physical / operational telemetry from the domain."""
        pass

    @abstractmethod
    def normalize(self, raw_data: Dict[str, Any]) -> DomainStateVector:
        """Converts raw domain metrics into a standardized DomainStateVector."""
        pass

    @abstractmethod
    def generate_candidates(self, current_state: DomainStateVector) -> List[ExecutionCandidate]:
        """Generates >= 3 domain-specific execution candidates."""
        pass

    @abstractmethod
    def predict(
        self,
        current_state: DomainStateVector,
        candidate: ExecutionCandidate,
    ) -> DomainStateVector:
        """Predicts expected next state given the candidate action."""
        pass

    @abstractmethod
    def execute(
        self,
        candidate: ExecutionCandidate,
    ) -> Dict[str, Any]:
        """Executes candidate action against the physical/simulated domain target."""
        pass

    @abstractmethod
    def validate(
        self,
        envelope: KrystalTransitionEnvelope,
    ) -> KrystalTransitionEnvelope:
        """Validates prediction vs observed after-state and populates residual and utility."""
        pass
