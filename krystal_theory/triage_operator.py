"""
==============================================================================
KRYSTAL-STACK FORMAL THEORY: BOUNDED TRIAGE OPERATOR & REACHABILITY (P1 RESOLUTION)
==============================================================================
Module: krystal_theory/triage_operator.py
Description: Deterministic bounded triage operator T(X) with hysteresis,
             explicit reachability graph G = (States, Mechanisms, Transitions),
             and state classification (THEORETICALLY_REACHABLE, RUNTIME_REACHABLE,
             CURRENTLY_ACTIVE, blocked_reason).

Epistemic Status: FORMALLY_DEFINED, PROPERTY_TESTED
Platform Invariant: VITAL_MAX_HP == 6
==============================================================================
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Any, List, Optional, Set, Tuple

from krystal_kernel.psychological_coping_cortex_mcp_bridge import (
    PsychologicalCopingMechanism,
    VITAL_MAX_HP
)
from .strain_operator import StrainState


class ReachabilityStatus(str, Enum):
    """Reachability status of a coping mechanism relative to the current state."""
    THEORETICALLY_REACHABLE = "THEORETICALLY_REACHABLE"
    RUNTIME_REACHABLE = "RUNTIME_REACHABLE"
    CURRENTLY_ACTIVE = "CURRENTLY_ACTIVE"
    BLOCKED = "BLOCKED"


@dataclass(frozen=True)
class MechanismDescriptor:
    """Formal declarative properties of a coping mechanism in the triage taxonomy."""
    mechanism: PsychologicalCopingMechanism
    priority_tier: int            # 1 (Critical Emergency) to 7 (Background Suppression)
    tau_on: float                 # Strain threshold to activate
    tau_off: float                # Strain threshold to deactivate (tau_off < tau_on)
    dependencies: Tuple[PsychologicalCopingMechanism, ...] = ()
    mutual_exclusions: Tuple[PsychologicalCopingMechanism, ...] = ()
    cooldown_ticks: int = 0
    evidence_requirement: str = "STATE_OBSERVABLE"


@dataclass
class MechanismRule:
    """Executable evaluation rule for mechanism reachability and activation."""
    descriptor: MechanismDescriptor
    active: bool = False
    cooldown_remaining: int = 0
    blocked_reason: Optional[str] = None


@dataclass
class ReachabilityResult:
    """Structured reachability analysis for a given evaluation step."""
    theoretically_reachable: List[str]
    runtime_reachable: List[str]
    currently_active: List[str]
    blocked_reasons: Dict[str, str]
    strain_normalized: float
    vital_max_hp: int = VITAL_MAX_HP

    def to_dict(self) -> Dict[str, Any]:
        return {
            "theoretically_reachable": self.theoretically_reachable,
            "runtime_reachable": self.runtime_reachable,
            "currently_active": self.currently_active,
            "blocked_reasons": self.blocked_reasons,
            "strain_normalized": self.strain_normalized,
            "vital_max_hp": self.vital_max_hp,
        }


class TriageOperator:
    """
    Deterministic Triage Operator T(X) -> Set[Mechanisms].

    Resolves P1 by enabling strain-activated reachability for previously unreachable
    mechanisms (HYPERVIGILANCE, RATIONALIZATION, REPRESSION) without naive `if strain > 0: activate_all()`.

    Enforces:
    1. Bounded activation functions A_m(X).
    2. Dual-threshold hysteresis (tau_on > tau_off) preventing chatter.
    3. Explicit reachability state classification.
    4. Deterministic tie-breaking and mutual exclusion resolution.
    """

    # Declarative taxonomy with explicit dual thresholds and constraints
    TAXONOMY_RULES: Dict[PsychologicalCopingMechanism, MechanismDescriptor] = {
        PsychologicalCopingMechanism.REGRESSION: MechanismDescriptor(
            mechanism=PsychologicalCopingMechanism.REGRESSION,
            priority_tier=1,
            tau_on=0.0,
            tau_off=0.0,
            dependencies=(),
            mutual_exclusions=(PsychologicalCopingMechanism.PROJECTION,),
            evidence_requirement="THERMAL_EMERGENCY_OR_NONFINITE"
        ),
        PsychologicalCopingMechanism.HYPERVIGILANCE: MechanismDescriptor(
            mechanism=PsychologicalCopingMechanism.HYPERVIGILANCE,
            priority_tier=2,
            tau_on=0.35,
            tau_off=0.20,
            dependencies=(),
            mutual_exclusions=(),
            cooldown_ticks=2,
            evidence_requirement="STRAIN_AND_HIGH_UNCERTAINTY"
        ),
        PsychologicalCopingMechanism.COMPARTMENTALIZATION: MechanismDescriptor(
            mechanism=PsychologicalCopingMechanism.COMPARTMENTALIZATION,
            priority_tier=3,
            tau_on=0.0,
            tau_off=0.0,
            dependencies=(),
            mutual_exclusions=(),
            evidence_requirement="MEMORY_CONTENTION_EXCEEDED"
        ),
        PsychologicalCopingMechanism.RATIONALIZATION: MechanismDescriptor(
            mechanism=PsychologicalCopingMechanism.RATIONALIZATION,
            priority_tier=4,
            tau_on=0.50,
            tau_off=0.30,
            dependencies=(),
            mutual_exclusions=(),
            cooldown_ticks=1,
            evidence_requirement="STRAIN_AND_LOSS_TOLERANCE"
        ),
        PsychologicalCopingMechanism.SUBLIMATION: MechanismDescriptor(
            mechanism=PsychologicalCopingMechanism.SUBLIMATION,
            priority_tier=5,
            tau_on=0.0,
            tau_off=0.0,
            dependencies=(),
            mutual_exclusions=(),
            evidence_requirement="CONTEXT_ENTROPY_EXCEEDED"
        ),
        PsychologicalCopingMechanism.REPRESSION: MechanismDescriptor(
            mechanism=PsychologicalCopingMechanism.REPRESSION,
            priority_tier=6,
            tau_on=0.65,
            tau_off=0.45,
            dependencies=(),
            mutual_exclusions=(PsychologicalCopingMechanism.PROJECTION,),
            cooldown_ticks=3,
            evidence_requirement="ACUTE_STRAIN_BACKGROUND_SUPPRESSION"
        ),
        PsychologicalCopingMechanism.PROJECTION: MechanismDescriptor(
            mechanism=PsychologicalCopingMechanism.PROJECTION,
            priority_tier=7,
            tau_on=0.0,
            tau_off=0.0,
            dependencies=(),
            mutual_exclusions=(PsychologicalCopingMechanism.REGRESSION, PsychologicalCopingMechanism.REPRESSION),
            evidence_requirement="NOMINAL_QUIESCENT_REGIME"
        ),
    }

    def __init__(self):
        self._state_rules: Dict[PsychologicalCopingMechanism, MechanismRule] = {
            m: MechanismRule(descriptor=desc) for m, desc in self.TAXONOMY_RULES.items()
        }
        assert VITAL_MAX_HP == 6, "Invariant VITAL_MAX_HP must remain 6"

    def reset_state(self) -> None:
        """Resets all dynamic hysteresis latches and cooldown counters."""
        for rule in self._state_rules.values():
            rule.active = False
            rule.cooldown_remaining = 0
            rule.blocked_reason = None

    def tick_cooldowns(self) -> None:
        """Decrements active cooldown timers deterministically."""
        for rule in self._state_rules.values():
            if rule.cooldown_remaining > 0:
                rule.cooldown_remaining -= 1

    def evaluate_reachability(
        self,
        context_entropy: float,
        memory_contention: float,
        thermal_pressure_pct: float,
        strain: StrainState,
        epistemic_uncertainty: float = 0.20,
        loss_tolerance: float = 0.50,
        safety_margin: float = 0.50,
        background_task_load: float = 0.20
    ) -> ReachabilityResult:
        """
        Evaluates reachability and active mechanisms for the given state vector X.

        State Vector X:
        - context_entropy (e): [0.0, 1.0]
        - memory_contention (c): [0.0, inf)
        - thermal_pressure_pct (theta): [0.0, 120.0]
        - strain (sigma): StrainState
        - epistemic_uncertainty (u): [0.0, 1.0]
        - loss_tolerance (l): [0.0, 1.0]
        - safety_margin (m_h): [0.0, 1.0]
        - background_task_load (b): [0.0, 1.0]
        """
        assert VITAL_MAX_HP == 6, "Invariant VITAL_MAX_HP must remain 6"

        sigma = strain.normalized_strain
        non_finite = not (
            math.isfinite(context_entropy)
            and math.isfinite(memory_contention)
            and math.isfinite(thermal_pressure_pct)
            and math.isfinite(sigma)
        )

        theoretically_reachable: List[str] = [m.value for m in PsychologicalCopingMechanism]
        runtime_reachable: List[str] = []
        blocked_reasons: Dict[str, str] = {}

        # 1. Evaluate prerequisites and candidate reachability per mechanism
        candidates: Dict[PsychologicalCopingMechanism, bool] = {}

        for mech, rule in self._state_rules.items():
            desc = rule.descriptor
            candidate = False
            reason: Optional[str] = None

            if non_finite:
                if mech == PsychologicalCopingMechanism.REGRESSION:
                    candidate = True
                else:
                    reason = "FAIL_SAFE_NON_FINITE_INPUT"
            elif mech == PsychologicalCopingMechanism.REGRESSION:
                if thermal_pressure_pct >= 85.0 or sigma >= 0.85:
                    candidate = True
                else:
                    reason = "THERMAL_BELOW_EMERGENCY"

            elif mech == PsychologicalCopingMechanism.HYPERVIGILANCE:
                # Dual threshold hysteresis
                was_active = rule.active
                threshold = desc.tau_off if was_active else desc.tau_on
                if sigma < threshold:
                    reason = f"STRAIN_BELOW_{'OFF' if was_active else 'ON'}_THRESHOLD"
                elif epistemic_uncertainty < 0.30:
                    reason = "UNCERTAINTY_TOO_LOW"
                elif safety_margin > 0.40:
                    reason = "SAFETY_MARGIN_ADEQUATE"
                elif rule.cooldown_remaining > 0:
                    reason = f"COOLDOWN_ACTIVE_{rule.cooldown_remaining}_TICKS"
                else:
                    candidate = True

            elif mech == PsychologicalCopingMechanism.COMPARTMENTALIZATION:
                if memory_contention >= 2.0:
                    candidate = True
                else:
                    reason = "MEMORY_CONTENTION_BELOW_LIMIT"

            elif mech == PsychologicalCopingMechanism.RATIONALIZATION:
                was_active = rule.active
                threshold = desc.tau_off if was_active else desc.tau_on
                if sigma < threshold:
                    reason = f"STRAIN_BELOW_{'OFF' if was_active else 'ON'}_THRESHOLD"
                elif loss_tolerance < 0.20:
                    reason = "LOSS_TOLERANCE_INSUFFICIENT"
                elif rule.cooldown_remaining > 0:
                    reason = f"COOLDOWN_ACTIVE_{rule.cooldown_remaining}_TICKS"
                else:
                    candidate = True

            elif mech == PsychologicalCopingMechanism.SUBLIMATION:
                if context_entropy >= 0.70 and memory_contention < 2.0:
                    candidate = True
                else:
                    reason = "CONTEXT_ENTROPY_BELOW_LIMIT_OR_CONTENTION_HIGH"

            elif mech == PsychologicalCopingMechanism.REPRESSION:
                was_active = rule.active
                threshold = desc.tau_off if was_active else desc.tau_on
                if sigma < threshold:
                    reason = f"STRAIN_BELOW_{'OFF' if was_active else 'ON'}_THRESHOLD"
                elif background_task_load < 0.40:
                    reason = "BACKGROUND_LOAD_TOO_LOW"
                elif rule.cooldown_remaining > 0:
                    reason = f"COOLDOWN_ACTIVE_{rule.cooldown_remaining}_TICKS"
                else:
                    candidate = True

            elif mech == PsychologicalCopingMechanism.PROJECTION:
                # Quiescent nominal lookahead
                if thermal_pressure_pct < 85.0 and memory_contention < 2.0 and context_entropy < 0.70 and sigma < 0.35:
                    candidate = True
                else:
                    reason = "SYSTEM_UNDER_STRESS_OR_NON_QUIESCENT"

            candidates[mech] = candidate
            if candidate:
                runtime_reachable.append(mech.value)
            elif reason:
                blocked_reasons[mech.value] = reason

        # 2. Filter runtime candidates through mutual exclusions and priority tiers
        currently_active: List[str] = []
        active_set: Set[PsychologicalCopingMechanism] = set()

        # Sort candidate mechanisms by priority tier
        sorted_candidates = sorted(
            [m for m, cand in candidates.items() if cand],
            key=lambda m: self.TAXONOMY_RULES[m].priority_tier
        )

        for mech in sorted_candidates:
            desc = self.TAXONOMY_RULES[mech]
            # Check mutual exclusions against already selected higher-priority mechanisms
            excluded = False
            for active_m in active_set:
                if active_m in desc.mutual_exclusions or mech in self.TAXONOMY_RULES[active_m].mutual_exclusions:
                    excluded = True
                    blocked_reasons[mech.value] = f"MUTUALLY_EXCLUSIVE_WITH_{active_m.value}"
                    break

            if not excluded:
                active_set.add(mech)
                currently_active.append(mech.value)

        # 3. Update internal hysteresis state & cooldowns
        for mech, rule in self._state_rules.items():
            if mech in active_set:
                rule.active = True
                rule.blocked_reason = None
            else:
                if rule.active:
                    # Deactivated this tick: initiate cooldown
                    rule.cooldown_remaining = rule.descriptor.cooldown_ticks
                rule.active = False
                rule.blocked_reason = blocked_reasons.get(mech.value)

        return ReachabilityResult(
            theoretically_reachable=theoretically_reachable,
            runtime_reachable=runtime_reachable,
            currently_active=currently_active,
            blocked_reasons=blocked_reasons,
            strain_normalized=sigma,
            vital_max_hp=VITAL_MAX_HP
        )
