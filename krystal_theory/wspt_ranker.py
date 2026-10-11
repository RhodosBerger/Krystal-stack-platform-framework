"""
==============================================================================
KRYSTAL-STACK FORMAL THEORY: SMITH'S RULE / WSPT SCHEDULER (P2 RESOLUTION)
==============================================================================
Module: krystal_theory/wspt_ranker.py
Description: Optimal scheduling derivation based on Smith's Rule (Weighted Shortest
             Processing Time - WSPT). Eliminates static lookup tables, formalizes
             deterministic tie-breaking, proves the w_COMP <= 0.24 bound, and
             provides bounded stress-dependent weight amplification.

Epistemic Status: FORMALLY_DEFINED, PROPERTY_TESTED
Platform Invariant: VITAL_MAX_HP == 6
==============================================================================
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Dict, Any, List, Optional, Tuple

from krystal_kernel.psychological_coping_cortex_mcp_bridge import (
    PsychologicalCopingMechanism,
    VITAL_MAX_HP
)


@dataclass
class SchedulableAction:
    """An action or task candidate submitted to the WSPT ranker."""
    id: str
    mechanism: PsychologicalCopingMechanism
    weight: float                   # Urgency / consequence weight w_i > 0
    processing_time: float          # Computational cost / duration p_i > 0 (us)
    safety_criticality: float = 0.0 # [0.0, 1.0] (1.0 = vital emergency)
    uncertainty: float = 0.0        # [0.0, 1.0] (0.0 = completely certain)
    vital_max_hp: int = VITAL_MAX_HP

    def __post_init__(self):
        if self.weight <= 0.0 or not math.isfinite(self.weight):
            raise ValueError(f"Weight must be finite and positive: {self.weight}")
        if self.processing_time <= 0.0 or not math.isfinite(self.processing_time):
            raise ValueError(f"Processing time must be finite and positive: {self.processing_time}")
        if not (0.0 <= self.safety_criticality <= 1.0):
            raise ValueError("safety_criticality must be in [0.0, 1.0]")
        if not (0.0 <= self.uncertainty <= 1.0):
            raise ValueError("uncertainty must be in [0.0, 1.0]")
        assert self.vital_max_hp == 6, "Invariant VITAL_MAX_HP must remain 6"

    @property
    def smith_ratio(self) -> float:
        """Smith priority index rho_i = w_i / p_i."""
        return self.weight / self.processing_time

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["mechanism"] = self.mechanism.value
        d["smith_ratio"] = self.smith_ratio
        return d


class StressWeightAmplifier:
    """
    Stress-dependent weight amplification operator:
    w_i' = w_i * g(sigma_strain)

    Requires:
    1. g(0) == 1.0 (no stress -> nominal weight)
    2. g(sigma) >= 1.0 for sigma >= 0
    3. Strictly monotonic in sigma
    4. Bounded: g(sigma) <= 1.0 + alpha (avoids uncontrolled priority explosion).

    Candidate Comparison:
    - Linear: g_lin(sigma) = 1 + alpha * sigma -> UNBOUNDED if sigma overflows.
    - Bounded Tanh (Selected): g(sigma) = 1 + alpha * tanh(beta * sigma) -> Bounded [1, 1+alpha].
    """

    # Sensitivity coefficients alpha per coping mechanism
    MECHANISM_ALPHA: Dict[PsychologicalCopingMechanism, float] = {
        PsychologicalCopingMechanism.REGRESSION: 2.5,
        PsychologicalCopingMechanism.HYPERVIGILANCE: 3.0,
        PsychologicalCopingMechanism.COMPARTMENTALIZATION: 1.5,
        PsychologicalCopingMechanism.RATIONALIZATION: 1.2,
        PsychologicalCopingMechanism.SUBLIMATION: 0.5,
        PsychologicalCopingMechanism.REPRESSION: 2.0,
        PsychologicalCopingMechanism.PROJECTION: 0.0,  # Nominal work should NOT gain priority under strain
    }

    @classmethod
    def compute_multiplier(
        cls,
        mechanism: PsychologicalCopingMechanism,
        sigma_strain: float,
        beta: float = 2.0
    ) -> float:
        """Calculates bounded saturation multiplier g(sigma)."""
        if not math.isfinite(sigma_strain) or sigma_strain <= 0.0:
            return 1.0
        alpha = cls.MECHANISM_ALPHA.get(mechanism, 1.0)
        return 1.0 + alpha * math.tanh(beta * sigma_strain)

    @classmethod
    def amplify_action(cls, action: SchedulableAction, sigma_strain: float) -> SchedulableAction:
        """Returns a new SchedulableAction with stress-amplified weight."""
        g = cls.compute_multiplier(action.mechanism, sigma_strain)
        return SchedulableAction(
            id=action.id,
            mechanism=action.mechanism,
            weight=action.weight * g,
            processing_time=action.processing_time,
            safety_criticality=action.safety_criticality,
            uncertainty=action.uncertainty,
            vital_max_hp=action.vital_max_hp
        )


def derive_w_comp_upper_bound() -> Dict[str, Any]:
    """
    Formal mathematical derivation of the w_COMP <= 0.24 bound from Theorem 2.6.

    Derivation:
    - Model execution durations (from 01_triazny_operator.md / 02_planovanie_a_poradie.md):
        pi_REG = 50.0 us
        pi_PROJ = 500.0 us
        pi_COMP = 120.0 us
        pi_SUB = 350.0 us
    - Normalized baseline weight: w_PROJ = 1.0
    - Legacy priority order: REG (rank 1) < PROJ (rank 2) < COMP (rank 3) < SUB (rank 4).
    - According to Smith's WSPT Rule, jobs are ordered in non-decreasing order of p_i / w_i:
        pi_REG / w_REG <= pi_PROJ / w_PROJ <= pi_COMP / w_COMP <= pi_SUB / w_SUB
    - Substituting known values:
        50 / w_REG <= 500 / 1.0 <= 120 / w_COMP <= 350 / w_SUB
    - The middle inequality requires:
        500 <= 120 / w_COMP  <=>  w_COMP <= 120 / 500 = 0.24

    Consequence of Exceeding Bound:
    If w_COMP > 0.24 (e.g. w_COMP = 0.50):
        rho_COMP = 0.50 / 120 = 0.004167 > 0.0020 = 1.0 / 500 = rho_PROJ
        Smith's Rule prioritizes COMP before PROJ!
        The legacy static table (PROJ rank 2, COMP rank 3) is violated.
    """
    pi_comp = 120.0
    pi_proj = 500.0
    bound = pi_comp / pi_proj
    return {
        "pi_comp_us": pi_comp,
        "pi_proj_us": pi_proj,
        "w_proj_normalized": 1.0,
        "w_comp_upper_bound": bound,
        "admissible_range": f"[0, {bound}]",
        "epistemic_classification": "MATHEMATICALLY_DERIVED_FROM_EMPIRICAL_PARAMETERS",
        "assumptions": [
            "Single-machine non-preemptive scheduling (Smith 1956)",
            "Processing times fixed at model constants pi_m",
            "Weight normalization w_PROJ == 1.0",
            "WSPT optimality holds for full 4-class batches"
        ],
        "consequence_if_exceeded": (
            "When w_COMP > 0.24, Smith index rho_COMP exceeds rho_PROJ, "
            "causing WSPT to schedule COMP before PROJ."
        )
    }


class SmithRanker:
    """
    Deterministically ranks schedulable actions using Smith's Rule (WSPT).

    Sorting Key Tuple:
    1. Smith ratio rho = w / p (descending: higher ratio executes earlier)
    2. Safety criticality (descending: higher criticality executes earlier)
    3. Epistemic uncertainty (ascending: lower uncertainty executes earlier)
    4. Processing time p (ascending: shorter job executes earlier)
    5. Stable lexicographical ID (ascending: guaranteed deterministic total order).
    """

    @classmethod
    def rank_actions(cls, actions: List[SchedulableAction]) -> List[Tuple[int, SchedulableAction]]:
        """
        Ranks a batch of actions deterministically.
        Returns list of (rank_1_indexed, action).
        """
        # Primary sort key: (-rho, -safety, uncertainty, p, id)
        def sort_key(act: SchedulableAction):
            return (
                -act.smith_ratio,
                -act.safety_criticality,
                act.uncertainty,
                act.processing_time,
                act.id
            )

        sorted_actions = sorted(actions, key=sort_key)
        return [(idx + 1, act) for idx, act in enumerate(sorted_actions)]

    @classmethod
    def select_top_action(cls, actions: List[SchedulableAction]) -> Optional[SchedulableAction]:
        """Returns the highest priority action, or None if actions is empty."""
        ranked = cls.rank_actions(actions)
        return ranked[0][1] if ranked else None
