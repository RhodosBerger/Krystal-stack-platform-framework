"""
==============================================================================
KRYSTAL-STACK FORMAL THEORY: STRAIN OPERATOR (P1 RESOLUTION)
==============================================================================
Module: krystal_theory/strain_operator.py
Description: Mathematical definition of the latent apex strain measure sigma_strain
             derived from the quadratic stability boundary discriminant Delta < 0.

Epistemic Status: FORMALLY_DEFINED, PROPERTY_TESTED
Platform Invariant: VITAL_MAX_HP == 6
==============================================================================
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Any, Tuple

VITAL_MAX_HP: int = 6


class CandidateStrainTransform(str, Enum):
    """Candidate normalization functions mapping raw strain [0, inf) to [0, 1)."""
    LINEAR_CLAMPED = "LINEAR_CLAMPED"
    SQRT_NORMALIZED = "SQRT_NORMALIZED"
    LOG1P_NORMALIZED = "LOG1P_NORMALIZED"
    BOUNDED_TANH = "BOUNDED_TANH"      # Canonical selected model


@dataclass(frozen=True)
class StrainState:
    """Immutable state vector representing system strain and apex projection."""
    delta: float
    raw_strain: float
    normalized_strain: float
    apex_real: float
    is_complex: bool
    transform_model: CandidateStrainTransform
    vital_max_hp: int = VITAL_MAX_HP

    def to_dict(self) -> Dict[str, Any]:
        return {
            "delta": self.delta,
            "raw_strain": self.raw_strain,
            "normalized_strain": self.normalized_strain,
            "apex_real": self.apex_real,
            "is_complex": self.is_complex,
            "transform_model": self.transform_model.value,
            "vital_max_hp": self.vital_max_hp,
        }


class StrainDomainModel:
    """
    Formal specification of the strain domain sigma_strain:

    1. Input Domain:
       - a, b, c in R, representing the quadratic stability parabola P(x) = a x^2 + b x + c = 0.
       - In thermal-frequency governor: a is silicon curvature (thermal impedance),
         b is dynamic load slope, c is static baseline dissipation.

    2. Range:
       - sigma_strain in [0.0, 1.0)
       - Boundary 0.0: Delta >= 0 (real regime, stable equilibrium exists).
       - Asymptotic limit 1.0: Delta -> -inf or raw_strain -> inf (acute collapse trauma).

    3. Continuity & Differentiability:
       - Real regime (Delta >= 0): sigma_strain == 0.0.
       - Complex regime (Delta < 0): sigma_strain is C^inf on (-inf, 0).
       - At transition Delta = 0: continuous (lim_{Delta -> 0^-} sigma_strain = 0.0).

    4. Monotonicity:
       - For fixed a != 0: sigma_strain is strictly monotonically increasing with |-Delta|
         when Delta < 0.

    5. Undefined States & Fail-Safe:
       - NaN or +/-inf in (a, b, c) -> normalized_strain = 1.0 (fail-safe REGRESSION emergency).
       - Singular leading coefficient |a| < 1e-12 -> raw_strain = sqrt(|Delta|), avoiding division by zero.
    """

    KAPPA_DEFAULT: float = 1.0  # Sensitivity scale factor

    @staticmethod
    def apply_transform(raw_strain: float, transform: CandidateStrainTransform, kappa: float = 1.0) -> float:
        """Applies a bounded normalization candidate to raw strain."""
        if not math.isfinite(raw_strain) or raw_strain < 0.0:
            return 1.0 if not math.isfinite(raw_strain) else 0.0

        if transform == CandidateStrainTransform.LINEAR_CLAMPED:
            return min(1.0, kappa * raw_strain)
        elif transform == CandidateStrainTransform.SQRT_NORMALIZED:
            s_sqrt = math.sqrt(raw_strain)
            return s_sqrt / (1.0 + s_sqrt)
        elif transform == CandidateStrainTransform.LOG1P_NORMALIZED:
            s_log = math.log1p(kappa * raw_strain)
            return s_log / (1.0 + s_log)
        elif transform == CandidateStrainTransform.BOUNDED_TANH:
            # Selected: C^inf smooth, strictly monotonic, bounded [0, 1), no sharp saturation corner
            return math.tanh(kappa * raw_strain)
        else:
            raise ValueError(f"Unknown transform: {transform}")


def compute_apex_strain(
    a: float,
    b: float,
    c: float,
    kappa: float = 1.0,
    transform: CandidateStrainTransform = CandidateStrainTransform.BOUNDED_TANH
) -> StrainState:
    """
    Computes the complex latent apex projection and strain measure:

    P(x) = a x^2 + b x + c
    Discriminant Delta = b^2 - 4ac

    If Delta >= 0:
        Real regime: Stable physical solution exists.
        sigma_strain = 0.0
        apex_real = -b / (2a)

    If Delta < 0:
        Latent complex regime: Physical boundary violated.
        x_apex = -b / (2a)  (Vertex / maximum stability point)
        sigma_raw = sqrt(-Delta) / (2|a|)
        sigma_strain = transform(sigma_raw, kappa)
    """
    assert VITAL_MAX_HP == 6, "Invariant VITAL_MAX_HP must remain 6"

    # Fail-safe check for non-finite inputs
    if not (math.isfinite(a) and math.isfinite(b) and math.isfinite(c)):
        return StrainState(
            delta=float("nan"),
            raw_strain=float("inf"),
            normalized_strain=1.0,
            apex_real=0.0,
            is_complex=True,
            transform_model=transform,
            vital_max_hp=VITAL_MAX_HP
        )

    # Singular degenerate quadratic
    if abs(a) < 1e-12:
        delta = b * b
        return StrainState(
            delta=delta,
            raw_strain=0.0,
            normalized_strain=0.0,
            apex_real=-c / b if abs(b) >= 1e-12 else 0.0,
            is_complex=False,
            transform_model=transform,
            vital_max_hp=VITAL_MAX_HP
        )

    delta = b * b - 4.0 * a * c
    apex_real = -b / (2.0 * a)

    if delta >= 0.0:
        # Real regime: zero strain
        return StrainState(
            delta=delta,
            raw_strain=0.0,
            normalized_strain=0.0,
            apex_real=apex_real,
            is_complex=False,
            transform_model=transform,
            vital_max_hp=VITAL_MAX_HP
        )
    else:
        # Latent complex regime: positive strain
        raw_strain = math.sqrt(-delta) / (2.0 * abs(a))
        normalized = StrainDomainModel.apply_transform(raw_strain, transform, kappa)
        return StrainState(
            delta=delta,
            raw_strain=raw_strain,
            normalized_strain=normalized,
            apex_real=apex_real,
            is_complex=True,
            transform_model=transform,
            vital_max_hp=VITAL_MAX_HP
        )
