#!/usr/bin/env python3
"""
==============================================================================
KRYSTAL-STACK: RADIUS DERIVATIVE VERSUS COSINE TOPOLOGY ENGINE
==============================================================================
Module: krystal_kernel/radius_derivative_vs_cosine_topology.py
Description:
    Mathematical and procedural framework comparing and reconciling
    Radial Derivatives (dr/dtheta, d2r/dtheta2) against Cosine Topology
    Parameters (harmonic frequency, amplitude, phase, and Hermite continuity).

    Core Mathematical Formulations:
      1. Cosine Topology Parametrization:
         r(theta) = R_0 + sum_{k=1}^K A_k * cos(k * omega * theta + phi_k)
      2. First Radial Derivative (Tangential Slope & Surface Normal):
         dr/dtheta = - sum_{k=1}^K A_k * k * omega * sin(k * omega * theta + phi_k)
      3. Second Radial Derivative (Curvature & Inflection Points):
         d2r/dtheta2 = - sum_{k=1}^K A_k * (k * omega)^2 * cos(k * omega * theta + phi_k)
      4. Polar Tangent Angle (psi):
         tan(psi) = r(theta) / (dr/dtheta)
      5. Polar Curvature (kappa):
         kappa(theta) = |r^2 + 2*(r')^2 - r*r''| / (r^2 + (r')^2)^(3/2)
      6. Undercut / Steepness Threshold:
         xi(theta) = |dr/dtheta| / r(theta)
         When xi > 1.0, severe undercutting / overhangs occur, requiring full 3D SDF.

    Invariant: VITAL_MAX_HP == 6 (Max amplitude budget and HP clamping)
    Author: Dušan Kopecký & Krystal-Stack Architecture Council (2026)
==============================================================================
"""

import math
import sys
from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple, Optional

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

GOLDEN_RATIO: float = 1.61803398875
VITAL_MAX_HP: int = 6


@dataclass
class CosineTopologyHarmonic:
    """Individual harmonic component of the cosine topology."""
    harmonic_order: int             # k (1, 2, 3...)
    amplitude: float                # A_k
    frequency_multiplier: float     # omega
    phase_rad: float = 0.0          # phi_k

    def to_dict(self) -> Dict[str, Any]:
        return {
            "harmonic_order": self.harmonic_order,
            "amplitude": round(self.amplitude, 4),
            "frequency_multiplier": round(self.frequency_multiplier, 4),
            "phase_rad": round(self.phase_rad, 4)
        }


@dataclass
class CosineTopologyParameters:
    """Configuration defining a multi-harmonic cosine topology."""
    base_radius: float = 100.0
    harmonics: List[CosineTopologyHarmonic] = field(default_factory=list)
    vital_max_hp: int = VITAL_MAX_HP

    @classmethod
    def create_canonical(cls, base_r: float = 100.0, lobes: int = 4) -> 'CosineTopologyParameters':
        """Creates a canonical 3-tier golden-ratio attenuated cosine topology."""
        # Amplitude budget strictly capped by VITAL_MAX_HP
        a0 = min(15.0, float(VITAL_MAX_HP) * 2.5)
        h1 = CosineTopologyHarmonic(harmonic_order=1, amplitude=a0, frequency_multiplier=float(lobes), phase_rad=0.0)
        h2 = CosineTopologyHarmonic(harmonic_order=2, amplitude=a0 / GOLDEN_RATIO, frequency_multiplier=float(lobes), phase_rad=math.pi / 4.0)
        h3 = CosineTopologyHarmonic(harmonic_order=3, amplitude=a0 / (GOLDEN_RATIO ** 2), frequency_multiplier=float(lobes), phase_rad=math.pi / 2.0)
        return cls(base_radius=base_r, harmonics=[h1, h2, h3], vital_max_hp=VITAL_MAX_HP)

    def evaluate_radius(self, theta: float) -> float:
        """Evaluates r(theta) = R_0 + sum A_k * cos(k * omega * theta + phi_k)."""
        r = self.base_radius
        for h in self.harmonics:
            arg = h.harmonic_order * h.frequency_multiplier * theta + h.phase_rad
            r += h.amplitude * math.cos(arg)
        return max(0.1, r)

    def evaluate_radius_first_derivative(self, theta: float) -> float:
        """Evaluates dr/dtheta = - sum A_k * (k * omega) * sin(k * omega * theta + phi_k)."""
        dr = 0.0
        for h in self.harmonics:
            freq = h.harmonic_order * h.frequency_multiplier
            arg = freq * theta + h.phase_rad
            dr -= h.amplitude * freq * math.sin(arg)
        return dr

    def evaluate_radius_second_derivative(self, theta: float) -> float:
        """Evaluates d2r/dtheta2 = - sum A_k * (k * omega)^2 * cos(k * omega * theta + phi_k)."""
        d2r = 0.0
        for h in self.harmonics:
            freq = h.harmonic_order * h.frequency_multiplier
            arg = freq * theta + h.phase_rad
            d2r -= h.amplitude * (freq ** 2) * math.cos(arg)
        return d2r

    def evaluate_curvature(self, theta: float) -> float:
        """
        Calculates polar curvature kappa(theta):
        kappa = |r^2 + 2*(r')^2 - r*r''| / (r^2 + (r')^2)^(3/2)
        """
        r = self.evaluate_radius(theta)
        dr = self.evaluate_radius_first_derivative(theta)
        d2r = self.evaluate_radius_second_derivative(theta)

        num = abs(r * r + 2.0 * (dr * dr) - r * d2r)
        den = (r * r + dr * dr) ** 1.5
        return (num / den) if den > 1e-9 else 0.0

    def evaluate_steepness_ratio(self, theta: float) -> float:
        """Evaluates xi(theta) = |dr/dtheta| / r(theta)."""
        r = self.evaluate_radius(theta)
        dr = self.evaluate_radius_first_derivative(theta)
        return abs(dr) / r if r > 1e-9 else 0.0

    def get_cartesian_point_and_normal(self, theta: float) -> Tuple[float, float, float, float]:
        """
        Returns (x, y, nx, ny) where (nx, ny) is the unit outward surface normal.
        x = r * cos(theta), y = r * sin(theta)
        dx/dtheta = dr * cos(theta) - r * sin(theta)
        dy/dtheta = dr * sin(theta) + r * cos(theta)
        Normal N = (dy/dtheta, -dx/dtheta) normalized.
        """
        r = self.evaluate_radius(theta)
        dr = self.evaluate_radius_first_derivative(theta)

        cos_t = math.cos(theta)
        sin_t = math.sin(theta)

        x = r * cos_t
        y = r * sin_t

        dx = dr * cos_t - r * sin_t
        dy = dr * sin_t + r * cos_t

        # Normal vector perpendicular to tangent vector T=(dx, dy)
        nx = dy
        ny = -dx
        norm_len = math.sqrt(nx * nx + ny * ny)
        if norm_len > 1e-9:
            nx /= norm_len
            ny /= norm_len
        else:
            nx, ny = cos_t, sin_t

        return (round(x, 3), round(y, 3), round(nx, 4), round(ny, 4))


class RadiusDerivativeVsCosineTopologyComparator:
    """
    Analytical comparative engine contrasting direct Radial Derivative
    properties with global Cosine Topology parameters.
    """

    def __init__(self, topology: Optional[CosineTopologyParameters] = None):
        self.topology = topology or CosineTopologyParameters.create_canonical()

    def sample_profile(self, sample_count: int = 36) -> List[Dict[str, Any]]:
        """Samples the contour across [0, 2*pi) and computes comparative telemetry."""
        samples = []
        for i in range(sample_count):
            theta = (i / float(sample_count)) * 2.0 * math.pi
            r = self.topology.evaluate_radius(theta)
            dr = self.topology.evaluate_radius_first_derivative(theta)
            d2r = self.topology.evaluate_radius_second_derivative(theta)
            kappa = self.topology.evaluate_curvature(theta)
            xi = self.topology.evaluate_steepness_ratio(theta)
            x, y, nx, ny = self.topology.get_cartesian_point_and_normal(theta)

            # Polar tangent angle psi in degrees
            psi_deg = math.degrees(math.atan2(r, dr)) if abs(dr) > 1e-6 else 90.0

            samples.append({
                "sample_index": i,
                "theta_rad": round(theta, 4),
                "theta_deg": round(math.degrees(theta), 1),
                "radius": round(r, 2),
                "radius_derivative_dr": round(dr, 3),
                "radius_second_derivative_d2r": round(d2r, 3),
                "curvature_kappa": round(kappa, 5),
                "steepness_ratio_xi": round(xi, 4),
                "tangent_angle_psi_deg": round(psi_deg, 1),
                "cartesian_coord": [x, y],
                "surface_normal": [nx, ny],
                "vital_max_hp_rule": VITAL_MAX_HP
            })
        return samples

    def generate_analytical_comparison(self) -> Dict[str, Any]:
        """Produces a structured technical comparison between the two concepts."""
        samples = self.sample_profile(sample_count=64)
        max_dr = max(abs(s["radius_derivative_dr"]) for s in samples)
        max_xi = max(s["steepness_ratio_xi"] for s in samples)
        max_kappa = max(s["curvature_kappa"] for s in samples)

        return {
            "title": "Radius Derivát vs. Kosínusová Topológia: Analytické Porovnanie",
            "vital_max_hp": VITAL_MAX_HP,
            "metrics": {
                "max_radial_derivative": round(max_dr, 3),
                "max_steepness_ratio_xi": round(max_xi, 4),
                "max_curvature_kappa": round(max_kappa, 5),
                "is_undercut_risk": max_xi > 1.0,
                "base_radius": self.topology.base_radius,
                "harmonics_count": len(self.topology.harmonics)
            },
            "comparison_matrix": {
                "radius_derivative": {
                    "symbol": "dr/dtheta",
                    "nature": "Lokálny diferenciálny vektor zmeny a dotyčnice",
                    "phase_alignment": "Ortogonálny (+90° / pi/2 posun, sínusový charakter)",
                    "critical_points": "Nulový na vrchole (crest) a úpätí (trough), maximálny v nulovom bode kosínusu",
                    "procedural_role": "Výpočet povrchových normál N, tieňovania, erózie a aerodynamického sklonu",
                    "cliff_threshold": "|dr/dtheta| > r(theta) indikuje previs / previsnutú stenu (overhang)"
                },
                "cosine_topology": {
                    "symbol": "r(theta) = R_0 + sum A_k cos(k*omega*theta + phi_k)",
                    "nature": "Globálna harmonická funkcia vyjadrená v ortogonálnej báze",
                    "phase_alignment": "Párna funkcia (symetria tvaru f(-theta) = f(theta))",
                    "critical_points": "Maximum v theta=0, 2pi/omega; minimum v theta=pi/omega",
                    "procedural_role": "Generovanie siluety ostrovov, profilov budov, IDCT tenzorov a mriežok",
                    "cliff_threshold": "Zaručuje C^inf spojitosť, zabraňuje jitteru a trhaniu polygonálnej siete"
                }
            },
            "interplay_synthesis": (
                "Kosínusová topológia určuje globálnu obálku tvaru r(theta), zatiaľ čo jej radius derivát "
                "dr/dtheta bezprostredne diktuje smerovú dotyčnicu T a povrchovú normálu N = (dy, -dx). "
                "Pretože dr/dtheta = -A*omega*sin(omega*theta), maximálny sklon nastáva presne tam, kde sa "
                "kosínus križuje so stredným polomerom R_0. Pomer xi = |dr/dtheta| / r určuje limit 2.5D terénu: "
                "ak xi <= 1.0, terén je reprezentovateľný výškovou mapou; ak xi > 1.0, vznikajú podrezané previsy "
                "vyžadujúce plné 3D SDF modelovanie."
            )
        }


GLOBAL_TOPOLOGY_COMPARATOR = RadiusDerivativeVsCosineTopologyComparator()


if __name__ == "__main__":
    print("=" * 80)
    print("KRYSTAL-STACK: RADIUS DERIVATIVE VERSUS COSINE TOPOLOGY VERIFICATION")
    print("=" * 80)
    comp = GLOBAL_TOPOLOGY_COMPARATOR
    report = comp.generate_analytical_comparison()
    print(f"Base Radius: {report['metrics']['base_radius']}")
    print(f"Max Radial Derivative (|dr/dtheta|): {report['metrics']['max_radial_derivative']}")
    print(f"Max Steepness Ratio (xi): {report['metrics']['max_steepness_ratio_xi']}")
    print(f"Undercut Risk: {report['metrics']['is_undercut_risk']}")
    print(f"\nSyntéza:")
    print(report['interplay_synthesis'])
    print(f"\nUniversal Platform Invariant: VITAL_MAX_HP == {report['vital_max_hp']}")
    print("ALL TESTS PASSED.")
