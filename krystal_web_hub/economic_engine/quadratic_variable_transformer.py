"""
KRYSTAL-STACK // QUADRATIC VARIABLE TRANSFORMER & CONVERSION BRIDGE
===================================================================
A unified mathematical manifold converting disparate platform variables
(Memory latency, VRAM, GPU frequency, Hamiltonian H, Market price, Game HP, Terrain)
through a canonical latent variable x ("predmet skúmania").

Mathematical Formulation:
  Forward:  V_i(x) = a_i * x^2 + b_i * x + c_i
  Inverse:  a_i * x^2 + b_i * x + (c_i - V_i) = 0
            Delta_i = b_i^2 - 4 * a_i * (c_i - V_i)
            x = (-b_i +- sqrt(Delta_i)) / (2 * a_i)

Invariant:
  VITAL_MAX_HP = 6 (Max HP rule strictly preserved)
"""

import math
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple

VITAL_MAX_HP: int = 6
GOLDEN_RATIO: float = 1.61803398875
INV_GOLDEN_RATIO: float = 0.61803398875  # phi^-1 (Aristotelian golden mean)


@dataclass
class QuadraticDomain:
    """Represents a physical or virtual domain governed by a quadratic curve."""
    domain_id: str
    name: str
    unit: str
    category: str  # 'hardware', 'thermodynamics', 'economy', 'gameplay', 'spatial'
    a: float  # Quadratic coefficient (curvature, second-order sensitivity)
    b: float  # Linear coefficient (rate of change / slope)
    c: float  # Base offset (idle / zero-state value)
    v_min: float
    v_max: float
    x_range: Tuple[float, float] = (0.0, 1.0)
    prefer_positive_root: bool = True
    description: str = ""

    def evaluate(self, x: float) -> float:
        """Calculates V(x) = a * x^2 + b * x + c, bounded by [v_min, v_max]."""
        val = self.a * (x ** 2) + self.b * x + self.c
        # Clamp to domain physical boundaries
        clamped = max(self.v_min, min(self.v_max, val))
        # Enforce platform vital invariant for gameplay HP
        if self.domain_id == "vital_hp":
            clamped = min(float(VITAL_MAX_HP), max(0.0, clamped))
        return round(clamped, 4)

    def solve_x(self, v: float) -> Tuple[Optional[float], float, bool]:
        """
        Solves a * x^2 + b * x + (c - v) = 0 for latent variable x.
        Returns: (selected_x, discriminant, is_real)
        """
        target_v = max(self.v_min, min(self.v_max, v))
        c_eff = self.c - target_v

        # Degenerate linear case (a == 0)
        if abs(self.a) < 1e-12:
            if abs(self.b) < 1e-12:
                return (0.0, 0.0, True)
            x_lin = -c_eff / self.b
            return (round(max(self.x_range[0], min(self.x_range[1], x_lin)), 5), 0.0, True)

        delta = (self.b ** 2) - 4.0 * self.a * c_eff

        if delta < 0:
            # Complex roots: project onto apex of parabola x = -b / (2a)
            apex_x = -self.b / (2.0 * self.a)
            clamped_apex = max(self.x_range[0], min(self.x_range[1], apex_x))
            return (round(clamped_apex, 5), round(delta, 5), False)

        sqrt_delta = math.sqrt(delta)
        x1 = (-self.b + sqrt_delta) / (2.0 * self.a)
        x2 = (-self.b - sqrt_delta) / (2.0 * self.a)

        # Filter roots within valid domain range
        r1_valid = self.x_range[0] <= x1 <= self.x_range[1]
        r2_valid = self.x_range[0] <= x2 <= self.x_range[1]

        if r1_valid and r2_valid:
            chosen = x1 if self.prefer_positive_root else x2
        elif r1_valid:
            chosen = x1
        elif r2_valid:
            chosen = x2
        else:
            # Pick closest root to valid range
            dist1 = min(abs(x1 - self.x_range[0]), abs(x1 - self.x_range[1]))
            dist2 = min(abs(x2 - self.x_range[0]), abs(x2 - self.x_range[1]))
            chosen = x1 if dist1 <= dist2 else x2
            chosen = max(self.x_range[0], min(self.x_range[1], chosen))

        return (round(chosen, 5), round(delta, 5), True)


class QuadraticVariableTransformer:
    """
    Central engine orchestrating quadratic cross-domain variable conversions
    through the normalized latent variable x.
    """

    def __init__(self):
        self._domains: Dict[str, QuadraticDomain] = self._init_canonical_domains()

    def _init_canonical_domains(self) -> Dict[str, QuadraticDomain]:
        return {
            "memory_latency_ns": QuadraticDomain(
                domain_id="memory_latency_ns",
                name="Pamäťová Latencia L1/L2 a DDR",
                unit="ns",
                category="hardware",
                a=85.0,     # Non-linear quadratic contention when bus saturates
                b=15.0,     # Linear transmission delay
                c=5.2,      # Baseline SRAM L1 access latency
                v_min=5.0,
                v_max=120.0,
                x_range=(0.0, 1.0),
                prefer_positive_root=True,
                description="Latencia pamäťového subsystému; rastie kvadraticky pri zapĺňaní vyrovnávacích pamätí."
            ),
            "vram_allocation_mb": QuadraticDomain(
                domain_id="vram_allocation_mb",
                name="Alokácia Zdieľanej VRAM (Host DDR)",
                unit="MB",
                category="hardware",
                a=3200.0,   # Parabolic scaling of textures and geometry buffers
                b=4800.0,
                c=256.0,    # Base framebuffer allocation
                v_min=256.0,
                v_max=8256.0,
                x_range=(0.0, 1.0),
                prefer_positive_root=True,
                description="Zdieľaná pamäť pre Vulkan Iris Xe a shadery; x=0 predstavuje minimum, x=1 plné zaťaženie."
            ),
            "gpu_clock_mhz": QuadraticDomain(
                domain_id="gpu_clock_mhz",
                name="Taktovacia Frekvencia Intel Iris Xe EUs",
                unit="MHz",
                category="hardware",
                a=-400.0,   # Downward curvature due to thermal throttling at high loads
                b=1050.0,
                c=800.0,    # Base clock
                v_min=800.0,
                v_max=1450.0,
                x_range=(0.0, 1.0),
                prefer_positive_root=False,
                description="Frekvencia 96 výpočtových jednotiek (EUs); dosahuje vrchol pred tepelným limitom."
            ),
            "hamiltonian_h": QuadraticDomain(
                domain_id="hamiltonian_h",
                name="Hamiltonián H (Fyzikálna Energia Systému)",
                unit="energy",
                category="thermodynamics",
                a=0.85,     # Harmonic potential energy (1/2 * k * x^2)
                b=0.20,     # Kinetic dissipation
                c=0.05,     # Zero-point vacuum fluctuation
                v_min=0.05,
                v_max=1.20,
                x_range=(0.0, 1.0),
                prefer_positive_root=True,
                description="Celková energia kyvadla a fázového priestoru v Krystal-Stack engine."
            ),
            "market_price_credits": QuadraticDomain(
                domain_id="market_price_credits",
                name="Ekonomická Cena Komodity (Bonding Curve)",
                unit="credits",
                category="economy",
                a=120.0,    # Quadratic bonding curve price acceleration
                b=30.0,     # Linear base cost
                c=10.0,     # Initial mint price
                v_min=10.0,
                v_max=160.0,
                x_range=(0.0, 1.0),
                prefer_positive_root=True,
                description="Kvadratická cenová krivka AMM zabezpečujúca likviditu bez hyperinflácie."
            ),
            "vital_hp": QuadraticDomain(
                domain_id="vital_hp",
                name="Životná Energia Kmeňa (Vital Max HP = 6)",
                unit="HP",
                category="gameplay",
                a=-3.5,     # Damage accelerates non-linearly near critical threshold
                b=-1.5,
                c=6.0,      # Max HP invariant
                v_min=0.0,
                v_max=6.0,
                x_range=(0.0, 1.0),
                prefer_positive_root=False,
                description="Platformové pravidlo 6 Max HP; x=0 znamená plné zdravie (6 HP), x=1 kritické vyčerpanie (1 HP)."
            ),
            "terrain_altitude_m": QuadraticDomain(
                domain_id="terrain_altitude_m",
                name="Procedurálna Nadmorská Výška (SDF Paraboloid)",
                unit="m",
                category="spatial",
                a=1200.0,   # Parabolic mountain slope
                b=600.0,
                c=50.0,     # Sea level base
                v_min=50.0,
                v_max=1850.0,
                x_range=(0.0, 1.0),
                prefer_positive_root=True,
                description="Geometrická výška terénu generovaná cez kvadratické signálne pole."
            ),
            "system_entropy_s": QuadraticDomain(
                domain_id="system_entropy_s",
                name="Systémová Entropia & Šum",
                unit="bits",
                category="thermodynamics",
                a=0.45,
                b=0.55,
                c=0.02,
                v_min=0.02,
                v_max=1.02,
                x_range=(0.0, 1.0),
                prefer_positive_root=True,
                description="Informačná entropia stavu vlákien a pamäťových blokov."
            )
        }

    def get_domains(self) -> Dict[str, Any]:
        """Returns catalog of all registered quadratic domains and formulas."""
        return {
            "vital_max_hp_rule": VITAL_MAX_HP,
            "golden_ratio_phi": GOLDEN_RATIO,
            "inv_golden_ratio": INV_GOLDEN_RATIO,
            "domains_count": len(self._domains),
            "domains": [
                {
                    "domain_id": d.domain_id,
                    "name": d.name,
                    "unit": d.unit,
                    "category": d.category,
                    "formula": f"V(x) = {d.a}*x^2 + {d.b}*x + {d.c}",
                    "coefficients": {"a": d.a, "b": d.b, "c": d.c},
                    "limits": {"min": d.v_min, "max": d.v_max},
                    "x_range": d.x_range,
                    "description": d.description
                }
                for d in self._domains.values()
            ]
        }

    def solve_latent_x(self, domain_id: str, value: float) -> Dict[str, Any]:
        """
        Solves a quadratic equation for the given domain value to extract x.
        x is the fundamental latent state under study ('predmet skúmania').
        """
        if domain_id not in self._domains:
            raise KeyError(f"Domain '{domain_id}' not found.")

        domain = self._domains[domain_id]
        chosen_x, delta, is_real = domain.solve_x(value)

        # Calculate vertex (apex of the parabola)
        apex_x = -domain.b / (2.0 * domain.a) if abs(domain.a) > 1e-12 else 0.0
        apex_v = domain.evaluate(apex_x)

        return {
            "domain_id": domain_id,
            "domain_name": domain.name,
            "input_value": value,
            "unit": domain.unit,
            "latent_x": chosen_x,
            "discriminant_delta": delta,
            "is_real_root": is_real,
            "parabola_apex": {"x": round(apex_x, 4), "value": apex_v},
            "formula_solved": f"{domain.a}*x^2 + {domain.b}*x + ({domain.c} - {value}) = 0",
            "vital_max_hp": VITAL_MAX_HP
        }

    def convert(self, source_domain_id: str, source_val: float, target_domain_id: str) -> Dict[str, Any]:
        """
        Converts variable from source domain to target domain via the latent bridge x:
        V_source -> x -> V_target.
        """
        if source_domain_id not in self._domains:
            raise KeyError(f"Source domain '{source_domain_id}' not found.")
        if target_domain_id not in self._domains:
            raise KeyError(f"Target domain '{target_domain_id}' not found.")

        source_domain = self._domains[source_domain_id]
        target_domain = self._domains[target_domain_id]

        chosen_x, delta, is_real = source_domain.solve_x(source_val)
        target_val = target_domain.evaluate(chosen_x if chosen_x is not None else 0.0)

        return {
            "source": {
                "domain_id": source_domain.domain_id,
                "name": source_domain.name,
                "value": source_val,
                "unit": source_domain.unit
            },
            "latent_bridge_x": chosen_x,
            "discriminant_delta": delta,
            "is_real_transformation": is_real,
            "target": {
                "domain_id": target_domain.domain_id,
                "name": target_domain.name,
                "value": target_val,
                "unit": target_domain.unit
            },
            "conversion_pipeline": f"{source_domain.unit} --[solve x]--> x={chosen_x} --[eval V(x)]--> {target_domain.unit}",
            "vital_max_hp_rule": VITAL_MAX_HP
        }

    def convert_all(self, source_domain_id: str, source_val: float) -> Dict[str, Any]:
        """
        Projects an observed value from any domain into ALL other platform domains
        simultaneously through the extracted latent parameter x.
        """
        if source_domain_id not in self._domains:
            raise KeyError(f"Source domain '{source_domain_id}' not found.")

        src = self._domains[source_domain_id]
        chosen_x, delta, is_real = src.solve_x(source_val)
        x_val = chosen_x if chosen_x is not None else 0.0

        projected = {}
        for d_id, dom in self._domains.items():
            projected[d_id] = {
                "name": dom.name,
                "value": dom.evaluate(x_val),
                "unit": dom.unit,
                "category": dom.category
            }

        return {
            "source_domain": source_domain_id,
            "source_value": source_val,
            "latent_x_extracted": x_val,
            "discriminant": delta,
            "is_real": is_real,
            "projected_variables": projected,
            "vital_max_hp_rule": VITAL_MAX_HP
        }

    def evaluate_at_x(self, x: float) -> Dict[str, Any]:
        """Evaluates all registered domains at an arbitrary latent coordinate x."""
        clamped_x = max(0.0, min(1.0, x))
        results = {}
        for d_id, dom in self._domains.items():
            results[d_id] = {
                "name": dom.name,
                "value": dom.evaluate(clamped_x),
                "unit": dom.unit,
                "category": dom.category
            }

        return {
            "latent_x": clamped_x,
            "is_golden_ratio_state": abs(clamped_x - INV_GOLDEN_RATIO) < 0.001,
            "variables": results,
            "vital_max_hp_rule": VITAL_MAX_HP
        }

    def evaluate_golden_mean(self) -> Dict[str, Any]:
        """
        Evaluates the platform equilibrium at the Aristotelian Golden Mean
        x = phi^-1 = 0.61803398875.
        """
        data = self.evaluate_at_x(INV_GOLDEN_RATIO)
        data["note"] = "Aristotelian Golden Mean Equilibrium (Harmonic balance between starvation and congestion)"
        return data


# Global singleton instance
GLOBAL_QUADRATIC_TRANSFORMER = QuadraticVariableTransformer()
