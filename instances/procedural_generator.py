"""
Krystal-Stack Platform Framework: Procedural Instance Generator
==============================================================
Synthesizes infinite, novel 3D geometric and artistic instances dynamically
using the 3D Gielis Superformula, 4D Toroidal Hopf projections, biomorphic
growth grammars, and cybernetic alchemical assembly.
"""

import math
import random
import sys
from typing import Dict, Any, List, Tuple, Optional

if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

class ProceduralInstanceGenerator:
    """
    Generates new, valid Krystal-Stack geometric and composite instances
    from mathematical seeds, parametric formulas, or semantic tags.
    """

    CATEGORIES = [
        "superformula_organic",
        "toroidal_hopf_manifold",
        "cybernetic_spire",
        "biomorphic_xenodrone",
        "alchemical_polyhedron",
        "hyperbolic_minimal_tpms"
    ]

    @staticmethod
    def superformula_2d(phi: float, m: float, n1: float, n2: float, n3: float, a: float = 1.0, b: float = 1.0) -> float:
        """Evaluates Johan Gielis' 2D Superformula radius r(phi)."""
        t1 = abs(math.cos(m * phi / 4.0) / a) ** n2
        t2 = abs(math.sin(m * phi / 4.0) / b) ** n3
        val = (t1 + t2)
        if val <= 1e-8:
            return 1e-4
        return val ** (-1.0 / n1)

    @classmethod
    def generate_superformula_instance(cls, seed: Optional[int] = None) -> Dict[str, Any]:
        """
        Synthesizes a novel organic 3D instance using spherical superformula coupling.
        Produces stars, polyhedrons, organic shells, and crystalline flowers.
        """
        rng = random.Random(seed)
        m = rng.choice([3, 4, 5, 6, 7, 8, 10, 12])
        n1 = round(rng.uniform(0.2, 5.0), 3)
        n2 = round(rng.uniform(0.5, 4.0), 3)
        n3 = round(rng.uniform(0.5, 4.0), 3)
        a = 1.0
        b = 1.0
        scale = round(rng.uniform(1.2, 2.2), 2)
        inst_id = f"PROC_SUPERFORMULA_M{m}_S{seed if seed is not None else rng.randint(100, 999)}"

        # Compute symmetry group
        symmetry_group = f"D{m} (Dihedral {m*2}-fold)" if m % 2 == 0 else f"C{m}v ({m}-fold Radial)"

        glyph_sets = [
            ["✦", "✧", "✶", "✹", "✺", "*", "+", "·"],
            ["◈", "◇", "◆", "⋄", "x", "+", ":", "."],
            ["⬡", "⬢", "⌬", "⎔", "O", "o", "°", "·"]
        ]
        glyphs = rng.choice(glyph_sets)

        return {
            "id": inst_id,
            "name": f"Procedural Gielis Manifold (m={m}, n1={n1})",
            "category": "superformula_organic",
            "generated": True,
            "seed": seed,
            "symmetry_group": symmetry_group,
            "formula": f"r(phi) = (|cos({m}*phi/4)|^{n2} + |sin({m}*phi/4)|^{n3})^(-1/{n1})",
            "default_params": {
                "m": m,
                "n1": n1,
                "n2": n2,
                "n3": n3,
                "a": a,
                "b": b,
                "scale": scale,
                "bound_radius": round(scale * 1.35, 2)
            },
            "composition_rules": {
                "mirror_folds": m if m <= 16 else 8,
                "fresnel_reflectance": round(rng.uniform(0.75, 0.95), 2),
                "depth_layers": rng.choice([16, 24, 32])
            },
            "preferred_glyphs": glyphs,
            "sdf_evaluator": {
                "type": "SUPERFORMULA_3D",
                "approx_lipschitz": 0.8
            }
        }

    @classmethod
    def generate_cyber_spire_instance(cls, seed: Optional[int] = None) -> Dict[str, Any]:
        """Synthesizes a procedural Cyberpunk Data Spire with random decks and fins."""
        rng = random.Random(seed)
        decks = rng.randint(3, 7)
        core_radius = round(rng.uniform(0.3, 0.6), 2)
        deck_radius = round(rng.uniform(core_radius + 0.4, core_radius + 1.0), 2)
        height = round(rng.uniform(3.5, 6.0), 2)
        fin_count = rng.choice([3, 4, 6, 8])
        inst_id = f"PROC_CYBER_SPIRE_D{decks}_F{fin_count}"

        return {
            "id": inst_id,
            "name": f"Modular Cyber Spire ({decks} Decks, {fin_count} Fins)",
            "category": "cybernetic_spire",
            "generated": True,
            "seed": seed,
            "symmetry_group": f"D{fin_count} (Radial Fin Symmetry)",
            "formula": f"Cylinder(r={core_radius}, h={height}) + RadialArray({fin_count}, Fins)",
            "default_params": {
                "decks": decks,
                "core_radius": core_radius,
                "deck_radius": deck_radius,
                "height": height,
                "fin_count": fin_count,
                "bound_radius": round(math.hypot(deck_radius, height * 0.5), 2)
            },
            "composition_rules": {
                "mirror_folds": fin_count,
                "fresnel_reflectance": 0.92,
                "depth_layers": 28
            },
            "preferred_glyphs": ["█", "▓", "▒", "|", "/", "-", "\\", ":", "."],
            "sdf_evaluator": {
                "type": "COMPOSITE_MODIFIER_STACK",
                "modifiers": ["ArrayModifier", "RadialMirror", "Bevel"]
            }
        }

    @classmethod
    def generate_alchemical_polyhedron_instance(cls, seed: Optional[int] = None) -> Dict[str, Any]:
        """Synthesizes a sacred alchemical geometry with twist and golden ratio harmonics."""
        rng = random.Random(seed)
        faces = rng.choice([6, 8, 12, 20])
        twist_rate = round(rng.uniform(0.3, 1.2), 2)
        scale = round(rng.uniform(1.2, 2.0), 2)
        inst_id = f"PROC_ALCHEMICAL_F{faces}_T{int(twist_rate*10)}"

        return {
            "id": inst_id,
            "name": f"Sacred Alchemical Manifold ({faces} Faces, Twist {twist_rate})",
            "category": "alchemical_polyhedron",
            "generated": True,
            "seed": seed,
            "symmetry_group": f"C{faces}v Harmonic",
            "formula": f"Polyhedron({faces}) twisted at {twist_rate} rad/y",
            "default_params": {
                "faces": faces,
                "twist_rate": twist_rate,
                "scale": scale,
                "phi_ratio": 1.6180339887,
                "bound_radius": round(scale * 1.2, 2)
            },
            "composition_rules": {
                "mirror_folds": min(12, faces),
                "fresnel_reflectance": 0.95,
                "depth_layers": 32
            },
            "preferred_glyphs": ["✦", "✧", "⬡", "◈", "◇", "°", "·"],
            "sdf_evaluator": {
                "type": "TWISTED_POLYHEDRON",
                "approx_lipschitz": 0.85
            }
        }

    @classmethod
    def generate_random_instance(cls, category: Optional[str] = None, seed: Optional[int] = None) -> Dict[str, Any]:
        """Generates a novel procedural instance in the requested or random category."""
        if seed is None:
            seed = random.randint(1000, 999999)
        rng = random.Random(seed)

        if not category:
            category = rng.choice(["superformula_organic", "cybernetic_spire", "alchemical_polyhedron"])

        if category == "cybernetic_spire":
            return cls.generate_cyber_spire_instance(seed)
        elif category == "alchemical_polyhedron":
            return cls.generate_alchemical_polyhedron_instance(seed)
        else: # Default superformula
            return cls.generate_superformula_instance(seed)
