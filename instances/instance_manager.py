"""
Krystal-Stack Platform Framework: Instance Manager
===================================================
Manages geometric and artistic instances for recursive augmented reality (AR)
mirroring, kaleidoscopic IFS folding, and Vulkan/Godot shader uniform composition.
"""

import json
import math
import os
import sys
from typing import Dict, List, Any, Optional

if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

CATALOG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "instance_database.json")

class InstanceManager:
    """
    Loads, queries, mutates, and composes geometric and artistic instances
    into unified AR rendering templates with mathematical composition rules.
    """
    def __init__(self, catalog_path: str = CATALOG_PATH):
        self.catalog_path = catalog_path
        self.catalog: Dict[str, Any] = self._load_catalog()

    def _load_catalog(self) -> Dict[str, Any]:
        if not os.path.exists(self.catalog_path):
            raise FileNotFoundError(f"Instance database not found at: {self.catalog_path}")
        with open(self.catalog_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def list_geometric_instances(self) -> List[Dict[str, Any]]:
        return self.catalog.get("geometric_instances", [])

    def list_artistic_instances(self) -> List[Dict[str, Any]]:
        return self.catalog.get("artistic_instances", [])

    def get_geometric_instance(self, instance_id: str) -> Optional[Dict[str, Any]]:
        for inst in self.list_geometric_instances():
            if inst.get("id") == instance_id:
                return inst
        return None

    def get_artistic_instance(self, instance_id: str) -> Optional[Dict[str, Any]]:
        for inst in self.list_artistic_instances():
            if inst.get("id") == instance_id:
                return inst
        return None

    def compose_template(
        self,
        geometric_id: str,
        artistic_id: str,
        mirror_folds: Optional[int] = None,
        recursion_depth: Optional[int] = None,
        fresnel_reflectance: Optional[float] = None,
        custom_notes: str = ""
    ) -> Dict[str, Any]:
        """
        Synthesizes a combined AR composition template based on geometric mirroring
        principles, dihedral symmetry groups, and shader uniforms.
        """
        geom = self.get_geometric_instance(geometric_id)
        if not geom:
            raise ValueError(f"Unknown geometric instance ID: '{geometric_id}'")

        art = self.get_artistic_instance(artistic_id)
        if not art:
            raise ValueError(f"Unknown artistic instance ID: '{artistic_id}'")

        comp_rules = geom.get("composition_rules", {})
        folds = mirror_folds if mirror_folds is not None else comp_rules.get("mirror_folds", 6)
        depth = recursion_depth if recursion_depth is not None else comp_rules.get("depth_layers", 24)
        fresnel = fresnel_reflectance if fresnel_reflectance is not None else comp_rules.get("fresnel_reflectance", 0.88)

        # Dihedral angle calculation: alpha = pi / N
        fold_angle_deg = 360.0 / (2.0 * folds) if folds > 0 else 60.0
        fold_angle_rad = math.pi / folds if folds > 0 else math.pi / 6.0

        # Golden ratio harmonic scaling
        phi = 1.61803398875
        harmonic_scale = 1.0 / (phi ** (1.0 / max(1, folds)))

        template = {
            "template_id": f"TPL_{geom['id']}_{art['id']}_{folds}F",
            "name": f"{geom['name']} × {art['name']}",
            "geometric_instance": geom,
            "artistic_instance": art,
            "composition_rules": {
                "mirror_folds": folds,
                "fold_angle_deg": round(fold_angle_deg, 3),
                "fold_angle_rad": round(fold_angle_rad, 5),
                "recursion_depth": depth,
                "fresnel_reflectance": fresnel,
                "golden_harmonic_scale": round(harmonic_scale, 5),
                "symmetry_group": f"D{folds} (Dihedral {folds * 2}-fold symmetry)"
            },
            "vulkan_shader_uniforms": {
                "u_mirror_folds": folds,
                "u_fold_angle": fold_angle_rad,
                "u_recursion_limit": depth,
                "u_fresnel_factor": fresnel,
                "u_chromatic_aberration": art.get("chromatic_aberration", 0.005),
                "u_scanline_density": art.get("scanline_density", 120.0),
                "u_primary_color": [
                    art["primary_palette"]["r"] / 255.0,
                    art["primary_palette"]["g"] / 255.0,
                    art["primary_palette"]["b"] / 255.0,
                    1.0
                ],
                "u_accent_color": [
                    art["accent_palette"]["r"] / 255.0,
                    art["accent_palette"]["g"] / 255.0,
                    art["accent_palette"]["b"] / 255.0,
                    1.0
                ]
            },
            "ascii_config": {
                "glyph_ramp": art.get("glyph_ramp", " ░▒▓█"),
                "preferred_glyphs": geom.get("preferred_glyphs", ["*", "+", "#"]),
                "hud_elements": art.get("hud_elements", [])
            },
            "custom_notes": custom_notes
        }

        return template

    def render_ascii_frame(self, template: Dict[str, Any], t: float = 0.0, width: int = 64, height: int = 24) -> str:
        """
        Generates a 2D ASCII preview applying kaleidoscopic dihedral mirror folds
        and the template's glyph ramp.
        """
        rules = template["composition_rules"]
        folds = rules["mirror_folds"]
        glyph_ramp = template["ascii_config"]["glyph_ramp"]
        preferred_glyphs = template["ascii_config"]["preferred_glyphs"]
        geom_id = template["geometric_instance"]["id"]

        lines = []
        aspect = 2.0  # Character aspect ratio compensation

        for y in range(height):
            line = []
            ny = (y / (height - 1)) * 2.0 - 1.0  # [-1, 1]
            for x in range(width):
                nx = ((x / (width - 1)) * 2.0 - 1.0) * aspect  # [-aspect, aspect]

                # 1. Polar conversion
                r = math.sqrt(nx * nx + ny * ny)
                theta = math.atan2(ny, nx)

                # 2. Dihedral mirror folding (Kaleidoscopic D_N symmetry)
                fold_sector = math.pi / max(1, folds)
                theta_mod = abs((theta % (2.0 * fold_sector)) - fold_sector)

                # 3. Folded Cartesian coordinates
                fx = r * math.cos(theta_mod)
                fy = r * math.sin(theta_mod)

                # 4. Geometric manifold evaluation
                val = 0.0
                if "OCTAHEDRON" in geom_id:
                    # Octahedral distance field approximation
                    val = abs(fx) + abs(fy) + 0.3 * math.sin(t * 1.5 + r * 4.0)
                elif "DODECAHEDRON" in geom_id:
                    phi = 1.618
                    val = max(abs(fx * phi + fy / phi), abs(fy * phi + 0.5))
                elif "TORUS" in geom_id:
                    val = abs(math.sin(r * 5.0 - t * 2.0) * math.cos(theta_mod * 3.0))
                elif "GYROID" in geom_id:
                    val = abs(math.sin(fx * 3.0 + t) * math.cos(fy * 3.0) + math.sin(r * 2.0))
                elif "MENGER" in geom_id:
                    # Recursive grid cavities
                    val = (int(abs(fx * 4.0 + t * 0.2)) % 3 == 1 and int(abs(fy * 4.0)) % 3 == 1) * 0.8 + 0.2 * r
                elif "KALEIDOSCOPIC" in geom_id:
                    val = math.sin(fx * 6.0 - t) * math.sin(fy * 6.0 + t) + 0.5 * math.cos(r * 8.0)
                elif "CALABI" in geom_id:
                    # Complex stereographic wave
                    val = abs(math.sin(fx * 4.0 + math.cos(t)) * math.cos(fy * 4.0 + math.sin(t)))
                else:
                    val = math.sin(r * 6.0 - t * 2.0)

                # Normalize and map to glyph
                norm_val = max(0.0, min(1.0, (val + 1.0) * 0.5))
                
                # Check for geometric boundary accents
                if abs(val - 1.0) < 0.1 and preferred_glyphs:
                    glyph = preferred_glyphs[int((x + y + int(t * 4)) % len(preferred_glyphs))]
                else:
                    glyph_idx = int(norm_val * (len(glyph_ramp) - 1))
                    glyph = glyph_ramp[glyph_idx]

                line.append(glyph)
            lines.append("".join(line))

        return "\n".join(lines)


if __name__ == "__main__":
    mgr = InstanceManager()
    print("[KRYSTAL] InstanceManager initialized successfully.")
    geoms = mgr.list_geometric_instances()
    arts = mgr.list_artistic_instances()
    print(f"Geometric Instances ({len(geoms)}): {[g['id'] for g in geoms]}")
    print(f"Artistic Instances  ({len(arts)}): {[a['id'] for a in arts]}")

    tpl = mgr.compose_template("KALEIDOSCOPIC_IFS", "CYBERPUNK_NEON_AR", mirror_folds=6)
    print(f"\nGenerated Template: {tpl['name']}")
    print(f"Symmetry: {tpl['composition_rules']['symmetry_group']}")
    print(f"Vulkan Shader Uniforms: {tpl['vulkan_shader_uniforms']}")
    print("\n--- ASCII Preview Frame (Kaleidoscopic IFS × Cyberpunk Neon AR) ---")
    print(mgr.render_ascii_frame(tpl, t=1.2, width=64, height=16))
