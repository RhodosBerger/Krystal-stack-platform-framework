"""
Krystal-Stack Platform Framework: Antigravity Prompt Engine
============================================================
Translates natural language prompts (multilingual EN/SK) into high-fidelity
geometric and artistic AR templates, Vulkan shader uniforms, and kaleidoscopic
mirroring configurations.
"""

import os
import sys
import re
import json
import math
from typing import Dict, Any, Optional, Tuple, List

if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from instances.instance_manager import InstanceManager, CATALOG_PATH

TEMPLATES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates")
os.makedirs(TEMPLATES_DIR, exist_ok=True)

class AntigravityPromptEngine:
    """
    Cognitive prompt processor that transforms natural language intent
    into rigorous mathematical compositions, dihedral symmetry folds,
    and Vulkan/Godot shader uniform configurations.
    """
    def __init__(self, catalog_path: str = CATALOG_PATH, templates_dir: str = TEMPLATES_DIR):
        self.instance_manager = InstanceManager(catalog_path)
        self.templates_dir = templates_dir
        self.geometry_keywords = {
            "PLATONIC_OCTAHEDRON": [
                "octahedron", "oktaeder", "ihlan", "crystal", "kryštál", "diamond", "diamant", "dual crystal"
            ],
            "PLATONIC_DODECAHEDRON": [
                "dodecahedron", "dvanásťsten", "pentagon", "pentagonálny", "golden ratio", "zlatý rez"
            ],
            "TORUS_KNOT_P3_Q5": [
                "torus", "knot", "uzol", "trefoil", "prstenec", "helix", "vortex"
            ],
            "HYPERBOLIC_GYROID": [
                "gyroid", "tpms", "minimal surface", "minimálny povrch", "hyperbolic", "hyperbolický"
            ],
            "MENGER_SPONGE_RECURSIVE": [
                "menger", "sponge", "špongia", "fractal", "fraktál", "recursive box", "rekurzia"
            ],
            "KALEIDOSCOPIC_IFS": [
                "kaleidoscopic", "kaleidoskop", "ifs", "dihedral", "sacred geometry", "posvätná geometria", "mandala"
            ],
            "CALABI_YAU_CROSS_SECTION": [
                "calabi", "yau", "calabi-yau", "manifold", "mnohotvárnosť", "superstring", "6d", "quantum space"
            ]
        }
        self.artistic_keywords = {
            "CYBERPUNK_NEON_AR": [
                "cyberpunk", "cyber", "neon", "neón", "synthwave", "retrofuturistic", "matrix", "cyan", "magenta"
            ],
            "BLUEPRINT_SCHEMATIC": [
                "blueprint", "schematic", "výkres", "technický", "iso", "cad", "wireframe", "grid"
            ],
            "BIOMECHANICAL_GIGER": [
                "giger", "biomechanical", "biomechanický", "biomech", "chitin", "organism", "alien", "organický", "ribbed"
            ],
            "HOLOGRAPHIC_QUANTUM": [
                "hologram", "holographic", "holografický", "quantum", "kvant", "laser", "interference", "anaglyph"
            ],
            "MONASTIC_ALCHEMICAL": [
                "alchemical", "alchýmia", "alchýmie", "alchym", "alchým", "monastic", "monastický", "gold", "zlato", "zlat", "runes", "runy", "sacred circle"
            ]
        }

    def parse_prompt(self, prompt: str) -> Dict[str, Any]:
        """
        Parses a natural language prompt into geometry, artistic style,
        mirror fold count, recursion depth, and custom modifications.
        """
        prompt_lower = prompt.lower()

        # 1. Match Geometric Instance
        selected_geom = "KALEIDOSCOPIC_IFS"  # Default
        max_geom_score = 0
        for geom_id, keywords in self.geometry_keywords.items():
            score = sum(1 for kw in keywords if kw in prompt_lower)
            if score > max_geom_score:
                max_geom_score = score
                selected_geom = geom_id

        # 2. Match Artistic Instance
        selected_art = "CYBERPUNK_NEON_AR"  # Default
        max_art_score = 0
        for art_id, keywords in self.artistic_keywords.items():
            score = sum(1 for kw in keywords if kw in prompt_lower)
            if score > max_art_score:
                max_art_score = score
                selected_art = art_id

        # 3. Detect Mirror Folds (Dihedral Symmetry D_N)
        mirror_folds = None
        # Match patterns like: "6-fold", "8 fold", "4 folds", "6-uholník", "8-uholník", "D6", "D8"
        fold_match = re.search(r'(\d+)[\s\-]*(?:fold|folds|násobn|uholník|hran|cíp)', prompt_lower)
        if not fold_match:
            fold_match = re.search(r'\bd(\d+)\b', prompt_lower)
        if fold_match:
            try:
                val = int(fold_match.group(1))
                if 2 <= val <= 32:
                    mirror_folds = val
            except ValueError:
                pass

        if mirror_folds is None:
            if "hexagonal" in prompt_lower or "šesť" in prompt_lower:
                mirror_folds = 6
            elif "octagonal" in prompt_lower or "osem" in prompt_lower:
                mirror_folds = 8
            elif "pentagon" in prompt_lower or "päť" in prompt_lower:
                mirror_folds = 5
            elif "quad" in prompt_lower or "štyri" in prompt_lower:
                mirror_folds = 4
            elif "triangle" in prompt_lower or "troj" in prompt_lower:
                mirror_folds = 3

        # 4. Detect Recursion Depth
        recursion_depth = None
        depth_match = re.search(r'(?:depth|hĺbka|level|úroveň|rekurzia|recursion)[\s\:\=]*(\d+)', prompt_lower)
        if depth_match:
            try:
                depth_val = int(depth_match.group(1))
                if 1 <= depth_val <= 64:
                    recursion_depth = depth_val
            except ValueError:
                pass

        # 5. Detect Fresnel / Reflectance Modifier
        fresnel = None
        if "high reflection" in prompt_lower or "vysoké zrkadlenie" in prompt_lower or "mirror" in prompt_lower:
            fresnel = 0.95
        elif "subtle" in prompt_lower or "jemné zrkadlenie" in prompt_lower or "matné" in prompt_lower:
            fresnel = 0.65

        # 6. Synthesize Template via InstanceManager
        template = self.instance_manager.compose_template(
            geometric_id=selected_geom,
            artistic_id=selected_art,
            mirror_folds=mirror_folds,
            recursion_depth=recursion_depth,
            fresnel_reflectance=fresnel,
            custom_notes=f"Generated via Antigravity Prompt: '{prompt}'"
        )

        return template

    def save_template(self, template: Dict[str, Any]) -> str:
        """
        Saves a generated template to the filesystem as JSON.
        """
        template_id = template.get("template_id", "unnamed_template")
        filepath = os.path.join(self.templates_dir, f"{template_id}.json")
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(template, f, indent=2, ensure_ascii=False)
        return filepath

    def list_saved_templates(self) -> List[str]:
        """
        Returns list of all saved template filenames.
        """
        if not os.path.exists(self.templates_dir):
            return []
        return [f for f in os.listdir(self.templates_dir) if f.endswith(".json")]

    def load_template(self, template_name_or_id: str) -> Optional[Dict[str, Any]]:
        """
        Loads a saved template by ID or filename.
        """
        if not template_name_or_id.endswith(".json"):
            template_name_or_id = f"{template_name_or_id}.json"
        filepath = os.path.join(self.templates_dir, template_name_or_id)
        if os.path.exists(filepath):
            with open(filepath, "r", encoding="utf-8") as f:
                return json.load(f)
        return None

    def export_vulkan_glsl_constants(self, template: Dict[str, Any]) -> str:
        """
        Exports uniform constants ready to paste into GLSL Vulkan compute or fragment shaders.
        """
        u = template["vulkan_shader_uniforms"]
        r = template["composition_rules"]
        return f"""// Auto-generated Vulkan GLSL Uniform Block by Antigravity Prompt Engine
// Template: {template['name']} ({r['symmetry_group']})
layout(push_constant) uniform AntigravityARBlock {{
    int   u_mirror_folds;          // = {u['u_mirror_folds']}
    float u_fold_angle;            // = {u['u_fold_angle']:.6f} rad ({r['fold_angle_deg']} deg)
    int   u_recursion_limit;       // = {u['u_recursion_limit']}
    float u_fresnel_factor;        // = {u['u_fresnel_factor']:.3f}
    float u_chromatic_aberration;  // = {u['u_chromatic_aberration']:.4f}
    float u_scanline_density;      // = {u['u_scanline_density']:.1f}
    vec4  u_primary_color;         // = vec4({u['u_primary_color'][0]:.3f}, {u['u_primary_color'][1]:.3f}, {u['u_primary_color'][2]:.3f}, 1.0)
    vec4  u_accent_color;          // = vec4({u['u_accent_color'][0]:.3f}, {u['u_accent_color'][1]:.3f}, {u['u_accent_color'][2]:.3f}, 1.0)
}} pushConstants;
"""


if __name__ == "__main__":
    engine = AntigravityPromptEngine()
    print("[KRYSTAL] Antigravity Prompt Engine initialized.")
    test_prompts = [
        "zrkadli 6-uholníkovú posvätnú geometriu v štýle alchýmie so zlatými runami a hĺbka 32",
        "recursive Menger sponge with 8-fold dihedral mirror in cyberpunk neon horizon",
        "Calabi-Yau 6D cross section with quantum holographic laser interference fringes",
        "trefoil torus knot technical blueprint schematic with CAD grid and 3 folds"
    ]

    for p in test_prompts:
        print(f"\n==========================================")
        print(f"PROMPT: \"{p}\"")
        tpl = engine.parse_prompt(p)
        saved_path = engine.save_template(tpl)
        print(f"-> Selected: {tpl['name']}")
        print(f"-> Symmetry: {tpl['composition_rules']['symmetry_group']} (Folds: {tpl['composition_rules']['mirror_folds']})")
        print(f"-> Saved: {os.path.basename(saved_path)}")
        print(f"-> Vulkan GLSL Snippet:\n{engine.export_vulkan_glsl_constants(tpl).strip()}")
        print("\nASCII Sample Preview (First 8 lines):")
        frame = engine.instance_manager.render_ascii_frame(tpl, t=0.5, width=64, height=8)
        print(frame)
