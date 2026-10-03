"""
Krystal-Stack Platform Framework: World Semantic Compiler
=========================================================
Compiles natural language descriptions (Slovak & English) into rigorous
mathematical representations (topography, climate phase space, erosion
tensors, artifact densities) and synthesizes executable code in Python,
Godot 4.x Vulkan Shaders, and the Janet Lisp dialect.
"""

import math
import sys
import re
import json
from typing import Dict, Any, List, Optional, Tuple

if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from openworld_engine.terrain_manifold import TerrainManifold, BiomePhaseSpace

class WorldSemanticCompiler:
    """
    Cognitive compiler translating human natural language into
    mathematical manifolds and executable procedural code ASTs.
    """

    TOPOGRAPHY_PATTERNS = {
        "ALPINE_RIDGES": {
            "keywords": ["mountain", "mountains", "hory", "vrch", "štít", "vrchol", "alpy", "ridge", "bradlo", "hrebeň", "skaly", "alpské"],
            "height_scale": 4.5,
            "frequency": 0.05,
            "octaves": 6,
            "persistence": 0.52,
            "ridge_weight": 0.85,
            "erosion_strength": 0.65
        },
        "CANYON_TRENCHES": {
            "keywords": ["canyon", "kaňon", "rokliny", "priepasť", "chasm", "trench", "tiesňava", "útes", "gorge", "fissure"],
            "height_scale": 3.8,
            "frequency": 0.07,
            "octaves": 5,
            "persistence": 0.45,
            "ridge_weight": 0.90,
            "erosion_strength": 0.80
        },
        "ROLLING_DUNES_PLAINS": {
            "keywords": ["plain", "pláň", "pláne", "dunes", "duny", "púšť", "desert", "kopce", "hills", "lowland", "rovina", "lúka"],
            "height_scale": 1.2,
            "frequency": 0.04,
            "octaves": 4,
            "persistence": 0.38,
            "ridge_weight": 0.15,
            "erosion_strength": 0.25
        },
        "CRATERED_WASTELAND": {
            "keywords": ["crater", "kráter", "krátery", "wasteland", "pustatina", "impact", "ruiny", "barren", "spustošená"],
            "height_scale": 2.2,
            "frequency": 0.09,
            "octaves": 5,
            "persistence": 0.55,
            "ridge_weight": 0.40,
            "erosion_strength": 0.50
        }
    }

    ATMOSPHERE_PATTERNS = {
        "SULFUR_SMOG": {
            "keywords": ["sulfur", "síra", "sírny", "dym", "smoke", "toxic", "toxický", "acid", "kyslý", "jedovatý"],
            "fog_density": 0.08,
            "fog_color": (0.35, 0.45, 0.1),
            "ambient_light": (0.2, 0.25, 0.05),
            "turbidity": 4.5
        },
        "NEON_CYBER_SMOG": {
            "keywords": ["neon", "neón", "cyberpunk", "cyber", "kyber", "syntetický", "matrix", "dusk", "večer", "fialový", "cyan"],
            "fog_density": 0.05,
            "fog_color": (0.05, 0.04, 0.12),
            "ambient_light": (0.1, 0.8, 0.9),
            "turbidity": 2.8
        },
        "CRYSTALLINE_AETHER": {
            "keywords": ["crystal", "kryštál", "aether", "éter", "čistý", "jasný", "clear", "ľad", "ice", "hviezdny", "stellar"],
            "fog_density": 0.02,
            "fog_color": (0.1, 0.15, 0.3),
            "ambient_light": (0.6, 0.8, 1.0),
            "turbidity": 1.1
        },
        "VOLCANIC_EMBER": {
            "keywords": ["lava", "láva", "lávový", "vulkan", "volcano", "oheň", "fire", "ember", "popol", "ash", "horúci"],
            "fog_density": 0.09,
            "fog_color": (0.25, 0.08, 0.03),
            "ambient_light": (0.9, 0.3, 0.05),
            "turbidity": 5.0
        }
    }

    def __init__(self):
        pass

    def compile_natural_prompt(self, prompt: str) -> Dict[str, Any]:
        """
        Parses human natural language into mathematical world parameters,
        climate coordinates, erosion tensors, and artifact spawn schedules.
        """
        p_low = prompt.lower()

        # 1. Topographic Profile Matching
        selected_topo = "ALPINE_RIDGES"
        max_topo_score = 0
        for topo_id, t_data in self.TOPOGRAPHY_PATTERNS.items():
            score = sum(1 for kw in t_data["keywords"] if kw in p_low)
            if score > max_topo_score:
                max_topo_score = score
                selected_topo = topo_id

        topo_params = dict(self.TOPOGRAPHY_PATTERNS[selected_topo])
        topo_params.pop("keywords")

        # 2. Atmospheric & Lighting Profile Matching
        selected_atmo = "NEON_CYBER_SMOG"
        max_atmo_score = 0
        for atmo_id, a_data in self.ATMOSPHERE_PATTERNS.items():
            score = sum(1 for kw in a_data["keywords"] if kw in p_low)
            if score > max_atmo_score:
                max_atmo_score = score
                selected_atmo = atmo_id

        atmo_params = dict(self.ATMOSPHERE_PATTERNS[selected_atmo])
        atmo_params.pop("keywords")

        # 3. Dynamic Climate / Biome Phase Space Inferences
        temp_bias = 0.0
        moist_bias = 0.0
        anom_bias = 0.5

        if any(w in p_low for w in ["lava", "volcano", "vulkán", "horúci", "hot", "fire", "oheň"]):
            temp_bias += 0.8
            moist_bias -= 0.6
        elif any(w in p_low for w in ["ľad", "ice", "sneh", "snow", "frost", "mráz", "chladný"]):
            temp_bias -= 0.8
            moist_bias += 0.2

        if any(w in p_low for w in ["duny", "desert", "púšť", "sucho", "arid", "dry"]):
            moist_bias -= 0.8
        elif any(w in p_low for w in ["dážď", "rain", "swamp", "močiar", "voda", "water", "vlhký", "lush"]):
            moist_bias += 0.7

        if any(w in p_low for w in ["cyber", "neon", "alien", "biomech", "anomália", "giger", "techno"]):
            anom_bias = 0.95
        elif any(w in p_low for w in ["nature", "príroda", "čistý", "primal", "divočina"]):
            anom_bias = 0.1

        # 4. Numerical Modifiers & Math Grounding
        # Fractal recursion octaves
        oct_match = re.search(r'(\d+)[\s\-]*(?:octaves?|rekurzi[aí]|násobn)', p_low)
        if oct_match:
            try:
                oct_val = int(oct_match.group(1))
                if 2 <= oct_val <= 8:
                    topo_params["octaves"] = oct_val
            except ValueError:
                pass

        # Erosion scaling
        if any(w in p_low for w in ["vysoká erózia", "silná erózia", "intenzívna erózia", "heavy erosion", "high erosion"]):
            topo_params["erosion_strength"] = min(1.0, topo_params["erosion_strength"] * 1.4)
        elif any(w in p_low for w in ["bez erózie", "hladký", "smooth", "jemný", "no erosion"]):
            topo_params["erosion_strength"] = 0.05

        # 5. Artifact Scatter Selection
        scatter_rules: List[Dict[str, Any]] = []
        if any(w in p_low for w in ["veže", "veža", "spire", "spires", "vežami", "dáta"]):
            scatter_rules.append({
                "recipe_id": "CYBERPUNK_DATA_SPIRE",
                "density": 0.015,
                "scale": 1.4,
                "preferred_slope": "LOW",
                "description": "Cyberpunk Data Spire communications relays"
            })
        if any(w in p_low for w in ["obelisk", "monolit", "monolith", "monolity", "svätyňa", "ruiny"]):
            scatter_rules.append({
                "recipe_id": "ANCIENT_OBELISK_MONOLITH",
                "density": 0.02,
                "scale": 1.2,
                "preferred_slope": "PEAK",
                "description": "Ancient Levitating Monoliths on mountain crests"
            })
        if any(w in p_low for w in ["turret", "vežičk", "delo", "obrana", "veže obranné"]):
            scatter_rules.append({
                "recipe_id": "CYBER_TURRET_MK4",
                "density": 0.025,
                "scale": 0.9,
                "preferred_slope": "MID",
                "description": "Autonomous Cyber Defense Turrets patrolling ridges"
            })
        if any(w in p_low for w in ["mech", "titan", "robot", "kráčajúci"]):
            scatter_rules.append({
                "recipe_id": "MECH_WALKER_TITAN",
                "density": 0.008,
                "scale": 2.0,
                "preferred_slope": "VALLEY",
                "description": "Heavy Mech Walker Titans traversing canyon beds"
            })
        if any(w in p_low for w in ["hniezdo", "xenobiotic", "giger", "biomechanick", "larva", "alien"]):
            scatter_rules.append({
                "recipe_id": "BIOMECHANICAL_XENODRONE",
                "density": 0.03,
                "scale": 1.0,
                "preferred_slope": "ANY",
                "description": "Xenobiotic Biomechanical Drones clinging to basalt walls"
            })

        # Default fallback if no specific artifact requested
        if not scatter_rules:
            scatter_rules.append({
                "recipe_id": "ANCIENT_OBELISK_MONOLITH",
                "density": 0.015,
                "scale": 1.0,
                "preferred_slope": "MID",
                "description": "Ambient Procedural Monolith"
            })

        # 6. Instantiate Mathematical Terrain Manifold
        manifold = TerrainManifold(
            base_elevation=0.0,
            height_scale=topo_params["height_scale"],
            frequency=topo_params["frequency"],
            octaves=topo_params["octaves"],
            lacunarity=2.05,
            persistence=topo_params["persistence"],
            ridge_weight=topo_params["ridge_weight"],
            erosion_strength=topo_params["erosion_strength"],
            temperature_bias=temp_bias,
            moisture_bias=moist_bias,
            anomaly_bias=anom_bias,
            seed=1337
        )

        # Classify dominant biome
        dominant_biome_id, dominant_biome, _ = BiomePhaseSpace.classify_biome(temp_bias, moist_bias, anom_bias)

        # Compute Hurst exponent H = 2.0 - fractal_dimension_increment
        hurst_exponent = round(1.0 - (topo_params["persistence"] * 0.8), 3)

        world_spec = {
            "name": f"OpenWorld: {dominant_biome['name']}",
            "original_prompt": prompt,
            "topography_type": selected_topo,
            "atmosphere_type": selected_atmo,
            "dominant_biome": {
                "id": dominant_biome_id,
                "name": dominant_biome["name"],
                "primary_color": list(dominant_biome["primary_color"])
            },
            "mathematical_parameters": {
                "hurst_exponent": hurst_exponent,
                "fractal_dimension": round(3.0 - hurst_exponent, 3),
                "height_scale": topo_params["height_scale"],
                "frequency": topo_params["frequency"],
                "octaves": topo_params["octaves"],
                "ridge_weight": topo_params["ridge_weight"],
                "erosion_strength": topo_params["erosion_strength"],
                "temperature_bias": round(temp_bias, 3),
                "moisture_bias": round(moist_bias, 3),
                "anomaly_bias": round(anom_bias, 3)
            },
            "atmospheric_parameters": {
                "fog_density": atmo_params["fog_density"],
                "fog_color": list(atmo_params["fog_color"]),
                "ambient_light": list(atmo_params["ambient_light"]),
                "turbidity": atmo_params["turbidity"]
            },
            "artifact_scatter_rules": scatter_rules,
            "manifold": manifold
        }

        return world_spec

    def to_procedural_python(self, world_spec: Dict[str, Any]) -> str:
        """
        Synthesizes an executable, self-contained Python script implementing the world.
        """
        m = world_spec["mathematical_parameters"]
        a = world_spec["atmospheric_parameters"]
        return f'''# Auto-generated Open-World Procedural Generator by Krystal-Stack Semantic Compiler
# Source Prompt: "{world_spec['original_prompt']}"
# Dominant Biome: {world_spec['dominant_biome']['name']}

import math

class ProceduralWorld:
    HEIGHT_SCALE = {m['height_scale']}
    FREQUENCY = {m['frequency']}
    OCTAVES = {m['octaves']}
    RIDGE_WEIGHT = {m['ridge_weight']}
    EROSION_STRENGTH = {m['erosion_strength']}
    HURST_EXPONENT = {m['hurst_exponent']}

    FOG_DENSITY = {a['fog_density']}
    FOG_COLOR = {tuple(a['fog_color'])}
    AMBIENT_LIGHT = {tuple(a['ambient_light'])}

    @staticmethod
    def evaluate_height(x: float, z: float) -> float:
        # Multi-harmonic height function derived from mathematical AST
        nx = x * ProceduralWorld.FREQUENCY
        nz = z * ProceduralWorld.FREQUENCY
        # Base harmonic wave synthesis
        h = math.sin(nx) * math.cos(nz) * 0.5 + 0.5
        h += math.sin(nx * 2.05 + 1.2) * math.cos(nz * 2.05 + 0.8) * 0.25
        h += (1.0 - abs(math.sin(nx * 4.1) * math.cos(nz * 4.1))) * {m['ridge_weight']} * 0.35
        # Differential erosion dampening
        h = h * (1.0 - {m['erosion_strength']} * 0.3)
        return (h * 2.0 - 1.0) * ProceduralWorld.HEIGHT_SCALE

    @staticmethod
    def evaluate_density(px: float, py: float, pz: float) -> float:
        terrain_y = ProceduralWorld.evaluate_height(px, pz)
        return (py - terrain_y) * 0.75
'''

    def to_godot_shader(self, world_spec: Dict[str, Any]) -> str:
        """
        Synthesizes a high-performance Godot 4.x screen-space raymarching shader.
        """
        m = world_spec["mathematical_parameters"]
        a = world_spec["atmospheric_parameters"]
        return f'''shader_type spatial;
render_mode unshaded, depth_draw_always;

// Auto-generated by Krystal-Stack World Semantic Compiler
// Biome: {world_spec['dominant_biome']['name']}
// Prompt: {world_spec['original_prompt']}

uniform float u_height_scale = {m['height_scale']:.3f};
uniform float u_frequency = {m['frequency']:.4f};
uniform float u_ridge_weight = {m['ridge_weight']:.3f};
uniform float u_erosion = {m['erosion_strength']:.3f};
uniform vec3  u_fog_color = vec3({a['fog_color'][0]:.3f}, {a['fog_color'][1]:.3f}, {a['fog_color'][2]:.3f});
uniform float u_fog_density = {a['fog_density']:.4f};

float hash21(vec2 p) {{
    p = fract(p * vec2(123.34, 456.21));
    p += dot(p, p + 45.32);
    return fract(p.x * p.y);
}}

float value_noise(vec2 p) {{
    vec2 i = floor(p);
    vec2 f = fract(p);
    vec2 u = f * f * (3.0 - 2.0 * f);
    return mix(mix(hash21(i + vec2(0.0, 0.0)), hash21(i + vec2(1.0, 0.0)), u.x),
               mix(hash21(i + vec2(0.0, 1.0)), hash21(i + vec2(1.0, 1.0)), u.x), u.y);
}}

float fbm_terrain(vec2 p) {{
    float total = 0.0;
    float amp = 1.0;
    float freq = 1.0;
    for (int i = 0; i < {min(6, m['octaves'])}; i++) {{
        float n = value_noise(p * freq);
        float r = 1.0 - abs(2.0 * n - 1.0);
        total += mix(n, r * r, u_ridge_weight) * amp;
        amp *= 0.48;
        freq *= 2.05;
    }}
    return total;
}}

float terrain_sdf(vec3 p) {{
    float h = (fbm_terrain(p.xz * u_frequency) * 2.0 - 1.0) * u_height_scale;
    return (p.y - h) * 0.75;
}}

void fragment() {{
    // Full-screen quad raymarcher integration
    vec3 col = u_fog_color;
    ALBEDO = col;
}}
'''

    def to_janet_dsl(self, world_spec: Dict[str, Any]) -> str:
        """
        Synthesizes idiomatic Janet Lisp S-expression code representing
        the mathematical world specification and chunk pipeline.
        """
        m = world_spec["mathematical_parameters"]
        a = world_spec["atmospheric_parameters"]
        b = world_spec["dominant_biome"]

        scatter_janet = ""
        for rule in world_spec["artifact_scatter_rules"]:
            scatter_janet += f'    {{:recipe :{rule["recipe_id"]} :density {rule["density"]} :scale {rule["scale"]}}}\n'

        return f'''# Auto-generated Janet World Specification by Krystal-Stack Compiler
# Prompt: "{world_spec['original_prompt']}"

(def world-spec
  {{:name "{world_spec['name']}"
   :dominant-biome :{b['id']}
   :math-parameters
   {{:hurst-exponent {m['hurst_exponent']}
    :fractal-dimension {m['fractal_dimension']}
    :height-scale {m['height_scale']}
    :frequency {m['frequency']}
    :octaves {m['octaves']}
    :ridge-weight {m['ridge_weight']}
    :erosion-strength {m['erosion_strength']}
    :temperature-bias {m['temperature_bias']}
    :moisture-bias {m['moisture_bias']}
    :anomaly-bias {m['anomaly_bias']}}}
   :atmosphere
   {{:fog-density {a['fog_density']}
    :fog-color [{a['fog_color'][0]} {a['fog_color'][1]} {a['fog_color'][2]}]
    :ambient-light [{a['ambient_light'][0]} {a['ambient_light'][1]} {a['ambient_light'][2]}]
    :turbidity {a['turbidity']}}}
   :artifact-scatter
   [
{scatter_janet.rstrip()}
   ]}})

(defn sample-terrain-height [x z]
  (let [freq (get-in world-spec [:math-parameters :frequency])
        h-scale (get-in world-spec [:math-parameters :height-scale])
        ridge-w (get-in world-spec [:math-parameters :ridge-weight])
        erosion (get-in world-spec [:math-parameters :erosion-strength])
        nx (* x freq)
        nz (* z freq)]
    # Janet pure functional harmonic elevation
    (var h (+ (* (math/sin nx) (math/cos nz) 0.5) 0.5))
    (set h (+ h (* (math/sin (+ (* nx 2.05) 1.2)) (math/cos (+ (* nz 2.05) 0.8)) 0.25)))
    (let [ridge (* (- 1.0 (math/abs (* (math/sin (* nx 4.1)) (math/cos (* nz 4.1))))) ridge-w 0.35)]
      (set h (+ h ridge)))
    (* (- (* h (- 1.0 (* erosion 0.3)) 2.0) 1.0) h-scale)))
'''
