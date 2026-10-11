#!/usr/bin/env python3
"""
==============================================================================
KRYSTAL-STACK: NATIVE LANGUAGE PREDICTION TENSOR GENERATOR & ENGINE PIPELINE
==============================================================================
Module: krystal_kernel/native_language_tensor_generator.py
Description: Mathematical tensor generator that models and predicts:
             1. Average weights of usage of native languages (Slovak, Czech,
                English) when communicating with local LLMs.
             2. Linguistic domain decomposition (CNC, Physics, Software,
                Alchemical, Geometry, Colloquial).
             3. Direct local pipeline synthesis of:
                - Engine Assets (Godot 4.x / Vulkan MultiMesh binary buffers & PBR shaders).
                - System Templates (Godot .tscn, Janet DSL .janet, Keras 3 .json, HTML5 Studio).
             4. Platform Invariant: VITAL_MAX_HP == 6 strictly enforced.

Author: Dušan Kopecký & Krystal-Stack Architecture Council (2026)
==============================================================================
"""

import os
import sys
import math
import time
import json
import struct
import random
import re
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Optional, Tuple, Union

WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

from krystal_kernel.scene_constructor_presets import VITAL_MAX_HP
from krystal_kernel.keras_tensor_math_engine import TensorMatrix
from krystal_kernel.procedural_asset_synthesizer import HarmonicColorTensor, GOLDEN_RATIO

# ── 1. NATIVE LANGUAGE VOCABULARY & DOMAIN REGISTERS ───────────────────────

DOMAINS = [
    "CNC_MACHINING",
    "THERMODYNAMICS",
    "SOFTWARE_ARCHITECTURE",
    "ALCHEMICAL_LORE",
    "SPATIAL_GEOMETRY",
    "COLLOQUIAL_PROMPT"
]

# Multilingual keywords (Slovak, Czech, English)
LEXICAL_LEXICON = {
    "CNC_MACHINING": [
        "rez", "vreteno", "otacky", "posuv", "frezovanie", "frezka", "nastroj", "platok", "chladenie",
        "opotrebovanie", "upinanie", "vibracie", "ssv", "g-kod", "cnc", "spindle", "feed", "speed",
        "carbide", "endmill", "flank", "wear", "chatter", "milling", "toolpath", "torque"
    ],
    "THERMODYNAMICS": [
        "teplota", "teplo", "gradient", "chladenie", "rozptyl", "odvod", "prehriatie", "tlak", "kvapalina",
        "entalpia", "zvar", "fyzika", "dynamika", "temperature", "heat", "thermal", "flux", "coolant",
        "pressure", "convection", "dissipation", "joule", "kelvin", "celsius"
    ],
    "SOFTWARE_ARCHITECTURE": [
        "tenzor", "matica", "jadro", "algoritmus", "pamät", "pamet", "vyrovnavacia", "vlakno", "vlakna",
        "latencia", "vypocet", "architektura", "skript", "bit", "bajt", "vykon", "ipc", "cache", "thread",
        "pipeline", "latency", "register", "simd", "avx", "allocation", "thrashing", "kernel", "engine"
    ],
    "ALCHEMICAL_LORE": [
        "krystal", "alchymia", "artefakt", "amulet", "rezonancia", "magia", "dusa", "aura", "runa",
        "kamen", "zlaty", "mysticky", "talisman", "vortex", "citadela", "crystal", "alchemy", "artifact",
        "relic", "resonance", "mystic", "rune", "obsidian", "aether", "soul", "patina"
    ],
    "SPATIAL_GEOMETRY": [
        "priestor", "geometria", "spirala", "uhol", "polomer", "plocha", "objem", "polygon", "mesh",
        "veza", "bastion", "monolit", "suradnice", "zlata", "geometry", "spiral", "fermat", "gielis",
        "phyllotaxis", "radius", "bastion", "monolith", "voronoi", "fractal", "mesh", "voxel"
    ],
    "COLLOQUIAL_PROMPT": [
        "vytvor", "sprav", "vygeneruj", "ukaz", "daj", "pomoz", "navrhni", "urob", "otestuj", "prosim",
        "rychlo", "dobre", "chcem", "create", "generate", "make", "build", "show", "help", "render",
        "please", "give", "execute", "synthesize", "prototype"
    ]
}


@dataclass
class LinguisticFeatureVector:
    """Extracted numeric representation of native language input."""
    raw_text: str
    token_count: int
    char_count: int
    language_code: str  # "sk", "cs", "en", "mixed"
    shannon_entropy: float
    domain_frequencies: Dict[str, float]
    imperative_density: float
    slavic_inflection_score: float

    def to_feature_list(self) -> List[float]:
        """Returns normalized 10-dimensional feature vector."""
        dom_vals = [self.domain_frequencies.get(d, 0.0) for d in DOMAINS]
        norm_tokens = min(1.0, self.token_count / 50.0)
        return dom_vals + [
            norm_tokens,
            self.imperative_density,
            self.slavic_inflection_score,
            min(1.0, self.shannon_entropy / 5.0)
        ]


# ── 2. PREDICTION TENSOR ENGINE ─────────────────────────────────────────────

class NativeLanguageTensorPredictor:
    """
    Computes statistical and predictive tensors representing the weights
    of native language usage when commanding local LLMs.
    """

    def __init__(self):
        # Rolling history of prompt feature vectors for moving average usage tensor
        self.history_vectors: List[LinguisticFeatureVector] = []
        self.max_history: int = 100

        # Calibrated projection matrix M_proj (6 Archetypes x 10 Features)
        # Maps linguistic features directly into engine archetype weights
        self.projection_matrix = TensorMatrix([
            [2.8, 1.2, 0.4, 0.0, 0.2, 0.5, 0.2, 0.8, 0.4, 0.1],  # CNC_TOOLING
            [1.1, 3.2, 0.2, 0.2, 0.4, 0.3, 0.1, 0.6, 0.3, 0.2],  # THERMAL_MATERIAL
            [0.1, 0.2, 3.5, 0.1, 0.8, 0.6, 0.4, 0.9, 0.2, 0.5],  # COMPUTE_NODE
            [0.0, 0.1, 0.2, 3.6, 0.4, 0.4, 0.3, 0.5, 0.6, 0.4],  # SKEUOMORPHIC_RELIC
            [0.3, 0.2, 0.6, 0.3, 3.8, 0.5, 0.2, 0.7, 0.4, 0.3],  # BIOME_GEOMETRY
            [0.5, 0.4, 1.2, 0.5, 0.6, 2.5, 0.5, 1.0, 0.5, 0.4]   # NEURAL_TENSOR_CORE
        ], name="M_lang_to_archetype")

    @classmethod
    def calculate_shannon_entropy(cls, text: str) -> float:
        """Computes information density H(text) = -sum(p * log2(p))."""
        if not text:
            return 0.0
        counts = {}
        for ch in text.lower():
            counts[ch] = counts.get(ch, 0) + 1
        n = len(text)
        ent = 0.0
        for c in counts.values():
            p = c / n
            ent -= p * math.log2(p)
        return round(ent, 3)

    @classmethod
    def detect_language(cls, text: str) -> str:
        """Detects Slovak, Czech, English, or Mixed based on characteristic diacritics & grammar."""
        sk_markers = {"ô", "ä", "ľ", "ĺ", "ŕ", "dz", "dž", "chcem", "vygeneruj", "nastroj"}
        cs_markers = {"ř", "ě", "ů", "chci", "nástroj", "udělej"}
        text_lower = text.lower()
        has_sk = any(m in text_lower for m in sk_markers)
        has_cs = any(m in text_lower for m in cs_markers)
        if has_sk:
            return "sk"
        if has_cs:
            return "cs"
        if re.search(r"\b(the|and|create|generate|tool|machine|temperature)\b", text_lower):
            return "en"
        return "multilingual"

    def extract_features(self, text: str) -> LinguisticFeatureVector:
        """Analyzes text and constructs the 10D feature vector."""
        tokens = re.findall(r"\w+", text.lower())
        token_count = len(tokens)
        char_count = len(text)
        entropy = self.calculate_shannon_entropy(text)
        lang = self.detect_language(text)

        # Domain frequencies
        domain_counts = {d: 0.0 for d in DOMAINS}
        for tok in tokens:
            for d, keywords in LEXICAL_LEXICON.items():
                if any(kw in tok for kw in keywords):
                    domain_counts[d] += 1.0

        # Normalize frequencies
        total_matched = sum(domain_counts.values()) or 1.0
        domain_freqs = {d: round(c / total_matched, 3) for d, c in domain_counts.items()}

        # Imperative density: command verbs ratio
        imp_words = LEXICAL_LEXICON["COLLOQUIAL_PROMPT"]
        imperative_count = sum(1 for tok in tokens if any(kw in tok for kw in imp_words))
        imp_density = round(imperative_count / max(1, token_count), 3)

        # Slavic inflection complexity (detecting non-ASCII case suffixes and declensions)
        diacritic_chars = len([ch for ch in text if ord(ch) > 127])
        inflection_score = min(1.0, round(diacritic_chars / max(1, char_count) * 4.0, 3))

        feat = LinguisticFeatureVector(
            raw_text=text,
            token_count=token_count,
            char_count=char_count,
            language_code=lang,
            shannon_entropy=entropy,
            domain_frequencies=domain_freqs,
            imperative_density=imp_density,
            slavic_inflection_score=inflection_score
        )

        self.history_vectors.append(feat)
        if len(self.history_vectors) > self.max_history:
            self.history_vectors.pop(0)

        return feat

    def predict_usage_tensor(self, feat: LinguisticFeatureVector) -> TensorMatrix:
        """
        Generates the 6-element prediction weight tensor W_usage = Softmax(M_proj * f).
        Represents the distribution of intended engine usage from native language commands.
        """
        f_vec = feat.to_feature_list()
        raw_logits = self.projection_matrix.vector_dot(f_vec)

        # Safe Softmax
        max_l = max(raw_logits)
        exps = [math.exp(l - max_l) for l in raw_logits]
        sum_exps = sum(exps)
        probs = [[round(e / sum_exps, 4)] for e in exps]

        return TensorMatrix(probs, name="W_native_language_usage")

    def compute_average_usage_tensor(self) -> TensorMatrix:
        """
        Computes the rolling average usage tensor W_avg across all received native prompts.
        """
        if not self.history_vectors:
            # Uniform prior
            return TensorMatrix([[1.0 / 6.0] for _ in range(6)], name="W_average_usage_prior")

        avg_probs = [0.0] * 6
        for f in self.history_vectors:
            f_vec = f.to_feature_list()
            logits = self.projection_matrix.vector_dot(f_vec)
            max_l = max(logits)
            exps = [math.exp(l - max_l) for l in logits]
            s = sum(exps)
            for i in range(6):
                avg_probs[i] += (exps[i] / s)

        n = len(self.history_vectors)
        res = [[round(p / n, 4)] for p in avg_probs]
        return TensorMatrix(res, name="W_average_usage_historical")

    def synthesize_engine_modulation_tensor(self, weights: TensorMatrix) -> TensorMatrix:
        """
        Computes the (6 x 8) Engine Modulation Tensor:
        Rows: 6 Archetypes
        Columns: [geom_complexity, roughness, metallic, emission, density_scale, shader_freq, instance_scale, vital_hp]
        """
        out_rows = []
        w_list = [weights.data[i][0] for i in range(6)]

        archetypes = [
            # [geom, rough, metal, emiss, density, freq, inst_scale, hp]
            [0.85, 0.25, 0.88, 3.2, 7.8, 14.5, 256.0, float(VITAL_MAX_HP)],   # CNC_TOOLING
            [0.40, 0.70, 0.45, 1.2, 8.5, 2.5, 128.0, float(VITAL_MAX_HP)],    # THERMAL_MATERIAL
            [0.92, 0.15, 0.30, 4.5, 2.3, 30.0, 512.0, float(VITAL_MAX_HP)],   # COMPUTE_NODE
            [0.65, 0.45, 0.75, 5.0, 4.2, 8.0, 64.0, float(VITAL_MAX_HP)],     # SKEUOMORPHIC_RELIC
            [0.95, 0.60, 0.10, 2.0, 1.8, 1.2, 1024.0, float(VITAL_MAX_HP)],   # BIOME_GEOMETRY
            [0.78, 0.35, 0.65, 6.5, 3.1, 20.0, 2048.0, float(VITAL_MAX_HP)]   # NEURAL_TENSOR_CORE
        ]

        for i in range(6):
            w = w_list[i]
            row = []
            for j in range(7):
                row.append(round(archetypes[i][j] * (0.5 + 0.5 * w), 3))
            # Vital HP is invariant: always 6
            row.append(float(VITAL_MAX_HP))
            out_rows.append(row)

        return TensorMatrix(out_rows, name="T_engine_modulation")


# ── 3. ASSET & TEMPLATE GENERATORS (LOCAL PIPELINE) ─────────────────────────

class LocalAssetAndTemplateCompiler:
    """
    Compiles production-grade engine assets and system templates
    directly from prediction tensors using pure local pipelines.
    """

    @classmethod
    def generate_godot_multimesh_binary_buffer(
        cls,
        weights: TensorMatrix,
        instance_count: int = 64
    ) -> bytes:
        """
        Generates 96-byte aligned binary MultiMesh instance buffer (16 transform + 4 color + 4 custom).
        Compatible with Godot 4.x Forward+ Vulkan pipeline.
        """
        buf = bytearray()
        w_vals = [weights.data[i][0] for i in range(6)]
        dominant_idx = w_vals.index(max(w_vals))

        rng = random.Random(int(sum(w_vals) * 100000))

        for i in range(instance_count):
            # Fermat phyllotaxis spiral coordinates
            theta = i * (math.pi * (3.0 - math.sqrt(5.0)))  # Golden angle in radians
            r = math.sqrt(i) * 3.5

            x = float(r * math.cos(theta))
            y = float(math.sin(i * 0.25) * 1.5)
            z = float(r * math.sin(theta))

            scale = float(0.8 + rng.uniform(-0.15, 0.25))

            # 4x4 Transform matrix in column-major float32
            transform = [
                scale, 0.0, 0.0, 0.0,
                0.0, scale, 0.0, 0.0,
                0.0, 0.0, scale, 0.0,
                x, y, z, 1.0
            ]

            # Harmonic Color RGBA
            color_tensor = HarmonicColorTensor.from_seed(dominant_idx * 1000 + i)
            rf, gf, bf = [c / 255.0 for c in color_tensor.rgb_tuple]
            color_rgba = [rf, gf, bf, 1.0]

            # Custom Data: [Vital_HP, Dominant_Weight, Entropy_Index, Shader_Phase]
            custom_data = [float(VITAL_MAX_HP), w_vals[dominant_idx], 0.15, float(i % 16)]

            # Pack exactly 24 Little-Endian Float32 values = 96 Bytes
            instance_floats = transform + color_rgba + custom_data
            buf.extend(struct.pack("<24f", *instance_floats))

        return bytes(buf)

    @classmethod
    def generate_godot_scene_template(
        cls,
        name: str,
        weights: TensorMatrix,
        instance_count: int = 64
    ) -> str:
        """Generates a complete, syntactically valid Godot 4.x .tscn scene template."""
        w_vals = [weights.data[i][0] for i in range(6)]
        dominant_archetype = DOMAINS[w_vals.index(max(w_vals))]
        color = HarmonicColorTensor.from_seed(42)
        rf, gf, bf = [c / 255.0 for c in color.rgb_tuple]

        return f"""[gd_scene load_steps=5 format=3 uid="uid://krystal_auto_{name.lower()}"]

# ==============================================================================
# KRYSTAL-STACK: PROCEDURALLY GENERATED SCENE TEMPLATE
# Target: Godot 4.x Forward+ Vulkan Renderer
# Invariant: VITAL_MAX_HP = 6
# Linguistic Focus: {dominant_archetype}
# ==============================================================================

[sub_resource type="StandardMaterial3D" id="Mat_{name}"]
albedo_color = Color({rf:.3f}, {gf:.3f}, {bf:.3f}, 1.0)
metallic = {min(0.95, max(0.1, w_vals[0] + w_vals[2])):.2f}
roughness = {max(0.05, 1.0 - w_vals[1]):.2f}
emission_enabled = true
emission = Color({rf * 0.8:.3f}, {gf * 0.8:.3f}, {bf * 0.8:.3f}, 1.0)
emission_energy_multiplier = {2.5 * w_vals[3] + 1.0:.2f}

[sub_resource type="BoxMesh" id="Mesh_{name}"]
material = SubResource("Mat_{name}")
size = Vector3(1.2, 2.4, 1.2)

[node name="{name}" type="Node3D"]

[node name="MultiMesh_{name}" type="MultiMeshInstance3D" parent="."]
# Binary buffer instances: {instance_count} (96 bytes/instance, L2 cache aligned)

[node name="Camera3D" type="Camera3D" parent="."]
transform = Transform3D(1, 0, 0, 0, 0.866, 0.5, 0, -0.5, 0.866, 0, 15, 25)
current = true

[node name="DirectionalLight3D" type="DirectionalLight3D" parent="."]
transform = Transform3D(0.707, -0.5, 0.5, 0, 0.707, 0.707, -0.707, -0.5, 0.5, 10, 20, 10)
light_color = Color(1.0, 0.95, 0.85, 1.0)
light_energy = 1.4
"""

    @classmethod
    def generate_janet_dsl_template(
        cls,
        name: str,
        weights: TensorMatrix
    ) -> str:
        """Generates an authoritative Janet S-Expression template with linguistic priors."""
        w_vals = [round(weights.data[i][0], 4) for i in range(6)]
        return f"""# ==============================================================================
# KRYSTAL-STACK: JANET DSL SYSTEM TEMPLATE
# Module: krystal_janet/{name.lower()}_spec.janet
# Invariant: VITAL-MAX-HP = 6
# ==============================================================================

(def VITAL-MAX-HP 6)

(def SYSTEM-PIPELINE-METRICS
  @{{:name "{name}"
    :timestamp {time.time()}
    :linguistic-usage-weights
    @{{:cnc-machining          {w_vals[0]}
      :thermodynamics          {w_vals[1]}
      :software-architecture   {w_vals[2]}
      :alchemical-lore         {w_vals[3]}
      :spatial-geometry        {w_vals[4]}
      :neural-tensor-core      {w_vals[5]}}}
    :hardware-affinity
    @{{:cache-line-bytes 64
      :instance-stride-bytes 96
      :uma-fast-path true}}
    :vital-health-guard
    @{{:current-hp 6
      :vital-max-hp VITAL-MAX-HP}}}})
"""

    @classmethod
    def generate_keras_architecture_template(
        cls,
        name: str,
        weights: TensorMatrix
    ) -> Dict[str, Any]:
        """Generates a Keras 3 Functional API graph JSON specification."""
        w_vals = [round(weights.data[i][0], 4) for i in range(6)]
        return {
            "model_type": "Keras_3_Functional_Template",
            "model_name": f"Krystal_{name}_TensorPredictorNet",
            "vital_max_hp": VITAL_MAX_HP,
            "linguistic_priors": {DOMAINS[i]: w_vals[i] for i in range(6)},
            "inputs": [
                {"name": "native_language_feature_input", "shape": [None, 10], "dtype": "float32"}
            ],
            "backbone": [
                {"type": "Dense", "units": 64, "activation": "gelu"},
                {"type": "LayerNormalization"},
                {"type": "Dense", "units": 32, "activation": "leaky_relu"}
            ],
            "heads": [
                {"name": "archetype_usage_logits", "units": 6, "activation": "softmax"},
                {"name": "vital_hp_distribution", "units": 7, "activation": "softmax"}
            ]
        }

    @classmethod
    def generate_html_studio_template(
        cls,
        name: str,
        weights: TensorMatrix
    ) -> str:
        """Generates an HTML5 Interactive Studio template with canvas and 6 Max HP HUD."""
        w_vals = [round(weights.data[i][0], 3) for i in range(6)]
        return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Krystal Studio: {name}</title>
  <style>
    body {{ margin: 0; background: #0a0a0f; color: #e2e8f0; font-family: monospace; overflow: hidden; }}
    #hud {{ position: absolute; top: 12px; left: 16px; background: rgba(15,15,22,0.85); padding: 12px 18px; border: 1px solid #00f5d4; border-radius: 6px; }}
    .hp-pip {{ display: inline-block; width: 14px; height: 14px; background: #00f5d4; margin-right: 4px; border-radius: 2px; }}
  </style>
</head>
<body>
  <div id="hud">
    <h3>{name} // Linguistic Tensor Studio</h3>
    <div>VITAL HP: <span class="hp-pip"></span><span class="hp-pip"></span><span class="hp-pip"></span><span class="hp-pip"></span><span class="hp-pip"></span><span class="hp-pip"></span> (6/6)</div>
    <div style="margin-top:6px; font-size:11px;">CNC: {w_vals[0]} | PHYS: {w_vals[1]} | COMP: {w_vals[2]} | ALCH: {w_vals[3]} | GEO: {w_vals[4]} | NET: {w_vals[5]}</div>
  </div>
  <canvas id="view" width="800" height="600" style="width:100vw; height:100vh;"></canvas>
  <script>
    const cv = document.getElementById("view");
    const ctx = cv.getContext("2d");
    function render() {{
      ctx.fillStyle = "#0a0a0f";
      ctx.fillRect(0, 0, cv.width, cv.height);
      ctx.strokeStyle = "#00f5d4";
      ctx.beginPath();
      ctx.arc(cv.width/2, cv.height/2, 120, 0, Math.PI * 2);
      ctx.stroke();
    }}
    render();
  </script>
</body>
</html>"""


# ── 4. END-TO-END PIPELINE ORCHESTRATOR ─────────────────────────────────────

@dataclass
class NativeLanguagePipelineResult:
    """Consolidated package containing predicted tensors, assets, and templates."""
    status: str
    prompt: str
    detected_language: str
    feature_vector: Dict[str, Any]
    usage_prediction_tensor: List[List[float]]
    average_usage_tensor: List[List[float]]
    engine_modulation_tensor: List[List[float]]
    binary_multimesh_bytes: int
    godot_scene_template: str
    janet_dsl_template: str
    keras_model_template: Dict[str, Any]
    html_studio_template: str
    execution_time_ms: float
    vital_hp: int = VITAL_MAX_HP
    vital_max_hp: int = VITAL_MAX_HP

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class NativeLanguageTensorPipeline:
    """
    Main entry point coordinating the complete native language tensor generator
    and local asset/template compilation pipeline.
    """

    def __init__(self):
        self.predictor = NativeLanguageTensorPredictor()
        self.compiler = LocalAssetAndTemplateCompiler()

    def process_prompt(
        self,
        prompt: str,
        name: str = "ProceduralConstruct",
        instance_count: int = 64
    ) -> NativeLanguagePipelineResult:
        """Executes the complete local prediction and asset generation pipeline."""
        t0 = time.time()

        # 1. Feature extraction
        features = self.predictor.extract_features(prompt)

        # 2. Tensor predictions
        w_pred = self.predictor.predict_usage_tensor(features)
        w_avg = self.predictor.compute_average_usage_tensor()
        t_mod = self.predictor.synthesize_engine_modulation_tensor(w_pred)

        # 3. Engine Assets synthesis
        multimesh_bin = self.compiler.generate_godot_multimesh_binary_buffer(w_pred, instance_count=instance_count)

        # 4. System Templates generation
        tscn_code = self.compiler.generate_godot_scene_template(name, w_pred, instance_count=instance_count)
        janet_code = self.compiler.generate_janet_dsl_template(name, w_pred)
        keras_spec = self.compiler.generate_keras_architecture_template(name, w_pred)
        html_code = self.compiler.generate_html_studio_template(name, w_pred)

        latency_ms = round((time.time() - t0) * 1000.0, 2)

        return NativeLanguagePipelineResult(
            status="SUCCESS",
            prompt=prompt,
            detected_language=features.language_code,
            feature_vector=asdict(features),
            usage_prediction_tensor=w_pred.to_list(),
            average_usage_tensor=w_avg.to_list(),
            engine_modulation_tensor=t_mod.to_list(),
            binary_multimesh_bytes=len(multimesh_bin),
            godot_scene_template=tscn_code,
            janet_dsl_template=janet_code,
            keras_model_template=keras_spec,
            html_studio_template=html_code,
            execution_time_ms=latency_ms,
            vital_hp=VITAL_MAX_HP,
            vital_max_hp=VITAL_MAX_HP
        )


# Global singleton pipeline instance
GLOBAL_NATIVE_TENSOR_PIPELINE = NativeLanguageTensorPipeline()
