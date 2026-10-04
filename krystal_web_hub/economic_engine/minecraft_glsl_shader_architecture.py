"""
Krystal Stack: Minecraft GLSL Shader Architecture & Godot 4.x Graphics Engine
=============================================================================
Provides data models, LabPBR 1.3 decoders, deferred pass pipeline simulation,
and shader pack profiles inspired by BSL, Complementary, and SEUS PTGI.

Invariants:
- VITAL_MAX_HP = 6
- GOLDEN_RATIO = 1.61803398875
"""

import math
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

GOLDEN_RATIO: float = 1.61803398875
INV_GOLDEN_RATIO: float = 1.0 / GOLDEN_RATIO
VITAL_MAX_HP: int = 6


@dataclass
class ShaderPackProfile:
    id: str
    name: str
    author: str
    style_category: str
    features: List[str]
    labpbr_version: str
    default_fog_density: float
    target_fps_iris_xe: int
    hp_cost: int = 1
    vital_max_hp: int = VITAL_MAX_HP

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "author": self.author,
            "style_category": self.style_category,
            "features": self.features,
            "labpbr_version": self.labpbr_version,
            "default_fog_density": self.default_fog_density,
            "target_fps_iris_xe": self.target_fps_iris_xe,
            "hp_cost": min(self.hp_cost, VITAL_MAX_HP),
            "vital_max_hp": VITAL_MAX_HP,
        }


@dataclass
class DeferredPassStage:
    id: str
    name: str
    description: str
    vram_cost_mb: float
    gpu_time_ms_iris_xe: float
    is_compute_pass: bool
    vital_max_hp: int = VITAL_MAX_HP

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "vram_cost_mb": round(self.vram_cost_mb, 2),
            "gpu_time_ms_iris_xe": round(self.gpu_time_ms_iris_xe, 2),
            "is_compute_pass": self.is_compute_pass,
            "vital_max_hp": VITAL_MAX_HP,
        }


@dataclass
class GodotPluginRecommendation:
    id: str
    name: str
    category: str
    purpose: str
    github_or_asset_lib: str
    iris_xe_benefit: str
    vital_max_hp: int = VITAL_MAX_HP

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "category": self.category,
            "purpose": self.purpose,
            "github_or_asset_lib": self.github_or_asset_lib,
            "iris_xe_benefit": self.iris_xe_benefit,
            "vital_max_hp": VITAL_MAX_HP,
        }


# Canonical Shader Pack Catalog
CANONICAL_SHADER_PACKS: List[ShaderPackProfile] = [
    ShaderPackProfile(
        id="bsl_v8",
        name="BSL Shaders v8.2",
        author="Capt Tatsu",
        style_category="Stylized Cinematic",
        features=[
            "Customizable Bloom & Cel-shading",
            "Volumetric Light Rays",
            "Real-time Depth of Field",
            "Motion Blur & TAA",
            "Foliage Wind Physics",
            "LabPBR 1.2 Support",
        ],
        labpbr_version="1.2",
        default_fog_density=0.015,
        target_fps_iris_xe=58,
        hp_cost=1,
    ),
    ShaderPackProfile(
        id="complementary_reimagined",
        name="Complementary Reimagined",
        author="EminGT",
        style_category="Authentic PBR Reimagined",
        features=[
            "Full LabPBR 1.3 Specular & Normal Decode",
            "Parallax Occlusion Mapping with Self-Shadowing",
            "Emerald Multi-Octave Gerstner Water",
            "Raymarched 3D Worley-Perlin Clouds",
            "Henyey-Greenstein Forward Light Scattering",
            "Dynamic Weather Wetness & Puddles",
        ],
        labpbr_version="1.3",
        default_fog_density=0.018,
        target_fps_iris_xe=54,
        hp_cost=2,
    ),
    ShaderPackProfile(
        id="seus_ptgi_hrr",
        name="SEUS PTGI HRR (Path Traced GI)",
        author="Sonic Ether",
        style_category="Photorealistic Voxel Ray Tracing",
        features=[
            "Software Voxel Path Tracing (No RTX required)",
            "DDA Raymarching in 3D Clipmap",
            "Colored Diffuse Light Bounces (Indirect GI)",
            "Screen-Space & Planar Reflections",
            "Spatio-Temporal Blue Noise Filtering",
            "LabPBR 1.3 Hardcoded Metals",
        ],
        labpbr_version="1.3",
        default_fog_density=0.012,
        target_fps_iris_xe=38,
        hp_cost=3,
    ),
    ShaderPackProfile(
        id="continuum_cinematic",
        name="Continuum 2.0 RT",
        author="Continuum Graphics",
        style_category="Hollywood CGI Physical Camera",
        features=[
            "Physical Camera Optics (F-Stop, Shutter, ISO)",
            "Full Stratospheric Atmospheric Scattering",
            "Beer-Lambert Underwater Light Extinction",
            "Subsurface Scattering (SSS) on Leaves & Organic Slabs",
            "High-Precision Parallax Displacement",
        ],
        labpbr_version="1.3",
        default_fog_density=0.022,
        target_fps_iris_xe=32,
        hp_cost=4,
    ),
    ShaderPackProfile(
        id="iris_vanilla_plus",
        name="Iris Vanilla+ Ultralight",
        author="Krystal Stack Team",
        style_category="Optimized Esports Performance",
        features=[
            "Bilateral Guided Volumetric Fog",
            "Screen-Space Ambient Occlusion (GTAO)",
            "Fast Kawase Dual-Blur Bloom",
            "Dynamic Water Waves & Shore Foam",
            "Low-Overhead Iris Xe Whisperer Thread Offload",
        ],
        labpbr_version="1.3",
        default_fog_density=0.008,
        target_fps_iris_xe=75,
        hp_cost=1,
    ),
]

# Canonical Deferred Pass Pipeline
CANONICAL_DEFERRED_STAGES: List[DeferredPassStage] = [
    DeferredPassStage(
        id="gbuffers_terrain",
        name="G-Buffers Terrain & Foliage",
        description="Encodes Albedo, Normal XY, Specular Smoothness, F0, Porosity and Lightmap",
        vram_cost_mb=64.0,
        gpu_time_ms_iris_xe=2.8,
        is_compute_pass=False,
    ),
    DeferredPassStage(
        id="gbuffers_water",
        name="G-Buffers Water Surface",
        description="Calculates Gerstner wave displacement, normal pertubation, and depth offset",
        vram_cost_mb=32.0,
        gpu_time_ms_iris_xe=1.2,
        is_compute_pass=False,
    ),
    DeferredPassStage(
        id="shadow_cascades",
        name="Cascaded Shadow Maps (CSM)",
        description="Renders 4-tier cascaded depth maps with warped near-field player bias",
        vram_cost_mb=48.0,
        gpu_time_ms_iris_xe=3.1,
        is_compute_pass=False,
    ),
    DeferredPassStage(
        id="voxel_gi_dda",
        name="3D Voxel GI Path Tracing",
        description="Compute shader voxel DDA traversal for indirect colored diffuse bounce lighting",
        vram_cost_mb=96.0,
        gpu_time_ms_iris_xe=4.5,
        is_compute_pass=True,
    ),
    DeferredPassStage(
        id="volumetric_fog_godrays",
        name="Volumetric Fog & Crepuscular Rays",
        description="Raymarches shadow cascade in view-space with bilateral upsampling",
        vram_cost_mb=24.0,
        gpu_time_ms_iris_xe=2.1,
        is_compute_pass=True,
    ),
    DeferredPassStage(
        id="screen_space_reflections",
        name="Hi-Z Screen-Space Reflections (SSR)",
        description="Cone-traced raymarching on depth buffer with roughness fallback",
        vram_cost_mb=32.0,
        gpu_time_ms_iris_xe=2.4,
        is_compute_pass=False,
    ),
    DeferredPassStage(
        id="post_process_composite",
        name="Composite, Dual-Bloom & ACES Tonemapping",
        description="Kawase pyramid blur, ACES filmic curve, TAA sub-pixel jitter resolve",
        vram_cost_mb=16.0,
        gpu_time_ms_iris_xe=1.6,
        is_compute_pass=False,
    ),
]

# Recommended Godot 4.x Plugins
CANONICAL_GODOT_PLUGINS: List[GodotPluginRecommendation] = [
    GodotPluginRecommendation(
        id="terrain_3d",
        name="Terrain3D (GDExtension)",
        category="World Generation & Terrains",
        purpose="High-performance Clipmap/CDLOD chunk terrain renderer with GPU splatting and LabPBR support.",
        github_or_asset_lib="https://github.com/TokisanGames/Terrain3D",
        iris_xe_benefit="Uses GPU compute tessellation and clipmap streaming, eliminating CPU draw-call bottleneck on Intel Iris Xe.",
    ),
    GodotPluginRecommendation(
        id="phantom_camera",
        name="Phantom Camera (Cinematic Rig)",
        category="Gameplay & Cameras",
        purpose="Cinemachine-inspired dynamic tracking, framing, spring arms, and smooth transitions.",
        github_or_asset_lib="https://github.com/ramok/phantom-camera",
        iris_xe_benefit="Zero-rendering overhead; delivers AAA camera control and visceral tweening.",
    ),
    GodotPluginRecommendation(
        id="zylann_voxel",
        name="Godot Voxel Tools (Zylann)",
        category="Voxel Engines & Destruction",
        purpose="Smooth & Blocky voxel streaming with Transvoxel LOD meshing and direct collision generation.",
        github_or_asset_lib="https://github.com/Zylann/godot_voxel",
        iris_xe_benefit="Native C++ threaded meshing allows Minecraft-style chunk modification at 60 FPS.",
    ),
    GodotPluginRecommendation(
        id="limbo_ai",
        name="LimboAI (Behavior Trees & HSM)",
        category="Artificial Intelligence & NPCs",
        purpose="Visual behavior tree editor and Hierarchical State Machine for creature, boss, and tribe logic.",
        github_or_asset_lib="https://github.com/limbonaut/limboai",
        iris_xe_benefit="Extremely lightweight GDExtension execution, leaving GPU cores 100% free for shaders.",
    ),
    GodotPluginRecommendation(
        id="compute_particles",
        name="Godot 4 GPUParticles3D Sub-Emitters",
        category="VFX & Atmospheric Motes",
        purpose="Handles millions of dust motes, pollen, embers, and water splashes using Vulkan compute.",
        github_or_asset_lib="Godot Built-in Core",
        iris_xe_benefit="Direct hardware compute acceleration on Intel Gen12 EU threads without RAM stall.",
    ),
]


class MinecraftGLSLShaderArchitectureEngine:
    """Core logic engine for Minecraft shader pack emulation, LabPBR decoding, and pipeline metrics."""

    def __init__(self):
        self.shader_packs = {p.id: p for p in CANONICAL_SHADER_PACKS}
        self.deferred_stages = {s.id: s for s in CANONICAL_DEFERRED_STAGES}
        self.godot_plugins = {pl.id: pl for pl in CANONICAL_GODOT_PLUGINS}

    def decode_labpbr_pixel(
        self,
        specular_r: float,
        specular_g: float,
        specular_b: float,
        specular_a: float,
        normal_r: float,
        normal_g: float,
        normal_b: float,
        normal_a: float,
    ) -> Dict[str, Any]:
        """
        Decodes raw texture pixel values according to LabPBR 1.3 specification.
        Inputs: normalized floats 0.0 to 1.0.
        """
        # 1. Specular R: Perceptual Smoothness -> Linear Roughness
        smoothness = max(0.0, min(1.0, specular_r))
        roughness = math.pow(1.0 - smoothness, 2.0)

        # 2. Specular G: F0 & Metals
        # LabPBR: 0..228 = Dielectric, 229..255 = Hardcoded Metal
        raw_g_byte = int(specular_g * 255.0)
        is_metal = raw_g_byte >= 229
        f0_reflectance = 1.0 if is_metal else (raw_g_byte / 228.0) * 0.08

        # 3. Specular B: Porosity
        porosity = max(0.0, min(1.0, specular_b))

        # 4. Specular A: Emission
        raw_a_byte = int(specular_a * 255.0)
        is_emissive = raw_a_byte < 255
        emissive_intensity = (raw_a_byte / 254.0) * 5.0 if is_emissive else 0.0

        # 5. Normal RG: Tangent Normal Vector
        norm_x = (normal_r * 2.0) - 1.0
        norm_y = (normal_g * 2.0) - 1.0
        norm_z = math.sqrt(max(0.0, 1.0 - (norm_x * norm_x + norm_y * norm_y)))

        # 6. Normal B: Ambient Occlusion
        ao = max(0.0, min(1.0, normal_b))

        # 7. Normal A: Height / Parallax Occlusion Depth
        height = max(0.0, min(1.0, normal_a))

        return {
            "smoothness": round(smoothness, 4),
            "linear_roughness": round(roughness, 4),
            "is_metal": is_metal,
            "f0_reflectance": round(f0_reflectance, 4),
            "porosity": round(porosity, 4),
            "is_emissive": is_emissive,
            "emissive_intensity": round(emissive_intensity, 4),
            "reconstructed_normal": [round(norm_x, 4), round(norm_y, 4), round(norm_z, 4)],
            "ambient_occlusion": round(ao, 4),
            "parallax_height": round(height, 4),
            "vital_max_hp": VITAL_MAX_HP,
        }

    def simulate_pipeline_run(
        self,
        profile_id: str = "complementary_reimagined",
        resolution_width: int = 1920,
        resolution_height: int = 1080,
    ) -> Dict[str, Any]:
        """
        Simulates the execution of the entire deferred rendering pipeline.
        Calculates total VRAM, frame time, projected FPS, and power envelope.
        """
        profile = self.shader_packs.get(profile_id, self.shader_packs["complementary_reimagined"])

        pixel_count = resolution_width * resolution_height
        base_pixels = 1920 * 1080
        res_scale = pixel_count / float(base_pixels)

        stages_breakdown = []
        total_gpu_time = 0.0
        total_vram = 0.0

        # Adjust stages according to profile characteristics
        for stage in CANONICAL_DEFERRED_STAGES:
            stage_time = stage.gpu_time_ms_iris_xe * res_scale
            stage_vram = stage.vram_cost_mb * res_scale

            # Profile specific multipliers
            if profile.id == "seus_ptgi_hrr" and stage.id == "voxel_gi_dda":
                stage_time *= 1.45
                stage_vram *= 1.25
            elif profile.id == "iris_vanilla_plus" and stage.id == "voxel_gi_dda":
                # Vanilla+ skips full voxel path tracing in favor of GTAO
                stage_time *= 0.15
                stage_vram *= 0.10

            stages_breakdown.append({
                "stage_id": stage.id,
                "name": stage.name,
                "gpu_time_ms": round(stage_time, 2),
                "vram_mb": round(stage_vram, 2),
                "is_compute": stage.is_compute_pass,
            })
            total_gpu_time += stage_time
            total_vram += stage_vram

        # Calculate Projected FPS
        projected_fps = int(1000.0 / max(total_gpu_time, 1.0))
        
        # Calculate Golden Mean Efficiency Score
        ideal_frame_time_60fps = 16.666
        efficiency_score = round(min(1.0, ideal_frame_time_60fps / max(total_gpu_time, 1.0)) * GOLDEN_RATIO, 3)

        return {
            "profile": profile.to_dict(),
            "resolution": f"{resolution_width}x{resolution_height}",
            "total_gpu_time_ms": round(total_gpu_time, 2),
            "total_vram_mb": round(total_vram, 2),
            "projected_fps": projected_fps,
            "target_fps_target": profile.target_fps_iris_xe,
            "efficiency_score": efficiency_score,
            "stages": stages_breakdown,
            "vital_max_hp_rule": VITAL_MAX_HP,
        }

    def get_full_catalog(self) -> Dict[str, Any]:
        """Returns all shader packs, stages, and Godot plugins."""
        return {
            "shader_packs": [p.to_dict() for p in self.shader_packs.values()],
            "deferred_stages": [s.to_dict() for s in self.deferred_stages.values()],
            "godot_plugins": [pl.to_dict() for pl in self.godot_plugins.values()],
            "vital_max_hp": VITAL_MAX_HP,
            "golden_ratio": GOLDEN_RATIO,
        }


# Global singleton
GLOBAL_MINECRAFT_SHADER_ENGINE = MinecraftGLSLShaderArchitectureEngine()
