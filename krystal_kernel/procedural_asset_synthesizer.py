#!/usr/bin/env python3
"""
==============================================================================
KRYSTAL-STACK: PROCEDURAL ASSET SYNTHESIZER
==============================================================================
Module: krystal_kernel/procedural_asset_synthesizer.py
Description: Generates a deterministic, random selection of unique procedural
             assets that fundamentally depend on Krystal Stack content:
             1. CNC Smart Cutting Tools (coupled with thermal dynamics & SSV).
             2. Material Substrates (coupled with Loewen-Shaw cutting physics).
             3. Skeuomorphic Cyber-Relics (coupled with durability pips).
             4. Computational Microarchitecture Nodes (coupled with IPC & cache).
             5. Procedural Biome Constructs (coupled with Fermat / Gielis math).
             
             Preserves platform invariant: VITAL_MAX_HP == 6.

Author: Dušan Kopecký & Krystal-Stack Architecture Council (2026)
==============================================================================
"""

import os
import sys
import math
import time
import random
from enum import Enum
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Optional, Tuple

WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

from krystal_kernel.scene_constructor_presets import VITAL_MAX_HP
from krystal_kernel.cnc_adaptive_thermal_governor import MATERIAL_CATALOG, WorkpieceMaterialProps

GOLDEN_RATIO = 1.61803398875
GOLDEN_ANGLE_DEG = 137.507764


class ProceduralAssetType(str, Enum):
    CNC_SMART_TOOL = "CNC_SMART_TOOL"
    MATERIAL_SUBSTRATE = "MATERIAL_SUBSTRATE"
    SKEUOMORPHIC_ARTIFACT = "SKEUOMORPHIC_ARTIFACT"
    COMPUTATIONAL_KERNEL_NODE = "COMPUTATIONAL_KERNEL_NODE"
    PROCEDURAL_BIOME_CONSTRUCT = "PROCEDURAL_BIOME_CONSTRUCT"


@dataclass
class HarmonicColorTensor:
    """Perceptually uniform color tensor derived via golden ratio harmonic spacing."""
    hue_deg: float
    saturation_pct: float
    lightness_pct: float
    rgb_hex: str
    rgb_tuple: Tuple[int, int, int]

    @classmethod
    def from_seed(cls, seed: int, sat_range: Tuple[float, float] = (75.0, 95.0),
                  light_range: Tuple[float, float] = (50.0, 70.0)) -> "HarmonicColorTensor":
        rng = random.Random(seed)
        hue = (seed * GOLDEN_ANGLE_DEG) % 360.0
        sat = rng.uniform(*sat_range)
        light = rng.uniform(*light_range)
        
        # Convert HSL to RGB
        c = (1.0 - abs(2.0 * (light / 100.0) - 1.0)) * (sat / 100.0)
        x = c * (1.0 - abs(((hue / 60.0) % 2.0) - 1.0))
        m = (light / 100.0) - c / 2.0
        
        h_prime = hue / 60.0
        if 0 <= h_prime < 1:
            r1, g1, b1 = c, x, 0.0
        elif 1 <= h_prime < 2:
            r1, g1, b1 = x, c, 0.0
        elif 2 <= h_prime < 3:
            r1, g1, b1 = 0.0, c, x
        elif 3 <= h_prime < 4:
            r1, g1, b1 = 0.0, x, c
        elif 4 <= h_prime < 5:
            r1, g1, b1 = x, 0.0, c
        else:
            r1, g1, b1 = c, 0.0, x
            
        r = int(round((r1 + m) * 255.0))
        g = int(round((g1 + m) * 255.0))
        b = int(round((b1 + m) * 255.0))
        rgb_hex = f"#{r:02x}{g:02x}{b:02x}"
        return cls(round(hue, 2), round(sat, 1), round(light, 1), rgb_hex, (r, g, b))


@dataclass
class ProceduralAsset:
    """Standardized multi-domain asset generated procedurally from Krystal Stack."""
    asset_id: str
    asset_type: ProceduralAssetType
    name: str
    seed: int
    properties: Dict[str, Any]
    color_tensor: Dict[str, Any]
    vital_hp: int
    vital_max_hp: int = VITAL_MAX_HP
    stack_dependencies: Dict[str, Any] = field(default_factory=dict)
    visual_descriptor: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ProceduralAssetSynthesizer:
    """
    Synthesizes unique procedural assets grounded in the actual physics,
    state-space observers, and runtime characteristics of Krystal Stack.
    """

    def __init__(self):
        self.tool_geometries = [
            ("Helical Endmill", 4, 30.0, "AlTiN_Nanocomposite"),
            ("Ballnose Profiler", 2, 45.0, "Diamond_PCD"),
            ("Bullnose Rougher", 5, 38.0, "TiAlN_MultiLayer"),
            ("Octa-Flute Chamfer", 8, 45.0, "TiCN_Thermal"),
            ("High-Feed Facemill", 6, 20.0, "CVD_Al2O3_Titanium")
        ]

        self.relic_titles = [
            ("Chrono-Harmonic Compass", "Temporal alignment of Vulkan frames"),
            ("Galvanic Alchemical Core", "Electrolyte charge buffer for thermal breathers"),
            ("Tachyon Memory Scriptor", "Zero-alloc cache-line ring buffer conduit"),
            ("Obsidian Canary Ward", "Anti-corruption sentinel for memory stability"),
            ("Aetheric Spindle Dynamo", "Piezo-electric chatter damper matrix")
        ]

        self.compute_nodes = [
            ("HyperVector-AVX512 Tile", 64, 4.2, "AVX-512 / FMA3"),
            ("Asynchronous Ring Arbiter", 16, 1.8, "L1d Cache Pipeline"),
            ("Symplectic Phase Predictor", 32, 2.5, "Float32 Tensor Contraction"),
            ("Zero-Overdraw Occlusion Gate", 8, 0.9, "UMA Unified Memory Lane")
        ]

    def synthesize_cnc_tool(self, seed: int, live_temp_c: float = 220.0) -> ProceduralAsset:
        """Synthesizes a CNC Smart Tool asset coupled with cutting thermal dynamics."""
        rng = random.Random(seed)
        geom_name, flutes, helix, coating = rng.choice(self.tool_geometries)
        diameter_mm = rng.choice([3.0, 4.0, 6.0, 8.0, 10.0, 12.0, 16.0])
        
        # Thermally conditioned tool wear VB and health
        thermal_strain = max(0.0, (live_temp_c - 200.0) / 500.0)
        wear_vb_mm = round(0.015 + rng.uniform(0.02, 0.15) + thermal_strain * 0.12, 4)
        
        # Invariant HP: degrades with thermal strain and wear, clamped to [1, 6]
        if wear_vb_mm < 0.08 and live_temp_c < 350.0:
            hp = 6
        elif wear_vb_mm < 0.15 and live_temp_c < 480.0:
            hp = 5
        elif wear_vb_mm < 0.22 and live_temp_c < 600.0:
            hp = 4
        elif wear_vb_mm < 0.30:
            hp = 3
        elif wear_vb_mm < 0.40:
            hp = 2
        else:
            hp = 1

        color = HarmonicColorTensor.from_seed(seed + 101, sat_range=(80, 100), light_range=(45, 65))
        
        # Procedural visual descriptor for image generator
        visual = {
            "archetype": "CUTTING_TOOL",
            "primary_shape": "cylinder_spiral",
            "diameter_pixels": int(diameter_mm * 2.5),
            "flute_spirals": flutes,
            "helix_angle_deg": helix,
            "coating_tint": color.rgb_hex,
            "heat_bluing_active": live_temp_c > 450.0,
            "glow_intensity": min(1.0, thermal_strain),
            "hud_chips": [
                f"D{diameter_mm:.1f}mm",
                f"{flutes}F",
                f"VB={wear_vb_mm:.3f}",
                f"{live_temp_c:.0f}°C"
            ]
        }

        return ProceduralAsset(
            asset_id=f"CNC-TOOL-{seed:06x}",
            asset_type=ProceduralAssetType.CNC_SMART_TOOL,
            name=f"{coating} {geom_name} Ø{diameter_mm}mm",
            seed=seed,
            properties={
                "diameter_mm": diameter_mm,
                "flute_count": flutes,
                "helix_angle_deg": helix,
                "coating": coating,
                "flank_wear_vb_mm": wear_vb_mm,
                "max_operating_temp_c": 750.0,
                "recommended_rpm_range": [8000, 18000],
                "ssv_wobble_amplitude_pct": 14.0
            },
            color_tensor=asdict(color),
            vital_hp=min(VITAL_MAX_HP, max(1, hp)),
            vital_max_hp=VITAL_MAX_HP,
            stack_dependencies={
                "coupled_module": "krystal_kernel.cnc_adaptive_thermal_governor",
                "thermal_reference_temp_c": live_temp_c,
                "state_variable": "flank_wear_vb",
                "ssv_frequency_hz": 2.2
            },
            visual_descriptor=visual
        )

    def synthesize_material_substrate(self, seed: int) -> ProceduralAsset:
        """Synthesizes a Material Substrate asset from the physical database."""
        rng = random.Random(seed)
        mat_keys = list(MATERIAL_CATALOG.keys())
        chosen_key = rng.choice(mat_keys)
        props: WorkpieceMaterialProps = MATERIAL_CATALOG[chosen_key]

        color = HarmonicColorTensor.from_seed(seed + 202, sat_range=(60, 90), light_range=(40, 70))
        
        # Hardness and machinability score derived from specific cutting force
        machinability_index = round(min(100.0, max(10.0, 250000.0 / (props.specific_cutting_force_kc_mpa + 1e-3))), 1)

        visual = {
            "archetype": "MATERIAL_INGOT",
            "primary_shape": "chamfered_block",
            "roughness": 0.35 if "ALUMINUM" in chosen_key else 0.65,
            "metallic": 0.85 if "STEEL" in chosen_key or "TITANIUM" in chosen_key else 0.40,
            "base_color": color.rgb_hex,
            "grain_frequency": round(props.density_rho / 1000.0, 2),
            "hud_chips": [
                props.name,
                f"kc={props.specific_cutting_force_kc_mpa:.0f}MPa",
                f"k={props.thermal_conductivity_k:.1f}W/mK",
                f"Mach={machinability_index}"
            ]
        }

        return ProceduralAsset(
            asset_id=f"MAT-SUB-{seed:06x}",
            asset_type=ProceduralAssetType.MATERIAL_SUBSTRATE,
            name=f"Substrate {props.name}",
            seed=seed,
            properties={
                "material_key": chosen_key,
                "specific_cutting_force_kc_mpa": props.specific_cutting_force_kc_mpa,
                "thermal_conductivity_k": props.thermal_conductivity_k,
                "density_rho": props.density_rho,
                "specific_heat_cp": props.specific_heat_cp,
                "thermal_diffusivity_alpha": round(props.thermal_diffusivity_alpha, 8),
                "machinability_index": machinability_index
            },
            color_tensor=asdict(color),
            vital_hp=VITAL_MAX_HP,  # Pristine material stock always starts at 6 HP
            vital_max_hp=VITAL_MAX_HP,
            stack_dependencies={
                "coupled_module": "krystal_kernel.cnc_adaptive_thermal_governor.MATERIAL_CATALOG",
                "heat_partition_equation": "Loewen-Shaw & Jaeger Moving Heat Source"
            },
            visual_descriptor=visual
        )

    def synthesize_skeuomorphic_artifact(self, seed: int) -> ProceduralAsset:
        """Synthesizes a Skeuomorphic Cyber-Relic asset with durability pips."""
        rng = random.Random(seed)
        title, lore = rng.choice(self.relic_titles)
        durability_hp = rng.randint(2, VITAL_MAX_HP)
        resonance_hz = round(rng.uniform(432.0, 963.0), 2)
        patina_age_cycles = rng.randint(120, 9800)

        color = HarmonicColorTensor.from_seed(seed + 303, sat_range=(85, 100), light_range=(50, 75))

        visual = {
            "archetype": "RELIC_AMULET",
            "primary_shape": "octagonal_conduit",
            "core_gem_color": color.rgb_hex,
            "gear_teeth": rng.choice([8, 12, 16]),
            "patina_intensity": round(min(1.0, patina_age_cycles / 10000.0), 2),
            "glow_pulse_rate_hz": round(resonance_hz / 200.0, 2),
            "hud_chips": [
                title,
                f"{resonance_hz}Hz",
                f"Durability: {durability_hp}/6 HP"
            ]
        }

        return ProceduralAsset(
            asset_id=f"RELIC-{seed:06x}",
            asset_type=ProceduralAssetType.SKEUOMORPHIC_ARTIFACT,
            name=f"{title} (Mk.{seed % 7 + 1})",
            seed=seed,
            properties={
                "lore_function": lore,
                "resonance_frequency_hz": resonance_hz,
                "patina_age_cycles": patina_age_cycles,
                "socket_type": "KRYSTAL_UMA_64B_ALIGNED"
            },
            color_tensor=asdict(color),
            vital_hp=min(VITAL_MAX_HP, max(1, durability_hp)),
            vital_max_hp=VITAL_MAX_HP,
            stack_dependencies={
                "coupled_module": "krystal_web_hub.economic_engine.SkeuomorphicProceduralEngine",
                "memory_alignment_bytes": 64
            },
            visual_descriptor=visual
        )

    def synthesize_kernel_node(self, seed: int, ipc: float = 1.65, cache_hit: float = 96.5) -> ProceduralAsset:
        """Synthesizes a Computational Microarchitecture Core Node asset."""
        rng = random.Random(seed)
        title, lanes, clock_ghz, isa = rng.choice(self.compute_nodes)
        entropy = round(rng.uniform(0.05, 0.45), 3)
        node_hp = VITAL_MAX_HP if cache_hit > 90.0 and ipc > 1.2 else rng.randint(3, 5)

        color = HarmonicColorTensor.from_seed(seed + 404, sat_range=(75, 95), light_range=(55, 75))

        visual = {
            "archetype": "MICRO_CORE",
            "primary_shape": "silicon_die_grid",
            "lane_count": lanes,
            "trace_color": color.rgb_hex,
            "core_clock_ghz": clock_ghz,
            "cache_saturation": round(cache_hit / 100.0, 2),
            "hud_chips": [
                isa,
                f"IPC={ipc:.2f}",
                f"L2={cache_hit:.1f}%",
                f"{lanes}-wide"
            ]
        }

        return ProceduralAsset(
            asset_id=f"CORE-NODE-{seed:06x}",
            asset_type=ProceduralAssetType.COMPUTATIONAL_KERNEL_NODE,
            name=f"{title} @ {clock_ghz}GHz",
            seed=seed,
            properties={
                "instruction_set": isa,
                "simd_lane_width": lanes,
                "target_clock_ghz": clock_ghz,
                "runtime_ipc": ipc,
                "l2_cache_hit_rate_pct": cache_hit,
                "entropy_score": entropy
            },
            color_tensor=asdict(color),
            vital_hp=min(VITAL_MAX_HP, max(1, node_hp)),
            vital_max_hp=VITAL_MAX_HP,
            stack_dependencies={
                "coupled_module": "krystal_kernel.keras_tensor_math_engine",
                "domain": "Domain B Software Computational Architecture",
                "spectral_radius_bound": 4.0
            },
            visual_descriptor=visual
        )

    def synthesize_biome_construct(self, seed: int) -> ProceduralAsset:
        """Synthesizes a 3D Procedural Biome Construct asset using Fermat/Gielis geometry."""
        rng = random.Random(seed)
        shapes = ["Fermat Phyllotaxis Spires", "Gielis Superformula Monolith", "Voronoi Alchemical Basin", "Hexagonal Bastion"]
        chosen_shape = rng.choice(shapes)
        instance_density = rng.randint(48, 512)
        height_m = round(rng.uniform(15.0, 120.0), 1)

        color = HarmonicColorTensor.from_seed(seed + 505, sat_range=(70, 95), light_range=(45, 65))

        visual = {
            "archetype": "BIOME_STRUCTURE",
            "primary_shape": "phyllotaxis_cluster",
            "structure_height_m": height_m,
            "instance_density": instance_density,
            "harmonic_tint": color.rgb_hex,
            "golden_ratio_applied": True,
            "hud_chips": [
                chosen_shape,
                f"H={height_m}m",
                f"N={instance_density}",
                "Φ=1.618"
            ]
        }

        return ProceduralAsset(
            asset_id=f"BIOME-GEO-{seed:06x}",
            asset_type=ProceduralAssetType.PROCEDURAL_BIOME_CONSTRUCT,
            name=f"{chosen_shape} [Seed {seed % 1000}]",
            seed=seed,
            properties={
                "geometric_class": chosen_shape,
                "height_m": height_m,
                "instance_density": instance_density,
                "golden_spiral_constant": GOLDEN_RATIO,
                "min_z_separation_m": 0.015
            },
            color_tensor=asdict(color),
            vital_hp=VITAL_MAX_HP,
            vital_max_hp=VITAL_MAX_HP,
            stack_dependencies={
                "coupled_module": "krystal_kernel.native_procedural_renderer",
                "pipeline": "Vulkan Forward+ MultiMesh Buffer",
                "struct_stride_bytes": 96
            },
            visual_descriptor=visual
        )

    def generate_random_asset_selection(
        self,
        count: int = 5,
        seed: Optional[int] = None,
        stack_state: Optional[Dict[str, Any]] = None
    ) -> List[ProceduralAsset]:
        """
        Generates a balanced, randomized selection of unique assets across all 5 domains,
        conditioning generation on live stack telemetry if provided.
        """
        actual_seed = seed if seed is not None else int(time.time() * 1000) % 1_000_000
        rng = random.Random(actual_seed)
        
        stack_state = stack_state or {}
        live_temp = float(stack_state.get("tip_temp_c", 220.0))
        live_ipc = float(stack_state.get("ipc", 1.65))
        live_cache = float(stack_state.get("cache_hit_pct", 96.5))

        types_pool = list(ProceduralAssetType)
        rng.shuffle(types_pool)

        assets: List[ProceduralAsset] = []
        for i in range(count):
            asset_seed = (actual_seed + i * 7919) % 10_000_000
            target_type = types_pool[i % len(types_pool)]
            
            if target_type == ProceduralAssetType.CNC_SMART_TOOL:
                assets.append(self.synthesize_cnc_tool(asset_seed, live_temp_c=live_temp))
            elif target_type == ProceduralAssetType.MATERIAL_SUBSTRATE:
                assets.append(self.synthesize_material_substrate(asset_seed))
            elif target_type == ProceduralAssetType.SKEUOMORPHIC_ARTIFACT:
                assets.append(self.synthesize_skeuomorphic_artifact(asset_seed))
            elif target_type == ProceduralAssetType.COMPUTATIONAL_KERNEL_NODE:
                assets.append(self.synthesize_kernel_node(asset_seed, ipc=live_ipc, cache_hit=live_cache))
            elif target_type == ProceduralAssetType.PROCEDURAL_BIOME_CONSTRUCT:
                assets.append(self.synthesize_biome_construct(asset_seed))

        return assets


# Global singleton synthesizer instance
GLOBAL_ASSET_SYNTHESIZER = ProceduralAssetSynthesizer()
