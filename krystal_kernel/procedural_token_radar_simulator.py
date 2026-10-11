#!/usr/bin/env python3
"""
==============================================================================
KRYSTAL-STACK: PROCEDURAL TOKEN RADAR SIMULATOR & WEBGL COMPOSITION ENGINE
==============================================================================
Module: krystal_kernel/procedural_token_radar_simulator.py
Description: Tokenizes procedural composition operations, partitions procedures
             between WebGL graphics rendering and tactical simulation, provides
             6-axis attribute radars for mortar ballistics, and executes
             statistical move simulation balancing spell capacity with game
             mechanics capacity to generate engaging visual storytelling.

Key Capabilities:
  1. Tokenization & Procedure Partitioning:
     - Compiles LLM directives into structured tokens.
     - Partitions into WebGL Composition Procs, Tactical Simulation Procs,
       and Narrative Temporal Procs.
  2. 6-Axis Mortar Attribute Radar (Attribute Radar Tensor):
     - Axes: Range, AoE Dispersion, Armor Penetration, Action Cost, Spell Affinity, Visual Drama.
     - Solves situational mortar positioning (parabolic arc, dead-zone, trench dampening).
  3. Statistical Capacity Allocation & Turn Simulation:
     - Mathematical distribution modeling balancing Spell Capacity vs Mechanics Capacity.
     - Evaluates hit probability, CEP dispersion, and expected damage.
  4. WebGL Narrative Story Synthesis:
     - Generates shader uniforms, trajectory arcs, and camera framing for WebGL Canvas.
  5. System Invariant: VITAL_MAX_HP = 6.

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

# ── 1. PROCEDURAL TOKENS & PARTITIONING ───────────────────────────────────────

class ProcedureDomain(str, Enum):
    WEBGL_COMPOSITION = "WEBGL_COMPOSITION"
    TACTICAL_SIMULATION = "TACTICAL_SIMULATION"
    NARRATIVE_STORY = "NARRATIVE_STORY"


@dataclass
class ProceduralToken:
    token_id: str
    domain: ProcedureDomain
    operation: str
    weight: float
    parameters: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "token_id": self.token_id,
            "domain": self.domain.value,
            "operation": self.operation,
            "weight": self.weight,
            "parameters": self.parameters
        }


class ProceduralCompositionTokenizer:
    """
    Parses LLM directives and game states into modular procedural tokens,
    cleanly partitioning WebGL graphics passes from statistical simulation.
    """

    @classmethod
    def tokenize_directive(
        cls,
        directive_text: str,
        context: str = "TACTICAL_ARENA"
    ) -> List[ProceduralToken]:
        tokens: List[ProceduralToken] = []
        d_lower = directive_text.lower()

        # ── WebGL Composition Tokens ──
        if any(w in d_lower for w in ["mortar", "mínomet", "minomet", "salv", "artiller", "delostrel"]):
            tokens.append(ProceduralToken(
                token_id="TOK_WGL_BALLISTIC_ARC",
                domain=ProcedureDomain.WEBGL_COMPOSITION,
                operation="RENDER_PARABOLIC_TRAJECTORY_RIBBON",
                weight=0.9,
                parameters={"arc_subdivisions": 64, "glow_color": [1.0, 0.45, 0.1, 1.0], "line_width": 2.5}
            ))
            tokens.append(ProceduralToken(
                token_id="TOK_WGL_SHOCKWAVE_DECAL",
                domain=ProcedureDomain.WEBGL_COMPOSITION,
                operation="SPAWN_RADIAL_SHOCKWAVE_RING",
                weight=0.85,
                parameters={"expansion_rate_m_s": 28.0, "max_radius_m": 4.5, "distortion": 0.4}
            ))

        if any(w in d_lower for w in ["crystal", "kryštál", "krystal", "aether", "spell", "kúzl", "kuzl"]):
            tokens.append(ProceduralToken(
                token_id="TOK_WGL_CRYSTAL_REFRACTION",
                domain=ProcedureDomain.WEBGL_COMPOSITION,
                operation="APPLY_PRISM_CAUSTICS_PASS",
                weight=0.8,
                parameters={"refraction_index": 1.54, "dispersion_power": 0.08}
            ))

        # Core WebGL PBR and Camera Framing
        tokens.append(ProceduralToken(
            token_id="TOK_WGL_GOLDEN_SPIRAL_CAMERA",
            domain=ProcedureDomain.WEBGL_COMPOSITION,
            operation="FRAME_COMPOSITION_RULE_OF_THIRDS",
            weight=0.75,
            parameters={"fov": 60.0, "target_focal_node": [0.333, 0.667]}
        ))

        # ── Tactical Simulation Tokens ──
        tokens.append(ProceduralToken(
            token_id="TOK_SIM_MORTAR_BALLISTICS",
            domain=ProcedureDomain.TACTICAL_SIMULATION,
            operation="SOLVE_PARABOLIC_DISPERSION_TENSOR",
            weight=0.95,
            parameters={"gravity": 9.81, "elevation_deg": 62.0, "plunging_fire": True}
        ))
        tokens.append(ProceduralToken(
            token_id="TOK_SIM_CAPACITY_BALANCE",
            domain=ProcedureDomain.TACTICAL_SIMULATION,
            operation="ALLOCATE_SPELL_VS_MECHANIC_BUDGET",
            weight=0.9,
            parameters={"max_vital_hp": VITAL_MAX_HP, "stamina_cost": 2, "mana_flux": 3}
        ))

        # ── Narrative Story Tokens ──
        tokens.append(ProceduralToken(
            token_id="TOK_NAR_TEMPORAL_ARC",
            domain=ProcedureDomain.NARRATIVE_STORY,
            operation="SYNTHESIZE_BEFORE_DURING_AFTER",
            weight=0.85,
            parameters={"phase_count": 3, "dramatic_intensity": 0.88}
        ))

        return tokens


# ── 2. MORTAR ATTRIBUTE RADAR TENSOR ──────────────────────────────────────────

@dataclass
class MortarAttributeRadar:
    """
    6-Axis Attribute Radar evaluating tactical effectiveness and visual composition.
    Axes (All normalized 0.0 .. 1.0):
      1. range_reach: Maximum ballistic trajectory distance
      2. aoe_dispersion: Area of effect blast radius and CEP ellipse
      3. armor_penetration: Kinetic / explosive punch against fortifications
      4. action_cost_efficiency: Stamina/AP economy (higher is cheaper/more efficient)
      5. spell_affinity: Synergy with elemental/magical enchantments
      6. visual_drama: Visual spectacle, shockwave impulse, and screen-space impact
    """
    range_reach: float
    aoe_dispersion: float
    armor_penetration: float
    action_cost_efficiency: float
    spell_affinity: float
    visual_drama: float
    composite_rating: float
    vital_max_hp: int = VITAL_MAX_HP

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def evaluate(
        cls,
        muzzle_velocity: float = 26.0,
        elevation_angle_deg: float = 62.0,
        blast_radius_m: float = 3.2,
        spell_capacity_pct: float = 50.0,
        mechanics_capacity_pct: float = 50.0
    ) -> "MortarAttributeRadar":
        # 1. Range reach via ballistic equation: R = (v^2 * sin(2*phi)) / g
        phi_rad = math.radians(elevation_angle_deg)
        theoretical_range = (muzzle_velocity ** 2 * math.sin(2.0 * phi_rad)) / 9.81
        norm_range = min(1.0, max(0.1, theoretical_range / 50.0))

        # 2. AoE Dispersion
        norm_aoe = min(1.0, max(0.1, blast_radius_m / 6.0))

        # 3. Armor Penetration (steep plunging angle increases top-down roof penetration)
        plunging_factor = math.sin(phi_rad)
        penetration = min(1.0, 0.4 + plunging_factor * 0.45 + (mechanics_capacity_pct / 100.0) * 0.15)

        # 4. Action Cost Efficiency (Higher mechanics capacity reduces fatigue)
        action_efficiency = min(1.0, max(0.2, (mechanics_capacity_pct / 100.0) * 0.8 + 0.2))

        # 5. Spell Affinity (Directly modulated by spell capacity)
        spell_aff = min(1.0, max(0.05, (spell_capacity_pct / 100.0) * 0.95))

        # 6. Visual Drama (Combined blast, spell affinity, and height apex)
        apex_h = (muzzle_velocity * math.sin(phi_rad)) ** 2 / (2.0 * 9.81)
        norm_apex = min(1.0, apex_h / 25.0)
        drama = min(1.0, 0.3 * norm_apex + 0.35 * norm_aoe + 0.35 * spell_aff)

        composite = round((norm_range + norm_aoe + penetration + action_efficiency + spell_aff + drama) / 6.0, 3)

        return cls(
            range_reach=round(norm_range, 3),
            aoe_dispersion=round(norm_aoe, 3),
            armor_penetration=round(penetration, 3),
            action_cost_efficiency=round(action_efficiency, 3),
            spell_affinity=round(spell_aff, 3),
            visual_drama=round(drama, 3),
            composite_rating=composite,
            vital_max_hp=VITAL_MAX_HP
        )


# ── 3. STATISTICAL CAPACITY MODEL & TURN SIMULATOR ────────────────────────────

@dataclass
class TurnSimulationOutcome:
    turn_id: str
    target_hex: List[int]
    firing_distance_m: float
    time_of_flight_s: float
    trajectory_apex_m: float
    hit_probability: float
    scatter_offset_m: float
    effective_damage: int               # Clamped 0..6
    remaining_hp_defender: int          # Clamped 0..6
    sensory_overpressure_kpa: float
    trench_cover_mitigation_pct: float
    spell_energy_expended: float
    mechanics_ap_expended: int
    temporal_story_log: Dict[str, str]
    vital_max_hp: int = VITAL_MAX_HP

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class StatisticalCapacitySimulator:
    """
    Executes rigorous probabilistic turn simulation balancing spell capacity
    and game mechanics capacity according to mathematical ballistics.
    """

    @classmethod
    def simulate_mortar_turn(
        cls,
        target_hex: Tuple[int, int] = (1, 1),
        current_defender_hp: int = 6,
        spell_capacity: float = 60.0,       # 0..100%
        mechanics_capacity: float = 70.0,   # 0..100%
        elevation_angle_deg: float = 65.0,
        salvo_count: int = 2,
        seed: int = 42
    ) -> TurnSimulationOutcome:
        rng = random.Random(seed)

        # 1. Coordinate & Ballistics calculation
        # Hex (1, 1) -> distance approx 18.0 m
        dist_m = math.sqrt(target_hex[0]**2 + target_hex[1]**2) * 12.0
        v0 = max(15.0, math.sqrt((dist_m * 9.81) / max(0.1, math.sin(math.radians(2.0 * elevation_angle_deg)))))
        phi_rad = math.radians(elevation_angle_deg)
        
        # Time of flight: t = 2 * v0 * sin(phi) / g
        tof = (2.0 * v0 * math.sin(phi_rad)) / 9.81
        apex = ((v0 * math.sin(phi_rad)) ** 2) / (2.0 * 9.81)

        # 2. Probability & CEP dispersion
        # Higher mechanics capacity increases aiming precision
        precision_factor = (mechanics_capacity / 100.0)
        base_sigma = 2.5 * (1.1 - precision_factor * 0.6)
        scatter_m = abs(rng.gauss(0.0, base_sigma))
        
        # Spell capacity grants elemental guided homing
        spell_guidance = (spell_capacity / 100.0) * 0.4
        effective_scatter = max(0.2, scatter_m * (1.0 - spell_guidance))

        # Hit probability within blast radius (3.0m)
        hit_prob = round(math.exp(- (effective_scatter ** 2) / (2.0 * 3.0 ** 2)), 3)

        # 3. Damage & Trench Cover mitigation
        trench_cover = 0.50 if (target_hex[0] + target_hex[1]) % 2 == 0 else 0.15
        raw_damage = 2 * salvo_count
        mitigated_damage = max(1, int(round(raw_damage * hit_prob * (1.0 - trench_cover * 0.6))))
        
        # Ensure VITAL_MAX_HP clamping
        effective_damage = min(VITAL_MAX_HP, mitigated_damage)
        remaining_hp = max(0, min(VITAL_MAX_HP, current_defender_hp - effective_damage))

        # 4. Sensory Overpressure
        overpressure = round(22.0 + (salvo_count * 14.5) * (spell_capacity / 50.0), 2)

        # 5. Narrative Storytelling Temporal Arc
        story = {
            "before": f"Mínometná batéria fixuje azimut na hex {target_hex}. Alokovaná kapacita: {spell_capacity:.0f}% kúziel a {mechanics_capacity:.0f}% mechaniky.",
            "during": f"Salva {salvo_count} granátov vystrelená pod uhlom {elevation_angle_deg}°. Vrchol trajektórie {apex:.1f}m, čas letu {tof:.2f}s.",
            "after": f"Dopad s rozptylom {effective_scatter:.2f}m. Spôsobené poškodenie {effective_damage} HP (zostáva {remaining_hp}/{VITAL_MAX_HP} HP). Rázová vlna: {overpressure} kPa."
        }

        return TurnSimulationOutcome(
            turn_id=f"TURN-MORTAR-{seed}",
            target_hex=list(target_hex),
            firing_distance_m=round(dist_m, 2),
            time_of_flight_s=round(tof, 2),
            trajectory_apex_m=round(apex, 2),
            hit_probability=hit_prob,
            scatter_offset_m=round(effective_scatter, 2),
            effective_damage=effective_damage,
            remaining_hp_defender=remaining_hp,
            sensory_overpressure_kpa=overpressure,
            trench_cover_mitigation_pct=round(trench_cover * 100.0, 1),
            spell_energy_expended=round(spell_capacity * 0.45, 1),
            mechanics_ap_expended=int(salvo_count + 1),
            temporal_story_log=story,
            vital_max_hp=VITAL_MAX_HP
        )


# ── 4. WEBGL NARRATIVE DIRECTIVE COMPOSITOR ───────────────────────────────────

@dataclass
class WebGLStoryDirectives:
    canvas_camera_target: List[float]
    trajectory_curve_coefficients: List[float]
    blast_radius_visual: float
    shockwave_distortion_amplitude: float
    pbr_palette_rgb: List[List[float]]
    narrative_caption: str
    vital_max_hp: int = VITAL_MAX_HP

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ProceduralTokenRadarOrchestrator:
    """
    Unified entry point tokenizing directives, synthesizing attribute radars,
    simulating statistical turns, and exporting WebGL composition story directives.
    """

    def process_turn_directive(
        self,
        prompt: str,
        target_hex: Tuple[int, int] = (1, 1),
        current_hp: int = 6,
        spell_capacity: float = 65.0,
        mechanics_capacity: float = 75.0,
        elevation_deg: float = 62.0,
        salvo_count: int = 2,
        seed: int = 42
    ) -> Dict[str, Any]:
        t0 = time.perf_counter()

        # 1. Tokenize procedural operations
        tokens = ProceduralCompositionTokenizer.tokenize_directive(prompt)

        # 2. Partition procedures by domain
        partitioned: Dict[str, List[Dict[str, Any]]] = {
            "webgl_composition": [t.to_dict() for t in tokens if t.domain == ProcedureDomain.WEBGL_COMPOSITION],
            "tactical_simulation": [t.to_dict() for t in tokens if t.domain == ProcedureDomain.TACTICAL_SIMULATION],
            "narrative_story": [t.to_dict() for t in tokens if t.domain == ProcedureDomain.NARRATIVE_STORY]
        }

        # 3. Compute 6-axis attribute radar
        radar = MortarAttributeRadar.evaluate(
            muzzle_velocity=26.0,
            elevation_angle_deg=elevation_deg,
            blast_radius_m=3.5,
            spell_capacity_pct=spell_capacity,
            mechanics_capacity_pct=mechanics_capacity
        )

        # 4. Statistical turn simulation
        sim_outcome = StatisticalCapacitySimulator.simulate_mortar_turn(
            target_hex=target_hex,
            current_defender_hp=current_hp,
            spell_capacity=spell_capacity,
            mechanics_capacity=mechanics_capacity,
            elevation_angle_deg=elevation_deg,
            salvo_count=salvo_count,
            seed=seed
        )

        # 5. Synthesize WebGL Story Directives
        webgl_story = WebGLStoryDirectives(
            canvas_camera_target=[float(target_hex[0]), 0.5, float(target_hex[1])],
            trajectory_curve_coefficients=[0.0, sim_outcome.trajectory_apex_m, sim_outcome.firing_distance_m],
            blast_radius_visual=3.5,
            shockwave_distortion_amplitude=round(sim_outcome.sensory_overpressure_kpa / 80.0, 3),
            pbr_palette_rgb=[
                [0.98, 0.65, 0.15],  # Mortar flash
                [0.15, 0.75, 0.95],  # Aetheric spell glow
                [0.10, 0.12, 0.16]   # Trench basalt
            ],
            narrative_caption=sim_outcome.temporal_story_log["after"],
            vital_max_hp=VITAL_MAX_HP
        )

        t1 = time.perf_counter()

        return {
            "status": "SUCCESS",
            "prompt": prompt,
            "tokens_count": len(tokens),
            "procedure_partitioning": partitioned,
            "mortar_radar": radar.to_dict(),
            "turn_simulation": sim_outcome.to_dict(),
            "webgl_story_directives": webgl_story.to_dict(),
            "execution_latency_ms": round((t1 - t0) * 1000.0, 2),
            "vital_max_hp": VITAL_MAX_HP
        }


GLOBAL_TOKEN_RADAR_ORCHESTRATOR = ProceduralTokenRadarOrchestrator()
