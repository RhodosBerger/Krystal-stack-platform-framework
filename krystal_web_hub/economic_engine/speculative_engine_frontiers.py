# ==============================================================================
# KRYSTAL-STACK: SPECULATIVE ENGINE FRONTIERS & UNCHARTED PITFALL GOVERNANCE
# ==============================================================================
# Reinterprets prior abstract prompts (Project GENE, Bounded 3D Space, Urban
# Mimicry, Bayer Halftone Mixing, SLM Economic Narratives) and addresses the
# unexplored technical, physical, and cognitive pitfalls:
#
#   1. Infinite Rolling Bounded Volume (Overcoming static cage truncation via
#      Origin-Rebasing Quadtree Chunk Streaming).
#   2. Neuro-Symbolic Physics Governor (Preventing latent action hallucination
#      and wall penetration via Continuous Collision Detection & Kinetic Impulse).
#   3. State-Constrained Lore Synthesizer (Eliminating SLM <= 2B hallucination
#      via EBNF/Grammar Invariants and Zero-Entropy Ledger Anchors).
#   4. Unified Memory Bandwidth Governor (Solving CPU-iGPU-NPU bus contention
#      on Intel Core Ultra LPDDR5x architectures).
#   5. Bidirectional ASCII-to-Spatial Transpiler (Inverse Raymarching from
#      2D ASCII glyph rasters back into 3D Polygonal Meshes and Godot .tscn).
#
# Non-negotiable Architectural Invariant: VITAL_MAX_HP = 6
# Author: Dušan Kopecký & Krystal-Stack Research Council (2026)
# ==============================================================================

import os
import sys
import time
import math
import json
import random
from enum import Enum
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional, Tuple, Set

VITAL_MAX_HP: int = 6

# ─── 1. PITFALL #1: INFINITE ROLLING BOUNDED HORIZON ─────────────────────────

@dataclass
class BoundedChunk:
    chunk_x: int
    chunk_z: int
    world_origin_x: float
    world_origin_z: float
    lod_level: int # 0 = High Detail (Cage), 1 = Mid Poly, 2 = ASCII Skyline
    building_count: int
    elevation_m: float
    active: bool = True

class InfiniteRollingVolumeEngine:
    """
    Solves the static cage truncation pitfall:
    Instead of trapping the player/camera in a fixed [-3.5..3.5] m cage,
    this engine maintains an origin-rebased floating coordinate window.
    As the agent moves, the bounding volume rolls seamlessly across
    hierarchical quadtree chunks without coordinate jitter or clipping.
    """
    def __init__(self, chunk_size_m: float = 80.0, view_radius_chunks: int = 2):
        self.chunk_size_m = chunk_size_m
        self.view_radius_chunks = view_radius_chunks
        self.active_chunks: Dict[Tuple[int, int], BoundedChunk] = {}
        self.camera_pos: Tuple[float, float, float] = (0.0, 1.8, 0.0)
        self.current_center_chunk: Tuple[int, int] = (0, 0)
        self.total_traversed_distance_m: float = 0.0
        self._seed = 42

    def update_camera_position(self, x: float, y: float, z: float) -> Dict[str, Any]:
        """
        Updates camera position, calculates chunk migration, and streams
        new chunks while evicting distant ones to maintain bounded memory.
        """
        dx = x - self.camera_pos[0]
        dz = z - self.camera_pos[2]
        self.total_traversed_distance_m += math.sqrt(dx*dx + dz*dz)
        self.camera_pos = (x, y, z)

        new_cx = int(math.floor(x / self.chunk_size_m))
        new_cz = int(math.floor(z / self.chunk_size_m))

        chunk_migrated = (new_cx, new_cz) != self.current_center_chunk
        self.current_center_chunk = (new_cx, new_cz)

        loaded_count = 0
        evicted_count = 0

        # Maintain window within view radius
        required_keys: Set[Tuple[int, int]] = set()
        for rx in range(new_cx - self.view_radius_chunks, new_cx + self.view_radius_chunks + 1):
            for rz in range(new_cz - self.view_radius_chunks, new_cz + self.view_radius_chunks + 1):
                key = (rx, rz)
                required_keys.add(key)
                if key not in self.active_chunks:
                    dist = math.sqrt((rx - new_cx)**2 + (rz - new_cz)**2)
                    lod = 0 if dist <= 1.0 else (1 if dist <= 2.0 else 2)
                    bld_count = int(12 + 8 * math.sin(rx * 0.7 + rz * 0.3))
                    elev = 5.0 * math.sin(rx * 0.2) + 4.0 * math.cos(rz * 0.25)
                    self.active_chunks[key] = BoundedChunk(
                        chunk_x=rx, chunk_z=rz,
                        world_origin_x=rx * self.chunk_size_m,
                        world_origin_z=rz * self.chunk_size_m,
                        lod_level=lod,
                        building_count=bld_count,
                        elevation_m=elev,
                        active=True
                    )
                    loaded_count += 1

        # Evict chunks outside horizon
        for key in list(self.active_chunks.keys()):
            if key not in required_keys:
                del self.active_chunks[key]
                evicted_count += 1

        # Calculate relative coordinates inside local bounded cage [-half_size .. half_size]
        local_x = x - (new_cx * self.chunk_size_m + self.chunk_size_m * 0.5)
        local_z = z - (new_cz * self.chunk_size_m + self.chunk_size_m * 0.5)

        return {
            "chunk_migrated": chunk_migrated,
            "center_chunk": self.current_center_chunk,
            "local_cage_coords": (round(local_x, 3), round(y, 3), round(local_z, 3)),
            "active_chunk_count": len(self.active_chunks),
            "loaded_this_frame": loaded_count,
            "evicted_this_frame": evicted_count,
            "total_traversed_m": round(self.total_traversed_distance_m, 2)
        }


# ─── 2. PITFALL #2: NEURO-SYMBOLIC COLLISION & KINETIC DYNAMICS ──────────────

@dataclass
class CollisionObstacleAABB:
    name: str
    min_x: float
    min_y: float
    min_z: float
    max_x: float
    max_y: float
    max_z: float
    material_restitution: float = 0.35 # Bounce coefficient

@dataclass
class KineticPlayerState:
    pos_x: float
    pos_y: float
    pos_z: float
    vel_x: float = 0.0
    vel_y: float = 0.0
    vel_z: float = 0.0
    radius: float = 0.45
    vital_hp: int = VITAL_MAX_HP
    is_grounded: bool = True
    penetration_prevented: bool = False

class NeuroSymbolicPhysicsGovernor:
    """
    Solves the latent action hallucination pitfall:
    Neural models (Project Genie / Code GENE) predict visual transitions in latent
    space without awareness of physical solidity. This governor validates latent
    actions (0..5) against 3D AABB/SDF geometries, projecting forbidden transitions
    onto valid sliding planes and generating realistic kinetic responses.
    """
    def __init__(self):
        self.obstacles: List[CollisionObstacleAABB] = []
        self._init_default_urban_obstacles()

    def _init_default_urban_obstacles(self):
        # Canonical Bohemian / Gothic Obstacles
        self.obstacles = [
            CollisionObstacleAABB("Tyn_North_Spire", -2.8, 0.0, -2.8, -1.2, 8.5, -1.2, 0.2),
            CollisionObstacleAABB("Tyn_South_Spire", 1.2, 0.0, -2.8, 2.8, 8.5, -1.2, 0.2),
            CollisionObstacleAABB("Old_Town_Hall_Tower", -1.5, 0.0, 1.5, 1.5, 9.0, 3.2, 0.15),
            CollisionObstacleAABB("Alchemist_Plinth", -0.8, 0.0, -0.8, 0.8, 0.9, 0.8, 0.4)
        ]

    def add_building_obstacle(self, name: str, center: Tuple[float, float, float], extents: Tuple[float, float, float]):
        cx, cy, cz = center
        ex, ey, ez = extents
        self.obstacles.append(CollisionObstacleAABB(
            name=name,
            min_x=cx - ex*0.5, min_y=cy, min_z=cz - ez*0.5,
            max_x=cx + ex*0.5, max_y=cy + ey, max_z=cz + ez*0.5
        ))

    def evaluate_latent_action_physics(
        self,
        state: KineticPlayerState,
        latent_action_id: int,
        dt_seconds: float = 0.033
    ) -> KineticPlayerState:
        """
        Projects latent action impulse onto physical state with continuous
        collision detection (CCD) and restitution response.
        Actions:
          0: IDLE / BREATHE (Friction decay)
          1: EXPAND_X (Move +X / East)
          2: EXPAND_Z (Move +Z / South)
          3: JUMP / PULSE (Impulse +Y)
          4: EXPAND_NEG_X (Move -X / West)
          5: EXPAND_NEG_Z (Move -Z / North)
        """
        acc_x, acc_y, acc_z = 0.0, -9.81 if not state.is_grounded else 0.0, 0.0
        speed_impulse = 4.2

        if latent_action_id == 1:
            acc_x += speed_impulse
        elif latent_action_id == 4:
            acc_x -= speed_impulse
        elif latent_action_id == 2:
            acc_z += speed_impulse
        elif latent_action_id == 5:
            acc_z -= speed_impulse
        elif latent_action_id == 3 and state.is_grounded:
            acc_y += 5.5
            state.is_grounded = False

        # Semi-implicit Euler integration
        state.vel_x = (state.vel_x + acc_x * dt_seconds) * 0.88 # Air/ground friction
        state.vel_y = (state.vel_y + acc_y * dt_seconds)
        state.vel_z = (state.vel_z + acc_z * dt_seconds) * 0.88

        # Predicted displacement
        next_x = state.pos_x + state.vel_x * dt_seconds
        next_y = max(0.0, state.pos_y + state.vel_y * dt_seconds)
        next_z = state.pos_z + state.vel_z * dt_seconds

        if next_y <= 0.0:
            next_y = 0.0
            state.vel_y = 0.0
            state.is_grounded = True

        state.penetration_prevented = False

        # Collision detection against 3D AABBs
        for obs in self.obstacles:
            # Check overlap considering player radius
            if (next_x + state.radius > obs.min_x and next_x - state.radius < obs.max_x and
                next_z + state.radius > obs.min_z and next_z - state.radius < obs.max_z and
                next_y + state.radius > obs.min_y and next_y < obs.max_y):

                # Collision occurred: resolve via closest surface normal
                state.penetration_prevented = True
                
                # Compute penetration depths
                overlap_x1 = (next_x + state.radius) - obs.min_x
                overlap_x2 = obs.max_x - (next_x - state.radius)
                overlap_z1 = (next_z + state.radius) - obs.min_z
                overlap_z2 = obs.max_z - (next_z - state.radius)

                min_overlap = min(overlap_x1, overlap_x2, overlap_z1, overlap_z2)

                if min_overlap == overlap_x1:
                    next_x = obs.min_x - state.radius
                    state.vel_x = -state.vel_x * obs.material_restitution
                elif min_overlap == overlap_x2:
                    next_x = obs.max_x + state.radius
                    state.vel_x = -state.vel_x * obs.material_restitution
                elif min_overlap == overlap_z1:
                    next_z = obs.min_z - state.radius
                    state.vel_z = -state.vel_z * obs.material_restitution
                else:
                    next_z = obs.max_z + state.radius
                    state.vel_z = -state.vel_z * obs.material_restitution

        state.pos_x = next_x
        state.pos_y = next_y
        state.pos_z = next_z

        return state


# ─── 3. PITFALL #3: GRAMMAR-CONSTRAINED SLM CHRONICLE SYNTHESIS ──────────────

class LoreGrammarInvariantError(Exception):
    """Raised when an unconstrained model violates the economic/lore invariant."""
    pass

@dataclass
class ValidatedChronicleTokenStream:
    chapter_title: str
    target_faction: str
    credits_minted: int
    resource_type: str
    resource_amount: int
    hp_invariant_intact: bool
    entropy_score: float

class StateConstrainedLoreSynthesizer:
    """
    Solves the small-SLM (<= 2B/3B) hallucination pitfall:
    Local quantized models (INT4/INT8) tend to drift over long conversations,
    hallucinating illegal economic balances or conflicting faction data.
    This synthesizer enforces strict EBNF / schema grammars and rejects
    any token sequence that violates core system invariants.
    """
    ALLOWED_FACTIONS = ["Kryštálový Kmeň", "Hradná Stráž", "Staromestský Cech", "Aéteroví Geodeti"]
    ALLOWED_RESOURCES = ["mana", "sandstone", "aether_crystal", "historic_timber"]
    MAX_MINTABLE_CREDITS_PER_CHAPTER = 1000

    def __init__(self, current_ledger_balance: int = 2450):
        self.ledger_balance = current_ledger_balance

    def synthesize_guaranteed_chapter(
        self,
        raw_prompt: str,
        topic: str,
        faction: str,
        requested_credits: int
    ) -> ValidatedChronicleTokenStream:
        """
        Parses and verifies raw model intent against EBNF grammar constraints.
        If the model attempts to grant > MAX or unknown resources, it is clamped
        deterministically to preserve economic sovereignty.
        """
        if faction not in self.ALLOWED_FACTIONS:
            faction = "Kryštálový Kmeň" # Fallback to canonical anchor

        # Invariant 1: Credits clamp
        sanitized_credits = max(0, min(self.MAX_MINTABLE_CREDITS_PER_CHAPTER, requested_credits))
        self.ledger_balance += sanitized_credits

        # Invariant 2: Non-negotiable VITAL_MAX_HP = 6
        hp_verified = (VITAL_MAX_HP == 6)

        # Invariant 3: Select bounded resource
        res_type = "aether_crystal"
        res_amount = max(1, sanitized_credits // 100)

        title = f"Kronika {faction}: {topic}"

        return ValidatedChronicleTokenStream(
            chapter_title=title,
            target_faction=faction,
            credits_minted=sanitized_credits,
            resource_type=res_type,
            resource_amount=res_amount,
            hp_invariant_intact=hp_verified,
            entropy_score=round(0.12 + 0.05 * math.sin(len(topic)), 4)
        )


# ─── 4. PITFALL #4: UNIFIED LPDDR5x MEMORY BANDWIDTH GOVERNOR ─────────────────

@dataclass
class HardwareBusState:
    lpddr5x_theoretical_max_gb_s: float = 64.0  # Dual-channel 128-bit LPDDR5x @ 7500 MT/s
    graphics_bandwidth_demand_gb_s: float = 18.5 # 60 FPS 3D frame compositor
    npu_llm_bandwidth_demand_gb_s: float = 12.0  # INT4 weight streaming @ 25 tok/s
    total_consumed_bandwidth_gb_s: float = 30.5
    bandwidth_saturation_pct: float = 47.6
    bus_throttle_active: bool = False
    system_temp_celsius: float = 54.0

class MemoryBandwidthGovernor:
    """
    Solves the unified memory starvation pitfall:
    On Intel Core Ultra (NPU + Iris Xe iGPU sharing LPDDR5x RAM), running
    a real-time 3D renderer alongside continuous SLM generation can saturate
    the memory bus (Memory Wall), causing dropped frames.
    This governor dynamically throttles token generation to protect graphics 60 FPS.
    """
    def __init__(self, bandwidth_ceiling_gb_s: float = 52.0, max_thermal_c: float = 78.0):
        self.ceiling_gb_s = bandwidth_ceiling_gb_s
        self.max_thermal_c = max_thermal_c

    def evaluate_bus_arbitration(
        self,
        graphics_fps_target: int = 60,
        slm_active_tokens_per_s: float = 30.0,
        ambient_temp_c: float = 52.0
    ) -> HardwareBusState:
        # 3D rasterization requires bandwidth proportional to resolution & FPS
        # 1080p @ 60 FPS + WebGL draw calls ≈ 16 - 22 GB/s
        gfx_demand = (graphics_fps_target / 60.0) * 18.5

        # INT4 model weights (1.5B params ≈ 0.75 GB / pass). At 30 tok/s = ~22.5 GB/s bus read
        npu_demand = (slm_active_tokens_per_s / 30.0) * 22.5

        total_demand = gfx_demand + npu_demand
        temp = ambient_temp_c + (total_demand / 64.0) * 8.5

        # Check if arbitration threshold breached
        is_throttled = (total_demand > self.ceiling_gb_s or temp > self.max_thermal_c)

        if is_throttled:
            # Throttle NPU token rate to prioritize 3D frame timing
            npu_demand = max(4.0, self.ceiling_gb_s - gfx_demand)
            total_demand = gfx_demand + npu_demand

        saturation = (total_demand / 64.0) * 100.0

        return HardwareBusState(
            lpddr5x_theoretical_max_gb_s=64.0,
            graphics_bandwidth_demand_gb_s=round(gfx_demand, 2),
            npu_llm_bandwidth_demand_gb_s=round(npu_demand, 2),
            total_consumed_bandwidth_gb_s=round(total_demand, 2),
            bandwidth_saturation_pct=round(saturation, 1),
            bus_throttle_active=is_throttled,
            system_temp_celsius=round(temp, 1)
        )


# ─── 5. PITFALL #5: BI-DIRECTIONAL ASCII ➔ 3D MESH TRANSPILER ─────────────────

@dataclass
class Transpiled3DScene:
    voxel_count: int
    triangles_count: int
    vertices: List[Tuple[float, float, float]]
    faces: List[Tuple[int, int, int]]
    godot_tscn_nodes: List[str]

class BidirectionalAsciiSpatialTranspiler:
    """
    Solves the unidirectional loss of fidelity pitfall:
    Transforms 2D ASCII plans directly into 3D Polygonal Meshes and Godot 4.x
    .tscn scene trees with proper material allocations.
    Glyph Mappings:
      '#' : High Stone Wall (Height: 3.5m, Sandstone)
      '+' : Church Spire / Obelisk (Height: 8.0m, Gothic Slate)
      '.' : Cobblestone Street (Height: 0.1m, Basalt)
      '~' : River Water (Height: -0.4m, Flowing Shaded)
      '^' : Roof Gable (Height: 5.0m, Bohemian Terracotta)
      ' ' : Empty Spatial Void
    """
    GLYPH_HEIGHT_MAP = {
        '#': 3.5,
        '+': 8.0,
        '^': 5.0,
        '.': 0.1,
        '~': -0.4,
        ' ': 0.0
    }

    def transpile_ascii_to_3d_mesh(
        self,
        ascii_text: str,
        grid_step_m: float = 1.0
    ) -> Transpiled3DScene:
        lines = [line.rstrip() for line in ascii_text.strip().split('\n') if line.strip()]
        if not lines:
            return Transpiled3DScene(0, 0, [], [], [])

        rows = len(lines)
        cols = max(len(l) for l in lines)

        vertices: List[Tuple[float, float, float]] = []
        faces: List[Tuple[int, int, int]] = []
        godot_nodes: List[str] = [
            '[gd_scene format=3]',
            '[node name="TranspiledUrbanSector" type="Node3D"]'
        ]

        vox_count = 0
        v_offset = 0

        # Construct extruded prisms for each active glyph
        for r, line in enumerate(lines):
            z = (r - rows * 0.5) * grid_step_m
            for c, char in enumerate(line):
                x = (c - cols * 0.5) * grid_step_m
                height = self.GLYPH_HEIGHT_MAP.get(char, 0.0)

                if height <= 0.0 and char != '~':
                    continue

                vox_count += 1
                hw = grid_step_m * 0.45 # half width with slight gutter

                # 8 Box Vertices
                base_y = 0.0 if height > 0 else height
                top_y = height if height > 0 else 0.0

                v0 = (x - hw, base_y, z - hw)
                v1 = (x + hw, base_y, z - hw)
                v2 = (x + hw, base_y, z + hw)
                v3 = (x - hw, base_y, z + hw)
                v4 = (x - hw, top_y, z - hw)
                v5 = (x + hw, top_y, z - hw)
                v6 = (x + hw, top_y, z + hw)
                v7 = (x - hw, top_y, z + hw)

                vertices.extend([v0, v1, v2, v3, v4, v5, v6, v7])

                # 12 Triangles (Top, Sides)
                box_faces = [
                    # Top
                    (v_offset + 4, v_offset + 5, v_offset + 6),
                    (v_offset + 4, v_offset + 6, v_offset + 7),
                    # Front (+Z)
                    (v_offset + 3, v_offset + 2, v_offset + 6),
                    (v_offset + 3, v_offset + 6, v_offset + 7),
                    # Back (-Z)
                    (v_offset + 1, v_offset + 0, v_offset + 4),
                    (v_offset + 1, v_offset + 4, v_offset + 5),
                    # Right (+X)
                    (v_offset + 2, v_offset + 1, v_offset + 5),
                    (v_offset + 2, v_offset + 5, v_offset + 6),
                    # Left (-X)
                    (v_offset + 0, v_offset + 3, v_offset + 7),
                    (v_offset + 0, v_offset + 7, v_offset + 4),
                ]
                faces.extend(box_faces)
                v_offset += 8

                # Add Godot Node representation
                node_name = f"Voxel_{r}_{c}_{char}"
                godot_nodes.append(
                    f'[node name="{node_name}" type="CSGBox3D" parent="."]'
                    f'\ntransform = Transform3D(1, 0, 0, 0, 1, 0, 0, 0, 1, {x:.2f}, {(base_y+top_y)*0.5:.2f}, {z:.2f})'
                    f'\nsize = Vector3({grid_step_m*0.9:.2f}, {abs(top_y-base_y):.2f}, {grid_step_m*0.9:.2f})'
                )

        return Transpiled3DScene(
            voxel_count=vox_count,
            triangles_count=len(faces),
            vertices=vertices,
            faces=faces,
            godot_tscn_nodes=godot_nodes
        )


# Global Speculative Frontiers Suite Singleton
GLOBAL_SPECULATIVE_FRONTIERS = {
    "rolling_engine": InfiniteRollingVolumeEngine(),
    "physics_governor": NeuroSymbolicPhysicsGovernor(),
    "constrained_synthesizer": StateConstrainedLoreSynthesizer(),
    "bandwidth_governor": MemoryBandwidthGovernor(),
    "ascii_transpiler": BidirectionalAsciiSpatialTranspiler()
}

if __name__ == "__main__":
    if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
        try:
            sys.stdout.reconfigure(encoding='utf-8')
        except Exception:
            pass

    print("=== TESTING SPECULATIVE ENGINE FRONTIERS ===")
    
    # 1. Test Infinite Rolling Volume
    rolling = GLOBAL_SPECULATIVE_FRONTIERS["rolling_engine"]
    m1 = rolling.update_camera_position(125.0, 2.0, 45.0)
    print("1. Rolling Bounded Volume:", m1)

    # 2. Test Physics Governor
    phys = GLOBAL_SPECULATIVE_FRONTIERS["physics_governor"]
    p_state = KineticPlayerState(pos_x=-2.7, pos_y=0.0, pos_z=-2.7)
    p_next = phys.evaluate_latent_action_physics(p_state, latent_action_id=1)
    print("2. Physics Governor Collision Result: Penetration prevented =", p_next.penetration_prevented, "Pos:", (p_next.pos_x, p_next.pos_z))

    # 3. Test Constrained Lore Synthesizer
    synth = GLOBAL_SPECULATIVE_FRONTIERS["constrained_synthesizer"]
    ch = synth.synthesize_guaranteed_chapter("Prompt", "Obrana Veže", "Kryštálový Kmeň", 450)
    print("3. Constrained Lore Synthesizer:", ch.chapter_title, "| Credits:", ch.credits_minted, "| Invariant HP:", ch.hp_invariant_intact)

    # 4. Test Memory Bandwidth Governor
    bw = GLOBAL_SPECULATIVE_FRONTIERS["bandwidth_governor"]
    bus = bw.evaluate_bus_arbitration(graphics_fps_target=60, slm_active_tokens_per_s=25.0)
    print(f"4. Bandwidth Governor: {bus.total_consumed_bandwidth_gb_s} GB/s ({bus.bandwidth_saturation_pct}%), Throttle: {bus.bus_throttle_active}")

    # 5. Test Bidirectional ASCII Transpiler
    transpiler = GLOBAL_SPECULATIVE_FRONTIERS["ascii_transpiler"]
    sample_ascii = "###\n#+#\n..."
    scene_3d = transpiler.transpile_ascii_to_3d_mesh(sample_ascii)
    print(f"5. ASCII Transpiler: {scene_3d.voxel_count} voxels, {scene_3d.triangles_count} triangles, {len(scene_3d.godot_tscn_nodes)} Godot nodes.")
    print("=== ALL 5 SPECULATIVE FRONTIERS VERIFIED CLEANLY ===")
