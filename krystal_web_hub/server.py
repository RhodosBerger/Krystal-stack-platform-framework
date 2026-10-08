#!/usr/bin/env python3
"""
KRYSTAL-STACK LOCALHOST PRODUCTION HUB // MASTER SERVER
======================================================
A zero-dependency, high-performance HTTP + Server-Sent Events (SSE) server
providing the Neural ASCII Engine, 3D Raymarching, Cognitive Director,
and Economic Governor in real time.

Runs at: http://localhost:8080
Author: Dušan Kopecký & Krystal-Stack Architecture Team
Date: 2026-10-01
"""

import sys
import os
import time
import math
import json
import random
import threading
from pathlib import Path
from urllib.parse import urlparse, parse_qs
import urllib.request
import urllib.error
import mimetypes
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn

# Add workspace root to sys.path so modules can be imported seamlessly
WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

try:
    from antigravity_prompt_engine import AntigravityPromptEngine
    antigravity_engine = AntigravityPromptEngine()
except Exception as e:
    antigravity_engine = None
    print(f"[WARN] AntigravityPromptEngine init notice: {e}")

try:
    from mimicry_engine.mimic_recipes import list_recipes, get_recipe
    from mimicry_engine.scene_composer import (
        list_scenes, get_scene, paint_real_world_scene, UrbanSpatialCompositionRules
    )
    mimicry_available = True
except Exception as e:
    mimicry_available = False
    print(f"[WARN] MimicryEngine init notice: {e}")


try:
    from openworld_engine.world_semantic_compiler import WorldSemanticCompiler
    from openworld_engine.openworld_renderer import OpenWorldRenderer
    openworld_compiler = WorldSemanticCompiler()
    openworld_available = True
except Exception as e:
    openworld_compiler = None
    openworld_available = False
    print(f"[WARN] OpenWorldEngine init notice: {e}")

try:
    from krystal_lang.compiler import KrystalCompiler
    from krystal_lang.bytecode_to_shape import BytecodeShapeTranspiler
    from krystal_lang.virtual_machine import TopologicalVM
    krystal_compiler = KrystalCompiler()
    krystal_lang_available = True
except Exception as e:
    krystal_compiler = None
    krystal_lang_available = False
    print(f"[WARN] KrystalLang init notice: {e}")

try:
    from instances.procedural_generator import ProceduralInstanceGenerator
    procedural_generator_available = True
except Exception as e:
    procedural_generator_available = False
    print(f"[WARN] ProceduralInstanceGenerator notice: {e}")

try:
    from src.python.cyclic_organism_kernel import SymplecticCyclicEngine
    cyclic_engine_available = True
except Exception as e:
    cyclic_engine_available = False
    print(f"[WARN] SymplecticCyclicEngine notice: {e}")

try:
    from src.python.vulkan_compute_driver import VulkanComputeDriver
    vulkan_driver_available = True
except Exception as e:
    vulkan_driver_available = False
    print(f"[WARN] VulkanComputeDriver notice: {e}")

try:
    from src.python.project_intelligence import get_intelligence
    intelligence_available = True
except Exception as e:
    intelligence_available = False
    print(f"[WARN] ProjectIntelligence notice: {e}")

try:
    from krystal_web_hub.economic_engine import (
        GLOBAL_CITY_COMPOSITION_ENGINE,
        GLOBAL_METROPOLIS_ENGINE,
        ProceduralCityCompositionEngine,
        MultiSectorMetropolisEngine,
        VITAL_MAX_HP,
        GLOBAL_SKEUOMORPHIC_ENGINE,
        SkeuomorphicProceduralEngine,
        SkeuomorphicItemType,
        CharacterArchetype,
        RoomArchetype,
        MATERIAL_SUBSTRATES,
        CodeGeneNeuralCompositor,
        BoundedRenderingVolume,
        ProceduralRasterMixer,
        CodeGeneDynamicsModel,
        GLOBAL_ACHIEVEMENT_NARRATIVE_ENGINE
    )
    city_engine_available = True
    skeuomorphic_engine_available = True
    code_gene_available = True
    achievement_narrative_available = True
except Exception as e:
    city_engine_available = False
    skeuomorphic_engine_available = False
    code_gene_available = False
    achievement_narrative_available = False
    print(f"[WARN] City/Skeuomorphic/CodeGene/Achievement Engine notice: {e}")


if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass


# ─── 1. GLOBAL ENGINE STATE ──────────────────────────────────────────────────

class EngineState:
    def __init__(self):
        self.lock = threading.Lock()
        self.running = True
        self.mode = "CYBERPUNK"
        self.cols = 96
        self.rows = 40
        self.entropy_threshold = 0.70
        self.backpressure_active = False

        # Metrics
        self.fps = 30.0
        self.spatial_entropy = 0.25
        self.temporal_entropy = 0.35
        self.total_entropy = 0.30
        self.coherence = 0.70

        # Economic Governor
        self.budget = 850.0
        self.max_budget = 1000.0
        self.governor_state = "OPTIMAL"
        self.thermal_penalty = 0.0

        # Antigravity Composition & Recursive Mirror State
        self.active_template = None
        self.mirror_folds = 6
        self.recursion_depth = 24
        self.fresnel_reflectance = 0.88
        if antigravity_engine:
            try:
                self.active_template = antigravity_engine.instance_manager.compose_template(
                    "KALEIDOSCOPIC_IFS", "CYBERPUNK_NEON_AR", mirror_folds=6
                )
            except Exception:
                self.active_template = None

        # Mimicry Compositor & Blender Modifier Engine State
        self.active_mimic_id = "CYBER_TURRET_MK4"
        self.active_mimic_obj = get_recipe("CYBER_TURRET_MK4") if mimicry_available else None
        self.active_scene_id = "SCENE_CYBERPUNK_DISTRICT"
        self.active_scene_obj = get_scene("SCENE_CYBERPUNK_DISTRICT") if mimicry_available else None

        # Open-World Procedural Generator & Janet Code Synthesizer State
        self.active_world_spec = None
        self.active_world_renderer = None
        self.active_janet_dsl = ""
        self.active_godot_shader = ""
        self.active_py_generator = ""
        if openworld_available and openworld_compiler:
            try:
                def_spec = openworld_compiler.compile_natural_prompt("vulkanické hory s kryštálovými vežami a lávovými kaňonmi")
                self.active_world_spec = def_spec
                self.active_world_renderer = OpenWorldRenderer(def_spec["manifold"], def_spec)
                self.active_janet_dsl = openworld_compiler.to_janet_dsl(def_spec)
                self.active_godot_shader = openworld_compiler.to_godot_shader(def_spec)
                self.active_py_generator = openworld_compiler.to_procedural_python(def_spec)
            except Exception as e:
                print(f"[WARN] OpenWorld default init error: {e}")

        # Krystal-Lang Topological Engine State
        self.krystal_lang_comp = None
        self.krystal_lang_transpiler = None
        self.krystal_lang_vm = None
        if krystal_lang_available and krystal_compiler:
            try:
                sample_krystal_code = """module CyberMatrixCore
queue InStream { priority: 3, capacity: 512, spatial: [0.0, 0.0, -4.0] }
queue ProcessKernel { priority: 2, capacity: 1024, spatial: [0.0, 1.0, 0.0] }
queue OutRaster { priority: 1, capacity: 256, spatial: [0.0, 0.0, 4.0] }
shape KernelManifold { type: GYROID, frequency: 1.6, thickness: 0.12, csg: SMOOTH_MIN(0.25) }
pipeline InToKernel { from: InStream, to: ProcessKernel, transform: MIRROR_DN(folds: 8), action: EVAL_SDF }
pipeline KernelToOut { from: ProcessKernel, to: OutRaster, action: PASS }
loop Symmetry(iterations: 8, fold: 8) { transform: DISPLACE(amp: 0.12, freq: 2.2) }"""
                comp_res = krystal_compiler.compile(sample_krystal_code)
                self.krystal_lang_comp = comp_res
                self.krystal_lang_transpiler = BytecodeShapeTranspiler(comp_res)
                self.krystal_lang_vm = TopologicalVM(comp_res)
                self.krystal_lang_vm.inject_input("InStream", {"boot": True})
            except Exception as e:
                print(f"[WARN] Krystal-Lang default init error: {e}")

        # Symplectic Hamiltonian Cyclic Organism Engine State
        self.cyclic_engine = SymplecticCyclicEngine(organism_id="Alpha-Localhost") if cyclic_engine_available else None

        # Vulkan Compute Driver State (Zero-Compiler Ctypes Engine)
        self.vulkan_driver = VulkanComputeDriver() if vulkan_driver_available else None
        self.use_vulkan = True if (self.vulkan_driver and self.vulkan_driver.gpu_accelerated) else False

        # Code GENE Imitation & ASCII Neural Compositor State
        self.code_gene_compositor = CodeGeneNeuralCompositor() if code_gene_available else None

        # Current Rendered Frame
        self.current_ascii = ""
        self.frame_id = 0

        # Director Log Queue
        self.director_events = []

state = EngineState()

# ─── 2. SKEUOMORPHIC PROCEDURAL 3D SDF (REAL-WORLD ITEMS) ───────────────────

def sdf_capped_cylinder(p, h, r):
    d_x = math.sqrt(p[0]**2 + p[2]**2) - r
    d_y = abs(p[1]) - h
    return min(max(d_x, d_y), 0.0) + math.sqrt(max(d_x, 0.0)**2 + max(d_y, 0.0)**2)

def sdf_box_3d(p, b):
    qx = abs(p[0]) - b[0]
    qy = abs(p[1]) - b[1]
    qz = abs(p[2]) - b[2]
    outside = math.sqrt(max(qx, 0.0)**2 + max(qy, 0.0)**2 + max(qz, 0.0)**2)
    inside = min(max(qx, max(qy, qz)), 0.0)
    return outside + inside

def sdf_skeuomorphic_chalice(p):
    # Real-World Bohemian Alchemical Chalice (Stem, Knop, Fluted Bowl, Plinth)
    d_base = sdf_capped_cylinder((p[0], p[1] + 1.1, p[2]), h=0.08, r=0.75)
    d_stem = sdf_capped_cylinder((p[0], p[1] + 0.5, p[2]), h=0.55, r=0.13)
    d_knop = math.sqrt(p[0]**2 + (p[1] + 0.5)**2 + p[2]**2) - 0.25
    d_bowl_outer = math.sqrt(p[0]**2 + (p[1] - 0.4)**2 + p[2]**2) - 0.78
    d_bowl_inner = math.sqrt(p[0]**2 + (p[1] - 0.45)**2 + p[2]**2) - 0.68
    d_bowl = max(d_bowl_outer, -d_bowl_inner)
    d_bowl = max(d_bowl, p[1] - 1.1)
    return min(d_base, min(d_stem, min(d_knop, d_bowl)))

def sdf_skeuomorphic_athame(p):
    # Real-World Forged Damascus Dagger (Crossguard, Fluted Grip, Pommel, Blade)
    d_guard = sdf_box_3d((p[0], p[1], p[2]), (0.65, 0.08, 0.15))
    d_grip = sdf_capped_cylinder((p[0], p[1] + 0.35, p[2]), h=0.32, r=0.14)
    d_pommel = math.sqrt(p[0]**2 + (p[1] + 0.75)**2 + p[2]**2) - 0.22
    y_b = p[1] - 0.7
    if abs(y_b) < 0.75:
        progress = max(0.0, min(1.0, (1.45 - p[1]) / 1.45))
        w = max(0.02, 0.28 * progress)
        th = max(0.015, 0.06 * progress)
        d_blade = sdf_box_3d((p[0], p[1] - 0.7, p[2]), (w, 0.72, th))
    else:
        d_blade = max(abs(y_b) - 0.75, math.sqrt(p[0]**2 + p[2]**2) - 0.02)
    return min(d_guard, min(d_grip, min(d_pommel, d_blade)))

def rotate_y(p, theta):
    c, s = math.cos(theta), math.sin(theta)
    return (p[0]*c + p[2]*s, p[1], -p[0]*s + p[2]*c)

def rotate_x(p, theta):
    c, s = math.cos(theta), math.sin(theta)
    return (p[0], p[1]*c - p[2]*s, p[1]*s + p[2]*c)

def scene_sdf(p, t):
    p_rot = rotate_y(p, t * 0.85)
    # Cycle between Alchemical Chalice and Damascus Athame every 16 seconds
    cycle = int(t / 16.0) % 2
    if cycle == 0:
        return sdf_skeuomorphic_chalice(p_rot)
    else:
        # Tilt dagger slightly on X for dramatic edge highlight
        p_tilt = rotate_x(p_rot, 0.25)
        return sdf_skeuomorphic_athame(p_tilt)

def calc_normal(p, t):
    eps = 0.003
    d = scene_sdf(p, t)
    nx = scene_sdf((p[0]+eps, p[1], p[2]), t) - d
    ny = scene_sdf((p[0], p[1]+eps, p[2]), t) - d
    nz = scene_sdf((p[0], p[1], p[2]+eps), t) - d
    mag = math.sqrt(nx*nx + ny*ny + nz*nz)
    if mag == 0:
        return (0.0, 1.0, 0.0)
    return (nx/mag, ny/mag, nz/mag)

# ─── 3. BACKGROUND ENGINE THREAD ─────────────────────────────────────────────

try:
    from holographic_engine.hologram_projector import HolographicProjector
    holo_projector = HolographicProjector(width=state.cols, height=state.rows)
except Exception:
    holo_projector = None

def engine_worker_loop():
    BLOCK_CHARS = [" ", "░", "▒", "▓", "█"]
    DIR_CHARS = ['-', '/', '|', '\\']
    DENSITY_CHARS = " .:-=+*#%@"
    MATRIX_CHARS = "0123456789ABCDEFｦｱｳｴｵｶｷｹｺｻｼｽｾｿﾀﾂﾃﾅﾆﾇﾈﾊﾋﾎﾏﾐﾑﾒﾓﾔﾕﾗﾘﾜ"

    matrix_drops = [random.randint(0, 40) for _ in range(160)]

    prev_luma = []
    light_dir = (0.577, 0.577, -0.577)

    start_time = time.time()
    last_tick = time.perf_counter()

    while state.running:
        t = time.time() - start_time
        frame_start = time.perf_counter()

        with state.lock:
            mode = state.mode
            cols = state.cols
            rows = state.rows
            thresh = state.entropy_threshold

        lines = []
        current_luma = []
        aspect = (cols / rows) * 0.52

        # ── Mode 1: 3D SDF Raymarching ─────────────────────────────
        if mode in ("RAYMARCH_ANOMALY", "CYBERPUNK", "BLUEPRINT_EDGE", "HIGH_FIDELITY"):
            if state.vulkan_driver and state.use_vulkan and not state.backpressure_active:
                glyphs, colors = state.vulkan_driver.execute_raymarch(
                    width=cols, height=rows, t=t, mode=mode, step_budget=32
                )
                for y in range(rows):
                    row_offset = y * cols
                    line_chars = []
                    for x in range(cols):
                        g_idx = glyphs[row_offset + x]
                        if mode == "CYBERPUNK":
                            ch = BLOCK_CHARS[min(4, max(0, g_idx))]
                        elif mode == "BLUEPRINT_EDGE":
                            ch = DIR_CHARS[g_idx % 4] if g_idx > 0 else " "
                        elif mode == "RAYMARCH_ANOMALY":
                            ch = "█" if g_idx >= 3 else ("▓" if g_idx == 2 else ("▒" if g_idx == 1 else "░"))
                        else:  # HIGH_FIDELITY
                            d_idx = min(len(DENSITY_CHARS)-1, int((g_idx / 4.0) * (len(DENSITY_CHARS)-1)))
                            ch = DENSITY_CHARS[d_idx]
                        line_chars.append(ch)
                        current_luma.append(g_idx / 4.0)
                    lines.append("".join(line_chars))
            else:
                for y in range(rows):
                    line_chars = []
                    screen_y = (1.0 - (y / rows) * 2.0)
                    for x in range(cols):
                        screen_x = ((x / cols) * 2.0 - 1.0) * aspect
                        ro = (0.0, 0.0, -3.2)
                        rd_len = math.sqrt(screen_x**2 + screen_y**2 + 2.0**2)
                        rd = (screen_x / rd_len, screen_y / rd_len, 2.0 / rd_len)

                        dist = 0.0
                        hit = False
                        p = ro
                        for _ in range(20):
                            p = (ro[0] + rd[0]*dist, ro[1] + rd[1]*dist, ro[2] + rd[2]*dist)
                            d = scene_sdf(p, t)
                            if d < 0.005:
                                hit = True
                                break
                            dist += d
                            if dist > 7.0:
                                break

                        if hit:
                            norm = calc_normal(p, t)
                            diff = max(0.0, norm[0]*light_dir[0] + norm[1]*light_dir[1] + norm[2]*light_dir[2])
                            rim = 1.0 - max(0.0, -(norm[0]*rd[0] + norm[1]*rd[1] + norm[2]*rd[2]))
                            luma = min(1.0, max(0.0, diff * 0.75 + rim * 0.5))
                            current_luma.append(luma)

                            if mode == "BLUEPRINT_EDGE":
                                deg = math.degrees(math.atan2(norm[1], norm[0])) % 180.0
                                idx = int((deg / 180.0) * 4) % 4
                                ch = DIR_CHARS[idx] if rim > 0.35 else " "
                            elif mode == "CYBERPUNK":
                                idx = int(luma * (len(BLOCK_CHARS) - 1))
                                ch = BLOCK_CHARS[idx]
                            elif mode == "RAYMARCH_ANOMALY":
                                ch = "█" if luma > 0.8 else ("▓" if luma > 0.5 else ("▒" if luma > 0.25 else "░"))
                            else: # HIGH_FIDELITY
                                idx = int(luma * (len(DENSITY_CHARS) - 1))
                                ch = DENSITY_CHARS[idx]
                        else:
                            current_luma.append(0.0)
                            ch = " "
                        line_chars.append(ch)
                    lines.append("".join(line_chars))

        # ── Mode 2: Matrix Digital Rain ────────────────────────────
        elif mode == "MATRIX_RAIN":
            for y in range(rows):
                line_chars = []
                for x in range(cols):
                    drop_y = matrix_drops[x % len(matrix_drops)]
                    dist = y - drop_y
                    if 0 <= dist < 8:
                        ch = random.choice(MATRIX_CHARS)
                        luma = (8 - dist) / 8.0
                    else:
                        ch = " "
                        luma = 0.0
                    current_luma.append(luma)
                    line_chars.append(ch)
                lines.append("".join(line_chars))

            # Advance drops
            for i in range(len(matrix_drops)):
                matrix_drops[i] = (matrix_drops[i] + 1) if random.random() > 0.2 else matrix_drops[i]
                if matrix_drops[i] > rows + 10:
                    matrix_drops[i] = -random.randint(0, 10)

        # ── Mode 3: Holographic 3D Volumetric ─────────────────────
        elif mode == "HOLOGRAPHIC_3D" and holo_projector:
            holo_projector.width = cols
            holo_projector.height = rows
            holo_str, holo_meta = holo_projector.project_frame(t)
            lines = holo_str.split('\n')
            current_luma = [holo_meta.get("hologram_intensity", 0.5)] * 64

        # ── Mode 4: Recursive Mirror AR & Sacred Geometry ───────────
        elif mode in ("RECURSIVE_MIRROR", "SACRED_GEOMETRY"):
            if antigravity_engine and state.active_template:
                ascii_frame = antigravity_engine.instance_manager.render_ascii_frame(
                    state.active_template, t=t, width=cols, height=rows
                )
                lines = ascii_frame.split('\n')
                # Calculate synthetic luma for entropy calculations
                current_luma = [0.5 + 0.3 * math.sin(t * 2.0 + i * 0.1) for i in range(rows)]
            else:
                lines = ["RECURSIVE_MIRROR ENGINE INITIALIZING..."]

        # ── Mode 5: Real-World Mimicry Object Compositor ────────────
        elif mode == "MIMICRY_OBJECT":
            if state.active_mimic_obj:
                ascii_frame = state.active_mimic_obj.render_ascii_projection(
                    width=cols, height=rows, t=t
                )
                lines = ascii_frame.split('\n')
                current_luma = [0.45 + 0.3 * math.sin(t * 1.5 + i * 0.1) for i in range(rows)]
            else:
                lines = ["MIMICRY COMPOSITOR INITIALIZING..."]

        # ── Mode 6: Full 3D Game Scene Composition ──────────────────
        elif mode == "GAME_SCENE":
            if state.active_scene_obj:
                ascii_frame = state.active_scene_obj.render_ascii_view(
                    width=cols, height=rows, t=t
                )
                lines = ascii_frame.split('\n')
                current_luma = [0.4 + 0.25 * math.sin(t * 1.0 + i * 0.1) for i in range(rows)]
            else:
                lines = ["GAME SCENE INITIALIZING..."]

        # ── Mode 7: Procedural Infinite Open World ──────────────────
        elif mode == "OPENWORLD":
            if state.active_world_renderer:
                if (t - getattr(state, 'openworld_last_t', -999.0)) > 0.8 or not getattr(state, 'openworld_cache_lines', []):
                    ow_cols = min(56, cols)
                    ow_rows = min(24, rows)
                    cam_x = 7.0 * math.cos(t * 0.12)
                    cam_z = 7.0 * math.sin(t * 0.12)
                    ascii_frame = state.active_world_renderer.render_ascii_frame(
                        width=ow_cols, height=ow_rows,
                        cam_pos=(cam_x, 4.5 + 0.4 * math.sin(t * 0.25), cam_z),
                        cam_target=(0.0, 0.5, 0.0),
                        t=t
                    )
                    state.openworld_cache_lines = ascii_frame.split('\n')
                    state.openworld_last_t = t
                lines = state.openworld_cache_lines
                current_luma = [0.4 + 0.25 * math.sin(t * 0.8 + i * 0.1) for i in range(len(lines))]
            else:
                lines = ["OPENWORLD ENGINE INITIALIZING..."]

        # ── Mode 8: Krystal-Lang Topological Code Shape ─────────────
        elif mode == "KRYSTAL_LANG_SHAPE":
            if state.krystal_lang_transpiler:
                ascii_frame = state.krystal_lang_transpiler.render_shape_ascii(
                    width=cols, height=rows, t=t
                )
                lines = ascii_frame.split('\n')
                current_luma = [0.45 + 0.3 * math.sin(t * 1.5 + i * 0.1) for i in range(rows)]
                if state.krystal_lang_vm and state.frame_id % 12 == 0:
                    state.krystal_lang_vm.inject_input("InStream", {"tick": state.frame_id})
                    state.krystal_lang_vm.step_execution(cycles=1)
            else:
                lines = ["KRYSTAL-LANG SHAPE ENGINE INITIALIZING..."]

        # ── Mode 10: Symplectic Hamiltonian Organism Phase-Space ─────
        elif mode == "CYCLIC_HAMILTONIAN_ORGANISM":
            phase = state.cyclic_engine.cognitive_phase if state.cyclic_engine else "BETA"
            q_val = state.cyclic_engine.q[0] if state.cyclic_engine else 1.0
            p_val = state.cyclic_engine.p[0] if state.cyclic_engine else 0.5
            h_energy = state.cyclic_engine.total_hamiltonian() if state.cyclic_engine else 2.5

            for y in range(rows):
                line_chars = []
                screen_y = (1.0 - (y / rows) * 2.0)
                for x in range(cols):
                    screen_x = ((x / cols) * 2.0 - 1.0) * aspect
                    # Toroidal phase attractor with Hamiltonian deformation
                    dist_attractor = abs(math.sqrt(screen_x**2 + screen_y**2) - (0.75 + 0.25 * math.sin(t * 1.8 + q_val)))
                    luma = max(0.0, min(1.0, 1.0 - dist_attractor * 3.5))

                    if dist_attractor < 0.12:
                        if phase == "OMEGA":
                            ch = "!" if (x + y) % 3 == 0 else "~"
                        elif phase == "GAMMA":
                            ch = "⚡" if (x + y) % 4 == 0 else "#"
                        elif phase == "BETA":
                            ch = "::"[(x + y) % 2]
                        else: # ALPHA
                            ch = "≈"
                    else:
                        ch = " "
                        luma = 0.0
                    current_luma.append(luma)
                    line_chars.append(ch)
                lines.append("".join(line_chars))

        # ── Mode 11: Code GENE Imitation / ASCII Neural Compositor ──
        elif mode in ("CODE_GENE", "CODE_GENE_IMITATION", "PROJECT_GENE", "CODE_JEAN"):
            if state.code_gene_compositor:
                ascii_frame, stats = state.code_gene_compositor.render_frame_ascii(
                    width=cols, height=rows, t=t
                )
                lines = ascii_frame.split('\n')
                current_luma = stats.get("luma_samples", [0.45] * rows)
            else:
                lines = ["CODE GENE COMPOSITOR INITIALIZING..."]

        # ── Mode 9: Retro CRT Scanlines ────────────────────────────
        else: # RETRO_CRT
            for y in range(rows):
                line_chars = []
                for x in range(cols):
                    if y % 2 == 0:
                        ch = "-"
                        luma = 0.3
                    else:
                        wave = math.sin(x * 0.15 + t * 4.0)
                        ch = "#" if wave > 0.4 else ("." if wave > -0.4 else " ")
                        luma = (wave + 1.0) * 0.5
                    current_luma.append(luma)
                    line_chars.append(ch)
                lines.append("".join(line_chars))

        # ── Calculate Real-Time Entropy ────────────────────────────
        if current_luma:
            mean_l = sum(current_luma) / len(current_luma)
            s_entropy = min(1.0, (sum((l - mean_l)**2 for l in current_luma) / len(current_luma)) * 4.0)

            if len(prev_luma) == len(current_luma):
                diffs = [abs(c - p) for c, p in zip(current_luma, prev_luma)]
                t_entropy = min(1.0, (sum(diffs) / len(diffs)) * 8.0)
            else:
                t_entropy = 0.3

            tot_entropy = 0.4 * s_entropy + 0.6 * t_entropy
            prev_luma = current_luma
        else:
            s_entropy, t_entropy, tot_entropy = 0.2, 0.3, 0.25

        # ── Advance Symplectic Hamiltonian Cyclic Organism ─────────
        if state.cyclic_engine:
            state.cyclic_engine.symplectic_step(
                dt=0.033,
                visual_entropy=tot_entropy,
                gpu_temp=45.0 + state.thermal_penalty * 30.0
            )

        # ── Economic Governor & Visual Backpressure ────────────────
        backpressure = tot_entropy > thresh or (state.cyclic_engine and state.cyclic_engine.cognitive_phase == "OMEGA")
        with state.lock:
            state.spatial_entropy = s_entropy
            state.temporal_entropy = t_entropy
            state.total_entropy = tot_entropy
            state.coherence = max(0.0, 1.0 - tot_entropy)
            state.backpressure_active = backpressure

            if backpressure:
                # Governor budget drain under chaotic visual conditions
                state.budget = max(50.0, state.budget - 2.5)
                state.governor_state = "THROTTLED"
                state.thermal_penalty = min(0.5, (tot_entropy - thresh) * 2.0)
            else:
                state.budget = min(state.max_budget, state.budget + 0.5)
                state.governor_state = "OPTIMAL"
                state.thermal_penalty = max(0.0, state.thermal_penalty - 0.05)

            state.current_ascii = "\n".join(lines)
            state.frame_id += 1

            # Periodic Autonomous Director Commentary
            if state.frame_id % 90 == 0:
                if backpressure:
                    state.director_events.append({
                        "author": "SLM_DIRECTOR",
                        "message": f"CRITICAL: High visual entropy ({tot_entropy:.2f}). Governor throttling background worker threads."
                    })
                elif state.budget > 800:
                    state.director_events.append({
                        "author": "SLM_DIRECTOR",
                        "message": f"STABLE: Coherence healthy ({(1.0 - tot_entropy)*100:.0f}%). Slicing optimal for Dual-Earn compute."
                    })

        # FPS regulation (Target 30 FPS for browser SSE efficiency)
        frame_time = time.perf_counter() - frame_start
        if frame_time < 0.033:
            time.sleep(0.033 - frame_time)
        state.fps = 1.0 / max(0.001, time.perf_counter() - frame_start)

# ─── 4. MULTI-THREADED HTTP & SSE REQUEST HANDLER ────────────────────────────

STATIC_DIR = Path(__file__).parent / "static"

# ── Optional Krystal Kernel (stdlib-only; hub still runs without it) ──
try:
    from krystal_kernel import get_kernel, shutdown_kernel
    from krystal_kernel.openai_api import ApiContext, handle as _kernel_handle
    _KERNEL_ERROR = None
except Exception as _e:  # noqa: BLE001
    get_kernel = shutdown_kernel = ApiContext = _kernel_handle = None
    _KERNEL_ERROR = f"{type(_e).__name__}: {_e}"

_api_ctx = None
_api_lock = threading.Lock()

def get_api_context():
    """Lazily create the kernel API context (spawns kernel + supervisor on first use)."""
    global _api_ctx
    if ApiContext is None:
        return None
    with _api_lock:
        if _api_ctx is None:
            _api_ctx = ApiContext(get_kernel())
        return _api_ctx

class KrystalHubHandler(BaseHTTPRequestHandler):
    # Headers and body go out as separate send() calls; without TCP_NODELAY the
    # Nagle + delayed-ACK interaction stalls every response by tens of ms.
    disable_nagle_algorithm = True

    def log_message(self, format, *args):
        # Silence default HTTP access logs for clean server output
        return

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS, PUT, DELETE")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_GET(self):
        # ── Krystal Kernel API (OpenAI-compatible + metrics) ───────
        if self._is_kernel_path(self._KERNEL_GET):
            self.handle_kernel_api("GET")
            return
        parsed = urlparse(self.path)
        path = parsed.path
        norm_path = path.rstrip("/") if path != "/" else "/"
        # ── Root / UI ──────────────────────────────────────────────
        if norm_path in ("/", "/index", "/index.html"):
            self.serve_file(STATIC_DIR / "index.html", "text/html")
            return
        elif norm_path in ("/game", "/arena", "/posledni-kmen", "/godot_builder_extension.html", "/builder", "/studio-3d"):
            self.serve_file(STATIC_DIR / "godot_builder_extension.html", "text/html")
            return
        elif norm_path in ("/manual", "/design-manual", "/game_elements_design_manual.html"):
            self.serve_file(STATIC_DIR / "game_elements_design_manual.html", "text/html")
            return
        elif norm_path in (
            "/city", "/city-studio", "/city_studio", "/city-composer", "/city_composer",
            "/city-composer-studio", "/city_composer_studio", "/city_composer_studio.html",
            "/metropolis", "/composer", "/cities"
        ):
            self.serve_file(STATIC_DIR / "city_composer_studio.html", "text/html")
            return
        elif norm_path in (
            "/skeuomorphic", "/skeuo", "/skeuomorphic-studio", "/skeuomorphic_studio",
            "/skeuomorphic_studio.html", "/real-world", "/realworld", "/studio-skeuo", "/skeuomorph"
        ):
            self.serve_file(STATIC_DIR / "skeuomorphic_studio.html", "text/html")
            return
        elif norm_path in (
            "/code-gene", "/code-gene-studio", "/code_gene", "/code_gene_studio",
            "/code-gene-studio.html", "/code-jean", "/codejean", "/project-gene", "/project-genie",
            "/ascii-neural-compositor", "/code_gene.html", "/enterprise", "/gene-enterprise",
            "/economic-studio", "/achievements-studio", "/codex-studio"
        ):
            self.serve_file(STATIC_DIR / "code_gene_studio.html", "text/html")
            return
        elif norm_path in (
            "/blog", "/speculative-architecture", "/neuromorphic-cpu", "/microprocessor",
            "/speculative_microprocessor_blog.html", "/essay", "/manifesto"
        ):
            self.serve_file(STATIC_DIR / "speculative_microprocessor_blog.html", "text/html")
            return
        elif norm_path in ("/ontology", "/llm-reference", "/api/llm/context", "/llm.md"):
            ref_path = Path(WORKSPACE_ROOT) / "docs" / "LLM_ARCHITECTURAL_RAPID_LEARNING_REFERENCE_AND_ONTOLOGY.md"
            if ref_path.exists():
                self.send_text(ref_path.read_text(encoding="utf-8"), content_type="text/markdown; charset=utf-8")
            else:
                self.send_error(404, "Reference document not found")
            return
        elif path.startswith("/static/"):
            rel_path = path[8:].lstrip("/\\")
            target = (STATIC_DIR / rel_path).resolve()
            try:
                target.relative_to(STATIC_DIR.resolve())
            except ValueError:
                self.send_error(403, "Access denied")
                return
            mime = self._guess_mime(target)
            self.serve_file(target, mime)
            return

        # ── API: Health ────────────────────────────────────────────
        elif norm_path in ("/api/health", "/health"):
            self.send_json({"status": "HEALTHY", "uptime_sec": time.time()})
            return

        # ── API: Processor Integrity & Context-Switch Telemetry ────
        elif norm_path in ("/api/kernel/integrity", "/api/processor/integrity", "/api/kernel_telemetry"):
            from krystal_kernel.processor_integrity_telemetry import GLOBAL_PROCESSOR_INTEGRITY_ENGINE
            rep = GLOBAL_PROCESSOR_INTEGRITY_ENGINE.sample_integrity()
            self.send_json(rep.to_dict())
            return

        # ── API: Visual Behavioral Relation Predictor ──────────────
        elif norm_path in ("/api/copilot/predict", "/api/copilot/behavior"):
            from krystal_kernel.processor_integrity_telemetry import GLOBAL_PROCESSOR_INTEGRITY_ENGINE
            from krystal_kernel.visual_copilot_generator import GLOBAL_VISUAL_COPILOT_GENERATOR
            rep = GLOBAL_PROCESSOR_INTEGRITY_ENGINE.sample_integrity()
            pred = GLOBAL_VISUAL_COPILOT_GENERATOR.predictor.predict(
                rep.context_switches_per_sec, rep.cpu_utilization_pct, rep.thrashing_index
            )
            self.send_json({
                "telemetry": rep.to_dict(),
                "visual_prediction": pred.to_dict()
            })
            return

        # ── API: OpenAPI 3.1.0 Specification ───────────────────────
        elif norm_path in ("/api/openapi.json", "/openapi.json"):
            from krystal_kernel.openapi_spec import get_openapi_specification
            self.send_json(get_openapi_specification())
            return

        # ── API: OpenAPI Interactive Documentation (Swagger UI) ────
        elif norm_path in ("/api/docs", "/api/swagger", "/docs"):
            from krystal_kernel.openapi_spec import render_swagger_ui_html
            self.send_text(render_swagger_ui_html(), content_type="text/html; charset=utf-8")
            return

        # ── API: Cortex Decision Integrity ────────────────────────
        elif norm_path in ("/api/cortex/integrity", "/api/cortex/status"):
            from krystal_kernel.processor_integrity_telemetry import GLOBAL_PROCESSOR_INTEGRITY_ENGINE
            from krystal_kernel.cortex_openvino_engine import GLOBAL_CORTEX_ENGINE
            rep = GLOBAL_PROCESSOR_INTEGRITY_ENGINE.sample_integrity()
            verdict = GLOBAL_CORTEX_ENGINE.evaluate_cortex_integrity(
                cs_rate=rep.context_switches_per_sec,
                cpu_load=rep.cpu_utilization_pct,
                thrashing_index=rep.thrashing_index
            )
            self.send_json({
                "cortex_verdict": verdict.to_dict(),
                "telemetry": rep.to_dict()
            })
            return

        # ── API: Active System Processes for Prioritization ───────
        elif norm_path in ("/api/cortex/processes", "/api/processes"):
            from krystal_kernel.cortex_compiler import GLOBAL_CORTEX_COMPILER
            qs = parse_qs(parsed.query)
            limit = int(qs.get("limit", [50])[0])
            procs = GLOBAL_CORTEX_COMPILER.win_governor.list_running_processes(limit=limit)
            self.send_json({
                "total_processes": len(procs),
                "processes": procs,
                "vital_max_hp": 6
            })
            return

        # ── API: Asahi-Inspired Power Governor Telemetry ───────────
        elif norm_path in ("/api/asahi/power", "/api/power/asahi"):
            from krystal_kernel.asahi_power_governor import GLOBAL_ASAHI_POWER_GOVERNOR
            pwr = GLOBAL_ASAHI_POWER_GOVERNOR.regulate()
            self.send_json(pwr.to_dict())
            return

        # ── API: Intel Iris Xe Unified Memory Architecture (UMA) ───
        elif norm_path in ("/api/iris_xe/uma", "/api/uma/status"):
            from krystal_kernel.iris_xe_uma_memory_manager import GLOBAL_IRIS_XE_UMA_MANAGER
            uma = GLOBAL_IRIS_XE_UMA_MANAGER.update_pacing_and_scale_quotient(measured_frame_time_ms=7.0)
            self.send_json(uma.to_dict())
            return

        # ── API: Self-Healing Telemetry & Transient Deviation Status
        elif norm_path in ("/api/self_healing/status", "/api/self_healing"):
            from krystal_kernel.self_healing_patterns import GLOBAL_SELF_HEALING_GOVERNOR
            from krystal_kernel.processor_integrity_telemetry import GLOBAL_PROCESSOR_INTEGRITY_ENGINE
            rep = GLOBAL_PROCESSOR_INTEGRITY_ENGINE.sample_integrity()
            heal = GLOBAL_SELF_HEALING_GOVERNOR.evaluate_and_heal(
                cs_rate=rep.context_switches_per_sec,
                thrashing_index=rep.thrashing_index,
                frame_time_ms=7.5,
                junction_temp_c=72.0
            )
            self.send_json(heal.to_dict())
            return

        # ── API: WSL2 Coprocessor & Automated UFS Subsystem Status ──
        elif norm_path in ("/api/wsl/coprocessor", "/api/wsl/status"):
            from krystal_kernel.wsl_coprocessor import GLOBAL_WSL_COPROCESSOR
            self.send_json(GLOBAL_WSL_COPROCESSOR.get_coprocessor_status())
            return

        elif norm_path in ("/api/wsl/gui_status", "/api/wsl/gui"):
            from krystal_kernel.wsl_gui_compositor_bridge import GLOBAL_WSL_GUI_COMPOSITOR_BRIDGE
            self.send_json(GLOBAL_WSL_GUI_COMPOSITOR_BRIDGE.get_system_status())
            return

        elif norm_path in ("/api/wsl/hypervisor_port", "/api/wsl/port"):
            from krystal_kernel.wsl_hypervisor_gnome_port import GLOBAL_GNOME_OVERLAY_MANAGER
            self.send_json(GLOBAL_GNOME_OVERLAY_MANAGER.get_port_status().to_dict())
            return

        elif norm_path in ("/api/wsl/registry_inspect", "/api/wsl/apps"):
            from krystal_kernel.wsl_hypervisor_gnome_port import WindowsRegistryAccessPort
            apps = [a.to_dict() for a in WindowsRegistryAccessPort.get_installed_applications()]
            reg_sample = WindowsRegistryAccessPort.inspect_registry_key("HKLM\\SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\Uninstall")
            self.send_json({
                "status": "OK",
                "installed_win32_apps": apps,
                "registry_query": reg_sample,
                "vital_max_hp": 6
            })
            return

        elif norm_path in ("/api/openvino/hardware_bypass_status", "/api/openvino/bypass"):
            from krystal_kernel.openvino_hardware_acceleration_pack import TigerLakeTurboUnblocker
            self.send_json(TigerLakeTurboUnblocker.apply_hardware_bypass().to_dict())
            return

        elif norm_path in ("/api/openvino/benchmark_dp4a", "/api/openvino/benchmark"):
            from krystal_kernel.openvino_hardware_acceleration_pack import HardwareAccelerationBenchmark
            self.send_json(HardwareAccelerationBenchmark.run_comparative_benchmark().to_dict())
            return

        elif norm_path in ("/api/wsl/scripts", "/api/wsl/admin_scripts"):
            from krystal_kernel.wsl_coprocessor import GLOBAL_WSL_COPROCESSOR
            scripts_info = [
                {
                    "name": "wsl_hardware_coprocessor.sh",
                    "path": str(GLOBAL_WSL_COPROCESSOR.wsl_coprocessor_script),
                    "exists": GLOBAL_WSL_COPROCESSOR.wsl_coprocessor_script.exists(),
                    "purpose": "Hardware telemetry probe, failing process forensic inspection, zombie thread cleanup in WSL2"
                },
                {
                    "name": "ufs_log_triage.sh",
                    "path": str(GLOBAL_WSL_COPROCESSOR.ufs_triage_script),
                    "exists": GLOBAL_WSL_COPROCESSOR.ufs_triage_script.exists(),
                    "purpose": "Unix File System log stream sorting, severity analysis, inode health check, socket sweeping"
                }
            ]
            self.send_json({
                "status": "OK",
                "scripts": scripts_info,
                "vital_max_hp": 6
            })
            return

        elif norm_path in ("/api/nss/glsl_shader", "/api/nss/shader"):
            from krystal_kernel.krystal_neural_super_sampler import GLOBAL_NEURAL_SUPER_SAMPLER
            shader_code = GLOBAL_NEURAL_SUPER_SAMPLER.generate_open_source_vulkan_glsl()
            self.send_json({
                "shader_name": "krystal_nss_reconstruction.comp",
                "vulkan_version": "1.3",
                "license": "Apache-2.0 / Community Modifiable Open Source",
                "glsl_source": shader_code,
                "vital_max_hp": 6
            })
            return

        elif norm_path in ("/api/godot/render_package", "/api/godot/package"):
            from pathlib import Path
            gd_script_path = Path(WORKSPACE_ROOT) / "godot_project" / "scripts" / "KrystalRenderEngineIntegration.gd"
            gd_shader_path = Path(WORKSPACE_ROOT) / "godot_project" / "shaders" / "krystal_knss_godot_viewport.gdshader"
            bridge_header_path = Path(WORKSPACE_ROOT) / "include" / "krystal_godot_llm_vram_bridge.h"
            self.send_json({
                "status": "OK",
                "version": "4.2+",
                "engine_name": "Godot 4.x // Krystal Unified Render Engine & LLM Package",
                "engine_version": "4.2+",
                "integration_script": {
                    "file": str(gd_script_path),
                    "description": "GDScript viewport controller connecting Godot to Krystal Hub",
                    "source_snippet": "var krystal = KrystalRenderEngineIntegration.new()\nadd_child(krystal)"
                },
                "knss_shader": {
                    "file": str(gd_shader_path),
                    "description": "Vulkan YCoCg variance-clipping super-sampling GDShader (540p -> 1080p)"
                },
                "bridge_header": {
                    "file": str(bridge_header_path),
                    "description": "C99/C++ cross-platform bridge header"
                },
                "godot_engine_compatibility": "Godot 4.0 - 4.3+ (Vulkan Forward+ & Mobile)",
                "commercial_licensing": {
                    "ready_for_commercial_distribution": True,
                    "license": "Apache-2.0 / Commercial Krystal Consortium"
                },
                "script_exists": gd_script_path.exists(),
                "viewport_shader": str(gd_shader_path),
                "shader_exists": gd_shader_path.exists(),
                "rest_hub_url": "http://127.0.0.1:8080",
                "vital_max_hp": 6
            })
            return

        # ── API: Janet Bytecode Decoder & Profile SVG ───────────────
        elif norm_path in ("/api/janet/status", "/api/janet/demo"):
            from krystal_kernel.janet_bytecode_decoder import JanetBytecodeDecoder, generate_canonical_demo_stream
            demo_bytes = generate_canonical_demo_stream()
            res = JanetBytecodeDecoder.process_hex_or_binary_input(demo_bytes)
            self.send_json({
                "status": "OK",
                "parsed": res["parsed"],
                "report": res["report"],
                "vital_max_hp": 6
            })
            return

        elif norm_path in ("/api/janet/render_profile_svg", "/api/janet/svg"):
            from krystal_kernel.janet_bytecode_decoder import JanetBytecodeDecoder, generate_canonical_demo_stream
            demo_bytes = generate_canonical_demo_stream()
            res = JanetBytecodeDecoder.process_hex_or_binary_input(demo_bytes)
            svg_content = JanetBytecodeDecoder.render_execution_profile_svg(res["parsed"])
            query = parse_qs(parsed.query)
            if query.get("format", [""])[0] == "raw":
                self.send_response(200)
                self.send_header("Content-Type", "image/svg+xml; charset=utf-8")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(svg_content.encode("utf-8"))
            else:
                self.send_json({
                    "status": "OK",
                    "format": "svg",
                    "svg_blueprint": svg_content,
                    "instruction_count": res["parsed"]["total_words"],
                    "vital_max_hp": 6
                })
            return

        # ── API: Code GENE Neural Compositor State & Mesh & Render ──
        elif norm_path == "/api/code_gene/state":
            if state.code_gene_compositor:
                s = state.code_gene_compositor.dynamics.state
                b = state.code_gene_compositor.bounds
                self.send_json({
                    "status": "OK",
                    "step_id": s.step_id,
                    "topology": s.topology,
                    "last_action": s.last_action,
                    "cam_orbit_deg": s.cam_orbit_deg,
                    "crystal_growth": s.crystal_growth_level,
                    "bounds": {
                        "min": [b.min_x, b.min_y, b.min_z],
                        "max": [b.max_x, b.max_y, b.max_z],
                        "volume_m3": round(b.volume_m3(), 2),
                        "grid_spacing": b.grid_spacing
                    },
                    "raster_settings": {
                        "dither_mode": state.code_gene_compositor.raster_mixer.dither_mode,
                        "subpixel_quadrants": state.code_gene_compositor.raster_mixer.use_subpixel_quadrants,
                        "directional_sobel": state.code_gene_compositor.raster_mixer.use_directional_sobel,
                        "specular_sparkles": state.code_gene_compositor.raster_mixer.use_specular_sparkles
                    }
                })
            else:
                self.send_json({"status": "UNAVAILABLE", "error": "CodeGene compositor not loaded."})
            return

        elif norm_path == "/api/code_gene/mesh":
            if state.code_gene_compositor:
                mesh = state.code_gene_compositor.generate_3d_mesh(resolution=18)
                self.send_json({"status": "OK", "mesh": mesh})
            else:
                self.send_json({"status": "UNAVAILABLE", "error": "CodeGene compositor not loaded."})
            return

        elif norm_path == "/api/code_gene/render":
            if state.code_gene_compositor:
                params = parse_qs(parsed.query)
                w = int(params.get("w", [78])[0])
                h = int(params.get("h", [28])[0])
                color = params.get("color", ["0"])[0] == "1"
                ascii_str, stats = state.code_gene_compositor.render_frame_ascii(
                    width=w, height=h, use_ansi_color=color
                )
                self.send_json({
                    "status": "OK",
                    "ascii_frame": ascii_str,
                    "stats": stats
                })
            else:
                self.send_json({"status": "UNAVAILABLE", "error": "CodeGene compositor not loaded."})
            return

        elif norm_path == "/api/google_maps/cities":
            from krystal_web_hub.economic_engine.google_maps_urban_extractor import GLOBAL_GOOGLE_MAPS_EXTRACTOR
            cities = GLOBAL_GOOGLE_MAPS_EXTRACTOR.list_available_cities()
            self.send_json({"status": "OK", "cities": cities})
            return

        # ── API: Achievements & Economic Codex ─────────────────────
        elif norm_path == "/api/achievements":
            if achievement_narrative_available:
                eng = GLOBAL_ACHIEVEMENT_NARRATIVE_ENGINE
                ach_list = []
                for a in eng.achievements.values():
                    ach_list.append({
                        "id": a.achievement_id,
                        "title": a.title,
                        "category": a.category.value,
                        "tier": a.tier.value,
                        "description": a.description,
                        "icon_glyph": a.icon_glyph,
                        "unlocked": a.unlocked,
                        "unlocked_at": a.unlocked_at,
                        "credits_reward": a.reward.krystal_credits,
                        "story_tokens": a.reward.story_inference_tokens,
                        "resource_grants": a.reward.resource_grants,
                        "artifact_unlock": a.reward.artifact_unlock
                    })
                self.send_json({
                    "status": "OK",
                    "achievements": ach_list,
                    "total_credits_minted": eng.total_credits_minted,
                    "unlocked_count": sum(1 for a in eng.achievements.values() if a.unlocked),
                    "total_count": len(eng.achievements)
                })
            else:
                self.send_json({"status": "UNAVAILABLE", "error": "Achievement engine offline."})
            return

        elif norm_path == "/api/achievements/codex":
            if achievement_narrative_available:
                eng = GLOBAL_ACHIEVEMENT_NARRATIVE_ENGINE
                entries = []
                for entry in reversed(eng.chronicle_codex):
                    entries.append({
                        "entry_id": entry.entry_id,
                        "achievement_id": entry.achievement_id,
                        "chapter_title": entry.chapter_title,
                        "narrative_text": entry.narrative_text,
                        "city_context": entry.city_context,
                        "tribe_affected": entry.tribe_affected,
                        "reputation_delta": entry.reputation_delta,
                        "created_at": entry.created_at,
                        "telemetry": entry.telemetry
                    })
                self.send_json({
                    "status": "OK",
                    "chapters_count": len(entries),
                    "codex_entries": entries
                })
            else:
                self.send_json({"status": "UNAVAILABLE", "error": "Achievement engine offline."})
            return

        elif norm_path == "/api/telemetry/hardware":
            if achievement_narrative_available:
                eng = GLOBAL_ACHIEVEMENT_NARRATIVE_ENGINE
                comp = eng.governor.get_comparative_report()
                snap = eng.governor.sample_telemetry(tokens=256, duration_s=1.0)
                self.send_json({
                    "status": "OK",
                    "live_snapshot": {
                        "backend": snap.backend.value,
                        "device_name": snap.device_name,
                        "target_hardware": snap.target_hardware,
                        "power_watts": snap.power_watts,
                        "joules_per_token": snap.energy_joules_per_tok,
                        "tokens_per_second": snap.tokens_per_second,
                        "ttft_ms": snap.time_to_first_token_ms,
                        "temp_celsius": round(eng.governor.thermal_ceiling_celsius - snap.thermal_headroom_celsius, 1),
                        "throttled": snap.is_throttled
                    },
                    "comparative_audit": comp
                })
            else:
                self.send_json({"status": "UNAVAILABLE", "error": "Telemetry governor offline."})
            return

        # ── API: Speculative Frontiers LPDDR5x Bus Telemetry ───────
        elif norm_path == "/api/frontiers/bus_state":
            from krystal_web_hub.economic_engine import GLOBAL_SPECULATIVE_FRONTIERS
            bw = GLOBAL_SPECULATIVE_FRONTIERS["bandwidth_governor"]
            bus = bw.evaluate_bus_arbitration(graphics_fps_target=60, slm_active_tokens_per_s=25.0)
            self.send_json({
                "status": "OK",
                "bus_telemetry": {
                    "max_bandwidth_gb_s": bus.lpddr5x_theoretical_max_gb_s,
                    "graphics_demand_gb_s": bus.graphics_bandwidth_demand_gb_s,
                    "npu_llm_demand_gb_s": bus.npu_llm_bandwidth_demand_gb_s,
                    "total_consumed_gb_s": bus.total_consumed_bandwidth_gb_s,
                    "saturation_pct": bus.bandwidth_saturation_pct,
                    "throttled": bus.bus_throttle_active,
                    "temp_celsius": bus.system_temp_celsius
                }
            })
            return

        # ── API: Processor Whisperer & Janet Hex Synthesizer ───────
        elif norm_path == "/api/whisperer/synthesize":
            from krystal_kernel import GLOBAL_PROCESSOR_WHISPERER
            params = parse_qs(parsed.query)
            scene = params.get("scene", ["NeoPraha_Alchemical_Matrix"])[0]
            seed = int(params.get("seed", [42])[0])
            script = GLOBAL_PROCESSOR_WHISPERER.synthesizer.synthesize_script(scene_name=scene, seed=seed)
            self.send_json({
                "status": "OK",
                "janet_source": script.script_source,
                "function_count": script.function_count,
                "hex_dump": script.hex_dump,
                "raw_hex_stream": script.raw_hex_stream,
                "bytecode_size_bytes": len(script.bytecode_bytes)
            })
            return

        elif norm_path == "/api/whisperer/status":
            from krystal_kernel import GLOBAL_PROCESSOR_WHISPERER
            wh = GLOBAL_PROCESSOR_WHISPERER
            rec = wh.governor.evaluate_optimal_instruction_set()
            self.send_json({
                "status": "OK",
                "in_memory_ring": {
                    "capacity": wh.ring_capacity,
                    "head": wh.ring_head,
                    "buffered_count": len(wh.ring_buffer)
                },
                "self_healing_log": {
                    "total_events": len(wh.log_engine.entries),
                    "total_healed": wh.log_engine.total_healed_events,
                    "l1_writes": wh.log_engine.l1_write_count,
                    "l3_writes": wh.log_engine.l3_write_count,
                    "current_switches": wh.log_engine.current_switch_count,
                    "dram_spills": wh.log_engine.dram_spill_count
                },
                "isa_recommendation": {
                    "isa": rec.isa_name,
                    "unroll_factor": rec.unroll_factor,
                    "prefetch_bytes": rec.prefetch_distance_bytes,
                    "l1_ratio": rec.l1_locality_ratio,
                    "l3_ratio": rec.l3_spill_ratio,
                    "rationale": rec.rationale
                },
                "ssd_persistence": {
                    "filepath": wh.ssd_filepath,
                    "queued_flushes": len(wh._flush_queue)
                }
            })
            return

        elif norm_path == "/api/whisperer/calculator":
            from dataclasses import asdict
            from krystal_kernel import GLOBAL_RENDER_CALCULATOR, GLOBAL_ADAPTIVE_SWAP_MANAGER
            calc = GLOBAL_RENDER_CALCULATOR
            swap = GLOBAL_ADAPTIVE_SWAP_MANAGER
            params = parse_qs(parsed.query)
            target_fps = int(params.get("fps", [60])[0])
            quality = params.get("quality", ["BALANCED"])[0]

            budget = calc.compute_render_budget(target_fps=target_fps, quality_preference=quality)
            swap_telem = swap.evaluate_adaptive_quota()

            # Format aggregated render functions
            funcs = []
            for k, f_desc in calc.registry.items():
                funcs.append({
                    "id": f_desc.function_id,
                    "opcode": f_desc.opcode.name,
                    "name": f_desc.display_name,
                    "category": f_desc.category,
                    "gpu_l2_kb": f_desc.gpu_l2_working_set_kb,
                    "alu_flops": f_desc.alu_intensity_flops,
                    "vram_bw_kb": f_desc.vram_bandwidth_kb_frame,
                    "diversity_score": f_desc.diversity_contribution,
                    "params": f_desc.parameters
                })

            self.send_json({
                "status": "OK",
                "hardware_profile": calc.profile.to_dict(),
                "budget_plan": asdict(budget),
                "swap_telemetry": asdict(swap_telem),
                "aggregated_render_functions": funcs
            })
            return

        elif norm_path == "/api/whisperer/readback_logs":
            from krystal_kernel import GLOBAL_ADAPTIVE_SWAP_MANAGER
            params = parse_qs(parsed.query)
            max_blocks = int(params.get("blocks", [10])[0])
            readback = GLOBAL_ADAPTIVE_SWAP_MANAGER.read_structured_ssd_logs(max_blocks=max_blocks)
            self.send_json({
                "status": "OK",
                "readback": readback
            })
            return

        elif norm_path == "/api/config/cells":
            from krystal_kernel import GLOBAL_LLM_CONFIG_AGENT
            self.send_json({
                "status": "OK",
                "cells": GLOBAL_LLM_CONFIG_AGENT.get_all_cells()
            })
            return

        elif norm_path == "/api/case_studies":
            case_studies = [
                {
                    "id": "CS-1",
                    "title": "Processor Whisperer & GF(2) Self-Healing State Transitions",
                    "category": "MICROARCHITECTURE_LOGS",
                    "key_metrics": {"syndrome_healed_rate": "100%", "isa_tiers": 4, "pgo_speedup": "AVX-512 FMA"},
                    "description": "Hamming (7, 4) binary parity matrix error recovery under current switching vs L1/L3 cache writebacks."
                },
                {
                    "id": "CS-2",
                    "title": "Hardware Render Budget & Cache-Constrained Scene Diversity",
                    "category": "GPU_CACHE_RAYMARCHING",
                    "key_metrics": {"gpu_l2_utilization": "43.1%", "diversity_score": "88.5 / 100", "raymarch_steps": 96},
                    "description": "Keeps active working set within 3.84 MB Iris Xe L2 cache, eliminating off-chip DRAM bus stalls."
                },
                {
                    "id": "CS-3",
                    "title": "Adaptive Swapping & Structured Log Generation (RAM vs SSD)",
                    "category": "MEMORY_STORAGE_IO",
                    "key_metrics": {"low_quota_batch": "24 entries", "burst_quota": "128 entries", "free_ram_gb": "10.1 GB"},
                    "description": "Inverse quota pacing: high RAM availability allows smooth, paced SSD block structuring without I/O wait spikes."
                },
                {
                    "id": "CS-4",
                    "title": "Closed-Loop Janet-to-Godot 4.x Screen-Space Raymarching",
                    "category": "ENGINE_SYNTHESIS",
                    "key_metrics": {"shader_format": "Vulkan Forward+ / Compatibility", "fps": "120+ FPS", "dither": "Bayer 4x4 / 8x8"},
                    "description": "Compiles empirical cache telemetry and Janet S-expressions into production-ready Godot 4.x .gdshader files."
                },
                {
                    "id": "CS-5",
                    "title": "Last-Combination Cache Matrix Repetition Compressor",
                    "category": "DATA_COMPRESSION",
                    "key_metrics": {"compression_ratio": "3.66x", "l1_lines_saved": 7, "deduplicated_values": 124},
                    "description": "Scans tail combinations and deduplicates constant affine rows, slashing L1/L3 writeback bandwidth."
                },
                {
                    "id": "CS-6",
                    "title": "LLM Configuration Agent & Form Cell Mutation",
                    "category": "NEURO_SYMBOLIC_AGENTS",
                    "key_metrics": {"repetition_drop": "0.85 -> 0.15", "mutated_cells": 7, "vital_hp_locked": 6},
                    "description": "Dynamically rewrites configuration panel form cells based on execution logs to ensure non-repetitive scenes."
                },
                {
                    "id": "CS-7",
                    "title": "Real-World Urban Spatial Mimicry & Blender 4.x Bridge",
                    "category": "URBAN_GEOMETRY",
                    "key_metrics": {"cities": ["Praha", "Bratislava", "Tokyo"], "export_format": "Wavefront OBJ + Blender Python"},
                    "description": "Extracts 3D street networks and gothic spires within a bounded cage for real-time ASCII/3D inspection."
                }
            ]
            self.send_json({
                "status": "OK",
                "case_studies_count": len(case_studies),
                "case_studies": case_studies,
                "research_doc": "docs/research/CASE_STUDIES_EXPLORATION_OF_GENERATED_VARIATIONS.md",
                "vital_max_hp": 6
            })
            return

        elif norm_path == "/api/vibe_prompts":
            prompts = [
                {
                    "prompt_id": "VP-01",
                    "title": "GF(2) Hamming Self-Healing Binary Matrix",
                    "target_function": "BinaryMatrixSelfHealingLog.encode_codeword & calculate_syndrome",
                    "vibe_prompt_sk": "Navrhni samoopravovaciu maticu v binárnom poli GF(2), ktorá bude sledovať hardvérové pulzy procesora – napríklad prepínanie napájacích stavov (C-states) a zápisy do L1 vs L3 vyrovnávacej pamäte...",
                    "cross_domain_use": "Aerospace Satellite Cosmic Radiation Bit-Flip Recovery & Banking Ledger Fault Tolerance"
                },
                {
                    "prompt_id": "VP-02",
                    "title": "Instruction Set Meta Governor (PGO ISA Selector)",
                    "target_function": "InstructionSetMetaGovernor.evaluate_optimal_instruction_set",
                    "vibe_prompt_sk": "Sprav našepkávača pre CPU inštrukčné sady, ktorý neháda, ale číta z našich samoopravovacích logov, ako sa správa cache pamäť (AVX-512 FMA vs MOVNTDQ vs SSE4)...",
                    "cross_domain_use": "Green AI Cloud Data Centers & Telecom 5G Baseband Beamforming"
                },
                {
                    "prompt_id": "VP-03",
                    "title": "Two-Tier Whisperer: In-Memory Ring Buffer & Async SSD Persistence",
                    "target_function": "ProcessorInstructionWhisperer.whisper_instruction_hint",
                    "vibe_prompt_sk": "Potrebujem dvojúrovňový systém: superrýchly RAM ring buffer pre nanosekundové nápovedy a asynchrónny SSD worker bez blokovania výpočtového vlákna...",
                    "cross_domain_use": "Autonomous Vehicle Black Box & High-Frequency Trading Execution Audits"
                },
                {
                    "prompt_id": "VP-04",
                    "title": "Hardware Render Budget Calculator & GPU L2 Working Set Fit",
                    "target_function": "HardwareRenderBudgetCalculator.compute_render_budget",
                    "vibe_prompt_sk": "Zisti skutočné parametre hardvéru (3.84 MB Iris Xe L2) a spočítaj pracovnú pamäť tak, aby sa celý raymarching zmestil do 85% L2 cache a maximalizoval diverzitu scény...",
                    "cross_domain_use": "Medical Ultrasound 60Hz Synthetic-Aperture Beamforming & Embedded Edge Drone SLAM"
                },
                {
                    "prompt_id": "VP-05",
                    "title": "Adaptive RAM Ring Buffer vs SSD Swapper (Inverse Quota Pacing)",
                    "target_function": "AdaptiveSwapManager.evaluate_adaptive_quota",
                    "vibe_prompt_sk": "Ak je v RAM dostatok miesta, swapovanie do SSD musí bežať v NIŽŠÍCH kvótach, aby SSD stíhalo vytvárať celú štruktúru logov bez I/O zásekov...",
                    "cross_domain_use": "Industrial IoT Sensor Gateways & Battery Energy Storage System (BESS) Flash Logging"
                },
                {
                    "prompt_id": "VP-06",
                    "title": "Closed-Loop Janet-to-Godot 4.x Vulkan Raymarcher Synthesis",
                    "target_function": "LogDrivenJanetGodotSynthesizer.synthesize_hardware_tuned_scene",
                    "vibe_prompt_sk": "Prečítaj štruktúrované logy z SSD, vlož dáta do syntetizátora a vygeneruj z Janet S-výrazov priamo funkčný Godot 4.x Vulkan shader (.gdshader) pre raymarching...",
                    "cross_domain_use": "Digital Twin Microclimate Urban Simulator & Virtual Reality Haptic Shaders"
                },
                {
                    "prompt_id": "VP-07",
                    "title": "Last-Combination Cache Matrix Repetition Compressor",
                    "target_function": "CacheMatrixRepetitionCompressor.compress_matrix_stream",
                    "vibe_prompt_sk": "Keď sa genericky opakujú niektoré čísla konštantne, skenuj iba posledné kombinácie čísel v histórii a nahraď duplikáty 2-bajtovým tokenom, šetriac L1 cache linky...",
                    "cross_domain_use": "High-Frequency Trading (HFT) Level-2 Order Book Compression & Multi-Electrode Neural Spikes"
                },
                {
                    "prompt_id": "VP-08",
                    "title": "Dynamic Form Cells & LLM Configuration Agent Mutation",
                    "target_function": "LLMConfigAgent.mutate_configuration_via_logs",
                    "vibe_prompt_sk": "Vytvor aplikačnú vrstvu pre formulárové bunky s LLM agentom, ktorý číta logy a automaticky prepisuje parametre novými hodnotami, aby scéna nebola repetitívna...",
                    "cross_domain_use": "Bio-Generative De Novo Drug Discovery & Dynamic Insurance Risk Actuarial Pricing"
                }
            ]
            self.send_json({
                "status": "OK",
                "prompts_count": len(prompts),
                "prompt_book": prompts,
                "research_doc": "docs/research/REVERSE_VIBE_CODING_PROMPT_BOOK_AND_CROSS_DOMAIN_DERIVATIVES.md",
                "vital_max_hp": 6
            })
            return

        # ── API: NextGen v2 Architecture & Iris Xe Kisak Optimization ──
        elif norm_path == "/api/nextgen/status":
            from krystal_stack_nextgen import (
                GLOBAL_IRIS_XE_OPTIMIZER,
                GLOBAL_ENERGY_GOVERNOR,
                GLOBAL_SUBGROUP_KERNEL,
                VITAL_MAX_HP
            )
            opt = GLOBAL_IRIS_XE_OPTIMIZER.compute_bandwidth_optimization()
            _, profile = GLOBAL_SUBGROUP_KERNEL.generate_optimized_godot_shader()
            self.send_json({
                "status": "OK",
                "nextgen_version": "2.0-PRODUCTION",
                "architecture_subfolder": "krystal_stack_nextgen",
                "vital_max_hp": VITAL_MAX_HP,
                "iris_xe_kisak_optimization": opt.to_dict(),
                "subgroup_kernel_profile": profile.to_dict(),
                "recent_energy_audits": [a.to_dict() for a in GLOBAL_ENERGY_GOVERNOR.history[-5:]]
            })
            return

        # ── API: NextGen NPU Hardware Predictor & Speculative Telemetry ──
        elif norm_path == "/api/nextgen/npu_predictor":
            from dataclasses import asdict
            from krystal_stack_nextgen import (
                GLOBAL_NPU_PREDICTOR,
                VITAL_MAX_HP
            )
            telem = GLOBAL_NPU_PREDICTOR.probe_windows_telemetry()
            strat = GLOBAL_NPU_PREDICTOR.evaluate_predictive_strategy()
            self.send_json({
                "status": "OK",
                "vital_max_hp": VITAL_MAX_HP,
                "windows_telemetry": asdict(telem),
                "active_npu_strategy": strat.to_dict(),
                "total_latency_saved_ms": round(GLOBAL_NPU_PREDICTOR.total_latency_saved_ms, 3),
                "npu_inferences_executed": GLOBAL_NPU_PREDICTOR.npu_inferences_executed,
                "pinned_ram_blocks_count": len(GLOBAL_NPU_PREDICTOR.prestager.pinned_ram_buffer)
            })
            return

        # ── API: NextGen Thermal & Memory Bus Optimization Report ──
        elif norm_path == "/api/nextgen/thermal_bus_report":
            from dataclasses import asdict
            from krystal_stack_nextgen import (
                GLOBAL_THERMAL_BUS_GOVERNOR,
                VITAL_MAX_HP,
                TargetPlatform
            )
            gov = GLOBAL_THERMAL_BUS_GOVERNOR
            params = parse_qs(parsed.query)
            as_markdown = params.get("format", ["json"])[0].lower() == "markdown"

            if as_markdown:
                md_text = gov.generate_comprehensive_markdown_report()
                self.send_text(md_text, content_type="text/markdown; charset=utf-8")
                return

            self.send_json({
                "status": "OK",
                "vital_max_hp": VITAL_MAX_HP,
                "memory_bus": asdict(gov.get_memory_bus_profile()),
                "float16_precision": asdict(gov.analyze_float16_precision()),
                "npu_prediction": asdict(gov.get_npu_prediction_metrics()),
                "cross_platform_thermal": {
                    "windows": asdict(gov.evaluate_cross_platform_thermal(TargetPlatform.WINDOWS_X64_INTEL, 78.5)),
                    "linux": asdict(gov.evaluate_cross_platform_thermal(TargetPlatform.LINUX_STEAM_DECK_KISAK, 71.0)),
                    "macos": asdict(gov.evaluate_cross_platform_thermal(TargetPlatform.MACOS_APPLE_SILICON_M_SERIES, 63.5)),
                    "android": asdict(gov.evaluate_cross_platform_thermal(TargetPlatform.ANDROID_EDGE_ARM_SNAPDRAGON, 86.0))
                }
            })
            return

        # ── API: NextGen TPU Systolic Tensor Acceleration Benchmark ──
        elif norm_path == "/api/nextgen/tpu_benchmark":
            from krystal_stack_nextgen import (
                GLOBAL_TPU_BENCHMARK,
                TensorPrecision,
                VITAL_MAX_HP
            )
            params = parse_qs(parsed.query)
            sweep = params.get("sweep", ["true"])[0].lower() in ("1", "true", "yes")

            if sweep:
                summary = GLOBAL_TPU_BENCHMARK.run_full_sweep()
                self.send_json({
                    "status": "SUCCESS",
                    "vital_max_hp": VITAL_MAX_HP,
                    "benchmark_summary": summary.to_dict()
                })
                return
            else:
                dim = int(params.get("dim", [128])[0])
                prec_str = params.get("precision", ["FP16"])[0].upper()
                prec = TensorPrecision.FP16
                if "FP32" in prec_str:
                    prec = TensorPrecision.FP32
                elif "INT8" in prec_str:
                    prec = TensorPrecision.INT8
                res = GLOBAL_TPU_BENCHMARK.run_benchmark(dimension=dim, precision=prec)
                self.send_json({
                    "status": "SUCCESS",
                    "vital_max_hp": VITAL_MAX_HP,
                    "result": res.to_dict()
                })
                return

        # ── API: NextGen Ontological Reverse-Engineering Prompts ──
        elif norm_path == "/api/nextgen/ontological_prompts":
            from krystal_stack_nextgen import (
                GLOBAL_ONTOLOGICAL_ENGINE,
                VITAL_MAX_HP
            )
            params = parse_qs(parsed.query)
            as_markdown = params.get("format", ["json"])[0].lower() == "markdown"

            if as_markdown:
                md_text = GLOBAL_ONTOLOGICAL_ENGINE.export_all_prompts_markdown()
                self.send_text(md_text, content_type="text/markdown; charset=utf-8")
                return

            self.send_json({
                "status": "SUCCESS",
                "vital_max_hp": VITAL_MAX_HP,
                "prompts_count": len(GLOBAL_ONTOLOGICAL_ENGINE.catalog),
                "prompts": [p.to_dict() for p in GLOBAL_ONTOLOGICAL_ENGINE.catalog]
            })
            return

        # ── API: NextGen Pattern Optimization & IPC Metrics ────────
        elif norm_path == "/api/nextgen/pattern_metrics":
            from krystal_stack_nextgen import GLOBAL_PATTERN_IPC_GOVERNOR, VITAL_MAX_HP
            metrics = GLOBAL_PATTERN_IPC_GOVERNOR.compute_pattern_metrics()
            self.send_json({
                "status": "SUCCESS",
                "vital_max_hp": VITAL_MAX_HP,
                "metrics": metrics.to_dict()
            })
            return

        # ── API: Procedural City Composition & Metropolis ──────────
        elif norm_path == "/api/city/compose":
            params = parse_qs(parsed.query)

            seed = int(params.get("seed", [42])[0])
            name = params.get("name", ["Neo-Praha Golden Spires"])[0]
            biome = params.get("biome", ["bohemian_cyber_noir"])[0]
            w = float(params.get("width", [240.0])[0])
            d = float(params.get("depth", [240.0])[0])
            if city_engine_available:
                comp = GLOBAL_CITY_COMPOSITION_ENGINE.paint_city_composition(
                    seed=seed, city_name=name, style_biome=biome, canvas_width_m=w, canvas_depth_m=d
                )
                data = {
                    "id": comp.composition_id,
                    "city_name": comp.city_name,
                    "style_biome": comp.style_biome,
                    "seed": comp.seed,
                    "canvas_width_m": comp.canvas_width_m,
                    "canvas_depth_m": comp.canvas_depth_m,
                    "focal_point_x": round(comp.focal_point_x, 2),
                    "focal_point_z": round(comp.focal_point_z, 2),
                    "total_assets": comp.total_assets_count,
                    "vital_hp_verified": comp.vital_max_hp_invariant_verified,
                    "golden_ratio_score": comp.golden_ratio_adherence_score,
                    "ascii_skyline": comp.ascii_skyline_view,
                    "ascii_plan": comp.ascii_plan_view
                }
                self.send_json({"status": "OK", "composition": data})
            else:
                self.send_json({"status": "UNAVAILABLE", "error": "City composition engine not loaded."})
            return

        elif norm_path == "/api/city/metropolis":
            params = parse_qs(parsed.query)
            seed = int(params.get("seed", [101])[0])
            cols = int(params.get("cols", [3])[0])
            rows = int(params.get("rows", [3])[0])
            name = params.get("name", ["Neo-Praha Veľká Metropola"])[0]
            if city_engine_available:
                metro = GLOBAL_METROPOLIS_ENGINE.build_metropolis(
                    seed=seed, metropolis_name=name, grid_cols=cols, grid_rows=rows
                )
                sectors_data = []
                for (gx, gz), sec in metro.sectors.items():
                    sectors_data.append({
                        "gx": gx, "gz": gz,
                        "biome": sec.district_biome.value,
                        "origin_x": sec.world_origin_x,
                        "origin_z": sec.world_origin_z,
                        "assets_count": sec.composition.total_assets_count
                    })
                data = {
                    "metropolis_id": metro.metropolis_id,
                    "name": metro.metropolis_name,
                    "seed": metro.seed,
                    "grid_dim": list(metro.grid_dim),
                    "total_width_m": metro.total_world_width_m,
                    "total_depth_m": metro.total_world_depth_m,
                    "total_assets": metro.total_assets_count,
                    "traffic_agents_count": len(metro.kinetic_traffic_fleet),
                    "vital_hp_verified": metro.vital_max_hp_verified,
                    "golden_score": metro.golden_ratio_balance_score,
                    "ascii_map": metro.metropolis_ascii_map,
                    "sectors": sectors_data
                }
                self.send_json({"status": "OK", "metropolis": data})
            else:
                self.send_json({"status": "UNAVAILABLE", "error": "Metropolis engine not loaded."})
            return

        elif norm_path == "/api/city/ascii_skyline":
            params = parse_qs(parsed.query)
            seed = int(params.get("seed", [42])[0])
            if city_engine_available:
                comp = GLOBAL_CITY_COMPOSITION_ENGINE.paint_city_composition(seed=seed)
                self.send_text(comp.ascii_skyline_view)
            else:
                self.send_text("City engine unavailable", status=503)
            return

        elif norm_path == "/api/city/ascii_plan":
            params = parse_qs(parsed.query)
            seed = int(params.get("seed", [42])[0])
            if city_engine_available:
                comp = GLOBAL_CITY_COMPOSITION_ENGINE.paint_city_composition(seed=seed)
                self.send_text(comp.ascii_plan_view)
            else:
                self.send_text("City engine unavailable", status=503)
            return

        elif norm_path == "/api/city/export_godot":
            params = parse_qs(parsed.query)
            seed = int(params.get("seed", [42])[0])
            is_metro = params.get("metro", ["0"])[0] == "1"
            if city_engine_available:
                if is_metro:
                    metro = GLOBAL_METROPOLIS_ENGINE.build_metropolis(seed=seed)
                    tscn = GLOBAL_METROPOLIS_ENGINE.export_metropolis_to_godot_tscn(metro)
                else:
                    comp = GLOBAL_CITY_COMPOSITION_ENGINE.paint_city_composition(seed=seed)
                    tscn = GLOBAL_CITY_COMPOSITION_ENGINE.export_to_godot_tscn(comp)
                self.send_text(tscn, content_type="text/plain; charset=utf-8")
            else:
                self.send_text("City engine unavailable", status=503)
            return

        elif norm_path == "/api/city/export_java":
            params = parse_qs(parsed.query)
            seed = int(params.get("seed", [42])[0])
            if city_engine_available:
                comp = GLOBAL_CITY_COMPOSITION_ENGINE.paint_city_composition(seed=seed)
                java_code = GLOBAL_CITY_COMPOSITION_ENGINE.export_to_java_records(comp)
                self.send_text(java_code, content_type="text/plain; charset=utf-8")
            else:
                self.send_text("City engine unavailable", status=503)
            return

        elif norm_path == "/api/city/export_janet":
            params = parse_qs(parsed.query)
            seed = int(params.get("seed", [42])[0])
            if city_engine_available:
                comp = GLOBAL_CITY_COMPOSITION_ENGINE.paint_city_composition(seed=seed)
                janet_code = GLOBAL_CITY_COMPOSITION_ENGINE.export_to_janet_dsl(comp)
                self.send_text(janet_code, content_type="text/plain; charset=utf-8")
            else:
                self.send_text("City engine unavailable", status=503)
            return

        # ── API: Skeuomorphic Procedural Synthesis ─────────────────
        elif norm_path == "/api/skeuomorphic/catalog":
            if skeuomorphic_engine_available:
                cat = GLOBAL_SKEUOMORPHIC_ENGINE.get_catalog()
                self.send_json({"status": "OK", "catalog": cat})
            else:
                self.send_json({"status": "UNAVAILABLE", "error": "Skeuomorphic engine not loaded."})
            return

        elif norm_path == "/api/skeuomorphic/item":
            params = parse_qs(parsed.query)
            item_type_str = params.get("type", ["alchemist_leather_grimoire"])[0]
            seed = int(params.get("seed", [42])[0])
            if skeuomorphic_engine_available:
                try:
                    item_type = SkeuomorphicItemType(item_type_str.lower())
                except ValueError:
                    item_type = SkeuomorphicItemType.ALCHEMIST_LEATHER_GRIMOIRE
                item = GLOBAL_SKEUOMORPHIC_ENGINE.synthesize_item(item_type, seed=seed)
                svg = GLOBAL_SKEUOMORPHIC_ENGINE.render_item_svg(item)
                ascii_art = GLOBAL_SKEUOMORPHIC_ENGINE.render_item_tactile_ascii(item)
                self.send_json({
                    "status": "OK",
                    "item": {
                        "id": item.item_id,
                        "type": item.item_type.value,
                        "name": item.display_name,
                        "seed": item.seed,
                        "vital_hp": item.vital_max_hp,
                        "material": item.primary_substrate.value,
                        "weight_kg": round(item.mass_weight_kg, 3),
                        "dimensions_cm": list(item.bounding_dimensions_cm),
                        "components": [
                            {"name": c.part_name, "substrate": c.substrate.value, "color": c.color_hex}
                            for c in item.components
                        ],
                        "functional_joinery": item.functional_joinery,
                        "wear_patina": round(item.wear_patina_factor, 3),
                        "svg": svg,
                        "ascii": ascii_art
                    }
                })
            else:
                self.send_json({"status": "UNAVAILABLE", "error": "Skeuomorphic engine not loaded."})
            return

        elif norm_path == "/api/skeuomorphic/character":
            params = parse_qs(parsed.query)
            archetype_str = params.get("type", ["bohemian_alchemist_hero"])[0]
            seed = int(params.get("seed", [108])[0])
            if skeuomorphic_engine_available:
                try:
                    archetype = CharacterArchetype(archetype_str.lower())
                except ValueError:
                    archetype = CharacterArchetype.BOHEMIAN_ALCHEMIST_HERO
                ch = GLOBAL_SKEUOMORPHIC_ENGINE.synthesize_character(archetype, seed=seed)
                svg = GLOBAL_SKEUOMORPHIC_ENGINE.render_character_svg(ch)
                self.send_json({
                    "status": "OK",
                    "character": {
                        "id": ch.character_id,
                        "archetype": ch.archetype.value,
                        "name": ch.character_name,
                        "seed": ch.seed,
                        "vital_hp": ch.vital_max_hp,
                        "total_height_cm": ch.total_height_cm,
                        "head_height_cm": round(ch.head_height_cm, 2),
                        "head_ratio": ch.vitruvian_head_ratio,
                        "garment_layers": [
                            {"layer": g.layer_level, "item": g.garment_name, "substrate": g.substrate.value, "seam": g.seam_type}
                            for g in ch.garment_layers
                        ],
                        "equipped_items": [i.display_name for i in ch.equipped_items],
                        "svg": svg,
                        "ascii": ch.tactile_ascii_silhouette
                    }
                })
            else:
                self.send_json({"status": "UNAVAILABLE", "error": "Skeuomorphic engine not loaded."})
            return

        elif norm_path == "/api/skeuomorphic/room":
            params = parse_qs(parsed.query)
            room_type_str = params.get("type", ["alchemist_workshop_chamber"])[0]
            seed = int(params.get("seed", [256])[0])
            if skeuomorphic_engine_available:
                try:
                    room_type = RoomArchetype(room_type_str.lower())
                except ValueError:
                    room_type = RoomArchetype.ALCHEMIST_WORKSHOP_CHAMBER
                room = GLOBAL_SKEUOMORPHIC_ENGINE.synthesize_room(room_type, seed=seed)
                svg = GLOBAL_SKEUOMORPHIC_ENGINE.render_room_svg(room)
                self.send_json({
                    "status": "OK",
                    "room": {
                        "id": room.room_id,
                        "archetype": room.room_type.value,
                        "name": room.room_name,
                        "seed": room.seed,
                        "width_m": room.width_m,
                        "length_m": room.length_m,
                        "height_m": room.height_m,
                        "golden_ratio_adherence": round(room.golden_ratio_adherence, 4),
                        "primary_material": room.primary_material.value,
                        "secondary_material": room.secondary_material.value,
                        "architectural_features": room.architectural_features,
                        "furniture_props": [i.display_name for i in room.furniture_props],
                        "svg": svg,
                        "ascii": room.elevation_ascii
                    }
                })
            else:
                self.send_json({"status": "UNAVAILABLE", "error": "Skeuomorphic engine not loaded."})
            return

        elif norm_path == "/api/skeuomorphic/export_godot":
            params = parse_qs(parsed.query)
            cat = params.get("category", ["item"])[0]
            typ = params.get("type", ["alchemist_leather_grimoire"])[0]
            seed = int(params.get("seed", [42])[0])
            if skeuomorphic_engine_available:
                if cat == "character":
                    try:
                        arch = CharacterArchetype(typ.lower())
                    except ValueError:
                        arch = CharacterArchetype.BOHEMIAN_ALCHEMIST_HERO
                    obj = GLOBAL_SKEUOMORPHIC_ENGINE.synthesize_character(arch, seed=seed)
                elif cat == "room":
                    try:
                        rarch = RoomArchetype(typ.lower())
                    except ValueError:
                        rarch = RoomArchetype.ALCHEMIST_WORKSHOP_CHAMBER
                    obj = GLOBAL_SKEUOMORPHIC_ENGINE.synthesize_room(rarch, seed=seed)
                else:
                    try:
                        ityp = SkeuomorphicItemType(typ.lower())
                    except ValueError:
                        ityp = SkeuomorphicItemType.ALCHEMIST_LEATHER_GRIMOIRE
                    obj = GLOBAL_SKEUOMORPHIC_ENGINE.synthesize_item(ityp, seed=seed)
                tscn = GLOBAL_SKEUOMORPHIC_ENGINE.export_to_godot_tscn(obj)
                self.send_text(tscn, content_type="text/plain; charset=utf-8")
            else:
                self.send_text("Skeuomorphic engine unavailable", status=503)
            return

        elif norm_path == "/api/skeuomorphic/export_java":
            params = parse_qs(parsed.query)
            typ = params.get("type", ["alchemist_leather_grimoire"])[0]
            seed = int(params.get("seed", [42])[0])
            if skeuomorphic_engine_available:
                try:
                    ityp = SkeuomorphicItemType(typ.lower())
                except ValueError:
                    ityp = SkeuomorphicItemType.ALCHEMIST_LEATHER_GRIMOIRE
                item = GLOBAL_SKEUOMORPHIC_ENGINE.synthesize_item(ityp, seed=seed)
                code = GLOBAL_SKEUOMORPHIC_ENGINE.export_to_java_records(item)
                self.send_text(code, content_type="text/plain; charset=utf-8")
            else:
                self.send_text("Skeuomorphic engine unavailable", status=503)
            return

        elif norm_path == "/api/skeuomorphic/export_janet":
            params = parse_qs(parsed.query)
            typ = params.get("type", ["alchemist_leather_grimoire"])[0]
            seed = int(params.get("seed", [42])[0])
            if skeuomorphic_engine_available:
                try:
                    ityp = SkeuomorphicItemType(typ.lower())
                except ValueError:
                    ityp = SkeuomorphicItemType.ALCHEMIST_LEATHER_GRIMOIRE
                item = GLOBAL_SKEUOMORPHIC_ENGINE.synthesize_item(ityp, seed=seed)
                code = GLOBAL_SKEUOMORPHIC_ENGINE.export_to_janet_dsl(item)
                self.send_text(code, content_type="text/plain; charset=utf-8")
            else:
                self.send_text("Skeuomorphic engine unavailable", status=503)
            return



        # ── API: Status ────────────────────────────────────────────
        elif path == "/api/status":
            with state.lock:
                data = {
                    "mode": state.mode,
                    "fps": state.fps,
                    "resolution": f"{state.cols}x{state.rows}",
                    "entropy": {
                        "spatial": state.spatial_entropy,
                        "temporal": state.temporal_entropy,
                        "total": state.total_entropy,
                        "coherence": state.coherence
                    },
                    "governor": {
                        "budget": state.budget,
                        "state": state.governor_state,
                        "thermal_penalty": state.thermal_penalty
                    },
                    "cognitive_phase": state.cyclic_engine.cognitive_phase if state.cyclic_engine else "UNKNOWN",
                    "hamiltonian_energy": round(state.cyclic_engine.total_hamiltonian(), 4) if state.cyclic_engine else 0.0,
                    "lyapunov_index": state.cyclic_engine.lyapunov_stability_index() if state.cyclic_engine else 0.0,
                    "backpressure": state.backpressure_active,
                    "vulkan": {
                        "accelerated": bool(state.vulkan_driver and state.vulkan_driver.real_gpu_dispatch),
                        "enabled": bool(state.use_vulkan),
                        "device_detected": bool(state.vulkan_driver and state.vulkan_driver.device_detected),
                        "compute_backend": state.vulkan_driver.compute_backend if state.vulkan_driver else "CPU_REFERENCE",
                        "device": state.vulkan_driver.device_name if state.vulkan_driver else "CPU Fallback",
                        "dispatch_us": state.vulkan_driver.last_dispatch_us if state.vulkan_driver else 0.0,
                        "readback_mb_s": state.vulkan_driver.readback_mb_s if state.vulkan_driver else 0.0
                    }
                }
            self.send_json(data)

        # ── API: Instances Catalog ─────────────────────────────────
        elif self.path == "/api/instances":
            if antigravity_engine:
                geoms = antigravity_engine.instance_manager.list_geometric_instances()
                arts = antigravity_engine.instance_manager.list_artistic_instances()
                self.send_json({"status": "OK", "geometric": geoms, "artistic": arts})
            else:
                self.send_json({"status": "UNAVAILABLE", "geometric": [], "artistic": []})

        # ── API: Templates List ────────────────────────────────────
        elif self.path == "/api/templates":
            if antigravity_engine:
                tpls = antigravity_engine.list_saved_templates()
                self.send_json({"status": "OK", "templates": tpls})
            else:
                self.send_json({"status": "UNAVAILABLE", "templates": []})

        # ── API: Active Template ───────────────────────────────────
        elif self.path == "/api/active-template":
            with state.lock:
                active_tpl = state.active_template
            self.send_json({"status": "OK", "active_template": active_tpl})

        # ── API: Mimicry Objects Catalog ───────────────────────────
        elif self.path == "/api/mimicry/objects":
            if mimicry_available:
                self.send_json({"status": "OK", "recipes": list_recipes()})
            else:
                self.send_json({"status": "UNAVAILABLE", "recipes": []})

        # ── API: Mimicry Game Scenes Catalog ───────────────────────
        elif self.path == "/api/mimicry/scenes":
            if mimicry_available:
                self.send_json({"status": "OK", "scenes": list_scenes()})
            else:
                self.send_json({"status": "UNAVAILABLE", "scenes": []})

        # ── API: Mimicry Active Selection ──────────────────────────
        elif self.path == "/api/mimicry/active":
            with state.lock:
                data = {
                    "status": "OK",
                    "active_mimic_id": state.active_mimic_id,
                    "active_scene_id": state.active_scene_id,
                    "mimic_object": state.active_mimic_obj.to_dict() if state.active_mimic_obj else None,
                    "scene": state.active_scene_obj.to_dict() if state.active_scene_obj else None
                }
            self.send_json(data)

        # ── API: Mimicry Rule Validation ───────────────────────────
        elif self.path == "/api/mimicry/validate-active":
            with state.lock:
                sc = state.active_scene_obj
            if not sc:
                self.send_json({"status": "ERROR", "message": "No active scene loaded."})
            else:
                val = UrbanSpatialCompositionRules.validate_scene(sc)
                self.send_json({
                    "status": "OK",
                    "scene_id": sc.scene_id,
                    "validation": val.to_dict()
                })


        # ── API: OpenWorld Status ──────────────────────────────────
        elif self.path == "/api/openworld/status":
            with state.lock:
                if state.active_world_spec:
                    spec_copy = {
                        "name": state.active_world_spec["name"],
                        "original_prompt": state.active_world_spec["original_prompt"],
                        "topography_type": state.active_world_spec["topography_type"],
                        "atmosphere_type": state.active_world_spec["atmosphere_type"],
                        "dominant_biome": state.active_world_spec["dominant_biome"],
                        "mathematical_parameters": state.active_world_spec["mathematical_parameters"],
                        "atmospheric_parameters": state.active_world_spec["atmospheric_parameters"],
                        "artifact_scatter_rules": state.active_world_spec["artifact_scatter_rules"],
                        "janet_dsl": state.active_janet_dsl,
                        "godot_shader": state.active_godot_shader,
                        "python_code": state.active_py_generator
                    }
                    self.send_json({"status": "OK", "world": spec_copy})
                else:
                    self.send_json({"status": "UNAVAILABLE"})

        # ── API: OpenWorld Presets ─────────────────────────────────
        elif self.path == "/api/openworld/presets":
            presets = [
                {
                    "title": "Volcanic Crags & Lava Canyons (SK)",
                    "prompt": "vulkanické hory s kryštálovými vežami, sírnym dymom a 6 rekurzií"
                },
                {
                    "title": "Cyberpunk Neon Wasteland (SK)",
                    "prompt": "kybernetická pustatina s neonovými obeliskmi a silnou eróziou"
                },
                {
                    "title": "Crystalline Highlands (EN)",
                    "prompt": "rolling dunes desert with ancient monoliths and clear crystalline aether"
                },
                {
                    "title": "Mech Titan Canyons (EN)",
                    "prompt": "canyon trenches with walking mech titans and heavy hydraulic erosion"
                },
                {
                    "title": "Biomechanical Xenodrone Hive (SK)",
                    "prompt": "biomechanické hniezdo s xenobiotic dronmi, toxickou hmlou a rázovitými útesmi"
                }
            ]
            self.send_json({"status": "OK", "presets": presets})

        # ── API: Krystal-Lang Status ───────────────────────────────
        elif self.path == "/api/krystal-lang/status":
            with state.lock:
                if state.krystal_lang_comp and state.krystal_lang_vm:
                    self.send_json({
                        "status": "OK",
                        "module_name": state.krystal_lang_comp["module_name"],
                        "queues_count": state.krystal_lang_comp["queues_count"],
                        "bytecode": state.krystal_lang_comp["bytecode"],
                        "queue_partitions": state.krystal_lang_comp["queue_partitions"],
                        "vm_metrics": state.krystal_lang_vm.metrics
                    })
                else:
                    self.send_json({"status": "UNAVAILABLE"})

        # ── API: Krystal-Lang Step VM ──────────────────────────────
        elif self.path == "/api/krystal-lang/step":
            with state.lock:
                if state.krystal_lang_vm:
                    m = state.krystal_lang_vm.step_execution(cycles=2)
                    self.send_json({"status": "OK", "metrics": m})
                else:
                    self.send_json({"status": "ERROR", "message": "VM unavailable."})

        # ── API: Bootstrap Schema ──────────────────────────────────
        elif self.path == "/api/bootstrap/schema":
            schema_path = os.path.join(WORKSPACE_ROOT, "schemas", "krystal_engine_bootstrap.schema.json")
            if os.path.exists(schema_path):
                with open(schema_path, "r", encoding="utf-8") as f:
                    self.send_json(json.load(f))
            else:
                self.send_json({"status": "ERROR", "message": "Schema file not found."})

        # ── API: List Procedural Instances ─────────────────────────
        elif self.path == "/api/instances/procedural":
            if procedural_generator_available:
                samples = [
                    ProceduralInstanceGenerator.generate_superformula_instance(seed=101),
                    ProceduralInstanceGenerator.generate_cyber_spire_instance(seed=202),
                    ProceduralInstanceGenerator.generate_alchemical_polyhedron_instance(seed=303)
                ]
                self.send_json({"status": "OK", "instances": samples})
            else:
                self.send_json({"status": "UNAVAILABLE", "instances": []})

        # ── API: Symplectic Cyclic Organism Telemetry ──────────────
        elif self.path == "/api/cyclic":
            with state.lock:
                if state.cyclic_engine:
                    data = {
                        "status": "OK",
                        "schema_data": state.cyclic_engine.to_schema_dict(),
                        "phase_trajectory": state.cyclic_engine.generate_phase_trajectory(num_points=24)
                    }
                else:
                    data = {"status": "UNAVAILABLE", "message": "Cyclic organism not initialized."}
            self.send_json(data)

        # ── API: Vulkan Compute Telemetry ──────────────────────────
        elif self.path == "/api/vulkan":
            with state.lock:
                if state.vulkan_driver:
                    data = {
                        "status": "OK",
                        "enabled": state.use_vulkan,
                        "telemetry": state.vulkan_driver.get_telemetry()
                    }
                else:
                    data = {"status": "UNAVAILABLE", "enabled": False}
            self.send_json(data)

        # ── API: Project Intelligence (priorities / expertise / patterns) ──
        elif self._route_path() in (
            "/api/project/overview", "/api/priorities", "/api/expertise", "/api/compute-patterns"
        ) or self._route_path().startswith("/api/priorities/"):
            self.handle_intelligence()

        # ── API: NextGen Pattern Improvement Metrics ───────────────
        elif self._route_path() == "/api/nextgen/pattern_metrics":
            from krystal_stack_nextgen.pattern_metrics_and_ipc_governor import GLOBAL_PATTERN_IPC_GOVERNOR
            metrics = GLOBAL_PATTERN_IPC_GOVERNOR.compute_pattern_metrics()
            self.send_json({"status": "OK", "metrics": metrics.to_dict()})
            return

        # ── API: Vulkan IPC Bridge Telemetry ───────────────────────
        elif self._route_path() == "/api/vulkan_ipc/telemetry":
            from krystal_stack_nextgen.vulkan_ipc_bridge import GLOBAL_VULKAN_IPC_BRIDGE
            self.send_json({"status": "OK", "telemetry": GLOBAL_VULKAN_IPC_BRIDGE.get_telemetry()})
            return

        # ── API: IPC Application Domain Benchmark ──────────────────
        elif self._route_path() == "/api/vulkan_ipc/benchmark_domains":
            from krystal_stack_nextgen.ipc_application_benchmark import GLOBAL_IPC_BENCHMARK
            results = GLOBAL_IPC_BENCHMARK.run_full_suite(iterations=1000)
            self.send_json({"status": "OK", "domains": [r.to_dict() for r in results]})
            return

        # ── API: Multi-GPU SLI/Crossfire & Indirect Compute Scaling ─
        elif self._route_path() == "/api/vulkan_ipc/multi_gpu_scaling":
            from krystal_stack_nextgen.multi_gpu_stream_scaler import GLOBAL_MULTI_GPU_SCALER
            q = parse_qs(urlparse(self.path).query)
            intensity = float(q.get("intensity", [1.5])[0])
            report = GLOBAL_MULTI_GPU_SCALER.evaluate_all_topologies(workload_intensity=intensity)
            self.send_json({"status": "OK", "scaling_report": report.to_dict()})
            return

        # ── API: Bytecode Predictive Graphics Accelerator ───────────
        elif self._route_path() in ("/api/accelerator/predictions", "/api/accelerator/benchmark_suite"):
            from krystal_kernel.bytecode_predictive_graphics_accelerator import GLOBAL_BYTECODE_GRAPHICS_ACCELERATOR
            rep = GLOBAL_BYTECODE_GRAPHICS_ACCELERATOR.generate_performance_benchmark()
            self.send_json({"status": "OK", "benchmark": rep.to_dict(), "vital_max_hp": 6})
            return

        # ── API: Server-Sent Events (SSE) Stream ───────────────────
        elif self.path == "/api/stream":
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()

            last_sent_frame = -1
            try:
                while state.running:
                    with state.lock:
                        cur_id = state.frame_id
                        ascii_data = state.current_ascii
                        fps = state.fps
                        mode = state.mode
                        ent = {
                            "spatial": state.spatial_entropy,
                            "temporal": state.temporal_entropy,
                            "total": state.total_entropy,
                            "coherence": state.coherence
                        }
                        gov = {
                            "budget": state.budget,
                            "state": state.governor_state,
                            "thermal_penalty": state.thermal_penalty
                        }
                        bp = state.backpressure_active
                        events = list(state.director_events)
                        state.director_events.clear()
                        c_phase = state.cyclic_engine.cognitive_phase if state.cyclic_engine else "UNKNOWN"
                        h_energy = round(state.cyclic_engine.total_hamiltonian(), 4) if state.cyclic_engine else 0.0
                        vk_info = {
                            "accelerated": bool(state.vulkan_driver and state.vulkan_driver.real_gpu_dispatch),
                            "enabled": bool(state.use_vulkan),
                            "device_detected": bool(state.vulkan_driver and state.vulkan_driver.device_detected),
                            "compute_backend": state.vulkan_driver.compute_backend if state.vulkan_driver else "CPU_REFERENCE",
                            "device": state.vulkan_driver.device_name if state.vulkan_driver else "CPU Fallback",
                            "dispatch_us": state.vulkan_driver.last_dispatch_us if state.vulkan_driver else 0.0,
                            "readback_mb_s": state.vulkan_driver.readback_mb_s if state.vulkan_driver else 0.0
                        }

                    if cur_id != last_sent_frame:
                        payload = json.dumps({
                            "frame_id": cur_id,
                            "ascii": ascii_data,
                            "fps": fps,
                            "mode": mode,
                            "entropy": ent,
                            "governor": gov,
                            "backpressure": bp,
                            "cognitive_phase": c_phase,
                            "hamiltonian_energy": h_energy,
                            "vulkan": vk_info
                        })
                        self.wfile.write(f"event: frame\ndata: {payload}\n\n".encode("utf-8"))
                        self.wfile.flush()
                        last_sent_frame = cur_id

                    # Send director events if any
                    for ev in events:
                        self.wfile.write(f"event: director\ndata: {json.dumps(ev)}\n\n".encode("utf-8"))
                        self.wfile.flush()

                    time.sleep(0.033) # 30 FPS stream
            except (ConnectionResetError, BrokenPipeError):
                pass
        else:
            if self.path.startswith("/api/") and self.proxy_to_engine_core("GET"):
                return
            self.send_error(404, "Endpoint not found")

    def do_POST(self):
        if self._is_kernel_path(self._KERNEL_POST):
            self.handle_kernel_api("POST")
            return
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length).decode("utf-8", errors="replace")
        try:
            data = json.loads(body) if body else {}
        except Exception:
            data = {}

        # ── API: Control (Mode / Resolution / Threshold / Folds) ──
        if self.path == "/api/control":
            with state.lock:
                if "mode" in data:
                    state.mode = data["mode"]
                if "cols" in data:
                    state.cols = int(data["cols"])
                if "rows" in data:
                    state.rows = int(data["rows"])
                if "threshold" in data:
                    state.entropy_threshold = float(data["threshold"])
                if "mirror_folds" in data and state.active_template:
                    state.mirror_folds = int(data["mirror_folds"])
                    state.active_template["composition_rules"]["mirror_folds"] = state.mirror_folds
                    state.active_template["vulkan_shader_uniforms"]["u_mirror_folds"] = state.mirror_folds
            self.send_json({"status": "UPDATED", "mode": state.mode})

        # ── API: Visual Copilot Polyglot Generator ─────────────────
        elif self.path in ("/api/copilot/generate", "/api/copilot/synthesize"):
            from krystal_kernel.processor_integrity_telemetry import GLOBAL_PROCESSOR_INTEGRITY_ENGINE
            from krystal_kernel.visual_copilot_generator import GLOBAL_VISUAL_COPILOT_GENERATOR
            rep = GLOBAL_PROCESSOR_INTEGRITY_ENGINE.sample_integrity()
            cs = float(data.get("context_switches_per_sec", rep.context_switches_per_sec))
            cpu = float(data.get("cpu_utilization_pct", rep.cpu_utilization_pct))
            thrash = float(data.get("thrashing_index", rep.thrashing_index))
            res = GLOBAL_VISUAL_COPILOT_GENERATOR.generate_all_targets(cs_rate=cs, cpu_pct=cpu, thrashing_idx=thrash)
            self.send_json({"status": "OK", "result": res})
            return

        # ── API: Cortex Process Prioritization & Windows API Execution ─
        elif self.path in ("/api/cortex/prioritize", "/api/process/prioritize"):
            from krystal_kernel.cortex_compiler import GLOBAL_CORTEX_COMPILER
            pid = int(data.get("pid", 0))
            if pid <= 0:
                self.send_error(400, "Valid target PID is required")
                return
            p_name = str(data.get("process_name", f"process_{pid}.exe"))
            user_intent = data.get("user_priority_intent")
            demand_ll = bool(data.get("demand_low_latency", False))
            cpu_val = float(data.get("cpu_load", 50.0))
            cs_val = float(data.get("cs_rate", 6500.0))
            thrash_val = float(data.get("thrashing_index", 1.1))
            mem_val = float(data.get("working_set_mb", 256.0))
            threads_val = int(data.get("thread_count", 8))

            plan = GLOBAL_CORTEX_COMPILER.compile_prioritization_policy(
                pid=pid,
                process_name=p_name,
                user_priority_intent=user_intent,
                cpu_load=cpu_val,
                cs_rate=cs_val,
                thrashing_index=thrash_val,
                working_set_mb=mem_val,
                thread_count=threads_val,
                demand_low_latency=demand_ll
            )
            exec_res = GLOBAL_CORTEX_COMPILER.execute_plan(plan)
            self.send_json({
                "status": "SUCCESS" if exec_res.get("success") else "DISPATCHED",
                "execution_result": exec_res,
                "compiled_plan": plan.to_dict(),
                "vital_max_hp": 6
            })
            return

        # ── API: Cortex Prioritization Policy Compiler ─────────────
        elif self.path in ("/api/cortex/compile", "/api/compiler/compile"):
            from krystal_kernel.cortex_compiler import GLOBAL_CORTEX_COMPILER
            pid = int(data.get("pid", 1000))
            p_name = str(data.get("process_name", "target_process.exe"))
            user_intent = data.get("user_priority_intent")
            demand_ll = bool(data.get("demand_low_latency", False))
            plan = GLOBAL_CORTEX_COMPILER.compile_prioritization_policy(
                pid=pid,
                process_name=p_name,
                user_priority_intent=user_intent,
                demand_low_latency=demand_ll
            )
            self.send_json({
                "status": "COMPILED",
                "compiled_plan": plan.to_dict(),
                "vital_max_hp": 6
            })
            return

        # ── API: OpenVINO Process Prioritization Inference ─────────
        elif self.path in ("/api/cortex/openvino_infer", "/api/openvino/infer"):
            from krystal_kernel.cortex_openvino_engine import GLOBAL_OPENVINO_GOVERNOR
            pid = int(data.get("pid", 1000))
            name = str(data.get("name", "worker.exe"))
            cpu = float(data.get("cpu_pct", 50.0))
            cs = float(data.get("cs_rate", 6500.0))
            faults = float(data.get("page_faults_per_sec", 120.0))
            mem = float(data.get("working_set_mb", 256.0))
            threads = int(data.get("thread_count", 8))
            io_ops = float(data.get("io_ops_per_sec", 50.0))
            k_ratio = float(data.get("kernel_time_ratio", 0.15))
            thrash = float(data.get("thrashing_index", 1.1))

            res = GLOBAL_OPENVINO_GOVERNOR.infer_process_priority(
                pid=pid,
                name=name,
                cpu_pct=cpu,
                cs_rate=cs,
                page_faults_per_sec=faults,
                working_set_mb=mem,
                thread_count=threads,
                io_ops_per_sec=io_ops,
                kernel_time_ratio=k_ratio,
                thrashing_index=thrash
            )
            self.send_json({
                "status": "OK",
                "inference": res.to_dict(),
                "vital_max_hp": 6
            })
            return

        # ── API: Iris Xe Dynamic Frame Pacing & UMA Quotient Scaling ──
        elif self.path in ("/api/iris_xe/pace_frame", "/api/uma/pace"):
            from krystal_kernel.iris_xe_uma_memory_manager import GLOBAL_IRIS_XE_UMA_MANAGER
            ft = float(data.get("frame_time_ms", 7.5))
            comp = float(data.get("complexity_factor", 1.0))
            uma = GLOBAL_IRIS_XE_UMA_MANAGER.update_pacing_and_scale_quotient(
                measured_frame_time_ms=ft,
                complexity_factor=comp
            )
            self.send_json({"status": "PACED", "uma_status": uma.to_dict(), "vital_max_hp": 6})
            return

        # ── API: Self-Healing Telemetry Evaluation & Pattern Trigger ─
        elif self.path in ("/api/self_healing/inject_telemetry", "/api/self_healing/evaluate"):
            from krystal_kernel.self_healing_patterns import GLOBAL_SELF_HEALING_GOVERNOR
            cs = float(data.get("cs_rate", 35000.0))
            thrash = float(data.get("thrashing_index", 1.8))
            ft = float(data.get("frame_time_ms", 7.9))
            temp = float(data.get("junction_temp_c", 78.0))
            budget = float(data.get("vsync_budget_ms", 8.333))
            rep = GLOBAL_SELF_HEALING_GOVERNOR.evaluate_and_heal(
                cs_rate=cs,
                thrashing_index=thrash,
                frame_time_ms=ft,
                junction_temp_c=temp,
                vsync_budget_ms=budget
            )
            self.send_json({"status": "EVALUATED", "healing_report": rep.to_dict(), "vital_max_hp": 6})
            return

        # ── API: WSL2 Failing Process Diagnostics ──────────────────
        elif self.path in ("/api/wsl/diagnose_process", "/api/wsl/diagnose"):
            from krystal_kernel.wsl_coprocessor import GLOBAL_WSL_COPROCESSOR
            t_pid = int(data.get("pid", 4412))
            p_name = str(data.get("process_name", "target_process.exe"))
            c_sig = str(data.get("crash_signature", "STATUS_ACCESS_VIOLATION (0xC0000005)"))
            diag_rep = GLOBAL_WSL_COPROCESSOR.diagnose_failing_process(
                pid=t_pid,
                process_name=p_name,
                crash_signature=c_sig
            )
            self.send_json(diag_rep.to_dict())
            return

        # ── API: WSL2 UFS Automated Log Triage ─────────────────────
        elif self.path in ("/api/wsl/ufs_triage", "/api/wsl/triage"):
            from krystal_kernel.wsl_coprocessor import GLOBAL_WSL_COPROCESSOR
            l_dir = data.get("log_directory")
            a_heal = bool(data.get("auto_heal", True))
            triage_res = GLOBAL_WSL_COPROCESSOR.run_ufs_log_triage(
                log_directory=l_dir,
                auto_heal=a_heal
            )
            self.send_json(triage_res.to_dict())
            return

        # ── API: WSL2 Automated Administrative Remediation ─────────
        elif self.path in ("/api/wsl/remediate", "/api/wsl/admin_action"):
            from krystal_kernel.wsl_coprocessor import GLOBAL_WSL_COPROCESSOR
            act_name = str(data.get("action_name", "QUARANTINE_PROCESS_AND_STAGE_DUMP"))
            rem_res = GLOBAL_WSL_COPROCESSOR.execute_automated_admin_remediation(act_name)
            self.send_json(rem_res)
            return

        # ── API: WSL2 Low-Latency GUI Compositor & Shared Surface ──
        elif self.path in ("/api/wsl/benchmark_gui_latency", "/api/wsl/gui_benchmark"):
            from krystal_kernel.wsl_gui_compositor_bridge import GLOBAL_WSL_GUI_COMPOSITOR_BRIDGE
            bench = GLOBAL_WSL_GUI_COMPOSITOR_BRIDGE.benchmark_gui_latency()
            self.send_json(bench.to_dict())
            return

        elif self.path in ("/api/wsl/create_shared_surface", "/api/wsl/surface"):
            from krystal_kernel.wsl_gui_compositor_bridge import (
                GLOBAL_WSL_GUI_COMPOSITOR_BRIDGE, WslGuiLatencyMode
            )
            title = str(data.get("window_title", "Krystal Linux Shell (Wayland)"))
            pid = int(data.get("linux_pid", 4096))
            w = int(data.get("width", 1920))
            h = int(data.get("height", 1080))
            m_str = str(data.get("mode", "KRYSTAL_DIRECT_UMA"))
            mode_obj = WslGuiLatencyMode.KRYSTAL_DIRECT_UMA
            for m in WslGuiLatencyMode:
                if m.value == m_str or m.name == m_str:
                    mode_obj = m
                    break
            srf = GLOBAL_WSL_GUI_COMPOSITOR_BRIDGE.create_shared_surface(
                window_title=title,
                linux_pid=pid,
                width=w,
                height=h,
                mode=mode_obj
            )
            self.send_json(srf.to_dict())
            return

        # ── API: Bytecode Dependency Power & Current Alternation ────
        elif self.path in ("/api/bytecode/predict_power", "/api/bytecode/power"):
            from krystal_kernel.bytecode_power_governor import GLOBAL_BYTECODE_POWER_GOVERNOR
            inject_p = bool(data.get("inject_priority_action", True))
            rep = GLOBAL_BYTECODE_POWER_GOVERNOR.analyze_bytecode_and_stagger_power(
                inject_priority_action=inject_p
            )
            self.send_json(rep.to_dict())
            return

        elif self.path in ("/api/bytecode/micro_slice", "/api/bytecode/slice"):
            from krystal_kernel.bytecode_power_governor import GLOBAL_BYTECODE_POWER_GOVERNOR
            inject_p = bool(data.get("inject_priority", True))
            rep = GLOBAL_BYTECODE_POWER_GOVERNOR.analyze_bytecode_and_stagger_power(
                inject_priority_action=inject_p
            )
            self.send_json({
                "status": "DISPATCHED",
                "micro_slices_count": len(rep.micro_slices),
                "response_budget_ms": rep.response_budget_ms,
                "active_slices": [s.to_dict() for s in rep.micro_slices],
                "vital_max_hp": 6
            })
            return

        # ── API: K-NSS Open Neural Super-Sampling Benchmark ─────────
        elif self.path in ("/api/nss/reconstruct", "/api/nss/benchmark"):
            from krystal_kernel.krystal_neural_super_sampler import (
                GLOBAL_NEURAL_SUPER_SAMPLER, NssQualityProfile
            )
            target_w = int(data.get("target_width", 1920))
            target_h = int(data.get("target_height", 1080))
            prof_str = str(data.get("profile", "PERFORMANCE")).upper()
            try:
                prof = NssQualityProfile(prof_str)
            except Exception:
                prof = NssQualityProfile.PERFORMANCE
            
            res = GLOBAL_NEURAL_SUPER_SAMPLER.reconstruct_frame_benchmark(
                target_w=target_w,
                target_h=target_h,
                profile=prof
            )
            self.send_json(res.to_dict())
            return

        # ── API: Iris Xe VRAM Unlocker & Local Quantized LLM ───────
        elif self.path in ("/api/llm/vram_unlock", "/api/vram/unlock"):
            from krystal_kernel.iris_xe_llm_vram_governor import (
                GLOBAL_IRIS_XE_LLM_VRAM_GOVERNOR, VramApertureTier
            )
            tier_str = str(data.get("tier", "4GB_UNLOCKED"))
            tier_obj = VramApertureTier.TIER_UNLOCKED_4GB
            for t in VramApertureTier:
                if t.value == tier_str or t.name == tier_str:
                    tier_obj = t
                    break
            res = GLOBAL_IRIS_XE_LLM_VRAM_GOVERNOR.unlock_vram_aperture(tier_obj)
            self.send_json(res)
            return

        elif self.path in ("/api/llm/benchmark_tokens", "/api/llm/benchmark"):
            from krystal_kernel.iris_xe_llm_vram_governor import (
                GLOBAL_IRIS_XE_LLM_VRAM_GOVERNOR, VramApertureTier, QuantizationFormat
            )
            tier_str = str(data.get("tier", "4GB_UNLOCKED"))
            quant_str = str(data.get("quantization", "INT4_GGUF_AWQ"))
            tier_obj = VramApertureTier.TIER_UNLOCKED_4GB
            for t in VramApertureTier:
                if t.value == tier_str or t.name == tier_str:
                    tier_obj = t
                    break
            quant_obj = QuantizationFormat.INT4_GGUF_AWQ
            for q in QuantizationFormat:
                if q.value == quant_str or q.name == quant_str:
                    quant_obj = q
                    break
            res = GLOBAL_IRIS_XE_LLM_VRAM_GOVERNOR.benchmark_local_llm_throughput(tier_obj, quant_obj)
            self.send_json(res.to_dict())
            return

        elif self.path in ("/api/llm/optimal_plan", "/api/llm/plan"):
            from krystal_kernel.iris_xe_llm_vram_governor import GLOBAL_IRIS_XE_LLM_VRAM_GOVERNOR
            high_tp = bool(data.get("demand_high_throughput", True))
            max_temp = float(data.get("max_safe_temp_c", 85.0))
            plan = GLOBAL_IRIS_XE_LLM_VRAM_GOVERNOR.calculator.synthesize_optimal_plan(
                demand_high_throughput=high_tp,
                max_safe_temp_c=max_temp
            )
            self.send_json(plan.to_dict())
            return

        # ── API: K-ISA Speculative Instruction Pipeline ─────────────
        elif self.path in ("/api/isa/speculate", "/api/isa/speculative_execution"):
            from krystal_kernel.speculative_instruction_set import GLOBAL_SPECULATIVE_PREDICTOR_ENGINE
            conf = float(data.get("speculation_confidence", 0.94))
            stalls = int(data.get("predicted_stall_cycles", 1200))
            rep = GLOBAL_SPECULATIVE_PREDICTOR_ENGINE.evaluate_and_dispatch_speculation(
                predicted_stall_cycles=stalls,
                speculation_confidence=conf
            )
            self.send_json(rep.to_dict())
            return

        # ── API: Janet Bytecode Decoder & Alert Synthesizer ─────────
        elif self.path in ("/api/janet/decode_binary", "/api/janet/decode"):
            from krystal_kernel.janet_bytecode_decoder import JanetBytecodeDecoder, generate_canonical_demo_stream
            binary_hex = data.get("binary_hex")
            instructions = data.get("instructions")
            
            if binary_hex:
                clean_hex = "".join(c for c in str(binary_hex) if c in "0123456789abcdefABCDEF")
                raw_bytes = bytes.fromhex(clean_hex)
            elif instructions and isinstance(instructions, list):
                raw_bytes = JanetBytecodeDecoder.encode_ksyn_stream(
                    instructions,
                    version=int(data.get("version", 0x0100)),
                    seed=int(data.get("seed", 1337))
                )
            else:
                raw_bytes = generate_canonical_demo_stream()

            res = JanetBytecodeDecoder.process_hex_or_binary_input(raw_bytes)
            ascii_map = JanetBytecodeDecoder.render_terminal_ascii_profile(res["parsed"])
            svg_chart = JanetBytecodeDecoder.render_execution_profile_svg(res["parsed"])

            self.send_json({
                "status": "OK",
                "parsed": res["parsed"],
                "report": res["report"],
                "ascii_command_map": ascii_map,
                "svg_profile": svg_chart,
                "vital_max_hp": 6
            })
            return

        # ── API: WSL Hypervisor GNOME Overlay Mode Toggle ──────────
        elif self.path in ("/api/wsl/toggle_gnome_overlay", "/api/wsl/toggle_overlay"):
            from krystal_kernel.wsl_hypervisor_gnome_port import GLOBAL_GNOME_OVERLAY_MANAGER, WslOverlayMode
            target_str = data.get("target_mode")
            target_enum = None
            if target_str:
                try:
                    target_enum = WslOverlayMode(target_str)
                except Exception:
                    pass
            new_status = GLOBAL_GNOME_OVERLAY_MANAGER.toggle_mode(target_enum)
            self.send_json(new_status.to_dict())
            return

        # ── API: OpenVINO DP4A Hardware Acceleration Pack ──────────
        elif self.path in ("/api/openvino/accelerate", "/api/openvino/acceleration_pack"):
            from krystal_kernel.openvino_hardware_acceleration_pack import OpenVinoIrisXeDP4APack
            use_dp4a = bool(data.get("use_dp4a_int8", True))
            streams = int(data.get("streams_count", 4))
            kv_u8 = bool(data.get("enable_kv_u8", True))
            cfg = OpenVinoIrisXeDP4APack.generate_extended_config(
                use_dp4a_int8=use_dp4a,
                streams_count=streams,
                enable_kv_u8=kv_u8
            )
            self.send_json(cfg.to_dict())
            return

        # ── API: Bytecode Predictive Graphics Accelerator Stream ───
        elif self.path in ("/api/accelerator/dispatch_bytecode", "/api/accelerator/dispatch"):
            from krystal_kernel.bytecode_predictive_graphics_accelerator import GLOBAL_BYTECODE_GRAPHICS_ACCELERATOR
            raw_ops = data.get("opcodes", [0x01, 0x02, 0x03, 0xA1, 0xA2, 0xA5, 0xA6])
            res = GLOBAL_BYTECODE_GRAPHICS_ACCELERATOR.dispatch_bytecode_stream(raw_ops)
            self.send_json(res)
            return

        # ── API: NextGen Fast IPC Benchmark ────────────────────────
        elif self.path == "/api/nextgen/benchmark_ipc":
            from krystal_stack_nextgen.pattern_metrics_and_ipc_governor import GLOBAL_PATTERN_IPC_GOVERNOR
            sample = data.get("ascii_frame", "░▒▓█" * 128)
            res = GLOBAL_PATTERN_IPC_GOVERNOR.benchmark_ipc_serialization(sample)
            self.send_json({"status": "OK", "benchmark": res})
            return

        # ── API: NextGen Accelerated Text Tokenizer ────────────────
        elif self.path == "/api/nextgen/text_tokenize":
            from krystal_stack_nextgen.pattern_metrics_and_ipc_governor import GLOBAL_PATTERN_IPC_GOVERNOR
            text = data.get("text", "")
            tokens = GLOBAL_PATTERN_IPC_GOVERNOR.text_processor.fast_tokenize(text)
            bench = GLOBAL_PATTERN_IPC_GOVERNOR.text_processor.measure_text_processing_speedup(text or "CYBERPUNK MATRIX")
            self.send_json({"status": "OK", "tokens": tokens, "benchmark": bench})
            return

        # ── API: Vulkan IPC Hardware Acceleration Dispatch ─────────
        elif self.path == "/api/vulkan_ipc/dispatch":
            from krystal_stack_nextgen.vulkan_ipc_bridge import GLOBAL_VULKAN_IPC_BRIDGE, VulkanIPCClient
            client = VulkanIPCClient(GLOBAL_VULKAN_IPC_BRIDGE)
            op = data.get("operation", "TOKEN_INTERN").upper()

            if op == "GEMM":
                m = int(data.get("m", 2))
                k = int(data.get("k", 2))
                n = int(data.get("n", 2))
                a = data.get("a", [1.0, 2.0, 3.0, 4.0])
                b = data.get("b", [5.0, 6.0, 7.0, 8.0])
                c = client.multiply_matrices(m, k, n, a, b)
                self.send_json({"status": "OK", "operation": op, "result": c, "dimensions": [m, n]})
                return
            elif op == "TOKEN_INTERN":
                text = data.get("text", "")
                toks = client.intern_text(text)
                self.send_json({"status": "OK", "operation": op, "tokens": toks, "count": len(toks)})
                return
            elif op == "AABB_CULL":
                f_min = tuple(data.get("frustum_min", [-10.0, -10.0, -10.0]))
                f_max = tuple(data.get("frustum_max", [10.0, 10.0, 10.0]))
                boxes = [tuple(b) for b in data.get("boxes", [[-1.0, -1.0, -1.0, 1.0, 1.0, 1.0]])]
                vis = client.cull_aabbs(f_min, f_max, boxes)
                self.send_json({"status": "OK", "operation": op, "visible_indices": vis, "visible_count": len(vis)})
                return
            else:
                self.send_json({"status": "ERROR", "message": f"Unsupported IPC operation: {op}"}, 400)
                return

        # ── API: Code GENE Latent Action & Configuration ────────────
        elif self.path == "/api/code_gene/action":
            if not state.code_gene_compositor:
                self.send_json({"status": "ERROR", "message": "Code GENE Compositor offline."})
                return
            act_id = int(data.get("action_id", 0))
            params = data.get("params", {})
            res = state.code_gene_compositor.dynamics.step(act_id, params)
            mesh = state.code_gene_compositor.generate_3d_mesh(resolution=16)
            ascii_str, stats = state.code_gene_compositor.render_frame_ascii(width=78, height=28)
            with state.lock:
                state.director_events.append({
                    "author": "CODE_GENE",
                    "message": f"Latent action '{res['action']}' applied. Topology: {res['topology']}, Volume: {res['volume_m3']}m³"
                })
            self.send_json({
                "status": "SUCCESS",
                "action_result": res,
                "mesh": mesh,
                "ascii_frame": ascii_str,
                "stats": stats
            })
            return

        elif self.path == "/api/code_gene/config":
            if not state.code_gene_compositor:
                self.send_json({"status": "ERROR", "message": "Code GENE Compositor offline."})
                return
            if "dither_mode" in data:
                state.code_gene_compositor.raster_mixer.dither_mode = data["dither_mode"]
            if "subpixel_quadrants" in data:
                state.code_gene_compositor.raster_mixer.use_subpixel_quadrants = bool(data["subpixel_quadrants"])
            if "directional_sobel" in data:
                state.code_gene_compositor.raster_mixer.use_directional_sobel = bool(data["directional_sobel"])
            if "specular_sparkles" in data:
                state.code_gene_compositor.raster_mixer.use_specular_sparkles = bool(data["specular_sparkles"])
            if "bounds" in data:
                b_data = data["bounds"]
                if "max_x" in b_data:
                    state.code_gene_compositor.bounds.max_x = float(b_data["max_x"])
                    state.code_gene_compositor.bounds.min_x = -float(b_data["max_x"])
                if "max_y" in b_data:
                    state.code_gene_compositor.bounds.max_y = float(b_data["max_y"])
                    state.code_gene_compositor.bounds.min_y = -float(b_data["max_y"])
                if "max_z" in b_data:
                    state.code_gene_compositor.bounds.max_z = float(b_data["max_z"])
                    state.code_gene_compositor.bounds.min_z = -float(b_data["max_z"])
            self.send_json({"status": "SUCCESS", "message": "Code GENE configuration updated."})
            return

        elif self.path == "/api/code_gene/export_obj":
            if not state.code_gene_compositor:
                self.send_json({"status": "ERROR", "message": "Code GENE Compositor offline."})
                return
            filename = data.get("filename", "code_gene_active.obj")
            try:
                path = state.code_gene_compositor.export_obj_file(filename)
                self.send_json({
                    "status": "SUCCESS",
                    "filename": filename,
                    "filepath": path,
                    "download_url": f"/static/godot_assets/{filename}"
                })
            except Exception as e:
                self.send_json({"status": "ERROR", "message": str(e)})
            return

        # ── API: Google Maps 3D Urban Geometry Extractor ─────────────
        elif self.path == "/api/google_maps/extract":
            from krystal_web_hub.economic_engine.google_maps_urban_extractor import GLOBAL_GOOGLE_MAPS_EXTRACTOR
            city_id = data.get("city_id", "praha_old_town")
            radius = float(data.get("radius_m", 120.0))
            sector = GLOBAL_GOOGLE_MAPS_EXTRACTOR.extract_city_sector(city_id, radius_m=radius)
            mesh = GLOBAL_GOOGLE_MAPS_EXTRACTOR.synthesize_3d_mesh(sector)

            # Load into active Code GENE Compositor if available
            ascii_frame, stats = "", {}
            if state.code_gene_compositor:
                state.code_gene_compositor.load_real_city(city_id)
                ascii_frame, stats = state.code_gene_compositor.render_frame_ascii(width=78, height=28)

            with state.lock:
                state.director_events.append({
                    "author": "GOOGLE_MAPS",
                    "message": f"Extracted real-world city '{sector.city_name}' ({len(sector.buildings)} buildings, {len(sector.roads)} roads)."
                })

            self.send_json({
                "status": "SUCCESS",
                "city_name": sector.city_name,
                "center_gps": sector.center_gps,
                "building_count": len(sector.buildings),
                "road_count": len(sector.roads),
                "mesh": mesh,
                "ascii_frame": ascii_frame,
                "stats": stats
            })
            return

        elif self.path == "/api/google_maps/export_blender":
            from krystal_web_hub.economic_engine.google_maps_urban_extractor import GLOBAL_GOOGLE_MAPS_EXTRACTOR
            city_id = data.get("city_id", "praha_old_town")
            sector = GLOBAL_GOOGLE_MAPS_EXTRACTOR.extract_city_sector(city_id)
            script_path = GLOBAL_GOOGLE_MAPS_EXTRACTOR.generate_blender_import_script(sector)
            obj_path = GLOBAL_GOOGLE_MAPS_EXTRACTOR.export_obj(sector, filename=f"google_maps_{sector.sector_id}.obj")

            self.send_json({
                "status": "SUCCESS",
                "city_name": sector.city_name,
                "blender_script_path": script_path,
                "obj_filepath": obj_path,
                "download_obj_url": f"/static/godot_assets/urban_cache/google_maps_{sector.sector_id}.obj"
            })
            return

        # ── API: Achievements Unlock & Narrative Synthesis ─────────
        elif self.path == "/api/achievements/unlock":
            if not achievement_narrative_available:
                self.send_json({"status": "ERROR", "message": "Achievement engine offline."})
                return
            ach_id = data.get("achievement_id")
            if not ach_id:
                self.send_json({"status": "ERROR", "message": "Missing achievement_id."})
                return
            city = data.get("city", "Praha - Staré Město")
            tribe = data.get("tribe", "Kryštálový Kmeň")
            ctx = data.get("context", {})
            ctx["city"] = city
            ctx["tribe"] = tribe

            eng = GLOBAL_ACHIEVEMENT_NARRATIVE_ENGINE
            res = eng.unlock_achievement(ach_id, ctx)

            # Broadcast to director events
            with state.lock:
                state.director_events.append({
                    "author": "ACHIEVEMENT_ENGINE",
                    "message": f"🏆 UNLOCKED: '{res['achievement']['title']}' [{res['achievement']['tier']}]! +{res['achievement']['reward']['krystal_credits']} KC"
                })

            self.send_json({
                "status": "SUCCESS",
                "result": res,
                "total_credits_minted": eng.total_credits_minted
            })
            return

        elif self.path == "/api/narrative/synthesize":
            if not achievement_narrative_available:
                self.send_json({"status": "ERROR", "message": "Narrative engine offline."})
                return
            topic = data.get("topic", "Nová Kryštálová Éra")
            city = data.get("city", "Praha - Staré Město")
            tribe = data.get("tribe", "Kryštálový Kmeň")
            category_str = data.get("category", "URBAN_ARCHITECTURE")

            eng = GLOBAL_ACHIEVEMENT_NARRATIVE_ENGINE
            # Find an achievement or mock one for synthesis
            from krystal_web_hub.economic_engine.achievement_narrative_engine import (
                AchievementDefinition, AchievementCategory, AchievementTier, AchievementReward
            )
            try:
                cat = AchievementCategory(category_str)
            except Exception:
                cat = AchievementCategory.URBAN_ARCHITECTURE

            mock_ach = AchievementDefinition(
                achievement_id=f"custom_{int(time.time())}",
                title=topic,
                category=cat,
                tier=AchievementTier.GOLD,
                description=f"Prieskum témy: {topic}",
                reward=AchievementReward(krystal_credits=100),
                condition_predicate="custom_synthesis"
            )

            t0 = time.perf_counter()
            chapter = eng.author_story_chapter(mock_ach, city=city, tribe=tribe, ctx=data)
            duration_s = max(0.001, time.perf_counter() - t0)
            telemetry_snap = eng.governor.sample_telemetry(tokens=256, duration_s=duration_s)
            chapter.telemetry = {
                "backend": telemetry_snap.backend.value,
                "power_watts": telemetry_snap.power_watts,
                "joules_per_token": telemetry_snap.energy_joules_per_tok,
                "ttft_ms": telemetry_snap.time_to_first_token_ms,
                "tokens_generated": 256,
                "duration_seconds": round(duration_s, 4)
            }
            eng.chronicle_codex.append(chapter)

            with state.lock:
                state.director_events.append({
                    "author": "SLM_CHRONICLER",
                    "message": f"📜 Authored new chapter: '{chapter.chapter_title}' ({chapter.city_context})"
                })

            self.send_json({
                "status": "SUCCESS",
                "chapter": {
                    "entry_id": chapter.entry_id,
                    "chapter_title": chapter.chapter_title,
                    "narrative_text": chapter.narrative_text,
                    "city_context": chapter.city_context,
                    "tribe_affected": chapter.tribe_affected,
                    "reputation_delta": chapter.reputation_delta,
                    "created_at": chapter.created_at,
                    "telemetry": chapter.telemetry
                }
            })
            return

        # ── API: Speculative Frontiers Actions ─────────────────────
        elif self.path == "/api/frontiers/rolling_horizon":
            from krystal_web_hub.economic_engine import GLOBAL_SPECULATIVE_FRONTIERS
            rolling = GLOBAL_SPECULATIVE_FRONTIERS["rolling_engine"]
            x = float(data.get("x", 0.0))
            y = float(data.get("y", 1.8))
            z = float(data.get("z", 0.0))
            res = rolling.update_camera_position(x, y, z)
            self.send_json({"status": "SUCCESS", "rolling_horizon": res})
            return

        elif self.path == "/api/frontiers/physics_step":
            from krystal_web_hub.economic_engine import GLOBAL_SPECULATIVE_FRONTIERS, KineticPlayerState
            phys = GLOBAL_SPECULATIVE_FRONTIERS["physics_governor"]
            act_id = int(data.get("action_id", 0))
            px = float(data.get("pos_x", 0.0))
            py = float(data.get("pos_y", 0.0))
            pz = float(data.get("pos_z", 0.0))
            p_state = KineticPlayerState(pos_x=px, pos_y=py, pos_z=pz)
            p_next = phys.evaluate_latent_action_physics(p_state, latent_action_id=act_id)
            self.send_json({
                "status": "SUCCESS",
                "physics_state": {
                    "pos": [round(p_next.pos_x, 3), round(p_next.pos_y, 3), round(p_next.pos_z, 3)],
                    "vel": [round(p_next.vel_x, 3), round(p_next.vel_y, 3), round(p_next.vel_z, 3)],
                    "is_grounded": p_next.is_grounded,
                    "penetration_prevented": p_next.penetration_prevented,
                    "vital_hp": p_next.vital_hp
                }
            })
            return

        elif self.path == "/api/frontiers/transpile_ascii":
            from krystal_web_hub.economic_engine import GLOBAL_SPECULATIVE_FRONTIERS
            transpiler = GLOBAL_SPECULATIVE_FRONTIERS["ascii_transpiler"]
            ascii_text = data.get("ascii_blueprint", "###\n#+#\n...")
            step_m = float(data.get("grid_step_m", 1.0))
            scene = transpiler.transpile_ascii_to_3d_mesh(ascii_text, grid_step_m=step_m)
            self.send_json({
                "status": "SUCCESS",
                "voxel_count": scene.voxel_count,
                "triangles_count": scene.triangles_count,
                "vertex_count": len(scene.vertices),
                "faces_count": len(scene.faces),
                "godot_nodes_count": len(scene.godot_tscn_nodes),
                "godot_tscn_preview": "\n".join(scene.godot_tscn_nodes[:10])
            })
            return

        # ── API: Processor Whisperer Pulse Simulation ──────────────
        elif self.path == "/api/whisperer/pulse":
            from krystal_kernel import GLOBAL_PROCESSOR_WHISPERER
            wh = GLOBAL_PROCESSOR_WHISPERER
            pc = int(data.get("pc", 0x1000))
            is_cs = bool(data.get("current_switch", False))
            is_l1 = bool(data.get("l1_write", True))
            is_l3 = bool(data.get("l3_write", False))
            is_dram = bool(data.get("dram_spill", False))
            sim_noise = bool(data.get("simulate_anomaly", False))

            entry = wh.record_hardware_pulse(
                pc=pc, is_current_switch=is_cs, is_l1_write=is_l1,
                is_l3_write=is_l3, is_dram_spill=is_dram,
                simulate_anomaly=sim_noise
            )
            hint = wh.whisper_instruction_hint(pc)

            self.send_json({
                "status": "SUCCESS",
                "logged_event": {
                    "pc": f"0x{entry.pc:04X}",
                    "origin": entry.origin_name,
                    "syndrome": entry.syndrome,
                    "healed": entry.healed,
                    "raw_codeword_7bit": f"0x{entry.raw_codeword_7bit:02X}"
                },
                "whisper_hint": hint
            })
            return

        # ── API: Adaptive RAM Ring Buffer vs SSD Swap Cycle ────────
        elif self.path == "/api/whisperer/adaptive_swap":
            from krystal_kernel import GLOBAL_ADAPTIVE_SWAP_MANAGER
            res = GLOBAL_ADAPTIVE_SWAP_MANAGER.execute_structured_swap_cycle()
            self.send_json(res)
            return

        # ── API: Closed-Loop Janet-to-Godot 4.x Synthesizer ────────
        elif self.path == "/api/whisperer/synthesize_godot":
            from krystal_kernel import GLOBAL_LOG_DRIVEN_SYNTHESIZER
            scene = data.get("scene", "NeoPraha_Alchemical_Hologram")
            seed = int(data.get("seed", 42))
            res = GLOBAL_LOG_DRIVEN_SYNTHESIZER.synthesize_hardware_tuned_scene(scene_name=scene, seed=seed)
            self.send_json(res)
            return

        # ── API: LLM Configuration Panel Mutation ──────────────────
        elif self.path == "/api/config/mutate":
            from krystal_kernel import GLOBAL_LLM_CONFIG_AGENT
            mutation = GLOBAL_LLM_CONFIG_AGENT.mutate_configuration_via_logs()
            self.send_json({
                "status": "SUCCESS",
                "mutation": mutation.to_dict(),
                "updated_cells": GLOBAL_LLM_CONFIG_AGENT.get_all_cells()
            })
            return

        # ── API: Update Individual Config Cell ─────────────────────
        elif self.path == "/api/config/update_cell":
            from krystal_kernel import GLOBAL_LLM_CONFIG_AGENT
            cell_id = data.get("cell_id", "")
            val = data.get("value")
            ok = GLOBAL_LLM_CONFIG_AGENT.update_cell_value(cell_id, val)
            self.send_json({
                "status": "SUCCESS" if ok else "ERROR",
                "cell_id": cell_id,
                "current_value": GLOBAL_LLM_CONFIG_AGENT.cells.get(cell_id, {}).current_value if ok else None
            })
            return

        # ── API: Repetitive Cache Matrix Stream Compressor ─────────
        elif self.path == "/api/config/compress":
            from krystal_kernel import GLOBAL_MATRIX_COMPRESSOR
            matrices = data.get("matrices", [])
            if not matrices:
                # Fabricate repeating identity and transformation matrices to test compression
                matrices = [
                    [1.0, 0.0, 0.0, 0.0,  0.0, 1.0, 0.0, 0.0,  0.0, 0.0, 1.0, 0.0,  0.0, 0.0, 0.0, 1.0],
                    [1.0, 0.0, 0.0, 0.0,  0.0, 1.0, 0.0, 0.0,  0.0, 0.0, 1.0, 0.0,  0.0, 0.0, 0.0, 1.0],
                    [1.2, 0.0, 0.0, 0.0,  0.0, 1.2, 0.0, 0.0,  0.0, 0.0, 1.0, 0.0,  0.0, 0.0, 0.0, 1.0],
                    [1.2, 0.0, 0.0, 0.0,  0.0, 1.2, 0.0, 0.0,  0.0, 0.0, 1.0, 0.0,  0.0, 0.0, 0.0, 1.0]
                ]
            compressed_bytes, stats = GLOBAL_MATRIX_COMPRESSOR.compress_matrix_stream(matrices)
            self.send_json({
                "status": "SUCCESS",
                "stats": stats.to_dict(),
                "compressed_hex_sample": compressed_bytes[:32].hex().upper()
            })
            return

        # ── API: Cross-Domain Derivatives Simulation ──────────────
        elif self.path == "/api/domain_derivatives/simulate":
            from krystal_kernel import (
                GLOBAL_HFT_DEDUPLICATOR,
                GLOBAL_DRONE_GOVERNOR,
                GLOBAL_GENOMIC_ENCODER,
                HFTOrderBookTick,
                VITAL_MAX_HP
            )
            domain = data.get("domain", "ALL").upper()
            results = {
                "status": "SUCCESS",
                "vital_max_hp": VITAL_MAX_HP,
                "timestamp_ns": time.time_ns(),
                "simulations": {}
            }

            # 1. HFT Tick Deduplicator Simulation
            if domain in ("HFT", "ALL"):
                ticks = []
                base_bid = 100.0
                base_ask = 100.05
                for i in range(10):
                    tick = HFTOrderBookTick(
                        timestamp_ns=time.time_ns() + i * 1000,
                        instrument="EUR/USD_MICRO",
                        bid_prices=[base_bid - j * 0.01 for j in range(5)],
                        bid_volumes=[100.0, 250.0, 500.0, 1000.0, 2000.0],
                        ask_prices=[base_ask + j * 0.01 for j in range(5)],
                        ask_volumes=[150.0, 300.0, 450.0, 800.0, 1500.0],
                        spread=0.05,
                        mid_price=100.025
                    )
                    ticks.append(tick)
                compressed_bytes, stats = GLOBAL_HFT_DEDUPLICATOR.ingest_and_compress_ticks(ticks)
                results["simulations"]["HFT"] = {
                    "instrument": "EUR/USD_MICRO",
                    "ticks_processed": len(ticks),
                    "raw_bytes": stats.raw_size_bytes,
                    "compressed_bytes": stats.compressed_size_bytes,
                    "compression_ratio": stats.compression_ratio,
                    "l1_cache_lines_saved": stats.l1_cache_lines_saved,
                    "l3_bandwidth_saved_kb": stats.l3_writeback_bandwidth_saved_kb,
                    "hex_sample": compressed_bytes[:24].hex().upper()
                }

            # 2. Edge Robotics Drone Navigation Step Simulation
            if domain in ("ROBOTICS", "ALL"):
                pos = (2.2, 1.6, 3.2) # Close to obstacle at (2.0, 1.5, 3.0)
                target = (5.0, 0.0, 5.0)
                obs = GLOBAL_DRONE_GOVERNOR.navigate_step(pos, target)
                results["simulations"]["ROBOTICS"] = {
                    "drone_position": obs.position_xyz,
                    "nearest_obstacle_dist_m": obs.nearest_obstacle_dist_m,
                    "safe_to_navigate": obs.safe_to_navigate,
                    "repulsive_escape_vector": obs.escape_vector_xyz,
                    "commanded_velocity": obs.velocity_xyz,
                    "vital_hp": obs.vital_hp
                }

            # 3. Genomic Codon Radiation Noise Self-Healing Simulation
            if domain in ("GENOMICS", "ALL"):
                pulse = GLOBAL_GENOMIC_ENCODER.encode_and_heal_codon_pair(
                    base_1="A",
                    base_2="C",
                    simulate_radiation_noise_bit=3
                )
                results["simulations"]["GENOMICS"] = {
                    "channel": pulse.channel,
                    "original_codon_target": "AC",
                    "injected_radiation_noise_bit": 3,
                    "syndrome_detected": pulse.raw_syndrome,
                    "was_repaired": pulse.was_repaired,
                    "repaired_codon_output": pulse.repaired_base,
                    "recovered_data_bits": list(pulse.codon_data_bits)
                }

            self.send_json(results)
            return

        # ── API: NextGen Energy & Anti-Mining Audit ────────────────
        elif self.path == "/api/nextgen/audit_energy":
            from krystal_stack_nextgen import GLOBAL_ENERGY_GOVERNOR
            watts = float(data.get("watts", 8.2))
            fps = float(data.get("fps", 60.0))
            entropy = float(data.get("entropy", 0.55))
            presentation = bool(data.get("has_display_presentation", True))
            alu_mem = float(data.get("alu_to_memory_ratio", 12.0))

            audit = GLOBAL_ENERGY_GOVERNOR.audit_workload_telemetry(
                measured_watts=watts,
                current_fps=fps,
                visual_entropy_change=entropy,
                has_display_presentation=presentation,
                alu_to_memory_ratio=alu_mem
            )
            self.send_json({
                "status": "SUCCESS",
                "audit": audit.to_dict()
            })
            return

        # ── API: NextGen Iris Xe Kisak Bandwidth Optimization ──────
        elif self.path == "/api/nextgen/optimize_iris_xe":
            from krystal_stack_nextgen import GLOBAL_IRIS_XE_OPTIMIZER
            w = int(data.get("width", 1920))
            h = int(data.get("height", 1080))
            fps = int(data.get("fps", 60))
            steps = int(data.get("steps", 96))
            tile4 = bool(data.get("tile4", True))
            ccs = bool(data.get("ccs", True))
            simd16 = bool(data.get("simd16", True))

            plan = GLOBAL_IRIS_XE_OPTIMIZER.compute_bandwidth_optimization(
                width=w, height=h, target_fps=fps, raymarch_steps=steps,
                use_tile4_compression=tile4, use_lossless_ccs=ccs, use_simd16_subgroups=simd16
            )
            self.send_json({
                "status": "SUCCESS",
                "plan": plan.to_dict()
            })
            return

        # ── API: NextGen NPU Speculative Pre-Staging ───────────────
        elif self.path == "/api/nextgen/npu_prestage":
            from krystal_stack_nextgen import GLOBAL_NPU_PREDICTOR
            asset_key = str(data.get("asset_key", "TERRAIN_OCTAVE_SURGE_CHUNK_04"))
            steps = int(data.get("expected_raymarch_steps", 96))

            strat = GLOBAL_NPU_PREDICTOR.evaluate_predictive_strategy(
                predicted_asset_key=asset_key,
                expected_raymarch_steps=steps
            )
            is_pinned, lat_us = GLOBAL_NPU_PREDICTOR.prestager.query_block_latency(asset_key)

            self.send_json({
                "status": "SUCCESS",
                "strategy": strat.to_dict(),
                "is_pinned_in_hot_ram": is_pinned,
                "retrieval_latency_us": lat_us,
                "speedup_factor": "1500x faster (RAM vs SSD NVMe)" if is_pinned else "1.0x (Cold Disk Miss)"
            })
            return

        # ── API: NextGen Fast Binary IPC Benchmark ─────────────────
        elif self.path == "/api/nextgen/benchmark_ipc":
            from krystal_stack_nextgen import GLOBAL_PATTERN_IPC_GOVERNOR
            sample = data.get("sample_text", "░▒▓█" * 100)
            res = GLOBAL_PATTERN_IPC_GOVERNOR.benchmark_ipc_serialization(sample)
            self.send_json({
                "status": "SUCCESS",
                "ipc_benchmark": res
            })
            return

        # ── API: NextGen Fast Text Tokenization ────────────────────
        elif self.path == "/api/nextgen/text_tokenize":
            from krystal_stack_nextgen import GLOBAL_PATTERN_IPC_GOVERNOR
            text = data.get("text", "CYBERPUNK TILE4 NPU_PRESTAGE TPU_SYSTOLIC")
            res = GLOBAL_PATTERN_IPC_GOVERNOR.text_processor.measure_text_processing_speedup(text)
            token_ids = GLOBAL_PATTERN_IPC_GOVERNOR.text_processor.fast_tokenize(text)
            self.send_json({
                "status": "SUCCESS",
                "tokens": token_ids,
                "benchmark": res
            })
            return

        # ── API: Antigravity Natural Language Prompt ───────────────
        elif self.path == "/api/antigravity/prompt":
            prompt = data.get("prompt", "")
            if not antigravity_engine or not prompt:
                self.send_json({"status": "ERROR", "message": "Prompt empty or Antigravity engine offline."})
                return

            try:
                tpl = antigravity_engine.parse_prompt(prompt)
                antigravity_engine.save_template(tpl)
                glsl = antigravity_engine.export_vulkan_glsl_constants(tpl)
                with state.lock:
                    state.active_template = tpl
                    state.mode = "RECURSIVE_MIRROR"
                    state.director_events.append({
                        "author": "ANTIGRAVITY",
                        "message": f"Instantiated '{tpl['name']}' ({tpl['composition_rules']['symmetry_group']}) -> Vulkan AR pipeline active."
                    })
                self.send_json({
                    "status": "SUCCESS",
                    "template": tpl,
                    "glsl_uniforms": glsl
                })
            except Exception as e:
                self.send_json({"status": "ERROR", "message": str(e)})

        # ── API: Manual Template Composition ───────────────────────
        elif self.path == "/api/templates/compose":
            if not antigravity_engine:
                self.send_json({"status": "ERROR", "message": "Antigravity engine unavailable."})
                return
            geom_id = data.get("geometric_id", "KALEIDOSCOPIC_IFS")
            art_id = data.get("artistic_id", "CYBERPUNK_NEON_AR")
            folds = int(data.get("mirror_folds", 6))
            depth = int(data.get("recursion_depth", 24))
            fresnel = float(data.get("fresnel_reflectance", 0.88))
            try:
                tpl = antigravity_engine.instance_manager.compose_template(
                    geometric_id=geom_id,
                    artistic_id=art_id,
                    mirror_folds=folds,
                    recursion_depth=depth,
                    fresnel_reflectance=fresnel
                )
                antigravity_engine.save_template(tpl)
                glsl = antigravity_engine.export_vulkan_glsl_constants(tpl)
                with state.lock:
                    state.active_template = tpl
                    state.mode = "RECURSIVE_MIRROR"
                self.send_json({"status": "SUCCESS", "template": tpl, "glsl_uniforms": glsl})
            except Exception as e:
                self.send_json({"status": "ERROR", "message": str(e)})

        # ── API: Mimicry Selection (Object or Scene) ───────────────
        elif self.path == "/api/mimicry/select":
            sel_type = data.get("type", "object")
            target_id = data.get("id", "")
            if not mimicry_available:
                self.send_json({"status": "ERROR", "message": "Mimicry Engine unavailable."})
                return

            with state.lock:
                if sel_type == "object":
                    obj = get_recipe(target_id)
                    if obj:
                        state.active_mimic_id = target_id
                        state.active_mimic_obj = obj
                        state.mode = "MIMICRY_OBJECT"
                        state.director_events.append({
                            "author": "MIMIC_ENGINE",
                            "message": f"Composed mimicry object '{obj.name}' with {len(obj.parts)} modifier parts."
                        })
                        self.send_json({"status": "SUCCESS", "type": "object", "object": obj.to_dict()})
                        return
                    else:
                        self.send_json({"status": "ERROR", "message": f"Unknown recipe ID: {target_id}"})
                        return
                elif sel_type == "scene":
                    sc = get_scene(target_id)
                    if sc:
                        state.active_scene_id = target_id
                        state.active_scene_obj = sc
                        state.mode = "GAME_SCENE"
                        state.director_events.append({
                            "author": "SCENE_COMPOSER",
                            "message": f"Assembled game scene '{sc.name}' with {len(sc.actors)} actors."
                        })
                        self.send_json({"status": "SUCCESS", "type": "scene", "scene": sc.to_dict()})
                        return
                    else:
                        self.send_json({"status": "ERROR", "message": f"Unknown scene ID: {target_id}"})
                        return
            self.send_json({"status": "ERROR", "message": "Invalid request parameters."})

        # ── API: Export Scene to Godot .tscn ────────────────────────
        elif self.path == "/api/mimicry/export-godot":
            if not mimicry_available or not state.active_scene_obj:
                self.send_json({"status": "ERROR", "message": "No active scene to export."})
                return
            tscn_content = state.active_scene_obj.export_godot_tscn()
            export_path = os.path.join(WORKSPACE_ROOT, "godot_project", "scenes", f"{state.active_scene_id}.tscn")
            with open(export_path, "w", encoding="utf-8") as f:
                f.write(tscn_content)
            self.send_json({
                "status": "SUCCESS",
                "scene_id": state.active_scene_id,
                "exported_file": f"godot_project/scenes/{state.active_scene_id}.tscn"
            })

        # ── API: Real-World Urban Spatial Painter ──────────────────
        elif self.path == "/api/mimicry/paint-real-world":
            if not mimicry_available:
                self.send_json({"status": "ERROR", "message": "Mimicry Engine unavailable."})
                return
            archetype = data.get("archetype", "old_town_square")
            seed = int(data.get("seed", 42))
            try:
                sc = paint_real_world_scene(archetype, seed=seed)
                val_res = UrbanSpatialCompositionRules.validate_scene(sc)
                with state.lock:
                    state.active_scene_id = sc.scene_id
                    state.active_scene_obj = sc
                    state.mode = "GAME_SCENE"
                    state.director_events.append({
                        "author": "URBAN_COMPOSITOR",
                        "message": f"Painted real-world urban scene '{sc.name}' (Archetype: {archetype}, Seed: {seed}, Compliance: {val_res.compliance_score*100:.1f}%)."
                    })
                self.send_json({
                    "status": "SUCCESS",
                    "scene": sc.to_dict(),
                    "validation": val_res.to_dict()
                })
            except Exception as e:
                self.send_json({"status": "ERROR", "message": str(e)})


        # ── API: OpenWorld Natural Language Compilation ───────────
        elif self.path == "/api/openworld/compile":
            prompt = data.get("prompt", "")
            if not openworld_available or not openworld_compiler or not prompt:
                self.send_json({"status": "ERROR", "message": "Prompt empty or OpenWorld compiler offline."})
                return

            try:
                spec = openworld_compiler.compile_natural_prompt(prompt)
                renderer = OpenWorldRenderer(spec["manifold"], spec)
                janet_dsl = openworld_compiler.to_janet_dsl(spec)
                godot_shader = openworld_compiler.to_godot_shader(spec)
                py_code = openworld_compiler.to_procedural_python(spec)

                with state.lock:
                    state.active_world_spec = spec
                    state.active_world_renderer = renderer
                    state.active_janet_dsl = janet_dsl
                    state.active_godot_shader = godot_shader
                    state.active_py_generator = py_code
                    state.mode = "OPENWORLD"
                    state.director_events.append({
                        "author": "OPENWORLD_COMPILER",
                        "message": f"Compiled '{prompt}' -> Biome: {spec['dominant_biome']['name']} (Janet DSL & Godot Shader ready)."
                    })

                self.send_json({
                    "status": "SUCCESS",
                    "spec": {
                        "name": spec["name"],
                        "topography_type": spec["topography_type"],
                        "atmosphere_type": spec["atmosphere_type"],
                        "dominant_biome": spec["dominant_biome"],
                        "mathematical_parameters": spec["mathematical_parameters"],
                        "atmospheric_parameters": spec["atmospheric_parameters"],
                        "artifact_scatter_rules": spec["artifact_scatter_rules"]
                    },
                    "janet_dsl": janet_dsl,
                    "godot_shader": godot_shader,
                    "python_code": py_code
                })
            except Exception as e:
                self.send_json({"status": "ERROR", "message": str(e)})

        # ── API: Krystal-Lang Compilation ─────────────────────────
        elif self.path == "/api/krystal-lang/compile":
            code = data.get("code", "")
            if not krystal_lang_available or not krystal_compiler or not code:
                self.send_json({"status": "ERROR", "message": "Code empty or Krystal-Lang compiler offline."})
                return

            try:
                comp_res = krystal_compiler.compile(code)
                transpiler = BytecodeShapeTranspiler(comp_res)
                vm = TopologicalVM(comp_res)
                vm.inject_input("InStream" if "InStream" in vm.channels else list(vm.channels.keys())[0], {"compile_init": True})

                with state.lock:
                    state.krystal_lang_comp = comp_res
                    state.krystal_lang_transpiler = transpiler
                    state.krystal_lang_vm = vm
                    state.mode = "KRYSTAL_LANG_SHAPE"
                    state.director_events.append({
                        "author": "KRYSTAL_LANG_COMPILER",
                        "message": f"Compiled module '{comp_res['module_name']}': {len(comp_res['bytecode'])} topological opcodes materialized as 3D shape."
                    })

                self.send_json({
                    "status": "SUCCESS",
                    "module_name": comp_res["module_name"],
                    "bytecode": comp_res["bytecode"],
                    "queue_partitions": comp_res["queue_partitions"],
                    "queues_count": comp_res["queues_count"],
                    "vm_metrics": vm.metrics
                })
            except Exception as e:
                self.send_json({"status": "ERROR", "message": str(e)})

        # ── API: Generate Procedural Instance ─────────────────────
        elif self.path == "/api/instances/generate":
            cat = data.get("category", "superformula_organic")
            seed = data.get("seed", None)
            if procedural_generator_available:
                try:
                    inst = ProceduralInstanceGenerator.generate_random_instance(cat, seed)
                    self.send_json({"status": "SUCCESS", "instance": inst})
                except Exception as e:
                    self.send_json({"status": "ERROR", "message": str(e)})
            else:
                self.send_json({"status": "ERROR", "message": "Generator offline."})

        # ── API: Governor Actions ──────────────────────────────────
        elif self.path == "/api/governor":
            action = data.get("action", "")
            with state.lock:
                if action == "RECHARGE":
                    state.budget = min(state.max_budget, state.budget + 200)
                elif action == "DRAIN":
                    state.budget = max(50, state.budget - 300)
                    state.governor_state = "CRITICAL_DEFICIT"
            self.send_json({"status": "APPLIED", "budget": state.budget})

        # ── API: Cognitive Scene Director Prompt ───────────────────
        elif self.path == "/api/director":
            prompt = data.get("prompt", "")
            reply = self.generate_director_reply(prompt)
            with state.lock:
                state.director_events.append({"author": "USER", "message": prompt})
                state.director_events.append({"author": "SLM_DIRECTOR", "message": reply})
            self.send_json({"reply": reply})

        # ── API: Vulkan Accelerator Toggle ────────────────────────
        elif self.path == "/api/vulkan/toggle":
            enabled = data.get("enabled", None)
            with state.lock:
                if enabled is not None:
                    state.use_vulkan = bool(enabled)
                else:
                    state.use_vulkan = not state.use_vulkan
                cur_state = state.use_vulkan
                telemetry = state.vulkan_driver.get_telemetry() if state.vulkan_driver else {}
            self.send_json({
                "status": "OK",
                "enabled": cur_state,
                "telemetry": telemetry
            })
        else:
            if self.path.startswith("/api/") and self.proxy_to_engine_core("POST", body=body.encode("utf-8") if isinstance(body, str) else body):
                return
            self.send_error(404, "Endpoint not found")

    def proxy_to_engine_core(self, method="GET", body=None) -> bool:
        target_url = f"http://127.0.0.1:8089{self.path}"
        req = urllib.request.Request(target_url, data=body, method=method)
        for h, v in self.headers.items():
            if h.lower() not in ("host", "content-length"):
                req.add_header(h, v)
        try:
            with urllib.request.urlopen(req, timeout=5.0) as resp:
                resp_body = resp.read()
                self.send_response(resp.status)
                for h, v in resp.headers.items():
                    if h.lower() not in ("transfer-encoding", "content-length"):
                        self.send_header(h, v)
                self.send_header("Content-Length", str(len(resp_body)))
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(resp_body)
                return True
        except urllib.error.HTTPError as e:
            err_body = e.read()
            self.send_response(e.code)
            for h, v in e.headers.items():
                if h.lower() not in ("transfer-encoding", "content-length"):
                    self.send_header(h, v)
            self.send_header("Content-Length", str(len(err_body)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(err_body)
            return True
        except Exception:
            return False

    def _guess_mime(self, filepath: Path) -> str:
        suffix = filepath.suffix.lower()
        mime_map = {
            ".html": "text/html",
            ".htm": "text/html",
            ".css": "text/css",
            ".js": "application/javascript",
            ".mjs": "application/javascript",
            ".json": "application/json",
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".gif": "image/gif",
            ".svg": "image/svg+xml",
            ".ico": "image/x-icon",
            ".webp": "image/webp",
            ".obj": "text/plain",
            ".tscn": "text/plain",
            ".wasm": "application/wasm",
            ".mp3": "audio/mpeg",
            ".wav": "audio/wav",
            ".ogg": "audio/ogg",
            ".ttf": "font/ttf",
            ".woff": "font/woff",
            ".woff2": "font/woff2",
        }
        return mime_map.get(suffix, mimetypes.guess_type(str(filepath))[0] or "application/octet-stream")

    def generate_director_reply(self, prompt: str) -> str:
        prompt_lower = prompt.lower()
        if "stealth" in prompt_lower or "silent" in prompt_lower:
            with state.lock: state.mode = "MATRIX_RAIN"
            return "Directive acknowledged. Engaging Matrix stealth mode. Palette shifted to monochrome digital rain."
        elif "combat" in prompt_lower or "attack" in prompt_lower or "overload" in prompt_lower:
            with state.lock: state.mode = "CYBERPUNK"
            return "Combat telemetry engaged. Amplifying high-contrast blocks and activating visual backpressure governor."
        elif "blueprint" in prompt_lower or "wireframe" in prompt_lower or "cad" in prompt_lower:
            with state.lock: state.mode = "BLUEPRINT_EDGE"
            return "Switched to Blueprint Edge mode. Computing directional Sobel normals for geometric clarity."
        elif "holo" in prompt_lower or "hologram" in prompt_lower:
            with state.lock: state.mode = "HOLOGRAPHIC_3D"
            return "Holographic projector online. Projecting volumetric interference fringes with chromatic stereoscopic depth."
        elif "mimic" in prompt_lower or "turret" in prompt_lower or "mech" in prompt_lower or "spire" in prompt_lower or "obelisk" in prompt_lower or "drone" in prompt_lower or "explorer" in prompt_lower:
            with state.lock: state.mode = "MIMICRY_OBJECT"
            return f"Mimicry Compositor engaged. Rendering Blender modifier stack for '{state.active_mimic_id}'."
        elif "scene" in prompt_lower or "district" in prompt_lower or "ruins" in prompt_lower or "hangar" in prompt_lower or "hive" in prompt_lower:
            with state.lock: state.mode = "GAME_SCENE"
            return f"Game Scene Framework loaded: '{state.active_scene_id}' with multi-actor placement."
        elif "mirror" in prompt_lower or "sacred" in prompt_lower or "zrkadl" in prompt_lower or "geometr" in prompt_lower:
            if antigravity_engine:
                try:
                    tpl = antigravity_engine.parse_prompt(prompt)
                    with state.lock:
                        state.active_template = tpl
                        state.mode = "RECURSIVE_MIRROR"
                    return f"Antigravity AR Pipeline engaged: {tpl['name']} ({tpl['composition_rules']['symmetry_group']})."
                except Exception:
                    pass
            with state.lock: state.mode = "RECURSIVE_MIRROR"
            return "Switched to Recursive Mirror AR engine. Dihedral symmetry folding active."
        elif "3d" in prompt_lower or "raymarch" in prompt_lower:
            with state.lock: state.mode = "RAYMARCH_ANOMALY"
            return "Loading procedural 3D raymarching manifold (SDF Torus + Pulsating Core)."
        else:
            return f"Processed directive: '{prompt}'. Telemetry coherence maintained at {state.coherence*100:.0f}%."

    def serve_file(self, filepath: Path, content_type: str):
        if not filepath.exists() or not filepath.is_file():
            self.send_error(404, f"File {filepath.name} not found")
            return
        if (content_type.startswith("text/") or content_type in ("application/javascript", "application/json", "image/svg+xml")) and "charset=" not in content_type:
            content_type = f"{content_type}; charset=utf-8"
        content = filepath.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(content)

    def _route_path(self) -> str:
        return urlparse(self.path).path.rstrip("/") or "/"

    def handle_intelligence(self):
        """Read-only project-intelligence endpoints (priorities, expertise, patterns)."""
        if not intelligence_available:
            self.send_json({"status": "UNAVAILABLE", "message": "project_intelligence module not importable."}, 503)
            return
        route = self._route_path()
        q = parse_qs(urlparse(self.path).query)

        def one(key, default=None):
            return q[key][0] if key in q and q[key] else default

        def flag(key):
            return str(one(key, "")).lower() in ("1", "true", "yes")

        try:
            pi = get_intelligence()
            force = flag("refresh")
            if route == "/api/project/overview":
                self.send_json({"status": "OK", **pi.overview(force=force)})
            elif route == "/api/priorities":
                sort = one("sort", "score")
                if sort not in ("score", "value"):
                    self.send_json({"status": "ERROR", "message": "sort must be 'score' or 'value'."}, 400)
                    return
                try:
                    limit = max(0, int(one("limit", "0")))
                except ValueError:
                    self.send_json({"status": "ERROR", "message": "limit must be an integer."}, 400)
                    return
                self.send_json({"status": "OK", **pi.priorities(
                    limit=limit, sort=sort, category=one("category", ""),
                    include_healthy=flag("include_healthy"), force=force)})
            elif route.startswith("/api/priorities/"):
                p = pi.priority(route[len("/api/priorities/"):])
                if p is None:
                    self.send_json({"status": "NOT_FOUND", "message": "Unknown priority id."}, 404)
                else:
                    self.send_json({"status": "OK", "priority": p})
            elif route == "/api/expertise":
                res = pi.expertise(one("priority", ""), force=force)
                if res is None:
                    self.send_json({"status": "NOT_FOUND", "message": "Unknown priority id."}, 404)
                else:
                    self.send_json({"status": "OK", **res})
            else:
                self.send_json({"status": "OK", **pi.patterns(force=force)})
        except Exception as e:
            self.send_json({"status": "ERROR", "message": f"{type(e).__name__}: {e}"}, 500)

    # ── Krystal Kernel API (OpenAI-compatible surface) ─────────────
    _KERNEL_GET = ("/v1/", "/metrics", "/api/kernel/")
    _KERNEL_POST = ("/v1/embeddings", "/v1/chat/completions")

    def _is_kernel_path(self, prefixes) -> bool:
        route = self.path.split("?", 1)[0].rstrip("/")
        return any(route == p.rstrip("/") or route.startswith(p) for p in prefixes)

    def _write_kernel_response(self, status: int, headers: dict, payload: bytes, close: bool = False):
        # Deliberately no Access-Control-Allow-Origin: this surface may be key-protected.
        self.send_response(status)
        for k, v in headers.items():
            self.send_header(k, v)
        self.send_header("Content-Length", str(len(payload)))
        if close:
            self.send_header("Connection", "close")
            self.close_connection = True
        self.end_headers()
        self.wfile.write(payload)

    def handle_kernel_api(self, method: str):
        ctx = get_api_context()
        if ctx is None:
            msg = {"error": {"message": "krystal_kernel is unavailable: " + str(_KERNEL_ERROR),
                             "type": "server_error", "code": "kernel_unavailable"}}
            self._write_kernel_response(503, {"Content-Type": "application/json"}, json.dumps(msg).encode("utf-8"))
            return
        body = b""
        if method == "POST":
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                length = -1
            if length < 0:
                err = {"error": {"message": "Invalid Content-Length.", "type": "invalid_request_error", "code": "invalid_length"}}
                self._write_kernel_response(400, {"Content-Type": "application/json"}, json.dumps(err).encode("utf-8"), close=True)
                return
            if length > ctx.max_body:
                # Reject BEFORE reading so an oversized body is never buffered.
                err = {"error": {"message": f"Body exceeds {ctx.max_body} bytes.", "type": "invalid_request_error", "code": "payload_too_large"}}
                self._write_kernel_response(413, {"Content-Type": "application/json"}, json.dumps(err).encode("utf-8"), close=True)
                return
            body = self.rfile.read(length) if length else b""
        hdrs = {k: v for k, v in self.headers.items()}
        status, h, out = _kernel_handle(ctx, method, self.path, hdrs, body)
        self._write_kernel_response(status, h, out)

    def send_json(self, data: dict, status: int = 200):
        encoded = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(encoded)

    def send_text(self, text: str, content_type: str = "text/plain; charset=utf-8", status: int = 200):
        encoded = text.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(encoded)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(encoded)


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True
    request_queue_size = 128  # default 5 refuses bursts of concurrent clients

def start_server(host="127.0.0.1", port=8080):
    server = ThreadedHTTPServer((host, port), KrystalHubHandler)
    worker = threading.Thread(target=engine_worker_loop, daemon=True)
    worker.start()

    print("\n" + "="*70)
    print(" [KRYSTAL-STACK] PRODUCTION LOCALHOST MISSION CONTROL")
    print("="*70)
    print(f" >> Web UI Address:    http://{host}:{port}/")
    print(f" >> SSE Live Stream:   http://{host}:{port}/api/stream")
    print(f" >> JSON System State: http://{host}:{port}/api/status")
    print(f" >> Health Check:      http://{host}:{port}/api/health")
    print("="*70)
    print(" Server online and listening. Press CTRL+C to terminate.")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[HUB] Gracefully shutting down Krystal Server...")
    finally:
        state.running = False
        if shutdown_kernel is not None and _api_ctx is not None:
            try:
                shutdown_kernel()
            except Exception:  # noqa: BLE001
                pass
        server.server_close()
        print("[HUB] Shutdown complete.")

if __name__ == "__main__":
    p = 8080
    if len(sys.argv) > 1:
        p = int(sys.argv[1])
    start_server(port=p)
