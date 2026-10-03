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
    from mimicry_engine.scene_composer import list_scenes, get_scene
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

        # Current Rendered Frame
        self.current_ascii = ""
        self.frame_id = 0

        # Director Log Queue
        self.director_events = []

state = EngineState()

# ─── 2. PROCEDURAL 3D SDF & PROCEDURAL GENERATORS ────────────────────────────

def sdf_torus(p, r1=1.1, r2=0.4):
    q_x = math.sqrt(p[0]**2 + p[2]**2) - r1
    q_y = p[1]
    return math.sqrt(q_x**2 + q_y**2) - r2

def sdf_sphere(p, r=0.6):
    return math.sqrt(p[0]**2 + p[1]**2 + p[2]**2) - r

def rotate_y(p, theta):
    c, s = math.cos(theta), math.sin(theta)
    return (p[0]*c + p[2]*s, p[1], -p[0]*s + p[2]*c)

def rotate_x(p, theta):
    c, s = math.cos(theta), math.sin(theta)
    return (p[0], p[1]*c - p[2]*s, p[1]*s + p[2]*c)

def scene_sdf(p, t):
    p_rot = rotate_y(p, t * 1.3)
    p_rot = rotate_x(p_rot, t * 0.7)
    d_torus = sdf_torus(p_rot, r1=1.0, r2=0.38)
    d_core = sdf_sphere(p, r=0.55 + 0.12 * math.sin(t * 3.5))
    k = 0.3
    h = max(k - abs(d_torus - d_core), 0.0) / k
    return min(d_torus, d_core) - h * h * k * 0.25

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
        # ── Root / UI ──────────────────────────────────────────────
        if path in ("/", "/index.html"):
            self.serve_file(STATIC_DIR / "index.html", "text/html")
            return
        elif path in ("/game", "/arena", "/posledni-kmen", "/godot_builder_extension.html", "/builder", "/studio-3d"):
            self.serve_file(STATIC_DIR / "godot_builder_extension.html", "text/html")
            return
        elif path in ("/manual", "/design-manual", "/game_elements_design_manual.html"):
            self.serve_file(STATIC_DIR / "game_elements_design_manual.html", "text/html")
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
        elif path == "/api/health":
            self.send_json({"status": "HEALTHY", "uptime_sec": time.time()})
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
        encoded = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
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
