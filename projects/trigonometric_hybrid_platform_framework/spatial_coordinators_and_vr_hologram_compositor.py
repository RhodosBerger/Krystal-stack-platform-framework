"""Multi-coordinator spatial architecture (Frostbite-style sector coordinators + Ubisoft-style
procedural LOD/frustum/instance culling), cycle-driven UI generator, VR hologram/holograph
ray-traced & animated profiler compositor, and CV + sensory telemetry mind-map synthesizer.
"""

from __future__ import annotations

import json
import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from krystal_kernel.constants import UMA_SATURATION_THRESHOLD, VITAL_MAX_HP, ContractViolation

from .clash_battle_royale import BOARD_RADIUS, ClashGame, board_cells, hex_distance
from .combat_turn_planner import DEFAULT_CACHE_DIR, probe_framework_dispositions
from .intel_accelerator_and_power_thread_governor import get_shared_ov_core
from .simulation_grid_3d import BABYLON_SRI, BABYLON_URL


# ---------------------------------------------------------------------------
# 1. Hierarchical Sector Coordinators + Procedural Render Culling & Instancing
# ---------------------------------------------------------------------------

SECTOR_COORDINATORS: Tuple[Dict[str, Any], ...] = (
    {"coord_id": "COORD_GLOBAL_DIRECTOR", "role": "Global Tick, Hash Chain & UMA < 0.78 Arbiter", "quadrant": "ALL"},
    {"coord_id": "COORD_SECTOR_NORTH_COMPUTE", "role": "Sublevel Squad, Destruction & Shader Cache (r > 0)", "quadrant": "NORTH"},
    {"coord_id": "COORD_SECTOR_EAST_SYNC", "role": "Sublevel Rollback & Netcode Interpolation (q > 0, r <= 0)", "quadrant": "EAST"},
    {"coord_id": "COORD_SECTOR_WEST_VERIFY", "role": "Sublevel Freivalds & Parity Barriers (q < 0, r >= 0)", "quadrant": "WEST"},
    {"coord_id": "COORD_SECTOR_SOUTH_OPS", "role": "Sublevel Telemetry & Zone Hazard Ring (r < 0, q <= 0)", "quadrant": "SOUTH"},
    {"coord_id": "COORD_PROCEDURAL_LOD_CULLER", "role": "GPU-Driven Rule Graph, Frustum Culling & Hex Instancing", "quadrant": "RENDER"},
    {"coord_id": "COORD_WORLD_DEFORMATION", "role": "Per-Hex Structural Integrity [0..6] & Voxel Crater Mask", "quadrant": "PHYSICS"},
)


def _assign_sector(q: int, r: int) -> str:
    if r > 0 and q >= 0:
        return "COORD_SECTOR_NORTH_COMPUTE"
    if q > 0 and r <= 0:
        return "COORD_SECTOR_EAST_SYNC"
    if q < 0 and r >= 0:
        return "COORD_SECTOR_WEST_VERIFY"
    return "COORD_SECTOR_SOUTH_OPS"


def evaluate_spatial_coordinators_and_procedural_render(
    board_radius: int = BOARD_RADIUS,
    camera_pos: Tuple[float, float, float] = (0.0, 11.0, -13.0),
    camera_fov_deg: float = 65.0,
    dirty_cells: Optional[List[Tuple[int, int]]] = None,
    uma_utilization: float = 0.35,
) -> Dict[str, Any]:
    """Partition space across hierarchical coordinators and compute procedural LOD + instancing stats."""
    if board_radius < 1 or not (20.0 <= camera_fov_deg <= 140.0):
        raise ContractViolation("Invalid spatial coordinator parameters", violation_code="ERR_COORD_PARAMS")

    cells = board_cells(board_radius)
    dirty_set = set(dirty_cells or [(0, 0), (1, 0), (-1, 0)])
    sq3 = math.sqrt(3.0)

    sector_buckets: Dict[str, Dict[str, Any]] = {
        c["coord_id"]: {"cells": 0, "dirty_cells": 0, "skipped_clean_eval": False}
        for c in SECTOR_COORDINATORS
        if c["quadrant"] in ("NORTH", "EAST", "WEST", "SOUTH")
    }

    # Camera forward vector toward origin (0, 0, 0)
    cx, cy, cz = camera_pos
    cam_norm = math.sqrt(cx * cx + cy * cy + cz * cz) or 1.0
    fwd = (-cx / cam_norm, -cy / cam_norm, -cz / cam_norm)
    cos_half_fov = math.cos(math.radians(camera_fov_deg * 0.55))

    lod_counts = {"LOD0_HIGH_240_TRIS": 0, "LOD1_MID_96_TRIS": 0, "LOD2_LOW_24_TRIS": 0, "CULLED_FRUSTUM": 0}
    total_tris = 0
    naive_tris = len(cells) * 240
    visible_instances = 0

    for q, r in cells:
        sec = _assign_sector(q, r)
        sector_buckets[sec]["cells"] += 1
        if (q, r) in dirty_set:
            sector_buckets[sec]["dirty_cells"] += 1

        wx, wy, wz = sq3 * (q + r * 0.5), 0.0, 1.5 * r
        vx, vy, vz = wx - cx, wy - cy, wz - cz
        dist = math.sqrt(vx * vx + vy * vy + vz * vz) or 1.0
        cos_ang = (vx * fwd[0] + vy * fwd[1] + vz * fwd[2]) / dist

        if cos_ang < cos_half_fov:
            lod_counts["CULLED_FRUSTUM"] += 1
            continue
        visible_instances += 1
        if dist < 12.5 and uma_utilization < UMA_SATURATION_THRESHOLD:
            lod_counts["LOD0_HIGH_240_TRIS"] += 1
            total_tris += 240
        elif dist < 17.5:
            lod_counts["LOD1_MID_96_TRIS"] += 1
            total_tris += 96
        else:
            lod_counts["LOD2_LOW_24_TRIS"] += 1
            total_tris += 24

    for sec_data in sector_buckets.values():
        sec_data["skipped_clean_eval"] = sec_data["dirty_cells"] == 0

    # Procedural instancing batches all identical LOD hex meshes + 1 unit cylinder batch = <= 4 draw calls
    instanced_draw_calls = sum(1 for k, v in lod_counts.items() if k != "CULLED_FRUSTUM" and v > 0) + 1
    naive_draw_calls = len(cells) + 8

    return {
        "vital_max_hp": VITAL_MAX_HP,
        "coordinators_total": len(SECTOR_COORDINATORS),
        "coordinators": list(SECTOR_COORDINATORS),
        "sector_partition": sector_buckets,
        "procedural_render_pipeline": {
            "total_cells": len(cells),
            "visible_instances": visible_instances,
            "lod_distribution": lod_counts,
            "naive_draw_calls": naive_draw_calls,
            "instanced_draw_calls": instanced_draw_calls,
            "draw_call_reduction_ratio": round(naive_draw_calls / max(1, instanced_draw_calls), 2),
            "naive_triangles": naive_tris,
            "active_lod_triangles": total_tris,
            "triangle_saving_pct": round(100.0 * (1.0 - total_tris / max(1, naive_tris)), 2),
        },
        "evidence": "DERIVED_PROCEDURAL_GEOMETRY_MODEL",
    }


# ---------------------------------------------------------------------------
# 2. Execution Cycles, Guard Conditions & Generative UI Schema
# ---------------------------------------------------------------------------

EXECUTION_CYCLES: Tuple[Dict[str, Any], ...] = (
    {
        "cycle_id": "CYCLE_C0_SCANOUT_VR_8_33MS",
        "period_ms": 8.33,
        "condition": "always_active; if uma >= 0.78 -> drop LOD0 to LOD1 & suppress hologram bloom",
        "bound_panel": "VR_HOLOGRAM_VIEWPORT",
    },
    {
        "cycle_id": "CYCLE_C1_SIM_ROLLBACK_16_67MS",
        "period_ms": 16.67,
        "condition": "dirty_sectors > 0 or pending_input_frames > 0 -> STEP_GATE_PROXY commit",
        "bound_panel": "SECTOR_COORDINATOR_HUD",
    },
    {
        "cycle_id": "CYCLE_C2_TELEMETRY_250MS",
        "period_ms": 250.0,
        "condition": "EWMA + 2*sigma_upper >= 0.78 -> immediate CPU fallback; 12 clean ticks -> upscale",
        "bound_panel": "MEMORY_AND_THREAD_GAUGES",
    },
    {
        "cycle_id": "CYCLE_C3_EPOCH_AUDIT_5000MS",
        "period_ms": 5000.0,
        "condition": "active_conflicts > 0 -> highlight FREIVALDS_AUDIT & refresh mind-map",
        "bound_panel": "MINDMAP_CAPABILITY_GRAPH",
    },
)


def compile_cycle_driven_ui_schema(uma_utilization: float = 0.35, vr_preferred: bool = True) -> Dict[str, Any]:
    """Generate declarative UI layout & widget tree from execution cycles and hardware thresholds."""
    disp = probe_framework_dispositions(uma_utilization=uma_utilization)
    uma_high = uma_utilization >= UMA_SATURATION_THRESHOLD
    widgets = [
        {
            "widget_id": "W01_VR_HOLOGRAM_STAGE",
            "mode": "WEBXR_IMMERSIVE_PREFERRED" if vr_preferred else "DESKTOP_3D_ORBIT",
            "lod_tier": "LOD1_CONSERVATIVE" if uma_high else "LOD0_FULL_HOLOGRAPH",
            "cycle": "CYCLE_C0_SCANOUT_VR_8_33MS",
        },
        {
            "widget_id": "W02_SECTOR_DIRTY_MATRIX",
            "coordinators": len(SECTOR_COORDINATORS),
            "usable_128b_frames": disp["usable_frame_budget"],
            "cycle": "CYCLE_C1_SIM_ROLLBACK_16_67MS",
        },
        {
            "widget_id": "W03_UMA_AND_ISA_STRIP",
            "devices": disp["openvino_devices"],
            "avx512f": disp["isa"]["flags"].get("AVX512F"),
            "alert_state": "UMA_SHADOW_THROTTLE" if uma_high else "NOMINAL",
            "cycle": "CYCLE_C2_TELEMETRY_250MS",
        },
        {
            "widget_id": "W04_SENSORY_MINDMAP_OVERLAY",
            "ontology_integrity": disp["ontology_integrity"],
            "conflicts": [k for k, v in disp["check_verdicts"].items() if v == "CONFLICT"],
            "cycle": "CYCLE_C3_EPOCH_AUDIT_5000MS",
        },
    ]
    return {
        "vital_max_hp": VITAL_MAX_HP,
        "vr_preferred": vr_preferred,
        "aesthetic_contract": {
            "background": "#030406",
            "hologram_primary": "#58D8FF",
            "hologram_secondary": "#50EEE0",
            "accent_gold": "#D8AE4B",
            "flat_crisp_borders_only": True,
        },
        "cycles": list(EXECUTION_CYCLES),
        "generated_widgets": widgets,
    }


# ---------------------------------------------------------------------------
# 3. VR Hologram / Holograph Compositor with Analytical Ray-Tracing & Animation
# ---------------------------------------------------------------------------

def build_and_trace_hologram_scene(
    phase_t: float = 0.25,
    uma_utilization: float = 0.35,
    ray_count: int = 16,
) -> Dict[str, Any]:
    """Construct animated 3D engine-profiling holograms and trace analytical rays against them."""
    if ray_count < 1 or ray_count > 256:
        raise ContractViolation("ray_count must be in [1, 256]", violation_code="ERR_HOLO_RAYS")

    game = ClashGame("SYNC", seed=6)
    sq3 = math.sqrt(3.0)
    nodes: List[Dict[str, Any]] = []

    # Animated holographic nodes: 8 arena units + 4 cardinal sector coordinator pylons
    for idx, (uid, u) in enumerate(sorted(game.state["units"].items())):
        q, r = u["pos"]
        bx, bz = sq3 * (q + r * 0.5), 1.5 * r
        bob_y = 0.8 + 0.18 * math.sin(2.0 * math.pi * phase_t + idx * 0.7)
        radius = 0.45
        nodes.append(
            {
                "hologram_id": f"HOLO_UNIT_{uid}",
                "kind": "HOLOGRAM_PILLAR",
                "center": [round(bx, 4), round(bob_y, 4), round(bz, 4)],
                "bounding_radius": radius,
                "hp": u["hp"],
                "color": "#58D8FF" if u["verdict"] != "CONFLICT" else "#FF307C",
            }
        )

    for idx, (cid, (px, pz)) in enumerate(
        [
            ("COORD_NORTH", (0.0, 5.2)),
            ("COORD_EAST", (5.2, 0.0)),
            ("COORD_WEST", (-5.2, 0.0)),
            ("COORD_SOUTH", (0.0, -5.2)),
        ]
    ):
        rot_rad = round((2.0 * math.pi * phase_t + idx * math.pi / 2.0) % (2.0 * math.pi), 4)
        nodes.append(
            {
                "hologram_id": f"HOLO_{cid}",
                "kind": "HOLOGRAPH_BEACON_RING",
                "center": [px, round(1.8 + 0.12 * math.cos(rot_rad), 4), pz],
                "bounding_radius": 0.65,
                "rotation_y_rad": rot_rad,
                "color": "#D8AE4B" if uma_utilization < UMA_SATURATION_THRESHOLD else "#F59E0B",
            }
        )

    # Analytical ray-sphere intersection from VR headset origin (0, 2.2, -7.5) sweeping across scene
    origin = np.array([0.0, 2.2, -7.5], dtype=np.float64)
    traced_rays: List[Dict[str, Any]] = []
    hits = 0
    for i in range(ray_count):
        az = -0.65 + (1.30 * i / max(1, ray_count - 1))
        direction = np.array([math.sin(az), -0.16, math.cos(az)], dtype=np.float64)
        direction /= np.linalg.norm(direction)

        best_t: Optional[float] = None
        hit_id: Optional[str] = None
        for nd in nodes:
            c = np.array(nd["center"], dtype=np.float64)
            rad = float(nd["bounding_radius"])
            oc = origin - c
            b = float(np.dot(oc, direction))
            c_term = float(np.dot(oc, oc) - rad * rad)
            disc = b * b - c_term
            if disc >= 0.0:
                t_cand = -b - math.sqrt(disc)
                if t_cand > 0.01 and (best_t is None or t_cand < best_t):
                    best_t = t_cand
                    hit_id = nd["hologram_id"]

        if best_t is not None:
            hits += 1
            hit_pt = (origin + best_t * direction).tolist()
        else:
            hit_pt = (origin + 14.0 * direction).tolist()
        traced_rays.append(
            {
                "ray_index": i,
                "origin": [round(v, 3) for v in origin.tolist()],
                "endpoint": [round(v, 3) for v in hit_pt],
                "hit_hologram": hit_id,
                "distance": None if best_t is None else round(best_t, 4),
            }
        )

    return {
        "vital_max_hp": VITAL_MAX_HP,
        "phase_t": phase_t,
        "hologram_nodes": nodes,
        "ray_trace_summary": {"rays_cast": ray_count, "hits": hits, "rays": traced_rays},
        "ui_schema": compile_cycle_driven_ui_schema(uma_utilization=uma_utilization, vr_preferred=True),
    }


def render_vr_hologram_profiler_html(uma_utilization: float = 0.35) -> str:
    """Render WebXR-capable Babylon.js holographic engine profiler (strict Krystal UI palette)."""
    scene_data = build_and_trace_hologram_scene(phase_t=0.15, uma_utilization=uma_utilization, ray_count=20)
    spatial = evaluate_spatial_coordinators_and_procedural_render(uma_utilization=uma_utilization)
    payload = json.dumps({"hologram": scene_data, "spatial": spatial}, ensure_ascii=False)
    return f"""<!DOCTYPE html>
<html lang="sk">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>Krystal · VR Holografický Profilovač Enginu</title>
<style>
  :root {{ --void:#030406; --navy:#101124; --panel:#11131D; --line:#292A38; --txt:#F1EFE8;
           --muted:#8D8997; --gold:#D8AE4B; --cyan:#58D8FF; --teal:#50EEE0; --mag:#FF307C; }}
  * {{ box-sizing:border-box; }}
  html, body {{ margin:0; height:100%; background:var(--void); color:var(--txt); font-family:"JetBrains Mono",Consolas,monospace; font-size:12px; }}
  header {{ height:68px; background:var(--navy); border-bottom:1px solid var(--line); display:flex; align-items:center; padding:0 24px; gap:16px; }}
  .brand {{ color:var(--gold); font-size:20px; font-weight:700; }}
  #wrap {{ display:grid; grid-template-columns:minmax(0,2.2fr) minmax(340px,1fr); height:calc(100% - 68px); }}
  #cv {{ width:100%; height:100%; display:block; outline:none; background:var(--void); }}
  aside {{ border-left:1px solid var(--line); padding:18px; overflow:auto; background:#070812; }}
  .card {{ background:var(--panel); border:1px solid var(--line); border-radius:2px; padding:12px; margin-bottom:12px; }}
  .card h4 {{ margin:0 0 8px; color:var(--cyan); font-size:12px; letter-spacing:0.12em; text-transform:uppercase; }}
  .row {{ display:flex; justify-content:space-between; border-top:1px solid var(--line); padding:4px 0; }}
  .row span:first-child {{ color:var(--muted); }} .row span:last-child {{ color:var(--teal); }}
  button {{ background:transparent; color:var(--gold); border:1px solid var(--gold); border-radius:2px; padding:7px 12px; cursor:pointer; font-family:inherit; }}
</style>
</head>
<body>
<header>
  <div>
    <div style="font-size:10px;color:var(--muted);letter-spacing:0.2em">WEBXR PREFERRED · HOLOGRAM &amp; RAY-TRACE COMPOSITOR</div>
    <div class="brand">VR Holografický Profilovač Enginu</div>
  </div>
  <div style="margin-left:auto;display:flex;gap:8px">
    <button id="toggleAnim">Pozastaviť / Spustiť animáciu</button>
    <button id="enterVr">Vstúpiť do VR (WebXR)</button>
  </div>
</header>
<div id="wrap">
  <canvas id="cv"></canvas>
  <aside id="rail"></aside>
</div>
<script src="vendor/babylon.js"></script>
<script>if (!window.BABYLON) {{ document.write('<script src="{BABYLON_URL}" integrity="{BABYLON_SRI}" crossorigin="anonymous"><\\/script>'); }}</script>
<script>
(function() {{
  const P = {payload};
  const pr = P.spatial.procedural_render_pipeline;
  document.getElementById("rail").innerHTML =
    `<div class="card"><h4>Koordinátory priestoru (Frostbite/Ubisoft štýl)</h4>` +
    `<div class="row"><span>Aktívne koordinátory</span><span>${{P.spatial.coordinators_total}}</span></div>` +
    `<div class="row"><span>Viditeľné inštancie</span><span>${{pr.visible_instances}} / ${{pr.total_cells}}</span></div>` +
    `<div class="row"><span>Redukcia draw-callov</span><span>${{pr.naive_draw_calls}} -> ${{pr.instanced_draw_calls}} (${{pr.draw_call_reduction_ratio}}x)</span></div>` +
    `<div class="row"><span>Úspora trojuholníkov LOD</span><span>${{pr.triangle_saving_pct}} %</span></div></div>` +
    `<div class="card"><h4>Trasovanie hologramov</h4>` +
    `<div class="row"><span>Holografické uzly</span><span>${{P.hologram.hologram_nodes.length}}</span></div>` +
    `<div class="row"><span>Lúče / Zásahy</span><span>${{P.hologram.ray_trace_summary.rays_cast}} / ${{P.hologram.ray_trace_summary.hits}}</span></div>` +
    `<div class="row"><span>Preferovaný režim</span><span>WebXR Immersive VR</span></div></div>`;

  if (!window.BABYLON) return;
  const canvas = document.getElementById("cv");
  const engine = new BABYLON.Engine(canvas, true);
  const scene = new BABYLON.Scene(engine);
  scene.clearColor = new BABYLON.Color4(0.012, 0.016, 0.024, 1.0);
  const cam = new BABYLON.ArcRotateCamera("cam", -Math.PI/2, 1.12, 14, BABYLON.Vector3.Zero(), scene);
  cam.attachControl(canvas, true);
  new BABYLON.HemisphericLight("light", new BABYLON.Vector3(0, 1, 0), scene).intensity = 0.85;

  let anim = true, t = 0;
  const meshes = P.hologram.hologram_nodes.map((n, idx) => {{
    const m = n.kind === "HOLOGRAM_PILLAR"
      ? BABYLON.MeshBuilder.CreateCylinder(n.hologram_id, {{diameter: n.bounding_radius*2, height: 1.2, tessellation: 6}}, scene)
      : BABYLON.MeshBuilder.CreateTorus(n.hologram_id, {{diameter: n.bounding_radius*2, thickness: 0.08, tessellation: 24}}, scene);
    m.position.set(n.center[0], n.center[1], n.center[2]);
    const mat = new BABYLON.StandardMaterial(n.hologram_id + "_m", scene);
    mat.emissiveColor = BABYLON.Color3.FromHexString(n.color);
    mat.wireframe = true;
    m.material = mat;
    return {{mesh: m, base: n.center[1], idx}};
  }});

  P.hologram.ray_trace_summary.rays.forEach((r, i) => {{
    const line = BABYLON.MeshBuilder.CreateLines("ray_" + i, {{
      points: [new BABYLON.Vector3(...r.origin), new BABYLON.Vector3(...r.endpoint)]
    }}, scene);
    line.color = BABYLON.Color3.FromHexString(r.hit_hologram ? "#50EEE0" : "#292A38");
  }});

  document.getElementById("toggleAnim").onclick = () => {{ anim = !anim; }};
  document.getElementById("enterVr").onclick = async () => {{
    if (scene.createDefaultXRExperienceAsync) {{
      try {{
        const xr = await scene.createDefaultXRExperienceAsync({{}});
        if (xr.baseExperience) await xr.baseExperience.enterXRAsync("immersive-vr", "local-floor");
      }} catch (e) {{ alert("WebXR headset nie je pripojený; beží 3D desktopový holografický náhľad."); }}
    }}
  }};

  engine.runRenderLoop(() => {{
    if (anim) {{
      t += 0.02;
      meshes.forEach(o => {{ o.mesh.position.y = o.base + 0.16 * Math.sin(t + o.idx * 0.7); o.mesh.rotation.y = t * 0.5; }});
    }}
    scene.render();
  }});
  window.addEventListener("resize", () => engine.resize());
}})();
</script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# 4. Computer Vision + Sensory Telemetry -> Mind-Map Capability Graph
# ---------------------------------------------------------------------------

def synthesize_vision_sensory_mindmap(uma_utilization: float = 0.35) -> Dict[str, Any]:
    """Run an OpenVINO Conv2D Sobel/Laplacian vision pass over spatial occupancy + telemetry into a capability mind map."""
    import openvino.opset13 as ops

    ov, core, _ = get_shared_ov_core(DEFAULT_CACHE_DIR)
    disp = probe_framework_dispositions(uma_utilization=uma_utilization)

    # Encode 37-hex arena into a 32x32 sensory occupancy image and run OpenVINO 2-filter CV edge/energy detector
    img = np.zeros((1, 1, 32, 32), dtype=np.float32)
    game = ClashGame("VERIFY", seed=6)
    for u in game.state["units"].values():
        gx = min(31, max(0, 16 + int(u["pos"][0]) * 4))
        gy = min(31, max(0, 16 + int(u["pos"][1]) * 4))
        img[0, 0, gy, gx] = float(u["hp"]) / float(VITAL_MAX_HP)

    sobel_x = np.array([[[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]]], dtype=np.float32)
    laplace = np.array([[[[0, 1, 0], [1, -4, 1], [0, 1, 0]]]], dtype=np.float32)
    kernels = np.concatenate([sobel_x, laplace], axis=0)  # (2, 1, 3, 3)

    param = ops.parameter([1, 1, 32, 32], np.float32, name="sensory_frame")
    cv_model = ov.Model([ops.convolution(param, ops.constant(kernels), [1, 1], [1, 1], [1, 1], [1, 1])], [param], "cv_mindmap_sensor")
    compiled = core.compile_model(cv_model, "CPU", {"INFERENCE_PRECISION_HINT": "f32"})
    feat = np.array(compiled.create_infer_request().infer({0: img})[compiled.output(0)], copy=True)

    edge_energy = round(float(np.linalg.norm(feat[0, 0])), 4)
    laplacian_peak = round(float(np.max(np.abs(feat[0, 1]))), 4)

    mindmap_nodes = [
        {"id": "ROOT", "label": "Krystal Sensory & Telemetry Mind-Map", "parent": None},
        {"id": "BR_VISION", "label": f"Počítačové videnie (Sobel={edge_energy}, Laplace={laplacian_peak})", "parent": "ROOT",
         "capability_fn": "virtual_convolution_pipeline:run_virtual_convolution_pipeline"},
        {"id": "BR_SILICON", "label": f"x86-64 & oneAPI Kremík ({', '.join(disp['openvino_devices'])}, AVX512={disp['isa']['flags'].get('AVX512F')})",
         "parent": "ROOT", "capability_fn": "x86_platform_findings:run_x86_platform_findings_computation"},
        {"id": "BR_SPATIAL", "label": "7 Priestorových Koordinátorov & VR Hologramy", "parent": "ROOT",
         "capability_fn": "spatial_coordinators_and_vr_hologram_compositor:build_and_trace_hologram_scene"},
        {"id": "BR_ARENA", "label": f"Hex Súboje & AI Reakcie (Rozpočet {disp['usable_frame_budget']} rámcov)", "parent": "ROOT",
         "capability_fn": "hex_combat_bandwidth_and_world_ai:resolve_action_with_world_and_ai_chain"},
        {"id": "BR_NEURO", "label": "Neuro-Symbolické Vetvenie & Citovaný Jazykový Model", "parent": "ROOT",
         "capability_fn": "neurosymbolic_genesis_discipline_and_llm:NeuroSymbolicMicroLLM"},
    ]

    mermaid = "\n".join(
        [
            "mindmap",
            "  root((Krystal Engine))",
            f"    Videnie_Sobel_{edge_energy}",
            "      run_virtual_convolution_pipeline",
            "    Kremik_x86_AVX512",
            "      run_x86_platform_findings_computation",
            "      IntelOneApiAccelerator",
            "    Priestor_a_VR_Hologram",
            "      evaluate_spatial_coordinators_and_procedural_render",
            "      build_and_trace_hologram_scene",
            "    Hex_Suboje_a_AI",
            "      compute_full_combat_bandwidth_solution",
            "      resolve_action_with_world_and_ai_chain",
            "    NeuroSymbolicky_Model",
            "      NeuroSymbolicMicroLLM",
        ]
    )

    return {
        "vital_max_hp": VITAL_MAX_HP,
        "cv_sensory_metrics": {"sobel_edge_energy": edge_energy, "laplacian_peak": laplacian_peak, "input_shape": [1, 1, 32, 32]},
        "mindmap_nodes": mindmap_nodes,
        "mermaid_mindmap": mermaid,
        "evidence": "LOCAL_MEASURED (OpenVINO CV pass) + DERIVED (mind-map graph)",
    }
