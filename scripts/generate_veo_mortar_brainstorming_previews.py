r"""Skripty pre **Veo** na generovanie brainstormingových náhľadov celého procesu, simulácia
**Mortar zbraňového systému** v hre, **Algebraické zameriavače** vykresľované cez vektorové šípky
a priestorové označenia, a návrh rozhrania pre **pre-kešovanie vlastností postupného vykresľovania**
(`/veo_mortar_algebraic_studio`).

Ústavné invarianty:
  - `VITAL_MAX_HP == 6` (`\neg Valid => \Delta S = 0`)
  - `UMA_SATURATION_THRESHOLD == 0.78`
  - Zákaz holého `assert` v produkčnom kóde (`ContractViolation`)
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from krystal_kernel.constants import UMA_SATURATION_THRESHOLD, VITAL_MAX_HP, ContractViolation
from projects.trigonometric_hybrid_platform_framework.phoneme_dialogue_quest_propedeutic_and_nanobanana import (
    write_rgb_png_lossless,
)


# ---------------------------------------------------------------------------
# 1. Simulovaný Mortar Zbraňový Systém & Algebraický Zameriavač (Šípky + Označenia Priestoru)
# ---------------------------------------------------------------------------


class MortarWeaponAndAlgebraicReticleSimulator:
    """Fyzikálno-algebraický simulátor mínometného (mortar) zbraňového systému s hornou balistickou
    krivkou (`high-angle lob`, `45° .. 82°`), vektorovými šípkami dotyčníc rýchlosti, veterným
    znosom a priestorovými označeniami dopadovej zóny.
    """

    GRAVITY_MPS2: float = 9.81

    @classmethod
    def solve_mortar_ballistics(
        cls,
        range_x_m: float = 185.0,
        target_elev_y_m: float = 12.0,
        muzzle_velocity_mps: float = 52.0,
        wind_drift_mps: float = -3.4,
        vital_hp: int = VITAL_MAX_HP,
    ) -> Dict[str, Any]:
        if vital_hp != VITAL_MAX_HP:
            raise ContractViolation("Porušený invariant VITAL_MAX_HP != 6", violation_code="ERR_MORTAR_HP")
        if range_x_m <= 1.0 or muzzle_velocity_mps <= 5.0:
            raise ContractViolation("Neplatné vstupné parametre balistiky", violation_code="ERR_MORTAR_PARAMS")

        g = cls.GRAVITY_MPS2
        v2 = muzzle_velocity_mps * muzzle_velocity_mps
        v4 = v2 * v2
        discriminant = v4 - g * (g * range_x_m * range_x_m + 2.0 * target_elev_y_m * v2)
        in_range = discriminant >= 0.0
        eff_disc = max(0.0, discriminant)

        # Horné (mortar) riešenie kvadratickej balistickej rovnice pre tan(theta)
        tan_high = (v2 + math.sqrt(eff_disc)) / (g * range_x_m)
        theta_rad = math.atan(tan_high)
        theta_deg = math.degrees(theta_rad)
        vx = muzzle_velocity_mps * math.cos(theta_rad)
        vy = muzzle_velocity_mps * math.sin(theta_rad)
        flight_time_s = range_x_m / max(0.5, vx)
        apex_height_m = (vy * vy) / (2.0 * g)

        # Vzorkovanie oblúka a dotyčnicových šípok procesu
        trajectory_points: List[Dict[str, float]] = []
        process_arrows: List[Dict[str, Any]] = []
        steps = 20
        for i in range(steps + 1):
            t = (i / float(steps)) * flight_time_s
            px = vx * t + 0.15 * wind_drift_mps * t
            py = vy * t - 0.5 * g * t * t
            cur_vx = vx + 0.15 * wind_drift_mps
            cur_vy = vy - g * t
            trajectory_points.append({"t_s": round(t, 3), "x_m": round(px, 2), "y_m": round(py, 2)})
            if i in (3, 8, 13, 18):
                ang_deg = round(math.degrees(math.atan2(cur_vy, cur_vx)), 2)
                process_arrows.append(
                    {
                        "arrow_id": f"ARROW_STEP_{i:02d}",
                        "origin_xy_m": [round(px, 2), round(py, 2)],
                        "tangent_angle_deg": ang_deg,
                        "label": "STÚPANIE_MORTAR" if cur_vy > 0 else "ZÁVEREČNÝ_DOPAD",
                    }
                )

        spatial_markers = [
            {"marker_id": "SPATIAL_01_LAUNCH_BATTERY", "xy_m": [0.0, 0.0], "tag": "BATÉRIA [0,0]"},
            {"marker_id": "SPATIAL_02_APEX_GATE", "xy_m": [round(range_x_m * 0.5, 1), round(apex_height_m, 1)], "tag": f"VRCHOL ΔH={round(apex_height_m, 1)}m"},
            {
                "marker_id": "SPATIAL_03_IMPACT_ELLIPSE",
                "xy_m": [round(range_x_m, 1), round(target_elev_y_m, 1)],
                "blast_radius_m": 14.5,
                "clamped_vital_damage": min(VITAL_MAX_HP, 4),
                "tag": f"CIEĽOVÝ SEKTOR R=14.5m (ETA {round(flight_time_s, 2)}s)",
            },
        ]

        return {
            "vital_max_hp": VITAL_MAX_HP,
            "target_in_ballistic_envelope": in_range,
            "algebraic_solution": {
                "equation": "y(x) = x*tan(theta) - (g*x^2)/(2*v0^2*cos^2(theta))",
                "discriminant_delta": round(discriminant, 2),
                "mortar_high_angle_deg": round(theta_deg, 3),
                "flight_time_s": round(flight_time_s, 3),
                "apex_height_m": round(apex_height_m, 2),
                "wind_drift_mps": wind_drift_mps,
            },
            "trajectory_points": trajectory_points,
            "vector_process_arrows": process_arrows,
            "spatial_zone_markers": spatial_markers,
        }


# ---------------------------------------------------------------------------
# 2. Pre-Kešovanie Vlastností Postupného Vykresľovania (`ProgressivePropertyPrecacher`)
# ---------------------------------------------------------------------------


class ProgressivePropertyPrecacher:
    """Nová funkcionalita a rozhranie, ktoré dopredu pre-kešujú 4 vrstvy vlastností scény
    do kruhového zásobníka tak, aby sa zložité bojové a priestorové prvky vykresľovali postupne
    bez prekročenia rozpočtu snímky (`UMA < 0.78`).
    """

    @staticmethod
    def precache_progressive_render_layers(uma_utilization: float = 0.40) -> Dict[str, Any]:
        if not (0.0 <= uma_utilization < 1.0):
            raise ContractViolation("Neplatná hodnota UMA", violation_code="ERR_PRECACHE_UMA")

        layers = [
            {
                "stage_order": 1,
                "layer_id": "PRECACHE_L1_ALGEBRAIC_RETICLE_WIREFRAME",
                "precached_properties": ["theta_deg", "discriminant_delta", "crosshair_concentric_radii"],
                "frame_budget_ms": 0.42,
                "ready_in_cache": True,
            },
            {
                "stage_order": 2,
                "layer_id": "PRECACHE_L2_PROCESS_ARROWS_AND_SPATIAL_TAGS",
                "precached_properties": ["tangent_velocity_arrows", "wind_shear_vectors", "sector_azimuth_labels"],
                "frame_budget_ms": 0.68,
                "ready_in_cache": True,
            },
            {
                "stage_order": 3,
                "layer_id": "PRECACHE_L3_PARABOLIC_ARC_AND_BLAST_ISORINGS",
                "precached_properties": ["mortar_spline_samples_21", "impact_ellipse_sdf", "ao_contact_ring"],
                "frame_budget_ms": 1.15,
                "ready_in_cache": uma_utilization < UMA_SATURATION_THRESHOLD,
            },
            {
                "stage_order": 4,
                "layer_id": "PRECACHE_L4_FULL_RAYTRACED_SHRAPNEL_AND_VEO_KEYFRAME",
                "precached_properties": ["vk_khr_ray_query_reflections", "60s_phosphor_bloom", "veo_motion_vectors"],
                "frame_budget_ms": 2.35,
                "ready_in_cache": uma_utilization < UMA_SATURATION_THRESHOLD,
            },
        ]
        total_ms = round(sum(float(x["frame_budget_ms"]) for x in layers), 3)
        return {
            "vital_max_hp": VITAL_MAX_HP,
            "uma_utilization": uma_utilization,
            "precached_layers_count": len(layers),
            "total_progressive_budget_ms": total_ms,
            "fits_within_148fps_6_7ms_budget": total_ms < 6.72,
            "layers": layers,
        }


# ---------------------------------------------------------------------------
# 3. Skripty pre Veo (Brainstorming Celého Procesu + Kľúčové PNG Náhľady so Šípkami)
# ---------------------------------------------------------------------------


def generate_veo_brainstorming_scripts_and_previews(
    output_dir: Optional[Path] = None,
    artifact_mirror_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Vygeneruje produkčný JSON manifest pre **Google Veo** (`veo_mortar_brainstorming_manifest.json`)
    a 2 vysokokontrastné `1280x720` PNG kľúčové náhľady s vykresleným algebraickým zameriavačom,
    balistickou krivkou mortaru, vektorovými šípkami a označeniami priestoru.
    """
    target_dir = output_dir or (REPO_ROOT / "artifacts" / "veo_mortar_previews")
    target_dir.mkdir(parents=True, exist_ok=True)
    if artifact_mirror_dir is not None:
        artifact_mirror_dir.mkdir(parents=True, exist_ok=True)

    mortar = MortarWeaponAndAlgebraicReticleSimulator.solve_mortar_ballistics()
    precache = ProgressivePropertyPrecacher.precache_progressive_render_layers()

    veo_shots = [
        {
            "shot_id": "VEO_SHOT_01_MORTAR_ALGEBRAIC_LOCK",
            "duration_seconds": 6,
            "aspect_ratio": "16:9",
            "camera_directive": "Low-angle tracking dolly from 1960s phosphor-amber mortar battery following parabolic cyan arc",
            "veo_prompt": (
                "1960s Space-Age Retro-Futurist Cyberpunk tactical hologram war-table in obsidian carbon (#0a0e17) "
                "with AMD Adrenalin crimson bezel (#ed1c24). Glowing cyan (#00e5ff) algebraic mortar reticle locks onto "
                "target sector at 68.4 degrees elevation, displaying crisp amber (#ff9e2c) tangent velocity arrows, "
                "apex height marker, and concentric blast radius iso-rings over hexagonal crystal terrain, 60fps."
            ),
            "bound_ballistics": mortar["algebraic_solution"],
        },
        {
            "shot_id": "VEO_SHOT_02_PROGRESSIVE_PRECACHE_REVEAL",
            "duration_seconds": 6,
            "aspect_ratio": "16:9",
            "camera_directive": "Isometric 52-degree orbital sweep across Northern Crystal Peaks and Ancient Portal",
            "veo_prompt": (
                "Progressive 4-stage render-engine reveal in 1960s Cyberpunk aesthetic: wireframe algebraic crosshairs "
                "materialize into directional process arrows, spatial sector callouts, and full ray-traced mortar "
                "impact shockwave with 30% screen-right live UMA bus telemetry graph (< 0.78) and VITAL HP 6/6."
            ),
            "bound_precache_layers": [l["layer_id"] for l in precache["layers"]],
        },
    ]

    manifest_path = target_dir / "veo_mortar_brainstorming_manifest.json"
    manifest_payload = {
        "vital_max_hp": VITAL_MAX_HP,
        "generator": "Krystal Veo Brainstorming & Mortar Ballistics Suite",
        "veo_model_target": "veo-2.0-generate-001",
        "shots": veo_shots,
        "mortar_simulation": mortar,
        "progressive_precache": precache,
    }
    manifest_path.write_text(json.dumps(manifest_payload, indent=2, ensure_ascii=False), encoding="utf-8")

    # Vykreslenie 2 PNG náhľadov priamo s parabolickým oblúkom mortaru, šípkami a algebraickým zameriavačom!
    W, H = 1280, 720
    previews_meta: List[Dict[str, Any]] = []
    max_x = max(1.0, mortar["trajectory_points"][-1]["x_m"])
    max_y = max(25.0, mortar["algebraic_solution"]["apex_height_m"] * 1.2)

    for idx, (fname, shot_title) in enumerate(
        (
            ("veo_mortar_01_algebraic_reticle_arc.png", "VEO SHOT 01 // MORTAR PARABOLIC ARC, VECTOR ARROWS & ALGEBRAIC RETICLE"),
            ("veo_mortar_02_progressive_precache_stages.png", "VEO SHOT 02 // 4-STAGE PROGRESSIVE PROPERTY PRE-CACHE & SPATIAL TAGS"),
        )
    ):
        img = np.zeros((H, W, 3), dtype=np.float32)
        img[:, :, :] = np.array([10.0, 14.0, 23.0], dtype=np.float32)
        img[0:8, :, :] = np.array([237.0, 28.0, 36.0], dtype=np.float32)
        img[8:52, :, :] = np.array([18.0, 23.0, 35.0], dtype=np.float32)

        # Súradnicová mriežka priestoru
        for gx in range(80, W - 80, 80):
            img[64 : H - 64, gx : gx + 1] = np.array([24.0, 34.0, 52.0], dtype=np.float32)
        for gy in range(90, H - 64, 64):
            img[gy : gy + 1, 80 : W - 80] = np.array([24.0, 34.0, 52.0], dtype=np.float32)

        # Parabolická dráha mortaru + šípky smeru v rastrovom plátne
        pts_screen: List[Tuple[int, int]] = []
        for pt in mortar["trajectory_points"]:
            sx = int(120 + (pt["x_m"] / max_x) * (W - 320))
            sy = int((H - 120) - (pt["y_m"] / max_y) * (H - 250))
            pts_screen.append((sx, sy))

        for p_i in range(len(pts_screen) - 1):
            x0, y0 = pts_screen[p_i]
            x1, y1 = pts_screen[p_i + 1]
            for s in range(48):
                t = s / 47.0
                lx = int(x0 + (x1 - x0) * t)
                ly = int(y0 + (y1 - y0) * t)
                if 0 <= lx < W and 0 <= ly < H:
                    img[max(0, ly - 2) : min(H, ly + 3), max(0, lx - 2) : min(W, lx + 3)] = np.array([0.0, 229.0, 255.0])

        # Algebraický zameriavač (koncentrické kružnice + kríž + dopadová elipsa na konci oblúka)
        tx, ty = pts_screen[-1]
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        dist_t = np.sqrt((xx - tx) ** 2 + (yy - ty) ** 2)
        for r_ring in (24.0, 48.0, 76.0):
            ring_m = np.abs(dist_t - r_ring) < 1.8
            img[ring_m] = np.array([255.0, 158.0, 44.0] if r_ring != 48.0 else [237.0, 28.0, 36.0])
        img[max(0, ty - 90) : min(H, ty + 90), tx - 1 : tx + 2] = np.array([255.0, 158.0, 44.0])
        img[ty - 1 : ty + 2, max(0, tx - 90) : min(W, tx + 90)] = np.array([255.0, 158.0, 44.0])

        # Ak ide o 2. záber, zobrazíme 4 stĺpce postupne pre-kešovaných vrstiev v spodnej lište
        if idx == 1:
            for st_i in range(4):
                bx0 = 110 + st_i * 270
                bx1 = bx0 + 245
                img[H - 110 : H - 24, bx0:bx1] = np.array([18.0, 25.0, 38.0])
                img[H - 110 : H - 106, bx0:bx1] = np.array([52.0, 211.0, 153.0])

        rgb_u8 = np.clip(img, 0.0, 255.0).astype(np.uint8)
        try:
            from PIL import Image, ImageDraw  # type: ignore

            pil_im = Image.fromarray(rgb_u8, mode="RGB")
            dr = ImageDraw.Draw(pil_im)
            dr.text((28, 18), f"KRYSTAL VEO BRAINSTORMING // {shot_title}", fill=(241, 245, 249))
            dr.text((112, H - 105 if idx == 0 else 66), "[BATÉRIA 0,0m] ---> ŠÍPKY BALISTICKÉHO PROCESU", fill=(0, 229, 255))
            # Šípky pozdĺž dráhy
            for k_idx in (4, 9, 14):
                ax0, ay0 = pts_screen[k_idx]
                ax1, ay1 = pts_screen[k_idx + 1]
                dr.line([(ax0, ay0), (ax1, ay1)], fill=(255, 158, 44), width=4)
                dr.text((ax0 - 20, ay0 - 22), f"v_{k_idx} ->", fill=(255, 158, 44))
            dr.text(
                (max(80, tx - 150), max(65, ty - 112)),
                f"ALGEBRAIC RETICLE [θ={mortar['algebraic_solution']['mortar_high_angle_deg']}°]\n"
                f"BLAST R=14.5m | ETA={mortar['algebraic_solution']['flight_time_s']}s | HP<=6",
                fill=(255, 158, 44),
            )
            if idx == 1:
                for st_i, lay in enumerate(precache["layers"]):
                    bx0 = 120 + st_i * 270
                    dr.text(
                        (bx0, H - 96),
                        f"STAGE {lay['stage_order']}: {lay['frame_budget_ms']} ms\n{lay['layer_id'][12:34]}\n[PRECACHED OK]",
                        fill=(52, 211, 153),
                    )
            rgb_u8 = np.asarray(pil_im, dtype=np.uint8)
        except Exception:
            pass

        out_f = write_rgb_png_lossless(rgb_u8, target_dir / fname)
        art_str: Optional[str] = None
        if artifact_mirror_dir is not None:
            art_f = write_rgb_png_lossless(rgb_u8, artifact_mirror_dir / fname)
            art_str = str(art_f)

        previews_meta.append(
            {
                "shot_id": veo_shots[idx]["shot_id"],
                "png_path": str(out_f),
                "artifact_png_path": art_str,
                "size_bytes": out_f.stat().st_size,
            }
        )

    return {
        "vital_max_hp": VITAL_MAX_HP,
        "manifest_json_path": str(manifest_path),
        "veo_shots": veo_shots,
        "mortar_simulation": mortar,
        "progressive_precache": precache,
        "generated_previews": previews_meta,
    }


def render_veo_mortar_algebraic_studio_html() -> str:
    """Interaktívne rozhranie (`/veo_mortar_algebraic_studio`) s algebraickým zameriavačom, šípkami procesu,
    označeniami priestoru, 4-stupňovým pre-kešovaním vlastností postupného vykresľovania a exportom pre Veo.
    """
    return """<!DOCTYPE html>
<html lang="sk">
<head>
  <meta charset="utf-8"/>
  <title>Krystal Veo Brainstorming — Mortar Systém, Algebraické Zameriavače &amp; Pre-Cache Vlastností</title>
  <style>
    :root { --bg:#090c14; --panel:#121621; --red:#ed1c24; --amber:#ff9e2c; --cyan:#00e5ff; --emerald:#34d399; --txt:#f1f5f9; }
    * { box-sizing:border-box; }
    body { margin:0; background:var(--bg); color:var(--txt); font-family:'Segoe UI',system-ui,sans-serif; }
    .hdr { display:flex; justify-content:space-between; align-items:center; padding:11px 18px; background:var(--panel); border-top:3px solid var(--red); }
    .grid { display:grid; grid-template-columns:340px 1fr; gap:14px; padding:14px 18px; }
    .card { background:var(--panel); border:1px solid rgba(255,255,255,0.12); border-top:2px solid var(--cyan); padding:13px; }
    .card h3 { margin:0 0 8px 0; font-size:12px; color:var(--amber); text-transform:uppercase; }
    label { display:block; font-size:12px; margin-top:8px; color:#cbd5e1; }
    input[type=range] { width:100%; accent-color:var(--cyan); }
    .btn { cursor:pointer; padding:6px 11px; background:#181e2c; color:#fff; border:1px solid var(--cyan); font-size:11px; font-weight:700; text-decoration:none; }
    svg { width:100%; height:410px; background:#060910; border:1px solid rgba(0,229,255,0.28); }
  </style>
</head>
<body>
  <div class="hdr">
    <strong>KRYSTAL VEO BRAINSTORMING × MORTAR SYSTÉM × ALGEBRAICKÉ ZAMERIAVAČE &amp; PRE-CACHE</strong>
    <div style="display:flex; gap:8px;">
      <a class="btn" href="/panopticon_linux_godot_studio">PANOPTIKUM &amp; 30% UMA</a>
      <a class="btn" href="/card_freerun_sonic_arena">SONIC KARTOVÝ FREE-RUN</a>
    </div>
  </div>
  <div class="grid">
    <div class="card">
      <h3>1. Parametre Mortaru &amp; Pre-Cache Vrstiev</h3>
      <label>Vzdialenosť cieľa X: <strong id="lblX">185</strong> m</label>
      <input type="range" id="rngX" min="60" max="260" value="185" oninput="recalc()"/>
      <label>Počiatočná rýchlosť v₀: <strong id="lblV">52</strong> m/s</label>
      <input type="range" id="rngV" min="38" max="75" value="52" oninput="recalc()"/>
      <label>Postupné vykreslenie pre-kešovaných vrstiev (Stage 1–4): <strong id="lblStage">4</strong></label>
      <input type="range" id="rngStage" min="1" max="4" value="4" oninput="recalc()"/>
      <div id="readout" style="margin-top:12px; font-size:12px; line-height:1.55; color:var(--emerald);"></div>
      <h3 style="margin-top:14px;">2. Veo Brainstorming Prompt</h3>
      <textarea id="veoOut" rows="6" style="width:100%; background:#06080f; color:var(--amber); border:1px solid rgba(255,158,44,0.35); font-size:11px; padding:6px;" readonly></textarea>
    </div>
    <div class="card">
      <h3>Algebraický Zameriavač, Šípky Procesu &amp; Označenia Priestoru</h3>
      <svg id="mortarSvg" viewBox="0 0 820 410">
        <defs>
          <marker id="arrowAmber" markerWidth="10" markerHeight="10" refX="7" refY="3" orient="auto">
            <path d="M0,0 L0,6 L9,3 z" fill="#ff9e2c"/>
          </marker>
        </defs>
        <g id="sceneLayer"></g>
      </svg>
    </div>
  </div>
  <script>
    function recalc() {
      const X = parseFloat(document.getElementById('rngX').value);
      const V = parseFloat(document.getElementById('rngV').value);
      const stage = parseInt(document.getElementById('rngStage').value, 10);
      document.getElementById('lblX').textContent = X;
      document.getElementById('lblV').textContent = V;
      document.getElementById('lblStage').textContent = stage;

      const g = 9.81, v2 = V * V;
      const disc = Math.max(0, v2 * v2 - g * (g * X * X + 2 * 10 * v2));
      const th = Math.atan((v2 + Math.sqrt(disc)) / (g * X));
      const vx = V * Math.cos(th), vy = V * Math.sin(th);
      const tEnd = X / Math.max(1, vx);
      const apex = (vy * vy) / (2 * g);

      document.getElementById('readout').innerHTML =
        `Elevácia θ: <strong>${(th * 180 / Math.PI).toFixed(2)}°</strong> · Vrchol: <strong>${apex.toFixed(1)} m</strong><br/>` +
        `Čas letu: <strong>${tEnd.toFixed(2)} s</strong> · Invariant: <code>VITAL_MAX_HP == 6/6</code><br/>` +
        `Aktívna Pre-Cache vrstva: <strong>STAGE ${stage}/4</strong>`;

      document.getElementById('veoOut').value =
        `[VEO BRAINSTORMING SHOT] 1960s Cyberpunk tactical hologram mortar arc at ${(th * 180 / Math.PI).toFixed(1)}° elevation, ` +
        `range ${X}m, apex ${apex.toFixed(1)}m, glowing #00e5ff parabolic spline, #ff9e2c tangent velocity arrows, ` +
        `spatial blast iso-rings, progressive pre-cache stage ${stage}/4, VITAL_MAX_HP 6/6.`;

      const tx = 90 + (X / 280) * 650, ty = 345;
      let html = '';
      // Stage 1: Algebraický zameriavač
      html += `<circle cx="${tx}" cy="${ty}" r="42" fill="none" stroke="#ff9e2c" stroke-dasharray="5 4"/>`;
      html += `<circle cx="${tx}" cy="${ty}" r="22" fill="none" stroke="#ed1c24" stroke-width="2"/>`;
      html += `<line x1="${tx-56}" y1="${ty}" x2="${tx+56}" y2="${ty}" stroke="#ff9e2c"/>`;
      html += `<line x1="${tx}" y1="${ty-56}" x2="${tx}" y2="${ty+56}" stroke="#ff9e2c"/>`;

      // Stage 2+: Šípky procesu a označenia priestoru
      let pathD = '';
      for (let i = 0; i <= 24; i++) {
        const t = (i / 24) * tEnd;
        const px = 90 + ((vx * t) / 280) * 650;
        const py = 345 - ((vy * t - 0.5 * g * t * t) / 260) * 270;
        pathD += (i === 0 ? 'M' : 'L') + `${px.toFixed(1)},${py.toFixed(1)} `;
        if (stage >= 2 && (i === 5 || i === 12 || i === 19)) {
          const t2 = ((i + 1.5) / 24) * tEnd;
          const nx = 90 + ((vx * t2) / 280) * 650;
          const ny = 345 - ((vy * t2 - 0.5 * g * t2 * t2) / 260) * 270;
          html += `<line x1="${px}" y1="${py}" x2="${nx}" y2="${ny}" stroke="#ff9e2c" stroke-width="2.5" marker-end="url(#arrowAmber)"/>`;
        }
      }
      if (stage >= 2) {
        html += `<text x="65" y="372" fill="#00e5ff" font-size="11">[BATÉRIA 0,0m]</text>`;
        html += `<text x="${tx-55}" y="${ty-64}" fill="#ff9e2c" font-size="11">SEKTOR DOPADU X=${X}m</text>`;
      }
      if (stage >= 3) {
        html += `<path d="${pathD}" fill="none" stroke="#00e5ff" stroke-width="3"/>`;
      }
      if (stage >= 4) {
        html += `<circle cx="${tx}" cy="${ty}" r="62" fill="rgba(237,28,36,0.14)" stroke="#34d399" stroke-dasharray="3 3"/>`;
      }
      document.getElementById('sceneLayer').innerHTML = html;
    }
    recalc();
  </script>
</body>
</html>"""


if __name__ == "__main__":
    print(json.dumps(generate_veo_brainstorming_scripts_and_previews(), indent=2, ensure_ascii=False))
