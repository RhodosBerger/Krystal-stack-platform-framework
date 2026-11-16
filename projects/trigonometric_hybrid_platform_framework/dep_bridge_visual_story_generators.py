"""Dependency Bridge Module 3: Streaming Visual Storyline Generators (`obrazový dej`).

Combines Bridges 1 & 2 with priority-stratified telemetry to yield concrete,
frame-by-frame visual action sequences (hex siege ballistic arc, UMA shadow transition,
and robotics LiDAR conflict halt) and compiles them into an SVG storyboard strip + JSON.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, Iterator, List

from krystal_kernel.constants import UMA_SATURATION_THRESHOLD, VITAL_MAX_HP, ContractViolation

from .dep_bridge_numeric_and_packet_abi import emit_governed_128b_event, sanitize_vector_f32
from .dep_bridge_symbolic_geometry_and_openvino import build_bridged_geometry_and_rig_state
from .fact_synthesis_engine import REPORTS


def generate_hex_siege_visual_story(total_frames: int = 6) -> Iterator[Dict[str, Any]]:
    """Python generator yielding a 6-keyframe tactical visual storyline in the hex arena."""
    if VITAL_MAX_HP != 6 or not (2 <= total_frames <= 16):
        raise ContractViolation("Invalid invariant or frame count in visual generator", violation_code="ERR_VIS_GEN")

    beats = [
        ("VYKRESTENIE_VYSADKU", 0.31, 30.0, (-2, 1), (0, 0), 6, "Jednotka C01 obsadzuje hex (-2,1), lúč 30°"),
        ("GONIOMETRICKY_ZAMER", 0.36, 58.3, (-1, 1), (0, 0), 6, "Natočenie lúča na klin #0 (FLANK_INTERCEPT)"),
        ("BALISTICKY_OBLUK",    0.48, 58.3, (-1, 0), (0, 0), 6, "Mínometná krivka letu nad stredovým hexom"),
        ("NARAZ_A_STIT_AI",     0.62, 64.0, (-1, 0), (0, 0), 4, "Zásah za 2 HP -> AI dvíha HORIZON_SHIELD (HP 4/6)"),
        ("SPICKA_UMA_TIEN",     0.84, 120.0, (0, -1), (0, 0), 4, "UMA 0.84 >= 0.78 -> OP_01 drôtený režim (5/20 vrstiev)"),
        ("ZUZENIE_ZONY_POTVRD", 0.38, 150.0, (0, -1), (0, 0), 4, "Pokles UMA -> návrat 10/20 vrstiev + 128B SHA-256 pečiatka"),
    ]

    for idx in range(total_frames):
        code, uma, theta, unit_hex, target_hex, enemy_hp, caption = beats[idx % len(beats)]
        t_norm = idx / float(max(1, total_frames - 1))
        # Parabolic mortar ballistic height z(t) = 4 * h_max * t * (1 - t)
        arc_height = sanitize_vector_f32([4.0 * 2.4 * t_norm * (1.0 - t_norm)])[0]
        geo = build_bridged_geometry_and_rig_state(
            arena_radius=3,
            theta_deg=theta,
            hip_deg=18.0 * math.sin(idx * 1.1),
            knee_deg=-28.0 - 10.0 * math.cos(idx * 1.1),
        )
        pkt = emit_governed_128b_event(3, 16777217, 17104902, uma, f"F{idx}:{code}")
        shadow = uma >= UMA_SATURATION_THRESHOLD

        yield {
            "frame_index": idx,
            "beat_code": code,
            "caption_sk": caption,
            "uma_utilization": uma,
            "visual_style": "WIREFRAME_SHADOW_STATE_A" if shadow else "HOLOGRAM_FULL_STATE_B",
            "godot_layer_mask": "0x1f" if shadow else "0x3ff",
            "theta_deg": theta,
            "goniometric_heading_xy": geo["goniometric_heading_xy"],
            "unit_hex": list(unit_hex),
            "target_hex": list(target_hex),
            "mortar_arc_height_m": arc_height,
            "rig_foot_xy": geo["rig_joints_xy"]["foot"],
            "enemy_hp_of_6": enemy_hp,
            "packet_sha256": pkt["frame_sha256_hex"],
        }


def generate_robotics_conflict_visual_story(total_frames: int = 4) -> Iterator[Dict[str, Any]]:
    """Python generator yielding a 4-keyframe Sim-to-Real LiDAR conflict visual storyline."""
    if total_frames < 2:
        raise ContractViolation("total_frames must be >= 2", violation_code="ERR_ROB_VIS_FRAMES")
    stages = [
        ("PRIbLIZENIE_NOMINAL", 3.6, 3.4, 1.0, "Kamera (3,6 m) a LiDAR (3,4 m) v zhode -> plynulý pohyb 1,0 m/s"),
        ("ROZPOR_SENZOROV",     4.2, 0.8, 0.0, "Rozpor 3,4 m > 1,0 m -> zákaz priemeru, bifurkácia a STOP (0,0 m/s)"),
        ("CDT_FYZICKA_SONDA",   4.2, 0.8, 0.0, "Fyzická CDT sonda potvrdzuje prekážku na 0,8 m v tieni VBlank"),
        ("BEZPECNY_OBCHVAT",    1.9, 1.8, 0.2, "Kolaps superpozície na LiDAR -> opatrný obchvat 0,2 m/s (HP 6/6)"),
    ]
    for idx in range(total_frames):
        code, d_cam, d_lid, vel, caption = stages[idx % len(stages)]
        gap = round(abs(d_cam - d_lid), 2)
        yield {
            "frame_index": idx,
            "beat_code": code,
            "caption_sk": caption,
            "camera_m": d_cam,
            "lidar_m": d_lid,
            "sensor_gap_m": gap,
            "conflict_halt": gap > 1.0,
            "velocity_mps": vel,
            "vital_max_hp": VITAL_MAX_HP,
        }


def export_visual_storyline_artifacts(
    svg_path: Path = REPORTS / "visual_storyline_strip.svg",
    json_path: Path = REPORTS / "visual_storyline_frames.json",
) -> Dict[str, Any]:
    """Consume both generators, render a 6-panel crisp SVG storyboard strip, and write JSON."""
    siege_frames: List[Dict[str, Any]] = list(generate_hex_siege_visual_story(6))
    rob_frames: List[Dict[str, Any]] = list(generate_robotics_conflict_visual_story(4))

    w_panel = 190
    total_w = w_panel * len(siege_frames) + 20
    svg_parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {total_w} 195" width="{total_w}" height="195">',
        '<rect width="100%" height="100%" fill="#0b1020"/>',
    ]
    for i, fr in enumerate(siege_frames):
        x0 = 10 + i * w_panel
        stroke = "#f59e0b" if fr["visual_style"].startswith("WIREFRAME") else "#38bdf8"
        cx, cy = x0 + 90, 92
        hx, hy = fr["goniometric_heading_xy"]
        rx, ry = cx + hx * 42, cy - hy * 42
        arc_y = cy - fr["mortar_arc_height_m"] * 14
        svg_parts.extend(
            [
                f'<rect x="{x0}" y="10" width="180" height="175" rx="6" fill="#111827" stroke="{stroke}" stroke-width="1.5"/>',
                f'<text x="{x0 + 8}" y="27" fill="#e5e7eb" font-family="monospace" font-size="10">#{i + 1} {fr["beat_code"][:16]}</text>',
                f'<circle cx="{cx}" cy="{cy}" r="36" fill="none" stroke="#374151" stroke-width="1"/>',
                f'<line x1="{cx}" y1="{cy}" x2="{rx:.1f}" y2="{ry:.1f}" stroke="#22c55e" stroke-width="2"/>',
                f'<path d="M {cx - 34} {cy + 12} Q {cx} {arc_y:.1f} {cx + 28} {cy}" fill="none" stroke="{stroke}" stroke-dasharray="3,2" stroke-width="1.8"/>',
                f'<circle cx="{cx + 28}" cy="{cy}" r="6" fill="#ef4444"/>',
                f'<text x="{x0 + 8}" y="156" fill="#9ca3af" font-family="monospace" font-size="9.5">UMA={fr["uma_utilization"]:.2f} | {fr["godot_layer_mask"]}</text>',
                f'<text x="{x0 + 8}" y="172" fill="#38bdf8" font-family="monospace" font-size="9.5">HP={fr["enemy_hp_of_6"]}/6 | z={fr["mortar_arc_height_m"]:.2f}m</text>',
            ]
        )
    svg_parts.append("</svg>")

    svg_path.parent.mkdir(parents=True, exist_ok=True)
    svg_path.write_text("\n".join(svg_parts) + "\n", encoding="utf-8")

    payload = {
        "bridge_id": "DEP_BRIDGE_03_VISUAL_STORY_GENERATORS",
        "vital_max_hp": VITAL_MAX_HP,
        "svg_path": str(svg_path),
        "hex_siege_frames": siege_frames,
        "robotics_conflict_frames": rob_frames,
    }
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return payload
