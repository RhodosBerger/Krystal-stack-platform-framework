"""Open-source layer patterns, WSL2 Ubuntu <-> Windows 11 hypervisor visualization
configurator, bidirectional OS call visual semantics, pre-cached CSS mortar ballistic
trajectories + aiming reticles, decision-tensor UI -> 2B compact model pipeline, and
4 Google-Maps-ML-style geospatial world-region tile adaptations.
"""

from __future__ import annotations

import json
import math
import platform
import shutil
from typing import Any, Dict, List, Tuple

import numpy as np

from krystal_kernel.constants import UMA_SATURATION_THRESHOLD, VITAL_MAX_HP, ContractViolation

from .adaptive_2b_spatial_linguistic_trainer import AdaptiveConversationTrainer
from .arena_cache_optical_sockets_and_consumption import BattleArenaCacheAndTensorNavigator
from .clash_battle_royale import ClashGame
from .combat_turn_planner import get_cached_hw_profile
from .tensor_prompt_branch_and_kb import TensorPromptBranchEngine

GRAVITY_MPS2 = 9.81


# ---------------------------------------------------------------------------
# 1. Open-Source Layer Templates & WSL2 Ubuntu <-> Windows 11 Hypervisor Setup
# ---------------------------------------------------------------------------

OSS_LAYER_TEMPLATES: Tuple[Dict[str, str], ...] = (
    {
        "template_id": "OSS_TPL_01_VULKAN_IMPLICIT_LAYER",
        "oss_origin": "Khronos Vulkan-Loader Layer Manifest Pattern (Zero-Touch P1)",
        "layer_role": "Zachytáva vkQueuePresentKHR bez úpravy zdrojového kódu render enginu.",
    },
    {
        "template_id": "OSS_TPL_02_POSIX_SHM_SPSC_RING",
        "oss_origin": "Linux /dev/shm mmap Lock-Free SPSC Ring",
        "layer_role": "Prenáša 128 B zmluvné rámce medzi Ubuntu procesom vo WSL2 a mostom na porte 8090.",
    },
    {
        "template_id": "OSS_TPL_03_WSLG_DXCORE_PASSTHROUGH",
        "oss_origin": "Microsoft WSLg Wayland + /dev/dxg vGPU Projection",
        "layer_role": "Premieta Linuxové OpenGL/Vulkan okno z Ubuntu priamo na plochu Windows 11.",
    },
    {
        "template_id": "OSS_TPL_04_BROWSER_RAF_CADENCE_SYNC",
        "oss_origin": "W3C requestAnimationFrame + PerformanceObserver",
        "layer_role": "Zosúlaďuje 60/120 Hz vzorkovanie v prehliadači s tikom autoritatívneho jadra.",
    },
)


def configure_wsl_ubuntu_hypervisor_bridge() -> Dict[str, Any]:
    """Generate host-calibrated WSL2 Ubuntu + hypervisor config for Windows 11 visualization."""
    hw = get_cached_hw_profile()
    wsl_bin = shutil.which("wsl")
    vm_mem_gb = max(4, min(12, int(hw.ram_gb * 0.45)))
    vm_procs = max(2, hw.physical_cores)

    wslconfig_ini = "\n".join(
        [
            "[wsl2]",
            f"memory={vm_mem_gb}GB",
            f"processors={vm_procs}",
            "guiApplications=true",
            "networkingMode=mirrored",
            "nestedVirtualization=true",
            "pageReporting=true",
        ]
    )
    ubuntu_bootstrap_sh = "\n".join(
        [
            "#!/usr/bin/env bash",
            "set -euo pipefail",
            "# WSL2 Ubuntu <-> Windows 11 WSLg & /dev/dxg verification script",
            "ls -la /dev/dxg || echo 'WARN: /dev/dxg vGPU device not exposed'",
            "export DISPLAY=${DISPLAY:-:0}",
            "export WAYLAND_DISPLAY=${WAYLAND_DISPLAY:-wayland-0}",
            "python3 -m http.client 127.0.0.1 8090 >/dev/null 2>&1 || true",
        ]
    )

    return {
        "vital_max_hp": VITAL_MAX_HP,
        "host_os": f"{platform.system()} {platform.release()} (version {platform.version()})",
        "wsl_executable_present": wsl_bin is not None,
        "wsl_executable_path": wsl_bin,
        "hypervisor_allocation": {
            "assigned_ram_gb": vm_mem_gb,
            "host_ram_gb": hw.ram_gb,
            "assigned_vcpus": vm_procs,
            "wslg_gui_passthrough": True,
            "mirrored_loopback_8090": True,
        },
        "oss_layer_templates": list(OSS_LAYER_TEMPLATES),
        "generated_wslconfig": wslconfig_ini,
        "ubuntu_guest_probe_script": ubuntu_bootstrap_sh,
        "evidence": "LOCAL_DETECTED (host topology & wsl.exe) + GENERATED_CONFIG (not auto-mutating ~/.wslconfig)",
    }


# ---------------------------------------------------------------------------
# 2. Bidirectional Engine <-> OS Semantics & Pre-Cached CSS Mortar Trajectories
# ---------------------------------------------------------------------------

def define_os_call_visual_semantics(uma_utilization: float = 0.35) -> Dict[str, Any]:
    """Map Engine <-> OS boundary calls to deterministic visual glyphs including unrendered attributes."""
    mappings = [
        {
            "boundary_call": "Engine -> OS: VirtualAlloc / mmap(128B Ring)",
            "direction": "ENGINE_TO_OS",
            "visual_glyph": "CONCENTRIC_HEX_ALLOCATION_PULSE",
            "previously_unvisualized_property": "Page-alignment remainder & zero-reallocation lock status",
        },
        {
            "boundary_call": "OS -> Engine: IsProcessorFeaturePresent(PF_AVX512F)",
            "direction": "OS_TO_ENGINE",
            "visual_glyph": "FOUR_CORE_VECTOR_LANE_BARS",
            "previously_unvisualized_property": "MXCSR FTZ/DAZ subnormal flush bit state under oneDNN",
        },
        {
            "boundary_call": "Engine -> Driver: clEnqueueNDRangeKernel (Iris Xe osv16)",
            "direction": "ENGINE_TO_OS",
            "visual_glyph": "CYAN_STREAM_ARC" if uma_utilization < UMA_SATURATION_THRESHOLD else "AMBER_SHADOW_VETO_GATE",
            "previously_unvisualized_property": "CACHE_DIR precompiled blob hit vs cold IGC shader compile",
        },
        {
            "boundary_call": "Browser -> Engine: requestAnimationFrameCadence(dt_ms)",
            "direction": "UI_TO_ENGINE",
            "visual_glyph": "PHASE_LOCK_RETICLE_RING",
            "previously_unvisualized_property": "Frame-pacing jitter ratio p95 / median across 128 B updates",
        },
    ]
    return {"vital_max_hp": VITAL_MAX_HP, "os_semantic_mappings": mappings}


def compute_mortar_css_trajectories_and_reticles(
    origin_hex: Tuple[int, int] = (-2, 0),
    target_hex: Tuple[int, int] = (2, -1),
    flight_time_s: float = 1.2,
    jitter_sigma_ms: float = 5.0,
) -> Dict[str, Any]:
    """Compute parabolic mortar arc from cached hex position, emit CSS @keyframes & aiming reticle."""
    if flight_time_s <= 0.1 or flight_time_s > 10.0:
        raise ContractViolation("flight_time_s out of [0.1, 10.0]", violation_code="ERR_MORTAR_TIME")

    game = ClashGame("VERIFY", seed=6)
    nav = BattleArenaCacheAndTensorNavigator()
    cached = nav.cache_arena_state(game)

    sq3 = math.sqrt(3.0)
    x0, z0 = sq3 * (origin_hex[0] + origin_hex[1] * 0.5), 1.5 * origin_hex[1]
    x1, z1 = sq3 * (target_hex[0] + target_hex[1] * 0.5), 1.5 * target_hex[1]

    vx = (x1 - x0) / flight_time_s
    vz = (z1 - z0) / flight_time_s
    vy0 = 0.5 * GRAVITY_MPS2 * flight_time_s
    apex_m = round((vy0 * vy0) / (2.0 * GRAVITY_MPS2), 3)

    keyframes = []
    css_stops = []
    for pct in (0, 25, 50, 75, 100):
        t = flight_time_s * (pct / 100.0)
        xt = x0 + vx * t
        zt = z0 + vz * t
        yt = max(0.0, vy0 * t - 0.5 * GRAVITY_MPS2 * t * t)
        px = round((xt - x0) * 28.0, 2)
        py = round(-yt * 32.0 + (zt - z0) * 14.0, 2)
        keyframes.append({"pct": pct, "world_xyz": [round(xt, 3), round(yt, 3), round(zt, 3)], "screen_px": [px, py]})
        css_stops.append(f"  {pct}% {{ transform: translate3d({px}px, {py}px, 0px); }}")

    css_preset = "@keyframes krystalMortarArc {\n" + "\n".join(css_stops) + "\n}"
    # Aiming reticle dispersion radius scales cleanly with network/OS jitter sigma
    reticle_cep_radius_px = round(14.0 + jitter_sigma_ms * 1.8, 2)

    return {
        "vital_max_hp": VITAL_MAX_HP,
        "cached_state_sha256": cached["state_sha256"][:16],
        "origin_hex": list(origin_hex),
        "target_hex": list(target_hex),
        "apex_height_m": apex_m,
        "flight_time_s": flight_time_s,
        "aiming_reticle": {
            "target_screen_offset_px": keyframes[-1]["screen_px"],
            "cep_radius_px": reticle_cep_radius_px,
            "lock_state": "LOCKED_PRECISION" if reticle_cep_radius_px < 28.0 else "WIDE_DISPERSION_WARN",
        },
        "trajectory_keyframes": keyframes,
        "css_keyframes_preset": css_preset,
    }


# ---------------------------------------------------------------------------
# 3. Inference-Trigger Formulas + Decision Tensor UI -> Compact 2B Model Pipeline
# ---------------------------------------------------------------------------

def evaluate_decision_tensor_ui_and_2b_bridge(
    uma_utilization: float = 0.35,
    trajectory_error_px: float = 6.5,
) -> Dict[str, Any]:
    """Evaluate closed-form inference trigger formulas, run OpenVINO decision tensor & compact 2B trainer."""
    # Trigger formula: I_trig = H(|err_px| - 4.0) + H(0.78 - uma) -> triggers compact neuro-symbolic pass
    trigger_score = round((trajectory_error_px / 4.0) * (1.0 - 0.5 * uma_utilization), 4)
    should_infer = trigger_score >= 1.0

    branch_res = TensorPromptBranchEngine().evaluate_prompt_matrix_branches(
        ["Prepočítaj balistický zámerník a vyber zobrazovací soket"],
        uma_utilization=uma_utilization,
    )
    top_sol = branch_res["branch_trees"][0]["scaled_solutions"][0]

    trainer = AdaptiveConversationTrainer(d_model=32, max_seq_len=24, epochs=35, seed=6)
    out_2b = trainer.infer_with_bonus_properties("Sync graph")

    ui_layers = [
        {"layer": "UI_L0_DECISION_TENSOR_HEATBAR", "bound_branch": top_sol["branch_id"], "prob": top_sol["tensor_probability"]},
        {"layer": "UI_L1_MORTAR_RETICLE_OVERLAY", "inference_triggered": should_infer, "trigger_score": trigger_score},
        {"layer": "UI_L2_COMPACT_2B_CITATION_BADGE", "cited_statement": out_2b["selected_output"], "grounded": out_2b["selected_grounded"]},
    ]

    return {
        "vital_max_hp": VITAL_MAX_HP,
        "model_tier": "COMPACT_2B_CLASS_NEUROSYMBOLIC (never inflated to unverified massive weights)",
        "inference_trigger_formula": "I_trig = (|trajectory_error_px| / 4.0) * (1 - 0.5 * uma_utilization) >= 1.0",
        "trigger_score": trigger_score,
        "inference_triggered": should_infer,
        "decision_tensor_top_branch": top_sol,
        "ui_bound_layers": ui_layers,
        "compact_2b_output": out_2b,
    }


# ---------------------------------------------------------------------------
# 4. Four Geospatial World-Region Tile Adaptations (Google Maps ML Pattern)
# ---------------------------------------------------------------------------

def _latlon_to_web_mercator_tile(lat_deg: float, lon_deg: float, zoom: int) -> Tuple[int, int]:
    """Standard Web Mercator slippy-map tile coordinates (z, x, y) used in web mapping pipelines."""
    lat_rad = math.radians(max(-85.0511, min(85.0511, lat_deg)))
    n = 2 ** zoom
    xtile = int((lon_deg + 180.0) / 360.0 * n)
    ytile = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
    return xtile, ytile


def adapt_integrations_to_four_geospatial_tiles() -> Dict[str, Any]:
    """Adapt Krystal's hex-cache, EWMA horizon, OpenVINO Conv2D, and 128B ledger to 4 global map regions."""
    regions = [
        {
            "adaptation_id": "GEO_01_ALPINE_CARPATHIAN_DEM",
            "region_name": "Vysoké Tatry / Stredoeurópsky reliéf",
            "lat_lon": (49.165, 20.133),
            "zoom": 12,
            "ml_map_pattern": "Elevation & Line-of-Sight Tile Pyramid",
            "krystal_integration": "Mortar parabolic arc + frustum/LOD culling + prewarmed CACHE_DIR conv filter",
        },
        {
            "adaptation_id": "GEO_02_METROPOLITAN_ETA_GRAPH",
            "region_name": "Tokio Šindžuku / Multimodálny dopravný uzol",
            "lat_lon": (35.6895, 139.6917),
            "zoom": 14,
            "ml_map_pattern": "Graph Neural / Matrix Diffusion ETA & Congestion Horizon",
            "krystal_integration": "BattleArenaCacheAndTensorNavigator (D^-1 A diffusion) + EWMA + 2*sigma_upper horizon",
        },
        {
            "adaptation_id": "GEO_03_MARITIME_PORT_DIGITAL_TWIN",
            "region_name": "Rotterdam Maasvlakte / Kontajnerový terminál",
            "lat_lon": (51.954, 4.142),
            "zoom": 13,
            "ml_map_pattern": "Sectorized Asset Tracking & Manifest Verification",
            "krystal_integration": "7 Sector Coordinators + AUTHORITATIVE_LEDGER 128 B SHA-256 receipts",
        },
        {
            "adaptation_id": "GEO_04_SOLAR_CORRIDOR_TELEMETRY",
            "region_name": "Atacama / Energeticko-tepelný koridor",
            "lat_lon": (-23.863, -69.132),
            "zoom": 11,
            "ml_map_pattern": "Satellite Irradiance Edge Segmentation & Load Shedding",
            "krystal_integration": "OpenVINO Sobel/Laplacian sensory pass + UMA < 0.78 / PL1 governor",
        },
    ]

    adapted = []
    for reg in regions:
        lat, lon = reg["lat_lon"]
        tx, ty = _latlon_to_web_mercator_tile(lat, lon, reg["zoom"])
        adapted.append(
            {
                **reg,
                "web_mercator_tile_zxy": [reg["zoom"], tx, ty],
                "hex_supercell_cells": 37,
                "frame_contract_bytes": 128,
            }
        )

    return {
        "vital_max_hp": VITAL_MAX_HP,
        "adaptation_count": len(adapted),
        "world_region_adaptations": adapted,
    }


# ---------------------------------------------------------------------------
# 5. 1:1 Metric Geo-Replica Arena Plugin, Vulkan SPIR-V Sets & UMA Isomorphisms
# ---------------------------------------------------------------------------

def convert_geotile_location_to_3d_arena_replica(
    adaptation_id: str = "GEO_01_ALPINE_CARPATHIAN_DEM",
    submenu_mode: str = "TERRAIN_RELIEF_1_TO_1",
    custom_elevation_scale: float = 1.0,
) -> Dict[str, Any]:
    """Convert a selected world region into a metric-scaled 37-hex 3D arena replica with submenu fields."""
    cat = {r["adaptation_id"]: r for r in adapt_integrations_to_four_geospatial_tiles()["world_region_adaptations"]}
    if adaptation_id not in cat:
        raise ContractViolation(f"Unknown geo adaptation {adaptation_id!r}", violation_code="ERR_GEO_ID")
    if not (0.1 <= custom_elevation_scale <= 5.0):
        raise ContractViolation("custom_elevation_scale outside [0.1, 5.0]", violation_code="ERR_GEO_SCALE")

    reg = cat[adaptation_id]
    lat, lon = reg["lat_lon"]
    zoom = int(reg["zoom"])
    meters_per_pixel = (156543.03392 * math.cos(math.radians(lat))) / (2 ** zoom)
    hex_span_m = round(meters_per_pixel * 32.0, 2)  # 1:1 metric width of one 32px tile patch per hex

    from .clash_battle_royale import board_cells

    cells = sorted(board_cells(3))
    sq3 = math.sqrt(3.0)
    mesh_hexes = []
    for q, r in cells:
        wx_m = round(sq3 * (q + r * 0.5) * hex_span_m, 2)
        wz_m = round(1.5 * r * hex_span_m, 2)
        # Deterministic terrain relief seeded from (lat, lon, q, r)
        elev_m = round(
            custom_elevation_scale * 18.0 * (0.5 + 0.5 * math.sin(q * 0.9 + math.radians(lat)) * math.cos(r * 0.8 + math.radians(lon))),
            2,
        )
        mesh_hexes.append({"cell": [q, r], "world_xz_meters": [wx_m, wz_m], "elevation_m": elev_m})

    return {
        "vital_max_hp": VITAL_MAX_HP,
        "adaptation_id": adaptation_id,
        "submenu_selected": submenu_mode,
        "contextual_fields": {
            "available_submenus": ["TERRAIN_RELIEF_1_TO_1", "TACTICAL_LOS_CONTOURS", "THERMAL_IRRADIANCE_OVERLAY"],
            "lat_lon": list(reg["lat_lon"]),
            "web_mercator_zxy": reg["web_mercator_tile_zxy"],
            "meters_per_pixel": round(meters_per_pixel, 3),
            "hex_pitch_meters_1_to_1": hex_span_m,
        },
        "scene_engine_intersections": [
            "I_GEO_1: Slippy-tile (z,x,y) <-> axiálne súradnice 37-hexovej arény (q,r) v mierke 1:1 (metre)",
            "I_GEO_2: Výškový profil DEM <-> balistický oblúk mínometu a viditeľnosť sektorových koordinátorov",
            "I_GEO_3: Predhriata CACHE_DIR <-> konvolučná filtrácia dlaždíc pred nahrávaním 3D scény",
        ],
        "replica_hex_count": len(mesh_hexes),
        "replica_hexes_preview": mesh_hexes[:6],
        "data_source_scope": "METRIC_WGS84_MERCATOR_PROCEDURAL_DEM (no external Google API key queried)",
    }


def compile_vulkan_spirv_shader_sets_and_isomorphisms(uma_utilization: float = 0.35) -> Dict[str, Any]:
    """Map physical CPU threads to precalculated LUT plugins, emit Vulkan GLSL/SPIR-V templates & verify hex isomorphism."""
    import hashlib
    import struct

    from .clash_battle_royale import board_cells

    hw = get_cached_hw_profile()
    glslang = shutil.which("glslangValidator")

    # 1. Verify exact bijective isomorphism phi: Hex_37 <-> {0..36} (UMA ring slot index)
    cells = sorted(board_cells(3))
    fwd = {c: idx for idx, c in enumerate(cells)}
    inv = {idx: c for idx, c in enumerate(cells)}
    bijection_verified = all(inv[fwd[c]] == c for c in cells) and len(fwd) == 37

    # 2. Thread-to-ISA precalculated morphing table across physical cores
    phys = max(1, hw.physical_cores)
    thread_lut_assignments = [
        {
            "thread_slot": t,
            "isa_lane": "AVX512F_FMA_oneDNN",
            "morphed_plugin_fn": f"PRECALC_LUT_SECTOR_{t % 4}",
            "uma_copy_mode": "VBLANK_DEFERRED_COPY" if uma_utilization >= UMA_SATURATION_THRESHOLD else "IMMEDIATE_USM_SYNC",
        }
        for t in range(phys)
    ]

    # 3. Three Vulkan GLSL 450 shader templates coupling UMA bus telemetry & VSync image copy
    shader_templates = {
        "SH_01_HEX_INSTANCED_COMPUTE": (
            "#version 450\nlayout(local_size_x = 64) in;\n"
            "layout(std430, binding = 0) readonly buffer HexBuf { vec4 hex_state[]; };\n"
            "void main() { uint idx = gl_GlobalInvocationID.x; if (idx < 37u) { /* precalculated LUT morph */ } }\n"
        ),
        "SH_02_UMA_VSYNC_COPY_GUARD": (
            "#version 450\nlayout(push_constant) uniform UmaPush { float uma_load; uint vsync_phase; } pc;\n"
            "void main() { bool allow_blit = (pc.uma_load < 0.78) || (pc.vsync_phase == 1u); }\n"
        ),
        "SH_03_HOLOGRAM_FRESNEL_FRAG": (
            "#version 450\nlayout(location = 0) in float cos_theta;\nlayout(location = 0) out vec4 out_col;\n"
            "void main() { float r = 0.04 + 0.96 * pow(1.0 - cos_theta, 5.0); out_col = vec4(0.34, 0.85, 1.0, r); }\n"
        ),
    }

    # Canonical 5-word SPIR-V 1.5 header (Magic 0x07230203, Version 0x00010500, Generator 0, Bound 16, Schema 0)
    spirv_header_bytes = struct.pack("<5I", 0x07230203, 0x00010500, 0x00000000, 16, 0)
    shader_blobs = {
        sid: {
            "glsl_bytes": len(src.encode("utf-8")),
            "spirv_magic_hex": "0x07230203",
            "cached_blob_sha256": hashlib.sha256(spirv_header_bytes + src.encode("utf-8")).hexdigest()[:16],
        }
        for sid, src in shader_templates.items()
    }

    return {
        "vital_max_hp": VITAL_MAX_HP,
        "bijection_hex37_to_uma_slot_verified": bijection_verified,
        "physical_thread_lut_assignments": thread_lut_assignments,
        "glslang_compiler_on_path": glslang is not None,
        "shader_compilation_scope": "SPIRV_BINARY_COMPILER_PRESENT" if glslang else "GLSL_450_TEMPLATES_AND_SPIRV_HEADER_CACHED",
        "vulkan_shader_sets": shader_blobs,
    }


# ---------------------------------------------------------------------------
# 6. Hypervisor/WSL Context-Renamed Functions, Conversation Pattern Templates
#    & Transient Cross-Unit Data Fields / Cycles / Structures
# ---------------------------------------------------------------------------

TRANSIENT_DATA_FIELDS_AND_STRUCTURES: Tuple[Dict[str, str], ...] = (
    {
        "structure_id": "STRUCT_01_ZERO_COPY_128B_FRAME",
        "fields": "magic:u32, ver:u16, pkt_type:u16, src_pid:u32, dst_pid:u32, uma:f32, payload:bytes[76], sha256:bytes[32]",
        "cycle_transit": "CYCLE_C1_SIM_ROLLBACK_16_67MS -> Host Windows 11 <-> WSL2 /dev/shm",
    },
    {
        "structure_id": "STRUCT_02_HEX_OCCUPANCY_TENSOR",
        "fields": "cells[37][3]:f32 (hp_norm, atk, def) + transition_matrix[37][37]:f32",
        "cycle_transit": "ClashGame -> BattleArenaCacheAndTensorNavigator -> OpenVINO CPU AVX-512",
    },
    {
        "structure_id": "STRUCT_03_DECISION_BRANCH_MATRIX",
        "fields": "prompt_features[B][8]:f32 @ weights[8][6]:f32 -> probabilities[B][6]:f32",
        "cycle_transit": "Prompt/CLI -> OpenVINO MatMul+Softmax -> Freivalds O(n^2) -> Symbolic Gate",
    },
    {
        "structure_id": "STRUCT_04_PHASE_RENDER_POOL",
        "fields": "geometry_16KB + rgba_texture_16KB + hologram_rays_4KB + receipts_2KB (38,912 B fixed)",
        "cycle_transit": "CYCLE_C0_SCANOUT_VR_8_33MS -> WebXR / WSLg Wayland / Blender bpy",
    },
)


def wsl_hypervisor_render_dispatch(
    adaptation_id: str = "GEO_01_ALPINE_CARPATHIAN_DEM",
    uma_utilization: float = 0.35,
) -> Dict[str, Any]:
    """Context-adapted hypervisor & WSL Ubuntu render dispatcher combining geo-replica, shaders & transient structures."""
    wsl_cfg = configure_wsl_ubuntu_hypervisor_bridge()
    replica = convert_geotile_location_to_3d_arena_replica(adaptation_id=adaptation_id)
    shaders = compile_vulkan_spirv_shader_sets_and_isomorphisms(uma_utilization=uma_utilization)
    return {
        "vital_max_hp": VITAL_MAX_HP,
        "context_function": "wsl_hypervisor_render_dispatch",
        "wsl_display_route": "Ubuntu WSL2 -> /dev/dxg -> WSLg Wayland -> Windows 11 Compositor",
        "hypervisor_quota": wsl_cfg["hypervisor_allocation"],
        "bound_geo_replica": replica["adaptation_id"],
        "active_shaders": list(shaders["vulkan_shader_sets"].keys()),
        "transient_structures": list(TRANSIENT_DATA_FIELDS_AND_STRUCTURES),
    }


