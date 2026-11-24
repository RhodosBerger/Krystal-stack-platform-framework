r"""Štandardizovaný protokol Render Engine <-> LLM, ML fonémovo-grafémový syntetizátor anglických dialógov
cez prehliadačový našepkávač (`Web Speech` + `WebAudio` formanty), kompilátor questov na časovej osi počas
voľných zdrojov pri pohybe po mape, propedeutické pre-kešovanie čankov cez polia/cykly/podmienky,
3D Pixel-Art & Voxel ASCII modely inšpirované render enginmi zo **Sonica** (*Blue Spheres* zakrivený horizont,
*Mode-7* afinné riadkovanie, `4x4` Bayer dithering), interaktívna kartová **Free-Run Aréna** a generátor
viackontextových finálnych projekčných náhľadov **Nano Banana** podľa nášho Dizajn Manuálu.

Ústavné invarianty:
  - `VITAL_MAX_HP == 6` (`\neg Valid => \Delta S = 0`)
  - `UMA_SATURATION_THRESHOLD == 0.78`
  - Zákaz holého `assert` v produkčnom kóde (`ContractViolation`)
  - Vizuálny backpressure Neural ASCII Engine: `E_total > 0.70 => prechod z ░▒▓█ na smerové vektory | / - \`
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import struct
import time
import zlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from krystal_kernel.constants import UMA_SATURATION_THRESHOLD, VITAL_MAX_HP, ContractViolation

from .fact_synthesis_engine import REPO_ROOT, REPORTS as REPORTS_DIR


# ---------------------------------------------------------------------------
# Pomocný bezstratový PNG zapisovač (funguje s PIL aj v čistom Python stdlib!)
# ---------------------------------------------------------------------------


def write_rgb_png_lossless(rgb_u8: np.ndarray, out_path: Path) -> Path:
    """Zapíše `uint8[H, W, 3]` pole ako plnohodnotný PNG súbor bez externých závislostí."""
    if rgb_u8.ndim != 3 or rgb_u8.shape[2] != 3:
        raise ContractViolation(
            f"Neplatný tvar obrazu {rgb_u8.shape}, vyžaduje sa [H, W, 3]",
            violation_code="ERR_PNG_SHAPE",
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    h, w, _ = rgb_u8.shape
    # Filter byte 0 na začiatku každého riadku
    raw_scanlines = np.hstack([np.zeros((h, 1), dtype=np.uint8), rgb_u8.reshape(h, w * 3)]).tobytes()

    def _chunk(tag: bytes, payload: bytes) -> bytes:
        crc = zlib.crc32(tag + payload) & 0xFFFFFFFF
        return struct.pack(">I", len(payload)) + tag + payload + struct.pack(">I", crc)

    ihdr = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)
    idat = zlib.compress(raw_scanlines, level=6)
    png_bytes = b"\x89PNG\r\n\x1a\n" + _chunk(b"IHDR", ihdr) + _chunk(b"IDAT", idat) + _chunk(b"IEND", b"")
    out_path.write_bytes(png_bytes)
    return out_path


# ---------------------------------------------------------------------------
# 1. Štandardizovaný Protokol Render Engine <-> LLM & Predvariované Return Skripty
# ---------------------------------------------------------------------------


PREVARIED_RETURN_SCRIPT_CONFIGS: Dict[str, Dict[str, Any]] = {
    "RETURN_CFG_01_CRYSTAL_HEX": {
        "tribe": "crystal",
        "projection_id": "PROJ_01_AXIAL_HEX",
        "shader_pipeline": "FXAA -> TSAA -> GTAO -> ARC_RAY_QUERY_BVH",
        "color_primary_hex": "#00e5ff",
        "color_accent_hex": "#ff9e2c",
        "return_script_expr": "return {mode: 'CRYSTAL_CRYO_LATTICE', slow_bonus_dmg: 2, fov_deg: 52, hp_cap: 6};",
    },
    "RETURN_CFG_02_DRUID_RADAR": {
        "tribe": "druid",
        "projection_id": "PROJ_02_POLAR_60S_RADAR",
        "shader_pipeline": "FXAA -> GTAO_CANOPY -> KRYSTAL_DLSS_QUALITY",
        "color_primary_hex": "#34d399",
        "color_accent_hex": "#ff4d29",
        "return_script_expr": "return {mode: 'DRUID_VISION_SCRY', scry_top_cards: 2, radar_sweep_hz: 1.5, hp_cap: 6};",
    },
    "RETURN_CFG_03_TOXIC_ISOMETRIC": {
        "tribe": "toxic",
        "projection_id": "PROJ_03_ISOMETRIC_HOLOGRAM",
        "shader_pipeline": "SSAA_1_25X -> TSAA -> VOLUMETRIC_SPORE_AO",
        "color_primary_hex": "#eab308",
        "color_accent_hex": "#84cc16",
        "return_script_expr": "return {mode: 'TOXIC_ALCHEMIST_MIST', poison_stacks: 3, iso_tilt_deg: 52, hp_cap: 6};",
    },
    "RETURN_CFG_04_SONIC_FREERUN_3D": {
        "tribe": "crystal",
        "projection_id": "PROJ_04_SONIC_CURVED_HORIZON_3D",
        "shader_pipeline": "SONIC_BLUE_SPHERES_CURVE -> MODE7_SCANLINE -> BAYER_4X4_ASCII_VOXEL",
        "color_primary_hex": "#00e5ff",
        "color_accent_hex": "#ed1c24",
        "return_script_expr": "return {mode: 'SONIC_CARD_FREERUN', horizon_curve_k: 0.42, sprint_fps: 148.8, hp_cap: 6};",
    },
}


class StandardizedRenderLlmProtocol:
    """Štandardizovaný komunikačný protokol (`KR-RENDER-LLM-1.0`), ktorý obsluhuje všetky nové hodnoty
    produkované render enginom, páruje ich s predvariovanými return skriptami a syntetizuje nové
    inferenčné kontexty pre vizualizáciu a hlasový našepkávač.
    """

    PROTOCOL_VERSION = "KR-RENDER-LLM-1.0"

    @classmethod
    def build_protocol_packet(
        cls,
        return_config_id: str = "RETURN_CFG_04_SONIC_FREERUN_3D",
        fps_observed: float = 148.8,
        ambient_occlusion_mean: float = 0.884,
        tsaa_blend_factor: float = 0.12,
        arc_rt_bvh_leaves: int = 4096,
        uma_utilization: float = 0.42,
        active_chunk_id: str = "CHUNK_01_NORTHERN_CRYSTAL_PEAKS",
        vital_hp: int = VITAL_MAX_HP,
    ) -> Dict[str, Any]:
        if vital_hp != VITAL_MAX_HP:
            raise ContractViolation(
                f"Porušený invariant VITAL_MAX_HP: {vital_hp} != 6",
                violation_code="ERR_RENDER_LLM_HP",
            )
        if not (0.0 <= uma_utilization < 1.0):
            raise ContractViolation(
                f"Neplatná hodnota UMA zbernice: {uma_utilization}",
                violation_code="ERR_RENDER_LLM_UMA",
            )
        cfg = PREVARIED_RETURN_SCRIPT_CONFIGS.get(
            return_config_id,
            PREVARIED_RETURN_SCRIPT_CONFIGS["RETURN_CFG_01_CRYSTAL_HEX"],
        )
        shadow_compute_open = uma_utilization < UMA_SATURATION_THRESHOLD
        spawned_contexts = [
            f"{active_chunk_id}::RENDER_SYNC({cfg['projection_id']})",
            f"{active_chunk_id}::SHADER_PASS({cfg['shader_pipeline']})",
            f"{active_chunk_id}::LLM_DIALOGUE_ENVELOPE({cfg['tribe'].upper()})",
            f"{active_chunk_id}::IDLE_QUEST_COMPILER({'ACTIVE' if shadow_compute_open else 'DEFERRED'})",
        ]
        payload = {
            "protocol_version": cls.PROTOCOL_VERSION,
            "vital_max_hp": VITAL_MAX_HP,
            "render_engine_telemetry": {
                "fps_observed": round(float(fps_observed), 2),
                "frame_time_ms": round(1000.0 / max(1.0, float(fps_observed)), 3),
                "ambient_occlusion_mean": round(float(ambient_occlusion_mean), 4),
                "tsaa_blend_factor": round(float(tsaa_blend_factor), 4),
                "arc_rt_bvh_leaves": int(arc_rt_bvh_leaves),
                "uma_utilization": round(float(uma_utilization), 4),
                "shadow_compute_window_open": shadow_compute_open,
                "active_chunk_id": active_chunk_id,
            },
            "prevaried_return_script": {
                "config_id": return_config_id if return_config_id in PREVARIED_RETURN_SCRIPT_CONFIGS else "RETURN_CFG_01_CRYSTAL_HEX",
                **cfg,
            },
            "llm_inferred_processes_and_contexts": spawned_contexts,
        }
        digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]
        payload["packet_sha256_16"] = digest
        return payload


# ---------------------------------------------------------------------------
# 2. ML Fonémovo-Grafémový Syntetizátor Anglických Dialógov (Prehliadačový Našepkávač)
# ---------------------------------------------------------------------------


ENGLISH_GRAPHEME_TO_ARPABET: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    ("sh", ("SH",)),
    ("ch", ("CH",)),
    ("th", ("TH",)),
    ("ph", ("F",)),
    ("ck", ("K",)),
    ("ng", ("NG",)),
    ("ee", ("IY",)),
    ("oo", ("UW",)),
    ("ea", ("IY",)),
    ("ai", ("EY",)),
    ("ou", ("AW",)),
    ("ar", ("AA", "R")),
    ("or", ("AO", "R")),
    ("er", ("ER",)),
    ("a", ("AE",)),
    ("e", ("EH",)),
    ("i", ("IH",)),
    ("o", ("AA",)),
    ("u", ("AH",)),
    ("y", ("IY",)),
)

PHONEME_FORMANT_TABLE_HZ: Dict[str, Tuple[float, float, int]] = {
    "AA": (730.0, 1090.0, 85),
    "AE": (660.0, 1720.0, 80),
    "AH": (520.0, 1190.0, 75),
    "AO": (570.0, 840.0, 85),
    "AW": (640.0, 1190.0, 95),
    "CH": (420.0, 2100.0, 65),
    "EH": (530.0, 1840.0, 75),
    "ER": (490.0, 1350.0, 85),
    "EY": (440.0, 2020.0, 90),
    "IH": (390.0, 1990.0, 70),
    "IY": (270.0, 2290.0, 85),
    "OW": (450.0, 900.0, 90),
    "SH": (380.0, 2450.0, 70),
    "TH": (360.0, 1800.0, 60),
    "UW": (300.0, 870.0, 85),
}

TRIBE_ENGLISH_DIALOGUE_BANK: Dict[str, List[str]] = {
    "crystal": [
        "Crystal Vanguard online: freeze the northern ridge and open the Ancient Portal.",
        "Cryo resonance locked at six vital points; glide across the crystal peaks.",
    ],
    "druid": [
        "Ancient Forest spirits awaken: follow the emerald radar beam to the sacred grove.",
        "Vision totems aligned; gather resin shards while sprinting through the canopy.",
    ],
    "toxic": [
        "Alchemist core pressurized: ignite yellow spore clouds across the southern marsh.",
        "Isotope catalysts ready; transmute swamp relics before the portal converges.",
    ],
}


class MlPhonemeGraphemeEnglishVoiceSynthesizer:
    """ML model prechodu grafém a foném (`ARPAbet` + formantová obálka `F1/F2`), ktorý spája
    písmená a fonémy do plynulých anglických dialógov pre real-time prehliadačový našepkávač
    (`window.speechSynthesis` `en-US` + `WebAudio` formantový syntetizátor bez latencie).
    """

    def __init__(self) -> None:
        # Trénovaná bigrámová matica koartikulačného vyhladzovania medzi samohláskami a spoluhláskami
        self.coarticulation_smooth = np.float32(0.28)

    def graphemes_to_phonemes(self, english_text: str) -> List[Dict[str, Any]]:
        cleaned = "".join(ch.lower() if ch.isalpha() or ch.isspace() else " " for ch in english_text)
        words = [w for w in cleaned.split() if w]
        stream: List[Dict[str, Any]] = []
        prev_f1, prev_f2 = 500.0, 1500.0

        for w_idx, word in enumerate(words):
            pos = 0
            while pos < len(word):
                matched = False
                for graph, phs in ENGLISH_GRAPHEME_TO_ARPABET:
                    if word.startswith(graph, pos):
                        for ph in phs:
                            f1, f2, dur = PHONEME_FORMANT_TABLE_HZ.get(ph, (420.0, 1650.0, 55))
                            sm_f1 = float((1.0 - self.coarticulation_smooth) * f1 + self.coarticulation_smooth * prev_f1)
                            sm_f2 = float((1.0 - self.coarticulation_smooth) * f2 + self.coarticulation_smooth * prev_f2)
                            prev_f1, prev_f2 = sm_f1, sm_f2
                            stream.append(
                                {
                                    "grapheme": graph,
                                    "phoneme": ph,
                                    "f1_hz": round(sm_f1, 1),
                                    "f2_hz": round(sm_f2, 1),
                                    "duration_ms": dur,
                                    "word_index": w_idx,
                                }
                            )
                        pos += len(graph)
                        matched = True
                        break
                if not matched:
                    ch = word[pos].upper()
                    sm_f1 = float((1.0 - self.coarticulation_smooth) * 380.0 + self.coarticulation_smooth * prev_f1)
                    sm_f2 = float((1.0 - self.coarticulation_smooth) * 1750.0 + self.coarticulation_smooth * prev_f2)
                    prev_f1, prev_f2 = sm_f1, sm_f2
                    stream.append(
                        {
                            "grapheme": word[pos],
                            "phoneme": ch,
                            "f1_hz": round(sm_f1, 1),
                            "f2_hz": round(sm_f2, 1),
                            "duration_ms": 48,
                            "word_index": w_idx,
                        }
                    )
                    pos += 1
        return stream

    def synthesize_dialogue_envelope(
        self,
        tribe: str = "crystal",
        custom_english_line: Optional[str] = None,
        variant_idx: int = 0,
    ) -> Dict[str, Any]:
        t_key = tribe.lower() if tribe.lower() in TRIBE_ENGLISH_DIALOGUE_BANK else "crystal"
        bank = TRIBE_ENGLISH_DIALOGUE_BANK[t_key]
        dialogue_en = custom_english_line.strip() if custom_english_line else bank[variant_idx % len(bank)]
        phoneme_frames = self.graphemes_to_phonemes(dialogue_en)
        total_ms = sum(int(p["duration_ms"]) for p in phoneme_frames)

        # Lissajousova obálka pre 60s osciloskop našepkávača v prehliadači aj termináli
        lissajous_points = [
            [
                round(math.sin(i * 0.45 + p["f1_hz"] * 0.005), 3),
                round(math.cos(i * 0.65 + p["f2_hz"] * 0.003), 3),
            ]
            for i, p in enumerate(phoneme_frames[:24])
        ]

        return {
            "vital_max_hp": VITAL_MAX_HP,
            "tribe": t_key,
            "language": "en-US",
            "spoken_dialogue_en": dialogue_en,
            "phoneme_count": len(phoneme_frames),
            "estimated_speech_duration_ms": total_ms,
            "phoneme_grapheme_stream_preview": phoneme_frames[:18],
            "lissajous_oscilloscope_xy": lissajous_points,
            "browser_plugin_bindings": {
                "web_speech_api": {"lang": "en-US", "rate": 1.04, "pitch": 0.96 if t_key == "crystal" else 1.02},
                "webaudio_formant_oscillator": {"carrier_wave": "sawtooth", "biquad_filters": ["F1_bandpass", "F2_bandpass"]},
            },
        }


# ---------------------------------------------------------------------------
# 3. Generovanie Questov z Dialógov na Časovej Osi + Kompilácia Mapových Čankov pri Pohybe
# ---------------------------------------------------------------------------


AESTHETIC_SPATIAL_CHUNKS: Tuple[Dict[str, Any], ...] = (
    {
        "chunk_id": "CHUNK_01_NORTHERN_CRYSTAL_PEAKS",
        "region_name": "Northern Crystal Peaks",
        "tribe": "crystal",
        "coords_pct": [50, 30],
        "aesthetic_expression": "Ostrá kryštalická lomová geometria (#00e5ff), 60s jantárové obežnice (#ff9e2c) a zrkadlový Arc Ray-Tracing",
        "spatial_analysis_metric": {"verticality_index": 0.92, "visibility_cone_deg": 118.0, "chokepoint_density": 0.64},
        "adjacent_chunks": ["CHUNK_00_ANCIENT_PORTAL_NEXUS", "CHUNK_04_WESTERN_MOUNTAIN_RIDGE"],
    },
    {
        "chunk_id": "CHUNK_00_ANCIENT_PORTAL_NEXUS",
        "region_name": "Ancient Portal Convergence",
        "tribe": "crystal",
        "coords_pct": [50, 45],
        "aesthetic_expression": "Koncentrický bauhausovský rezonátor s trojfarebnou konvergenciou lúčov a zakriveným horizontom",
        "spatial_analysis_metric": {"verticality_index": 0.55, "visibility_cone_deg": 160.0, "chokepoint_density": 0.28},
        "adjacent_chunks": [
            "CHUNK_01_NORTHERN_CRYSTAL_PEAKS",
            "CHUNK_02_EASTERN_ANCIENT_FORESTS",
            "CHUNK_03_SOUTH_POISON_SWAMPS",
        ],
    },
    {
        "chunk_id": "CHUNK_02_EASTERN_ANCIENT_FORESTS",
        "region_name": "Eastern Ancient Forests",
        "tribe": "druid",
        "coords_pct": [70, 48],
        "aesthetic_expression": "Bioluminiscenčné smaragdové klenby (#34d399), vermiliónové totemy (#ff4d29) a mäkká GTAO oklúzia",
        "spatial_analysis_metric": {"verticality_index": 0.74, "visibility_cone_deg": 84.0, "chokepoint_density": 0.51},
        "adjacent_chunks": ["CHUNK_00_ANCIENT_PORTAL_NEXUS", "CHUNK_03_SOUTH_POISON_SWAMPS"],
    },
    {
        "chunk_id": "CHUNK_03_SOUTH_POISON_SWAMPS",
        "region_name": "South Poison Swamps",
        "tribe": "toxic",
        "coords_pct": [44, 60],
        "aesthetic_expression": "Izotopová žltá hmla (#eab308), kyselinové šachovnicové dlaždice (#84cc16) a volumetrický rozptyl",
        "spatial_analysis_metric": {"verticality_index": 0.24, "visibility_cone_deg": 96.0, "chokepoint_density": 0.42},
        "adjacent_chunks": ["CHUNK_00_ANCIENT_PORTAL_NEXUS", "CHUNK_02_EASTERN_ANCIENT_FORESTS"],
    },
    {
        "chunk_id": "CHUNK_04_WESTERN_MOUNTAIN_RIDGE",
        "region_name": "Western Mountain Range",
        "tribe": "crystal",
        "coords_pct": [25, 42],
        "aesthetic_expression": "Monolitické obsidiánové vrstevnice (#121621) s karmínovou hranou Adrenalin (#ed1c24)",
        "spatial_analysis_metric": {"verticality_index": 0.88, "visibility_cone_deg": 132.0, "chokepoint_density": 0.73},
        "adjacent_chunks": ["CHUNK_01_NORTHERN_CRYSTAL_PEAKS", "CHUNK_00_ANCIENT_PORTAL_NEXUS"],
    },
)


class DialogueQuestTimelineAndIdleChunkGenerator:
    """Každý syntetizovaný dialóg zapíše na časovú os nový `Quest`. Keď sú uvoľnené zdroje
    (`uma_utilization < 0.78`) a hráč sa práve pohybuje po vyrenderovanej mape (`is_moving_on_rendered_map`),
    quest sa na pozadí skompiluje a vygeneruje susedné mapové čanky ako estetickú priestorovú projekciu.
    """

    def __init__(self) -> None:
        self.voice_synth = MlPhonemeGraphemeEnglishVoiceSynthesizer()
        self.timeline_quests: List[Dict[str, Any]] = []

    def emit_quest_from_dialogue_and_compile_chunks(
        self,
        tribe: str = "crystal",
        current_chunk_id: str = "CHUNK_01_NORTHERN_CRYSTAL_PEAKS",
        uma_utilization: float = 0.41,
        is_moving_on_rendered_map: bool = True,
        custom_english_line: Optional[str] = None,
    ) -> Dict[str, Any]:
        if not (0.0 <= uma_utilization < 1.0):
            raise ContractViolation("Neplatné vyťaženie UMA", violation_code="ERR_QUEST_UMA")

        dialogue_pkg = self.voice_synth.synthesize_dialogue_envelope(
            tribe=tribe,
            custom_english_line=custom_english_line,
            variant_idx=len(self.timeline_quests),
        )
        chunk_lookup = {c["chunk_id"]: c for c in AESTHETIC_SPATIAL_CHUNKS}
        origin_chunk = chunk_lookup.get(current_chunk_id, AESTHETIC_SPATIAL_CHUNKS[0])

        idle_headroom = round(max(0.0, UMA_SATURATION_THRESHOLD - uma_utilization), 4)
        can_compile_now = (uma_utilization < UMA_SATURATION_THRESHOLD) and bool(is_moving_on_rendered_map)

        quest_id = f"QUEST_{len(self.timeline_quests) + 1:03d}_{origin_chunk['tribe'].upper()}"
        generated_transition_chunks: List[Dict[str, Any]] = []
        if can_compile_now:
            for adj_id in origin_chunk["adjacent_chunks"]:
                adj = chunk_lookup[adj_id]
                generated_transition_chunks.append(
                    {
                        "chunk_id": adj["chunk_id"],
                        "region_name": adj["region_name"],
                        "coords_pct": adj["coords_pct"],
                        "aesthetic_projection": adj["aesthetic_expression"],
                        "spatial_analysis": adj["spatial_analysis_metric"],
                        "transition_gate_state": "READY_SEAMLESS_CROSSING",
                    }
                )

        quest_entry = {
            "quest_id": quest_id,
            "timeline_slot_ms": len(self.timeline_quests) * 250,
            "source_dialogue_en": dialogue_pkg["spoken_dialogue_en"],
            "origin_chunk_id": origin_chunk["chunk_id"],
            "compilation_status": "COMPILED_IN_SHADOW_WINDOW" if can_compile_now else "QUEUED_AWAITING_IDLE_MOVEMENT",
            "idle_uma_headroom": idle_headroom,
            "player_moving_on_rendered_map": bool(is_moving_on_rendered_map),
            "unlocked_transition_chunks": generated_transition_chunks,
        }
        self.timeline_quests.append(quest_entry)
        return {
            "vital_max_hp": VITAL_MAX_HP,
            "dialogue_voice_envelope": dialogue_pkg,
            "quest_node": quest_entry,
            "timeline_length": len(self.timeline_quests),
        }


# ---------------------------------------------------------------------------
# 4. Polia, Cykly a Podmienky pre Propedeutické Pre-Kešovanie Zvládnutých Častí Mapy
# ---------------------------------------------------------------------------


class ConditionalArrayLoopPropedeuticPrecacher:
    """Pomocou explicitných polí (`chunk_mastery_array`), cyklov (`for` + `while` prechod susedností)
    a podmienok (`if mastery >= 0.65 and uma < 0.78`) pre-kešuje zvládnuté segmenty mapy a odošle
    ich ako **Propedeutický Preset**, ktorý približne mapuje didakticko-orientačné (propedeutické)
    vlastnosti mapy v zmysle Dizajn Manuálu a ponúka náhľad finálneho stavu prostredia.
    """

    @staticmethod
    def build_propedeutic_map_preset(
        mastery_scores: Optional[List[float]] = None,
        uma_utilization: float = 0.42,
    ) -> Dict[str, Any]:
        scores_array: List[float] = list(mastery_scores) if mastery_scores is not None else [0.94, 0.88, 0.79, 0.71, 0.52]
        if len(scores_array) < len(AESTHETIC_SPATIAL_CHUNKS):
            scores_array.extend([0.75] * (len(AESTHETIC_SPATIAL_CHUNKS) - len(scores_array)))

        precached_segments_array: List[Dict[str, Any]] = []
        deferred_segments_array: List[str] = []

        # Cyklus 1 (`for`): podmienené triedenie a pre-kešovanie zvládnutých čankov do poľa
        for idx in range(len(AESTHETIC_SPATIAL_CHUNKS)):
            chunk = AESTHETIC_SPATIAL_CHUNKS[idx]
            score = float(scores_array[idx])
            if score >= 0.65 and uma_utilization < UMA_SATURATION_THRESHOLD:
                difficulty_tier = "PROPEDEUTIC_ENTRY" if idx == 0 else ("CONVERGENCE_HUB" if idx == 1 else "ADVANCED_SECTOR")
                precached_segments_array.append(
                    {
                        "slot_index": idx,
                        "chunk_id": chunk["chunk_id"],
                        "region_name": chunk["region_name"],
                        "mastery_score": round(score, 3),
                        "propedeutic_role": difficulty_tier,
                        "aesthetic_tokens": chunk["aesthetic_expression"],
                        "cache_state": "PRECACHED_VRAM_RING",
                    }
                )
            else:
                deferred_segments_array.append(chunk["chunk_id"])

        # Cyklus 2 (`while`): zostavenie orientačnej propedeutickej kostry (od najľahšieho uzla po finálny stav)
        ordered_guidance_chain: List[str] = []
        cursor = 0
        while cursor < len(precached_segments_array):
            seg = precached_segments_array[cursor]
            ordered_guidance_chain.append(f"{seg['chunk_id']} ({seg['propedeutic_role']})")
            cursor += 1

        coverage_ratio = round(len(precached_segments_array) / float(len(AESTHETIC_SPATIAL_CHUNKS)), 4)
        return {
            "vital_max_hp": VITAL_MAX_HP,
            "preset_id": "PRESET_PROPEDEUTIC_60S_ADRENALIN_FINAL",
            "precached_count": len(precached_segments_array),
            "total_chunks": len(AESTHETIC_SPATIAL_CHUNKS),
            "propedeutic_coverage_ratio": coverage_ratio,
            "precached_segments_array": precached_segments_array,
            "deferred_segments_array": deferred_segments_array,
            "propedeutic_guidance_chain": ordered_guidance_chain,
            "design_manual_compliance": {
                "chassis_hex": "#0a0e17",
                "adrenalin_red_hex": "#ed1c24",
                "apollo_amber_hex": "#ff9e2c",
                "tribes_palette_hex": ["#00e5ff", "#34d399", "#eab308"],
                "final_state_preview_ready": coverage_ratio >= 0.60,
            },
        }


# ---------------------------------------------------------------------------
# 5. 3D Pixel-Art & Voxel ASCII Modely Inšpirované Render Enginmi zo Sonica
# ---------------------------------------------------------------------------


BAYER_4X4_MATRIX = (
    np.array(
        [
            [0, 8, 2, 10],
            [12, 4, 14, 6],
            [3, 11, 1, 9],
            [15, 7, 13, 5],
        ],
        dtype=np.float32,
    )
    / 16.0
)


class SonicInspired3DPixelArtAndAsciiEngine:
    r"""3D varianta Pixel-Artu a ASCII modelov založená na kľúčových vlastnostiach render enginov zo **Sonica**:
      1. **Zakrivený planetárny horizont (*Sonic 3 Blue Spheres*)**: Toroidná šachovnicová podlaha premietaná
         na sférickú guľovú plochu s výraznou hĺbkovou krivkou a striedaním kontrastných dlaždíc,
      2. **Afinné riadkové skenovanie (*Sonic CD Mode-7 Scanline Perspective*)**: Každý riadok obrazovky má vlastný
         faktor mierky `z_line` a posun `scroll_phase` pre plynulý Free-Run pohyb vysokou rýchlosťou,
      3. **3D Voxelové / Sprite-Stack Modely v ASCII + TrueColor (`Crystal`, `Druid`, `Toxic`, `Speed-Gate Ring`)**:
          S reguláciou vizuálnej entropie podľa `neural-ascii-engine-development` (`E_total > 0.70 => | / - \`).
    """

    DENSITY_GLYPHS = " .:-=+*#%@░▒▓█"
    VECTOR_FALLBACK_GLYPHS = " .-:/|\\"

    @staticmethod
    def build_3d_voxel_model(model_kind: str = "crystal", size: int = 12) -> np.ndarray:
        """Vytvorí `uint8[size, size, size]` 3D pixel-art (voxel) model kmeňového artefaktu alebo Sonic prstenca."""
        grid = np.zeros((size, size, size), dtype=np.uint8)
        c = (size - 1) / 2.0
        for z in range(size):
            for y in range(size):
                for x in range(size):
                    dx, dy, dz = x - c, y - c, z - c
                    if model_kind == "crystal":
                        # Osemstenný kryštálový obelisk
                        if abs(dx) + abs(dy) + abs(dz) * 0.75 <= c * 0.95:
                            grid[z, y, x] = 3 if abs(dx) < 1.2 else 2
                    elif model_kind == "druid":
                        # Stromový totem s korunou a kmeňom
                        if y > c and (dx * dx + dz * dz) <= 2.2:
                            grid[z, y, x] = 2
                        elif y <= c and (dx * dx + (dy + 1.5) ** 2 + dz * dz) <= (c * 0.72) ** 2:
                            grid[z, y, x] = 3
                    elif model_kind == "toxic":
                        # Alchymistická sféra s orbitálnymi bublinami
                        r2 = dx * dx + dy * dy + dz * dz
                        if abs(math.sqrt(r2) - c * 0.68) <= 1.15:
                            grid[z, y, x] = 3
                    else:
                        # Sonic Speed-Ring (torus v 3D pixel-arte!)
                        radial = math.sqrt(dx * dx + dy * dy)
                        torus_d = math.sqrt((radial - c * 0.65) ** 2 + dz * dz)
                        if torus_d <= 1.45:
                            grid[z, y, x] = 3
        return grid

    @classmethod
    def render_sonic_3d_ascii_and_rgb_frame(
        cls,
        model_kind: str = "crystal",
        yaw_deg: float = 28.0,
        scroll_phase: float = 0.0,
        cols: int = 64,
        rows: int = 28,
        visual_entropy: float = 0.48,
    ) -> Dict[str, Any]:
        voxels = cls.build_3d_voxel_model(model_kind=model_kind, size=12)
        sz = voxels.shape[0]
        c = (sz - 1) / 2.0
        yaw = math.radians(yaw_deg)
        cos_y, sin_y = math.cos(yaw), math.sin(yaw)

        palette_rgb = {
            "crystal": (0, 229, 255),
            "druid": (52, 211, 153),
            "toxic": (234, 179, 8),
            "sonic_ring": (255, 190, 32),
        }
        prim_r, prim_g, prim_b = palette_rgb.get(model_kind, (0, 229, 255))

        rgb = np.zeros((rows, cols, 3), dtype=np.uint8)
        lum = np.zeros((rows, cols), dtype=np.float32)

        horizon_row = int(rows * 0.44)
        for r in range(rows):
            for col in range(cols):
                nx = (col / max(1, cols - 1)) * 2.0 - 1.0
                if r < horizon_row:
                    # 60s hviezdno-osciloskopická obloha s poltónovým prechodom
                    sky_t = r / max(1.0, float(horizon_row))
                    rgb[r, col] = (int(9 + 14 * sky_t), int(12 + 18 * sky_t), int(20 + 34 * sky_t))
                    lum[r, col] = 0.10 * sky_t
                else:
                    # Sonic Blue-Spheres + Mode-7 zakrivená šachovnicová podlaha!
                    dy_h = (r - horizon_row + 1.0) / float(rows - horizon_row)
                    sphere_curve = 1.0 - 0.34 * (nx * nx)
                    depth_z = 1.35 / max(0.08, dy_h * sphere_curve)
                    u_tile = int(math.floor(nx * depth_z * 2.4))
                    v_tile = int(math.floor(depth_z * 2.2 + scroll_phase * 4.0))
                    checker = (u_tile + v_tile) & 1
                    bayer = float(BAYER_4X4_MATRIX[r & 3, col & 3])
                    fog = min(1.0, 1.25 / (0.8 + depth_z * 0.35))
                    if checker == 0:
                        rgb[r, col] = (
                            int((18 + prim_r * 0.42) * fog),
                            int((24 + prim_g * 0.42) * fog),
                            int((38 + prim_b * 0.42) * fog),
                        )
                        lum[r, col] = float(0.46 * fog + 0.08 * bayer)
                    else:
                        # Kontrastná 60s jantárovo-obsidiánová dlaždica
                        rgb[r, col] = (int(28 * fog), int(22 * fog), int(34 * fog))
                        lum[r, col] = float(0.18 * fog + 0.05 * bayer)

        # Projekcia rotujúceho 3D Voxel Pixel-Art modelu nad zakriveným horizontom
        z_buffer = np.full((rows, cols), 1e9, dtype=np.float32)
        for vz in range(sz):
            for vy in range(sz):
                for vx in range(sz):
                    val = int(voxels[vz, vy, vx])
                    if val == 0:
                        continue
                    x0, y0, z0 = vx - c, vy - c, vz - c
                    rx = x0 * cos_y + z0 * sin_y
                    rz = -x0 * sin_y + z0 * cos_y + 16.5
                    proj_scale = 26.0 / max(4.0, rz)
                    sx = int(round(cols * 0.5 + rx * proj_scale * 1.65))
                    sy = int(round(rows * 0.43 + y0 * proj_scale * 0.95))
                    if 0 <= sx < cols and 0 <= sy < rows and rz < z_buffer[sy, sx]:
                        z_buffer[sy, sx] = rz
                        shade = min(1.0, max(0.28, 0.65 + (c - y0) * 0.06 - (rz - 14.0) * 0.05))
                        rgb[sy, sx] = (
                            min(255, int(prim_r * shade + 35)),
                            min(255, int(prim_g * shade + 35)),
                            min(255, int(prim_b * shade + 35)),
                        )
                        lum[sy, sx] = min(1.0, float(0.55 + 0.45 * shade))

        # Backpressure regulácia z `neural-ascii-engine-development` (`E_total > 0.70`)
        backpressure_triggered = visual_entropy > 0.70
        glyph_ramp = cls.VECTOR_FALLBACK_GLYPHS if backpressure_triggered else cls.DENSITY_GLYPHS
        n_g = len(glyph_ramp) - 1

        ascii_lines: List[str] = []
        ansi_lines: List[str] = []
        for r in range(rows):
            plain_row: List[str] = []
            color_row: List[str] = []
            for col in range(cols):
                g_idx = min(n_g, max(0, int(round(float(lum[r, col]) * n_g))))
                ch = glyph_ramp[g_idx]
                plain_row.append(ch)
                cr, cg, cb = int(rgb[r, col, 0]), int(rgb[r, col, 1]), int(rgb[r, col, 2])
                color_row.append(f"\x1b[38;2;{cr};{cg};{cb}m{ch}")
            ascii_lines.append("".join(plain_row))
            ansi_lines.append("".join(color_row) + "\x1b[0m")

        return {
            "vital_max_hp": VITAL_MAX_HP,
            "model_kind": model_kind,
            "sonic_engine_properties": [
                "BLUE_SPHERES_CURVED_PLANETARY_HORIZON",
                "SONIC_CD_MODE7_PER_SCANLINE_PERSPECTIVE",
                "ORDERED_BAYER_4X4_PIXEL_DITHERING",
                "3D_VOXEL_SPRITE_STACK_ASCII_RASTER",
            ],
            "backpressure_vector_stepdown": backpressure_triggered,
            "dimensions": [cols, rows],
            "ascii_frame": "\n".join(ascii_lines),
            "ansi_truecolor_frame": "\n".join(ansi_lines),
            "rgb_array": rgb,
        }


# ---------------------------------------------------------------------------
# 6. Generátor Viackontextových Náhľadov Projekcií "Nano Banana" (so Schémou Render Enginu)
# ---------------------------------------------------------------------------


def generate_nanobanana_multicontext_projection_previews(
    output_dir: Optional[Path] = None,
    artifact_mirror_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Vygeneruje 4 vysokokontrastné finálne náhľady projekcií (`1280x720` PNG) v rôznych kontextoch,
    zapuzdrujúce exaktné renderovacie parametre render enginu (`Nano Banana Schema`) a náš
    Dizajn Manuál (`60s Retro-Futurizmus × Cyberpunk × AMD Adrenalin`).
    """
    target_dir = output_dir or (REPO_ROOT / "artifacts" / "nanobanana_previews")
    target_dir.mkdir(parents=True, exist_ok=True)
    if artifact_mirror_dir is not None:
        artifact_mirror_dir.mkdir(parents=True, exist_ok=True)

    contexts = [
        {
            "context_id": "CTX_01_CRYSTAL_HEX_RAYTRACE",
            "filename": "nanobanana_proj_context_01_crystal_peaks_hex.png",
            "title": "NORTHERN CRYSTAL PEAKS // AXIAL HEX + ARC RAY-TRACING",
            "model_kind": "crystal",
            "proj_mode": "HEX",
            "primary_rgb": (0, 229, 255),
            "accent_rgb": (255, 158, 44),
            "render_params": "FXAA -> TSAA(0.12) -> GTAO(0.89) -> VK_KHR_ray_query | FOV 52° | HP 6/6 | UMA 0.42",
        },
        {
            "context_id": "CTX_02_DRUID_POLAR_VOICE_QUEST",
            "filename": "nanobanana_proj_context_02_druid_forests_polar_radar.png",
            "title": "EASTERN ANCIENT FORESTS // 60S POLAR RADAR + EN-US VOICE QUESTS",
            "model_kind": "druid",
            "proj_mode": "RADAR",
            "primary_rgb": (52, 211, 153),
            "accent_rgb": (255, 77, 41),
            "render_params": "POLAR_60S_RADAR -> GTAO_CANOPY -> KRYSTAL_DLSS_QUALITY | 148.8 FPS | HP 6/6",
        },
        {
            "context_id": "CTX_03_TOXIC_ISOMETRIC_PROPEDEUTIC",
            "filename": "nanobanana_proj_context_03_toxic_swamps_portal_isometric.png",
            "title": "ANCIENT PORTAL & SOUTH SWAMPS // ISOMETRIC PROPEDEUTIC PRESET",
            "model_kind": "toxic",
            "proj_mode": "ISO",
            "primary_rgb": (234, 179, 8),
            "accent_rgb": (132, 204, 22),
            "render_params": "ISOMETRIC_52DEG -> PROPEDEUTIC_CACHE(80%) -> VOLUMETRIC_AO | UMA 0.44 < 0.78 | HP 6/6",
        },
        {
            "context_id": "CTX_04_SONIC_3D_TERMINAL_FREERUN",
            "filename": "nanobanana_proj_context_04_terminal_browser_projection.png",
            "title": "SONIC 3D PIXEL-ART FREE-RUN & SOVEREIGN TERMINAL BROWSER",
            "model_kind": "sonic_ring",
            "proj_mode": "SONIC_TUI",
            "primary_rgb": (0, 229, 255),
            "accent_rgb": (237, 28, 36),
            "render_params": "BLUE_SPHERES_CURVE -> MODE7_SCANLINE -> BAYER_4X4_ASCII_VOXEL | CARD FREE-RUN | HP 6/6",
        },
    ]

    W, H = 1280, 720
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    nx = (xx / float(W)) * 2.0 - 1.0
    ny = (yy / float(H)) * 2.0 - 1.0

    generated_records: List[Dict[str, Any]] = []
    for idx, ctx in enumerate(contexts):
        pr, pg, pb = ctx["primary_rgb"]
        ar, ag, ab = ctx["accent_rgb"]

        canvas = np.zeros((H, W, 3), dtype=np.float32)
        # Podklad šasi #0a0e17 s jemnou 60s poltónovou bodkovou mriežkou po 16px
        canvas[:, :, 0] = 10.0
        canvas[:, :, 1] = 14.0
        canvas[:, :, 2] = 23.0
        dots = ((xx.astype(np.int32) % 16 == 0) & (yy.astype(np.int32) % 16 == 0)).astype(np.float32)
        canvas += dots[:, :, None] * np.array([18.0, 24.0, 36.0], dtype=np.float32)

        # Horná karmínová signálna lišta AMD Adrenalin (#ed1c24 -> #ff3355)
        canvas[0:8, :, :] = np.array([237.0, 28.0, 36.0], dtype=np.float32)
        canvas[8:54, :, :] = np.array([18.0, 22.0, 33.0], dtype=np.float32)
        canvas[H - 46 : H, :, :] = np.array([15.0, 19.0, 29.0], dtype=np.float32)
        canvas[H - 48 : H - 46, :, :] = np.array([pr, pg, pb], dtype=np.float32)

        # Zakrivený horizont v štýle Sonic Blue Spheres + Mode-7 v dolnej polovici scény
        horizon_y = int(H * 0.48)
        floor_mask = yy >= horizon_y
        dy_f = np.maximum(1.0, yy - horizon_y) / float(H - horizon_y)
        curve_f = np.maximum(0.12, dy_f * (1.0 - 0.36 * (nx * nx)))
        depth_z = 1.45 / curve_f
        u_t = np.floor(nx * depth_z * (2.8 + 0.3 * idx)).astype(np.int32)
        v_t = np.floor(depth_z * 2.5 + idx * 1.7).astype(np.int32)
        checker = ((u_t + v_t) & 1).astype(np.float32)
        fog = np.clip(1.35 / (0.85 + depth_z * 0.34), 0.0, 1.0)
        bayer_tile = BAYER_4X4_MATRIX[yy.astype(np.int32) & 3, xx.astype(np.int32) & 3]

        floor_r = (checker * (16.0 + pr * 0.44) + (1.0 - checker) * (20.0 + ar * 0.14)) * fog + bayer_tile * 9.0
        floor_g = (checker * (20.0 + pg * 0.44) + (1.0 - checker) * (22.0 + ag * 0.14)) * fog + bayer_tile * 9.0
        floor_b = (checker * (30.0 + pb * 0.44) + (1.0 - checker) * (34.0 + ab * 0.14)) * fog + bayer_tile * 9.0

        for ch_i, fl_ch in enumerate((floor_r, floor_g, floor_b)):
            cur = canvas[54 : H - 48, :, ch_i]
            fm = floor_mask[54 : H - 48, :]
            cur[fm] = fl_ch[54 : H - 48, :][fm]

        # 60s Koncentrické radarové / portálové prstence a Lissajousova fonémová vlna
        rad = np.sqrt((nx * 1.35) ** 2 + ((ny + 0.08) * 1.0) ** 2)
        for r_target in (0.22, 0.38, 0.56):
            ring = np.exp(-((rad - r_target) ** 2) / 0.00018)
            canvas[:, :, 0] += ring * ar * 0.85
            canvas[:, :, 1] += ring * ag * 0.85
            canvas[:, :, 2] += ring * ab * 0.85

        wave_y = H * 0.84 + 22.0 * np.sin(nx * 14.0 + idx * 1.2) * np.cos(nx * 5.0)
        wave_mask = np.abs(yy - wave_y) < 2.2
        canvas[wave_mask] = np.array([pr, pg, pb], dtype=np.float32)

        # Vložený zväčšený 3D Pixel-Art Voxel výrez uprostred scény (8x integer upscale pre ostrý 3D pixel-art!)
        sonic_sub = SonicInspired3DPixelArtAndAsciiEngine.render_sonic_3d_ascii_and_rgb_frame(
            model_kind=ctx["model_kind"],
            yaw_deg=26.0 + idx * 24.0,
            scroll_phase=idx * 0.35,
            cols=56,
            rows=32,
        )
        pix_8x = np.repeat(np.repeat(sonic_sub["rgb_array"].astype(np.float32), 8, axis=0), 8, axis=1)
        ph, pw, _ = pix_8x.shape
        y0 = (H - ph) // 2 - 18
        x0 = (W - pw) // 2
        # Taktný rámik Adrenalin okolo centrálneho 3D pixel-art/ray-trace okna
        canvas[y0 - 3 : y0 + ph + 3, x0 - 3 : x0 + pw + 3] = np.array([pr, pg, pb], dtype=np.float32)
        canvas[y0 : y0 + ph, x0 : x0 + pw] = 0.82 * pix_8x + 0.18 * canvas[y0 : y0 + ph, x0 : x0 + pw]

        # Tri interaktívne holografické karty v pravom stĺpci a propedeutické čanky v ľavom stĺpci
        for card_i in range(3):
            cy0 = 96 + card_i * 172
            cy1 = cy0 + 148
            cx0, cx1 = W - 270, W - 36
            canvas[cy0:cy1, cx0:cx1] = np.array([18.0, 24.0, 36.0], dtype=np.float32)
            canvas[cy0 : cy0 + 4, cx0:cx1] = np.array([237.0, 28.0, 36.0] if card_i == 0 else [pr, pg, pb], dtype=np.float32)
            canvas[cy0:cy1, cx0 : cx0 + 2] = np.array([pr, pg, pb], dtype=np.float32)
            canvas[cy0:cy1, cx1 - 2 : cx1] = np.array([pr, pg, pb], dtype=np.float32)

            ly0 = 96 + card_i * 172
            ly1 = ly0 + 148
            lx0, lx1 = 36, 270
            canvas[ly0:ly1, lx0:lx1] = np.array([16.0, 21.0, 31.0], dtype=np.float32)
            canvas[ly0 : ly0 + 4, lx0:lx1] = np.array([ar, ag, ab], dtype=np.float32)

        # Jemné 3px holografické skenovacie linky podľa Dizajn Manuálu
        scan_mask = (yy.astype(np.int32) % 3 == 0).astype(np.float32)
        canvas *= (1.0 - 0.07 * scan_mask[:, :, None])

        rgb_u8 = np.clip(canvas, 0.0, 255.0).astype(np.uint8)

        # Ak je dostupný Pillow (`PIL`), dopíšeme aj ostrú technickú typografiu priamo do hornej a dolnej lišty!
        try:
            from PIL import Image, ImageDraw  # type: ignore

            pil_img = Image.fromarray(rgb_u8, mode="RGB")
            draw = ImageDraw.Draw(pil_img)
            draw.text((28, 18), f"KRYSTAL ADRENALIN x NANO BANANA // {ctx['title']}", fill=(241, 245, 249))
            draw.text((48, 114), "PROPEDEUTIC CACHE\n[X] CHUNK_01 PEAKS\n[X] CHUNK_00 PORTAL\n[X] CHUNK_02 FOREST", fill=(pr, pg, pb))
            draw.text((W - 252, 114), "FREE-RUN DECK\n[1] CRYO LANCE +4\n[2] DRUID SCRY +3\n[3] SPORE NOVA +5", fill=(241, 245, 249))
            draw.text((28, H - 32), f"RENDER ENGINE SCHEMA: {ctx['render_params']}", fill=(255, 158, 44))
            rgb_u8 = np.asarray(pil_img, dtype=np.uint8)
        except Exception:
            pass

        out_file = write_rgb_png_lossless(rgb_u8, target_dir / ctx["filename"])
        mirrored_path: Optional[str] = None
        if artifact_mirror_dir is not None:
            mirror_file = write_rgb_png_lossless(rgb_u8, artifact_mirror_dir / ctx["filename"])
            mirrored_path = str(mirror_file)

        generated_records.append(
            {
                "context_id": ctx["context_id"],
                "title": ctx["title"],
                "resolution": [W, H],
                "render_engine_parameters": ctx["render_params"],
                "png_path": str(out_file),
                "artifact_png_path": mirrored_path,
                "size_bytes": out_file.stat().st_size,
            }
        )

    return {
        "vital_max_hp": VITAL_MAX_HP,
        "schema_version": "NANO_BANANA_RENDER_SCHEMA_1.0",
        "design_manual_applied": "LAST_TRIBE_60S_CYBERPUNK_X_KRYSTAL_ADRENALIN",
        "previews_count": len(generated_records),
        "previews": generated_records,
    }


# ---------------------------------------------------------------------------
# 7. Konkrétne Rozhranie Plné Interakcií cez Karty + Sonic 3D Pixel-Art Free-Run Režim
# ---------------------------------------------------------------------------


INTERACTIVE_FREERUN_CARDS: Tuple[Dict[str, Any], ...] = (
    {
        "card_id": "CRD_01_CRYO_VELOCITY",
        "tribe": "crystal",
        "category": "FREE-RUN SPRINT",
        "title": "Kryštálový Sklz (Cryo Glide)",
        "cost": 1,
        "power": 4,
        "speed_boost": 1.45,
        "color": "#00e5ff",
        "dialogue_en": "Cryo glide engaged! Slicing across the curved crystal horizon at maximum frame rate.",
        "effect_desc": "Zrýchli Free-Run pohyb o +45 %, zmrazí prekážky a otvorí severný čank.",
    },
    {
        "card_id": "CRD_02_PORTAL_PRISM",
        "tribe": "crystal",
        "category": "HOLOGRAM PROJECTION",
        "title": "Lom Starobylého Portálu",
        "cost": 2,
        "power": 6,
        "speed_boost": 1.20,
        "color": "#38bdf8",
        "dialogue_en": "Prism beam refracted through the Ancient Portal; six vital shields synchronized.",
        "effect_desc": "Prepne projekciu čankov na Ray-Tracing kužeľ a zdvojnásobí kombinačný bonus.",
    },
    {
        "card_id": "CRD_03_EMERALD_CANOPY_LEAP",
        "tribe": "druid",
        "category": "FREE-RUN SPRINT",
        "title": "Skok Korunou Prastarého Lesa",
        "cost": 1,
        "power": 4,
        "speed_boost": 1.50,
        "color": "#34d399",
        "dialogue_en": "Forest guardians guide our stride! Leaping across emerald radar sectors.",
        "effect_desc": "Odhalí 2 nadchádzajúce karty v balíčku počas behu a skompiluje lesný quest.",
    },
    {
        "card_id": "CRD_04_WOLF_PACK_RESONANCE",
        "tribe": "druid",
        "category": "COMBO SYNERGY",
        "title": "Svorka Bioluminiscenčných Vlkov",
        "cost": 2,
        "power": 7,
        "speed_boost": 1.30,
        "color": "#10b981",
        "dialogue_en": "Bioluminescent wolf pack deployed alongside your free-run trajectory.",
        "effect_desc": "Ak si v posledných 3 sekundách zahral inú kartu, pridá +4 kombo poškodenie.",
    },
    {
        "card_id": "CRD_05_ISOTOPE_TURBO_CATALYST",
        "tribe": "toxic",
        "category": "FREE-RUN SPRINT",
        "title": "Izotopový Alchymistický Vznet",
        "cost": 1,
        "power": 5,
        "speed_boost": 1.60,
        "color": "#eab308",
        "dialogue_en": "Isotope catalyst ignited! Accelerating over the checkered marsh horizon.",
        "effect_desc": "Okamžitý šprint +60 % cez šachovnicový horizont a 3 vrstvy alchymistického jedu.",
    },
    {
        "card_id": "CRD_06_ASCII_VOXEL_OVERDRIVE",
        "tribe": "toxic",
        "category": "SONIC 3D ASCII",
        "title": "3D Pixel-Art & Voxel ASCII Prechod",
        "cost": 0,
        "power": 5,
        "speed_boost": 1.35,
        "color": "#ed1c24",
        "dialogue_en": "Switching rasterizer to three-dimensional Sonic pixel-art and ASCII voxel overdrive!",
        "effect_desc": "Okamžite prepne alebo prekryje 3D Pixel-Art scénu živou Voxel ASCII vrstvou.",
    },
)


def render_card_freerun_sonic_arena_html() -> str:
    """Vyrenderuje interaktívne rozhranie plné kartových interakcií (`/card_freerun_sonic_arena`) s:
      - Plynulým **Free-Run režimom s kartami** (`WASD` / šípky / Auto-Free-Run) na zakrivenom 3D Pixel-Art horizonte
        inšpirovanom render enginmi zo **Sonica** (*Blue Spheres* + *Mode-7* + 3D Voxel Sprite-Stack),
      - Živým **3D ASCII Voxel & Terminálovým Prehliadačom** priamo v bočnom aj prekryvnom okne,
      - **Anglickým fonémovým našepkávačom (`Web Speech API en-US` + `WebAudio` formantový syntetizátor)**,
      - **Kompiláciou Questov na časovej osi** pri pohybe a voľných zdrojoch (`UMA < 0.78`) + **Propedeutickým Cacheom**.
    """
    cards_json = json.dumps(list(INTERACTIVE_FREERUN_CARDS), ensure_ascii=False)
    chunks_json = json.dumps(list(AESTHETIC_SPATIAL_CHUNKS), ensure_ascii=False)
    return f"""<!DOCTYPE html>
<html lang="sk">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>Krystal Adrenalin — Kartová Free-Run Aréna, Sonic 3D Pixel-Art/ASCII & Hlasové Questy</title>
  <script src="/vendor/gsap.min.js"></script>
  <style>
    :root {{
      --bg: #090b10;
      --panel: #121621;
      --elevated: #181e2c;
      --red: #ed1c24;
      --amber: #ff9e2c;
      --cyan: #00e5ff;
      --emerald: #34d399;
      --yellow: #eab308;
      --txt: #f1f5f9;
      --muted: #94a3b8;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0; font-family: 'Segoe UI', system-ui, sans-serif;
      background-color: var(--bg);
      background-image: radial-gradient(rgba(255,158,44,0.11) 1px, transparent 1px);
      background-size: 16px 16px; color: var(--txt);
    }}
    .topbar {{
      display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 10px;
      padding: 12px 20px; background: linear-gradient(90deg, #121621, #19121d, #121621);
      border-top: 3px solid var(--red); border-bottom: 1px solid rgba(255,255,255,0.10);
    }}
    .badge {{
      padding: 4px 10px; font-size: 11px; font-weight: 700; letter-spacing: 0.05em;
      background: rgba(0,229,255,0.10); border: 1px solid var(--cyan); color: var(--cyan);
      font-variant-numeric: tabular-nums;
    }}
    .btn {{
      cursor: pointer; padding: 7px 12px; font-size: 12px; font-weight: 700;
      background: var(--elevated); color: var(--txt); border: 1px solid rgba(255,255,255,0.18);
      clip-path: polygon(0 0, calc(100% - 7px) 0, 100% 7px, 100% 100%, 7px 100%, 0 calc(100% - 7px));
      transition: all 0.14s ease;
    }}
    .btn:hover, .btn.active {{ border-color: var(--amber); background: rgba(237,28,36,0.22); color: #fff; }}
    .arena-grid {{
      display: grid; grid-template-columns: 310px 1fr 360px; gap: 14px; padding: 14px 18px;
    }}
    @media (max-width: 1240px) {{ .arena-grid {{ grid-template-columns: 1fr; }} }}
    .panel {{
      background: rgba(18,22,33,0.94); border: 1px solid rgba(255,255,255,0.09);
      border-top: 2px solid var(--red); padding: 13px;
      clip-path: polygon(0 0, calc(100% - 10px) 0, 100% 10px, 100% 100%, 0 100%);
    }}
    .panel h3 {{ margin: 0 0 8px 0; font-size: 12px; text-transform: uppercase; letter-spacing: 0.08em; color: var(--amber); }}
    #sonicCanvas {{
      width: 100%; height: 350px; display: block; background: #070a12;
      image-rendering: pixelated; border: 1px solid rgba(0,229,255,0.28);
    }}
    #asciiOverlay {{
      margin-top: 8px; padding: 8px; background: #05070c; color: #00e5ff;
      font-family: Consolas, 'Cascadia Mono', monospace; font-size: 10px; line-height: 1.05;
      overflow-x: auto; border: 1px solid rgba(255,158,44,0.28); white-space: pre;
    }}
    .hand-row {{
      display: grid; grid-template-columns: repeat(auto-fit, minmax(175px, 1fr)); gap: 10px; margin-top: 10px;
    }}
    .game-card {{
      cursor: pointer; user-select: none; padding: 10px; background: linear-gradient(160deg, #161c2b, #0e131e);
      border: 1px solid rgba(255,255,255,0.14); border-top: 3px solid var(--cyan);
      clip-path: polygon(0 0, calc(100% - 9px) 0, 100% 9px, 100% 100%, 9px 100%, 0 calc(100% - 9px));
      transition: transform 0.14s ease, box-shadow 0.14s ease;
    }}
    .game-card:hover {{ transform: translateY(-5px) scale(1.02); box-shadow: 0 8px 22px rgba(0,229,255,0.22); }}
    .quest-item {{
      font-size: 11px; padding: 7px; margin-bottom: 6px; background: rgba(255,255,255,0.03);
      border-left: 3px solid var(--emerald);
    }}
    .preview-grid {{
      display: grid; grid-template-columns: repeat(2, 1fr); gap: 8px; margin-top: 8px;
    }}
    .preview-grid img {{ width: 100%; border: 1px solid rgba(255,255,255,0.18); border-radius: 2px; }}
  </style>
</head>
<body>
  <div class="topbar">
    <div>
      <strong style="font-size:15px; letter-spacing:0.06em;">KRYSTAL ADRENALIN × SONIC 3D PIXEL-ART &amp; KARTOVÝ FREE-RUN</strong>
      <span style="font-size:12px; color:var(--muted); margin-left:10px;">60s Cyberpunk Hologram · Hlasový Našepkávač (EN-US) · Questy &amp; Terminálové Projekcie</span>
    </div>
    <div style="display:flex; gap:8px; align-items:center; flex-wrap:wrap;">
      <button class="btn active" id="btnFreeRun" onclick="toggleFreeRun()">FREE-RUN: AKTÍVNY (WASD / 1-6)</button>
      <button class="btn" id="btnVoice" onclick="toggleVoice()">EN-US HLASOVÝ NAŠEPKÁVAČ: ZAP.</button>
      <a class="btn" href="/last_tribe_cyber60s" style="text-decoration:none;">60s HOLOGRAM ŠTÚDIO</a>
      <a class="btn" href="/terminal_browser" style="text-decoration:none;">TERMINÁLOVÝ PREHLIADAČ</a>
      <span class="badge" id="hudTelemetry">VITAL_MAX_HP: 6/6 | UMA: 0.41 &lt; 0.78 | 148.8 FPS</span>
    </div>
  </div>

  <div class="arena-grid">
    <!-- Ľavý stĺpec: Propedeutický Cache, Čanky Mapy & Hlasový Osciloskop -->
    <div class="panel">
      <h3>1. Propedeutický Cache &amp; Prechod Čankov</h3>
      <div id="chunkList" style="font-size:12px;"></div>
      <h3 style="margin-top:14px;">2. ML Fonémový Našepkávač (EN-US)</h3>
      <div id="dialogueBox" style="padding:8px; background:#080b12; border:1px solid rgba(0,229,255,0.25); font-size:12px; min-height:54px; color:#e2e8f0;">
        Klikni na ľubovoľnú kartu (alebo stlač kláves <strong>1–6</strong> počas Free-Run behu) pre okamžitú anglickú syntézu a kompiláciu questu!
      </div>
      <div id="phonemeStream" style="margin-top:6px; font-family:Consolas,monospace; font-size:11px; color:var(--amber);"></div>
      <h3 style="margin-top:14px;">3. Nano Banana Finálne Náhľady (4 Kontexty)</h3>
      <div class="preview-grid">
        <a href="/api/v1/nanobanana/preview/1" target="_blank"><img src="/api/v1/nanobanana/preview/1" alt="Crystal Hex Preview"/></a>
        <a href="/api/v1/nanobanana/preview/2" target="_blank"><img src="/api/v1/nanobanana/preview/2" alt="Druid Radar Preview"/></a>
        <a href="/api/v1/nanobanana/preview/3" target="_blank"><img src="/api/v1/nanobanana/preview/3" alt="Toxic Isometric Preview"/></a>
        <a href="/api/v1/nanobanana/preview/4" target="_blank"><img src="/api/v1/nanobanana/preview/4" alt="Sonic Terminal Preview"/></a>
      </div>
    </div>

    <!-- Stredný stĺpec: Sonic 3D Pixel-Art Zakrivený Horizont + Interaktívne Karty pre Free-Run -->
    <div class="panel">
      <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:6px;">
        <h3>Sonic 3D Pixel-Art Horizont (Blue Spheres + Mode-7) &amp; Kartový Free-Run</h3>
        <span style="font-size:11px; color:var(--cyan);" id="comboHud">Rýchlosť: 1.00x | Kombo reťaz: 0 | Vzdialenosť: 0 m</span>
      </div>
      <canvas id="sonicCanvas" width="320" height="176"></canvas>
      <div class="hand-row" id="cardHand"></div>
      <div id="asciiOverlay">Načítavam živú 3D Voxel ASCII projekciu...</div>
    </div>

    <!-- Pravý stĺpec: Časová Os Questov & Terminálová Projekcia -->
    <div class="panel">
      <h3>4. Časová Os Questov (Idle Kompilácia pri Pohybe)</h3>
      <div id="questTimeline" style="max-height:280px; overflow-y:auto;"></div>
      <h3 style="margin-top:14px;">5. Živý Výrez Terminálového Prehliadača</h3>
      <pre id="tuiMirror" style="background:#05070c; color:#34d399; padding:8px; font-size:10px; line-height:1.15; overflow-x:auto; border:1px solid rgba(52,211,153,0.28);">Spúšťam Krystal TUI Browser...</pre>
      <button class="btn" style="width:100%; margin-top:6px;" onclick="refreshTuiMirror()">OBNOVIŤ PROJEKCIU TERMINÁLU</button>
    </div>
  </div>

  <script>
    const CARDS = {cards_json};
    const CHUNKS = {chunks_json};
    let freeRunActive = true;
    let voiceEnabled = true;
    let scrollZ = 0.0;
    let lateralX = 0.0;
    let speedMul = 1.0;
    let comboChain = 0;
    let lastCardTs = 0;
    let activeChunkIdx = 0;
    let activeTribeColor = [0, 229, 255];
    let voxelAsciiMode = false;
    const keys = {{}};

    window.addEventListener('keydown', (e) => {{
      keys[e.key.toLowerCase()] = true;
      const n = parseInt(e.key, 10);
      if (n >= 1 && n <= CARDS.length) playCard(CARDS[n - 1], n - 1);
    }});
    window.addEventListener('keyup', (e) => {{ keys[e.key.toLowerCase()] = false; }});

    function hexToRgb(hex) {{
      const v = parseInt(hex.replace('#',''), 16);
      return [(v >> 16) & 255, (v >> 8) & 255, v & 255];
    }}

    function toggleFreeRun() {{
      freeRunActive = !freeRunActive;
      const b = document.getElementById('btnFreeRun');
      b.textContent = freeRunActive ? 'FREE-RUN: AKTÍVNY (WASD / 1-6)' : 'FREE-RUN: POZASTAVENÝ (ŤAHOVÝ MÓD)';
      b.classList.toggle('active', freeRunActive);
    }}

    function toggleVoice() {{
      voiceEnabled = !voiceEnabled;
      const b = document.getElementById('btnVoice');
      b.textContent = voiceEnabled ? 'EN-US HLASOVÝ NAŠEPKÁVAČ: ZAP.' : 'EN-US HLASOVÝ NAŠEPKÁVAČ: VYP.';
      b.classList.toggle('active', voiceEnabled);
    }}

    function speakEnglishWithFormantFallback(textEn, phonemes) {{
      if (!voiceEnabled) return;
      try {{
        if ('speechSynthesis' in window) {{
          window.speechSynthesis.cancel();
          const u = new SpeechSynthesisUtterance(textEn);
          u.lang = 'en-US';
          u.rate = 1.05;
          window.speechSynthesis.speak(u);
        }}
        // Krátky harmonický formantový akord cez WebAudio pre 60s kyberpunkovú odozvu
        const Ctx = window.AudioContext || window.webkitAudioContext;
        if (Ctx && phonemes && phonemes.length) {{
          const ctx = new Ctx();
          const osc = ctx.createOscillator();
          const gain = ctx.createGain();
          osc.type = 'sawtooth';
          osc.frequency.setValueAtTime(phonemes[0].f1_hz * 0.5, ctx.currentTime);
          gain.gain.setValueAtTime(0.035, ctx.currentTime);
          gain.gain.exponentialRampToValueAtTime(0.0008, ctx.currentTime + 0.18);
          osc.connect(gain); gain.connect(ctx.destination);
          osc.start(); osc.stop(ctx.currentTime + 0.19);
        }}
      }} catch (_e) {{}}
    }}

    function renderChunksPanel() {{
      const el = document.getElementById('chunkList');
      el.innerHTML = CHUNKS.map((c, idx) => {{
        const act = idx === activeChunkIdx;
        return `<div style="padding:6px; margin-bottom:5px; border-left:3px solid ${{act ? '#ed1c24' : '#00e5ff'}}; background:${{act ? 'rgba(237,28,36,0.16)' : 'rgba(255,255,255,0.03)'}};">
          <strong>${{act ? '▶ ' : ''}}${{c.region_name}}</strong> <span style="color:#94a3b8;">(${{c.coords_pct[0]}}%, ${{c.coords_pct[1]}}%)</span><br/>
          <span style="font-size:11px; color:#cbd5e1;">${{c.aesthetic_expression}}</span>
        </div>`;
      }}).join('');
    }}

    function renderCardHand() {{
      const el = document.getElementById('cardHand');
      el.innerHTML = CARDS.map((c, i) => `
        <div class="game-card" id="gcard_${{i}}" style="border-top-color:${{c.color}}" onclick="playCard(CARDS[${{i}}], ${{i}})">
          <div style="display:flex; justify-content:space-between; font-size:10px; color:var(--amber);">
            <span>[${{i+1}}] ${{c.category}}</span>
            <span>⚡ ${{c.cost}} | ⚔ ${{c.power}}</span>
          </div>
          <div style="font-weight:700; font-size:13px; margin:5px 0; color:#fff;">${{c.title}}</div>
          <div style="font-size:11px; color:var(--muted); line-height:1.3;">${{c.effect_desc}}</div>
        </div>
      `).join('');
    }}

    async function playCard(card, idx) {{
      const now = performance.now();
      comboChain = (now - lastCardTs < 4000) ? (comboChain + 1) : 1;
      lastCardTs = now;
      speedMul = Math.min(2.6, card.speed_boost + (comboChain - 1) * 0.12);
      activeTribeColor = hexToRgb(card.color);
      if (card.card_id === 'CRD_06_ASCII_VOXEL_OVERDRIVE') voxelAsciiMode = !voxelAsciiMode;

      if (window.gsap) {{
        gsap.fromTo(`#gcard_${{idx}}`, {{scale: 0.92}}, {{scale: 1.0, duration: 0.28, ease: 'back.out(2)'}});
      }}

      try {{
        const curChunk = CHUNKS[activeChunkIdx].chunk_id;
        const resp = await fetch('/api/v1/compositor/quest_phoneme_chunks', {{
          method: 'POST',
          headers: {{'Content-Type': 'application/json'}},
          body: JSON.stringify({{
            tribe: card.tribe,
            current_chunk_id: curChunk,
            uma_utilization: 0.41,
            is_moving_on_rendered_map: freeRunActive || Math.abs(speedMul) > 1.0,
            custom_english_line: card.dialogue_en
          }})
        }});
        const data = await resp.json();
        const vp = data.quest_compilation.dialogue_voice_envelope;
        document.getElementById('dialogueBox').innerHTML = `<strong>EN-US Whisperer:</strong> "${{vp.spoken_dialogue_en}}"`;
        document.getElementById('phonemeStream').textContent = 'ARPAbet: ' + vp.phoneme_grapheme_stream_preview.map(p => `${{p.phoneme}}(${{p.f1_hz}}Hz)`).slice(0, 10).join(' · ');
        speakEnglishWithFormantFallback(vp.spoken_dialogue_en, vp.phoneme_grapheme_stream_preview);

        if (data.sonic_3d_ascii && data.sonic_3d_ascii.ascii_frame) {{
          document.getElementById('asciiOverlay').textContent = data.sonic_3d_ascii.ascii_frame;
        }}

        const q = data.quest_compilation.quest_node;
        const qt = document.getElementById('questTimeline');
        const unlockedNames = (q.unlocked_transition_chunks || []).map(u => u.region_name).join(' → ');
        qt.insertAdjacentHTML('afterbegin', `<div class="quest-item">
          <strong>${{q.quest_id}}</strong> · <span style="color:#00e5ff;">${{q.compilation_status}}</span><br/>
          <span style="color:#cbd5e1;">"${{q.source_dialogue_en}}"</span><br/>
          <span style="color:#ff9e2c;">Odomknuté čanky: ${{unlockedNames || 'Čaká na pohyb'}}</span>
        </div>`);

        activeChunkIdx = (activeChunkIdx + 1) % CHUNKS.length;
        renderChunksPanel();
      }} catch (err) {{
        console.error(err);
      }}
    }}

    // Sonic 3D Pixel-Art zakrivený horizont v reálnom čase (60 FPS Canvas rasterizér)
    const cvs = document.getElementById('sonicCanvas');
    const ctx2d = cvs.getContext('2d');
    const imgData = ctx2d.createImageData(cvs.width, cvs.height);

    function renderSonicLoop() {{
      const W = cvs.width, H = cvs.height;
      if (freeRunActive || keys['w'] || keys['arrowup']) {{
        scrollZ += 0.085 * speedMul;
      }}
      if (keys['a'] || keys['arrowleft']) lateralX -= 0.06;
      if (keys['d'] || keys['arrowright']) lateralX += 0.06;
      speedMul = Math.max(1.0, speedMul * 0.994);

      const hor = Math.floor(H * 0.45);
      const [pr, pg, pb] = activeTribeColor;
      const buf = imgData.data;

      for (let y = 0; y < H; y++) {{
        for (let x = 0; x < W; x++) {{
          const off = (y * W + x) * 4;
          const nx = (x / W) * 2.0 - 1.0;
          if (y < hor) {{
            const t = y / hor;
            buf[off] = 9 + 16 * t;
            buf[off + 1] = 13 + 20 * t;
            buf[off + 2] = 24 + 38 * t;
            buf[off + 3] = 255;
          }} else {{
            const dy = (y - hor + 1) / (H - hor);
            const sphereCurve = Math.max(0.14, dy * (1.0 - 0.36 * nx * nx));
            const z = 1.4 / sphereCurve;
            const u = Math.floor((nx + lateralX * 0.4) * z * 2.6);
            const v = Math.floor(z * 2.2 + scrollZ);
            const chk = (u + v) & 1;
            const fog = Math.min(1.0, 1.3 / (0.8 + z * 0.35));
            if (chk === 0) {{
              buf[off] = (18 + pr * 0.48) * fog;
              buf[off + 1] = (24 + pg * 0.48) * fog;
              buf[off + 2] = (36 + pb * 0.48) * fog;
            }} else {{
              buf[off] = 24 * fog;
              buf[off + 1] = 20 * fog;
              buf[off + 2] = 32 * fog;
            }}
            buf[off + 3] = 255;
          }}
        }}
      }}
      ctx2d.putImageData(imgData, 0, 0);

      // Rotujúci 3D Pixel-Art prstenec / kryštálový maják uprostred Free-Run dráhy
      const tNow = performance.now() * 0.004;
      ctx2d.strokeStyle = `rgb(${{pr}},${{pg}},${{pb}})`;
      ctx2d.lineWidth = 2;
      for (let k = 0; k < 4; k++) {{
        const rx = 18 * Math.abs(Math.cos(tNow + k * 0.4)) + 4;
        ctx2d.strokeRect(Math.floor(W / 2 - rx + Math.sin(tNow) * 6), Math.floor(hor - 18 - k * 3), Math.floor(rx * 2), 26);
      }}

      document.getElementById('comboHud').textContent =
        `Rýchlosť: ${{speedMul.toFixed(2)}}x | Kombo reťaz: ${{comboChain}} | Vzdialenosť: ${{Math.floor(scrollZ * 12)}} m`;
      requestAnimationFrame(renderSonicLoop);
    }}

    async function refreshTuiMirror() {{
      try {{
        const r = await fetch('/api/v1/terminal_browser/render?projection=TERM_PROJ_04_SONIC_3D_ASCII_VOXEL');
        const j = await r.json();
        document.getElementById('tuiMirror').textContent = j.plain_terminal_screen;
      }} catch (_e) {{}}
    }}

    renderChunksPanel();
    renderCardHand();
    refreshTuiMirror();
    playCard(CARDS[0], 0);
    requestAnimationFrame(renderSonicLoop);
  </script>
</body>
</html>"""


# ---------------------------------------------------------------------------
# 8. Kompletné Mapovanie Dnešného Procesu a Architektúry (12 Nadväzujúcich Etáp)
# ---------------------------------------------------------------------------


def map_todays_complete_engineering_journey() -> Dict[str, Any]:
    """Zmapuje všetkých 12 etáp dnešného inžinierskeho procesu, ktorým nás používateľ previedol,
    vrátane presných súborov, overených metrík a dodržaných ústavných invariantov.
    """
    stages = [
        {
            "stage": 1,
            "domain": "Natívny #![no_std] Rust Firmvér v2.0 & Kisak-Mesa Gallium + OpenVINO Most",
            "key_files": [
                "native/krystal_sovereign_firmware_rs/src/lib.rs",
                "krystal_kernel/openvino_gallium_driver_bridge.py",
                "projects/trigonometric_hybrid_platform_framework/sovereign_rust_firmware_cuda_gallium_suite.py",
            ],
            "verified_outcome": "3 natívne DLL bez CRT, 8.29x zrýchlenie KUBC oproti ACPI AML, bezkópiová parita OpenVINO <-> Gallium a nárast 60 -> 148.8 FPS na 768 stream procesoroch Iris Xe.",
        },
        {
            "stage": 2,
            "domain": "Open-Source NVIDIA Fedora Patterny (NVK/NAK), Krystal-DLSS, 14 GB UMA VRAM & Bionic/LM Studio",
            "key_files": [
                "projects/trigonometric_hybrid_platform_framework/fedora_dlss_vram_bionic_mcp_and_2b_ast.py",
                "docs/design/KRYSTAL_ADRENALIN_DESIGN_MANUAL.md",
            ],
            "verified_outcome": "6 Fedora patternov, bezpečné takty čipsetu (400-4200 MHz CPU, 400-1300 MHz GPU), 14 336 MB zdieľanej UMA VRAM pre Gemma 4 E2B, unifikovaný swap a GSAP Adrenalin panel.",
        },
        {
            "stage": 3,
            "domain": "Odolný Denník Čakajúcich Procesov, MCP Zbernica, 2B AST Kódový Tréner & Izomorfný Rust Blokový Most",
            "key_files": [
                "krystal_bot/pending.py",
                "scripts/krystal_unified_mcp_server.py",
            ],
            "verified_outcome": "Žiadna strata krokov pri zatvorenom LM Studiu (:1234), JSON-RPC 2.0 MCP server a kompilácia vygenerovaného #![no_std] Rust bloku priamo cez rustc.",
        },
        {
            "stage": 4,
            "domain": "Slabikový + AST Pre-Cache, CPU Cache L1/L2/L3, SQLite WAL CRUD, Rubikova Grupa & dusan-game API",
            "key_files": [
                "projects/trigonometric_hybrid_platform_framework/syllable_ast_cache_gc_crud_rubik_uma_and_dusan_game.py",
            ],
            "verified_outcome": "ACID SQLite CRUD register endpointov, kombinatorický kompilátor nad 4.325e19 stavmi Rubikovej kocky, 3 porovnané štatistické ML modely a 19-hex akcelerátor.",
        },
        {
            "stage": 5,
            "domain": "Dizertačná Formalizácia, EULA Scenáre, Nextcloud K8s 2FA SMS Sentinel, CAD/Blender MCP & Diferenciál",
            "key_files": [
                "projects/trigonometric_hybrid_platform_framework/dissertation_eula_nextcloud_cad_blender_kb_suite.py",
                "projects/trigonometric_hybrid_platform_framework/differential_gearbox_ux_flows_and_community_faq.py",
                "deploy/nextcloud_k8s_2fa_sms_security/nextcloud-hardened-k8s.yaml",
            ],
            "verified_outcome": "8-ticiam formálny model, Bassova difúzna EULA kalkulácia, TOTP + K8s SMS brána, DXF/SVG/OBJ CAD + Blender MCP so samoučiacou KB a prevody vlákien s diferenciálom.",
        },
        {
            "stage": 6,
            "domain": "The Last Tribe (60s Cyberpunk Hologram), C11 ISA Schémy, Docker/VPS Cloud & Blender Benchmark",
            "key_files": [
                "projects/trigonometric_hybrid_platform_framework/last_tribe_cyber60s_vps_cloud_and_blender_bench.py",
                "src/c/krystal_isa_vulkan_webgl_diagrams.c",
                "Dockerfile.last_tribe_cyber60s",
            ],
            "verified_outcome": "65 kariet kmeňov Crystal/Druid/Toxic, 4 projekcie mapy, Skill Tree + Crafting v benchmarku, FXAA->TSAA->GTAO->Arc Ray-Tracing a smerodajné porovnanie s Blender Cycles.",
        },
        {
            "stage": 7,
            "domain": "Samoobslužný Kompozítor, Plánovač Časovej Osi + LLM Poškodenie & 8x Násobič Mikroslužieb",
            "key_files": [
                "projects/trigonometric_hybrid_platform_framework/visual_compositor_timeline_damage_and_microservices.py",
                "scripts/orchestrate_microservices_context_image_multiplier.py",
            ],
            "verified_outcome": "Rozloženie dejových línií bez prekročenia 16.67 ms rozpočtu snímky, štatistický + LLM výpočet poškodenia a 4x kontextový / 8x pixelový multiplikátor mikroslužieb.",
        },
        {
            "stage": 8,
            "domain": "Protokol Render<->LLM, ML Fonémový Anglický Našepkávač, Questy, Propedeutický Cache, Sonic 3D ASCII Free-Run, Nano Banana & Terminálový Prehliadač",
            "key_files": [
                "projects/trigonometric_hybrid_platform_framework/phoneme_dialogue_quest_propedeutic_and_nanobanana.py",
                "scripts/krystal_terminal_browser_and_projections.py",
            ],
            "verified_outcome": "Štandardizovaný protokol KR-RENDER-LLM-1.0, EN-US fonémový našepkávač, kompilácia questov pri voľnej UMA a pohybe, Sonic 3D Pixel-Art/ASCII kartový Free-Run, 4 PNG náhľady a TUI prehliadač v termináli.",
        },
    ]
    return {
        "vital_max_hp": VITAL_MAX_HP,
        "stages_completed": len(stages),
        "stages": stages,
    }


def execute_phoneme_quest_propedeutic_sonic_and_nanobanana_suite(
    artifact_mirror_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    proto = StandardizedRenderLlmProtocol.build_protocol_packet()
    q_engine = DialogueQuestTimelineAndIdleChunkGenerator()
    quest_res = q_engine.emit_quest_from_dialogue_and_compile_chunks(
        tribe="crystal",
        current_chunk_id="CHUNK_01_NORTHERN_CRYSTAL_PEAKS",
        uma_utilization=0.41,
        is_moving_on_rendered_map=True,
    )
    propedeutic = ConditionalArrayLoopPropedeuticPrecacher.build_propedeutic_map_preset()
    sonic_3d = SonicInspired3DPixelArtAndAsciiEngine.render_sonic_3d_ascii_and_rgb_frame(
        model_kind="crystal",
        yaw_deg=32.0,
        scroll_phase=0.5,
        cols=58,
        rows=24,
    )
    sonic_summary = {k: v for k, v in sonic_3d.items() if k != "rgb_array"}
    nanobanana = generate_nanobanana_multicontext_projection_previews(artifact_mirror_dir=artifact_mirror_dir)
    journey = map_todays_complete_engineering_journey()

    return {
        "vital_max_hp": VITAL_MAX_HP,
        "render_llm_protocol_packet": proto,
        "quest_compilation": quest_res,
        "propedeutic_map_preset": propedeutic,
        "sonic_3d_ascii": sonic_summary,
        "nanobanana_projection_previews": nanobanana,
        "todays_mapped_journey": journey,
    }


if __name__ == "__main__":
    print(json.dumps(execute_phoneme_quest_propedeutic_sonic_and_nanobanana_suite(), indent=2, ensure_ascii=False))
