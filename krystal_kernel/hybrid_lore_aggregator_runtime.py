#!/usr/bin/env python3
"""
KRYSTAL-STACK: HYBRID LORE-LLM AGGREGATOR, V-RAY ALGEBRAIC SIGNAL PIPELINE,
MARIÁŠ CARD STRATEGY, GRID-TO-BINARY CMS BRIDGE & K8S/FAIL2BAN RUNTIME COMPILER
================================================================================
Unifies:
  1. Algebraic V-Ray-Style Render Signal Reader (irradiance, GI bounce, GGX
     roughness, caustic flux, and SPIR-V shader taxonomy weights).
  2. Fantasy Cognition Genesis & Dynamic Lore-LLM / Mortar Choice Hierarchy
     (rebuilds storyline perception at every new game start with learned
     cognition adjustments and endless branching choices).
  3. Mariáš Algebraic Card/Turn Strategy & Semantic Click Animation Queue
     (handles suit/trump 'tromf' algebra, King+Ober 20/40 marriage 'hláška',
     pending action queue, and semantic animation descriptors upon click).
  4. Grid Mechanical Movement -> Binary Bitboard CMS Bridge (emulates mechanical
     movements on an 8x8 64-bit grid, packing kinematic states into compact
     `KMB1` binary frames that synchronize deterministically with Strapi CMS,
     MedusaJS, and WooCommerce).
  5. Taxonomy Portable Runtime Compiler & Zero-Trust Security Generator
     (Kubernetes Default-Deny NetworkPolicy + Hardened Deployment + Fail2Ban
     jails + nftables default-drop firewall) with full CLI validation.

System Invariant: VITAL_MAX_HP = 6
Author: Dušan Kopecký & Krystal-Stack Architecture Council (2026)
"""

from __future__ import annotations

import argparse
import base64
import dataclasses
import hashlib
import json
import math
import struct
import sys
import time
import urllib.request
import zlib
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

VITAL_MAX_HP: int = 6

REPO_ROOT = Path(__file__).resolve().parent.parent
TAXONOMY_SHADER_DIR = REPO_ROOT / "src" / "shaders" / "taxonomy"
DEFAULT_SECURITY_DIR = REPO_ROOT / "deploy" / "security"
DEFAULT_RUNTIME_DIST = REPO_ROOT / "dist" / "taxonomy_hybrid_runtime"


class PerceptionLens(str, Enum):
    SOLAR_CLARITY = "SOLAR_CLARITY"
    CAUSTIC_SUSPICION = "CAUSTIC_SUSPICION"
    UMBRAL_NOIR_MORTAR = "UMBRAL_NOIR_MORTAR"
    PRISMATIC_OVERDRIVE = "PRISMATIC_OVERDRIVE"


class MariasSuit(str, Enum):
    SRDCIA_HEARTS = "HEARTS_SRDCIA"     # Classic 40-point trump marriage
    ZELENE_LEAVES = "LEAVES_ZELENE"
    ZALUDE_ACORNS = "ACORNS_ZALUDE"
    GULE_BELLS = "BELLS_GULE"


class CMSPlatform(str, Enum):
    STRAPI_V5 = "strapi_v5"
    WOOCOMMERCE_REST = "woocommerce_rest"
    MEDUSA_OPEN_COMMERCE = "medusa_open_commerce"


@dataclass
class RenderPipelineTerms:
    direct_irradiance: float = 1.35
    gi_bounce_luminance: float = 0.62
    specular_roughness: float = 0.24
    caustic_photon_flux: float = 0.48
    ao_occlusion: float = 0.18
    mortar_elevation_deg: float = 42.5
    frame_time_ms: float = 8.33

    def to_dict(self) -> Dict[str, float]:
        return dataclasses.asdict(self)


@dataclass
class VRayAlgebraicSignal:
    total_radiance_term: float
    luminance_contrast_ratio: float
    chromatic_entropy: float
    mortar_ballistic_impulse: float
    perception_shift_index: float
    active_lens: str
    taxonomy_kernel_weights: Dict[str, float]
    vital_max_hp: int = VITAL_MAX_HP

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


class VRayAlgebraicSignalExtractor:
    CANONICAL_TAXONOMY_KERNELS = (
        "branch_heavy",
        "fp32_poly",
        "int8_dot",
        "mem_stream",
    )

    def __init__(self, taxonomy_dir: Path = TAXONOMY_SHADER_DIR) -> None:
        self.taxonomy_dir = taxonomy_dir

    def inspect_taxonomy_kernels(self) -> Dict[str, Dict[str, Any]]:
        inventory: Dict[str, Dict[str, Any]] = {}
        for name in self.CANONICAL_TAXONOMY_KERNELS:
            comp_path = self.taxonomy_dir / f"{name}.comp"
            spv_path = self.taxonomy_dir / f"{name}.spv"
            spv_exists = spv_path.exists()
            spv_bytes = spv_path.stat().st_size if spv_exists else 0
            sha_short = hashlib.sha256(spv_path.read_bytes()).hexdigest()[:12] if spv_exists else ""
            inventory[name] = {
                "comp_present": comp_path.exists(),
                "spv_present": spv_exists,
                "spv_size_bytes": spv_bytes,
                "spv_sha256_12": sha_short,
            }
        return inventory

    def evaluate(self, terms: RenderPipelineTerms) -> VRayAlgebraicSignal:
        inv = self.inspect_taxonomy_kernels()
        total_spv = sum(max(v["spv_size_bytes"], 1) for v in inv.values())
        weights = {k: round(max(v["spv_size_bytes"], 1) / total_spv, 4) for k, v in inv.items()}

        alpha = max(0.02, min(1.0, terms.specular_roughness))
        ao_clamp = max(0.0, min(0.95, terms.ao_occlusion))
        direct_term = terms.direct_irradiance * (1.0 - ao_clamp)
        indirect_term = terms.gi_bounce_luminance * (1.0 - 0.5 * alpha)
        caustic_term = terms.caustic_photon_flux / (1.0 + alpha * alpha)
        total_radiance = round(direct_term + indirect_term + caustic_term, 4)

        contrast_ratio = round((direct_term + caustic_term + 1e-4) / (indirect_term + 0.15), 4)
        chromatic_entropy = round(
            min(1.0, abs(math.sin(total_radiance * math.pi * 0.35)) * 0.65 + (1.0 - alpha) * 0.35),
            4,
        )
        rad_2theta = math.radians(max(5.0, min(85.0, terms.mortar_elevation_deg)) * 2.0)
        mortar_impulse = round(math.sin(rad_2theta) * (1.0 + weights.get("fp32_poly", 0.25)), 4)
        perception_shift = round(
            min(1.0, 0.4 * chromatic_entropy + 0.35 * (mortar_impulse / 1.3) + 0.25 * min(1.0, caustic_term)),
            4,
        )

        if ao_clamp >= 0.45 or total_radiance < 1.1:
            lens = PerceptionLens.UMBRAL_NOIR_MORTAR.value
        elif caustic_term >= 0.55 and alpha < 0.3:
            lens = PerceptionLens.CAUSTIC_SUSPICION.value
        elif perception_shift >= 0.68:
            lens = PerceptionLens.PRISMATIC_OVERDRIVE.value
        else:
            lens = PerceptionLens.SOLAR_CLARITY.value

        return VRayAlgebraicSignal(
            total_radiance_term=total_radiance,
            luminance_contrast_ratio=contrast_ratio,
            chromatic_entropy=chromatic_entropy,
            mortar_ballistic_impulse=mortar_impulse,
            perception_shift_index=perception_shift,
            active_lens=lens,
            taxonomy_kernel_weights=weights,
            vital_max_hp=VITAL_MAX_HP,
        )


class FantasyCognitionGenesisEngine:
    """
    Rebuilds the player's storyline perception at every new game beginning using
    Fantasy LLM cognition practices (hermeneutic, tactical-stoic, alchemical-intuitive,
    dialectic-trickster) and updates learned cognition weights across turns.
    """

    ARCHETYPES = (
        ("ALCHEMICAL_HERMENEUT", "Reads hidden rune causality in light refractions"),
        ("MORTAR_STRATEGIST", "Calculates siege arcs and deterministic supply lines"),
        ("MARIAS_TRICKSTER", "Plays concealed marriage combinations to flip allegiances"),
        ("SOVEREIGN_CHRONICLER", "Balances guild commerce with sacred citadel oaths"),
    )

    def __init__(self) -> None:
        self.learned_weights: Dict[str, float] = {
            "analytic_rigor": 0.50,
            "mythic_empathy": 0.50,
            "tactical_audacity": 0.50,
            "perceptual_plasticity": 0.50,
        }
        self.run_index: int = 0

    def begin_new_storyline(self, seed_phrase: str, signal: VRayAlgebraicSignal) -> Dict[str, Any]:
        self.run_index += 1
        digest = hashlib.sha256(f"{seed_phrase}:{self.run_index}:{signal.active_lens}".encode("utf-8")).digest()
        idx = digest[0] % len(self.ARCHETYPES)
        archetype_id, archetype_desc = self.ARCHETYPES[idx]

        # Learned cognition adjustment based on render entropy & run seed
        self.learned_weights["perceptual_plasticity"] = round(
            min(1.0, 0.35 + (digest[1] / 255.0) * 0.45 + signal.perception_shift_index * 0.2), 4
        )
        self.learned_weights["tactical_audacity"] = round(
            min(1.0, 0.30 + (digest[2] / 255.0) * 0.40 + signal.mortar_ballistic_impulse * 0.2), 4
        )

        opening_worldview = (
            f"Run #{self.run_index} [{archetype_id}]: Under {signal.active_lens}, the Citadel timeline "
            f"reconfigures—{archetype_desc.lower()}, shifting initial quest allegiances."
        )
        return {
            "run_index": self.run_index,
            "seed_phrase": seed_phrase,
            "cognition_archetype": archetype_id,
            "cognition_practice": archetype_desc,
            "learned_cognition_weights": dict(self.learned_weights),
            "opening_perception_prologue": opening_worldview,
            "vital_max_hp": VITAL_MAX_HP,
        }


class MariasAlgebraicCardTurnEngine:
    """
    Mariáš-inspired algebraic card & turn engine:
      - Evaluates card value algebra + Hláška/Marriage (King + Ober of same suit:
        40 points in Trump suit, 20 points in non-trump suit).
      - Maintains a Pending Action Queue (`actions_pending`) staged during turns.
      - Upon click confirmation (`resolve_click`), emits a deterministic Semantic
        Animation Descriptor (curve, shader glyph aura, duration, narrative beat).
    """

    RANK_POINTS = {
        "ACE_ESO": 10,
        "TEN_DESIATKA": 10,
        "KING_KRAL": 4,
        "OBER_HORNIK": 3,
        "UNTER_DOLNIK": 2,
        "NINE_DEVINA": 0,
    }

    def __init__(self, trump_suit: MariasSuit = MariasSuit.SRDCIA_HEARTS) -> None:
        self.trump_suit = trump_suit
        self.turn_number: int = 0
        self.pending_actions: List[Dict[str, Any]] = []
        self.score_ledger: int = 0

    def stage_card_play(
        self,
        suit: MariasSuit,
        rank: str,
        hand_ranks_in_suit: List[str],
        target_objective: str,
    ) -> Dict[str, Any]:
        self.turn_number += 1
        base_pts = self.RANK_POINTS.get(rank, 4)
        is_trump = suit == self.trump_suit

        # Mariáš marriage (Hláška): King + Ober in same suit
        has_marriage = (
            rank in ("KING_KRAL", "OBER_HORNIK")
            and "KING_KRAL" in hand_ranks_in_suit
            and "OBER_HORNIK" in hand_ranks_in_suit
        )
        marriage_bonus = (40 if is_trump else 20) if has_marriage else 0
        algebraic_move_power = round((base_pts * (1.5 if is_trump else 1.0)) + marriage_bonus, 2)

        action_id = f"marias_turn_{self.turn_number}_{suit.name.lower()}_{rank.lower()}"
        staged = {
            "action_id": action_id,
            "turn": self.turn_number,
            "state": "ACTION_PENDING_CLICK",
            "suit": suit.value,
            "rank": rank,
            "is_trump": is_trump,
            "marias_marriage_declared": has_marriage,
            "marriage_bonus_points": marriage_bonus,
            "algebraic_move_power": algebraic_move_power,
            "target_objective": target_objective,
        }
        self.pending_actions.append(staged)
        return staged

    def resolve_click_semantic_animation(self, action_id: Optional[str] = None) -> Dict[str, Any]:
        if not self.pending_actions:
            return {"status": "EMPTY_QUEUE", "vital_max_hp": VITAL_MAX_HP}

        if action_id:
            idx = next((i for i, a in enumerate(self.pending_actions) if a["action_id"] == action_id), 0)
        else:
            idx = 0
        action = self.pending_actions.pop(idx)
        self.score_ledger += int(action["algebraic_move_power"])

        is_marriage = action["marias_marriage_declared"]
        semantic_gesture = "ROYAL_MARIAS_CONJUNCTION_ARC" if is_marriage else "TACTICAL_TALON_STRIKE"
        shader_effect = "GOLD_TRUMP_CAUSTIC_BLOOM" if action["is_trump"] else "COBALT_SIGIL_TRAIL"

        return {
            "status": "CLICK_RESOLVED_ANIMATING",
            "resolved_action": action,
            "total_algebraic_score": self.score_ledger,
            "remaining_pending_count": len(self.pending_actions),
            "semantic_animation": {
                "gesture_archetype": semantic_gesture,
                "shader_taxonomy_pass": "fp32_poly" if is_marriage else "branch_heavy",
                "visual_aura": shader_effect,
                "bezier_control_points": [[0.0, 0.0], [0.28, 0.92], [0.74, 1.08], [1.0, 1.0]],
                "duration_ms": 680 if is_marriage else 420,
                "semantic_meaning": (
                    f"Declared Mariáš ({action['marriage_bonus_points']} pts) locks objective '{action['target_objective']}'"
                    if is_marriage
                    else f"Card strike ({action['algebraic_move_power']} power) advances '{action['target_objective']}'"
                ),
            },
            "vital_max_hp": VITAL_MAX_HP,
        }


class GridMechanicalBitboardBridge:
    """
    Solves the Engine <-> Headless CMS Bridge problem:
    Emulates mechanical actuator/unit movements on an 8x8 (64-cell) grid, packing
    each transition into a deterministic 24-byte `KMB1` binary frame:
      [4B magic 'KMB1'][8B uint64 bitboard][2B from/to cell][2B impulse_q8][4B trump_crc32][4B hp_flags]
    This compact binary payload syncs losslessly to Strapi CMS / Medusa / WooCommerce
    as base64/hex telemetry and decodes back into exact mechanical & lore state.
    """

    MAGIC = b"KMB1"

    def __init__(self) -> None:
        # Initial 8x8 bitboard occupancy (4 corner citadel automatons)
        self.bitboard: int = (1 << 0) | (1 << 7) | (1 << 56) | (1 << 63)

    @staticmethod
    def coord_to_bit(x: int, y: int) -> int:
        cx = max(0, min(7, x))
        cy = max(0, min(7, y))
        return cy * 8 + cx

    def emulate_mechanical_move(
        self,
        from_xy: Tuple[int, int],
        to_xy: Tuple[int, int],
        mechanical_torque_nm: float = 48.5,
    ) -> Dict[str, Any]:
        src_bit = self.coord_to_bit(*from_xy)
        dst_bit = self.coord_to_bit(*to_xy)

        # Clear source bit, set destination bit on 64-bit mechanical grid
        self.bitboard &= ~(1 << src_bit)
        self.bitboard |= (1 << dst_bit)

        manhattan = abs(to_xy[0] - from_xy[0]) + abs(to_xy[1] - from_xy[1])
        torque_q8 = min(65535, max(0, int(round(mechanical_torque_nm * 100.0))))
        hp_flags = (VITAL_MAX_HP & 0xFF) | ((manhattan & 0xFF) << 8)

        core_bytes = struct.pack(">4sQBBH", self.MAGIC, self.bitboard, src_bit, dst_bit, torque_q8)
        checksum = zlib.crc32(core_bytes) & 0xFFFFFFFF
        binary_frame = core_bytes + struct.pack(">II", checksum, hp_flags)

        return {
            "grid_dimensions": "8x8_UINT64_BITBOARD",
            "from_cell": {"x": from_xy[0], "y": from_xy[1], "bit_index": src_bit},
            "to_cell": {"x": to_xy[0], "y": to_xy[1], "bit_index": dst_bit},
            "manhattan_displacement": manhattan,
            "bitboard_uint64_hex": f"0x{self.bitboard:016X}",
            "binary_frame_bytes_len": len(binary_frame),
            "binary_frame_hex": binary_frame.hex(),
            "binary_frame_b64": base64.b64encode(binary_frame).decode("ascii"),
            "crc32": f"0x{checksum:08X}",
            "cms_bridge_envelope": {
                "strapi_field": "mechanical_bitboard_packet",
                "commerce_unlock_trigger": (self.bitboard & (1 << dst_bit)) != 0 and manhattan >= 2,
            },
            "vital_max_hp": VITAL_MAX_HP,
        }

    @classmethod
    def decode_binary_frame(cls, frame_hex: str) -> Dict[str, Any]:
        raw = bytes.fromhex(frame_hex)
        if len(raw) != 24 or raw[:4] != cls.MAGIC:
            raise ValueError("Invalid KMB1 mechanical grid binary frame")
        magic, bitboard, src_bit, dst_bit, torque_q8, crc, hp_flags = struct.unpack(">4sQBBHII", raw)
        calc_crc = zlib.crc32(raw[:16]) & 0xFFFFFFFF
        if calc_crc != crc:
            raise ValueError("CRC32 mismatch in KMB1 binary frame")
        vital_hp = hp_flags & 0xFF
        manhattan = (hp_flags >> 8) & 0xFF
        return {
            "magic": magic.decode("ascii"),
            "bitboard_uint64": bitboard,
            "src_bit": src_bit,
            "dst_bit": dst_bit,
            "mechanical_torque_nm": round(torque_q8 / 100.0, 2),
            "manhattan_displacement": manhattan,
            "crc_verified": True,
            "vital_max_hp": vital_hp,
        }


@dataclass
class ChoiceNode:
    choice_id: str
    tier: str
    label: str
    mortar_vector_weight: float
    moral_polarity: float
    mutated_objective: str
    sub_choices: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


class DynamicLoreChoiceMortarEngine:
    def __init__(
        self,
        ollama_url: str = "http://127.0.0.1:11434/api/generate",
        ollama_model: str = "llama3.2:3b",
    ) -> None:
        self.ollama_url = ollama_url
        self.ollama_model = ollama_model
        self.step_counter: int = 0
        self.moral_alignment: float = 0.0
        self.history: List[Dict[str, Any]] = []

    def _try_ollama_lore(self, prompt: str, timeout_s: float = 1.2) -> Optional[str]:
        payload = json.dumps({"model": self.ollama_model, "prompt": prompt, "stream": False}).encode("utf-8")
        req = urllib.request.Request(
            self.ollama_url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout_s) as resp:
                body = json.loads(resp.read().decode("utf-8"))
                text = str(body.get("response", "")).strip()
                return text if text else None
        except Exception:
            return None

    def step_lore_quest(
        self,
        quest_slug: str,
        selected_choice_id: str,
        signal: VRayAlgebraicSignal,
        cms_artifacts: Optional[List[Dict[str, Any]]] = None,
        attempt_live_ollama: bool = False,
    ) -> Dict[str, Any]:
        self.step_counter += 1
        cms_artifacts = cms_artifacts or []
        featured_artifact = cms_artifacts[0].get("title", "Krystal Relic Core") if cms_artifacts else "Krystal Relic Core"

        polarity_delta = (signal.perception_shift_index - 0.5) * 0.4
        if "harmonic" in selected_choice_id:
            polarity_delta += 0.25
        elif "mortar" in selected_choice_id:
            polarity_delta -= 0.15
        self.moral_alignment = round(max(-1.0, min(1.0, self.moral_alignment + polarity_delta)), 4)

        lens = signal.active_lens
        objective_map = {
            PerceptionLens.SOLAR_CLARITY.value: f"Stabilize the {featured_artifact} beacon across the Citadel ley-grid.",
            PerceptionLens.CAUSTIC_SUSPICION.value: f"Refract hidden faction ciphers inside {featured_artifact} before trade lock.",
            PerceptionLens.UMBRAL_NOIR_MORTAR.value: f"Triangulate Mortar ballistic arc ({signal.mortar_ballistic_impulse:.2f}) through shadow corridors.",
            PerceptionLens.PRISMATIC_OVERDRIVE.value: f"Collapse multi-timeline branches around {featured_artifact} at peak radiance ({signal.total_radiance_term:.2f}).",
        }
        active_objective = objective_map.get(lens, objective_map[PerceptionLens.SOLAR_CLARITY.value])

        narrative = None
        llm_source = "krystal_algebraic_synthesizer"
        if attempt_live_ollama:
            prompt = (
                f"Quest={quest_slug} Lens={lens} Radiance={signal.total_radiance_term} "
                f"Artifact={featured_artifact} Choice={selected_choice_id}. Write two concise sentences of reactive fantasy lore."
            )
            narrative = self._try_ollama_lore(prompt)
            if narrative:
                llm_source = f"ollama:{self.ollama_model}"

        if not narrative:
            narrative = (
                f"[{lens} // Step {self.step_counter}] Render pipeline radiance ({signal.total_radiance_term:.2f}) "
                f"and entropy ({signal.chromatic_entropy:.2f}) reshape your perception of {quest_slug}: "
                f"{featured_artifact} resonates with Mortar impulse {signal.mortar_ballistic_impulse:.2f}."
            )

        hierarchy: List[ChoiceNode] = [
            ChoiceNode(
                choice_id=f"step_{self.step_counter}_harmonic_seal",
                tier="PRIMARY_DOCTRINE",
                label=f"Attune {featured_artifact} via {lens} resonance",
                mortar_vector_weight=round(0.35 + signal.chromatic_entropy * 0.4, 3),
                moral_polarity=0.65,
                mutated_objective=active_objective,
                sub_choices=[
                    {
                        "choice_id": f"step_{self.step_counter}_harmonic_sub_1",
                        "label": "Channel indirect GI bounce to reveal allied sigils",
                        "perception_cost": round(signal.perception_shift_index * 0.5, 3),
                    },
                    {
                        "choice_id": f"step_{self.step_counter}_harmonic_sub_2",
                        "label": "Declare Mariáš Trump Marriage to bind merchant council",
                        "perception_cost": 0.15,
                    },
                ],
            ),
            ChoiceNode(
                choice_id=f"step_{self.step_counter}_mortar_barrage",
                tier="MORTAR_TACTICAL",
                label=f"Deploy Mortar ballistic lock (impulse {signal.mortar_ballistic_impulse:.2f})",
                mortar_vector_weight=signal.mortar_ballistic_impulse,
                moral_polarity=-0.45,
                mutated_objective=f"Breach outer bastion and seize {featured_artifact} supply route.",
                sub_choices=[
                    {
                        "choice_id": f"step_{self.step_counter}_mortar_sub_1",
                        "label": "Advance mechanical unit across 8x8 bitboard grid",
                        "perception_cost": 0.3,
                    }
                ],
            ),
            ChoiceNode(
                choice_id=f"step_{self.step_counter}_ephemeral_reframe",
                tier="EPHEMERAL_PERCEPTION",
                label="Invert perception lens & renegotiate faction terms",
                mortar_vector_weight=round(signal.perception_shift_index, 3),
                moral_polarity=0.1,
                mutated_objective=f"Reinterpret {quest_slug} memory graph into an open-commerce pact.",
                sub_choices=[
                    {
                        "choice_id": f"step_{self.step_counter}_ephemeral_sub_1",
                        "label": "Sync binary KMB1 state packet to Strapi & Medusa ledger",
                        "perception_cost": 0.2,
                    }
                ],
            ),
        ]

        record = {
            "status": "OK",
            "quest_slug": quest_slug,
            "step": self.step_counter,
            "llm_source": llm_source,
            "moral_alignment": self.moral_alignment,
            "active_perception_lens": lens,
            "active_objective": active_objective,
            "narrative_text": narrative,
            "vray_signal": signal.to_dict(),
            "choice_hierarchy": [c.to_dict() for c in hierarchy],
            "vital_max_hp": VITAL_MAX_HP,
        }
        self.history.append({"step": self.step_counter, "choice": selected_choice_id, "lens": lens})
        return record


class HeadlessCMSCommerceConnector:
    def __init__(
        self,
        strapi_base_url: str = "http://127.0.0.1:1337",
        medusa_base_url: str = "http://127.0.0.1:9000",
        woocommerce_base_url: str = "http://127.0.0.1:8088",
    ) -> None:
        self.strapi_base_url = strapi_base_url.rstrip("/")
        self.medusa_base_url = medusa_base_url.rstrip("/")
        self.woocommerce_base_url = woocommerce_base_url.rstrip("/")

    def _probe_json(self, url: str, timeout_s: float = 0.6) -> Optional[Dict[str, Any]]:
        req = urllib.request.Request(url, headers={"Accept": "application/json"}, method="GET")
        try:
            with urllib.request.urlopen(req, timeout=timeout_s) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception:
            return None

    def fetch_unified_catalog(
        self,
        probe_network: bool = False,
        latest_binary_packet_hex: Optional[str] = None,
    ) -> Dict[str, Any]:
        strapi_url = f"{self.strapi_base_url}/api/lore-quests?populate=*"
        medusa_url = f"{self.medusa_base_url}/store/products"
        woo_url = f"{self.woocommerce_base_url}/wp-json/wc/v3/products"

        live_strapi = self._probe_json(strapi_url) if probe_network else None
        live_medusa = self._probe_json(medusa_url) if probe_network else None
        live_woo = self._probe_json(woo_url) if probe_network else None

        lore_entries = (
            live_strapi.get("data", [])
            if isinstance(live_strapi, dict) and live_strapi.get("data")
            else [
                {
                    "id": "strapi-quest-01",
                    "slug": "citadel-well-of-souls",
                    "title": "Citadel Well of Souls Codex",
                    "cms_provider": CMSPlatform.STRAPI_V5.value,
                    "faction": "Kryštálový Kmeň",
                    "mechanical_bitboard_packet": latest_binary_packet_hex or "KMB1_READY",
                },
                {
                    "id": "strapi-quest-02",
                    "slug": "alchemical-ruins-convoy",
                    "title": "Sacred Alchemical Ruins Convoy",
                    "cms_provider": CMSPlatform.STRAPI_V5.value,
                    "faction": "V-Ray Optic Guild",
                    "mechanical_bitboard_packet": latest_binary_packet_hex or "KMB1_READY",
                },
            ]
        )

        commerce_items = [
            {
                "sku": "MEDUSA-KRYSTAL-LENS-01",
                "title": "Prismatic Raymarch Relic Lens",
                "platform": CMSPlatform.MEDUSA_OPEN_COMMERCE.value,
                "price_credits": 240,
                "live_synced": live_medusa is not None,
            },
            {
                "sku": "WOO-MORTAR-GYRO-02",
                "title": "Ballistic Mortar Stabilizer Sigil",
                "platform": CMSPlatform.WOOCOMMERCE_REST.value,
                "price_credits": 180,
                "live_synced": live_woo is not None,
            },
        ]

        return {
            "status": "OK",
            "vital_max_hp": VITAL_MAX_HP,
            "endpoints": {
                "strapi_cms": strapi_url,
                "medusa_commerce": medusa_url,
                "woocommerce_rest": woo_url,
            },
            "connection_modes": {
                "strapi_live": live_strapi is not None,
                "medusa_live": live_medusa is not None,
                "woocommerce_live": live_woo is not None,
                "fallback_cache_active": not (live_strapi and live_medusa and live_woo),
            },
            "lore_entries": lore_entries,
            "commerce_artifacts": commerce_items,
            "frontend_fetch_contract": {
                "hub_status_get": "/api/hybrid/status",
                "lore_step_post": "/api/hybrid/lore_step",
                "marias_card_post": "/api/hybrid/marias_turn",
                "grid_binary_post": "/api/hybrid/grid_move",
                "cms_sync_post": "/api/hybrid/cms_sync",
                "runtime_compile_post": "/api/hybrid/compile_runtime",
            },
        }


class TaxonomyRuntimeSecurityBuilder:
    def __init__(
        self,
        security_dir: Path = DEFAULT_SECURITY_DIR,
        runtime_dist_dir: Path = DEFAULT_RUNTIME_DIST,
    ) -> None:
        self.security_dir = security_dir
        self.runtime_dist_dir = runtime_dist_dir

    def generate_k8s_manifest(self) -> str:
        return """apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: krystal-hybrid-default-deny-all
  namespace: krystal-sovereign
spec:
  podSelector: {}
  policyTypes:
    - Ingress
    - Egress
---
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: krystal-hybrid-strict-allowlist
  namespace: krystal-sovereign
spec:
  podSelector:
    matchLabels:
      app: krystal-hybrid-engine
  policyTypes:
    - Ingress
    - Egress
  ingress:
    - from:
        - podSelector:
            matchLabels:
              role: krystal-ingress-gateway
      ports:
        - protocol: TCP
          port: 8080
  egress:
    - to:
        - podSelector:
            matchLabels:
              app: strapi-cms
      ports:
        - protocol: TCP
          port: 1337
    - to:
        - podSelector:
            matchLabels:
              app: medusa-commerce
      ports:
        - protocol: TCP
          port: 9000
    - to:
        - podSelector:
            matchLabels:
              app: ollama-llm
      ports:
        - protocol: TCP
          port: 11434
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: krystal-hybrid-engine
  namespace: krystal-sovereign
spec:
  replicas: 2
  selector:
    matchLabels:
      app: krystal-hybrid-engine
  template:
    metadata:
      labels:
        app: krystal-hybrid-engine
    spec:
      automountServiceAccountToken: false
      securityContext:
        runAsNonRoot: true
        runAsUser: 10001
        seccompProfile:
          type: RuntimeDefault
      containers:
        - name: krystal-hybrid-runtime
          image: krystal-stack/hybrid-taxonomy-runtime:latest
          ports:
            - containerPort: 8080
          env:
            - name: VITAL_MAX_HP
              value: "6"
          securityContext:
            allowPrivilegeEscalation: false
            readOnlyRootFilesystem: true
            capabilities:
              drop: ["ALL"]
"""

    def generate_fail2ban_and_firewall(self) -> Dict[str, str]:
        jail_conf = """# Krystal-Stack Absolute Firewall & Fail2Ban Hardening Jail
[DEFAULT]
bantime  = 86400
findtime = 300
maxretry = 3
banaction = nftables-multiport
backend  = auto

[krystal-hybrid-api]
enabled  = true
port     = 80,443,8080,1337,9000
filter   = krystal-hybrid-api
logpath  = /var/log/krystal/hybrid_access.log
maxretry = 3
bantime  = 172800

[krystal-strapi-auth]
enabled  = true
port     = 1337
filter   = krystal-strapi-auth
logpath  = /var/log/strapi/access.log
maxretry = 4
bantime  = 86400
"""
        nftables_conf = """#!/usr/sbin/nft -f
# Krystal-Stack Sovereign Default-Drop Firewall (nftables)
flush ruleset

table inet krystal_absolute_firewall {
    set fail2ban_blocklist {
        type ipv4_addr
        flags interval, timeout
    }

    chain input {
        type filter hook input priority 0; policy drop;

        iif "lo" accept
        ct state established,related accept
        ct state invalid drop
        ip saddr @fail2ban_blocklist drop

        tcp dport { 443, 8080 } ct state new limit rate 30/second burst 60 packets accept
    }

    chain forward {
        type filter hook forward priority 0; policy drop;
    }

    chain output {
        type filter hook output priority 0; policy accept;
    }
}
"""
        return {"jail_conf": jail_conf, "nftables_conf": nftables_conf}

    def compile_portable_runtime(self, extractor: VRayAlgebraicSignalExtractor) -> Dict[str, Any]:
        self.security_dir.mkdir(parents=True, exist_ok=True)
        self.runtime_dist_dir.mkdir(parents=True, exist_ok=True)
        shaders_out = self.runtime_dist_dir / "taxonomy_shaders"
        shaders_out.mkdir(parents=True, exist_ok=True)

        kernel_inv = extractor.inspect_taxonomy_kernels()
        copied_kernels: List[str] = []
        for name in kernel_inv:
            src_spv = extractor.taxonomy_dir / f"{name}.spv"
            if src_spv.exists():
                dst_spv = shaders_out / f"{name}.spv"
                dst_spv.write_bytes(src_spv.read_bytes())
                copied_kernels.append(name)

        k8s_path = self.security_dir / "k8s_zero_trust_firewall.yaml"
        jail_path = self.security_dir / "fail2ban_krystal_jail.conf"
        nft_path = self.security_dir / "krystal_absolute_firewall.nft"

        k8s_path.write_text(self.generate_k8s_manifest(), encoding="utf-8")
        fw_files = self.generate_fail2ban_and_firewall()
        jail_path.write_text(fw_files["jail_conf"], encoding="utf-8")
        nft_path.write_text(fw_files["nftables_conf"], encoding="utf-8")

        manifest = {
            "runtime_name": "krystal-hybrid-taxonomy-runtime",
            "compiled_at_epoch": round(time.time(), 3),
            "vital_max_hp": VITAL_MAX_HP,
            "copied_taxonomy_spv_kernels": copied_kernels,
            "taxonomy_inventory": kernel_inv,
            "security_artifacts": {
                "kubernetes_manifest": str(k8s_path),
                "fail2ban_jail": str(jail_path),
                "nftables_firewall": str(nft_path),
            },
        }
        manifest_path = self.runtime_dist_dir / "runtime_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        manifest["manifest_path"] = str(manifest_path)
        return manifest

    def validate_all(self, extractor: VRayAlgebraicSignalExtractor) -> Dict[str, Any]:
        kernel_inv = extractor.inspect_taxonomy_kernels()
        k8s_path = self.security_dir / "k8s_zero_trust_firewall.yaml"
        jail_path = self.security_dir / "fail2ban_krystal_jail.conf"
        nft_path = self.security_dir / "krystal_absolute_firewall.nft"

        k8s_text = k8s_path.read_text(encoding="utf-8") if k8s_path.exists() else ""
        jail_text = jail_path.read_text(encoding="utf-8") if jail_path.exists() else ""
        nft_text = nft_path.read_text(encoding="utf-8") if nft_path.exists() else ""

        checks = {
            "invariant_vital_max_hp_6": VITAL_MAX_HP == 6,
            "taxonomy_spv_all_present": all(v["spv_present"] for v in kernel_inv.values()),
            "k8s_default_deny_ingress_egress": (
                "krystal-hybrid-default-deny-all" in k8s_text
                and "readOnlyRootFilesystem: true" in k8s_text
                and 'drop: ["ALL"]' in k8s_text
            ),
            "fail2ban_jails_configured": (
                "[krystal-hybrid-api]" in jail_text
                and "[krystal-strapi-auth]" in jail_text
                and "banaction = nftables-multiport" in jail_text
            ),
            "nftables_policy_drop_enforced": (
                "policy drop;" in nft_text and "fail2ban_blocklist" in nft_text
            ),
        }
        return {
            "valid": all(checks.values()),
            "checks": checks,
            "vital_max_hp": VITAL_MAX_HP,
        }


class KrystalHybridPlatformFacade:
    def __init__(self) -> None:
        self.signal_extractor = VRayAlgebraicSignalExtractor()
        self.cognition_genesis = FantasyCognitionGenesisEngine()
        self.marias_engine = MariasAlgebraicCardTurnEngine()
        self.grid_bridge = GridMechanicalBitboardBridge()
        self.lore_engine = DynamicLoreChoiceMortarEngine()
        self.cms_connector = HeadlessCMSCommerceConnector()
        self.runtime_builder = TaxonomyRuntimeSecurityBuilder()

    def get_platform_capabilities(self) -> Dict[str, Any]:
        return {
            "framework": "Krystal-Stack Hybrid Aggregation Platform",
            "vital_max_hp": VITAL_MAX_HP,
            "engine_bridge": "Godot 4.x Forward+ / Vulkan + 8x8 KMB1 Binary Bitboard Grid Bridge",
            "render_signal_algebra": "V-Ray Light-Transport Decomposition (Direct, GI, GGX, Caustic, Mortar Arc)",
            "fantasy_cognition_and_cards": [
                "Per-New-Game Storyline Perception Seed & Learned Cognition Weights",
                "Mariáš Algebraic Card/Turn Engine (20/40 Hláška Marriage + Pending Action Queue + Semantic Click Animations)",
            ],
            "cms_and_commerce": ["Strapi v5 Headless CMS", "MedusaJS Open-Source Headless E-Commerce", "WooCommerce REST v3"],
            "security_and_runtime": [
                "Portable SPIR-V Taxonomy Runtime Bundle",
                "Kubernetes Default-Deny NetworkPolicy + Non-Root ReadOnly PodSecurity",
                "Fail2Ban Multi-Port Jail + nftables Default-Drop Sovereign Firewall",
            ],
            "cli_definable_flags": [
                "--validate-all",
                "--compile-runtime",
                "--new-game-seed <phrase>",
                "--step-quest <slug>",
                "--play-marias-card <RANK>",
                "--resolve-click",
                "--emulate-grid-move <x1,y1:x2,y2>",
                "--sync-cms",
                "--capabilities",
                "--json",
            ],
        }


GLOBAL_HYBRID_PLATFORM = KrystalHybridPlatformFacade()


def build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="krystal-hybrid-runtime",
        description="CLI Validator, Fantasy Lore-LLM, Mariáš Strategy, Grid Binary CMS Bridge & K8s/Fail2Ban Runtime Builder",
    )
    parser.add_argument("--capabilities", action="store_true", help="Print full platform framework reach & CLI flags")
    parser.add_argument("--compile-runtime", action="store_true", help="Compile taxonomy runtime bundle and K8s/Fail2Ban firewall configs")
    parser.add_argument("--validate-all", action="store_true", help="Validate SPIR-V taxonomy shaders, HP=6 invariant, K8s NetworkPolicy, Fail2Ban & nftables")
    parser.add_argument("--new-game-seed", type=str, default="", help="Generate fresh Fantasy LLM cognition & storyline perception at game start")
    parser.add_argument("--play-marias-card", type=str, default="", help="Stage Mariáš card rank (e.g. KING_KRAL, OBER_HORNIK, ACE_ESO) into pending actions")
    parser.add_argument("--suit", type=str, default="HEARTS_SRDCIA", help="Suit for --play-marias-card")
    parser.add_argument("--resolve-click", action="store_true", help="Resolve pending card click and emit semantic animation")
    parser.add_argument("--emulate-grid-move", type=str, default="", help="Emulate 8x8 mechanical grid move 'x1,y1:x2,y2' into KMB1 binary CMS packet")
    parser.add_argument("--sync-cms", action="store_true", help="Run backend/frontend fetch pipeline for Strapi + MedusaJS + WooCommerce")
    parser.add_argument("--probe-network", action="store_true", help="Attempt live HTTP probes for Strapi/Medusa/Woo/Ollama")
    parser.add_argument("--step-quest", type=str, default="", help="Execute a Lore-LLM + V-Ray Algebraic Signal + Mortar choice step")
    parser.add_argument("--choice", type=str, default="init_harmonic", help="Selected choice ID for --step-quest")
    parser.add_argument("--irradiance", type=float, default=1.35)
    parser.add_argument("--gi-bounce", type=float, default=0.62)
    parser.add_argument("--roughness", type=float, default=0.24)
    parser.add_argument("--caustic-flux", type=float, default=0.48)
    parser.add_argument("--ao", type=float, default=0.18)
    parser.add_argument("--mortar-angle", type=float, default=42.5)
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON output")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_cli_parser()
    args = parser.parse_args(argv)
    facade = GLOBAL_HYBRID_PLATFORM

    terms = RenderPipelineTerms(
        direct_irradiance=args.irradiance,
        gi_bounce_luminance=args.gi_bounce,
        specular_roughness=args.roughness,
        caustic_photon_flux=args.caustic_flux,
        ao_occlusion=args.ao,
        mortar_elevation_deg=args.mortar_angle,
    )
    sig = facade.signal_extractor.evaluate(terms)
    output: Dict[str, Any] = {"vital_max_hp": VITAL_MAX_HP}
    latest_bin_hex: Optional[str] = None

    if args.capabilities:
        output["capabilities"] = facade.get_platform_capabilities()

    if args.compile_runtime:
        output["compiled_runtime"] = facade.runtime_builder.compile_portable_runtime(facade.signal_extractor)

    if args.new_game_seed:
        output["storyline_genesis"] = facade.cognition_genesis.begin_new_storyline(args.new_game_seed, sig)

    if args.emulate_grid_move:
        left, right = args.emulate_grid_move.split(":")
        fx, fy = (int(v) for v in left.split(","))
        tx, ty = (int(v) for v in right.split(","))
        grid_res = facade.grid_bridge.emulate_mechanical_move((fx, fy), (tx, ty))
        output["grid_mechanical_binary"] = grid_res
        latest_bin_hex = grid_res["binary_frame_hex"]

    if args.play_marias_card:
        suit_enum = next((s for s in MariasSuit if s.value == args.suit or s.name == args.suit), MariasSuit.SRDCIA_HEARTS)
        output["marias_staged"] = facade.marias_engine.stage_card_play(
            suit=suit_enum,
            rank=args.play_marias_card,
            hand_ranks_in_suit=["KING_KRAL", "OBER_HORNIK", args.play_marias_card],
            target_objective="Citadel Ley-Grid Sanctum",
        )

    if args.resolve_click:
        output["marias_click_animation"] = facade.marias_engine.resolve_click_semantic_animation()

    if args.sync_cms:
        output["cms_commerce_sync"] = facade.cms_connector.fetch_unified_catalog(
            probe_network=args.probe_network,
            latest_binary_packet_hex=latest_bin_hex,
        )

    if args.step_quest:
        catalog = facade.cms_connector.fetch_unified_catalog(
            probe_network=args.probe_network,
            latest_binary_packet_hex=latest_bin_hex,
        )
        output["lore_step"] = facade.lore_engine.step_lore_quest(
            quest_slug=args.step_quest,
            selected_choice_id=args.choice,
            signal=sig,
            cms_artifacts=catalog["commerce_artifacts"],
            attempt_live_ollama=args.probe_network,
        )

    if args.validate_all:
        validation = facade.runtime_builder.validate_all(facade.signal_extractor)
        output["validation"] = validation
        if not validation["valid"]:
            print(json.dumps(output, indent=2))
            return 1

    if len(output) == 1:
        output["capabilities"] = facade.get_platform_capabilities()

    if args.json:
        print(json.dumps(output, indent=2))
    else:
        print(json.dumps(output, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
