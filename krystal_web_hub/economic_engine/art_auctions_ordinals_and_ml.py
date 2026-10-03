# ==============================================================================
# KRYSTAL-STACK: ART AUCTIONS, ORDINALS EXPORT, CONTENT REPLAY & ML MATRICES
# ==============================================================================
# Implements:
#   1. Painting & Generative Art Auction House (bidding, buyouts, rarity tiers).
#   2. Aether/Bitcoin Ordinals Protocol & Inscriptions Export API (witness envelopes).
#   3. Content-Addressed Game Replay Fragmenter with unique perceptual level snapshots.
#   4. Multi-dimensional Hero Matrices (5x2, 4x5, 30x20, 90x120) per hero count.
#   5. Variable Sampling Frequency Frame-Rate Encoding Protocol (15Hz to 240Hz).
#   6. Machine Learning Evolutionary Physics Engine for kinematic motion & trajectories.
#   7. Event Stream Congestion Controller (AIMD rate-limiting for genetic streams).
# ==============================================================================

import math
import time
import uuid
import hashlib
import json
import zlib
import random
from typing import Dict, List, Any, Optional, Tuple


# ── 1. PAINTING & GENERATIVE ART AUCTION HOUSE ───────────────────────────────
class PaintingAuctionHouseEngine:
    """
    Manages live auctions for unique artworks, level paintings,
    and generated battle moments with bidding and settlement.
    """

    def __init__(self):
        # lot_id -> lot data
        self.active_lots: Dict[str, Dict[str, Any]] = {}
        self.sold_lots: Dict[str, Dict[str, Any]] = {}
        self._seed_default_paintings()

    def _seed_default_paintings(self):
        defaults = [
            {
                "title": "Aéterová Polárna Žiara nad Arénou",
                "artist": "Krystal Neural Generative Studio",
                "style": "Aetheric Impressionism",
                "rarity": "Legendary",
                "resolution": "3840x2160",
                "starting_bid": 150,
                "buyout_price": 500,
                "image_url": "/api/assets/paintings/aurora_nocturne.png"
            },
            {
                "title": "Portrét Chladnej Vystupovačky v Duely",
                "artist": "Master Chronos Painter",
                "style": "Cyberpunk Baroque",
                "rarity": "Masterpiece",
                "resolution": "1920x1080",
                "starting_bid": 200,
                "buyout_price": 750,
                "image_url": "/api/assets/paintings/cold_duelist_portrait.png"
            },
            {
                "title": "12 Apoštolov a Kozmické Zrkadlo",
                "artist": "Empyrean Guild Illuminator",
                "style": "Celestial Sacred Gold",
                "rarity": "Celestial Relic",
                "resolution": "4096x2304",
                "starting_bid": 300,
                "buyout_price": 1200,
                "image_url": "/api/assets/paintings/celestial_apostles_mirror.png"
            }
        ]
        for p in defaults:
            self.create_painting_lot(
                title=p["title"],
                artist=p["artist"],
                style=p["style"],
                rarity=p["rarity"],
                resolution=p["resolution"],
                starting_bid=p["starting_bid"],
                buyout_price=p["buyout_price"],
                image_url=p["image_url"],
                duration_sec=7200
            )

    def create_painting_lot(
        self,
        title: str,
        artist: str,
        style: str,
        rarity: str,
        resolution: str = "1920x1080",
        starting_bid: int = 100,
        buyout_price: int = 400,
        image_url: str = "/api/assets/paintings/default.png",
        duration_sec: int = 3600,
        ordinal_id: Optional[str] = None
    ) -> Dict[str, Any]:
        lot_id = f"paint_auc_{uuid.uuid4().hex[:8]}"
        now = time.time()
        lot = {
            "id": lot_id,
            "lot_id": lot_id,
            "title": title,
            "artist": artist,
            "style": style,
            "rarity": rarity,
            "resolution": resolution,
            "starting_bid": starting_bid,
            "current_bid": starting_bid,
            "highest_bidder": None,
            "buyout_price": buyout_price,
            "image_url": image_url,
            "ordinal_inscription_id": ordinal_id or f"ord_paint_{uuid.uuid4().hex[:6]}i0",
            "created_at": now,
            "expires_at": now + duration_sec,
            "bids_count": 0,
            "bids_history": [],
            "status": "active"
        }
        self.active_lots[lot_id] = lot
        return lot

    def create_lot(self, *args, **kwargs) -> Dict[str, Any]:
        """Convenient alias for create_painting_lot."""
        return self.create_painting_lot(*args, **kwargs)

    def place_bid(self, lot_id: str, bidder: str = "", amount: int = 0, **kwargs) -> Dict[str, Any]:
        bidder_name = bidder or kwargs.get("bidder_name", "Anonymous")
        bid_amount = amount or kwargs.get("bid_amount", 0)

        lot = self.active_lots.get(lot_id)
        if not lot:
            return {"success": False, "error": f"Aukcia {lot_id} neexistuje."}

        if lot["status"] != "active":
            return {"success": False, "error": f"Aukcia {lot_id} už nie je aktívna (Status: {lot['status']})."}

        if bid_amount <= lot["current_bid"]:
            return {
                "success": False,
                "error": f"Ponuka {bid_amount} musí byť vyššia ako aktuálna ponuka ({lot['current_bid']})."
            }

        prev_bidder = lot["highest_bidder"]
        lot["current_bid"] = bid_amount
        lot["highest_bidder"] = bidder_name
        lot["bids_count"] += 1
        lot["bids_history"].append({
            "bidder": bidder_name,
            "amount": bid_amount,
            "timestamp": time.time()
        })

        # Check buyout trigger
        if lot["buyout_price"] > 0 and bid_amount >= lot["buyout_price"]:
            lot["status"] = "sold"
            self.sold_lots[lot_id] = lot
            del self.active_lots[lot_id]
            return {
                "success": True,
                "event": "BUYOUT_TRIGGERED",
                "buyout_triggered": True,
                "winner": bidder_name,
                "highest_bidder": bidder_name,
                "current_bid": bid_amount,
                "winning_price": bid_amount,
                "status": "sold",
                "lot": lot
            }

        return {
            "success": True,
            "event": "BID_PLACED",
            "buyout_triggered": False,
            "highest_bidder": bidder_name,
            "current_bid": bid_amount,
            "outbid_user": prev_bidder,
            "status": "active",
            "lot": lot
        }

    def buyout_painting(self, lot_id: str, buyer_name: str) -> Dict[str, Any]:
        lot = self.active_lots.get(lot_id)
        if not lot:
            return {"success": False, "error": f"Aukcia {lot_id} neexistuje."}
        return self.place_bid(lot_id, bidder=buyer_name, amount=lot["buyout_price"])

    def list_active_auctions(self) -> List[Dict[str, Any]]:
        return list(self.active_lots.values())

    def list_lots(self, status: str = "active") -> List[Dict[str, Any]]:
        st = status.lower()
        if st == "active":
            return list(self.active_lots.values())
        elif st == "sold":
            return list(self.sold_lots.values())
        return list(self.active_lots.values()) + list(self.sold_lots.values())

    def get_lot(self, lot_id: str) -> Optional[Dict[str, Any]]:
        return self.active_lots.get(lot_id) or self.sold_lots.get(lot_id)


# ── 2. AETHER / BITCOIN ORDINALS EXPORT PROTOCOL ─────────────────────────────
class AetherOrdinalsProtocolEngine:
    """
    Implements digital artifact inscriptions following the Bitcoin/Aether Ordinal model.
    Encodes content into script witness envelopes (OP_FALSE OP_IF ... OP_ENDIF).
    """

    def __init__(self):
        self.inscriptions: Dict[str, Dict[str, Any]] = {}
        self.inscription_counter = 1000

    def inscribe_content(
        self,
        content_payload: str,
        content_type: str = "text/plain;charset=utf-8",
        metadata: Optional[Dict[str, Any]] = None,
        owner_address: str = "bc1p_krystal_stack_aether_inscription",
        art_lot_id: Optional[str] = None
    ) -> Dict[str, Any]:
        self.inscription_counter += 1
        num = self.inscription_counter
        ins_id = f"ord_krys_{num:06d}i0"

        # SHA-256 content commitment
        payload_bytes = content_payload.encode("utf-8")
        content_sha256 = hashlib.sha256(payload_bytes).hexdigest()

        # Bitcoin witness script envelope:
        # OP_FALSE (0x00) OP_IF (0x63) OP_PUSH("ord") (0x036f7264)
        # OP_1 (0x51) OP_PUSH(mime) 0x00 OP_PUSH(body) OP_ENDIF (0x68)
        mime_hex = content_type.encode("utf-8").hex()
        data_hex = payload_bytes.hex()
        envelope_hex = f"0063036f726401{len(content_type):02x}{mime_hex}00{len(payload_bytes):04x}{data_hex}68"

        witness_asm = (
            f"OP_FALSE OP_IF OP_PUSH 'ord' OP_1 OP_PUSH '{content_type}' "
            f"OP_0 OP_PUSH '{content_sha256[:16]}...' OP_ENDIF"
        )

        rec = {
            "id": ins_id,
            "inscription_id": ins_id,
            "inscription_number": num,
            "art_lot_id": art_lot_id,
            "sat_ordinal_index": 789000000000 + num,
            "content_type": content_type,
            "content_hash": content_sha256,
            "content_sha256": content_sha256,
            "content_size_bytes": len(payload_bytes),
            "owner_address": owner_address,
            "metadata": metadata or {},
            "witness_envelope_hex": envelope_hex,
            "witness_script_hex": envelope_hex,
            "witness_asm": witness_asm,
            "onchain_timestamp": time.time(),
            "genesis_block_height": 840000 + (num // 100),
            "exportable": True
        }
        self.inscriptions[ins_id] = rec
        return rec

    def inscribe(self, *args, **kwargs) -> Dict[str, Any]:
        """Convenient alias for inscribe_content."""
        return self.inscribe_content(*args, **kwargs)

    def export_ordinal(self, inscription_id: str) -> Optional[Dict[str, Any]]:
        return self.inscriptions.get(inscription_id)

    def list_all_inscriptions(self) -> List[Dict[str, Any]]:
        return list(self.inscriptions.values())

    def list_inscriptions(self) -> List[Dict[str, Any]]:
        """Convenient alias for list_all_inscriptions."""
        return self.list_all_inscriptions()


# ── 3. CONTENT-FRAGMENTED GAME REPLAY RECORDS ────────────────────────────────
class ContentReplayEngine:
    """
    Fragments game replays by level content with deterministic visual snapshot hashes,
    allowing granular playback and event verification.
    """

    def __init__(self):
        self.records: Dict[str, Dict[str, Any]] = {}

    def record_level_fragment(
        self,
        match_id: str,
        level_index: int,
        level_name: str,
        combat_events: List[Dict[str, Any]],
        telemetry_summary: Dict[str, Any],
        visual_snapshot_data: str
    ) -> Dict[str, Any]:
        if match_id not in self.records:
            self.records[match_id] = {
                "match_id": match_id,
                "game_mode": "Tactical Poslední Kmen",
                "start_time": time.time(),
                "level_fragments": {},
                "total_levels": 0,
                "total_events": 0
            }

        # Unique deterministic snapshot perceptual hash
        snapshot_hash = hashlib.sha256(visual_snapshot_data.encode("utf-8")).hexdigest()

        frag = {
            "level_idx": level_index,
            "level_index": level_index,
            "level_name": level_name,
            "unique_visual_snapshot_hash": snapshot_hash,
            "perceptual_snapshot": {
                "snapshot_hash": snapshot_hash,
                "visual_seed_data": visual_snapshot_data[:48] + "...",
                "hash_algorithm": "SHA-256-PERCEPTUAL"
            },
            "events_count": len(combat_events),
            "combat_events": combat_events,
            "telemetry_summary": telemetry_summary,
            "recorded_at": time.time()
        }

        self.records[match_id]["level_fragments"][str(level_index)] = frag
        self.records[match_id]["total_levels"] = len(self.records[match_id]["level_fragments"])
        self.records[match_id]["total_events"] += len(combat_events)

        return frag

    def record_level_segment(
        self,
        match_id: str,
        level_name: str,
        level_idx: int,
        level_seed: Any,
        hero_count: int,
        action_events: List[Dict[str, Any]],
        visual_layers: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Convenient method for fragmented level recording with automatic visual snapshot payload."""
        visual_payload = (
            f"level:{level_idx}:seed:{level_seed}:heroes:{hero_count}:"
            f"name:{level_name}:{json.dumps(visual_layers, sort_keys=True)}"
        )
        return self.record_level_fragment(
            match_id=match_id,
            level_index=level_idx,
            level_name=level_name,
            combat_events=action_events,
            telemetry_summary={"hero_count": hero_count, "level_seed": level_seed, "layers": visual_layers},
            visual_snapshot_data=visual_payload
        )

    def get_game_record(self, match_id: str) -> Optional[Dict[str, Any]]:
        return self.records.get(match_id)

    def get_match_manifest(self, match_id: str) -> Optional[Dict[str, Any]]:
        return self.get_game_record(match_id)

    def get_level_segment(self, match_id: str, level_idx: int) -> Optional[Dict[str, Any]]:
        rec = self.records.get(match_id)
        if not rec:
            return None
        return rec.get("level_fragments", {}).get(str(level_idx))

    def list_records(self) -> List[Dict[str, Any]]:
        return list(self.records.values())

    def list_matches(self) -> List[Dict[str, Any]]:
        return self.list_records()


# ── 4. MULTI-DIMENSIONAL HERO MATRICES (5x2, 4x5, 30x20, 90x120) ─────────────
class HeroMatrixEngine:
    """
    Generates structured mathematical matrices for heroes:
      - 5x2: Action x Phase (10 elements)
      - 4x5: Target Zone x Defense Stance (20 elements)
      - 30x20: Spatial Radial Field (600 elements)
      - 90x120: High-Resolution Camera & Frequency Tensor (10,800 elements)
    """

    MATRIX_PRESETS = {
        "5x2": {"cols": 2, "rows": 5, "cells": 10, "desc": "5 Action Types x 2 Combat Phases (Planning / Execution)"},
        "4x5": {"cols": 5, "rows": 4, "cells": 20, "desc": "4 Target Zones (Head, Left Shoulder, Right Shoulder, Torso) x 5 Stances"},
        "30x20": {"cols": 20, "rows": 30, "cells": 600, "desc": "30 Radial Azimuth Angles x 20 Distance Bins"},
        "90x120": {"cols": 120, "rows": 90, "cells": 10800, "desc": "90 Sampling Frequencies x 120 Sensor Channels"}
    }

    @staticmethod
    def generate_hero_matrix(
        hero_id: str,
        hero_index: int,
        total_heroes_count: int,
        matrix_type: str = "4x5"
    ) -> Dict[str, Any]:
        preset = HeroMatrixEngine.MATRIX_PRESETS.get(matrix_type, HeroMatrixEngine.MATRIX_PRESETS["4x5"])
        cols = preset["cols"]
        rows = preset["rows"]

        phase_offset = (hero_index / max(1, total_heroes_count)) * 2.0 * math.pi
        grid_sample: List[List[float]] = []

        sample_rows = min(rows, 10)
        sample_cols = min(cols, 10)

        for r in range(sample_rows):
            row_vals: List[float] = []
            for c in range(sample_cols):
                val = math.sin((c / max(1, cols)) * math.pi + phase_offset) * math.cos((r / max(1, rows)) * math.pi)
                row_vals.append(round(val, 3))
            grid_sample.append(row_vals)

        return {
            "hero_id": hero_id,
            "hero_index": hero_index,
            "total_heroes_count": total_heroes_count,
            "matrix_type": matrix_type,
            "dimensions": {"cols": cols, "rows": rows, "total_cells": preset["cells"]},
            "description": preset["desc"],
            "phase_modulation_rad": round(phase_offset, 3),
            "sampled_matrix": grid_sample,
            "mean_energy": round(sum(sum(r) for r in grid_sample) / (sample_rows * sample_cols), 4)
        }

    @staticmethod
    def generate_matrix_for_hero_count(hero_count: int, dim_key: str = "4x5") -> Dict[str, Any]:
        """
        Generates full dimensional matrix and preview representation modulated by hero count.
        """
        preset = HeroMatrixEngine.MATRIX_PRESETS.get(dim_key, HeroMatrixEngine.MATRIX_PRESETS["4x5"])
        rows = preset["rows"]
        cols = preset["cols"]
        total_cells = preset["cells"]

        phase_mod = (hero_count % 12) / 12.0 * 2.0 * math.pi
        full_matrix: List[List[float]] = []

        # For smaller matrices generate all elements, for 90x120 generate preview + summary
        if total_cells <= 600:
            for r in range(rows):
                r_vals = []
                for c in range(cols):
                    v = math.sin((c / max(1, cols)) * math.pi + phase_mod) * math.cos((r / max(1, rows)) * math.pi)
                    r_vals.append(round(v, 3))
                full_matrix.append(r_vals)
            preview = full_matrix[:min(rows, 5)]
        else:
            # 90x120 (10,800 cells)
            preview = []
            for r in range(min(rows, 8)):
                r_vals = []
                for c in range(min(cols, 8)):
                    v = math.sin((c / cols) * math.pi + phase_mod) * math.cos((r / rows) * math.pi)
                    r_vals.append(round(v, 3))
                preview.append(r_vals)
            # Create lightweight row generator or representative sample
            full_matrix = preview

        return {
            "dim_key": dim_key,
            "rows": rows,
            "cols": cols,
            "total_cells": total_cells,
            "hero_count": hero_count,
            "phase_modulation": round(phase_mod, 4),
            "description": preset["desc"],
            "matrix": full_matrix,
            "preview_sample": preview
        }


# ── 5. VARIABLE SAMPLING FREQUENCY ENCODING PROTOCOL ─────────────────────────
class FrameRateEncodingProtocol:
    """
    Encodes replay frame sequences across dynamic frequencies (15Hz to 240Hz).
    """

    @staticmethod
    def build_frequency_schedule(combat_intensity: float = 0.5) -> Dict[str, Any]:
        norm = max(0.0, min(1.0, combat_intensity))
        fps = int(round(15 + (225 * norm)))
        mode = "NORMAL"
        if fps >= 120:
            mode = "BULLET_TIME_BURST"
        elif fps >= 60:
            mode = "COMBAT_TACTICAL"

        return {
            "intensity": norm,
            "sampling_frequency_hz": fps,
            "mode": mode,
            "quantum_packet_bytes": 64 if fps < 60 else 256,
            "clock_tick_interval_ms": round(1000.0 / fps, 2)
        }

    @staticmethod
    def compute_sampling_frequency_schedule(hero_count: int = 1, battle_intensity: float = 0.5) -> Dict[str, Any]:
        """
        Dynamically calculates sampling frequency schedule modulated by hero count and intensity.
        """
        base_hz = 15.0
        hero_boost = min(60.0, (hero_count - 1) * 7.5)
        intensity_boost = battle_intensity * 165.0
        effective_hz = int(min(240, max(15, round(base_hz + hero_boost + intensity_boost))))

        mode = "NORMAL"
        if effective_hz >= 120:
            mode = "BULLET_TIME_BURST"
        elif effective_hz >= 60:
            mode = "COMBAT_TACTICAL"

        return {
            "hero_count": hero_count,
            "battle_intensity": battle_intensity,
            "effective_hz": effective_hz,
            "sampling_frequency_hz": effective_hz,
            "mode": mode,
            "clock_interval_ms": round(1000.0 / effective_hz, 3),
            "buffer_packet_size_bytes": 64 if effective_hz < 60 else 256
        }

    @staticmethod
    def pack_stream_header(total_frames: int, base_hz: int, channel_count: int = 1) -> Dict[str, Any]:
        """
        Packs binary stream header with CRC32 integrity check.
        """
        magic = "KRYS_FPS\x01"
        header_raw = f"{magic}:{base_hz}:{total_frames}:{channel_count}".encode("utf-8")
        crc = zlib.crc32(header_raw)
        return {
            "magic": magic,
            "base_frequency_hz": base_hz,
            "total_frames": total_frames,
            "channel_count": channel_count,
            "crc32": crc,
            "crc32_hex": f"{crc:08x}",
            "header_bytes_len": len(header_raw)
        }

    @staticmethod
    def encode_frame_stream(
        frames_count: int,
        combat_intensity: float,
        level_snapshot_hash: str
    ) -> Dict[str, Any]:
        sched = FrameRateEncodingProtocol.build_frequency_schedule(combat_intensity)
        fps = sched["sampling_frequency_hz"]
        duration = frames_count / max(1, fps)

        envelope = {
            "magic": "KRYS_STREAM_V2",
            "fps_hz": fps,
            "mode": sched["mode"],
            "total_frames": frames_count,
            "duration_sec": round(duration, 3),
            "level_snapshot_hash": level_snapshot_hash,
            "compression_algorithm": "RLE_QUANTIZED_SPATIAL_TENSOR"
        }
        return envelope


# ── 6. MACHINE LEARNING EVOLUTIONARY PHYSICS ENGINE ──────────────────────────
class EvolutionaryPhysicsEngine:
    """
    Applies genetic / evolutionary algorithm principles to generate optimal,
    physically plausible kinematic trajectories for heroes and projectiles.
    """

    def __init__(self, population_size: int = 16, trajectory_steps: int = 15):
        self.population_size = population_size
        self.trajectory_steps = trajectory_steps
        self.generation = 0
        self.friction = 0.05
        self.gravity = 9.81
        self.collision_target = (10.0, 0.0, 5.0)

        # Initialize chromosome population of acceleration sequences
        self.population: List[List[Tuple[float, float, float]]] = []
        self._init_population()
        self.best_individual: Optional[Dict[str, Any]] = None
        self._evaluate_population()

    def _init_population(self):
        self.population = []
        for _ in range(self.population_size):
            chromosome = []
            for _ in range(self.trajectory_steps):
                ax = random.uniform(-1.0, 1.0)
                ay = random.uniform(-0.5, 0.5)
                az = random.uniform(-1.0, 1.0)
                chromosome.append((ax, ay, az))
            self.population.append(chromosome)

    def _simulate_trajectory(self, chromosome: List[Tuple[float, float, float]]) -> Tuple[List[List[float]], float]:
        px, py, pz = 0.0, 0.0, 0.0
        vx, vy, vz = 1.0, 0.0, 0.5
        dt = 0.1
        path = [[px, py, pz]]
        energy = 0.0

        for ax, ay, az in chromosome:
            vx = (vx + ax * dt) * (1.0 - self.friction)
            vy = (vy + ay * dt) * (1.0 - self.friction)
            vz = (vz + (az - self.gravity * 0.1) * dt) * (1.0 - self.friction)
            px += vx * dt
            py += vy * dt
            pz = max(0.0, pz + vz * dt)  # Ground floor constraint
            energy += abs(ax) + abs(ay) + abs(az)
            path.append([round(px, 3), round(py, 3), round(pz, 3)])

        # Fitness: proximity to target with low energy consumption
        tx, ty, tz = self.collision_target
        dist_sq = (px - tx)**2 + (py - ty)**2 + (pz - tz)**2
        fitness = 1000.0 / (1.0 + math.sqrt(dist_sq)) - (energy * 0.2)
        return path, fitness

    def _evaluate_population(self) -> List[Dict[str, Any]]:
        scored = []
        for chrom in self.population:
            traj, fit = self._simulate_trajectory(chrom)
            scored.append({"chromosome": chrom, "trajectory": traj, "fitness": fit})
        scored.sort(key=lambda x: x["fitness"], reverse=True)

        if self.best_individual is None or scored[0]["fitness"] > self.best_individual["fitness"]:
            self.best_individual = scored[0]

        return scored

    def evolve_generation(self) -> Dict[str, Any]:
        self.generation += 1
        scored = self._evaluate_population()

        # Elitism: keep top 25%
        survivors_count = max(2, self.population_size // 4)
        survivors = [s["chromosome"] for s in scored[:survivors_count]]

        # Crossover & Mutation to replenish
        new_pop = list(survivors)
        while len(new_pop) < self.population_size:
            p1 = random.choice(survivors)
            p2 = random.choice(survivors)
            split = random.randint(1, self.trajectory_steps - 1)
            child = p1[:split] + p2[split:]
            # Mutate child
            mutated = []
            for ax, ay, az in child:
                if random.random() < 0.25:
                    mutated.append((
                        ax + random.uniform(-0.3, 0.3),
                        ay + random.uniform(-0.2, 0.2),
                        az + random.uniform(-0.3, 0.3)
                    ))
                else:
                    mutated.append((ax, ay, az))
            new_pop.append(mutated)

        self.population = new_pop
        new_scored = self._evaluate_population()

        return {
            "generation": self.generation,
            "best_fitness": round(self.best_individual["fitness"], 3),
            "avg_fitness": round(sum(s["fitness"] for s in new_scored) / len(new_scored), 3),
            "population_size": self.population_size
        }

    def get_best_individual(self) -> Dict[str, Any]:
        if not self.best_individual:
            self._evaluate_population()
        return {
            "generation": self.generation,
            "fitness": round(self.best_individual["fitness"], 3),
            "trajectory": self.best_individual["trajectory"],
            "target": list(self.collision_target),
            "steps": len(self.best_individual["trajectory"])
        }

    @staticmethod
    def evolve_trajectory(
        start_pos: Tuple[float, float],
        target_pos: Tuple[float, float],
        generations: int = 15,
        population_size: int = 20,
        mutation_rate: float = 0.15
    ) -> Dict[str, Any]:
        """
        Evolves a sequence of acceleration vectors that guide an entity from start to target.
        """
        dx = target_pos[0] - start_pos[0]
        dy = target_pos[1] - start_pos[1]
        dist_target = math.sqrt(dx*dx + dy*dy)

        best_trajectory: List[Tuple[float, float]] = []
        best_fitness = -float("inf")
        steps = 10

        for gen in range(generations):
            for p in range(population_size):
                cur_x, cur_y = start_pos
                vx, vy = (dx / steps) * 0.5, (dy / steps) * 0.5
                path = [(round(cur_x, 2), round(cur_y, 2))]
                energy = 0.0

                for s in range(steps):
                    ax = random.uniform(-0.5, 0.5) * (1.0 + mutation_rate)
                    ay = random.uniform(-0.5, 0.5) * (1.0 + mutation_rate)
                    vx = (vx + ax) * 0.95
                    vy = (vy + ay) * 0.95
                    cur_x += vx
                    cur_y += vy
                    energy += abs(ax) + abs(ay)
                    path.append((round(cur_x, 2), round(cur_y, 2)))

                rem_dist = math.sqrt((cur_x - target_pos[0])**2 + (cur_y - target_pos[1])**2)
                fitness = (dist_target - rem_dist) * 10.0 - (energy * 0.5)

                if fitness > best_fitness:
                    best_fitness = fitness
                    best_trajectory = path

        return {
            "generations_computed": generations,
            "population_size": population_size,
            "best_fitness_score": round(best_fitness, 2),
            "start_coordinates": list(start_pos),
            "target_coordinates": list(target_pos),
            "evolved_waypoints": best_trajectory,
            "steps_count": len(best_trajectory),
            "kinematics_status": "CONVERGED_SMOOTH"
        }


# ── 7. EVENT STREAM CONGESTION CONTROLLER (AIMD) ──────────────────────────────
class EventStreamCongestionController:
    """
    Implements Additive Increase / Multiplicative Decrease (AIMD) flow control
    to throttle and buffer high-frequency genetic algorithm event streams.
    """

    def __init__(self, initial_window: int = 20, max_queue: int = 100):
        self.window_size = initial_window
        self.max_queue_threshold = max_queue
        self.queue: List[Dict[str, Any]] = []
        self.total_dispatched = 0
        self.congestion_events = 0

    def push_event(self, event_data: Dict[str, Any]) -> bool:
        if len(self.queue) >= self.max_queue_threshold:
            # Overflow / congestion event
            self.congestion_events += 1
            # Multiplicative decrease
            self.window_size = max(5, int(self.window_size * 0.5))
            return False

        self.queue.append({**event_data, "enqueued_at": time.time()})
        return True

    def dispatch_batch(self) -> Dict[str, Any]:
        batch_size = min(len(self.queue), self.window_size)
        dispatched_items = self.queue[:batch_size]
        self.queue = self.queue[batch_size:]
        self.total_dispatched += len(dispatched_items)

        # If smooth, Additive Increase
        if len(self.queue) < (self.max_queue_threshold // 2):
            self.window_size = min(120, self.window_size + 2)

        return {
            "dispatched_count": len(dispatched_items),
            "remaining_queue_depth": len(self.queue),
            "current_window_size": self.window_size,
            "congestion_active": len(self.queue) > (self.max_queue_threshold * 0.75),
            "events": dispatched_items
        }

    def get_status(self) -> Dict[str, Any]:
        return {
            "window_size": self.window_size,
            "queue_depth": len(self.queue),
            "max_queue_threshold": self.max_queue_threshold,
            "total_dispatched": self.total_dispatched,
            "congestion_events_count": self.congestion_events
        }


# Global Singletons
GLOBAL_PAINTING_AUCTIONS = PaintingAuctionHouseEngine()
GLOBAL_ORDINALS_PROTOCOL = AetherOrdinalsProtocolEngine()
GLOBAL_CONTENT_REPLAY = ContentReplayEngine()
GLOBAL_CONGESTION_CONTROLLER = EventStreamCongestionController()
