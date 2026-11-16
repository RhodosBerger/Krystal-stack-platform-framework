"""Hexagon combat presets, combinatorial matchup equations, maximum combat bandwidth
solver, and action-coupled procedural world effects + reactive AI chains.

Constitutional invariants:
  - VITAL_MAX_HP == 6 (all HP mutations clamped to [0, 6])
  - Render(S) != S; invalid primary action raises ContractViolation with Delta S == 0
  - Every primary action, world effect, and AI reaction emits a 128 B SHA-256 frame receipt.
"""

from __future__ import annotations

import copy
import hashlib
import itertools
import json
import math
import statistics
import time
from typing import Any, Dict, List, Optional, Tuple

from krystal_kernel.constants import (
    CACHE_LINE_BYTES,
    FRAME_SIZE_BYTES,
    UMA_SATURATION_THRESHOLD,
    VITAL_MAX_HP,
    ContractViolation,
)

from .adaptive_cross_genre_sync import HORIZON_MAX, HORIZON_MIN, TICK_MS
from .clash_battle_royale import (
    BOARD_RADIUS,
    CARDS,
    DIRS,
    FACTIONS,
    ROUND_BUDGET_FRAMES,
    ClashGame,
    _pack_frame,
    board_cells,
    hex_distance,
)
from .x86_platform_findings import measure_x05_128b_binary_frame_vs_json


# ---------------------------------------------------------------------------
# 1. Exact Hexagonal Geometry, Combinatorics & Full-Parameter Bandwidth Equations
# ---------------------------------------------------------------------------

def hex_cell_count(radius: int) -> int:
    """Centered hexagonal number: N_hex(R) = 1 + 3 * R * (R + 1)."""
    if radius < 0:
        raise ContractViolation("radius must be >= 0", violation_code="ERR_HEX_RADIUS")
    return 1 + 3 * radius * (radius + 1)


def hex_undirected_edge_count(radius: int) -> int:
    """Exact undirected adjacency edges on a radius-R hex board: E_adj(R) = 3 * R * (3 * R + 1)."""
    if radius < 0:
        raise ContractViolation("radius must be >= 0", violation_code="ERR_HEX_RADIUS")
    return 3 * radius * (3 * radius + 1)


def verify_hex_combinatorial_identities(max_radius: int = 4) -> Dict[str, Any]:
    """Prove closed-form N_hex(R) and E_adj(R) against explicit board enumeration."""
    rows = []
    all_match = True
    for r in range(max_radius + 1):
        cells = set(board_cells(r))
        enum_edges = 0
        for q, s in cells:
            for dq, ds in DIRS:
                if (q + dq, s + ds) in cells:
                    enum_edges += 1
        undirected_enum = enum_edges // 2
        n_formula = hex_cell_count(r)
        e_formula = hex_undirected_edge_count(r)
        ok = (len(cells) == n_formula) and (undirected_enum == e_formula)
        all_match = all_match and ok
        rows.append(
            {
                "radius": r,
                "cells_enumerated": len(cells),
                "cells_formula": n_formula,
                "undirected_edges_enumerated": undirected_enum,
                "undirected_edges_formula": e_formula,
                "directed_attack_arcs": enum_edges,
                "exact_match": ok,
            }
        )
    return {"verdict": "VERIFIED_WITH_SCOPE" if all_match else "FALSIFIED", "radii_checked": rows}


def measure_clash_engine_action_throughput(iterations: int = 200) -> Dict[str, Any]:
    """Measure real ClashGame.apply + 128 B SHA-256 commit throughput on this host CPU."""
    samples_actions_per_sec: List[float] = []
    for run_idx in range(10):
        t0 = time.perf_counter()
        executed = 0
        for k in range(max(10, iterations // 4)):
            g = ClashGame("SYNC", seed=6 + run_idx + k)
            g.apply({"type": "move", "unit": "C01", "to": [3, -1]})
            g.apply({"type": "card", "card": "HORIZON_SHIELD", "target": "C01"})
            g.apply({"type": "move", "unit": "C02", "to": [1, 0]})
            g.apply({"type": "end_turn"})
            executed += int(len(g.state["log"]))
        dt = max(time.perf_counter() - t0, 1e-9)
        samples_actions_per_sec.append(executed / dt)

    med = statistics.median(samples_actions_per_sec)
    return {
        "engine_commits_per_sec_median": round(med, 1),
        "engine_commits_per_sec_p95": round(sorted(samples_actions_per_sec)[int(0.95 * len(samples_actions_per_sec))], 1),
        "n_batches": len(samples_actions_per_sec),
        "evidence": "LOCAL_MEASURED",
    }


def compute_full_combat_bandwidth_solution(
    board_radius: int = BOARD_RADIUS,
    factions_count: int = 4,
    units_per_faction: int = 2,
    round_budget_frames: int = ROUND_BUDGET_FRAMES,
    uma_utilization: float = 0.35,
    tick_hz: float = 60.0,
    subscribed_clients: int = 4,
    one_way_rtt_ms: float = 25.0,
    jitter_upper_sigma_ms: float = 5.0,
    misprediction_prob: float = 0.15,
    frame_bytes: int = FRAME_SIZE_BYTES,
    measured_pack_fps: Optional[float] = None,
    measured_engine_aps: Optional[float] = None,
) -> Dict[str, Any]:
    """Evaluate the complete closed-form hex combat & bandwidth equation system across all parameters."""
    if board_radius < 1 or factions_count < 1 or units_per_faction < 1 or tick_hz <= 0 or frame_bytes != 128:
        raise ContractViolation("Invalid parameter in combat bandwidth equation", violation_code="ERR_BW_PARAMS")
    if not (0.0 <= uma_utilization <= 1.0) or not (0.0 <= misprediction_prob <= 1.0):
        raise ContractViolation("Probability / UMA ratio outside [0, 1]", violation_code="ERR_BW_RATIO")

    n_cells = hex_cell_count(board_radius)
    e_undir = hex_undirected_edge_count(board_radius)
    e_dir = 2 * e_undir
    total_units = factions_count * units_per_faction
    if total_units > n_cells:
        raise ContractViolation("total_units exceeds hex cell count", violation_code="ERR_BW_OCCUPANCY")

    # Effective per-faction 128 B frame budget under Sovereign P3 (UMA < 0.78)
    b_eff = round_budget_frames if uma_utilization < UMA_SATURATION_THRESHOLD else (round_budget_frames // 2)
    min_card_cost = min(c["cost"] for c in CARDS.values())  # 4 frames (HORIZON_SHIELD)

    # Max discrete state-writing events per faction per round:
    # Up to U_f unit actions (cost 1 each) + floor((B_eff - U_f) / min_card_cost) cards
    unit_acts_per_faction = min(units_per_faction, b_eff)
    max_cards_per_faction = max(0, (b_eff - unit_acts_per_faction) // min_card_cost)
    max_events_per_round = factions_count * (unit_acts_per_faction + max_cards_per_faction) + 1  # +1 END_TURN
    max_frames_per_round = factions_count * b_eff + 1

    # Rollback horizon H(RTT, sigma_+)
    raw_h = math.ceil((one_way_rtt_ms + 2.0 * jitter_upper_sigma_ms) / TICK_MS)
    horizon_h = max(HORIZON_MIN, min(HORIZON_MAX, raw_h))

    # Wire fanout + rollback replay factor: 1 ingress + M_clients egress + p_mis * H resimulated frames
    amplification = 1.0 + float(subscribed_clients) + misprediction_prob * float(horizon_h)
    wire_bytes_per_sec = tick_hz * max_frames_per_round * frame_bytes * amplification
    wire_mbps = (wire_bytes_per_sec * 8.0) / 1_000_000.0
    cache_lines_per_sec = wire_bytes_per_sec / CACHE_LINE_BYTES

    pack_fps = measured_pack_fps
    if pack_fps is None:
        pack_fps = measure_x05_128b_binary_frame_vs_json(runs=10, batch=300)["binary_frames_per_sec_median"]
    eng_aps = measured_engine_aps
    if eng_aps is None:
        eng_aps = measure_clash_engine_action_throughput(iterations=80)["engine_commits_per_sec_median"]

    bottleneck_commits_per_sec = min(float(pack_fps), float(eng_aps))
    effective_commits_per_arena_sec = tick_hz * max_events_per_round * (1.0 + misprediction_prob * horizon_h)
    max_concurrent_arenas = max(1, int(bottleneck_commits_per_sec // effective_commits_per_arena_sec))
    silicon_ceiling_mb_per_sec = (bottleneck_commits_per_sec * frame_bytes) / (1024 ** 2)

    # Pairwise faction combat combinations C(F, 2) and unit placement combinations P(N_hex, U)
    faction_pairings = math.comb(factions_count, 2)
    placement_ways = math.perm(n_cells, total_units)

    equations_symbolic = {
        "E1_hex_cells": "N_hex(R) = 1 + 3*R*(R + 1)",
        "E2_adjacency_arcs": "E_dir(R) = 2 * E_adj(R) = 6*R*(3*R + 1)",
        "E3_uma_frame_budget": "B_f(uma) = B_0 if uma < 0.78 else floor(B_0 / 2)",
        "E4_round_frames_cap": "F_round = F * B_f(uma) + 1",
        "E5_round_events_cap": "A_round = F * (min(U_f, B_f) + floor((B_f - min(U_f, B_f)) / c_card_min)) + 1",
        "E6_rollback_horizon": "H = clamp(ceil((RTT_ms + 2*sigma_upper_ms) / 16.667), 1, 8)",
        "E7_wire_bandwidth_Bps": "BW_wire = f_tick * F_round * L_frame * (1 + M_clients + p_mis * H)",
        "E8_silicon_arena_cap": "Arenas_max = floor(min(Phi_engine, Phi_x86_pack) / (f_tick * A_round * (1 + p_mis * H)))",
    }

    return {
        "vital_max_hp": VITAL_MAX_HP,
        "parameters": {
            "R_board_radius": board_radius,
            "F_factions": factions_count,
            "U_f_units_per_faction": units_per_faction,
            "U_total_units": total_units,
            "B_0_base_budget_frames": round_budget_frames,
            "B_eff_uma_budget_frames": b_eff,
            "c_card_min_frames": min_card_cost,
            "L_frame_bytes": frame_bytes,
            "uma_utilization": uma_utilization,
            "f_tick_hz": tick_hz,
            "M_subscribed_clients": subscribed_clients,
            "one_way_rtt_ms": one_way_rtt_ms,
            "jitter_upper_sigma_ms": jitter_upper_sigma_ms,
            "H_rollback_horizon_ticks": horizon_h,
            "p_misprediction": misprediction_prob,
        },
        "equations": equations_symbolic,
        "combinatorics": {
            "hex_cells": n_cells,
            "undirected_hex_edges": e_undir,
            "directed_attack_arcs": e_dir,
            "faction_duel_pairings": faction_pairings,
            "ordered_unit_placements_perm": placement_ways,
        },
        "bandwidth_solution": {
            "max_frames_per_round": max_frames_per_round,
            "max_discrete_events_per_round": max_events_per_round,
            "single_arena_wire_bytes_per_sec": round(wire_bytes_per_sec, 1),
            "single_arena_wire_mbps": round(wire_mbps, 4),
            "single_arena_x86_cache_lines_per_sec": round(cache_lines_per_sec, 1),
            "measured_x86_pack_frames_per_sec": round(float(pack_fps), 1),
            "measured_engine_commits_per_sec": round(float(eng_aps), 1),
            "host_max_concurrent_60hz_arenas": max_concurrent_arenas,
            "host_silicon_max_combat_bw_mb_per_sec": round(silicon_ceiling_mb_per_sec, 3),
        },
        "evidence": "DERIVED equations calibrated by LOCAL_MEASURED Phi_engine and Phi_x86_pack",
    }


# ---------------------------------------------------------------------------
# 2. Six Hex Combat Presets & Matchup Combinations
# ---------------------------------------------------------------------------

HEX_COMBAT_PRESETS: Dict[str, Dict[str, Any]] = {
    "PRESET_01_CORNER_SIEGE_BURST": {
        "player_faction": "COMPUTE",
        "round": 1,
        "zone_radius": 3,
        "uma": 0.28,
        "rtt_ms": 12.0,
        "positions": {"C03": [0, 1], "C04": [1, 1], "C01": [1, 0], "C02": [2, 0]},
        "hp_overrides": {},
        "description": "Rýchly stret COMPUTE vs. SYNC pri nízkom UMA (povolená karta GPU_ALLOCATE +1 ATK).",
    },
    "PRESET_02_CENTER_HEX_CRUCIBLE": {
        "player_faction": "SYNC",
        "round": 4,
        "zone_radius": 2,
        "uma": 0.42,
        "rtt_ms": 24.0,
        "positions": {
            "C01": [0, 0], "C02": [1, 0],
            "C03": [0, 1], "C04": [-1, 1],
            "C05": [-1, 0], "C06": [0, -1],
            "C07": [1, -1], "C08": [2, -1],
        },
        "hp_overrides": {},
        "description": "Maximálna hustota susediacich súbojov v strede hexagónovej siete (polomer <= 1).",
    },
    "PRESET_03_SHRINKING_ZONE_COLLAPSE": {
        "player_faction": "VERIFY",
        "round": 7,
        "zone_radius": 1,
        "uma": 0.35,
        "rtt_ms": 30.0,
        "positions": {"C05": [-2, 0], "C06": [-1, 0], "C01": [0, 0], "C03": [0, 1]},
        "hp_overrides": {"C05": 4, "C01": 5},
        "description": "Zúžená bezpečná zóna R_z=1; jednotka C05 mimo zóny musí kombinovať opravu CONFLICT a ústup do stredu.",
    },
    "PRESET_04_DUAL_CONFLICT_ATTRITION": {
        "player_faction": "OPS",
        "round": 5,
        "zone_radius": 2,
        "uma": 0.33,
        "rtt_ms": 20.0,
        "positions": {"C07": [0, -1], "C08": [1, -1], "C05": [-1, 0], "C06": [-1, 1]},
        "hp_overrides": {"C08": 3, "C05": 3},
        "description": "Opotrebovávací súboj dvoch frakcií s aktívnymi CONFLICT jednotkami (C08 vs. C05) a kartou FREIVALDS_AUDIT.",
    },
    "PRESET_05_UMA_SHADOW_LOCKSTEP": {
        "player_faction": "SYNC",
        "round": 3,
        "zone_radius": 3,
        "uma": 0.84,
        "rtt_ms": 18.0,
        "positions": {"C01": [1, 0], "C02": [1, -1], "C03": [0, 1], "C04": [2, -1]},
        "hp_overrides": {},
        "description": "Preťažená zbernica UMA (0.84 >= 0.78): GPU_ALLOCATE zablokovaná, rozpočet krátený na 8 rámcov/frakciu.",
    },
    "PRESET_06_WAN_JITTER_TOURNAMENT": {
        "player_faction": "COMPUTE",
        "round": 2,
        "zone_radius": 3,
        "uma": 0.50,
        "rtt_ms": 85.0,
        "positions": {"C03": [0, 2], "C04": [0, 1], "C07": [0, -1], "C08": [1, -1]},
        "hp_overrides": {},
        "description": "Turnajový WAN profil s vysokou latenciou (RTT=85 ms), horizontom H=7 a ochranou cez HORIZON_SHIELD.",
    },
}


def instantiate_combat_preset(preset_id: str, seed: int = 6) -> Tuple[ClashGame, Dict[str, Any]]:
    """Build a valid ClashGame configured to one of the 6 tactical hex presets."""
    if preset_id not in HEX_COMBAT_PRESETS:
        raise ContractViolation(f"Unknown combat preset {preset_id!r}", violation_code="ERR_PRESET_ID")
    spec = HEX_COMBAT_PRESETS[preset_id]
    game = ClashGame(spec["player_faction"], seed=seed)
    game.state["round"] = int(spec["round"])
    game.state["zone_radius"] = int(spec["zone_radius"])
    if spec["uma"] >= UMA_SATURATION_THRESHOLD:
        game.state["budget"] = {f: ROUND_BUDGET_FRAMES // 2 for f in FACTIONS}

    # Place specified units cleanly without collisions
    used = set()
    for uid, pos in spec["positions"].items():
        pt = (int(pos[0]), int(pos[1]))
        if pt not in game.cells or pt in used:
            raise ContractViolation(f"Invalid preset coordinate {pt} for {uid}", violation_code="ERR_PRESET_POS")
        used.add(pt)
        game.state["units"][uid]["pos"] = [pt[0], pt[1]]

    # Relocate any remaining units whose default corner collided with preset positions
    for uid, u in sorted(game.state["units"].items()):
        if uid in spec["positions"]:
            continue
        pt = tuple(u["pos"])
        if pt in used:
            free_cell = next(c for c in sorted(game.cells) if c not in used)
            u["pos"] = [free_cell[0], free_cell[1]]
            used.add(free_cell)
        else:
            used.add(pt)

    for uid, hp_val in spec["hp_overrides"].items():
        if not (1 <= int(hp_val) <= VITAL_MAX_HP):
            raise ContractViolation("Preset HP override out of [1, 6]", violation_code="ERR_PRESET_HP")
        game.state["units"][uid]["hp"] = int(hp_val)

    bw = compute_full_combat_bandwidth_solution(
        board_radius=BOARD_RADIUS,
        uma_utilization=float(spec["uma"]),
        one_way_rtt_ms=float(spec["rtt_ms"]),
    )
    return game, {"preset_id": preset_id, "spec": spec, "bandwidth_profile": bw["bandwidth_solution"]}


def evaluate_all_faction_duel_combinations() -> Dict[str, Any]:
    """Simulate all C(4, 2) = 6 pairwise faction combat combinations and preset bandwidth profiles."""
    factions = list(FACTIONS.keys())
    duels = []
    for fa, fb in itertools.combinations(factions, 2):
        g = ClashGame(fa, seed=6)
        ua = [u for u in g.state["units"].values() if u["faction"] == fa]
        ub = [u for u in g.state["units"].values() if u["faction"] == fb]
        # Place lead units adjacent at (0,0) and (1,0) for direct combat exchange metric
        dmg_a_to_b = sum(max(1, a["atk"] - b["def"]) for a, b in zip(ua, ub))
        dmg_b_to_a = sum(max(1, b["atk"] - a["def"]) for a, b in zip(ua, ub))
        rounds_to_elim_b = math.ceil(sum(b["hp"] for b in ub) / max(1, dmg_a_to_b))
        rounds_to_elim_a = math.ceil(sum(a["hp"] for a in ua) / max(1, dmg_b_to_a))
        duels.append(
            {
                "pair": f"{fa}_vs_{fb}",
                "damage_per_round_a_to_b": dmg_a_to_b,
                "damage_per_round_b_to_a": dmg_b_to_a,
                "rounds_to_eliminate_b": rounds_to_elim_b,
                "rounds_to_eliminate_a": rounds_to_elim_a,
                "structural_advantage": fa if rounds_to_elim_b < rounds_to_elim_a else (fb if rounds_to_elim_a < rounds_to_elim_b else "SYMMETRIC"),
            }
        )

    preset_summaries = []
    for pid in HEX_COMBAT_PRESETS:
        g, meta = instantiate_combat_preset(pid)
        preset_summaries.append({"preset_id": pid, "state_sha256": g.state_hash()[:16], **meta})

    return {
        "vital_max_hp": VITAL_MAX_HP,
        "combinatorial_identities": verify_hex_combinatorial_identities(max_radius=4),
        "pairwise_faction_duels": duels,
        "presets": preset_summaries,
    }


# ---------------------------------------------------------------------------
# 3. Generated World Effects & Action-Coupled AI Reactions
# ---------------------------------------------------------------------------

def generate_world_hex_influences(
    seed: int = 6,
    round_num: int = 1,
    zone_radius: int = BOARD_RADIUS,
    uma_utilization: float = 0.35,
) -> Dict[Tuple[int, int], Dict[str, Any]]:
    """Deterministically generate world hex effects over the 37-cell board from seed + hardware state."""
    cells = board_cells(BOARD_RADIUS)
    uma_high = uma_utilization >= UMA_SATURATION_THRESHOLD
    world: Dict[Tuple[int, int], Dict[str, Any]] = {}

    for q, r in cells:
        d = hex_distance((q, r), (0, 0))
        digest = hashlib.sha256(f"{seed}:{round_num}:{q}:{r}".encode("utf-8")).digest()[0]
        if d > zone_radius:
            kind = "ZONE_STORM_DRAIN"
            effect = "Mimo bezpečnej zóny: -1 HP na konci kola a okamžitá strata 1 rámca pri vstupe."
        elif (q, r) == (0, 0) or (d == 1 and (q + r) == 0):
            kind = "CACHE_RESONANCE_WELL"
            effect = "Rezonančná studňa CACHE_DIR: jednotka získava +1 dočasnú obranu (tmp_def) pri vstupe."
        elif (abs(q) == 2 and r == 0) or (q == 0 and abs(r) == 2):
            kind = "THERMAL_BACKPRESSURE_VENT" if uma_high else "UMA_FLUX_CONDUIT"
            effect = (
                "Tepelný pretlak zbernice (UMA >= 0.78): ruší útočný bonus tmp_atk na tomto hexe."
                if uma_high
                else " Tok UMA (< 0.78): udeľuje +1 dočasný útok (tmp_atk) pri vstupe alebo karte."
            )
        elif digest % 5 == 0:
            kind = "FREIVALDS_PARITY_NEXUS"
            effect = "Paritný uzol: pri útoku z/na tento hex lieči +1 HP vlastnej CONFLICT jednotke (max 6)."
        else:
            kind = "STABLE_BASALT_HEX"
            effect = "Stabilný čadičový hex bez dodatočnej odchýlky stavu."
        world[(q, r)] = {"cell": [q, r], "effect_type": kind, "description": effect}
    return world


def resolve_action_with_world_and_ai_chain(
    game: ClashGame,
    action: Dict[str, Any],
    uma_utilization: float = 0.35,
) -> Dict[str, Any]:
    """Execute player action, apply deterministic world hex influence, and trigger AI counter-reaction."""
    pre_hash = game.state_hash()
    player_fac = game.state["player_faction"]

    # 1. Primary authoritative action (raises ContractViolation with Delta S == 0 if invalid!)
    primary_receipt = game.apply(action, player_fac)

    world_map = generate_world_hex_influences(
        seed=game.state["seed"],
        round_num=game.state["round"],
        zone_radius=game.state["zone_radius"],
        uma_utilization=uma_utilization,
    )
    world_events: List[Dict[str, Any]] = []
    ai_reactions: List[Dict[str, Any]] = []

    # 2. Evaluate world influence triggered by the committed action
    kind = action.get("type")
    focus_uid = action.get("unit") if kind in ("move", "attack") else action.get("target")
    if focus_uid and focus_uid in game.state["units"]:
        u = game.state["units"][str(focus_uid)]
        cell = tuple(u["pos"])
        w_tile = world_map.get(cell, {"effect_type": "STABLE_BASALT_HEX"})
        w_type = w_tile["effect_type"]

        if kind == "move" and u["hp"] > 0:
            if w_type == "CACHE_RESONANCE_WELL":
                u["tmp_def"] += 1
                world_events.append({"effect_type": w_type, "unit": u["unit_id"], "delta": "+1 tmp_def", **_pack_frame(f"WORLD|{w_type}|{u['unit_id']}")})
            elif w_type == "UMA_FLUX_CONDUIT":
                u["tmp_atk"] += 1
                world_events.append({"effect_type": w_type, "unit": u["unit_id"], "delta": "+1 tmp_atk", **_pack_frame(f"WORLD|{w_type}|{u['unit_id']}")})
            elif w_type == "THERMAL_BACKPRESSURE_VENT":
                u["tmp_atk"] = 0
                world_events.append({"effect_type": w_type, "unit": u["unit_id"], "delta": "tmp_atk->0", **_pack_frame(f"WORLD|{w_type}|{u['unit_id']}")})
            elif w_type == "ZONE_STORM_DRAIN" and game.state["budget"][player_fac] > 0:
                game.state["budget"][player_fac] -= 1
                world_events.append({"effect_type": w_type, "unit": u["unit_id"], "delta": "-1 budget_frame", **_pack_frame(f"WORLD|{w_type}|{u['unit_id']}")})

        elif kind == "attack":
            tgt_u = game.state["units"][str(action["target"])]
            t_cell = tuple(tgt_u["pos"])
            if world_map.get(cell, {}).get("effect_type") == "FREIVALDS_PARITY_NEXUS" or world_map.get(t_cell, {}).get("effect_type") == "FREIVALDS_PARITY_NEXUS":
                if u["verdict"] == "CONFLICT" and u["hp"] < VITAL_MAX_HP:
                    u["hp"] = min(VITAL_MAX_HP, u["hp"] + 1)
                    world_events.append({"effect_type": "FREIVALDS_PARITY_NEXUS", "unit": u["unit_id"], "delta": "+1 hp (CONFLICT parity)", **_pack_frame(f"WORLD|PARITY|{u['unit_id']}")})

    # 3. Immediate AI Counter-Reaction coupled directly to the player's action
    if not game.state.get("winner") and kind in ("move", "attack", "card"):
        if kind == "attack":
            rival_uid = str(action["target"])
            rival = game.state["units"][rival_uid]
            rival_fac = rival["faction"]
            actor_u = game.state["units"][str(action["unit"])]
            # If target survived and has budget: shield if low HP, else immediate counter-attack if unacted
            if rival["hp"] > 0:
                if rival["hp"] <= 3 and game.state["budget"][rival_fac] >= CARDS["HORIZON_SHIELD"]["cost"]:
                    rec = game.apply({"type": "card", "card": "HORIZON_SHIELD", "target": rival_uid}, rival_fac)
                    ai_reactions.append({"ai_faction": rival_fac, "reaction_type": "AI_EMERGENCY_HORIZON_SHIELD", "receipt": rec})
                elif not rival["acted"] and actor_u["hp"] > 0 and hex_distance(tuple(rival["pos"]), tuple(actor_u["pos"])) == 1 and game.state["budget"][rival_fac] >= 1:
                    rec = game.apply({"type": "attack", "unit": rival_uid, "target": actor_u["unit_id"]}, rival_fac)
                    ai_reactions.append({"ai_faction": rival_fac, "reaction_type": "AI_IMMEDIATE_COUNTER_STRIKE", "receipt": rec})

        elif kind == "card" and action.get("card") == "GPU_ALLOCATE":
            # Nearest rival faction raises HORIZON_SHIELD against GPU offensive spike
            actor_t = game.state["units"][str(action["target"])]
            rivals = [
                v for v in game.state["units"].values()
                if v["faction"] != player_fac and v["hp"] > 0 and game.state["budget"][v["faction"]] >= CARDS["HORIZON_SHIELD"]["cost"]
            ]
            if rivals:
                nearest = min(rivals, key=lambda e: (hex_distance(tuple(actor_t["pos"]), tuple(e["pos"])), e["unit_id"]))
                rec = game.apply({"type": "card", "card": "HORIZON_SHIELD", "target": nearest["unit_id"]}, nearest["faction"])
                ai_reactions.append({"ai_faction": nearest["faction"], "reaction_type": "AI_COUNTER_SHIELD_VS_GPU_SPIKE", "receipt": rec})

        elif kind == "move":
            moved_u = game.state["units"][str(action["unit"])]
            if hex_distance(tuple(moved_u["pos"]), (0, 0)) <= 1:
                # Rival unit intercepts center occupation
                occ = {tuple(v["pos"]) for v in game.state["units"].values() if v["hp"] > 0}
                rivals = [
                    v for v in sorted(game.state["units"].values(), key=lambda x: x["unit_id"])
                    if v["faction"] != player_fac and v["hp"] > 0 and not v["acted"] and game.state["budget"][v["faction"]] >= 1
                ]
                for rv in rivals:
                    rpos = tuple(rv["pos"])
                    cur_d = hex_distance(rpos, tuple(moved_u["pos"]))
                    steps = [
                        (hex_distance((rpos[0] + dq, rpos[1] + dr), tuple(moved_u["pos"])), (rpos[0] + dq, rpos[1] + dr))
                        for dq, dr in DIRS
                        if (rpos[0] + dq, rpos[1] + dr) in game.cells and (rpos[0] + dq, rpos[1] + dr) not in occ
                    ]
                    closer = [c for d_new, c in sorted(steps) if d_new < cur_d]
                    if closer:
                        rec = game.apply({"type": "move", "unit": rv["unit_id"], "to": [closer[0][0], closer[0][1]]}, rv["faction"])
                        ai_reactions.append({"ai_faction": rv["faction"], "reaction_type": "AI_CENTER_INTERCEPT_MANEUVER", "receipt": rec})
                        break

    # Final invariant check
    for u in game.state["units"].values():
        if not (0 <= u["hp"] <= VITAL_MAX_HP):
            raise ContractViolation("HP invariant breached after world/AI chain", violation_code="ERR_WORLD_AI_HP")

    return {
        "vital_max_hp": VITAL_MAX_HP,
        "pre_state_sha256": pre_hash,
        "post_state_sha256": game.state_hash(),
        "primary_receipt": primary_receipt,
        "world_events": world_events,
        "ai_reactions": ai_reactions,
    }
