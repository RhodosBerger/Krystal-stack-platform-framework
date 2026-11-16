"""Capability-driven combat and turn planner for Clash Battle Royale.

Connects real disposed framework and hardware properties (OpenVINO/oneAPI devices,
Win32 ISA flags, warm CACHE_DIR presence, UMA saturation ceiling < 0.78, and
consistency-bot verdicts C01..C08) directly to turn planning in ClashGame.

Constitutional rules:
  - PROPOSER = CapabilityCombatTurnPlanner
  - VERIFIER = ClashGame.apply() executed on a shadow clone before any commit
  - VITAL_MAX_HP == 6; invalid proposal -> ContractViolation and Delta S == 0
  - Sovereign P3: when UMA >= 0.78 or GPU is absent, GPU_ALLOCATE is vetoed and
    the usable 128 B frame budget is halved (16 -> 8) to preserve scanout headroom.
"""

from __future__ import annotations

import copy
import ctypes
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from krystal_kernel.constants import UMA_SATURATION_THRESHOLD as UMA_SATURATION_CEILING, VITAL_MAX_HP, ContractViolation
from krystal_kernel.hwprofile import detect_profile

from .clash_battle_royale import (
    ACTION_COST,
    BOARD_RADIUS,
    CARDS,
    DIRS,
    FACTIONS,
    ROUND_BUDGET_FRAMES,
    ClashGame,
    hex_distance,
)
from .custom_allocator import FactDrivenAllocator, check_ontology_integrity
from .fact_synthesis_engine import REPO_ROOT, run_fact_synthesis

DEFAULT_CACHE_DIR = REPO_ROOT / "artifacts" / "cache" / "openvino_blobs"
_HW_MEMO: Dict[str, Any] = {}


def get_cached_hw_profile():
    """Return process-cached HardwareProfile to avoid repeated OpenCL driver re-initialization."""
    if "hw" not in _HW_MEMO:
        _HW_MEMO["hw"] = detect_profile(use_cached_calibration=True)
    return _HW_MEMO["hw"]


def probe_isa_flags() -> Dict[str, Any]:
    """Query OS-level ISA feature flags on Windows via IsProcessorFeaturePresent."""
    if "isa" in _HW_MEMO:
        return _HW_MEMO["isa"]
    # Win32 PF_* constants: 10=XMMI64(SSE2), 13=SSE3, 39=AVX, 40=AVX2, 41=AVX512F
    win_pf = {"SSE2": 10, "SSE3": 13, "AVX": 39, "AVX2": 40, "AVX512F": 41}
    flags: Dict[str, Optional[bool]] = {k: None for k in win_pf}
    source = "UNAVAILABLE_NON_WIN32"
    try:
        k32 = getattr(ctypes, "windll", None)
        if k32 is not None and hasattr(k32, "kernel32"):
            fn = k32.kernel32.IsProcessorFeaturePresent
            flags = {name: bool(fn(code)) for name, code in win_pf.items()}
            source = "WIN32_IsProcessorFeaturePresent"
    except Exception:
        source = "PROBE_ERROR"
    res = {"flags": flags, "source": source, "evidence": "LOCAL_MEASURED" if source.startswith("WIN32") else "UNKNOWN"}
    _HW_MEMO["isa"] = res
    return res


def probe_framework_dispositions(
    uma_utilization: float = 0.35,
    expected_inferences: int = 600,
    latency_budget_ms: float = 10.0,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    override_devices: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Build the verified capability envelope ('disponované vlastnosti frameworku')."""
    if not (0.0 <= uma_utilization <= 1.0):
        raise ContractViolation(f"uma_utilization={uma_utilization} outside [0.0, 1.0]", violation_code="ERR_UMA_RANGE")

    hw = get_cached_hw_profile()
    devices = list(override_devices) if override_devices is not None else list(hw.openvino_devices)
    isa = probe_isa_flags()
    syn = run_fact_synthesis(write=False)
    ont = check_ontology_integrity(syn)
    alloc = FactDrivenAllocator(syn["facts"]).allocate(expected_inferences, latency_budget_ms)

    blob_files = list(cache_dir.glob("*")) if cache_dir.exists() else []
    cache_bytes = sum(f.stat().st_size for f in blob_files if f.is_file())
    verdicts = {
        f["fact_id"].split("_")[1]: f["value"]
        for f in syn["facts"]
        if f["fact_id"].startswith("F_C0") and f["fact_id"].endswith("_verdict")
    }

    gpu_available = "GPU" in devices
    uma_ok = uma_utilization < UMA_SATURATION_CEILING
    sync_verified = verdicts.get("C01") == "VERIFIED_WITH_SCOPE"
    usable_frames = ROUND_BUDGET_FRAMES if uma_ok else (ROUND_BUDGET_FRAMES // 2)

    card_gates = {
        "GPU_ALLOCATE": {
            "unlocked": bool(gpu_available and uma_ok),
            "reason": (
                "GPU Iris Xe present and UMA < 0.78"
                if (gpu_available and uma_ok)
                else ("CHRONOMETRIC_SHADOW_VETO_P3: UMA >= 0.78" if gpu_available else "GPU device unavailable")
            ),
        },
        "FREIVALDS_AUDIT": {
            "unlocked": True,
            "reason": "Freivalds O(n^2) verifier active; applies only to own CONFLICT units with HP < 6",
        },
        "STRICT_JOIN": {
            "unlocked": bool(sync_verified),
            "reason": "C01 sync authority verified" if sync_verified else "C01 not verified",
        },
        "HORIZON_SHIELD": {
            "unlocked": True,
            "reason": "AdaptiveHorizonController active (12-tick downshift hysteresis)",
        },
    }

    return {
        "vital_max_hp": VITAL_MAX_HP,
        "cpu_name": hw.cpu_name,
        "physical_cores": hw.physical_cores,
        "logical_cores": hw.logical_cores,
        "ram_gb": hw.ram_gb,
        "openvino_devices": devices,
        "npu_present": hw.npu_present,
        "isa": isa,
        "uma_utilization": round(uma_utilization, 4),
        "uma_ceiling": UMA_SATURATION_CEILING,
        "uma_headroom_ok": uma_ok,
        "usable_frame_budget": usable_frames,
        "reserved_uma_frames": ROUND_BUDGET_FRAMES - usable_frames,
        "cache_dir": str(cache_dir),
        "warm_cache_blob_count": len(blob_files),
        "warm_cache_bytes": cache_bytes,
        "allocator_recommendation": alloc,
        "ontology_integrity": ont["verdict"],
        "check_verdicts": verdicts,
        "card_gates": card_gates,
        "evidence": "LOCAL_MEASURED",
    }


class CapabilityCombatTurnPlanner:
    """Proposes and shadow-verifies ClashGame combat turns governed by framework capabilities."""

    def _adjacent_enemies(self, state: Dict[str, Any], faction: str, pos: Tuple[int, int]) -> List[Dict[str, Any]]:
        return sorted(
            (
                u for u in state["units"].values()
                if u["faction"] != faction and u["hp"] > 0 and hex_distance(pos, tuple(u["pos"])) == 1
            ),
            key=lambda e: (e["hp"], -e["atk"], e["unit_id"]),
        )

    def _threat_at(self, state: Dict[str, Any], faction: str, pos: Tuple[int, int], unit_def: int) -> int:
        total = 0
        for e in state["units"].values():
            if e["faction"] != faction and e["hp"] > 0 and hex_distance(pos, tuple(e["pos"])) == 1:
                total += max(1, e["atk"] + e["tmp_atk"] - unit_def)
        return total

    def _plan_cards(
        self,
        shadow: ClashGame,
        faction: str,
        dispositions: Dict[str, Any],
        remaining_frames: int,
    ) -> Tuple[List[Dict[str, Any]], List[str], int]:
        actions: List[Dict[str, Any]] = []
        notes: List[str] = []
        gates = dispositions["card_gates"]
        own = [
            shadow.state["units"][uid]
            for uid in sorted(shadow.state["units"])
            if shadow.state["units"][uid]["faction"] == faction and shadow.state["units"][uid]["hp"] > 0
        ]
        if not own:
            return actions, notes, remaining_frames

        # 1. Repair damaged CONFLICT unit via FREIVALDS_AUDIT (5 frames)
        for u in own:
            cost = CARDS["FREIVALDS_AUDIT"]["cost"]
            if (
                gates["FREIVALDS_AUDIT"]["unlocked"]
                and u["verdict"] == "CONFLICT"
                and u["hp"] < VITAL_MAX_HP
                and remaining_frames >= cost + 2
            ):
                act = {"type": "card", "card": "FREIVALDS_AUDIT", "target": u["unit_id"]}
                shadow.apply(act, faction)
                actions.append(act)
                remaining_frames -= cost
                notes.append(f"FREIVALDS_AUDIT lieči {u['unit_id']} (+1 HP, check v stave CONFLICT).")

        # 2. Offensive boost GPU_ALLOCATE (8 frames) if unlocked and an enemy is within strike range (<= 2 hexes)
        enemies = [v for v in shadow.state["units"].values() if v["faction"] != faction and v["hp"] > 0]
        if enemies and gates["GPU_ALLOCATE"]["unlocked"]:
            cost = CARDS["GPU_ALLOCATE"]["cost"]
            striker = min(
                own,
                key=lambda u: (min(hex_distance(tuple(u["pos"]), tuple(e["pos"])) for e in enemies), u["unit_id"]),
            )
            min_dist = min(hex_distance(tuple(striker["pos"]), tuple(e["pos"])) for e in enemies)
            if min_dist <= 2 and remaining_frames >= cost + 2:
                act = {"type": "card", "card": "GPU_ALLOCATE", "target": striker["unit_id"]}
                shadow.apply(act, faction)
                actions.append(act)
                remaining_frames -= cost
                notes.append(f"GPU_ALLOCATE posilňuje {striker['unit_id']} (+1 ATK cez Iris Xe, UMA < 0.78).")
        elif not gates["GPU_ALLOCATE"]["unlocked"]:
            notes.append(f"GPU_ALLOCATE zablokovaná: {gates['GPU_ALLOCATE']['reason']}.")

        # 3. Defensive card if own units are threatened or clustered
        threatened = [u for u in own if self._threat_at(shadow.state, faction, tuple(u["pos"]), u["def"] + u["tmp_def"]) > 0]
        if len(own) >= 2 and hex_distance(tuple(own[0]["pos"]), tuple(own[1]["pos"])) <= 1:
            cost = CARDS["STRICT_JOIN"]["cost"]
            if gates["STRICT_JOIN"]["unlocked"] and threatened and remaining_frames >= cost + len(own):
                act = {"type": "card", "card": "STRICT_JOIN", "target": own[0]["unit_id"]}
                shadow.apply(act, faction)
                actions.append(act)
                remaining_frames -= cost
                notes.append(f"STRICT_JOIN kryje susediacu dvojicu od {own[0]['unit_id']} (+1 DEF).")
        elif threatened:
            cost = CARDS["HORIZON_SHIELD"]["cost"]
            target_u = min(threatened, key=lambda u: (u["hp"], u["unit_id"]))
            if gates["HORIZON_SHIELD"]["unlocked"] and remaining_frames >= cost + len(own):
                act = {"type": "card", "card": "HORIZON_SHIELD", "target": target_u["unit_id"]}
                shadow.apply(act, faction)
                actions.append(act)
                remaining_frames -= cost
                notes.append(f"HORIZON_SHIELD chráni ohrozenú jednotku {target_u['unit_id']} (+1 DEF).")

        return actions, notes, remaining_frames

    def _best_unit_step(
        self,
        shadow: ClashGame,
        faction: str,
        uid: str,
        remaining_frames: int,
    ) -> Optional[Tuple[Dict[str, Any], Dict[str, Any]]]:
        u = shadow.state["units"][uid]
        if u["hp"] <= 0 or u["acted"] or remaining_frames < 1:
            return None
        enemies = [v for v in shadow.state["units"].values() if v["faction"] != faction and v["hp"] > 0]
        if not enemies:
            return None

        pos = tuple(u["pos"])
        adj = self._adjacent_enemies(shadow.state, faction, pos)
        if adj:
            # Prefer lethal elimination, otherwise lowest HP target for focus-fire
            best_target = None
            best_score = -10_000.0
            for e in adj:
                dmg = max(1, u["atk"] + u["tmp_atk"] - e["def"] - e["tmp_def"])
                lethal = 1 if e["hp"] <= dmg else 0
                score = lethal * 100.0 + dmg * 15.0 - e["hp"] * 2.0
                if score > best_score:
                    best_score = score
                    best_target = (e, dmg, lethal)
            assert best_target is not None
            e, dmg, lethal = best_target
            return (
                {"type": "attack", "unit": uid, "target": e["unit_id"]},
                {"expected_damage": dmg, "lethal": bool(lethal), "zone_save": False, "score": round(best_score, 2)},
            )

        # Evaluate legal 1-hex moves
        next_zone_r = max(0, BOARD_RADIUS - shadow.state["round"] // 3)
        cur_outside = hex_distance(pos, (0, 0)) > shadow.state["zone_radius"]
        occ = {tuple(v["pos"]) for v in shadow.state["units"].values() if v["hp"] > 0}
        candidates: List[Tuple[float, Tuple[int, int], bool]] = []
        for dq, dr in DIRS:
            c = (pos[0] + dq, pos[1] + dr)
            if c not in shadow.cells or c in occ:
                continue
            in_curr_zone = hex_distance(c, (0, 0)) <= shadow.state["zone_radius"]
            in_next_zone = hex_distance(c, (0, 0)) <= next_zone_r
            dist_enemy = min(hex_distance(c, tuple(e["pos"])) for e in enemies)
            threat = self._threat_at(shadow.state, faction, c, u["def"] + u["tmp_def"])
            survivable = 1 if threat < u["hp"] else -3
            zone_save = cur_outside and in_curr_zone
            score = (
                (40.0 if in_curr_zone else -40.0)
                + (20.0 if in_next_zone else -10.0)
                + survivable * 12.0
                - dist_enemy * 6.0
                - hex_distance(c, (0, 0)) * 1.5
            )
            candidates.append((score, c, zone_save))
        if not candidates:
            return None
        candidates.sort(key=lambda item: (-item[0], item[1][0], item[1][1]))
        score, dest, zone_save = candidates[0]
        return (
            {"type": "move", "unit": uid, "to": [dest[0], dest[1]]},
            {"expected_damage": 0, "lethal": False, "zone_save": zone_save, "score": round(score, 2)},
        )

    def plan_turn(
        self,
        game: ClashGame,
        faction: Optional[str] = None,
        uma_utilization: float = 0.35,
        expected_inferences: int = 600,
        commit: bool = False,
        override_devices: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Compute, shadow-verify, and optionally commit a full turn plan for `faction`."""
        target_faction = faction or game.state["player_faction"]
        if target_faction not in FACTIONS:
            raise ContractViolation(f"Unknown faction {target_faction}", violation_code="ERR_PLANNER_FACTION")
        if game.state.get("winner"):
            raise ContractViolation("Cannot plan turn: game already finished", violation_code="ERR_PLANNER_GAME_OVER")

        pre_hash = game.state_hash()
        dispositions = probe_framework_dispositions(
            uma_utilization=uma_utilization,
            expected_inferences=expected_inferences,
            override_devices=override_devices,
        )

        # Shadow clone guarantees PROPOSER != VERIFIER and Delta S == 0 on dry-run or failure
        shadow = copy.deepcopy(game)
        avail_budget = min(shadow.state["budget"][target_faction], dispositions["usable_frame_budget"])

        planned_actions: List[Dict[str, Any]] = []
        step_metrics: List[Dict[str, Any]] = []
        card_actions, rationale, rem = self._plan_cards(shadow, target_faction, dispositions, avail_budget)
        for ca in card_actions:
            planned_actions.append(ca)
            step_metrics.append({"action": ca, "cost_frames": CARDS[ca["card"]]["cost"], "kind": "card"})

        for uid in sorted(u for u, v in shadow.state["units"].items() if v["faction"] == target_faction):
            choice = self._best_unit_step(shadow, target_faction, uid, rem)
            if choice is None:
                continue
            act, meta = choice
            shadow.apply(act, target_faction)
            cost = ACTION_COST[act["type"]]
            rem -= cost
            planned_actions.append(act)
            step_metrics.append({"action": act, "cost_frames": cost, "kind": act["type"], **meta})
            if act["type"] == "attack":
                rationale.append(
                    f"ÚTOK {uid} -> {act['target']} (očakávané poškodenie {meta['expected_damage']}, eliminácia={meta['lethal']})."
                )
            else:
                rationale.append(f"POHYB {uid} -> {act['to']} (záchrana zo zóny={meta['zone_save']}, skóre={meta['score']}).")

        # Independent replay audit from fresh clone
        verifier_clone = copy.deepcopy(game)
        receipts: List[Dict[str, Any]] = []
        for act in planned_actions:
            rec = verifier_clone.apply(act, target_faction)
            receipts.append(rec)
        for u in verifier_clone.state["units"].values():
            if not (0 <= u["hp"] <= VITAL_MAX_HP):
                raise ContractViolation("HP invariant breach in shadow verification", violation_code="ERR_PLANNER_HP")

        frames_used = sum(int(r["frames"]) for r in receipts)
        projected_hash = verifier_clone.state_hash()

        committed_receipts: List[Dict[str, Any]] = []
        if commit:
            for act in planned_actions:
                committed_receipts.append(game.apply(act, target_faction))
        else:
            if game.state_hash() != pre_hash:
                raise ContractViolation("Dry-run mutated authoritative state", violation_code="ERR_PLANNER_MUTATION")

        return {
            "vital_max_hp": VITAL_MAX_HP,
            "verdict": "VERIFIED_WITH_SCOPE",
            "proposer": "CapabilityCombatTurnPlanner",
            "verifier": "ClashGame.apply(shadow_clone)",
            "committed": commit,
            "faction": target_faction,
            "round": game.state["round"],
            "pre_state_sha256": pre_hash,
            "projected_state_sha256": projected_hash,
            "post_state_sha256": game.state_hash(),
            "frames_used": frames_used,
            "usable_frame_budget": avail_budget,
            "reserved_uma_frames": dispositions["reserved_uma_frames"],
            "expected_damage_total": sum(m.get("expected_damage", 0) for m in step_metrics),
            "expected_eliminations": sum(1 for m in step_metrics if m.get("lethal")),
            "planned_actions": planned_actions,
            "step_metrics": step_metrics,
            "shadow_receipts": receipts,
            "committed_receipts": committed_receipts,
            "rationale": rationale,
            "dispositions_summary": {
                "openvino_devices": dispositions["openvino_devices"],
                "avx512f": dispositions["isa"]["flags"].get("AVX512F"),
                "uma_utilization": dispositions["uma_utilization"],
                "uma_headroom_ok": dispositions["uma_headroom_ok"],
                "warm_cache_blob_count": dispositions["warm_cache_blob_count"],
                "allocator_device": dispositions["allocator_recommendation"].get("device"),
                "card_gates": dispositions["card_gates"],
            },
        }
