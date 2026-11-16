"""Clash Battle Royale engine — authoritative state for the arena UI.

The UI only proposes actions; this engine validates and commits them (Render(S) != S).
Framework elements become game mechanics:
  units      = consistency-bot checks (verdict -> defense), 4 factions = 4 framework layers
  memory     = per-round budget of 128 B frames; every committed action is packed into one
               real 128 B frame (96 B payload + 32 B SHA-256 tag) and costs frames
  cards      = configurable framework elements (horizon controller, strict join, allocator, Freivalds)
  zone       = shrinking safe radius (battle royale); outside units lose 1 HP per round
Invariants: HP in [0, 6]; an invalid action raises ContractViolation and leaves state unchanged.
"""

from __future__ import annotations

import copy
import hashlib
import json
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from krystal_kernel.constants import VITAL_MAX_HP, ContractViolation

REPO_ROOT = Path(__file__).resolve().parents[2]
BOARD_RADIUS = 3
ROUND_BUDGET_FRAMES = 16
MAX_ROUNDS = 12
FRAME_BYTES, PAYLOAD_BYTES = 128, 96
DIRS = [(1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1)]

FACTIONS = {
    "SYNC": {"checks": ("C01", "C02"), "corner": (3, 0), "color": "#58D8FF"},
    "COMPUTE": {"checks": ("C03", "C04"), "corner": (0, 3), "color": "#FF307C"},
    "VERIFY": {"checks": ("C05", "C06"), "corner": (-3, 0), "color": "#50EEE0"},
    "OPS": {"checks": ("C07", "C08"), "corner": (0, -3), "color": "#D8AE4B"},
}
DEFENSE_BY_VERDICT = {"VERIFIED_WITH_SCOPE": 2, "MEASURED_WITH_SCOPE": 2, "SUPPORTED_WITH_SCOPE": 1}
CARDS = {
    "HORIZON_SHIELD": {"cost": 4, "effect": "+1 defense to one own unit this round",
                       "framework_ref": "adaptive_cross_genre_sync.AdaptiveHorizonController"},
    "STRICT_JOIN": {"cost": 6, "effect": "+1 defense to an own unit and its adjacent allies this round",
                    "framework_ref": "adaptive_cross_genre_sync.join_sync_policy"},
    "GPU_ALLOCATE": {"cost": 8, "effect": "+1 attack to one own unit this round",
                     "framework_ref": "custom_allocator.FactDrivenAllocator"},
    "FREIVALDS_AUDIT": {"cost": 5, "effect": "heal 1 HP on an own unit whose check is in CONFLICT",
                        "framework_ref": "matrix_control_processes.freivalds_verify"},
}
ACTION_COST = {"move": 1, "attack": 1}
FRAMEWORK_STEP = {"move": "PAIR_JOIN_ARBITER -> STEP_GATE_PROXY (position proposal)",
                  "attack": "PAIR_JOIN_ARBITER -> STEP_GATE_PROXY (state-writing interaction)",
                  "card": "POLICY_CONTROLLER -> STEP_GATE_PROXY (configuration change)"}


def hex_distance(a: Tuple[int, int], b: Tuple[int, int]) -> int:
    dq, dr = a[0] - b[0], a[1] - b[1]
    return (abs(dq) + abs(dr) + abs(dq + dr)) // 2


def board_cells(radius: int = BOARD_RADIUS) -> List[Tuple[int, int]]:
    return [(q, r) for q in range(-radius, radius + 1) for r in range(-radius, radius + 1)
            if hex_distance((q, r), (0, 0)) <= radius]


def _load_verdicts() -> Dict[str, Dict[str, str]]:
    p = REPO_ROOT / "reports" / "consistency_bot_latest.json"
    if not p.exists():
        return {}
    data = json.loads(p.read_text(encoding="utf-8"))
    return {c["check_id"]: {"verdict": c["verdict"], "claim": c["claim"]} for c in data["checks"]}


def _pack_frame(payload: str) -> Dict[str, str]:
    raw = payload.encode("utf-8")[:PAYLOAD_BYTES].ljust(PAYLOAD_BYTES, b"\x00")
    tag = hashlib.sha256(raw).digest()
    frame = raw + tag
    if len(frame) != FRAME_BYTES:
        raise ContractViolation("frame size != 128", violation_code="ERR_CLASH_FRAME")
    return {"frame_sha256_prefix": tag.hex()[:16], "frame_bytes": str(len(frame))}


class ClashGame:
    def __init__(self, player_faction: str = "SYNC", seed: int = 6,
                 verdicts: Optional[Dict[str, Dict[str, str]]] = None) -> None:
        if player_faction not in FACTIONS:
            raise ContractViolation(f"Unknown faction {player_faction}", violation_code="ERR_CLASH_FACTION")
        verdicts = _load_verdicts() if verdicts is None else verdicts
        self.cells = set(board_cells())
        units: Dict[str, Dict[str, Any]] = {}
        for fac, spec in FACTIONS.items():
            corner = spec["corner"]
            inward = (corner[0] - (corner[0] > 0) + (corner[0] < 0), corner[1] - (corner[1] > 0) + (corner[1] < 0))
            for cid, pos in zip(spec["checks"], (corner, inward)):
                v = verdicts.get(cid, {"verdict": "UNKNOWN", "claim": cid})
                units[cid] = {"unit_id": cid, "faction": fac, "name": v["claim"][:48], "verdict": v["verdict"],
                              "hp": VITAL_MAX_HP, "atk": 1 if v["verdict"] in ("UNKNOWN", "FALSIFIED") else 2,
                              "def": DEFENSE_BY_VERDICT.get(v["verdict"], 0), "pos": list(pos),
                              "tmp_def": 0, "tmp_atk": 0, "acted": False}
        self.state: Dict[str, Any] = {
            "vital_max_hp": VITAL_MAX_HP, "seed": seed, "round": 1, "player_faction": player_faction,
            "zone_radius": BOARD_RADIUS, "budget": {f: ROUND_BUDGET_FRAMES for f in FACTIONS},
            "units": units, "log": [], "winner": None,
        }

    # ------------------------------------------------------------------ helpers
    def state_hash(self) -> str:
        return hashlib.sha256(json.dumps(self.state, sort_keys=True).encode("utf-8")).hexdigest()

    def _unit(self, uid: str) -> Dict[str, Any]:
        u = self.state["units"].get(uid)
        if u is None or u["hp"] <= 0:
            raise ContractViolation(f"Unit {uid} missing or eliminated", violation_code="ERR_CLASH_UNIT")
        return u

    def _occupied(self) -> Dict[Tuple[int, int], str]:
        return {tuple(u["pos"]): uid for uid, u in self.state["units"].items() if u["hp"] > 0}

    def _commit(self, faction: str, kind: str, cost: int, desc: str, mutate) -> Dict[str, Any]:
        if self.state["budget"][faction] < cost:
            raise ContractViolation(f"{faction} budget {self.state['budget'][faction]} < cost {cost}",
                                    violation_code="ERR_CLASH_BUDGET")
        mutate()
        self.state["budget"][faction] -= cost
        rec = {"round": self.state["round"], "faction": faction, "action": desc, "frames": cost,
               "framework_step": FRAMEWORK_STEP[kind], **_pack_frame(f"{self.state['round']}|{faction}|{desc}")}
        self.state["log"].append(rec)
        self.state["log"] = self.state["log"][-40:]
        return rec

    # ------------------------------------------------------------------ actions
    def apply(self, action: Dict[str, Any], faction: Optional[str] = None) -> Dict[str, Any]:
        """Validate first, mutate only on success (Delta S == 0 on any violation)."""
        faction = faction or self.state["player_faction"]
        if self.state["winner"]:
            raise ContractViolation("Game over", violation_code="ERR_CLASH_OVER")
        kind = action.get("type")
        if kind == "end_turn":
            if faction != self.state["player_faction"]:
                raise ContractViolation("Only the player ends the turn", violation_code="ERR_CLASH_TURN")
            return self.end_turn()
        if kind in ("move", "attack"):
            u = self._unit(str(action.get("unit")))
            if u["faction"] != faction:
                raise ContractViolation("Not your unit", violation_code="ERR_CLASH_OWNER")
            if u["acted"]:
                raise ContractViolation("Unit already acted this round", violation_code="ERR_CLASH_ACTED")
            if kind == "move":
                to = tuple(int(v) for v in action.get("to", ()))
                if to not in self.cells or hex_distance(tuple(u["pos"]), to) != 1 or to in self._occupied():
                    raise ContractViolation(f"Illegal move to {to}", violation_code="ERR_CLASH_MOVE")

                def m():
                    u["pos"], u["acted"] = list(to), True
                return self._commit(faction, "move", ACTION_COST["move"], f"MOVE {u['unit_id']}->{to}", m)
            t = self._unit(str(action.get("target")))
            if t["faction"] == faction or hex_distance(tuple(u["pos"]), tuple(t["pos"])) != 1:
                raise ContractViolation("Target must be an adjacent enemy", violation_code="ERR_CLASH_TARGET")
            dmg = max(1, u["atk"] + u["tmp_atk"] - t["def"] - t["tmp_def"])

            def a():
                t["hp"] = max(0, min(VITAL_MAX_HP, t["hp"] - dmg))
                u["acted"] = True
            return self._commit(faction, "attack", ACTION_COST["attack"],
                                f"ATTACK {u['unit_id']}->{t['unit_id']} dmg={dmg}", a)
        if kind == "card":
            cid = str(action.get("card"))
            card = CARDS.get(cid)
            if card is None:
                raise ContractViolation(f"Unknown card {cid}", violation_code="ERR_CLASH_CARD")
            t = self._unit(str(action.get("target")))
            if t["faction"] != faction:
                raise ContractViolation("Cards target own units", violation_code="ERR_CLASH_CARD_TARGET")
            if cid == "FREIVALDS_AUDIT" and t["verdict"] != "CONFLICT":
                raise ContractViolation("Audit only repairs units in CONFLICT", violation_code="ERR_CLASH_AUDIT")

            def c():
                if cid == "HORIZON_SHIELD":
                    t["tmp_def"] += 1
                elif cid == "STRICT_JOIN":
                    for o in self.state["units"].values():
                        if o["faction"] == faction and o["hp"] > 0 and hex_distance(tuple(o["pos"]), tuple(t["pos"])) <= 1:
                            o["tmp_def"] += 1
                elif cid == "GPU_ALLOCATE":
                    t["tmp_atk"] += 1
                elif cid == "FREIVALDS_AUDIT":
                    t["hp"] = min(VITAL_MAX_HP, t["hp"] + 1)
            return self._commit(faction, "card", card["cost"], f"CARD {cid}->{t['unit_id']}", c)
        raise ContractViolation(f"Unknown action {kind}", violation_code="ERR_CLASH_ACTION")

    # ------------------------------------------------------------------ AI + round
    def _ai_turn(self, faction: str) -> None:
        for uid in sorted(u for u, v in self.state["units"].items() if v["faction"] == faction):
            u = self.state["units"][uid]
            if u["hp"] <= 0 or u["acted"] or self.state["budget"][faction] < 1:
                continue
            enemies = [v for v in self.state["units"].values() if v["faction"] != faction and v["hp"] > 0]
            if not enemies:
                return
            adj = sorted((e for e in enemies if hex_distance(tuple(u["pos"]), tuple(e["pos"])) == 1),
                         key=lambda e: (e["hp"], e["unit_id"]))
            if adj:
                self.apply({"type": "attack", "unit": uid, "target": adj[0]["unit_id"]}, faction)
                continue
            next_r = max(0, BOARD_RADIUS - self.state["round"] // 3)
            occ = self._occupied()
            options = []
            for dq, dr in DIRS:
                c = (u["pos"][0] + dq, u["pos"][1] + dr)
                if c in self.cells and c not in occ:
                    near = min(hex_distance(c, tuple(e["pos"])) for e in enemies)
                    options.append((hex_distance(c, (0, 0)) > next_r, near, c))
            if options:
                options.sort()
                self.apply({"type": "move", "unit": uid, "to": list(options[0][2])}, faction)

    def end_turn(self) -> Dict[str, Any]:
        for fac in FACTIONS:
            if fac != self.state["player_faction"]:
                self._ai_turn(fac)
        r = self.state["zone_radius"]
        zone_hits = []
        for u in self.state["units"].values():
            if u["hp"] > 0 and hex_distance(tuple(u["pos"]), (0, 0)) > r:
                u["hp"] = max(0, u["hp"] - 1)
                zone_hits.append(u["unit_id"])
        self.state["round"] += 1
        self.state["zone_radius"] = max(0, BOARD_RADIUS - (self.state["round"] - 1) // 3)
        self.state["budget"] = {f: ROUND_BUDGET_FRAMES for f in FACTIONS}
        for u in self.state["units"].values():
            u["tmp_def"] = u["tmp_atk"] = 0
            u["acted"] = False
        alive = {u["faction"] for u in self.state["units"].values() if u["hp"] > 0}
        if len(alive) <= 1:
            self.state["winner"] = next(iter(alive), "NONE")
        elif self.state["round"] > MAX_ROUNDS:
            totals = {f: sum(u["hp"] for u in self.state["units"].values() if u["faction"] == f) for f in FACTIONS}
            self.state["winner"] = max(sorted(totals), key=lambda f: totals[f])
        rec = {"round": self.state["round"] - 1, "faction": "ARENA", "action": f"END_TURN zone_hits={zone_hits}",
               "frames": 0, "framework_step": "KERNEL_STATE commit (round)", **_pack_frame(f"END|{self.state['round']}")}
        self.state["log"].append(rec)
        return rec

    def public_state(self) -> Dict[str, Any]:
        s = copy.deepcopy(self.state)
        s["cells"] = [list(c) for c in sorted(self.cells)]
        s["cards"] = CARDS
        s["factions"] = {f: {"color": v["color"], "checks": list(v["checks"])} for f, v in FACTIONS.items()}
        s["state_sha256"] = self.state_hash()
        return s


class ClashRegistry:
    """In-memory game store for the HTTP server (bounded, oldest evicted)."""

    def __init__(self, cap: int = 32) -> None:
        self._games: "OrderedDict[str, ClashGame]" = OrderedDict()
        self._lock = threading.Lock()
        self._cap = cap
        self._counter = 0

    def new(self, faction: str = "SYNC", seed: int = 6) -> Tuple[str, ClashGame]:
        g = ClashGame(faction, seed)
        with self._lock:
            self._counter += 1
            gid = f"G{self._counter:05d}"
            self._games[gid] = g
            while len(self._games) > self._cap:
                self._games.popitem(last=False)
        return gid, g

    def get(self, gid: str) -> ClashGame:
        with self._lock:
            g = self._games.get(gid)
        if g is None:
            raise ContractViolation(f"Unknown game {gid}", violation_code="ERR_CLASH_GAME")
        return g


GLOBAL_CLASH_REGISTRY = ClashRegistry()
