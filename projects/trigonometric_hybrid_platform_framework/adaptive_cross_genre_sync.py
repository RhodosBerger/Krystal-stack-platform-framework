"""Adaptive Cross-Genre Synchronization Module.

Defines:
  1. Typed connection registry (nodes + edges) with authority validation.
  2. Per-genre sync policy and strictest-policy join for cross-genre interactions.
  3. Adaptive rollback-horizon controller (EWMA latency + 2 sigma, 12-tick down-hysteresis).
  4. Deterministic rollback simulation used as the cheapest discriminating test.

Evidence class of the simulation: SIMULATED_DETERMINISTIC (synthetic latency,
no real network). It verifies the rollback/re-simulation logic, not WAN behavior.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from collections import deque
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from krystal_kernel.constants import VITAL_MAX_HP, ContractViolation

TICK_MS: float = 1000.0 / 60.0  # authoritative 60 Hz tick
WIRE_FRAME_BYTES: int = 128
HORIZON_MIN: int = 1
HORIZON_MAX: int = 8
DOWNSHIFT_DWELL_TICKS: int = 12
HEX_CELLS: int = 19


# ---------------------------------------------------------------------------
# 1. SYNC POLICY PER GENRE + STRICTEST JOIN
# ---------------------------------------------------------------------------

class SyncMode(str, Enum):
    ROLLBACK = "ROLLBACK"
    INPUT_DELAY_LOCKSTEP = "INPUT_DELAY_LOCKSTEP"
    TURN_LOCKSTEP = "TURN_LOCKSTEP"
    AUTHORITATIVE_LEDGER = "AUTHORITATIVE_LEDGER"
    OBSERVER_INTERP = "OBSERVER_INTERP"


# Higher rank = stricter consistency requirement. Observers cannot write.
_STRICTNESS: Dict[SyncMode, int] = {
    SyncMode.ROLLBACK: 1,
    SyncMode.INPUT_DELAY_LOCKSTEP: 2,
    SyncMode.TURN_LOCKSTEP: 3,
    SyncMode.AUTHORITATIVE_LEDGER: 4,
}

GENRE_SYNC_POLICY: Dict[str, SyncMode] = {
    "GENRE_01_TACTICAL_HEX_CARD": SyncMode.TURN_LOCKSTEP,
    "GENRE_02_RTS_STRATEGIC_3D_GRID": SyncMode.INPUT_DELAY_LOCKSTEP,
    "GENRE_03_ACTION_BALLISTIC_6D": SyncMode.ROLLBACK,
    "GENRE_04_RPG_QUEST_NARRATIVE": SyncMode.AUTHORITATIVE_LEDGER,
    "GENRE_05_SIMULATION_DIGITAL_TWIN": SyncMode.OBSERVER_INTERP,
}


def join_sync_policy(genre_a: str, genre_b: str) -> SyncMode:
    """Shared-entity interaction uses the strictest policy of the participants."""
    for g in (genre_a, genre_b):
        if g not in GENRE_SYNC_POLICY:
            raise ContractViolation(f"Unknown genre: {g}", violation_code="ERR_SYNC_UNKNOWN_GENRE")
    mode_a, mode_b = GENRE_SYNC_POLICY[genre_a], GENRE_SYNC_POLICY[genre_b]
    if SyncMode.OBSERVER_INTERP in (mode_a, mode_b):
        raise ContractViolation(
            "Observer genre cannot participate in a state-writing interaction (Delta S == 0).",
            violation_code="ERR_SYNC_OBSERVER_WRITE",
        )
    return mode_a if _STRICTNESS[mode_a] >= _STRICTNESS[mode_b] else mode_b


# ---------------------------------------------------------------------------
# 2. CONNECTION REGISTRY (NODES + EDGES) WITH AUTHORITY VALIDATION
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SyncNode:
    node_id: str
    role: str  # CLIENT | OBSERVER | CONTROL | AUTHORITY | STATE | PROJECTION
    anchor: str


@dataclass(frozen=True)
class SyncConnection:
    conn_id: str
    source: str
    target: str
    payload: str
    transport: str  # WIRE_128B | LOCAL_CALL
    frame_bytes: int
    on_failure: str


SYNC_NODES: Tuple[SyncNode, ...] = (
    SyncNode("CLIENT_ACTION_6D", "CLIENT", "GoniometricProposerKernel (BALLISTIC / TRAJECTORY)"),
    SyncNode("CLIENT_RTS_GRID", "CLIENT", "Gamesa3DGridVulkanStrategicParserEngine"),
    SyncNode("CLIENT_HEX_CARD", "CLIENT", "Posledni Kmen ruleset (9 tiles / 19 hexes)"),
    SyncNode("CLIENT_RPG_QUEST", "CLIENT", "HybridLoreAggregatorRuntime"),
    SyncNode("CLIENT_TWIN_OBSERVER", "OBSERVER", "BlenderCoreStudioBridge / RSS twin"),
    SyncNode("PAIR_JOIN_ARBITER", "CONTROL", "join_sync_policy()"),
    SyncNode("JITTER_METER", "CONTROL", "EWMA latency estimator (cf. RecurrentVariationsStreamDecoder)"),
    SyncNode("POLICY_CONTROLLER", "CONTROL", "AdaptiveHorizonController"),
    SyncNode("STEP_GATE_PROXY", "AUTHORITY", "StepExecutionGateProxy"),
    SyncNode("KERNEL_STATE", "STATE", "Authoritative state S"),
    SyncNode("DISPLAY_PROJECTION", "PROJECTION", "Render(S) != S, 5 display surfaces"),
    SyncNode("RSS_TELEMETRY", "PROJECTION", "/api/v1/runtime/rss_feed.xml"),
)

_CLIENTS = ("CLIENT_ACTION_6D", "CLIENT_RTS_GRID", "CLIENT_HEX_CARD", "CLIENT_RPG_QUEST")

SYNC_CONNECTIONS: Tuple[SyncConnection, ...] = tuple(
    [
        SyncConnection(
            f"E0{i + 1}", c, "PAIR_JOIN_ARBITER", "action proposal (input, tick, genre)",
            "WIRE_128B", WIRE_FRAME_BYTES, "drop frame, client keeps prediction",
        )
        for i, c in enumerate(_CLIENTS)
    ]
    + [
        SyncConnection("E05", "PAIR_JOIN_ARBITER", "JITTER_METER", "arrival timestamps", "LOCAL_CALL", 0,
                       "hold last estimate"),
        SyncConnection("E06", "JITTER_METER", "POLICY_CONTROLLER", "EWMA mean and sigma (ms)", "LOCAL_CALL", 0,
                       "hold current horizon"),
        SyncConnection("E07", "POLICY_CONTROLLER", "PAIR_JOIN_ARBITER", "rollback horizon / input delay",
                       "LOCAL_CALL", 0, "fallback to HORIZON_MAX"),
        SyncConnection("E08", "PAIR_JOIN_ARBITER", "STEP_GATE_PROXY", "joined proposal + strictest SyncMode",
                       "LOCAL_CALL", 0, "reject, Delta S == 0"),
        SyncConnection("E09", "STEP_GATE_PROXY", "KERNEL_STATE", "validated commit", "LOCAL_CALL", 0,
                       "no commit, Delta S == 0"),
        SyncConnection("E10", "KERNEL_STATE", "DISPLAY_PROJECTION", "confirmed snapshot + checksum", "LOCAL_CALL",
                       0, "display keeps last confirmed frame"),
        SyncConnection("E11", "KERNEL_STATE", "RSS_TELEMETRY", "commit receipts", "LOCAL_CALL", 0,
                       "buffer receipts"),
    ]
    + [
        SyncConnection(
            f"E{12 + i}", "DISPLAY_PROJECTION", c, "confirmed tick + state checksum",
            "WIRE_128B", WIRE_FRAME_BYTES, "client resyncs from next snapshot",
        )
        for i, c in enumerate(_CLIENTS + ("CLIENT_TWIN_OBSERVER",))
    ]
)


def validate_connection_graph(
    nodes: Tuple[SyncNode, ...] = SYNC_NODES,
    edges: Tuple[SyncConnection, ...] = SYNC_CONNECTIONS,
) -> Dict[str, Any]:
    """Validates authority rules; raises ContractViolation on the first breach."""
    by_id = {n.node_id: n for n in nodes}
    ids = [e.conn_id for e in edges]
    if len(ids) != len(set(ids)):
        raise ContractViolation("Duplicate connection IDs", violation_code="ERR_SYNC_DUP_EDGE")
    adjacency: Dict[str, List[str]] = {n: [] for n in by_id}
    for e in edges:
        if e.source not in by_id or e.target not in by_id:
            raise ContractViolation(f"{e.conn_id}: unknown endpoint", violation_code="ERR_SYNC_UNKNOWN_NODE")
        if e.source == e.target:
            raise ContractViolation(f"{e.conn_id}: self loop", violation_code="ERR_SYNC_SELF_LOOP")
        if e.transport == "WIRE_128B" and e.frame_bytes != WIRE_FRAME_BYTES:
            raise ContractViolation(f"{e.conn_id}: wire frame != 128B", violation_code="ERR_SYNC_FRAME_SIZE")
        if e.target == "KERNEL_STATE" and e.source != "STEP_GATE_PROXY":
            raise ContractViolation(
                f"{e.conn_id}: only STEP_GATE_PROXY may write KERNEL_STATE",
                violation_code="ERR_SYNC_AUTHORITY_BYPASS",
            )
        src_role = by_id[e.source].role
        if src_role == "CLIENT" and e.target != "PAIR_JOIN_ARBITER":
            raise ContractViolation(
                f"{e.conn_id}: clients may only send proposals to PAIR_JOIN_ARBITER",
                violation_code="ERR_SYNC_CLIENT_ROUTE",
            )
        if src_role == "OBSERVER":
            raise ContractViolation(f"{e.conn_id}: observer cannot emit", violation_code="ERR_SYNC_OBSERVER_WRITE")
        adjacency[e.source].append(e.target)

    # Every writing client must reach KERNEL_STATE through the gate.
    for n in nodes:
        if n.role != "CLIENT":
            continue
        seen, queue = {n.node_id}, deque([n.node_id])
        while queue:
            cur = queue.popleft()
            for nxt in adjacency[cur]:
                if nxt not in seen:
                    seen.add(nxt)
                    queue.append(nxt)
        if "KERNEL_STATE" not in seen or "STEP_GATE_PROXY" not in seen:
            raise ContractViolation(f"{n.node_id}: no gated path to KERNEL_STATE",
                                    violation_code="ERR_SYNC_UNREACHABLE")
    return {
        "node_count": len(nodes),
        "edge_count": len(edges),
        "wire_edges": sum(1 for e in edges if e.transport == "WIRE_128B"),
        "graph_valid": True,
    }


# ---------------------------------------------------------------------------
# 3. ADAPTIVE HORIZON CONTROLLER
# ---------------------------------------------------------------------------

class AdaptiveHorizonController:
    """horizon = clamp(ceil((ewma_mean + 2*sigma) / TICK_MS), 1, 8).

    Upshift immediately; downshift by 1 only after 12 consecutive ticks with a lower target.
    """

    def __init__(self, alpha: float = 0.1, initial_horizon: int = 2) -> None:
        self.alpha = alpha
        self.mean: Optional[float] = None
        self.var: float = 0.0
        self.horizon = initial_horizon
        self._below_count = 0

    def observe(self, one_way_latency_ms: float) -> None:
        if self.mean is None:
            self.mean = one_way_latency_ms
            return
        diff = one_way_latency_ms - self.mean
        self.mean += self.alpha * diff
        # Upper semi-variance: only late arrivals widen the horizon. A latency drop
        # must not cause an upshift (observed failure mode of symmetric EWMA variance).
        if diff > 0:
            self.var = (1.0 - self.alpha) * (self.var + self.alpha * diff * diff)
        else:
            self.var = (1.0 - self.alpha) * self.var

    def target(self) -> int:
        if self.mean is None:
            return self.horizon
        raw = math.ceil((self.mean + 2.0 * math.sqrt(self.var)) / TICK_MS)
        return max(HORIZON_MIN, min(HORIZON_MAX, raw))

    def tick(self) -> int:
        tgt = self.target()
        if tgt > self.horizon:
            self.horizon = tgt
            self._below_count = 0
        elif tgt < self.horizon:
            self._below_count += 1
            if self._below_count >= DOWNSHIFT_DWELL_TICKS:
                self.horizon -= 1
                self._below_count = 0
        else:
            self._below_count = 0
        return self.horizon


# ---------------------------------------------------------------------------
# 4. DETERMINISTIC ROLLBACK SIMULATION (CHEAPEST DISCRIMINATING TEST)
# ---------------------------------------------------------------------------

State = Tuple[Tuple[int, int], Tuple[int, int]]  # ((pos_a, hp_a), (pos_b, hp_b))


def _step(state: State, inputs: Tuple[int, int]) -> State:
    """0 idle, 1 move, 2 attack, 3 rest. Simultaneous resolution, HP in [0, 6]."""
    (pa, ha), (pb, hb) = state
    ia, ib = inputs
    pa2 = (pa + 1) % HEX_CELLS if ia == 1 else pa
    pb2 = (pb + 1) % HEX_CELLS if ib == 1 else pb
    ha2 = min(VITAL_MAX_HP, max(0, ha - (1 if ib == 2 else 0) + (1 if ia == 3 else 0)))
    hb2 = min(VITAL_MAX_HP, max(0, hb - (1 if ia == 2 else 0) + (1 if ib == 3 else 0)))
    return ((pa2, ha2), (pb2, hb2))


def _checksum(state: State) -> str:
    return hashlib.sha256(json.dumps(state).encode("utf-8")).hexdigest()


def simulate_adaptive_rollback(
    ticks: int = 2000,
    one_way_latency_ms: float = 40.0,
    jitter_sigma_ms: float = 6.0,
    seed: int = 6,
) -> Dict[str, Any]:
    """Client A predicts remote inputs of B, rolls back on misprediction, stalls beyond horizon.

    Pass: zero checksum desync against the authoritative kernel trace on every tick.
    """
    rng = random.Random(seed)
    inputs_a = [rng.choice((0, 1, 2, 3)) for _ in range(ticks)]
    inputs_b = [rng.choice((0, 1, 2, 3)) for _ in range(ticks)]
    latencies = [max(0.0, rng.gauss(one_way_latency_ms, jitter_sigma_ms)) for _ in range(ticks)]

    state0: State = ((0, VITAL_MAX_HP), (9, VITAL_MAX_HP))
    kernel_sums: List[str] = []
    s = state0
    for k in range(ticks):
        s = _step(s, (inputs_a[k], inputs_b[k]))
        kernel_sums.append(_checksum(s))

    arrivals: Dict[int, List[int]] = {}
    for k in range(ticks):
        arrivals.setdefault(k + math.ceil(latencies[k] / TICK_MS), []).append(k)

    ctrl = AdaptiveHorizonController()
    known_b: Dict[int, int] = {}
    used_b: List[int] = []
    snapshots: List[State] = [state0]
    t_local = 0
    confirmed = 0
    stalls = rollbacks = resim_ticks = 0
    max_h = ctrl.horizon
    max_hp_seen = VITAL_MAX_HP

    def predict(k: int) -> int:
        for j in range(k - 1, -1, -1):
            if j in known_b:
                return known_b[j]
        return 0

    wall = 0
    wall_cap = ticks * 20 + 1000
    while True:
        rollback_from: Optional[int] = None
        for k in arrivals.get(wall, []):
            known_b[k] = inputs_b[k]
            ctrl.observe(latencies[k])
            if k < t_local and used_b[k] != inputs_b[k]:
                rollback_from = k if rollback_from is None else min(rollback_from, k)
        if rollback_from is not None:
            rollbacks += 1
            for j in range(rollback_from, t_local):
                b_val = known_b.get(j, predict(j))
                used_b[j] = b_val
                snapshots[j + 1] = _step(snapshots[j], (inputs_a[j], b_val))
                resim_ticks += 1
        while confirmed < ticks and confirmed in known_b:
            confirmed += 1

        max_h = max(max_h, ctrl.tick())

        if t_local < ticks:
            if t_local - confirmed < ctrl.horizon:
                b_val = known_b.get(t_local, predict(t_local))
                used_b.append(b_val)
                nxt = _step(snapshots[t_local], (inputs_a[t_local], b_val))
                max_hp_seen = max(max_hp_seen, nxt[0][1], nxt[1][1])
                snapshots.append(nxt)
                t_local += 1
            else:
                stalls += 1

        if t_local == ticks and confirmed == ticks:
            break
        wall += 1
        if wall > wall_cap:
            raise ContractViolation("Simulation did not converge", violation_code="ERR_SYNC_NO_CONVERGENCE")

    if max_hp_seen > VITAL_MAX_HP:
        raise ContractViolation("HP invariant breached in simulation", violation_code="ERR_SYNC_HP_BREACH")

    desync = sum(1 for k in range(ticks) if _checksum(snapshots[k + 1]) != kernel_sums[k])
    return {
        "ticks": ticks,
        "one_way_latency_ms": one_way_latency_ms,
        "jitter_sigma_ms": jitter_sigma_ms,
        "seed": seed,
        "wall_ticks_elapsed": wall,
        "stalls": stalls,
        "rollback_events": rollbacks,
        "resimulated_ticks": resim_ticks,
        "final_horizon": ctrl.horizon,
        "max_horizon": max_h,
        "desync_ticks": desync,
        "max_hp_observed": max_hp_seen,
        "kernel_final_checksum": kernel_sums[-1],
        "client_final_checksum": _checksum(snapshots[-1]),
        "evidence_level": "SIMULATED_DETERMINISTIC_SYNTHETIC_LATENCY",
    }


def describe_adaptive_sync_architecture() -> Dict[str, Any]:
    return {
        "vital_max_hp": VITAL_MAX_HP,
        "nodes": [asdict(n) for n in SYNC_NODES],
        "connections": [asdict(e) for e in SYNC_CONNECTIONS],
        "graph_validation": validate_connection_graph(),
        "genre_sync_policy": {g: m.value for g, m in GENRE_SYNC_POLICY.items()},
    }
