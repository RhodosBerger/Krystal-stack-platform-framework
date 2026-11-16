"""Unified arena caching, algebraic tensor movement, multiplayer entity scripting,
turn/render consumption graph + optimizer, .kopt optical binary codec, State A/B
editable CSS wireframes, lighting/object breakpoints, display sockets, and scaled
option-geometry holograms.
"""

from __future__ import annotations

import hashlib
import json
import math
import struct
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from krystal_kernel.constants import VITAL_MAX_HP, ContractViolation

from .clash_battle_royale import BOARD_RADIUS, CARDS, DIRS, FACTIONS, ClashGame, board_cells, hex_distance
from .combat_turn_planner import DEFAULT_CACHE_DIR
from .fact_synthesis_engine import REPORTS, run_fact_synthesis
from .intel_accelerator_and_power_thread_governor import get_shared_ov_core
from .neurosymbolic_genesis_discipline_and_llm import NeuroSymbolicMicroLLM

KOPT_MAGIC = b"KOPT0001"
KOPT_RECORD_FMT = "<4f4I"  # ior, r0, cos_theta, fresnel_r, lod_tier, rays, tris, flags = 32 bytes


# ---------------------------------------------------------------------------
# 1. Compact-Model (2B-class) Neuro-Symbolic Interaction Scaffolding
# ---------------------------------------------------------------------------

def evaluate_compact_model_interaction_scaffolding(prompt_prefix: str = "NumPy BLAS") -> Dict[str, Any]:
    """Combine local micro-transformer + deterministic fact routing to achieve verified domain answers."""
    syn = run_fact_synthesis(write=False)
    llm = NeuroSymbolicMicroLLM(d_model=32, max_seq_len=24, seed=6)
    train_meta = llm.train_and_compile(epochs=40, lr=0.08, device="CPU")
    gen = llm.generate_and_verify(prompt_prefix)

    return {
        "vital_max_hp": VITAL_MAX_HP,
        "architecture_principle": (
            "Namiesto ukladania faktov do desiatok miliárd váh (30B+) oddeľuje architektúra "
            "jazykovú formuláciu (kompaktný 2B/mikro-model) od deterministického výpočtu, citácií a brány verifikátora."
        ),
        "scope_boundary_note": (
            "V rámci overenej domény frameworku dosahuje výstup 0 % nepodložených tvrdení vďaka verify_answer(); "
            "mimo lokálnej bázy faktov sa model nesnaží nahrádzať všeobecnú svetovú encyklopédiu 30B+ modelov."
        ),
        "atomic_facts_indexed": len(syn["facts"]),
        "micro_llm_metrics": train_meta,
        "verified_interaction": gen,
    }


# ---------------------------------------------------------------------------
# 2. Battle Arena Cache Representation & Algebraic Tensor Navigation
# ---------------------------------------------------------------------------

class BattleArenaCacheAndTensorNavigator:
    """Caches 37-hex topology/adjacency tensors and computes movement via OpenVINO opset13 MatMul diffusion."""

    def __init__(self, radius: int = BOARD_RADIUS) -> None:
        self.radius = radius
        self.cells: List[Tuple[int, int]] = sorted(board_cells(radius))
        self.cell_to_idx: Dict[Tuple[int, int], int] = {c: i for i, c in enumerate(self.cells)}
        n = len(self.cells)
        adj = np.zeros((n, n), dtype=np.float32)
        for i, (q, r) in enumerate(self.cells):
            adj[i, i] = 1.0  # self-retention
            for dq, dr in DIRS:
                nb = (q + dq, r + dr)
                if nb in self.cell_to_idx:
                    adj[i, self.cell_to_idx[nb]] = 1.0
        # Row-normalized transition operator D^-1 A
        row_sums = np.sum(adj, axis=1, keepdims=True)
        self.transition_matrix: np.ndarray = (adj / np.maximum(row_sums, 1.0)).astype(np.float32)
        self.topology_sha256: str = hashlib.sha256(self.transition_matrix.tobytes()).hexdigest()
        self._state_cache: Dict[str, Dict[str, Any]] = {}

    def cache_arena_state(self, game: ClashGame) -> Dict[str, Any]:
        """Store compact tensor snapshot of ClashGame keyed by authoritative state_hash."""
        sh = game.state_hash()
        if sh in self._state_cache:
            entry = dict(self._state_cache[sh])
            entry["cache_hit"] = True
            return entry

        n = len(self.cells)
        occ_tensor = np.zeros((n, 3), dtype=np.float32)  # [hp_norm, atk, def]
        for u in game.state["units"].values():
            if u["hp"] > 0:
                idx = self.cell_to_idx[tuple(u["pos"])]
                occ_tensor[idx] = [u["hp"] / float(VITAL_MAX_HP), float(u["atk"]), float(u["def"])]

        snap = {
            "state_sha256": sh,
            "topology_sha256": self.topology_sha256,
            "round": game.state["round"],
            "zone_radius": game.state["zone_radius"],
            "occupancy_tensor_bytes": int(occ_tensor.nbytes),
            "active_units": int(np.count_nonzero(occ_tensor[:, 0])),
            "cache_hit": False,
        }
        self._state_cache[sh] = snap
        return snap

    def compute_algebraic_tensor_move(
        self,
        game: ClashGame,
        unit_id: str,
        diffusion_steps: int = 3,
        apply_move: bool = False,
    ) -> Dict[str, Any]:
        """Diffuse spatial potential V_{k+1} = T @ V_k in OpenVINO opset13 to choose optimal legal hex step."""
        import openvino.opset13 as ops

        u = game.state["units"].get(unit_id)
        if u is None or u["hp"] <= 0:
            raise ContractViolation(f"Unit {unit_id} missing or eliminated", violation_code="ERR_NAV_UNIT")

        n = len(self.cells)
        potential = np.zeros((n, 1), dtype=np.float32)
        zr = game.state["zone_radius"]
        for i, cell in enumerate(self.cells):
            d0 = hex_distance(cell, (0, 0))
            potential[i, 0] = (3.0 - 0.6 * d0) if d0 <= zr else -4.0

        for other in game.state["units"].values():
            if other["hp"] <= 0 or other["unit_id"] == unit_id:
                continue
            idx = self.cell_to_idx[tuple(other["pos"])]
            if other["faction"] != u["faction"]:
                # Attractive if weaker, repulsive if dangerous
                potential[idx, 0] += 2.5 if other["hp"] <= u["atk"] + 1 else 0.8
            else:
                potential[idx, 0] += 0.4

        ov, core, _ = get_shared_ov_core(DEFAULT_CACHE_DIR)
        v_param = ops.parameter([n, 1], np.float32, name="potential")
        t_const = ops.constant(self.transition_matrix)
        cur = v_param
        for _ in range(max(1, diffusion_steps)):
            cur = ops.matmul(t_const, cur, False, False)
        ov_model = ov.Model([cur], [v_param], "arena_tensor_navigator")
        compiled = core.compile_model(ov_model, "CPU", {"INFERENCE_PRECISION_HINT": "f32"})
        diffused = np.array(compiled.create_infer_request().infer({0: potential})[compiled.output(0)], copy=True).reshape(n)

        occupied = {tuple(v["pos"]) for v in game.state["units"].values() if v["hp"] > 0}
        pos = tuple(u["pos"])
        legal_neighbors: List[Tuple[float, Tuple[int, int]]] = []
        for dq, dr in DIRS:
            nb = (pos[0] + dq, pos[1] + dr)
            if nb in self.cell_to_idx and nb not in occupied:
                legal_neighbors.append((float(diffused[self.cell_to_idx[nb]]), nb))

        if not legal_neighbors:
            return {"unit_id": unit_id, "selected_to": None, "applied": False, "reason": "no_free_neighbor"}

        legal_neighbors.sort(key=lambda x: (-x[0], x[1][0], x[1][1]))
        best_score, best_cell = legal_neighbors[0]
        receipt = None
        if apply_move and not u["acted"]:
            receipt = game.apply({"type": "move", "unit": unit_id, "to": [best_cell[0], best_cell[1]]}, u["faction"])

        return {
            "vital_max_hp": VITAL_MAX_HP,
            "unit_id": unit_id,
            "from": list(pos),
            "selected_to": [best_cell[0], best_cell[1]],
            "tensor_potential_score": round(best_score, 4),
            "neighbor_candidates": [
                {"cell": [c[0], c[1]], "potential": round(sc, 4), "norm_scale_0_to_6": round(max(0.0, min(6.0, sc + 3.0)), 2)}
                for sc, c in legal_neighbors
            ],
            "applied": bool(receipt is not None),
            "receipt": receipt,
        }


# ---------------------------------------------------------------------------
# 3. Multiplayer Player-Entity Scripting & Turn/Render Consumption Graph
# ---------------------------------------------------------------------------

ENTITY_PERSONAS: Dict[str, str] = {
    "SYNC": "ZONE_CONTROL_TACTICIAN",
    "COMPUTE": "AGGRESSIVE_GPU_BURSTER",
    "VERIFY": "CONFLICT_PARITY_GUARDIAN",
    "OPS": "FRAME_BUDGET_CONSERVER",
}


class TurnConsumptionAndRenderAuditGraph:
    """Logs every multiplayer entity turn + allocation into a render/memory cost graph and optimizes expensive actions."""

    RENDER_COST_TABLE = {
        "move": {"tris": 240, "rays": 4, "shader_passes": 1},
        "attack": {"tris": 720, "rays": 18, "shader_passes": 3},
        "card:HORIZON_SHIELD": {"tris": 480, "rays": 12, "shader_passes": 2},
        "card:STRICT_JOIN": {"tris": 960, "rays": 24, "shader_passes": 3},
        "card:GPU_ALLOCATE": {"tris": 1200, "rays": 32, "shader_passes": 4},
        "card:FREIVALDS_AUDIT": {"tris": 640, "rays": 16, "shader_passes": 2},
    }

    def __init__(self) -> None:
        self.nodes: List[Dict[str, Any]] = []
        self.edges: List[Tuple[int, int]] = []

    @classmethod
    def _cost_score(cls, tris: int, rays: int, passes: int) -> float:
        return round(tris * 0.01 + rays * 0.5 + passes * 4.0, 3)

    def record_action(self, faction: str, action: Dict[str, Any], frames_spent: int) -> Dict[str, Any]:
        key = f"card:{action['card']}" if action["type"] == "card" else str(action["type"])
        spec = self.RENDER_COST_TABLE.get(key, {"tris": 240, "rays": 4, "shader_passes": 1})
        score = self._cost_score(spec["tris"], spec["rays"], spec["shader_passes"])
        idx = len(self.nodes)
        if idx > 0:
            self.edges.append((idx - 1, idx))
        node = {
            "node_id": idx,
            "faction": faction,
            "persona": ENTITY_PERSONAS.get(faction, "GENERIC_PLAYER"),
            "action_key": key,
            "action": action,
            "frames_128b_spent": frames_spent,
            "render_tris": spec["tris"],
            "render_rays": spec["rays"],
            "shader_passes": spec["shader_passes"],
            "render_cost_score": score,
        }
        self.nodes.append(node)
        return node

    def derive_render_optimization_procedure(self) -> Dict[str, Any]:
        """Identify the costliest combat render actions and compute Before -> After optimization savings."""
        if not self.nodes:
            return {"nodes_logged": 0, "most_expensive": [], "before_total_cost": 0.0, "after_total_cost": 0.0}

        ranked = sorted(self.nodes, key=lambda n: -n["render_cost_score"])
        before_total = sum(n["render_cost_score"] for n in self.nodes)
        after_total = 0.0
        steps: List[Dict[str, Any]] = []

        for n in self.nodes:
            if n["render_cost_score"] >= 20.0:
                # Apply instanced LOD1 + half-ray hologram clamp + fused single-pass shader
                opt_tris = int(n["render_tris"] * 0.45)
                opt_rays = max(4, n["render_rays"] // 2)
                opt_passes = max(1, n["shader_passes"] - 1)
                opt_score = self._cost_score(opt_tris, opt_rays, opt_passes)
                steps.append(
                    {
                        "node_id": n["node_id"],
                        "action_key": n["action_key"],
                        "before_cost": n["render_cost_score"],
                        "after_cost": opt_score,
                        "remediation": "LOD0->LOD1 hex instancing + 50% ray-bundle decimation + shader pass fusion",
                    }
                )
                after_total += opt_score
            else:
                after_total += n["render_cost_score"]

        return {
            "nodes_logged": len(self.nodes),
            "edges_logged": len(self.edges),
            "top_expensive_actions": ranked[:3],
            "before_total_render_cost": round(before_total, 3),
            "after_total_render_cost": round(after_total, 3),
            "render_cost_reduction_pct": round(100.0 * (1.0 - after_total / max(before_total, 1e-9)), 2),
            "optimization_procedure_steps": steps,
        }


def run_multiplayer_entity_script_round(seed: int = 6) -> Dict[str, Any]:
    """Run scripted multiplayer player-entities across all 4 factions and audit consumption + render cost."""
    game = ClashGame("COMPUTE", seed=seed)
    nav = BattleArenaCacheAndTensorNavigator()
    graph = TurnConsumptionAndRenderAuditGraph()

    # Player-entity COMPUTE uses GPU_ALLOCATE card + tensor-navigated move
    rec_card = game.apply({"type": "card", "card": "GPU_ALLOCATE", "target": "C03"}, "COMPUTE")
    graph.record_action("COMPUTE", {"type": "card", "card": "GPU_ALLOCATE", "target": "C03"}, int(rec_card["frames"]))
    mv = nav.compute_algebraic_tensor_move(game, "C03", apply_move=True)
    if mv["applied"] and mv["receipt"]:
        graph.record_action("COMPUTE", {"type": "move", "unit": "C03", "to": mv["selected_to"]}, int(mv["receipt"]["frames"]))

    # Other 3 multiplayer entity personas act via deterministic scripts
    for fac, card_id, uid in [("SYNC", "HORIZON_SHIELD", "C01"), ("VERIFY", "STRICT_JOIN", "C05"), ("OPS", "HORIZON_SHIELD", "C07")]:
        rc = game.apply({"type": "card", "card": card_id, "target": uid}, fac)
        graph.record_action(fac, {"type": "card", "card": card_id, "target": uid}, int(rc["frames"]))
        mv_f = nav.compute_algebraic_tensor_move(game, uid, apply_move=True)
        if mv_f["applied"] and mv_f["receipt"]:
            graph.record_action(fac, {"type": "move", "unit": uid, "to": mv_f["selected_to"]}, int(mv_f["receipt"]["frames"]))

    cache_first = nav.cache_arena_state(game)
    cache_second = nav.cache_arena_state(game)
    audit = graph.derive_render_optimization_procedure()
    return {
        "vital_max_hp": VITAL_MAX_HP,
        "entity_personas": ENTITY_PERSONAS,
        "arena_cache_probe": {"first_hit": cache_first["cache_hit"], "second_hit": cache_second["cache_hit"], "state_sha256": cache_second["state_sha256"]},
        "consumption_audit": audit,
    }


# ---------------------------------------------------------------------------
# 4. Optical Knowledge Synthesis & .kopt Little-Endian Binary Codec
# ---------------------------------------------------------------------------

def synthesize_and_write_optical_binary(
    path: Path = REPORTS / "krystal_arena_optical.kopt",
) -> Dict[str, Any]:
    """Compute Schlick Fresnel + hologram LOD optical records and pack into a SHA-256-verified .kopt binary."""
    if struct.calcsize(KOPT_RECORD_FMT) != 32:
        raise ContractViolation("KOPT_RECORD_FMT size != 32 B", violation_code="ERR_KOPT_FMT_SIZE")
    materials = [
        ("BASALT_HEX_MATTE", 1.52, 0.75, 0, 8, 240, 1),
        ("CYAN_HOLOGRAM_GLASS", 1.45, 0.35, 0, 20, 480, 3),
        ("GOLD_AUTHORITY_SPECULAR", 2.40, 0.60, 1, 12, 96, 5),
        ("CONFLICT_MAGENTA_PLASMA", 1.33, 0.20, 1, 16, 120, 7),
    ]

    records_meta: List[Dict[str, Any]] = []
    payload_chunks: List[bytes] = []
    for name, ior, cos_th, lod, rays, tris, flags in materials:
        r0 = ((1.0 - ior) / (1.0 + ior)) ** 2
        fresnel = r0 + (1.0 - r0) * ((1.0 - cos_th) ** 5)  # Schlick's approximation
        rec_bytes = struct.pack(KOPT_RECORD_FMT, ior, r0, cos_th, fresnel, lod, rays, tris, flags)
        payload_chunks.append(rec_bytes)
        records_meta.append(
            {"material": name, "ior": round(ior, 3), "r0": round(r0, 5), "fresnel_reflectance": round(fresnel, 5), "lod": lod, "rays": rays, "tris": tris}
        )

    header = KOPT_MAGIC + struct.pack("<II", 1, len(payload_chunks))
    body = header + b"".join(payload_chunks)
    digest = hashlib.sha256(body).digest()
    full_blob = body + digest

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(full_blob)
    return {
        "path": str(path),
        "total_bytes": len(full_blob),
        "record_count": len(payload_chunks),
        "sha256": digest.hex(),
        "records": records_meta,
    }


def read_optical_binary(path: Path = REPORTS / "krystal_arena_optical.kopt") -> Dict[str, Any]:
    """Read and cryptographically verify a .kopt optical binary container."""
    raw = path.read_bytes()
    if len(raw) < 16 + 32 or raw[:8] != KOPT_MAGIC:
        raise ContractViolation("Invalid .kopt header or magic", violation_code="ERR_KOPT_MAGIC")
    body, stored_sha = raw[:-32], raw[-32:]
    if hashlib.sha256(body).digest() != stored_sha:
        raise ContractViolation(".kopt SHA-256 integrity mismatch", violation_code="ERR_KOPT_HASH")
    version, count = struct.unpack("<II", body[8:16])
    expected_len = 16 + count * 32
    if len(body) != expected_len:
        raise ContractViolation(".kopt truncated record table", violation_code="ERR_KOPT_LEN")

    decoded = []
    for i in range(count):
        off = 16 + i * 32
        ior, r0, cos_th, fresnel, lod, rays, tris, flags = struct.unpack(KOPT_RECORD_FMT, body[off : off + 32])
        decoded.append(
            {
                "index": i,
                "ior": round(ior, 4),
                "r0": round(r0, 5),
                "cos_theta": round(cos_th, 4),
                "fresnel_reflectance": round(fresnel, 5),
                "lod": lod,
                "rays": rays,
                "tris": tris,
                "flags": flags,
            }
        )
    return {"version": version, "record_count": count, "sha256_verified": True, "records": decoded}


# ---------------------------------------------------------------------------
# 5. Breakpoints, Display Sockets, Option-Geometry Holograms & State A/B Wireframe UI
# ---------------------------------------------------------------------------

LIGHTING_AND_VIEW_BREAKPOINTS: Tuple[Dict[str, Any], ...] = (
    {"bp_id": "BP_360_MONO_TACTICAL", "min_px": 360, "lighting": "DIRECTIONAL_FLAT_KEY", "socket": "SOCKET_WIREFRAME_PRISM"},
    {"bp_id": "BP_768_AMBER_DUEL", "min_px": 768, "lighting": "WARM_AMBER_CONTRA_RIM", "socket": "SOCKET_VOXEL_PILLAR"},
    {"bp_id": "BP_1280_CYAN_STUDIO", "min_px": 1280, "lighting": "THREE_POINT_CYAN_GOLD", "socket": "SOCKET_HOLOGRAM_GLYPH"},
    {"bp_id": "BP_1920_WEBXR_DOME", "min_px": 1920, "lighting": "HEMISPHERIC_FRESNEL_DOME", "socket": "SOCKET_HOLOGRAM_GLYPH"},
)

DISPLAY_SOCKETS: Dict[str, Dict[str, Any]] = {
    "SOCKET_WIREFRAME_PRISM": {"object_mesh": "6-hranný drôtený hranol", "tris_per_unit": 24, "light_rig": "1x kľúčové svetlo"},
    "SOCKET_VOXEL_PILLAR": {"object_mesh": "Vrstvený voxelový stĺp podľa HP [1..6]", "tris_per_unit": 96, "light_rig": "Jantárové protisvetlo"},
    "SOCKET_HOLOGRAM_GLYPH": {"object_mesh": "Rotujúci Fresnelov torus + lúčový zväzok", "tris_per_unit": 240, "light_rig": "Trojbodový azúrovo-zlatý rig"},
}


def build_option_geometry_holograms(seed: int = 6) -> Dict[str, Any]:
    """Build geometric hologram descriptors for arena options mapped onto the [0..6] VITAL_MAX_HP scale."""
    game = ClashGame("SYNC", seed=seed)
    nav = BattleArenaCacheAndTensorNavigator()
    mv = nav.compute_algebraic_tensor_move(game, "C01", apply_move=False)
    holograms = []
    for idx, cand in enumerate(mv["neighbor_candidates"]):
        scale_val = float(cand["norm_scale_0_to_6"])
        holograms.append(
            {
                "option_id": f"OPT_HEX_{cand['cell'][0]}_{cand['cell'][1]}",
                "target_cell": cand["cell"],
                "geometry_primitive": "HEX_PYRAMID_HOLOGRAM" if idx == 0 else "WIRE_RING_MARKER",
                "scale_0_to_6": scale_val,
                "height_m": round(0.25 + 0.22 * scale_val, 3),
                "recommended": idx == 0,
            }
        )
    return {
        "vital_max_hp": VITAL_MAX_HP,
        "scale_bounds": [0, VITAL_MAX_HP],
        "breakpoints": list(LIGHTING_AND_VIEW_BREAKPOINTS),
        "sockets": DISPLAY_SOCKETS,
        "option_holograms": holograms,
    }


def render_wireframe_sockets_and_holograms_html() -> str:
    """Render editable State A (Before) / State B (After) CSS wireframe + socket switcher UI."""
    mp = run_multiplayer_entity_script_round(seed=6)
    holo = build_option_geometry_holograms(seed=6)
    data_json = json.dumps({"mp": mp, "holo": holo}, ensure_ascii=False)
    return f"""<!DOCTYPE html>
<html lang="sk">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>Krystal · Wireframe State A/B, Sokety &amp; Hologramy Možností</title>
<style>
  :root {{
    --bg:#070812; --surface:#11131D; --wire:#58D8FF; --accent:#D8AE4B; --txt:#F1EFE8; --muted:#8D8997; --stroke:1px;
  }}
  body.state-b {{
    --bg:#030406; --surface:#0C1418; --wire:#50EEE0; --accent:#43D264; --stroke:2px;
  }}
  * {{ box-sizing:border-box; }}
  body {{ margin:0; background:var(--bg); color:var(--txt); font-family:"JetBrains Mono",Consolas,monospace; font-size:12px; padding:24px; }}
  header {{ display:flex; flex-wrap:wrap; gap:12px; align-items:center; border-bottom:var(--stroke) solid var(--wire); padding-bottom:14px; margin-bottom:20px; }}
  .title {{ font-size:18px; color:var(--accent); font-weight:700; }}
  .grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(300px,1fr)); gap:16px; }}
  .frame {{ background:var(--surface); border:var(--stroke) dashed var(--wire); border-radius:2px; padding:14px; }}
  .frame h3 {{ margin:0 0 10px; color:var(--wire); font-size:12px; text-transform:uppercase; letter-spacing:0.12em; }}
  .bar {{ height:10px; border:1px solid var(--wire); margin-top:4px; }}
  .bar > div {{ height:100%; background:var(--accent); }}
  button, select {{ background:transparent; color:var(--txt); border:1px solid var(--wire); border-radius:2px; padding:6px 12px; cursor:pointer; font-family:inherit; }}
</style>
</head>
<body>
<header>
  <div>
    <div style="font-size:10px;color:var(--muted)">EDITOVATEĽNÝ WIREFRAME · STATE A (PRED) / STATE B (PO OPTIMALIZÁCII)</div>
    <div class="title">Sokety Bojového Deja, Breakpointy Svetla &amp; Hologramy Možností</div>
  </div>
  <div style="margin-left:auto;display:flex;gap:8px">
    <button id="btnState">Prepnúť State A / State B (Before/After)</button>
    <select id="selSocket">
      <option value="SOCKET_WIREFRAME_PRISM">SOCKET_WIREFRAME_PRISM</option>
      <option value="SOCKET_VOXEL_PILLAR">SOCKET_VOXEL_PILLAR</option>
      <option value="SOCKET_HOLOGRAM_GLYPH">SOCKET_HOLOGRAM_GLYPH</option>
    </select>
  </div>
</header>
<div class="grid" id="out"></div>
<script>
const D = {data_json};
let isAfter = false;
function draw() {{
  const sockId = document.getElementById("selSocket").value;
  const sock = D.holo.sockets[sockId];
  const audit = D.mp.consumption_audit;
  const costVal = isAfter ? audit.after_total_render_cost : audit.before_total_render_cost;
  document.getElementById("out").innerHTML =
    `<div class="frame"><h3>Aktívny Soket &amp; Svetlo (${{isAfter ? "STATE B · AFTER" : "STATE A · BEFORE"}})</h3>` +
    `<p>Soket: <b>${{sockId}}</b> (${{sock.object_mesh}})</p>` +
    `<p>Svetelný rig: ${{sock.light_rig}} · Trojuholníky/jednotku: ${{sock.tris_per_unit}}</p>` +
    `<p>Celková cena renderu kola: <b>${{costVal}}</b> (úspora ${{audit.render_cost_reduction_pct}} %)</p></div>` +
    `<div class="frame"><h3>Hologramy geometrie možností (škála 0 .. ${{D.holo.vital_max_hp}})</h3>` +
    D.holo.option_holograms.map(h =>
      `<div style="margin-bottom:10px">${{h.option_id}} · ${{h.geometry_primitive}} · <b>${{h.scale_0_to_6}} / 6</b>` +
      `<div class="bar"><div style="width:${{(h.scale_0_to_6 / 6)*100}}%"></div></div></div>`).join("") + `</div>` +
    `<div class="frame"><h3>Breakpointy zobrazenia</h3>` +
    D.holo.breakpoints.map(b => `<div>&ge; ${{b.min_px}}px: ${{b.bp_id}} -> ${{b.lighting}} (${{b.socket}})</div>`).join("") + `</div>`;
}}
document.getElementById("btnState").onclick = () => {{ isAfter = !isAfter; document.body.classList.toggle("state-b", isAfter); draw(); }};
document.getElementById("selSocket").onchange = draw;
draw();
</script>
</body>
</html>
"""
