"""Standalone card grid for the adaptive cross-genre sync architecture.

Follows .agents/rules/krystal-ui-system-ai-style-contract.md:
  - hierarchy WORLD -> FACTION -> SYSTEM -> ACTION -> TELEMETRY
  - three display modes: World / Arena / Hybrid
  - KSArtifactCard: 1px border, 2px radius, no shadows, no glass, 4px spacing grid
  - metric tags [MEASURED] / [MODELED] / [TARGET]
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List

from krystal_kernel.constants import VITAL_MAX_HP

from .adaptive_cross_genre_sync import (
    _STRICTNESS,
    GENRE_SYNC_POLICY,
    SYNC_CONNECTIONS,
    SYNC_NODES,
    SyncMode,
    simulate_adaptive_rollback,
    validate_connection_graph,
)

_ROLE_LAYER = {
    "STATE": "WORLD",
    "PROJECTION": "WORLD",
    "CLIENT": "FACTION",
    "OBSERVER": "FACTION",
    "CONTROL": "SYSTEM",
    "AUTHORITY": "SYSTEM",
}

_CLIENT_GENRE = {
    "CLIENT_HEX_CARD": "GENRE_01_TACTICAL_HEX_CARD",
    "CLIENT_RTS_GRID": "GENRE_02_RTS_STRATEGIC_3D_GRID",
    "CLIENT_ACTION_6D": "GENRE_03_ACTION_BALLISTIC_6D",
    "CLIENT_RPG_QUEST": "GENRE_04_RPG_QUEST_NARRATIVE",
    "CLIENT_TWIN_OBSERVER": "GENRE_05_SIMULATION_DIGITAL_TWIN",
}

SIM_PROFILES = (
    ("LAN", 10.0, 2.0),
    ("REGIONAL", 40.0, 6.0),
    ("WAN", 100.0, 20.0),
)


@lru_cache(maxsize=32)
def _cached_simulation(ticks: int, latency_ms: float, jitter_ms: float, seed: int) -> Dict[str, Any]:
    """The simulation is fully deterministic for a given argument tuple, so recomputing it
    on every page request was redundant. Results are read-only for callers."""
    return simulate_adaptive_rollback(ticks=ticks, one_way_latency_ms=latency_ms, jitter_sigma_ms=jitter_ms, seed=seed)


def build_card_dataset(sim_ticks: int = 1000) -> Dict[str, Any]:
    graph = validate_connection_graph()
    cards: List[Dict[str, Any]] = []

    out_deg = {n.node_id: 0 for n in SYNC_NODES}
    in_deg = {n.node_id: 0 for n in SYNC_NODES}
    for e in SYNC_CONNECTIONS:
        out_deg[e.source] += 1
        in_deg[e.target] += 1

    for n in SYNC_NODES:
        genre = _CLIENT_GENRE.get(n.node_id)
        mode = GENRE_SYNC_POLICY[genre].value if genre else None
        links = [e.conn_id for e in SYNC_CONNECTIONS if n.node_id in (e.source, e.target)]
        if genre:
            links.append(genre)
        cards.append({
            "id": n.node_id,
            "kind": "NODE",
            "layer": _ROLE_LAYER[n.role],
            "role": n.role,
            "sync_mode": mode,
            "links": links,
            "title": n.node_id.replace("_", " "),
            "subtitle": n.anchor,
            "metrics": [
                {"label": "IN", "value": in_deg[n.node_id], "tag": ""},
                {"label": "OUT", "value": out_deg[n.node_id], "tag": ""},
            ] + ([{"label": "GENRE", "value": genre, "tag": ""}] if genre else []),
        })

    for g, m in GENRE_SYNC_POLICY.items():
        cards.append({
            "id": g,
            "kind": "GENRE",
            "layer": "FACTION",
            "role": "GENRE",
            "sync_mode": m.value,
            "links": [cid for cid, gen in _CLIENT_GENRE.items() if gen == g],
            "title": g.split("_", 2)[2].replace("_", " "),
            "subtitle": f"Sync mode: {m.value}",
            "metrics": [
                {"label": "STRICTNESS", "value": _STRICTNESS.get(m, "READ-ONLY"), "tag": ""},
                {"label": "HP CAP", "value": VITAL_MAX_HP, "tag": ""},
            ],
        })

    for e in SYNC_CONNECTIONS:
        cards.append({
            "id": e.conn_id,
            "kind": "CONNECTION",
            "layer": "ACTION",
            "role": "CONNECTION",
            "sync_mode": None,
            "links": [e.source, e.target],
            "transport": e.transport,
            "title": f"{e.conn_id} · {e.source} → {e.target}",
            "subtitle": e.payload,
            "metrics": [
                {"label": "TRANSPORT", "value": e.transport, "tag": ""},
                {"label": "FRAME", "value": f"{e.frame_bytes} B" if e.frame_bytes else "—", "tag": ""},
                {"label": "ON FAILURE", "value": e.on_failure, "tag": ""},
            ],
        })

    total_desync = 0
    for name, lat, jit in SIM_PROFILES:
        r = _cached_simulation(sim_ticks, lat, jit, 6)
        total_desync += r["desync_ticks"]
        cards.append({
            "id": f"SIM_{name}",
            "kind": "TELEMETRY",
            "layer": "TELEMETRY",
            "role": "TELEMETRY",
            "sync_mode": SyncMode.ROLLBACK.value,
            "links": ["GENRE_03_ACTION_BALLISTIC_6D", "CLIENT_ACTION_6D", "JITTER_METER", "POLICY_CONTROLLER"],
            "title": f"Rollback simulation · {name}",
            "subtitle": f"one-way {lat} ms, sigma {jit} ms, {sim_ticks} ticks, synthetic latency (simulated, not a network measurement)",
            "metrics": [
                {"label": "DESYNC TICKS", "value": r["desync_ticks"], "tag": "MODELED"},
                {"label": "MAX HORIZON", "value": r["max_horizon"], "tag": "MODELED"},
                {"label": "ROLLBACKS", "value": r["rollback_events"], "tag": "MODELED"},
                {"label": "STALLS", "value": r["stalls"], "tag": "MODELED"},
                {"label": "MAX HP", "value": r["max_hp_observed"], "tag": "MODELED"},
            ],
        })

    return {
        "vital_max_hp": VITAL_MAX_HP,
        "graph": graph,
        "total_sim_desync": total_desync,
        "card_count": len(cards),
        "cards": cards,
    }


_TEMPLATE = r"""<!DOCTYPE html>
<html lang="sk">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>Krystal · Adaptive Cross-Genre Sync · Card Grid</title>
<style>
  :root {
    --ks-black:#030406; --ks-void:#070812; --ks-navy:#101124; --ks-panel:#11131D; --ks-panel-soft:#171827;
    --ks-line:#292A38; --ks-text:#F1EFE8; --ks-muted:#8D8997; --ks-gold:#D8AE4B; --ks-gold-hi:#F0D071;
    --ks-cyan:#58D8FF; --ks-teal:#50EEE0; --ks-magenta:#FF307C; --ks-green:#43D264; --ks-violet:#7548D8;
    --ks-red:#FF4757; --ks-yellow:#F59E0B;
    --radius-ui:2px; --radius-control:4px;
    --f-serif:"Cormorant Garamond","Cinzel",Georgia,serif;
    --f-cond:"Bebas Neue","Oswald","Arial Narrow",sans-serif;
    --f-mono:"JetBrains Mono","IBM Plex Mono",Consolas,monospace;
    --bg:var(--ks-void); --surface:var(--ks-panel); --accent:var(--ks-cyan); --title-font:var(--f-cond);
  }
  body.mode-world  { --bg:var(--ks-navy);  --surface:var(--ks-panel-soft); --accent:var(--ks-gold); --title-font:var(--f-serif); }
  body.mode-arena  { --bg:var(--ks-black); --surface:var(--ks-panel);      --accent:var(--ks-cyan); --title-font:var(--f-mono); }
  body.mode-hybrid { --bg:var(--ks-void);  --surface:var(--ks-panel);      --accent:var(--ks-cyan); --title-font:var(--f-cond); }
  * { box-sizing:border-box; }
  body { margin:0; background:var(--bg); color:var(--ks-text); font-family:var(--f-mono); font-size:14px; transition:background 240ms; }
  nav.ks-nav { height:80px; background:var(--ks-navy); border-bottom:1px solid var(--ks-line); display:flex; align-items:center; gap:24px; padding:0 32px; }
  .ks-brand { font-family:var(--f-serif); font-size:28px; color:var(--ks-gold); letter-spacing:0.04em; }
  .ks-chapter { font-family:var(--f-cond); font-size:12px; letter-spacing:0.24em; text-transform:uppercase; color:var(--ks-muted); }
  .ks-controls { margin-left:auto; display:flex; gap:16px; align-items:center; }
  .ks-seg { display:flex; border:1px solid var(--ks-line); border-radius:var(--radius-control); }
  .ks-seg button { background:transparent; color:var(--ks-muted); border:0; border-right:1px solid var(--ks-line); padding:8px 12px;
                   font-family:var(--f-cond); font-size:13px; letter-spacing:0.16em; text-transform:uppercase; cursor:pointer; transition:color 160ms, background 160ms; }
  .ks-seg button:last-child { border-right:0; }
  .ks-seg button.active { color:var(--ks-black); background:var(--accent); }
  main { display:grid; grid-template-columns:1fr; gap:24px; padding:32px; }
  body.mode-hybrid main { grid-template-columns:minmax(0,2fr) minmax(320px,1fr); }
  .ks-group { margin-bottom:48px; }
  .ks-group h2 { font-family:var(--title-font); font-weight:400; font-size:44px; margin:4px 0 16px; color:var(--ks-text); }
  .ks-grid { display:grid; grid-template-columns:repeat(auto-fill,minmax(256px,1fr)); gap:16px; }
  body.density-list .ks-grid { grid-template-columns:1fr; gap:8px; }
  body.density-list .ks-card { display:grid; grid-template-columns:240px 1fr 2fr; gap:16px; align-items:start; }
  .ks-card { background:var(--surface); border:1px solid var(--ks-line); border-radius:var(--radius-ui); padding:16px; cursor:pointer; transition:border-color 160ms; }
  .ks-card:hover { border-color:var(--accent); }
  .ks-card.selected { border-color:var(--ks-magenta); }
  .ks-card .eyebrow { font-family:var(--f-cond); font-size:11px; letter-spacing:0.24em; text-transform:uppercase; color:var(--accent); }
  .ks-card h3 { font-family:var(--title-font); font-weight:400; font-size:20px; margin:8px 0 4px; word-break:break-word; }
  .ks-card p { color:var(--ks-muted); font-size:12px; margin:0 0 12px; }
  .ks-metric { display:flex; justify-content:space-between; gap:8px; border-top:1px solid var(--ks-line); padding:4px 0; font-size:12px; }
  .ks-metric .v { color:var(--ks-teal); text-align:right; word-break:break-word; }
  .tag { font-size:10px; letter-spacing:0.20em; color:var(--ks-yellow); margin-left:4px; }
  aside.ks-rail { display:none; }
  body.mode-hybrid aside.ks-rail { display:block; position:sticky; top:32px; align-self:start; }
  .ks-terminal { background:var(--ks-black); border:1px solid var(--ks-line); border-radius:var(--radius-ui); padding:16px; font-size:12px; color:var(--ks-teal); min-height:240px; }
  .ks-terminal h3 { font-family:var(--title-font); font-weight:400; font-size:20px; color:var(--ks-text); margin:8px 0 4px; }
  .ks-terminal p { color:var(--ks-muted); margin:0 0 12px; }
  .ks-terminal .eyebrow { font-family:var(--f-cond); font-size:11px; letter-spacing:0.24em; text-transform:uppercase; color:var(--accent); }
  .ks-trail { margin-bottom:12px; color:var(--ks-muted); font-size:11px; }
  .ks-related { margin-top:16px; }
  .ks-related .ks-chapter { margin-top:12px; }
  .ks-link, .ks-crumb { display:block; width:100%; text-align:left; background:transparent; color:var(--ks-text); border:1px solid var(--ks-line);
                        border-radius:var(--radius-ui); padding:8px 12px; margin-top:4px; font-family:var(--f-mono); font-size:11px; cursor:pointer; transition:border-color 160ms; }
  .ks-crumb { display:inline-block; width:auto; padding:4px 8px; margin:0; color:var(--ks-muted); }
  .ks-link:hover, .ks-crumb:hover { border-color:var(--accent); }
  .ks-group-head { cursor:pointer; user-select:none; }
  .ks-hud { position:fixed; right:16px; bottom:16px; background:var(--ks-black); border:1px solid var(--ks-line); border-radius:var(--radius-ui);
            padding:8px 12px; font-size:11px; color:var(--ks-muted); }
  .ok { color:var(--ks-green); } .bad { color:var(--ks-red); }

  /* ── KS HEX GRID ─────────────────────────────────────────────────────────
     Pointy-top hexagons, offset rows via float + shape-outside (pure CSS, responsive).
     1px structural border = outer hex in --ks-line + inner hex inset 1px in surface.
     No shadows, no blur, no rounded corners (design manual §4). Spacing on 4px grid. */
  .ks-hexgrid {
    --hex-s: 128px;                                   /* hex width  */
    --hex-m: 4px;                                     /* gap        */
    --hex-h: calc(var(--hex-s) * 1.1547);             /* height = s * 2/sqrt(3) */
    --hex-f: calc(1.732 * var(--hex-s) + 4 * var(--hex-m) - 1px); /* row-pair period */
    font-size: 0;                                     /* kill inline-block whitespace */
  }
  .ks-hexgrid::before {
    content: ""; float: left; height: 120%;
    width: calc(var(--hex-s) / 2 + var(--hex-m));
    shape-outside: repeating-linear-gradient(#0000 0 calc(var(--hex-f) - 3px), #000 0 var(--hex-f));
  }
  .ks-hex {
    position: relative; display: inline-block; vertical-align: top;
    width: var(--hex-s); height: var(--hex-h);
    margin: var(--hex-m); margin-bottom: calc(var(--hex-m) - var(--hex-s) * 0.2885);
    clip-path: polygon(50% 0, 100% 25%, 100% 75%, 50% 100%, 0 75%, 0 25%);
    background: var(--ks-line);                       /* acts as the 1px border */
    cursor: pointer; transition: background 160ms;
    font-size: 11px;
  }
  .ks-hex-inner {
    position: absolute; inset: 1px;
    clip-path: polygon(50% 0, 100% 25%, 100% 75%, 50% 100%, 0 75%, 0 25%);
    background: var(--surface);
    display: flex; flex-direction: column; align-items: center; justify-content: center;
    gap: 4px; padding: 24px 12px; text-align: center; overflow: hidden;
  }
  .ks-hex .hx-kind  { font-family: var(--f-cond); font-size: 10px; letter-spacing: 0.24em; text-transform: uppercase; color: var(--accent); }
  .ks-hex .hx-title { font-family: var(--f-mono); font-size: 11px; color: var(--ks-text); line-height: 1.2; word-break: break-word; }
  .ks-hex .hx-metric{ font-family: var(--f-mono); font-size: 10px; color: var(--ks-teal); }
  .ks-hex:hover            { background: var(--accent); }
  .ks-hex.selected         { background: var(--ks-magenta); }
  .ks-hex.layer-WORLD     .hx-kind { color: var(--ks-gold); }   /* narrative layer = gold   */
  .ks-hex.layer-TELEMETRY .hx-kind { color: var(--ks-green); }  /* status layer = green     */
  body.density-hex .ks-hexgrid { --hex-s: 128px; }
  @media (max-width: 720px) { .ks-hexgrid { --hex-s: 96px; } }

  /* ── KS HEX GRID 3D ──────────────────────────────────────────────────────
     Same hex lattice on a tilted plane. Each tile is lifted by --z (graph degree of the
     card, capped at 10) * 8px. Depth is conveyed by perspective + elevation only:
     no shadows, no glow, no blur (design manual §4). Motion 240ms (manual §8). */
  .ks-hex3d-stage { perspective: 1200px; perspective-origin: 50% 0; padding: 48px 0 96px; overflow: hidden; }
  .ks-hexgrid.ks-hex3d { transform: rotateX(52deg) rotateZ(-8deg); transform-origin: 50% 0; transform-style: preserve-3d; }
  .ks-hex3d .ks-hex { transform: translateZ(calc(var(--z, 0) * 8px)); transition: transform 240ms, background 160ms; }
  .ks-hex3d .ks-hex:hover,
  .ks-hex3d .ks-hex.selected { transform: translateZ(calc(var(--z, 0) * 8px + 24px)); }
  .ks-z-legend { font-family: var(--f-mono); font-size: 11px; color: var(--ks-muted); margin-bottom: 8px; }
</style>
</head>
<body class="mode-hybrid density-grid">
<nav class="ks-nav">
  <div>
    <div class="ks-chapter">Krystal Stack · Adaptive Cross-Genre Sync</div>
    <div class="ks-brand">Card Grid</div>
  </div>
  <div class="ks-controls">
    <div class="ks-seg" id="seg-hier">
      <button data-v="manual" class="active">Manuál</button>
      <button data-v="role">Rola</button>
      <button data-v="sync">Sync režim</button>
    </div>
    <div class="ks-seg" id="seg-mode">
      <button data-v="world">World</button>
      <button data-v="arena">Arena</button>
      <button data-v="hybrid" class="active">Hybrid</button>
    </div>
    <div class="ks-seg" id="seg-density">
      <button data-v="grid" class="active">Grid</button>
      <button data-v="list">List</button>
      <button data-v="hex">Hex</button>
      <button data-v="hex3d">Hex 3D</button>
    </div>
  </div>
</nav>
<main>
  <section id="groups"></section>
  <aside class="ks-rail">
    <div class="ks-chapter">Detail karty</div>
    <div class="ks-terminal" id="detail">Vyber kartu.</div>
  </aside>
</main>
<div class="ks-hud" id="hud"></div>
<script>
const DATA = __DATA__;
const state = { hier: "manual", selected: null, density: "grid", collapsed: new Set(), trail: [] };
const BY_ID = Object.fromEntries(DATA.cards.map(c => [c.id, c]));
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));

const LAYERS = ["WORLD","FACTION","SYSTEM","ACTION","TELEMETRY"];
const LAYER_EYEBROW = {WORLD:"Stav a projekcie", FACTION:"Žánre a klienti", SYSTEM:"Riadenie a autorita", ACTION:"Spojenia", TELEMETRY:"Simulácia"};

function groups() {
  const cards = DATA.cards;
  if (state.hier === "manual") {
    return LAYERS.map(l => ({ eyebrow: LAYER_EYEBROW[l], title: l, cards: cards.filter(c => c.layer === l) }));
  }
  if (state.hier === "role") {
    const roles = [...new Set(cards.map(c => c.role))];
    return roles.map(r => ({ eyebrow: "Rola", title: r, cards: cards.filter(c => c.role === r) }));
  }
  const modes = ["AUTHORITATIVE_LEDGER","TURN_LOCKSTEP","INPUT_DELAY_LOCKSTEP","ROLLBACK","OBSERVER_INTERP"];
  const out = modes.map(m => ({ eyebrow: "Sync režim", title: m, cards: cards.filter(c => c.sync_mode === m) }));
  out.push({ eyebrow: "Bez režimu", title: "INFRAŠTRUKTÚRA", cards: cards.filter(c => !c.sync_mode) });
  return out.filter(g => g.cards.length);
}

function cardHtml(c) {
  const metrics = c.metrics.map(m =>
    `<div class="ks-metric"><span>${esc(m.label)}${m.tag ? `<span class="tag">[${esc(m.tag)}]</span>` : ""}</span><span class="v">${esc(m.value)}</span></div>`
  ).join("");
  return `<div class="ks-card${state.selected === c.id ? " selected" : ""}" data-id="${esc(c.id)}">
      <div><div class="eyebrow">${esc(c.kind)} · ${esc(c.layer)}</div><h3>${esc(c.title)}</h3></div>
      <p>${esc(c.subtitle)}</p><div>${metrics}</div></div>`;
}

function hexHtml(c) {
  const m = c.metrics[0];
  const short = c.kind === "CONNECTION" ? c.id : c.title;
  return `<div class="ks-hex layer-${esc(c.layer)}${state.selected === c.id ? " selected" : ""}" data-id="${esc(c.id)}" title="${esc(c.title)}" style="--z:${Math.min((c.links || []).length, 10)}">
      <div class="ks-hex-inner"><span class="hx-kind">${esc(c.kind)}</span><span class="hx-title">${esc(short)}</span>
      ${m ? `<span class="hx-metric">${esc(m.label)} ${esc(m.value)}</span>` : ""}</div></div>`;
}

function render() {
  const hex = state.density === "hex" || state.density === "hex3d";
  const hex3d = state.density === "hex3d";
  document.getElementById("groups").innerHTML = groups().map(g => {
    const closed = state.collapsed.has(g.title);
    const hexBody = `<div class="ks-hexgrid${hex3d ? " ks-hex3d" : ""}">${g.cards.map(hexHtml).join("")}</div>`;
    const body = closed ? "" : (hex ? (hex3d ? `<div class="ks-z-legend">Výška Z = počet väzieb karty v grafe (× 8 px, max 10)</div><div class="ks-hex3d-stage">${hexBody}</div>` : hexBody)
                                     : `<div class="ks-grid">${g.cards.map(cardHtml).join("")}</div>`);
    return `<div class="ks-group"><div class="ks-chapter">${esc(g.eyebrow)} · ${g.cards.length}</div>
      <h2 class="ks-group-head" data-group="${esc(g.title)}">${closed ? "▸" : "▾"} ${esc(g.title)}</h2>${body}</div>`;
  }).join("");
  renderDetail();
  const g = DATA.graph;
  document.getElementById("hud").innerHTML =
    `HP CAP ${DATA.vital_max_hp} · GRAPH <span class="${g.graph_valid ? "ok" : "bad"}">${g.graph_valid ? "VALID" : "INVALID"}</span>` +
    ` · ${g.node_count} NODES · ${g.edge_count} EDGES · SIM DESYNC <span class="${DATA.total_sim_desync === 0 ? "ok" : "bad"}">${DATA.total_sim_desync}</span> [MODELED]` +
    ` · DEPTH ${state.trail.length}`;
}

/* Drill-down: detail rail shows the selected card, its related cards (graph neighbours)
   and a breadcrumb trail of the navigation path. Every related card is clickable. */
function renderDetail() {
  const el = document.getElementById("detail");
  const c = BY_ID[state.selected];
  if (!c) { el.innerHTML = "Vyber kartu."; return; }
  const trail = state.trail.map((id, i) =>
    `<button class="ks-crumb" data-id="${esc(id)}">${i + 1}. ${esc(id)}</button>`).join(" › ");
  const metrics = c.metrics.map(m =>
    `<div class="ks-metric"><span>${esc(m.label)}${m.tag ? `<span class="tag">[${esc(m.tag)}]</span>` : ""}</span><span class="v">${esc(m.value)}</span></div>`).join("");
  const rel = (c.links || []).map(id => BY_ID[id]).filter(Boolean);
  const relByLayer = LAYERS.map(l => {
    const items = rel.filter(r => r.layer === l);
    return items.length ? `<div class="ks-chapter">${esc(l)} · ${items.length}</div>` +
      items.map(r => `<button class="ks-link" data-id="${esc(r.id)}">${esc(r.kind)} · ${esc(r.kind === "CONNECTION" ? r.title : r.id)}</button>`).join("") : "";
  }).join("");
  el.innerHTML = `<div class="ks-trail">${trail}</div>
    <div class="eyebrow">${esc(c.kind)} · ${esc(c.layer)} · ${esc(c.role)}</div>
    <h3>${esc(c.title)}</h3><p>${esc(c.subtitle)}</p>${metrics}
    <div class="ks-related"><div class="ks-chapter">Súvisiace karty · ${rel.length}</div>${relByLayer}</div>`;
}

function select(id) {
  if (!BY_ID[id]) return;
  state.selected = id;
  if (state.trail[state.trail.length - 1] !== id) state.trail.push(id);
  if (state.trail.length > 8) state.trail.shift();
  render();
}

document.addEventListener("click", (ev) => {
  const head = ev.target.closest("[data-group]");
  if (head) {
    const t = head.dataset.group;
    state.collapsed.has(t) ? state.collapsed.delete(t) : state.collapsed.add(t);
    render(); return;
  }
  const item = ev.target.closest("[data-id]");
  if (item) select(item.dataset.id);
});

function bindSeg(id, fn) {
  document.querySelectorAll(`#${id} button`).forEach(b => b.addEventListener("click", () => {
    document.querySelectorAll(`#${id} button`).forEach(x => x.classList.remove("active"));
    b.classList.add("active"); fn(b.dataset.v);
  }));
}
bindSeg("seg-hier", v => { state.hier = v; render(); });
bindSeg("seg-mode", v => { document.body.classList.remove("mode-world","mode-arena","mode-hybrid"); document.body.classList.add("mode-" + v); });
bindSeg("seg-density", v => { document.body.classList.remove("density-grid","density-list","density-hex","density-hex3d"); document.body.classList.add("density-" + v); state.density = v; render(); });
render();
</script>
</body>
</html>
"""


def render_adaptive_sync_card_grid_html(sim_ticks: int = 1000) -> str:
    data = build_card_dataset(sim_ticks=sim_ticks)
    payload = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    return _TEMPLATE.replace("__DATA__", payload)


def export_standalone_html(path: Path | None = None) -> Path:
    target = path or Path(__file__).resolve().parents[2] / "frontend" / "adaptive_sync_card_grid.html"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(render_adaptive_sync_card_grid_html(), encoding="utf-8")
    return target


if __name__ == "__main__":
    print(export_standalone_html())
