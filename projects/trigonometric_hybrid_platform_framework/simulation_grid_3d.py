"""Babylon.js 3D replay grid of the adaptive rollback simulation logic.

Visualizes, tick by tick, the authoritative kernel state against the client's first
(possibly mispredicted) state on a 19-cell hex board (radius 2). Positions 0..18 of the
simulation map onto hex cells in spiral order. Design: Arena mode of the Krystal UI
manual (black, cyan/magenta/teal, monospace, 1px lines, no shadows/glow).

Evidence class of all numbers shown: [MODELED] — synthetic latency, deterministic simulation.
"""

from __future__ import annotations

import hashlib
import json
import urllib.request
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Tuple

from krystal_kernel.constants import VITAL_MAX_HP, ContractViolation

from .adaptive_cross_genre_sync import HEX_CELLS, simulate_adaptive_rollback

BABYLON_VERSION = "9.36.0"
BABYLON_URL = f"https://cdn.jsdelivr.net/npm/babylonjs@{BABYLON_VERSION}/babylon.js"
BABYLON_SHA256 = "7edf5405501eee0ea55e31d7501f4ac17c7ca87dfc8e3227bb3cd99499309dbc"
BABYLON_SRI = "sha384-QWFG07sio34fL3sNY9wd1SSoDvD7YVMX3067rJICw3vO2LQQY5m53yB70YlKXnRX"
VENDOR_PATH = Path(__file__).resolve().parents[2] / "frontend" / "vendor" / "babylon.js"

PROFILES = (("LAN", 10.0, 2.0), ("REGIONAL", 40.0, 6.0), ("WAN", 100.0, 20.0))


def ensure_babylon_vendor(path: Path = VENDOR_PATH) -> Dict[str, Any]:
    """Downloads the pinned Babylon.js bundle if missing and verifies its SHA-256."""
    if not path.exists():
        req = urllib.request.Request(BABYLON_URL, headers={"User-Agent": "KrystalVendorFetch/1.0"})
        data = urllib.request.urlopen(req, timeout=60).read()
        digest = hashlib.sha256(data).hexdigest()
        if digest != BABYLON_SHA256:
            raise ContractViolation(
                f"Babylon.js hash mismatch: {digest} != {BABYLON_SHA256}", violation_code="ERR_VENDOR_HASH"
            )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != BABYLON_SHA256:
        raise ContractViolation(f"Vendored Babylon.js hash mismatch: {digest}", violation_code="ERR_VENDOR_HASH")
    return {"path": str(path), "version": BABYLON_VERSION, "sha256": digest, "bytes": path.stat().st_size}


def hex_spiral_axial(radius: int = 2) -> List[Tuple[int, int]]:
    """Axial (q, r) coordinates in spiral order: center, ring 1 (6), ring 2 (12) -> 19 cells."""
    dirs = [(1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1)]
    cells = [(0, 0)]
    for k in range(1, radius + 1):
        q, r = -k, k  # start at direction 4 scaled by k
        for d in range(6):
            for _ in range(k):
                cells.append((q, r))
                q, r = q + dirs[d][0], r + dirs[d][1]
    return cells


@lru_cache(maxsize=8)
def _trace(ticks: int, lat: float, jit: float) -> Dict[str, Any]:
    return simulate_adaptive_rollback(ticks=ticks, one_way_latency_ms=lat, jitter_sigma_ms=jit, seed=6, record_trace=True)


def build_trace_dataset(ticks: int = 600) -> Dict[str, Any]:
    cells = hex_spiral_axial(2)
    if len(cells) != HEX_CELLS:
        raise ContractViolation("Hex board size != simulation HEX_CELLS", violation_code="ERR_HEX_BOARD")
    profiles: Dict[str, Any] = {}
    for name, lat, jit in PROFILES:
        r = _trace(ticks, lat, jit)
        profiles[name] = {
            "one_way_latency_ms": lat,
            "jitter_sigma_ms": jit,
            "metrics": {k: r[k] for k in ("desync_ticks", "rollback_events", "stalls", "max_horizon", "max_hp_observed")},
            "trace": r["trace"],
        }
    return {"vital_max_hp": VITAL_MAX_HP, "ticks": ticks, "hex_cells": cells, "profiles": profiles}


_TEMPLATE = r"""<!DOCTYPE html>
<html lang="sk">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>Krystal · Simulation Grid 3D (Babylon.js)</title>
<style>
  :root { --ks-black:#030406; --ks-panel:#11131D; --ks-line:#292A38; --ks-text:#F1EFE8; --ks-muted:#8D8997;
          --ks-cyan:#58D8FF; --ks-teal:#50EEE0; --ks-magenta:#FF307C; --ks-green:#43D264; --ks-yellow:#F59E0B; --ks-red:#FF4757;
          --f-cond:"Bebas Neue","Oswald","Arial Narrow",sans-serif; --f-mono:"JetBrains Mono","IBM Plex Mono",Consolas,monospace; }
  * { box-sizing:border-box; }
  html, body { margin:0; height:100%; background:var(--ks-black); color:var(--ks-text); font-family:var(--f-mono); font-size:12px; }
  #app { display:grid; grid-template-columns:minmax(0,7fr) minmax(320px,3fr); grid-template-rows:1fr 96px; height:100%; }
  #stage { grid-row:1; grid-column:1; position:relative; border-right:1px solid var(--ks-line); }
  #renderCanvas { width:100%; height:100%; display:block; outline:none; }
  #rail { grid-row:1 / span 2; grid-column:2; padding:24px; overflow:auto; }
  #timeline { grid-row:2; grid-column:1; border-top:1px solid var(--ks-line); padding:8px 16px; display:grid; grid-template-rows:48px 32px; gap:8px; }
  #strip { width:100%; height:48px; display:block; }
  .ks-chapter { font-family:var(--f-cond); font-size:11px; letter-spacing:0.24em; text-transform:uppercase; color:var(--ks-muted); margin:16px 0 8px; }
  .ks-title { font-family:var(--f-cond); font-size:28px; letter-spacing:0.08em; text-transform:uppercase; color:var(--ks-cyan); margin:0; }
  .ks-metric { display:flex; justify-content:space-between; border-top:1px solid var(--ks-line); padding:4px 0; }
  .ks-metric .v { color:var(--ks-teal); }
  .tag { color:var(--ks-yellow); font-size:10px; letter-spacing:0.2em; margin-left:4px; }
  .ks-stat { display:grid; grid-template-columns:repeat(6, 1fr); gap:4px; margin:4px 0 8px; }
  .ks-stat span { height:8px; border:1px solid var(--ks-line); border-radius:2px; }
  .ks-stat.a span.on { background:var(--ks-cyan); border-color:var(--ks-cyan); }
  .ks-stat.b span.on { background:var(--ks-magenta); border-color:var(--ks-magenta); }
  .controls { display:flex; gap:8px; align-items:center; }
  button, select { background:transparent; color:var(--ks-text); border:1px solid var(--ks-line); border-radius:4px; padding:4px 12px;
                   font-family:var(--f-cond); font-size:13px; letter-spacing:0.16em; text-transform:uppercase; cursor:pointer; transition:border-color 160ms; }
  button:hover, select:hover { border-color:var(--ks-cyan); }
  input[type=range] { flex:1; accent-color:var(--ks-cyan); }
  .legend div { display:flex; gap:8px; align-items:center; padding:2px 0; }
  .sw { width:12px; height:12px; border:1px solid var(--ks-line); border-radius:2px; display:inline-block; }
  #noBabylon { position:absolute; inset:0; display:none; align-items:center; justify-content:center; color:var(--ks-red); padding:32px; text-align:center; }
</style>
</head>
<body>
<div id="app">
  <div id="stage"><canvas id="renderCanvas"></canvas><div id="noBabylon">Babylon.js sa nenačítal (chýba frontend/vendor/babylon.js a CDN nie je dostupné).</div></div>
  <div id="timeline">
    <canvas id="strip"></canvas>
    <div class="controls">
      <button id="play">Play</button>
      <select id="speed"><option value="1">1×</option><option value="4" selected>4×</option><option value="16">16×</option></select>
      <select id="profile"></select>
      <input id="scrub" type="range" min="0" value="0" />
      <span id="tickLabel">0</span>
    </div>
  </div>
  <aside id="rail">
    <div class="ks-chapter">Arena · Babylon.js __BABYLON_VERSION__</div>
    <h1 class="ks-title">Simulation Grid 3D</h1>
    <div class="ks-chapter">Kernel HP (autoritatívny stav)</div>
    <div>Hráč A</div><div class="ks-stat a" id="hpA"></div>
    <div>Hráč B</div><div class="ks-stat b" id="hpB"></div>
    <div class="ks-chapter">Tick</div><div id="tickInfo"></div>
    <div class="ks-chapter">Profil · metriky simulácie</div><div id="metrics"></div>
    <div class="ks-chapter">Legenda</div>
    <div class="legend">
      <div><span class="sw" style="background:#58D8FF"></span> A — kernel</div>
      <div><span class="sw" style="background:#FF307C"></span> B — kernel</div>
      <div><span class="sw" style="border-color:#F59E0B"></span> Predikcia klienta, neskôr opravená rollbackom</div>
      <div><span class="sw" style="background:#50EEE0"></span> Horizont rollbacku (pás dole)</div>
    </div>
    <div class="ks-chapter">Poznámka</div>
    <div style="color:var(--ks-muted)">Pozície 0–18 sú mapované na 19 hexov v špirálovom poradí. Všetky čísla sú [MODELED]: deterministická simulácia s umelou latenciou, nie meranie siete.</div>
  </aside>
</div>
<script src="vendor/babylon.js"></script>
<script>
if (!window.BABYLON) {
  document.write('<script src="__BABYLON_URL__" integrity="__BABYLON_SRI__" crossorigin="anonymous"><\/script>');
}
</script>
<script>
const DATA = __DATA__;
(function () {
  if (!window.BABYLON) { document.getElementById("noBabylon").style.display = "flex"; return; }
  const C = (hex) => BABYLON.Color3.FromHexString(hex);
  const canvas = document.getElementById("renderCanvas");
  const engine = new BABYLON.Engine(canvas, true, { preserveDrawingBuffer: false, stencil: false });
  const scene = new BABYLON.Scene(engine);
  scene.clearColor = new BABYLON.Color4(0.012, 0.016, 0.024, 1);
  const camera = new BABYLON.ArcRotateCamera("cam", -Math.PI / 2, Math.PI / 3.2, 12, BABYLON.Vector3.Zero(), scene);
  camera.lowerRadiusLimit = 6; camera.upperRadiusLimit = 24; camera.attachControl(canvas, true);
  const light = new BABYLON.HemisphericLight("h", new BABYLON.Vector3(0, 1, 0), scene); light.intensity = 0.9;

  const SIZE = 1.0, SQ3 = Math.sqrt(3);
  const centers = DATA.hex_cells.map(([q, r]) => new BABYLON.Vector3(SIZE * SQ3 * (q + r / 2), 0, SIZE * 1.5 * r));

  function hexMesh(name, center, radius, color, y) {
    const pos = [center.x, y, center.z], idx = [];
    for (let i = 0; i < 6; i++) {
      const a = Math.PI / 180 * (60 * i - 30);
      pos.push(center.x + radius * Math.cos(a), y, center.z + radius * Math.sin(a));
    }
    for (let i = 1; i <= 6; i++) idx.push(0, i, i === 6 ? 1 : i + 1);
    const m = new BABYLON.Mesh(name, scene), vd = new BABYLON.VertexData();
    vd.positions = pos; vd.indices = idx; const normals = []; BABYLON.VertexData.ComputeNormals(pos, idx, normals); vd.normals = normals;
    vd.applyToMesh(m);
    const mat = new BABYLON.StandardMaterial(name + "m", scene);
    mat.diffuseColor = color; mat.specularColor = BABYLON.Color3.Black(); mat.backFaceCulling = false;
    m.material = mat; return m;
  }
  function hexOutline(name, center, radius, color, y) {
    const pts = [];
    for (let i = 0; i <= 6; i++) { const a = Math.PI / 180 * (60 * (i % 6) - 30); pts.push(new BABYLON.Vector3(center.x + radius * Math.cos(a), y, center.z + radius * Math.sin(a))); }
    const l = BABYLON.MeshBuilder.CreateLines(name, { points: pts }, scene); l.color = color; return l;
  }

  const tiles = centers.map((c, i) => hexMesh("tile" + i, c, SIZE * 0.96, C("#11131D"), 0));
  centers.forEach((c, i) => hexOutline("edge" + i, c, SIZE * 0.96, C("#292A38"), 0.002));
  const warnRings = centers.map((c, i) => { const l = hexOutline("warn" + i, c, SIZE * 0.82, C("#F59E0B"), 0.004); l.isVisible = false; return l; });

  function token(name, hex, alpha) {
    const m = BABYLON.MeshBuilder.CreateCylinder(name, { diameter: 0.55, height: 0.1, tessellation: 6 }, scene);
    const mat = new BABYLON.StandardMaterial(name + "m", scene);
    mat.diffuseColor = C(hex); mat.emissiveColor = C(hex).scale(0.35); mat.specularColor = BABYLON.Color3.Black(); mat.alpha = alpha;
    m.material = mat; return m;
  }
  const pillar = (name, hex) => { const p = token(name, hex, 1); return p; };
  const A = pillar("A", "#58D8FF"), B = pillar("B", "#FF307C");
  const ghostA = token("gA", "#F59E0B", 0.35), ghostB = token("gB", "#F59E0B", 0.35);

  const sel = document.getElementById("profile"), scrub = document.getElementById("scrub");
  Object.keys(DATA.profiles).forEach(n => { const o = document.createElement("option"); o.value = n; o.textContent = n; sel.appendChild(o); });
  sel.value = "REGIONAL";
  scrub.max = DATA.ticks - 1;
  const segs = (id) => { const el = document.getElementById(id); el.innerHTML = Array.from({length: 6}, () => "<span></span>").join(""); return [...el.children]; };
  const segA = segs("hpA"), segB = segs("hpB");

  function place(mesh, cell, hp, lift) {
    const c = centers[cell]; const h = 0.12 + hp * 0.18;   // pillar height encodes HP (<= 6)
    mesh.scaling.y = h / 0.1; mesh.position.set(c.x + lift, h / 2, c.z);
  }

  function drawStrip() {
    const cv = document.getElementById("strip"), ctx = cv.getContext("2d");
    cv.width = cv.clientWidth; cv.height = 48;
    const p = DATA.profiles[sel.value].trace, n = DATA.ticks, W = cv.width;
    ctx.fillStyle = "#030406"; ctx.fillRect(0, 0, W, 48);
    ctx.fillStyle = "#F59E0B";
    p.corrected.forEach((c, k) => { if (c) ctx.fillRect(Math.floor(k / n * W), 0, 1, 12); });
    ctx.strokeStyle = "#50EEE0"; ctx.lineWidth = 1; ctx.beginPath();
    p.horizon.forEach((h, k) => { const x = k / n * W, y = 46 - (h / 8) * 30; k ? ctx.lineTo(x, y) : ctx.moveTo(x, y); });
    ctx.stroke();
    const x = Number(scrub.value) / n * W; ctx.fillStyle = "#FF307C"; ctx.fillRect(Math.floor(x), 0, 1, 48);
  }

  function show(k) {
    const prof = DATA.profiles[sel.value], t = prof.trace;
    const kr = t.kernel[k], fp = t.first_prediction[k], corr = t.corrected[k];
    place(A, kr[0], kr[1], -0.12); place(B, kr[2], kr[3], 0.12);
    ghostA.isVisible = ghostB.isVisible = corr;
    warnRings.forEach(w => w.isVisible = false);
    if (corr) { place(ghostA, fp[0], fp[1], -0.12); place(ghostB, fp[2], fp[3], 0.12); warnRings[fp[0]].isVisible = true; warnRings[fp[2]].isVisible = true; }
    segA.forEach((s, i) => s.classList.toggle("on", i < kr[1]));
    segB.forEach((s, i) => s.classList.toggle("on", i < kr[3]));
    document.getElementById("tickLabel").textContent = k;
    document.getElementById("tickInfo").innerHTML =
      `<div class="ks-metric"><span>TICK</span><span class="v">${k}</span></div>` +
      `<div class="ks-metric"><span>HORIZONT</span><span class="v">${t.horizon[k]}</span></div>` +
      `<div class="ks-metric"><span>PREDIKCIA OPRAVENÁ</span><span class="v" style="color:${corr ? "#F59E0B" : "#43D264"}">${corr ? "ÁNO" : "NIE"}</span></div>` +
      `<div class="ks-metric"><span>KERNEL A / B</span><span class="v">pos ${kr[0]} hp ${kr[1]} · pos ${kr[2]} hp ${kr[3]}</span></div>`;
    const m = prof.metrics;
    document.getElementById("metrics").innerHTML =
      `<div class="ks-metric"><span>LATENCIA / σ</span><span class="v">${prof.one_way_latency_ms} / ${prof.jitter_sigma_ms} ms</span></div>` +
      Object.entries(m).map(([kk, v]) => `<div class="ks-metric"><span>${kk.toUpperCase()}<span class="tag">[MODELED]</span></span><span class="v">${v}</span></div>`).join("");
    drawStrip();
  }

  let playing = false, acc = 0;
  document.getElementById("play").onclick = (e) => { playing = !playing; e.target.textContent = playing ? "Pause" : "Play"; };
  scrub.oninput = () => show(Number(scrub.value));
  sel.onchange = () => show(Number(scrub.value));
  scene.onBeforeRenderObservable.add(() => {
    if (!playing) return;
    acc += engine.getDeltaTime() / (1000 / 60) * Number(document.getElementById("speed").value) / 4;
    if (acc >= 1) { const step = Math.floor(acc); acc -= step; scrub.value = (Number(scrub.value) + step) % DATA.ticks; show(Number(scrub.value)); }
  });
  show(0);
  engine.runRenderLoop(() => scene.render());
  window.addEventListener("resize", () => { engine.resize(); drawStrip(); });
})();
</script>
</body>
</html>
"""


def render_simulation_grid_3d_html(ticks: int = 600) -> str:
    payload = json.dumps(build_trace_dataset(ticks), separators=(",", ":")).replace("</", "<\\/")
    return (
        _TEMPLATE.replace("__DATA__", payload)
        .replace("__BABYLON_VERSION__", BABYLON_VERSION)
        .replace("__BABYLON_URL__", BABYLON_URL)
        .replace("__BABYLON_SRI__", BABYLON_SRI)
    )


def export_standalone_html(path: Path | None = None) -> Path:
    ensure_babylon_vendor()
    target = path or VENDOR_PATH.parent.parent / "simulation_grid_3d.html"
    target.write_text(render_simulation_grid_3d_html(), encoding="utf-8")
    return target


if __name__ == "__main__":
    print(export_standalone_html())
