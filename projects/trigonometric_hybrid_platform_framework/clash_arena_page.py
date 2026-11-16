"""Clash Battle Royale arena page + control-center panel (Babylon.js, Krystal UI Hybrid mode).

The page is a pure projection: every click becomes a POST to the authoritative engine
(clash_battle_royale.ClashGame) and the returned state is rendered (Render(S) != S).
The control-center rail shows known hardware facts and allocator predictions from
/api/v1/runtime/facts and /api/v1/runtime/allocate.
"""

from __future__ import annotations

from .simulation_grid_3d import BABYLON_SRI, BABYLON_URL, BABYLON_VERSION

_TEMPLATE = r"""<!DOCTYPE html>
<html lang="sk">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>Krystal · Clash Battle Royale</title>
<style>
  :root { --ks-black:#030406; --ks-void:#070812; --ks-navy:#101124; --ks-panel:#11131D; --ks-line:#292A38; --ks-text:#F1EFE8;
          --ks-muted:#8D8997; --ks-gold:#D8AE4B; --ks-cyan:#58D8FF; --ks-teal:#50EEE0; --ks-magenta:#FF307C;
          --ks-green:#43D264; --ks-yellow:#F59E0B; --ks-red:#FF4757;
          --f-serif:"Cormorant Garamond","Cinzel",Georgia,serif; --f-cond:"Bebas Neue","Oswald","Arial Narrow",sans-serif;
          --f-mono:"JetBrains Mono","IBM Plex Mono",Consolas,monospace; }
  * { box-sizing:border-box; }
  html, body { margin:0; height:100%; background:var(--ks-void); color:var(--ks-text); font-family:var(--f-mono); font-size:12px; }
  nav { height:80px; background:var(--ks-navy); border-bottom:1px solid var(--ks-line); display:flex; align-items:center; gap:24px; padding:0 32px; }
  .brand { font-family:var(--f-serif); font-size:32px; color:var(--ks-gold); }
  .chapter { font-family:var(--f-cond); font-size:11px; letter-spacing:0.24em; text-transform:uppercase; color:var(--ks-muted); }
  #app { display:grid; grid-template-columns:minmax(0,2fr) minmax(360px,1fr); height:calc(100% - 80px); }
  #stage { position:relative; border-right:1px solid var(--ks-line); background:var(--ks-black); }
  #c { width:100%; height:100%; display:block; outline:none; }
  #hud { position:absolute; left:16px; bottom:16px; background:var(--ks-black); border:1px solid var(--ks-line); border-radius:2px; padding:8px 12px; color:var(--ks-muted); }
  #rail { overflow:auto; padding:24px; }
  .card { background:var(--ks-panel); border:1px solid var(--ks-line); border-radius:2px; padding:12px; margin-bottom:8px; cursor:pointer; transition:border-color 160ms; }
  .card:hover { border-color:var(--ks-cyan); } .card.sel { border-color:var(--ks-magenta); }
  .card .t { font-family:var(--f-cond); font-size:16px; letter-spacing:0.12em; }
  .card .m { color:var(--ks-muted); font-size:11px; margin-top:4px; }
  .row { display:flex; justify-content:space-between; border-top:1px solid var(--ks-line); padding:4px 0; }
  .v { color:var(--ks-teal); }
  .budget { display:grid; grid-template-columns:repeat(16,1fr); gap:4px; margin:8px 0; }
  .budget span { height:8px; border:1px solid var(--ks-line); border-radius:2px; } .budget span.on { background:var(--ks-cyan); border-color:var(--ks-cyan); }
  button, select { background:transparent; color:var(--ks-text); border:1px solid var(--ks-line); border-radius:4px; padding:8px 16px;
                   font-family:var(--f-cond); font-size:14px; letter-spacing:0.16em; text-transform:uppercase; cursor:pointer; transition:border-color 160ms; }
  button:hover { border-color:var(--ks-cyan); } button.gold { background:var(--ks-gold); color:var(--ks-black); border-color:var(--ks-gold); }
  .term { background:var(--ks-black); border:1px solid var(--ks-line); border-radius:2px; padding:12px; max-height:220px; overflow:auto; color:var(--ks-teal); font-size:11px; }
  .err { color:var(--ks-red); min-height:16px; margin:8px 0; }
  .tag { color:var(--ks-yellow); font-size:10px; letter-spacing:0.2em; }
</style>
</head>
<body>
<nav>
  <div><div class="chapter">Krystal Stack · Framework Arena</div><div class="brand">Clash Battle Royale</div></div>
  <div style="margin-left:auto;display:flex;gap:8px;align-items:center">
    <select id="faction"><option>SYNC</option><option>COMPUTE</option><option>VERIFY</option><option>OPS</option></select>
    <button class="gold" id="newGame">Nová hra</button>
    <button id="endTurn">Koniec kola</button>
  </div>
</nav>
<div id="app">
  <div id="stage"><canvas id="c"></canvas><div id="hud"></div></div>
  <aside id="rail">
    <div class="chapter">Kolo · zóna · pamäťový rozpočet (128 B rámce)</div>
    <div id="status"></div><div class="budget" id="budget"></div>
    <div class="err" id="err"></div>
    <div class="chapter">Vybraná jednotka</div><div id="unit">Klikni na svoju jednotku.</div>
    <div class="chapter" style="margin-top:16px">Karty = konfigurovateľné prvky frameworku</div><div id="cards"></div>
    <div class="chapter" style="margin-top:16px">Potvrdenia ťahov (receipts)</div><div class="term" id="log"></div>
    <div class="chapter" style="margin-top:16px">Riadiace centrum · známe vlastnosti hardvéru</div><div id="cc">Načítavam…</div>
  </aside>
</div>
<script src="vendor/babylon.js"></script>
<script>if (!window.BABYLON) { document.write('<script src="__BABYLON_URL__" integrity="__BABYLON_SRI__" crossorigin="anonymous"><\/script>'); }</script>
<script>
(function () {
  const $ = (id) => document.getElementById(id);
  const esc = (s) => String(s ?? "").replace(/[&<>"']/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
  let gameId = null, S = null, selUnit = null, selCard = null;
  if (!window.BABYLON) { $("hud").textContent = "Babylon.js sa nenačítal."; return; }

  const engine = new BABYLON.Engine($("c"), true), scene = new BABYLON.Scene(engine);
  scene.clearColor = new BABYLON.Color4(0.012, 0.016, 0.024, 1);
  const cam = new BABYLON.ArcRotateCamera("cam", -Math.PI / 2, Math.PI / 3.4, 15, BABYLON.Vector3.Zero(), scene);
  cam.attachControl($("c"), true); cam.lowerRadiusLimit = 8; cam.upperRadiusLimit = 30;
  new BABYLON.HemisphericLight("h", new BABYLON.Vector3(0, 1, 0), scene).intensity = 0.9;
  const C = (h) => BABYLON.Color3.FromHexString(h), SQ3 = Math.sqrt(3);
  const pos = (q, r) => new BABYLON.Vector3(SQ3 * (q + r / 2), 0, 1.5 * r);
  const dist = (a, b) => (Math.abs(a[0]-b[0]) + Math.abs(a[1]-b[1]) + Math.abs(a[0]+a[1]-b[0]-b[1])) / 2;

  function hexMesh(name, c, rad, color, y) {
    const p = [c.x, y, c.z], idx = [];
    for (let i = 0; i < 6; i++) { const a = Math.PI / 180 * (60 * i - 30); p.push(c.x + rad * Math.cos(a), y, c.z + rad * Math.sin(a)); }
    for (let i = 1; i <= 6; i++) idx.push(0, i, i === 6 ? 1 : i + 1);
    const m = new BABYLON.Mesh(name, scene), vd = new BABYLON.VertexData(), n = [];
    vd.positions = p; vd.indices = idx; BABYLON.VertexData.ComputeNormals(p, idx, n); vd.normals = n; vd.applyToMesh(m);
    const mat = new BABYLON.StandardMaterial(name + "m", scene); mat.diffuseColor = color; mat.specularColor = BABYLON.Color3.Black(); mat.backFaceCulling = false;
    m.material = mat; return m;
  }
  function outline(name, c, rad, color, y) {
    const pts = []; for (let i = 0; i <= 6; i++) { const a = Math.PI / 180 * (60 * (i % 6) - 30); pts.push(new BABYLON.Vector3(c.x + rad * Math.cos(a), y, c.z + rad * Math.sin(a))); }
    const l = BABYLON.MeshBuilder.CreateLines(name, { points: pts }, scene); l.color = color; l.isPickable = false; return l;
  }

  let tiles = [], zoneLines = [], unitMeshes = [];
  function buildBoard() {
    tiles.forEach(t => t.dispose()); tiles = [];
    S.cells.forEach(([q, r]) => {
      const t = hexMesh(`t${q}_${r}`, pos(q, r), 0.95, C("#11131D"), 0); t.metadata = { cell: [q, r] }; tiles.push(t);
      outline(`o${q}_${r}`, pos(q, r), 0.95, C("#292A38"), 0.002);
    });
  }
  function render() {
    tiles.forEach(t => { const out = dist(t.metadata.cell, [0, 0]) > S.zone_radius; t.material.diffuseColor = out ? C("#2a0c12") : C("#11131D"); });
    zoneLines.forEach(l => l.dispose()); zoneLines = [];
    S.cells.filter(c => dist(c, [0, 0]) === S.zone_radius).forEach(([q, r]) => zoneLines.push(outline(`z${q}_${r}`, pos(q, r), 0.8, C("#F59E0B"), 0.004)));
    unitMeshes.forEach(m => m.dispose()); unitMeshes = [];
    Object.values(S.units).filter(u => u.hp > 0).forEach(u => {
      const h = 0.15 + u.hp * 0.15, m = BABYLON.MeshBuilder.CreateCylinder(u.unit_id, { diameter: 0.7, height: h, tessellation: 6 }, scene);
      const p = pos(u.pos[0], u.pos[1]); m.position.set(p.x, h / 2, p.z); m.metadata = { unit: u.unit_id };
      const mat = new BABYLON.StandardMaterial(u.unit_id + "m", scene), col = C(S.factions[u.faction].color);
      mat.diffuseColor = col; mat.emissiveColor = col.scale(selUnit === u.unit_id ? 0.8 : 0.25); mat.specularColor = BABYLON.Color3.Black();
      m.material = mat; unitMeshes.push(m);
    });
    const pf = S.player_faction, b = S.budget[pf];
    $("status").innerHTML = `<div class="row"><span>KOLO</span><span class="v">${S.round}</span></div>` +
      `<div class="row"><span>POLOMER ZÓNY</span><span class="v">${S.zone_radius}</span></div>` +
      `<div class="row"><span>FRAKCIA</span><span class="v" style="color:${S.factions[pf].color}">${pf}</span></div>` +
      (S.winner ? `<div class="row"><span>VÍŤAZ</span><span class="v" style="color:#F0D071">${esc(S.winner)}</span></div>` : "");
    $("budget").innerHTML = Array.from({length: 16}, (_, i) => `<span class="${i < b ? "on" : ""}"></span>`).join("");
    const u = selUnit && S.units[selUnit];
    $("unit").innerHTML = u ? `<div class="card sel"><div class="t">${esc(u.unit_id)} · ${esc(u.faction)}</div><div class="m">${esc(u.name)}</div>` +
      `<div class="row"><span>HP</span><span class="v">${u.hp} / ${S.vital_max_hp}</span></div>` +
      `<div class="row"><span>ÚTOK / OBRANA</span><span class="v">${u.atk}+${u.tmp_atk} / ${u.def}+${u.tmp_def}</span></div>` +
      `<div class="row"><span>VERDIKT KONTROLY</span><span class="v">${esc(u.verdict)}</span></div>` +
      `<div class="row"><span>ŤAH V KOLE</span><span class="v">${u.acted ? "použitý" : "voľný"}</span></div></div>` : "Klikni na svoju jednotku.";
    $("cards").innerHTML = Object.entries(S.cards).map(([id, c]) =>
      `<div class="card${selCard === id ? " sel" : ""}" data-card="${esc(id)}"><div class="t">${esc(id)} · ${c.cost} rámcov</div>` +
      `<div class="m">${esc(c.effect)}</div><div class="m">↳ ${esc(c.framework_ref)}</div></div>`).join("");
    $("log").innerHTML = S.log.slice(-14).reverse().map(l =>
      `<div>[${l.round}] ${esc(l.faction)} · ${esc(l.action)} · ${l.frames} rámcov · ${esc(l.framework_step)} · #${esc(l.frame_sha256_prefix)}</div>`).join("");
    $("hud").innerHTML = `STATE ${S.state_sha256.slice(0, 12)} · HP CAP ${S.vital_max_hp} · klik: jednotka → hex (pohyb) / nepriateľ (útok) / karta → vlastná jednotka`;
  }

  async function post(url, body) {
    const r = await fetch(url, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
    const d = await r.json(); if (!r.ok) throw d; return d;
  }
  async function act(action) {
    try { const d = await post("/api/v1/runtime/clash/action", { game_id: gameId, action }); S = d.state; $("err").textContent = ""; }
    catch (e) { $("err").textContent = `${e.violation_code || "ERROR"}: ${e.error || e} (stav sa nezmenil)`; }
    render();
  }
  async function newGame() {
    const d = await post("/api/v1/runtime/clash/new", { faction: $("faction").value, seed: 6 });
    gameId = d.game_id; S = d.state; selUnit = selCard = null; buildBoard(); render();
  }

  scene.onPointerObservable.add((pi) => {
    if (pi.type !== BABYLON.PointerEventTypes.POINTERTAP || !S || !pi.pickInfo.hit) return;
    const md = pi.pickInfo.pickedMesh.metadata || {};
    if (md.unit) {
      const u = S.units[md.unit];
      if (u.faction === S.player_faction) {
        if (selCard) { act({ type: "card", card: selCard, target: md.unit }); selCard = null; return; }
        selUnit = md.unit; render(); return;
      }
      if (selUnit) act({ type: "attack", unit: selUnit, target: md.unit });
      return;
    }
    if (md.cell && selUnit) act({ type: "move", unit: selUnit, to: md.cell });
  });
  document.addEventListener("click", (e) => { const c = e.target.closest("[data-card]"); if (c) { selCard = selCard === c.dataset.card ? null : c.dataset.card; render(); } });
  $("newGame").onclick = newGame;
  $("endTurn").onclick = () => act({ type: "end_turn" });

  async function controlCenter() {
    try {
      const f = await (await fetch("/api/v1/runtime/facts")).json();
      const val = (id) => (f.facts.find(x => x.fact_id === id) || {}).value;
      const kern = (dev) => ((f.facts.find(x => x.fact_id.startsWith("F_SW_") && x.fact_id.endsWith("_" + dev)) || {}).value || {}).kernel;
      let html = ["CPU", "GPU"].map(d => `<div class="card"><div class="t">${d}</div>` +
        `<div class="row"><span>KERNEL</span><span class="v">${esc(kern(d))}</span></div>` +
        `<div class="row"><span>MEDIÁN / P95 <span class="tag">[MEASURED]</span></span><span class="v">${val("F_C03_" + d + "_median_ms")} / ${val("F_C03_" + d + "_p95_ms")} ms</span></div>` +
        `<div class="row"><span>KOMPILÁCIA</span><span class="v">${val("F_C03_" + d + "_compile_ms")} ms</span></div></div>`).join("");
      html += `<div class="row"><span>BLAS PRAH K <span class="tag">[MEASURED]</span></span><span class="v">${val("F_C06_threshold_k")}</span></div>`;
      html += `<div class="row"><span>INTEGRITA ONTOLÓGIE</span><span class="v">${esc(f.ontology_integrity.verdict)}</span></div>`;
      const ns = [100, 530, 2000, 10000], preds = await Promise.all(ns.map(n => fetch(`/api/v1/runtime/allocate?n=${n}`).then(r => r.json())));
      html += `<div class="chapter" style="margin-top:12px">Predikcia alokácie <span class="tag">[DERIVED]</span></div>` +
        preds.map((p, i) => `<div class="row"><span>n = ${ns[i]}</span><span class="v">${esc(p.device)} · ${p.table ? Object.entries(p.table).map(([d, t]) => d + " " + t.cost_ms + " ms").join(" / ") : esc(p.reason)}</span></div>`).join("");
      $("cc").innerHTML = html;
    } catch (e) { $("cc").textContent = "Riadiace centrum nedostupné: " + e; }
  }
  newGame(); controlCenter();
  engine.runRenderLoop(() => scene.render());
  window.addEventListener("resize", () => engine.resize());
})();
</script>
</body>
</html>
"""


def render_clash_arena_html() -> str:
    return (_TEMPLATE.replace("__BABYLON_URL__", BABYLON_URL).replace("__BABYLON_SRI__", BABYLON_SRI)
            .replace("__BABYLON_VERSION__", BABYLON_VERSION))
