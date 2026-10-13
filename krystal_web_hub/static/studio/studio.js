"use strict";
let CULL = true;
/* Krystal Studio – SAP-style shell + Adobe/Inkscape/GIMP hybrid shortcuts */
const $ = (s, r = document) => r.querySelector(s);
const cv = $("#cv"), ctx = cv.getContext("2d");
const S = { objs: [], sel: new Set(), tool: "move", zoom: 1, pan: { x: 40, y: 40 }, hist: [], hi: -1, id: 1, drag: null, space: false, doc: { w: 1280, h: 720 } };

/* ---------- Tools ---------- */
const TOOLS = [
  { id: "move", ic: "↖", key: "v", name: "Výber / Move" },
  { id: "rect", ic: "▭", key: "r", name: "Obdĺžnik" },
  { id: "ellipse", ic: "◯", key: "e", name: "Elipsa" },
  { id: "line", ic: "╱", key: "l", name: "Čiara" },
  { id: "brush", ic: "✎", key: "b", name: "Štetec" },
  { id: "text", ic: "T", key: "t", name: "Text" },
  { id: "hand", ic: "✋", key: "h", name: "Posun" },
  { id: "zoom", ic: "🔍", key: "z", name: "Lupa" },
];
const tb = $("#toolbox");
TOOLS.forEach(t => { const b = document.createElement("button"); b.className = "tool"; b.dataset.t = t.id; b.title = `${t.name} (${t.key.toUpperCase()})`; b.innerHTML = `${t.ic}<small>${t.key.toUpperCase()}</small>`; b.onclick = () => setTool(t.id); tb.append(b); });
function setTool(id) { S.tool = id; document.querySelectorAll(".tool").forEach(b => b.classList.toggle("active", b.dataset.t === id)); $("#st-tool").textContent = "Tool: " + TOOLS.find(t => t.id === id).name; cv.style.cursor = id === "hand" ? "grab" : id === "move" ? "default" : "crosshair"; }

/* ---------- Document ops ---------- */
const style = () => ({ fill: $("#fill").value, stroke: $("#stroke").value, sw: +$("#sw").value, op: $("#opacity").value / 100 });
function commit(label) { S.hist = S.hist.slice(0, S.hi + 1); S.hist.push({ label, data: JSON.stringify(S.objs) }); S.hi = S.hist.length - 1; refresh(); }
function restore(i) { if (i < 0 || i >= S.hist.length) return; S.hi = i; S.objs = JSON.parse(S.hist[i].data); S.sel = new Set([...S.sel].filter(id => S.objs.some(o => o.id === id))); refresh(); }
const undo = () => restore(S.hi - 1), redo = () => restore(S.hi + 1);
const selected = () => S.objs.filter(o => S.sel.has(o.id));
function add(o) { o.id = S.id++; o.name = o.type + " " + o.id; o.vis = true; Object.assign(o, style()); S.objs.push(o); S.sel = new Set([o.id]); return o; }
function del() { S.objs = S.objs.filter(o => !S.sel.has(o.id)); S.sel.clear(); commit("Zmazať"); }
function dup() { const c = selected().map(o => ({ ...JSON.parse(JSON.stringify(o)), id: S.id++, x: (o.x || 0) + 16, y: (o.y || 0) + 16 })); c.forEach(o => o.name += " kópia"); S.objs.push(...c); S.sel = new Set(c.map(o => o.id)); commit("Duplikovať"); }
function zorder(dir) { const ids = S.sel; if (dir > 0) { for (let i = S.objs.length - 2; i >= 0; i--) if (ids.has(S.objs[i].id) && !ids.has(S.objs[i + 1].id)) [S.objs[i], S.objs[i + 1]] = [S.objs[i + 1], S.objs[i]]; } else { for (let i = 1; i < S.objs.length; i++) if (ids.has(S.objs[i].id) && !ids.has(S.objs[i - 1].id)) [S.objs[i], S.objs[i - 1]] = [S.objs[i - 1], S.objs[i]]; } commit(dir > 0 ? "Vyššie" : "Nižšie"); }
function nudge(dx, dy) { selected().forEach(o => { o.x += dx; o.y += dy; }); commit("Posun"); }
function swapColors() { const f = $("#fill"), s = $("#stroke"); [f.value, s.value] = [s.value, f.value]; }
function align(mode) { const sel = selected(); if (!sel.length) return; const W = S.doc.w, H = S.doc.h; sel.forEach(o => { const w = o.w || 0, h = o.h || 0; if (mode === "cx") o.x = (W - w) / 2; if (mode === "cy") o.y = (H - h) / 2; if (mode === "l") o.x = 0; if (mode === "t") o.y = 0; }); commit("Zarovnať"); }

/* ---------- View ---------- */
const W2S = (x, y) => ({ x: x * S.zoom + S.pan.x, y: y * S.zoom + S.pan.y });
const S2W = (x, y) => ({ x: (x - S.pan.x) / S.zoom, y: (y - S.pan.y) / S.zoom });
function setZoom(z, cx = cv.width / 2, cy = cv.height / 2) { z = Math.max(.05, Math.min(32, z)); const w = S2W(cx, cy); S.zoom = z; S.pan.x = cx - w.x * z; S.pan.y = cy - w.y * z; $("#zoomlbl").textContent = Math.round(z * 100) + " %"; draw(); }
function fit() { const z = Math.min((cv.width - 80) / S.doc.w, (cv.height - 80) / S.doc.h); S.zoom = z; S.pan = { x: (cv.width - S.doc.w * z) / 2, y: (cv.height - S.doc.h * z) / 2 }; setZoom(z); }
function resize() { const r = cv.parentElement.getBoundingClientRect(); cv.width = r.width; cv.height = r.height; draw(); }
addEventListener("resize", resize);

/* ---------- Render ---------- */
function bounds(o) { if (o.type === "brush") { const xs = o.pts.map(p => p[0]), ys = o.pts.map(p => p[1]); const x = Math.min(...xs), y = Math.min(...ys); return { x, y, w: Math.max(...xs) - x, h: Math.max(...ys) - y }; } return { x: o.x, y: o.y, w: o.w || 0, h: o.h || 0 }; }
function drawObj(o) {
  ctx.save(); ctx.globalAlpha = o.op; ctx.fillStyle = o.fill; ctx.strokeStyle = o.stroke; ctx.lineWidth = o.sw; ctx.lineCap = ctx.lineJoin = "round";
  if (o.type === "rect") { ctx.fillRect(o.x, o.y, o.w, o.h); if (o.sw) ctx.strokeRect(o.x, o.y, o.w, o.h); }
  else if (o.type === "ellipse") { ctx.beginPath(); ctx.ellipse(o.x + o.w / 2, o.y + o.h / 2, Math.abs(o.w / 2), Math.abs(o.h / 2), 0, 0, 7); ctx.fill(); if (o.sw) ctx.stroke(); }
  else if (o.type === "line") { ctx.beginPath(); ctx.moveTo(o.x, o.y); ctx.lineTo(o.x + o.w, o.y + o.h); ctx.stroke(); }
  else if (o.type === "brush") { ctx.beginPath(); o.pts.forEach((p, i) => i ? ctx.lineTo(p[0], p[1]) : ctx.moveTo(p[0], p[1])); ctx.lineWidth = Math.max(1, o.sw); ctx.stroke(); }
  else if (o.type === "text") { ctx.font = "32px Inter, sans-serif"; ctx.textBaseline = "top"; ctx.fillText(o.text, o.x, o.y); o.w = ctx.measureText(o.text).width; o.h = 36; }
  ctx.restore();
}
function draw(exporting) {
  ctx.setTransform(1, 0, 0, 1, 0, 0); ctx.clearRect(0, 0, cv.width, cv.height);
  if (!exporting) { ctx.setTransform(S.zoom, 0, 0, S.zoom, S.pan.x, S.pan.y); ctx.fillStyle = "#14161f"; ctx.fillRect(0, 0, S.doc.w, S.doc.h); if ($("#grid").checked && S.zoom > .4) { ctx.strokeStyle = "#ffffff10"; ctx.lineWidth = 1 / S.zoom; ctx.beginPath(); for (let x = 0; x <= S.doc.w; x += 32) { ctx.moveTo(x, 0); ctx.lineTo(x, S.doc.h); } for (let y = 0; y <= S.doc.h; y += 32) { ctx.moveTo(0, y); ctx.lineTo(S.doc.w, y); } ctx.stroke(); } }
  const vx0 = -S.pan.x / S.zoom, vy0 = -S.pan.y / S.zoom, vx1 = vx0 + cv.width / S.zoom, vy1 = vy0 + cv.height / S.zoom;
  S.objs.forEach(o => { if (!o.vis) return; if (CULL && !exporting) { const b = bounds(o), p = o.sw + 2; if (Math.max(b.x, b.x + b.w) + p < vx0 || Math.min(b.x, b.x + b.w) - p > vx1 || Math.max(b.y, b.y + b.h) + p < vy0 || Math.min(b.y, b.y + b.h) - p > vy1) return; } drawObj(o); });
  if (exporting) return;
  S.objs.filter(o => S.sel.has(o.id)).forEach(o => { const b = bounds(o); ctx.strokeStyle = "#22d3ee"; ctx.lineWidth = 1 / S.zoom; ctx.setLineDash([4 / S.zoom, 3 / S.zoom]); ctx.strokeRect(b.x, b.y, b.w, b.h); ctx.setLineDash([]); });
  ctx.setTransform(1, 0, 0, 1, 0, 0);
}
function hit(x, y) { for (let i = S.objs.length - 1; i >= 0; i--) { const o = S.objs[i]; if (!o.vis) continue; const b = bounds(o), p = 4 / S.zoom; if (x >= Math.min(b.x, b.x + b.w) - p && x <= Math.max(b.x, b.x + b.w) + p && y >= Math.min(b.y, b.y + b.h) - p && y <= Math.max(b.y, b.y + b.h) + p) return o; } return null; }

/* ---------- Mouse ---------- */
const snap = v => $("#snap").checked ? Math.round(v / 8) * 8 : v;
cv.addEventListener("pointerdown", e => {
  cv.setPointerCapture(e.pointerId);
  const w = S2W(e.offsetX, e.offsetY), t = S.space || e.button === 1 ? "hand" : S.tool;
  S.drag = { t, sx: e.offsetX, sy: e.offsetY, wx: snap(w.x), wy: snap(w.y), pan0: { ...S.pan }, moved: false };
  if (t === "move") { const o = hit(w.x, w.y); if (o) { if (e.shiftKey) S.sel.has(o.id) ? S.sel.delete(o.id) : S.sel.add(o.id); else if (!S.sel.has(o.id)) S.sel = new Set([o.id]); S.drag.last = w; } else if (!e.shiftKey) S.sel.clear(); refresh(true); }
  else if (t === "rect" || t === "ellipse" || t === "line") S.drag.o = add({ type: t, x: S.drag.wx, y: S.drag.wy, w: 0, h: 0 });
  else if (t === "brush") S.drag.o = add({ type: "brush", pts: [[w.x, w.y]] });
  else if (t === "text") { const s = prompt("Text:"); if (s) { add({ type: "text", x: w.x, y: w.y, w: 0, h: 0, text: s }); commit("Text"); } S.drag = null; }
  else if (t === "zoom") { setZoom(S.zoom * (e.altKey ? .5 : 2), e.offsetX, e.offsetY); S.drag = null; }
});
cv.addEventListener("pointermove", e => {
  const w = S2W(e.offsetX, e.offsetY); $("#st-pos").textContent = `${w.x | 0}, ${w.y | 0}`;
  const d = S.drag; if (!d) return; d.moved = true;
  if (d.t === "hand") { S.pan.x = d.pan0.x + e.offsetX - d.sx; S.pan.y = d.pan0.y + e.offsetY - d.sy; }
  else if (d.t === "move" && d.last) { const dx = snap(w.x) - snap(d.last.x), dy = snap(w.y) - snap(d.last.y); if (dx || dy) { selected().forEach(o => { if (o.type === "brush") o.pts.forEach(p => { p[0] += dx; p[1] += dy; }); else { o.x += dx; o.y += dy; } }); d.last = w; } }
  else if (d.o && d.o.type === "brush") d.o.pts.push([w.x, w.y]);
  else if (d.o) { let ww = snap(w.x) - d.wx, hh = snap(w.y) - d.wy; if (e.shiftKey && d.t !== "line") { const m = Math.max(Math.abs(ww), Math.abs(hh)); ww = Math.sign(ww || 1) * m; hh = Math.sign(hh || 1) * m; } d.o.w = ww; d.o.h = hh; }
  requestDraw();
});
cv.addEventListener("pointerup", () => { const d = S.drag; S.drag = null; if (!d || !d.moved) return; if (d.t === "move") commit("Presunúť"); else if (d.o) commit("Kresliť " + d.o.type); });
cv.addEventListener("wheel", e => { e.preventDefault(); if (e.ctrlKey || e.altKey) setZoom(S.zoom * (e.deltaY < 0 ? 1.1 : 1 / 1.1), e.offsetX, e.offsetY); else { S.pan.x -= e.shiftKey ? e.deltaY : e.deltaX; S.pan.y -= e.shiftKey ? 0 : e.deltaY; draw(); } }, { passive: false });

/* ---------- Panels ---------- */
function refresh(skipHist) {
  const ul = $("#layers"); ul.innerHTML = "";
  [...S.objs].reverse().forEach(o => { const li = document.createElement("li"); li.className = S.sel.has(o.id) ? "sel" : ""; li.innerHTML = `<span class="eye ${o.vis ? "" : "off"}">👁</span><span>${o.name}</span>`; li.onclick = e => { if (e.target.classList.contains("eye")) o.vis = !o.vis; else S.sel = new Set([o.id]); refresh(true); }; ul.append(li); });
  const s = selected(); $("#props").innerHTML = s.length ? (s.length > 1 ? `<dt>Výber</dt><dd>${s.length} objektov</dd>` : (b => `<dt>Názov</dt><dd>${s[0].name}</dd><dt>X, Y</dt><dd>${b.x | 0}, ${b.y | 0}</dd><dt>Š × V</dt><dd>${Math.abs(b.w) | 0} × ${Math.abs(b.h) | 0}</dd><dt>Typ</dt><dd>${s[0].type}</dd>`)(bounds(s[0]))) : "<dt>Výber</dt><dd>—</dd>";
  $("#history").innerHTML = S.hist.map((h, i) => `<li class="${i === S.hi ? "cur" : ""}">${h.label}</li>`).join("");
  draw();
}

/* ---------- SAP-style data views ---------- */
const DATA = {
  projects: [["ID", "Projekt", "Vlastník", "Stav"], ["P-1001", "Parametrický render set", "studio", "Aktívny"], ["P-1002", "Monetizačné vizuály", "marketing", "Aktívny"], ["P-1003", "Poslední Kmen UI", "design", "Revízia"]],
  renders: [["Job", "Preset", "Rozlíšenie", "Stav"], ["R-204", "Isometric city", "3840×2160", "Hotovo"], ["R-205", "Tier matrix", "2048×1152", "Beží"], ["R-206", "Tokenomics", "2048×1152", "Čaká"]],
  billing: [["Tier", "Cena", "Render kredity", "Stav"], ["Developer Free", "0 €", "100", "Aktívny"], ["Indie Studio", "49 €", "5 000", "Dostupný"], ["Enterprise Pro", "499 €", "100 000", "Dostupný"]],
  audit: [["Čas", "Používateľ", "Akcia", "Objekt"], [new Date().toISOString().slice(0, 19), "studio", "Otvorený workspace", "P-1001"]],
};
Object.entries(DATA).forEach(([k, rows]) => { $("#tbl-" + k).innerHTML = `<thead><tr>${rows[0].map(h => `<th>${h}</th>`).join("")}</tr></thead><tbody>${rows.slice(1).map(r => `<tr>${r.map((c, i) => `<td>${/Aktívny|Hotovo|Beží/.test(c) && i === r.length - 1 ? `<span class="badge">${c}</span>` : /Čaká|Revízia|Dostupný/.test(c) && i === r.length - 1 ? `<span class="badge warn">${c}</span>` : c}</td>`).join("")}</tr>`).join("")}</tbody>`; });
function showView(v) { document.querySelectorAll(".view").forEach(x => x.classList.toggle("active", x.id === "view-" + v)); document.querySelectorAll(".nav-item").forEach(x => x.classList.toggle("active", x.dataset.view === v)); if (v === "canvas") resize(); }
document.querySelectorAll(".nav-item").forEach(b => b.onclick = () => showView(b.dataset.view));

/* ---------- Commands & shortcuts (single registry => menu, palette, help) ---------- */
const exportPNG = () => { const c = document.createElement("canvas"); c.width = S.doc.w; c.height = S.doc.h; const o = c.getContext("2d"); o.fillStyle = "#14161f"; o.fillRect(0, 0, c.width, c.height); renderTo(o); const a = document.createElement("a"); a.download = "krystal-studio.png"; a.href = c.toDataURL(); a.click(); };
function renderTo(c) { const real = ctx; const orig = drawObj; /* temporary context swap */ S.objs.forEach(o => { if (!o.vis) return; c.save(); c.globalAlpha = o.op; c.fillStyle = o.fill; c.strokeStyle = o.stroke; c.lineWidth = o.sw; c.lineCap = c.lineJoin = "round"; if (o.type === "rect") { c.fillRect(o.x, o.y, o.w, o.h); if (o.sw) c.strokeRect(o.x, o.y, o.w, o.h); } else if (o.type === "ellipse") { c.beginPath(); c.ellipse(o.x + o.w / 2, o.y + o.h / 2, Math.abs(o.w / 2), Math.abs(o.h / 2), 0, 0, 7); c.fill(); if (o.sw) c.stroke(); } else if (o.type === "line") { c.beginPath(); c.moveTo(o.x, o.y); c.lineTo(o.x + o.w, o.y + o.h); c.stroke(); } else if (o.type === "brush") { c.beginPath(); o.pts.forEach((p, i) => i ? c.lineTo(p[0], p[1]) : c.moveTo(p[0], p[1])); c.stroke(); } else if (o.type === "text") { c.font = "32px Inter, sans-serif"; c.textBaseline = "top"; c.fillText(o.text, o.x, o.y); } c.restore(); }); }
const save = () => { localStorage.setItem("krystal-studio", JSON.stringify({ objs: S.objs, id: S.id })); msg("Uložené"); };
const load = () => { const d = JSON.parse(localStorage.getItem("krystal-studio") || "null"); if (d) { S.objs = d.objs; S.id = d.id; commit("Načítať"); } };
const msg = t => $("#st-msg").textContent = t;
const toggle = id => { const e = $(id); e.checked = !e.checked; draw(); };

const CMDS = [
  ["Súbor", "Uložiť", "Ctrl+S", save], ["Súbor", "Načítať", "Ctrl+O", load], ["Súbor", "Exportovať PNG", "Ctrl+Shift+E", exportPNG],
  ["Úpravy", "Späť", "Ctrl+Z", undo], ["Úpravy", "Znova", "Ctrl+Shift+Z", redo], ["Úpravy", "Duplikovať", "Ctrl+D", dup], ["Úpravy", "Zmazať", "Delete", del],
  ["Úpravy", "Vybrať všetko", "Ctrl+A", () => { S.sel = new Set(S.objs.map(o => o.id)); refresh(true); }], ["Úpravy", "Zrušiť výber", "Ctrl+Shift+A", () => { S.sel.clear(); refresh(true); }],
  ["Úpravy", "Prehodiť farby", "X", swapColors], ["Úpravy", "Predvolené farby", "D", () => { $("#fill").value = "#ffffff"; $("#stroke").value = "#000000"; }],
  ["Objekt", "Presunúť vyššie", "Ctrl+]", () => zorder(1)], ["Objekt", "Presunúť nižšie", "Ctrl+[", () => zorder(-1)],
  ["Objekt", "Zarovnať na stred X", "Alt+H", () => align("cx")], ["Objekt", "Zarovnať na stred Y", "Alt+V", () => align("cy")],
  ["Zobrazenie", "Priblížiť", "Ctrl+=", () => setZoom(S.zoom * 1.25)], ["Zobrazenie", "Oddialiť", "Ctrl+-", () => setZoom(S.zoom / 1.25)], ["Zobrazenie", "Prispôsobiť oknu", "Ctrl+0", fit], ["Zobrazenie", "100 %", "Ctrl+1", () => setZoom(1)],
  ["Zobrazenie", "Mriežka", "Ctrl+'", () => toggle("#grid")], ["Zobrazenie", "Snap", "Ctrl+;", () => toggle("#snap")],
  ["Zobrazenie", "Skryť panely", "Tab", () => document.querySelectorAll(".sidenav,.inspector,.toolbox,.optionsbar").forEach(e => e.classList.toggle("hidden")) || resize()],
  ["Pomoc", "Benchmark (naive vs sync)", "Ctrl+Shift+B", runBench],
  ["Pomoc", "Paleta príkazov", "Ctrl+K", () => openPal()], ["Pomoc", "Klávesové skratky", "F1", () => $("#help").classList.toggle("hidden")],
  ...TOOLS.map(t => ["Nástroje", t.name, t.key.toUpperCase(), () => setTool(t.id)]),
  ["Navigácia", "Canvas", "Alt+1", () => showView("canvas")], ["Navigácia", "Projekty", "Alt+2", () => showView("projects")], ["Navigácia", "Render Queue", "Alt+3", () => showView("renders")], ["Navigácia", "Billing", "Alt+4", () => showView("billing")], ["Navigácia", "Audit", "Alt+5", () => showView("audit")],
];
const matches = (e, combo) => { const p = combo.split("+"), k = p.pop(); const ctrl = p.includes("Ctrl"), sh = p.includes("Shift"), alt = p.includes("Alt"); const key = e.key.length === 1 ? e.key.toLowerCase() : e.key; return key === (k.length === 1 ? k.toLowerCase() : k) && (e.ctrlKey || e.metaKey) === ctrl && e.shiftKey === sh && e.altKey === alt; };
// menu bar
const mb = $("#menubar"); [...new Set(CMDS.map(c => c[0]))].filter(g => g !== "Navigácia").forEach(g => { const m = document.createElement("div"); m.className = "menu"; m.innerHTML = `<button>${g}</button><div class="dropdown"></div>`; CMDS.filter(c => c[0] === g).forEach(c => { const b = document.createElement("button"); b.innerHTML = `<span>${c[1]}</span><span>${c[2]}</span>`; b.onclick = () => { m.classList.remove("open"); c[3](); }; m.lastChild.append(b); }); m.firstChild.onclick = e => { e.stopPropagation(); const o = m.classList.contains("open"); document.querySelectorAll(".menu").forEach(x => x.classList.remove("open")); if (!o) m.classList.add("open"); }; mb.append(m); });
addEventListener("click", () => document.querySelectorAll(".menu").forEach(x => x.classList.remove("open")));
// help
$("#help-body").innerHTML = [...new Set(CMDS.map(c => c[0]))].map(g => `<h3>${g}</h3>` + CMDS.filter(c => c[0] === g).map(c => `<div><span>${c[1]}</span><kbd>${c[2]}</kbd></div>`).join("")).join("") + `<h3>Myš</h3><div><span>Posun plátna</span><kbd>Space+drag / MMB</kbd></div><div><span>Zoom</span><kbd>Ctrl+koliesko</kbd></div><div><span>Štvorec/kruh</span><kbd>Shift pri ťahu</kbd></div><div><span>Pridať do výberu</span><kbd>Shift+klik</kbd></div><div><span>Oddialiť (Lupa)</span><kbd>Alt+klik</kbd></div><div><span>Posun o 1/10 px</span><kbd>Šípky / Shift+šípky</kbd></div>`;
$("#help").onclick = e => e.target.id === "help" && $("#help").classList.add("hidden");
// palette
let pSel = 0, pList = [];
function openPal() { $("#palette").classList.remove("hidden"); $("#pal-input").value = ""; $("#pal-input").focus(); renderPal(); }
function renderPal() { const q = $("#pal-input").value.toLowerCase(); pList = CMDS.filter(c => (c[0] + " " + c[1]).toLowerCase().includes(q)); pSel = Math.min(pSel, Math.max(0, pList.length - 1)); $("#pal-list").innerHTML = pList.map((c, i) => `<li class="${i === pSel ? "sel" : ""}" data-i="${i}"><span>${c[0]} › ${c[1]}</span><kbd>${c[2]}</kbd></li>`).join(""); }
function runPal(i) { const c = pList[i]; $("#palette").classList.add("hidden"); if (c) c[3](); }
$("#pal-input").oninput = () => { pSel = 0; renderPal(); };
$("#pal-input").onkeydown = e => { if (e.key === "ArrowDown") pSel = Math.min(pSel + 1, pList.length - 1); else if (e.key === "ArrowUp") pSel = Math.max(0, pSel - 1); else if (e.key === "Enter") return runPal(pSel); else if (e.key === "Escape") return $("#palette").classList.add("hidden"); else return; e.preventDefault(); renderPal(); };
$("#pal-list").onclick = e => { const li = e.target.closest("li"); if (li) runPal(+li.dataset.i); };
$("#palette").onclick = e => e.target.id === "palette" && $("#palette").classList.add("hidden");
$("#btn-palette").onclick = openPal; $("#btn-help").onclick = () => $("#help").classList.toggle("hidden");

addEventListener("keydown", e => {
  if (e.target.tagName === "INPUT" && e.target.type !== "checkbox" && e.target.type !== "range" && e.target.type !== "color") return;
  if (e.code === "Space") { S.space = true; e.preventDefault(); return; }
  if (e.key === "Escape") { document.querySelectorAll(".overlay").forEach(o => o.classList.add("hidden")); S.sel.clear(); refresh(true); return; }
  const ar = { ArrowLeft: [-1, 0], ArrowRight: [1, 0], ArrowUp: [0, -1], ArrowDown: [0, 1] }[e.key];
  if (ar && S.sel.size) { e.preventDefault(); const n = e.shiftKey ? 10 : 1; return nudge(ar[0] * n, ar[1] * n); }
  if (e.key === "Backspace") { e.preventDefault(); return del(); }
  for (const c of CMDS) if (matches(e, c[2])) { e.preventDefault(); c[3](); return; }
});
addEventListener("keyup", e => { if (e.code === "Space") S.space = false; });
["fill", "stroke", "sw", "opacity"].forEach(id => $("#" + id).addEventListener("input", () => { if (S.sel.size) { const st = style(); selected().forEach(o => Object.assign(o, st)); draw(); } }));
["grid", "snap"].forEach(id => $("#" + id).addEventListener("change", draw));

/* ---------- Frame sync: dirty flag + rAF coalescing (1 draw / vsync) ---------- */
let dirty = false;
function requestDraw() { if (dirty) return; dirty = true; requestAnimationFrame(() => { dirty = false; draw(); }); }

/* ---------- GL-mark style benchmark: naive vs time-synchronised ---------- */
async function runBench() {
  showView("canvas");
  const keep = { objs: S.objs, sel: S.sel, zoom: S.zoom, pan: { ...S.pan }, cull: CULL };
  const N = 3000; let seed = 1337; const rnd = () => (seed = (seed * 1664525 + 1013904223) >>> 0) / 4294967296;
  const T = ["rect", "ellipse", "line"];
  S.objs = Array.from({ length: N }, (_, i) => ({ id: i, name: "b" + i, vis: true, type: T[i % 3], x: rnd() * 4000 - 1500, y: rnd() * 2500 - 900, w: 20 + rnd() * 120, h: 20 + rnd() * 120, fill: `hsl(${rnd() * 360},70%,55%)`, stroke: "#fff", sw: 1, op: .8 }));
  S.sel = new Set(); S.zoom = 0.5; S.pan = { x: 0, y: 0 };
  const DUR = 2500;
  const stats = (ft, cpu) => { const s = [...ft].sort((a, b) => a - b), m = ft.reduce((a, b) => a + b, 0) / ft.length, sd = Math.sqrt(ft.reduce((a, b) => a + (b - m) ** 2, 0) / ft.length); const c = cpu.reduce((a, b) => a + b, 0) / cpu.length; return { fps: 1000 / m, avg: m, p99: s[Math.floor(s.length * .99)] || m, jitter: sd, cpu: c, frames: ft.length }; };
  const step = (i, mode) => { S.pan.x = -Math.sin(i / 40) * 600; S.pan.y = -Math.cos(i / 55) * 400; const t0 = performance.now(); draw(); return performance.now() - t0; };
  // A) naive: full redraw, no culling, driven by timer
  const runA = () => new Promise(res => { CULL = false; const ft = [], cpu = []; let last = performance.now(), i = 0; const t0 = last; (function tick() { const now = performance.now(); ft.push(now - last); last = now; cpu.push(step(i++)); if (now - t0 < DUR) setTimeout(tick, 0); else res(stats(ft.slice(1), cpu)); })(); });
  // B) synchronised: rAF pacing + viewport culling
  const runB = () => new Promise(res => { CULL = true; const ft = [], cpu = []; let last = 0, i = 0, t0 = 0; (function tick(now) { if (!t0) t0 = now; if (last) ft.push(now - last); last = now; cpu.push(step(i++)); if (now - t0 < DUR) requestAnimationFrame(tick); else res(stats(ft, cpu)); })(performance.now()); requestAnimationFrame(function w() { }); });
  msg("Benchmark A (naive)…"); const a = await runA();
  msg("Benchmark B (synced)…"); const b = await new Promise(r => requestAnimationFrame(() => r(runB())));
  Object.assign(S, { objs: keep.objs, sel: keep.sel, zoom: keep.zoom, pan: keep.pan }); CULL = keep.cull; refresh(true);
  // score: higher is better; penalise CPU cost and jitter
  const score = s => Math.round(1e4 / (s.cpu + s.jitter * .25 + 1e-3));
  const sa = score(a), sb = score(b), win = sb > sa;
  const row = (n, s, sc) => `<tr><td>${n}</td><td>${s.fps.toFixed(1)}</td><td>${s.avg.toFixed(2)}</td><td>${s.p99.toFixed(2)}</td><td>${s.jitter.toFixed(2)}</td><td>${s.cpu.toFixed(2)}</td><td><b>${sc}</b></td></tr>`;
  $("#tbl-audit").innerHTML = `<thead><tr><th>Mód (${N} objektov)</th><th>FPS</th><th>Frame ms</th><th>p99 ms</th><th>Jitter ms</th><th>CPU ms/frame</th><th>Skóre</th></tr></thead><tbody>${row("A · naive (setTimeout, bez cullingu)", a, sa)}${row("B · synchronizované (rAF + dirty + culling)", b, sb)}<tr><td colspan=7>${win ? "✅ B vyhráva" : "⚠ B nevyhralo"}: skóre ×${(sb / sa).toFixed(2)}, CPU/frame ×${(a.cpu / b.cpu).toFixed(2)} nižšie</td></tr></tbody>`;
  showView("audit"); msg(win ? `Vyhrali sme ×${(sb / sa).toFixed(2)}` : "Bez zlepšenia");
  return { a, b, win };
}

/* ---------- Boot ---------- */
setTool("move"); resize(); S.hist.push({ label: "Nový dokument", data: "[]" }); S.hi = 0; fit(); refresh(true);
