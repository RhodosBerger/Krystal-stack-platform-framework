/* ── KRYSTAL-STACK CLIENT ENGINE (SSE & REST) ────────────────────────── */

let eventSource = null;
let isPaused = false;
let currentMode = "CYBERPUNK";
let scanlinesActive = true;

// ── 1. Initialization ───────────────────────────────────────────────────

document.addEventListener("DOMContentLoaded", () => {
  initClock();
  initEventSource();
  fetchVulkanStatus();
  loadIntel("score");
  setInterval(() => loadIntel(intelSort), 60000);
});

function initClock() {
  const clockEl = document.getElementById("liveClock");
  setInterval(() => {
    const now = new Date();
    clockEl.textContent = now.toTimeString().split(" ")[0] + " // SEC-7";
  }, 1000);
}

// ── 1b. Project intelligence (priorities + expertise) ────────────────────

let intelSort = "score";
let intelFocus = "";

function intelRow(listEl, name, status, cls) {
  const row = document.createElement("div");
  row.className = "triad-item";
  const n = document.createElement("span");
  n.className = "triad-name";
  n.textContent = name;
  const s = document.createElement("span");
  s.className = "triad-status " + (cls || "cyan");
  s.textContent = status;
  row.append(n, s);
  listEl.appendChild(row);
  return row;
}

async function loadIntel(sort) {
  intelSort = sort || "score";
  const badge = document.getElementById("intelBadge");
  const list = document.getElementById("intelList");
  if (!list) return;
  try {
    const res = await fetch(`/api/priorities?limit=6&sort=${encodeURIComponent(intelSort)}`);
    const data = await res.json();
    if (data.status !== "OK") throw new Error(data.message || data.status);
    list.replaceChildren();
    data.priorities.forEach((p) => {
      const metric = intelSort === "value" ? `${p.value_per_effort.toFixed(1)}/pt` : p.score.toFixed(1);
      const row = intelRow(list, `${p.rank}. ${p.id}`, `${metric} · ${p.effort.size}`, p.effort.size === "S" ? "green" : "cyan");
      row.style.cursor = "pointer";
      row.title = p.title;
      row.addEventListener("click", () => loadExpertise(p.id));
    });
    badge.textContent = `${data.count} OPEN`;
    if (!intelFocus && data.priorities.length) loadExpertise(data.priorities[0].id);
    else loadExpertise(intelFocus);
  } catch (e) {
    badge.textContent = "API ERROR";
    list.replaceChildren();
    intelRow(list, "priorities", String(e.message || e), "cyan");
  }
}

async function loadExpertise(priorityId) {
  intelFocus = priorityId || "";
  const el = document.getElementById("intelExpertise");
  const focus = document.getElementById("intelFocus");
  if (!el) return;
  try {
    const res = await fetch("/api/expertise" + (intelFocus ? `?priority=${encodeURIComponent(intelFocus)}` : ""));
    const data = await res.json();
    if (data.status !== "OK") throw new Error(data.message || data.status);
    el.replaceChildren();
    focus.textContent = intelFocus ? `· ${intelFocus}` : "";
    const tracks = data.required_tracks || data.tracks.slice(0, 5);
    tracks.forEach((t) => {
      const gap = t.staffing_signal === "gap";
      const row = intelRow(el, t.title, t.staffing_signal.replace(/_/g, " ").toUpperCase(), gap ? "cyan" : "green");
      row.title = (t.engagement_options || []).join(" | ");
    });
  } catch (e) {
    el.replaceChildren();
    intelRow(el, "expertise", String(e.message || e), "cyan");
  }
}

// ── 2. Server-Sent Events (SSE) Stream ───────────────────────────────────

function initEventSource() {
  const badge = document.getElementById("connStatusBadge");
  const text = document.getElementById("connStatusText");

  if (eventSource) {
    eventSource.close();
  }

  eventSource = new EventSource("/api/stream");

  eventSource.onopen = () => {
    badge.style.borderColor = "rgba(0, 255, 136, 0.4)";
    badge.style.color = "var(--green)";
    text.textContent = "SSE STREAM ACTIVE";
    appendLog("Connected to Krystal Neural Core Stream (SSE).");
  };

  eventSource.onerror = (err) => {
    badge.style.borderColor = "rgba(255, 0, 85, 0.4)";
    badge.style.color = "var(--magenta)";
    text.textContent = "RECONNECTING...";
  };

  eventSource.addEventListener("frame", (event) => {
    if (isPaused) return;

    try {
      const payload = JSON.parse(event.data);
      renderFrame(payload);
    } catch (e) {
      console.error("Payload parse error:", e);
    }
  });

  eventSource.addEventListener("director", (event) => {
    try {
      const data = JSON.parse(event.data);
      appendDirectorLine(data.author || "DIRECTOR", data.message);
    } catch (e) {}
  });
}

// ── 3. Frame Rendering & Gauges ──────────────────────────────────────────

function renderFrame(payload) {
  // Update ASCII Canvas
  const canvas = document.getElementById("asciiCanvas");
  if (payload.ascii) {
    canvas.textContent = payload.ascii;
  }

  // Update FPS
  if (payload.fps) {
    document.getElementById("fpsDisplay").textContent = payload.fps.toFixed(1);
  }

  // Update HUD
  if (payload.mode) {
    document.getElementById("hudMode").textContent = payload.mode;
    highlightActiveButton(payload.mode);
  }

  // Update Entropy Metrics
  if (payload.entropy) {
    const s = payload.entropy.spatial || 0;
    const t = payload.entropy.temporal || 0;
    const tot = payload.entropy.total || 0;
    const coh = payload.entropy.coherence || (1.0 - tot);

    document.getElementById("txtSpatialEntropy").textContent = s.toFixed(2);
    document.getElementById("txtTemporalEntropy").textContent = t.toFixed(2);
    document.getElementById("txtTotalEntropy").textContent = tot.toFixed(2);

    document.getElementById("fillSpatial").style.width = `${Math.min(100, s * 100)}%`;
    document.getElementById("fillTemporal").style.width = `${Math.min(100, t * 100)}%`;
    document.getElementById("fillTotal").style.width = `${Math.min(100, tot * 100)}%`;

    document.getElementById("hudEntropy").textContent = tot.toFixed(2);
    document.getElementById("hudCoherence").textContent = `${Math.round(coh * 100)}%`;

    const bpEl = document.getElementById("hudBackpressure");
    if (payload.backpressure) {
      bpEl.textContent = "ACTIVE (THROTTLED)";
      bpEl.style.color = "var(--magenta)";
    } else {
      bpEl.textContent = "NORMAL (STABLE)";
      bpEl.style.color = "var(--green)";
    }
  }

  // Update Economic Governor
  if (payload.governor) {
    const g = payload.governor;
    document.getElementById("govBudget").textContent = Math.round(g.budget);
    document.getElementById("fillBudget").style.width = `${Math.min(100, (g.budget / 1000) * 100)}%`;
    document.getElementById("govState").textContent = g.state || "OPTIMAL";
    document.getElementById("govPenalty").textContent = (g.thermal_penalty || 0).toFixed(2);
  }

  // Update Cognitive Brainwave & Hamiltonian Energy
  if (payload.cognitive_phase) {
    const bwEl = document.getElementById("brainwaveStatus");
    if (bwEl) {
      let icon = "::";
      if (payload.cognitive_phase === "ALPHA") icon = "≈";
      else if (payload.cognitive_phase === "GAMMA") icon = "⚡";
      else if (payload.cognitive_phase === "OMEGA") icon = "Ω";
      bwEl.textContent = `${payload.cognitive_phase} (${icon})`;
      if (payload.cognitive_phase === "OMEGA") bwEl.style.color = "var(--magenta)";
      else if (payload.cognitive_phase === "GAMMA") bwEl.style.color = "var(--cyan)";
      else bwEl.style.color = "var(--gold, #ffd700)";
    }
  }
  if (payload.hamiltonian_energy !== undefined) {
    const hEl = document.getElementById("hamiltonianDisplay");
    if (hEl) hEl.textContent = Number(payload.hamiltonian_energy).toFixed(2);
  }

  // Update Vulkan probe / kernel backend telemetry
  if (payload.vulkan) {
    updateVulkanUI(payload.vulkan.enabled, payload.vulkan);
  }
}

// ── 4. Controls & Interactions ──────────────────────────────────────────

function setMode(mode) {
  currentMode = mode;
  highlightActiveButton(mode);

  fetch("/api/control", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ mode: mode })
  })
  .then(res => res.json())
  .then(data => {
    appendLog(`Switched style to ${mode}`);
  })
  .catch(err => console.error("Error setting mode:", err));
}

function highlightActiveButton(mode) {
  const map = {
    "CYBERPUNK": "btnCyberpunk",
    "BLUEPRINT_EDGE": "btnBlueprint",
    "HIGH_FIDELITY": "btnHighFidelity",
    "RAYMARCH_ANOMALY": "btnRaymarch",
    "MATRIX_RAIN": "btnMatrix",
    "RETRO_CRT": "btnRetro",
    "HOLOGRAPHIC_3D": "btnHolo",
    "RECURSIVE_MIRROR": "btnMirror",
    "SACRED_GEOMETRY": "btnMirror",
    "MIMICRY_OBJECT": "btnMimic",
    "GAME_SCENE": "btnScene",
    "OPENWORLD": "btnOpenWorld",
    "KRYSTAL_LANG_SHAPE": "btnKrystalLang",
    "CYCLIC_HAMILTONIAN_ORGANISM": "btnCyclicOrganism"
  };

  document.querySelectorAll(".mode-btn").forEach(btn => btn.classList.remove("active"));
  const targetId = map[mode];
  if (targetId) {
    const el = document.getElementById(targetId);
    if (el) el.classList.add("active");
  }
}

function updateConfig(param, value) {
  const payload = {};
  payload[param] = parseFloat(value);

  if (param === "cols") {
    document.getElementById("valCols").textContent = value;
    document.getElementById("resTag").textContent = `${value}x${document.getElementById("sliderRows").value} CHARS`;
  } else if (param === "rows") {
    document.getElementById("valRows").textContent = value;
    document.getElementById("resTag").textContent = `${document.getElementById("sliderCols").value}x${value} CHARS`;
  } else if (param === "threshold") {
    document.getElementById("valEntropyThresh").textContent = value.toFixed(2);
  }

  fetch("/api/control", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload)
  });
}

function sendDirective() {
  const input = document.getElementById("directorInput");
  const text = input.value.trim();
  if (!text) return;

  appendDirectorLine("USER", text);
  input.value = "";

  fetch("/api/director", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ prompt: text })
  })
  .then(res => res.json())
  .then(data => {
    if (data.reply) {
      appendDirectorLine("SLM_DIRECTOR", data.reply);
    }
  });
}

function triggerGovernor(action) {
  fetch("/api/governor", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ action: action })
  })
  .then(res => res.json())
  .then(data => {
    appendLog(`Governor action triggered: ${action}`);
  });
}

function togglePause() {
  isPaused = !isPaused;
  const btn = document.getElementById("btnPause");
  btn.textContent = isPaused ? "RESUME" : "PAUSE";
  btn.style.color = isPaused ? "var(--yellow)" : "var(--text-main)";
}

function toggleScanlines() {
  scanlinesActive = !scanlinesActive;
  const el = document.querySelector(".cyber-scanlines");
  el.style.display = scanlinesActive ? "block" : "none";
}

function captureSnapshot() {
  const canvas = document.getElementById("asciiCanvas");
  const text = canvas.textContent;
  const blob = new Blob([text], { type: "text/plain" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = `krystal_snapshot_${Date.now()}.txt`;
  a.click();
  appendLog("Snapshot downloaded to disk.");
}

function appendDirectorLine(author, msg) {
  const term = document.getElementById("directorTerminal");
  const line = document.createElement("div");
  line.className = "terminal-line";
  line.innerHTML = `<span class="prompt">[${author}]</span> ${msg}`;
  term.appendChild(line);
  term.scrollTop = term.scrollHeight;
}

function appendLog(msg) {
  const logs = document.getElementById("sysLogs");
  const now = new Date().toTimeString().split(" ")[0];
  const div = document.createElement("div");
  div.textContent = `[${now}] ${msg}`;
  logs.appendChild(div);
  logs.scrollTop = logs.scrollHeight;
}

// ── 5. Antigravity Prompt & Template Studio ──────────────────────────────

function submitAntigravityPrompt() {
  const input = document.getElementById("agPromptInput");
  const prompt = input.value.trim();
  if (!prompt) return;

  appendLog(`Synthesizing Antigravity Prompt: "${prompt}"`);
  appendDirectorLine("ANTIGRAVITY", `Parsing geometric intent: "${prompt}"...`);

  fetch("/api/antigravity/prompt", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ prompt: prompt })
  })
  .then(res => res.json())
  .then(data => {
    if (data.status === "SUCCESS") {
      const tpl = data.template;
      appendLog(`Instantiated ${tpl.name} (${tpl.composition_rules.symmetry_group})`);
      appendDirectorLine("SLM_DIRECTOR", `Composed ${tpl.name}. Dihedral mirror order: ${tpl.composition_rules.mirror_folds}. Vulkan pipeline updated.`);
      
      // Update UI sliders and selectors
      if (tpl.geometric_instance) {
        document.getElementById("selGeom").value = tpl.geometric_instance.id;
      }
      if (tpl.artistic_instance) {
        document.getElementById("selArt").value = tpl.artistic_instance.id;
      }
      if (tpl.composition_rules) {
        const folds = tpl.composition_rules.mirror_folds;
        document.getElementById("sliderFolds").value = folds;
        document.getElementById("valFolds").textContent = folds;
        document.getElementById("valFoldSymmetry").textContent = `D${folds} / ${folds * 2}-fold`;
        
        const depth = tpl.composition_rules.recursion_depth;
        document.getElementById("sliderRecursion").value = depth;
        document.getElementById("valRecursion").textContent = depth;
      }

      // Display GLSL push constants
      if (data.glsl_uniforms) {
        document.getElementById("glslCodeBlock").textContent = data.glsl_uniforms;
      }

      // Switch mode button to active
      setMode("RECURSIVE_MIRROR");
    } else {
      appendLog(`Antigravity Error: ${data.message || 'Unknown error'}`);
    }
  })
  .catch(err => console.error("Error submitting Antigravity prompt:", err));
}

function applyQuickPrompt(promptText) {
  document.getElementById("agPromptInput").value = promptText;
  submitAntigravityPrompt();
}

function composeFromSelectors() {
  const geomId = document.getElementById("selGeom").value;
  const artId = document.getElementById("selArt").value;
  const folds = parseInt(document.getElementById("sliderFolds").value);
  const depth = parseInt(document.getElementById("sliderRecursion").value);

  fetch("/api/templates/compose", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      geometric_id: geomId,
      artistic_id: artId,
      mirror_folds: folds,
      recursion_depth: depth
    })
  })
  .then(res => res.json())
  .then(data => {
    if (data.status === "SUCCESS") {
      appendLog(`Custom manifold composed: ${data.template.name}`);
      if (data.glsl_uniforms) {
        document.getElementById("glslCodeBlock").textContent = data.glsl_uniforms;
      }
      setMode("RECURSIVE_MIRROR");
    }
  })
  .catch(err => console.error("Error composing manifold:", err));
}

function updateFoldSlider(val) {
  document.getElementById("valFolds").textContent = val;
  document.getElementById("valFoldSymmetry").textContent = `D${val} / ${val * 2}-fold`;
  fetch("/api/control", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ mirror_folds: parseInt(val) })
  });
}

function updateRecursionSlider(val) {
  document.getElementById("valRecursion").textContent = val;
  composeFromSelectors();
}

// ── 6. Blender Modifier & Mimicry Compositor Studio ──────────────────────

const MIMIC_MODIFIERS_MAP = {
  "CYBER_TURRET_MK4": ["Bevel (r:0.06)", "Array (Barrels x2)", "Mirror (Ammo X)", "Smooth Union (k:0.1)"],
  "MECH_WALKER_TITAN": ["Deform (Taper Y)", "Boolean (Visor Cut)", "Mirror (Legs X)", "Mirror (Feet X)"],
  "ANCIENT_OBELISK_MONOLITH": ["Array (Plinth Steps x3)", "Displace (Runic Relief)", "Bevel (Crystal)"],
  "BIOMECHANICAL_XENODRONE": ["Displace (BioVeins)", "Array (Spine Ribs x4)", "Mirror (Mandibles X)"],
  "CYBERPUNK_DATA_SPIRE": ["Array (Decks x5)", "Mirror (Coolant Fins X)", "Smooth Union (Core)"],
  "RETRO_SOLAR_EXPLORER": ["Array (Ion Radial x4)", "Mirror (Solar Wings X)", "Solidify (Dish)"]
};

function onSelectMimicObject(recipeId) {
  updateModifierPills(recipeId);
  renderCurrentMimicObject();
}

function onSelectGameScene(sceneId) {
  renderCurrentGameScene();
}

function updateModifierPills(recipeId) {
  const container = document.getElementById("modifierPills");
  if (!container) return;
  const mods = MIMIC_MODIFIERS_MAP[recipeId] || ["Modifier Stack (DAG Active)"];
  container.innerHTML = mods.map(m => `<span class="mod-pill active">${m}</span>`).join(" ");
}

function renderCurrentMimicObject() {
  const recipeId = document.getElementById("selMimicObj").value;
  appendLog(`Selecting Mimic Object: ${recipeId}`);

  fetch("/api/mimicry/select", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ type: "object", id: recipeId })
  })
  .then(res => res.json())
  .then(data => {
    if (data.status === "SUCCESS") {
      appendLog(`Mimic object active: ${data.object.name}`);
      setMode("MIMICRY_OBJECT");
    }
  })
  .catch(err => console.error("Error selecting mimic object:", err));
}

function renderCurrentGameScene() {
  const sceneId = document.getElementById("selGameScene").value;
  appendLog(`Assembling Game Scene: ${sceneId}`);

  fetch("/api/mimicry/select", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ type: "scene", id: sceneId })
  })
  .then(res => res.json())
  .then(data => {
    if (data.status === "SUCCESS") {
      appendLog(`Game scene active: ${data.scene.name} (${data.scene.actors_count} actors)`);
      setMode("GAME_SCENE");
    }
  })
  .catch(err => console.error("Error selecting game scene:", err));
}

function exportActiveSceneToGodot() {
  fetch("/api/mimicry/export-godot", {
    method: "POST",
    headers: { "Content-Type": "application/json" }
  })
  .then(res => res.json())
  .then(data => {
    if (data.status === "SUCCESS") {
      appendLog(`Exported Godot scene: ${data.exported_file}`);
      appendDirectorLine("GODOT_BRIDGE", `Successfully generated ${data.exported_file}. Ready for Godot 4.x engine.`);
    } else {
      appendLog(`Godot Export Error: ${data.message || 'Unknown'}`);
    }
  })
  .catch(err => console.error("Error exporting Godot scene:", err));
}

// ── 7. Open-World Semantic Compiler & Janet Code Synthesizer ─────────────

let activeOpenWorldData = null;

function applyOwPrompt(promptText) {
  const input = document.getElementById("owPromptInput");
  if (input) {
    input.value = promptText;
    submitOpenWorldPrompt();
  }
}

function submitOpenWorldPrompt() {
  const input = document.getElementById("owPromptInput");
  const prompt = input ? input.value.trim() : "";
  if (!prompt) return;

  const btn = document.getElementById("btnOwSubmit");
  if (btn) btn.textContent = "COMPILING...";

  appendLog(`Compiling Open-World Prompt: "${prompt}"`);

  fetch("/api/openworld/compile", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ prompt: prompt })
  })
  .then(res => res.json())
  .then(data => {
    if (btn) btn.textContent = "COMPILE TO MATH & JANET";
    if (data.status === "SUCCESS") {
      activeOpenWorldData = data;
      appendLog(`Compiled to Biome: ${data.spec.dominant_biome.name}`);
      appendDirectorLine("OPENWORLD", `Instantiated infinite terrain manifold for '${data.spec.name}'. Topography: ${data.spec.topography_type}, Atmosphere: ${data.spec.atmosphere_type}.`);
      updateOwCodeViewers(data);
      setMode("OPENWORLD");
    } else {
      appendLog(`Open-World Error: ${data.message || 'Unknown'}`);
    }
  })
  .catch(err => {
    if (btn) btn.textContent = "COMPILE TO MATH & JANET";
    console.error("Error compiling open-world prompt:", err);
  });
}

function updateOwCodeViewers(data) {
  const mathViewer = document.getElementById("owMathViewer");
  const janetViewer = document.getElementById("owJanetViewer");
  const godotViewer = document.getElementById("owGodotViewer");
  const pythonViewer = document.getElementById("owPythonViewer");

  if (mathViewer && data.spec) {
    mathViewer.textContent = JSON.stringify(data.spec, null, 2);
  }
  if (janetViewer && data.janet_dsl) {
    janetViewer.textContent = data.janet_dsl;
  }
  if (godotViewer && data.godot_shader) {
    godotViewer.textContent = data.godot_shader;
  }
  if (pythonViewer && data.python_code) {
    pythonViewer.textContent = data.python_code;
  }
}

function switchOwTab(tabKey) {
  const tabs = ["math", "janet", "godot", "python"];
  tabs.forEach(t => {
    const btn = document.getElementById(`tabBtn${t.charAt(0).toUpperCase() + t.slice(1)}`);
    const content = document.getElementById(`tabContent${t.charAt(0).toUpperCase() + t.slice(1)}`);
    if (btn) {
      if (t === tabKey) btn.classList.add("active");
      else btn.classList.remove("active");
    }
    if (content) {
      content.style.display = (t === tabKey) ? "block" : "none";
    }
  });
}

// ── 8. Krystal-Lang Topological Compiler & Virtual Machine ───────────────

function compileKrystalLang() {
  const codeArea = document.getElementById("klCodeInput");
  const code = codeArea ? codeArea.value.trim() : "";
  if (!code) return;

  const btn = document.getElementById("btnKlCompile");
  if (btn) btn.textContent = "COMPILING...";

  appendLog("Compiling Krystal-Lang source into topological bytecode & 3D shape...");

  fetch("/api/krystal-lang/compile", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ code: code })
  })
  .then(res => res.json())
  .then(data => {
    if (btn) btn.textContent = "COMPILE & SHAPE VM";
    if (data.status === "SUCCESS") {
      appendLog(`Module '${data.module_name}' compiled: ${data.bytecode.length} opcodes across ${data.queue_partitions.length} stages.`);
      appendDirectorLine("KRYSTAL_VM", `Topological manifold generated. VM active with ${data.queues_count} partitioned queue streams.`);
      updateKlMetrics(data.vm_metrics);
      setMode("KRYSTAL_LANG_SHAPE");
    } else {
      appendLog(`Krystal-Lang Error: ${data.message || 'Unknown'}`);
    }
  })
  .catch(err => {
    if (btn) btn.textContent = "COMPILE & SHAPE VM";
    console.error("Error compiling Krystal-Lang code:", err);
  });
}

function stepKrystalVM() {
  fetch("/api/krystal-lang/step")
  .then(res => res.json())
  .then(data => {
    if (data.status === "OK") {
      updateKlMetrics(data.metrics);
      appendLog(`Injected packet into Krystal-VM -> Throughput: ${data.metrics.average_throughput_pps} pps`);
    }
  })
  .catch(err => console.error("Error stepping VM:", err));
}

function updateKlMetrics(metrics) {
  const label = document.getElementById("klVmMetrics");
  if (label && metrics) {
    label.textContent = `Cycles: ${metrics.cycles} | Packets: ${metrics.packets_processed} | Rate: ${metrics.average_throughput_pps} pps`;
  }
}

// ── 9. Vulkan Compute Hardware Accelerator ──────────────────────────────

function fetchVulkanStatus() {
  fetch("/api/vulkan")
    .then(res => res.json())
    .then(data => {
      if (data.status === "OK") {
        updateVulkanUI(data.enabled, data.telemetry);
      }
    })
    .catch(() => {});
}

function toggleVulkan() {
  const btn = document.getElementById("btnToggleVk");
  if (btn) btn.textContent = "SWITCHING...";

  fetch("/api/vulkan/toggle", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({})
  })
  .then(res => res.json())
  .then(data => {
    if (btn) btn.textContent = "TOGGLE VULKAN-PATH KERNEL / CPU";
    if (data.status === "OK") {
      updateVulkanUI(data.enabled, data.telemetry);
      appendLog(`Vulkan-path kernel: ${data.enabled ? "ENABLED (CPU-emulated kernel, device probed)" : "DISABLED (CPU reference path)"}`);
    }
  })
  .catch(err => {
    if (btn) btn.textContent = "TOGGLE VULKAN-PATH KERNEL / CPU";
    console.error("Error toggling Vulkan:", err);
  });
}

// Status is derived from what is actually true: real_gpu_dispatch is only set
// by the driver once a genuine vkCmdDispatch path exists. Until then the
// 'Vulkan' path is a device probe plus a CPU-emulated kernel.
function updateVulkanUI(enabled, t) {
  t = t || {};
  const badge = document.getElementById("vkActiveBadge");
  const pill = document.getElementById("vkPillStatus");
  const deviceEl = document.getElementById("vkDeviceName");
  const latencyEl = document.getElementById("vkLatency");
  const readbackEl = document.getElementById("vkReadback");

  const real = Boolean(t.real_gpu_dispatch ?? t.accelerated);
  const detected = Boolean(t.device_detected);
  const device = t.device ?? t.device_name;
  const latencyUs = t.dispatch_us ?? t.last_dispatch_us ?? t.dispatch_latency_us;

  let label, color, bg, border;
  if (real) {
    label = "ACTIVE (GPU)"; color = "#00ff88"; bg = "rgba(0, 255, 136, 0.15)"; border = "rgba(0, 255, 136, 0.4)";
  } else if (enabled) {
    label = detected ? "DETECTED \u00b7 CPU KERNEL" : "CPU KERNEL";
    color = "#ffaa00"; bg = "rgba(255, 170, 0, 0.15)"; border = "rgba(255, 170, 0, 0.4)";
  } else {
    label = "CPU REFERENCE PATH"; color = "#9aa4b2"; bg = "rgba(154, 164, 178, 0.12)"; border = "rgba(154, 164, 178, 0.35)";
  }
  if (badge) {
    badge.textContent = label;
    badge.style.background = bg;
    badge.style.color = color;
    badge.style.borderColor = border;
  }
  if (pill) {
    pill.textContent = label;
    pill.style.color = color;
  }

  if (deviceEl && device) deviceEl.textContent = device;
  if (latencyEl && latencyUs !== undefined) {
    latencyEl.textContent = `${(Number(latencyUs) / 1000).toFixed(1)} ms`;
  }
  if (readbackEl && t.readback_mb_s !== undefined) {
    readbackEl.textContent = `${Number(t.readback_mb_s).toFixed(1)} MB/s`;
  }
}
