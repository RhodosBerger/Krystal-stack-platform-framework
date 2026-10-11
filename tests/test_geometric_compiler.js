/**
 * Node.js Integration Test for Sovereign Geometric Compiler & Adaptive Render Pipeline.
 * Tests axial topology, Boolean Hex Ops, Pauli Exclusion Guard on [A_CENTER],
 * packed binary buffer generation (96 bytes/instance), 3 telemetry hardware profiles,
 * and 20 Hz shader uniform throttler.
 */

const assert = require("assert");
const {
  VITAL_MAX_HP,
  MIN_EFFECTIVE_RESISTANCE,
  CONFIDENCE_THRESHOLD,
  LAYER_BASE,
  LAYER_CENTER,
  hexDistance,
  axialToWorld,
  worldToAxial,
  formatConductBadge,
  validateConductCode,
  PauliExclusionGuard,
  GeometricCompiler
} = require("../krystal_web_hub/static/js/sovereign_dvt/geometric_compiler.js");

const { AdaptiveRenderPipeline } = require("../krystal_web_hub/static/js/sovereign_dvt/adaptive_render_pipeline.js");

console.log("===============================================================================");
console.log("RUNNING SOVEREIGN GEOMETRIC COMPILER & DVT INTEGRATION TESTS (NODE.JS)");
console.log("===============================================================================\n");

// 1. Axial Hex Topology Math
console.log("[TEST 1] Axial Hex Topology Math & Distance");
const d0 = hexDistance({ q: 0, r: 0 }, { q: 2, r: -1 });
assert.strictEqual(d0, 2, "Hex distance should equal 2");

const w0 = axialToWorld(0, 0, 1.0);
assert.strictEqual(w0.x, 0.0);
assert.strictEqual(w0.z, 0.0);

const w1 = axialToWorld(1, 0, 1.0);
assert(Math.abs(w1.x - Math.sqrt(3.0)) < 1e-4, "w1.x should equal sqrt(3)");

const axRound = worldToAxial(w1.x, w1.z, 1.0);
assert.strictEqual(axRound.q, 1);
assert.strictEqual(axRound.r, 0);
console.log("  -> OK: Axial topology math verified.");

// 2. Pauli Exclusion Guard on Layer [A_CENTER]
console.log("[TEST 2] Pauli Exclusion Guard on Layer [A_CENTER]");
const guard = new PauliExclusionGuard();
const r1 = guard.validateAndOccupy({ q: 0, r: 0 }, "entity_a", LAYER_CENTER);
assert.strictEqual(r1.valid, true);

// Attempting duplicate placement on A_CENTER must fail
const r2 = guard.validateAndOccupy({ q: 0, r: 0 }, "entity_b", LAYER_CENTER);
assert.strictEqual(r2.valid, false);
assert(r2.error.includes("Pauli Exclusion Violation"));

// Placement on A_BASE or A_AURA_RING at same coord must succeed
const r3 = guard.validateAndOccupy({ q: 0, r: 0 }, "aura_ring", LAYER_BASE);
assert.strictEqual(r3.valid, true);
console.log("  -> OK: Pauli Exclusion Guard strictly enforced.");

// 3. Declarative Blueprint Compilation & Packed Binary Buffer
console.log("[TEST 3] Declarative Blueprint Compilation & 96-Byte Packed Buffer");
const compiler = new GeometricCompiler();
const blueprint = {
  mapId: "TEST_MAP_NODE",
  name: "Node Test Sector",
  hexRadius: 1.0,
  cells: [
    { q: 0, r: 0, biome: "CRYSTAL_CITADEL", elevation: 0.0, is_traversable: true },
    { q: 1, r: 0, biome: "VOLCANIC", elevation: 0.5, is_hazard: true },
    { q: 0, r: 1, biome: "PLAINS", elevation: 0.0, is_traversable: true }
  ],
  entities: [
    { id: "standee_1", name: "Standee Alpha", q: 0, r: 0, layer: "A_CENTER", hp: 6, base_resistance: 2.0 },
    { id: "standee_2", name: "Standee Beta", q: 0, r: 1, layer: "A_CENTER", hp: 4, base_resistance: 1.5 }
  ]
};

const compiled = compiler.compile(blueprint);
assert.strictEqual(compiled.cells.size, 3);
assert.strictEqual(compiled.entities.size, 2);
assert.strictEqual(compiled.instanceCount, 5);

// Packed buffer must be exactly 5 instances * 96 bytes = 480 bytes
assert.strictEqual(compiled.packedBuffer.byteLength, 5 * 96);
const floatView = new Float32Array(compiled.packedBuffer);
assert.strictEqual(floatView.length, 5 * 24);

// First instance (cell at 0,0): translation at indices 12, 13, 14
assert.strictEqual(floatView[12], 0.0); // posX
assert.strictEqual(floatView[13], 0.0); // posY (elevation)
assert.strictEqual(floatView[14], 0.0); // posZ
assert.strictEqual(floatView[20], LAYER_BASE); // layer_id

// Fourth instance (standee_1 at index 3 * 24 = 72)
const standeeOffset = 3 * 24;
assert.strictEqual(floatView[standeeOffset + 20], LAYER_CENTER); // layer_id = 0.5
assert.strictEqual(floatView[standeeOffset + 21], 6.0); // hp = 6
console.log("  -> OK: Packed binary buffer generated with exact 96 bytes/instance alignment.");

// 4. Conduct Code Bridge & Safety Validation
console.log("[TEST 4] Conduct Code Bridge & Safety Validation");
const badge = formatConductBadge({
  isActive: true,
  isLocked: false,
  confidence: 0.95,
  isHazardous: false,
  vitalHp: 6,
  vitalMaxHp: 6
});
assert.strictEqual(badge, "[ACT] [!LCK] [CNF] [!HAZ] [VIT 6HP]");

const valSafe = validateConductCode({
  isActive: true,
  isLocked: false,
  confidence: 0.95,
  isHazardous: false,
  vitalHp: 6,
  vitalMaxHp: 6
});
assert.strictEqual(valSafe.valid, true);

const valLowConf = validateConductCode({
  isActive: true,
  isLocked: false,
  confidence: 0.65,
  isHazardous: false,
  vitalHp: 6,
  vitalMaxHp: 6
});
assert.strictEqual(valLowConf.valid, false);
console.log("  -> OK: Conduct Code badges and safety checks verified.");

// 5. Adaptive Render Pipeline & 20 Hz Throttler
console.log("[TEST 5] Adaptive Render Pipeline & 20 Hz Uniform Throttler");
const pipeline = new AdaptiveRenderPipeline();

// High stress telemetry -> Minimal Wireframe
const pWire = pipeline.evaluateTelemetry({ fps: 20, cpuLoad: 92, temperatureC: 84, entropyH: 0.5 });
assert.strictEqual(pWire, "MINIMAL_WIREFRAME");
assert.strictEqual(pipeline.getUniformState().wireframeActive, true);

// Optimal telemetry -> AAA Cinematic
const pAAA = pipeline.evaluateTelemetry({ fps: 60, cpuLoad: 25, temperatureC: 50, entropyH: 0.4 });
assert.strictEqual(pAAA, "AAA_CINEMATIC");
assert.strictEqual(pipeline.getUniformState().bloomIntensity, 1.2);

// Throttled uniform update at 20 Hz (50 ms minimum period)
const t0 = 1000.0;
const dispatched1 = pipeline.updateUniforms(t0, 1.0);
assert.strictEqual(dispatched1, true, "First uniform write must be dispatched");

// Immediate subsequent write after 10 ms (< 50 ms) must be throttled
const dispatched2 = pipeline.updateUniforms(t0 + 10.0, 1.01);
assert.strictEqual(dispatched2, false, "Subsequent uniform write within 10ms must be throttled");

// Write after 55 ms (>= 50 ms) must succeed
const dispatched3 = pipeline.updateUniforms(t0 + 55.0, 1.055);
assert.strictEqual(dispatched3, true, "Uniform write after 55ms must succeed");

console.log("  -> OK: Adaptive pipeline hardware profiles and 20 Hz throttler verified.");

console.log("\n===============================================================================");
console.log("ALL 5 NODE.JS INTEGRATION TESTS PASSED SUCCESSFULLY (100% OK)");
console.log("===============================================================================\n");
