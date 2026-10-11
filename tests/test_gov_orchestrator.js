/**
 * Node.js Integration tests for Krystal-Stack Government Orchestrator.
 */

const assert = require('assert');
const { get5DHealthVector, appendAuditEntry, state } = require('../krystal_kernel/gov_orchestrator.js');

console.log('===============================================================================');
console.log('RUNNING GOV ORCHESTRATOR & 5D HEALTH VECTOR INTEGRATION TESTS (NODE.JS)');
console.log('===============================================================================');

// [TEST 1] 5D Health Vector continuous evaluation
const hv = get5DHealthVector();
assert.strictEqual(typeof hv.liveness, 'number');
assert.strictEqual(typeof hv.readiness, 'number');
assert.strictEqual(typeof hv.resource_integrity, 'number');
assert.strictEqual(typeof hv.degradation_score, 'number');
assert.strictEqual(typeof hv.sla_compliance, 'number');
assert.strictEqual(typeof hv.composite_health, 'number');
assert.ok(hv.composite_health > 0.9, 'Initial pristine health must exceed 0.90');
console.log('[TEST 1] 5D Health Vector evaluation -> OK');

// [TEST 2] Cryptographic Ledger Chaining
const prevCount = state.ledger.length;
const entry = appendAuditEntry('TEST_EVENT_GOV', 'CORR-TEST-1234', { test: true });
assert.strictEqual(state.ledger.length, prevCount + 1);
assert.strictEqual(entry.event_type, 'TEST_EVENT_GOV');
assert.strictEqual(entry.prev_hash, state.ledger[prevCount - 1].hash);
assert.strictEqual(entry.hash.length, 64); // SHA-256 hex string length
console.log('[TEST 2] Cryptographic Ledger Chaining -> OK');

// [TEST 3] Throttling Degradation Impact
state.cgroups_nr_throttled = 45;
const throttledHv = get5DHealthVector();
assert.strictEqual(throttledHv.degradation_score, 0.65);
assert.strictEqual(throttledHv.resource_integrity, 0.35);
assert.ok(throttledHv.composite_health < hv.composite_health);
console.log('[TEST 3] Silent Throttling Degradation Impact -> OK');

console.log('===============================================================================');
console.log('ALL 3 NODE.JS GOV ORCHESTRATOR TESTS PASSED SUCCESSFULLY (100% OK)');
console.log('===============================================================================');
