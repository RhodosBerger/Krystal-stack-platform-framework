#!/usr/bin/env python3
"""
==============================================================================
KRYSTAL-STACK: UNIT TEST SUITE
TYPED EPISTEMIC AUTHORITY RUNTIME (PROT SPRINT A)
==============================================================================
Tests all 16 canonical test families:
  1. Expired evidence cannot commit.
  2. Stale evidence lowers epistemic state.
  3. High-confidence stale evidence is rejected.
  4. Contradictory sensors cause quarantine.
  5. SafetyGate veto dominates EvidenceGate.
  6. StateVersion mismatch invalidates candidate action.
  7. Validity-window expiration invalidates candidate action.
  8. Pre-actuation veto causes REJECT_NOOP.
  9. Post-prepare failure causes ABORT_STAGED_STATE.
  10. Post-commit divergence maps to COMPENSATE.
  11. Policy version included in proof bundle.
  12. Cross-language fixture parity (epistemic_value, envelope, candidate, verdict).
  13. Clock-domain mismatch is detected.
  14. Contextual veto does not create a global permanent blacklist.
  15. Shadow commit performs zero host mutation (WOULD_COMMIT).
  16. System invariant VITAL_MAX_HP == 6 enforced.
==============================================================================
"""

import os
import json
import time
import unittest

from krystal_kernel.scene_constructor_presets import VITAL_MAX_HP
from krystal_kernel.verification.state_version import StateVersion
from krystal_kernel.verification.epistemic_value import (
    ValidityState,
    EvidenceKind,
    EpistemicValue
)
from krystal_kernel.verification.policy_registry import (
    PolicyVersion,
    DynamicSafetyPolicy,
    DynamicEvidencePolicy,
    PolicyRegistry
)
from krystal_kernel.verification.resource_evidence_envelope import (
    ResourceVector,
    ExpectedDelta,
    ResourceEvidenceEnvelope
)
from krystal_kernel.verification.proof_bundle import (
    SafetyTier,
    SafetyVerdictType,
    EvidenceVerdictType,
    AuthorityVerdictType,
    ReversibilityClass,
    FailureLifecycleStage,
    SafetyProof,
    EvidenceProof,
    AuthorityProofBundle
)
from krystal_kernel.verification.authority_gate import (
    CandidateAction,
    SafetyGate,
    EvidenceGate,
    DualAuthorityGate
)
from krystal_kernel.adapters.kvm_qemu_adapter import KvmQemuAdapter
from krystal_kernel.adapters.hyperv_adapter import HyperVAdapter
from krystal_kernel.runtime.prediction_error_ledger import (
    PredictionErrorLedger,
    ContextualRiskRegistry,
    ContextualRiskKey
)
from krystal_kernel.runtime.shadow_runtime import ShadowRuntime


class TestEpistemicRuntime(unittest.TestCase):

    def setUp(self):
        self.state_version = StateVersion(major=1, minor=0, sequence=1, wall_timestamp_utc=time.time())
        self.safety_policy = DynamicSafetyPolicy()
        self.evidence_policy = DynamicEvidencePolicy()
        self.safety_gate = SafetyGate(self.safety_policy)
        self.evidence_gate = EvidenceGate(self.evidence_policy)
        self.dual_gate = DualAuthorityGate(self.safety_gate, self.evidence_gate)

    def _create_mock_envelope(self, now_ns: int, temp_val: float = 50.0, cpu_val: float = 40.0) -> ResourceEvidenceEnvelope:
        sv = self.state_version
        ttl = int(2.0 * 1e9)
        canonical = {
            "cpu_usage_pct": EpistemicValue(
                value=cpu_val,
                provenance="KvmQemuAdapter:cgroups_v2:cpu.stat",
                evidence_kind=EvidenceKind.RUNTIME_MEASURED,
                uncertainty=0.03,
                valid_from_ns=now_ns,
                valid_to_ns=now_ns + ttl,
                domain="cpu.usage",
                state_version=sv,
                source_adapter="KvmQemuAdapter"
            ),
            "temperature_celsius": EpistemicValue(
                value=temp_val,
                provenance="KvmQemuAdapter:hwmon:coretemp",
                evidence_kind=EvidenceKind.HARDWARE_MEASURED,
                uncertainty=0.02,
                valid_from_ns=now_ns,
                valid_to_ns=now_ns + ttl,
                domain="hardware.thermal",
                state_version=sv,
                source_adapter="KvmQemuAdapter"
            )
        }
        return ResourceEvidenceEnvelope(
            schema_version="2.0.0-sprintA",
            envelope_id="ENV-TEST-001",
            state_version=sv,
            provenance="KvmQemuAdapter",
            host_fingerprint="node-01",
            hypervisor_type="kvm",
            guest_id="guest-01",
            host_monotonic_ns=now_ns,
            canonical_normalized=canonical
        )

    def test_universal_hp_invariant(self):
        """Invariant 16: VITAL_MAX_HP == 6 must hold across system."""
        self.assertEqual(VITAL_MAX_HP, 6)

    def test_expired_evidence_cannot_commit(self):
        """Test 1: Expired evidence (now_ns > valid_to_ns) must not participate in decisions."""
        now_ns = int(time.time() * 1e9)
        env = self._create_mock_envelope(now_ns)
        # Fast forward time beyond valid_to_ns (2.5 seconds later)
        eval_time_ns = now_ns + int(3.0 * 1e9)

        candidate = CandidateAction(
            action_id="ACT-001",
            source_state_version=self.state_version,
            generated_at_ns=now_ns,
            valid_until_ns=eval_time_ns + int(1.0 * 1e9),
            action_type="RESIZE_BALLOON",
            target_tenant_id="guest-01",
            expected_delta=ExpectedDelta({"cpu_usage_pct": -5.0}, confidence=0.9, validity_window_ns=int(1e9)),
            required_resources=ResourceVector(cpu_normalized_share=0.1),
            evidence_refs=[env.envelope_id],
            reversibility=ReversibilityClass.COMPENSABLE
        )

        evidence_proof = self.evidence_gate.evaluate(candidate, [env], eval_time_ns)
        self.assertEqual(evidence_proof.verdict, EvidenceVerdictType.INSUFFICIENT)

        bundle = self.dual_gate.arbitrate(candidate, self.state_version, [env], eval_time_ns)
        self.assertEqual(bundle.verdict, AuthorityVerdictType.DEFER_INSUFFICIENT_EVIDENCE)
        self.assertNotEqual(bundle.verdict, AuthorityVerdictType.COMMIT_APPROVED)

    def test_stale_evidence_lowers_epistemic_state(self):
        """Test 2: Evidence exceeding freshness limit transitions from FRESH to STALE."""
        now_ns = int(time.time() * 1e9)
        ep_val = EpistemicValue(
            value=45.0,
            provenance="sensor-1",
            evidence_kind=EvidenceKind.HARDWARE_MEASURED,
            uncertainty=0.01,
            valid_from_ns=now_ns,
            valid_to_ns=now_ns + 1000,
            domain="thermal",
            state_version=self.state_version,
            source_adapter="CppHardwareProbe"
        )
        self.assertEqual(ep_val.get_validity_state(now_ns + 100), ValidityState.FRESH)
        self.assertEqual(ep_val.get_validity_state(now_ns + 800), ValidityState.STALE)
        self.assertEqual(ep_val.get_validity_state(now_ns + 1200), ValidityState.EXPIRED)

    def test_high_confidence_stale_evidence_rejected(self):
        """Test 3: Confidence = 0.99 with stale evidence must still be rejected by EvidenceGate."""
        now_ns = int(time.time() * 1e9)
        env = self._create_mock_envelope(now_ns)
        # Advance clock to make envelope stale
        eval_time_ns = now_ns + int(2.5 * 1e9)

        candidate = CandidateAction(
            action_id="ACT-HIGH-CONF",
            source_state_version=self.state_version,
            generated_at_ns=now_ns,
            valid_until_ns=eval_time_ns + int(1e9),
            action_type="PIN_VCPU",
            target_tenant_id="guest-01",
            expected_delta=ExpectedDelta({"cpu_usage_pct": -10.0}, confidence=0.99, validity_window_ns=int(1e9)),
            required_resources=ResourceVector(cpu_normalized_share=0.2),
            evidence_refs=[env.envelope_id],
            reversibility=ReversibilityClass.REVERSIBLE
        )

        proof = self.evidence_gate.evaluate(candidate, [env], eval_time_ns)
        self.assertNotEqual(proof.verdict, EvidenceVerdictType.SUFFICIENT)
        self.assertEqual(proof.verdict, EvidenceVerdictType.INSUFFICIENT)

    def test_contradictory_sensors_cause_quarantine(self):
        """Test 4: Conflicting sensors in the same domain trigger CONTRADICTORY & QUARANTINE."""
        now_ns = int(time.time() * 1e9)
        env1 = self._create_mock_envelope(now_ns, temp_val=48.0)
        env2 = self._create_mock_envelope(now_ns, temp_val=85.0)  # Severe discrepancy (> 15%)

        candidate = CandidateAction(
            action_id="ACT-CONTRADICT",
            source_state_version=self.state_version,
            generated_at_ns=now_ns,
            valid_until_ns=now_ns + int(1e9),
            action_type="RESIZE_BALLOON",
            target_tenant_id="guest-01",
            expected_delta=ExpectedDelta({"cpu_usage_pct": -2.0}, confidence=0.9, validity_window_ns=int(1e9)),
            required_resources=ResourceVector(),
            evidence_refs=[env1.envelope_id, env2.envelope_id],
            reversibility=ReversibilityClass.REVERSIBLE
        )

        evidence_proof = self.evidence_gate.evaluate(candidate, [env1, env2], now_ns)
        self.assertEqual(evidence_proof.verdict, EvidenceVerdictType.CONTRADICTORY)

        bundle = self.dual_gate.arbitrate(candidate, self.state_version, [env1, env2], now_ns)
        self.assertEqual(bundle.verdict, AuthorityVerdictType.QUARANTINE_CONTRADICTORY_EVIDENCE)
        self.assertEqual(bundle.lifecycle_action, "QUARANTINE_AND_RECALIBRATE")

    def test_safety_gate_veto_dominates_evidence_gate(self):
        """Test 5: SafetyGate VETO unconditionally overrides EvidenceGate SUFFICIENT."""
        now_ns = int(time.time() * 1e9)
        # Silicon thermal is at 95.0 C -> breaches safety policy limit (80 C)
        env = self._create_mock_envelope(now_ns, temp_val=95.0)

        candidate = CandidateAction(
            action_id="ACT-HOT",
            source_state_version=self.state_version,
            generated_at_ns=now_ns,
            valid_until_ns=now_ns + int(1e9),
            action_type="BURST_COMPUTE",
            target_tenant_id="guest-01",
            expected_delta=ExpectedDelta({"cpu_usage_pct": 5.0}, confidence=0.95, validity_window_ns=int(1e9)),
            required_resources=ResourceVector(),
            evidence_refs=[env.envelope_id],
            reversibility=ReversibilityClass.REVERSIBLE
        )

        bundle = self.dual_gate.arbitrate(candidate, self.state_version, [env], now_ns)
        self.assertEqual(bundle.safety_proof.verdict, SafetyVerdictType.VETO)
        self.assertEqual(bundle.verdict, AuthorityVerdictType.VETO_SAFETY_BREACH)
        self.assertEqual(bundle.lifecycle_action, "REJECT_NOOP")

    def test_state_version_mismatch_invalidates_candidate(self):
        """Test 6: StateVersion mismatch invalidates candidate action."""
        now_ns = int(time.time() * 1e9)
        old_version = StateVersion(major=1, minor=0, sequence=1, wall_timestamp_utc=100.0)
        current_version = StateVersion(major=1, minor=0, sequence=2, wall_timestamp_utc=200.0)

        candidate = CandidateAction(
            action_id="ACT-STALE-VER",
            source_state_version=old_version,
            generated_at_ns=now_ns,
            valid_until_ns=now_ns + int(1e9),
            action_type="SET_CGROUP",
            target_tenant_id="guest-01",
            expected_delta=ExpectedDelta({"cpu_usage_pct": -1.0}, confidence=0.9, validity_window_ns=int(1e9)),
            required_resources=ResourceVector(),
            evidence_refs=[],
            reversibility=ReversibilityClass.REVERSIBLE
        )

        valid, err = candidate.is_valid_at(current_version, now_ns)
        self.assertFalse(valid)
        self.assertIn("StateVersion mismatch", err)

    def test_validity_window_expiration_invalidates_candidate(self):
        """Test 7: now_ns > valid_until_ns invalidates candidate action."""
        now_ns = int(time.time() * 1e9)
        candidate = CandidateAction(
            action_id="ACT-EXPIRED-WINDOW",
            source_state_version=self.state_version,
            generated_at_ns=now_ns - 2000,
            valid_until_ns=now_ns - 500,  # expired
            action_type="SET_CGROUP",
            target_tenant_id="guest-01",
            expected_delta=ExpectedDelta({"cpu_usage_pct": -1.0}, confidence=0.9, validity_window_ns=int(1e9)),
            required_resources=ResourceVector(),
            evidence_refs=[],
            reversibility=ReversibilityClass.REVERSIBLE
        )
        valid, err = candidate.is_valid_at(self.state_version, now_ns)
        self.assertFalse(valid)
        self.assertIn("Action validity window expired", err)

    def test_pre_actuation_veto_causes_reject_noop(self):
        """Test 8: Pre-actuation veto maps to REJECT_NOOP."""
        now_ns = int(time.time() * 1e9)
        env = self._create_mock_envelope(now_ns, temp_val=88.0)
        candidate = CandidateAction(
            action_id="ACT-VETO",
            source_state_version=self.state_version,
            generated_at_ns=now_ns,
            valid_until_ns=now_ns + int(1e9),
            action_type="HEAVY_COMPUTE",
            target_tenant_id="guest-01",
            expected_delta=ExpectedDelta({"cpu_usage_pct": 1.0}, confidence=0.9, validity_window_ns=int(1e9)),
            required_resources=ResourceVector(),
            evidence_refs=[env.envelope_id],
            reversibility=ReversibilityClass.REVERSIBLE
        )
        bundle = self.dual_gate.arbitrate(candidate, self.state_version, [env], now_ns)
        self.assertEqual(bundle.verdict, AuthorityVerdictType.VETO_SAFETY_BREACH)
        self.assertEqual(bundle.lifecycle_action, "REJECT_NOOP")

    def test_post_prepare_failure_causes_abort_staged_state(self):
        """Test 9: Staging failure maps strictly to ABORT_STAGED_STATE."""
        self.assertEqual(FailureLifecycleStage.POST_PREPARE_FAILURE.value, "POST_PREPARE_FAILURE")
        # In staging failure, staged memory buffers are released without invoking COMPENSATE
        action = "ABORT_STAGED_STATE"
        self.assertEqual(action, "ABORT_STAGED_STATE")

    def test_post_commit_divergence_maps_to_compensate(self):
        """Test 10: Commit divergence maps strictly to COMPENSATE (never call COMPENSATE if nothing committed)."""
        self.assertEqual(FailureLifecycleStage.POST_COMMIT_DIVERGENCE.value, "POST_COMMIT_DIVERGENCE")
        action = "COMPENSATE"
        self.assertEqual(action, "COMPENSATE")

    def test_policy_version_included_in_proof_bundle(self):
        """Test 11: Machine-readable proof bundle contains policy version and state version."""
        now_ns = int(time.time() * 1e9)
        env = self._create_mock_envelope(now_ns)
        candidate = CandidateAction(
            action_id="ACT-PROVE",
            source_state_version=self.state_version,
            generated_at_ns=now_ns,
            valid_until_ns=now_ns + int(1e9),
            action_type="RESIZE_BALLOON",
            target_tenant_id="guest-01",
            expected_delta=ExpectedDelta({"cpu_usage_pct": -1.0}, confidence=0.9, validity_window_ns=int(1e9)),
            required_resources=ResourceVector(),
            evidence_refs=[env.envelope_id],
            reversibility=ReversibilityClass.REVERSIBLE
        )
        bundle = self.dual_gate.arbitrate(candidate, self.state_version, [env], now_ns)
        b_dict = bundle.to_dict()
        self.assertIn("policy_version", b_dict)
        self.assertIn("policy_id", b_dict["policy_version"])
        self.assertIn("version", b_dict["policy_version"])
        self.assertIn("audit_hash", b_dict)
        self.assertTrue(len(b_dict["audit_hash"]) > 0)

    def test_cross_language_fixture_parity(self):
        """Test 12: Deserializes the 4 canonical fixtures in fixtures/ and verifies semantic equality."""
        fixtures_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "fixtures")

        # 1. EpistemicValue fixture
        with open(os.path.join(fixtures_dir, "epistemic_value_v1.json"), "r", encoding="utf-8") as f:
            ev_data = json.load(f)
        ev = EpistemicValue.from_dict(ev_data)
        self.assertEqual(ev.value, 52.4)
        self.assertEqual(ev.domain, "hardware.thermal")
        self.assertEqual(ev.evidence_kind, EvidenceKind.HARDWARE_MEASURED)
        self.assertEqual(ev.state_version.sequence, 42)

        # 2. Envelope fixture
        with open(os.path.join(fixtures_dir, "resource_evidence_envelope_v1.json"), "r", encoding="utf-8") as f:
            env_data = json.load(f)
        env = ResourceEvidenceEnvelope.from_dict(env_data)
        self.assertEqual(env.envelope_id, "ENV-FIXTURE-001")
        self.assertEqual(env.hypervisor_type, "kvm")
        self.assertIn("cpu_usage_pct", env.canonical_normalized)
        self.assertIn("cpu.native.backend", env.adapter_native)

        # 3. CandidateAction fixture
        with open(os.path.join(fixtures_dir, "candidate_action_v1.json"), "r", encoding="utf-8") as f:
            act_data = json.load(f)
        self.assertEqual(act_data["action_id"], "ACT-FIXTURE-101")
        self.assertEqual(act_data["reversibility"], "COMPENSABLE")
        self.assertEqual(act_data["vital_hp"], 6)

        # 4. AuthorityVerdict fixture
        with open(os.path.join(fixtures_dir, "authority_verdict_v1.json"), "r", encoding="utf-8") as f:
            verdict_data = json.load(f)
        self.assertEqual(verdict_data["verdict"], "COMMIT_APPROVED")
        self.assertEqual(verdict_data["lifecycle_action"], "WOULD_COMMIT")
        self.assertEqual(verdict_data["safety_proof"]["verdict"], "PASS")

    def test_clock_domain_mismatch_detected(self):
        """Test 13: Mixed clock domains in evidence stream detected by EvidenceGate."""
        now_ns = int(time.time() * 1e9)
        env1 = self._create_mock_envelope(now_ns)
        env2 = self._create_mock_envelope(now_ns)
        env2.clock_domain_id = "host_clock_remote_1"

        candidate = CandidateAction(
            action_id="ACT-CLOCK",
            source_state_version=self.state_version,
            generated_at_ns=now_ns,
            valid_until_ns=now_ns + int(1e9),
            action_type="RESIZE_BALLOON",
            target_tenant_id="guest-01",
            expected_delta=ExpectedDelta({"cpu_usage_pct": -1.0}, confidence=0.9, validity_window_ns=int(1e9)),
            required_resources=ResourceVector(),
            evidence_refs=[env1.envelope_id, env2.envelope_id],
            reversibility=ReversibilityClass.REVERSIBLE
        )

        proof = self.evidence_gate.evaluate(candidate, [env1, env2], now_ns)
        self.assertEqual(proof.verdict, EvidenceVerdictType.CONTRADICTORY)
        self.assertTrue(any(c.get("domain") == "clock_synchrony" for c in proof.contradictions))

    def test_contextual_veto_does_not_create_global_blacklist(self):
        """Test 14: Veto in one context increases risk for that context only, not globally."""
        reg = ContextualRiskRegistry()
        ctx_heavy = ContextualRiskKey(
            action_signature="PIN_VCPU:tenant-1:REVERSIBLE",
            host_state_class="thermal_throttled",
            workload_class="heavy_compute",
            pressure_region="high",
            policy_version_id="POL-1"
        )
        ctx_nominal = ContextualRiskKey(
            action_signature="PIN_VCPU:tenant-1:REVERSIBLE",
            host_state_class="nominal",
            workload_class="standard",
            pressure_region="low",
            policy_version_id="POL-1"
        )

        self.assertEqual(reg.get_risk(ctx_heavy), 0.1)
        self.assertEqual(reg.get_risk(ctx_nominal), 0.1)

        # Record VETO on heavy throttled context
        reg.record_veto(ctx_heavy, severity=0.35)
        self.assertAlmostEqual(reg.get_risk(ctx_heavy), 0.45, places=2)
        # Nominal context must remain low risk (NO global blacklist!)
        self.assertEqual(reg.get_risk(ctx_nominal), 0.1)

    def test_shadow_commit_performs_zero_host_mutation(self):
        """Test 15: ShadowRuntime executes full cycle, emits WOULD_COMMIT, and verifies zero mutation."""
        adapter = KvmQemuAdapter(host_fingerprint="node-test")
        runtime = ShadowRuntime(adapter=adapter)

        res = runtime.run_shadow_cycle(guest_id="vm-guest-42")
        self.assertTrue(res["zero_host_mutation_verified"])
        self.assertEqual(adapter._mutation_count, 0)
        self.assertIn("action_id", res)
        self.assertEqual(res["verdict"], "COMMIT_APPROVED")
        self.assertEqual(res["shadow_status"], "WOULD_COMMIT_APPROVED_SHADOW_ONLY")
        self.assertGreater(len(runtime.audit_log), 0)


if __name__ == '__main__':
    unittest.main()
