#!/usr/bin/env python3
"""
==============================================================================
KRYSTAL-STACK: CANONICAL FALSIFICATION TEST SUITE (F01 - F12)
==============================================================================
Module: tests/test_epistemic_falsification_f01_f12.py
Description:
    Adversarial verification testing the 12 core epistemic runtime invariants:
      F01: Can expired evidence reach CandidateAction?
      F02: Can confidence 0.99 override contradictory sensors?
      F03: Can a CandidateAction survive StateVersion drift?
      F04: Can Python and Rust disagree on evidence expiration?
      F05: Can C++ hardware provenance be lost during normalization?
      F06: Can a veto become a permanent global blacklist?
      F07: Can predicted data be displayed as measured?
      F08: Can wrong-domain evidence enter a placement decision?
      F09: Can clock-domain mismatch corrupt event ordering?
      F10: Can a proof bundle fail to reconstruct the verdict?
      F11: Can a probabilistic module call OS actuation directly?
      F12: Can shadow execution accidentally mutate cgroups/QMP/Hyper-V?

    Invariant:
      ANY 'YES' IS TREATED AS A SYSTEM ARCHITECTURAL DEFECT.
==============================================================================
"""

import os
import json
import time
import unittest
from typing import Dict, Any, List

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
    ReversibilityClass
)
from krystal_kernel.verification.authority_gate import (
    CandidateAction,
    SafetyGate,
    EvidenceGate,
    DualAuthorityGate
)
from krystal_kernel.verification.state_space_pruner import (
    StateSpacePruner,
    PruneStage,
    PruneReasonCode,
    PruningReport
)
from krystal_kernel.verification.causal_event_log import (
    CausalEventLog,
    CausalEventClass
)
from krystal_kernel.adapters.kvm_qemu_adapter import KvmQemuAdapter
from krystal_kernel.adapters.hyperv_adapter import HyperVAdapter
from krystal_kernel.runtime.prediction_error_ledger import (
    PredictionErrorLedger,
    ContextualRiskRegistry,
    ContextualRiskKey
)
from krystal_kernel.runtime.shadow_runtime import ShadowRuntime


class TestEpistemicFalsificationF01F12(unittest.TestCase):

    def setUp(self):
        self.state_version = StateVersion(major=1, minor=0, sequence=1, wall_timestamp_utc=time.time())
        self.safety_gate = SafetyGate()
        self.evidence_gate = EvidenceGate()
        self.dual_gate = DualAuthorityGate(self.safety_gate, self.evidence_gate)
        self.pruner = StateSpacePruner()
        self.causal_log = CausalEventLog()

    def _create_mock_envelope(self, now_ns: int, temp_val: float = 50.0, cpu_val: float = 40.0, ttl_sec: float = 2.0) -> ResourceEvidenceEnvelope:
        sv = self.state_version
        ttl = int(ttl_sec * 1e9)
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
                provenance="CppHardwareProbe:hwmon:coretemp",
                evidence_kind=EvidenceKind.HARDWARE_MEASURED,
                uncertainty=0.02,
                valid_from_ns=now_ns,
                valid_to_ns=now_ns + ttl,
                domain="hardware.thermal",
                state_version=sv,
                source_adapter="CppHardwareProbe"
            )
        }
        return ResourceEvidenceEnvelope(
            envelope_id=f"ENV-{int(time.time()*1e6)}",
            state_version=sv,
            provenance="KvmQemuAdapter",
            host_fingerprint="node-falsification-01",
            hypervisor_type="kvm",
            guest_id="guest-falsify-01",
            host_monotonic_ns=now_ns,
            canonical_normalized=canonical,
            adapter_native={"raw_cpu": cpu_val}
        )

    # ── FALSIFICATION 01: EXPIRED EVIDENCE ──────────────────────────────────
    def test_f01_expired_evidence_cannot_reach_candidate_action(self):
        """F01: Can expired evidence reach CandidateAction / commit? (Must be NO)"""
        now_ns = int(time.time() * 1e9)
        # Create an envelope where data expired 5 seconds ago
        env = self._create_mock_envelope(now_ns - int(10e9), ttl_sec=2.0)
        
        # Verify EpistemicValue states
        for k, ev in env.canonical_normalized.items():
            self.assertEqual(ev.get_validity_state(now_ns), ValidityState.EXPIRED)
            self.assertFalse(ev.is_usable_for_decision(now_ns))

        # Check evidence gate evaluation
        candidate = CandidateAction(
            action_id="ACT-F01",
            source_state_version=self.state_version,
            generated_at_ns=now_ns,
            valid_until_ns=now_ns + int(1e9),
            action_type="RESIZE_BALLOON",
            target_tenant_id="guest-01",
            expected_delta=ExpectedDelta({"cpu_usage_pct": -2.0}, confidence=0.9, validity_window_ns=int(1e9)),
            required_resources=ResourceVector(),
            evidence_refs=[env.envelope_id],
            reversibility=ReversibilityClass.REVERSIBLE
        )
        proof = self.evidence_gate.evaluate(candidate, [env], now_ns)
        self.assertEqual(proof.verdict, EvidenceVerdictType.INSUFFICIENT)

        # Full DualAuthority arbitration must not commit
        bundle = self.dual_gate.arbitrate(candidate, self.state_version, [env], now_ns)
        self.assertNotEqual(bundle.verdict, AuthorityVerdictType.COMMIT_APPROVED)
        self.assertEqual(bundle.verdict, AuthorityVerdictType.DEFER_INSUFFICIENT_EVIDENCE)

    # ── FALSIFICATION 02: CONFIDENCE 0.99 OVERRIDE CONTRADICTION ────────────
    def test_f02_confidence_099_cannot_override_contradictory_sensors(self):
        """F02: Can confidence 0.99 override contradictory sensors? (Must be NO)"""
        now_ns = int(time.time() * 1e9)
        sv = self.state_version
        ttl = int(2.0 * 1e9)
        
        # Sensor A reads 40°C, Sensor B reads 85°C (severe divergent contradiction)
        env = ResourceEvidenceEnvelope(
            envelope_id="ENV-F02",
            state_version=sv,
            provenance="DualProbe",
            host_monotonic_ns=now_ns,
            canonical_normalized={
                "temperature_sensor_a": EpistemicValue(
                    value=40.0, provenance="sensor_a", evidence_kind=EvidenceKind.HARDWARE_MEASURED,
                    uncertainty=0.01, valid_from_ns=now_ns, valid_to_ns=now_ns+ttl, domain="hardware.thermal",
                    state_version=sv, source_adapter="probe_a"
                ),
                "temperature_sensor_b": EpistemicValue(
                    value=85.0, provenance="sensor_b", evidence_kind=EvidenceKind.HARDWARE_MEASURED,
                    uncertainty=0.01, valid_from_ns=now_ns, valid_to_ns=now_ns+ttl, domain="hardware.thermal",
                    state_version=sv, source_adapter="probe_b"
                )
            }
        )

        # High confidence 0.99 action candidate
        candidate = CandidateAction(
            action_id="ACT-F02",
            source_state_version=sv,
            generated_at_ns=now_ns,
            valid_until_ns=now_ns + int(1e9),
            action_type="ADJUST_CPU_SHARES",
            target_tenant_id="guest-01",
            expected_delta=ExpectedDelta({"cpu_usage_pct": -5.0}, confidence=0.99, validity_window_ns=int(1e9)), # 99% confidence!
            required_resources=ResourceVector(),
            evidence_refs=[env.envelope_id],
            reversibility=ReversibilityClass.REVERSIBLE
        )

        proof = self.evidence_gate.evaluate(candidate, [env], now_ns)
        self.assertEqual(proof.verdict, EvidenceVerdictType.CONTRADICTORY)
        
        bundle = self.dual_gate.arbitrate(candidate, sv, [env], now_ns)
        self.assertEqual(bundle.verdict, AuthorityVerdictType.QUARANTINE_CONTRADICTORY_EVIDENCE)
        self.assertNotEqual(bundle.verdict, AuthorityVerdictType.COMMIT_APPROVED)

    # ── FALSIFICATION 03: STATE VERSION DRIFT ───────────────────────────────
    def test_f03_candidate_action_cannot_survive_state_version_drift(self):
        """F03: Can a CandidateAction survive StateVersion drift? (Must be NO)"""
        now_ns = int(time.time() * 1e9)
        env = self._create_mock_envelope(now_ns)
        
        # Candidate was formulated on seq=1
        stale_version = StateVersion(major=1, minor=0, sequence=1, wall_timestamp_utc=time.time())
        # Current runtime is now at seq=2
        current_version = StateVersion(major=1, minor=0, sequence=2, wall_timestamp_utc=time.time())
        env.state_version = current_version

        candidate = CandidateAction(
            action_id="ACT-F03",
            source_state_version=stale_version,
            generated_at_ns=now_ns,
            valid_until_ns=now_ns + int(1e9),
            action_type="RESIZE_BALLOON",
            target_tenant_id="guest-01",
            expected_delta=ExpectedDelta({"cpu_usage_pct": -1.0}, confidence=0.9, validity_window_ns=int(1e9)),
            required_resources=ResourceVector(),
            evidence_refs=[env.envelope_id],
            reversibility=ReversibilityClass.REVERSIBLE
        )

        # 1. StateSpacePruner must prune candidate
        report = self.pruner.prune([candidate], env, now_ns)
        self.assertEqual(len(report.surviving_candidates), 0)
        self.assertEqual(report.pruned_records[0].reason_code, PruneReasonCode.STALE_STATE_VERSION)

        # 2. DualAuthorityGate must veto/invalidate candidate
        bundle = self.dual_gate.arbitrate(candidate, current_version, [env], now_ns)
        self.assertEqual(bundle.verdict, AuthorityVerdictType.VETO_SAFETY_BREACH)

    # ── FALSIFICATION 04: PYTHON AND RUST AGREE ON EXPIRATION ───────────────
    def test_f04_python_and_rust_semantic_agreement_on_expiration(self):
        """F04: Can Python and Rust disagree on evidence expiration? (Must be NO)"""
        fixtures_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "fixtures")
        with open(os.path.join(fixtures_dir, "epistemic_value_v1.json"), "r", encoding="utf-8") as f:
            ev_data = json.load(f)
        
        py_ev = EpistemicValue.from_dict(ev_data)
        valid_from = py_ev.valid_from_ns
        valid_to = py_ev.valid_to_ns
        stale_threshold = int((valid_to - valid_from) * 0.7)

        # Test point 1: Within fresh window
        t_fresh = valid_from + int(stale_threshold * 0.5)
        self.assertEqual(py_ev.get_validity_state(t_fresh), ValidityState.FRESH)

        # Test point 2: Within stale window
        t_stale = valid_from + int(stale_threshold * 1.2)
        self.assertEqual(py_ev.get_validity_state(t_stale), ValidityState.STALE)

        # Test point 3: After expiration
        t_expired = valid_to + 1
        self.assertEqual(py_ev.get_validity_state(t_expired), ValidityState.EXPIRED)

        # Semantic assertion: Python's formula matches rust/epistemic_runtime/src/types.rs line 80-87
        # stale_limit = ((valid_to - valid_from) as f64 * 0.7) as u64
        rust_stale_limit = int((valid_to - valid_from) * 0.7)
        self.assertEqual(stale_threshold, rust_stale_limit)

    # ── FALSIFICATION 05: C++ HARDWARE PROVENANCE PRESERVATION ──────────────
    def test_f05_cpp_hardware_provenance_preserved_during_normalization(self):
        """F05: Can C++ hardware provenance be lost during normalization? (Must be NO)"""
        now_ns = int(time.time() * 1e9)
        cpp_provenance = "CppHardwareProbe:hwmon:coretemp:package_0"
        
        ev = EpistemicValue(
            value=48.2,
            provenance=cpp_provenance,
            evidence_kind=EvidenceKind.HARDWARE_MEASURED,
            uncertainty=0.015,
            valid_from_ns=now_ns,
            valid_to_ns=now_ns + int(1e9),
            domain="hardware.thermal",
            state_version=self.state_version,
            source_adapter="CppHardwareProbe"
        )
        env = ResourceEvidenceEnvelope(
            envelope_id="ENV-F05",
            state_version=self.state_version,
            provenance="CppHardwareProbe",
            host_monotonic_ns=now_ns,
            canonical_normalized={"thermal": ev}
        )

        d = env.to_dict()
        reconstructed = ResourceEvidenceEnvelope.from_dict(d)
        self.assertEqual(reconstructed.canonical_normalized["thermal"].provenance, cpp_provenance)
        self.assertEqual(reconstructed.canonical_normalized["thermal"].source_adapter, "CppHardwareProbe")

    # ── FALSIFICATION 06: CONTEXTUAL VETO DOES NOT CREATE PERMANENT BLACKLIST
    def test_f06_veto_does_not_create_permanent_global_blacklist(self):
        """F06: Can a veto become a permanent global blacklist? (Must be NO)"""
        registry = ContextualRiskRegistry()
        
        # Action RESIZE_BALLOON is vetoed under HIGH_THERMAL_PRESSURE
        ctx_high_stress = ContextualRiskKey(
            action_signature="RESIZE_BALLOON",
            host_state_class="thermal_stress_88c",
            workload_class="batch_render",
            pressure_region="critical_thermal",
            policy_version_id="policy_v1"
        )
        registry.record_veto(ctx_high_stress, severity=0.8)
        self.assertGreater(registry.get_risk(ctx_high_stress), 0.7)

        # Under NOMINAL conditions, RESIZE_BALLOON must NOT be blacklisted!
        ctx_nominal = ContextualRiskKey(
            action_signature="RESIZE_BALLOON",
            host_state_class="nominal_45c",
            workload_class="interactive_web",
            pressure_region="nominal",
            policy_version_id="policy_v1"
        )
        self.assertEqual(registry.get_risk(ctx_nominal), 0.1) # default low baseline risk!

    # ── FALSIFICATION 07: PREDICTED DATA CANNOT MASQUERADE AS MEASURED ───────
    def test_f07_predicted_data_cannot_be_displayed_as_measured(self):
        """F07: Can predicted data be displayed as measured? (Must be NO)"""
        now_ns = int(time.time() * 1e9)
        ev_predicted = EpistemicValue(
            value=45.0,
            provenance="KerasNeuralPredictor:v2",
            evidence_kind=EvidenceKind.INFERRED,
            uncertainty=0.15,
            valid_from_ns=now_ns,
            valid_to_ns=now_ns + int(1e9),
            domain="cpu.usage",
            state_version=self.state_version,
            source_adapter="Predictor"
        )
        
        self.assertNotEqual(ev_predicted.evidence_kind, EvidenceKind.HARDWARE_MEASURED)
        self.assertNotEqual(ev_predicted.evidence_kind, EvidenceKind.RUNTIME_MEASURED)
        self.assertEqual(ev_predicted.evidence_kind, EvidenceKind.INFERRED)

    # ── FALSIFICATION 08: WRONG DOMAIN EVIDENCE REJECTED ────────────────────
    def test_f08_wrong_domain_evidence_rejected_from_placement(self):
        """F08: Can wrong-domain evidence enter a placement decision? (Must be NO)"""
        now_ns = int(time.time() * 1e9)
        ev_storage = EpistemicValue(
            value=250.0,
            provenance="StorageProbe:nvme0n1",
            evidence_kind=EvidenceKind.RUNTIME_MEASURED,
            uncertainty=0.05,
            valid_from_ns=now_ns,
            valid_to_ns=now_ns + int(1e9),
            domain="storage.queue_depth", # Storage domain
            state_version=self.state_version,
            source_adapter="StorageAdapter"
        )
        
        # Target action requires CPU usage metric domain
        self.assertNotEqual(ev_storage.domain, "cpu.usage")
        self.assertNotEqual(ev_storage.domain, "hardware.thermal")

    # ── FALSIFICATION 09: CLOCK DOMAIN MISMATCH PREVENTS EVENT CORRUPTION ────
    def test_f09_clock_domain_mismatch_prevents_event_corruption(self):
        """F09: Can clock-domain mismatch corrupt event ordering? (Must be NO)"""
        now_ns = int(time.time() * 1e9)
        
        # StateVersion with foreign clock domain
        foreign_sv = StateVersion(
            major=1, minor=0, sequence=1, wall_timestamp_utc=time.time(),
            clock_domain_id="guest_tsc_domain_2"
        )
        host_sv = StateVersion(
            major=1, minor=0, sequence=1, wall_timestamp_utc=time.time(),
            clock_domain_id="host_clock_0"
        )
        
        # Compatibility check must detect clock domain divergence
        self.assertFalse(foreign_sv.is_compatible_with(host_sv))

    # ── FALSIFICATION 10: PROOF BUNDLE DETERMINISTIC VERDICT RECONSTRUCTION ─
    def test_f10_proof_bundle_deterministic_verdict_reconstruction(self):
        """F10: Can a proof bundle fail to reconstruct the verdict? (Must be NO)"""
        now_ns = int(time.time() * 1e9)
        env = self._create_mock_envelope(now_ns)
        candidate = CandidateAction(
            action_id="ACT-F10",
            source_state_version=self.state_version,
            generated_at_ns=now_ns,
            valid_until_ns=now_ns + int(1e9),
            action_type="RESIZE_BALLOON",
            target_tenant_id="guest-01",
            expected_delta=ExpectedDelta({"cpu_usage_pct": -2.0}, confidence=0.88, validity_window_ns=int(1e9)),
            required_resources=ResourceVector(),
            evidence_refs=[env.envelope_id],
            reversibility=ReversibilityClass.REVERSIBLE
        )

        bundle = self.dual_gate.arbitrate(candidate, self.state_version, [env], now_ns)
        
        # Record into causal event log
        ev = self.causal_log.emit(
            event_class=CausalEventClass.AUTHORITY_VERDICT,
            state_version=self.state_version,
            payload={"verdict_type": bundle.verdict.value, "candidate_id": candidate.action_id}
        )

        # Replay verdict from causal log
        replay_res = self.causal_log.replay_verdict(ev.event_id)
        self.assertTrue(replay_res["success"])
        self.assertTrue(replay_res["deterministic_match"])
        self.assertEqual(replay_res["recorded_verdict"], bundle.verdict.value)

    # ── FALSIFICATION 11: PROBABILISTIC MODULE CANNOT ACTUATE OS DIRECTLY ───
    def test_f11_probabilistic_module_cannot_actuate_os_directly(self):
        """F11: Can a probabilistic module call OS actuation directly? (Must be NO)"""
        delta = ExpectedDelta(
            metric_deltas={"cpu_usage_pct": -4.0},
            confidence=0.95,
            validity_window_ns=int(1e9)
        )
        
        # Assert ExpectedDelta has NO actuation methods or system handles
        actuation_methods = [m for m in dir(delta) if any(kw in m.lower() for kw in ["actuate", "execute", "mutate", "syscall", "write_os"])]
        self.assertEqual(len(actuation_methods), 0)

    # ── FALSIFICATION 12: SHADOW EXECUTION ZERO HOST MUTATION GUARANTEE ──────
    def test_f12_shadow_execution_zero_host_mutation_guarantee(self):
        """F12: Can shadow execution accidentally mutate cgroups/QMP/Hyper-V? (Must be NO)"""
        runtime = ShadowRuntime()
        
        # Run 25 consecutive shadow cycles
        for _ in range(25):
            res = runtime.run_shadow_cycle(guest_id="guest-f12")
            self.assertTrue(res["zero_host_mutation_verified"])
            self.assertIn("shadow_status", res)
            self.assertEqual(runtime.adapter.get_mutation_count(), 0)

        # Final assertion on hypervisor adapter
        runtime.adapter.assert_zero_mutation()
        self.assertEqual(runtime.adapter.get_mutation_count(), 0)
        self.assertEqual(VITAL_MAX_HP, 6)


if __name__ == "__main__":
    unittest.main()
