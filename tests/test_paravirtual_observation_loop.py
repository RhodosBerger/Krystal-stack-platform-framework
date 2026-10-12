#!/usr/bin/env python3
"""
UNIT & INTEGRATION TESTS: PARAVIRTUAL OBSERVATION LOOP & AUTHORITATIVE TRANSITION
=============================================================================
Tests the decomposed architecture:
  1. Cross-platform telemetry normalization (KVM QEMU vs Windows Hyper-V).
  2. Zero-mutation invariant during observation loop.
  3. Decomposed two-gate engine:
       - Hard Safety Gate in {PASS, VETO}
       - Evidence Gate in {SUFFICIENT, INSUFFICIENT, CONTRADICTORY}
       - AuthorityVerdict = f(SafetyGate, EvidenceGate) -> 4 distinct outcomes.
  4. Precise 3-stage failure lifecycle:
       - PRE-ACTUATION VETO -> REJECT_NOOP (zero compensation, zero OS mutation).
       - POST-PREPARE FAILURE -> ABORT_STAGED_STATE (release in-memory buffers).
       - POST-COMMIT DIVERGENCE -> COMPENSATE (inverse rollback actuation).
  5. 4-tier guardrails (MemorySafety, ResourceSafety, PhysicalSafety, SystemInvariants).
  6. Dynamic policies without hardcoded magic numbers.
  7. Contextual Risk Learning from VETO (Risk(a | context), no global blacklisting).
  8. Antigravity Candidate Generator (typed candidate + requirements + falsification).
=============================================================================
"""

import os
import sys
import time
import unittest

WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

from krystal_kernel.scene_constructor_presets import VITAL_MAX_HP
from krystal_kernel.verification.paravirtual_envelope import (
    CandidateAction,
    EvidenceRequirements,
    FalsificationExperiment,
    ReversibilityClass,
    SafetyTier,
    SafetyVerdictType,
    EvidenceVerdictType,
    AuthorityVerdictType,
    ResourceEvidenceEnvelope,
    DynamicSafetyPolicy,
    ContextualRiskRegistry,
    HardSafetyGate,
    EvidenceGate,
    AuthoritativeDecisionEngine,
)
from krystal_kernel.verification.hypervisor_adapters import (
    KvmQemuAdapter,
    HyperVAdapter,
)
from krystal_kernel.verification.antigravity_candidate_generator import (
    AntigravityCandidateGenerator,
)


class TestParavirtualAdaptersAndNormalization(unittest.TestCase):
    """Validates cross-platform telemetry ingestion and canonical normalization."""

    def test_01_kvm_adapter_canonical_normalization(self):
        adapter = KvmQemuAdapter(host_fingerprint="node-arch-01")
        env = adapter.sample_telemetry(guest_id="guest-linux-01")

        self.assertEqual(env.hypervisor_type, "kvm")
        self.assertEqual(env.schema_version, "1.0.0")
        self.assertIn("cpu_usage_pct", env.telemetry)
        self.assertIn("memory_rss_bytes", env.telemetry)
        self.assertIn("temperature_celsius", env.telemetry)
        self.assertEqual(env.epistemic_tag, "[CANONICAL_INVARIANT]")
        adapter.assert_zero_mutation()

    def test_02_hyperv_adapter_canonical_normalization(self):
        adapter = HyperVAdapter(host_fingerprint="node-win11-01")
        env = adapter.sample_telemetry(guest_id="guest-win-01")

        self.assertEqual(env.hypervisor_type, "hyper-v")
        self.assertEqual(env.schema_version, "1.0.0")
        self.assertIn("cpu_usage_pct", env.telemetry)
        self.assertIn("memory_rss_bytes", env.telemetry)
        self.assertIn("temperature_celsius", env.telemetry)
        self.assertEqual(env.epistemic_tag, "[CANONICAL_INVARIANT]")
        adapter.assert_zero_mutation()

    def test_03_cross_platform_schema_equivalence(self):
        kvm_adapter = KvmQemuAdapter()
        hyperv_adapter = HyperVAdapter()

        env_kvm = kvm_adapter.sample_telemetry("vm-1")
        env_hyperv = hyperv_adapter.sample_telemetry("vm-2")

        # Canonical metric keys must be identical across Linux and Windows
        self.assertEqual(
            set(env_kvm.telemetry.keys()),
            set(env_hyperv.telemetry.keys())
        )
        self.assertEqual(
            set(env_kvm.uncertainty_vector.keys()),
            set(env_hyperv.uncertainty_vector.keys())
        )


class TestDecomposedGatesAndAuthorityVerdict(unittest.TestCase):
    """Validates SafetyGate x EvidenceGate composition into 4 authority outcomes."""

    def setUp(self):
        self.policy = DynamicSafetyPolicy()
        self.risk_registry = ContextualRiskRegistry()
        self.engine = AuthoritativeDecisionEngine(
            policy=self.policy,
            risk_registry=self.risk_registry
        )

    def _create_candidate(self, state_version: int = 1, vital_hp: int = VITAL_MAX_HP) -> CandidateAction:
        return CandidateAction(
            action_id="CAND-TEST-01",
            state_version=state_version,
            action_type="RESIZE_BALLOON",
            target_tenant_id="tenant-db",
            expected_delta={"cpu_pct": -5.0},
            required_resources={"ram_mb": 512.0},
            evidence_refs=["ENV-REF-01"],
            reversibility=ReversibilityClass.COMPENSABLE,
            evidence_requirements=EvidenceRequirements(
                max_acceptable_age_seconds=2.0,
                mandatory_dimensions=["cpu_usage_pct", "memory_rss_bytes"]
            ),
            vital_hp=vital_hp
        )

    def _create_envelope(self, env_id: str = "ENV-REF-01", cpu: float = 45.0, age_sec: float = 0.1) -> ResourceEvidenceEnvelope:
        return ResourceEvidenceEnvelope(
            envelope_id=env_id,
            timestamp=time.time() - age_sec,
            hypervisor_type="kvm",
            guest_id="tenant-db",
            telemetry={"cpu_usage_pct": cpu, "memory_rss_bytes": 1024 * 1024 * 1024}
        )

    def test_04_outcome_commit_approved(self):
        """SafetyGate = PASS, EvidenceGate = SUFFICIENT -> COMMIT_APPROVED."""
        candidate = self._create_candidate()
        env = self._create_envelope()
        host_telemetry = {
            "free_ram_bytes": 8 * 1024 * 1024 * 1024,
            "temperature_celsius": 55.0,
            "package_power_watts": 18.0
        }

        verdict = self.engine.evaluate_candidate(
            candidate=candidate,
            envelopes={env.envelope_id: env},
            host_telemetry=host_telemetry
        )

        self.assertEqual(verdict.verdict, AuthorityVerdictType.COMMIT_APPROVED)
        self.assertEqual(verdict.lifecycle_action, "PROCEED_TO_RTP_PREPARE")
        self.assertTrue(verdict.safety_verdict.is_pass)
        self.assertTrue(verdict.evidence_verdict.is_sufficient)

    def test_05_outcome_defer_insufficient_evidence(self):
        """SafetyGate = PASS, EvidenceGate = INSUFFICIENT -> DEFER_INSUFFICIENT_EVIDENCE."""
        candidate = self._create_candidate()
        # Create stale envelope (age = 5.0s > 2.0s requirement)
        stale_env = self._create_envelope(age_sec=5.0)
        host_telemetry = {
            "free_ram_bytes": 8 * 1024 * 1024 * 1024,
            "temperature_celsius": 55.0,
            "package_power_watts": 18.0
        }

        verdict = self.engine.evaluate_candidate(
            candidate=candidate,
            envelopes={stale_env.envelope_id: stale_env},
            host_telemetry=host_telemetry
        )

        self.assertEqual(verdict.verdict, AuthorityVerdictType.DEFER_INSUFFICIENT_EVIDENCE)
        self.assertEqual(verdict.lifecycle_action, "TRIGGER_ACTIVE_MICRO_PROBE")
        self.assertTrue(verdict.safety_verdict.is_pass)
        self.assertEqual(verdict.evidence_verdict.verdict, EvidenceVerdictType.INSUFFICIENT)

    def test_06_outcome_quarantine_contradictory_evidence(self):
        """SafetyGate = PASS, EvidenceGate = CONTRADICTORY -> QUARANTINE_CONTRADICTORY_EVIDENCE."""
        candidate = self._create_candidate()
        candidate.evidence_refs = ["ENV-SENS-1", "ENV-SENS-2"]
        # Two sensors reporting wildly contradictory CPU (15% vs 85%, delta=70% > 35%)
        env1 = self._create_envelope(env_id="ENV-SENS-1", cpu=15.0)
        env2 = self._create_envelope(env_id="ENV-SENS-2", cpu=85.0)

        host_telemetry = {
            "free_ram_bytes": 8 * 1024 * 1024 * 1024,
            "temperature_celsius": 55.0,
            "package_power_watts": 18.0
        }

        verdict = self.engine.evaluate_candidate(
            candidate=candidate,
            envelopes={env1.envelope_id: env1, env2.envelope_id: env2},
            host_telemetry=host_telemetry
        )

        self.assertEqual(verdict.verdict, AuthorityVerdictType.QUARANTINE_CONTRADICTORY_EVIDENCE)
        self.assertEqual(verdict.lifecycle_action, "QUARANTINE_AND_RECALIBRATE")
        self.assertEqual(verdict.evidence_verdict.verdict, EvidenceVerdictType.CONTRADICTORY)

    def test_07_outcome_veto_safety_breach_pre_actuation(self):
        """SafetyGate = VETO -> VETO_SAFETY_BREACH with REJECT_NOOP."""
        candidate = self._create_candidate()
        env = self._create_envelope()
        # Host temperature breaches dynamic policy threshold
        host_telemetry = {
            "free_ram_bytes": 8 * 1024 * 1024 * 1024,
            "temperature_celsius": 89.5,  # Exceeds policy ceiling
            "package_power_watts": 18.0,
            "host_state_class": "THERMAL_PRESSURE",
            "workload_class": "HEAVY",
            "pressure_region": "HIGH",
            "policy_version": "v1.0.0"
        }

        verdict = self.engine.evaluate_candidate(
            candidate=candidate,
            envelopes={env.envelope_id: env},
            host_telemetry=host_telemetry
        )

        self.assertEqual(verdict.verdict, AuthorityVerdictType.VETO_SAFETY_BREACH)
        self.assertEqual(verdict.lifecycle_action, "REJECT_NOOP")
        self.assertTrue(verdict.safety_verdict.is_veto)
        self.assertEqual(verdict.safety_verdict.violation.tier, SafetyTier.PHYSICAL_SAFETY)


class TestFourTierGuardrailsAndLifecycle(unittest.TestCase):
    """Tests MemorySafety, ResourceSafety, PhysicalSafety, SystemInvariants, and Failure Lifecycle."""

    def setUp(self):
        self.policy = DynamicSafetyPolicy()
        self.engine = AuthoritativeDecisionEngine(policy=self.policy)

    def test_08_memory_safety_guardrail(self):
        """Invalid tenant descriptor length triggers MemorySafety veto."""
        candidate = CandidateAction(
            action_id="CAND-MEM",
            state_version=1,
            action_type="SET_CGROUP",
            target_tenant_id="X" * 150,  # Exceeds 128 character buffer bound
            expected_delta={},
            required_resources={},
            evidence_refs=[],
            reversibility=ReversibilityClass.REVERSIBLE,
            evidence_requirements=EvidenceRequirements(),
            vital_hp=VITAL_MAX_HP
        )
        verdict = self.engine.evaluate_candidate(candidate, {}, {})
        self.assertTrue(verdict.safety_verdict.is_veto)
        self.assertEqual(verdict.safety_verdict.violation.tier, SafetyTier.MEMORY_SAFETY)

    def test_09_resource_safety_guardrail(self):
        """Allocation violating host memory headroom floor triggers ResourceSafety veto."""
        candidate = CandidateAction(
            action_id="CAND-RAM",
            state_version=1,
            action_type="RESIZE_BALLOON",
            target_tenant_id="guest-01",
            expected_delta={},
            required_resources={"ram_mb": 7500.0},  # Requesting 7.5GB
            evidence_refs=[],
            reversibility=ReversibilityClass.COMPENSABLE,
            evidence_requirements=EvidenceRequirements(),
            vital_hp=VITAL_MAX_HP
        )
        # Host has 8GB free; remaining 0.5GB breaches 1.0GB headroom floor
        host_telemetry = {"free_ram_bytes": 8 * 1024 * 1024 * 1024}
        verdict = self.engine.evaluate_candidate(candidate, {}, host_telemetry)
        self.assertTrue(verdict.safety_verdict.is_veto)
        self.assertEqual(verdict.safety_verdict.violation.tier, SafetyTier.RESOURCE_SAFETY)

    def test_10_system_invariants_guardrail(self):
        """Stale state version or vital_hp != 6 triggers SystemInvariants veto."""
        candidate_stale = CandidateAction(
            action_id="CAND-STALE",
            state_version=99,  # Current is 1
            action_type="SET_CGROUP",
            target_tenant_id="guest-01",
            expected_delta={},
            required_resources={},
            evidence_refs=[],
            reversibility=ReversibilityClass.REVERSIBLE,
            evidence_requirements=EvidenceRequirements(),
            vital_hp=VITAL_MAX_HP
        )
        verdict = self.engine.evaluate_candidate(candidate_stale, {}, {})
        self.assertTrue(verdict.safety_verdict.is_veto)
        self.assertEqual(verdict.safety_verdict.violation.tier, SafetyTier.SYSTEM_INVARIANTS)

        candidate_hp = CandidateAction(
            action_id="CAND-HP",
            state_version=1,
            action_type="SET_CGROUP",
            target_tenant_id="guest-01",
            expected_delta={},
            required_resources={},
            evidence_refs=[],
            reversibility=ReversibilityClass.REVERSIBLE,
            evidence_requirements=EvidenceRequirements(),
            vital_hp=4  # Violation of VITAL_MAX_HP == 6
        )
        verdict_hp = self.engine.evaluate_candidate(candidate_hp, {}, {})
        self.assertTrue(verdict_hp.safety_verdict.is_veto)
        self.assertEqual(verdict_hp.safety_verdict.violation.tier, SafetyTier.SYSTEM_INVARIANTS)

    def test_11_precise_failure_lifecycle_actions(self):
        """Verifies distinct lifecycle operations for PRE-ACTUATION, POST-PREPARE, and POST-COMMIT."""
        # 1. Pre-actuation veto: REJECT_NOOP
        dummy_verdict = self.engine.evaluate_candidate(
            candidate=CandidateAction("C1", 999, "A", "T", {}, {}, [], ReversibilityClass.REVERSIBLE, EvidenceRequirements()),
            envelopes={},
            host_telemetry={}
        )
        pre_res = self.engine.handle_pre_actuation_veto(dummy_verdict)
        self.assertEqual(pre_res["status"], "REJECT_NOOP")
        self.assertFalse(pre_res["compensation_required"])
        self.assertFalse(pre_res["os_mutated"])

        # 2. Post-prepare failure: ABORT_STAGED_STATE
        prep_res = self.engine.handle_post_prepare_failure(staged_descriptor_id="STAGED-4082")
        self.assertEqual(prep_res["status"], "ABORT_STAGED_STATE")
        self.assertTrue(prep_res["memory_released"])

        # 3. Post-commit divergence: COMPENSATE
        applied = CandidateAction("C2", 1, "RESIZE_BALLOON", "T", {}, {}, [], ReversibilityClass.COMPENSABLE, EvidenceRequirements())
        commit_res = self.engine.handle_post_commit_divergence(applied)
        self.assertEqual(commit_res["status"], "COMPENSATE")
        self.assertTrue(commit_res["compensation_executed"])


class TestContextualRiskAndAntigravityGenerator(unittest.TestCase):
    """Tests Contextual Risk Learning and Antigravity Candidate Generator."""

    def test_12_contextual_risk_learning_not_global_blacklisting(self):
        """Veto in high pressure penalizes specific context tuple, but leaves low pressure clean."""
        registry = ContextualRiskRegistry()
        engine = AuthoritativeDecisionEngine(risk_registry=registry)

        candidate = CandidateAction(
            action_id="CAND-01",
            state_version=1,
            action_type="RESIZE_BALLOON",
            target_tenant_id="db-vm",
            expected_delta={},
            required_resources={"ram_mb": 1024.0},
            evidence_refs=[],
            reversibility=ReversibilityClass.COMPENSABLE,
            evidence_requirements=EvidenceRequirements(),
            vital_hp=VITAL_MAX_HP
        )

        # Trigger thermal veto in HIGH pressure context
        high_press_host = {
            "temperature_celsius": 92.0,  # Veto
            "host_state_class": "OVERHEATED",
            "workload_class": "BATCH",
            "pressure_region": "HIGH_THERMAL",
            "policy_version": "v1.0.0"
        }
        verdict = engine.evaluate_candidate(candidate, {}, high_press_host)
        self.assertTrue(verdict.safety_verdict.is_veto)

        action_sig = candidate.action_signature()

        # Contextual risk is elevated for the overheated context
        risk_high = registry.get_contextual_risk(
            action_signature=action_sig,
            host_state_class="OVERHEATED",
            workload_class="BATCH",
            pressure_region="HIGH_THERMAL",
            policy_version="v1.0.0"
        )
        self.assertGreater(risk_high, 0.0)

        # But contextual risk remains 0.0 for nominal context (NO global eternal blacklisting!)
        risk_nominal = registry.get_contextual_risk(
            action_signature=action_sig,
            host_state_class="NOMINAL",
            workload_class="BATCH",
            pressure_region="LOW_PRESSURE",
            policy_version="v1.0.0"
        )
        self.assertEqual(risk_nominal, 0.0)

    def test_13_antigravity_candidate_generator_formulation(self):
        """Antigravity generates typed candidate action with evidence requirements and falsification test."""
        generator = AntigravityCandidateGenerator()
        adapter = KvmQemuAdapter()
        env = adapter.sample_telemetry("tenant-web-01")

        candidate = generator.generate_candidate(
            tenant_id="tenant-web-01",
            action_type="RESIZE_BALLOON",
            current_state_version=1,
            desired_effect="Reclaim unused balloon memory",
            resource_request={"ram_mb": 512.0},
            active_envelopes=[env],
            context={"host_state_class": "NOMINAL", "workload_class": "WEB", "pressure_region": "LOW"}
        )

        self.assertIsInstance(candidate, CandidateAction)
        self.assertEqual(candidate.target_tenant_id, "tenant-web-01")
        self.assertEqual(candidate.state_version, 1)
        self.assertEqual(candidate.reversibility, ReversibilityClass.COMPENSABLE)

        # Verify Evidence Requirements are attached
        self.assertIsNotNone(candidate.evidence_requirements)
        self.assertIn("cpu_usage_pct", candidate.evidence_requirements.mandatory_dimensions)

        # Verify Falsification Experiment is attached
        self.assertIsNotNone(candidate.falsification_experiment)
        self.assertIn("HYP-", candidate.falsification_experiment.hypothesis_id)
        self.assertEqual(candidate.falsification_experiment.metric_key, "guest_swap_bytes")
        self.assertEqual(candidate.falsification_experiment.calibration_tag, "[CALIBRATION_HYPOTHESIS]")


if __name__ == "__main__":
    unittest.main()
