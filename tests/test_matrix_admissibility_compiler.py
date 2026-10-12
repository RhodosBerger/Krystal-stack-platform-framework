#!/usr/bin/env python3
"""
UNIT TEST SUITE: DETERMINISTIC MATRIX ADMISSIBILITY & TRANSITION COMPILER
=============================================================================
Tests:
  1. State and action vectorization into TensorMatrix objects.
  2. M_safe * [s; a] <= c_limits evaluation and slack margin calculation.
  3. Row-specific constraint vetoes (thermal breach, headroom floor, vital invariant).
  4. State transition projection s_{k+1} = T * s_k + B * a.
  5. Evidence admissibility matrix evaluation (sufficient, insufficient, contradictory).
  6. Quadratic cross-workload interference penalty calculation.
  7. End-to-end CompiledRtpMatrixPlan synthesis.
  8. C-ABI / SIMD flat array export.
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
    ReversibilityClass,
    ResourceEvidenceEnvelope,
    DynamicSafetyPolicy,
    AuthorityVerdictType
)
from krystal_kernel.verification.matrix_admissibility_compiler import (
    MatrixAdmissibilityCompiler,
    CompiledRtpMatrixPlan
)


class TestMatrixAdmissibilityCompiler(unittest.TestCase):
    """Test suite for MatrixAdmissibilityCompiler."""

    def setUp(self):
        self.policy = DynamicSafetyPolicy()
        self.compiler = MatrixAdmissibilityCompiler(self.policy)

    def _create_candidate(
        self,
        vcpu: float = 1.0,
        ram_mb: float = 512.0,
        state_version: int = 1,
        vital_hp: int = VITAL_MAX_HP
    ) -> CandidateAction:
        return CandidateAction(
            action_id="CAND-MAT-01",
            state_version=state_version,
            action_type="RESIZE_BALLOON",
            target_tenant_id="tenant-vm-01",
            expected_delta={"cpu_pct": -5.0},
            required_resources={"vcpu": vcpu, "ram_mb": ram_mb, "io_weight": 100.0, "balloon_delta_mb": 0.0},
            evidence_refs=["ENV-REF-01"],
            reversibility=ReversibilityClass.COMPENSABLE,
            evidence_requirements=EvidenceRequirements(
                max_acceptable_age_seconds=2.0,
                mandatory_dimensions=["cpu_usage_pct", "memory_rss_bytes"]
            ),
            vital_hp=vital_hp
        )

    def _create_envelope(self, env_id: str = "ENV-REF-01", cpu: float = 45.0, age: float = 0.1) -> ResourceEvidenceEnvelope:
        return ResourceEvidenceEnvelope(
            envelope_id=env_id,
            timestamp=time.time() - age,
            guest_id="tenant-vm-01",
            telemetry={"cpu_usage_pct": cpu, "memory_rss_bytes": 1024 * 1024 * 1024}
        )

    def test_01_invariant_constant(self):
        self.assertEqual(VITAL_MAX_HP, 6)

    def test_02_vector_compilation(self):
        host_telemetry = {
            "cpu_usage_pct": 38.5,
            "free_ram_bytes": 4096 * 1024 * 1024,
            "temperature_celsius": 52.0,
            "package_power_watts": 18.0,
            "vital_hp": 6
        }
        s = self.compiler.compile_state_vector(host_telemetry, current_state_version=1)
        self.assertEqual(s.shape, (6, 1))
        self.assertEqual(s.data[0][0], 38.5)
        self.assertEqual(s.data[1][0], 4096.0)

        candidate = self._create_candidate(vcpu=2.0, ram_mb=1024.0)
        a = self.compiler.compile_action_vector(candidate)
        self.assertEqual(a.shape, (4, 1))
        self.assertEqual(a.data[0][0], 2.0)
        self.assertEqual(a.data[1][0], 1024.0)

    def test_03_safety_constraint_evaluation_nominal(self):
        candidate = self._create_candidate(vcpu=1.0, ram_mb=512.0)
        env = self._create_envelope()
        host_telemetry = {
            "cpu_usage_pct": 30.0,
            "free_ram_bytes": 8192 * 1024 * 1024,
            "temperature_celsius": 50.0,
            "package_power_watts": 15.0,
            "current_state_version": 1
        }

        plan = self.compiler.compile_and_verify(
            candidate=candidate,
            envelopes={env.envelope_id: env},
            host_telemetry=host_telemetry
        )

        self.assertTrue(plan.is_committable)
        self.assertEqual(plan.authority_verdict, AuthorityVerdictType.COMMIT_APPROVED)
        # All slack margins must be non-negative
        for name, margin in plan.slack_margins.items():
            self.assertGreaterEqual(margin, 0.0, f"Slack margin for {name} should be >= 0")

    def test_04_thermal_breach_veto_in_matrix(self):
        candidate = self._create_candidate(vcpu=1.0, ram_mb=512.0)
        env = self._create_envelope()
        # High temperature that breaches thermal ceiling in row 2 of M_safe
        host_telemetry = {
            "cpu_usage_pct": 30.0,
            "free_ram_bytes": 8192 * 1024 * 1024,
            "temperature_celsius": 88.0,  # Breaches limit
            "package_power_watts": 15.0,
            "current_state_version": 1
        }

        plan = self.compiler.compile_and_verify(
            candidate=candidate,
            envelopes={env.envelope_id: env},
            host_telemetry=host_telemetry
        )

        self.assertFalse(plan.is_committable)
        self.assertEqual(plan.authority_verdict, AuthorityVerdictType.VETO_SAFETY_BREACH)
        self.assertLess(plan.slack_margins["PHYSICAL_THERMAL_CEILING"], 0.0)

    def test_05_ram_headroom_breach_veto_in_matrix(self):
        # Requesting 7500MB when free RAM is 8000MB breaches 1024MB headroom floor
        candidate = self._create_candidate(vcpu=1.0, ram_mb=7500.0)
        env = self._create_envelope()
        host_telemetry = {
            "cpu_usage_pct": 30.0,
            "free_ram_bytes": 8000 * 1024 * 1024,
            "temperature_celsius": 50.0,
            "package_power_watts": 15.0,
            "current_state_version": 1
        }

        plan = self.compiler.compile_and_verify(
            candidate=candidate,
            envelopes={env.envelope_id: env},
            host_telemetry=host_telemetry
        )

        self.assertFalse(plan.is_committable)
        self.assertEqual(plan.authority_verdict, AuthorityVerdictType.VETO_SAFETY_BREACH)
        self.assertLess(plan.slack_margins["RAM_HEADROOM_FLOOR"], 0.0)

    def test_06_state_transition_projection(self):
        T, B = self.compiler.compile_transition_matrices()
        self.assertEqual(T.shape, (6, 6))
        self.assertEqual(B.shape, (6, 4))

        candidate = self._create_candidate(vcpu=1.0, ram_mb=512.0)
        env = self._create_envelope()
        host_telemetry = {
            "cpu_usage_pct": 30.0,
            "free_ram_bytes": 8192 * 1024 * 1024,
            "temperature_celsius": 50.0,
            "package_power_watts": 15.0,
            "current_state_version": 1
        }

        plan = self.compiler.compile_and_verify(
            candidate=candidate,
            envelopes={env.envelope_id: env},
            host_telemetry=host_telemetry
        )

        self.assertIn("cpu_pct", plan.predicted_next_state)
        self.assertIn("free_ram_mb", plan.predicted_next_state)
        self.assertIn("temp_c", plan.predicted_next_state)
        self.assertEqual(plan.predicted_next_state["vital_hp"], 6.0)

    def test_07_evidence_insufficient_deferral(self):
        candidate = self._create_candidate()
        # Stale envelope (age=5.0s > 2.0s)
        stale_env = self._create_envelope(age=5.0)
        host_telemetry = {
            "cpu_usage_pct": 30.0,
            "free_ram_bytes": 8192 * 1024 * 1024,
            "temperature_celsius": 50.0,
            "package_power_watts": 15.0,
            "current_state_version": 1
        }

        plan = self.compiler.compile_and_verify(
            candidate=candidate,
            envelopes={stale_env.envelope_id: stale_env},
            host_telemetry=host_telemetry
        )

        self.assertFalse(plan.is_committable)
        self.assertEqual(plan.authority_verdict, AuthorityVerdictType.DEFER_INSUFFICIENT_EVIDENCE)

    def test_08_evidence_contradiction_quarantine(self):
        candidate = self._create_candidate()
        candidate.evidence_refs = ["ENV-S1", "ENV-S2"]
        # Discrepant CPU (20% vs 80%, delta=60% > 35%)
        env1 = self._create_envelope(env_id="ENV-S1", cpu=20.0)
        env2 = self._create_envelope(env_id="ENV-S2", cpu=80.0)
        host_telemetry = {
            "cpu_usage_pct": 30.0,
            "free_ram_bytes": 8192 * 1024 * 1024,
            "temperature_celsius": 50.0,
            "package_power_watts": 15.0,
            "current_state_version": 1
        }

        plan = self.compiler.compile_and_verify(
            candidate=candidate,
            envelopes={env1.envelope_id: env1, env2.envelope_id: env2},
            host_telemetry=host_telemetry
        )

        self.assertFalse(plan.is_committable)
        self.assertEqual(plan.authority_verdict, AuthorityVerdictType.QUARANTINE_CONTRADICTORY_EVIDENCE)

    def test_09_cross_workload_interference_calculation(self):
        tenant_features = {"llc_intensity": 0.8, "bus_intensity": 0.6, "io_intensity": 0.4}
        active_neighbors = [
            {"llc_intensity": 0.5, "bus_intensity": 0.4, "io_intensity": 0.2},
            {"llc_intensity": 0.3, "bus_intensity": 0.3, "io_intensity": 0.1}
        ]
        penalty = self.compiler.calculate_interference_penalty(tenant_features, active_neighbors)
        self.assertGreater(penalty, 0.0)
        self.assertLessEqual(penalty, 1.0)

    def test_10_c_abi_simd_export(self):
        export = self.compiler.export_c_abi_matrices()
        self.assertIn("M_safe_flat", export)
        self.assertIn("c_limits_flat", export)
        self.assertIn("T_transition_flat", export)
        self.assertIn("B_actuation_flat", export)
        self.assertEqual(export["M_safe_rows"], 6)
        self.assertEqual(export["M_safe_cols"], 10)
        self.assertEqual(len(export["M_safe_flat"]), 60)
        self.assertEqual(len(export["c_limits_flat"]), 6)
        self.assertEqual(len(export["T_transition_flat"]), 36)
        self.assertEqual(len(export["B_actuation_flat"]), 24)
        self.assertEqual(export["simd_alignment_bytes"], 64)


if __name__ == "__main__":
    unittest.main()
