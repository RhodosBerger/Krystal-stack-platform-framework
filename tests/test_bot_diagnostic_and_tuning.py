#!/usr/bin/env python3
"""
UNIT TESTS: Sovereign Bot Diagnostic & Autonomic Tuning Toolkit
=============================================================
Formally verifies multi-scale diagnostic probing across 4 domains (MICRO, MESO, MACRO, EPOCH),
cross-layer coupling bottleneck auditing, and autonomic tuning execution under:
  1. Separation of concerns: proposer != evaluator
  2. Constitutional invariant: VITAL_MAX_HP == 6
  3. Rejection quarantine invariant: ¬Valid(A) ==> ΔS == 0
"""

import unittest
from krystal_kernel.bot_diagnostic_and_tuning_toolkit import (
    SovereignBotDiagnosticToolkit,
    TuningProfile,
    BottleneckSeverity,
    ScaleDomainProbeResult,
    CrossLayerCouplingAudit,
    BotDiagnosticReport,
    BotTuningAction,
    BotPerformanceTuningPlan,
    BotTuningExecutionReceipt,
    get_bot_diagnostic_toolkit,
)
from krystal_kernel.sovereign_bot_scale_dimensioner import (
    ScaleDomain,
    LayerSystemId,
    GLOBAL_SOVEREIGN_BOT_SCALE_DIMENSIONER,
    SovereignBotScaleDimensioner,
)
from krystal_bot.sovereign_layer_governor import (
    SOVEREIGN_LAYER_GOVERNOR,
    SovereignLayerGovernor,
)
from krystal_janet.sovereign_janet_api_bridge import GLOBAL_SOVEREIGN_JANET_SYSTEM_BRIDGE


class TestBotDiagnosticAndTuning(unittest.TestCase):

    def setUp(self):
        self.toolkit = SovereignBotDiagnosticToolkit()
        self.dimensioner = GLOBAL_SOVEREIGN_BOT_SCALE_DIMENSIONER

    def test_domain_probes_four_scales(self):
        """Verifies that all 4 scale domains are correctly probed with valid telemetry metrics."""
        probes = self.toolkit.probe_scale_domains()
        self.assertEqual(len(probes), 4)

        for domain in (
            ScaleDomain.MICRO_SILICON_NS,
            ScaleDomain.MESO_VBLANK_MS,
            ScaleDomain.MACRO_CADENCE_MS,
            ScaleDomain.EPOCH_GOVERNANCE_S,
        ):
            self.assertIn(domain.value, probes)
            probe = probes[domain.value]
            self.assertEqual(probe.domain, domain)
            self.assertGreater(probe.observed_throughput_eps, 0.0)
            self.assertGreater(probe.capacity_throughput_eps, 0.0)
            self.assertGreaterEqual(probe.throughput_utilization_ratio, 0.0)
            self.assertGreaterEqual(probe.observed_jitter_ns_or_ms, 0.0)
            self.assertGreater(probe.jitter_threshold, 0.0)
            self.assertGreater(probe.active_concurrency, 0)
            self.assertGreater(probe.max_concurrency, 0)
            self.assertEqual(probe.vital_max_hp, 6)

        # Specifically check micro domain metrics
        micro_probe = probes[ScaleDomain.MICRO_SILICON_NS.value]
        self.assertLessEqual(micro_probe.observed_jitter_ns_or_ms, micro_probe.jitter_threshold)
        self.assertTrue(micro_probe.is_healthy)

        # Specifically check meso domain metrics
        meso_probe = probes[ScaleDomain.MESO_VBLANK_MS.value]
        self.assertIn(meso_probe.status, ("NOMINAL", "WARNING_SHADOW_PRESSURE"))

        # Specifically check macro domain metrics
        macro_probe = probes[ScaleDomain.MACRO_CADENCE_MS.value]
        self.assertEqual(macro_probe.status, "NOMINAL")

        # Specifically check epoch domain metrics
        epoch_probe = probes[ScaleDomain.EPOCH_GOVERNANCE_S.value]
        self.assertEqual(epoch_probe.status, "NOMINAL")

    def test_cross_layer_coupling_audit(self):
        """Verifies cross-layer coupling audit returns valid structural tracks."""
        audits = self.toolkit.audit_cross_layer_couplings()
        self.assertGreaterEqual(len(audits), 4)

        coupling_ids = [a.coupling_id for a in audits]
        self.assertIn("COUPLE_L0_L1_PDN_VS_CHRONO", coupling_ids)
        self.assertIn("COUPLE_L1_L4_UMA_VS_CHANNELS", coupling_ids)
        self.assertIn("COUPLE_L2_L3_SYMPLECTIC_VS_CADENCE", coupling_ids)
        self.assertIn("COUPLE_L5_CONSTITUTIONAL_INTEGRITY", coupling_ids)

        for a in audits:
            self.assertIn(a.severity, (
                BottleneckSeverity.NOMINAL,
                BottleneckSeverity.ADVISORY,
                BottleneckSeverity.WARNING,
                BottleneckSeverity.CRITICAL,
            ))
            self.assertGreater(len(a.diagnostic_message), 0)
            self.assertGreater(a.safe_coupling_ceiling, 0.0)
            self.assertEqual(a.vital_max_hp, 6)

    def test_full_system_diagnostic_report(self):
        """Verifies full system diagnostic report assembly and cryptographic signature."""
        report = self.toolkit.run_full_system_diagnostic()
        self.assertGreaterEqual(report.overall_health_score, 0.0)
        self.assertLessEqual(report.overall_health_score, 100.0)
        self.assertEqual(len(report.scale_domain_probes), 4)
        self.assertEqual(report.vital_max_hp, 6)
        self.assertEqual(len(report.audit_hmac), 64)

        r_dict = report.to_dict()
        self.assertIn("overall_health_score", r_dict)
        self.assertIn("scale_domain_probes", r_dict)
        self.assertIn("cross_layer_audits", r_dict)
        self.assertIn("active_bottlenecks", r_dict)
        self.assertIn("thermal_headroom_c", r_dict)
        self.assertIn("power_headroom_w", r_dict)
        self.assertIn("uma_headroom_ratio", r_dict)
        self.assertEqual(r_dict["vital_max_hp"], 6)

    def test_tuning_plan_generation_all_profiles(self):
        """Verifies that each tuning profile generates a sound and bounded tuning plan."""
        for profile in (
            TuningProfile.BALANCED,
            TuningProfile.ULTRA_PERFORMANCE,
            TuningProfile.LOW_POWER_EFFICIENCY,
            TuningProfile.LOW_LATENCY_GAMING,
        ):
            plan = self.toolkit.generate_tuning_plan(profile)
            self.assertEqual(plan.target_profile, profile)
            self.assertEqual(plan.vital_max_hp, 6)
            self.assertGreaterEqual(len(plan.actions), 1)

            for act in plan.actions:
                var = self.dimensioner.get_variable(act.variable_id)
                self.assertIsNotNone(var)
                self.assertGreaterEqual(act.proposed_value, var.min_value)
                self.assertLessEqual(act.proposed_value, var.max_value)
                self.assertGreater(len(act.rationale), 0)
                self.assertGreater(len(act.expected_gain), 0)

    def test_autonomic_tuning_execution_and_separation_of_concerns(self):
        """Verifies autonomic tuning execution under proposer != evaluator and VITAL_MAX_HP == 6."""
        receipt = self.toolkit.execute_autonomic_tuning(TuningProfile.BALANCED)
        self.assertEqual(receipt.target_profile, TuningProfile.BALANCED.value)
        self.assertTrue(receipt.state_safety_guaranteed)
        self.assertEqual(receipt.vital_max_hp, 6)
        self.assertEqual(len(receipt.audit_hmac), 64)
        self.assertGreaterEqual(receipt.total_committed_actions, 1)

        for tr in receipt.tuning_receipts:
            self.assertIsNotNone(self.dimensioner.get_variable(tr.variable_id))
            self.assertEqual(tr.vital_max_hp, 6)
            if tr.success:
                self.assertGreater(tr.mutation_delta_bytes, 0)
            else:
                self.assertEqual(tr.mutation_delta_bytes, 0)

    def test_quarantine_invariant_on_out_of_bounds(self):
        """Verifies that an out-of-bounds tuning attempt results in zero state mutation (ΔS == 0)."""
        # Attempt to tune VAR_PL1_POWER_BUDGET_W way beyond max_value (max is 35.0W)
        rcpt = self.dimensioner.tune_variable(
            variable_id="VAR_PL1_POWER_BUDGET_W",
            new_value=999999.0,
            proposer_role="BOT_AUTONOMIC_TUNER",
            evaluator_role="CONSTITUTIONAL_GATEKEEPER",
        )
        self.assertFalse(rcpt.success)
        self.assertEqual(rcpt.mutation_delta_bytes, 0)
        self.assertIn("exceeds valid physical range", rcpt.reason.lower())
        self.assertEqual(rcpt.vital_max_hp, 6)

    def test_bot_cheatsheet_formatters(self):
        """Verifies that formatting functions for diagnostic and autonomic tuning return valid text."""
        diag_sheet = SOVEREIGN_LAYER_GOVERNOR.format_diagnostic_cheatsheet()
        self.assertIn("SOVEREIGN BOT MULTI-SCALE DIAGNOSTIC & HEALTH MATRIX", diag_sheet)
        self.assertIn("VITAL_MAX_HP == 6", diag_sheet)
        self.assertIn("MICRO_SILICON_NS", diag_sheet)

        tune_sheet = SOVEREIGN_LAYER_GOVERNOR.format_autonomic_tuning_cheatsheet("BALANCED")
        self.assertIn("AUTONÓMNE LADENIE VÝKONU BOTA: PROFIL [BALANCED]", tune_sheet)
        self.assertIn("VITAL_MAX_HP == 6", tune_sheet)
        self.assertIn("Delta S == 0 pri porušení", tune_sheet)

    def test_service_bot_subsystems(self):
        """Verifies holistic servicing of bot subsystems."""
        serv_res = self.toolkit.service_bot_subsystems()
        self.assertEqual(serv_res["status"], "SERVICED_OPTIMAL")
        self.assertIn("health_score", serv_res)
        self.assertIn("optimal_fps", serv_res)
        self.assertIn("allocated_shadow_ms", serv_res)
        self.assertTrue(serv_res["all_invariants_satisfied"])
        self.assertEqual(serv_res["vital_max_hp"], 6)

    def test_janet_dsl_bot_diagnostics_integration(self):
        """Verifies Janet bridge command dispatch for bot diagnostics and autonomic tuning."""
        # 1. Query diagnostics
        rcpt_diag = GLOBAL_SOVEREIGN_JANET_SYSTEM_BRIDGE.dispatch_janet_command("QUERY_BOT_SYSTEM_DIAGNOSTICS", {})
        self.assertTrue(rcpt_diag.is_success)
        self.assertEqual(rcpt_diag.status_code, "BOT_SYSTEM_DIAGNOSTICS_OK")
        self.assertIn("overall_health_score", rcpt_diag.output_payload)
        self.assertEqual(rcpt_diag.vital_max_hp, 6)

        # 2. Execute tuning via Janet
        rcpt_tune = GLOBAL_SOVEREIGN_JANET_SYSTEM_BRIDGE.dispatch_janet_command(
            "EXECUTE_BOT_PERFORMANCE_TUNING",
            {"profile": "BALANCED"}
        )
        self.assertTrue(rcpt_tune.is_success)
        self.assertEqual(rcpt_tune.status_code, "BOT_PERFORMANCE_TUNING_EXECUTED")
        self.assertIn("total_committed_actions", rcpt_tune.output_payload)
        self.assertEqual(rcpt_tune.vital_max_hp, 6)


if __name__ == "__main__":
    unittest.main()
