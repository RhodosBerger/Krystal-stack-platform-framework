"""
tests/test_sovereign_bot_scale_dimensioner.py

Unit test suite verifying the Sovereign Bot Scale Dimensioner and Multi-Layer Governing Variables.
Strictly enforces VITAL_MAX_HP == 6, bounds constraints, and Delta S == 0 on unauthorized mutation.
"""

import unittest
from krystal_kernel.constants import VITAL_MAX_HP, ContractViolation
from krystal_kernel.sovereign_bot_scale_dimensioner import (
    ScaleDomain,
    LayerSystemId,
    ScaleProfile,
    LayerGoverningVariable,
    VariableTuningReceipt,
    OptimalOperatingReceipt,
    SovereignBotScaleDimensioner,
    GLOBAL_SOVEREIGN_BOT_SCALE_DIMENSIONER
)


class TestSovereignBotScaleDimensioner(unittest.TestCase):
    """Test suite for SovereignBotScaleDimensioner and multi-layer variable governance."""

    def setUp(self) -> None:
        self.dimensioner = SovereignBotScaleDimensioner()

    def test_scale_domains_and_capacity_envelopes(self) -> None:
        """Verifies dimensioning across the 4 hierarchical scale domains."""
        profiles = self.dimensioner.list_scale_profiles()
        self.assertEqual(len(profiles), 4)

        domains = {p.domain for p in profiles}
        expected_domains = {
            ScaleDomain.MICRO_SILICON_NS,
            ScaleDomain.MESO_VBLANK_MS,
            ScaleDomain.MACRO_CADENCE_MS,
            ScaleDomain.EPOCH_GOVERNANCE_S
        }
        self.assertEqual(domains, expected_domains)

        for p in profiles:
            self.assertEqual(p.vital_max_hp, VITAL_MAX_HP)
            self.assertGreater(p.max_concurrency, 0)
            self.assertGreater(p.throughput_events_per_sec, 0.0)
            self.assertGreater(p.max_latency_budget_ms, 0.0)

        # Micro scale: sub-50ns
        micro = self.dimensioner.get_scale_profile(ScaleDomain.MICRO_SILICON_NS)
        self.assertEqual(micro.max_concurrency, 1024)
        self.assertEqual(micro.max_latency_budget_ms, 0.00005)

    def test_governing_variables_registry_and_layer_coverage(self) -> None:
        """Verifies presence and valid bounds of all 28 governing variables across L0..L5."""
        variables = self.dimensioner.list_variables_by_layer()
        self.assertEqual(len(variables), 28)

        # Verify all 6 layers are represented
        layers_present = {v.layer_id for v in variables}
        self.assertEqual(len(layers_present), 6)

        for v in variables:
            self.assertEqual(v.vital_max_hp, VITAL_MAX_HP)
            self.assertLessEqual(v.min_value, v.nominal_value)
            self.assertLessEqual(v.nominal_value, v.max_value)
            self.assertLessEqual(v.min_value, v.current_value)
            self.assertLessEqual(v.current_value, v.max_value)
            self.assertTrue(len(v.governing_equation) > 0)

    def test_immutable_laws_cannot_be_mutated(self) -> None:
        """Verifies that constitutional immutable laws reject mutations with Delta S == 0."""
        # 1. Attempt to mutate VITAL_MAX_HP (6 -> 7)
        r1 = self.dimensioner.tune_variable(
            variable_id="VAR_VITAL_MAX_HP",
            new_value=7.0,
            proposer_role="EXPLORE_BOT",
            evaluator_role="GATEKEEPER"
        )
        self.assertFalse(r1.success)
        self.assertEqual(r1.committed_value, 6.0)
        self.assertEqual(r1.mutation_delta_bytes, 0)
        self.assertIn("constitutionally immutable", r1.reason)

        # 2. Attempt to disable separation of concerns
        r2 = self.dimensioner.tune_variable(
            variable_id="VAR_SEPARATION_OF_CONCERNS_ENFORCED",
            new_value=0.0,
            proposer_role="EXPLORE_BOT",
            evaluator_role="GATEKEEPER"
        )
        self.assertFalse(r2.success)
        self.assertEqual(r2.committed_value, 1.0)
        self.assertEqual(r2.mutation_delta_bytes, 0)

        # 3. Attempt to alter Delta S == 0 rule
        r3 = self.dimensioner.tune_variable(
            variable_id="VAR_DELTA_S_ON_BREACH",
            new_value=128.0,
            proposer_role="EXPLORE_BOT",
            evaluator_role="GATEKEEPER"
        )
        self.assertFalse(r3.success)
        self.assertEqual(r3.committed_value, 0.0)

    def test_separation_of_concerns_proposer_equals_evaluator(self) -> None:
        """Verifies rejection with Delta S == 0 if proposer attempts self-certification."""
        r = self.dimensioner.tune_variable(
            variable_id="VAR_ALPHA_CADENCE_PERIOD_MS",
            new_value=120.0,
            proposer_role="SAME_AGENT",
            evaluator_role="SAME_AGENT"
        )
        self.assertFalse(r.success)
        self.assertEqual(r.mutation_delta_bytes, 0)
        self.assertIn("forbidden self-certification", r.reason)

    def test_out_of_bounds_rejection(self) -> None:
        """Verifies that requested values exceeding physical boundaries are rejected."""
        # Minimum dwell frames is 12 (P8); try to set 5
        r = self.dimensioner.tune_variable(
            variable_id="VAR_DWELL_FRAME_LIMIT",
            new_value=5.0,
            proposer_role="OPERATOR",
            evaluator_role="GATEKEEPER"
        )
        self.assertFalse(r.success)
        self.assertEqual(r.mutation_delta_bytes, 0)
        self.assertIn("exceeds valid physical range", r.reason)

    def test_successful_variable_tuning(self) -> None:
        """Verifies successful modification of a mutable variable within valid bounds."""
        var = self.dimensioner.get_variable("VAR_ALPHA_CADENCE_PERIOD_MS")
        old_val = var.current_value
        new_val = 110.0

        r = self.dimensioner.tune_variable(
            variable_id="VAR_ALPHA_CADENCE_PERIOD_MS",
            new_value=new_val,
            proposer_role="OPERATOR",
            evaluator_role="GATEKEEPER"
        )
        self.assertTrue(r.success)
        self.assertEqual(r.committed_value, new_val)
        self.assertEqual(r.mutation_delta_bytes, 8)
        self.assertEqual(var.current_value, new_val)

        # Restore
        self.dimensioner.tune_variable("VAR_ALPHA_CADENCE_PERIOD_MS", old_val, "OP", "GATE")

    def test_cross_layer_constraints(self) -> None:
        """Verifies cross-layer constraint solver (e.g. VBlank shadow reduction)."""
        # Setting refresh rate to 144 Hz reduces frame time to 6.94 ms
        # 6.94 - 12.80 active scan - 0.366 guard band < 3.50 ms -> Must be rejected!
        r = self.dimensioner.tune_variable(
            variable_id="VAR_DISPLAY_REFRESH_RATE_HZ",
            new_value=144.0,
            proposer_role="OPERATOR",
            evaluator_role="GATEKEEPER"
        )
        self.assertFalse(r.success)
        self.assertEqual(r.mutation_delta_bytes, 0)
        self.assertIn("cross-layer", r.reason.lower())

    def test_solve_optimal_operating_point(self) -> None:
        """Verifies calculation of Pareto-optimal operating profile across layers."""
        rec = self.dimensioner.solve_optimal_operating_point(workload_intensity=0.6)
        self.assertEqual(rec.vital_max_hp, VITAL_MAX_HP)
        self.assertTrue(rec.all_invariants_satisfied)
        self.assertGreaterEqual(rec.allocated_shadow_ms, 3.50)
        self.assertLess(rec.uma_utilization, 0.78)
        self.assertEqual(rec.det_j_conservation, 1.0)
        self.assertTrue(len(rec.audit_hmac) == 64)
        self.assertEqual(rec.cross_layer_coordination_status, "OPTIMAL_COORDINATED")


if __name__ == "__main__":
    unittest.main()
