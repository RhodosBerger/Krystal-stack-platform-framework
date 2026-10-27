#!/usr/bin/env python3
"""
================================================================================
UNIT TESTS: SOVEREIGN BEHAVIORAL SCHEMA COMPILER & VULKAN UMA SYNTHESIS
================================================================================
Module:     tests/test_sovereign_behavioral_compiler.py
Author:     Antigravity Cognitive Invariants Group (2026)
Invariants: VITAL_MAX_HP == 6 | Proposer != Evaluator | Delta S == 0 on breach
================================================================================
"""

import unittest
from krystal_kernel.constants import VITAL_MAX_HP, ContractViolation
from krystal_kernel.sovereign_behavioral_compiler import (
    CeilingDomain,
    SublevelDomainId,
    ResourceSwapKind,
    CanonicalPhaseId,
    PotentialRangeMetrics,
    BehavioralSchema,
    PhaseConfirmationResult,
    SublevelDomainPacket,
    EconomicCostQuote,
    UmaRelativeCorrelation,
    BehavioralValidatorEngine,
    VectorPhaseRagEngine,
    ComputationalDifferenceServo,
    SublevelDomainRouter,
    HypervisorEconomicResourceGovernor,
    VulkanUmaRuntimePredictor,
    SovereignBehavioralSchemaCompiler,
    GLOBAL_BEHAVIORAL_SCHEMA_COMPILER,
)


class TestBehavioralValidatorEngine(unittest.TestCase):
    """Verifies Potential Range matrix evaluation across 7 ceiling domains."""

    def setUp(self) -> None:
        self.validator = BehavioralValidatorEngine(vital_max_hp=VITAL_MAX_HP)

    def test_thermal_ceiling_classification(self) -> None:
        """High temperature should trigger THERMAL_POWER_BOUND classification."""
        metrics = self.validator.evaluate_potential_range(
            observed_ops=850.0,
            theoretical_ceiling_ops=1000.0,
            uma_pressure=0.40,
            temperature_c=74.0  # Close to 75°C
        )
        self.assertEqual(metrics.ceiling_domain, CeilingDomain.THERMAL_POWER_BOUND)
        self.assertEqual(metrics.efficiency_ratio, 0.85)

    def test_memory_bound_classification(self) -> None:
        """UMA pressure >= 0.78 must trigger MEMORY_BOUND."""
        metrics = self.validator.evaluate_potential_range(
            observed_ops=600.0,
            theoretical_ceiling_ops=1000.0,
            uma_pressure=0.82,
            temperature_c=50.0
        )
        self.assertEqual(metrics.ceiling_domain, CeilingDomain.MEMORY_BOUND)

    def test_compute_bound_classification(self) -> None:
        """High efficiency (> 0.85) under nominal conditions triggers COMPUTE_BOUND."""
        metrics = self.validator.evaluate_potential_range(
            observed_ops=950.0,
            theoretical_ceiling_ops=1000.0,
            uma_pressure=0.30,
            temperature_c=45.0
        )
        self.assertEqual(metrics.ceiling_domain, CeilingDomain.COMPUTE_BOUND)
        self.assertEqual(metrics.efficiency_ratio, 0.95)

    def test_non_positive_ceiling_rejection(self) -> None:
        """Non-positive theoretical ceiling must raise ContractViolation."""
        with self.assertRaises(ContractViolation):
            self.validator.evaluate_potential_range(100.0, 0.0, 0.2, 40.0)


class TestVectorPhaseRagEngine(unittest.TestCase):
    """Verifies vector phase space matching and resource swap confirmation."""

    def setUp(self) -> None:
        self.rag = VectorPhaseRagEngine(vital_max_hp=VITAL_MAX_HP)

    def test_confirmed_canonical_practice_retrieval(self) -> None:
        """Vector close to canonical practice yields confirmation."""
        # Candidate vector nearly identical to CONFIRMED_VULKAN_STAGING_SHADOW
        cand_vec = [0.11, 0.24, 0.16, 0.89, 0.06, 0.11, 0.81, 0.94]
        res = self.rag.evaluate_resource_swap(
            swap_kind=ResourceSwapKind.VULKAN_STAGING_TO_SPSC,
            candidate_phase_vector=cand_vec
        )
        self.assertTrue(res.is_confirmed)
        self.assertGreaterEqual(res.phase_concurrence, 0.85)
        self.assertEqual(res.retrieved_practice_id, "CONFIRMED_VULKAN_STAGING_SHADOW")
        self.assertTrue(bool(res.hmac_digest))

    def test_divergent_practice_rejection(self) -> None:
        """Orthogonal/divergent phase vector is rejected with is_confirmed == False."""
        cand_vec = [-0.9, -0.9, -0.9, 0.0, 0.0, 0.0, -0.5, -0.5]
        res = self.rag.evaluate_resource_swap(
            swap_kind=ResourceSwapKind.VULKAN_STAGING_TO_SPSC,
            candidate_phase_vector=cand_vec
        )
        self.assertFalse(res.is_confirmed)
        self.assertLess(res.phase_concurrence, 0.85)


class TestComputationalDifferenceServo(unittest.TestCase):
    """Verifies PID difference servo regulation and anti-windup."""

    def setUp(self) -> None:
        self.servo = ComputationalDifferenceServo(vital_max_hp=VITAL_MAX_HP)

    def test_servo_adjustment_positive_error(self) -> None:
        """When observed is below target, servo emits positive pacing boost."""
        adj = self.servo.calculate_servo_adjustment(target_value=1000.0, observed_value=800.0, dt_s=0.016)
        self.assertGreater(adj, 0.0)

    def test_servo_anti_windup_clamping(self) -> None:
        """Large continuous errors must clamp integral term."""
        for _ in range(50):
            self.servo.calculate_servo_adjustment(target_value=2000.0, observed_value=100.0, dt_s=0.05)
        self.assertLessEqual(abs(self.servo.integral_error), 10.0)

    def test_servo_reset(self) -> None:
        """reset_servo sets integral and derivative states back to 0."""
        self.servo.calculate_servo_adjustment(100.0, 50.0)
        self.servo.reset_servo()
        self.assertEqual(self.servo.integral_error, 0.0)
        self.assertEqual(self.servo.last_error, 0.0)


class TestSublevelDomainRouter(unittest.TestCase):
    """Verifies inter-domain packet dispatch and HMAC signatures."""

    def setUp(self) -> None:
        self.router = SublevelDomainRouter(vital_max_hp=VITAL_MAX_HP)

    def test_packet_dispatch(self) -> None:
        """Packets between sublevels are logged with HMAC-SHA256 signatures."""
        pkt = self.router.dispatch_packet(
            source=SublevelDomainId.L2_FPGA_MAILBOX,
            target=SublevelDomainId.L3_DMA_SHM_RINGS,
            opcode=0x4B,
            payload_u64=0x12345678
        )
        self.assertEqual(len(self.router.packet_journal), 1)
        self.assertEqual(pkt.opcode, 0x4B)
        self.assertEqual(len(pkt.hmac_signature), 64)


class TestHypervisorEconomicResourceGovernor(unittest.TestCase):
    """Verifies economic pricing quotes and account debits."""

    def setUp(self) -> None:
        self.gov = HypervisorEconomicResourceGovernor(initial_balance=50.0, vital_max_hp=VITAL_MAX_HP)

    def test_quote_and_charge_success(self) -> None:
        """Affordable operations debit balance and return is_affordable=True."""
        quote = self.gov.quote_and_charge(
            operation_id="OP-ECON-01",
            uma_bytes=1024 * 1024 * 10,  # 10 MB
            energy_uj=50.0,
            cycle_ns=100_000             # 100 us
        )
        self.assertTrue(quote.is_affordable)
        self.assertGreater(quote.cost_micro_credits, 0.0)
        self.assertLess(quote.account_balance, 50.0)

    def test_quote_and_charge_unaffordable(self) -> None:
        """Expensive operation exceeding balance is rejected without debiting."""
        quote = self.gov.quote_and_charge(
            operation_id="OP-EXPENSIVE",
            uma_bytes=1024 * 1024 * 1024 * 100,  # 100 GB
            energy_uj=1_000_000.0,
            cycle_ns=1_000_000_000
        )
        self.assertFalse(quote.is_affordable)
        self.assertEqual(quote.account_balance, 50.0)


class TestVulkanUmaRuntimePredictor(unittest.TestCase):
    """Verifies Vulkan fetch pressure prediction and UMA relative synthesis."""

    def setUp(self) -> None:
        self.predictor = VulkanUmaRuntimePredictor(vital_max_hp=VITAL_MAX_HP)

    def test_fetch_pressure_prediction_with_swapchain(self) -> None:
        """Swapchain presentation adds significant scanout UMA pressure."""
        p_no_swap = self.predictor.predict_vulkan_fetch_pressure(descriptor_count=10, staging_buffer_mb=2.0, is_swapchain_presenting=False)
        p_swap = self.predictor.predict_vulkan_fetch_pressure(descriptor_count=10, staging_buffer_mb=2.0, is_swapchain_presenting=True)
        self.assertGreater(p_swap, p_no_swap)
        self.assertGreaterEqual(p_swap, 0.35)

    def test_uma_relatives_correlation_synthesis(self) -> None:
        """Co-access patterns generate sorted UMA relative correlation entries."""
        for _ in range(5):
            self.predictor.record_co_access("VULKAN_USM_HEAP", "DMA_SPSC_RING")
        self.predictor.record_co_access("VULKAN_USM_HEAP", "LLM_KV_CACHE")

        relatives = self.predictor.compute_uma_relatives()
        self.assertEqual(len(relatives), 2)
        top = relatives[0]
        self.assertEqual(top.region_a, "DMA_SPSC_RING")
        self.assertEqual(top.region_b, "VULKAN_USM_HEAP")
        self.assertGreater(top.contention_risk, 0.0)


class TestSovereignBehavioralSchemaCompiler(unittest.TestCase):
    """Verifies full end-to-end schema compilation and authorization."""

    def setUp(self) -> None:
        self.compiler = SovereignBehavioralSchemaCompiler(vital_max_hp=VITAL_MAX_HP)

    def test_end_to_end_compilation_and_authorization(self) -> None:
        """Complete pipeline executes and authorizes safe schema."""
        schema = BehavioralSchema(
            schema_id="SCHEMA-01",
            title="Dispatch zero-copy Vulkan compute stage",
            target_sublevel=SublevelDomainId.L5_KSL_CORE_RUNTIME,
            pre_condition_phase=[0.10, 0.25, 0.15, 0.90, 0.05, 0.10, 0.80, 0.95],
            max_cycle_quota_ns=2_000_000,
            max_energy_budget_uj=45.0,
            max_uma_bytes=1024 * 1024 * 2,
            is_reversible=True
        )

        result = self.compiler.compile_and_authorize_schema(
            schema=schema,
            swap_kind=ResourceSwapKind.VULKAN_STAGING_TO_SPSC,
            observed_ops=920.0,
            theoretical_ceiling_ops=1000.0,
            current_uma=0.35,
            current_temp_c=48.0
        )

        self.assertTrue(result["is_authorized"])
        self.assertEqual(result["canonical_phase"], CanonicalPhaseId.CANONICAL_PHASE_04_VULKAN_UMA_SYNTHESIS.value)
        self.assertTrue(result["rag_confirmation"].is_confirmed)
        self.assertTrue(result["economic_quote"].is_affordable)
        self.assertGreater(result["compilation_latency_us"], 0.0)
        self.assertEqual(result["vital_max_hp"], VITAL_MAX_HP)

    def test_rejection_on_uma_bus_saturation(self) -> None:
        """High UMA bus pressure (>= 0.78) causes schema rejection."""
        schema = BehavioralSchema(
            schema_id="SCHEMA-HIGH-UMA",
            title="Saturating pipeline",
            target_sublevel=SublevelDomainId.L5_KSL_CORE_RUNTIME,
            pre_condition_phase=[0.10, 0.25, 0.15, 0.90, 0.05, 0.10, 0.80, 0.95],
            max_cycle_quota_ns=2_000_000,
            max_energy_budget_uj=45.0,
            max_uma_bytes=1024 * 1024 * 2,
            is_reversible=True
        )

        result = self.compiler.compile_and_authorize_schema(
            schema=schema,
            swap_kind=ResourceSwapKind.VULKAN_STAGING_TO_SPSC,
            observed_ops=500.0,
            theoretical_ceiling_ops=1000.0,
            current_uma=0.85,  # Exceeds 0.78
            current_temp_c=48.0
        )
        self.assertFalse(result["is_authorized"])


if __name__ == "__main__":
    unittest.main()
