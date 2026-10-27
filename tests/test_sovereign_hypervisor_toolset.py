#!/usr/bin/env python3
"""
================================================================================
UNIT TESTS: SOVEREIGN HYPERVISOR TOOLSET & CONTROL PLANE
================================================================================
Module:     tests/test_sovereign_hypervisor_toolset.py
Author:     Antigravity Cognitive Invariants Group (2026)
Invariants: VITAL_MAX_HP == 6 | Proposer != Evaluator | Delta S == 0 on breach
================================================================================
"""

import unittest

from krystal_kernel.constants import VITAL_MAX_HP, ContractViolation
from krystal_kernel.sovereign_subsystem_turn_coordinator import (
    OperationReversibilityClass,
    ArchitectureProfile,
)
from krystal_kernel.sovereign_uma_inspector_governor import (
    HexMemoryCellState,
    GovernorAdviceKind,
    HardwarePriorityLane,
)
from krystal_kernel.sovereign_hypervisor_toolset import (
    HypervisorGuestDomain,
    HypercallKind,
    HypercallSafetyVerdict,
    HypercallCandidate,
    HypercallTurnToken,
    HypercallExecutionReceipt,
    HypervisorDomainLockArbiter,
    HypervisorReversibilityCoordinator,
    HypervisorUmaHexGovernor,
    HypervisorTurnDiagnostics,
    HypervisorComponentAblationTool,
    SovereignHypervisorConsoleTool,
    GLOBAL_SOVEREIGN_HYPERVISOR_TOOL,
)


class TestSovereignHypervisorToolset(unittest.TestCase):
    """Formal unit test suite for the Sovereign Hypervisor Toolset."""

    def setUp(self) -> None:
        self.tool = SovereignHypervisorConsoleTool()

    def test_constitutional_invariant_and_health(self) -> None:
        """Constitutional invariant VITAL_MAX_HP == 6 must hold across all components."""
        health = self.tool.audit_hypervisor_health()
        self.assertTrue(health["constitutional_invariant_verified"])
        self.assertEqual(health["vital_max_hp"], VITAL_MAX_HP)
        self.assertEqual(health["status"], "HYPERVISOR_SOVEREIGN_NOMINAL")

        # Must reject invalid HP
        with self.assertRaises(ContractViolation):
            SovereignHypervisorConsoleTool(vital_max_hp=5)

    def test_authority_separation_gate(self) -> None:
        """Proposer == Evaluator must be rejected immediately."""
        candidate = HypercallCandidate(
            hypercall_id="HC-TEST-AUTH",
            guest_domain=HypervisorGuestDomain.DOMU_WSL_NETWORK,
            opcode=HypercallKind.TELEMETRY_BURST,
            proposer_id="AGENT_SAME",
            evaluator_id="AGENT_SAME",
            estimated_latency_us=100.0,
            estimated_energy_uj=2.0
        )
        ok, tok, verdict, reason = self.tool.request_and_grant_hypercall_turn(candidate)
        self.assertFalse(ok)
        self.assertIsNone(tok)
        self.assertEqual(verdict, HypercallSafetyVerdict.REJECTED_AUTHORITY_VIOLATION)

    def test_conflict_aware_domain_locks(self) -> None:
        """Disjoint domains execute concurrently; overlapping domains are rejected with lock conflict."""
        # Candidate 1: Dom0 acquires SPINDLE_MOTOR
        c1 = HypercallCandidate(
            hypercall_id="HC-DOM0-01",
            guest_domain=HypervisorGuestDomain.DOM0_CENTRAL_MACHINE,
            opcode=HypercallKind.DMA_BUFFER_MAP,
            proposer_id="PROP_DOM0",
            evaluator_id="EVAL_AUDIT",
            estimated_latency_us=150.0,
            estimated_energy_uj=5.0,
            resource_domains={"SPINDLE_MOTOR"}
        )
        ok1, tok1, verd1, _ = self.tool.request_and_grant_hypercall_turn(c1)
        self.assertTrue(ok1)
        self.assertIsNotNone(tok1)

        # Candidate 2: DomU Vulkan acquires GPU_QUEUE_0 (disjoint domain -> must succeed)
        c2 = HypercallCandidate(
            hypercall_id="HC-VULKAN-01",
            guest_domain=HypervisorGuestDomain.DOMU_VULKAN_GRAPHICS,
            opcode=HypercallKind.SHARED_PAGE_GRANT,
            proposer_id="PROP_VK",
            evaluator_id="EVAL_AUDIT",
            estimated_latency_us=180.0,
            estimated_energy_uj=6.0,
            resource_domains={"GPU_QUEUE_0"}
        )
        ok2, tok2, verd2, _ = self.tool.request_and_grant_hypercall_turn(c2)
        self.assertTrue(ok2)
        self.assertIsNotNone(tok2)

        # Candidate 3: DomU WSL contends on SPINDLE_MOTOR (held by Dom0 -> must be rejected)
        c3 = HypercallCandidate(
            hypercall_id="HC-WSL-01",
            guest_domain=HypervisorGuestDomain.DOMU_WSL_NETWORK,
            opcode=HypercallKind.DMA_BUFFER_MAP,
            proposer_id="PROP_WSL",
            evaluator_id="EVAL_AUDIT",
            estimated_latency_us=120.0,
            estimated_energy_uj=3.0,
            resource_domains={"SPINDLE_MOTOR"}
        )
        ok3, tok3, verd3, reason3 = self.tool.request_and_grant_hypercall_turn(c3)
        self.assertFalse(ok3)
        self.assertIsNone(tok3)
        self.assertEqual(verd3, HypercallSafetyVerdict.REJECTED_DOMAIN_LOCK_CONFLICT)
        self.assertIn("SPINDLE_MOTOR", reason3)

        # Clean up active tokens
        self.tool.yield_hypercall_commit(tok1, 140.0)
        self.tool.yield_hypercall_commit(tok2, 170.0)

    def test_reversible_hypercall_lifecycle(self) -> None:
        """Verifies reversible hypercall commit produces Delta S == 1 and cryptographic HMAC receipt."""
        c = HypercallCandidate(
            hypercall_id="HC-REV-01",
            guest_domain=HypervisorGuestDomain.DOMU_TELEMETRY_HUB,
            opcode=HypercallKind.TELEMETRY_BURST,
            proposer_id="PROP_HUB",
            evaluator_id="EVAL_AUDIT",
            estimated_latency_us=200.0,
            estimated_energy_uj=4.0,
            priority=3,
            resource_domains={"TELEMETRY_BUS"}
        )
        ok, tok, _, _ = self.tool.request_and_grant_hypercall_turn(c)
        self.assertTrue(ok)
        self.assertIsNotNone(tok)

        receipt = self.tool.yield_hypercall_commit(tok, actual_latency_us=160.0)
        self.assertEqual(receipt.status, "COMMITTED")
        self.assertEqual(receipt.delta_s, 1)
        self.assertFalse(receipt.compensation_performed)
        self.assertGreater(len(receipt.hmac_digest), 32)
        self.assertEqual(receipt.vital_max_hp, VITAL_MAX_HP)

    def test_compensatable_hypercall_abort(self) -> None:
        """Compensatable hypercall abort must execute registered compensator with Delta S == 0."""
        compensated_calls = []

        def custom_compensator(payload):
            compensated_calls.append(payload.get("dma_buffer_id"))
            return True

        self.tool.reversibility_engine.register_compensator("CUSTOM_DMA_COMPENSATOR", custom_compensator)

        c = HypercallCandidate(
            hypercall_id="HC-COMP-01",
            guest_domain=HypervisorGuestDomain.DOM0_CENTRAL_MACHINE,
            opcode=HypercallKind.DMA_BUFFER_MAP,
            proposer_id="PROP_DMA",
            evaluator_id="EVAL_AUDIT",
            estimated_latency_us=250.0,
            estimated_energy_uj=8.0,
            resource_domains={"DMA_CH_0"},
            reversibility=OperationReversibilityClass.COMPENSATABLE,
            compensating_action="CUSTOM_DMA_COMPENSATOR",
            payload={"dma_buffer_id": "BUF_0xDEADBEEF"}
        )
        ok, tok, _, _ = self.tool.request_and_grant_hypercall_turn(c)
        self.assertTrue(ok)

        receipt = self.tool.abort_hypercall_rollback(tok, reason="CRC_CHECKSUM_ERROR")
        self.assertEqual(receipt.delta_s, 0)
        self.assertTrue(receipt.compensation_performed)
        self.assertIn("COMPENSATED", receipt.status)
        self.assertEqual(compensated_calls, ["BUF_0xDEADBEEF"])

    def test_irreversible_hypercall_safety_interlock(self) -> None:
        """Irreversible hypercalls without pre_commit_validated == True must be rejected."""
        # Without pre-commit validation -> REJECTED
        c_unsafe = HypercallCandidate(
            hypercall_id="HC-IRR-UNSAFE",
            guest_domain=HypervisorGuestDomain.DOM0_CENTRAL_MACHINE,
            opcode=HypercallKind.DEVICE_STROBE_PHYSICAL,
            proposer_id="PROP_CNC",
            evaluator_id="EVAL_AUDIT",
            estimated_latency_us=100.0,
            estimated_energy_uj=2.0,
            reversibility=OperationReversibilityClass.IRREVERSIBLE,
            pre_commit_validated=False
        )
        ok, tok, verd, reason = self.tool.request_and_grant_hypercall_turn(c_unsafe)
        self.assertFalse(ok)
        self.assertEqual(verd, HypercallSafetyVerdict.REJECTED_PRECOMMIT_VALIDATION_REQUIRED)

        # With pre-commit validation -> ACCEPTED
        c_safe = HypercallCandidate(
            hypercall_id="HC-IRR-SAFE",
            guest_domain=HypervisorGuestDomain.DOM0_CENTRAL_MACHINE,
            opcode=HypercallKind.DEVICE_STROBE_PHYSICAL,
            proposer_id="PROP_CNC",
            evaluator_id="EVAL_AUDIT",
            estimated_latency_us=100.0,
            estimated_energy_uj=2.0,
            reversibility=OperationReversibilityClass.IRREVERSIBLE,
            pre_commit_validated=True,
            resource_domains={"GPIO_STROBE_PIN"}
        )
        ok_s, tok_s, verd_s, _ = self.tool.request_and_grant_hypercall_turn(c_safe)
        self.assertTrue(ok_s)
        self.assertEqual(verd_s, HypercallSafetyVerdict.ACCEPTED_AND_DRAFTED)
        self.tool.yield_hypercall_commit(tok_s, 90.0)

    def test_uma_hex_memory_governor(self) -> None:
        """Tests paravirtualized UMA hex grid memory allocation and ASCII visual matrix."""
        # Allocate 8 cells to Dom0
        cells = self.tool.uma_governor.allocate_guest_cells(
            guest=HypervisorGuestDomain.DOM0_CENTRAL_MACHINE,
            cell_count=8,
            pressure=0.50
        )
        self.assertEqual(len(cells), 8)

        # Evaluate pressure
        press = self.tool.uma_governor.evaluate_guest_memory_pressure(HypervisorGuestDomain.DOM0_CENTRAL_MACHINE)
        self.assertEqual(press["total_cells"], 64)
        self.assertEqual(press["occupied_cells"], 8)
        self.assertAlmostEqual(press["occupancy_ratio"], 8 / 64, places=2)

        # Render ASCII hex matrix
        ascii_grid = self.tool.inspect_uma_hex_matrix()
        self.assertIn("HYPERVISOR UMA HEX MEMORY MATRIX", ascii_grid)
        self.assertIn("Row 00:", ascii_grid)

        # Free cells
        self.tool.uma_governor.free_guest_cells(HypervisorGuestDomain.DOM0_CENTRAL_MACHINE, cells)
        press_after = self.tool.uma_governor.evaluate_guest_memory_pressure(HypervisorGuestDomain.DOM0_CENTRAL_MACHINE)
        self.assertEqual(press_after["occupied_cells"], 0)

    def test_turn_utility_density_and_vute(self) -> None:
        """Verifies Verified Useful Time Efficiency (VUTE) and Turn Utility Density (TUD)."""
        c = HypercallCandidate(
            hypercall_id="HC-TUD-01",
            guest_domain=HypervisorGuestDomain.DOMU_LLM_COGNITION,
            opcode=HypercallKind.TELEMETRY_BURST,
            proposer_id="PROP_LLM",
            evaluator_id="EVAL_AUDIT",
            estimated_latency_us=400.0,
            estimated_energy_uj=15.0,
            priority=5,
            resource_domains={"LLM_CACHE_0"}
        )
        ok, tok, _, _ = self.tool.request_and_grant_hypercall_turn(c)
        self.assertTrue(ok)

        # Latency = 300us on allocated ~480us -> VUTE ~ 0.625, TUD = 5 * VUTE ~ 3.125
        self.tool.yield_hypercall_commit(tok, actual_latency_us=300.0)

        guest_info = self.tool.inspect_guest_domains()
        llm_metrics = guest_info["guest_domains"][HypervisorGuestDomain.DOMU_LLM_COGNITION.value]
        self.assertGreater(llm_metrics["average_turn_efficiency"], 0.0)
        self.assertGreater(llm_metrics["average_vute"], 0.0)
        self.assertGreater(llm_metrics["average_tud"], 0.0)
        self.assertEqual(llm_metrics["total_committed"], 1)

    def test_hypervisor_component_ablation_study(self) -> None:
        """Verifies 2x2 factorial hypervisor ablation decomposition."""
        report = self.tool.run_hypervisor_component_ablation(repetitions=3)

        self.assertTrue(report.constitutional_invariant_verified)
        self.assertIn("STRONG_PASS", report.verdict)
        self.assertEqual(report.epistemic_scope, "f(HypervisorDAG, GuestDomainTopology, QuantumPolicy, HypercallDistribution)")

        # Verify positive gains
        self.assertGreater(report.gain_quantum, 0.0)
        self.assertGreater(report.gain_parallelism, 0.0)
        self.assertGreater(report.gain_total, 0.0)

        # Mathematical decomposition check: Gain_total == Gain_quantum + Gain_parallelism + InteractionGain
        sum_gains = round(report.gain_quantum + report.gain_parallelism + report.gain_interaction, 4)
        self.assertAlmostEqual(report.gain_total, sum_gains, places=3)

        # Check virtualization hardware recommendations
        self.assertEqual(report.primary_gain_driver, "DOMAIN_PARALLELISM")
        self.assertIn("tier1_hardware_vt_domain_arbiter", report.virtualization_hardware_recommendations)
        self.assertIn("tier2_paravirt_countdown_timer", report.virtualization_hardware_recommendations)
        self.assertIn("tier3_interaction_synergy", report.virtualization_hardware_recommendations)


if __name__ == "__main__":
    unittest.main()
