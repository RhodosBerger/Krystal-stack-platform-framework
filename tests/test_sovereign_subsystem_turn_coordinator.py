#!/usr/bin/env python3
"""
================================================================================
UNIT TESTS: SOVEREIGN SUBSYSTEM TURN COORDINATOR & AGILE DRAFTING INTERPOLATION
================================================================================
Module:     tests/test_sovereign_subsystem_turn_coordinator.py
Author:     Antigravity Cognitive Invariants Group (2026)
Invariants: VITAL_MAX_HP == 6 | Proposer != Evaluator | Delta S == 0 on breach
================================================================================
"""

import time
import unittest
from typing import Set

from krystal_kernel.constants import (
    VITAL_MAX_HP,
    UMA_SATURATION_THRESHOLD,
    ARRHENIUS_MAX_TEMP_C,
    CapabilityScope,
    ContractViolation,
)
from krystal_kernel.sovereign_runtime import SubsystemNodeId
from krystal_kernel.sovereign_subsystem_turn_coordinator import (
    TurnState,
    OperationSafetyVerdict,
    CandidateOperation,
    AgileDraftedEnvelope,
    SelectionEvaluationResult,
    TurnToken,
    InstructionReceipt,
    ProceduralAgileDraftingInterpolator,
    SafeOperationSelectionPack,
    InstructionalDependencyCoordinator,
    GLOBAL_SUBSYSTEM_TURN_COORDINATOR,
)


class TestProceduralAgileDraftingInterpolator(unittest.TestCase):
    """Verifies EMA smoothing, running variance, and agile procedural drafting."""

    def setUp(self) -> None:
        self.interpolator = ProceduralAgileDraftingInterpolator(vital_max_hp=VITAL_MAX_HP)

    def test_ema_and_variance_convergence(self) -> None:
        """Streaming samples should smoothly converge EMA and calculate variance."""
        metric = "CHRONO_SOC_latency_us"
        for _ in range(10):
            self.interpolator.update_telemetry_metric(metric, 100.0)

        ema = self.interpolator.get_ema(metric)
        self.assertAlmostEqual(ema, 100.0, delta=1.0)
        self.assertAlmostEqual(self.interpolator.get_std_dev(metric), 0.0, delta=1.0)

        # Introduce variance
        self.interpolator.update_telemetry_metric(metric, 150.0)
        self.assertGreater(self.interpolator.get_std_dev(metric), 0.0)

    def test_cubic_smoothstep_boundaries(self) -> None:
        """Cubic Hermite should be 0 at t=0, 1 at t=1, and monotonic in between."""
        self.assertEqual(self.interpolator.cubic_smoothstep(0.0), 0.0)
        self.assertEqual(self.interpolator.cubic_smoothstep(1.0), 1.0)
        self.assertEqual(self.interpolator.cubic_smoothstep(0.5), 0.5)
        # Clamping
        self.assertEqual(self.interpolator.cubic_smoothstep(-0.5), 0.0)
        self.assertEqual(self.interpolator.cubic_smoothstep(1.5), 1.0)

    def test_agile_quantum_shrinks_under_uma_pressure(self) -> None:
        """High UMA bus pressure should procedurally shrink the allocated quantum slice."""
        q_low_uma = self.interpolator.draft_quantum_slice_us(
            subsystem=SubsystemNodeId.CHRONO_SOC,
            priority=2,
            current_uma=0.10,
            base_quantum_us=1000.0
        )
        q_high_uma = self.interpolator.draft_quantum_slice_us(
            subsystem=SubsystemNodeId.CHRONO_SOC,
            priority=2,
            current_uma=0.75,  # Very close to 0.78 threshold
            base_quantum_us=1000.0
        )
        self.assertGreater(q_low_uma, q_high_uma)

    def test_thermal_headroom_drafting(self) -> None:
        """Thermal headroom should decrease as current temperature approaches Arrhenius limit."""
        headroom_cool = self.interpolator.draft_thermal_headroom_c(current_temp_c=35.0)
        headroom_warm = self.interpolator.draft_thermal_headroom_c(current_temp_c=70.0)
        self.assertGreater(headroom_cool, headroom_warm)
        self.assertGreaterEqual(headroom_warm, 0.0)


class TestSafeOperationSelectionPack(unittest.TestCase):
    """Verifies all safety gates in the selection pack."""

    def setUp(self) -> None:
        self.interpolator = ProceduralAgileDraftingInterpolator(vital_max_hp=VITAL_MAX_HP)
        self.selection_pack = SafeOperationSelectionPack(
            interpolator=self.interpolator,
            vital_max_hp=VITAL_MAX_HP
        )

    def _create_valid_candidate(self) -> CandidateOperation:
        return CandidateOperation(
            operation_id="OP-001",
            title="Dispatch microcode test slice",
            subsystem=SubsystemNodeId.CHRONO_SOC,
            proposer_id="PROPOSER_BOT_01",
            evaluator_id="VERIFIER_BOT_02",  # Separation of concerns
            required_capability=CapabilityScope.CAN_EXECUTE,
            estimated_latency_us=250.0,
            estimated_energy_uj=12.0,
            priority=2,
            depends_on=[],
            resource_domains={"CHRONO_MAILBOX"}
        )

    def test_successful_selection_and_envelope_generation(self) -> None:
        """A valid candidate operation should be authorized with an agile drafted envelope."""
        cand = self._create_valid_candidate()
        result = self.selection_pack.evaluate_candidate(
            candidate=cand,
            current_uma=0.35,
            current_temp_c=42.0,
            resolved_dependencies=set(),
            active_locked_domains=set()
        )
        self.assertTrue(result.is_authorized)
        self.assertEqual(result.verdict, OperationSafetyVerdict.ACCEPTED_AND_DRAFTED)
        self.assertIsNotNone(result.drafted_envelope)
        self.assertGreater(result.drafted_envelope.allocated_quantum_us, 0.0)
        self.assertTrue(bool(result.hmac_digest))

    def test_authority_separation_violation_rejection(self) -> None:
        """Proposer == Evaluator must be rejected immediately."""
        cand = self._create_valid_candidate()
        cand.evaluator_id = cand.proposer_id  # Violation
        result = self.selection_pack.evaluate_candidate(
            candidate=cand,
            current_uma=0.30,
            current_temp_c=40.0,
            resolved_dependencies=set(),
            active_locked_domains=set()
        )
        self.assertFalse(result.is_authorized)
        self.assertEqual(result.verdict, OperationSafetyVerdict.REJECTED_AUTHORITY_SEPARATION)

    def test_thermal_gate_rejection(self) -> None:
        """Temperature exceeding Arrhenius limit must be rejected."""
        cand = self._create_valid_candidate()
        result = self.selection_pack.evaluate_candidate(
            candidate=cand,
            current_uma=0.30,
            current_temp_c=78.5,  # Exceeds ARRHENIUS_MAX_TEMP_C (75.0)
            resolved_dependencies=set(),
            active_locked_domains=set()
        )
        self.assertFalse(result.is_authorized)
        self.assertEqual(result.verdict, OperationSafetyVerdict.REJECTED_THERMAL_HEADROOM)

    def test_bus_saturation_gate_rejection(self) -> None:
        """UMA bus pressure exceeding 0.78 must be rejected."""
        cand = self._create_valid_candidate()
        result = self.selection_pack.evaluate_candidate(
            candidate=cand,
            current_uma=0.82,  # Exceeds 0.78
            current_temp_c=40.0,
            resolved_dependencies=set(),
            active_locked_domains=set()
        )
        self.assertFalse(result.is_authorized)
        self.assertEqual(result.verdict, OperationSafetyVerdict.REJECTED_BUS_SATURATED)

    def test_unresolved_dependency_rejection(self) -> None:
        """Candidate with missing dependency receipts must be rejected."""
        cand = self._create_valid_candidate()
        cand.depends_on = ["DEP_PREDECESSOR_INSTR_01"]
        result = self.selection_pack.evaluate_candidate(
            candidate=cand,
            current_uma=0.30,
            current_temp_c=40.0,
            resolved_dependencies=set(),  # Missing dependency
            active_locked_domains=set()
        )
        self.assertFalse(result.is_authorized)
        self.assertEqual(result.verdict, OperationSafetyVerdict.REJECTED_DEPENDENCY_UNRESOLVED)

    def test_resource_domain_conflict_rejection(self) -> None:
        """Candidate requiring an already locked resource domain must be rejected."""
        cand = self._create_valid_candidate()
        cand.resource_domains = {"CHRONO_MAILBOX", "DMA_SPSC_RING"}
        active_locks = {"DMA_SPSC_RING"}  # Conflict
        result = self.selection_pack.evaluate_candidate(
            candidate=cand,
            current_uma=0.30,
            current_temp_c=40.0,
            resolved_dependencies=set(),
            active_locked_domains=active_locks
        )
        self.assertFalse(result.is_authorized)
        self.assertEqual(result.verdict, OperationSafetyVerdict.REJECTED_DOMAIN_CONFLICT)


class TestInstructionalDependencyCoordinator(unittest.TestCase):
    """Verifies subsystem turn logic, mutual exclusion, receipts, and rollback."""

    def setUp(self) -> None:
        self.coordinator = InstructionalDependencyCoordinator(vital_max_hp=VITAL_MAX_HP)

    def _make_candidate(
        self,
        op_id: str,
        subsystem: SubsystemNodeId,
        depends_on: list[str] = None
    ) -> CandidateOperation:
        return CandidateOperation(
            operation_id=op_id,
            title=f"Task {op_id}",
            subsystem=subsystem,
            proposer_id=f"PROP_{subsystem.value}",
            evaluator_id="INDEPENDENT_AUDITOR",
            required_capability=CapabilityScope.CAN_EXECUTE,
            estimated_latency_us=300.0,
            estimated_energy_uj=15.0,
            depends_on=depends_on or []
        )

    def test_turn_grant_and_mutual_exclusion(self) -> None:
        """Granting turn token sets bus master; concurrent grant is locked out."""
        cand1 = self._make_candidate("TASK-01", SubsystemNodeId.CHRONO_SOC)
        cand2 = self._make_candidate("TASK-02", SubsystemNodeId.GODOT_HOLOBRIDGE)

        # Grant turn 1
        ok1, token1, _ = self.coordinator.request_and_grant_turn(cand1)
        self.assertTrue(ok1)
        self.assertIsNotNone(token1)
        self.assertTrue(self.coordinator.is_bus_master_active())
        self.assertEqual(self.coordinator.get_subsystem_state(SubsystemNodeId.CHRONO_SOC), TurnState.TURN_GRANTED)

        # Attempt concurrent grant for candidate 2 -> Must be rejected
        ok2, token2, res2 = self.coordinator.request_and_grant_turn(cand2)
        self.assertFalse(ok2)
        self.assertIsNone(token2)
        self.assertEqual(res2.verdict, OperationSafetyVerdict.REJECTED_BUS_SATURATED)

    def test_full_turn_lifecycle_yield_commit(self) -> None:
        """Granted turn transitions to EXECUTING_SLICE and gracefully yields with Delta S == 1."""
        cand = self._make_candidate("TASK-YIELD", SubsystemNodeId.WSL_BRIDGE)
        ok, token, _ = self.coordinator.request_and_grant_turn(cand)
        self.assertTrue(ok)

        # Begin slice
        started = self.coordinator.begin_execution_slice(token)
        self.assertTrue(started)
        self.assertEqual(self.coordinator.get_subsystem_state(SubsystemNodeId.WSL_BRIDGE), TurnState.EXECUTING_SLICE)

        # Yield commit
        receipt = self.coordinator.yield_turn_commit(
            token=token,
            latency_us=210.0,
            energy_uj=9.5,
            output_payload_hash="e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        )
        self.assertEqual(receipt.delta_s, 1)
        self.assertEqual(receipt.status, "COMMITTED")
        self.assertFalse(self.coordinator.is_bus_master_active())
        self.assertEqual(self.coordinator.get_subsystem_state(SubsystemNodeId.WSL_BRIDGE), TurnState.TURN_YIELDED)
        self.assertIn("TASK-YIELD", self.coordinator.resolved_instruction_ids)

    def test_turn_abort_and_rollback_delta_s_zero(self) -> None:
        """Aborted turn must enforce Delta S == 0 and record rollback."""
        cand = self._make_candidate("TASK-FAIL", SubsystemNodeId.TERRAIN_MANIFOLD)
        ok, token, _ = self.coordinator.request_and_grant_turn(cand)
        self.assertTrue(ok)

        # Abort
        receipt = self.coordinator.abort_turn_rollback(token, reason="HARDWARE_PARITY_FAULT")
        self.assertEqual(receipt.delta_s, 0)
        self.assertIn("ROLLED_BACK", receipt.status)
        self.assertFalse(self.coordinator.is_bus_master_active())
        self.assertEqual(self.coordinator.get_subsystem_state(SubsystemNodeId.TERRAIN_MANIFOLD), TurnState.TURN_EXPIRED_ROLLBACK)
        self.assertEqual(self.coordinator.total_turns_rolled_back, 1)

    def test_instructional_dependency_sequencing(self) -> None:
        """Dependent task cannot get a turn until predecessor commits its receipt."""
        cand_pred = self._make_candidate("PRED-01", SubsystemNodeId.CHRONO_SOC)
        cand_dep = self._make_candidate("DEP-02", SubsystemNodeId.SLOVAK_LLM_CASCADE, depends_on=["PRED-01"])

        # Try to grant dependent before predecessor commits
        ok_dep, _, res_dep = self.coordinator.request_and_grant_turn(cand_dep)
        self.assertFalse(ok_dep)
        self.assertEqual(res_dep.verdict, OperationSafetyVerdict.REJECTED_DEPENDENCY_UNRESOLVED)
        self.assertEqual(self.coordinator.get_subsystem_state(SubsystemNodeId.SLOVAK_LLM_CASCADE), TurnState.AWAITING_DEPENDENCY)

        # Now execute and commit predecessor
        ok_p, tok_p, _ = self.coordinator.request_and_grant_turn(cand_pred)
        self.assertTrue(ok_p)
        self.coordinator.yield_turn_commit(tok_p, latency_us=150.0, energy_uj=5.0, output_payload_hash="abc")

        # Now dependent should be granted its turn!
        ok_dep2, tok_dep2, res_dep2 = self.coordinator.request_and_grant_turn(cand_dep)
        self.assertTrue(ok_dep2)
        self.assertIsNotNone(tok_dep2)
        self.assertEqual(res_dep2.verdict, OperationSafetyVerdict.ACCEPTED_AND_DRAFTED)

    def test_quantum_expiry_watchdog_enforcement(self) -> None:
        """Watchdog check should forcefully abort turn if quantum expired."""
        cand = self._make_candidate("TASK-TIMEOUT", SubsystemNodeId.VPP_GRID_BROKER)
        ok, token, _ = self.coordinator.request_and_grant_turn(cand)
        self.assertTrue(ok)

        # Simulate current time past expiration
        expired = self.coordinator.check_and_enforce_quantum_expiry(current_time_ns=token.expires_at_ns + 1000)
        self.assertTrue(expired)
        self.assertFalse(self.coordinator.is_bus_master_active())
        self.assertEqual(self.coordinator.get_subsystem_state(SubsystemNodeId.VPP_GRID_BROKER), TurnState.TURN_EXPIRED_ROLLBACK)


if __name__ == "__main__":
    unittest.main()
