"""
tests/test_unified_dependency_coordinator.py

Authoritative Unit Tests for Unified Cross-Stream Dependency Coordinator
and Safe Real-Time Intervention Actuator.
========================================================================
Validates stream mapping, multi-coordinator coupled safety audits,
Two-Phase Actuation Pipeline (Separation of Concerns, physical limits),
Delta S == 0 on any breach, and holistic stream synchronization cycles.
"""

import unittest
from krystal_kernel.constants import VITAL_MAX_HP, ContractViolation
from krystal_kernel.unified_dependency_coordinator import (
    RuntimeStreamKind,
    KeyDependencyCoordinatorId,
    StreamElementMapping,
    InterventionProposal,
    RealtimeInterventionReceipt,
    UnifiedStreamSyncReceipt,
    SafeRealtimeInterventionActuator,
    UnifiedCrossStreamDependencyCoordinator,
    GLOBAL_UNIFIED_DEPENDENCY_COORDINATOR
)
from krystal_kernel.sovereign_bot_scale_dimensioner import GLOBAL_SOVEREIGN_BOT_SCALE_DIMENSIONER


class TestUnifiedDependencyCoordinator(unittest.TestCase):
    """Test suite verifying cross-stream dependency coordination and safe real-time actuation."""

    def setUp(self) -> None:
        self.coordinator = GLOBAL_UNIFIED_DEPENDENCY_COORDINATOR
        self.dimensioner = GLOBAL_SOVEREIGN_BOT_SCALE_DIMENSIONER

    def test_01_stream_element_mappings_count_and_invariants(self) -> None:
        """Verifies all 15 elements across 5 streams and 6 coordinators with VITAL_MAX_HP == 6."""
        elements = self.coordinator.list_elements_by_stream()
        self.assertEqual(len(elements), 15)

        # Verify all 5 streams are represented
        streams_found = {m.stream for m in elements}
        self.assertEqual(len(streams_found), len(RuntimeStreamKind))
        for kind in RuntimeStreamKind:
            self.assertIn(kind, streams_found)

        # Verify all 6 coordinators are represented as primary
        primary_coords = {m.primary_coordinator for m in elements}
        self.assertEqual(len(primary_coords), len(KeyDependencyCoordinatorId))

        # Check constitutional invariant on every mapping
        for m in elements:
            self.assertEqual(m.vital_max_hp, VITAL_MAX_HP)
            self.assertLessEqual(m.safe_min, m.safe_max)

    def test_02_stream_and_coordinator_filtering(self) -> None:
        """Tests element queries by stream kind and coordinator ID."""
        pdn_elems = self.coordinator.list_elements_by_stream(RuntimeStreamKind.SILICON_PDN_TELEMETRY)
        self.assertEqual(len(pdn_elems), 3)

        render_elems = self.coordinator.list_elements_by_stream(RuntimeStreamKind.CHRONOMETRIC_RENDER_STREAM)
        self.assertEqual(len(render_elems), 3)

        voltage_coord_elems = self.coordinator.list_elements_by_coordinator(
            KeyDependencyCoordinatorId.PDN_VOLTAGE_STABILITY_COORDINATOR
        )
        # Should include primary and coupled elements
        self.assertGreaterEqual(len(voltage_coord_elems), 3)

    def test_03_successful_safe_realtime_interventions(self) -> None:
        """Tests authorized real-time interventions committed within safe operating limits."""
        # 1. Safe intervention on PDN impedance (e.g. adjust from 50.0 to 52.0 Ohm)
        var_z0 = self.dimensioner.get_variable("VAR_PDN_IMPEDANCE_Z0")
        orig_z0 = var_z0.current_value

        receipt_z0 = self.coordinator.intervene_realtime(
            element_id="ELEM_PDN_IMPEDANCE",
            delta=2.0,
            rationale="Optimize RLC damping for transient surge",
            proposer_role="PDN_FREQUENCY_TRACKER",
            evaluator_role="SAFETY_CONSTITUTIONAL_AUDITOR"
        )
        self.assertTrue(receipt_z0.authorized)
        self.assertEqual(receipt_z0.status, "COMMITTED_SAFE")
        self.assertEqual(receipt_z0.mutation_delta_bytes, 8)
        self.assertAlmostEqual(receipt_z0.committed_value, orig_z0 + 2.0, places=4)
        self.assertAlmostEqual(var_z0.current_value, orig_z0 + 2.0, places=4)

        # Rollback intervention to preserve baseline state
        self.coordinator.intervene_realtime(
            element_id="ELEM_PDN_IMPEDANCE",
            delta=-2.0,
            rationale="Rollback damping to baseline",
            proposer_role="PDN_FREQUENCY_TRACKER",
            evaluator_role="SAFETY_CONSTITUTIONAL_AUDITOR"
        )
        self.assertAlmostEqual(var_z0.current_value, orig_z0, places=4)

    def test_04_rejection_separation_of_concerns(self) -> None:
        """Ensures that proposer == evaluator is strictly rejected with Delta S == 0."""
        var = self.dimensioner.get_variable("VAR_PDN_IMPEDANCE_Z0")
        initial_val = var.current_value

        receipt = self.coordinator.intervene_realtime(
            element_id="ELEM_PDN_IMPEDANCE",
            delta=5.0,
            rationale="Unilateral self-certified adjustment",
            proposer_role="AUTONOMOUS_ACTUATOR",
            evaluator_role="AUTONOMOUS_ACTUATOR"  # Breach!
        )
        self.assertFalse(receipt.authorized)
        self.assertEqual(receipt.status, "REJECTED_SOC")
        self.assertEqual(receipt.mutation_delta_bytes, 0)
        self.assertEqual(var.current_value, initial_val)  # Delta S == 0

    def test_05_rejection_direct_bounds_breach(self) -> None:
        """Ensures that projected values exceeding the element's safe envelope are rejected."""
        var = self.dimensioner.get_variable("VAR_DISPLAY_REFRESH_RATE_HZ")
        initial_hz = var.current_value  # 60.0

        # Safe max is 120.0 Hz. Requesting +84.0 Hz would yield 144.0 Hz (breach)
        receipt = self.coordinator.intervene_realtime(
            element_id="ELEM_DISPLAY_REFRESH_HZ",
            delta=84.0,
            rationale="Attempt 144Hz high refresh without adequate blanking",
            proposer_role="GAMING_CLIENT",
            evaluator_role="GATEKEEPER"
        )
        self.assertFalse(receipt.authorized)
        self.assertEqual(receipt.status, "REJECTED_BOUNDS")
        self.assertEqual(receipt.mutation_delta_bytes, 0)
        self.assertEqual(var.current_value, initial_hz)  # Delta S == 0

    def test_06_rejection_coupled_voltage_ceiling_breach(self) -> None:
        """Ensures voltage interventions exceeding silicon dielectric limit 1.00V are quarantined."""
        var = self.dimensioner.get_variable("VAR_VDD_NOMINAL_VOLTS")
        initial_v = var.current_value  # 0.85V

        # Propose delta of +0.17V -> 1.02V (within [0.65, 1.05] envelope, but breaches 1.00V sustained ceiling)
        receipt = self.coordinator.intervene_realtime(
            element_id="ELEM_CORE_VOLTAGE",
            delta=0.17,
            rationale="Extreme turbo boost over-voltage",
            proposer_role="PERFORMANCE_TUNER",
            evaluator_role="GATEKEEPER"
        )
        self.assertFalse(receipt.authorized)
        self.assertEqual(receipt.status, "QUARANTINED_SAFETY_BREACH")
        self.assertIn("exceeds safe sustained ceiling", receipt.reason)
        self.assertEqual(receipt.mutation_delta_bytes, 0)
        self.assertEqual(var.current_value, initial_v)  # Delta S == 0

    def test_07_rejection_dwell_frame_underflow(self) -> None:
        """Ensures dwell frame count cannot be reduced below 12 (KSPP P8)."""
        var = self.dimensioner.get_variable("VAR_DWELL_FRAME_LIMIT")
        initial_dwell = var.current_value  # 12.0

        # Propose -4.0 frames -> 8.0 frames (breaches minimum of 12)
        receipt = self.coordinator.intervene_realtime(
            element_id="ELEM_DWELL_FRAME_COUNT",
            delta=-4.0,
            rationale="Attempt fast aggressive upscale",
            proposer_role="FAST_UPSCALE_OBSERVER",
            evaluator_role="GATEKEEPER"
        )
        self.assertFalse(receipt.authorized)
        self.assertEqual(receipt.mutation_delta_bytes, 0)
        self.assertEqual(var.current_value, initial_dwell)  # Delta S == 0

    def test_08_rejection_unknown_element(self) -> None:
        """Ensures unknown element IDs are rejected cleanly without mutating any state."""
        receipt = self.coordinator.intervene_realtime(
            element_id="ELEM_NONEXISTENT_MAGIC_VARIABLE",
            delta=10.0,
            rationale="Attempt adjusting non-existent element",
            proposer_role="UNKNOWN_AGENT",
            evaluator_role="GATEKEEPER"
        )
        self.assertFalse(receipt.authorized)
        self.assertEqual(receipt.status, "REJECTED_UNKNOWN")
        self.assertEqual(receipt.mutation_delta_bytes, 0)

    def test_09_holistic_stream_sync_cycle(self) -> None:
        """Tests the continuous holistic synchronization cycle across all 5 unified streams."""
        sync_receipt = self.coordinator.sync_all_streams_cycle()

        self.assertIsInstance(sync_receipt, UnifiedStreamSyncReceipt)
        self.assertEqual(sync_receipt.total_elements_monitored, 15)
        self.assertTrue(sync_receipt.all_streams_in_safe_envelope)
        self.assertGreaterEqual(sync_receipt.chattering_suppression_ratio, 0.80)
        self.assertGreaterEqual(sync_receipt.vblank_shadow_headroom_ms, 3.50)
        self.assertLessEqual(sync_receipt.uma_bus_contention_ratio, 0.78)
        self.assertEqual(sync_receipt.det_j_preservation, 1.0)
        self.assertEqual(sync_receipt.vital_max_hp, VITAL_MAX_HP)
        self.assertTrue(len(sync_receipt.audit_hash) > 0)

    def test_10_constitutional_invariant_enforcement(self) -> None:
        """Verifies ContractViolation is raised if VITAL_MAX_HP != 6 is attempted."""
        with self.assertRaises(ContractViolation):
            UnifiedCrossStreamDependencyCoordinator(vital_max_hp=5)

        with self.assertRaises(ContractViolation):
            SafeRealtimeInterventionActuator(
                element_mappings=self.coordinator.mappings,
                vital_max_hp=7
            )

        with self.assertRaises(ContractViolation):
            StreamElementMapping(
                element_id="ELEM_TEST",
                stream=RuntimeStreamKind.SILICON_PDN_TELEMETRY,
                name="Test Element",
                primary_coordinator=KeyDependencyCoordinatorId.PDN_VOLTAGE_STABILITY_COORDINATOR,
                coupled_coordinators=(),
                governing_variable_id="VAR_VDD_NOMINAL_VOLTS",
                safe_min=1.0,
                safe_max=0.5,  # min > max violation!
                unit="V",
                description="Invalid envelope",
                vital_max_hp=VITAL_MAX_HP
            )


if __name__ == "__main__":
    unittest.main()
