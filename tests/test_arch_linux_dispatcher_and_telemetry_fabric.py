#!/usr/bin/env python3
"""
Unit and integration test suite for Arch Linux API Dispatcher,
Telemetry Fabric, Desired vs Observed State Resource Contract,
Diagnostic Governor (Observe -> Explain -> Intervene),
Communication Integrity Plane, and Calibrated Physical Mathematics.
"""

import os
import sys
import time
import unittest

WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

from krystal_kernel.scene_constructor_presets import VITAL_MAX_HP
from krystal_kernel.telemetry_fabric_and_resource_contract import (
    CanonicalTelemetryPacket,
    DesiredResourceState,
    ObservedResourceState,
    ResourceContract,
    DiagnosticGovernor,
    GovernorExecutionMode,
    CommunicationFlow,
    CommunicationIntegrityPlane,
    TelemetryFabric
)
from krystal_kernel.arch_linux_dispatcher import (
    ArchLinuxProfileType,
    ArchLinuxProfileConfig,
    ARCH_LINUX_PROFILES,
    SystemdUnitGenerator,
    PhysicalMathematicsSensor,
    CalibratedFaultInjectionController,
    ArchLinuxApiDispatcher
)


class TestArchLinuxDispatcherAndTelemetryFabric(unittest.TestCase):
    """Test suite validating Arch Linux Dispatcher, Telemetry Fabric & Resource Integrity."""

    def setUp(self):
        self.dispatcher = ArchLinuxApiDispatcher(default_profile=ArchLinuxProfileType.GOV_DETERMINISTIC_RT)
        self.fabric = self.dispatcher.telemetry_fabric

    def test_canonical_telemetry_contract_integrity(self):
        """Verify that CanonicalTelemetryPacket computes deterministic SHA-256 and detects tampering."""
        pkt = CanonicalTelemetryPacket(
            packet_id="PKT-TEST-001",
            timestamp=1700000000.0,
            correlation_id="CORR-TEST-001",
            source_node="arch-rt-node-01",
            profile_active="GOV_DETERMINISTIC_RT",
            cpu_usage_pct=35.0,
            cgroups_nr_throttled=0,
            cgroups_throttled_time_usec=0,
            memory_current_bytes=400000000,
            memory_limit_bytes=1073741824,
            queue_depth=10,
            p95_latency_ms=12.0,
            p99_latency_ms=22.0,
            physical_temperature_celsius=38.5,
            thermal_partition_q_ratio=0.68,
            effective_resistance_ohms=2.0,
            effective_current_amps=6.0,
            health_vector={"composite_health": 0.98}
        )
        h1 = pkt.calculate_hash()
        self.assertEqual(len(h1), 64)

        # Re-calculating with identical values must yield exact same hash
        h2 = pkt.calculate_hash()
        self.assertEqual(h1, h2)

        # Altering a single physical metric must alter hash
        pkt.physical_temperature_celsius = 78.5
        h3 = pkt.calculate_hash()
        self.assertNotEqual(h1, h3)

    def test_resource_contract_desired_vs_observed_drift(self):
        """Verify Resource Contract flags cgroups throttling, memory high, and thermal drift."""
        contract = self.fabric.resource_contract
        corr = "CORR-DRIFT-TEST"

        # 1. Nominal observed state (zero drift)
        nominal_obs = ObservedResourceState(
            cpu_cores_active=[2, 3],
            cpu_usage_pct=45.0,
            cgroups_nr_throttled=0,
            cgroups_throttled_time_usec=0,
            memory_current_bytes=500 * 1024 * 1024,
            queue_depth=20,
            p95_latency_ms=15.0,
            p99_latency_ms=30.0,
            physical_temperature_celsius=45.0,
            effective_current_amps=5.0
        )
        drift1 = contract.evaluate_drift(nominal_obs, correlation_id=corr)
        self.assertFalse(drift1.has_drift)
        self.assertEqual(drift1.drift_score, 0.0)
        self.assertEqual(len(drift1.violations), 0)

        # 2. Throttling and thermal overshoot injection
        anomalous_obs = ObservedResourceState(
            cpu_cores_active=[2, 3],
            cpu_usage_pct=99.0,
            cgroups_nr_throttled=15,
            cgroups_throttled_time_usec=15000,
            memory_current_bytes=900 * 1024 * 1024,  # Exceeds memory_high (800MB)
            queue_depth=250,                         # Exceeds max_queue_depth (200)
            p95_latency_ms=45.0,                     # Exceeds budget (25ms)
            p99_latency_ms=85.0,
            physical_temperature_celsius=82.0,       # Exceeds ceiling (75C)
            effective_current_amps=14.0              # Exceeds max current (12A)
        )
        drift2 = contract.evaluate_drift(anomalous_obs, correlation_id=corr)
        self.assertTrue(drift2.has_drift)
        self.assertGreaterEqual(drift2.drift_score, 0.85)
        self.assertGreaterEqual(len(drift2.violations), 5)
        self.assertTrue(any("CGROUPS_THROTTLE_DRIFT" in v for v in drift2.violations))
        self.assertTrue(any("THERMAL_CEILING_DRIFT" in v for v in drift2.violations))

    def test_diagnostic_governor_observe_explain_intervene(self):
        """Verify strict sequence: Observe -> Explain -> Intervene."""
        governor = DiagnosticGovernor(mode=GovernorExecutionMode.DIAGNOSTIC_ONLY)
        telemetry = {
            "queue_depth": 35,
            "nr_throttled_delta": 25,
            "cpu_usage_pct": 98.0,
            "memory_usage_pct": 50.0,
            "loopback_rtt_ms": 0.1,
            "tcp_retransmits": 0,
            "intent_confidence": 0.95
        }

        # 1. In DIAGNOSTIC_ONLY mode: Observe & Explain, but DO NOT intervene
        prop1 = governor.process_anomaly(telemetry, correlation_id="CORR-DIAG-01")
        self.assertEqual(prop1.winning_hypothesis.code, "H2_CGROUPS_CPU_THROTTLING")
        self.assertEqual(prop1.recommended_action, "CGROUP_QUOTA_RESET_PREEMPT_RT")
        self.assertFalse(prop1.is_executed, "Intervention MUST NOT execute in DIAGNOSTIC_ONLY mode")
        self.assertEqual(len(governor.executed_interventions), 0)

        # 2. Switch to BOUNDED_AUTONOMOUS_INTERVENTION mode
        governor.set_execution_mode(GovernorExecutionMode.BOUNDED_AUTONOMOUS_INTERVENTION)
        prop2 = governor.process_anomaly(telemetry, correlation_id="CORR-AUTO-02")
        self.assertTrue(prop2.is_executed, "Intervention MUST execute once mode is enabled and hypothesis is confirmed")
        self.assertEqual(len(governor.executed_interventions), 1)

    def test_communication_integrity_plane(self):
        """Verify Communication Integrity Plane permits declared flows and blocks rogue traffic."""
        plane = CommunicationIntegrityPlane()
        corr = "CORR-NET-TEST"

        # Authorized declared flow (API Dispatcher -> Worker via socket)
        auth_res = plane.validate_packet_flow(
            source="API_DISPATCHER",
            destination="WORKER_CORE",
            protocol="UNIX_DOMAIN_SOCKET",
            port_or_socket="/run/krystal/worker.sock",
            direction="BIDIRECTIONAL",
            rate_req_sec=1200,
            correlation_id=corr
        )
        self.assertTrue(auth_res.is_authorized)
        self.assertEqual(auth_res.reason, "AUTHORIZED_DECLARED_FLOW")

        # Rogue undeclared flow (e.g., unauthorized external exfiltration)
        rogue_res = plane.validate_packet_flow(
            source="WORKER_CORE",
            destination="UNKNOWN_REMOTE_IP",
            protocol="TCP_RAW",
            port_or_socket="4444",
            direction="EGRESS",
            rate_req_sec=10,
            correlation_id=corr
        )
        self.assertFalse(rogue_res.is_authorized)
        self.assertIn("UNDECLARED_COMMUNICATION_FLOW", rogue_res.reason)
        self.assertEqual(len(plane.violations), 1)

    def test_arch_linux_profile_switching_and_systemd_artifacts(self):
        """Verify dynamic profile switching and systemd slice generation."""
        # 1. Switch to EDGE_FLEET_HEADLESS
        res1 = self.dispatcher.switch_profile(ArchLinuxProfileType.EDGE_FLEET_HEADLESS)
        self.assertEqual(res1["status"], "PROFILE_ACTIVATED")
        self.assertEqual(self.dispatcher.active_profile_type, ArchLinuxProfileType.EDGE_FLEET_HEADLESS)
        self.assertEqual(self.dispatcher.profile_config.allowed_cpus, "0,1")
        self.assertEqual(self.dispatcher.telemetry_fabric.desired.memory_max_bytes, 512 * 1024 * 1024)

        # 2. Switch back to GOV_DETERMINISTIC_RT
        res2 = self.dispatcher.switch_profile(ArchLinuxProfileType.GOV_DETERMINISTIC_RT)
        self.assertEqual(res2["status"], "PROFILE_ACTIVATED")
        self.assertEqual(self.dispatcher.profile_config.target_kernel, "linux-rt")

        # 3. Verify systemd slice artifact generation
        artifacts = self.dispatcher.get_systemd_artifacts()
        slice_content = artifacts["slice_content"]
        self.assertIn("AllowedCPUs=2,3", slice_content)
        self.assertIn("MemorySwapMax=0", slice_content)  # Zero swap
        self.assertIn("krystal-rt.slice", artifacts["slice_filename"])

        service_content = artifacts["service_content"]
        self.assertIn("CPUSchedulingPolicy=fifo", service_content)
        self.assertIn("CPUSchedulingPriority=80", service_content)
        self.assertIn("LimitMEMLOCK=infinity", service_content)

    def test_physical_mathematics_as_telemetry_sensor(self):
        """Verify Physical Mathematics acts as a hardware telemetry sensor."""
        sensor = PhysicalMathematicsSensor(base_resistance=2.0)

        # Nominal cutting conditions (12V on 2.0 Ohm -> 6A)
        nominal = sensor.observe_physical_state(applied_voltage=12.0, cutting_heat_watts=200.0)
        self.assertFalse(nominal.is_physical_hazard)
        self.assertAlmostEqual(nominal.current_amps, 6.0, places=2)
        self.assertIn("[ACT]", nominal.conduct_badge)
        self.assertIn("[!HAZ]", nominal.conduct_badge)

        # Overload surge (36V on 2.0 Ohm -> 18A > 10A ceiling)
        surge = sensor.observe_physical_state(applied_voltage=36.0, cutting_heat_watts=800.0)
        self.assertTrue(surge.is_physical_hazard)
        self.assertGreater(surge.current_amps, 10.0)
        self.assertGreater(surge.temperature_celsius, 75.0)
        self.assertIn("[!ACT]", surge.conduct_badge)
        self.assertIn("[HAZ]", surge.conduct_badge)
        self.assertIn("PHYSICAL_OVERLOAD", surge.signature)

    def test_calibrated_fault_injection_controller(self):
        """Verify Fault Injection Lab accurately calibrates the Hypothesis Engine."""
        calibrator = self.dispatcher.calibrator

        record = calibrator.run_calibrated_experiment(
            experiment_name="PROBE_CPU_THROTTLING_SIGNATURE",
            target_container="krystal_worker_01",
            fault_type="CPU_QUOTA_STEAL",
            expected_hypothesis_code="H2_CGROUPS_CPU_THROTTLING"
        )
        self.assertTrue(record["hypothesis_matched"])
        self.assertEqual(record["detected_hypothesis"], "H2_CGROUPS_CPU_THROTTLING")
        self.assertGreaterEqual(record["confidence_score"], 0.80)
        self.assertEqual(record["recovery_status"], "VERIFIED")
        self.assertLess(record["recovery_duration_ms"], 50.0)

    def test_dispatched_transaction_flow(self):
        """Verify complete transaction dispatch through Arch Linux API Dispatcher."""
        res = self.dispatcher.dispatch_transaction(
            payload_command="CNC_MILL_TITANIUM_TURBINE_BLADE",
            applied_voltage=12.0,
            cutting_heat_watts=220.0
        )
        self.assertEqual(res["status"], "SUCCESS")
        self.assertTrue(res["job_id"].startswith("JOB-"))
        self.assertEqual(res["vital_hp"], VITAL_MAX_HP)
        self.assertIn("conduct_badge", res["physical_observation"])
        self.assertEqual(res["telemetry_fabric"]["status"], "OBSERVATION_INGESTED")


if __name__ == "__main__":
    unittest.main()
