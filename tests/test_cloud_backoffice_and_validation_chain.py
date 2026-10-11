#!/usr/bin/env python3
"""
Unit and Integration Tests for Krystal-Stack Cloud Back-Office & Validation Chains
=============================================================================
Verifies:
  1. All 7 links of the deterministic ValidationChain pipeline:
     - IngressAuthenticationValidator
     - ConductCodeInvariantValidator
     - PhysicalParameterValidator
     - CgroupsResourceContractValidator
     - PauliExclusionSpatialValidator
     - CryptographicAuditLedgerValidator
     - SlaLatencyBoundValidator
  2. Invariant preservation: VITAL_MAX_HP == 6, defensive fallback [VIT 4HP]
  3. SHA-256 HMAC cryptographic certificate signing & verification
  4. Fault injections: rate-limit, thermal breach, Pauli collision, SLA breach
  5. CloudBackofficeService KPI aggregation, ledger explorer, and compliance reporting
=============================================================================
"""

import os
import sys
import unittest
import time
import json
import hashlib
import hmac

WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

from krystal_kernel.scene_constructor_presets import VITAL_MAX_HP
from krystal_kernel.validation_chain_pipeline import (
    ValidationChainPipeline,
    ValidationCertificate,
    ValidationLinkStatus,
    IngressAuthenticationValidator,
    ConductCodeInvariantValidator,
    PhysicalParameterValidator,
    CgroupsResourceContractValidator,
    PauliExclusionSpatialValidator,
    CryptographicAuditLedgerValidator,
    SlaLatencyBoundValidator,
    GLOBAL_VALIDATION_CHAIN_PIPELINE
)
from krystal_web_hub.cloud_backoffice_service import (
    CloudBackofficeService,
    GLOBAL_CLOUD_BACKOFFICE_SERVICE
)


class TestValidationChainPipeline(unittest.TestCase):
    """Verifies deterministic multi-link validation chain execution and safety fallbacks."""

    def setUp(self):
        self.pipeline = ValidationChainPipeline()
        self.nominal_payload = {
            "token": "BEARER_KRYSTAL_GOV_ENTERPRISE_KEY_2026",
            "actor_id": "GOV_AUDITOR_UNIT_ALPHA",
            "conduct_badges": ["[ACT]", "[LCK]", "[CNF 0.99]", "[VITAL_MAX_HP 6]"],
            "vital_hp": 6,
            "coolant_pressure_mpa": 2.4,
            "current_amps": 5.2,
            "resistance_ohms": 4.0,
            "temperature_c": 54.0,
            "throttle_time_ms": 0.0,
            "cpu_throttling_count": 0,
            "active_entities": [
                {"entity_id": "E1", "q": 0, "r": 0, "layer": "A_CENTER"},
                {"entity_id": "E2", "q": 1, "r": -1, "layer": "A_CENTER"},
                {"entity_id": "E3", "q": 0, "r": 1, "layer": "A_CENTER"}
            ],
            "queue_depth": 14,
            "p95_latency_ms": 12.8,
            "p99_latency_ms": 22.4
        }

    def test_nominal_chain_execution_passes_all_links(self):
        """A compliant payload must pass all 7 validation links with VITAL_MAX_HP == 6."""
        cert = self.pipeline.execute_chain(self.nominal_payload)

        self.assertTrue(cert.is_valid, f"Nominal chain failed: {cert.rejection_reason}")
        self.assertEqual(cert.vital_max_hp, VITAL_MAX_HP)
        self.assertEqual(cert.vital_max_hp, 6)
        self.assertEqual(len(cert.link_results), 7)

        # Verify all links passed
        for link in cert.link_results:
            self.assertEqual(link.status, ValidationLinkStatus.PASSED)
            self.assertIn("latency_ms", link.details)

        # Verify cryptographic signature
        self.assertTrue(cert.crypto_signature.startswith("HMAC-SHA256:"))
        self.assertTrue(cert.verify_signature())

    def test_ingress_rate_limiting_fault(self):
        """Flooding requests from a single actor must trigger rate limit rejection."""
        pipeline = ValidationChainPipeline()
        actor = "MALICIOUS_FLOOD_ACTOR_01"
        payload = dict(self.nominal_payload, actor_id=actor)

        # Ingress validator allows max 100 requests per minute
        ingress = IngressAuthenticationValidator(max_requests_per_minute=5)
        isolated_pipeline = ValidationChainPipeline(links=[ingress])

        for _ in range(5):
            cert = isolated_pipeline.execute_chain(payload)
            self.assertTrue(cert.is_valid)

        # 6th request must be rejected
        blocked_cert = isolated_pipeline.execute_chain(payload)
        self.assertFalse(blocked_cert.is_valid)
        self.assertIn("Rate limit exceeded", blocked_cert.rejection_reason)
        self.assertEqual(blocked_cert.vital_max_hp, 0)

    def test_conduct_code_invariants_and_low_confidence_fallback(self):
        """Low confidence ([CNF < 0.85]) or missing badges trigger fallback to 4 HP, not crash."""
        payload = dict(self.nominal_payload)
        payload["conduct_badges"] = ["[ACT]", "[CNF 0.60]"]  # Missing [LCK], low confidence

        cert = self.pipeline.execute_chain(payload)
        self.assertTrue(cert.is_valid)  # Allowed under defensive fallback
        self.assertEqual(cert.vital_max_hp, 4)  # Clamped to 4 HP
        self.assertIn("[VIT 4HP]", cert.conduct_badges)

        # When [HAZ] is present, pipeline enters emergency interlock
        hazard_payload = dict(self.nominal_payload)
        hazard_payload["conduct_badges"] = ["[ACT]", "[LCK]", "[HAZ]"]
        hazard_cert = self.pipeline.execute_chain(hazard_payload)
        self.assertFalse(hazard_cert.is_valid)
        self.assertIn("Hazard interlock active", hazard_cert.rejection_reason)

    def test_physical_parameter_thermal_and_current_breach(self):
        """Spindle temperature > 75C or current > 10A must trigger thermal governor fallback."""
        overheat_payload = dict(self.nominal_payload)
        overheat_payload["temperature_c"] = 88.5  # Critical overheat

        cert = self.pipeline.execute_chain(overheat_payload)
        # Must not crash; triggers DEFENSIVE_FALLBACK with vital HP = 4
        self.assertTrue(cert.is_valid)
        self.assertEqual(cert.vital_max_hp, 4)
        
        # Verify physical link result specifically
        phys_link = next(r for r in cert.link_results if r.link_name == "PhysicalParameterValidator")
        self.assertEqual(phys_link.status, ValidationLinkStatus.DEFENSIVE_FALLBACK)
        self.assertIn("temperature_c 88.5 exceeds safe limit", phys_link.message)

    def test_pauli_exclusion_spatial_collision(self):
        """Two entities attempting to occupy the same (q, r) hex on layer A_CENTER must be rejected."""
        collision_payload = dict(self.nominal_payload)
        collision_payload["active_entities"] = [
            {"entity_id": "DRUID_01", "q": 2, "r": -1, "layer": "A_CENTER"},
            {"entity_id": "TOXIC_01", "q": 2, "r": -1, "layer": "A_CENTER"}  # Same hex!
        ]

        cert = self.pipeline.execute_chain(collision_payload)
        self.assertFalse(cert.is_valid)
        self.assertIn("Pauli Exclusion Violation", cert.rejection_reason)

        pauli_link = next(r for r in cert.link_results if r.link_name == "PauliExclusionSpatialValidator")
        self.assertEqual(pauli_link.status, ValidationLinkStatus.REJECTED)

    def test_sla_latency_bounds_breach(self):
        """Queue depth > 200 or p95 > 25ms triggers SLA rejection."""
        sla_breach_payload = dict(self.nominal_payload)
        sla_breach_payload["queue_depth"] = 285  # Over limit

        cert = self.pipeline.execute_chain(sla_breach_payload)
        self.assertFalse(cert.is_valid)
        self.assertIn("SLA queue depth exceeded", cert.rejection_reason)


class TestCloudBackofficeService(unittest.TestCase):
    """Verifies Cloud Back-Office management service, ledger audit and compliance reports."""

    def setUp(self):
        self.service = CloudBackofficeService()

    def test_dashboard_summary_structure(self):
        """Dashboard must return 5D health vector, ledger height, SLA metrics, and vital invariant."""
        dash = self.service.get_dashboard_summary()

        self.assertEqual(dash["status"], "ONLINE")
        self.assertEqual(dash["vital_max_hp"], VITAL_MAX_HP)
        self.assertEqual(dash["vital_max_hp"], 6)
        self.assertIn("health_vector", dash)
        self.assertIn("cryptographic_ledger", dash)
        self.assertTrue(dash["cryptographic_ledger"]["integrity_verified"])
        self.assertGreaterEqual(dash["pass_rate_pct"], 0.0)
        self.assertLessEqual(dash["pass_rate_pct"], 100.0)

    def test_live_validation_and_history(self):
        """Executing live validation must append to audit history and update pass rates."""
        initial_history_len = len(self.service.validation_history)
        payload = {
            "token": "BEARER_GOV_KEY",
            "actor_id": "TEST_UNIT",
            "conduct_badges": ["[ACT]", "[LCK]", "[CNF 0.95]"],
            "vital_hp": 6,
            "temperature_c": 50.0,
            "current_amps": 4.5,
            "resistance_ohms": 2.0,
            "queue_depth": 10,
            "p95_latency_ms": 11.2,
            "p99_latency_ms": 19.5
        }

        res = self.service.run_live_validation(payload)
        self.assertTrue(res["is_valid"])
        self.assertEqual(len(self.service.validation_history), initial_history_len + 1)

        # Check list_validation_chains
        recent = self.service.list_validation_chains(limit=5)
        self.assertGreater(len(recent), 0)
        self.assertEqual(recent[0]["certificate_id"], res["certificate_id"])

    def test_cryptographic_audit_ledger_verification(self):
        """Ledger blocks must have valid SHA-256 block continuity and integrity."""
        ledger_info = self.service.get_ledger_blocks(limit=10)
        self.assertTrue(ledger_info["integrity_verified"])
        self.assertGreaterEqual(ledger_info["total_blocks"], 1)

        blocks = ledger_info["blocks"]
        for b in blocks:
            self.assertIn("block_index", b)
            self.assertIn("entry_hash", b)
            self.assertIn("previous_hash", b)

    def test_compliance_reporting_iso13849_and_nist_sp800_53(self):
        """Compliance report export must support both JSON and Markdown formats."""
        # JSON format
        json_report = self.service.export_compliance_report(format_type="json")
        self.assertIn("report_id", json_report)
        self.assertEqual(json_report["regulatory_frameworks"]["iso_13849_1"], "PL d / Cat 3 (Hardware Watchdog)")
        self.assertEqual(json_report["regulatory_frameworks"]["vital_invariant_guard"], "VITAL_MAX_HP == 6 STRICT")
        self.assertTrue(json_report["cryptographic_hash"].startswith("SHA256:"))

        # Markdown format
        md_report = self.service.export_compliance_report(format_type="markdown")
        self.assertIn("# KRYSTAL-STACK // COMPLIANCE & FORENSIC AUDIT REPORT", md_report["markdown_content"])
        self.assertIn("ISO 13849-1", md_report["markdown_content"])
        self.assertIn("NIST SP 800-53", md_report["markdown_content"])


if __name__ == "__main__":
    unittest.main()
