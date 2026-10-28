#!/usr/bin/env python3
"""
================================================================================
KSL v0.2 — HTE-001 PREFLIGHT & REPRODUCIBILITY TEST SUITE
================================================================================
Module: tests/test_hte001_preflight.py

Verifies:
  1. Hte001PreflightChecker runs all 7 checks and produces a signed receipt.
  2. Stratified status: [PRE_FLIGHT_VERIFIED_EMULATED] with physical_run_allowed=False in dev.
  3. Enforce secure key fails closed when environment key is missing.
  4. Enforce physical hardware flag behaves properly (fails when physical board missing).
  5. Tampered manifest hash is detected and fails preflight.
  6. Repro standalone evaluator rejects unexecuted template receipts (Anti-Promotion).
  7. Repro standalone evaluator validates complete and compliant physical receipts.
  8. Repro standalone evaluator rejects exact-zero and envelope breaches.
  9. JSON schemas in repro/HTE001/schemas are syntactically valid and cover required properties.
  10. Constitutional Sentinel P10: VITAL_MAX_HP == 6.
================================================================================
"""

import json
import unittest
import tempfile
import time
import hmac
import hashlib
from pathlib import Path

from krystal_kernel.constants import VITAL_MAX_HP
from krystal_kernel.hte001_preflight import (
    Hte001PreflightChecker,
    EXPECTED_MANIFEST_DIGEST,
    EXPECTED_HARNESS_DIGEST,
)
from repro.HTE001.standalone_evaluator import (
    evaluate_single_receipt,
    evaluate_batch_run,
    canonical_jcs_serialize,
)


class TestHte001PreflightAndReproducibility(unittest.TestCase):
    """Test suite for HTE-001 preflight and reproducibility artifacts."""

    def setUp(self):
        self.repo_root = Path(__file__).parent.parent
        self.manifest_path = self.repo_root / "artifacts" / "hte001" / "HTE001_RUN_MANIFEST.json"
        self.harness_path = self.repo_root / "krystal_kernel" / "hte001_hardware_transition_harness.py"
        self.artifacts_dir = self.repo_root / "artifacts" / "hte001"
        self.repro_dir = self.repo_root / "repro" / "HTE001"

    def _create_mock_valid_physical_receipt(self, manifest: dict, secret_key: bytes = b"test_secret") -> dict:
        """Helper to create a fully completed, signed physical execution receipt."""
        receipt = {
            "experiment_id": "HTE-001",
            "run_id": "HTE-001-RUN-0001-MOCK",
            "sequence": 1,
            "software_commit": manifest["software_commit"],
            "manifest_hash": EXPECTED_MANIFEST_DIGEST,
            "ksl_source_hash": manifest["ksl_source_hash"],
            "kir_hash": manifest["kir_hash"],
            "kmicroisa_hash": manifest["kmicroisa_hash"],
            "rtl_hash": manifest["rtl_source_hash"],
            "bitstream_hash": manifest["bitstream_hash"],
            "board_id": manifest["board_model"],
            "fpga_part": manifest["fpga_part"],
            "clock_frequency": manifest["clock_frequency_hz"],
            "vital_max_hp": 6,
            "exact_zero_pass": True,
            "normalized_residuals_pass": True,
            "transition_pass": True,
            "acquisition_valid": True,
            "receipt_integrity": "VERIFIED",
            "evidence_class": "FPGA_BOARD_MEASURED",
            "semantic_verdict": "COMMITTED",
            "raw_residuals": {
                "memory": 0.0,
                "energy": 0.012,
                "latency": 3.2,
                "thermal": 0.12,
                "queue": 0.01,
                "compute": 0.0,
            },
            "normalized_residuals": {
                "energy": 0.24,
                "latency": 0.32,
                "thermal": 0.24,
                "queue": 0.20,
                "compute": 0.0,
            },
            "raw_predictions": {
                "memory": 0.0,
                "energy": 0.05,
                "latency": 10.0,
                "thermal": 0.5,
                "queue": 0.05,
                "compute": 1.0,
            },
            "raw_measurements": {
                "memory": 0.0,
                "energy": 0.062,
                "latency": 13.2,
                "thermal": 0.62,
                "queue": 0.06,
                "compute": 1.0,
            },
            "timestamp_iso": "2026-10-28T12:00:00Z",
            "timestamp_ns": 1793187600000000000,
        }
        jcs = canonical_jcs_serialize(receipt)
        receipt["hmac_signature"] = hmac.new(secret_key, jcs, hashlib.sha256).hexdigest()
        return receipt

    def test_01_constitutional_sentinel_holds(self):
        """Principle 10: VITAL_MAX_HP == 6."""
        self.assertEqual(VITAL_MAX_HP, 6)

    def test_02_preflight_checks_pass_in_emulated_environment(self):
        """Verifies that all preflight checks pass in dev with [PRE_FLIGHT_VERIFIED_EMULATED]."""
        checker = Hte001PreflightChecker(
            manifest_path=str(self.manifest_path),
            harness_path=str(self.harness_path),
            artifacts_dir=str(self.artifacts_dir),
            enforce_physical_hardware=False,
            enforce_secure_key=False,
        )
        passed, receipt = checker.run_all_checks()
        self.assertTrue(passed, f"Preflight checks failed: {receipt.overall_status}")
        self.assertEqual(receipt.overall_status, "[PRE_FLIGHT_VERIFIED_EMULATED]")
        self.assertFalse(receipt.physical_run_allowed)
        self.assertEqual(receipt.signature_provenance, "DEV_FALLBACK_INSECURE_TEST_KEY")
        self.assertEqual(receipt.manifest_hash, EXPECTED_MANIFEST_DIGEST)
        self.assertEqual(receipt.harness_hash, EXPECTED_HARNESS_DIGEST)
        self.assertTrue(receipt.manifest_verified)
        self.assertTrue(receipt.harness_verified)
        self.assertTrue(receipt.software_identity_verified)
        self.assertEqual(receipt.vital_max_hp, 6)
        self.assertEqual(receipt.theory_nodes_total, 56)
        self.assertEqual(len(receipt.hmac_signature), 64)
        self.assertIn("EMULATION NOTICE", receipt.epistemic_warning)

    def test_03_preflight_fails_closed_when_enforcing_secure_key(self):
        """Verifies P0 Provenance: enforce_secure_key=True raises ValueError if env secret is missing."""
        checker = Hte001PreflightChecker(
            manifest_path=str(self.manifest_path),
            harness_path=str(self.harness_path),
            artifacts_dir=str(self.artifacts_dir),
            enforce_physical_hardware=False,
            enforce_secure_key=True,
        )
        with self.assertRaises(ValueError) as ctx:
            checker.run_all_checks()
        self.assertIn("P0 PROVENANCE AUTHENTICATION", str(ctx.exception))

    def test_04_preflight_fails_when_enforcing_missing_physical_hardware(self):
        """Verifies that enforce_physical_hardware=True fails gracefully if board not present."""
        checker = Hte001PreflightChecker(
            manifest_path=str(self.manifest_path),
            harness_path=str(self.harness_path),
            artifacts_dir=str(self.artifacts_dir),
            enforce_physical_hardware=True,
            enforce_secure_key=False,
        )
        passed, receipt = checker.run_all_checks()
        hw_check = next(c for c in receipt.checks if c["check_name"] == "HARDWARE_INTERFACE_PROBE")
        if not receipt.hardware_present:
            self.assertFalse(passed)
            self.assertFalse(hw_check["passed"])
            self.assertEqual(receipt.overall_status, "[PRE-FLIGHT FAILED]")
            self.assertFalse(receipt.physical_run_allowed)

    def test_05_tampered_manifest_fails_preflight(self):
        """Verifies that a tampered manifest file is caught by hash verification."""
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            f.write(json.dumps({"experiment_id": "TAMPERED-001"}))
            tampered_path = f.name

        try:
            checker = Hte001PreflightChecker(
                manifest_path=tampered_path,
                harness_path=str(self.harness_path),
                artifacts_dir=str(self.artifacts_dir),
                enforce_physical_hardware=False,
                enforce_secure_key=False,
            )
            passed, receipt = checker.run_all_checks()
            self.assertFalse(passed)
            self.assertFalse(receipt.manifest_verified)
            self.assertEqual(receipt.overall_status, "[PRE-FLIGHT FAILED]")
        finally:
            Path(tampered_path).unlink(missing_ok=True)

    def test_06_repro_standalone_evaluator_rejects_unexecuted_template(self):
        """Verifies P0 Anti-Promotion: standalone_evaluator rejects RECEIPT_TEMPLATE.json."""
        with open(self.repro_dir / "HTE001_RUN_MANIFEST.json", "r", encoding="utf-8") as f:
            manifest = json.load(f)
        with open(self.repro_dir / "RECEIPT_TEMPLATE.json", "r", encoding="utf-8") as f:
            template = json.load(f)

        passed, notes = evaluate_single_receipt(template, manifest)
        self.assertFalse(passed)
        self.assertTrue(any("PHYSICAL_EVIDENCE_INCOMPLETE" in n for n in notes))
        self.assertTrue(any("Unexecuted receipt template detected" in n for n in notes))

    def test_07_repro_standalone_evaluator_validates_complete_physical_receipt(self):
        """Verifies that standalone_evaluator validates a genuine completed physical receipt."""
        with open(self.repro_dir / "HTE001_RUN_MANIFEST.json", "r", encoding="utf-8") as f:
            manifest = json.load(f)

        secret = b"test_evaluator_secret"
        valid_receipt = self._create_mock_valid_physical_receipt(manifest, secret)

        passed, notes = evaluate_single_receipt(valid_receipt, manifest, secret_key=secret)
        self.assertTrue(passed, f"Evaluator failed on mock receipt: {notes}")
        self.assertTrue(any("I_0 Exact Zero Invariant holds" in n for n in notes))
        self.assertTrue(any("Normalized envelope compliance" in n for n in notes))
        self.assertTrue(any("HMAC-SHA256 signature verified" in n for n in notes))

    def test_08_repro_standalone_evaluator_rejects_exact_zero_breach(self):
        """Verifies that standalone_evaluator rejects non-zero memory delta (I_0 breach)."""
        with open(self.repro_dir / "HTE001_RUN_MANIFEST.json", "r", encoding="utf-8") as f:
            manifest = json.load(f)

        receipt = self._create_mock_valid_physical_receipt(manifest)
        receipt["raw_residuals"]["memory"] = 64.0  # Invariant breach!
        passed, notes = evaluate_single_receipt(receipt, manifest)
        self.assertFalse(passed)
        self.assertTrue(any("I_0 Exact Zero Violation" in n for n in notes))

    def test_09_repro_standalone_evaluator_rejects_envelope_breach(self):
        """Verifies that standalone_evaluator rejects normalized residual > 1.0 (I_+ breach)."""
        with open(self.repro_dir / "HTE001_RUN_MANIFEST.json", "r", encoding="utf-8") as f:
            manifest = json.load(f)

        receipt = self._create_mock_valid_physical_receipt(manifest)
        receipt["normalized_residuals"]["energy"] = 1.5  # Breach!
        passed, notes = evaluate_single_receipt(receipt, manifest)
        self.assertFalse(passed)
        self.assertTrue(any("Envelope breach on 'energy'" in n for n in notes))

    def test_10_repro_json_schemas_are_valid(self):
        """Verifies that both manifest and receipt JSON schemas exist and parse correctly."""
        manifest_schema_path = self.repro_dir / "schemas" / "hte001_manifest_schema.json"
        receipt_schema_path = self.repro_dir / "schemas" / "hte001_receipt_schema.json"

        self.assertTrue(manifest_schema_path.exists())
        self.assertTrue(receipt_schema_path.exists())

        with open(manifest_schema_path, "r", encoding="utf-8") as f:
            m_schema = json.load(f)
            self.assertEqual(m_schema["title"], "Hte001RunManifestSchema")
            self.assertIn("required", m_schema)

        with open(receipt_schema_path, "r", encoding="utf-8") as f:
            r_schema = json.load(f)
            self.assertEqual(r_schema["title"], "Hte001HardwareTransitionReceiptSchema")
            self.assertIn("required", r_schema)


if __name__ == "__main__":
    unittest.main()
