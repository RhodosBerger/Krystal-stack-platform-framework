#!/usr/bin/env python3
"""
================================================================================
KSL v0.2 — HTE-001 PREFLIGHT & REPRODUCIBILITY TEST SUITE
================================================================================
Module: tests/test_hte001_preflight.py

Verifies:
  1. Hte001PreflightChecker runs all 7 checks and produces a signed receipt.
  2. Enforce physical hardware flag behaves properly (fails when physical board missing).
  3. Tampered manifest hash is detected and fails preflight.
  4. Repro standalone evaluator validates compliant receipts and rejects envelope breaches.
  5. JSON schemas in repro/HTE001/schemas are syntactically valid and cover required properties.
  6. Constitutional Sentinel P10: VITAL_MAX_HP == 6.
================================================================================
"""

import json
import unittest
import tempfile
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

    def test_01_constitutional_sentinel_holds(self):
        """Principle 10: VITAL_MAX_HP == 6."""
        self.assertEqual(VITAL_MAX_HP, 6)

    def test_02_preflight_checks_pass_in_emulated_environment(self):
        """Verifies that all preflight checks pass in the standard development environment."""
        checker = Hte001PreflightChecker(
            manifest_path=str(self.manifest_path),
            harness_path=str(self.harness_path),
            artifacts_dir=str(self.artifacts_dir),
            enforce_physical_hardware=False,
        )
        passed, receipt = checker.run_all_checks()
        self.assertTrue(passed, f"Preflight checks failed: {receipt.overall_status}")
        self.assertEqual(receipt.overall_status, "[PRE-FLIGHT VERIFIED]")
        self.assertEqual(receipt.manifest_hash, EXPECTED_MANIFEST_DIGEST)
        self.assertEqual(receipt.harness_hash, EXPECTED_HARNESS_DIGEST)
        self.assertTrue(receipt.manifest_verified)
        self.assertTrue(receipt.harness_verified)
        self.assertTrue(receipt.software_identity_verified)
        self.assertEqual(receipt.vital_max_hp, 6)
        self.assertEqual(receipt.theory_nodes_total, 56)
        self.assertTrue(len(receipt.hmac_signature) == 64)
        self.assertIn("ANTI-PROMOTION NOTICE", receipt.epistemic_warning)

    def test_03_preflight_fails_when_enforcing_missing_physical_hardware(self):
        """Verifies that enforce_physical_hardware=True fails gracefully if board not present."""
        # On dev environment /dev/i2c-1 is typically absent
        checker = Hte001PreflightChecker(
            manifest_path=str(self.manifest_path),
            harness_path=str(self.harness_path),
            artifacts_dir=str(self.artifacts_dir),
            enforce_physical_hardware=True,
        )
        passed, receipt = checker.run_all_checks()
        # If running in environment without real hardware, it must return False
        hw_check = next(c for c in receipt.checks if c["check_name"] == "HARDWARE_INTERFACE_PROBE")
        if not receipt.hardware_present:
            self.assertFalse(passed)
            self.assertFalse(hw_check["passed"])
            self.assertEqual(receipt.overall_status, "[PRE-FLIGHT FAILED]")

    def test_04_tampered_manifest_fails_preflight(self):
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
            )
            passed, receipt = checker.run_all_checks()
            self.assertFalse(passed)
            self.assertFalse(receipt.manifest_verified)
            self.assertEqual(receipt.overall_status, "[PRE-FLIGHT FAILED]")
        finally:
            Path(tampered_path).unlink(missing_ok=True)

    def test_05_repro_standalone_evaluator_validates_template_receipt(self):
        """Verifies that standalone_evaluator validates the compliant template receipt."""
        with open(self.repro_dir / "HTE001_RUN_MANIFEST.json", "r", encoding="utf-8") as f:
            manifest = json.load(f)
        with open(self.repro_dir / "RECEIPT_TEMPLATE.json", "r", encoding="utf-8") as f:
            receipt = json.load(f)

        passed, notes = evaluate_single_receipt(receipt, manifest)
        self.assertTrue(passed, f"Evaluator failed on template: {notes}")
        self.assertTrue(any("I_0 Exact Zero Invariant holds" in n for n in notes))
        self.assertTrue(any("Normalized envelope compliance" in n for n in notes))

    def test_06_repro_standalone_evaluator_rejects_exact_zero_breach(self):
        """Verifies that standalone_evaluator rejects non-zero memory delta (I_0 breach)."""
        with open(self.repro_dir / "HTE001_RUN_MANIFEST.json", "r", encoding="utf-8") as f:
            manifest = json.load(f)
        with open(self.repro_dir / "RECEIPT_TEMPLATE.json", "r", encoding="utf-8") as f:
            receipt = json.load(f)

        # Mutate memory residual
        receipt["raw_residuals"]["memory"] = 64.0  # Invariant breach!
        passed, notes = evaluate_single_receipt(receipt, manifest)
        self.assertFalse(passed)
        self.assertTrue(any("I_0 Exact Zero Violation" in n for n in notes))

    def test_07_repro_standalone_evaluator_rejects_envelope_breach(self):
        """Verifies that standalone_evaluator rejects normalized residual > 1.0 (I_+ breach)."""
        with open(self.repro_dir / "HTE001_RUN_MANIFEST.json", "r", encoding="utf-8") as f:
            manifest = json.load(f)
        with open(self.repro_dir / "RECEIPT_TEMPLATE.json", "r", encoding="utf-8") as f:
            receipt = json.load(f)

        # Mutate energy residual to 1.5 (> 1.0)
        receipt["normalized_residuals"]["energy"] = 1.5
        passed, notes = evaluate_single_receipt(receipt, manifest)
        self.assertFalse(passed)
        self.assertTrue(any("Envelope breach on 'energy'" in n for n in notes))

    def test_08_repro_json_schemas_are_valid(self):
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
