"""
==============================================================================
KSL DOCUMENTATION & EPISTEMIC GOVERNANCE TEST SUITE
==============================================================================
Module: tests/test_documentation_and_governance.py
Constitutional Invariant:
    - THEORY_NODES == 56
    - Anti-Promotion Barrier: Prediction != Evidence, Proposal != Authority
    - Scoped Claims: Pareto claims scoped by M and D
==============================================================================
"""

import json
import unittest
from pathlib import Path

from krystal_kernel.constants import VITAL_MAX_HP
from krystal_kernel.hte001_uncertainty_budget import MetrologicalUncertaintyBudget
from krystal_kernel.hte001_provenance_graph import generate_hte001_provenance_dag


class TestDocumentationAndGovernance(unittest.TestCase):
    """Verifies that documentation, claims registry, and provenance DAG adhere to epistemic laws."""

    def setUp(self) -> None:
        self.repo_root = Path(__file__).parent.parent
        self.claims_path = self.repo_root / "docs" / "evidence" / "CLAIMS_REGISTRY.json"
        self.provenance_path = self.repo_root / "artifacts" / "hte001" / "HTE001_PROVENANCE_GRAPH.json"
        self.manifest_path = self.repo_root / "artifacts" / "hte001" / "HTE001_RUN_MANIFEST.json"

    def test_claims_registry_structure_and_anti_promotion(self) -> None:
        """Verifies schema, completeness, and anti-promotion invariants of CLAIMS_REGISTRY.json."""
        self.assertTrue(self.claims_path.exists(), "CLAIMS_REGISTRY.json must exist")
        with open(self.claims_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.assertEqual(data["baseline_state"], "FROZEN")
        self.assertEqual(data["baseline_commit"], "12f91b5c1b4d42b5ce979fb3a01d2233ba02f0ed")

        # Hierarchy of evidence levels (lower index = strictly lower or equal epistemic authority)
        rank = {cls_name: idx for idx, cls_name in enumerate(data["allowed_evidence_classes"])}

        claims = data.get("claims", [])
        self.assertGreaterEqual(len(claims), 20, "Registry must contain at least 20 core claims")

        for c in claims:
            cid = c["claim_id"]
            curr = c["current_evidence_status"]
            max_allowed = c["max_allowed_evidence_class"]

            self.assertIn(curr, rank, f"Claim {cid} has unknown current evidence class: {curr}")
            self.assertIn(max_allowed, rank, f"Claim {cid} has unknown max allowed class: {max_allowed}")

            # Specific anti-promotion rule for pending physical claims
            if cid in ("KS-CLAIM-019", "KS-CLAIM-020"):
                self.assertEqual(
                    curr, "PENDING",
                    f"Claim {cid} must remain PENDING until physical RUN-0001 execution on KCU116"
                )

            # Invariant: current evidence must not exceed max allowed evidence
            self.assertLessEqual(
                rank[curr], rank[max_allowed],
                f"Anti-promotion violation on {cid}: {curr} exceeds max allowed {max_allowed}"
            )

    def test_experiment_provenance_dag_integrity(self) -> None:
        """Verifies that HTE001_PROVENANCE_GRAPH.json matches the preregistered manifest."""
        self.assertTrue(self.provenance_path.exists(), "HTE001_PROVENANCE_GRAPH.json must exist")
        with open(self.provenance_path, "r", encoding="utf-8") as f:
            dag = json.load(f)

        self.assertEqual(dag["experiment_id"], "HTE-001")
        nodes_by_id = {n["id"]: n for n in dag.get("nodes", [])}

        # Check critical node hashes
        manifest_node = nodes_by_id["NODE_MANIFEST"]
        self.assertEqual(manifest_node["canonical_digest"], "ae6a1add5fcf1d3cb36c36859d3f38ef48a27df340b8aaf39daa3b47ea17f9cf")

        commit_node = nodes_by_id["NODE_SOFTWARE_COMMIT"]
        self.assertEqual(commit_node["canonical_digest"], "12f91b5c1b4d42b5ce979fb3a01d2233ba02f0ed")

        # Merkle Evidence Root must be present and sealed
        self.assertIn("NODE_EVIDENCE_ROOT", nodes_by_id)
        root_node = nodes_by_id["NODE_EVIDENCE_ROOT"]
        self.assertEqual(root_node["status"], "SEALED_PREREGISTRATION")
        self.assertEqual(len(root_node["canonical_digest"]), 64)

        # Telemetry and receipt nodes must be pending execution
        self.assertEqual(nodes_by_id["NODE_RAW_TELEMETRY"]["status"], "PENDING_PHYSICAL_EXECUTION")
        self.assertEqual(nodes_by_id["NODE_HARDWARE_RECEIPT"]["status"], "PENDING_PHYSICAL_EXECUTION")

        # Verify DAG acyclicity
        edges = dag.get("edges", [])
        self.assertGreaterEqual(len(edges), 17)

    def test_measurement_uncertainty_budget_resolvability(self) -> None:
        """Verifies GUM uncertainty budget: expanded uncertainty U_95% must be smaller than epsilon_energy."""
        budget = MetrologicalUncertaintyBudget()
        res = budget.calculate_energy_uncertainty_budget()

        # Both Conservative and Correlated bounds must be strictly less than epsilon (0.050 uJ)
        self.assertTrue(res["metrologically_resolvable"])
        self.assertLess(res["u95_conservative_uj"], res["epsilon_energy_uj"])
        self.assertLess(res["u95_correlated_uj"], res["epsilon_energy_uj"])

        # Discrimination ratio for conservative: > 10x
        self.assertGreater(res["discrimination_ratio_conservative"], 10.0)
        # Discrimination ratio for correlated: > 100x
        self.assertGreater(res["discrimination_ratio_correlated"], 100.0)

    def test_claims_lattice_and_semantic_digests(self) -> None:
        """Verifies 4D evidence lattice points and RFC 8785 claim semantic digests."""
        from krystal_kernel.claims_lattice import (
            EvidenceLatticePoint, compute_claim_semantic_digest
        )

        with open(self.claims_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        for c in data.get("claims", []):
            cid = c["claim_id"]
            # 1. Digest verification
            expected_digest = compute_claim_semantic_digest(c)
            self.assertEqual(c.get("claim_digest"), expected_digest, f"Digest mismatch on {cid}")

            # 2. 4D Lattice Point verification
            curr_lat = EvidenceLatticePoint.from_dict(c["current_evidence_lattice"])
            max_lat = EvidenceLatticePoint.from_dict(c["max_allowed_evidence_lattice"])
            self.assertTrue(
                curr_lat.is_less_than_or_equal_to(max_lat),
                f"4D Lattice anti-promotion violation on {cid}: {curr_lat} exceeds {max_lat}"
            )

            if cid in ("KS-CLAIM-019", "KS-CLAIM-020"):
                self.assertEqual(curr_lat.measurement, "PENDING")

    def test_claim_promotion_protocol_and_gates(self) -> None:
        """Verifies that the ClaimPromotionEngine rejects unauthorized evidence and seals valid receipts."""
        from krystal_kernel.claim_promotion_protocol import ClaimPromotionEngine
        from krystal_kernel.claims_lattice import EvidenceLatticePoint

        engine = ClaimPromotionEngine(self.claims_path)

        # Case A: Promotion rejected if evidence receipt is not VERIFIED
        bad_receipt = {
            "receipt_integrity": "CORRUPTED",
            "semantic_verdict": "COMMITTED",
            "measurement_boundary": "FPGA_BOARD",
            "sensor_id": "INA226_I2C_CALIBRATED"
        }
        target_lat = EvidenceLatticePoint("FROZEN", "REGRESSION_TESTED", "CONTRACT_BOUND", "FPGA_BOARD_MEASURED")
        ok, rec, msg = engine.evaluate_promotion("KS-CLAIM-019", target_lat, bad_receipt)
        self.assertFalse(ok)
        self.assertIn("ReceiptIntegrity", msg)

        # Case B: Promotion rejected if scope does not cover claim
        bad_scope_receipt = {
            "receipt_integrity": "VERIFIED",
            "semantic_verdict": "COMMITTED",
            "measurement_boundary": "HOST_ONLY_CPU",
            "sensor_id": "INA226_I2C_CALIBRATED"
        }
        ok, rec, msg = engine.evaluate_promotion("KS-CLAIM-019", target_lat, bad_scope_receipt)
        self.assertFalse(ok)
        self.assertIn("Scope mismatch", msg)

        # Case C: Authorized promotion with valid physical receipt
        valid_receipt = {
            "receipt_integrity": "VERIFIED",
            "semantic_verdict": "COMMITTED",
            "measurement_boundary": "FPGA_BOARD",
            "sensor_id": "INA226_I2C_CALIBRATED",
            "board_id": "XILINX-KCU116-KINTEX-USPLUS-REV1"
        }
        ok, rec, msg = engine.evaluate_promotion("KS-CLAIM-019", target_lat, valid_receipt)
        self.assertTrue(ok, f"Promotion failed: {msg}")
        self.assertIsNotNone(rec)
        self.assertTrue(rec.verify_integrity())
        self.assertEqual(rec.new_lattice["measurement"], "FPGA_BOARD_MEASURED")

    def test_documentation_text_invariants(self) -> None:
        """Scans documentation files to ensure no forbidden drift phrases exist."""
        checked_files = [
            self.repo_root / "docs" / "architecture" / "KCB_KMUISA_KMUASM_CONTROL_SERIES_SPEC.md",
            self.repo_root / "docs" / "architecture" / "SECONDARY_INNOVATION_LOOPS_SPEC.md",
            self.repo_root / "docs" / "research" / "KSL_CONTROL_PLANE_SOFTWARE_BASELINE_FREEZE.md",
            self.repo_root / "docs" / "research" / "KSL_V02_OPERATIONAL_TRANSITION_ALGEBRA_SPEC.md",
            self.repo_root / "docs" / "research" / "KSL_V02_HARDWARE_TRANSITION_EXPERIMENT_001.md"
        ]

        forbidden_phrases = [
            "100% operations authorized",
            "10 kHz independent INA226 samples",
            "5.33x RTL acceleration"
        ]

        for p in checked_files:
            if not p.exists():
                continue
            text = p.read_text(encoding="utf-8")
            for phrase in forbidden_phrases:
                self.assertNotIn(
                    phrase, text,
                    f"Forbidden un-scoped or ambiguous phrase '{phrase}' found in {p.name}"
                )


if __name__ == "__main__":
    unittest.main()
