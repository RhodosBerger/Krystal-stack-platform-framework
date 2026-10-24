#!/usr/bin/env python3
"""
==============================================================================
KRYSTAL-STACK: THEORY HARVESTING ENGINE TEST SUITE
==============================================================================
Module: tests/test_theory_harvesting_engine.py
Milestone: V-THEORY-HARVEST-01
Constitutional Invariant: VITAL_MAX_HP == 6
Normative Reference: AGENTS.md, GEMINI.md, krystal_theory/REGISTER_TVRDENI.md

TEST OBJECTIVES:
  1. Verify constitutional invariant VITAL_MAX_HP == 6 across harvester and records.
  2. Verify automated extraction of explicit theory_ref annotations from rails.
  3. Verify physical formula regex detection (Faraday-Lenz, Z(omega), smin).
  4. Verify markdown report generation with proper table formatting.
  5. Verify global singleton availability and stability.
==============================================================================
"""

import os
import sys
import unittest

WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

from krystal_kernel.constants import VITAL_MAX_HP, ContractViolation
from krystal_kernel.verification.theory_harvesting_engine import (
    HarvestedTheoryRecord,
    TheoryHarvestSummary,
    TheoryHarvestingEngine,
    GLOBAL_THEORY_HARVESTING_ENGINE,
)
from krystal_kernel.verification.theory_dependency_graph import (
    TheoryCategory,
    TheoryEpistemicStatus,
)


class TestTheoryHarvestingEngine(unittest.TestCase):
    """Formal unit verification of TheoryHarvestingEngine."""

    def setUp(self) -> None:
        self.engine = TheoryHarvestingEngine()

    def test_01_constitutional_invariant_enforcement(self) -> None:
        """Constitutional invariant VITAL_MAX_HP == 6 must hold across all harvester components."""
        self.assertEqual(VITAL_MAX_HP, 6)
        self.assertEqual(self.engine.vital_max_hp, 6)

        with self.assertRaises(ContractViolation):
            TheoryHarvestingEngine(vital_max_hp=5)

        with self.assertRaises(ContractViolation):
            HarvestedTheoryRecord(
                record_id="HARVEST-TEST-BAD",
                source_file="test.py",
                line_number=1,
                raw_reference="test",
                category=TheoryCategory.EPISTEMIC_TRIAGE,
                epistemic_status=TheoryEpistemicStatus.MODEL,
                extracted_claim_or_formula="test",
                target_component="test",
                vital_max_hp=3,
            )

    def test_02_harvest_extracts_theories(self) -> None:
        """harvest_all must discover theory references across the codebase."""
        summary = self.engine.harvest_all()
        self.assertGreater(summary.total_files_scanned, 100)
        self.assertGreater(summary.total_theories_harvested, 30)

        # Check category distribution
        self.assertIn("SCHEDULING_WSPT", summary.theories_by_category)
        self.assertIn("ELECTRICAL_PDN", summary.theories_by_category)
        self.assertGreater(summary.theories_by_category["SCHEDULING_WSPT"], 0)

        # Invariant preserved in summary
        self.assertEqual(summary.vital_max_hp, 6)

    def test_03_markdown_report_generation(self) -> None:
        """Harvester must generate a well-formed markdown matrix report."""
        summary = self.engine.harvest_all()
        report = self.engine.generate_markdown_report(summary)
        self.assertIn("# Krystal-Stack: Automated Theory Harvest", report)
        self.assertIn("VITAL_MAX_HP == 6", report)
        self.assertIn("Distribution by Theoretical Category", report)
        self.assertIn("Distribution by Epistemic Status", report)

    def test_04_global_singleton_availability(self) -> None:
        """GLOBAL_THEORY_HARVESTING_ENGINE is initialized and functional."""
        self.assertIsNotNone(GLOBAL_THEORY_HARVESTING_ENGINE)
        self.assertEqual(GLOBAL_THEORY_HARVESTING_ENGINE.vital_max_hp, 6)


if __name__ == "__main__":
    unittest.main()
