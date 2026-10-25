"""
tests/test_sovereign_guidebook_registry.py

Unit and invariant verification suite for SovereignGuidebookRegistry.
Constitutional Invariant: VITAL_MAX_HP == 6
"""

import unittest
from krystal_kernel.constants import VITAL_MAX_HP, ContractViolation
from krystal_kernel.sovereign_guidebook_registry import (
    GuidebookCategory,
    ReadingDifficulty,
    InteractiveSandboxPreset,
    GuidebookSection,
    SovereignGuidebookRegistry,
    GLOBAL_SOVEREIGN_GUIDEBOOK_REGISTRY
)


class TestSovereignGuidebookRegistry(unittest.TestCase):

    def setUp(self) -> None:
        self.registry = SovereignGuidebookRegistry()

    def test_01_registry_initialization_and_count(self) -> None:
        """Verifies canonical sections are loaded and vital_max_hp is preserved."""
        sections = self.registry.list_sections()
        self.assertGreaterEqual(len(sections), 8)
        self.assertEqual(self.registry.vital_max_hp, VITAL_MAX_HP)

    def test_02_section_lookup_and_content(self) -> None:
        """Verifies section lookup by ID retrieves correct metadata and invariants."""
        sec = self.registry.get_section("layer-02-symplectic-integrator")
        self.assertIsNotNone(sec)
        self.assertEqual(sec.category, GuidebookCategory.SOVEREIGN_6_LAYERS)
        self.assertIn("det(J) == 1.000000", sec.constitutional_invariants)
        self.assertIn("symplectic", sec.tags)
        self.assertIsNotNone(sec.code_snippet)

    def test_03_category_filtering(self) -> None:
        """Verifies filtering sections by category works as expected."""
        layers = self.registry.list_sections(category=GuidebookCategory.SOVEREIGN_6_LAYERS)
        self.assertGreaterEqual(len(layers), 6)
        for s in layers:
            self.assertEqual(s.category, GuidebookCategory.SOVEREIGN_6_LAYERS)

    def test_04_fulltext_search(self) -> None:
        """Verifies fulltext search scores titles, tags, and summary matches."""
        results = self.registry.search_sections("PDN")
        self.assertGreaterEqual(len(results), 1)
        self.assertEqual(results[0].section_id, "layer-00-silicon-pdn")

        results_detj = self.registry.search_sections("det(J)")
        self.assertGreaterEqual(len(results_detj), 1)
        self.assertEqual(results_detj[0].section_id, "layer-02-symplectic-integrator")

    def test_05_export_catalog_json(self) -> None:
        """Verifies JSON catalog payload adheres to UI consumption format."""
        catalog = self.registry.export_catalog_json()
        self.assertEqual(catalog["vital_max_hp"], VITAL_MAX_HP)
        self.assertIn("sections", catalog)
        self.assertIn("categories", catalog)
        self.assertEqual(len(catalog["sections"]), len(self.registry.list_sections()))

    def test_06_constitutional_invariant_breach_prevention(self) -> None:
        """Verifies ContractViolation is raised if vital_max_hp != 6 is attempted."""
        with self.assertRaises(ContractViolation):
            SovereignGuidebookRegistry(vital_max_hp=5)

        with self.assertRaises(ContractViolation):
            GuidebookSection(
                section_id="invalid-sec",
                category=GuidebookCategory.ARCHITECTURE_FOUNDATIONS,
                title="Invalid",
                subtitle="Invalid",
                summary_markdown="Invalid",
                reading_difficulty=ReadingDifficulty.INTRODUCTORY,
                order_index=99,
                tags=("invalid",),
                constitutional_invariants=("breach",),
                theory_reference_id=None,
                code_snippet=None,
                vital_max_hp=4
            )


if __name__ == "__main__":
    unittest.main()
