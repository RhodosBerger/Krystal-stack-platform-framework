#!/usr/bin/env python3
"""
================================================================================
UNIT TESTS: SOVEREIGN OPERATIONAL PROMPTBOOK ENGINE
================================================================================
Module:     tests/test_sovereign_promptbook.py
Author:     Antigravity Cognitive Invariants Group (2026)
Invariants: VITAL_MAX_HP == 6 | Proposer != Evaluator | Delta S == 0 on breach
================================================================================
"""

import unittest
from krystal_kernel.constants import VITAL_MAX_HP, ContractViolation
from krystal_kernel.sovereign_promptbook import (
    PromptbookCategory,
    PromptbookRecipe,
    RenderedPrompt,
    SovereignPromptbookRegistry,
    GLOBAL_SOVEREIGN_PROMPTBOOK,
)


class TestSovereignPromptbookRegistry(unittest.TestCase):
    """Verifies recipe registry, input slot rendering, and invariants."""

    def setUp(self) -> None:
        self.registry = SovereignPromptbookRegistry(vital_max_hp=VITAL_MAX_HP)

    def test_registry_contains_todays_seven_inventions(self) -> None:
        """The registry must contain all 7 foundational invention recipes."""
        self.assertEqual(len(self.registry.recipes), 7)
        expected_ids = [
            "RECIPE-01-OPERATIONS-DECOMPOSER",
            "RECIPE-02-BOT-AXIOM-COMPOSER",
            "RECIPE-03-AGILE-TURN-ARBITRATOR",
            "RECIPE-04-UMA-HEX-INSPECTOR-GOVERNOR",
            "RECIPE-05-MICRO-OP-KNAPSACK-BOT",
            "RECIPE-06-BEHAVIORAL-VULKAN-SYNTHESIS",
            "RECIPE-07-RUNTIME-CONTROL-PLANE-BENCHMARK",
        ]
        for rid in expected_ids:
            recipe = self.registry.get_recipe(rid)
            self.assertIsNotNone(recipe, f"Missing recipe: {rid}")
            self.assertEqual(recipe.vital_max_hp, VITAL_MAX_HP)
            self.assertEqual(len(recipe.hmac_signature), 64)

    def test_filter_by_category(self) -> None:
        """Filtering by category returns matching subset."""
        decomp_recipes = self.registry.list_recipes_by_category(PromptbookCategory.OPERATIONS_DECOMPOSITION)
        self.assertEqual(len(decomp_recipes), 1)
        self.assertEqual(decomp_recipes[0].recipe_id, "RECIPE-01-OPERATIONS-DECOMPOSER")

    def test_render_prompt_successful_interpolation(self) -> None:
        """Rendering a recipe with all required slots produces valid RenderedPrompt."""
        slots = {
            "goal_description": "Run M4.1 Physical Validation Benchmark",
            "domain_name": "SCIENTIFIC_EVIDENCE",
            "current_state_json": '{"temperature_c": 45.0, "uma_pressure": 0.35}'
        }
        rendered = self.registry.render_prompt("RECIPE-01-OPERATIONS-DECOMPOSER", slots)
        self.assertEqual(rendered.recipe_id, "RECIPE-01-OPERATIONS-DECOMPOSER")
        self.assertIn("Run M4.1 Physical Validation Benchmark", rendered.user_text)
        self.assertIn("SCIENTIFIC_EVIDENCE", rendered.user_text)
        self.assertGreater(rendered.rendered_at_ns, 0)
        self.assertEqual(len(rendered.hmac_digest), 64)
        self.assertEqual(rendered.vital_max_hp, VITAL_MAX_HP)

    def test_render_prompt_missing_slot_raises_contract_violation(self) -> None:
        """Missing a required slot must raise ContractViolation."""
        incomplete_slots = {
            "goal_description": "Run M4.1"
            # Missing domain_name and current_state_json
        }
        with self.assertRaises(ContractViolation):
            self.registry.render_prompt("RECIPE-01-OPERATIONS-DECOMPOSER", incomplete_slots)

    def test_render_prompt_unknown_recipe_raises_contract_violation(self) -> None:
        """Requesting an unknown recipe ID must raise ContractViolation."""
        with self.assertRaises(ContractViolation):
            self.registry.render_prompt("RECIPE-UNKNOWN-99", {})

    def test_markdown_summary_export(self) -> None:
        """Markdown export generates comprehensive summary with table and directives."""
        md = self.registry.export_promptbook_markdown_summary()
        self.assertIn("# KRYSTAL STACK: SOVEREIGN OPERATIONAL PROMPTBOOK", md)
        self.assertIn("RECIPE-01-OPERATIONS-DECOMPOSER", md)
        self.assertIn("RECIPE-06-BEHAVIORAL-VULKAN-SYNTHESIS", md)
        self.assertIn("VITAL_MAX_HP == 6", md)

    def test_constitutional_invariant_breach_prevention(self) -> None:
        """Initializing with VITAL_MAX_HP != 6 must raise ContractViolation."""
        with self.assertRaises(ContractViolation):
            SovereignPromptbookRegistry(vital_max_hp=5)


if __name__ == "__main__":
    unittest.main()
