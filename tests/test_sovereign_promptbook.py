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

    def test_canonical_recipe_json_digest_sensitivity(self) -> None:
        """Modifying any field (system_directive, user_template, etc.) must alter the HMAC digest."""
        import hmac
        import hashlib
        from krystal_kernel.sovereign_promptbook import canonical_recipe_json

        key = b"test_secret_key_1234567890"
        bytes_orig = canonical_recipe_json(
            recipe_id="TEST-01",
            title="Test Recipe",
            category="OPERATIONS_DECOMPOSITION",
            system_directive="Directive A",
            user_template="Template A",
            input_slots=["slot1"],
            bound_invariants=["VITAL_MAX_HP == 6"],
            expected_output_format="JSON",
            falsification_condition="Error",
            schema_version="2.0",
            vital_max_hp=6
        )
        sig_orig = hmac.new(key, bytes_orig, hashlib.sha256).hexdigest()

        # Mutate directive only
        bytes_mutated = canonical_recipe_json(
            recipe_id="TEST-01",
            title="Test Recipe",
            category="OPERATIONS_DECOMPOSITION",
            system_directive="Directive B (Mutated)",
            user_template="Template A",
            input_slots=["slot1"],
            bound_invariants=["VITAL_MAX_HP == 6"],
            expected_output_format="JSON",
            falsification_condition="Error",
            schema_version="2.0",
            vital_max_hp=6
        )
        sig_mutated = hmac.new(key, bytes_mutated, hashlib.sha256).hexdigest()
        self.assertNotEqual(sig_orig, sig_mutated, "Digest must change when system_directive changes")

    def test_rendered_prompt_content_digest_sensitivity(self) -> None:
        """RenderedPrompt digest must change when text content changes, even if length is identical."""
        slots1 = {
            "goal_description": "GoalAAAA",
            "domain_name": "DOMAIN_A",
            "current_state_json": '{"k":"v1"}'
        }
        slots2 = {
            "goal_description": "GoalBBBB",  # Exactly same character length as GoalAAAA
            "domain_name": "DOMAIN_A",
            "current_state_json": '{"k":"v1"}'
        }
        r1 = self.registry.render_prompt("RECIPE-01-OPERATIONS-DECOMPOSER", slots1)
        r2 = self.registry.render_prompt("RECIPE-01-OPERATIONS-DECOMPOSER", slots2)
        self.assertEqual(len(r1.user_text), len(r2.user_text))
        self.assertNotEqual(r1.hmac_digest, r2.hmac_digest, "Digest must differ even when text length is identical")

    def test_constraint_plane_executable_invariants_precondition(self) -> None:
        """Constraint Plane must block rendering with ContractViolation if a HARD invariant fails."""
        # RECIPE-04 has UMA_PRESSURE < 0.78 invariant bound
        # High UMA pressure (0.95 >= 0.90 ceiling) must trigger invariant violation
        violating_slots = {
            "active_slots_list": "0x00, 0x01, 0x02",
            "sampled_uma": 0.95,
            "current_uma": 0.95,
            "sampled_temp": 45.0,
            "is_vblank": True,
            "architecture_profile_json": "{}"
        }
        with self.assertRaises(ContractViolation) as cm:
            self.registry.render_prompt("RECIPE-04-UMA-HEX-INSPECTOR-GOVERNOR", violating_slots)
        self.assertIn("Constraint Plane precondition failed", str(cm.exception))

        # Safe UMA pressure (0.42 < 0.90 ceiling) must pass
        safe_slots = {
            "active_slots_list": "0x00, 0x01, 0x02",
            "sampled_uma": 0.42,
            "current_uma": 0.42,
            "sampled_temp": 45.0,
            "is_vblank": True,
            "architecture_profile_json": "{}"
        }
        rendered = self.registry.render_prompt("RECIPE-04-UMA-HEX-INSPECTOR-GOVERNOR", safe_slots)
        self.assertIsNotNone(rendered)

    def test_operational_ast_proposal_compilation(self) -> None:
        """compile_to_operational_ast produces typed AST proposal maintaining Proposer != Evaluator."""
        slots = {
            "goal_description": "Execute KSL AST bridge validation",
            "domain_name": "COMPILER_FRONTEND",
            "current_state_json": "{}"
        }
        rendered = self.registry.render_prompt("RECIPE-01-OPERATIONS-DECOMPOSER", slots)
        ast_prop = self.registry.compile_to_operational_ast(rendered)
        self.assertEqual(ast_prop.recipe_id, "RECIPE-01-OPERATIONS-DECOMPOSER")
        self.assertEqual(ast_prop.proposer_role, "LLM_PROMPTBOOK_COMPILER_FRONTEND")
        self.assertEqual(ast_prop.evaluator_role, "AEP_CONSTITUTIONAL_EVALUATOR")
        self.assertIn("VITAL_MAX_HP == 6", ast_prop.constraint_set)
        self.assertIn("Q_time_ns", ast_prop.resource_budget)
        self.assertEqual(len(ast_prop.hmac_ast_signature), 64)

    def test_execution_receipt_provenance_chain(self) -> None:
        """create_execution_receipt links recipe digest, rendered prompt digest, and operation hash."""
        slots = {
            "goal_description": "Test Provenance Chain",
            "domain_name": "TEST_DOMAIN",
            "current_state_json": "{}"
        }
        rendered = self.registry.render_prompt("RECIPE-01-OPERATIONS-DECOMPOSER", slots)
        receipt = self.registry.create_execution_receipt(
            rendered_prompt=rendered,
            operation_hash="0xabcd1234deadbeef00112233445566778899aabbccddeeff",
            status="COMMITTED"
        )
        self.assertEqual(receipt.recipe_id, rendered.recipe_id)
        self.assertEqual(receipt.recipe_digest, rendered.recipe_digest)
        self.assertEqual(receipt.rendered_prompt_digest, rendered.hmac_digest)
        self.assertEqual(receipt.operation_hash, "0xabcd1234deadbeef00112233445566778899aabbccddeeff")
        self.assertTrue(receipt.invariants_verified)
        self.assertEqual(len(receipt.hmac_receipt_signature), 64)


if __name__ == "__main__":
    unittest.main()
