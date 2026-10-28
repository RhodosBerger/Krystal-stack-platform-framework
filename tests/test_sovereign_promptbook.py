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

import hashlib
import os
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
        self.assertEqual(receipt.sequence_number, 1)
        self.assertEqual(receipt.execution_epoch, 1)
        self.assertEqual(receipt.parent_receipt_hash, "0" * 64)
        self.assertTrue(receipt.nonce.startswith("nonce_"))
        self.assertGreater(len(receipt.invariant_set_hash), 0)

    def test_hardening_fail_closed_in_production(self) -> None:
        """Missing secure key in production mode must raise ContractViolation (Fail-Closed)."""
        import os
        from krystal_kernel.sovereign_promptbook import resolve_promptbook_secret_key, KeyProvenance

        old_mode = os.environ.get("KRYSTAL_MODE")
        old_key = os.environ.get("KRYSTAL_PROMPTBOOK_SECRET_KEY")
        try:
            os.environ["KRYSTAL_MODE"] = "production"
            if "KRYSTAL_PROMPTBOOK_SECRET_KEY" in os.environ:
                del os.environ["KRYSTAL_PROMPTBOOK_SECRET_KEY"]

            # Without explicit key or env key, must fail closed
            with self.assertRaises(ContractViolation) as cm:
                resolve_promptbook_secret_key(explicit_key=None, enforce_production_fail_closed=True)
            self.assertIn("Fail-Closed enforced", str(cm.exception))

            # With explicit KMS/TPM key, must succeed with SECURE_EXPLICIT
            explicit_kms_key = b"kms_root_hardware_security_module_key_32b"
            k, prov = resolve_promptbook_secret_key(explicit_key=explicit_kms_key)
            self.assertEqual(k, explicit_kms_key)
            self.assertEqual(prov, KeyProvenance.SECURE_EXPLICIT)
        finally:
            if old_mode is not None:
                os.environ["KRYSTAL_MODE"] = old_mode
            else:
                os.environ.pop("KRYSTAL_MODE", None)
            if old_key is not None:
                os.environ["KRYSTAL_PROMPTBOOK_SECRET_KEY"] = old_key

    def test_hardening_hkdf_domain_separation(self) -> None:
        """Domain-separated keys derived via HKDF-SHA256 must be distinct and non-interchangeable."""
        from krystal_kernel.sovereign_promptbook import hkdf_derive_domain_key
        import hmac
        import hashlib

        root = b"master_cryptographic_root_seed_2026_kspp"
        k_recipe = hkdf_derive_domain_key(root, "krystal-domain-recipe-v1")
        k_render = hkdf_derive_domain_key(root, "krystal-domain-render-v1")
        k_ast = hkdf_derive_domain_key(root, "krystal-domain-ast-v1")
        k_receipt = hkdf_derive_domain_key(root, "krystal-domain-receipt-v1")

        # All derived keys must be distinct 32-byte keys
        keys = [k_recipe, k_render, k_ast, k_receipt]
        self.assertEqual(len(set(keys)), 4)
        for k in keys:
            self.assertEqual(len(k), 32)

        # Proving: Compromise(K_recipe) !implies Forge(TransitionReceipt)
        message = b"canonical_receipt_payload_data_block"
        receipt_sig = hmac.new(k_receipt, message, hashlib.sha256).hexdigest()
        recipe_sig = hmac.new(k_recipe, message, hashlib.sha256).hexdigest()
        self.assertNotEqual(receipt_sig, recipe_sig)

    def test_hardening_rfc8785_canonical_jcs_serialization(self) -> None:
        """RFC 8785 JCS must preserve Unicode verbatim (no NFC/NFD normalization), format floats, and sort keys by UTF-16."""
        from krystal_kernel.sovereign_promptbook import canonical_jcs_serialize

        # 1. Unicode verbatim preservation: RFC 8785 Section 3.2.2 explicitly forbids mutating strings via NFC
        # JCS(NFC(x)) != JCS(x) in general: decomposed and precomposed MUST remain distinct verbatim bytes
        decomposed = "Krystal \u0073\u030c"  # s + combining caron
        precomposed = "Krystal \u0161"        # š
        self.assertNotEqual(canonical_jcs_serialize(decomposed), canonical_jcs_serialize(precomposed))
        self.assertEqual(canonical_jcs_serialize(decomposed), b'"Krystal s\xcc\x8c"')
        self.assertEqual(canonical_jcs_serialize(precomposed), b'"Krystal \xc5\xa1"')

        # 2. Float -0.0 normalization (RFC 8785 Section 3.2.2.3)
        self.assertEqual(canonical_jcs_serialize(-0.0), b"0")
        self.assertEqual(canonical_jcs_serialize(0.0), b"0")

        # 3. Integer-valued float format
        self.assertEqual(canonical_jcs_serialize(1.0), b"1")

        # 4. UTF-16 code unit key sorting (RFC 8785 Section 3.2.3)
        data = {"z": 1, "a": 2, "m": 3}
        self.assertEqual(canonical_jcs_serialize(data), b'{"a":2,"m":3,"z":1}')

        # 5. Whitespace elimination
        data_nested = {"outer": [1, 2, {"inner": "val"}]}
        self.assertEqual(canonical_jcs_serialize(data_nested), b'{"outer":[1,2,{"inner":"val"}]}')

    def test_hardening_receipt_hash_chaining_and_replay_protection(self) -> None:
        """Receipt chain must enforce strict sequence numbers and parent receipt hash-chaining."""
        slots = {
            "goal_description": "Chain Task Alpha",
            "domain_name": "CHAIN_TEST",
            "current_state_json": "{}"
        }
        r1 = self.registry.render_prompt("RECIPE-01-OPERATIONS-DECOMPOSER", slots)
        rcpt1 = self.registry.create_execution_receipt(r1, "0xhash1", "COMMITTED")
        self.assertEqual(rcpt1.sequence_number, 1)
        self.assertEqual(rcpt1.parent_receipt_hash, "0" * 64)

        rcpt2 = self.registry.create_execution_receipt(r1, "0xhash2", "COMMITTED")
        self.assertEqual(rcpt2.sequence_number, 2)
        # rcpt2 parent must equal rcpt1's HMAC signature
        self.assertEqual(rcpt2.parent_receipt_hash, rcpt1.hmac_receipt_signature)

        rcpt3 = self.registry.create_execution_receipt(r1, "0xhash3", "COMMITTED")
        self.assertEqual(rcpt3.sequence_number, 3)
        self.assertEqual(rcpt3.parent_receipt_hash, rcpt2.hmac_receipt_signature)

        # Epoch advance re-anchors the chain
        new_epoch = self.registry.advance_execution_epoch()
        self.assertEqual(new_epoch, 2)
        rcpt_e2 = self.registry.create_execution_receipt(r1, "0xhash4", "COMMITTED")
        self.assertEqual(rcpt_e2.execution_epoch, 2)
        self.assertEqual(rcpt_e2.sequence_number, 1)
        self.assertTrue(rcpt_e2.parent_receipt_hash.startswith("GENESIS_EPOCH_2_"))

    def test_hardening_invariant_definition_versioning_and_identity(self) -> None:
        """InvariantDefinition identity ensures that changing invariant version or severity alters InvariantSetHash."""
        from krystal_kernel.sovereign_promptbook import (
            InvariantDefinition,
            InvariantSeverity,
            compute_invariant_set_hash
        )

        d1 = InvariantDefinition(
            invariant_id="VITAL_MAX_HP == 6",
            version="1.0",
            severity=InvariantSeverity.HARD,
            predicate_digest="digest_hp_v1",
            input_schema={"hp": "int"},
            output_schema={"valid": "bool"},
            description="HP Contract"
        )
        d2 = InvariantDefinition(
            invariant_id="IS_ACYCLIC_DAG",
            version="1.0",
            severity=InvariantSeverity.HARD,
            predicate_digest="digest_dag_v1",
            input_schema={"dag": "list"},
            output_schema={"acyclic": "bool"},
            description="DAG Contract"
        )
        hash_v1 = compute_invariant_set_hash([d1, d2])

        # Version bump on d1
        d1_bumped = InvariantDefinition(
            invariant_id="VITAL_MAX_HP == 6",
            version="2.0",  # Bumped version
            severity=InvariantSeverity.HARD,
            predicate_digest="digest_hp_v1",
            input_schema={"hp": "int"},
            output_schema={"valid": "bool"},
            description="HP Contract"
        )
        hash_v2 = compute_invariant_set_hash([d1_bumped, d2])
        self.assertNotEqual(hash_v1, hash_v2, "InvariantSetHash must change when any invariant version changes")

    def test_hardening_second_validation_gate_untrusted_llm_proposal(self) -> None:
        """Second validation gate checks untrusted LLM output, validates schema, evaluates invariants, and emits trusted AST."""
        slots = {
            "goal_description": "Validate LLM Gate",
            "domain_name": "GATE_TEST",
            "current_state_json": "{}"
        }
        rendered = self.registry.render_prompt("RECIPE-01-OPERATIONS-DECOMPOSER", slots)

        # 1. Valid proposal from untrusted LLM output
        valid_llm_output = {
            "candidate_set": ["SUBTASK_1_SPEC", "SUBTASK_2_EXEC"],
            "resource_budget": {
                "Q_time_ns": 1_000_000.0,
                "Q_memory_bytes": 2048.0,
                "Q_energy_uj": 50.0
            },
            "falsification_oracle": {
                "predicate_id": "IS_ACYCLIC_DAG",
                "expected_result": True,
                "evaluator_rule": "Reject if cycle detected"
            }
        }
        trusted_ast = self.registry.validate_untrusted_llm_proposal(valid_llm_output, rendered)
        self.assertEqual(trusted_ast.proposer_role, "UNTRUSTED_LLM_PROPOSAL_VALIDATED")
        self.assertEqual(trusted_ast.evaluator_role, "AEP_CONSTITUTIONAL_EVALUATOR")
        self.assertEqual(len(trusted_ast.candidate_set), 2)
        self.assertTrue(trusted_ast.falsification_oracle.expected_result)

        # 2. Malformed JSON raises ContractViolation
        with self.assertRaises(ContractViolation) as cm:
            self.registry.validate_untrusted_llm_proposal("INVALID_NOT_JSON", rendered)
        self.assertIn("failed JSON parsing", str(cm.exception))

        # 3. Missing candidate_set raises ContractViolation
        invalid_proposal_missing_candidates = {
            "resource_budget": {"Q_time_ns": 1000.0, "Q_memory_bytes": 1024.0, "Q_energy_uj": 10.0}
        }
        with self.assertRaises(ContractViolation) as cm:
            self.registry.validate_untrusted_llm_proposal(invalid_proposal_missing_candidates, rendered)
        self.assertIn("missing or empty candidate_set", str(cm.exception))

        # 4. Negative quota values raise ContractViolation
        invalid_proposal_negative_quota = {
            "candidate_set": ["OP_1"],
            "resource_budget": {"Q_time_ns": -50.0, "Q_memory_bytes": 1024.0, "Q_energy_uj": 10.0}
        }
        with self.assertRaises(ContractViolation) as cm:
            self.registry.validate_untrusted_llm_proposal(invalid_proposal_negative_quota, rendered)
        self.assertIn("non-positive or invalid budget quota", str(cm.exception))

        # 5. Invariant breach in context raises ContractViolation (e.g. cycle detected)
        with self.assertRaises(ContractViolation) as cm:
            self.registry.validate_untrusted_llm_proposal(
                valid_llm_output,
                rendered,
                execution_context={"cycle_detected": True}  # Triggers IS_ACYCLIC_DAG violation
            )
        self.assertIn("Second validation gate failed", str(cm.exception))

    def test_hardening_falsification_oracle_executable_contract(self) -> None:
        """FalsificationOracleContract evaluates execution contexts and enforces commit or rollback."""
        from krystal_kernel.sovereign_promptbook import FalsificationOracleContract

        oracle = FalsificationOracleContract(
            predicate_id="IS_ACYCLIC_DAG",
            expected_result=True,
            evaluator_rule="Graph must be strictly acyclic",
            evaluator_predicate=lambda ctx: not bool(ctx.get("cycle", False))
        )
        # Passing context: cycle is False -> oracle returns True
        self.assertTrue(oracle.evaluate({"cycle": False}))

        # Failing context: cycle is True -> oracle returns False (blocks commit, Delta S == 0)
        self.assertFalse(oracle.evaluate({"cycle": True}))

    def test_hardening_asymmetric_external_audit_signature(self) -> None:
        """sign_receipt_for_external_audit provides third-party non-repudiation without revealing HMAC secret."""
        slots = {
            "goal_description": "Audit Proof Task",
            "domain_name": "AUDIT_DOMAIN",
            "current_state_json": "{}"
        }
        rendered = self.registry.render_prompt("RECIPE-01-OPERATIONS-DECOMPOSER", slots)
        rcpt = self.registry.create_execution_receipt(rendered, "0xop_audit_hash", "COMMITTED")
        self.assertIsNone(rcpt.external_audit_signature)

        # Mock asymmetric signer simulating Ed25519 audit key
        def mock_ed25519_signer(payload_bytes: bytes) -> tuple[str, str]:
            sig_hex = f"ed25519_sig_{hashlib.sha256(payload_bytes).hexdigest()}"
            return (sig_hex, "ed25519-rfc8032")

        audited_rcpt = self.registry.sign_receipt_for_external_audit(rcpt, mock_ed25519_signer)
        self.assertIsNotNone(audited_rcpt.external_audit_signature)
        self.assertTrue(audited_rcpt.external_audit_signature.startswith("ed25519_sig_"))
        self.assertEqual(audited_rcpt.audit_scheme, "ed25519-rfc8032")
        self.assertEqual(audited_rcpt.receipt_id, rcpt.receipt_id)
        self.assertEqual(audited_rcpt.hmac_receipt_signature, rcpt.hmac_receipt_signature)


if __name__ == "__main__":
    unittest.main()
