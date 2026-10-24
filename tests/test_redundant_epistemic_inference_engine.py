"""
tests/test_redundant_epistemic_inference_engine.py

Unit test suite verifying redundant epistemic multi-lens inference across
all project aspects, definitions, visions, and roadmap milestones.
Strictly enforces VITAL_MAX_HP == 6 and Delta S == 0 on violation.
"""

import unittest
from krystal_kernel.constants import VITAL_MAX_HP, ContractViolation
from krystal_kernel.redundant_epistemic_inference_engine import (
    AspectCategory,
    InferenceLens,
    InferenceEpistemicStatus,
    AspectDefinition,
    EpistemicInferenceClaim,
    RedundantInferenceSynthesis,
    RedundantEpistemicInferenceEngine,
    GLOBAL_REDUNDANT_INFERENCE_ENGINE
)


class TestRedundantEpistemicInferenceEngine(unittest.TestCase):
    """Test cases for RedundantEpistemicInferenceEngine."""

    def setUp(self) -> None:
        self.engine = RedundantEpistemicInferenceEngine()

    def test_engine_initialization_and_catalog(self) -> None:
        """Verifies that all 15 aspects are cataloged across all 7 categories."""
        aspect_ids = self.engine.list_aspect_ids()
        self.assertEqual(len(aspect_ids), 15)

        categories_represented = {self.engine.get_aspect(a_id).category for a_id in aspect_ids} # type: ignore
        self.assertEqual(len(categories_represented), len(AspectCategory))
        self.assertEqual(self.engine.vital_max_hp, VITAL_MAX_HP)

    def test_constitutional_invariant_strictness(self) -> None:
        """Ensures that any tampering with VITAL_MAX_HP raises ContractViolation."""
        with self.assertRaises(ContractViolation):
            RedundantEpistemicInferenceEngine(vital_max_hp=5)

        with self.assertRaises(ContractViolation):
            AspectDefinition(
                aspect_id="INVALID_ASPECT",
                name="Invalid Aspect",
                category=AspectCategory.CONSTITUTIONAL_KERNEL,
                primary_definition="Test",
                governing_invariant="None",
                empirical_reports=(),
                roadmap_phase="TEST",
                vital_max_hp=7
            )

        with self.assertRaises(ContractViolation):
            EpistemicInferenceClaim(
                claim_id="INVALID_CLAIM",
                aspect_id="ASPECT_P1_P10_SOVEREIGNTY",
                lens=InferenceLens.FORMAL_INVARIANT,
                inferred_statement="Test",
                status=InferenceEpistemicStatus.MATHEMATICALLY_DERIVED,
                evidence_basis="None",
                falsifier="None",
                confidence=1.0,
                vital_max_hp=0
            )

    def test_infer_single_aspect_across_all_five_lenses(self) -> None:
        """Verifies that a single aspect produces claims from all 5 distinct lenses."""
        synthesis = self.engine.infer_aspect_across_lenses("ASPECT_HET_CROSSOVER_PHASE")
        self.assertIsInstance(synthesis, RedundantInferenceSynthesis)
        self.assertEqual(len(synthesis.claims_by_lens), 5)

        expected_lenses = {
            InferenceLens.FORMAL_INVARIANT,
            InferenceLens.EMPIRICAL_SILICON,
            InferenceLens.CYBERNETIC_FEEDBACK,
            InferenceLens.ADVERSARIAL_PARANOIA,
            InferenceLens.EMERGENT_TELEOLOGY
        }
        self.assertEqual(set(synthesis.claims_by_lens.keys()), expected_lenses)

        for lens, claim in synthesis.claims_by_lens.items():
            self.assertEqual(claim.lens, lens)
            self.assertGreaterEqual(claim.confidence, 0.90)
            self.assertEqual(claim.vital_max_hp, VITAL_MAX_HP)
            self.assertTrue(len(claim.falsifier) > 0)

    def test_cross_lens_agreement_and_soundness(self) -> None:
        """Verifies that cross-lens agreement meets or exceeds the 80% consensus threshold."""
        synthesis = self.engine.infer_aspect_across_lenses("ASPECT_CHRONO_SYMPLECTIC_SOC")
        self.assertGreaterEqual(synthesis.cross_lens_agreement, 0.80)
        self.assertTrue(synthesis.is_constitutionally_sound)
        self.assertEqual(synthesis.vital_max_hp, VITAL_MAX_HP)
        self.assertIn("compute density", synthesis.systemic_synergy)

    def test_infer_all_aspects_coverage(self) -> None:
        """Verifies that infer_all_aspects runs over all 15 aspects and produces 75 claims."""
        syntheses = self.engine.infer_all_aspects()
        self.assertEqual(len(syntheses), 15)

        total_claims = sum(len(s.claims_by_lens) for s in syntheses.values())
        self.assertEqual(total_claims, 75)

        for aspect_id, synthesis in syntheses.items():
            self.assertTrue(synthesis.is_constitutionally_sound)
            self.assertEqual(synthesis.vital_max_hp, VITAL_MAX_HP)

    def test_compute_global_synergy_matrix(self) -> None:
        """Verifies the global cross-domain synergy matrix calculation."""
        matrix = self.engine.compute_global_synergy_matrix()
        self.assertEqual(matrix["total_aspects_evaluated"], 15)
        self.assertEqual(matrix["total_epistemic_claims_generated"], 75)
        self.assertEqual(matrix["constitutionally_sound_aspects"], 15)
        self.assertEqual(matrix["soundness_percentage"], 100.0)
        self.assertTrue(matrix["vital_max_hp_verified"])
        self.assertGreaterEqual(len(matrix["cross_domain_links"]), 5)

    def test_immutable_claim_integrity(self) -> None:
        """Ensures that EpistemicInferenceClaim instances cannot be modified post-creation."""
        synthesis = self.engine.infer_aspect_across_lenses("ASPECT_SUB50NS_PULSE_LOGGER")
        claim = synthesis.claims_by_lens[InferenceLens.EMPIRICAL_SILICON]
        with self.assertRaises(AttributeError):
            claim.confidence = 0.5  # type: ignore

    def test_unknown_aspect_rejection_delta_s_zero(self) -> None:
        """Ensures that requesting inference on an invalid aspect raises ContractViolation and does not alter catalog."""
        catalog_before = len(self.engine.list_aspect_ids())
        with self.assertRaises(ContractViolation):
            self.engine.infer_aspect_across_lenses("NON_EXISTENT_ASPECT_XYZ")
        catalog_after = len(self.engine.list_aspect_ids())
        self.assertEqual(catalog_before, catalog_after)


if __name__ == "__main__":
    unittest.main()
