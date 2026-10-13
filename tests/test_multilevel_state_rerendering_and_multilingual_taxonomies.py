#!/usr/bin/env python3
"""
==============================================================================
KRYSTAL-STACK: UNIT TESTS FOR MULTI-LEVEL STATE RE-RENDERING & MULTILINGUAL TAXONOMIES
==============================================================================
File: tests/test_multilevel_state_rerendering_and_multilingual_taxonomies.py
Covers:
  - 5-Tier State Hierarchy (Silicon, Pipeline, Epistemic, Linguistic, Visual)
  - Hierarchical dirty-state cascade propagation (Level 0 through Level 4)
  - Multilingual Language Locales (SK, CS, EN, DE, LA, JA)
  - Linguistic Registers & Semantic Intent classification
  - Lexicon bridge and localized status re-synthesis
  - Rendered ASCII HUD formatting with constitutional invariant
  - Constitutional Invariant: VITAL_MAX_HP == 6
==============================================================================
"""

import unittest
from krystal_kernel.scene_constructor_presets import VITAL_MAX_HP
from krystal_kernel.multilevel_state_rerendering_and_multilingual_taxonomies import (
    SystemStateLevelTaxon,
    LanguageLocaleTaxon,
    LinguisticRegisterTaxon,
    SemanticIntentTaxon,
    MULTILINGUAL_LEXICON,
    MultiLevelSystemState,
    MultiLevelReRenderReport,
    MultilingualIntentClassifier,
    MultiLevelStateReRenderingEngine,
    GLOBAL_MULTILEVEL_STATE_RERENDERING_ENGINE
)


class TestMultiLevelStateReRenderingAndMultilingualTaxonomies(unittest.TestCase):
    """Validation test suite for multi-level state re-rendering and multilingual taxonomies."""

    def setUp(self):
        self.engine = MultiLevelStateReRenderingEngine()

    def test_constitutional_invariant(self):
        """Verify that VITAL_MAX_HP == 6 holds strictly across all structures."""
        self.assertEqual(VITAL_MAX_HP, 6)
        report = self.engine.trigger_hierarchical_rerender(SystemStateLevelTaxon.LEVEL_0_HARDWARE_SILICON)
        self.assertEqual(report.vital_max_hp, 6)
        self.assertIn("6/6 HP", report.rendered_ascii_hud)

    def test_state_level_taxons_completeness(self):
        """Verify all 5 hierarchical state levels are defined."""
        self.assertEqual(len(SystemStateLevelTaxon), 5)
        self.assertIn(SystemStateLevelTaxon.LEVEL_0_HARDWARE_SILICON, SystemStateLevelTaxon)
        self.assertIn(SystemStateLevelTaxon.LEVEL_1_COMPUTE_PIPELINE, SystemStateLevelTaxon)
        self.assertIn(SystemStateLevelTaxon.LEVEL_2_EPISTEMIC_AUTHORITY, SystemStateLevelTaxon)
        self.assertIn(SystemStateLevelTaxon.LEVEL_3_SEMANTIC_LINGUISTIC, SystemStateLevelTaxon)
        self.assertIn(SystemStateLevelTaxon.LEVEL_4_VISUAL_COMPOSITOR, SystemStateLevelTaxon)

    def test_multilingual_locales_and_registers(self):
        """Verify all 6 language locales and 4 registers are defined."""
        self.assertEqual(len(LanguageLocaleTaxon), 6)
        locales = [LanguageLocaleTaxon.SK, LanguageLocaleTaxon.CS, LanguageLocaleTaxon.EN,
                   LanguageLocaleTaxon.DE, LanguageLocaleTaxon.LA, LanguageLocaleTaxon.JA]
        for loc in locales:
            self.assertIn(loc, LanguageLocaleTaxon)

        self.assertEqual(len(LinguisticRegisterTaxon), 4)
        self.assertEqual(len(SemanticIntentTaxon), 5)

    def test_multilingual_intent_classifier(self):
        """Verify language locale detection and semantic intent parsing."""
        # Locale Detection
        self.assertEqual(MultilingualIntentClassifier.detect_locale("prekresli stav systému"), LanguageLocaleTaxon.SK)
        self.assertEqual(MultilingualIntentClassifier.detect_locale("překresli stav"), LanguageLocaleTaxon.CS)
        self.assertEqual(MultilingualIntentClassifier.detect_locale("rerender system status"), LanguageLocaleTaxon.EN)
        self.assertEqual(MultilingualIntentClassifier.detect_locale("neuzeichnen zustand"), LanguageLocaleTaxon.DE)
        self.assertEqual(MultilingualIntentClassifier.detect_locale("reddere statusque"), LanguageLocaleTaxon.LA)
        self.assertEqual(MultilingualIntentClassifier.detect_locale("システム再描画"), LanguageLocaleTaxon.JA)

        # Intent Classification
        self.assertEqual(MultilingualIntentClassifier.classify_intent("prekresli viewport"), SemanticIntentTaxon.RE_RENDER_TRIGGER)
        self.assertEqual(MultilingualIntentClassifier.classify_intent("spusti workload"), SemanticIntentTaxon.COMMAND_ACTUATE)
        self.assertEqual(MultilingualIntentClassifier.classify_intent("aký je stav pamäte"), SemanticIntentTaxon.TELEMETRY_INSPECT)
        self.assertEqual(MultilingualIntentClassifier.classify_intent("skontroluj invariant a audit"), SemanticIntentTaxon.GOVERNANCE_AUDIT)
        self.assertEqual(MultilingualIntentClassifier.classify_intent("simuluj alternatívnu hypotézu"), SemanticIntentTaxon.HYPOTHESIS_EXPLORE)

    def test_hierarchical_rerender_cascade_levels(self):
        """Verify dirty level propagates downwards to all subsequent levels."""
        # Level 0 dirty -> all 5 levels affected
        rep0 = self.engine.trigger_hierarchical_rerender(SystemStateLevelTaxon.LEVEL_0_HARDWARE_SILICON)
        self.assertEqual(len(rep0.affected_levels_cascaded), 5)
        self.assertIn(SystemStateLevelTaxon.LEVEL_0_HARDWARE_SILICON, rep0.affected_levels_cascaded)
        self.assertIn(SystemStateLevelTaxon.LEVEL_4_VISUAL_COMPOSITOR, rep0.affected_levels_cascaded)

        # Level 3 dirty -> only Level 3 and Level 4 affected
        rep3 = self.engine.trigger_hierarchical_rerender(SystemStateLevelTaxon.LEVEL_3_SEMANTIC_LINGUISTIC)
        self.assertEqual(len(rep3.affected_levels_cascaded), 2)
        self.assertIn(SystemStateLevelTaxon.LEVEL_3_SEMANTIC_LINGUISTIC, rep3.affected_levels_cascaded)
        self.assertIn(SystemStateLevelTaxon.LEVEL_4_VISUAL_COMPOSITOR, rep3.affected_levels_cascaded)
        self.assertNotIn(SystemStateLevelTaxon.LEVEL_0_HARDWARE_SILICON, rep3.affected_levels_cascaded)

    def test_multilingual_localization_coverage(self):
        """Verify each of the 6 languages produces localized output."""
        for loc in LanguageLocaleTaxon:
            rep = self.engine.trigger_hierarchical_rerender(
                dirty_level=SystemStateLevelTaxon.LEVEL_2_EPISTEMIC_AUTHORITY,
                target_locale=loc
            )
            self.assertEqual(rep.target_locale, loc)
            self.assertGreater(len(rep.localized_status_text), 0)
            self.assertGreater(len(rep.localized_authority_banner), 0)
            self.assertGreater(len(rep.rendered_ascii_hud), 0)
            self.assertIn(f"[{loc.value}]", rep.rendered_ascii_hud)

    def test_elevated_thermal_warning_localization(self):
        """Verify elevated thermal triggers localized hardware warnings."""
        self.engine.update_hardware_state(thermal=82.0, power=35.0, isa="AVX512_VNNI")

        rep_sk = self.engine.trigger_hierarchical_rerender(
            dirty_level=SystemStateLevelTaxon.LEVEL_0_HARDWARE_SILICON,
            target_locale=LanguageLocaleTaxon.SK
        )
        self.assertIn("Varovanie", rep_sk.localized_silicon_summary)

        rep_en = self.engine.trigger_hierarchical_rerender(
            dirty_level=SystemStateLevelTaxon.LEVEL_0_HARDWARE_SILICON,
            target_locale=LanguageLocaleTaxon.EN
        )
        self.assertIn("Warning", rep_en.localized_silicon_summary)

    def test_report_serialization(self):
        """Verify dictionary serialization matches schema requirements."""
        rep = self.engine.trigger_hierarchical_rerender(
            dirty_level=SystemStateLevelTaxon.LEVEL_1_COMPUTE_PIPELINE,
            target_locale=LanguageLocaleTaxon.SK
        )
        data = rep.to_dict()
        self.assertEqual(data["dirty_trigger_level"], "LEVEL_1_COMPUTE_PIPELINE")
        self.assertEqual(data["target_locale"], "SK")
        self.assertEqual(data["vital_max_hp"], 6)
        self.assertIn("rendered_ascii_hud", data)

    def test_global_singleton(self):
        """Verify the global singleton is operational."""
        self.assertIsNotNone(GLOBAL_MULTILEVEL_STATE_RERENDERING_ENGINE)
        rep = GLOBAL_MULTILEVEL_STATE_RERENDERING_ENGINE.trigger_hierarchical_rerender(
            SystemStateLevelTaxon.LEVEL_3_SEMANTIC_LINGUISTIC
        )
        self.assertEqual(rep.vital_max_hp, 6)


if __name__ == "__main__":
    unittest.main()
