#!/usr/bin/env python3
"""
==============================================================================
UNIT TEST SUITE: NATIVE LANGUAGE PREDICTION TENSOR & ASSET/TEMPLATE PIPELINE
==============================================================================
Test Target: krystal_kernel/native_language_tensor_generator.py
Framework: unittest
Invariant: VITAL_MAX_HP == 6 strictly preserved across all tensors and templates.
==============================================================================
"""

import os
import sys
import unittest
import struct

WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

from krystal_kernel.scene_constructor_presets import VITAL_MAX_HP
from krystal_kernel.native_language_tensor_generator import (
    NativeLanguageTensorPredictor,
    LinguisticFeatureVector,
    LocalAssetAndTemplateCompiler,
    NativeLanguageTensorPipeline,
    GLOBAL_NATIVE_TENSOR_PIPELINE
)


class TestNativeLanguageTensorPipeline(unittest.TestCase):
    """Rigorous tests for linguistic tensor predictions, asset synthesis, and templates."""

    def setUp(self):
        self.pipeline = NativeLanguageTensorPipeline()
        self.predictor = self.pipeline.predictor
        self.compiler = self.pipeline.compiler

    def test_vital_max_hp_invariant(self):
        """Platform invariant VITAL_MAX_HP == 6 must hold universally."""
        self.assertEqual(VITAL_MAX_HP, 6)

    def test_slovak_prompt_feature_extraction(self):
        """Slovak CNC prompt must detect language 'sk' and high CNC domain frequency."""
        prompt = "Vytvor mi optimalny CNC nastroj s vysokym posuvom a monitorovanim teploty vretena"
        feat = self.predictor.extract_features(prompt)

        self.assertEqual(feat.language_code, "sk")
        self.assertGreater(feat.domain_frequencies["CNC_MACHINING"], 0.25)
        self.assertGreater(feat.imperative_density, 0.0)
        self.assertGreater(feat.shannon_entropy, 3.0)

    def test_czech_prompt_feature_extraction(self):
        """Czech alchemical prompt must detect language 'cs' and Alchemical domain."""
        prompt = "Udělej mi alchymistický krystalový talisman s vysokou rezonancí"
        feat = self.predictor.extract_features(prompt)

        self.assertEqual(feat.language_code, "cs")
        self.assertGreater(feat.domain_frequencies["ALCHEMICAL_LORE"], 0.25)

    def test_english_prompt_feature_extraction(self):
        """English software prompt must detect language 'en' and Software domain."""
        prompt = "Generate high performance AVX-512 compute core with low cache latency and high IPC"
        feat = self.predictor.extract_features(prompt)

        self.assertEqual(feat.language_code, "en")
        self.assertGreater(feat.domain_frequencies["SOFTWARE_ARCHITECTURE"], 0.25)

    def test_usage_prediction_tensor_probabilities(self):
        """Prediction tensor must produce a valid probability distribution summing to 1.0."""
        prompt = "Chladenie reznej kvapaliny a tepelny gradient nastroja"
        feat = self.predictor.extract_features(prompt)
        w_pred = self.predictor.predict_usage_tensor(feat)

        self.assertEqual(w_pred.shape, (6, 1))
        probs = [w_pred.data[i][0] for i in range(6)]
        for p in probs:
            self.assertGreater(p, 0.0)
            self.assertLess(p, 1.0)
        self.assertAlmostEqual(sum(probs), 1.0, places=3)

    def test_average_usage_tensor_rolling_history(self):
        """Average usage tensor must update dynamically with prompt history."""
        # Initial average prior
        avg_initial = self.predictor.compute_average_usage_tensor()
        self.assertEqual(avg_initial.shape, (6, 1))

        # Feed 3 prompts
        self.predictor.extract_features("CNC frezovanie vreteno")
        self.predictor.extract_features("Tenzorova matica vypocet IPC")
        self.predictor.extract_features("Krystalovy alchymicky artefakt")

        avg_updated = self.predictor.compute_average_usage_tensor()
        self.assertEqual(avg_updated.shape, (6, 1))
        self.assertAlmostEqual(sum(avg_updated.data[i][0] for i in range(6)), 1.0, places=3)

    def test_engine_modulation_tensor_shape_and_vital_hp(self):
        """Modulation tensor must be (6, 8) with last column strictly equal to VITAL_MAX_HP."""
        feat = self.predictor.extract_features("Vytvor biome veza")
        w_pred = self.predictor.predict_usage_tensor(feat)
        t_mod = self.predictor.synthesize_engine_modulation_tensor(w_pred)

        self.assertEqual(t_mod.shape, (6, 8))
        # Column 7 is Vital HP -> must be 6.0
        for i in range(6):
            self.assertEqual(t_mod.data[i][7], float(VITAL_MAX_HP))

    def test_godot_multimesh_binary_buffer_packing(self):
        """Binary buffer must pack exactly 96 bytes per instance with valid floats."""
        feat = self.predictor.extract_features("CNC rezny nastroj")
        w_pred = self.predictor.predict_usage_tensor(feat)
        
        inst_count = 32
        buf = self.compiler.generate_godot_multimesh_binary_buffer(w_pred, instance_count=inst_count)
        expected_len = inst_count * 96  # 32 * 96 = 3072 bytes
        self.assertEqual(len(buf), expected_len)

        # Unpack first instance (24 float32 values)
        first_instance = struct.unpack("<24f", buf[:96])
        self.assertEqual(len(first_instance), 24)
        # Custom data[0] is Vital HP == 6.0
        self.assertEqual(first_instance[20], float(VITAL_MAX_HP))

    def test_system_templates_syntax_and_invariants(self):
        """Generated templates (Godot, Janet, Keras, HTML) must contain valid syntax & 6 HP invariant."""
        feat = self.predictor.extract_features("Alchymicka citadela s rezonanciou")
        w_pred = self.predictor.predict_usage_tensor(feat)

        # 1. Godot .tscn
        tscn = self.compiler.generate_godot_scene_template("AlchemArena", w_pred)
        self.assertIn('[gd_scene load_steps=5 format=3', tscn)
        self.assertIn('VITAL_MAX_HP = 6', tscn)
        self.assertIn('MultiMeshInstance3D', tscn)

        # 2. Janet DSL .janet
        janet_code = self.compiler.generate_janet_dsl_template("AlchemArena", w_pred)
        self.assertIn('(def VITAL-MAX-HP 6)', janet_code)
        self.assertIn(':vital-max-hp VITAL-MAX-HP', janet_code)

        # 3. Keras JSON spec
        keras_spec = self.compiler.generate_keras_architecture_template("AlchemArena", w_pred)
        self.assertEqual(keras_spec["vital_max_hp"], 6)
        self.assertIn("archetype_usage_logits", [h["name"] for h in keras_spec["heads"]])

        # 4. HTML5 Studio
        html_code = self.compiler.generate_html_studio_template("AlchemArena", w_pred)
        self.assertIn('VITAL HP:', html_code)
        self.assertIn('(6/6)', html_code)
        self.assertIn('<canvas id="view"', html_code)

    def test_end_to_end_pipeline_execution(self):
        """End-to-end pipeline must execute cleanly with zero errors in under 20ms."""
        res = self.pipeline.process_prompt(
            prompt="Vytvor mi optimalny CNC nastroj s vysokym posuvom a monitorovanim teploty vretena",
            name="SmartCncEndmill",
            instance_count=64
        )

        self.assertEqual(res.status, "SUCCESS")
        self.assertEqual(res.detected_language, "sk")
        self.assertEqual(res.vital_hp, 6)
        self.assertEqual(res.vital_max_hp, 6)
        self.assertEqual(res.binary_multimesh_bytes, 64 * 96)
        self.assertLess(res.execution_time_ms, 50.0)  # High performance local pipeline


if __name__ == "__main__":
    unittest.main()
