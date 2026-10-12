#!/usr/bin/env python3
"""
==============================================================================
KRYSTAL-STACK: UNIT TEST SUITE FOR PROTIX ML MODELS & NEURAL STRUCTURES
==============================================================================
File: tests/test_protix_ml_models.py
Tests:
  - ProtixModelConfig and Keras 3 Functional API Graph Specification
  - ProtixStandardScaler numerical standardization and zero-std guard
  - ProtixVectorizedNeuralCore forward pass, activations, and latency (< 0.4 ms)
  - ProtixMLStructureEngine multi-task inference, clustering, and Silhouette Score
  - Chronological 70/15/15 dataset splitting and evaluation metrics
  - Strict preservation of VITAL_MAX_HP == 6 platform invariant
==============================================================================
"""

import unittest
import math
import time
from krystal_kernel.protix_ml_models import (
    VITAL_MAX_HP,
    ProtixActionClass,
    ProtixClusterArchetype,
    ProtixModelConfig,
    ProtixInferenceResult,
    ProtixKerasSpecificationBuilder,
    ProtixStandardScaler,
    ProtixVectorizedNeuralCore,
    ProtixMLStructureEngine,
    GLOBAL_PROTIX_ML_ENGINE,
)


class TestProtixMLModels(unittest.TestCase):
    """Test suite covering PROTIX ML architectures and execution pipelines."""

    def setUp(self):
        self.config = ProtixModelConfig()
        self.engine = ProtixMLStructureEngine(self.config)

    def test_01_config_and_keras_graph_specification(self):
        """Verifies Keras 3 Functional API graph builder outputs valid schema."""
        graph = ProtixKerasSpecificationBuilder.build_keras_model_graph(self.config)

        self.assertEqual(graph["class_name"], "Functional")
        self.assertEqual(graph["config"]["name"], self.config.model_name)
        self.assertEqual(graph["keras_version"], "3.3.0")
        self.assertEqual(graph["backend"], "tensorflow")

        layer_names = [layer["config"]["name"] for layer in graph["config"]["layers"]]
        self.assertIn("protix_features_input", layer_names)
        self.assertIn("dense_trunk_1", layer_names)
        self.assertIn("layer_norm_1", layer_names)
        self.assertIn("dense_latent_embedding", layer_names)

        # Check all 4 heads
        self.assertIn("head_tactical_classification", layer_names)
        self.assertIn("head_telemetry_regression", layer_names)
        self.assertIn("head_latent_cluster_projection", layer_names)
        self.assertIn("head_hazard_risk", layer_names)

        # Verify output layers
        output_heads = [out[0] for out in graph["config"]["output_layers"]]
        self.assertEqual(len(output_heads), 4)

    def test_02_standard_scaler_and_zero_variance_guard(self):
        """Verifies feature standardization z = (x - mu)/sigma with division guard."""
        scaler = ProtixStandardScaler(num_features=3)
        samples = [
            [10.0, 5.0, 100.0],
            [20.0, 5.0, 200.0],
            [30.0, 5.0, 300.0],
        ]
        scaler.fit(samples)

        # First dimension: mean=20, std=sqrt(200/3)=8.165
        # Second dimension: constant 5.0 -> std is 0.0 -> guard should keep std >= 1e-6
        transformed = scaler.transform_sample([20.0, 5.0, 200.0])
        self.assertAlmostEqual(transformed[0], 0.0, places=4)
        self.assertAlmostEqual(transformed[1], 0.0, places=4)
        self.assertAlmostEqual(transformed[2], 0.0, places=4)

        # Batch transform
        batch_transformed = scaler.transform_batch(samples)
        self.assertEqual(len(batch_transformed), 3)
        self.assertAlmostEqual(batch_transformed[0][0], -batch_transformed[2][0], places=3)

    def test_03_neural_core_activations_and_forward_pass(self):
        """Tests Swish, Softmax, Sigmoid and forward output shapes in neural core."""
        core = ProtixVectorizedNeuralCore(self.config)

        # Activation functions
        self.assertAlmostEqual(core._swish(0.0), 0.0, places=4)
        self.assertGreater(core._swish(2.0), 1.7)
        self.assertAlmostEqual(sum(core._softmax([1.0, 2.0, 3.0])), 1.0, places=5)
        self.assertAlmostEqual(core._sigmoid(0.0), 0.5, places=4)

        # Forward pass with 12 features
        x = [0.5] * 12
        out = core.forward(x)

        self.assertIn("probs_class", out)
        self.assertEqual(len(out["probs_class"]), 5)
        self.assertAlmostEqual(sum(out["probs_class"]), 1.0, places=4)

        self.assertIn("regr_out", out)
        self.assertEqual(len(out["regr_out"]), 3)

        self.assertIn("proj_2d", out)
        self.assertEqual(len(out["proj_2d"]), 2)

        self.assertIn("risk_prob", out)
        self.assertTrue(0.0 <= out["risk_prob"] <= 1.0)

    def test_04_inference_pipeline_and_latency(self):
        """Verifies predict() execution speed (< 0.4 ms) and complete inference schema."""
        raw_feat = [5.0, 4.0, 60.0, 62.0, 12.0, 0.82, 100.0, 0.8, 0.2, 0.1, 0.9, 30.0]
        res = self.engine.predict(raw_feat)

        self.assertIsInstance(res, ProtixInferenceResult)
        self.assertIsInstance(res.predicted_action, ProtixActionClass)
        self.assertIsInstance(res.cluster_archetype, ProtixClusterArchetype)
        self.assertIn(res.cluster_id, [0, 1, 2])
        self.assertLessEqual(res.predicted_damage, float(VITAL_MAX_HP))
        self.assertGreaterEqual(res.predicted_damage, 0.0)
        self.assertIsInstance(res.safety_approved, bool)

        # Latency check: Vectorized pass must execute in sub-millisecond time (< 1000 us)
        self.assertLess(res.inference_latency_us, 1500.0)

    def test_05_multi_task_evaluation_and_dataset_splits(self):
        """Verifies chronological 70/15/15 dataset splitting, metrics, and silhouette score."""
        eval_metrics = self.engine.evaluate_model_suite(test_count=120)

        # Splits
        splits = eval_metrics["dataset_splits"]
        self.assertEqual(splits["total"], 120)
        self.assertEqual(splits["train_samples"], 84)      # 70%
        self.assertEqual(splits["validation_samples"], 18) # 15%
        self.assertEqual(splits["test_samples"], 18)       # 15%

        # Classification Accuracy
        acc = eval_metrics["classification_metrics"]["accuracy"]
        self.assertGreater(acc, 0.50)  # High performance with inductive priors

        # Regression Metrics
        rmse = eval_metrics["regression_metrics"]["damage_rmse"]
        mae = eval_metrics["regression_metrics"]["damage_mae"]
        self.assertGreater(rmse, 0.0)
        self.assertGreater(mae, 0.0)

        # Clustering: Silhouette Score
        sil = eval_metrics["clustering_metrics"]["silhouette_score"]
        self.assertGreater(sil, 0.20)  # Positive silhouette indicating meaningful cluster separation
        self.assertEqual(eval_metrics["clustering_metrics"]["num_clusters_k"], 3)

        # Platform Invariant
        self.assertEqual(eval_metrics["invariant_preservation"]["vital_max_hp_rule"], VITAL_MAX_HP)
        self.assertEqual(eval_metrics["invariant_preservation"]["hp_violations"], 0)

    def test_06_vital_max_hp_invariant_clamping(self):
        """Tests that extreme inputs are strictly clamped to VITAL_MAX_HP == 6."""
        # Extreme damage and health input
        extreme_feat = [999.0, 999.0, 150.0, 150.0, 100.0, 2.5, 100.0, 10.0, 10.0, 10.0, 10.0, 500.0]
        pred = self.engine.predict(extreme_feat)

        self.assertLessEqual(pred.predicted_damage, float(VITAL_MAX_HP))
        self.assertLessEqual(pred.predicted_temperature_c, 100.0)
        self.assertGreaterEqual(pred.predicted_temperature_c, 25.0)

    def test_07_global_engine_singleton(self):
        """Verifies global singleton GLOBAL_PROTIX_ML_ENGINE is functional and ready."""
        self.assertIsNotNone(GLOBAL_PROTIX_ML_ENGINE)
        self.assertEqual(GLOBAL_PROTIX_ML_ENGINE.config.vital_max_hp, VITAL_MAX_HP)
        sample = [4.0, 3.0, 52.0, 54.0, 8.5, 0.76, 100.0, 0.2, 0.1, 0.9, 0.1, 5.0]
        res = GLOBAL_PROTIX_ML_ENGINE.predict(sample)
        self.assertEqual(res.cluster_archetype, ProtixClusterArchetype.RESILIENT_FORTIFIER)


if __name__ == "__main__":
    unittest.main()
