#!/usr/bin/env python3
"""
==============================================================================
KRYSTAL-STACK: UNIT TEST SUITE
FACTOGRAPHIC SOURCE NETWORK & MULTI-SCALE LEARNING ENGINE
==============================================================================
Module: tests/test_factographic_network_learning.py
==============================================================================
"""

import time
import unittest

from krystal_kernel.scene_constructor_presets import VITAL_MAX_HP
from krystal_kernel.verification.epistemic_value import ValidityState, EvidenceKind
from krystal_kernel.factographic_network_learning_engine import (
    FactographicScale,
    RelationType,
    FactographicSourceNode,
    FactographicRelationEdge,
    FactographicSourceNetwork,
    FactographicLearningCalibrator,
    GLOBAL_FACTOGRAPHIC_LEARNING_ENGINE
)


class TestFactographicNetworkLearning(unittest.TestCase):

    def setUp(self):
        self.network = FactographicSourceNetwork()
        self.calibrator = FactographicLearningCalibrator(self.network)

    def test_canonical_nodes_and_scales_registered(self):
        """All 4 canonical scales (micro, meso, macro, ontological) exist in network."""
        nodes = self.network.list_nodes()
        self.assertGreaterEqual(len(nodes), 8)

        scales = {n.scale for n in nodes}
        self.assertIn(FactographicScale.MICRO_SCALE, scales)
        self.assertIn(FactographicScale.MESO_SCALE, scales)
        self.assertIn(FactographicScale.MACRO_SCALE, scales)
        self.assertIn(FactographicScale.ONTOLOGICAL_SCALE, scales)

    def test_consensus_evaluation_nominal(self):
        """Under nominal conditions without divergence, network consensus is > 0.85."""
        consensus = self.network.evaluate_consensus()
        self.assertGreater(consensus.overall_consensus_score, 0.85)
        self.assertEqual(len(consensus.active_contradictions), 0)
        self.assertEqual(consensus.vital_hp_invariant, VITAL_MAX_HP)

    def test_cross_scale_contradiction_detection(self):
        """Injecting sensor divergence along a corroborating edge triggers contradiction penalty."""
        # Inject severe divergence on CPU CoreTemp (normal is ~51.2°C; inject 92.0°C)
        self.network.update_node_value("SRC-MICRO-CORETEMP", 92.0)
        
        consensus = self.network.evaluate_consensus()
        self.assertGreater(len(consensus.active_contradictions), 0)
        contra = consensus.active_contradictions[0]
        self.assertEqual(contra["edge_id"], "EDGE-CORETEMP-DRIFT")
        self.assertGreater(contra["divergence_pct"], contra["tolerance_pct"])
        # Consensus score must drop
        self.assertLess(consensus.overall_consensus_score, 0.85)

    def test_multi_scale_feature_pyramid(self):
        """Feature pyramid captures micro, meso, and macro scale vectors."""
        pyramid = self.calibrator.sample_pyramid()
        self.assertGreater(len(pyramid.micro_vector), 0)
        self.assertGreater(len(pyramid.meso_vector), 0)
        self.assertGreater(len(pyramid.macro_vector), 0)
        self.assertGreater(pyramid.cross_scale_coupling_ratio, 0.0)

    def test_self_calibration_loss_reduction(self):
        """Self-calibration adjusts learning rates and decreases loss."""
        rec = self.calibrator.calibrate()
        self.assertLessEqual(rec.post_calibration_loss, rec.pre_calibration_loss)
        self.assertGreater(rec.learning_rate_micro, 0.0)
        self.assertGreater(rec.learning_rate_meso, 0.0)
        self.assertGreater(rec.learning_rate_macro, 0.0)
        self.assertGreater(rec.prediction_horizon_adjusted_ms, 500.0)

    def test_family_s_invariant_protection(self):
        """Feature Family S: Learning cannot mutate authority or violate VITAL_MAX_HP == 6."""
        rec = self.calibrator.calibrate(mock_divergence_injection=88.5)
        self.assertEqual(rec.invariant_status, f"PROTECTED_VITAL_HP_{VITAL_MAX_HP}")
        self.assertFalse(rec.authority_mutation_attempted)
        self.assertEqual(VITAL_MAX_HP, 6)

    def test_global_singleton_status(self):
        """Global singleton returns structured engine status dictionary."""
        status = GLOBAL_FACTOGRAPHIC_LEARNING_ENGINE.get_status()
        self.assertTrue(status["success"])
        self.assertEqual(status["engine"], "FactographicNetworkLearningEngine")
        self.assertEqual(status["vital_max_hp_rule"], 6)
        self.assertIn("consensus", status)
        self.assertIn("pyramid", status)


if __name__ == "__main__":
    unittest.main()
