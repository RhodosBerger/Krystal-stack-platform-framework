#!/usr/bin/env python3
"""
Unit tests for Radius Derivative vs Cosine Topology Engine.
"""

import unittest
import math
from krystal_kernel.radius_derivative_vs_cosine_topology import (
    CosineTopologyHarmonic,
    CosineTopologyParameters,
    RadiusDerivativeVsCosineTopologyComparator,
    GLOBAL_TOPOLOGY_COMPARATOR,
    VITAL_MAX_HP
)


class TestRadiusDerivativeVsCosineTopology(unittest.TestCase):
    def setUp(self):
        self.topology = CosineTopologyParameters.create_canonical(base_r=100.0, lobes=4)
        self.comparator = RadiusDerivativeVsCosineTopologyComparator(self.topology)

    def test_canonical_parameters_creation(self):
        self.assertEqual(self.topology.base_radius, 100.0)
        self.assertEqual(len(self.topology.harmonics), 3)
        self.assertEqual(self.topology.vital_max_hp, 6)

    def test_cosine_topology_radius_evaluation(self):
        r_0 = self.topology.evaluate_radius(0.0)
        # At theta=0, all cos terms contribute positively
        self.assertGreater(r_0, self.topology.base_radius)

    def test_radius_derivative_orthogonality(self):
        # In a single harmonic r(theta) = R_0 + A * cos(omega * theta):
        # dr/dtheta at theta = 0 must be 0 (derivative of cos at 0 is -sin(0) = 0)
        single_h = CosineTopologyParameters(
            base_radius=50.0,
            harmonics=[CosineTopologyHarmonic(harmonic_order=1, amplitude=10.0, frequency_multiplier=2.0, phase_rad=0.0)]
        )
        dr_0 = single_h.evaluate_radius_first_derivative(0.0)
        self.assertAlmostEqual(dr_0, 0.0, places=5)

        # Peak derivative occurs at theta = pi / (2 * omega) = pi / 4
        dr_peak = single_h.evaluate_radius_first_derivative(math.pi / 4.0)
        self.assertAlmostEqual(abs(dr_peak), 20.0, places=4)

    def test_curvature_and_normal_vector_normalization(self):
        kappa = self.topology.evaluate_curvature(0.5)
        self.assertGreater(kappa, 0.0)

        x, y, nx, ny = self.topology.get_cartesian_point_and_normal(1.0)
        norm_len = math.sqrt(nx * nx + ny * ny)
        self.assertAlmostEqual(norm_len, 1.0, places=3)

    def test_steepness_ratio_and_undercut_detection(self):
        xi = self.topology.evaluate_steepness_ratio(0.2)
        self.assertGreaterEqual(xi, 0.0)

    def test_comparator_profile_and_analytical_report(self):
        report = self.comparator.generate_analytical_comparison()
        self.assertIn("metrics", report)
        self.assertIn("comparison_matrix", report)
        self.assertIn("interplay_synthesis", report)
        self.assertEqual(report["vital_max_hp"], 6)

    def test_global_singleton(self):
        samples = GLOBAL_TOPOLOGY_COMPARATOR.sample_profile(sample_count=12)
        self.assertEqual(len(samples), 12)
        self.assertEqual(samples[0]["vital_max_hp_rule"], 6)


if __name__ == '__main__':
    unittest.main()
