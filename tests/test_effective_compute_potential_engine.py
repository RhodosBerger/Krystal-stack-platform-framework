"""
Unit Test Suite for Effective Compute Potential (ECP) & Effective Hardware Unlock (EHU).

Verification of Theory Node 52:
THEORY-EFFECTIVE-COMPUTE-POTENTIAL-AND-HARDWARE-UNLOCK
Constitutional Invariant: VITAL_MAX_HP == 6
"""

import unittest
from krystal_kernel.effective_compute_potential_engine import (
    EcpMultiFactorDecomposition,
    EhuDimensionBreakdown,
    LatencyProfile,
    EcpAuditReceipt,
    EffectiveComputePotentialEngine,
    GLOBAL_ECP_ENGINE,
)
from krystal_kernel.verification.theory_dependency_graph import GLOBAL_THEORY_DEPENDENCY_GRAPH
from krystal_kernel.constants import VITAL_MAX_HP


class TestEffectiveComputePotentialEngine(unittest.TestCase):
    """Test suite for Theory Node 52 verification."""

    def setUp(self):
        self.engine = EffectiveComputePotentialEngine()

    def test_theory_node_52_registration(self):
        """Verify that Theory Node 52 is registered in the global dependency graph without cycles."""
        theory_id = "THEORY-EFFECTIVE-COMPUTE-POTENTIAL-AND-HARDWARE-UNLOCK"
        self.assertIn(theory_id, GLOBAL_THEORY_DEPENDENCY_GRAPH._nodes)
        node = GLOBAL_THEORY_DEPENDENCY_GRAPH._nodes[theory_id]
        self.assertEqual(node.epistemic_status.name, "DOKAZANE")
        self.assertFalse(GLOBAL_THEORY_DEPENDENCY_GRAPH.has_cycles())

    def test_ecp_multi_factor_calculation(self):
        """Test Effective Compute Potential calculation across 6 governing factors."""
        factors = EcpMultiFactorDecomposition(
            peak_capability_raw=100.0,
            utilization_factor=0.70,
            scheduling_efficiency=0.85,
            memory_efficiency=0.75,
            thermal_stability=0.80,
            prediction_quality=0.92,
        )
        # Expected: 100 * 0.70 * 0.85 * 0.75 * 0.80 * 0.92 = 32.844
        expected = round(100.0 * 0.70 * 0.85 * 0.75 * 0.80 * 0.92, 3)
        self.assertEqual(factors.effective_potential, expected)

    def test_cpg_gap_calculation_and_unlock(self):
        """Test Compute Potential Gap (CPG) calculation: CPG = 1 - (Observed / Potential)."""
        # Baseline: Potential = 100, Observed = 62 => CPG = 38%
        cpg_base = self.engine.compute_cpg(observed_useful_work=62.0, estimated_effective_potential=100.0)
        self.assertEqual(cpg_base, 0.38)

        # Adaptive: Potential = 100, Observed = 78 => CPG = 22%
        cpg_adapt = self.engine.compute_cpg(observed_useful_work=78.0, estimated_effective_potential=100.0)
        self.assertEqual(cpg_adapt, 0.22)

        # Capacity unlocked: 38% - 22% = 16 percentage points
        unlocked_points = round((cpg_base - cpg_adapt) * 100.0, 1)
        self.assertEqual(unlocked_points, 16.0)

    def test_ehu_effective_hardware_unlock_evaluation(self):
        """Test Effective Hardware Unlock (EHU) calculation: EHU = (Adaptive / Baseline) - 1."""
        # Baseline = 100, Adaptive = 124 => EHU = +24%
        ehu = self.engine.compute_ehu(useful_perf_baseline=100.0, useful_perf_adaptive=124.0)
        self.assertEqual(ehu, 24.0)

    def test_full_workload_transition_audit_and_receipt(self):
        """Test full closed-loop workload transition evaluation and cryptographic receipt generation."""
        baseline_factors = EcpMultiFactorDecomposition(
            peak_capability_raw=100.0,
            utilization_factor=0.60,
            scheduling_efficiency=0.75,
            memory_efficiency=0.65,
            thermal_stability=0.70,
            prediction_quality=0.75,
        )
        adaptive_factors = EcpMultiFactorDecomposition(
            peak_capability_raw=100.0,
            utilization_factor=0.78,
            scheduling_efficiency=0.90,
            memory_efficiency=0.82,
            thermal_stability=0.88,
            prediction_quality=0.95,
        )
        baseline_lat = LatencyProfile(p50_us=450.0, p95_us=1200.0, p99_us=2400.0, jitter_us=180.0, deadline_miss_rate=0.045)
        adaptive_lat = LatencyProfile(p50_us=380.0, p95_us=920.0, p99_us=1450.0, jitter_us=65.0, deadline_miss_rate=0.008)

        receipt = self.engine.evaluate_workload_transition(
            workload_id="WORKLOAD-GEMM-AVX512-IRISXE",
            baseline_factors=baseline_factors,
            adaptive_factors=adaptive_factors,
            baseline_energy_joules=100.0,
            adaptive_energy_joules=82.0,
            baseline_latency=baseline_lat,
            adaptive_latency=adaptive_lat,
        )

        self.assertIsInstance(receipt, EcpAuditReceipt)
        self.assertEqual(receipt.vital_max_hp, 6)
        self.assertTrue(receipt.is_verified)
        self.assertGreater(receipt.ehu_breakdown.ehu_total, 0.0)
        self.assertGreater(receipt.ehu_breakdown.ehu_energy, 20.0) # ~ 21.95%
        self.assertGreater(receipt.ehu_breakdown.ehu_latency, 20.0) # ~ 23.3% P95 reduction
        self.assertGreater(receipt.cpg_reduction_points, 0.0)
        self.assertEqual(len(receipt.hmac_digest), 64)


if __name__ == "__main__":
    unittest.main()
