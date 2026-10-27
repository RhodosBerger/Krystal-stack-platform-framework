#!/usr/bin/env python3
"""
==============================================================================
KRYSTAL-STACK: UNIT TESTS FOR DATACENTER POWER CASCADE SIMULATOR
==============================================================================
Module: tests/test_datacenter_power_cascade_simulator.py
Description:
    Comprehensive unit tests for the 5 000 GPU & AMD EPYC datacenter power model:
      - Baseline power and economic calculations
      - Conservative, Balanced, and Experimental adaptive regimes
      - Marginal Power Utility (MPU) calculation
      - Phase-Aware Power Orchestration (P3 Chronometric Multiplexing)
      - Intra-Rack Power Budget Arbitrage (SavedPower_A -> UsefulWork_B)
      - Cryptographic Datacenter Economic Receipts and Invariant VITAL_MAX_HP == 6.

Author: Dušan Kopecký & Krystal-Stack Architecture Council (2026)
==============================================================================
"""

import unittest
from krystal_kernel.constants import VITAL_MAX_HP, ContractViolation
from krystal_kernel.datacenter_power_cascade_simulator import (
    PowerRegime,
    PhaseType,
    DatacenterTopology,
    DatacenterPowerMetrics,
    DatacenterEconomicReceipt,
    DatacenterPowerCascadeSimulator,
    GLOBAL_DATACENTER_POWER_CASCADE_SIMULATOR
)


class TestDatacenterPowerCascadeSimulator(unittest.TestCase):
    """Test suite for the 5 000 GPU & AMD EPYC datacenter power cascade engine."""

    def setUp(self) -> None:
        self.simulator = GLOBAL_DATACENTER_POWER_CASCADE_SIMULATOR
        self.topology = self.simulator.topology

    def test_topology_dimensions(self) -> None:
        """Verifies physical hardware dimensions: 625 nodes, 5000 GPUs, 625 EPYC CPUs."""
        self.assertEqual(self.topology.node_count, 625)
        self.assertEqual(self.topology.gpus_per_node, 8)
        self.assertEqual(self.topology.total_gpus, 5000)
        self.assertEqual(self.topology.total_cpus, 625)
        self.assertEqual(self.topology.pue, 1.20)
        self.assertEqual(self.topology.electricity_price_eur_kwh, 0.12)

    def test_baseline_metrics_and_economics(self) -> None:
        """Verifies exact Baseline regime power load and financial costs."""
        metrics = self.simulator.calculate_regime_metrics(PowerRegime.BASELINE)

        self.assertEqual(metrics.regime, PowerRegime.BASELINE)
        self.assertEqual(metrics.gpu_total_mw, 1.800)
        self.assertEqual(metrics.cpu_total_mw, 0.1875)
        self.assertEqual(metrics.other_total_mw, 0.125)
        self.assertEqual(metrics.it_load_mw, 2.1125)
        self.assertEqual(metrics.facility_power_mw, 2.535)

        # Financial costs
        self.assertEqual(metrics.cost_per_hour_eur, 304.20)
        self.assertEqual(metrics.cost_per_day_eur, 7300.80)
        self.assertEqual(metrics.cost_per_year_eur, 2664792.00)
        self.assertEqual(metrics.model_savings_per_year_eur, 0.0)
        self.assertEqual(metrics.delta_percentage, 0.0)
        self.assertEqual(metrics.normalized_unit_cost, 1.0)
        self.assertEqual(metrics.vital_max_hp, VITAL_MAX_HP)

    def test_conservative_adaptive_regime(self) -> None:
        """Verifies Conservative Adaptive regime (-8.28% facility power reduction)."""
        metrics = self.simulator.calculate_regime_metrics(PowerRegime.CONSERVATIVE_ADAPTIVE)

        self.assertEqual(metrics.gpu_total_mw, 1.650)
        self.assertEqual(metrics.cpu_total_mw, 0.16875)
        self.assertEqual(metrics.other_total_mw, 0.11875)
        self.assertEqual(metrics.it_load_mw, 1.9375)
        self.assertEqual(metrics.facility_power_mw, 2.325)

        self.assertEqual(metrics.cost_per_hour_eur, 279.00)
        self.assertEqual(metrics.cost_per_year_eur, 2444040.00)
        self.assertEqual(metrics.model_savings_per_year_eur, 220752.00)
        self.assertEqual(metrics.delta_percentage, -8.28)
        self.assertAlmostEqual(metrics.normalized_unit_cost, 0.9172, places=3)

    def test_balanced_adaptive_regime(self) -> None:
        """Verifies Balanced Adaptive regime (-16.57% facility power reduction)."""
        metrics = self.simulator.calculate_regime_metrics(PowerRegime.BALANCED_ADAPTIVE)

        self.assertEqual(metrics.gpu_total_mw, 1.500)
        self.assertEqual(metrics.cpu_total_mw, 0.150)
        self.assertEqual(metrics.other_total_mw, 0.1125)
        self.assertEqual(metrics.it_load_mw, 1.7625)
        self.assertEqual(metrics.facility_power_mw, 2.115)

        self.assertEqual(metrics.cost_per_hour_eur, 253.80)
        self.assertEqual(metrics.cost_per_year_eur, 2223288.00)
        self.assertEqual(metrics.model_savings_per_year_eur, 441504.00)
        self.assertEqual(metrics.delta_percentage, -16.57)
        self.assertAlmostEqual(metrics.normalized_unit_cost, 0.8343, places=3)

    def test_experimental_efficiency_regime(self) -> None:
        """Verifies Experimental Efficiency Envelope (-24.41% facility power reduction)."""
        metrics = self.simulator.calculate_regime_metrics(PowerRegime.EXPERIMENTAL_EFFICIENCY)

        self.assertEqual(metrics.gpu_total_mw, 1.350)
        self.assertEqual(metrics.cpu_total_mw, 0.140625)
        self.assertEqual(metrics.other_total_mw, 0.10625)
        self.assertEqual(metrics.it_load_mw, 1.596875)
        self.assertEqual(metrics.facility_power_mw, 1.91625)

        self.assertEqual(metrics.cost_per_hour_eur, 229.95)
        self.assertEqual(metrics.cost_per_year_eur, 2014362.00)
        self.assertEqual(metrics.model_savings_per_year_eur, 650430.00)
        self.assertEqual(metrics.delta_percentage, -24.41)
        self.assertAlmostEqual(metrics.normalized_unit_cost, 0.7559, places=3)

    def test_marginal_power_utility_evaluation(self) -> None:
        """Verifies calculation of Marginal Power Utility (MPU = Delta W / Delta Joules)."""
        # Step 1: 270W -> 300W produces +0.07 useful work units per second
        mpu_low = self.simulator.calculate_marginal_power_utility(
            current_watts=270.0,
            proposed_watts=300.0,
            current_work=0.90,
            proposed_work=0.97,
            delta_time_s=1.0
        )
        self.assertGreater(mpu_low, 0.0)

        # Step 2: 330W -> 360W produces only +0.02 useful work units per second
        mpu_high = self.simulator.calculate_marginal_power_utility(
            current_watts=330.0,
            proposed_watts=360.0,
            current_work=1.00,
            proposed_work=1.02,
            delta_time_s=1.0
        )
        # Diminishing returns: MPU at top power is significantly lower than at mid power
        self.assertLess(mpu_high, mpu_low)

    def test_chronometric_phase_power_orchestration(self) -> None:
        """Verifies phase-aware power allocation (CPU, GPU, Fabric)."""
        # Phase A: CPU Preprocessing -> GPU should be throttled
        gpu_a, cpu_a, fab_a = self.simulator.orchestrate_phase_power(PhaseType.CPU_PREPROCESSING)
        self.assertEqual(gpu_a, 80.0)
        self.assertEqual(cpu_a, 300.0)

        # Phase C: GPU Compute -> GPU at peak compute, CPU throttled
        gpu_c, cpu_c, fab_c = self.simulator.orchestrate_phase_power(PhaseType.GPU_COMPUTE)
        self.assertEqual(gpu_c, 360.0)
        self.assertEqual(cpu_c, 120.0)

        # Phase D: Barrier Synchronization -> GPU throttled, Fabric active
        gpu_d, cpu_d, fab_d = self.simulator.orchestrate_phase_power(PhaseType.COLLECTIVE_SYNC)
        self.assertEqual(gpu_d, 90.0)
        self.assertEqual(fab_d, 120.0)

    def test_intra_rack_power_arbitrage(self) -> None:
        """Verifies rack-level power reallocation from memory-bound to compute-bound GPUs."""
        # 32 GPU rack: 16 memory-bound, 16 compute-bound
        res = self.simulator.arbitrate_rack_power(
            memory_bound_gpus=16,
            compute_bound_gpus=16,
            total_rack_budget_w=11520.0
        )
        self.assertTrue(res["within_budget"])
        self.assertEqual(res["mem_gpu_power_w"], 270.0)
        self.assertGreater(res["compute_gpu_power_w"], 360.0)
        self.assertEqual(res["reallocated_watts"], 1440.0)  # 16 * 90W
        self.assertLessEqual(res["total_consumed_w"], 11520.0)
        self.assertEqual(res["vital_max_hp"], VITAL_MAX_HP)

    def test_economic_receipt_generation_and_invariants(self) -> None:
        """Verifies generation of HMAC-SHA256 authenticated DatacenterEconomicReceipt."""
        rcpt = self.simulator.generate_economic_receipt(
            regime=PowerRegime.BALANCED_ADAPTIVE,
            duration_hours=2.0,
            verified_work_units=10_000_000.0,
            workload_class="TRANSFORMER_AUTOREGRESSIVE_INFERENCE"
        )

        self.assertEqual(rcpt.active_gpus, 5000)
        self.assertEqual(rcpt.active_nodes, 625)
        self.assertEqual(rcpt.operating_regime, PowerRegime.BALANCED_ADAPTIVE)
        self.assertEqual(rcpt.facility_power_mw, 2.115)
        self.assertAlmostEqual(rcpt.energy_cost_eur, 507.60, places=2)  # 253.80 * 2
        self.assertAlmostEqual(rcpt.cost_per_million_units_eur, 50.76, places=2)
        self.assertEqual(len(rcpt.receipt_hmac_sha256), 64)
        self.assertEqual(rcpt.vital_max_hp, VITAL_MAX_HP)

    def test_constitutional_invariant_rejection(self) -> None:
        """Verifies simulator rejects initialization with vital_max_hp != 6."""
        with self.assertRaises(ContractViolation):
            DatacenterPowerCascadeSimulator(vital_max_hp=5)


if __name__ == "__main__":
    unittest.main()
