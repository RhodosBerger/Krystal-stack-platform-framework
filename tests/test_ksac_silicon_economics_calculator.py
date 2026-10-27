#!/usr/bin/env python3
"""
==============================================================================
UNIT TESTS: KSAC SILICON ECONOMICS v2 AUDITED CALCULATOR
==============================================================================
Tests verification for:
  - Silicon die area calculations across TSMC 7nm, TSMC 16nm, GF 22FDX
  - FPGA resource utilization (LUTs, FFs, BRAMs, DSPs)
  - Latency separation: T_core (16 ns) vs T_system (24 ns) & SRAM bandwidth
  - Incremental Control Energy (E_decision = 2.88 nJ) and EDP
  - Verified Hardware Leverage (VHL = 1.4852)
  - Audited Murphy Random Yield (~99.4%) and Compound Yield (~89.5%)
  - Prototype NRE budget and industrial customer payback
  - Theory Node 55 registration in TheoryDependencyGraph
==============================================================================
"""

import unittest
from krystal_kernel.ksac_silicon_economics_calculator import (
    SiliconProcessNode,
    KsacAreaCalculator,
    KsacYieldCalculator,
    KsacSystemPerformanceCalculator,
    KsacFinancialCalculator,
    GLOBAL_KSAC_AREA_CALCULATOR,
    GLOBAL_KSAC_YIELD_CALCULATOR,
    GLOBAL_KSAC_SYSTEM_PERFORMANCE_CALCULATOR,
    GLOBAL_KSAC_FINANCIAL_CALCULATOR,
)
from krystal_kernel.verification.theory_dependency_graph import (
    GLOBAL_THEORY_DEPENDENCY_GRAPH,
    TheoryEpistemicStatus,
)


class TestKsacSiliconEconomicsCalculator(unittest.TestCase):
    """Test suite for audited KSAC silicon economics, area, latency, yield and VHL."""

    def setUp(self) -> None:
        self.area_calc = GLOBAL_KSAC_AREA_CALCULATOR
        self.yield_calc = GLOBAL_KSAC_YIELD_CALCULATOR
        self.perf_calc = GLOBAL_KSAC_SYSTEM_PERFORMANCE_CALCULATOR
        self.fin_calc = GLOBAL_KSAC_FINANCIAL_CALCULATOR

    def test_silicon_area_across_nodes(self) -> None:
        """Verifies area scaling across TSMC 7nm, TSMC 16nm, and GF 22FDX."""
        area_7nm = self.area_calc.calculate_asic_area(SiliconProcessNode.TSMC_7NM)
        area_16nm = self.area_calc.calculate_asic_area(SiliconProcessNode.TSMC_16NM)
        area_22fdx = self.area_calc.calculate_asic_area(SiliconProcessNode.GF_22FDX)

        self.assertLess(area_7nm.die_area_with_margin_mm2, area_16nm.die_area_with_margin_mm2)
        self.assertLess(area_16nm.die_area_with_margin_mm2, area_22fdx.die_area_with_margin_mm2)

        # 16nm die area falls in the range ~5 to 15 mm^2 (exact ~7.14 mm^2)
        self.assertGreater(area_16nm.die_area_with_margin_mm2, 5.0)
        self.assertLess(area_16nm.die_area_with_margin_mm2, 15.0)

        # 768 KiB SRAM Scratchpad area must be positive and non-zero
        self.assertGreater(area_16nm.sram_scratchpad_768k_mm2, 0.3)
        self.assertLess(area_16nm.sram_scratchpad_768k_mm2, 1.5)

    def test_fpga_resource_utilization(self) -> None:
        """Verifies feasibility on Xilinx Kintex UltraScale+ FPGA."""
        fpga_res = self.area_calc.calculate_fpga_utilization()
        self.assertLess(fpga_res["lut_pct"], 25.0) # < 25% LUTs
        self.assertLess(fpga_res["bram_pct"], 35.0) # < 35% BRAMs
        self.assertEqual(fpga_res["vital_max_hp"], 6)

    def test_audited_wafer_dpw_and_compound_yield(self) -> None:
        """Verifies audited DPW formula and separation of random defect vs compound yield."""
        report = self.yield_calc.calculate_yield_and_die_cost(SiliconProcessNode.TSMC_16NM)
        # DPW on 300mm wafer with A=7.139 mm^2 is ~9652
        self.assertGreaterEqual(report.gross_dies_per_wafer, 9500)
        self.assertLessEqual(report.gross_dies_per_wafer, 9800)

        # Murphy Random Defect Yield is very high for small die (~99.43%)
        self.assertGreater(report.murphy_random_yield_pct, 99.0)

        # Compound Manufacturing Yield (SRAM, Parametric, Edge) is ~89.5%
        self.assertGreater(report.compound_manufacturing_yield_pct, 85.0)
        self.assertLess(report.compound_manufacturing_yield_pct, 95.0)

        # Good dies per wafer
        self.assertGreater(report.good_dies_per_wafer, 8000)

        # Unit costs
        self.assertLess(report.raw_die_cost_eur, 1.00) # Under €1 raw die in 16nm volume
        self.assertLess(report.packaged_and_tested_cost_eur, 15.00)

    def test_latency_separation_and_bandwidth(self) -> None:
        """Verifies separation of T_core pipeline vs T_system and SRAM bandwidth."""
        perf = self.perf_calc.calculate_system_performance(fabric_clock_mhz=500.0)
        self.assertEqual(perf.t_core_pipeline_ns, 16.0) # Exactly 8 cycles @ 500 MHz
        self.assertEqual(perf.t_system_end_to_end_us, 0.024)
        self.assertGreater(perf.speedup_core_vs_software_x, 5000.0)
        self.assertEqual(perf.sram_scratchpad_bandwidth_gb_s, 48.0)

    def test_incremental_control_energy_and_edp(self) -> None:
        """Verifies Incremental Control Energy (2.88 nJ) and Energy-Delay Product (EDP)."""
        perf = self.perf_calc.calculate_system_performance()
        self.assertAlmostEqual(perf.e_decision_incremental_nj, 2.88, places=2)
        # EDP improvement vs software daemon must exceed 10^8
        self.assertGreater(perf.edp_improvement_vs_software_x, 1.0e8)

    def test_verified_hardware_leverage_vhl(self) -> None:
        """Verifies Master Metric: Verified Hardware Leverage (VHL) Stratification."""
        perf = self.perf_calc.calculate_system_performance()
        # VHL_S = 1.4852 (+48.52% useful work per Joule)
        self.assertAlmostEqual(perf.vhl_s_simulation, 1.4852, places=3)
        # VHL_F FPGA Target >= 1.120
        self.assertGreaterEqual(perf.vhl_f_fpga_target, 1.120)
        # VHL_P Silicon Target >= 1.350
        self.assertGreaterEqual(perf.vhl_p_physical_target, 1.350)
        # Control Leverage Factor CLF: Delta Useful Work >> E_control
        self.assertGreater(perf.clf_control_leverage_factor, 1.0e6)
        self.assertEqual(perf.vital_max_hp, 6)

    def test_financial_feasibility_and_roi(self) -> None:
        """Verifies prototype NRE budget and customer ROI payback."""
        fin = self.fin_calc.calculate_financials()
        self.assertLessEqual(fin.total_prototype_nre_eur, 350000.0)
        self.assertLess(fin.edge_customer_payback_months, 1.0)

    def test_theory_node_55_and_56_registration(self) -> None:
        """Verifies that Theory Nodes 55 and 56 are certified and acyclic."""
        node_55 = GLOBAL_THEORY_DEPENDENCY_GRAPH.get_theory(
            "THEORY-KSAC-SILICON-ECONOMICS-AND-AREA-YIELD-CALCULATIONS"
        )
        self.assertIsNotNone(node_55)
        self.assertEqual(node_55.epistemic_status, TheoryEpistemicStatus.DOKAZANE)

        node_56 = GLOBAL_THEORY_DEPENDENCY_GRAPH.get_theory(
            "THEORY-KSAC-VHL-STRATIFICATION-AND-CONTROL-LEVERAGE-FACTOR"
        )
        self.assertIsNotNone(node_56)
        self.assertEqual(node_56.epistemic_status, TheoryEpistemicStatus.DOKAZANE)

        self.assertFalse(GLOBAL_THEORY_DEPENDENCY_GRAPH.has_cycles())


if __name__ == "__main__":
    unittest.main()

